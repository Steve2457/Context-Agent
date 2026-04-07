import uuid
import numpy as np
import ollama
from sklearn.metrics.pairwise import cosine_similarity

# Wrapper for Ollama embedding model to mimic SentenceTransformer's interface
class OllamaEmbeddingWrapper:
    def __init__(self, model_name='dengcao/Qwen3-Embedding-0.6B:F16'):
        self.model_name = model_name
        self.client = ollama.Client()
        self._dimension = None

    def encode(self, text):
        """Encodes a single string of text using the Ollama model."""
        if not text or not text.strip():
            # Return a zero vector if the text is empty
            return np.zeros(self.get_sentence_embedding_dimension())
        
        response = self.client.embeddings(model=self.model_name, prompt=text)
        return np.array(response['embedding'])

    def get_sentence_embedding_dimension(self):
        """Gets the embedding dimension by sending a test query to the model."""
        if self._dimension is None:
            try:
                response = self.client.embeddings(model=self.model_name, prompt="test")
                self._dimension = len(response['embedding'])
            except Exception as e:
                print(f"Error: Could not connect to Ollama to determine embedding dimension for model '{self.model_name}'.")
                print("Please ensure Ollama is running and the model is available.")
                raise e
        return self._dimension

# Use the Ollama wrapper for embeddings
embedding_model = OllamaEmbeddingWrapper(model_name='dengcao/Qwen3-Embedding-0.6B:F16')

class Node:
    """Represents a node in the conversation tree, storing a full turn."""
    def __init__(self, user_message=None, assistant_message=None, system_message=None, parent=None, branch_id=None):
        self.id = str(uuid.uuid4())
        self.user_message = user_message
        self.assistant_message = assistant_message
        self.system_message = system_message # For the root node
        self.parent = parent
        self.children = []
        self.summary = None
        self.branch_id = branch_id
        self.vector = self._embed_turn()

    def _embed_turn(self):
        """Embeds the combined content of the turn into a vector."""
        # Combine user and assistant messages for a more representative embedding
        user_content = self.user_message.get("content", "") if self.user_message else ""
        assistant_content = self.assistant_message.get("content", "") if self.assistant_message else ""
        system_content = self.system_message.get("content", "") if self.system_message else ""
        
        combined_content = f"User: {user_content}\nAssistant: {assistant_content}".strip()

        if system_content and not combined_content:
             combined_content = system_content

        if combined_content:
            return embedding_model.encode(combined_content)
        return np.zeros(embedding_model.get_sentence_embedding_dimension())

    def add_child(self, user_message, assistant_message, branch_id):
        """Adds a child node to the current node."""
        child_node = Node(user_message=user_message, assistant_message=assistant_message, parent=self, branch_id=branch_id)
        self.children.append(child_node)
        return child_node

class ConversationTree:
    """Manages the conversation history as a tree with branching support."""

    def __init__(self, initial_message: str = ""):
        # Root and main branch setup
        main_branch_id = str(uuid.uuid4())
        self.root = Node(system_message={"role": "system", "content": initial_message}, branch_id=main_branch_id)
        self.branches = {main_branch_id: [self.root]}  # branch_id -> list[Node]
        self.branch_origins = {main_branch_id: None}   # branch_id -> fork node id (None for main)
        self.active_branch_id = main_branch_id
        self.current_node = self.root
        self.node_map = {self.root.id: self.root}

    # ---- Internal helpers ----
    def _add_node_to_tree(self, user_message, assistant_message, parent_node, branch_id):
        new_node = parent_node.add_child(user_message, assistant_message, branch_id=branch_id)
        if branch_id not in self.branches:
            self.branches[branch_id] = []
        self.branches[branch_id].append(new_node)
        self.node_map[new_node.id] = new_node
        return new_node

    # ---- Branch management ----
    def create_branch(self, fork_node: Node) -> str:
        new_branch_id = str(uuid.uuid4())
        self.branches[new_branch_id] = []
        self.branch_origins[new_branch_id] = fork_node.id
        print(f"Branch {new_branch_id} created, forking from node {fork_node.id}")
        return new_branch_id

    def switch_branch(self, branch_id: str):
        if branch_id not in self.branches:
            raise ValueError(f"Branch with id {branch_id} not found.")
        self.active_branch_id = branch_id
        if self.branches[branch_id]:
            self.current_node = self.branches[branch_id][-1]
        print(f"Switched to branch {branch_id}")

    def prepare_for_new_turn(self, fork_from_node_id: str | None = None):
        if fork_from_node_id:
            fork_node = self.node_map.get(fork_from_node_id)
            if not fork_node:
                raise ValueError(f"Node with id {fork_from_node_id} not found.")
            new_branch_id = self.create_branch(fork_node)
            self.switch_branch(new_branch_id)
            self.current_node = fork_node

    # ---- Turn management ----
    def add_turn(self, user_content: str, assistant_content: str) -> Node:
        parent_node = self.current_node
        user_message = {"role": "user", "content": user_content}
        assistant_message = {"role": "assistant", "content": assistant_content}
        self.current_node = self._add_node_to_tree(user_message, assistant_message, parent_node, self.active_branch_id)
        return self.current_node

    # ---- Retrieval & history ----
    def search_similar_nodes(self, query_content: str, k: int = 3):
        all_nodes = [n for branch_nodes in self.branches.values() for n in branch_nodes]
        searchable = [n for n in all_nodes if not n.system_message]
        if not searchable:
            return []
        query_vector = embedding_model.encode(query_content)
        node_vectors = np.array([n.vector for n in searchable])
        similarities = cosine_similarity(query_vector.reshape(1, -1), node_vectors)[0]
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        return [(searchable[i], similarities[i].item()) for i in top_k_idx]

    def get_all_nodes(self):
        if not self.root:
            return []
        nodes = []
        queue = [self.root]
        visited = {self.root.id}
        while queue:
            curr = queue.pop(0)
            nodes.append(curr)
            for child in curr.children:
                if child.id not in visited:
                    queue.append(child)
                    visited.add(child.id)
        return nodes

    def get_current_history(self, all_trees, current_tree_index: int):
        history = []
        node = self.current_node
        while node is not None:
            if node.assistant_message:
                history.append(node.assistant_message)
            if node.user_message:
                history.append(node.user_message)
            if node.system_message:
                history.append(node.system_message)
            node = node.parent
        history.reverse()

        other_topics_texts = []
        for i, tree in enumerate(all_trees):
            if i == current_tree_index:
                continue
            tree_summaries = [n.summary for n in tree.get_all_nodes() if n.summary]
            if tree_summaries:
                unique_tree_summaries = sorted(set(tree_summaries))
                topic_summary_text = f"Topic {i + 1}:\n- " + "\n- ".join(unique_tree_summaries)
                other_topics_texts.append(topic_summary_text)

        other_branch_texts = []
        for branch_id, nodes in self.branches.items():
            if branch_id == self.active_branch_id or not nodes:
                continue
            branch_summaries = [n.summary for n in nodes if n.summary]
            if branch_summaries:
                unique_branch_summaries = sorted(set(branch_summaries))
                branch_summary_text = f"Inactive Branch {branch_id[:8]}:\n- " + "\n- ".join(unique_branch_summaries)
                other_branch_texts.append(branch_summary_text)

        summary_sections = []
        if other_topics_texts:
            summary_sections.append("The user also discussed the following content in other topics:\n" + "\n\n".join(other_topics_texts))
        if other_branch_texts:
            summary_sections.append("This topic also has the following inactive branches:\n" + "\n\n".join(other_branch_texts))
        if summary_sections:
            summaries_text = "\n\n".join(summary_sections)
            summary_message = {"role": "system", "content": summaries_text}
            if len(history) > 1:
                history.insert(len(history) - 1, summary_message)
            else:
                history.insert(0, summary_message)
        return history

    # ---- Printing ----
    def print_tree(self, is_current_tree: bool = False):
        title = "--- Conversation Tree (Current) ---" if is_current_tree else "--- Conversation Tree ---"
        print(f"\n{title}")
        self._print_node_recursive(self.root, 0, set())
        print("-------------------------\n")

    def print_branches(self):
        print("\n=== Branch Overview ===")
        for idx, (branch_id, nodes) in enumerate(self.branches.items(), start=1):
            short_id = branch_id[:8]
            active_mark = " *ACTIVE*" if branch_id == self.active_branch_id else ""
            fork_from = self.branch_origins.get(branch_id)
            fork_short = fork_from[:8] if fork_from else "root"
            print(f"Branch {idx}: {short_id}{active_mark} (fork from {fork_short}) | turns: {len(nodes)}")
            if not nodes:
                print("  (no turns yet)")
                continue
            for i, n in enumerate(nodes):
                if n.system_message:
                    continue
                user_text = n.user_message.get('content', '') if n.user_message else ''
                asst_text = n.assistant_message.get('content', '') if n.assistant_message else ''
                user_disp = (user_text[:40] + '...') if len(user_text) > 40 else user_text
                asst_disp = (asst_text[:40] + '...') if len(asst_text) > 40 else asst_text
                if n.summary:
                    summary_part = f" | S: {n.summary[:35]}..." if len(n.summary) > 35 else f" | S: {n.summary}"
                else:
                    summary_part = ""
                print(f"  - Turn {i}: U: {user_disp.replace(chr(10), ' ')} | A: {asst_disp.replace(chr(10), ' ')}{summary_part}")
        print("=======================\n")

    def _print_node_recursive(self, node: Node, level: int, visited_nodes: set):
        if node.id in visited_nodes:
            return
        visited_nodes.add(node.id)
        indent = "  " * level
        current_marker = "  <-- current" if node == self.current_node else ""
        branch_marker = f" (Branch: {node.branch_id[:8]})" if node.branch_id else ""
        if node.system_message:
            content = node.system_message.get("content", "").strip()
            display_content = (content[:70] + '...') if len(content) > 70 else content
            print(f"{indent}- System: {display_content}{branch_marker}{current_marker}")
        if node.user_message:
            user_content = node.user_message.get("content", "").strip()
            assistant_content = node.assistant_message.get("content", "").strip() if node.assistant_message else ""
            display_user = (user_content[:35] + '...') if len(user_content) > 35 else user_content
            display_assistant = (assistant_content[:35] + '...') if len(assistant_content) > 35 else assistant_content
            print(f"{indent}- Turn:{branch_marker}{current_marker}")
            print(f"{indent}  - User: {display_user.replace(chr(10), ' ')}")
            print(f"{indent}  - Assistant: {display_assistant.replace(chr(10), ' ')}")
            if node.summary:
                summary_indent = "  " * (level + 1)
                print(f"{summary_indent}[Summary: {node.summary}]")
        for child in node.children:
            self._print_node_recursive(child, level + 1, visited_nodes)

    # ---- Utility ----
    def find_leaf_nodes(self):
        return [node for node in self.node_map.values() if not node.children]