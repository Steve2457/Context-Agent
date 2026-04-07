import json
from src.conversation_tree import ConversationTree, Node

class BranchManager:
    """
    Manages branch creation and switching decisions within a conversation tree
    using RAG and a lightweight LLM.
    """
    def __init__(self, tree: ConversationTree, llm_client):
        self.tree = tree
        self.llm_client = llm_client

    def _is_ancestor(self, potential_ancestor_node, current_node):
        """Checks if a node is an ancestor of another node."""
        node = current_node
        while node.parent:
            if node.parent.id == potential_ancestor_node.id:
                return True
            node = node.parent
        return False

    def decide_branch(self, user_query):
        """
        Decides whether to continue on the current branch or create a new one.
        Returns a dictionary with the decision ('action') and relevant metadata.
        """
        # Optimization: Skip RAG for the very first user message in a tree
        if self.tree.current_node == self.tree.root:
            return {"action": "CONTINUE", "fork_node_id": None}

        print("\n--- Branch Manager: Deciding branch ---")
        print(f"User Query: \"{user_query}\"")

        similar_nodes = self.tree.search_similar_nodes(user_query, k=3)

        if not similar_nodes:
            print("No similar nodes found. Defaulting to CONTINUE.")
            return {"action": "CONTINUE", "fork_node_id": None}

        print("\n[RAG] Top similar nodes found:")
        for i, (node, score) in enumerate(similar_nodes):
            user_content = node.user_message.get('content', '')[:50] if node.user_message else "[...]"
            print(f"  {i+1}. Node ID: {node.id[:8]}, Similarity: {score:.4f}, Content: \"User: {user_content}...\"")

        most_relevant_node, similarity = similar_nodes[0]
        current_branch_latest_node = self.tree.current_node
        
        should_consult_llm = False
        
        print(f"\n[Heuristic Analysis]")
        if similarity > 0.6:  # Similarity threshold is configurable
            is_on_current_branch = most_relevant_node.branch_id == current_branch_latest_node.branch_id
            is_direct_parent = current_branch_latest_node.parent and most_relevant_node.id == current_branch_latest_node.parent.id

            if not is_on_current_branch:
                # Case 1: Most relevant node is on a different branch.
                # This is a strong signal for a potential branch switch or a new fork.
                is_leaf = not most_relevant_node.children
                print(f"Most relevant node ({most_relevant_node.id[:8]}) is on a different branch ({most_relevant_node.branch_id[:8]}). Leaf node: {is_leaf}.")
                should_consult_llm = True
            elif self._is_ancestor(most_relevant_node, current_branch_latest_node) and not is_direct_parent:
                # Case 2: Most relevant node is an ancestor on the same branch, but not the direct parent.
                print(f"Most relevant node ({most_relevant_node.id[:8]}) is an ancestor on the current branch.")
                should_consult_llm = True
            else:
                print("Most relevant node is not a significant fork point (e.g., direct parent).")
        else:
            print(f"Similarity ({similarity:.4f}) is below the threshold.")

        print(f"Final decision to consult LLM: {should_consult_llm}")

        # If the heuristic suggests a potential fork or switch, consult the LLM
        if should_consult_llm:
            print("\n[Decision] Heuristic suggests a potential branch change. Consulting LLM...")
            # Step 3: Use a lightweight LLM for the final decision
            decision = self._get_llm_decision(user_query, similar_nodes, current_branch_latest_node)
            print(f"[LLM Decision] Action: {decision.get('action')}, Fork Node ID: {decision.get('fork_node_id')}, Reason: {decision.get('reason')}")
            print("---------------------------------------\n")
            return decision

        # Default action is to continue
        print("\n[Decision] Heuristic suggests continuing on the current branch.")
        print("---------------------------------------\n")
        return {"action": "CONTINUE", "fork_node_id": None}

    def _get_llm_decision(self, user_query, similar_nodes, current_node):
        """
        Constructs a prompt and gets a decision from the lightweight LLM.
        """
        # --- This function's logic needs to be updated to handle turn-based nodes ---

        # Existing Branches
        branch_info = [f"branch_{i+1}: {branch_id[:8]}" for i, branch_id in enumerate(self.tree.branches.keys())]
        existing_branches = "\n".join(branch_info)

        # Current Path Summaries (from branch root/fork point to current node)
        # Traverse from current node back to the root collecting nodes
        full_path_nodes = []
        node = current_node
        while node:
            full_path_nodes.append(node)
            node = node.parent
        full_path_nodes.reverse()

        # Extract summaries for every node along the path (skip empty summaries)
        path_summaries = [
            {
                "node_id": n.id,
                "summary": n.summary
            }
            for n in full_path_nodes if getattr(n, 'summary', None)
        ]
        # If no summaries exist yet, provide structural placeholders so the LLM still knows path depth
        if not path_summaries:
            path_summaries = [
                {
                    "node_id": n.id,
                    "summary": ""
                } for n in full_path_nodes if not n.system_message
            ]
        current_path_json = json.dumps(path_summaries, indent=2)

        # Retrieved History Nodes (from RAG) - use summaries instead of full turn content
        rag_results_list = [
            {
                "node_id": n.id,
                "summary": (n.summary if getattr(n, 'summary', None) else ""),
                "similarity": round(s, 2)
            }
            for n, s in similar_nodes
        ]
        rag_results_json = json.dumps(rag_results_list, indent=2)

        # Get the most relevant node ID from RAG to use in the example
        most_relevant_rag_node_id = similar_nodes[0][0].id if similar_nodes else "node_abc"


        prompt = f"""System Prompt
                # Role and Task
                You are a Dialogue Flow Controller. Your core task is to analyze the user's query and conversational context to determine their navigational intent. You MUST output ONLY a single JSON object as your decision, with no additional content.


                # Core Decision Rules
                Decisions must be based on comparing "Retrieved History Nodes" with the "Current Path", following these specific rules:

                1. If the user's query is most relevant to a "historical node" (retrieved ancestor node), shows a tendency to diverge from the "Current Path", and the content of the current path provides no substantial help in answering the new query (i.e., the presence or absence of current path content makes no significant difference to the answer) → MUST create a new branch.
                2. If the user's new query is highly similar to a historical node in a non-current topic branch, or if the user explicitly expresses a desire to return to a existing topic branch, and providing the context of the previously existing topic branch is obviously helpful for answering this new query. → MUST switch to the branch that the retrieved history node belongs to.
                3. If the user's query is a logical continuation of the "last turn in the Current Path" and the current path context helps better answer the new query → continue along the current path.
                


                # Input Information
                ## Existing Branches
                {existing_branches}

                ## Current Path Summaries
                {current_path_json}

                ## Retrieved History Nodes
                {rag_results_json}

                ## New User Query
                "{user_query}"


                # Output Requirements
                Choose one of the following actions and output your decision as a single JSON object with these fields:
                - CONTINUE: User continues the current topic. Use ONLY when the query is a direct, incremental continuation of the "last turn in the Current Path".
                → JSON structure: {{"action": "CONTINUE", "reason": "[Explanation for continuing]"}}
                - CREATE_BRANCH: User wants to diverge from a past decision point. Must provide the fork node ID (fork_node_id). Use when the user clearly "backtracks" or "pivots" to explore an alternative path from an earlier conversation node (default choice).
                → JSON structure: {{"action": "CREATE_BRANCH", "fork_node_id": "[ID of most relevant historical node]", "reason": "[Explanation for creating new branch]"}}
                - SWITCH_BRANCH: User wants to switch to another existing branch and providing the context of the previously existing topic branch is obviously helpful for answering this new query. Must provide the target branch ID that the retrieved history node belongs to .
                → JSON structure: {{"action": "SWITCH_BRANCH", "target_branch_id": "[Target branch ID]", "reason": "[Explanation for switching]"}}


                # Example References
                ## Example 1: Create New Branch
                Query: "I think Beijing is too cold. Let's check out Guangzhou instead."
                Decision:
                {{
                "action": "CREATE_BRANCH",
                "fork_node_id": "{most_relevant_rag_node_id}",
                "reason": "User rejects the current path ('too cold') and pivots to an alternative ('Guangzhou') from the retrieved node '{most_relevant_rag_node_id}'. Additionally, previous discussions about Beijing travel plans provide no help in formulating a new plan for Guangzhou."
                }}

                ## Example 2: Continue Current Branch
                Query: "Okay, besides the Palace Museum, what other historical sites do you recommend in Beijing?"
                Decision:
                {{
                "action": "CONTINUE",
                "reason": "The query is a direct continuation of the current topic (Beijing attractions)."
                }}

                ## Example 3: Switch to Existing Branch
                Query: "Let's pause on Beijing for now and go back to the Shanghai plan we discussed."
                Decision:
                {{
                "action": "SWITCH_BRANCH",
                "target_branch_id": "branch_xyz",
                "reason": "User explicitly requests to resume another existing branch ('Shanghai plan'). The previous discussion about Shanghai is obviously helpful for answering the new query."
                }}


                Please output the final decision as a JSON object based on the provided context.
            """

        try:
            # Assuming the llm_client has a `generate` method that takes a prompt
            response = self.llm_client.chat(
                model='gemma3:12b',
                messages=[{'role': 'user', 'content': prompt}],
                format='json'
            )
            decision = json.loads(response['message']['content'])
            
            # Post-process LLM decision to inject correct IDs and prevent hallucination
            action = decision.get('action')
            
            if action == "CREATE_BRANCH":
                if similar_nodes:
                    most_relevant_node = similar_nodes[0][0]
                    decision['fork_node_id'] = most_relevant_node.id
                    print(f"[Branch Manager] Overriding fork_node_id with RAG result: {most_relevant_node.id[:8]}")
            
            if action == "SWITCH_BRANCH":
                if similar_nodes:
                    most_relevant_node = similar_nodes[0][0]
                    decision['target_branch_id'] = most_relevant_node.branch_id
                    print(f"[Branch Manager] Overriding target_branch_id with RAG result: {most_relevant_node.branch_id[:8]}")

            # Validate the decision
            if 'action' in decision and decision['action'] in ["CREATE_BRANCH", "CONTINUE", "SWITCH_BRANCH"]:
                if decision['action'] == "CREATE_BRANCH" and not decision.get('fork_node_id'):
                    # Fallback if fork_node_id is missing
                    print("[LLM Validation] CREATE_BRANCH action is missing 'fork_node_id'. Defaulting to CONTINUE.")
                    return {"action": "CONTINUE", "fork_node_id": None}
                return decision
            else:
                print(f"[LLM Validation] Invalid action received: {decision.get('action')}. Defaulting to CONTINUE.")
                return {"action": "CONTINUE", "fork_node_id": None}

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM decision: {e}")
            # Fallback to default behavior
            return {"action": "CONTINUE", "fork_node_id": None}
        
        return {"action": "CONTINUE", "fork_node_id": None}
