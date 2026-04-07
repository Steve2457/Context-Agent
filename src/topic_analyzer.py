from model.ollama import OllamaClient
from src.conversation_tree import ConversationTree
ollama_client = OllamaClient()



TOPIC_ANALYSIS_PROMPT = """
# STRICT INSTRUCTION - EXECUTE ONLY THE FOLLOWING LOGIC CHAIN
Act as a dialogue topic consistency adjudicator.Your task is to objectively score the semantic relationship between a new query from user and conversation history summary of dialogues between user and AI assistant. You MUST perform **exactly three steps**:

1️⃣ [Theme Check] Does the new query discuss the **SAME physical/conceptual core object** as history?
   ✅ Valid: "battery life" → "charging speed" (core object = battery)
   ❌ Invalid: "Beijing weather" → "Shanghai weather" (core object changed)
   ⚠️ Rule: Disregard surface differences (tools/locations/times). 
           e.g., "Python data cleaning" vs "Excel data cleaning" → ❌

2️⃣ [Continuity Check] Does the new query **depend on historical context**?
   ✅ Valid: "How fast does it charge?" (refers to prior "battery")
   ❌ Invalid: "Recommend restaurants" (no contextual link)
   ⚠️ Rule: Specially verify pronouns (it/this/that/them etc.), probing words (how/why), some specific signpost words (such as "return to", "previously mentioned", etc.), logical progression

3️⃣ [Final Judgment] Output "yes" ONLY if both steps pass ✅, otherwise "no"

# ANTI-ERROR PROTOCOLS (Critical for lightweight LLMs)
🔴 ABSOLUTELY PROHIBITED:
   • No keyword matching (e.g., "weather" in different cities)
   • No intent speculation (textual content only)

🔵 Core Object Definition (Key innovation):
   - Physical: Devices/items/body parts (iPhone battery, car engine)
   - Conceptual: Problems/tasks/themes (data cleaning, travel planning)
   - ✦ Critical: Core object changes when tools/locations shift ✦

# EXTENDED EXAMPLE BANK 
| History Summary          | New Query               | Theme | Continuity | Output |
|--------------------------|-------------------------|-------|------------|--------|
| "iPhone 15 battery life" | "Charging speed?"       | ✅    | ✅         | yes    |
| "Beijing weather today"  | "Shanghai temperature?" | ❌    | ❌         | no     |
| "Python Pandas cleaning" | "Excel missing values"  | ❌    | ❌         | no     |
| "Avatar movie effects"   | "Cameron's next film?"  | ✅    | ✅         | yes    |
| "Diabetes diet tips"     | "Exercise recommendations"| ✅  | ✅         | yes    |
| "Laptop overheating"     | "Phone thermal issues"  | ❌    | ❌         | no     |

# OUTPUT REQUIREMENTS
■ ONLY output SINGLE word: `yes` or `no` WITHOUT any extra characters(no spaces/punctuation)

# CURRENT INPUT
Now, please start comparing the history summary and the new query:
History Summary: {summary}
New Query: {query}

"""

def _get_topic_scores(branch_summary: str, new_query: str) -> bool:
  
    if not branch_summary.strip():
        return False

    prompt = TOPIC_ANALYSIS_PROMPT.format(summary=branch_summary, query=new_query)

    try:
        response = ollama_client.chat(
            model='gemma3:12b',
            messages=[{'role': 'user', 'content': prompt}]
        )
        content = response['message']['content']

        text = (content or "").strip().lower()
      
        if "```" in text:
        
            tokens = [t for t in text.replace('`', ' ').split()] 
        else:
            tokens = text.split()

        decision = next((t for t in tokens if t in ("yes", "no")), tokens[0] if tokens else "")
        return decision == "yes"
    except Exception as e:
        print(f"An error occurred when calling the LLM for topic judgment.: {e}")
        return False

def _check_tree_for_topic_match(tree: ConversationTree, new_query: str) -> bool:
    
    leaf_nodes = tree.find_leaf_nodes()

    if not leaf_nodes:
        return False

    for leaf in leaf_nodes:
        
        summaries = []
        current = leaf
        while current:
            if current.summary:
                summaries.append(current.summary)
            current = current.parent

        branch_summary = " ".join(reversed(summaries))

        if not branch_summary.strip():
            continue

        is_same_topic = _get_topic_scores(branch_summary, new_query)
        print(f"  - Branch (Leaf ID: ...{str(leaf.id)[-6:]}): Judgment={'yes' if is_same_topic else 'no'}")

        if is_same_topic:
          
            return True

    return False

def analyze_topic_and_get_branch(
    trees: list[ConversationTree], 
    current_tree_index: int, 
    new_query: str
) -> tuple[int, bool]:
    
    
    current_tree = trees[current_tree_index]

    if not current_tree.root.children:
        return current_tree_index, False
    
    print(f"\nAnalyzing the topic of the new query and the current conversation branch #{current_tree_index}...")
    if _check_tree_for_topic_match(current_tree, new_query):
        
        print(f"The query belongs to the current topic. Continue in branch #{current_tree_index}.")
        return current_tree_index, False

  
    print("The query does not belong to the current topic. Checking other branches now....")
    for i, tree in enumerate(trees):
        if i == current_tree_index:
            continue 

        print(f"Analyzing the topic related to branch #{i}...")
        if _check_tree_for_topic_match(tree, new_query):
            print(f"Find the matching topic in branch #{i}. Switch to this branch.")
            return i, False

    print("No matching topics were found in any existing branches. A new conversation branch is being created.。")
    system_prompt = trees[0].root.system_message['content'] if trees and trees[0].root.system_message else "You are a helpful assistant."
    new_tree = ConversationTree(initial_message=system_prompt)
    trees.append(new_tree)
    new_tree_index = len(trees) - 1
    
    return new_tree_index, True 