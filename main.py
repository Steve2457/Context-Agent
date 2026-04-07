import os
import argparse
from model.ollama import OllamaClient, chat_with_ollama as ollama_chat
from model.openai import OpenAIChatClient
from model.zhipu import ZhipuClient, chat_with_zhipu as zhipu_chat

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from src.summarizer import summarize_turn
from src.topic_analyzer import analyze_topic_and_get_branch
from src.conversation_tree import Node, ConversationTree
from src.direct import DirectHistory
from src.batch_process import batch_process
from src.branch_manager import BranchManager
from src.token_counter import compute_context_tokens

# --- global configs (set via CLI) ---
USE_MODEL = None
OPENAI_MODEL = None
OLLAMA_MODEL = None
ZHIPU_MODEL = None

# --- model clients (initialized in main) ---
ollama_client = None
openai_client = None
zhipu_client = None

def chat_with_ollama(history):
    
    return ollama_chat(history, model=OLLAMA_MODEL)

def chat_with_openai(history):
    
    return openai_client.chat(history)

def chat_with_zhipu(history):

    return zhipu_client.chat(history)

def interactive_chat(smart_mode: bool):
    """Runs the chat in interactive mode."""
    system_prompt = "You are a helpful assistant."
    
    if smart_mode:
        # Start with a single conversation tree for smart mode
        trees = [ConversationTree(system_prompt)]
        current_tree_index = 0
        # Create a branch manager for the initial tree
        branch_managers = {0: BranchManager(trees[0], ollama_client)}
    else:
        # Use direct history for non-smart mode
        conversation_history = DirectHistory(system_prompt)

    print(f"Starting chat with {USE_MODEL.upper()} in {'SMART' if smart_mode else 'DIRECT'} mode.")
    print("Type 'exit' to end.")
    if smart_mode:
        print("Type 'view_trees' to see all trees. Type 'view_branches' to see branches in current tree.")
    print(f"System: {system_prompt}")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting chat.")
            break
        
        if smart_mode:
            if user_input.lower() == 'view_trees':
                for i, tree in enumerate(trees):
                    tree.print_tree(is_current_tree=(i == current_tree_index))
                continue
            if user_input.lower() == 'view_branches':
                trees[current_tree_index].print_branches()
                continue

        if smart_mode:
            # Analyze the topic to decide which tree to use
            new_tree_index, is_new_tree = analyze_topic_and_get_branch(
                trees, current_tree_index, user_input
            )
            current_tree_index = new_tree_index

            # If a new tree was created, it needs a new branch manager
            if is_new_tree:
                branch_managers[current_tree_index] = BranchManager(trees[current_tree_index], ollama_client)

            # --- Branching Logic ---
            current_tree = trees[current_tree_index]
            branch_manager = branch_managers[current_tree_index]

            # 1. Decide on the branch *before* calling the LLM
            decision = branch_manager.decide_branch(user_input)
            
            # 2. Prepare the tree context based on the decision
            action = decision.get('action')
            if action == 'CREATE_BRANCH':
                current_tree.prepare_for_new_turn(fork_from_node_id=decision.get('fork_node_id'))
            elif action == 'SWITCH_BRANCH':
                current_tree.switch_branch(branch_id=decision.get('target_branch_id'))

            # 3. Get the clean history from the (potentially new) branch
            history_for_llm = current_tree.get_current_history(trees, current_tree_index)
            history_for_llm.append({"role": "user", "content": user_input})

            context_token_count = compute_context_tokens(history_for_llm[:-1], model=OLLAMA_MODEL if USE_MODEL=="ollama" else 'gemma3:4b')
            print(f"[Context tokens]: {context_token_count}")

            # 4. Call the LLM with the prepared context
            print("Assistant: Thinking...")
            if USE_MODEL == "openai":
                assistant_response = chat_with_openai(history_for_llm)
            elif USE_MODEL == "zhipu":
                assistant_response = chat_with_zhipu(history_for_llm)
            else:
                assistant_response = chat_with_ollama(history_for_llm)
            
            print(f"Assistant: {assistant_response}")

            # 5. Add the completed turn to the current node of the tree
            new_turn_node = current_tree.add_turn(
                user_content=user_input,
                assistant_content=assistant_response
            )

            # 6. Summarize the turn
            print("Summarizing turn...")
            summary = summarize_turn(user_input, assistant_response)
            new_turn_node.summary = summary
            print("Summary complete.")

        else: # direct mode
            conversation_history.add_user_message(user_input)
            current_history = conversation_history.get_current_history()
            
            context_token_count = compute_context_tokens(current_history[:-1], model=OLLAMA_MODEL if USE_MODEL=="ollama" else 'gemma3:4b')
            print(f"[Context tokens]: {context_token_count}")
        

            print("Assistant: Thinking...")
            if USE_MODEL == "openai":
                assistant_response = chat_with_openai(current_history)
            elif USE_MODEL == "zhipu":
                assistant_response = chat_with_zhipu(current_history)
            else:
                assistant_response = chat_with_ollama(current_history)
            
            print(f"Assistant: {assistant_response}")
            conversation_history.add_assistant_message(assistant_response)

def main():
    """Main function to run the chat application."""

    global USE_MODEL, OPENAI_MODEL, OLLAMA_MODEL, ZHIPU_MODEL, openai_client, ollama_client, zhipu_client

    parser = argparse.ArgumentParser(description="TS-IR chat runner")
    parser.add_argument(
        "--use-model",
        choices=["ollama", "openai", "zhipu"],
        required=True,
        help="Select the backend model: ollama | openai | zhipu",
    )
    parser.add_argument(
        "--smart-context",
        dest="smart_context",
        action="store_true",
        help="Enable intelligent context management",
    )
    parser.add_argument(
        "--no-smart-context",
        dest="smart_context",
        action="store_false",
        help="Disable intelligent context management",
    )
    parser.set_defaults(smart_context=True)

    parser.add_argument(
        "--openai-model",
        default=None,
        help="OpenAI conversation model names, such as gpt-3.5-turbo",
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Ollama conversation model names, such as gemma3:12b",
    )
    parser.add_argument(
        "--zhipu-model",
        default=None,
        help=" GLM conversation model names, such as GLM-4-AirX",
    )
    parser.add_argument(
        "--input-path",
        default="input",
        help="Input path: Supports directories (batch .jsonl) or a single .jsonl file",
    )

    args = parser.parse_args()

  
    RUN_BATCH_MODE = True
    
    SMART_CONTEXT = args.smart_context

    USE_MODEL = args.use_model
    OPENAI_MODEL = args.openai_model
    OLLAMA_MODEL = args.ollama_model
    ZHIPU_MODEL = args.zhipu_model

    if USE_MODEL == "openai" and not OPENAI_MODEL:
        raise SystemExit("Error: Please select an OpenAI model name, for example --openai-model gpt-4o")
    if USE_MODEL == "ollama" and not OLLAMA_MODEL:
        raise SystemExit("Error: Please select an Ollama model name, for example --ollama-model gemma3:12b")
    if USE_MODEL == "zhipu" and not ZHIPU_MODEL:
        raise SystemExit("Error: Please select the model name of Zhipu, for example --zhipu-model GLM-4-AirX")

    openai_client = OpenAIChatClient(model=OPENAI_MODEL) if USE_MODEL == "openai" else None
    zhipu_client = ZhipuClient(model=ZHIPU_MODEL) if USE_MODEL == "zhipu" else None
  
    ollama_client = OllamaClient()

    INPUT_PATH = args.input_path

    if USE_MODEL == "openai":
        model_name_for_folder = OPENAI_MODEL
    elif USE_MODEL == "zhipu":
        model_name_for_folder = ZHIPU_MODEL
    else:
   
        model_name_for_folder = OLLAMA_MODEL

    if SMART_CONTEXT:
        OUTPUT_DIRECTORY_PATH = os.path.join("output", "smart", model_name_for_folder)
        context_prefix = "S-"
    else:
        OUTPUT_DIRECTORY_PATH = os.path.join("output", "direct", model_name_for_folder)
        context_prefix = "D-"
   
    model_prefix = f"{USE_MODEL}-"
    FILENAME_PREFIX = model_prefix + context_prefix

    if RUN_BATCH_MODE:
        print(f"Run in batch mode, input path: '{INPUT_PATH}'")
        batch_process(
            INPUT_PATH,
            OUTPUT_DIRECTORY_PATH,
            SMART_CONTEXT,
            USE_MODEL,
            chat_with_ollama,
            chat_with_openai,
            ollama_client,
            FILENAME_PREFIX
        )
    else:
        print("Running in interactive mode.")
        interactive_chat(SMART_CONTEXT)


if __name__ == "__main__":
    main()
