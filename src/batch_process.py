import os
import json
import glob
import time
from tqdm import tqdm
from src.summarizer import summarize_turn
from src.topic_analyzer import analyze_topic_and_get_branch
from src.conversation_tree import ConversationTree
from src.direct import DirectHistory
from src.token_counter import compute_context_tokens
from src.branch_manager import BranchManager


def call_llm_with_retry(chat_function, history, max_retries=3, initial_delay=5):
    """
    Calls the language model with a retry mechanism for handling rate limit errors.

    Args:
        chat_function (callable): The function that calls the LLM.
        history (list): The conversation history to send to the LLM.
        max_retries (int): Maximum number of retries.
        initial_delay (int): Initial delay in seconds before the first retry.

    Returns:
        str: The response from the LLM or an error message.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        response = chat_function(history)
        # Check if the response indicates a rate limit or client error (e.g., "429 Client Error")
        if isinstance(response, str) and "429 Client Error" in response:
            print(f"  Server error detected. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
            print(f"  Error details: {response}")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
            continue
        return response

    print("  Failed to get a valid response after multiple retries. Skipping this turn.")
    return "Error: No response received after multiple retries."


def batch_process(input_path, output_dir, smart_mode: bool, USE_MODEL, chat_with_ollama, chat_with_openai, llm_client, filename_prefix=""):
    """
    Processes conversations from a jsonl file or a directory of jsonl files in batch mode.

    Args:
        input_path (str): Path to the input jsonl file or directory.
        output_dir (str): Directory to save the output json files.
        smart_mode (bool): Whether to use the smart context strategy.
        USE_MODEL (str): Model selection string, e.g., 'ollama' or 'openai' or 'zhipu'.
        chat_with_ollama (callable): Function to call Ollama API.
        chat_with_openai (callable): Function to call OpenAI API.
        llm_client: The language model client for BranchManager.
        filename_prefix (str): A prefix for the output filenames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    completed_ids = set()
    existing_outputs = glob.glob(os.path.join(output_dir, f"{filename_prefix}*.json"))
    for path in existing_outputs:
        fname = os.path.basename(path)
        if not fname.endswith('.json'):
            continue
        
        if fname.startswith(filename_prefix):
            conv_id = fname[len(filename_prefix):-5]
        else:
            conv_id = fname[:-5]
        if conv_id:
            completed_ids.add(conv_id)

    if os.path.isdir(input_path):
        input_files = glob.glob(os.path.join(input_path, "*.jsonl"))
        if not input_files:
            print(f"Error: No .jsonl files found in the directory '{input_path}'.")
            return
    elif os.path.isfile(input_path) and input_path.endswith(".jsonl"):
        input_files = [input_path]
    else:
        print(f"Error: The provided input path '{input_path}' is not a valid .jsonl file or directory.")
        return

    total_turns = 0
    conversations_to_process = []
    print("Pre-scanning files to count total turns...")
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    conversation_id = data.get('conversation_id')
                
                    if str(conversation_id) in completed_ids:
                       
                        continue

                    user_turns = data.get('user_turns', [])
                    total_turns += len(user_turns)
                    conversations_to_process.append(data)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {input_file}")
    
    if total_turns == 0:
        print("No new user turns found to process.")
        return

    with tqdm(total=total_turns, desc="Processing Turns") as pbar:
        for data in conversations_to_process:
            conversation_id = data['conversation_id']
         
            user_turns = sorted(data['user_turns'], key=lambda x: x['turn_id'])
            
            pbar.set_description(f"Conv: {conversation_id}")

            system_prompt = "You are a helpful assistant."
            if smart_mode:
             
                trees = [ConversationTree(system_prompt)]
                current_tree_index = 0
                branch_managers = {0: BranchManager(trees[0], llm_client)}
            else:
              
                conversation_history = DirectHistory(system_prompt)
            
            output_turns = []
            
            for turn in user_turns:
                user_input = turn['content']
                
                if smart_mode:
                    # Analyze topic to decide which tree to use
                    new_tree_index, is_new_tree = analyze_topic_and_get_branch(
                        trees, current_tree_index, user_input
                    )
                    current_tree_index = new_tree_index
                    
                    if is_new_tree:
                        branch_managers[current_tree_index] = BranchManager(trees[current_tree_index], llm_client)

                    current_tree = trees[current_tree_index]
                    branch_manager = branch_managers[current_tree_index]

                    decision = branch_manager.decide_branch(user_input)
                    
                    action = decision.get('action')
                    if action == 'CREATE_BRANCH':
                        current_tree.prepare_for_new_turn(fork_from_node_id=decision.get('fork_node_id'))
                    elif action == 'SWITCH_BRANCH':
                        current_tree.switch_branch(branch_id=decision.get('target_branch_id'))

                  
                    history_for_llm = current_tree.get_current_history(trees, current_tree_index)
                    history_for_llm.append({"role": "user", "content": user_input})
                    
                    context_tokens = compute_context_tokens(history_for_llm[:-1], model='gemma3:12b')

                    if USE_MODEL == "openai":
                        assistant_response = call_llm_with_retry(chat_with_openai, history_for_llm)
                    elif USE_MODEL == "zhipu":
                        from model.zhipu import chat_with_zhipu
                        assistant_response = call_llm_with_retry(chat_with_zhipu, history_for_llm)
                    else: # Default to ollama
                        assistant_response = call_llm_with_retry(chat_with_ollama, history_for_llm)

                    new_turn_node = current_tree.add_turn(
                        user_content=user_input,
                        assistant_content=assistant_response
                    )

                    summary = summarize_turn(user_input, assistant_response)
                    new_turn_node.summary = summary
                
                else: 
                    conversation_history.add_user_message(user_input)
                    current_history = conversation_history.get_current_history()
                    context_tokens = compute_context_tokens(current_history[:-1], model='gemma3:12b')

                    if USE_MODEL == "openai":
                        assistant_response = call_llm_with_retry(chat_with_openai, current_history)
                    elif USE_MODEL == "zhipu":
                        from model.zhipu import chat_with_zhipu
                        assistant_response = call_llm_with_retry(chat_with_zhipu, current_history)
                    else:
                        assistant_response = call_llm_with_retry(chat_with_ollama, current_history)
                    
                    conversation_history.add_assistant_message(assistant_response)

                output_turns.append({"role": "user", "turn_id": turn['turn_id'], "content": user_input})
                output_turns.append({"role": "assistant", "content": assistant_response, "context_tokens": context_tokens})
                pbar.update(1)
           
            output_data = {
                "conversation_id": conversation_id,
                "metadata": data.get("metadata", {}),
                "turns": output_turns
            }
           
            output_filename = f"{filename_prefix}{conversation_id}.json"
            output_filepath = os.path.join(output_dir, output_filename)
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                json.dump(output_data, outfile, ensure_ascii=False, indent=2)

    print(f"\nBatch processing complete. Outputs are in '{output_dir}'.")