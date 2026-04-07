from model.ollama import OllamaClient
ollama_client = OllamaClient()

SUMMARIZATION_PROMPT = """
# Role and Goal
You are a conversation summary generator. Your sole task is to compress a single conversation between [USER] and [ASSISTANT] into the following format. This object will be used by another language model to quickly understand the context of past interactions.You only need to output the content of summary_for_llm, with no other content whatsoever.

# Output Format
summary_for_llm: (string) An information-concentrated summary designed for AI, written in the third person. This summary must integrate the user's intention and the LLM's actions into a single declarative sentence. This is the most critical field for providing context to downstream LLMs. If specific numbers are involved, the summary must include important data.

# Example
## Input Conversation:
[USER]:
I'm trying to understand the difference between 'git merge' and 'git rebase'. Can you explain it to me in simple terms, maybe with an analogy?

[ASSISTANT]:
Certainly. You can think of a project's history as a storybook.

- **`git merge`** is like connecting two separate story chapters (branches) with a new page that says, "At this point, the stories from Chapter A and Chapter B merge." It creates a new "merge commit" and preserves the exact history of both branches, resulting in a history graph that looks like a diamond or a fork.

- **`git rebase`** is like taking all the pages of your Chapter B and placing them one by one at the end of Chapter A. It rewrites the history of your branch to make it look as if all your work was done in a straight line. This creates a cleaner, linear history.

In short, `merge` preserves history, while `rebase` rewrites history for the sake of linearity. For team collaboration, `merge` is usually safer.

## Expected Output:
  summary_for_llm: "The user asked about the difference between 'git merge' and 'git rebase', and the AI assistant clarified their respective functions and different impacts on commit history by providing definitions and analogies."

# Task
Now, please analyze the following conversation:
[USER]:
{user_message}
[ASSISTANT]:
{assistant_message}
"""

def summarize_turn(user_message, assistant_message):
    prompt = SUMMARIZATION_PROMPT.format(user_message=user_message, assistant_message=assistant_message)
    
    try:
        response = ollama_client.chat(
            model='gemma3:4b',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"An error occurred during summarization: {e}"
