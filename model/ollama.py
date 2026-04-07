from typing import List, Dict, Optional
import ollama


class OllamaClient:
    

    def __init__(self):
        self._client = ollama.Client()

    def chat(self, model: str, messages: List[Dict], format: Optional[str] = None):
        kwargs = {"model": model, "messages": messages}
        if format is not None:
            kwargs["format"] = format
        return self._client.chat(**kwargs)


def chat_with_ollama(history: List[Dict], model: str = "gemma3:12b") -> str:
    
    try:
        client = ollama.Client()
        response = client.chat(model=model, messages=history)
        return response["message"]["content"]
    except Exception as e:
        return f"An error occurred: {e}"




