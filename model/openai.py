from typing import List, Dict, Optional
import os

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


class OpenAIChatClient:
    

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = "gpt-3.5-turbo", timeout: int = 60):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model
        self.timeout = timeout

        if OpenAI is None:
            raise RuntimeError("The openai library is not installed. Please install it first using pip install openai`。");

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, history: List[Dict], model: Optional[str] = None) -> str:
        if not self.api_key:
            return "Error: OPENAI_API_KEY is not set。"

        use_model = model or self.model

        try:
            resp = self._client.chat.completions.create(
                model=use_model,
                messages=history,
                timeout=self.timeout,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"An error occurred: {e}"


def chat_with_openai(history: List[Dict], api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = "gpt-3.5-turbo") -> str:
    
    try:
        client = OpenAIChatClient(api_key=api_key, base_url=base_url, model=model)
        return client.chat(history)
    except Exception as e:
        return f"An error occurred: {e}"


