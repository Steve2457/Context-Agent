from typing import List, Dict, Optional
import os

try:
    from zai import ZhipuAiClient  
except Exception:  # pragma: no cover
    ZhipuAiClient = None  # type: ignore


class ZhipuClient:


    def __init__(self, api_key: Optional[str] = None, model: str = "GLM-4-AirX", timeout: int = 60):
        self.api_key = api_key or os.getenv("ZAI_API_KEY", "")
        self.model = model
        self.timeout = timeout

        if ZhipuAiClient is None:
            raise RuntimeError("The zai library is not installed. Please first install it with pip install zai and configure the ZAI_API_KEY.")

        self._client = ZhipuAiClient(api_key=self.api_key)

    def chat(self, history: List[Dict], model: Optional[str] = None) -> str:
        if not self.api_key:
            return "Error: ZAI_API_KEY is not set"

        use_model = model or self.model

        try:
            resp = self._client.chat.completions.create(
                model=use_model,
                messages=history,
            )
            
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"An error occurred: {e}"


def chat_with_zhipu(history: List[Dict], model: str = "GLM-4-AirX") -> str:
   
    try:
        client = ZhipuClient(model=model)
        return client.chat(history)
    except Exception as e:
        return f"An error occurred: {e}"


