import json
import urllib.request
import urllib.error


def _ollama_tokenize_count(text: str, model: str = 'gemma3:12b') -> int:
    """Return token count using Ollama's /api/tokenize endpoint; fallback to a rough estimate if unavailable."""
    try:
        payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            url="http://localhost:11434/api/tokenize",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            tokens = data.get("tokens")
            if isinstance(tokens, list):
                return len(tokens)
            if isinstance(data.get("token_count"), int):
                return int(data["token_count"])
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        pass
    return len(text.split())


def compute_context_tokens(history: list, model: str = 'gemma3:4b') -> int:
    """
    Compute the number of tokens used as context for the current response.
    Includes: system prompt and all previous turns.
    Excludes: the current user's query and the model's reply.

    Args:
        history: The messages passed into chat (ordered list of dicts with 'role' and 'content').
        model: Ollama model name for tokenization.

    Returns:
        int: Estimated token count for the context only.
    """
    if not history:
        return 0
    if history[-1].get("role") == "user":
        context_messages = history[:-1]
    else:
        context_messages = history
    total = 0
    for m in context_messages:
        total += _ollama_tokenize_count(m.get("content", ""), model=model)
    return total
