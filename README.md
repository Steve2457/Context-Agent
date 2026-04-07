# Context-Agent & NTM Benchmark
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

This project introduces the **Context-Agent** and the **Non-linear Task Multiturn Dialogue (NTM) benchmark**.

## Table of Contents

- [Requirements](#requirements)
- [Ollama Setup (for local models)](#ollama-setup-for-local-models)
- [Dataset Format](#dataset-format-inputjsonl)
- [Quickstart](#quickstart)
- [Output Layout](#output-layout)
- [Evaluation](#evaluation)

## Requirements

- Python 3.10+
- For local models, refer to the **Ollama Setup** section.
- For cloud models, ensure you have the corresponding API keys set as environment variables:
  - `OPENAI_API_KEY` for OpenAI
  - `ZAI_API_KEY` for Zhipu

Alternatively, you can place a `.env` file in the project root; `main.py` will load it automatically if present.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Ollama Setup (for local models)

1.  **Install Ollama:** Follow the official instructions at [ollama.com](https://ollama.com).
2.  **Pull a model:** Before running `main.py` with Ollama, ensure you have pulled the desired model. For example:
    ```bash
    ollama pull qwen3:30b
    ```
3.  **Ensure Ollama is running:** The script expects the Ollama server to be available at its default address `http://localhost:11434`.

## Dataset Format (`input/*.jsonl`)

Each line in the dataset is a JSON object with the following structure:

```json
{
  "conversation_id": "<string|number>",
  "user_turns": [
    { "turn_id": 1, "content": "..." },
    { "turn_id": 2, "content": "..." }
  ],
  "metadata": { "optional": true }
}
```

Example datasets are provided under `input/`.

## Quickstart

The main entry point for the project is `main.py`. It processes JSONL datasets and generates conversation JSON files under `output/...`.

**Minimal run (local Ollama, smart context ON by default):**

```bash
python main.py
```

**Common flags:**

- `--use-model {ollama|openai|zhipu}`: Specifies the backend model to use.
- `--smart-context` / `--no-smart-context`: Enables/disables dynamic tree context management.
- `--input-path <path>`: Path to a JSONL file or a folder containing JSONL files.
- `--ollama-model <name>`: E.g., `qwen3:30b`.
- `--openai-model <name>`: E.g., `gpt-4.1` or any gateway model ID.
- `--zhipu-model <name>`: E.g., `GLM-4`.

## Output Layout

Output files are saved based on the `--smart-context` flag:

- **Smart Context**: `output/smart/<model_name>/`
- **Direct Context**: `output/direct/<model_name>/`

Filename convention: `<backend>-<S|D>-<conversation_id>.json` where `S` = smart, `D` = direct. Each output JSON contains:

```json
{
  "conversation_id": "...",
  "metadata": { },
  "turns": [
    { "role": "user", "turn_id": 1, "content": "..." },
    { "role": "assistant", "content": "...", "context_tokens": 123 }
  ]
}
```

`context_tokens` counts tokens for the context portion only (excludes current query and reply) via `src/token_counter.py`.

## Evaluation

Use `evaluate.py` to score conversations with a judge model. It pairs each conversation’s final assistant reply with a checkpoint question from `input/dataset-full.jsonl` and asks the judge to output "x/y" satisfied goals.

**Basic usage:**

```bash
# Evaluate a specific output folder with OpenAI judge
python evaluate.py --mode output/smart/gpt-4.1 --judge openai
```

**Key flags:**

- `--mode`: One of `direct`, `smart`, `rag`, a subfolder under `output/direct`, or any folder path containing conversation JSONs.
- `--judge {openai|ollama}` and `--judge-model <name>`.
- `--openai-base-url <url>` to point to OpenAI-compatible gateways.
- `--limit <N>` to evaluate a subset.

**Artifacts:**

- CSV with per-conversation rows: `evaluation_result/<label>_<suffix>_scores.csv`
- Summary JSON without per-conversation details: `evaluation_result/<label>_<suffix>_score.json`






