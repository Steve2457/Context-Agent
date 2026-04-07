import os
import json
import csv
import argparse
from typing import List, Dict, Any, Tuple, Optional

from model.openai import OpenAIChatClient
from model.ollama import chat_with_ollama as ollama_chat
import sys
import time
from datetime import timedelta

OPENAI_MODEL_DEFAULT = "gpt-4.1"
OLLAMA_CHAT_MODEL_DEFAULT = "gemma3:12b"


def load_dataset_jsonl(path: str) -> List[Dict[str, Any]]:
	items = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
				items.append(obj)
			except json.JSONDecodeError:
				continue
	return items


def read_conversation_file(path: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def get_last_assistant_turn(turns: List[Dict[str, Any]]) -> str:
	# Find the last message with role == "assistant"
	for msg in reversed(turns):
		if msg.get("role") == "assistant":
			return str(msg.get("content", "")).strip()
	return ""


def sum_context_tokens(turns: List[Dict[str, Any]]) -> int:
	total = 0
	for msg in turns or []:
		val = msg.get("context_tokens")
		if isinstance(val, (int, float)):
			try:
				total += int(val)
			except Exception:
				continue
	return total


def build_prompt(assistant_answer: str, checkpoint_question: str) -> str:
	system_prompt = (
		"You are a strict evaluator. The user provides (1) the assistant's final reply in braces and "
		"(2) a checkpoint question that may contain multiple atomic goals (e.g., 'replacement product (in blue)', "
		"'all shipping fees waived', 'no request for VIP service upgrade'). Split the checkpoint into minimal, independent "
		"goals and check each goal against ONLY the assistant's reply. Count how many goals are clearly satisfied. "
		"Output two integers in the exact format 'x/y' with no extra text, where x = number of satisfied goals and y = total goals. "
		"If something is unclear or unspecified in the reply, consider that goal NOT satisfied."
	)
	# Wrap assistant answer in braces and append checkpoint question
	return (
		f"System: {system_prompt}\n"
		f"Assistant final reply: {{{assistant_answer}}}\n"
		f"Checkpoint question: {checkpoint_question}"
	)


def call_llm_judge(judge: str, prompt: str, *, api_key: str = "", judge_model: Optional[str] = None, openai_base_url: Optional[str] = None, timeout: int = 30) -> str:
	history = [{"role": "user", "content": prompt}]
	j = (judge or "").strip().lower()

	if j == "openai":
		client = OpenAIChatClient(api_key=api_key or None, base_url=openai_base_url or None, model=judge_model or OPENAI_MODEL_DEFAULT, timeout=timeout)
		return client.chat(history)
	if j == "ollama":
		return ollama_chat(history, model=judge_model or OLLAMA_CHAT_MODEL_DEFAULT)
	client = OpenAIChatClient(api_key=api_key or None, base_url=openai_base_url or None, model=judge_model or OPENAI_MODEL_DEFAULT, timeout=timeout)
	return client.chat(history)


def parse_score_pair(text: str) -> Tuple[int, int]:
	"""Parse a reply like '2/3' or '2 3' into (score, max_score)."""
	if not text:
		return (0, 0)
	t = text.strip()
	if "/" in t:
		parts = t.split("/")
		if len(parts) >= 2:
			try:
				a = int("".join(ch for ch in parts[0] if ch.isdigit()))
				b = int("".join(ch for ch in parts[1] if ch.isdigit()))
				return (a, b)
			except Exception:
				pass
	# Fallback: extract first two integers
	nums = []
	cur = ""
	for ch in t:
		if ch.isdigit():
			cur += ch
		elif cur:
			nums.append(int(cur))
			cur = ""
		if len(nums) >= 2:
			break
	if cur and len(nums) < 2:
		nums.append(int(cur))
	if len(nums) >= 2:
		return (nums[0], nums[1])
	if len(nums) == 1:
		return (nums[0], 1)
	return (0, 0)


def collect_conversation_files(mode: str) -> Tuple[str, List[str]]:
	base = os.path.dirname(os.path.abspath(__file__))
	
	if mode == "direct":
		folder = os.path.join(base, "output", "direct")
	elif mode == "smart":
		folder = os.path.join(base, "output", "smart")
	elif mode == "rag":
		folder = os.path.join(base, "output_rag")
	elif mode == "output_rag":
		folder = os.path.join(base, "output_rag")
	else:
		if os.path.isabs(mode):
			folder = mode
		else:
			if ("/" in mode) or (os.sep in mode):
				folder = os.path.join(base, mode)
			else:
				folder = os.path.join(base, "output", "direct", mode)
	if not os.path.isdir(folder):
		raise FileNotFoundError(f"Output folder not found: {folder}")
	files = [
		os.path.join(folder, name)
		for name in os.listdir(folder)
		if name.endswith(".json")
	]
	return folder, files


def build_conv_index(files: List[str]) -> Dict[str, str]:
	index: Dict[str, str] = {}
	for path in files:
		try:
			data = read_conversation_file(path)
			cid = data.get("conversation_id")
			if not cid:
				# try parse from filename (smart/rag use <uuid>.json)
				cid = os.path.splitext(os.path.basename(path))[0]
				if cid.startswith("D-"):
					cid = cid[2:]
			if cid:
				index[cid] = path
		except Exception:
			continue
	return index


def _fmt_eta(seconds: float) -> str:
	if seconds < 0 or seconds == float("inf"):
		return "--:--:--"
	return str(timedelta(seconds=int(seconds)))


def _print_progress(done: int, total: int, start_ts: float) -> None:
	if total <= 0:
		return
	elapsed = max(0.0, time.time() - start_ts)
	rate = elapsed / done if done > 0 else 0.0
	remaining = (total - done) * rate if rate > 0 else 0.0
	pct = done / total
	bar_width = 30
	filled = int(bar_width * pct)
	bar = "#" * filled + "-" * (bar_width - filled)
	msg = f"[{bar}] {done}/{total} ({pct*100:5.1f}%) ETA: {_fmt_eta(remaining)}"
	sys.stdout.write("\r" + msg)
	sys.stdout.flush()


def evaluate(mode: str, judge: str, api_key: str, limit: Optional[int] = None, *, judge_model: Optional[str] = None, openai_base_url: Optional[str] = None) -> Dict[str, Any]:
	base = os.path.dirname(os.path.abspath(__file__))
	dataset_path = os.path.join(base, "input", "dataset-full.jsonl")
	ds = load_dataset_jsonl(dataset_path)

	folder, files = collect_conversation_files(mode)
	conv_index = build_conv_index(files)

	# Pre-compute items that have corresponding output conversations
	items_to_process: List[Dict[str, Any]] = [
		it for it in ds if str(it.get("conversation_id")) in conv_index
	]
	if limit is not None:
		items_to_process = items_to_process[: max(0, int(limit))]

	results: List[Dict[str, Any]] = []
	sum_raw = 0
	sum_max = 0
	total = len(items_to_process)

	print(f"Evaluating {total} conversations in folder: '{folder}' using judge='{judge}' model='{judge_model or ''}' ...")
	start_ts = time.time()
	done = 0
	if total > 0:
		_print_progress(done, total, start_ts)

	for item in items_to_process:
		cid = str(item.get("conversation_id"))
		cq = str(item.get("checkpoint_question", "")).strip()
		conv_path = conv_index.get(cid)
		if not conv_path:
			# Shouldn't happen due to pre-filtering, but keep safe
			continue

		try:
			conv = read_conversation_file(conv_path)
			turns = conv.get("turns", [])
			ctx_tokens_sum = sum_context_tokens(turns)
			ans = get_last_assistant_turn(turns)
			prompt = build_prompt(ans, cq)
			reply = call_llm_judge(judge, prompt, api_key=api_key, judge_model=judge_model, openai_base_url=openai_base_url)
			got, full = parse_score_pair(reply)
			percent = (got / full * 100.0) if full > 0 else 0.0
			results.append({
				"conversation_id": cid,
				"score": got,
				"max": full,
				"percent": percent,
				"context_tokens_sum": ctx_tokens_sum,
				"raw_reply": reply,
			})
			sum_raw += got
			sum_max += full
		except Exception as e:
			# Record failure as 0 and continue
			results.append({
				"conversation_id": cid,
				"score": 0,
				"max": 0,
				"percent": 0.0,
				"context_tokens_sum": 0,
				"error": str(e),
			})
		finally:
			done += 1
			_print_progress(done, total, start_ts)

	if total > 0:
		# End the progress line
		sys.stdout.write("\n")
		sys.stdout.flush()

	average_percent = (
		(sum((r.get("percent", 0.0) for r in results)) / total) if total > 0 else 0.0
	)
	return {
		"mode": mode,
		"folder_path": folder,
		"total": total,
		"sum_score": sum_raw,
		"sum_max": sum_max,
		"average_percent": average_percent,
		"judge": judge,
		"judge_model": judge_model or "",
		"results": results,
	}


def write_results(mode: str, data: Dict[str, Any]) -> str:
	base = os.path.dirname(os.path.abspath(__file__))
	out_dir = os.path.join(base, "evaluation_result")
	os.makedirs(out_dir, exist_ok=True)

	folder_path = data.get("folder_path")

	fname = f"{mode}_score.json"
	try:
		if folder_path:
			folder_abs = os.path.abspath(str(folder_path))
			label = os.path.basename(folder_abs.rstrip("/")) or "output"
			
			output_root = os.path.join(base, "output")
			direct_root = os.path.join(output_root, "direct")
			smart_root = os.path.join(output_root, "smart")
			rag_root = os.path.join(base, "output_rag")
			folder_norm = os.path.normpath(folder_abs)
			if folder_norm.startswith(os.path.normpath(direct_root) + os.sep):
				fname = f"{label}_direct_score.json"
			elif folder_norm.startswith(os.path.normpath(smart_root) + os.sep):
				fname = f"{label}_smart_score.json"
			elif folder_norm.startswith(os.path.normpath(rag_root) + os.sep):
				fname = f"{label}_rag_score.json"
			else:
			
				fname = f"{label}_score.json"
	except Exception:
		pass

	path = os.path.join(out_dir, fname)
	
	summary_only = {
		"mode": data.get("mode"),
		"folder_path": data.get("folder_path"),
		"total": data.get("total"),
		"sum_score": data.get("sum_score"),
		"sum_max": data.get("sum_max"),
		"average_percent": data.get("average_percent"),
		"judge": data.get("judge"),
		"judge_model": data.get("judge_model"),
	}
	with open(path, "w", encoding="utf-8") as f:
		json.dump(summary_only, f, ensure_ascii=False, indent=2)
	return path


def write_results_csv(mode: str, data: Dict[str, Any]) -> str:
	base = os.path.dirname(os.path.abspath(__file__))
	out_dir = os.path.join(base, "evaluation_result")
	os.makedirs(out_dir, exist_ok=True)

	folder_path = data.get("folder_path")

	fname = f"{mode}_scores.csv"
	try:
		if folder_path:
			folder_abs = os.path.abspath(str(folder_path))
			label = os.path.basename(folder_abs.rstrip("/")) or "output"
		
			output_root = os.path.join(base, "output")
			direct_root = os.path.join(output_root, "direct")
			smart_root = os.path.join(output_root, "smart")
			rag_root = os.path.join(base, "output_rag")
			folder_norm = os.path.normpath(folder_abs)
			if folder_norm.startswith(os.path.normpath(direct_root) + os.sep):
				fname = f"{label}_direct_scores.csv"
			elif folder_norm.startswith(os.path.normpath(smart_root) + os.sep):
				fname = f"{label}_smart_scores.csv"
			elif folder_norm.startswith(os.path.normpath(rag_root) + os.sep):
				fname = f"{label}_rag_scores.csv"
			else:
				
				fname = f"{label}_scores.csv"
	except Exception:
		pass

	path = os.path.join(out_dir, fname)
	rows = data.get("results") or []
	fieldnames = [
		"conversation_id",
		"score",
		"max",
		"percent",
		"context_tokens_sum",
		"raw_reply",
	]
	with open(path, "w", encoding="utf-8", newline="") as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for r in rows:
			writer.writerow({
				"conversation_id": r.get("conversation_id"),
				"score": r.get("score"),
				"max": r.get("max"),
				"percent": r.get("percent"),
				"context_tokens_sum": r.get("context_tokens_sum"),
				"raw_reply": r.get("raw_reply"),
			})
	return path



def main():
	parser = argparse.ArgumentParser(description="Evaluate conversations with selectable LLM judge (openai|ollama).")
	parser.add_argument(
		"--mode",
		default="direct",
		help=(
			"Folder to evaluate: 'direct'|'smart'|'rag', or a relative/absolute folder path containing .json conversations."
		),
	)
	parser.add_argument(
		"--api-key",
		default="",
		help=(
			"API key for the selected judge. For openai, will fallback to OPENAI_API_KEY env var."
		),
	)
	parser.add_argument(
		"--judge",
		default="openai",
		choices=["openai", "ollama"],
		help="Which LLM to use as the judge.",
	)
	parser.add_argument(
		"--judge-model",
		default=None,
		help="Underlying model name for the judge (e.g., gpt-3.5-turbo, gemma3:12b)",
	)
	parser.add_argument(
		"--openai-base-url",
		default=os.environ.get("OPENAI_BASE_URL", None),
		help="Optional custom base URL for OpenAI-compatible endpoints.",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Optionally limit the number of conversations to evaluate (for quick tests)",
	)
	args = parser.parse_args()

	judge = str(args.judge).strip().lower()
	api_key = str(args.api_key or "")
	if judge == "openai":
		api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
		if not api_key:
			raise SystemExit("Missing OpenAI API key. Provide via --api-key or OPENAI_API_KEY env var.")
	else:
		
		api_key = ""

	summary = evaluate(
		args.mode,
		judge,
		api_key,
		limit=args.limit,
		judge_model=args.judge_model,
		openai_base_url=args.openai_base_url,
	)
	csv_path = write_results_csv(args.mode, summary)
	json_path = write_results(args.mode, summary)
	print(f"Wrote CSV to: {csv_path}")
	print(f"Wrote summary JSON to: {json_path}")
	print(
		f"Mode={summary['mode']} Total={summary['total']} Average={summary['average_percent']:.3f}"
	)


if __name__ == "__main__":
	main()

