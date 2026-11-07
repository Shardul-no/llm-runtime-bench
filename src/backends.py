"""Streaming clients for llama.cpp's llama-server and Ollama. Timing is measured client-side."""
import json
import time
import uuid

import requests

FILLER = (
    "The history of computing spans mechanical calculators, vacuum tube machines, transistors, "
    "integrated circuits and modern parallel accelerators. Engineers balanced speed, power and "
    "cost while software grew from machine code to high level languages and distributed systems. "
)


def build_prompt(target_tokens):
    """Roughly target_tokens long (~0.75 words/token). Actual counts come from the server.
    A unique nonce at the start defeats prompt caching between runs."""
    nonce = f"[run {uuid.uuid4().hex[:8]}] "
    tail = "\n\nWrite a long, detailed essay continuing this topic."
    words_needed = max(int(target_tokens * 0.75) - 20, 5)
    words = (FILLER * (words_needed // 30 + 2)).split()[:words_needed]
    return nonce + " ".join(words) + tail


def _lines(resp):
    for line in resp.iter_lines():
        if not line:
            continue
        if line.startswith(b"data: "):
            line = line[6:]
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def run_llamacpp(url, prompt, max_new_tokens, **_):
    payload = {"prompt": prompt, "n_predict": max_new_tokens, "temperature": 0.0,
               "top_k": 1, "seed": 0, "stream": True, "cache_prompt": False, "ignore_eos": True}
    t_start = time.perf_counter()
    t_first, final = None, {}
    with requests.post(f"{url}/completion", json=payload, stream=True, timeout=900) as r:
        r.raise_for_status()
        for d in _lines(r):
            if d.get("content") and t_first is None:
                t_first = time.perf_counter()
            if d.get("stop"):
                final = d
    t_end = time.perf_counter()
    return t_start, t_first, t_end, final.get("tokens_evaluated", 0), final.get("tokens_predicted", 0)


def run_ollama(url, prompt, max_new_tokens, model="llama3.2:3b", num_ctx=4096, **_):
    payload = {"model": model, "prompt": prompt, "stream": True, "keep_alive": "30m",
               "options": {"temperature": 0.0, "seed": 0, "num_predict": max_new_tokens, "num_ctx": num_ctx}}
    t_start = time.perf_counter()
    t_first, final = None, {}
    with requests.post(f"{url}/api/generate", json=payload, stream=True, timeout=900) as r:
        r.raise_for_status()
        for d in _lines(r):
            if d.get("response") and t_first is None:
                t_first = time.perf_counter()
            if d.get("done"):
                final = d
    t_end = time.perf_counter()
    return t_start, t_first, t_end, final.get("prompt_eval_count", 0), final.get("eval_count", 0)


BACKENDS = {
    "llamacpp": (run_llamacpp, "http://127.0.0.1:8080"),
    "ollama": (run_ollama, "http://127.0.0.1:11434"),
}
