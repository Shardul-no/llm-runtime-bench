# Local LLM Runtime Benchmark: llama.cpp vs Ollama on an 8 GB Laptop GPU

Scripts and raw data for the paper *Power State Matters More Than Runtime Choice: Measuring llama.cpp and Ollama on an RTX 4060 Laptop GPU* ([link to preprint / DOI to be added]).

The harness compares `llama-server` (llama.cpp) and Ollama serving the **same GGUF weights**, and records time to first token (TTFT), decode throughput, GPU board energy, temperature, clocks and VRAM, both plugged in and on battery, plus a sustained-load test.

## Main findings

- Plugged in, decode throughput differed by at most 0.6% between runtimes for both models.
- llama.cpp had lower TTFT in every cell, by 9 to 73 ms.
- Unplugging the charger cut decode throughput by 45 to 51% (3B) and 36 to 38% (8B).
- Results are from one machine and one session per model; see the paper's limitations.

## Hardware and software used

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB |
| Driver / CUDA | 592.00 / CUDA 13.1 |
| OS | Windows 11 (build 10.0.26200) |
| CPU | [add CPU model] |
| llama.cpp | `llama-server` build 12184, commit c0ab6e5c0, [add build variant, e.g. CUDA] |
| Ollama | 0.40.15 |
| Python | 3.13.2 |

## Models (not included in this repo)

Download the GGUF files yourself; they are large and have their own licenses.

- Llama-3.2-3B-Instruct, Q4_K_M: [add Hugging Face repo and file name]
- Llama-3.1-8B-Instruct, Q4_K_M: [add Hugging Face repo and file name]

## Repository layout

```
src/
  run_all.py         # full pipeline: sysinfo, idle baseline, servers, benchmarks, analysis, zip
  run_benchmark.py   # one backend/condition: cooldown, warm-ups, timed runs
  gpu_monitor.py     # NVML sampler (100 ms): power, temp, clocks, VRAM, throttle flags
  backends.py        # streaming clients for llama-server and Ollama
  analyze.py         # summary.csv and charts
data/
  llama32_3b/        # results.csv, summary.csv, sysinfo.json, run_log.txt, Modelfile, charts
  llama31_8b/        # same files for the 8B session
docs/
  DATA_NOTES.md      # read before using the CSVs
```

## Setup

Requires an NVIDIA GPU with NVML, Python 3, a `llama-server` binary, and Ollama installed.

```
pip install -r requirements.txt
```

## Reproducing the runs

Full run (asks you to plug or unplug the charger for each condition):

```
python run_all.py --gguf <path-to-model.gguf> --llama-server <path-to-llama-server.exe> --ollama-model <your-ollama-tag> --backends llamacpp ollama --sustained-runs 240
```

Quick smoke test:

```
python run_all.py --gguf <path-to-model.gguf> --llama-server <path-to-llama-server.exe> --ollama-model <your-ollama-tag> --quick
```

Re-analyze an existing results file:

```
python analyze.py --inputs bench_out\<folder>\results.csv --outdir bench_out\<folder>
```

Use a distinct `--ollama-model` tag for each model so labels in the CSV are correct. Each run writes `results.csv` (one row per request), `summary.csv`, `sysinfo.json`, `run_log.txt` and charts.

## Protocol summary

- Prompt targets of about 128, 512 and 2048 tokens, context window 4096, greedy decoding, 256 new tokens.
- Per cell: 3 warm-up requests and 10 timed requests, after the GPU cools to idle temperature plus 6 C.
- Conditions: plugged in (default power settings), then battery. Sustained test (240 consecutive requests per runtime, no cooldown) plugged in only.
- The harness checks that the model is resident in VRAM and aborts otherwise.