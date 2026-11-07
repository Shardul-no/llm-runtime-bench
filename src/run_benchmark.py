"""Run the benchmark against a running llama-server or Ollama and append rows to a CSV.

Example:
  python run_benchmark.py --backend llamacpp --model llama3.2-3b-q4km --condition plugged_default
  python run_benchmark.py --backend ollama   --model llama3.2:3b      --condition plugged_default
"""
import argparse
import csv
import os
import time
from datetime import datetime

from backends import BACKENDS, build_prompt
from gpu_monitor import GPUMonitor

FIELDS = ["timestamp", "gpu", "power_limit_w", "backend", "model", "condition", "target_ctx",
          "run", "elapsed_s", "prompt_tokens", "gen_tokens", "ttft_ms", "decode_tps", "total_s",
          "energy_j", "tokens_per_joule", "mean_power_w", "max_temp_c", "mean_sm_clock_mhz",
          "vram_start_gb", "peak_vram_gb", "thermal_throttle", "power_throttle", "n_samples"]


def wait_for_cooldown(mon, target, timeout):
    if target <= 0:
        return
    t0 = time.time()
    while mon.temp() > target:
        if time.time() - t0 > timeout:
            print(f"  (cooldown timeout, GPU still {mon.temp()}C)")
            return
        print(f"  cooling: {mon.temp()}C -> {target}C")
        time.sleep(5)


def check_on_gpu(mon, min_vram_gb):
    """Abort if the model clearly isn't resident in VRAM (e.g. CPU-only llama.cpp build)."""
    if min_vram_gb <= 0:
        return
    used = mon._read()[4]
    print(f"  GPU residency check: {used:.2f} GB VRAM in use (need >= {min_vram_gb:.2f} GB)")
    if used < min_vram_gb:
        raise SystemExit(
            "\nERROR: the model does not appear to be loaded on the GPU (VRAM use is near idle).\n"
            "For llama.cpp this usually means a CPU-only build: download the Windows CUDA build, "
            "and check the server log for 'offloaded ... layers to GPU'.\n"
            "(Override with --min-vram-gb 0 only if you intentionally benchmark the CPU.)")


def one_request(fn, mon, url, ctx, args):
    prompt = build_prompt(ctx)
    mon.start()
    time.sleep(0.3)  # a few idle samples before the request
    t_start, t_first, t_end, n_prompt, n_gen = fn(
        url, prompt, args.max_new_tokens, model=args.model, num_ctx=args.num_ctx)
    time.sleep(0.3)
    mon.stop()
    if t_first is None or n_gen < 2:
        raise RuntimeError("No tokens received - is the server running and the model loaded?")
    stats = mon.summarize(t_start, t_end)
    decode_time = t_end - t_first
    row = {
        "prompt_tokens": n_prompt, "gen_tokens": n_gen,
        "ttft_ms": (t_first - t_start) * 1000,
        "decode_tps": (n_gen - 1) / decode_time if decode_time > 0 else float("nan"),
        "total_s": t_end - t_start,
    }
    row.update(stats)
    if stats.get("energy_j"):
        row["tokens_per_joule"] = n_gen / stats["energy_j"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=list(BACKENDS))
    ap.add_argument("--model", default="llama3.2:3b", help="Ollama model tag, or a label for llama.cpp")
    ap.add_argument("--url", default=None)
    ap.add_argument("--condition", default="plugged_default",
                    help="free-text label, e.g. plugged_default, battery, powerlimit_60w")
    ap.add_argument("--ctx", type=int, nargs="+", default=[128, 512, 2048])
    ap.add_argument("--runs", type=int, default=10, help="timed runs per context size")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--num-ctx", type=int, default=4096, help="Ollama context window (match llama-server -c)")
    ap.add_argument("--cooldown-temp", type=int, default=55, help="wait for GPU <= this C before each block; 0 disables")
    ap.add_argument("--cooldown-timeout", type=int, default=300)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--min-vram-gb", type=float, default=1.2,
                    help="abort if VRAM in use after warm-up is below this (0 disables)")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    fn, default_url = BACKENDS[args.backend]
    url = args.url or default_url
    mon = GPUMonitor(args.gpu)
    meta = {"gpu": mon.name(), "power_limit_w": mon.power_limit_w(), "backend": args.backend,
            "model": args.model, "condition": args.condition}
    print(f"GPU: {meta['gpu']}  power limit: {meta['power_limit_w']} W")

    new_file = not os.path.exists(args.out)
    t_origin = time.perf_counter()
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for ctx in args.ctx:
            print(f"\n=== target context {ctx} tokens ===")
            wait_for_cooldown(mon, args.cooldown_temp, args.cooldown_timeout)
            for i in range(args.warmup):
                print(f"  warmup {i + 1}/{args.warmup}")
                one_request(fn, mon, url, ctx, args)
            if args.warmup > 0:
                check_on_gpu(mon, args.min_vram_gb)
            for i in range(1, args.runs + 1):
                row = one_request(fn, mon, url, ctx, args)
                row.update(meta)
                row.update({"timestamp": datetime.now().isoformat(timespec="seconds"),
                            "target_ctx": ctx, "run": i,
                            "elapsed_s": time.perf_counter() - t_origin})
                w.writerow({k: row.get(k, "") for k in FIELDS})
                f.flush()
                if i == 1 and args.warmup == 0:
                    check_on_gpu(mon, args.min_vram_gb)
                print(f"  run {i}/{args.runs}: TTFT {row['ttft_ms']:.0f} ms | {row['decode_tps']:.1f} tok/s | "
                      f"{row.get('tokens_per_joule', float('nan')):.2f} tok/J | "
                      f"{row.get('max_temp_c', '?')}C | peak VRAM {row.get('peak_vram_gb', float('nan')):.2f} GB")
    mon.close()
    print(f"\nDone. Results appended to {args.out}")


if __name__ == "__main__":
    main()
