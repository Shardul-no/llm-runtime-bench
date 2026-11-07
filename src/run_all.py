#!/usr/bin/env python3
"""One command to run the whole experiment and bundle everything to send back.

Example (Windows or Linux):
  python run_all.py --gguf C:\\models\\Llama-3.2-3B-Instruct-Q4_K_M.gguf --llama-server C:\\llama.cpp\\llama-server.exe

Quick smoke test first (about 3 minutes):
  python run_all.py --gguf <path> --llama-server <path> --quick
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

import requests

HERE = Path(__file__).parent.resolve()
LLAMA_URL = "http://127.0.0.1:8080"
OLLAMA_URL = "http://127.0.0.1:11434"
LOGF = None


def log(msg=""):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode(), flush=True)
    if LOGF:
        LOGF.write(msg + "\n")
        LOGF.flush()


def sh(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        return ((r.stdout or "") + (r.stderr or "")).strip()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def stream_run(cmd):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env=env)
    for line in p.stdout:
        log(line.rstrip())
    if p.wait() != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")


def wait_http(url, timeout, proc=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            if requests.get(url, timeout=3).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def battery_state():
    try:
        import psutil
        b = psutil.sensors_battery()
        return None if b is None else {"plugged_in": b.power_plugged, "percent": b.percent}
    except Exception:  # noqa: BLE001
        return None


def measure_idle(seconds=10):
    """Idle baseline (nothing loaded): power, temperature, VRAM."""
    try:
        from gpu_monitor import GPUMonitor
        m = GPUMonitor(0)
        m.start()
        t0 = time.perf_counter()
        time.sleep(seconds)
        t1 = time.perf_counter()
        m.stop()
        st = m.summarize(t0, t1)
        m.close()
        return {"idle_power_w": round(st["mean_power_w"], 1), "idle_temp_c": st["max_temp_c"],
                "idle_vram_gb": round(st["peak_vram_gb"], 2)}
    except Exception as e:  # noqa: BLE001
        log(f"(idle baseline failed: {e})")
        return {}


def collect_sysinfo(args):
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor(),
        "nvidia_smi": sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit,"
                          "power.max_limit,clocks.max.sm", "--format=csv"]),
        "nvidia_smi_full": sh(["nvidia-smi"]),  # includes the Pwr:Usage/Cap column
        "nvidia_smi_power": sh(["nvidia-smi", "-q", "-d", "POWER"]),
        "llama_server_version": sh([args.llama_server, "--version"]),
        "ollama_version": sh(["ollama", "--version"]),
        "gguf_file": Path(args.gguf).name,
        "gguf_size_gb": round(Path(args.gguf).stat().st_size / 1024 ** 3, 2),
        "ctx_window": args.num_ctx,
        "battery_at_start": battery_state(),
        "args": {k: str(v) for k, v in vars(args).items()},
    }


# ---------------- servers ----------------
def start_llama(args, outdir):
    logf = open(outdir / "llama_server.log", "w", encoding="utf-8", errors="replace")
    cmd = [args.llama_server, "-m", args.gguf, "-ngl", "99", "-c", str(args.num_ctx), "--port", "8080"]
    log(f"Starting llama-server: {' '.join(cmd)}")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    if not wait_http(LLAMA_URL + "/health", 300, p):
        p.terminate()
        raise RuntimeError("llama-server did not become ready. See llama_server.log in the output folder.")
    return p


def stop_proc(p):
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()
    time.sleep(3)  # let VRAM free up


def ensure_ollama(args, outdir):
    started = None
    if not wait_http(OLLAMA_URL + "/api/tags", 3):
        log("Starting `ollama serve`...")
        started = subprocess.Popen(["ollama", "serve"], stdout=open(outdir / "ollama_serve.log", "w", encoding="utf-8", errors="replace"),
                                   stderr=subprocess.STDOUT)
        if not wait_http(OLLAMA_URL + "/api/tags", 60, started):
            raise RuntimeError("Could not start Ollama.")
    mf = outdir / "Modelfile"
    mf.write_text(f"FROM {Path(args.gguf).resolve().as_posix()}\n", encoding="utf-8")
    r = subprocess.run(["ollama", "create", args.ollama_model, "-f", str(mf)], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    log((r.stdout or "") + (r.stderr or ""))
    if r.returncode != 0:
        raise RuntimeError("`ollama create` failed (see output above).")
    return started


def unload_ollama(model):
    try:
        requests.post(OLLAMA_URL + "/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
    except requests.RequestException:
        pass
    time.sleep(3)


def bench(backend, model, condition, args, outdir, ctx, runs, warmup, cooldown):
    cmd = [sys.executable, str(HERE / "run_benchmark.py"), "--backend", backend, "--model", model,
           "--condition", condition, "--num-ctx", str(args.num_ctx), "--out", str(outdir / "results.csv"),
           "--ctx", *map(str, ctx), "--runs", str(runs), "--warmup", str(warmup),
           "--cooldown-temp", str(cooldown)]
    stream_run(cmd)


def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="path to the Q4_K_M GGUF file")
    ap.add_argument("--llama-server", default="llama-server", help="path to llama-server (or on PATH)")
    ap.add_argument("--ollama-model", default="bench-llama32-3b-q4km")
    ap.add_argument("--conditions", nargs="+", default=["plugged_default", "battery"],
                    help="you will be asked to set up each one (e.g. plugged_default battery)")
    ap.add_argument("--backends", nargs="+", default=["llamacpp", "ollama"], choices=["llamacpp", "ollama"])
    ap.add_argument("--ctx", type=int, nargs="+", default=[128, 512, 2048])
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--sustained-runs", type=int, default=60)
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--cooldown-temp", type=int, default=None,
                    help="wait for GPU <= this C before each block (default: idle temp + 6)")
    ap.add_argument("--quick", action="store_true", help="tiny smoke test to check the setup works")
    args = ap.parse_args()

    if args.quick:
        args.runs, args.warmup, args.ctx, args.conditions = 2, 1, [128], args.conditions[:1]
    outdir = Path(args.out_dir or (HERE / "bench_out" / datetime.now().strftime("%Y%m%d_%H%M%S")))
    outdir.mkdir(parents=True, exist_ok=True)
    LOGF = open(outdir / "run_log.txt", "a", encoding="utf-8", errors="replace")

    if not Path(args.gguf).exists():
        sys.exit(f"GGUF not found: {args.gguf}")

    log(f"Output folder: {outdir}")
    sysinfo = collect_sysinfo(args)
    sysinfo["conditions"] = []
    log("Measuring idle baseline (10 s, keep the GPU idle)...")
    sysinfo["idle_baseline"] = measure_idle()
    log(f"Idle baseline: {sysinfo['idle_baseline']}")
    if args.cooldown_temp is None:
        idle_t = sysinfo["idle_baseline"].get("idle_temp_c", 50)
        args.cooldown_temp = int(idle_t) + 6
    log(f"Cooldown target before each block: {args.cooldown_temp} C")
    sysinfo["args"]["cooldown_temp"] = str(args.cooldown_temp)
    (outdir / "sysinfo.json").write_text(json.dumps(sysinfo, indent=2), encoding="utf-8")
    log(json.dumps({k: sysinfo[k] for k in ["os", "nvidia_smi", "llama_server_version", "ollama_version"]}, indent=2))

    if not args.quick:
        log("\nRough time estimate: 45-90 minutes. Keep the laptop awake, close other apps, "
            "and don't use the GPU meanwhile.")

    ollama_started = None
    try:
        if "ollama" in args.backends:
            ollama_started = ensure_ollama(args, outdir)
            unload_ollama(args.ollama_model)

        for ci, cond in enumerate(args.conditions):
            if cond.startswith("battery"):
                msg = "UNPLUG the charger now (battery mode)."
            elif ci == 0:
                msg = "Plug in the charger and set your normal/performance power mode."
            else:
                msg = f"Set up the laptop for condition '{cond}' (power mode / plug state)."
            input(f"\n[{cond}] {msg}\nPress Enter when ready... ")
            bs = battery_state()
            sysinfo["conditions"].append({"condition": cond, "battery": bs,
                                          "time": datetime.now().isoformat(timespec="seconds")})
            (outdir / "sysinfo.json").write_text(json.dumps(sysinfo, indent=2), encoding="utf-8")
            if bs and cond.startswith("battery") and bs["plugged_in"]:
                log("WARNING: condition is 'battery' but the charger appears to be plugged in.")

            for backend in args.backends:
                log(f"\n########## {backend} | {cond} ##########")
                proc = None
                try:
                    if backend == "llamacpp":
                        proc = start_llama(args, outdir)
                        model = Path(args.gguf).stem
                    else:
                        model = args.ollama_model
                    bench(backend, model, cond, args, outdir, args.ctx, args.runs, args.warmup, args.cooldown_temp)
                    if ci == 0 and not args.quick:
                        log(f"\n--- sustained load test ({args.sustained_runs} back-to-back runs) ---")
                        bench(backend, model, f"sustained_{cond}", args, outdir, [512],
                              args.sustained_runs, 2, 0)
                finally:
                    if backend == "llamacpp":
                        stop_proc(proc)
                    else:
                        unload_ollama(args.ollama_model)
    finally:
        if ollama_started:
            stop_proc(ollama_started)

    log("\nAnalyzing...")
    stream_run([sys.executable, str(HERE / "analyze.py"), "--inputs", str(outdir / "results.csv"),
                "--outdir", str(outdir)])

    bundle = outdir / "bench_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["results.csv", "summary.csv", "sysinfo.json", "run_log.txt", "Modelfile"]:
            if (outdir / f).exists():
                z.write(outdir / f, f)
        for f in outdir.glob("chart_*.png"):
            z.write(f, f.name)

if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if LOGF:
            LOGF.write("\nCRASH:\n" + traceback.format_exc())
            LOGF.flush()
        raise
