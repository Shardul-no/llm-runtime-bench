"""Summarize results.csv: mean/std table plus charts (PNG)."""
import argparse
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", default=["results.csv"])
ap.add_argument("--outdir", default=".")
args = ap.parse_args()

files = [f for p in args.inputs for f in glob.glob(p)]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df["label"] = df["backend"] + " | " + df["condition"]
is_sus = df["condition"].astype(str).str.startswith("sustained")
main_df = df[~is_sus]
sus_df = df[is_sus] if is_sus.any() else df

metrics = ["ttft_ms", "decode_tps", "tokens_per_joule", "mean_power_w", "max_temp_c",
           "mean_sm_clock_mhz", "peak_vram_gb"]
g = df.groupby(["label", "model", "target_ctx"])[metrics].agg(["mean", "std", "count"])
g.to_csv(f"{args.outdir}/summary.csv")
print(g[[(m, s) for m in ["ttft_ms", "decode_tps", "tokens_per_joule"] for s in ["mean", "std"]]].round(2))


def bar(metric, ylabel, fname):
    m = main_df.groupby(["target_ctx", "label"])[metric].mean().unstack()
    s = main_df.groupby(["target_ctx", "label"])[metric].std().unstack()
    ax = m.plot(kind="bar", yerr=s, capsize=3, figsize=(7, 4))
    ax.set_xlabel("Target prompt length (tokens)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/{fname}", dpi=200)
    plt.close()


bar("ttft_ms", "Time to first token (ms)", "chart_ttft.png")
bar("decode_tps", "Decode throughput (tokens/s)", "chart_tps.png")
bar("tokens_per_joule", "Energy efficiency (tokens/J)", "chart_energy.png")
bar("peak_vram_gb", "Peak GPU memory used (GB)", "chart_vram.png")

# Sustained-load drift: use a run with many consecutive requests (e.g. --runs 100 --cooldown-temp 0)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for label, d in sus_df.groupby("label"):
    d = d.sort_values("elapsed_s")
    axes[0].plot(d["elapsed_s"], d["decode_tps"], label=label, marker=".")
    axes[1].plot(d["elapsed_s"], d["max_temp_c"], label=label, marker=".")
    axes[2].plot(d["elapsed_s"], d["mean_sm_clock_mhz"], label=label, marker=".")
for ax, yl in zip(axes, ["Decode tokens/s", "Max GPU temp (C)", "Mean SM clock (MHz)"]):
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel(yl)
axes[0].legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{args.outdir}/chart_sustained.png", dpi=200)
print("\nSaved summary.csv and chart_*.png")
