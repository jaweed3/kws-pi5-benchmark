#!/usr/bin/env python3
"""Generate latency figure for the KWS paper (Pi 5, benchmark results)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "text.usetex": False,
})

res = json.load(open("/Users/wedjaw/research/paper_audio_kws/bench/results/benchmark_kws_pi.json"))
mixed = json.load(open("/Users/wedjaw/research/paper_audio_kws/bench/results/benchmark_mixed_pi.json"))

rows = [
    ("TFLite\n(fp32)", res["tflite"]["mean_ms"], res["tflite"]["ci95_ms"], res["tflite"]["fps"]),
    ("TFLite\n(int8 convs)", mixed["tflite"]["mean_ms"], mixed["tflite"]["ci95_ms"], mixed["tflite"]["fps"]),
    ("ONNX Runtime\n(fp32)", res["onnx"]["mean_ms"], res["onnx"]["ci95_ms"], res["onnx"]["fps"]),
]

labels = [r[0] for r in rows]
means = [r[1] for r in rows]
cis = [r[2] for r in rows]
fps = [r[3] for r in rows]

x = np.arange(len(rows))
err_lo = np.array([m - c[0] for m, c in zip(means, cis)])
err_hi = np.array([c[1] - m for m, c in zip(means, cis)])

fig, ax = plt.subplots(figsize=(3.4, 2.2), dpi=300)
bars = ax.bar(x, means, width=0.55, color=["#4C72B0", "#55A868", "#C44E52"],
              edgecolor="black", linewidth=0.4, yerr=[err_lo, err_hi],
              error_kw=dict(elinewidth=0.8, capsize=3, capthick=0.8))
ax.set_ylabel("Mean latency (ms)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 0.28)
ax.tick_params(axis="both", labelsize=8)
for xi, m, f in zip(x, means, fps):
    ax.text(xi, m + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(xi, m - 0.018, f"{f:,.0f} FPS", ha="center", va="top", fontsize=7,
            color="white", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("/Users/wedjaw/research/paper_audio_kws/paper/figures/latency.pdf")
print("figure saved")
