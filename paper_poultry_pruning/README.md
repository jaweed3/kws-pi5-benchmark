# Paper 2: Pruning + INT8 Co-Design for Poultry Fecal Disease on Edge CPU

> Status: CONCEPT — Ide 1 dari 4 ide next-paper. Kamu fokus `paper_poultry_edge` (AMICT Sep 1) dulu. Folder ini siap eksekusi mulai 2 Sep.

## One-liner
Paper 1 menemukan INT8 PTQ di CPU malah 6-30x lebih lambat (dequantize overhead). Paper 2 menjawab: **structured channel pruning sebelum INT8 bisa mengembalikan speedup** — karena pruning mengurangi operasi yang sensitif kuantisasi (depthwise bottleneck) dan mengurangi total quantize/dequantize calls.

## Posisi vs Paper 1
| Paper 1 (AMICT Sep 1) | Paper 2 (MLCIPR Oct 5 / MCSoC Dec) |
|---|---|
| Karakterisasi: 3 arsitektur + INT8 + Pareto + Grad-CAM | Optimasi: pruning 30/50% + INT8 + recovery Pareto |
| Temuan negatif: INT8 lambat di CPU | Solusi: pruning+INT8 co-design, latency recovery 3-8x |
| CPU Intel i5 | CPU Intel + Pi 5 Cortex-A76 (dua hardware point) |

Unique angle: **Pertama yang mengukur interaksi pruning x quantization untuk poultry fecal domain di edge CPU.** Literatur pruning (MobileNetV2 catfish 2025, HAPE TODAES 2025, MOEA pruning 2024) tidak pernah ukur efeknya ke INT8 latency di CPU tanpa VNNI.

## Struktur folder
```
paper_poultry_pruning/
  README.md              # ini
  outline.md             # outline paper lengkap (target 4-6 hal IEEE)
  plan.md                # step-by-step eksekusi bite-sized
  paper/                 # LaTeX IEEEtran (nanti: main.tex, refs.bib, figures)
  code/                  # pruning_pipeline.py (extend dari ../paper_poultry_edge/train_pipeline.py)
  results/               # JSON latency/size/accuracy per (model, sparsity, quant)
  figures/               # Pareto baru, pruning-vs-accuracy, latency recovery
```

## Dataset & Model reuse
- Dataset: Machuve 8,770 gambar (5,614 train / 1,402 val / 1,754 test) — sama persis Paper 1, tidak perlu cari data baru.
- Model: 3 checkpoint FP32 >97% dari Paper 1 (MobileNetV2 2.2M, ShuffleNetV2 1.26M, EfficientNet-B0 4.0M) — reuse, tinggal prune + fine-tune 10 epoch.

## Venue target
- **Primary: MLCIPR 2026 Nanjing** — IEEE, Scopus, deadline ~5 Okt 2026 (verifikasi di iccipr.org / wikicfp 2 minggu sebelum). 4-6 hal.
- **Fallback: MCSoC 2026 Shanghai** — 14-17 Des 2026, Round 2 buka ~10 Agu (cek mcsoc-forum.org). 4-6 hal.
- **ArXiv:** boleh upload setelah submit (jangan sebelum submit kalau venue double-blind).

## Timeline (mulai 2 Sep, setelah AMICT submit)
- 2-4 Sep: Setup pruning pipeline + smoke test 30% di Mac (10 iter)
- 5-7 Sep: Full prune 30% & 50% untuk 3 model + fine-tune 10 epoch di RTX 4060 (ingat kill LM Studio!)
- 8-10 Sep: ONNX export → INT8 PTQ → benchmark 200 runs di CPU + Pi 5
- 11-14 Sep: Figures + tulis Results/Discussion
- 15-20 Sep: Intro/Related Work/Method + proofread anti AI slop
- 21-25 Sep: Final compile + submit MLCIPR

Total effort solo: ~3 minggu, 2 jam/hari (rotasi Paper Senin/Rabu/Jumat).

## Aturan eksekusi
- Jangan mulai sebelum `paper_poultry_edge` submit 1 Sep. Freeze folder ini sampai itu.
- Satu eksperimen = satu JSON di results/ dengan seed tetap (42) + raw latency array (bukan cuma mean).
- Semua klaim di paper harus ada angka dari results/ — no hand-waving.
