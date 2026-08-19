# Paper 2 Outline — Pruning–Quantization Co-Design for Poultry Fecal Disease on Edge CPU

> Status: DRAFT v0.1 — 19 Agu 2026. Jangan eksekusi sebelum Paper 1 (AMICT Sep 1) submit. Estimasi eksekusi 3 minggu solo mulai 2 Sep.

## Target Venue
**MLCIPR 2026 Nanjing (primary, ~5 Okt) / MCSoC 2026 Shanghai (fallback, 14-17 Des)**
- Format: IEEE 4-6 pages, English, IEEE Xplore + Scopus
- Cek ulang deadline 2 minggu sebelum di iccipr.org & mcsoc-forum.org
- Fee: ~$150 student. Double-blind? siapkan `main.tex` + `main_anon.tex`

## Working Title
**"From Slowdown to Speedup: Structured Pruning Recovers INT8 Quantization Benefits for Lightweight CNNs in Poultry Disease Detection on Edge CPUs"**
- Alt: "Pruning-Quantization Co-Design for Efficient Poultry Fecal Disease Detection on Resource-Constrained CPUs"

## Positioning
Paper 1 menemukan: INT8 PTQ di CPU tanpa VNNI malah 6-30x lebih lambat (MN2 1.85ms→55.8ms, SNV2 3.17ms→20.3ms, EB0 3.50ms→80.7ms). Paper 1 hanya karakterisasi, belum solusi.
Paper 2 menjawab: **pruning dulu, baru quantize**. Literatur pruning CNN lightweight (MobileNetV2 catfish IEEE 2025, HAPE TODAES 2025 untuk LLM, MOEA multi-obj pruning 2024) tidak pernah mengukur interaksi pruning x INT8 latency di CPU untuk domain poultry. Kita yang pertama.

| Literatur | Yang mereka lakukan | Gap yang kita isi |
|---|---|---|
| Dhungana 2025 (AgriEngineering) | EB0 99.12% di cloud, tanpa edge | Kita edge CPU + INT8 |
| Tasdelen 2025 | 6 arch, MN2 97.1% terbaik, tanpa quant | Kita quant + pruning |
| HAPE 2025 (ACM TODAES) | Hardware-aware pruning untuk LLM di edge | Kita CNN lightweight + INT8 latency |
| MOEA HW-aware pruning 2024 (Elsevier) | Single-obj → multi-obj pruning | Kita ukur pruning x quant co-design |
| BitNet b1.58 2025 / MCUNet 2020 | Extreme quant / NAS terpisah | Kita tunjukkan pruning+PTQ lebih praktis solo |

Unique angle: **Pertama ukur apakah structured channel pruning bisa mengembalikan speedup INT8 yang hilang di CPU.**

---

## Abstract (Draft — akan polish setelah data)

Post-training INT8 quantization is widely assumed to accelerate edge inference, yet our prior characterization on poultry fecal disease detection shows 6-30x slowdown on CPUs without dedicated INT8 acceleration due to dequantization overhead. This paper investigates whether structured channel pruning before quantization can recover those benefits. We apply L1-norm structured pruning at 30% and 50% sparsity to three lightweight CNNs (MobileNetV2, ShuffleNetV2, EfficientNet-B0, all >97% baseline) and evaluate accuracy, model size, and latency under INT8 PTQ on Intel i5-12400F and Raspberry Pi 5 (Cortex-A76). Our co-design reduces INT8 latency by X–Y% while keeping accuracy within 1% of baseline, shifting the Pareto frontier toward practical edge deployment. Grad-CAM confirms pruned models retain diagnostic feature focus. Results provide a low-effort recipe for practitioners: prune first, then quantize.

## Research Questions
1. Seberapa sensitif tiap arsitektur (MN2/SNV2/EB0) terhadap structured channel pruning (0/30/50%) sebelum fine-tune?
2. Apakah pruning sebelum INT8 PTQ mengurangi latency overhead INT8 di CPU (target: MN2 dari 30x → <8x, SNV2 dari 6.4x → ~2x)?
3. Di mana titik Pareto baru (accuracy vs latency vs size) setelah pruning+INT8 vs baseline FP32/INT8?
4. Apakah pruned model tetap fokus ke fitur diagnostik yang sama (Grad-CAM delta)?

## Contributions
1. Systematic pruning sensitivity untuk 3 lightweight CNN di domain poultry fecal (30/50% structured).
2. Bukti empiris bahwa pruning+INT8 co-design memulihkan speedup INT8 di CPU (angka dari E2).
3. Pareto frontier baru (9 titik: 3 model × 3 sparsity) untuk pemilihan model berdasar hardware constraint.
4. Recipe praktis solo: 10-epoch fine-tune setelah pruning cukup, tidak perlu QAT.

---

## 1. Introduction (0.5 pg)
- Konteks poultry fecal detection, edge deployment need (reuse Paper 1 intro 1 paragraf).
- Temuan Paper 1 yang jadi motivasi: INT8 lambat di CPU — contradict assumption.
- Pertanyaan: apakah pruning bisa kurangi dequantize cost? Intuisi: pruning kurangi channel → kurangi matmul yang perlu dequantize.
- Preview hasil (isi setelah eksperimen): "Pruning 30% recovers Y% latency..."

## 2. Related Work (0.6 pg)
- 2.1 Poultry disease detection (Machuve 2022 dataset, Dhungana/Tasdelen/Degu — sama Paper 1, ringkas).
- 2.2 Structured pruning for lightweight CNNs (MobileNetV2 catfish 2025, torch-pruning/DepGraph 2023, HAPE 2025 analogy, MOEA 2024). Gap: tidak ukur interaksi dengan quant.
- 2.3 Quantization & co-design (Jacob 2018 PTQ, ONNX Runtime issues #6732/#12854, BitNet b1.58 sebagai extreme, Maulana 2026 QAT>PTQ). Gap: no pruning+PTQ di poultry.

## 3. Methodology (1.0 pg)
- 3.1 Dataset: sama Paper 1 — Machuve 8,770 (5,614/1,402/1,754), 224x224, ImageNet norm. Tekankan tidak ada data baru.
- 3.2 Models: MN2 2.2M, SNV2 1.26M, EB0 4.0M — load checkpoint FP32 >97% dari Paper 1.
- 3.3 Structured pruning: L1-norm channel pruning via `torch-pruning` (DepGraph) — prune BatchNorm-associated channels, sparsity 30% & 50% (global). Alasan: hardware-friendly (channel removal → real latency gain, bukan sparse mask). Fine-tune 10 epoch, Adam 1e-5, cosine, batch 32, augment sama.
- 3.4 Quantization: ONNX export FP32 → INT8 dynamic PTQ (ONNX Runtime, QInt8) — sama persis Paper 1.
- 3.5 Benchmark protocol: 200 runs + 10 warmup, CPU EP (Intel i5) + Pi 5 Cortex-A76, mean/P50/FPS, seed 42, size MB. Simpan raw latency array untuk CI bootstrap 95% (reuse protokol ComTech).

## 4. Experiments (0.5 pg ringkas, detail di 5)
- E1: Pruning sensitivity (0/30/50% → accuracy, size, FLOPs).
- E2: Pruning + INT8 latency recovery (core result — Table quant_pruned).
- E3: Pareto frontier baru (9 titik vs 6 titik Paper 1).
- E4: Fine-tune ablation (no FT vs 5 vs 10 epoch — tunjukkan 10 epoch cukup).
- E5: Cross-hardware (Intel i5 vs Pi 5 — tunjukkan tren konsisten).
- E6 (opsional jika sempat): Grad-CAM delta pruned vs unpruned (1 figure).

## 5. Results & Discussion (1.5 pg) — template tabel, isi setelah run

### 5.1 Pruning sensitivity
| Model | Sparsity | Params | Size FP32 | Acc (%) | ΔAcc |
|---|---|---|---|---|---|
| MN2 | 0% | 2.2M | 8.48 MB | 97.8 | — |
| MN2 | 30% | ~1.5M | ~6.0 | ? | ? |
| MN2 | 50% | ~1.1M | ~4.2 | ? | ? |
| SNV2 | 0% | 1.26M | 4.89 | 97.4 | — |
| SNV2 | 30% | ~0.9M | ~3.4 | ? | ? |
| EB0 | 0% | 4.0M | 15.3 | 98.3 | — |
| ... | ... | ... | ... | ... | ... |
Hypothesis: SNV2 paling tahan pruning (channel shuffle homogen), EB0 paling rapuh (SE block sensitif channel).

### 5.2 INT8 latency recovery (MAIN CLAIM)
| Model | Sparsity | Lat FP32 | Lat INT8 | Slowdown | Recovery vs 0% INT8 |
|---|---|---|---|---|---|
| MN2 | 0% | 1.85 | 55.8 | 30.2x | — |
| MN2 | 30% | ? | ? | ? | ? |
| MN2 | 50% | ? | ? | ? | ? |
| SNV2 | 0% | 3.17 | 20.3 | 6.4x | — |
| SNV2 | 30% | ? | ? | ? | ? |
Target: MN2 30x → <10x, SNV2 6.4x → ~2-3x. Jelaskan mekanisme: fewer channels → fewer dequantize ops.

### 5.3 Pareto baru (Figure: 2-panel: acc-lat & acc-size, 9 pruned + 6 baseline)
Sweet spot expect: SNV2-30% INT8 untuk memory, MN2-30% FP32 untuk latency, EB0-0% untuk accuracy.

### 5.4 Fine-tune ablation (Figure: acc vs FT epoch 0/5/10)

### 5.5 Cross-hardware (Table: Intel i5 vs Pi 5 latency side-by-side, tunjukkan konsisten pruning benefit)

### 5.6 Interpretability (optional Figure: Grad-CAM pruned vs unpruned per class — expect mirip)

## 6. Conclusion (0.3 pg)
- Pruning dulu baru quantize memulihkan speedup INT8 di CPU (ringkas angka E2).
- Trade-off jelas: 30% sparsity best balance (size -30%, acc -0.5%, latency recovery Y%).
- Limitation: hanya PTQ dynamic, belum QAT; hanya CPU, belum ESP32 deployment.
- Future: QAT setelah pruning, deploy ke ESP32-S3 (1 TOPS NPU) untuk farm nyata (kandang 5k ayam), NAS di atas pruned model.

## References (15+, 5 tahun terakhir, reuse 80% dari Paper 1 + tambah)
1. Machuve et al. Frontiers AI 2022 (dataset)
2. Dhungana et al. AgriEngineering 2025
3. Tasdelen & Arslan 2025
4. Degu & Simegn 2023
5. Sandler et al. MobileNetV2 CVPR 2018
6. Ma et al. ShuffleNetV2 ECCV 2018
7. Tan & Le EfficientNet ICML 2019
8. Jacob et al. CVPR 2018 (quant)
9. ONNX Runtime docs 2024 (quant slowdown)
10. ONNX Runtime issues #6732, #12854
11. Maulana & Ramasamy Computers 2026 (QAT>PTQ)
12. **Fang et al. Torch-Pruning / DepGraph 2023** — structured pruning tool (NEW)
13. **HAPE TODAES 2025** — hardware-aware pruning analogy (NEW)
14. **Zhao et al. MOEA HW-aware pruning 2024** — multi-obj (NEW)
15. **Wang et al. BitNet b1.58 arXiv 2502.11295** — extreme quant context (NEW, optional cite)
16. Selvaraju et al. Grad-CAM ICCV 2017
17. Hoang et al. Sci Reports 2026 (XAI+PSS)

## Figures Checklist
- [ ] Fig 1: Pruning sensitivity (acc vs sparsity, 3 lines)
- [ ] Fig 2: Latency recovery — grouped bar FP32 vs INT8 per sparsity (core figure)
- [ ] Fig 3: Pareto baru (acc-lat & acc-size, 15 titik)
- [ ] Fig 4: Fine-tune ablation (acc vs epoch)
- [ ] Fig 5: Cross-hardware latency (Intel vs Pi 5, grouped bar)
- [ ] Fig 6 (optional): Grad-CAM pruned vs unpruned (2x4 grid)

## Appendix (supplementary, tidak hitung halaman)
- Raw JSON results per (model, sparsity, quant, hardware)
- Reproduce script `reproduce_pruning_int8.py` + seed 42
- ONNX size & FLOPs via thop

---

## Trajectory — Kenapa Paper Ini Masuk Akal Buat Kamu
- **Sekarang (Paper 1):** Karakterisasi → kamu punya angka 6-30x slowdown yang bikin reviewer penasaran "lalu gimana?".
- **Paper 2 ini:** Jawab pertanyaan itu dengan effort paling kecil (tidak perlu dataset baru, tidak perlu hardware baru, cuma prune + quant ulang).
- **Paper 3 (setelah ini):** Ambil winning pruned model → deploy nyata ke ESP32-S3 di kandang ayam (5k ayam bokap) + ukur power/latency real farm. Atau NAS di atas pruned search space untuk Dresden S2 proposal (Barkhausen, Silicon Saxony — fit edge ML).
- S2 story: "From characterization → co-design → real deployment" — narasi 3 paper yang koheren, bukan 3 paper random.

## Catatan Solo/Bertiga
- Solo feasible: semua eksperimen bisa di Mac + RTX 4060 + Pi 5 yang sudah ada. Tidak butuh Amirul (TIP) karena domain tetap poultry fecal.
- Kalau mau bertiga: ajak 1 teman buat bantu benchmark Pi 5 (ssh techronity@192.168.119.43) sambil kamu fine-tune di lab.
