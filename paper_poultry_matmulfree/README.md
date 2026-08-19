# Paper 3 Concept — MatMul-Free CNN for Poultry Fecal Disease on ESP32-S3

> Status: CONCEPT DRAFT v0.1 — 19 Agu 2026. Planning only. Eksekusi setelah Paper 2 submit (estimasi Nov 2026–Q1 2027). Jangan mulai sebelum Paper 1 AMICT Sep 1 & Paper 2 MLCIPR Okt selesai.

## Trajectory 3 Paper (kenapa urutan ini masuk akal)
```
Paper 1 (AMICT Sep 1) : Characterization  — "INT8 di CPU malah 6-30x lambat"
        ↓ negative result jadi motivasi
Paper 2 (MLCIPR Okt)  : Co-design        — "Pruning sebelum INT8 pulihkan speedup"
        ↓ sisa bottleneck: MatMul itu sendiri masih ada
Paper 3 (Q1 2027)     : Elimination      — "Hilang MatMul sekalian → ternary + additive-only"
```
Narasi koheren untuk S2 Dresden (Barkhausen/Silicon Saxony): edge ML yang makin ekstrem constraint-nya.

## One-liner
Ganti MatMul di MobileNetV2/ShuffleNetV2 dengan **ternary convolution (-1/0/+1) + additive accumulation** (ala BitNet b1.58 & MatMul-free LM 2406.02528), deploy di **ESP32-S3 8MB PSRAM** (Xtensa LX7, 240 MHz, tensor arena ~510KB), ukur latency/power/accuracy vs Paper 1 & 2. Target: **<1MB model, <100ms latency, accuracy drop <2%**.

## Optimization Framing — Kenapa Ini Tetap Paper Optimization
Ini bukan paper LLM, bukan paper arsitektur baru. Ini paper **inference optimization level operator**, level paling dalam setelah compression.

Taxonomi edge optimization ada tiga level dan kamu sudah naik satu per satu. Level satu adalah compression: Paper 1 karakterisasi INT8 PTQ yang ternyata 6 sampai 30x lebih lambat di CPU karena cost dequantize, Paper 2 co-design pruning 30 dan 50 persen dulu baru INT8 untuk recover speedup. Level dua adalah architecture: TinyNAS dan sejenisnya yang cari arsitektur paling pas untuk hardware tertentu. Level tiga adalah operator dan system: di sini MatMul itu sendiri dihilangkan dan diganti additive-only. Jadi MatMul-free bukan ganti topik, ini kelanjutan logis dari optimization yang sama dengan constraint makin ekstrem.

Untuk reviewer Scopus, framing harus eksplisit sebagai optimization dengan metrik Pareto. Kita tidak klaim accuracy tertinggi. Kita klaim tradeoff paling efisien. Di Paper 1 Pareto sudah ada latency vs accuracy vs size di CPU Intel. Di Paper 2 Pareto bergeser karena pruning. Di Paper 3 Pareto bergeser lagi karena ternary 1.58-bit yang compress 5x vs FP32 dan mengganti multiply-accumulate jadi add saja. Di paper asli 2406.02528 untuk LLM 2.7B hasilnya 3.79ms vs Transformer++ 13.87ms dan hemat SRAM 61 persen, BitNet b1.58 2B4T 2025 membuktikan ternary jalan real di CPU via bitnet.cpp bukan cuma di FPGA. Untuk CNN poultry kita lakukan hal yang sama tapi scope sempit: ganti depthwise bottleneck di MobileNetV2 dulu, keep pointwise dan classifier tetap FP32, ukur di Pi 5 Cortex-A76 dan ESP32-S3.

Detail yang bikin klaim optimization ini valid ada empat. Pertama baseline apple to apple: bandingkan ternary conv vs FP32 vs INT8 vs pruned plus INT8 dari Paper 2 di hardware dan dataset Machuve 8.7k yang sama. Kedua metrik lengkap: latency, size, peak RAM arena, power mW, dan accuracy, lalu plot Pareto bukan single number. Ketiga scope incremental: mulai depthwise-only ternary, kalau drop di bawah 2 persen baru expand ke semua conv, ini yang bikin solo feasible. Keempat hardware proof bertingkat: primary di Pi 5 dengan fake-quant ternary tetap valid untuk conference, ESP32-S3 sebagai stretch goal untuk showcase kandang.

## Mekanisme Inti (yang perlu dipahami sebelum planning)
- **MatMul-free LM (2406.02528):** Ganti dense layer MatMul dengan BitLinear — weight ternary {-1,0,+1} via absmean quantization, forward jadi pure addition (ternary accumulation), bukan multiply-accumulate. Di LLM 2.7B: 3.79ms vs Transformer++ 13.87ms, hemat SRAM 61%.
- **BitNet b1.58 (2025):** 2B4T open-weights, weight ternary, inference di CPU via bitnet.cpp — prove ternary bisa jalan real di edge CPU (bukan cuma FPGA).
- **Untuk CNN poultry:** Ganti `nn.Conv2d` depthwise/pointwise di MN2 dengan `TernaryConv2d` — weight ternarized per forward, activation tetap FP8/INT8, accum via integer add. Tidak perlu ganti seluruh network, cukup depthwise bottleneck yang paling mahal.

## Hardware Target & Constraint Real
- **ESP32-S3:** Xtensa LX7 dual-core 240MHz, 512KB SRAM + 8MB PSRAM, ESP-NN library (optimized kernels), ESP-DL/TFLite-Micro sebagai runtime.
- **Arena:** Paper 2 MN2 INT8 ~2.3MB → harus <1MB setelah ternary (ternary 1.58-bit = ~5x compression vs FP32).
- **Power:** RescueVision kamu 549mW di Pi — target ESP32-S3 <300mW (ukur via USB power meter).
- **Fallback hardware:** Jika ESP32-S3 terlalu sempit, Pi 5 Cortex-A76 tetap jadi baseline (Paper 1/2), ESP32 sebagai "stretch goal" — paper tetap valid kalau hanya Pi 5 ternary vs INT8.

## Experiment Ideas (4–5, prioritas)

### E1 — Ternary Sensitivity per Layer (wajib, paling murah)
Prune dulu? Tidak — langsung ternarize per block. Ukur accuracy drop kalau hanya depthwise vs pointwise vs semua conv di-MatMul-free. Expected: depthwise paling tahan (sparse), pointwise SE paling sensitif.
- Tool: custom `TernaryConv2d` (inspired ridgerchu/matmulfreellm BitLinear, microsoft/BitNet absmean).
- Metric: acc vs layer config (0/ depthwise-only / all).

### E2 — Full MatMul-Free PoultryNet (core claim)
Train 3 varian: MN2-Ternary, SNV2-Ternary, EB0-Ternary (ternary depthwise+pointwise, keep FC FP32). Fine-tune 20 epoch dari checkpoint Paper 1 (LR 5e-5). Bandingkan dengan Paper 1 FP32, Paper 1 INT8, Paper 2 Pruned+INT8.
- Metric: acc, size, latency (Mac CPU + Pi 5), model MB. Target acc 95–97% (drop <2% dari 97.8%).

### E3 — Latency & Power di ESP32-S3 (showcase farm nyata)
Export TFLite-Micro / ESP-DL, flash ke ESP32-S3, benchmark 100 runs, ukur latency + power + peak RAM (arena). Bandingkan dengan TFLite INT8 baseline di ESP32-S3.
- Metric: latency ms, power mW, RAM KB, FPS. Expected: ternary <100ms vs INT8 ~250ms.

### E4 — Pareto 3-Generasi (figure flagship)
Plot acc vs latency vs size untuk 15 titik Paper 1/2 + 3 titik ternary Paper 3. Tunjukkan frontier baru: ternary mendominasi size-latency corner.
- Figure: 2-panel Pareto (acc-lat, acc-size), warna per generasi.

### E5 — Grad-CAM Consistency (optional, jika sempat)
Ternary model masih fokus ke fecal color/texture yang sama? Bandingkan Grad-CAM ternary vs FP32.

## Tooling & Repo Referensi
- `ridgerchu/matmulfreellm` — BitLinear impl, ternary accumulation, Transformers-compatible.
- `microsoft/BitNet` + `bitnet.cpp` — absmean quantization, CPU inference ternary, 2B4T weights sebagai proof.
- `espressif/esp-dl` / `tflite-micro` / `esp-nn` — runtime ESP32-S3.
- `torch-pruning` — jika mau combine pruning Paper 2 + ternary Paper 3 (future).

## Risks & Mitigasi
- **Accuracy drop >5%:** Mitigasi — keep classifier FC di FP32, atau hanya depthwise ternary (hybrid), atau QAT ternary (ste 1 epoch). Tulis ablation.
- **ESP32-S3 toolchain ribet:** Mitigasi — Pi 5 ternary CPU sebagai primary result, ESP32 sebagai secondary/stretch. Paper tetap Scopus tanpa ESP32, tapi dengan ESP32 lebih kuat story farm deployment (kandang 5k ayam bokap).
- **Kernel ternary belum ada untuk ESP32:** Mitigasi — pakai bitnet.cpp CPU kernel sebagai referensi, port minimal ke Xtensa via ESP-NN add ops. Jika tidak sempat, simulasi ternary di Pi 5 (fake-quant) + report theoretical FLOPs/adds tetap valid untuk conference.
- **Effort besar:** Ini Paper 3, bukan next. Jangan eksekusi sebelum Paper 2 selesai. Estimasi 5–6 minggu solo.

## Venue & Timeline
- **Venue:** TinyML Summit / MCSoC 2027 / IEEE Access Edge — Scopus, 4–6 hal. Deadline Q1 2027 (cek wikicfp 1 bulan sebelum).
- **Timeline:** Start Nov 2026 (setelah MLCIPR result), 5 minggu: 2 minggu E1-E2, 1 minggu E3 deploy, 2 minggu tulis.
- **ArXiv:** Upload setelah submit (double-blind check).

## Kaitan ke S2 & Farm
- S2 Dresden Barkhausen: MatMul-free + RISC-V Vector adalah topik yang pas untuk Silicon Saxony — tunjukkan kamu bisa eliminate MatMul, bukan cuma optimize.
- Farm showcase: Deploy ternary model di ESP32-S3 di kandang Marzuqi (Jl. Raya Baruh Sampang) — live demo untuk beasiswa & portfolio.

## Next Step Planning Only (sekarang jangan code)
- [ ] Simpan concept ini di `paper_poultry_matmulfree/` (done)
- [ ] Tulis outline.md full setelah Paper 2 outline final (Nov)
- [ ] Buat plan.md bite-sized setelah Paper 2 submit
- [ ] Kumpulkan 5–10 paper referensi ternary/MCU di `literature/` subfolder Paper 3
