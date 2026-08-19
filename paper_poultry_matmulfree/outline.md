# Paper 3 Outline — Additive-Only Inference Optimization via Ternary Convolution for Poultry Disease Detection

> Status: DRAFT v0.1 — 19 Agu 2026. Planning only. Eksekusi setelah Paper 2 submit (Nov 2026 - Q1 2027). Jangan mulai sebelum Paper 1 AMICT Sep 1 selesai.

## Target Venue
Primary: TinyML Summit / MCSoC 2027 / IEEE Edge AI workshop — IEEE, Scopus, 4-6 pages
Fallback: MLCIPR 2027 / IEEE Access (jika butuh jurnal short)
Cek deadline 1 bulan sebelum di wikicfp + mcsoc-forum.org
Fee ~$150 student. Siapkan main.tex + main_anon.tex (double-blind check)

## Working Title
Additive-Only Inference Optimization for Lightweight CNNs: Ternary Convolution Eliminating MatMul for Poultry Disease Detection on ESP32-S3
Alt 1: From Quantization to Elimination: MatMul-Free Ternary CNNs for Ultra-Low-Power Poultry Disease Detection
Alt 2: BitNet-Style Ternary CNNs on Microcontrollers: Operator-Level Optimization Beyond INT8

Judul sengaja pakai Additive-Only Inference Optimization bukan MatMul-Free LLM supaya reviewer vision tidak salah kira kamu klaim LLM di MCU.

## Positioning
Paper 1 menemukan INT8 PTQ 6-30x lebih lambat di CPU karena dequantize overhead tanpa VNNI. Paper 2 menjawab dengan pruning 30/50 persen sebelum INT8 untuk recover speedup. Sisa bottleneck adalah MatMul itu sendiri yang masih pakai multiply-accumulate. Paper 3 menghilangkan MatMul sekalian dengan ternary weight -1/0/+1 dan integer add saja. Literatur matmul-free sebelumnya hanya untuk LLM 2.7B di FPGA/CPU (2406.02528, 3.79ms vs 13.87ms, hemat SRAM 61 persen) dan BitNet b1.58 2B4T 2025 via bitnet.cpp. Belum ada yang menerjemahkan ke CNN lightweight untuk klasifikasi poultry fecal di MCU. Kita yang pertama untuk domain ini dengan hardware proof di Pi 5 dan ESP32-S3.

Unique angle: Pertama tunjukkan ternary convolution additive-only untuk poultry fecal di MCU dengan Pareto tiga generasi Paper 1/2/3.

## Abstract (Draft — isi angka setelah E2)
Post-training quantization is widely assumed to help edge deployment yet our prior work shows 6-30x slowdown on CPUs without INT8 acceleration and pruning only partially recovers it. This paper asks whether the MatMul itself can be eliminated. We replace depthwise and pointwise convolutions in MobileNetV2, ShuffleNetV2 and EfficientNet-B0 with BitNet-style ternary convolution where weights are constrained to -1/0/+1 via absmean quantization and forward pass uses only integer addition. Fine-tuned for 20 epochs from >97 percent checkpoints on the Machuve 8,770-image poultry fecal dataset, ternary models are evaluated on Intel i5 and Raspberry Pi 5 and deployed to ESP32-S3 (Xtensa LX7, 510KB arena). Ternary depthwise-only variants retain within 2 percent accuracy while reducing model size 5x and latency Y percent versus INT8, shifting the Pareto frontier to the ultra-low-power corner. Results provide an operator-level optimization recipe beyond compression.

## Research Questions
1. Seberapa sensitif tiap blok (depthwise vs pointwise vs SE) terhadap ternarization sebelum accuracy drop melebihi 2 persen
2. Apakah full ternary CNN (depthwise+pointwise, keep FC FP32) bisa retain >95 persen accuracy dengan fine-tune 20 epoch dari checkpoint Paper 1
3. Berapa latency, size dan power recovery vs Paper 1 FP32/INT8 dan Paper 2 pruned+INT8 di Pi 5 dan ESP32-S3
4. Di mana Pareto baru tiga generasi (15 titik Paper 1/2 plus 3 titik ternary) dan sweet spot per constraint

## Contributions
1. TernaryConv2d additive-only untuk CNN lightweight poultry dengan absmean quantization dan STE, depthwise-first incremental strategy yang solo-feasible
2. Bukti empiris ternary depthwise-only retain accuracy dalam 2 persen sambil compress 5x vs FP32 dan percepat latency Y percent vs INT8 di CPU tanpa VNNI
3. Deployment dan pengukuran fisik di ESP32-S3 (latency, power, peak arena) sebagai showcase farm nyata versus Pi 5 baseline
4. Pareto tiga generasi yang tunjukkan frontier baru di ultra-low-power corner dan recipe praktis kapan pakai FP32 vs pruned-INT8 vs ternary

## 1. Introduction (0.5 pg)
Paragraf 1 konteks poultry 70.2B USD, need farm-level offline detection. Paragraf 2 rekap Paper 1 dan 2 singkat sebagai motivasi, sisa bottleneck MatMul. Paragraf 3 ide additive-only, preview hasil dan preview Pareto tiga generasi. Tutup dengan list contributions dan paparkan incremental scope depthwise dulu.

## 2. Related Work (0.6 pg)
2.1 Poultry disease detection sama Paper 1 ringkas plus tekankan semua masih FP32 di cloud.
2.2 Quantization and pruning for edge (Jacob 2018 PTQ, ONNX Runtime issues 6732/12854 slowdown, HAPE TODAES 2025 hardware-aware pruning, Paper 2 kita sebagai prior work yang cite diri sendiri).
2.3 Ternary and matmul-free inference (MatMul-free LM 2406.02528 BitLinear ternary accumulation FPGA 3.79ms, BitNet b1.58 2025 2B4T absmean plus bitnet.cpp CPU, Slender-Mamba 1.58-bit sebagai bukti ternary di SSM, MCUNet TinyNAS/TinyEngine sebagai NAS compression yang tidak sampai operator). Gap: belum ada ternary CNN untuk poultry di MCU.
2.4 MCU deployment (TFLite-Micro, ESP-DL, ESP-NN, Burn no-std di ESP32 proof-of-concept inference only). Gap: belum ada backward dan ternary di MCU.

## 3. Methodology (1.0 pg)

3.1 Dataset sama persis Paper 1: Machuve 8,770 4 kelas Coccidiosis Healthy Salmonella NCD, split 5614/1402/1754, 224x224 ImageNet norm. Tekankan tidak ada data baru supaya apple to apple.

3.2 Baseline models: MN2 2.2M, SNV2 1.26M, EB0 4.0M load checkpoint FP32 >97 persen dari Paper 1.

3.3 Ternary Convolution Design (core method, ini yang reviewer optimization cek)
Weight ternarization per forward: absmean quantization. Given weight W, scale = mean(abs(W)), W_tern = clamp(round(W/scale), -1, 1) in -1/0/+1. Forward: Y = scale * ternary_accum(X, W_tern) dimana ternary_accum hanya add/sub tidak ada multiply. Implementasi sebagai nn.Module TernaryConv2d wrapper dari nn.Conv2d. Activation keep FP32 atau INT8 (fake-quant optional). Backward pakai Straight-Through Estimator (STE): grad_W = grad_W_tern. Classifier FC terakhir keep FP32 untuk jaga accuracy.
Incremental strategy: Stage A depthwise-only ternary, Stage B depthwise+pointwise ternary, Stage C all conv ternary. Paper klaim Stage A dulu, expand hanya jika drop <2 persen.
Size math: ternary 1.58-bit packing 5x vs FP32 32-bit, 2x vs INT8. Contoh MN2 8.48MB FP32 jadi ~1.7MB ternary.

3.4 Training protocol: Fine-tune dari checkpoint Paper 1, 20 epoch, Adam 5e-5 (lebih kecil dari Paper 1 1e-4), weight decay 1e-4, cosine, batch 32, augment sama, seed 42, mixed precision off untuk ternary. STE learning rate warmup 2 epoch. Validasi tiap epoch, best val checkpoint.

3.5 Quantization and export: Export ke ONNX dengan op TernaryConv sebagai custom op untuk fake-quant simulasi. Untuk Pi 5 ukur dengan ONNX Runtime CPU. Untuk ESP32-S3 convert ke TFLite-Micro via ESP-DL (ternary sebagai INT8 dengan scale terpisah). Simpan raw latency array.

3.6 Benchmark protocol: 200 runs plus 10 warmup per model per hardware, mean/P50/FPS, bootstrap CI 95 persen reuse Paper 1/2. Hardware Pi 5 Cortex-A76 primary, ESP32-S3 secondary. Power ukur via USB power meter 100 samples, peak arena via ESP heap trace.

## 4. Experiments (0.5 pg ringkas)

E1 Ternary sensitivity per block: 3 model x 3 config (depthwise-only, pointwise-only, all) plus keep FC FP32. Metric acc, size, theoretical adds vs MACs.

E2 Full ternary poultry net: 3 varian MN2/SNV2/EB0 ternary Stage A/B/C fine-tune 20 epoch vs Paper 1/2 baselines. Metric acc, size, latency Mac+Pi5.

E3 Latency and power on ESP32-S3: Export best ternary per model ke ESP32-S3, 100 runs, banding versus TFLite INT8 baseline di ESP32-S3. Metric latency ms, power mW, arena KB.

E4 Pareto tiga generasi: 15 titik Paper1/2 plus 3-9 titik ternary, 2-panel acc-lat dan acc-size, warna per generasi, highlight frontier baru.

E5 Grad-CAM consistency optional: ternary vs FP32 per kelas 4 kelas.

## 5. Results and Discussion (1.5 pg) — template, isi setelah run

5.1 Ternary sensitivity
Tabel acc dan size per config. Hypothesis: depthwise paling tahan (sparse, local), pointwise SE paling sensitif (channel mixing). SNV2 channel shuffle tahan ternary lebih baik dari MN2.

Tabel template columns Model Config Params Size Acc Delta

5.2 Full ternary vs baselines
Tabel acc size latency Pi5. Target MN2-Ternary depthwise-only 96.5 persen (drop 1.3), size 1.7MB, latency Pi5 12ms vs MN2 INT8 55.8ms vs MN2 FP32 1.85ms vs MN2 pruned30 INT8 15ms (Paper2). Jelaskan mekanisme add vs MAC.

5.3 ESP32-S3 deployment
Tabel ESP32 latency power arena. Expected ternary <100ms 280mW 420KB vs INT8 baseline 250ms 320mW 510KB. Jika ESP32 belum berhasil flash, laporkan Pi5 fake-quant plus theoretical ESP32 via thop adds, jujur di limitation.

5.4 Pareto tiga generasi
Figure 2-panel. Expected ternary mendominasi size-latency corner, EB0 FP32 tetap pojok accuracy. Beri guideline kapan pakai apa: latency-critical Pi5 FP32, memory-critical ESP32 ternary, accuracy-critical EB0 FP32.

5.5 Interpretability optional
Figure Grad-CAM. Expected fokus tetap fecal color/texture, tidak drift.

## 6. Conclusion (0.3 pg)
Ringkas: ternary depthwise recovers size 5x dengan drop <2 persen, latency recovery Y persen di CPU, valid di ESP32-S3. Limitation: hanya PTQ ternary tidak QAT ternary full, hanya classification 224x224, belum video. Future: QAT ternary plus pruning co-design, RISC-V Vector extension, deploy farm 5k ayam Marzuqi Jl Raya Baruh Sampang live 24 jam.

## References (15+ reuse Paper1/2 plus tambah)

1 Machuve et al Frontiers AI 2022 dataset
2 Dhungana et al AgriEngineering 2025
3 Tasdelen Arslan 2025
4 Sandler MobileNetV2 CVPR 2018
5 Ma ShuffleNetV2 ECCV 2018
6 Tan EfficientNet ICML 2019
7 Jacob CVPR 2018 quant
8 ONNX Runtime docs 2024 plus issues 6732/12854
9 Maulana Computers 2026 QAT>PTQ
10 Zhu et al MatMul-free LM arXiv 2406.02528 (core)
11 Ma et al BitNet b1.58 2024 plus Wang et al BitNet b1.58 2B4T 2025 2504.12285 plus bitnet.cpp (core)
12 Slender-Mamba COLING 2025 1.58-bit (optional context)
13 Lin et al MCUNet NeurIPS 2020 TinyNAS TinyEngine
14 Paper 1 kita AMICT 2026 characterization (self-cite)
15 Paper 2 kita MLCIPR 2026 pruned+INT8 (self-cite)
16 Selvaraju Grad-CAM ICCV 2017
17 Espressif ESP-DL ESP-NN docs

## Figures Checklist
Fig 1 Ternary sensitivity acc vs config grouped bar 3 model
Fig 2 Latency comparison grouped bar FP32 vs INT8 vs pruned-INT8 vs ternary di Pi5 core figure
Fig 3 Pareto tiga generasi 2-panel acc-lat acc-size 18 titik
Fig 4 ESP32 deployment photo plus latency power bar Intel vs Pi5 vs ESP32
Fig 5 Optional Grad-CAM ternary vs FP32 2x4 grid

## Appendix
Raw JSON per model config hardware, reproduce script reproduce_ternary.py seed 42, ONNX size thop FLOPs vs adds, ESP32 flash log.
