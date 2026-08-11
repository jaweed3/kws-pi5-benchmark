# Paper A — Benchmark Keyword Spotting (Audio ML) di Edge
**Status: RISET SELESAI — siap eksekusi setelah ComTech di-submit**

## Konteks
- ComTech #16124 (image edge benchmark) lagi direview — habis submit, garap ini.
- Pertanyaan yang mendorong: image ML udah mature di edge, audio ML belum.
  Paper ini = bukti kuantitatif gap itu + benchmark sistematis pertama (kita) buat KWS di Pi 5.

## Judul kerja
"Benchmarking Keyword Spotting Inference on ARM64 Edge Devices:
TensorFlow Lite vs ONNX Runtime on Raspberry Pi 5"

## Story / Gap
- Image di edge: banyak benchmark (termasuk paper ComTech kita). Audio di edge: jarang, fragmented.
- KWS = workload representatif audio (always-on, real-time, resource-constrained).
- Yang sudah ada: Hello Edge (MCU, DS-CNN, 2018); paper IEEE 10174531
  (pipeline optimization KWS di Pi 4 Cortex-A72). **Belum ada** benchmark framework
  comparison (TFLite vs ONNX) KWS di Pi 5 dengan latency distribution + resource profiling.

## Dataset
- **Google Speech Commands v2** (Warden 2018, arXiv 1804.03209)
  - Lisensi: CC BY 4.0 — aman untuk publikasi
  - ~105k utterances, 35 kata; standar KWS: subset 10-12 kata (yes, no, up, down, left, right, on, off, stop, go)
  - Download: HuggingFace `google/speech_commands` (atau storage.googleapis.com official)
  - Ukuran ~2.3GB — download di Mac, subset kecil yang dipakai kirim ke Pi (disk Pi cuma 3GB free!)

## Model & Runtime
- Baseline: **DS-CNN** (Hello Edge, Zhang et al. 2017/2018, arXiv 1711.07128)
  - TFLite: TF Lite Model Maker / google-research kws_streaming (pretrained tersedia)
  - ONNX: export dari training atau konversi TFLite→ONNX
- Runtime di Pi 5: **TFLite** vs **ONNX Runtime (CPU)**
- Opsional (extension): Burn via burn-onnx codegen — angle "Rust native audio di edge",
  bisa jadi pembeda kalau sempat; JANGAN over-commit kalau waktu mepet.
- Preprocessing (MFCC/spectrogram): definisikan jelas di paper — diukur terpisah dari model
  inference, atau end-to-end kalau model-nya input raw audio. Konsistensi wajib.

## Metrik (reuse protokol ComTech)
- Latency distribution: mean/median/p95/p99/std (raw samples disimpan)
- Throughput: FPS / inferences per second
- Akurasi: top-1 pada test set Speech Commands
- Resource: RSS memory, CPU%, suhu SoC (vcgencmd)
- Setup: warmup + N iterasi; sequential (jangan paralel — inflasi 2-4x)

## Related work kunci (untuk Related Work section)
1. Warden 2018 — Speech Commands dataset (arXiv 1804.03209)
2. Zhang et al. 2018 — Hello Edge: KWS on Microcontrollers, DS-CNN (arXiv 1711.07128)
3. IEEE 10174531 — Concurrent Pipeline Stages Optimization for Embedded KWS (Pi 4, A72)
   → **bedain angle: kita Pi 5 A76 + framework comparison + latency distribution + resource**
4. ComTech #16124 (kita sendiri) — image benchmark, extension alami ke audio

## Konferensi kandidat (IEEE, beginner-friendly)
- **MCSoC 2026** — Shanghai, 14-17 Des 2026. Round 2 buka ~10 Agustus 2026.
  Deadline round 2: VERIFIKASI di mcsoc-forum.org saat buka.
- **MLCIPR 2026** — Nanjing, 25-27 Des 2026. Deadline: VERIFIKASI di iccipr.org.
- Prioritas: MCSoC (timeline paling pas — round 2 buka tepat setelah ComTech submit).

## Timeline
- 31 Jul - 3 Agt: **SUBMIT COMTECH** (prioritas #1, deadline 3 Agt)
- 4-8 Agt: setup benchmark audio di Pi 5 (dataset subset, model DS-CNN TFLite+ONNX, script)
- 9-12 Agt: jalankan benchmark, kumpulkan data, log JSON
- 13-20 Agt: tulis paper (IEEE template), figures, tables
- ~21 Agt: submit MCSoC round 2 (kalau deadline masih buka — verifikasi)

## Risiko & mitigasi
1. **Disk Pi 3GB free** → dataset di Mac, cuma subset kecil (beberapa MB) ke Pi.
   Kalau perlu, bersih-bersih lagi (cache pip, ~/.cache).
2. **Model DS-CNN nggak ada pretrained ONNX siap pakai** → train singkat di Mac
   (Model Maker, mini dataset) atau konversi TFLite→ONNX. Untuk latency, arsitektur
   yang penting; akurasi pakai model yang ditrain standar.
3. **Deadline konferensi belum terverifikasi** → cek mcsoc-forum.org (10 Agt) & iccipr.org.
4. **Over-scope** → JANGAN tambah ESP32 / Burn / continual learning di paper ini.
   Tetap: TFLite vs ONNX, Pi 5, KWS. Simpel = jadi.

## Infra yang sudah ada (reuse!)
- Pi 5: techronity@192.168.119.43 (venv drone_venv, uv, benchmark script pattern)
- Script benchmark `benchmark_face.py` → template untuk `benchmark_kws.py`
- Protokol + dokumentasi pattern dari ComTech & drone_face
- Mac: tectonic, matplotlib, folder pattern ~/research/paper_audio_kws/
