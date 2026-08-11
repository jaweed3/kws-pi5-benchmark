# Literature — Paper A: KWS Audio Edge Benchmark (Pi 5, TFLite vs ONNX)

Folder: `pdfs/` (open access, sudah terdownload)
Folder: `paywalled/` (perlu akses Sevilla — cek README di bawah)

## Urutan baca (rekomendasi)

### Blok 1 — Fundamental (wajib, walau di luar 5 tahun)
| # | File | Paper | Kenapa |
|---|------|-------|--------|
| 1 | `zhang2018_hello_edge_ds-cnn.pdf` | Zhang et al., "Hello Edge: Keyword Spotting on Microcontrollers", 2018 | Ngedefine DS-CNN — model baseline kita. Arsitektur + training. |
| 2 | `warden2018_speech_commands.pdf` | Warden, "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition", 2018 | Dataset kita (v2). Sitasi wajib di paper. |

### Blok 2 — Benchmark methodology (baca sebelum nulis script benchmark)
| # | File | Paper | Kenapa |
|---|------|-------|--------|
| 3 | `mlperf_tiny_2021.pdf` | Banbury et al., "MLPerf Tiny Benchmark", 2021 | Standar benchmark TinyML: KWS = salah satu task-nya. Metrik accuracy/latency/energy. |
| 4 | `quantization_characterization_2023.pdf` | "Performance Characterization of using Quantization for DNN Inference on Edge Devices", 2023 | Paling dekat dengan kita: bandingin TFLite/ONNX/OpenVINO/PyTorch di Pi (ARM) pake MLPerf. Framework comparison di Pi. |
| 5 | `benchmarking_edge_ai_platforms_2024.pdf` | "Benchmarking Edge AI Platforms for High-Performance ML Inference", 2024 | Benchmark SoC edge, metodologi latency distribution. |

### Blok 3 — KWS spesifik (positioning)
| # | File | Paper | Kenapa |
|---|------|-------|--------|
| 6 | `wang2022_pruning_quant_kws_mcu.pdf` | Wang & Li, "KWS System and Evaluation of Pruning and Quantization on Low-power MCUs", 2022 | KWS + kompresi di edge. Bandingin angle sama kita. |
| 7 | `kws_end_to_end_efficiency_2025.pdf` | "End-to-End Efficiency in Keyword Spotting: System-Level Approach for Embedded MCUs", 2025 | Paling baru (Sep 2025). Bandingin DS-CNN vs LiCoNet vs TENet. Wajib biar claim "belum ada" kita tetap valid. |

### Blok 4 — Konteks luas (skim saja)
| # | File | Paper | Kenapa |
|---|------|-------|--------|
| 8 | `edge_ai_survey_2025_mdpi.pdf` | "Edge AI in Practice: Survey and Deployment Framework", 2025 | Survey 39 hal. Cari kalimat positioning + framework deployment. |

## Paywalled — sudah terdownload (via akses Sevilla)

| File | Paper | Kenapa penting |
|------|-------|----------------|
| `ieee2023_concurrent_pipeline_kws_pi4.pdf` (7p) | Concurrent Pipeline Stages Optimization for Embedded KWS (IEEE AIIoT 2023, Pi 4 Cortex-A72, DOI 10.1109/AIIoT58121.2023.10174531) | **POSITIONING UTAMA KITA.** Pi 4 + pipeline optimization; kita beda di Pi 5 + framework comparison. |
| `springer2023_smart_home_kws_fullbook_LNNS660.pdf` (600p) | Lecture Notes in Networks and Systems 660 (Springer, 2023) — **full book**, chapter 6 = Small-Footprint KWS in Smart Home IoT Devices (10.1007/978-981-99-1203-2_6) | TFLite di Pi, stream handling. Baca chapter 6, sisanya skip. |
| `icml2023_dissecting_efficient_wakeword.pdf` (14p) | Dissecting Efficient Architectures for Wake-Word Detection (ICML 2023) | Analisis efisiensi arsitektur wake-word. |

Catatan: file Springer itu full proceedings 600 hal — ekstrak/PDF chapter 6 aja kalau mau ringan dibaca.

## Catatan

- `kws_end_to_end_efficiency_2025.pdf` cuma 4 halaman — kemungkinan short paper/workshop. Kalau mau isi penuhnya, cek versi terbaru di arXiv.
- Verifikasi ulang sebelum sitasi: cek DOI dan halaman asli di venue-nya (aturan: jangan sitasi referensi yang belum diverifikasi).
