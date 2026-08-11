# Research Archive

Arsip riset pribadi — paper, kode benchmark, figure, dan literatur.

## Isi

| Folder | Isi |
|---|---|
| `paper_audio_kws/` | Paper A: *Benchmarking Keyword Spotting Architectures Across TensorFlow Lite and ONNX Runtime on a Raspberry Pi 5* |
| `paper_kombucha_edge/` | Paper B (draft): kombucha ML + edge (kolaborasi Amirul Al Aziz) |

## paper_audio_kws

- `paper/` — source LaTeX (`main.tex`), PDF final (`main.pdf`), `refs.bib`, figure scripts (`gen_figure_*.py`), review manuscript `.doc` (EDAS)
- `bench/code/` — pipeline: `train_models.py` → `export_model.py` → `quantize_model.py` → `benchmark_kws.py`
- `bench/results/` — raw benchmark JSON (latency, fp32/int8)
- `bench/models/` — model artifacts: MLPerf reference + trained (keras/tflite/onnx), per-arsitektur
- `literature/` — literatur open-access (PDF) + skrip download

### Catatan penting

- Dataset Speech Commands v2 (5.6GB) dan `data_cache.npz` (cache MFCC, ~166MB/model) TIDAK di-commit — regenerable lewat `bench/` scripts. Download SCv2 lalu jalankan pipeline.
- `literature/paywalled/` (18MB) sengaja tidak di-commit (lisensi) — ada lokal.
- Status venue: ICAITech 2026 (submitted, review via EDAS), MLCIPR fallback. Paper bersifat double-blind sampai diterima — repo ini private, jangan di-public sebelum acceptance.

## Status

- Paper A: submitted ICAITech (deadline 12 Aug 2026, keputusan 15 Sep)
- Paper B: outline siap, eksperimen wet lab menunggu batch (target Okt 2026)
