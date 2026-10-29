# Benchmarking Keyword Spotting Architectures across TensorFlow Lite and ONNX Runtime on a Raspberry Pi 5

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
<!-- Zenodo badge placeholder: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxxx) -->

Official artifact repository for the paper:
> **Benchmarking Keyword Spotting Architectures across TensorFlow Lite and ONNX Runtime on a Raspberry Pi 5**  
> *Fatih Jawwad Al Mumtaz and Amirul Al Aziz*  
> Universitas Darussalam Gontor, Ponorogo, Indonesia  
> Accepted at the 8th International Conference on Artificial Intelligence and Information Technology (ICAITech 2026).

---

## Overview

This repository provides the complete experimental artifact, model pipeline, and benchmarking suite to evaluate four Keyword Spotting (KWS) architectures on a **Raspberry Pi 5** (quad-core Arm Cortex-A76 at 2.4 GHz):
1. **DS-CNN** (MLPerf Tiny reference, 22.6k parameters)
2. **DNN** (4-layer dense baseline, 88.4k parameters)
3. **LSTM** (Single-layer LSTM, 20.0k parameters)
4. **CRNN** (CNN--GRU hybrid, 56.8k parameters)

Each model is evaluated under three deployment configurations:
- **TensorFlow Lite (float32)** via LiteRT (`ai-edge-litert`)
- **TensorFlow Lite (full int8)** post-training quantization
- **ONNX Runtime (float32)** (`onnxruntime` CPUExecutionProvider)

Across two dataset evaluation tasks:
- **10-Command Subset** (4,074 Speech Commands v2 test clips)
- **Full 12-Class Task** (12,005 test clips including unknown and silence)
- **Thread Sensitivity Sweep** (1, 2, and 4 threads pinned)
- **On-board Power/Energy Profiling** via Raspberry Pi 5 PMIC ADC

---

## Repository Structure

```text
kws-pi5-benchmark/
├── paper/                  # Camera-ready LaTeX sources, compiled PDF, figures & bib
│   ├── main.tex            # IEEE conference manuscript source (6 pages)
│   ├── main.pdf            # Compiled publication PDF (IEEE PDF eXpress compliant)
│   ├── refs.bib            # Bibliography
│   ├── figures/            # Generated figures
│   └── gen_figure_*.py     # Reproduction scripts for paper figures
├── bench/
│   ├── code/               # Python scripts for training, export, PTQ, and benchmarking
│   │   ├── benchmark_kws.py      # Standard 1,000 warmup + 1,000 measured latency runner
│   │   ├── bench_threads.py      # Thread sweep benchmark (1, 2, 4 threads)
│   │   ├── evaluate.py           # Top-1 accuracy evaluator (10-command & 12-class)
│   │   ├── quantize_dscnn.py     # True full-int8 quantization for DS-CNN
│   │   ├── train_models.py       # Training pipeline for DNN, LSTM, CRNN
│   │   └── mfcc.py               # MLPerf Tiny compatible NumPy MFCC extractor
│   ├── models/             # Exported models (.tflite, .onnx, .keras)
│   │   ├── kws_ref_model_fp32.tflite
│   │   ├── kws_ref_model_fullint8.tflite
│   │   ├── kws_ref_model_fp32.onnx
│   │   └── trained/        # Models trained from scratch (dnn, lstm, crnn)
│   └── results/            # Raw measurement JSONs and evaluation logs
├── literature/             # Related literature references and download script
├── CITATION.cff            # Citation metadata
└── LICENSE                 # GNU Affero General Public License v3.0 (AGPLv3)
```

---

## Quickstart & Reproduction

### Prerequisites
- Single-Board Computer: **Raspberry Pi 5** (Debian GNU/Linux 64-bit)
- Python $\ge$ 3.10 with `onnxruntime`, `ai-edge-litert`, `numpy`, `scipy`

Set CPU governor to performance:
```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 1. Run Latency Benchmark
To benchmark latency under the exact protocol reported in the paper (1,000 warmup + 1,000 timed passes with IQR outlier filtering and 10,000-resample bootstrap):
```bash
cd bench/code
python benchmark_kws.py \
  --tflite ../models/trained/crnn/crnn_fp32.tflite \
  --onnx ../models/trained/crnn/crnn_fp32.onnx \
  --tag crnn_fp32 \
  --output ../results/bench_crnn_fp32.json
```

### 2. Thread Sensitivity Sweep
```bash
python bench_threads.py --models dscnn,crnn --threads 1,2,4 --output ../results/bench_threads.json
```

### 3. Evaluate Top-1 Accuracy (10-Command & 12-Class)
```bash
python evaluate.py --model ../models/trained/crnn/crnn_fp32.tflite --backend tflite --wavlist ../data/testlist_10.tsv
python evaluate.py --model ../models/trained/crnn/crnn_fp32.tflite --backend tflite --wavlist ../data/testlist_12.tsv
```

---

## Citation

```bibtex
@inproceedings{almumtaz2026kws,
  author    = {Fatih Jawwad Al Mumtaz and Amirul Al Aziz},
  title     = {Benchmarking Keyword Spotting Architectures across {TensorFlow Lite} and {ONNX Runtime} on a {Raspberry Pi 5}},
  booktitle = {Proceedings of the 8th International Conference on Artificial Intelligence and Information Technology (ICAITech)},
  year      = {2026}
}
```
