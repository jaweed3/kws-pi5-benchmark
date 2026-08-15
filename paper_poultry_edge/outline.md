# Paper Outline: Edge-Optimized Lightweight Architectures for Poultry Fecal Disease Detection

## Working Title
**"Quantization-Aware Deployment of Lightweight CNNs for Poultry Disease Detection on Resource-Constrained Edge Devices"**

---

## Abstract (Draft)
Early detection of poultry diseases through fecal image analysis enables timely intervention and reduces economic losses. While deep learning achieves high accuracy on this task, most approaches rely on heavy architectures deployed on cloud infrastructure — impractical for small-scale poultry farms with limited connectivity and computing resources. This paper presents a systematic characterization of three lightweight convolutional neural architectures (MobileNetV2, ShuffleNetV2, and EfficientNet-Lite0) for poultry fecal disease classification across four categories: Coccidiosis, Newcastle Disease, Salmonella, and Healthy. We evaluate each architecture under post-training quantization (PTQ) and quantization-aware training (QAT) to assess the accuracy-latency-memory trade-off on resource-constrained hardware. Our results demonstrate that [architecture X] achieves [Y]% accuracy with INT8 quantization, enabling real-time inference on Raspberry Pi 4 at [Z] ms latency with [W] MB model size, while maintaining accuracy within [V]% of the floating-point baseline. We further analyze quantization sensitivity across architectures and characterize the Pareto frontier for edge poultry disease detection.

---

## 1. Introduction

### 1.1 Background
- Poultry disease causes significant economic losses, especially in developing countries
- Small-scale farmers lack access to veterinary expertise
- Fecal image analysis as non-invasive diagnostic method
- Existing solutions require cloud connectivity — unavailable in rural farms

### 1.2 Problem Statement
- High-accuracy models (Xception: 98.24%, 23M params) are too heavy for edge deployment
- Lightweight models exist but haven't been systematically characterized for this domain
- Quantization sensitivity across different architectures is unknown
- No Pareto frontier analysis for accuracy-latency-memory trade-off

### 1.3 Research Questions
1. How does quantization sensitivity vary across lightweight architectures on poultry fecal domain?
2. Where does each architecture sit on the accuracy-latency-memory Pareto frontier?
3. How do different architectures learn domain-specific features (color, texture, shape)?
4. How does small dataset size (6,812 images) affect generalization across architectures?
5. Is QAT more effective than PTQ for this specific domain?

### 1.4 Contributions
1. Systematic characterization of quantization sensitivity across three lightweight architectures
2. Accuracy-latency Pareto frontier for edge poultry disease detection
3. Empirical comparison of PTQ vs QAT on small-domain dataset
4. Open-source deployment pipeline for edge poultry disease detection

---

## 2. Related Work

### 2.1 Poultry Disease Detection
- Machuve et al. (2022): MobileNetV2 (98.02%), Xception (98.24%) on fecal images
- Wati et al. (2023): YOLOv7 + Soft-NMS for fecal detection
- DeepPoultryInsight: Vision Transformer for fecal classification
- [Add 5-8 more from recent survey papers]

### 2.2 Lightweight CNN Architectures
- MobileNetV2 (Sandler et al., 2018): Inverted residuals, depthwise separable conv
- ShuffleNetV2 (Ma et al., 2018): Channel shuffle, four guiding principles
- EfficientNet (Tan & Le, 2019): Compound scaling
- EfficientNet-Lite (Google, 2020): Edge-optimized variant

### 2.3 Model Compression for Edge Deployment
- Quantization: INT8, FP16, dynamic range
- Pruning: structured vs unstructured
- Knowledge distillation
- [Focus on domain-specific quantization sensitivity studies]

### 2.4 Edge AI in Agriculture
- Previous deployments on Raspberry Pi, Jetson, mobile devices
- Latency/power requirements for real-time agricultural applications

---

## 3. Methodology

### 3.1 Dataset
- **Source:** Machuve et al. (2021), Zenodo repository
- **Size:** 6,812 annotated fecal images
- **Classes:** Coccidiosis, Salmonella, Newcastle Disease, Healthy
- **Split:** 70% train / 15% validation / 15% test (stratified)
- **Preprocessing:** Resize to 224×224, normalization (ImageNet stats)

### 3.2 Model Architectures

#### 3.2.1 MobileNetV2 (Baseline)
- Inverted residuals with linear bottlenecks
- Width multiplier: 1.0
- Input: 224×224×3
- Params: 3.4M, FLOPs: 300M

#### 3.2.2 ShuffleNetV2
- Channel shuffle + split operation
- 2× width (equivalent to MobileNetV2 capacity)
- Input: 224×224×3
- Params: 2.3M, FLOPs: 150M

#### 3.2.3 EfficientNet-Lite0 (Primary)
- Compound scaling: resolution=224, width=1.0, depth=1.0
- MBConv blocks with squeeze-and-excitation
- Input: 224×224×3
- Params: 4.7M, FLOPs: 400M

### 3.3 Training Protocol
- **Pretrained:** ImageNet (transfer learning)
- **Optimizer:** Adam (lr=1e-4, weight_decay=1e-4)
- **Scheduler:** Cosine annealing, 50 epochs
- **Batch size:** 32
- **Augmentation:** Random horizontal flip, rotation (±15°), color jitter, random crop
- **Loss:** Cross-entropy
- **Framework:** PyTorch → ONNX export

### 3.4 Quantization Methods

#### 3.4.1 Post-Training Quantization (PTQ)
- ONNX Runtime quantization tool
- Dynamic range quantization ( activations: dynamic, weights: static )
- Static quantization with calibration dataset (500 samples)

#### 3.4.2 Quantization-Aware Training (QAT)
- Torch quantization aware training
- 10 additional fine-tuning epochs
- Fake quantization nodes inserted during forward pass

### 3.5 Evaluation Metrics

#### Accuracy Metrics
- Top-1 accuracy (overall)
- Per-class precision, recall, F1
- Confusion matrix
- AUC-ROC (one-vs-rest)

#### Efficiency Metrics
- Model size (MB)
- Inference latency (ms) — averaged over 1000 runs
- Throughput (images/sec)
- Power consumption (mW) — measured on target hardware

#### Hardware Targets
- **Primary:** Raspberry Pi 4 (4GB) — ARM Cortex-A72
- **Secondary:** Laptop CPU (Apple M4) — for reproducibility
- **Optional:** Jetson Nano — for comparison

### 3.6 Interpretability Analysis
- Grad-CAM visualization per architecture
- Feature map analysis (layer-wise)
- Comparison of learned features across architectures

---

## 4. Experiments

### Experiment 1: Baseline Training
- Train all three architectures (FP32)
- Report accuracy, training curves
- Establish baselines

### Experiment 2: PTQ Comparison
- Apply PTQ to all three architectures
- Report accuracy drop per architecture
- **RQ1:** Quantization sensitivity characterization

### Experiment 3: QAT Comparison
- Apply QAT to all three architectures
- Compare PTQ vs QAT accuracy recovery
- **RQ5:** QAT effectiveness on small-domain dataset

### Experiment 4: Hardware Benchmarking
- Deploy INT8 models to Raspberry Pi 4
- Measure latency, throughput, power
- **RQ2:** Pareto frontier construction

### Experiment 5: Interpretability
- Grad-CAM for all architectures
- Feature map visualization
- **RQ3:** Domain-specific feature learning

### Experiment 6: Dataset Size Sensitivity
- Train on 25%, 50%, 75%, 100% of data
- Compare generalization across architectures
- **RQ4:** Small dataset generalization

---

## 5. Results (Expected)

### 5.1 Baseline Accuracy
| Model | FP32 Accuracy | Params | FLOPs |
|-------|---------------|--------|-------|
| MobileNetV2 | ~98% | 3.4M | 300M |
| ShuffleNetV2 | ~96% | 2.3M | 150M |
| EfficientNet-Lite0 | ~98% | 4.7M | 400M |

### 5.2 Quantization Impact
| Model | PTQ INT8 Acc | Δ | QAT INT8 Acc | Δ |
|-------|--------------|---|--------------|---|
| MobileNetV2 | ~96% | -2% | ~97% | -1% |
| ShuffleNetV2 | ~95% | -1% | ~96% | 0% |
| EfficientNet-Lite0 | ~95% | -3% | ~97% | -1% |

### 5.3 Edge Performance (Raspberry Pi 4)
| Model | Size (MB) | Latency (ms) | FPS | Power (mW) |
|-------|-----------|--------------|-----|------------|
| MobileNetV2 INT8 | ~3.5 | ~15 | ~67 | ~800 |
| ShuffleNetV2 INT8 | ~2.5 | ~10 | ~100 | ~700 |
| EfficientNet-Lite0 INT8 | ~5.0 | ~20 | ~50 | ~900 |

---

## 6. Discussion

### 6.1 Quantization Sensitivity
- Which architecture is most robust to INT8?
- Why? (homogeneous vs heterogeneous operations)
- Implications for edge deployment

### 6.2 Pareto Analysis
- Sweet spot identification
- Which architecture for which constraint?
  - Memory-constrained → ShuffleNetV2
  - Latency-constrained → ShuffleNetV2
  - Accuracy-constrained → EfficientNet-Lite0
  - Balanced → MobileNetV2

### 6.3 Feature Learning Analysis
- Grad-CAM insights
- Do architectures learn different features?
- Domain-specific adaptation from ImageNet

### 6.4 Practical Implications
- For small-scale farmers: what hardware + model combo is optimal?
- Deployment pipeline recommendations
- Cost analysis

---

## 7. Conclusion

### Summary
- Three architectures characterized on poultry fecal domain
- Quantization sensitivity varies by architecture type
- Pareto frontier identified for edge deployment
- QAT recommended for small-domain datasets

### Future Work
- Federated learning across multiple farms
- Multi-task: disease detection + severity grading
- Deployment on even smaller devices (ESP32-CAM, STM32)

---

## References (Key Papers to Cite)

1. Machuve et al. (2022). Poultry diseases diagnostics models using deep learning. Frontiers in AI.
2. Sandler et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR.
3. Ma et al. (2018). ShuffleNetV2: Practical Guidelines for Efficient CNN Architecture Design. ECCV.
4. Tan & Le (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML.
5. Jacob et al. (2018). Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR.
6. [Add more from related work section]

---

## Experimental Checklist

- [ ] Download dataset from Zenodo
- [ ] Set up training environment (PyTorch, ONNX Runtime)
- [ ] Train MobileNetV2 (FP32 baseline)
- [ ] Train ShuffleNetV2 (FP32 baseline)
- [ ] Train EfficientNet-Lite0 (FP32 baseline)
- [ ] Export all models to ONNX
- [ ] Apply PTQ to all models
- [ ] Apply QAT to all models
- [ ] Set up Raspberry Pi 4 benchmarking
- [ ] Run hardware benchmarks
- [ ] Generate Grad-CAM visualizations
- [ ] Run dataset size sensitivity experiments
- [ ] Write paper
- [ ] Submit to venue

---

## Target Venues

1. **IEEE Access** — open access, good for engineering/ML
2. **MDPI AgriEngineering** — domain-specific, already published poultry papers
3. **Computers and Electronics in Agriculture** — high impact, agriculture + tech
4. **ICDLT / ICAITech** — conference option

---

*Draft created: August 15, 2026*
*Status: Outline complete, ready for experiment execution*
