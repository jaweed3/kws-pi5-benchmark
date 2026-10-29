# Conference Video Presentation Script & Q&A Guide

**Paper Title:** Benchmarking Keyword Spotting Architectures across TensorFlow Lite and ONNX Runtime on a Raspberry Pi 5  
**Conference:** 2026 8th International Conference on Artificial Intelligence and Information Technology (ICAITech)  
**Authors:** Fatih Jawwad Al Mumtaz and Amirul Al Aziz  
**Affiliation:** Faculty of Science and Technology, Universitas Darussalam Gontor, Ponorogo, Indonesia  
**Permanent DOI:** [10.5281/zenodo.23765521](https://doi.org/10.5281/zenodo.23765521)  
**Repository:** [https://github.com/jaweed3/kws-pi5-benchmark](https://github.com/jaweed3/kws-pi5-benchmark)  
**Target Duration:** 8 to 10 minutes (~1,200 – 1,400 words spoken at ~130–140 WPM)  
**Slide Deck:** `deck/slides.pdf` (16:9 Widescreen, 11 slides)

---

## Recording Checklist & Technical Recommendations

1. **Display & Resolution:** Open `deck/slides.pdf` in Adobe Acrobat / Preview / Okular in Full Screen presentation mode (`Ctrl+L` or `Cmd+Shift+F`) at 1920×1080 (16:9).
2. **Audio Setup:** Use a dedicated USB microphone or headset. Position it 10–15 cm from your mouth, slightly off-axis to avoid plosive pops ("p", "b").
3. **Pacing:** Speak clearly, deliberately, and with steady rhythm. Pause for 1–2 seconds when transitioning slides to allow the visual slide to register.
4. **Camera / Web-cam:** Position the camera at eye level. Look into the camera lens periodically, especially during the introduction (Slide 1) and conclusion (Slides 10–11).

---

## Slide-by-Slide Narration Script

```
================================================================================
SLIDE 1: TITLE SLIDE [00:00 - 00:45]
Visual: Slide 1 (Title, Authors, Affiliation, ICAITech 2026, DOI & AGPLv3 badge)
================================================================================
```

**[Spoken Script - English]**
> *"Good day, distinguished session chairs, fellow researchers, and conference attendees.*
>
> *My name is Fatih Jawwad Al Mumtaz, and on behalf of my co-author Amirul Al Aziz from Universitas Darussalam Gontor, Indonesia, I am pleased to present our paper entitled:*
>
> **'Benchmarking Keyword Spotting Architectures across TensorFlow Lite and ONNX Runtime on a Raspberry Pi 5.'**
>
> *In edge AI and TinyML, deployment decisions are frequently guided by rules of thumb rather than rigorous empirical verification. In this presentation, we systematically unpack how model topology, runtime engine design, and hardware thread pools interact on modern 64-bit Arm single-board computers."*

**[Presenter Cues / Catatan Presenter (ID)]**
- Buka presentasi dengan tenang dan percaya diri.
- Sorot judul paper dan afiliasi UNIDA Gontor.
- Tekankan bahwa riset ini membongkar asumsi umum (*rules of thumb*) dalam deployment TinyML.

---

```
================================================================================
SLIDE 2: CONTEXT & MOTIVATION [00:45 - 01:40]
Visual: Slide 2 (Two side-by-side cards: KWS workload & SDG 3, bottom alert box)
================================================================================
```

**[Spoken Script - English]**
> *"Let us begin with the operational context.*
>
> *Keyword Spotting—or KWS—is the canonical always-on edge audio workload. It continuously monitors streaming audio for wake-words or clinical acoustic cues under strict millisecond latency and milliwatt power budgets.*
>
> *Deploying KWS directly on-device aligns directly with UN Sustainable Development Goal 3—Good Health and Well-Being. By processing audio entirely on the edge, we eliminate cloud dependency for low-connectivity clinics and ensure zero telemetry exposure for sensitive biometric voice data.*
>
> *However, there is an unspoken assumption in the TinyML community: practitioners routinely assume TensorFlow Lite is universally the fastest inference engine on Arm processors. But on modern 64-bit Arm platforms—such as the Cortex-A76 cores on the Raspberry Pi 5—does TFLite actually hold an uncontested advantage over ONNX Runtime across diverse model topologies? That is the empirical question we set out to answer."*

**[Presenter Cues / Catatan Presenter (ID)]**
- Hubungkan KWS dengan privasi data dan SDG 3 (kesehatan / klinik terpencil tanpa internet).
- Tekankan *The Problem*: kebiasaan developer menganggap TFLite otomatis lebih cepat di Arm tanpa menguji ONNX Runtime.

---

```
================================================================================
SLIDE 3: BENCHMARK MODEL ZOO & RECONSTRUCTION [01:40 - 02:45]
Visual: Slide 3 (Table of 4 models, two bottom cards detailing reconstruction & PTQ)
================================================================================
```

**[Spoken Script - English]**
> *"To ensure a rigorous comparison, we evaluated four distinct architectural paradigms, all standardized on a 49-by-10 MFCC spectro-temporal input and a 12-way softmax classification head over Speech Commands v2.*
>
> *First, DS-CNN—the canonical depthwise-separable convolutional reference from MLPerf Tiny with 22.6 thousand parameters.*
> *Second, a 4-layer fully connected DNN baseline serving as our parameter-heavy lower bound.*
> *Third, a compact unrolled LSTM network with under 20 thousand parameters.*
> *And fourth, a hybrid CRNN combining convolutional feature extractors with recurrent GRU temporal modeling.*
>
> *Crucially, during our audit of the official MLPerf Tiny reference artifact, we discovered that it is actually a mixed-precision graph—containing int8 convolutional kernels but float32 auxiliary layers. To eliminate cross-format translation artifacts, we decompiled and reconstructed a mathematically pristine float32 Keras baseline. From this gold standard, we derived both ONNX graphs and true full-int8 flatbuffers calibrated over 1,000 representative audio clips."*

**[Presenter Cues / Catatan Presenter (ID)]**
- Jelaskan 4 famili arsitektur: CNN (DS-CNN), MLP (DNN), RNN (LSTM), dan Hybrid (CRNN).
- Sorot temuan kritis: artifact resmi MLPerf ternyata *mixed-precision*, sehingga kami rekonstruksi ke true-fp32 agar perbandingannya 100% *apple-to-apple*.

---

```
================================================================================
SLIDE 4: CONTROLLED BENCHMARK PROTOCOL [02:45 - 03:45]
Visual: Slide 4 (Hardware specs on left, statistical protocol on right)
================================================================================
```

**[Spoken Script - English]**
> *"Turning to our experimental protocol: reliable edge benchmarking requires eliminating non-deterministic system noise.*
>
> *All experiments were conducted on a Raspberry Pi 5 featuring the Broadcom BCM2712 SoC with quad-core Arm Cortex-A76 processors at 2.4 GHz, running 64-bit Debian Linux. The CPU governor was strictly pinned to 'performance' to prevent dynamic clock stepping.*
>
> *We evaluated Google's latest LiteRT runtime—version 2.1.6—against ONNX Runtime 1.28.*
>
> *Statistically, each measurement is preceded by 1,000 unmeasured warm-up passes, followed by 1,000 sequential measurement iterations. Outliers were filtered using the standard 1.5-IQR rule, and 95% confidence intervals were generated via 10,000-resample bootstrapping.*
>
> *Furthermore, we evaluated accuracy across two distinct benchmarks: the standard 10-command subset of 4,074 clips, and the full real-world 12-class task of 12,005 clips including over 6,900 unknown words and 1,000 background silence recordings."*

**[Presenter Cues / Catatan Presenter (ID)]**
- Tekankan kontrol suhu dan clock: CPU governor di-pin ke *performance*.
- Sebutkan angka statistik: 1.000 warmup, 1.000 run berurutan, IQR outlier removal, 10k bootstrap CI.
- Jelaskan beda subset 10-command vs full 12-class test set.

---

```
================================================================================
SLIDE 5: ACCURACY EVALUATION [03:45 - 04:50]
Visual: Slide 5 (Comprehensive Accuracy Table, 3 key takeaway boxes)
================================================================================
```

**[Spoken Script - English]**
> *"On Slide 5, we examine the accuracy results.*
>
> *First, we established strict numerical parity: TFLite float32 and ONNX float32 produce identical top-1 classifications across all 12,005 clips, with maximum absolute logit differences below 3 times 10 to the minus 6.*
>
> *Second, look at the post-training quantization penalty. Convolutional representations demonstrate exceptional resilience: DS-CNN loses only 0.28 percentage points on the full 12-class task, while CRNN loses only 0.48 points.*
>
> *In stark contrast, recurrent architectures show severe sensitivity to standard affine post-training quantization. The LSTM drops by 3.05 points on the 10-command subset and over 4.1 points on the full task. This demonstrates that without Quantization-Aware Training, recurrent state accumulations suffer heavy compounding round-off errors in fixed-point representations."*

**[Presenter Cues / Catatan Presenter (ID)]**
- Poin 1: Paritas numerik sempurna antara TFLite dan ONNX di level float32.
- Poin 2: DS-CNN dan CRNN sangat tahan kuantisasi INT8 (<0.5% drop).
- Poin 3: LSTM anjlok >4% karena akumulasi rounding error pada recurrent cell.

---

```
================================================================================
SLIDE 6: LATENCY BENCHMARK & REVERSAL [04:50 - 05:55]
Visual: Slide 6 (Latency Table, Multi-model Bar Chart, Full-int8 callout)
================================================================================
```

**[Spoken Script - English]**
> *"Slide 6 presents our first major finding: the architectural reversal in inference latency.*
>
> *When running default float32 execution, ONNX Runtime outperforms TFLite on DS-CNN by 19%—achieving 0.1702 milliseconds compared to TFLite's 0.2107 milliseconds.*
>
> *However, this advantage does not generalize! On every other architecture, TensorFlow Lite decisively wins:*
> - *DNN is 1.81 times faster under TFLite;*
> - *LSTM is over 2 times faster—0.37 milliseconds versus 0.78 milliseconds;*
> - *And CRNN is 1.4 times faster.*
>
> *Why does this inversion happen? DS-CNN has 5 sequential depthwise-separable stages where kernel dispatch dominates; ONNX's aggressive graph optimizer fuses these operations effectively. Conversely, DNN and recurrent networks are bounded by dense matrix multiplication and sequential data dependencies, where TFLite's lean runtime overhead and optimized XNNPACK kernels achieve superior execution.*
>
> *Furthermore, full-int8 quantization on TFLite accelerates DS-CNN to just 0.1268 milliseconds—25% faster than ONNX float32."*

**[Presenter Cues / Catatan Presenter (ID)]**
- Sorot fenomena *Architectural Reversal*: ONNX hanya menang pada DS-CNN (rasio 0.81), tetapi kalah telak pada DNN, LSTM, dan CRNN (TFLite 1.4 - 2.1x lebih cepat).
- Jelaskan mekanismenya: *kernel fusion* vs *lightweight dispatch overhead* di XNNPACK.

---

```
================================================================================
SLIDE 7: THREAD SENSITIVITY SWEEP [05:55 - 07:05]
Visual: Slide 7 (Iso-thread table for T=1, 2, 4, two analysis cards)
================================================================================
```

**[Spoken Script - English]**
> *"Slide 7 reveals an even deeper architectural discovery.*
>
> *When we investigated why ONNX appeared faster on DS-CNN in default out-of-the-box configurations, we uncovered a hidden confound: ONNX Runtime silently spawns a 4-thread worker pool matching the Pi 5's quad-core count, whereas TFLite defaults to single-threaded execution.*
>
> *To isolate true kernel efficiency, we conducted an iso-threaded sweep pinning both runtimes to exactly 1, 2, and 4 threads.*
>
> *The result was striking: under strictly controlled iso-threaded conditions, TensorFlow Lite is faster than ONNX Runtime across ALL architectures at EVERY thread count.*
> *At T equals 1, TFLite executes DS-CNN in 0.2119 milliseconds versus 0.4379 milliseconds for ONNX—a 2.07 times speedup! At T equals 4, TFLite reaches 0.078 milliseconds versus 0.168 milliseconds for ONNX.*
>
> *Additionally, we observed diminishing returns: on CRNN, scaling TFLite from 2 to 4 threads actually increased latency from 0.187 to 0.219 milliseconds due to thread synchronization overhead exceeding kernel compute time on small sub-60-thousand parameter models."*

**[Presenter Cues / Catatan Presenter (ID)]**
- Ini adalah salah satu kontribusi orisinal terpenting paper: *The Threading Confound*.
- Dalam kondisi default, ONNX memakai 4 core sedangkan TFLite 1 core.
- Ketika di-pin sama-sama 1 core ($T=1$), TFLite sebenarnya 2.07x lebih kencang dari ONNX di DS-CNN!

---

```
================================================================================
SLIDE 8: ON-BOARD PMIC ENERGY MEASUREMENT [07:05 - 08:05]
Visual: Slide 8 (Energy table, PMIC telemetry note, Race-to-Sleep paradox box)
================================================================================
```

**[Spoken Script - English]**
> *"Next, we examine on-chip power and energy consumption. Rather than relying on external USB inline meters that conflate board peripherals, we tapped directly into the Raspberry Pi 5's internal DA9091 PMIC telemetry via the on-chip 12-channel ADC.*
>
> *This revealed a fascinating power versus energy paradox.*
>
> *If you look solely at active delta power, TFLite draws higher instantaneous power than ONNX—for instance, 1.59 Watts versus 1.18 Watts on DS-CNN.*
>
> *However, because TFLite finishes the inference workload twice as fast, total energy consumed per inference is 35% lower—0.34 millijoules for TFLite versus 0.53 millijoules for ONNX.*
>
> *This provides empirical proof of the 'Race-to-Sleep' paradigm on modern edge SBCs: pushing the processor to complete bursts at maximum efficiency and returning promptly to low-power idle saves substantially more battery life than running at lower instantaneous power for a longer duration."*

**[Presenter Cues / Catatan Presenter (ID)]**
- Sorot paradoks daya vs energi: TFLite *power*-nya lebih tinggi (1.59W vs 1.18W), tapi *energi total per inferensi* 35% lebih hemat (0.34 mJ vs 0.53 mJ).
- Hubungkan dengan konsep *Race-to-Sleep* di perangkat IoT/Edge berdaya baterai.

---

```
================================================================================
SLIDE 9: PARETO FRONTIER [08:05 - 08:50]
Visual: Slide 9 (Scatter plot accuracy vs parameter count, Trade-offs list)
================================================================================
```

**[Spoken Script - English]**
> *"Slide 9 visualizes the Accuracy versus Parameter Size Pareto Frontier.*
>
> *The four models occupy distinct architectural niches:*
> - *CRNN establishes the top accuracy frontier at 92.56% on the 10-command benchmark and 92.08% on the full 12-class task with 56.8 thousand parameters.*
> - *LSTM demonstrates high parameter efficiency—achieving over 90% accuracy with under 20 thousand parameters—though constrained by quantization drop.*
> - *DS-CNN occupies the overall sweet spot: 22.6 thousand parameters, 95.5% full-task accuracy, and sub-0.13 millisecond latency under full-int8 quantization.*
> - *Finally, the dense MLP baseline confirms that naive feed-forward structures scale poorly for spectral audio data."*

**[Presenter Cues / Catatan Presenter (ID)]**
- Tunjukkan posisi tiap model pada grafik Pareto.
- Sebutkan bahwa DS-CNN adalah *sweet spot* terbaik untuk efisiensi vs akurasi.

---

```
================================================================================
SLIDE 10: CONCLUSION & PRACTICAL GUIDELINES [08:50 - 09:35]
Visual: Slide 10 (4 Core Findings, Actionable Recommendation box)
================================================================================
```

**[Spoken Script - English]**
> *"To conclude, our study offers four clear guidelines for edge AI engineers and researchers:*
>
> *First: Runtime speedups do not generalize across model topologies. Defaulting to ONNX Runtime only benefits multi-threaded convolution graphs; for dense and recurrent structures, TensorFlow Lite remains 1.4 to 2.1 times faster.*
>
> *Second: Thread parity is paramount. When benchmarked on identical core allocations, TFLite consistently outperforms ONNX across all four architectures.*
>
> *Third: Full-int8 quantization provides an immediate 1.7 to 2.5 times latency reduction with negligible accuracy loss on convolutional and hybrid models, but recurrent models require quantization-aware training.*
>
> *And fourth: Prioritize completion latency over instantaneous power to minimize total energy drain via the Race-to-Sleep advantage."*

**[Presenter Cues / Catatan Presenter (ID)]**
- Rangkum 4 temuan utama secara ringkas, padat, dan berbobot.
- Tekankan pesan kunci: *"Choose your runtime based on operator distribution, not convention."*

---

```
================================================================================
SLIDE 11: OPEN SCIENCE & THANK YOU [09:35 - 10:00]
Visual: Slide 11 (Zenodo DOI, GitHub Repo, AGPLv3, Contact Info)
================================================================================
```

**[Spoken Script - English]**
> *"In the spirit of open science and full reproducibility, our complete pipeline—including model checkpoints, quantization recipes, telemetry logging tools, and the full test suite—is publicly available on GitHub under the GNU AGPLv3 license and permanently archived on Zenodo under DOI 10.5281/zenodo.23765521.*
>
> *Thank you very much for your time and attention. We look forward to your questions and discussions."*

**[Presenter Cues / Catatan Presenter (ID)]**
- Tutup dengan ucapan terima kasih kepada session chair dan audiens.
- Sebutkan link GitHub dan Zenodo DOI untuk meyakinkan reviewer/audiens bahwa riset ini 100% *reproducible*.

---

## Anticipated Q&A & Reviewer Discussion Guide

Below are the most probable technical questions that session chairs, reviewers, or attendees may ask, along with comprehensive, evidence-based answers.

---

### Question 1: "Why does ONNX Runtime appear faster than TensorFlow Lite on DS-CNN in default settings, but slower under thread-pinned settings?"

**Presenter Answer:**
> *"Thank you for this insightful question. This highlights our key finding regarding runtime default configurations.*
>
> *By default, ONNX Runtime inspects the host hardware and automatically instantiates an intra-op thread pool matching the machine's online core count—which is 4 cores on the Raspberry Pi 5. In contrast, TensorFlow Lite / LiteRT defaults to single-threaded execution unless explicitly configured via runtime threading options.*
>
> *When we controlled for this confound by pinning both engines to identical thread counts ($T=1, 2, 4$), TensorFlow Lite was faster across every single thread level. At $T=1$, TFLite achieved 0.2119 ms compared to 0.4379 ms for ONNX—over $2.07\times$ faster. The initial apparent advantage of ONNX on DS-CNN was entirely an artifact of 4 cores executing against 1 core, rather than intrinsic execution engine superiority."*

---

### Question 2: "Why did you replace the official MLPerf Tiny DS-CNN model with your own reconstructed version?"

**Presenter Answer:**
> *"When examining the official MLPerf Tiny reference submission (`kws_ref_model.tflite`), we inspected the internal flatbuffer schema and discovered it was not a homogeneous float32 model. Instead, it was an intermediate mixed-precision artifact where the core convolution operators were quantized to 8-bit integers while subsequent operators and tensor outputs remained float32.*
>
> *Attempting to convert this mixed-precision flatbuffer directly to ONNX produced non-standard dequantize-re-quantize operator chains that severely penalized ONNX execution. To maintain strict scientific parity, we decompiled the topology, reconstructed an identical true float32 Keras model, verified that its parameter count (22,604) and architecture matched the reference exactly, and then derived both pure ONNX and pure INT8 artifacts from that identical baseline."*

---

### Question 3: "Why did you report two different accuracy test sets (10-command subset vs 12-class full task)?"

**Presenter Answer:**
> *"In the literature, many TinyML papers report accuracy exclusively on the 10-target command subset (4,074 audio clips). While this tests wake-word discrimination, real-world deployment requires an edge model to reject arbitrary conversational speech and background ambient noise.*
>
> *Therefore, we benchmarked both:*
> 1. *The 10-command subset (4,074 clips) for direct comparison with prior TinyML publications; and*
> 2. *The full 12-class task (12,005 clips), which incorporates 6,931 'unknown' vocabulary utterances and 1,000 background silence/noise samples.*
>
> *Interestingly, this revealed that while LSTM scored 90.48% on the 10-command subset, its INT8 post-training quantization penalty escalated from $-3.05$ percentage points on the subset to $-4.12$ points on the full task, proving that open-vocabulary background noise rejection is substantially more vulnerable to quantization noise."*

---

### Question 4: "Why use internal PMIC ADC readings rather than standard external USB power meters?"

**Presenter Answer:**
> *"External USB inline meters (such as UM25C or MakerHawk meters) measure the total gross current drawn at the 5V input rail. On a single-board computer like the Raspberry Pi 5, this 5V reading includes board-level voltage regulators, HDMI display controllers, Ethernet PHYs, USB 3.0 bridge chips, and active cooling fans.*
>
> *These peripheral baselines inject significant noise and disguise subtle milliwatt differences during sub-millisecond inference bursts. By reading the DA9091 PMIC ADC directly via `vcgencmd pmic_read_adc`, we measured the internal core power rails ($V_{\text{core}}$, memory LPDDR4X, and SoC internal logic) at sub-millisecond granularity, isolating pure processor inference energy without peripheral skew."*

---

### Question 5: "Why did CRNN scale worse at 4 threads than at 2 threads under TensorFlow Lite?"

**Presenter Answer:**
> *"CRNN has 56,844 parameters, and its execution latency is already around 0.18 ms. At this ultra-low latency regime, dividing small matrix multiplications across 4 worker threads introduces thread barrier, cache ping-pong, and scheduling overhead that outweighs the parallel compute benefit.*
>
> *Specifically, on Cortex-A76 cores, thread synchronization and context handoff can take several microseconds. When kernel execution takes only tens of microseconds, multi-threading yields diminishing returns or net slowdowns. For compact edge models, 2 threads represents the empirical sweet spot."*

---

### Question 6: "What is the difference between Google LiteRT and legacy `tflite-runtime`?"

**Presenter Answer:**
> *"Google recently rebranded and re-architected TensorFlow Lite into 'LiteRT' (`ai-edge-litert`). LiteRT features updated XNNPACK SIMD micro-kernels optimized specifically for 64-bit Arm NEON and FP16 vector instructions on newer Armv8.2-A architectures such as the Cortex-A76.*
>
> *By using `ai-edge-litert 2.1.6`, our benchmark reflects the state-of-the-art production execution environment for edge deployments in 2026."*

---

### Question 7: "What are the practical deployment takeaways for battery-operated IoT devices?"

**Presenter Answer:**
> *"The primary takeaway is the validation of the 'Race-to-Sleep' strategy:*
>
> *Even though TFLite draws $35\%$ more peak power than ONNX ($1.59$ W vs $1.18$ W on DS-CNN), it finishes in less than half the time ($0.21$ ms vs $0.44$ ms). Because energy equals power integrated over time ($E = P \times t$), TFLite saves $35\%$ of total battery energy per audio inference.*
>
> *In an always-on device that wakes up once every 100 milliseconds to classify an audio frame, faster execution allows the CPU to spend over $99.8\%$ of its life cycle in ultra-low-power idle sleep states, extending battery longevity significantly."*
