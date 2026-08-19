# Paper 3 Plan — Additive-Only Ternary CNN (MatMul-Free)

> Status: READY TO PLAN — eksekusi Nov 2026 setelah Paper 2 MLCIPR submit. Jangan mulai sebelum Paper 1 AMICT Sep 1.
> Goal: Buktikan ternary convolution -1/0/+1 additive-only bisa ganti MatMul di CNN poultry dengan drop <2 persen, size 5x lebih kecil, deploy di ESP32-S3.

**Approach:** Reuse 3 checkpoint >97 persen Paper 1, bungkus Conv2d depthwise/pointwise dengan TernaryConv2d wrapper (absmean + STE), fine-tune 20 epoch incremental (depthwise dulu baru expand), benchmark Pi5 + ESP32-S3, Pareto tiga generasi.

**Why incremental:** Depthwise paling tahan ternary (sparse local filter), kalau langsung all conv drop bisa 5 persen dan paper gagal. Stage A dulu = solo feasible.

---

## Phase 0 — Setup & Pemahaman Mekanisme (Nov Week 1, 3 hari)

### Task 1: Kumpulkan referensi ternary di literature folder
```
mkdir -p ~/research/paper_poultry_matmulfree/literature
# download 5 PDF:
# 2406.02528 MatMul-free LM, BitNet b1.58 2504.12285, bitnet.cpp report, Slender-Mamba COLING, MCUNet
cp ~/Jurnal\ Bacaan/*.pdf ~/research/paper_poultry_matmulfree/literature/  # jika ada
ls -lh ~/research/paper_poultry_matmulfree/literature/
```
Output: 5 PDF + satu halaman ringkasan perbedaan BitLinear vs TernaryConv2d.

### Task 2: Buat skeleton TernaryConv2d di code/ternary_conv.py (Smoke, belum train)
Wrapper minimal yang bisa di-import tanpa error.

```python
# ~/research/paper_poultry_matmulfree/code/ternary_conv.py
import torch, torch.nn as nn

def absmean_quantize(w):
    scale = w.abs().mean().clamp(min=1e-5)
    w_tern = (w / scale).round().clamp(-1, 1)  # -1/0/+1
    return w_tern, scale

class TernaryConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv
        self.weight = conv.weight
        self.bias = conv.bias
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

    def forward(self, x):
        w_tern, scale = absmean_quantize(self.weight)
        # STE: forward pakai tern, backward pakai FP32
        w_ste = (w_tern * scale - self.weight).detach() + self.weight
        # untuk smoke: tetap pakai F.conv2d dengan w_ste (nanti ganti additive kernel di Pi5/ESP32)
        return nn.functional.conv2d(x, w_ste, self.bias, self.stride, self.padding, self.dilation, self.groups)
```

Test smoke:
```
python -c "import torch; from code.ternary_conv import TernaryConv2d; print('ok')"
python code/ternary_conv.py  # test forward 1 batch 224x224
```
Expected: ok, output shape sama.

### Task 3: Buat script convert_model.py — ganti Conv di MN2 secara selektif
```
# code/convert_model.py — fungsi convert_mobilenetv2(stage='depthwise-only')
# depthwise = conv dengan groups == in_channels
# pointwise = 1x1 conv
# keep last classifier classifier[1] tetap FP32
```
Test: load checkpoint Paper 1 MN2, convert depthwise-only, cek params dan forward satu batch, simpan JSON size.

---

## Phase 1 — E1 Ternary Sensitivity per Block (Nov Week 1-2, 3 hari di RTX 4060)

### Task 4: E1 — Ukur sensitivity 3 model x 3 config tanpa fine-tune (zero-shot ternarization)
```
for model in mobilenetv2 shufflenetv2 efficientnet_b0:
  for config in depthwise-only pointwise-only all:
    python code/convert_model.py --model $model --config $config --no-finetune --eval test
```
Output: results/e1_sensitivity.json — acc, size MB, theoretical adds vs MACs per model config.
Expected: MN2 depthwise-only drop 1-2 persen, all drop 6-8 persen. SNV2 tahan lebih. EB0 SE drop paling gede di all.
Gagal gate: jika depthwise-only drop >3 persen, cek absmean scale per-channel vs per-tensor (fix ke per-channel).

### Task 5: Pilih Stage untuk E2 berdasar E1
Aturan: pilih config dengan drop <2 persen dan size <2MB untuk lanjut fine-tune. Kemungkinan pemenang depthwise-only. Tulis di results/e1_decision.md satu paragraf.

---

## Phase 2 — E2 Full Ternary Fine-tune (Nov Week 2-3, 5 hari di RTX 4060)

> Kill LM Studio sebelum train: ssh labti@100.102.6.78 "taskkill //IM lm-studio* //F; nvidia-smi"

### Task 6: Fine-tune Stage pemenang 20 epoch dari checkpoint Paper 1
```
python code/train_ternary.py --model mobilenetv2 --config depthwise-only --epochs 20 --lr 5e-5 --batch 32 --seed 42
python code/train_ternary.py --model shufflenetv2 --config depthwise-only --epochs 20 --lr 5e-5
python code/train_ternary.py --model efficientnet_b0 --config depthwise-only --epochs 20 --lr 5e-5
# 3 runs, tiap run save best val checkpoint ke models/ternary/{model}_depthwise.pth
# log history ke results/e2_history_{model}.json
# LR 5e-5 lebih kecil dari Paper1 1e-4 karena STE butuh update halus
```
Expected: MN2 96.5 persen (drop 1.3), SNV2 96.0 (drop 1.4), EB0 97.0 (drop 1.3). Jika drop >2, extend ke 30 epoch atau keep pointwise FP32.

### Task 7: E2 expanded — jika Stage A sukses, coba Stage B (depthwise+pointwise) satu model saja (MN2) sebagai ablation
```
python code/train_ternary.py --model mobilenetv2 --config depthwise+pointwise --epochs 20 --lr 3e-5
```
Tulis ablation table Stage A vs Stage B. Jika Stage B drop >4 persen, stop dan report di limitation (tidak perlu paksa).

---

## Phase 3 — E2 Benchmark & E3 ESP32 Deployment (Nov Week 3-4, 4 hari)

### Task 8: Export ONNX fake-quant ternary + benchmark Pi5
```
python code/export_ternary.py --model mobilenetv2 --config depthwise-only --export onnx
python code/benchmark_ternary.py --hardware cpu --runs 200 --warmup 10 --out results/bench_ternary_cpu.json
scp models/ternary/*.onnx techronity@192.168.119.43:/tmp/
ssh techronity@192.168.119.43 "python3 ~/bench_ternary.py --runs 200 --out /tmp/bench_ternary_pi5.json"
scp techronity@192.168.119.43:/tmp/bench_ternary_pi5.json results/
cat results/bench_ternary_pi5.json | python -m json.tool | head -n 50
```
Metric: latency Pi5, size MB, acc. Bandingkan dengan results Paper1 bench_cpu.json dan Paper2 bench_cpu.json (reuse).
Success: MN2 ternary Pi5 ~12ms vs INT8 55.8ms vs pruned+INT8 15ms. Size 1.7MB vs 8.48MB FP32.

### Task 9: E3 — Convert ke ESP32-S3 (TFLite-Micro / ESP-DL)
```
python code/export_ternary.py --target esp32 --format tflite
# generate models/esp32/{model}_ternary.tflite + arena estimate
# flash via esptool / esp-idf
idf.py -p /dev/ttyUSB0 flash monitor  # di lab atau via ssh jika ESP32 colok ke Pi5
# atau simulasi: jika tidak ada board, report theoretical via esp-dl quantization tool + thop adds
```
Metric: latency ms, power mW (USB meter 100 samples), arena KB, FPS.
Stretch goal — paper tetap valid kalau hanya Pi5, tapi tulis jujur di limitation dan appendix flash log.

### Task 10: Custom kernel detail — apa yang kita build bersama (ini jawab pertanyaan bisa bantu kernel ga)
Kita tidak tulis kernel Xtensa assembly dari nol. Kita tulis 3 lapis:
Lapis 1 Python TernaryConv2d (sudah Task2) — untuk training dan fake-quant.
Lapis 2 CPU additive kernel di Pi5 (optional optimization) — loop Python numba / torch custom op yang ganti matmul dengan add/sub berdasar w_tern == 1 / -1, skip dimana 0. Ini yang bikin latency Pi5 turun real bukan cuma teori. Aku bantu skeleton numba.
Lapis 3 ESP32 kernel — pakai ESP-NN add ops existing, tidak tulis intrinsic baru. Ternary weight di-pack 2-bit (00=0,01=1,11=-1) dan unpack di runtime jadi branch add/sub. Jika mentok, fallback simulasi tetap publishable.
Kita pair-program Task8-9: kamu pegang training, aku pegang kernel skeleton dan benchmark script. Estimasi kernel work 2-3 hari, bukan 2 minggu.

---

## Phase 4 — Analysis & Figures (Nov Week 4, 3 hari)

### Task 11: Generate figures 5 figure
```
python code/gen_figures_ternary.py --all
# figures/fig1_sensitivity.png (grouped bar acc per config)
# figures/fig2_latency_compare.png (FP32 vs INT8 vs pruned-INT8 vs ternary di Pi5 — CORE)
# figures/fig3_pareto_3gen.png (18 titik 2-panel)
# figures/fig4_esp32.png (Pi5 vs ESP32 latency/power)
# figures/fig5_gradcam_ternary.png (optional)
ls -lh figures/
```

### Task 12: E4 Pareto tiga generasi
Script gen_figures hitung frontier non-dominated. Tulis satu paragraf guideline: kapan pakai FP32 (accuracy), pruned-INT8 (balanced Pi5), ternary (ultra-low-power ESP32).

---

## Phase 5 — Writing (Dec Week 1-2, 2 minggu)

### Task 13: Tulis paper LaTeX reuse Paper1 preamble
```
cp ~/research/paper_poultry_edge/paper/main.tex paper/main.tex
# ganti Title, Abstract, Method 3.3 ternary design, Results 5.1-5.4
cd paper && tectonic main.tex && pdfinfo main.pdf | grep Pages
```
Target 5 halaman, anti AI slop setiap klaim ada angka dari results/*.json, cite 17 refs termasuk 2406.02528 dan bitnet.cpp.

### Task 14: Submit TinyML/MCSoC
Cek deadline wikicfp 2 minggu sebelum, upload paper/main.pdf ke EDAS, simpan confirmation.

---

## Validation Checklist
- [ ] results/e1_sensitivity.json 9 entries (3x3), depthwise-only drop <2 persen
- [ ] results/e2_history JSON 3 model fine-tune 20 epoch, best acc >95 persen
- [ ] results/bench_ternary_pi5.json raw 200 latencies per entry, size ternary ~1.7MB
- [ ] figures 5 PNG 300dpi, Fig2 core latency compare ada 4 bar per model
- [ ] tectonic 0 error Pages 4-6 semua cite resolved 17 refs
- [ ] ESP32 flash log atau theoretical fallback jujur di appendix

## Risks & Mitigasi
- Drop >5 persen all-conv ternary: fallback depthwise-only, tulis ablation, paper tetap jadi.
- ESP32 arena 510KB overflow: reduce input 224->160 atau channel last stage FP32, tulis trade-off.
- Kernel numba tidak speedup Pi5: tetap report fake-quant latency plus theoretical adds (adds = MACs - multiplies), tetap valid optimization paper.
- Over-scope QAT ternary: jangan, PTQ ternary + STE fine-tune 20 epoch cukup untuk conference.

## 5 Experiment Details Ringkas (untuk diskusi)
1 E1 sensitivity paling murah, wajib dulu untuk pilih stage
2 E2 full ternary core claim, 20 epoch dari checkpoint >97 persen
3 E3 ESP32 deployment showcase farm, stretch tapi bikin S2 story kuat
4 E4 Pareto tiga generasi flagship figure, warna per generasi
5 E5 Grad-CAM optional, bukti ternary tidak drift fokus
