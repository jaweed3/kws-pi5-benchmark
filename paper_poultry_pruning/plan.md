# Paper 2 Plan — Pruning–Quantization Co-Design (Ide 1)

> Status: READY TO EXECUTE — mulai 2 Sep setelah `paper_poultry_edge` submit AMICT 1 Sep. Jangan mulai sebelum itu.

**Goal:** Buktikan structured channel pruning (30/50%) sebelum INT8 PTQ bisa **memulihkan speedup** yang hilang (6–30× slowdown di Paper 1) — dengan effort solo minimal (reuse dataset + checkpoint + Pi 5 yang sudah ada).

**Approach:** Ambil 3 checkpoint FP32 >97% dari Paper 1 → prune channel via torch-pruning (DepGraph) → fine-tune 10 epoch → ONNX export → INT8 PTQ (ONNX Runtime QInt8) → benchmark 200 runs di Intel i5 + Pi 5 Cortex-A76 → Pareto baru.

**Why this works (mekanisme):** INT8 lambat di CPU karena dequantize INT8→FP32 sebelum matmul (tanpa VNNI). Pruning mengurangi jumlah channel → mengurangi jumlah matmul/dequantize calls secara linear.

---

## Phase 0 — Setup (2 Sep, 2 jam)

### Task 1: Clone tooling + verifikasi checkpoint Paper 1
```
pip install torch-pruning==1.6.x onnx onnxruntime thop
ls ~/research/paper_poultry_edge/models/fp32/  # harus ada mobilenetv2.pth, shufflenetv2.pth, efficientnet_b0.pth
ls ~/research/paper_poultry_edge/paper/main.tex
```
Jika checkpoint belum ada: `python ~/research/paper_poultry_edge/train_pipeline.py --eval-only`

### Task 2: Buat `code/pruning_pipeline.py` — skeleton
- Copy `~/research/paper_poultry_edge/train_pipeline.py` → `~/research/paper_poultry_pruning/code/pruning_pipeline.py`
- Tambah import: `import torch_pruning as tp`
- Fungsi baru: `prune_model(model, name, sparsity=0.3)` → pakai `tp.pruner.MetaPruner` L1-norm, prune LayerNorm/BatchNorm-coupled channels.
- Test: `python code/pruning_pipeline.py --smoke --model mobilenetv2 --sparsity 0.3` → harus print `Params: 2.2M → 1.5M`

---

## Phase 1 — Pruning + Fine-tune (3–7 Sep, di RTX 4060 100.102.6.78)

> ⚠️ Sebelum train: `ssh labti@100.102.6.78 "taskkill //IM lm-studio* //F; nvidia-smi"` — kill LM Studio biar VRAM 6.8GB free. Batch 32 biar 15 it/s.

### Task 3: E1 — Pruning sensitivity (6 runs)
Matrix: 3 model × 3 sparsity (0%, 30%, 50%) = 9 checkpoint, tapi 0% reuse Paper 1 → 6 prune baru.
```
for sparsity in 0.3 0.5:
  for model in mobilenetv2 shufflenetv2 efficientnet_b0:
    python code/pruning_pipeline.py --model $model --sparsity $sparsity --finetune 10 --lr 1e-5
```
Output: `results/pruning_sensitivity.json` → {model, sparsity, params, size_MB, acc, val_acc}
Expected: SNV2 paling tahan (97.4→96.8 di 30%), EB0 paling drop (98.3→96.5 di 50% karena SE block).

### Task 4: E4 — Fine-tune ablation (validasi 10 epoch cukup)
Untuk MN2-30% saja:
```
python code/pruning_pipeline.py --model mobilenetv2 --sparsity 0.3 --finetune 0  # no FT
python code/pruning_pipeline.py --model mobilenetv2 --sparsity 0.3 --finetune 5
python code/pruning_pipeline.py --model mobilenetv2 --sparsity 0.3 --finetune 10
```
Catat acc 0/5/10 epoch. Expected: 0 epoch drop 3–5%, 5 epoch recover 80%, 10 epoch plateau.

---

## Phase 2 — Quantization + Benchmark (8–10 Sep)

### Task 5: E2 — ONNX export + INT8 PTQ untuk semua pruned model
```
python code/pruning_pipeline.py --export-onnx --quantize --sparsity all
# generate: models/onnx/{model}_{sparsity}_fp32.onnx  +  models/ptq/{model}_{sparsity}_int8.onnx
ls -lh models/onnx/ models/ptq/
```
Expected size: MN2-30% FP32 ~6.0MB → INT8 ~1.7MB (3.5×).

### Task 6: Benchmark di Intel i5 (Mac) — 200 runs + 10 warmup
```
python code/pruning_pipeline.py --benchmark --hardware cpu --runs 200 --warmup 10
# output: results/bench_cpu.json → {model, sparsity, quant(fp32/int8), mean_ms, p50_ms, fps, raw_latencies[]}
```
Success criteria: MN2 INT8 dari 55.8ms (0%) → <15ms (30%) → slowdown 30× → <8×. SNV2 INT8 20.3ms → <8ms (2–3×).

### Task 7: Benchmark di Pi 5 (Cortex-A76) — cross-hardware
```
scp models/ptq/*.onnx techronity@192.168.119.43:/tmp/
ssh techronity@192.168.119.43 "python3 ~/bench_pruned.py --runs 200 --out /tmp/bench_pi5.json"
scp techronity@192.168.119.43:/tmp/bench_pi5.json results/bench_pi5.json
cat results/bench_pi5.json | python -m json.tool | head -n 40
```
Expected tren sama: pruning benefit konsisten di Pi 5, bahkan lebih besar karena Cortex-A76 lebih lemah dequantize.

---

## Phase 3 — Analysis + Figures (11–14 Sep)

### Task 8: Figure pipeline — 5 figure wajib
```
python code/gen_figures.py --all
# generates:
# figures/fig1_pruning_sensitivity.png  (acc vs sparsity, 3 lines)
# figures/fig2_latency_recovery.png     (grouped bar FP32 vs INT8 per sparsity — CORE FIGURE)
# figures/fig3_pareto_new.png           (acc vs latency + acc vs size, 15 titik vs 6 baseline)
# figures/fig4_finetune_ablation.png    (acc vs FT epoch)
# figures/fig5_cross_hardware.png       (Intel i5 vs Pi 5 side-by-side)
ls -lh figures/
```
Setiap figure harus punya caption angka, bukan "significantly better".

### Task 9: E3 — Pareto frontier baru
Script `gen_figures.py` otomatis hitung Pareto front (non-dominated points). Catat sweet spot:
- Latency-critical: MN2-30% FP32 (~1.4ms)
- Memory-constrained: SNV2-50% INT8 (~1.0MB)
- Accuracy-critical: EB0-0% FP32 (98.3%)

### Task 10: E6 optional — Grad-CAM delta (jika sempat)
```
python code/gen_gradcam_grid.py --models mobilenetv2 --sparsity 0 0.3 --classes cocci healthy ncd salmo
# output: figures/gradcam_pruned_vs_unpruned.png (2 row × 4 col)
```
Expected: fokus tetap di warna fecal, tidak drift ke background.

---

## Phase 4 — Writing (15–25 Sep)

### Task 11: Tulis paper LaTeX (IEEEtran 4–6 hal)
```
cp ~/research/paper_poultry_edge/paper/main.tex paper/main.tex  # reuse preamble
# edit sesuai outline.md: Title baru, Abstract baru, Method 3.3 pruning, Results 5.1-5.5
cd paper && tectonic main.tex && pdfinfo main.pdf | grep Pages
```
Target 5.0 halaman. Anti AI slop: setiap klaim ada angka dari `results/*.json`.

### Task 12: Submit MLCIPR (~5 Okt)
```
# cek deadline: curl -s https://iccipr.org | grep -i deadline
# upload paper/main.pdf ke EDAS, simpan confirmation
cp EDAS_confirmation.pdf ~/research/paper_poultry_pruning/paper/
```

---

## Validation Checklist (semua harus PASS sebelum submit)
- [ ] `results/pruning_sensitivity.json` ada 9 entries (3×3), acc >95% di 30% sparsity
- [ ] `results/bench_cpu.json` raw_latencies length 200 per entry
- [ ] MN2 INT8 slowdown turun dari 30.2× (Paper 1) → <10× (Paper 2)
- [ ] `tectonic main.tex` 0 error, Pages 4–6, semua \cite resolved (15+ refs)
- [ ] 5 figures PNG exist, 300 dpi, caption konsisten dengan Table angka

## Risks & Mitigasi
- **Pruning EB0 drop acc >3%:** Turunkan sparsity EB0 ke 20% (bukan 30%), atau ganti pruner ke Hessian-aware (HAP-E). Tulis di limitation.
- **Pi 5 disk 3GB free:** Kirim model on-demand via scp, jangan simpan Speech Commands di Pi.
- **LM Studio makan VRAM di lab:** Selalu kill sebelum train, atau pakai `CUDA_VISIBLE_DEVICES=0 python ...` dengan batch 8 fallback.
- **HAPE paper adalah LLM (bukan CNN):** Cite sebagai analogy hardware-aware, bukan claim kita pakai HAPE exact. Tool kita `torch-pruning`.

## 4–5 Ideas for Experiment (ringkas, detail di outline.md)
1. **E1 Pruning sensitivity** — 3 model × 30/50% → acc/size/FLOPs trade-off (wajib)
2. **E2 Latency recovery** — INT8 latency pruned vs unpruned di 2 hardware (CORE CLAIM)
3. **E3 Pareto baru** — 15 titik vs 6 baseline → sweet spot per constraint (wajib)
4. **E4 Fine-tune ablation** — 0/5/10 epoch → buktikan 10 epoch cukup solo (cepat)
5. **E5 Cross-hardware** — Intel i5 vs Pi 5 → generalisasi pruning benefit (kuat untuk reviewer edge)

## Trajectory Solo → S2 Dresden
- **Paper 1 (AMICT Sep 1):** Characterization — "INT8 slow di CPU"
- **Paper 2 (MLCIPR Okt / MCSoC Des):** Co-design — "Pruning recovers INT8 speedup" (ini)
- **Paper 3 (Q1 2027):** Real deployment — pruned winner → ESP32-S3 di kandang 5k ayam + power measure. Atau TinyNAS di atas pruned model untuk S2 proposal Barkhausen (Silicon Saxony).

## Handoff Note Template (jika terputus tengah Task)
```
Progress: Task 6 done 50% (MN2-30% bench ok, SNV2 pending)
Next 2-min: python code/pruning_pipeline.py --benchmark --model shufflenetv2 --sparsity 0.3 --quant int8
Blocker: none
Branch: paper_poultry_pruning/main
```
