# BACKUP & RECOVERY — kws-pi5-benchmark

## Backup utama (full history, semua branch)

- File: `~/backups/kws-pi5-benchmark/kws-pi5-benchmark_20260915_1930.bundle` (31MB)
- SHA256: `e07c9f5162f16cd4c44b98cff4be3b58aae52a7c22a32fd48f5b7473d237daa9`
- Isi: `main` + `revise-icaitech-minor`, verified (`git bundle verify` OK)
- Restore: `git clone ~/backups/kws-pi5-benchmark/kws-pi5-benchmark_20260915_1930.bundle kws-restore`

## Remote (lapis kedua)

- GitHub private `jaweed3/kws-pi5-benchmark`, branch `revise-icaitech-minor` = `7d1a81c`

## Snapshot anonim (revisi review, BUKAN backup)

- `/tmp/kws-anon-revision.zip` (3.6MB, 64 files) — KWS-only, scrubbed
  (author block, .doc, hostname techronity, path lokal). Leak-check NOL match.
- SEMENTARA di /tmp (hilang saat restart). Kalau mau arsipkan, pindahkan ke
  `~/backups/kws-pi5-benchmark/` sebelum restart.

## Yang TIDAK ikut backup manapun (regenerable, by design)

- `bench/data/` (SCv2 5.6GB + cache MFCC) — download ulang + `train_models.py`
- `literature/paywalled/` (18MB, lisensi) — ada lokal di mesin asal
- `data/dscnn_rep.npz`, `data/testlist_*.tsv` — hasil `make_rep_dscnn.py` / `make_testlists.py`
