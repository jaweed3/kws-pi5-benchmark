#!/bin/bash
# Download open-access PDFs for Paper A — KWS audio edge benchmark
# Target: ~/research/paper_audio_kws/literature/pdfs/
set -e
cd "$(dirname "$0")/pdfs"

dl() {
  local url="$1"
  local name="$2"
  echo "=== $name ==="
  curl -sL --max-time 120 -o "$name" "$url" && echo "  OK ($(du -h "$name" | cut -f1))" || echo "  FAIL"
}

# --- Fundamental (pre-2021, but required) ---
dl "https://arxiv.org/pdf/1804.03209v1" "warden2018_speech_commands.pdf"
dl "https://arxiv.org/pdf/1711.07128v4" "zhang2018_hello_edge_ds-cnn.pdf"

# --- Recent (2021+) ---
dl "https://arxiv.org/pdf/2106.07597v2" "mlperf_tiny_2021.pdf"
dl "https://arxiv.org/pdf/2208.02765v1" "wang2022_pruning_quant_kws_mcu.pdf"
dl "https://arxiv.org/pdf/2303.05016v1" "quantization_characterization_2023.pdf"
dl "https://arxiv.org/pdf/2409.14803v1" "benchmarking_edge_ai_platforms_2024.pdf"
dl "https://arxiv.org/pdf/2509.07051v1" "kws_end_to_end_efficiency_2025.pdf"
dl "https://www.mdpi.com/2079-9292/14/24/4877/pdf" "edge_ai_survey_2025_mdpi.pdf"

echo "---"
echo "Done. Files:"
ls -la
