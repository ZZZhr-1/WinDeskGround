#!/bin/bash

# Ensure we are running from the directory where the script is located
cd "$(dirname "$0")"

# Example usage of the experiment generator
# Make sure you have sentence_sim.json and windows_background.jpg in the root or adjust paths

METADATA="../sentence_sim.json"
WALLPAPER="../windows_background.jpg"
OUT_DIR="experiment_data"

# Generate 50 samples per condition for all experiments
python3 experiment_generator.py \
    --metadata "$METADATA" \
    --wallpaper "$WALLPAPER" \
    --out_dir "$OUT_DIR" \
    --image_root "../windows" \
    --n_per_condition 5 \
    --experiments all

echo "Experiments generated in $OUT_DIR"
