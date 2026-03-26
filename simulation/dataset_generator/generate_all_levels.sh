#!/bin/bash

METADATA="sentence_sim.json"
WALLPAPER="windows_background.jpg"
N=5

# Ensure output directory exists
mkdir -p output

for LEVEL in L1 L2 L3; do
    echo "============================================"
    echo "Generating $N samples for difficulty $LEVEL..."
    echo "============================================"
    
    OUT_DIR="output/test_${LEVEL}"
    
    python simulation/dataset_generator/run.py \
        --metadata "$METADATA" \
        --out "$OUT_DIR" \
        --n $N \
        --difficulty "$LEVEL" \
        --wallpaper "$WALLPAPER"
        
    echo "Finished $LEVEL. Saved to $OUT_DIR"
    echo ""
done

echo "All levels generated successfully."
