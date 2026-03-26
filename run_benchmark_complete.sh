#!/bin/bash

# run_benchmark_complete.sh
# Usage: ./run_benchmark_complete.sh [output_dir]

export CUDA_VISIBLE_DEVICES=0,1,2,3

# ================= Configuration =================
OUTPUT_DIR=${1:-"eval_outputs"}
METADATA_PATH="/data/home/zhr/multiwindows/sentence_sim.json"
WALLPAPER_PATH="/data/home/zhr/multiwindows/windows_background.jpg"
IMAGE_ROOT="/data/home/zhr/multiwindows/windows"

# Sample Counts (Adjust as needed)
# For quick debug: use small numbers (e.g. 10, 20)
# For full run: use large numbers (e.g. 100, 200)
SAMPLES_PER_CONDITION=100
SAMPLES_PER_LEVEL=100

# Set to true to traverse all metadata samples as targets (overrides counts above)
TRAVERSE_MODE=true
if [ "$TRAVERSE_MODE" = true ]; then
    # Traverse mode args
    ARGS_SENS="--traverse"
    ARGS_SIM="--traverse"
    echo ">> [Config] Traverse Mode ENABLED (Iterating all metadata samples)"
else
    # Random sampling args
    ARGS_SENS="--n_per_condition $SAMPLES_PER_CONDITION"
    ARGS_SIM="--n $SAMPLES_PER_LEVEL"
    echo ">> [Config] Random Sampling Mode (Sens=$SAMPLES_PER_CONDITION, Sim=$SAMPLES_PER_LEVEL)"
fi

# Script Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="/data/home/zhr/multiwindows"
GEN_SCRIPT_SENS="$PROJECT_ROOT/sensitivity_analysis/experiment_generator.py"
GEN_SCRIPT_SIM="$PROJECT_ROOT/simulation/dataset_generator/run.py"
EVAL_CMD_GEN="$PROJECT_ROOT/eval/generate_commands.py"

# ================= Step 1: Data Generation =================
if [[ "$2" == "--skip-gen" ]]; then
    echo "########################################################"
    echo "# Step 1: Data Generation SKIPPED (via --skip-gen)"
    echo "########################################################"
else
    echo "########################################################"
    echo "# Step 1: Generating Datasets (Images won't be deleted yet)"
    echo "########################################################"

    # 1.1 Sensitivity Analysis
    echo ">> Generating Sensitivity Analysis Data..."
    python "$GEN_SCRIPT_SENS" \
        --metadata "$METADATA_PATH" \
        --wallpaper "$WALLPAPER_PATH" \
        --image_root "$IMAGE_ROOT" \
        --out_dir "$OUTPUT_DIR/sensitivity" \
        $ARGS_SENS \
        --experiments all
    # Note: Removed --model_path and --delete_images to keep images for offline eval

    # 1.2 Simulation Levels (L1-L5)
    DIFFICULTIES=("L1" "L2" "L3" "L4" "L5")
    echo ">> Generating Simulation Levels (L1-L5)..."
    for DIFF in "${DIFFICULTIES[@]}"; do
        echo "   Generating $DIFF..."
        python "$GEN_SCRIPT_SIM" \
            --metadata "$METADATA_PATH" \
            --wallpaper "$WALLPAPER_PATH" \
            --image_root "$IMAGE_ROOT" \
            --out "$OUTPUT_DIR/simulation/$DIFF" \
            $ARGS_SIM \
            --difficulty "$DIFF"
        # Note: Removed --model_path
    done
fi

# ================= Step 2: Multi-Model Evaluation =================
echo ""
echo "########################################################"
echo "# Step 2: Generating Evaluation Commands"
echo "########################################################"

EVAL_SCRIPT_NAME="run_eval_jobs.sh"

# Logic: If 3rd arg is provided, use it as models list. 
# Else if --skip-gen is used, default to new models (infigui seed ui-tars-api)
# Else (full run), default to all models (empty arg).

MODELS_ARG=""
if [[ -n "$3" ]]; then
    MODELS_ARG="--models $3"
elif [[ "$2" == "--skip-gen" ]]; then
    MODELS_ARG="--models infigui seed ui-tars-api"
    echo ">> [Config] Defaulting to NEW models for skip-gen mode: infigui seed ui-tars-api"
fi

python "$EVAL_CMD_GEN" --data_dir "$OUTPUT_DIR" $MODELS_ARG > "$EVAL_SCRIPT_NAME"

NUM_JOBS=$(wc -l < "$EVAL_SCRIPT_NAME")
echo "Generated $NUM_JOBS evaluation jobs in $EVAL_SCRIPT_NAME"

echo ""
echo "########################################################"
echo "# Step 3: Running Evaluations (Sequential or Parallel)"
echo "########################################################"

# Determine Number of GPUs based on CUDA_VISIBLE_DEVICES or nvidia-smi
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    IFS=',' read -ra ADDR <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#ADDR[@]}
    echo ">> [Config] Using CUDA_VISIBLE_DEVICES limit. Available GPUs: $NUM_GPUS"
else
    # Auto-detect number of GPUs
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    else
        NUM_GPUS=1
    fi
fi

# Fallback
if [[ -z "$NUM_GPUS" ]] || [[ "$NUM_GPUS" -eq 0 ]]; then
    NUM_GPUS=1
fi

echo "Total Evaluation Jobs: $NUM_JOBS"
echo "Concurrency Level: $NUM_GPUS"

# Check if 'parallel' is installed for speedup
if command -v parallel &> /dev/null; then
    echo "GNU Parallel found. Distributing jobs..."
    
    # Use GNU Parallel to distribute jobs
    # 1. grep -v '^#': Remove comment lines
    # 2. grep -v '^[[:space:]]*$': Remove empty lines
    # 3. tr -d '\r': Remove carriage returns (Windows compat)
    # 4. parallel: Run jobs. 
    #    CUDA_VISIBLE_DEVICES=$(({%} - 1)) maps the parallel slot (1-based) to 0-based index. 
    #    We use 'eval {}' to handle cases where parallel might quote the input string or if there are special chars.
    
    grep -v '^#' "$EVAL_SCRIPT_NAME" | grep -v '^[[:space:]]*$' | tr -d '\r' | \
    parallel --jobs "$NUM_GPUS" --bar 'export CUDA_VISIBLE_DEVICES=$(({%} - 1)); eval {}'
else
    echo "GNU Parallel NOT found. Running jobs sequentially (this might take a while)..."
    echo "If you have multiple GPUs, install 'parallel' to speed this up significantly."
    chmod +x "$EVAL_SCRIPT_NAME"
    ./"$EVAL_SCRIPT_NAME"
fi

# ================= Step 4: Cleanup (Optional) =================
# echo ""
# echo "########################################################"
# echo "# Step 4: Cleanup"
# echo "########################################################"
# read -p "Do you want to delete generated images to save space? [y/N] " -n 1 -r
# echo
# if [[ $REPLY =~ ^[Yy]$ ]]; then
#     echo "Deleting images in $OUTPUT_DIR..."
#     find "$OUTPUT_DIR" -name "*.jpg" -delete
#     find "$OUTPUT_DIR" -name "*.png" -delete
#     echo "Images deleted. JSON results preserved."
# else
#     echo "Images preserved."
# fi

echo ""
echo "Benchmark Completed Successfully!"
echo "Check results in $OUTPUT_DIR"
