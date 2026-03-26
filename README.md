# MultiWindows Benchmark for GUI Grounding

A reproducible benchmark toolkit for evaluating GUI grounding models under multi-window scenes.

This repository provides:
- controlled data generation for sensitivity analysis
- difficulty-level simulation tasks (L1-L5)
- unified multi-model evaluation scripts
- result summarization utilities

## Highlights

- Controlled experiment axes: occlusion, semantic distraction, clutter
- Difficulty curriculum: L1 to L5 simulation levels
- Unified evaluation pipeline for local/API models
- Batch command generation for parallel GPU execution

## Project Structure

```text
.
├── run_benchmark_complete.sh          # End-to-end pipeline
├── eval/
│   ├── evaluate_model.py              # Evaluate one model on one test set
│   ├── generate_commands.py           # Generate evaluation job commands
│   ├── summarize_benchmark.py         # Aggregate results into tables
│   ├── models.py                      # Model adapters
│   └── utils.py                       # Output parsing helpers
├── sensitivity_analysis/
│   ├── experiment_generator.py        # Sensitivity dataset generation
│   ├── controlled_analysis.py         # Controlled-condition analysis
│   ├── slicing_analysis.py            # Slicing analysis
│   └── run_experiments.sh
└── simulation/dataset_generator/
    ├── run.py                         # Difficulty-level dataset generation
    ├── generator.py                   # Core sample generation
    ├── config.py                      # Difficulty configs
    └── README.md
```

## Quick Start

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch transformers pillow tqdm pandas python-dotenv
```

If you use API-based models, also install the matching SDK required by your adapter implementation.

### 2. Prepare Assets

Expected key inputs (currently used by the scripts):
- metadata json: sentence_sim.json
- wallpaper image: windows_background.jpg
- window image root: windows/

### 3. Run Full Benchmark

```bash
bash run_benchmark_complete.sh eval_outputs
```

### 4. Skip Data Generation and Only Evaluate

```bash
bash run_benchmark_complete.sh eval_outputs --skip-gen
```

### 5. Evaluate Specific Models Only

```bash
bash run_benchmark_complete.sh eval_outputs --skip-gen "infigui seed ui-tars-api"
```

## Reproducible Commands

### Generate Sensitivity Data Only

```bash
python sensitivity_analysis/experiment_generator.py \
  --metadata sentence_sim.json \
  --wallpaper windows_background.jpg \
  --image_root windows \
  --out_dir eval_outputs/sensitivity \
  --traverse \
  --experiments all
```

### Generate Simulation Data by Difficulty

```bash
for d in L1 L2 L3 L4 L5; do
  python simulation/dataset_generator/run.py \
    --metadata sentence_sim.json \
    --wallpaper windows_background.jpg \
    --image_root windows \
    --out eval_outputs/simulation/$d \
    --traverse \
    --difficulty $d
done
```

### Generate Evaluation Jobs

```bash
python eval/generate_commands.py --data_dir eval_outputs > run_eval_jobs.sh
bash run_eval_jobs.sh
```

### Summarize Results

```bash
python eval/summarize_benchmark.py --output_dir eval_outputs
```

## Output Layout

```text
eval_outputs/
├── sensitivity/
│   ├── occlusion/<condition>/
│   ├── semantic/<condition>/
│   ├── clutter/<condition>/
│   └── baseline/single_window/
└── simulation/
    ├── L1/
    ├── L2/
    ├── L3/
    ├── L4/
    └── L5/
```

Each condition/level directory contains generated samples and evaluation files such as:
- test.json
- eval_<model>.json

## Configuration Notes

Current scripts include machine-local absolute paths in some places (for example model paths and project root). Before open-sourcing, replace them with one of:
- environment variables
- CLI arguments
- a versioned config template (without secrets)

## Privacy and De-identification Checklist

Before pushing to GitHub:
- remove all secrets from .env
- avoid committing local absolute paths
- avoid exposing private model endpoints or internal model IDs
- exclude generated artifacts and temporary outputs

## License

Add your license here (for example MIT or Apache-2.0).

## Citation

If this benchmark supports a paper or report, add citation info here.
