<div align="center">

# MultiWindows: GUI Grounding Benchmark in Multi-Window Scenes

<p>
Controlled data generation, difficulty-level simulation, and unified evaluation for desktop GUI grounding.
</p>

<p>
  <a href="#quick-start">Quick Start</a> •
  <a href="#benchmark-at-a-glance">Results</a> •
  <a href="#visual-overview">Visual Overview</a> •
  <a href="#reproducibility">Reproducibility</a>
</p>

</div>

![Pipeline](assets/figure_pipeline.png)

## Overview

MultiWindows is a reproducible benchmark toolkit for evaluating GUI grounding models in realistic multi-window desktop environments.

It includes:
- sensitivity-controlled experiments for occlusion, semantic distraction, and clutter
- simulation difficulty levels from L1 to L5
- unified evaluation scripts for multiple local or API-based models
- automatic summary tables for cross-model comparison

## Benchmark At A Glance

Source: eval_outputs/benchmark_pivot.csv

### Difficulty Trend (L1 -> L5)

| Model | L1 | L2 | L3 | L4 | L5 |
|---|---:|---:|---:|---:|---:|
| infigui | 0.771 | 0.616 | 0.438 | 0.214 | 0.103 |
| os-atlas | 0.509 | 0.373 | 0.267 | 0.113 | 0.050 |
| seeclick | 0.207 | 0.091 | 0.040 | 0.018 | 0.010 |
| uground | 0.835 | 0.631 | 0.439 | 0.204 | 0.071 |
| ui-tars | 0.818 | 0.598 | 0.395 | 0.204 | 0.113 |

## Visual Overview

This section follows the same logic as the benchmark workflow: data design -> difficulty behavior -> controlled diagnosis -> qualitative case study.

### 1) Dataset Design And Generation Pipeline

| Pipeline | Distribution |
|---|---|
| ![Pipeline](assets/figure_pipeline.png) | ![Distribution](assets/figure_distribution.png) |

### 2) Difficulty-Level Behavior

![Difficulty](assets/difficulty_analysis.png)

### 3) Controlled Diagnostic Analysis

| Controlled Analysis | Performance Gap |
|---|---|
| ![Controlled](assets/controlled_analysis.png) | ![Performance Gap](assets/figure_gap.png) |

### 4) Qualitative Case Study

![Case Study](assets/case_study_viz.png)

## Quick Start

### Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch transformers pillow tqdm pandas python-dotenv
```

### Required Inputs

- metadata: sentence_sim.json
- wallpaper: windows_background.jpg
- GUI image root: windows/

### Run End To End

```bash
bash run_benchmark_complete.sh eval_outputs
```

### Evaluate Only (skip generation)

```bash
bash run_benchmark_complete.sh eval_outputs --skip-gen
```

### Evaluate Selected Models

```bash
bash run_benchmark_complete.sh eval_outputs --skip-gen "infigui seed ui-tars-api"
```

## Reproducibility

### Generate Sensitivity Data

```bash
python sensitivity_analysis/experiment_generator.py \
  --metadata sentence_sim.json \
  --wallpaper windows_background.jpg \
  --image_root windows \
  --out_dir eval_outputs/sensitivity \
  --traverse \
  --experiments all
```

### Generate Simulation Data (L1-L5)

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

### Summarize Benchmark

```bash
python eval/summarize_benchmark.py --output_dir eval_outputs
```

## Repository Layout

```text
.
├── assets/                              # Homepage figures
├── eval/                                # Evaluation and model adapters
├── sensitivity_analysis/                # Controlled sensitivity experiments
├── simulation/dataset_generator/        # Multi-window synthesis pipeline
├── run_benchmark_complete.sh            # End-to-end benchmark entry
└── eval_outputs/                        # Generated results and summaries
```

## Data And Outputs

```text
eval_outputs/
├── benchmark_summary.csv
├── benchmark_pivot.csv
├── sensitivity/
│   ├── baseline/single_window/
│   ├── occlusion/*
│   ├── semantic/*
│   └── clutter/*
└── simulation/
    ├── L1/
    ├── L2/
    ├── L3/
    ├── L4/
    └── L5/
```

## Privacy Checklist Before Open Source

- remove all secrets from .env
- replace machine-local absolute paths with env vars or config files
- avoid exposing private endpoint names or internal model identifiers
- keep generated artifacts and temporary files out of version control

## License

Please add your project license (for example MIT or Apache-2.0).
