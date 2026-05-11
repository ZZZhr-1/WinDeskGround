import os
import sys
import json
import argparse
import random
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add parent directory to path to import simulation modules
# We need to add 'mutilwindows/simulation/dataset_generator' to path. 
# This script is in mutilwindows/sensitivity_analysis/experiment_generator.py
# parents[0] = sensitivity_analysis
# parents[1] = mutilwindows
sys.path.append(str(Path(__file__).resolve().parents[1] / "simulation" / "dataset_generator"))
# Add eval directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "eval"))

from generator import generate_one_sample, save_sample
from loader import load_metadata

# BatchEvaluator is deprecated in favor of offline evaluation
# BatchEvaluator = None

def run_experiment(experiment_name, conditions, args, metadata, wallpaper, evaluator=None):
    print(f"Starting Experiment: {experiment_name}")
    
    # Build lookup for fast retrieval by image path (Critical for using similar_contents in sampler)
    path_to_meta = {m['image_path']: m for m in metadata}

    manifest = []
    all_samples = []
    base_out_dir = Path(args.out_dir) / experiment_name
    base_out_dir.mkdir(parents=True, exist_ok=True)

    
    if evaluator:
        evaluator.output_dir = base_out_dir
        evaluator.results = [] # Reset results for new experiment
        evaluator.total_samples = 0
        evaluator.correct_samples = 0
        evaluator.buffer = []
    
    summary = {}
    global_idx = 0
    
    for condition_name, config_override in conditions.items():
        print(f"  Generating condition: {condition_name}")
        
        # Create condition specific folder
        cond_dir = base_out_dir / condition_name
        cond_dir.mkdir(exist_ok=True)
        
        if evaluator:
             # Flush any previous data
            if evaluator.buffer:
                evaluator.flush()
            
            # Point to condition directory
            evaluator.output_dir = cond_dir
            evaluator.results = []
            evaluator.total_samples = 0
            evaluator.correct_samples = 0
            evaluator.buffer = []

    for condition_name, config_override in conditions.items():
        print(f"  Generating condition: {condition_name}")
        
        # Create condition specific folder
        cond_dir = base_out_dir / condition_name
        cond_dir.mkdir(exist_ok=True)
        
        # If evaluator is present, reset it for each condition
        if evaluator:
             # Flush any previous data
            if evaluator.buffer:
                evaluator.flush()
            
            # Point to condition directory
            evaluator.output_dir = cond_dir
            evaluator.results = []
            evaluator.total_samples = 0
            evaluator.correct_samples = 0
            evaluator.buffer = []

        # Generate batch
        # Determine number of samples
        n_samples = len(metadata) if args.traverse else args.n_per_condition
        if args.traverse:
            print(f"  Traverse mode: Generating {n_samples} samples per condition.")
            
        current_batch_samples = []
        for i in tqdm(range(n_samples), desc=f"{experiment_name}/{condition_name}"):
            try:
                # Use sequential index if traversing, else (if you want random) None or keep i for determinism
                # To match run.py logic:
                target_idx = i if args.traverse else None 

                # Generate sample
                out, debug, windows, cg = generate_one_sample(
                    metadata, 
                    wallpaper, 
                    difficulty='L3', # Base template
                    image_root=args.image_root,
                    path_to_meta=path_to_meta,
                    config_override=config_override,
                    target_index=target_idx
                )
                
                # Save
                img_path, debug_path, meta_path, rec = save_sample(
                    global_idx, 
                    cond_dir, 
                    out, 
                    debug, 
                    windows, 
                    cg
                )

                # Keep only absolute path or relative to output
                manifest.append({
                    'id': global_idx,
                    'experiment': experiment_name,
                    'condition': condition_name,
                    'image_path': str(img_path),
                    'meta_path': str(meta_path),
                    'config': config_override
                })
                
                rec['img_filename'] = os.path.basename(img_path)
                all_samples.append(rec)
                current_batch_samples.append(rec)
                
                if evaluator:
                    evaluator.add_sample(rec)

                global_idx += 1
                
            except Exception as e:
                print(f"Error generating sample {global_idx}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save partial test.json for this condition (Crucial for per-condition offline eval)
        with open(cond_dir / 'test.json', 'w') as f:
            json.dump(current_batch_samples, f, indent=2)

        # Post-Condition Cleanup: Delete images to save space
        if args.delete_images:
            print("  Cleaning up images for this condition...")
            for sample in current_batch_samples:
                try:
                    p = cond_dir / sample['img_filename']
                    if p.exists():
                        p.unlink()
                    # Also delete debug image
                    d = cond_dir / sample['img_filename'].replace('.jpg', '_debug.jpg')
                    if d.exists():
                        d.unlink()
                except Exception as e:
                     print(f"Error deleting {p}: {e}")

        # Flush results for this condition
        if evaluator:
            evaluator.flush()
            
            # Add to summary
            acc = evaluator.correct_samples / evaluator.total_samples if evaluator.total_samples > 0 else 0
            summary[condition_name] = {
                "accuracy": acc,
                "correct": evaluator.correct_samples,
                "total": evaluator.total_samples
            }
            print(f"  Condition {condition_name} Accuracy: {acc:.4f}")

    # Save manifest
    with open(base_out_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Save test.json for evaluation (though images might be gone)
    with open(base_out_dir / 'test.json', 'w') as f:
        json.dump(all_samples, f, indent=2)

    if evaluator:
        # Save experiment summary
        with open(base_out_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Experiment {experiment_name} completed. Summary saved to {base_out_dir / 'summary.json'}")
    else:
        print(f"Experiment {experiment_name} completed. Manifest and test.json saved.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True, help='Path to sentence_sim.json')
    parser.add_argument('--wallpaper', required=True, help='Path to wallpaper image')
    parser.add_argument('--out_dir', default='experiment_output', help='Output directory')
    parser.add_argument('--image_root', default=None, help='Root directory for window images')
    parser.add_argument('--n_per_condition', type=int, default=50, help='Samples per condition')
    parser.add_argument('--traverse', action='store_true', help='If set, traverse all metadata items as targets. Overrides n_per_condition.')
    parser.add_argument('--experiments', nargs='+', default=['all'], help='Which experiments to run')
    parser.add_argument('--model_path', default=None, help='Path to Qwen2-VL model for online evaluation')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for evaluation and cleanup')
    
    parser.add_argument('--delete_images', action='store_true', help='Delete images after evaluation to save space')
    
    args = parser.parse_args()
    
    # Load resources
    metadata = load_metadata(args.metadata)
    wallpaper = Image.open(args.wallpaper).convert('RGBA')
    
    # Initialize Evaluator if model path provided
    evaluator = None
    # if args.model_path:
    #     if BatchEvaluator:
    #         # Initialize with dummy output dir, will be updated in run_experiment
    #         evaluator = BatchEvaluator(args.model_path, args.out_dir, batch_size=args.batch_size)
    #     else:
    #         print("Error: BatchEvaluator not available. Please check imports.")
    #         return

    # Define Experiments
    experiments = {}
    
    # Define experiments first
    # 1. Occlusion Sensitivity
    experiments['occlusion'] = {
        'visible_100_100': {'visible_min': 1.0, 'visible_max': 1.0, 'n_windows': 2, 'sim_level': 1},
        'visible_80_90':  {'visible_min': 0.8, 'visible_max': 0.9, 'n_windows': 2, 'sim_level': 1},
        'visible_80_70':  {'visible_min': 0.7, 'visible_max': 0.8, 'n_windows': 2, 'sim_level': 1},
        'visible_70_50':  {'visible_min': 0.5, 'visible_max': 0.7, 'n_windows': 2, 'sim_level': 1},
        'visible_30_50':  {'visible_min': 0.3, 'visible_max': 0.5, 'n_windows': 2, 'sim_level': 1},
    }
    
    # B. Semantic Distraction
    experiments['semantic'] = {
        'sim_level_1': {'visible_min': 1.0, 'n_windows': 2, 'sim_level': 1},
        'sim_level_2': {'visible_min': 1.0, 'n_windows': 2, 'sim_level': 2},
        'sim_level_3': {'visible_min': 1.0, 'n_windows': 2, 'sim_level': 3},
        'sim_level_4': {'visible_min': 1.0, 'n_windows': 2, 'sim_level': 4},
        'sim_level_5': {'visible_min': 1.0, 'n_windows': 2, 'sim_level': 5}, 
    }
    
    # C. Clutter/Crowding
    experiments['clutter'] = {
        'win_2':  {'visible_min': 1.0, 'n_windows': 2,  'sim_level': 1},
        'win_3':  {'visible_min': 1.0, 'n_windows': 3,  'sim_level': 1},
        'win_5':  {'visible_min': 1.0, 'n_windows': 5,  'sim_level': 1},
        'win_8':  {'visible_min': 1.0, 'n_windows': 8,  'sim_level': 1},
        'win_12': {'visible_min': 1.0, 'n_windows': 12, 'sim_level': 1},
    }

    # 4. Baseline (Single Window)
    experiments['baseline'] = {
        'single_window': {'visible_min': 1.0, 'visible_max': 1.0, 'n_windows': 1, 'sim_level': 0}
    }
    
    to_run = experiments.keys() if 'all' in args.experiments else args.experiments
    
    
    for exp_name in to_run:
        if exp_name in experiments:
            run_experiment(exp_name, experiments[exp_name], args, metadata, wallpaper, evaluator=evaluator)
        else:
            print(f"Unknown experiment: {exp_name}")

if __name__ == '__main__':
    main()
