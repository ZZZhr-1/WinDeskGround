import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from analysis_utils import load_dataset_from_dir

def run_slicing_analysis(dataset_dir, prediction_file, out_dir):
    print(f"Loading dataset from {dataset_dir}...")
    samples = load_dataset_from_dir(dataset_dir, prediction_file)
    print(f"Loaded {len(samples)} samples.")
    
    if not samples:
        print("No samples found.")
        return

    # Convert to DataFrame for easier slicing
    data = []
    for s in samples:
        row = s['metrics'].copy()
        row['id'] = s['id']
        row['success'] = s.get('success', False)
        row['iou'] = s.get('iou', 0.0)
        data.append(row)
        
    df = pd.DataFrame(data)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Analysis 1: Occlusion Sensitivity ---
    # Slice: Low Semantic Distraction (sim_count <= 1)
    # Binning: visible_ratio
    print("Analyzing Occlusion Sensitivity...")
    slice_occ = df[df['sim_count'] <= 1].copy()
    if not slice_occ.empty:
        # Bin visible_ratio
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
        labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        slice_occ['vis_bin'] = pd.cut(slice_occ['visible_ratio'], bins=bins, labels=labels)
        
        stats_occ = slice_occ.groupby('vis_bin')['success'].mean().reset_index()
        stats_occ.to_csv(out_dir / 'sensitivity_occlusion.csv', index=False)
        print("Saved sensitivity_occlusion.csv")
    else:
        print("Not enough data for Occlusion slice (sim_count <= 1)")

    # --- Analysis 2: Semantic Distraction ---
    # Slice: High Visibility (visible_ratio > 0.8)
    # Binning: sim_count
    print("Analyzing Semantic Distraction...")
    slice_sem = df[df['visible_ratio'] > 0.8].copy()
    if not slice_sem.empty:
        stats_sem = slice_sem.groupby('sim_count')['success'].mean().reset_index()
        stats_sem.to_csv(out_dir / 'sensitivity_semantic.csv', index=False)
        print("Saved sensitivity_semantic.csv")
    else:
        print("Not enough data for Semantic slice (visible_ratio > 0.8)")

    # --- Analysis 3: Scale Sensitivity ---
    # Slice: High Visibility, Low Distraction
    print("Analyzing Scale Sensitivity...")
    slice_scale = df[(df['visible_ratio'] > 0.8) & (df['sim_count'] <= 1)].copy()
    if not slice_scale.empty:
        # Bin target_rel_screen
        # It's usually small, e.g. 0.01 to 0.2
        # Let's use quantiles or fixed small bins
        bins = [0, 0.01, 0.05, 0.1, 0.2, 1.0]
        labels = ['<1%', '1-5%', '5-10%', '10-20%', '>20%']
        slice_scale['scale_bin'] = pd.cut(slice_scale['target_rel_screen'], bins=bins, labels=labels)
        
        stats_scale = slice_scale.groupby('scale_bin')['success'].mean().reset_index()
        stats_scale.to_csv(out_dir / 'sensitivity_scale.csv', index=False)
        print("Saved sensitivity_scale.csv")
    else:
        print("Not enough data for Scale slice")

    # --- Analysis 4: Error Attribution Matrix ---
    # For failed samples, categorize the likely cause
    print("Generating Error Attribution Matrix...")
    failed = df[df['success'] == False].copy()
    
    def attribute_error(row):
        reasons = []
        if row['visible_ratio'] < 0.4:
            reasons.append('Occlusion')
        if row['sim_count'] >= 2:
            reasons.append('Semantic')
        if row['target_rel_screen'] < 0.01:
            reasons.append('SmallScale')
        if row['n_windows'] > 10:
            reasons.append('Clutter')
            
        if not reasons:
            return 'Unknown/Other'
        return '+'.join(reasons)

    if not failed.empty:
        failed['primary_cause'] = failed.apply(attribute_error, axis=1)
        attribution = failed['primary_cause'].value_counts().reset_index()
        attribution.columns = ['cause', 'count']
        attribution.to_csv(out_dir / 'error_attribution.csv', index=False)
        print("Saved error_attribution.csv")
        
    print(f"Analysis complete. Results saved to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True, help='Directory containing desktop_*.json files (recursive)')
    parser.add_argument('--predictions', required=True, help='JSON file with predictions {image_path: [x,y,w,h]}')
    parser.add_argument('--out_dir', default='analysis_results', help='Output directory for CSV reports')
    
    args = parser.parse_args()
    
    run_slicing_analysis(args.dataset_dir, args.predictions, args.out_dir)

if __name__ == '__main__':
    main()
