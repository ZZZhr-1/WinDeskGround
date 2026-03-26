import argparse
import json
import pandas as pd
from pathlib import Path
from analysis_utils import load_experiment_results, analyze_sensitivity

def run_controlled_analysis(experiment_dir, prediction_file, out_dir):
    print(f"Analyzing Controlled Experiment in {experiment_dir}...")
    
    # Load results using the manifest-based loader
    results = load_experiment_results(experiment_dir, prediction_file)
    
    if not results:
        print("No results found.")
        return
        
    df = pd.DataFrame(results)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by experiment type
    experiments = df['experiment'].unique()
    
    for exp_name in experiments:
        print(f"Processing Experiment: {exp_name}")
        exp_data = df[df['experiment'] == exp_name]
        
        # Calculate stats per condition
        stats = exp_data.groupby('condition')['success'].mean().reset_index()
        
        # Save report
        csv_path = out_dir / f'controlled_{exp_name}.csv'
        stats.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path}")
        
    print(f"Controlled analysis complete. Results saved to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', required=True, help='Root directory of the experiment output (containing manifest.json)')
    parser.add_argument('--predictions', required=True, help='JSON file with predictions {image_path: [x,y,w,h]}')
    parser.add_argument('--out_dir', default='analysis_results', help='Output directory for CSV reports')
    
    args = parser.parse_args()
    
    run_controlled_analysis(args.experiment_dir, args.predictions, args.out_dir)

if __name__ == '__main__':
    main()
