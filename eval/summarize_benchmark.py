import os
import json
import argparse
import pandas as pd
from pathlib import Path

def summarize_benchmark(output_dir):
    output_dir = Path(output_dir)
    data = []

    # Walk through to find evaluation json files
    # Pattern: eval_<model_name>.json
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.startswith('eval_') and f.endswith('.json'):
                model_name = f[5:-5] # remove eval_ and .json
                filepath = Path(root) / f
                
                try:
                    with open(filepath, 'r') as json_file:
                        res = json.load(json_file)
                    
                    # Determine context (Sensitivity vs Simulation)
                    rel_path = Path(root).relative_to(output_dir)
                    parts = rel_path.parts
                    
                    category = parts[0] if parts else "unknown"
                    
                    if category == 'sensitivity':
                        # sensitivity/<experiment>/<condition>
                        if len(parts) >= 3:
                            experiment = parts[1]
                            condition = parts[2]
                        else:
                            experiment = "unknown"
                            condition = str(rel_path)
                    elif category == 'simulation':
                        # simulation/<level>
                        experiment = "difficulty"
                        if len(parts) >= 2:
                            condition = parts[1] # L1, L2...
                        else:
                            condition = str(rel_path)
                    else:
                        experiment = category
                        condition = str(rel_path)

                    entry = {
                        "Model": model_name,
                        "Category": category,
                        "Experiment": experiment,
                        "Condition": condition,
                        "Accuracy": res.get("accuracy", 0.0),
                        "Total": res.get("total", 0)
                    }
                    data.append(entry)
                    
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

    if not data:
        print("No evaluation results found.")
        return

    df = pd.DataFrame(data)
    
    # Sort
    df = df.sort_values(by=["Category", "Experiment", "Condition", "Model"])
    
    print("\nBenchmark Summary:")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    
    # Save CSV
    csv_path = output_dir / "benchmark_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to {csv_path}")

    # Pivot table for better comparing models
    try:
        pivot = df.pivot_table(index=["Category", "Experiment", "Condition"], columns="Model", values="Accuracy")
        print("\nAccuracy Comparison Table:")
        print("=" * 100)
        print(pivot.to_string())
        print("=" * 100)
        pivot.to_csv(output_dir / "benchmark_pivot.csv")
    except Exception as e:
        print(f"Could not create pivot table: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='../eval_outputs_sim_only_wo_pre', help='Root directory of eval outputs')
    args = parser.parse_args()
    
    summarize_benchmark(args.output_dir)
