import os
import argparse

def generate_commands(base_output_dir, model_configs):
    """
    base_output_dir: directory where experiment_generator outputs are located
    model_configs: dict of {model_name: {path: ..., type: ...}}
    """
    
    # Define experiment paths structure based on experiment_generator.py and run.py
    # Structure:
    # <base_output_dir>/sensitivity/<exp_group>/<condition>/test.json
    # <base_output_dir>/sensitivity/baseline/single_window/test.json
    # <base_output_dir>/simulation/<level>/test.json
    
    cmds = []
    
    # 1. Sensitivity Analysis Experiments
    sensitivity_dir = os.path.join(base_output_dir, "sensitivity")
    if os.path.exists(sensitivity_dir):
        for exp_group in os.listdir(sensitivity_dir): # e.g., occlusion, semantic, baseline
            group_path = os.path.join(sensitivity_dir, exp_group)
            if not os.path.isdir(group_path): continue
            
            if exp_group == 'manifest.json' or exp_group == 'summary.json': continue

            for condition in os.listdir(group_path): # e.g., visible_90_100
                cond_path = os.path.join(group_path, condition)
                test_file = os.path.join(cond_path, "test.json")
                
                if os.path.exists(test_file):
                    for model_name, conf in model_configs.items():
                        out_file = os.path.join(cond_path, f"eval_{model_name}.json")
                        
                        cmd = f"python eval/evaluate_model.py --model_type {conf['type']} --model_path {conf['path']} --data_file {test_file} --image_root {cond_path} --output_file {out_file}"
                        cmds.append(cmd)

    # 2. Simulation Experiments (levels)
    simulation_dir = os.path.join(base_output_dir, "simulation")
    if os.path.exists(simulation_dir):
        for level in os.listdir(simulation_dir): # L1, L2...
            level_path = os.path.join(simulation_dir, level)
            test_file = os.path.join(level_path, "test.json")
            
            if os.path.exists(test_file):
                for model_name, conf in model_configs.items():
                    out_file = os.path.join(level_path, f"eval_{model_name}.json")
                    
                    cmd = f"python eval/evaluate_model.py --model_type {conf['type']} --model_path {conf['path']} --data_file {test_file} --image_root {level_path} --output_file {out_file}"
                    cmds.append(cmd)

    return cmds

if __name__ == "__main__":
    # Example Configuration
    # You should update paths to your actual model locations
    MODELS = {
        "os-atlas": {"type": "os-atlas", "path": "/data/home/zhr/models/OS-Atlas-Base-7B"},
        "seeclick": {"type": "seeclick", "path": "/data/home/zhr/models/SeeClick"}, 
        "uground":  {"type": "uground", "path": "/data/home/zhr/models/UGround-V1-7B"},
        "ui-tars":  {"type": "ui-tars", "path": "/data/home/zhr/models/UI-TARS-1.5-7B"},
        "infigui":  {"type": "infigui", "path": "/data/home/zhr/models/InfiGUI-G1-7B"},
        "seed":     {"type": "seed", "path": "doubao-seed-1-6-vision-250815"},
        "ui-tars-api": {"type": "ui-tars-api", "path": "doubao-1-5-ui-tars-250428"}
    }
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="experiment_results", help="Root dir of generated data")
    parser.add_argument("--models", nargs="+", help="List of models to generate commands for. If not specified, all models are used.")
    args = parser.parse_args()
    
    # Filter models if specified
    if args.models:
        # Validate keys
        valid_keys = set(MODELS.keys())
        req_keys = set(args.models)
        invalid = req_keys - valid_keys
        if invalid:
            print(f"Warning: The following requested models are not defined: {invalid}")
        
        # Filter
        active_models = {k: v for k, v in MODELS.items() if k in req_keys}
    else:
        active_models = MODELS

    commands = generate_commands(args.data_dir, active_models)
    
    print(f"# Generated {len(commands)} evaluation commands.")
    print("# You can run them in parallel. Example using GNU parallel:")
    print("# cat commands.txt | parallel -j 4")
    print("")
    for cmd in commands:
        print(cmd)
