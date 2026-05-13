import json
import argparse
import os
import sys
from tqdm import tqdm
from PIL import Image
import warnings

# Add parent directory to path to allow imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model

warnings.filterwarnings("ignore")

def evaluate_dataset(model, data_path, image_root, output_path):
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return

    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Filter or process if needed. The structure is list of dicts.
    # Each item has "img_filename", "instruction", "bbox" (absolute or [x,y,w,h])?
    # Based on generator.py, 'bbox' is [x, y, w, h] (absolute pixels)
    
    results = []
    correct_count = 0
    total_count = 0
    
    for item in tqdm(data, desc=f"Evaluating {os.path.basename(data_path)}"):
        try:
            filename = item.get("img_filename") or item.get("image_path")
            if os.path.isabs(filename) and image_root not in filename:
                 img_path = filename
            else:
                 img_path = os.path.join(image_root, os.path.basename(filename))
            
            if not os.path.exists(img_path):
                if os.path.exists(item.get("image_path")):
                    img_path = item.get("image_path")
                else: 
                    # print(f"Image not found: {img_path}")
                    continue

            instruction = item.get("instruction")
            if not instruction: continue
            
            gt_bbox_raw = item.get("bbox") # [x, y, w, h] from generator.py
            if not gt_bbox_raw and "gt_bbox" in item:
                 # fallback if 'bbox' missing but 'gt_bbox' present [x1, y1, x2, y2]
                 x1, y1, x2, y2 = item["gt_bbox"]
                 gt_bbox_raw = [x1, y1, x2-x1, y2-y1]

            if not gt_bbox_raw:
                # Last resort: try computing from metadata if available (rare in unified json)
                if "windows" in item:
                     target = next((w for w in item['windows'] if w.get('is_target')), None)
                     if target and 'target_bbox_proj' in target:
                         x1, y1, x2, y2 = target['target_bbox_proj']
                         gt_bbox_raw = [x1, y1, x2-x1, y2-y1]
            
            if not gt_bbox_raw: continue

            # Get Image Size for normalization
            with Image.open(img_path) as img:
                w, h = img.size
            
            # Normalize to [x_min, y_min, x_max, y_max] relative (0-1)
            raw_x, raw_y, raw_w, raw_h = gt_bbox_raw
            gt_norm = [
                raw_x / w,
                raw_y / h,
                (raw_x + raw_w) / w,
                (raw_y + raw_h) / h
            ]
            
            # Ensure valid range
            gt_norm = [min(max(v, 0.0), 1.0) for v in gt_norm]
            
            # Predict
            pred_point, raw_output = model.predict(img_path, instruction)
            
            is_correct = False
            if pred_point:
                px, py = pred_point
                if gt_norm[0] <= px <= gt_norm[2] and gt_norm[1] <= py <= gt_norm[3]:
                    is_correct = True
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            results.append({
                "id": item.get("id", total_count),
                "image_path": img_path,
                "instruction": instruction,
                "gt_bbox_norm": gt_norm,
                "pred_point": pred_point,
                "is_correct": is_correct,
                "raw_output": raw_output
            })
            
        except Exception as e:
            print(f"Error processing item: {e}")
            import traceback
            traceback.print_exc()

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Evaluation Complete. Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct_count,
            "total": total_count,
            "results": results
        }, f, indent=2)
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate GUI Grounding Models")
    parser.add_argument('--model_type', required=True, choices=['os-atlas', 'uground', 'seeclick', 'ui-tars', 'infigui', 'seed', 'ui-tars-api'], help="Model type")
    parser.add_argument('--model_path', required=True, help="Path to model weights")
    parser.add_argument('--data_file', required=True, help="Path to test.json file")
    parser.add_argument('--image_root', default='', help="Root directory for images if paths are relative")
    parser.add_argument('--output_file', required=True, help="Path to save result json")
    
    args = parser.parse_args()
    
    # Initialize Model
    model = get_model(args.model_type, args.model_path)
    
    # Run Evaluation
    evaluate_dataset(model, args.data_file, args.image_root, args.output_file)

if __name__ == "__main__":
    main()
