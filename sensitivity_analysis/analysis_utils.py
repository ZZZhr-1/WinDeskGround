import json
import numpy as np
from pathlib import Path

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    Boxes are (x, y, w, h).
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to x1, y1, x2, y2
    b1_x1, b1_y1, b1_x2, b1_y2 = x1, y1, x1 + w1, y1 + h1
    b2_x1, b2_y1, b2_x2, b2_y2 = x2, y2, x2 + w2, y2 + h2
    
    # Intersection
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Union
    b1_area = w1 * h1
    b2_area = w2 * h2
    union_area = b1_area + b2_area - inter_area
    
    if union_area == 0:
        return 0.0
        
    return inter_area / union_area

def load_experiment_results(experiment_dir, prediction_file):
    """
    Load ground truth from manifest and predictions from a file.
    prediction_file should be a JSON mapping image_path -> predicted_bbox [x,y,w,h]
    """
    manifest_path = Path(experiment_dir) / 'manifest.json'
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    with open(prediction_file, 'r') as f:
        predictions = json.load(f)
        
    results = []
    for item in manifest:
        img_path = item['image_path']
        
        # Load GT
        with open(item['meta_path'], 'r') as f:
            meta = json.load(f)
            
        gt_bbox = meta.get('gt_bbox') # [x1, y1, x2, y2] format in JSON usually? 
        # Wait, generator saves as tuple(map(int, target_bbox_proj)) which is x1,y1,x2,y2
        # But calculate_iou expects x,y,w,h? 
        # Let's check generator.py: project_bbox returns x1,y1,x2,y2.
        # So gt_bbox is x1,y1,x2,y2.
        
        # Convert GT to x,y,w,h for IoU function if needed, or adjust IoU function.
        # Let's adjust IoU function to handle x1,y1,x2,y2 if that's what we have.
        # Actually, let's standardize.
        
        if gt_bbox:
            gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1
            gt_rect = (gt_x1, gt_y1, gt_w, gt_h)
        else:
            gt_rect = None
            
        pred_rect = predictions.get(img_path) # Assuming [x,y,w,h] or [x1,y1,x2,y2]?
        # User needs to define this.
        
        iou = 0.0
        if gt_rect and pred_rect:
            # Assuming pred is also x,y,w,h for now
            iou = calculate_iou(gt_rect, pred_rect)
            
        item['iou'] = iou
        item['success'] = (iou >= 0.5)
        results.append(item)
        
    return results

def analyze_sensitivity(results, variable_name):
    """
    Group results by condition and calculate mean success rate.
    """
    groups = {}
    for r in results:
        cond = r['condition']
        if cond not in groups:
            groups[cond] = []
        groups[cond].append(r['success'])
        
    stats = {}
    for cond, values in groups.items():
        stats[cond] = sum(values) / len(values)
        
    return stats

def load_dataset_from_dir(dataset_dir, prediction_file=None):
    """
    Load all desktop_*.json files from a directory (recursive) and optionally match with predictions.
    Returns a list of sample dictionaries with 'metrics' populated.
    """
    dataset_dir = Path(dataset_dir)
    json_files = sorted(list(dataset_dir.rglob('desktop_*.json')))
    
    predictions = {}
    if prediction_file and Path(prediction_file).exists():
        with open(prediction_file, 'r') as f:
            predictions = json.load(f)
            
    samples = []
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                meta = json.load(f)
                
            # Basic info
            img_path = meta.get('image_path', '')
            # Normalize path for matching if needed (e.g. absolute vs relative)
            # For now assume exact string match or filename match
            
            # Extract Metrics
            target_win = next((w for w in meta['windows'] if w.get('is_target')), None)
            
            if not target_win:
                continue
                
            # 1. Occlusion: Visible Ratio
            visible_ratio = target_win.get('visible_ratio', 1.0)
            
            # 2. Scale: Relative Screen Area
            target_rel_screen = target_win.get('target_rel_screen', 0.0)
            
            # 3. Clutter: Number of windows
            n_windows = len(meta['windows'])
            
            # 4. Semantic: Similarity Count
            # Count how many other windows have the same 'type'
            target_type = target_win.get('type', 'unknown')
            sim_count = sum(1 for w in meta['windows'] if not w.get('is_target') and w.get('type') == target_type)
            
            sample = {
                'id': jf.stem,
                'image_path': img_path,
                'meta_path': str(jf),
                'metrics': {
                    'visible_ratio': visible_ratio,
                    'target_rel_screen': target_rel_screen,
                    'n_windows': n_windows,
                    'sim_count': sim_count,
                    'target_type': target_type
                }
            }
            
            # Match Prediction
            # Try full path match first, then filename match
            pred_rect = predictions.get(img_path)
            if pred_rect is None:
                pred_rect = predictions.get(Path(img_path).name)
                
            # Calculate IoU
            gt_bbox = meta.get('gt_bbox') # [x1, y1, x2, y2]
            if gt_bbox:
                gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
                gt_w = gt_x2 - gt_x1
                gt_h = gt_y2 - gt_y1
                gt_rect = (gt_x1, gt_y1, gt_w, gt_h)
                
                if pred_rect:
                    iou = calculate_iou(gt_rect, pred_rect)
                    sample['iou'] = iou
                    sample['success'] = (iou >= 0.5)
                else:
                    sample['iou'] = 0.0
                    sample['success'] = False
            
            samples.append(sample)
            
        except Exception as e:
            print(f"Error loading {jf}: {e}")
            
    return samples
