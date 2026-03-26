# generator.py - orchestrates dataset generation
import random, json, os, math
from pathlib import Path
from PIL import Image
from loader import load_metadata, load_window_image
from placement import sample_size, sample_position, assign_z_order, place_occluder
from geometry import project_bbox, compute_visible_ratio
from compositor import composite_desktop
# from clutter import edge_density, add_clutter, add_notification # Removed
from sampler import sample_windows
from debug_visualizer import draw_debug_info
from config import SCREEN_W, SCREEN_H, DIFFICULTY_LEVELS

def generate_one_sample(metadata, wallpaper, difficulty='L3', image_root=None, path_to_meta=None, config_override=None, target_index=None):
    if config_override:
        conf = config_override
    else:
        conf = DIFFICULTY_LEVELS[difficulty]
        
    visible_min = conf['visible_min']
    visible_max = conf.get('visible_max', 1.0)
    
    # 1. Sample Windows
    selected = sample_windows(metadata, conf, image_root=image_root, path_to_meta=path_to_meta, target_index=target_index)
    windows = []
    
    # 2. Initial Placement
    for i, win in enumerate(selected):
        # Target is usually the first one from sample_windows (based on sampler logic)
        is_target = (i == 0)

        # Load image first to get natural size (avoid stretching)
        try:
            # Load the original image as specified in metadata
            # We MUST use the exact image path from metadata because gt_bbox is tied to that specific layout.
            img = load_window_image(win['image_path'], root=image_root)
            
            # No resizing is performed here, relies on native resolution or loader handling.
                
            w, h = img.size
        except Exception as e:
            # Fallback
            print(f"Warning: Failed to load image {win['image_path']}: {e}")
            w, h = (1280, 720)
            img = Image.new('RGBA', (w,h), (200,200,200,255))
        
        # Center bias for easier levels
        center_bias = (difficulty in ['L1', 'L2'] and is_target)
        
        x, y = sample_position(win['type'], w, h, center_bias=center_bias)
            
        windows.append({
            'meta': win, 
            'window_image': img, 
            'placed_bbox': (x,y,w,h), 
            'is_target': is_target,
            'z': 0 # Placeholder
        })

    # 3. Construct Occlusion (Deterministic)
    target_win = next((w for w in windows if w['is_target']), None)
    if target_win:
        # Calculate Target BBox Projection
        if 'gt_bbox' in target_win['meta']:
            target_win['target_bbox_proj'] = project_bbox(target_win['meta']['gt_bbox'], target_win['placed_bbox'])
        else:
            # Fallback if no gt_bbox, use full window
            tx, ty, tw, th = target_win['placed_bbox']
            target_win['target_bbox_proj'] = (tx, ty, tx+tw, ty+th)
            
        # Determine desired visibility
        target_vis = random.uniform(visible_min, visible_max)
        
        # If we need significant occlusion (e.g. < 95% visible)
        if target_vis < 0.95:
            tx1, ty1, tx2, ty2 = target_win['target_bbox_proj']
            target_w = tx2 - tx1
            target_h = ty2 - ty1
            target_area = target_w * target_h
            
            needed_occ_area = target_area * (1.0 - target_vis)
            
            # Select distractors to use as occluders
            distractors = [w for w in windows if not w['is_target']]
            # Sort by area descending to use big windows for occlusion
            distractors.sort(key=lambda w: w['placed_bbox'][2] * w['placed_bbox'][3], reverse=True)
            
            active_occluders = []
            background_windows = []
            
            if distractors and needed_occ_area > 10: # Ignore tiny occlusion needs
                # Use the largest distractor as the main occluder
                occ = distractors.pop(0)
                active_occluders.append(occ)
                
                # Calculate position to achieve overlap
                ratio = target_w / target_h if target_h > 0 else 1.0
                w_ov = math.sqrt(needed_occ_area * ratio)
                h_ov = w_ov / ratio
                
                # Clamp to dimensions
                ox, oy, ow, oh = occ['placed_bbox']
                w_ov = min(w_ov, target_w, ow)
                h_ov = min(h_ov, target_h, oh)
                
                # Re-calculate h_ov if w_ov was clamped, to maintain area
                if w_ov > 0:
                    h_ov = min(needed_occ_area / w_ov, target_h, oh)
                
                # Randomize corner to avoid bias
                corner = random.choice(['br', 'bl', 'tr', 'tl'])
                
                new_ox, new_oy = ox, oy
                
                if corner == 'br': 
                    new_ox = tx2 - w_ov
                    new_oy = ty2 - h_ov
                elif corner == 'bl': 
                    new_ox = tx1 + w_ov - ow
                    new_oy = ty2 - h_ov
                elif corner == 'tr':
                    new_ox = tx2 - w_ov
                    new_oy = ty1 + h_ov - oh
                elif corner == 'tl': 
                    new_ox = tx1 + w_ov - ow
                    new_oy = ty1 + h_ov - oh
                    
                # Apply position
                occ['placed_bbox'] = (int(new_ox), int(new_oy), ow, oh)
                
            background_windows = distractors # The rest
            
            # Assign Z-order: Background < Target < Occluders
            current_z = 0
            
            # Ensure similar distractors are visible if possible (put them on top of background)
            similar_distractors = [w for w in background_windows if w['meta'].get('type') == target_win['meta'].get('type')]
            other_background = [w for w in background_windows if w not in similar_distractors]
            
            for w in other_background:
                w['z'] = current_z
                current_z += 1
                
            for w in similar_distractors:
                w['z'] = current_z
                current_z += 1
            
            target_win['z'] = current_z
            current_z += 1
            
            for w in active_occluders:
                w['z'] = current_z
                current_z += 1
                
        else:
            # High visibility requested
            
            distractors = [w for w in windows if not w['is_target']]
            
            # Semantic Distraction Fix: Ensure specific distracting elements are not covered by the target
            if distractors and target_win:
                tx, ty, tw, th = target_win['placed_bbox']
                target_rect = (tx, ty, tx+tw, ty+th)
                
                for w in distractors:
                    if 'distractor_element' in w['meta'] and 'bbox' in w['meta']['distractor_element']:
                        # Calculate absolute bbox of the distracting element
                        dw, dh = w['placed_bbox'][2], w['placed_bbox'][3]
                        dx, dy = w['placed_bbox'][0], w['placed_bbox'][1]
                        
                        rel_bbox = w['meta']['distractor_element']['bbox']
                        # bbox is [x1, y1, x2, y2] normalized
                        ex1 = dx + rel_bbox[0] * dw
                        ey1 = dy + rel_bbox[1] * dh
                        ex2 = dx + rel_bbox[2] * dw
                        ey2 = dy + rel_bbox[3] * dh
                        
                        elem_area = (ex2 - ex1) * (ey2 - ey1)
                        if elem_area <= 0: continue

                        # Check overlap with target
                        ix1 = max(ex1, tx)
                        iy1 = max(ey1, ty)
                        ix2 = min(ex2, tx + tw)
                        iy2 = min(ey2, ty + th)
                        
                        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        overlap_ratio = inter_area / elem_area
                        
                        # If more than 20% of the key element is covered, try to move the window
                        if overlap_ratio > 0.20:
                            # Try to resample position
                            for _ in range(15):
                                new_dx, new_dy = sample_position(w['meta'].get('type',''), dw, dh)
                                
                                # Check new overlap
                                n_ex1 = new_dx + rel_bbox[0] * dw
                                n_ey1 = new_dy + rel_bbox[1] * dh
                                n_ex2 = new_dx + rel_bbox[2] * dw
                                n_ey2 = new_dy + rel_bbox[3] * dh
                                
                                n_ix1 = max(n_ex1, tx)
                                n_iy1 = max(n_ey1, ty)
                                n_ix2 = min(n_ex2, tx + tw)
                                n_iy2 = min(n_ey2, ty + th)
                                
                                n_inter_area = max(0, n_ix2 - n_ix1) * max(0, n_iy2 - n_iy1)
                                
                                if (n_inter_area / elem_area) < 0.1: # Acceptable
                                    w['placed_bbox'] = (new_dx, new_dy, dw, dh)
                                    break

            similar_distractors = [w for w in distractors if w['meta'].get('type') == target_win['meta'].get('type')]
            other_distractors = [w for w in distractors if w not in similar_distractors]
            
            current_z = 0
            for w in other_distractors:
                w['z'] = current_z
                current_z += 1
                
            for w in similar_distractors:
                w['z'] = current_z
                current_z += 1
                
            target_win['z'] = current_z # Top
            
    # Final visibility check (just to update metrics)
    windows.sort(key=lambda x: x['z'])
    if target_win and 'gt_bbox' in target_win['meta']:
        proj = project_bbox(target_win['meta']['gt_bbox'], target_win['placed_bbox'])
        vis = compute_visible_ratio(proj, windows, SCREEN_W, SCREEN_H, windows.index(target_win))
        target_win['target_bbox_proj'] = proj
        target_win['visible_ratio'] = vis

    # 5. Composite
    out = composite_desktop(wallpaper.copy(), windows)
    
    # Debug
    debug = draw_debug_info(out, windows)
    
    return out, debug, windows, 0.0

def save_sample(idx, out_dir, img, debug_img, windows, clutter):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f'desktop_{idx:06d}.jpg'
    debug_path = out_dir / f'desktop_{idx:06d}_debug.jpg'
    meta_path = out_dir / f'desktop_{idx:06d}.json'
    img.save(img_path, quality=90)
    debug_img.save(debug_path, quality=90)
    
    rec = {
        'image_path': str(img_path), 
        'debug_path': str(debug_path), 
        'windows': []
    }
    
    # Extract target info to root level
    target_win = next((w for w in windows if w.get('is_target')), None)
    if target_win:
        if 'target_bbox_proj' in target_win:
            rec['gt_bbox'] = tuple(map(int, target_win['target_bbox_proj']))
            # Add bbox in [x, y, w, h] format for compatibility
            x1, y1, x2, y2 = rec['gt_bbox']
            rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
        if 'content' in target_win['meta']:
            rec['gt_content'] = target_win['meta']['content']
            rec['instruction'] = target_win['meta']['content']
            
    for w in windows:
        info = {'image_path': w['meta'].get('image_path',''), 'type': w['meta'].get('type',''), 'placed_bbox': w['placed_bbox'], 'z': w.get('z',0)}
        if w.get('is_target'):
            info['is_target'] = True
            if 'target_bbox_proj' in w:
                info['target_bbox_proj'] = tuple(map(int, w['target_bbox_proj']))
                info['visible_ratio'] = float(w.get('visible_ratio', 0.0))
                x1,y1,x2,y2 = info['target_bbox_proj']
                area = max(0,(x2-x1)) * max(0,(y2-y1))
                info['target_rel_screen'] = float(area) / float(SCREEN_W*SCREEN_H)
        rec['windows'].append(info)
    with open(meta_path, 'w', encoding='utf8') as f:
        json.dump(rec, f, indent=2)
    return str(img_path), str(debug_path), str(meta_path), rec

def generate_dataset(metadata_path, wallpaper_path, out_dir, n, difficulty='L3', seed=None, image_root=None, evaluator=None, traverse=False):
    if seed is not None:
        random.seed(seed)
    metadata = load_metadata(metadata_path)
    
    if traverse:
        n = len(metadata)
        print(f"Traverse mode enabled. Generating {n} samples (one per metadata item).")

    # Build lookup for fast retrieval by image path
    path_to_meta = {m['image_path']: m for m in metadata}
    
    if wallpaper_path and os.path.exists(wallpaper_path):
        wallpaper = Image.open(wallpaper_path).convert('RGBA')
    else:
        # Create default wallpaper
        wallpaper = Image.new('RGBA', (SCREEN_W, SCREEN_H), (50, 100, 150, 255))
    
    all_samples = []
    out_dir = Path(out_dir)

    for i in range(n):
        target_idx = i if traverse else None
        img, debug, windows, cg = generate_one_sample(
            metadata, 
            wallpaper, 
            difficulty, 
            image_root=image_root, 
            path_to_meta=path_to_meta,
            target_index=target_idx
        )
        img_path, debug_path, meta_path, rec = save_sample(i, out_dir, img, debug, windows, cg)
        
        rec['img_filename'] = os.path.basename(img_path)
        all_samples.append(rec)

        if evaluator:
            evaluator.add_sample(rec)

        if (i+1) % 10 == 0:
            print(f'Generated {i+1}/{n}')

    # Process remaining
    if evaluator:
        evaluator.flush()

    # Save final results
    with open(out_dir / 'test.json', 'w') as f:
        json.dump(all_samples, f, indent=2)
        
    if evaluator:
        print(f"Generation and Evaluation completed.")
