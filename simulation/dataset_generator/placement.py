# placement.py - size selection, sampling position, occlusion application
import random
from config import CATEGORY_POSITION_PRIORS, WINDOW_SIZES, SCREEN_W, SCREEN_H

def sample_size(window_entry, scale_preference='medium'):
    # scale_preference could be used to bias selection, but for now we pick randomly
    s = random.choice(WINDOW_SIZES)
    if s == 'original':
        if 'orig_size' in window_entry:
            return tuple(window_entry['orig_size'])
        return (1280,720)
    return s

def sample_position(category, w, h, center_bias=False, use_position_prior=True):
    if not use_position_prior:
        # Ablation mode: remove all spatial priors and sample from full screen.
        px1, px2, py1, py2 = (0.0, 1.0, 0.0, 1.0)
    elif center_bias:
        # Bias towards center
        px1, px2, py1, py2 = (0.15, 0.85, 0.15, 0.85)
    else:
        # Use category priors or full screen
        px1, px2, py1, py2 = CATEGORY_POSITION_PRIORS.get(category, (0.0, 1.0, 0.0, 1.0))
        
    x_min = int(px1 * SCREEN_W)
    x_max = int(px2 * SCREEN_W - w)
    y_min = int(py1 * SCREEN_H)
    y_max = int(py2 * SCREEN_H - h)
    
    # Ensure valid range
    if x_max < x_min: x_max = x_min
    if y_max < y_min: y_max = y_min
    
    x = random.randint(x_min, x_max)
    y = random.randint(y_min, y_max)
    
    # Ensure within screen
    x = max(0, min(SCREEN_W - w, x))
    y = max(0, min(SCREEN_H - h, y))
    
    return x, y

def assign_z_order(windows):
    if all('z' in w for w in windows):
        windows.sort(key=lambda x: x['z'])
    else:
        random.shuffle(windows)
        
    for i, w in enumerate(windows):
        w['z'] = i
    return windows

def place_occluder(target_bbox, occluder_size, screen_w, screen_h):
    # target_bbox: (x1, y1, x2, y2) - absolute screen coordinates
    tx1, ty1, tx2, ty2 = target_bbox
    ow, oh = occluder_size
    
    p_x = random.uniform(tx1, tx2)
    p_y = random.uniform(ty1, ty2)
    
    # Position occluder such that it covers p_x, p_y
    # ox <= p_x <= ox + ow  =>  p_x - ow <= ox <= p_x
    # oy <= p_y <= oy + oh  =>  p_y - oh <= oy <= p_y
    
    min_ox = p_x - ow
    max_ox = p_x
    min_oy = p_y - oh
    max_oy = p_y
    
    ox = int(random.uniform(min_ox, max_ox))
    oy = int(random.uniform(min_oy, max_oy))
    
    # Clamp to screen
    ox = max(0, min(screen_w - ow, ox))
    oy = max(0, min(screen_h - oh, oy))
    
    return ox, oy
