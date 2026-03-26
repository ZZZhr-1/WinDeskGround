# geometry.py - bbox projection and visible ratio calc
import numpy as np

def project_bbox(gt_bbox, placed_bbox):
    x, y, w, h = placed_bbox
    x1 = x + int(gt_bbox[0] * w)
    y1 = y + int(gt_bbox[1] * h)
    x2 = x + int(gt_bbox[2] * w)
    y2 = y + int(gt_bbox[3] * h)
    return (x1, y1, x2, y2)

def mask_from_bbox(screen_w, screen_h, bbox):
    mask = np.zeros((screen_h, screen_w), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(screen_w, int(x2)); y2 = min(screen_h, int(y2))
    if x2>x1 and y2>y1:
        mask[y1:y2, x1:x2] = 1
    return mask

def compute_visible_ratio(target_bbox, placed_windows, screen_w, screen_h, window_index):
    target_mask = mask_from_bbox(screen_w, screen_h, target_bbox)
    if target_mask.sum() == 0:
        return 0.0
    occluder_mask = np.zeros_like(target_mask)
    for w in placed_windows[window_index+1:]:
        x, y, ww, hh = w['placed_bbox']
        ob = (x, y, x+ww, y+hh)
        occluder_mask = occluder_mask | mask_from_bbox(screen_w, screen_h, ob)
    visible = target_mask & (~occluder_mask)
    return float(visible.sum()) / float(target_mask.sum())
