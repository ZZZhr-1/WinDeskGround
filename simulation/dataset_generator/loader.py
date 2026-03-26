# loader.py - load metadata and images
import json
from pathlib import Path
from PIL import Image

def load_metadata(path):
    path = Path(path)
    with open(path, 'r', encoding='utf8') as f:
        data = json.load(f)
    for e in data:
        if 'image_path' not in e or 'gt_bbox' not in e or 'type' not in e:
            raise ValueError('Each metadata entry must have image_path, gt_bbox, and type fields.')
    return data

def load_window_image(path, root=None):
    if root:
        # If path is absolute or already contains root, this might duplicate.
        # But usually path is relative like "Advanced_Tools/..."
        # and root is "windows" or "../windows"
        path = Path(root) / path
    return Image.open(path).convert('RGBA')
