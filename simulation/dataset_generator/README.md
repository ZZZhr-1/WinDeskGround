# Desktop Composite Dataset Generator

Modular generator to create synthetic desktop images with multiple windows and remapped target bboxes.
Includes configurable difficulty levels, visual distractors, occlusion control, and debug visualization.

## Structure
- config.py
- loader.py
- geometry.py
- clutter.py
- similarity.py
- placement.py
- compositor.py
- sampler.py
- generator.py
- debug_visualizer.py
- run.py

## Usage
1. Put your `metadata.json` describing window images (see example below).
2. Run: `python run.py --metadata /path/to/metadata.json --wallpaper /path/to/wallpaper.jpg --out ./out --n 100 --difficulty L3`

Metadata example:
```json
[
  {
    "image_path": "Advanced_Tools/original/client_original_20250603_230025.png",
    "content": "the 'Views' dropdown menu ...",
    "gt_bbox": [0.84, 0.07, 0.96, 0.14],
    "type": "Advanced_Tools",
    "orig_size": [1366,768]
  }
]
```

## Notes
- Requires: Pillow, numpy. OpenCV optional (for faster ops).
- The package is modular; swap out similarity, clutter, or placement modules as needed.
