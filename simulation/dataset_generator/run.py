# run.py - command line launcher
import argparse
import os
import sys
from pathlib import Path

# Add eval directory to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "eval"))

from generator import generate_dataset

# BatchEvaluator is deprecated in favor of offline evaluation
BatchEvaluator = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, required=True, help='metadata json path')
    parser.add_argument('--wallpaper', type=str, default=None, help='wallpaper image (2560x1440 recommended)')
    parser.add_argument('--out', type=str, required=True, help='output directory')
    parser.add_argument('--n', type=int, default=100, help='number of samples')
    parser.add_argument('--difficulty', type=str, default='L3', help='difficulty level L1..L5')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--image_root', type=str, default='windows', help='root directory for window images')
    parser.add_argument('--traverse', action='store_true', help='If set, traverse all metadata items as targets one by one. Overrides --n.')
    # parser.add_argument('--model_path', type=str, default=None, help='Path to Qwen2-VL model for online evaluation')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for evaluation')
    args = parser.parse_args()
    
    evaluator = None
    # if args.model_path:
    #     if BatchEvaluator:
    #         evaluator = BatchEvaluator(args.model_path, args.out, batch_size=args.batch_size)
    #     else:
    #         print("Error: BatchEvaluator not available.")

    generate_dataset(args.metadata, args.wallpaper, args.out, args.n, difficulty=args.difficulty, seed=args.seed, image_root=args.image_root, evaluator=evaluator, traverse=args.traverse)
