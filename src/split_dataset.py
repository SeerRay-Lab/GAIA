"""Shuffle a JSON list and split it into train / val files.

Usage:
    python src/split_dataset.py <input.json> [output_prefix] [train_ratio]

Writes <prefix>_train.json and <prefix>_val.json (default prefix = input path,
default ratio = 0.8).
"""

import os
import sys
import json
import random


def split(input_path, output_path, train_ratio=0.8):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records from {input_path}")

    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    train_data, val_data = data[:split_idx], data[split_idx:]

    base, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".json"
    train_path, val_path = f"{base}_train{ext}", f"{base}_val{ext}"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"Train -> {train_path} ({len(train_data)})")
    print(f"Val   -> {val_path} ({len(val_data)})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else inp
    ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    split(inp, out, ratio)
