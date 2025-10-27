#!/usr/bin/env python3
"""
Read an RGB image where background is white and a path is drawn in red.
Extract red pixel coordinates and append them to points.txt (one point per line: "x y").

Usage:
    python scripts/extract_red_path.py --input path.png --output points.txt --step 1 --threshold 200 --append

Options:
- --input: input image path
- --output: output points file (default: points.txt)
- --step: sample every N pixels along x and y to reduce output (default 1 = all)
- --threshold: minimum red channel value to consider a pixel "red" (0-255)
- --append: if set, append to existing file; otherwise overwrite
"""
import argparse
from PIL import Image
import numpy as np


def extract_red_pixels(image_path, threshold=200, step=1):
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)
    # arr shape: (H, W, 3)
    H, W, C = arr.shape
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]

    # Condition for red pixel: red > threshold and green,blue are small compared to red
    mask = (red >= threshold) & (green < threshold // 2) & (blue < threshold // 2)

    points = []
    for y in range(0, H, step):
        for x in range(0, W, step):
            if mask[y, x]:
                # Use x y coordinates (float not necessary, but we'll output floats)
                points.append((float(x), float(y)))
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--output', default='points.txt', help='Output points file')
    parser.add_argument('--step', type=int, default=1, help='Sampling step in pixels')
    parser.add_argument('--threshold', type=int, default=200, help='Red threshold (0-255)')
    parser.add_argument('--append', action='store_true', help='Append to output file instead of overwrite')
    parser.add_argument('--scale_max', type=float, default=5.0, help='Scale output coordinates into [0,scale_max] range')
    parser.add_argument('--radius', type=float, default=0.0, help='Minimum distance between output points; points closer than radius will be filtered out (in scaled units)')
    parser.add_argument('--origin', choices=['top-left', 'bottom-left'], default='bottom-left', help='Origin for coordinates: top-left (image coords) or bottom-left (default)')
    args = parser.parse_args()

    points = extract_red_pixels(args.input, threshold=args.threshold, step=args.step)

    # Scale points to [0, scale_max] range
    img = Image.open(args.input).convert('RGB')
    arr = np.array(img)
    H, W = arr.shape[0], arr.shape[1]
    scale_max = float(args.scale_max)
    if scale_max <= 0:
        scale_max = 5.0

    scaled_points = []
    for x, y in points:
        # Convert pixel coords depending on origin choice
        if args.origin == 'bottom-left':
            # bottom-left origin: x same, y inverted (y' = (H-1 - y))
            px = x
            py = (H - 1.0) - y
        else:
            px = x
            py = y

        sx = (px / max(1.0, W - 1)) * scale_max
        sy = (py / max(1.0, H - 1)) * scale_max
        scaled_points.append((sx, sy))

    # If radius filtering requested, perform greedy filtering in SCALED coordinate space
    if args.radius is not None and args.radius > 0.0:
        kept = []
        for p in scaled_points:
            x, y = p
            too_close = False
            for qx, qy in kept:
                dx = x - qx
                dy = y - qy
                if (dx * dx + dy * dy) <= (args.radius * args.radius):
                    too_close = True
                    break
            if not too_close:
                kept.append(p)
        scaled_points = kept

    if args.append:
        mode = 'a'
    else:
        mode = 'w'

    with open(args.output, mode) as f:
        for x, y in scaled_points:
            f.write(f"{x} {y}\n")

    print(f"Wrote {len(scaled_points)} points to {args.output}")


if __name__ == '__main__':
    main()
