# -*- coding: utf-8 -*-
"""Fuse one registered infrared/visible pair with PARF.

Example:
    python parf_fuse.py --ir IR.png --vis VIS.png --out fused.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from parf_operator import PARF


def load_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser(description="PARF infrared/visible image fusion")
    parser.add_argument("--ir", required=True, help="path to the infrared image")
    parser.add_argument("--vis", required=True, help="path to the visible image")
    parser.add_argument("--out", required=True, help="output image path")
    parser.add_argument("--device", default="auto",
                        help="cpu, cuda, or auto (default)")
    parser.add_argument("--relation-input", default="raw",
                        choices=["raw", "residual"],
                        help="domain used to compute Gamma (default: raw)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ir = load_gray(args.ir)
    vis = load_gray(args.vis)
    if ir.shape != vis.shape:
        raise ValueError(
            f"IR and VIS sizes differ: {ir.shape} vs {vis.shape}. "
            "Please register/resize them first."
        )

    model = PARF(relation_input=args.relation_input).to(device).eval()
    ir_t = torch.from_numpy(ir)[None, None].to(device)
    vis_t = torch.from_numpy(vis)[None, None].to(device)

    with torch.no_grad():
        fused = model.fuse(ir_t, vis_t)

    out = fused[0, 0].detach().cpu().numpy()
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="L").save(out_path)
    print(f"saved: {out_path}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
