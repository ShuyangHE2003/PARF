

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from parf_operator import PARF

REQUIRED = (
    "gf_radius",
    "gf_eps",
    "saliency_radius",
    "lambda_s",
    "lambda_g",
    "base_eps",
    "detail_eps",
    "lambdas",
    "kappa",
    "resolution_scale",
    "tau_gate",
    "relation_input",
    "rho0",
    "gamma_tau",
    "target_percentile",
    "mask_radius",
)


def load_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    missing = []
    for key in REQUIRED:
        value = cfg.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(key)

    if missing:
        raise SystemExit(
            "Config is incomplete. Fill these fields in "
            f"{path}:\n  " + "\n  ".join(missing)
        )

    cfg["lambdas"] = [float(v) for v in cfg["lambdas"]]
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="PARF infrared/visible image fusion (strict config mode)"
    )
    parser.add_argument("--config", required=True,
                        help="fully filled YAML config")
    parser.add_argument("--ir", required=True, help="path to the infrared image")
    parser.add_argument("--vis", required=True, help="path to the visible image")
    parser.add_argument("--out", required=True, help="output image path")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
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

    model = PARF(**{k: cfg[k] for k in REQUIRED}).to(device).eval()
    ir_t = torch.from_numpy(ir)[None, None].to(device)
    vis_t = torch.from_numpy(vis)[None, None].to(device)

    with torch.no_grad():
        fused = model.fuse(ir_t, vis_t)

    out = fused[0, 0].detach().cpu().numpy()
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(out_path)
    print(f"saved: {out_path}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
