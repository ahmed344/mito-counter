#!/usr/bin/env python3
import argparse
import os
from typing import Iterable, Tuple

import cv2
import numpy as np
from skimage import exposure
import tifffile as tif


DEFAULT_INPUT_ROOT = "/workspaces/mito-counter/data/Calpaine_3/Processed"


def find_tiff_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            lower = name.lower()
            if lower.endswith((".tif", ".tiff")) and "_corrected" not in lower:
                yield os.path.join(dirpath, name)


def shading_correct_flatfield(
    raw: np.ndarray, sigma: int = 400, eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_f = raw.astype(np.float32)

    illum = cv2.GaussianBlur(
        raw_f, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT
    )

    scale = np.median(illum[illum > 0]) if np.any(illum > 0) else np.median(illum)
    corrected = raw_f / (illum + eps) * scale

    p1, p99 = np.percentile(corrected, (1, 99.8))
    corrected_u16 = exposure.rescale_intensity(
        corrected, in_range=(p1, p99), out_range=(0, 65535)
    ).astype(np.uint16)

    return corrected, corrected_u16, illum


def build_corrected_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}_corrected{ext}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Background-correct TIFF images (no denoise).")
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT, help="Root folder of TIFF files.")
    parser.add_argument("--sigma", type=int, default=400, help="Gaussian sigma for background.")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tiff_files = list(find_tiff_files(args.input_root))
    if not tiff_files:
        print(f"No TIFF files found under: {args.input_root}")
        return

    for path in tiff_files:
        out_path = build_corrected_path(path)
        if args.dry_run:
            print(f"[DRY RUN] {path} -> {out_path}")
            continue

        raw_img = tif.imread(path)
        _, corrected_u16, _ = shading_correct_flatfield(raw_img, sigma=args.sigma)
        tif.imwrite(out_path, corrected_u16, photometric="minisblack")
        print(f"Corrected {path} -> {out_path}")


if __name__ == "__main__":
    main()
