#!/usr/bin/env python3
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Tuple

import cv2
import numpy as np
from skimage import exposure
import tifffile as tif


DEFAULT_INPUT_ROOT = "/workspaces/mito-counter/data/Calpaine_3/Processed"


def find_tiff_files(root: str) -> Iterable[str]:
    """Yield TIFF file paths under a root folder.

    Args:
        root (str): Root directory to search for TIFF files.

    Returns:
        Iterable[str]: TIFF file paths excluding already-corrected outputs.
    """
    # Walk the directory tree and filter valid TIFFs.
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            lower = name.lower()
            if lower.endswith((".tif", ".tiff")) and "_corrected" not in lower:
                yield os.path.join(dirpath, name)


def shading_correct_flatfield(
    raw: np.ndarray, sigma: int = 400, eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply flat-field background correction with a Gaussian illumination model.

    Args:
        raw (np.ndarray): 2D image array.
        sigma (int): Gaussian sigma for illumination estimation.
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        Tuple of (corrected_float, corrected_u8, illumination_estimate).
    """
    # Work in float for stable correction math.
    raw_f = raw.astype(np.float32)

    # Estimate illumination via heavy Gaussian blur.
    illum = cv2.GaussianBlur(
        raw_f, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT
    )

    # Flat-field division with a robust scale factor.
    scale = np.median(illum[illum > 0]) if np.any(illum > 0) else np.median(illum)
    corrected = raw_f / (illum + eps) * scale

    # Robust contrast stretch to avoid outlier domination.
    p1, p99 = np.percentile(corrected, (1, 99.8))
    corrected_u8 = exposure.rescale_intensity(
        corrected, in_range=(p1, p99), out_range=(0, 255)
    ).astype(np.uint8)

    return corrected, corrected_u8, illum


def build_corrected_path(input_path: str) -> str:
    """Construct the output path with a _corrected suffix.

    Args:
        input_path (str): Original TIFF file path.

    Returns:
        str: Output path with _corrected appended before the extension.
    """
    base, ext = os.path.splitext(input_path)
    return f"{base}_corrected{ext}"


def process_file(path: str, sigma: int) -> str:
    """Read, correct background, and write a corrected TIFF.

    Args:
        path (str): Input TIFF path.
        sigma (int): Gaussian sigma for illumination estimation.

    Returns:
        str: Output path of the corrected TIFF.
    """
    # Derive output path and load the image.
    out_path = build_corrected_path(path)
    raw_img = tif.imread(path)
    # Apply background correction and save.
    _, corrected_u8, _ = shading_correct_flatfield(raw_img, sigma=sigma)
    tif.imwrite(out_path, corrected_u8, photometric="minisblack")
    return out_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description="Background-correct TIFF images (no denoise).")
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT, help="Root folder of TIFF files.")
    parser.add_argument("--sigma", type=int, default=400, help="Gaussian sigma for background.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of parallel workers.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print planned outputs.")
    return parser.parse_args()


def main() -> None:
    """Run the background correction pipeline.

    Args:
        None.

    Returns:
        None.
    """
    # Collect candidate TIFF files.
    args = parse_args()
    # Gather all candidate input files once.
    tiff_files = list(find_tiff_files(args.input_root))
    if not tiff_files:
        print(f"No TIFF files found under: {args.input_root}")
        return

    # If dry-run, just show planned outputs.
    if args.dry_run:
        for path in tiff_files:
            out_path = build_corrected_path(path)
            print(f"[DRY RUN] {path} -> {out_path}")
        return

    # Process images in parallel for speed.
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(process_file, path, args.sigma): path for path in tiff_files
        }
        # Report results as tasks complete.
        for future in as_completed(future_map):
            path = future_map[future]
            try:
                out_path = future.result()
            except Exception as exc:
                print(f"[ERROR] {path} -> {exc}")
                continue
            print(f"Corrected {path} -> {out_path}")


if __name__ == "__main__":
    main()
