#!/usr/bin/env python3
"""Render mitochondria segmentation QC overlays for the CAPN3 patient dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "Calpaine_3_patients" / "Processed"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data" / "Calpaine_3_patients" / "results" / "overlay_qc"
)
MITO_ALPHA = 0.28
EXCLUDED_SOURCE_SUFFIXES = (
    "_segmented",
    "_cells",
    "_center_cell_mask",
    "_cell_mask",
    "_sarcomere",
    "_sarcomere_corrected",
    "_sarcomere_segmented",
    "_sarcomere_mask",
    "_mbands",
    "_zbands",
)
EXCLUDED_SOURCE_STEMS = {
    "cell_mask",
    "sarcomere_mask",
    "mbands",
    "zbands",
}


@dataclass(frozen=True)
class OverlayInput:
    """Container for one patient overlay rendering job.

    Args:
        source_path (Path): Path to the corrected source TIFF image.
        segmented_path (Path): Path to the mitochondria segmentation TIFF.
        metadata_path (Path): Path to the corresponding base JSON metadata.
        output_path (Path): Path where the rendered PNG will be written.
        image_label (str): Base image stem shared by the related input files.

    Returns:
        None: Dataclass field container.
    """

    source_path: Path
    segmented_path: Path
    metadata_path: Path
    output_path: Path
    image_label: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for patient QC overlay rendering.

    Args:
        None: This function reads arguments from the process command line.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Render CAPN3 patient QC overlays by alpha-blending mitochondria "
            "segmentations over percentile-scaled corrected images."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root scanned recursively for *_corrected.tif images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where mirrored PNG overlays are written.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Optional corrected TIFF to process instead of scanning --input-root.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of discovered images to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace overlay PNGs that already exist.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be zero or greater")
    return args


def strip_corrected_suffix(path: Path) -> str:
    """Remove the corrected suffix from a source image stem.

    Args:
        path (Path): Corrected source TIFF path.

    Returns:
        str: Base image label used by the JSON and segmentation files.
    """
    stem = path.stem
    if stem.lower().endswith("_corrected"):
        return stem[: -len("_corrected")]
    return stem


def is_primary_source_image(path: Path) -> bool:
    """Determine whether a path is a primary corrected source TIFF.

    Args:
        path (Path): Candidate filesystem path.

    Returns:
        bool: True when the path is a corrected TIFF and not a derived image.
    """
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return False
    if not path.stem.lower().endswith("_corrected"):
        return False

    base_stem = strip_corrected_suffix(path).lower()
    if base_stem in EXCLUDED_SOURCE_STEMS:
        return False
    return not base_stem.endswith(EXCLUDED_SOURCE_SUFFIXES)


def discover_source_images(
    input_root: Path,
    input_file: Path | None,
    limit: int | None,
) -> list[Path]:
    """Discover primary corrected patient TIFF images.

    Args:
        input_root (Path): Root recursively scanned for corrected TIFF files.
        input_file (Path | None): Optional single corrected TIFF to process.
        limit (int | None): Optional maximum number of paths to return.

    Returns:
        list[Path]: Sorted corrected source image paths.
    """
    if input_file is not None:
        if not input_file.is_file():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        if not is_primary_source_image(input_file):
            raise ValueError(f"Input file is not a corrected source TIFF: {input_file}")
        source_paths = [input_file]
    else:
        if not input_root.is_dir():
            raise FileNotFoundError(f"Input root not found: {input_root}")
        source_paths = sorted(
            path
            for path in input_root.rglob("*")
            if path.is_file() and is_primary_source_image(path)
        )

    return source_paths if limit is None else source_paths[:limit]


def build_overlay_input(
    source_path: Path,
    input_root: Path,
    output_root: Path,
) -> OverlayInput:
    """Resolve required sibling files and the mirrored patient output path.

    Args:
        source_path (Path): Corrected source TIFF path.
        input_root (Path): Processed-data root used to compute relative paths.
        output_root (Path): Root directory for rendered overlay PNGs.

    Returns:
        OverlayInput: Validated paths for one rendering job.
    """
    image_label = strip_corrected_suffix(source_path)
    metadata_path = source_path.with_name(f"{image_label}.json")
    segmented_path = source_path.with_name(f"{image_label}_segmented.tif")

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata JSON not found: {metadata_path}")
    if not segmented_path.is_file():
        raise FileNotFoundError(
            f"Segmented mitochondria TIFF not found: {segmented_path}"
        )

    try:
        relative_parent = source_path.resolve().parent.relative_to(
            input_root.resolve()
        )
    except ValueError as exc:
        raise ValueError(
            f"Source image is outside input root and cannot be mirrored: {source_path}"
        ) from exc

    output_path = output_root / relative_parent / f"{image_label}_overlay.png"
    return OverlayInput(
        source_path=source_path,
        segmented_path=segmented_path,
        metadata_path=metadata_path,
        output_path=output_path,
        image_label=image_label,
    )


def load_tiff(path: Path) -> np.ndarray:
    """Load an image array from a TIFF file.

    Args:
        path (Path): TIFF image path.

    Returns:
        np.ndarray: Loaded image array.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return tiff.imread(str(path))


def to_uint8_display(image: np.ndarray) -> np.ndarray:
    """Percentile-scale an image into a display-ready uint8 array.

    Args:
        image (np.ndarray): Numeric grayscale or RGB-like image array.

    Returns:
        np.ndarray: Display-ready uint8 array with the original dimensions.
    """
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"Unsupported non-numeric image dtype: {image.dtype}")

    values = image.astype(np.float32)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Image contains no finite pixel values")

    low = float(np.percentile(finite_values, 1.0))
    high = float(np.percentile(finite_values, 99.0))
    if high <= low:
        high = low + 1.0
    scaled = np.nan_to_num(
        np.clip((values - low) / (high - low), 0.0, 1.0),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    return (scaled * 255.0).astype(np.uint8)


def ensure_rgb_display(
    image: np.ndarray,
    percentile_scale: bool = True,
) -> np.ndarray:
    """Convert a grayscale or RGB-like image to an RGB uint8 array.

    Args:
        image (np.ndarray): Input grayscale or RGB-like image.
        percentile_scale (bool): Whether to percentile-scale uint8 input too.

    Returns:
        np.ndarray: RGB uint8 image with shape ``(height, width, 3)``.
    """
    if image.dtype == np.uint8 and not percentile_scale:
        display = image.copy()
    else:
        display = to_uint8_display(image)
    if display.ndim == 2:
        return np.repeat(display[:, :, None], 3, axis=2)
    if display.ndim == 3 and display.shape[2] >= 3:
        return display[:, :, :3].copy()
    raise ValueError(f"Unsupported image shape for RGB display: {display.shape}")


def render_overlay(
    source_image: np.ndarray,
    segmented_image: np.ndarray,
    alpha: float = MITO_ALPHA,
) -> np.ndarray:
    """Blend a colored segmentation over a percentile-scaled source image.

    Args:
        source_image (np.ndarray): Corrected grayscale or RGB source image.
        segmented_image (np.ndarray): Colored segmentation visualization.
        alpha (float): Segmentation opacity in the closed interval [0, 1].

    Returns:
        np.ndarray: Rendered RGB uint8 overlay.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Alpha must be between 0 and 1, received: {alpha}")

    source_rgb = ensure_rgb_display(source_image, percentile_scale=True)
    segmentation_rgb = ensure_rgb_display(
        segmented_image,
        percentile_scale=False,
    )
    if source_rgb.shape != segmentation_rgb.shape:
        raise ValueError(
            "Source and segmentation shapes must match: "
            f"source={source_rgb.shape}, segmented={segmentation_rgb.shape}"
        )

    segmentation_mask = np.any(segmentation_rgb > 0, axis=2)
    overlay = source_rgb.astype(np.float32)
    overlay[segmentation_mask] = (
        (1.0 - alpha) * overlay[segmentation_mask]
        + alpha * segmentation_rgb[segmentation_mask].astype(np.float32)
    )
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def save_overlay_png(image_rgb: np.ndarray, output_path: Path) -> None:
    """Save an RGB overlay as a PNG image.

    Args:
        image_rgb (np.ndarray): RGB uint8 overlay image.
        output_path (Path): Destination PNG path.

    Returns:
        None: The image is written to disk.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = cv2.imwrite(
        str(output_path),
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
    )
    if not written:
        raise RuntimeError(f"Failed to write overlay PNG: {output_path}")


def process_one_overlay(job: OverlayInput) -> None:
    """Render and save one patient segmentation overlay.

    Args:
        job (OverlayInput): Validated paths for one overlay job.

    Returns:
        None: One overlay PNG is written to disk.
    """
    source_image = load_tiff(job.source_path)
    segmented_image = load_tiff(job.segmented_path)
    overlay = render_overlay(source_image, segmented_image)
    save_overlay_png(overlay, job.output_path)


def main() -> None:
    """Run batch patient QC overlay rendering with per-image error handling.

    Args:
        None: This function reads configuration from command-line arguments.

    Returns:
        None: Prints per-image statuses and a final processing summary.
    """
    args = parse_args()
    try:
        source_paths = discover_source_images(
            input_root=args.input_root,
            input_file=args.input_file,
            limit=args.limit,
        )
    except Exception as exc:
        print(f"[ERROR] Discovery failed: {exc}")
        print("[SUMMARY] OK=0 SKIP=0 ERROR=1 TOTAL=0")
        return

    ok_count = 0
    skip_count = 0
    error_count = 0
    for source_path in source_paths:
        try:
            job = build_overlay_input(
                source_path=source_path,
                input_root=args.input_root,
                output_root=args.output_root,
            )
            if job.output_path.is_file() and not args.overwrite:
                skip_count += 1
                print(f"[SKIP] {job.output_path}")
                continue

            process_one_overlay(job)
            ok_count += 1
            print(f"[OK] {job.output_path}")
        except Exception as exc:
            error_count += 1
            print(f"[ERROR] {source_path} -> {exc}")

    print(
        f"[SUMMARY] OK={ok_count} SKIP={skip_count} ERROR={error_count} "
        f"TOTAL={len(source_paths)}"
    )


if __name__ == "__main__":
    main()
