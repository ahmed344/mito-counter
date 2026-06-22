#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "Calpaine_3" / "Processed"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "Calpaine_3" / "results" / "overlay_qc"
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
    """Container for one Calpaine_3 overlay rendering job.

    Args:
        source_path (Path): Path to the source grayscale or RGB TIFF image.
        segmented_path (Path): Path to the mitochondria segmentation TIFF visualization.
        metadata_path (Path): Path to the per-image JSON metadata file.
        output_path (Path): Path where the rendered PNG overlay will be written.
        image_label (str): Image stem shared by source, metadata, and segmentation files.

    Returns:
        None: Dataclass field container.
    """

    source_path: Path
    segmented_path: Path
    metadata_path: Path
    output_path: Path
    image_label: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Calpaine_3 QC overlay script.

    Args:
        None: This function reads arguments from the process command line.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Render Calpaine_3 full-frame QC overlays by alpha-blending mitochondria "
            "segmentation visualizations over source images."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory scanned recursively for primary TIFF images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where PNG overlay outputs are written.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Optional single TIFF image to process instead of scanning --input-root.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of images processed after discovery.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite overlay files even when the PNG already exists.",
    )
    return parser.parse_args()


def strip_source_suffix(path: Path) -> str:
    """Return the image label for either plain or corrected source names.

    Args:
        path (Path): Source image path whose stem may end with ``_corrected``.

    Returns:
        str: Image label used to find sibling metadata and segmentation files.
    """
    stem = path.stem
    if stem.endswith("_corrected"):
        return stem[: -len("_corrected")]
    return stem


def is_primary_source_image(path: Path) -> bool:
    """Check whether a TIFF path is a primary corrected image.

    Args:
        path (Path): Candidate TIFF path.

    Returns:
        bool: ``True`` when the path should be considered for overlay rendering.
    """
    name_lower = path.name.lower()
    if not (name_lower.endswith("_corrected.tif") or name_lower.endswith("_corrected.tiff")):
        return False
    stem = strip_source_suffix(path)
    stem_lower = stem.lower()
    if stem_lower in EXCLUDED_SOURCE_STEMS:
        return False
    return not stem_lower.endswith(EXCLUDED_SOURCE_SUFFIXES)


def discover_source_images(
    input_root: Path,
    input_file: Path | None,
    limit: int | None,
) -> list[Path]:
    """Discover corrected TIFF images for Calpaine_3 overlay rendering.

    Args:
        input_root (Path): Root directory scanned recursively for corrected TIFF images.
        input_file (Path | None): Optional single corrected TIFF image to process.
        limit (int | None): Optional maximum number of image paths to return.

    Returns:
        list[Path]: Sorted list of corrected source image paths.
    """
    if input_file is not None:
        if not input_file.is_file():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        if not is_primary_source_image(input_file):
            raise ValueError(f"Input file is not a corrected source TIFF: {input_file}")
        source_paths = [input_file]
    else:
        source_paths = sorted(
            path
            for path in input_root.rglob("*")
            if path.is_file() and is_primary_source_image(path)
        )

    if limit is not None:
        return source_paths[:limit]
    return source_paths


def build_overlay_input(
    source_path: Path,
    input_root: Path,
    output_root: Path,
) -> OverlayInput:
    """Resolve sibling files and output path for one Calpaine_3 source image.

    Args:
        source_path (Path): Path to the source TIFF image.
        input_root (Path): Root processed directory used to compute relative outputs.
        output_root (Path): Root directory for saved overlay PNGs.

    Returns:
        OverlayInput: Resolved input bundle for rendering one overlay.
    """
    image_label = strip_source_suffix(source_path)
    segmented_path = source_path.with_name(f"{image_label}_segmented.tif")
    metadata_path = source_path.with_name(f"{image_label}.json")
    if not segmented_path.is_file():
        raise FileNotFoundError(f"Segmented mitochondria TIFF not found: {segmented_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata JSON not found: {metadata_path}")

    relative_parent = source_path.parent.relative_to(input_root)
    output_path = output_root / relative_parent / f"{image_label}_overlay.png"
    return OverlayInput(
        source_path=source_path,
        segmented_path=segmented_path,
        metadata_path=metadata_path,
        output_path=output_path,
        image_label=image_label,
    )


def load_grayscale_or_rgb(path: Path) -> np.ndarray:
    """Read an image from a TIFF file.

    Args:
        path (Path): TIFF path to read.

    Returns:
        np.ndarray: Loaded image array.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return tiff.imread(str(path))


def to_uint8_display(image: np.ndarray) -> np.ndarray:
    """Convert an image array into a display-ready uint8 representation.

    Args:
        image (np.ndarray): Input grayscale or RGB-like image array.

    Returns:
        np.ndarray: Display-ready uint8 image preserving dimensionality.
    """
    if image.dtype == np.uint8:
        return image.copy()
    arr = image.astype(np.float32)
    lo = float(np.percentile(arr, 1.0))
    hi = float(np.percentile(arr, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def ensure_rgb_display(image: np.ndarray) -> np.ndarray:
    """Ensure an image is represented as an RGB uint8 array.

    Args:
        image (np.ndarray): Input grayscale or RGB-like image array.

    Returns:
        np.ndarray: RGB uint8 image with shape ``(H, W, 3)``.
    """
    display = to_uint8_display(image)
    if display.ndim == 2:
        return np.repeat(display[:, :, None], 3, axis=2)
    if display.ndim == 3 and display.shape[2] >= 3:
        return display[:, :, :3].copy()
    raise ValueError(f"Unsupported image shape for RGB display: {display.shape}")


def blend_color_layer(
    canvas_rgb: np.ndarray,
    mask: np.ndarray,
    color_rgb: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend one RGB layer into an image under a boolean mask.

    Args:
        canvas_rgb (np.ndarray): Base RGB float image array with shape ``(H, W, 3)``.
        mask (np.ndarray): Boolean mask specifying pixels to blend.
        color_rgb (np.ndarray): RGB image with shape ``(H, W, 3)``.
        alpha (float): Blend opacity in the range ``[0, 1]``.

    Returns:
        np.ndarray: Blended RGB float image array.
    """
    if alpha <= 0.0 or not np.any(mask):
        return canvas_rgb
    color_float = color_rgb.astype(np.float32)
    if color_float.shape != canvas_rgb.shape:
        raise ValueError(
            "Per-pixel color layer shape must match canvas shape: "
            f"canvas={canvas_rgb.shape}, color={color_float.shape}"
        )
    canvas_rgb[mask] = (
        (1.0 - alpha) * canvas_rgb[mask] + alpha * color_float[mask]
    )
    return canvas_rgb


def render_overlay(source_image: np.ndarray, segmented_rgb: np.ndarray) -> np.ndarray:
    """Render a Calpaine_3 QC overlay from a source image and segmentation visualization.

    Args:
        source_image (np.ndarray): Source grayscale or RGB image.
        segmented_rgb (np.ndarray): RGB-like mitochondria segmentation visualization.

    Returns:
        np.ndarray: Rendered RGB uint8 overlay image.
    """
    base_rgb = ensure_rgb_display(source_image).astype(np.float32)
    mito_rgb = ensure_rgb_display(segmented_rgb)
    if base_rgb.shape[:2] != mito_rgb.shape[:2]:
        raise ValueError(
            "Source image and mito segmentation shapes must match: "
            f"source={base_rgb.shape}, segmented={mito_rgb.shape}"
        )

    mito_mask = np.any(mito_rgb > 0, axis=2)
    overlay = blend_color_layer(base_rgb.copy(), mito_mask, mito_rgb, alpha=MITO_ALPHA)
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def save_overlay_png(image_rgb: np.ndarray, output_path: Path) -> None:
    """Write one RGB overlay image to a PNG file.

    Args:
        image_rgb (np.ndarray): RGB uint8 image to save.
        output_path (Path): Output PNG path.

    Returns:
        None: Writes the image to disk.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(
        str(output_path),
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
    )
    if not success:
        raise RuntimeError(f"Failed to write overlay PNG: {output_path}")


def process_one_overlay(job: OverlayInput) -> None:
    """Render and save one Calpaine_3 overlay image.

    Args:
        job (OverlayInput): Input bundle describing one image to render.

    Returns:
        None: Writes one overlay PNG to disk.
    """
    source_image = load_grayscale_or_rgb(job.source_path)
    segmented_rgb = load_grayscale_or_rgb(job.segmented_path)
    overlay_rgb = render_overlay(
        source_image=source_image,
        segmented_rgb=segmented_rgb,
    )
    save_overlay_png(overlay_rgb, job.output_path)


def main() -> None:
    """Run the Calpaine_3 batch QC overlay rendering pipeline.

    Args:
        None: This function does not accept arguments directly.

    Returns:
        None: Discovers images, renders overlays, and writes PNG files to disk.
    """
    args = parse_args()
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")

    source_paths = discover_source_images(
        input_root=args.input_root,
        input_file=args.input_file,
        limit=args.limit,
    )
    if not source_paths:
        raise ValueError(f"No source TIFF files found under: {args.input_root}")

    written_count = 0
    skipped_count = 0
    failed_count = 0
    for source_path in source_paths:
        try:
            job = build_overlay_input(
                source_path=source_path,
                input_root=args.input_root,
                output_root=args.output_root,
            )
        except Exception as exc:
            failed_count += 1
            print(f"[ERROR] {source_path} -> {exc}")
            continue

        if job.output_path.is_file() and not args.overwrite:
            skipped_count += 1
            print(f"[SKIP] {job.output_path}")
            continue

        try:
            process_one_overlay(job)
        except Exception as exc:
            failed_count += 1
            print(f"[ERROR] {source_path} -> {exc}")
            continue

        written_count += 1
        print(f"[OK] {job.output_path}")

    print(
        f"Finished overlays: written={written_count}, skipped={skipped_count}, "
        f"failed={failed_count}, total={len(source_paths)}"
    )


if __name__ == "__main__":
    main()
