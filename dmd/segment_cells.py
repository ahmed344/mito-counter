#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff


SegmentationMeta = dict[str, float | str | bool]


DEFAULT_INPUT_ROOT = Path("/workspaces/mito-counter/data/DMD/Processed")
EXCLUDED_STEM_SUFFIXES = (
    "_corrected",
    "_segmented",
    "_cells",
    "_center_cell_mask",
    "_sarcomere",
    "_sarcomere_corrected",
    "_sarcomere_segmented",
    "_cell_mask",
    "_sarcomere_mask",
    "_mbands",
    "_zbands",
)
EXCLUDED_STEMS = {
    "cell_mask",
    "sarcomere_mask",
    "mbands",
    "zbands",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        None: This function reads arguments from the process command line.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Segment cell regions from grayscale TIFF images. By default, the script "
            "processes all source TIFF files under the DMD processed root and writes "
            "'*_cells.tif' masks next to each input image."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory scanned recursively for source TIFF images.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Optional single TIFF image path to process instead of scanning --input-root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output TIFF path used only together with --input-file.",
    )
    parser.add_argument(
        "--center-only",
        action="store_true",
        help="If set, keep only the best-scoring cell closest to the image center.",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=7,
        help="Odd Gaussian blur kernel size.",
    )
    parser.add_argument(
        "--close-kernel",
        type=int,
        default=9,
        help="Odd elliptical kernel size for morphological close.",
    )
    parser.add_argument(
        "--open-kernel",
        type=int,
        default=9,
        help="Odd elliptical kernel size for morphological open.",
    )
    parser.add_argument(
        "--close-iterations",
        type=int,
        default=2,
        help="Morphological close iterations.",
    )
    parser.add_argument(
        "--open-iterations",
        type=int,
        default=1,
        help="Morphological open iterations.",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=5000,
        help="Minimum connected-component area in pixels retained in the final mask.",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.01,
        help="Minimum connected-component area ratio relative to the full image.",
    )
    parser.add_argument(
        "--center-distance-weight",
        type=float,
        default=0.25,
        help="Penalty weight for normalized distance from image center when --center-only is used.",
    )
    parser.add_argument(
        "--max-hole-area",
        type=int,
        default=50000,
        help="Maximum enclosed hole area in pixels to fill in both cells and background.",
    )
    return parser.parse_args()


def ensure_positive_odd(value: int, name: str) -> int:
    """Return a positive odd kernel size.

    Args:
        value (int): Input kernel size candidate.
        name (str): Name of the parameter for error messages.

    Returns:
        int: Validated positive odd kernel size.
    """
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")
    if value % 2 == 0:
        value += 1
    return value


def load_grayscale_image(path: Path) -> np.ndarray:
    """Read a grayscale image from TIFF.

    Args:
        path (Path): Input image path.

    Returns:
        np.ndarray: Grayscale image array with shape (H, W).
    """
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    image = tiff.imread(str(path))
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {image.shape}.")
    return image


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an image array to uint8 for OpenCV operations.

    Args:
        image (np.ndarray): Input grayscale image array.

    Returns:
        np.ndarray: Converted uint8 image array with shape (H, W).
    """
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.uint16:
        return (image / 257.0).astype(np.uint8)
    image_float = image.astype(np.float32)
    image_float -= float(image_float.min())
    peak = float(image_float.max())
    if peak <= 0.0:
        return np.zeros_like(image_float, dtype=np.uint8)
    image_float = image_float * (255.0 / peak)
    return image_float.astype(np.uint8)


def inverse_otsu_threshold(image_u8: np.ndarray, blur_kernel: int) -> tuple[np.ndarray, float]:
    """Threshold a grayscale image with inverse Otsu.

    Args:
        image_u8 (np.ndarray): Grayscale image in uint8 format.
        blur_kernel (int): Odd Gaussian blur kernel size.

    Returns:
        tuple[np.ndarray, float]: Binary foreground mask (uint8 0/255) and Otsu threshold.
    """
    blurred = cv2.GaussianBlur(image_u8, (blur_kernel, blur_kernel), 0)
    otsu_threshold, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return mask, float(otsu_threshold)


def fill_small_holes(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    """Fill enclosed foreground holes up to a maximum area.

    Args:
        mask (np.ndarray): Binary mask array with values 0 or 255.
        max_hole_area (int): Maximum enclosed hole area in pixels to fill.

    Returns:
        np.ndarray: Binary mask array with only small enclosed holes filled.
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    if max_hole_area <= 0:
        return mask_u8

    background = (mask_u8 == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=8)
    border_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])

    filled = mask_u8.copy()
    for label_id in range(1, num_labels):
        if label_id in border_labels:
            continue
        hole_area = int(stats[label_id, cv2.CC_STAT_AREA])
        if hole_area <= max_hole_area:
            filled[labels == label_id] = 255
    return filled


def fill_small_holes_in_background(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    """Fill enclosed background holes up to a maximum area.

    Args:
        mask (np.ndarray): Binary mask array with values 0 or 255.
        max_hole_area (int): Maximum enclosed background-hole area in pixels to fill.

    Returns:
        np.ndarray: Binary mask where only small enclosed holes in the background are filled.
    """
    inverted = cv2.bitwise_not((mask > 0).astype(np.uint8) * 255)
    filled_background = fill_small_holes(inverted, max_hole_area=max_hole_area)
    return cv2.bitwise_not(filled_background)


def clean_mask(
    mask: np.ndarray,
    close_kernel: int,
    open_kernel: int,
    close_iterations: int,
    open_iterations: int,
    max_hole_area: int,
) -> np.ndarray:
    """Clean a binary mask with morphology and selective hole filling in both classes.

    Args:
        mask (np.ndarray): Raw binary mask with values 0 or 255.
        close_kernel (int): Elliptical kernel size for close.
        open_kernel (int): Elliptical kernel size for open.
        close_iterations (int): Number of close iterations.
        open_iterations (int): Number of open iterations.
        max_hole_area (int): Maximum enclosed hole area in pixels to fill.

    Returns:
        np.ndarray: Cleaned binary mask with values 0 or 255.
    """
    close_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    open_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_se, iterations=close_iterations)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_se, iterations=open_iterations)

    filled_cells = fill_small_holes(cleaned, max_hole_area=max_hole_area)
    return fill_small_holes_in_background(filled_cells, max_hole_area=max_hole_area)


def filter_components(
    mask: np.ndarray,
    min_component_area: int,
    min_area_ratio: float,
) -> tuple[np.ndarray, SegmentationMeta]:
    """Keep all sufficiently large connected components in a binary mask.

    Args:
        mask (np.ndarray): Binary foreground mask with values 0 or 255.
        min_component_area (int): Minimum area in pixels for retained components.
        min_area_ratio (float): Minimum area ratio relative to total image area.

    Returns:
        tuple[np.ndarray, SegmentationMeta]: Filtered binary mask (0/255) and summary metadata.
    """
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        raise RuntimeError("No foreground component found after thresholding.")

    image_area = float(binary.shape[0] * binary.shape[1])
    filtered = np.zeros_like(binary, dtype=np.uint8)
    kept_count = 0
    total_area = 0.0

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        area_ratio = area / image_area
        if area < min_component_area:
            continue
        if area_ratio < min_area_ratio:
            continue
        filtered[labels == label_id] = 255
        kept_count += 1
        total_area += float(area)

    if kept_count == 0:
        raise RuntimeError(
            "No valid cell components found. Try reducing --min-component-area or "
            "--min-area-ratio."
        )

    return filtered, {
        "component_count": float(kept_count),
        "total_area": float(total_area),
        "total_area_ratio": float(total_area / image_area),
    }


def select_center_component(
    mask: np.ndarray,
    center_distance_weight: float,
) -> tuple[np.ndarray, SegmentationMeta]:
    """Select the best-scoring central component from a filtered binary mask.

    Args:
        mask (np.ndarray): Filtered binary mask with values 0 or 255.
        center_distance_weight (float): Weight for normalized center-distance penalty.

    Returns:
        tuple[np.ndarray, SegmentationMeta]: Selected binary mask (0/255) and metadata for
        the chosen component.
    """
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        raise RuntimeError("No filtered foreground component available for center selection.")

    height, width = binary.shape
    image_area = float(height * width)
    center_xy = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    max_dist = float(np.linalg.norm(center_xy))

    best_label = -1
    best_score = -np.inf
    best_meta: SegmentationMeta = {}

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        area_ratio = area / image_area
        centroid_xy = np.array(centroids[label_id], dtype=np.float64)
        dist_norm = float(np.linalg.norm(centroid_xy - center_xy) / (max_dist + 1e-8))
        score = float(area_ratio - center_distance_weight * dist_norm)
        if score > best_score:
            best_score = score
            best_label = label_id
            best_meta = {
                "selected_area": float(area),
                "selected_area_ratio": float(area_ratio),
                "centroid_x": float(centroid_xy[0]),
                "centroid_y": float(centroid_xy[1]),
                "distance_norm": float(dist_norm),
                "score": float(score),
            }

    if best_label < 0:
        raise RuntimeError("Failed to select a center cell from the filtered components.")

    selected = np.where(labels == best_label, 255, 0).astype(np.uint8)
    return selected, best_meta


def segment_cells(
    image: np.ndarray,
    blur_kernel: int,
    close_kernel: int,
    open_kernel: int,
    close_iterations: int,
    open_iterations: int,
    max_hole_area: int,
    min_component_area: int,
    min_area_ratio: float,
    center_distance_weight: float,
    center_only: bool,
) -> tuple[np.ndarray, SegmentationMeta]:
    """Run cell segmentation on a grayscale image.

    Args:
        image (np.ndarray): Input grayscale image array with shape (H, W).
        blur_kernel (int): Odd Gaussian blur kernel size.
        close_kernel (int): Odd close kernel size for morphology.
        open_kernel (int): Odd open kernel size for morphology.
        close_iterations (int): Number of close iterations.
        open_iterations (int): Number of open iterations.
        max_hole_area (int): Maximum enclosed hole area in pixels to fill.
        min_component_area (int): Minimum area in pixels for retained components.
        min_area_ratio (float): Minimum retained area ratio to total image area.
        center_distance_weight (float): Penalty weight for center distance.
        center_only (bool): Whether to retain only the best central component.

    Returns:
        tuple[np.ndarray, SegmentationMeta]: Final binary mask (0/255) and segmentation
        metadata describing the retained components.
    """
    image_u8 = to_uint8(image)
    raw_mask, otsu_value = inverse_otsu_threshold(
        image_u8,
        blur_kernel=blur_kernel,
    )
    cleaned = clean_mask(
        raw_mask,
        close_kernel=close_kernel,
        open_kernel=open_kernel,
        close_iterations=close_iterations,
        open_iterations=open_iterations,
        max_hole_area=max_hole_area,
    )
    filtered, meta = filter_components(
        cleaned,
        min_component_area=min_component_area,
        min_area_ratio=min_area_ratio,
    )
    meta["otsu_threshold"] = float(otsu_value)
    meta["threshold_inverted"] = True
    meta["threshold_mode"] = "inverse"

    if center_only:
        selected, center_meta = select_center_component(
            filtered,
            center_distance_weight=center_distance_weight,
        )
        meta.update(center_meta)
        return selected, meta

    return filtered, meta


def is_source_image(path: Path) -> bool:
    """Return whether a TIFF path should be treated as a primary source image.

    Args:
        path (Path): Candidate filesystem path.

    Returns:
        bool: True when the path is a source TIFF, not a derived pipeline output, and has
        a sibling JSON metadata file matching the TIFF stem.
    """
    if not path.is_file():
        return False
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return False

    stem_lower = path.stem.lower()
    if stem_lower in EXCLUDED_STEMS:
        return False
    if stem_lower.endswith(EXCLUDED_STEM_SUFFIXES):
        return False
    return path.with_suffix(".json").is_file()


def build_output_path(input_path: Path) -> Path:
    """Construct the output path for a cell mask TIFF.

    Args:
        input_path (Path): Input image path.

    Returns:
        Path: Output path with a '_cells.tif' suffix.
    """
    stem = input_path.stem
    for suffix in ("_corrected", "_segmented", "_cells"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return input_path.parent / f"{stem}_cells.tif"


def resolve_input_images(input_root: Path, input_file: Path | None) -> list[Path]:
    """Resolve source TIFF images from one file or a root directory.

    Args:
        input_root (Path): Root directory scanned recursively when ``input_file`` is absent.
        input_file (Path | None): Optional single TIFF path to process.

    Returns:
        list[Path]: Sorted list of source TIFF files to process.
    """
    if input_file is not None:
        if not is_source_image(input_file):
            raise ValueError(
                "The provided --input-file must be a source TIFF, not a derived pipeline output."
            )
        return [input_file]

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root directory not found: {input_root}")

    return sorted(path for path in input_root.rglob("*") if is_source_image(path))


def write_mask(path: Path, mask: np.ndarray) -> None:
    """Write a binary mask TIFF to disk.

    Args:
        path (Path): Output TIFF path.
        mask (np.ndarray): Binary mask image with values 0 or 255.

    Returns:
        None: This function writes output to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(path), mask.astype(np.uint8))


def print_summary(
    input_path: Path,
    output_path: Path,
    meta: SegmentationMeta,
    center_only: bool,
) -> None:
    """Print a per-image segmentation summary.

    Args:
        input_path (Path): Input TIFF path.
        output_path (Path): Output TIFF path.
        meta (SegmentationMeta): Segmentation metadata.
        center_only (bool): Whether center-only mode was used.

    Returns:
        None: This function writes summary lines to stdout.
    """
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Otsu threshold: {meta['otsu_threshold']:.3f}")
    print(f"Threshold mode: {meta['threshold_mode']}")
    print(f"Kept components: {int(meta['component_count'])}")
    print(f"Kept total area: {meta['total_area']:.0f} px")
    print(f"Kept total area ratio: {meta['total_area_ratio']:.4f}")
    if center_only:
        print(f"Selected area: {meta['selected_area']:.0f} px")
        print(f"Selected area ratio: {meta['selected_area_ratio']:.4f}")
        print(f"Selected centroid: ({meta['centroid_x']:.2f}, {meta['centroid_y']:.2f})")
        print(f"Normalized center distance: {meta['distance_norm']:.4f}")
        print(f"Selection score: {meta['score']:.6f}")
    print("")


def main() -> None:
    """Run the command-line cell-segmentation workflow.

    Args:
        None: This function reads command-line arguments and processes TIFF images.

    Returns:
        None: This function writes output files and prints progress summaries.
    """
    args = parse_args()

    if args.output is not None and args.input_file is None:
        raise ValueError("--output can only be used together with --input-file.")

    blur_kernel = ensure_positive_odd(args.blur_kernel, "blur-kernel")
    close_kernel = ensure_positive_odd(args.close_kernel, "close-kernel")
    open_kernel = ensure_positive_odd(args.open_kernel, "open-kernel")
    if args.close_iterations < 0 or args.open_iterations < 0:
        raise ValueError("close-iterations and open-iterations must be >= 0.")
    if args.max_hole_area < 0:
        raise ValueError("max-hole-area must be >= 0.")
    if args.min_component_area <= 0:
        raise ValueError("min-component-area must be > 0.")
    if not 0.0 <= args.min_area_ratio <= 1.0:
        raise ValueError("min-area-ratio must be in [0, 1].")

    image_paths = resolve_input_images(args.input_root, args.input_file)
    if not image_paths:
        raise FileNotFoundError(f"No source TIFF images found under: {args.input_root}")

    print(
        f"Processing {len(image_paths)} image(s) from "
        f"{args.input_file if args.input_file is not None else args.input_root}"
    )

    succeeded = 0
    failed: list[tuple[Path, str]] = []

    for index, image_path in enumerate(image_paths, start=1):
        output_path = args.output if args.input_file is not None and args.output is not None else build_output_path(image_path)
        print(f"[{index}/{len(image_paths)}] Segmenting {image_path}")
        try:
            image = load_grayscale_image(image_path)
            mask, meta = segment_cells(
                image=image,
                blur_kernel=blur_kernel,
                close_kernel=close_kernel,
                open_kernel=open_kernel,
                close_iterations=args.close_iterations,
                open_iterations=args.open_iterations,
                max_hole_area=args.max_hole_area,
                min_component_area=args.min_component_area,
                min_area_ratio=args.min_area_ratio,
                center_distance_weight=args.center_distance_weight,
                center_only=args.center_only,
            )
            write_mask(output_path, mask)
            print_summary(
                input_path=image_path,
                output_path=output_path,
                meta=meta,
                center_only=args.center_only,
            )
            succeeded += 1
        except Exception as exc:
            failed.append((image_path, str(exc)))
            print(f"Skipped: {image_path}")
            print(f"Reason: {exc}")
            print("")

    print(f"Completed: {succeeded}/{len(image_paths)} image(s) segmented successfully.")
    if failed:
        print(f"Skipped: {len(failed)} image(s).")
        for failed_path, reason in failed:
            print(f"  {failed_path}: {reason}")


if __name__ == "__main__":
    main()
