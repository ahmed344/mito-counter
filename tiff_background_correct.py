#!/usr/bin/env python3
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

import cv2
import numpy as np
import tifffile as tif
import yaml


DEFAULT_INPUT_ROOTS = {
    "calpaine_3": "/workspaces/mito-counter/data/Calpaine_3/Processed",
    "dmd": "/workspaces/mito-counter/data/DMD/Processed",
}
DEFAULT_INPUT_ROOT = DEFAULT_INPUT_ROOTS["calpaine_3"]
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tiff_background_correct.yaml"
)
# Derived pipeline products under Processed/ must not be re-corrected.
EXCLUDED_STEM_SUFFIXES = (
    "_corrected",
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


@dataclass(frozen=True)
class CorrectionConfig:
    """Validated background-correction configuration.

    Args:
        dataset (str): Dataset preset name.
        input_root (Optional[str]): Root directory scanned for TIFF images.
        input_file (Optional[str]): Optional single TIFF path.
        gaussian_enabled (bool): Whether Gaussian flat-field correction is applied.
        sigma (int): Gaussian illumination-estimation sigma.
        background_downsample_factor (int): Coarse-resolution factor used for
            Gaussian illumination estimation.
        clahe_enabled (bool): Whether CLAHE is applied after correction.
        clahe_clip_limit (float): CLAHE contrast clip limit.
        clahe_tile_grid_size (tuple[int, int]): CLAHE grid columns and rows.
        minmax_stretch_enabled (bool): Whether exact min-max stretching is applied.
        percentile_stretch_enabled (bool): Whether percentile stretching is applied.
        percentile_lower (float): Lower percentile used for stretching.
        percentile_upper (float): Upper percentile used for stretching.
        stretch_output_range (tuple[float, float]): Stretch output minimum and maximum.
        workers (int): Parallel worker count.
        dry_run (bool): Whether to print outputs without processing images.

    Returns:
        None: Dataclass instances contain immutable validated settings.
    """

    dataset: str
    input_root: Optional[str]
    input_file: Optional[str]
    gaussian_enabled: bool
    sigma: int
    background_downsample_factor: int
    clahe_enabled: bool
    clahe_clip_limit: float
    clahe_tile_grid_size: tuple[int, int]
    minmax_stretch_enabled: bool
    percentile_stretch_enabled: bool
    percentile_lower: float
    percentile_upper: float
    stretch_output_range: tuple[float, float]
    workers: int
    dry_run: bool


def _is_source_tiff(filename: str) -> bool:
    """Return whether a filename is a source TIFF rather than a derived product.

    Args:
        filename (str): Basename of a candidate TIFF file.

    Returns:
        bool: ``True`` when the file is a ``.tif``/``.tiff`` whose stem does not
            end with a known derived-pipeline suffix.
    """
    lower = filename.lower()
    if not lower.endswith((".tif", ".tiff")):
        return False
    stem = os.path.splitext(lower)[0]
    return not stem.endswith(EXCLUDED_STEM_SUFFIXES)


def find_tiff_files(root: str) -> Iterable[str]:
    """Yield TIFF file paths under a root folder.

    Args:
        root (str): Root directory to search for TIFF files.

    Returns:
        Iterable[str]: TIFF file paths excluding derived pipeline products
            (``*_corrected``, ``*_segmented``, cell/sarcomere masks, etc.).
    """
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if _is_source_tiff(name):
                yield os.path.join(dirpath, name)


def resolve_input_root(input_root: Optional[str], dataset: str) -> str:
    """Resolve the input root from CLI override or dataset preset.

    Args:
        input_root (Optional[str]): User-provided input root override.
        dataset (str): Dataset key used to select default processed root.

    Returns:
        str: Effective input root path for TIFF discovery.
    """
    if input_root:
        return input_root
    return DEFAULT_INPUT_ROOTS.get(dataset, DEFAULT_INPUT_ROOT)


def load_yaml_config(config_path: str) -> dict[str, Any]:
    """Load background-correction settings from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict[str, Any]: Parsed top-level configuration mapping.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing background-correction config: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Background-correction config must be a YAML mapping")
    return config


def _config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return and validate one YAML configuration section.

    Args:
        config (dict[str, Any]): Parsed top-level configuration.
        name (str): Required section name.

    Returns:
        dict[str, Any]: Validated section mapping.
    """
    section = config.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return section


def resolve_config(
    args: argparse.Namespace, yaml_config: dict[str, Any]
) -> CorrectionConfig:
    """Resolve CLI overrides over YAML settings and validate the result.

    Args:
        args (argparse.Namespace): Parsed command-line overrides.
        yaml_config (dict[str, Any]): Parsed YAML configuration.

    Returns:
        CorrectionConfig: Fully resolved and validated settings.
    """
    paths = _config_section(yaml_config, "paths")
    processing = _config_section(yaml_config, "processing")
    runtime = _config_section(yaml_config, "runtime")

    dataset = args.dataset if args.dataset is not None else paths.get("dataset")
    if dataset not in DEFAULT_INPUT_ROOTS:
        raise ValueError(
            f"dataset must be one of {sorted(DEFAULT_INPUT_ROOTS)}, got {dataset!r}"
        )

    if args.input_file is not None:
        input_root = paths.get("input_root")
        input_file = args.input_file
    elif args.input_root is not None:
        input_root = args.input_root
        input_file = None
    elif args.dataset is not None:
        input_root = None
        input_file = None
    else:
        input_root = paths.get("input_root")
        input_file = paths.get("input_file")
    for name, value in (("input_root", input_root), ("input_file", input_file)):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{name} must be a path string or null")

    gaussian_enabled = (
        args.gaussian
        if args.gaussian is not None
        else processing.get("gaussian_enabled", True)
    )
    if not isinstance(gaussian_enabled, bool):
        raise ValueError("gaussian_enabled must be true or false")

    sigma = args.sigma if args.sigma is not None else processing.get("sigma")
    if isinstance(sigma, bool) or not isinstance(sigma, int) or sigma <= 0:
        raise ValueError("sigma must be a positive integer")

    background_downsample_factor = (
        args.background_downsample_factor
        if args.background_downsample_factor is not None
        else processing.get("background_downsample_factor", 1)
    )
    if (
        isinstance(background_downsample_factor, bool)
        or not isinstance(background_downsample_factor, int)
        or background_downsample_factor <= 0
    ):
        raise ValueError("background_downsample_factor must be a positive integer")

    clahe_enabled = (
        args.clahe
        if args.clahe is not None
        else processing.get("clahe_enabled")
    )
    if not isinstance(clahe_enabled, bool):
        raise ValueError("clahe_enabled must be true or false")

    clahe_clip_limit = (
        args.clahe_clip_limit
        if args.clahe_clip_limit is not None
        else processing.get("clahe_clip_limit")
    )
    if (
        isinstance(clahe_clip_limit, bool)
        or not isinstance(clahe_clip_limit, (int, float))
        or clahe_clip_limit <= 0
    ):
        raise ValueError("clahe_clip_limit must be a positive number")

    tile_grid = (
        args.clahe_tile_size
        if args.clahe_tile_size is not None
        else processing.get("clahe_tile_grid_size")
    )
    if (
        not isinstance(tile_grid, (list, tuple))
        or len(tile_grid) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in tile_grid
        )
    ):
        raise ValueError(
            "clahe_tile_grid_size must contain two positive integers"
        )

    minmax_stretch_enabled = (
        args.minmax_stretch
        if args.minmax_stretch is not None
        else processing.get("minmax_stretch_enabled")
    )
    if not isinstance(minmax_stretch_enabled, bool):
        raise ValueError("minmax_stretch_enabled must be true or false")

    percentile_stretch_enabled = (
        args.percentile_stretch
        if args.percentile_stretch is not None
        else processing.get("percentile_stretch_enabled")
    )
    if not isinstance(percentile_stretch_enabled, bool):
        raise ValueError("percentile_stretch_enabled must be true or false")

    enabled_contrast_methods = sum(
        (
            clahe_enabled,
            minmax_stretch_enabled,
            percentile_stretch_enabled,
        )
    )
    if enabled_contrast_methods > 1:
        raise ValueError(
            "Enable at most one contrast method: CLAHE, min-max stretching, "
            "or percentile stretching"
        )

    percentile_lower = (
        args.percentile_lower
        if args.percentile_lower is not None
        else processing.get("percentile_lower")
    )
    percentile_upper = (
        args.percentile_upper
        if args.percentile_upper is not None
        else processing.get("percentile_upper")
    )
    if (
        isinstance(percentile_lower, bool)
        or isinstance(percentile_upper, bool)
        or not isinstance(percentile_lower, (int, float))
        or not isinstance(percentile_upper, (int, float))
        or not 0 <= percentile_lower < percentile_upper <= 100
    ):
        raise ValueError(
            "percentile bounds must satisfy 0 <= lower < upper <= 100"
        )

    stretch_output_range = (
        args.stretch_output_range
        if args.stretch_output_range is not None
        else processing.get("stretch_output_range")
    )
    if (
        not isinstance(stretch_output_range, (list, tuple))
        or len(stretch_output_range) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in stretch_output_range
        )
        or stretch_output_range[0] >= stretch_output_range[1]
    ):
        raise ValueError(
            "stretch_output_range must contain two increasing numeric values"
        )

    configured_workers = (
        args.workers if args.workers is not None else runtime.get("workers")
    )
    workers = (
        max(1, (os.cpu_count() or 1) - 1)
        if configured_workers is None
        else configured_workers
    )
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer or null")

    dry_run = (
        args.dry_run if args.dry_run is not None else runtime.get("dry_run")
    )
    if not isinstance(dry_run, bool):
        raise ValueError("dry_run must be true or false")

    return CorrectionConfig(
        dataset=dataset,
        input_root=input_root,
        input_file=input_file,
        gaussian_enabled=gaussian_enabled,
        sigma=sigma,
        background_downsample_factor=background_downsample_factor,
        clahe_enabled=clahe_enabled,
        clahe_clip_limit=float(clahe_clip_limit),
        clahe_tile_grid_size=(int(tile_grid[0]), int(tile_grid[1])),
        minmax_stretch_enabled=minmax_stretch_enabled,
        percentile_stretch_enabled=percentile_stretch_enabled,
        percentile_lower=float(percentile_lower),
        percentile_upper=float(percentile_upper),
        stretch_output_range=(
            float(stretch_output_range[0]),
            float(stretch_output_range[1]),
        ),
        workers=workers,
        dry_run=dry_run,
    )


def shading_correct_flatfield(
    raw: np.ndarray,
    sigma: int = 400,
    eps: float = 1e-6,
    background_downsample_factor: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply flat-field background correction with a Gaussian illumination model.

    Args:
        raw (np.ndarray): 2D image array.
        sigma (int): Gaussian sigma for illumination estimation.
        eps (float): Small epsilon to avoid division by zero.
        background_downsample_factor (int): Positive factor used to reduce the
            image before blurring. The Gaussian sigma is reduced by the same
            factor, and the illumination estimate is restored to full resolution.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Corrected floating-point
            pixels, corrected pixels in the input dtype, and the illumination
            estimate.
    """
    # Work in float for stable correction math.
    raw_f = raw.astype(np.float32)

    if (
        isinstance(background_downsample_factor, bool)
        or not isinstance(background_downsample_factor, int)
        or background_downsample_factor <= 0
    ):
        raise ValueError("background_downsample_factor must be a positive integer")

    # Estimate slowly varying illumination at a cheaper coarse resolution.
    if background_downsample_factor == 1:
        illum = cv2.GaussianBlur(
            raw_f,
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT,
        )
    else:
        height, width = raw_f.shape
        coarse_size = (
            max(1, int(round(width / background_downsample_factor))),
            max(1, int(round(height / background_downsample_factor))),
        )
        coarse = cv2.resize(raw_f, coarse_size, interpolation=cv2.INTER_AREA)
        coarse_sigma = sigma / background_downsample_factor
        coarse_illum = cv2.GaussianBlur(
            coarse,
            ksize=(0, 0),
            sigmaX=coarse_sigma,
            sigmaY=coarse_sigma,
            borderType=cv2.BORDER_REFLECT,
        )
        illum = cv2.resize(
            coarse_illum, (width, height), interpolation=cv2.INTER_LINEAR
        )

    # Flat-field division with a robust scale factor.
    scale = np.median(illum[illum > 0]) if np.any(illum > 0) else np.median(illum)
    corrected = raw_f / (illum + eps) * scale

    if np.issubdtype(raw.dtype, np.integer):
        dtype_limits = np.iinfo(raw.dtype)
        corrected_native = np.clip(
            np.rint(corrected), dtype_limits.min, dtype_limits.max
        ).astype(raw.dtype)
    elif np.issubdtype(raw.dtype, np.floating):
        corrected_native = corrected.astype(raw.dtype)
    else:
        raise TypeError(f"Unsupported TIFF image dtype: {raw.dtype}")

    return corrected, corrected_native, illum


def apply_clahe(
    image: np.ndarray, clip_limit: float, tile_grid_size: tuple[int, int]
) -> np.ndarray:
    """Apply contrast-limited adaptive histogram equalization to an image.

    Args:
        image (np.ndarray): Corrected 2D uint8 or uint16 image.
        clip_limit (float): Positive CLAHE contrast clip limit.
        tile_grid_size (tuple[int, int]): Positive grid columns and rows.

    Returns:
        np.ndarray: CLAHE-enhanced image with the same dtype and shape.
    """
    if image.ndim != 2:
        raise ValueError(f"CLAHE requires a 2D grayscale image, got shape {image.shape}")
    if image.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise TypeError(
            f"CLAHE requires uint8 or uint16 pixels, got {image.dtype}"
        )
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=tile_grid_size
    )
    return clahe.apply(np.ascontiguousarray(image))


def linear_rescale(
    image: np.ndarray,
    input_range: tuple[float, float],
    output_range: tuple[float, float],
) -> np.ndarray:
    """Linearly rescale and clip image intensities into an output interval.

    Args:
        image (np.ndarray): Numeric source image.
        input_range (tuple[float, float]): Intensities mapped to output endpoints.
        output_range (tuple[float, float]): Desired output minimum and maximum.

    Returns:
        np.ndarray: Rescaled image with the source dtype and shape.
    """
    if not (
        np.issubdtype(image.dtype, np.integer)
        or np.issubdtype(image.dtype, np.floating)
    ):
        raise TypeError(f"Stretching requires numeric real pixels, got {image.dtype}")
    if not np.isfinite(image).all():
        raise ValueError("Stretching requires finite image pixels")

    input_min, input_max = input_range
    output_min, output_max = output_range
    if output_min >= output_max:
        raise ValueError("Stretch output minimum must be less than its maximum")
    if np.issubdtype(image.dtype, np.integer):
        dtype_limits = np.iinfo(image.dtype)
        if output_min < dtype_limits.min or output_max > dtype_limits.max:
            raise ValueError(
                f"Output range {output_range} does not fit dtype {image.dtype}"
            )

    if not np.isfinite(input_min) or not np.isfinite(input_max):
        raise ValueError("Stretch input range must be finite")
    if input_max <= input_min:
        fill_value = np.rint(output_min) if np.issubdtype(
            image.dtype, np.integer
        ) else output_min
        return np.full(image.shape, fill_value, dtype=image.dtype)

    normalized = (image.astype(np.float64) - input_min) / (input_max - input_min)
    stretched = (
        np.clip(normalized, 0.0, 1.0) * (output_max - output_min) + output_min
    )
    if np.issubdtype(image.dtype, np.integer):
        stretched = np.rint(stretched)
    return stretched.astype(image.dtype)


def apply_minmax_stretch(
    image: np.ndarray, output_range: tuple[float, float]
) -> np.ndarray:
    """Stretch an image's exact minimum and maximum to an output range.

    Args:
        image (np.ndarray): Numeric source image.
        output_range (tuple[float, float]): Desired output minimum and maximum.

    Returns:
        np.ndarray: Min-max-stretched image with the source dtype and shape.
    """
    if image.size == 0:
        raise ValueError("Cannot stretch an empty image")
    return linear_rescale(
        image,
        input_range=(float(np.min(image)), float(np.max(image))),
        output_range=output_range,
    )


def apply_percentile_stretch(
    image: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
    output_range: tuple[float, float],
) -> np.ndarray:
    """Stretch configured percentile bounds and clip intensities outside them.

    Args:
        image (np.ndarray): Numeric source image.
        lower_percentile (float): Lower percentile in the inclusive 0-100 range.
        upper_percentile (float): Upper percentile in the inclusive 0-100 range.
        output_range (tuple[float, float]): Desired output minimum and maximum.

    Returns:
        np.ndarray: Percentile-stretched image with the source dtype and shape.
    """
    if image.size == 0:
        raise ValueError("Cannot stretch an empty image")
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError(
            "Percentile bounds must satisfy 0 <= lower < upper <= 100"
        )
    if not np.isfinite(image).all():
        raise ValueError("Percentile stretching requires finite image pixels")
    input_min, input_max = np.percentile(
        image, (lower_percentile, upper_percentile)
    )
    return linear_rescale(
        image,
        input_range=(float(input_min), float(input_max)),
        output_range=output_range,
    )


def build_corrected_path(input_path: str) -> str:
    """Construct the output path with a _corrected suffix.

    Args:
        input_path (str): Original TIFF file path.

    Returns:
        str: Output path with _corrected appended before the extension.
    """
    base, ext = os.path.splitext(input_path)
    return f"{base}_corrected{ext}"


def process_file(
    path: str,
    gaussian_enabled: bool,
    sigma: int,
    background_downsample_factor: int,
    clahe_enabled: bool,
    clahe_clip_limit: float,
    clahe_tile_grid_size: tuple[int, int],
    minmax_stretch_enabled: bool,
    percentile_stretch_enabled: bool,
    percentile_lower: float,
    percentile_upper: float,
    stretch_output_range: tuple[float, float],
) -> str:
    """Read, optionally correct and enhance, and write a corrected TIFF.

    Args:
        path (str): Input TIFF path.
        gaussian_enabled (bool): Whether to apply Gaussian flat-field correction.
        sigma (int): Gaussian sigma for illumination estimation.
        background_downsample_factor (int): Coarse-resolution factor used for
            Gaussian illumination estimation.
        clahe_enabled (bool): Whether to apply CLAHE after correction.
        clahe_clip_limit (float): CLAHE contrast clip limit.
        clahe_tile_grid_size (tuple[int, int]): CLAHE grid columns and rows.
        minmax_stretch_enabled (bool): Whether to apply exact min-max stretching.
        percentile_stretch_enabled (bool): Whether to apply percentile stretching.
        percentile_lower (float): Lower stretching percentile.
        percentile_upper (float): Upper stretching percentile.
        stretch_output_range (tuple[float, float]): Stretch output endpoints.

    Returns:
        str: Output path of the corrected TIFF.
    """
    out_path = build_corrected_path(path)
    raw_img = tif.imread(path)
    if gaussian_enabled:
        _, corrected_native, _ = shading_correct_flatfield(
            raw_img,
            sigma=sigma,
            background_downsample_factor=background_downsample_factor,
        )
    else:
        corrected_native = raw_img
    if clahe_enabled:
        corrected_native = apply_clahe(
            corrected_native,
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_grid_size,
        )
    elif minmax_stretch_enabled:
        corrected_native = apply_minmax_stretch(
            corrected_native, output_range=stretch_output_range
        )
    elif percentile_stretch_enabled:
        corrected_native = apply_percentile_stretch(
            corrected_native,
            lower_percentile=percentile_lower,
            upper_percentile=percentile_upper,
            output_range=stretch_output_range,
        )
    tif.imwrite(out_path, corrected_native, photometric="minisblack")
    return out_path


def resolve_target_files(input_file: Optional[str], input_root: str) -> Iterable[str]:
    """Resolve target TIFF files from a single file or a root folder.

    Args:
        input_file (Optional[str]): Optional path to one TIFF file to process.
        input_root (str): Root directory scanned when ``input_file`` is not provided.

    Returns:
        Iterable[str]: Iterable of TIFF file paths to process.
    """
    if input_file:
        return [input_file]
    return list(find_tiff_files(input_root))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Apply YAML-configured optional Gaussian flat-field correction and "
            "one optional contrast method to TIFF images, writing "
            "'*_corrected.tif' beside each source."
        ),
        epilog=(
            "Examples:\n"
            "  Use all settings from the default YAML:\n"
            "    python tiff_background_correct.py\n"
            "  Process one image:\n"
            "    python tiff_background_correct.py --input-file "
            "/workspaces/mito-counter/data/DMD/Processed/DMD/EOM/EOM_DMD_1-1900X-0011.tif "
            "--sigma 500\n"
            "  Disable Gaussian flat-field correction:\n"
            "    python tiff_background_correct.py --no-gaussian\n"
            "  Disable CLAHE:\n"
            "    python tiff_background_correct.py --no-clahe\n"
            "  Apply percentile stretching instead of CLAHE:\n"
            "    python tiff_background_correct.py --no-clahe "
            "--percentile-stretch --percentile-lower 1 --percentile-upper 99.8\n"
            "  Process a dataset tree:\n"
            "    python tiff_background_correct.py --dataset dmd --dry-run\n"
            "\n"
            "Notes:\n"
            "  - CLI processing and path options override values from --config.\n"
            "  - --input-file overrides --input-root and --dataset scanning.\n"
            "  - --help is available as: -h or --help."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML configuration path (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DEFAULT_INPUT_ROOTS.keys()),
        default=None,
        help="Dataset preset used when --input-root is not provided.",
    )
    parser.add_argument(
        "--input-root",
        default=None,
        help="Root folder of TIFF files. Overrides --dataset preset when provided.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Single TIFF file path to process. Overrides --input-root and --dataset scanning.",
    )
    parser.add_argument(
        "--gaussian",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable Gaussian flat-field background correction.",
    )
    parser.add_argument(
        "--sigma",
        type=int,
        default=None,
        help="Gaussian sigma override for background estimation.",
    )
    parser.add_argument(
        "--background-downsample-factor",
        type=int,
        default=None,
        help=(
            "Positive coarse-resolution factor for background estimation; "
            "1 uses full resolution."
        ),
    )
    parser.add_argument(
        "--clahe",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable CLAHE after background correction.",
    )
    parser.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=None,
        help="Positive CLAHE clip-limit override.",
    )
    parser.add_argument(
        "--clahe-tile-size",
        type=int,
        nargs=2,
        metavar=("COLUMNS", "ROWS"),
        default=None,
        help="CLAHE tile-grid override as two positive integers.",
    )
    parser.add_argument(
        "--minmax-stretch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable exact per-image min-max stretching.",
    )
    parser.add_argument(
        "--percentile-stretch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable percentile-based stretching.",
    )
    parser.add_argument(
        "--percentile-lower",
        type=float,
        default=None,
        help="Lower percentile override for percentile stretching.",
    )
    parser.add_argument(
        "--percentile-upper",
        type=float,
        default=None,
        help="Upper percentile override for percentile stretching.",
    )
    parser.add_argument(
        "--stretch-output-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Output range override for min-max or percentile stretching.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker override.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable printing planned outputs without processing.",
    )
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
    yaml_config = load_yaml_config(args.config)
    config = resolve_config(args, yaml_config)
    input_root = resolve_input_root(config.input_root, config.dataset)
    # Gather all candidate input files once.
    tiff_files = list(resolve_target_files(config.input_file, input_root))
    if not tiff_files:
        if config.input_file:
            print(f"Input TIFF file not found: {config.input_file}")
        else:
            print(f"No TIFF files found under: {input_root}")
        return

    # If dry-run, just show planned outputs.
    if config.dry_run:
        for path in tiff_files:
            out_path = build_corrected_path(path)
            print(f"[DRY RUN] {path} -> {out_path}")
        return

    # Process images in parallel for speed.
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        future_map = {
            executor.submit(
                process_file,
                path,
                config.gaussian_enabled,
                config.sigma,
                config.background_downsample_factor,
                config.clahe_enabled,
                config.clahe_clip_limit,
                config.clahe_tile_grid_size,
                config.minmax_stretch_enabled,
                config.percentile_stretch_enabled,
                config.percentile_lower,
                config.percentile_upper,
                config.stretch_output_range,
            ): path
            for path in tiff_files
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
