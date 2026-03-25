#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import zlib

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
import tifffile as tiff

MEASUREMENTS_CSV = Path(
    "/workspaces/mito-counter/data/Calpaine_3/results/measurments_cleaned.csv"
)
STATISTICS_CSV = Path("/workspaces/mito-counter/data/Calpaine_3/results/statistics.csv")
PROCESSED_ROOT = Path("/workspaces/mito-counter/data/Calpaine_3/Processed")
OUTPUT_DIR = Path("/workspaces/mito-counter/data/Calpaine_3/results/figures/examples")

SAMPLE_PER_TAIL = 20
GRID_COLS_PER_TAIL = 4
LOW_QUANTILE = 0.20
HIGH_QUANTILE = 0.80
RANDOM_SEED = 42
MASK_ALPHA = 0.35
MASK_INNER_MARGIN_PX = 12

CONDITION_TO_FOLDER = {
    "Wildtype": "WT",
    "Calpain_3_Knockout": "KO-C3",
}

MUSCLE_TO_FOLDER = {
    "Soleus": "SOL",
    "Tibialis Anterior": "TA",
}


@dataclass
class ExampleTile:
    """Container for a rendered example tile.

    Args:
        image_rgb (np.ndarray): Rendered RGB tile image as uint8 array with shape (H, W, 3).
        image_name (str): Source image stem used in measurements.
        mito_id (int): Mitochondrion ID used to recover the instance mask.
        value (float): Measurement value for this selected example.
        tail (str): Tail label, either ``"low"`` or ``"high"``.
        condition (str): Biological condition name.
        muscle (str): Muscle name.
        block (int | str): Block identifier from the measurements table.

    Returns:
        None: Dataclass field container.
    """

    image_rgb: np.ndarray
    image_name: str
    mito_id: int
    value: float
    tail: str
    condition: str
    muscle: str
    block: int | str


def sanitize_name(name: str) -> str:
    """Convert free text to a filesystem-safe slug.

    Args:
        name (str): Raw text name to sanitize.

    Returns:
        str: Lowercase slug with non-alphanumeric characters replaced by underscores.
    """

    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def infer_measurements_from_statistics(
    statistics_df: pd.DataFrame, measurements_df: pd.DataFrame
) -> list[str]:
    """Build the measurement list from statistics, excluding count columns.

    Args:
        statistics_df (pd.DataFrame): Dataframe loaded from statistics CSV.
        measurements_df (pd.DataFrame): Dataframe loaded from cleaned measurements CSV.

    Returns:
        list[str]: Ordered list of measurement column names that exist in measurements data and are not count metrics.
    """

    if "Measurment" not in statistics_df.columns:
        raise KeyError("statistics.csv is missing required column: Measurment")

    result: list[str] = []
    seen: set[str] = set()
    excluded_measurements = {"count", "connected_parts"}
    for metric in statistics_df["Measurment"].dropna().astype(str):
        if metric.lower() in excluded_measurements:
            continue
        if metric not in measurements_df.columns:
            continue
        if metric in seen:
            continue
        seen.add(metric)
        result.append(metric)
    return result


def id_to_rgb(mito_id: int) -> tuple[int, int, int]:
    """Convert a mitochondrion ID into the segmentation visualization color.

    Args:
        mito_id (int): Instance ID value from the metrics table.

    Returns:
        tuple[int, int, int]: RGB color tuple used in ``*_segmented.tif`` visualization.
    """

    return (
        int((mito_id * 37 + 23) % 255),
        int((mito_id * 17 + 91) % 255),
        int((mito_id * 29 + 47) % 255),
    )


def resolve_image_paths(condition: str, muscle: str, image_name: str) -> tuple[Path, Path]:
    """Resolve corrected and segmented image paths for one row.

    Args:
        condition (str): Condition name from cleaned measurements CSV.
        muscle (str): Muscle name from cleaned measurements CSV.
        image_name (str): Image stem from cleaned measurements CSV.

    Returns:
        tuple[Path, Path]: Tuple of ``(corrected_tif_path, segmented_tif_path)``.
    """

    if condition not in CONDITION_TO_FOLDER:
        raise KeyError(f"Unsupported condition in row: {condition}")
    if muscle not in MUSCLE_TO_FOLDER:
        raise KeyError(f"Unsupported muscle in row: {muscle}")

    root = PROCESSED_ROOT / CONDITION_TO_FOLDER[condition] / MUSCLE_TO_FOLDER[muscle]
    corrected_path = root / f"{image_name}_corrected.tif"
    segmented_path = root / f"{image_name}_segmented.tif"
    return corrected_path, segmented_path


def to_uint8_display(image: np.ndarray) -> np.ndarray:
    """Convert an image array into display-ready uint8 values.

    Args:
        image (np.ndarray): Input image array in grayscale or RGB-like format.

    Returns:
        np.ndarray: Output array in uint8 preserving original dimensionality.
    """

    if image.dtype == np.uint8:
        return image.copy()

    arr = image.astype(np.float32)
    lo = float(np.percentile(arr, 1.0))
    hi = float(np.percentile(arr, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return arr


def build_overlay_tile(base_crop: np.ndarray, mask_crop: np.ndarray, alpha: float) -> np.ndarray:
    """Create an RGB tile with transparent yellow mask overlay.

    Args:
        base_crop (np.ndarray): Cropped corrected image (grayscale or RGB-like).
        mask_crop (np.ndarray): Cropped boolean mask for one mitochondrion.
        alpha (float): Transparency factor for yellow mask blend in ``[0, 1]``.

    Returns:
        np.ndarray: RGB uint8 tile array with segmentation overlay.
    """

    img = to_uint8_display(base_crop)
    if img.ndim == 2:
        rgb = np.repeat(img[:, :, None], 3, axis=2)
    elif img.ndim == 3 and img.shape[2] >= 3:
        rgb = img[:, :, :3].copy()
    else:
        raise ValueError(f"Unsupported crop shape for display: {img.shape}")

    blend = rgb.astype(np.float32)
    yellow = np.array([255.0, 255.0, 0.0], dtype=np.float32)
    blend[mask_crop] = (1.0 - alpha) * blend[mask_crop] + alpha * yellow
    return np.clip(blend, 0.0, 255.0).astype(np.uint8)


def _clamp_square_start(start: int, side: int, max_len: int) -> int:
    """Clamp square crop start coordinate within image bounds.

    Args:
        start (int): Proposed crop start coordinate along one axis.
        side (int): Square side length in pixels.
        max_len (int): Full image length in pixels for the same axis.

    Returns:
        int: Clamped coordinate satisfying ``0 <= start <= max_len - side``.
    """

    return max(0, min(start, max_len - side))


def compute_centered_square_bbox(
    mask: np.ndarray, inner_margin_px: int
) -> tuple[int, int, int, int] | None:
    """Compute centered square crop that keeps mask away from tile borders.

    Args:
        mask (np.ndarray): Boolean mask array with shape (H, W).
        inner_margin_px (int): Required minimal pixel margin between mask and tile edges.

    Returns:
        tuple[int, int, int, int] | None: Square bounding box ``(x0, y0, x1, y1)`` inclusive, or ``None`` if no valid square can satisfy edge-margin constraints.
    """

    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None

    min_y = int(ys.min())
    max_y = int(ys.max())
    min_x = int(xs.min())
    max_x = int(xs.max())
    image_h, image_w = mask.shape
    max_side = min(image_h, image_w)

    object_h = max_y - min_y + 1
    object_w = max_x - min_x + 1
    side = max(object_h, object_w) + (2 * inner_margin_px)
    side = min(max(1, side), max_side)

    center_y = 0.5 * (min_y + max_y)
    center_x = 0.5 * (min_x + max_x)

    for _ in range(16):
        x0 = int(round(center_x - (side / 2.0)))
        y0 = int(round(center_y - (side / 2.0)))
        x0 = _clamp_square_start(x0, side, image_w)
        y0 = _clamp_square_start(y0, side, image_h)
        x1 = x0 + side - 1
        y1 = y0 + side - 1

        local_min_x = min_x - x0
        local_max_x = max_x - x0
        local_min_y = min_y - y0
        local_max_y = max_y - y0
        touches_edge = (
            local_min_x <= 0
            or local_min_y <= 0
            or local_max_x >= (side - 1)
            or local_max_y >= (side - 1)
        )
        has_margin = (
            local_min_x >= inner_margin_px
            and local_min_y >= inner_margin_px
            and local_max_x <= (side - 1 - inner_margin_px)
            and local_max_y <= (side - 1 - inner_margin_px)
        )
        if not touches_edge and has_margin:
            return x0, y0, x1, y1

        if side >= max_side:
            break
        side = min(max_side, side + max(2, 2 * inner_margin_px))

    return None


def compute_tail_pools(
    data: pd.DataFrame, measurement: str
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Compute full low/high tail pools for one measurement.

    Args:
        data (pd.DataFrame): Full cleaned measurements dataframe.
        measurement (str): Measurement column to evaluate.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, float, float]: Tuple of
        ``(low_pool_df, high_pool_df, q20, q80)``.
    """

    work = data.copy()
    work[measurement] = pd.to_numeric(work[measurement], errors="coerce")
    work = work.dropna(subset=[measurement])
    if work.empty:
        return work, work, float("nan"), float("nan")

    q20 = float(work[measurement].quantile(LOW_QUANTILE))
    q80 = float(work[measurement].quantile(HIGH_QUANTILE))
    low = work[work[measurement] <= q20]
    high = work[work[measurement] >= q80]
    low = low.reset_index(drop=True)
    high = high.reset_index(drop=True)
    return low, high, q20, q80


def select_valid_tiles_with_resampling(
    pool_df: pd.DataFrame,
    measurement: str,
    tail: str,
    image_cache: dict[Path, np.ndarray],
    target_count: int,
    random_seed: int,
) -> list[ExampleTile]:
    """Select valid tiles from a tail pool with replacement-style resampling.

    Args:
        pool_df (pd.DataFrame): Candidate rows from one tail pool.
        measurement (str): Measurement column name to read value from.
        tail (str): Tail label, either ``"low"`` or ``"high"``.
        image_cache (dict[Path, np.ndarray]): In-memory cache of loaded TIFF arrays keyed by path.
        target_count (int): Number of valid tiles requested from this pool.
        random_seed (int): Seed used for deterministic candidate shuffling.

    Returns:
        list[ExampleTile]: List of valid rendered tiles, up to ``target_count``.
    """

    if pool_df.empty or target_count <= 0:
        return []

    rng = np.random.default_rng(random_seed)
    order = rng.permutation(len(pool_df))
    selected: list[ExampleTile] = []
    for idx in order:
        row = pool_df.iloc[int(idx)]
        tile = row_to_tile(row, measurement, tail, image_cache)
        if tile is None:
            continue
        selected.append(tile)
        if len(selected) >= target_count:
            break

    selected = sorted(
        selected,
        key=lambda tile: tile.value,
        reverse=(tail == "high"),
    )
    return selected


def row_to_tile(
    row: pd.Series,
    measurement: str,
    tail: str,
    image_cache: dict[Path, np.ndarray],
) -> ExampleTile | None:
    """Render one sampled row into an overlay tile.

    Args:
        row (pd.Series): One sampled dataframe row.
        measurement (str): Measurement column name to read value from.
        tail (str): Tail label, either ``"low"`` or ``"high"``.
        image_cache (dict[Path, np.ndarray]): In-memory cache of loaded TIFF arrays keyed by path.

    Returns:
        ExampleTile | None: Rendered tile metadata object, or ``None`` if row cannot be resolved.
    """

    condition = str(row["Condition"])
    muscle = str(row["Muscle"])
    image_name = str(row["image"])
    mito_id = int(row["Id"])
    metric_value = float(row[measurement])
    block = row.get("Block", "")

    corrected_path, segmented_path = resolve_image_paths(condition, muscle, image_name)
    if not corrected_path.exists():
        print(f"[WARN] Missing corrected image: {corrected_path}")
        return None
    if not segmented_path.exists():
        print(f"[WARN] Missing segmented image: {segmented_path}")
        return None

    if corrected_path not in image_cache:
        image_cache[corrected_path] = tiff.imread(str(corrected_path))
    if segmented_path not in image_cache:
        image_cache[segmented_path] = tiff.imread(str(segmented_path))

    corrected = image_cache[corrected_path]
    segmented = image_cache[segmented_path]

    if segmented.ndim != 3 or segmented.shape[2] < 3:
        print(f"[WARN] Unexpected segmented shape for {segmented_path}: {segmented.shape}")
        return None

    rgb = id_to_rgb(mito_id)
    mask = np.all(segmented[:, :, :3] == np.array(rgb, dtype=segmented.dtype), axis=2)
    if not np.any(mask):
        print(
            f"[WARN] No pixels found for image={image_name}, id={mito_id}, "
            f"measurement={measurement}, tail={tail}"
        )
        return None

    bbox = compute_centered_square_bbox(mask, MASK_INNER_MARGIN_PX)
    if bbox is None:
        print(
            f"[WARN] No valid square crop for image={image_name}, id={mito_id}, "
            f"measurement={measurement}, tail={tail}. Skipping example."
        )
        return None

    x0, y0, x1, y1 = bbox
    corrected_crop = corrected[y0 : y1 + 1, x0 : x1 + 1]
    mask_crop = mask[y0 : y1 + 1, x0 : x1 + 1]
    tile = build_overlay_tile(corrected_crop, mask_crop, alpha=MASK_ALPHA)

    return ExampleTile(
        image_rgb=tile,
        image_name=image_name,
        mito_id=mito_id,
        value=metric_value,
        tail=tail,
        condition=condition,
        muscle=muscle,
        block=block,
    )


def _render_measurement_grid(
    measurement: str,
    low_tiles: list[ExampleTile],
    high_tiles: list[ExampleTile],
    q20: float,
    q80: float,
    show_tile_titles: bool,
) -> plt.Figure:
    """Render one low-vs-high tile grid figure.

    Args:
        measurement (str): Measurement name used for title and output filename.
        low_tiles (list[ExampleTile]): Tiles from the low 20% tail.
        high_tiles (list[ExampleTile]): Tiles from the high 20% tail.
        q20 (float): 20th percentile threshold for the measurement.
        q80 (float): 80th percentile threshold for the measurement.
        show_tile_titles (bool): Whether to draw image and ID title text on each tile.

    Returns:
        plt.Figure: Rendered matplotlib figure object.
    """

    low_rows = max(1, int(np.ceil(len(low_tiles) / float(GRID_COLS_PER_TAIL))))
    high_rows = max(1, int(np.ceil(len(high_tiles) / float(GRID_COLS_PER_TAIL))))
    n_rows = max(low_rows, high_rows)
    total_cols = GRID_COLS_PER_TAIL * 2
    fig, axes = plt.subplots(
        n_rows,
        total_cols,
        figsize=(24.0, 2.7 * n_rows + 1.5),
        squeeze=False,
        constrained_layout=True,
    )

    axes[0][0].set_title(f"Low 20% (<= {q20:.4g})", fontsize=14, pad=8)
    axes[0][GRID_COLS_PER_TAIL].set_title(f"High 20% (>= {q80:.4g})", fontsize=14, pad=8)

    shadow = [patheffects.withStroke(linewidth=2.2, foreground="black")]

    for row_idx in range(n_rows):
        for col_idx in range(total_cols):
            axes[row_idx][col_idx].axis("off")

    for idx, tile in enumerate(low_tiles):
        row_idx = idx // GRID_COLS_PER_TAIL
        col_idx = idx % GRID_COLS_PER_TAIL
        if row_idx < n_rows:
            ax = axes[row_idx][col_idx]
            ax.imshow(tile.image_rgb)
            if show_tile_titles:
                label = f"{tile.image_name}\nId: {tile.mito_id}"
                ax.text(
                    0.02,
                    0.98,
                    label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.5,
                    color="white",
                    path_effects=shadow,
                )

    for idx, tile in enumerate(high_tiles):
        row_idx = idx // GRID_COLS_PER_TAIL
        col_idx = GRID_COLS_PER_TAIL + (idx % GRID_COLS_PER_TAIL)
        if row_idx < n_rows:
            ax = axes[row_idx][col_idx]
            ax.imshow(tile.image_rgb)
            if show_tile_titles:
                label = f"{tile.image_name}\nId: {tile.mito_id}"
                ax.text(
                    0.02,
                    0.98,
                    label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.5,
                    color="white",
                    path_effects=shadow,
                )

    fig.suptitle(f"{measurement} examples", fontsize=18)
    return fig


def save_measurement_grids(
    measurement: str,
    low_tiles: list[ExampleTile],
    high_tiles: list[ExampleTile],
    q20: float,
    q80: float,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save labeled and unlabeled low-vs-high tile grids for one measurement.

    Args:
        measurement (str): Measurement name used for title and output filename.
        low_tiles (list[ExampleTile]): Tiles from the low 20% tail.
        high_tiles (list[ExampleTile]): Tiles from the high 20% tail.
        q20 (float): 20th percentile threshold for the measurement.
        q80 (float): 80th percentile threshold for the measurement.
        output_dir (Path): Directory where figure PNG files should be saved.

    Returns:
        tuple[Path, Path]: Tuple of ``(labeled_figure_path, unlabeled_figure_path)``.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    slug = sanitize_name(measurement)
    labeled_path = output_dir / f"{slug}_examples_with_titles.png"
    unlabeled_path = output_dir / f"{slug}_examples_no_titles.png"

    labeled_fig = _render_measurement_grid(
        measurement=measurement,
        low_tiles=low_tiles,
        high_tiles=high_tiles,
        q20=q20,
        q80=q80,
        show_tile_titles=True,
    )
    labeled_fig.savefig(labeled_path, dpi=300)
    plt.close(labeled_fig)

    unlabeled_fig = _render_measurement_grid(
        measurement=measurement,
        low_tiles=low_tiles,
        high_tiles=high_tiles,
        q20=q20,
        q80=q80,
        show_tile_titles=False,
    )
    unlabeled_fig.savefig(unlabeled_path, dpi=300)
    plt.close(unlabeled_fig)

    return labeled_path, unlabeled_path


def save_manifest(
    measurement: str, tiles: list[ExampleTile], output_dir: Path, figure_paths: tuple[Path, Path]
) -> Path:
    """Save a CSV manifest for selected examples in one measurement.

    Args:
        measurement (str): Measurement name linked to this manifest.
        tiles (list[ExampleTile]): Rendered low/high tiles that were successfully exported.
        output_dir (Path): Output directory for the manifest CSV.
        figure_paths (tuple[Path, Path]): Pair of figure PNG paths associated with this manifest.

    Returns:
        Path: Saved manifest CSV path.
    """

    labeled_figure_path, unlabeled_figure_path = figure_paths
    rows: list[dict[str, object]] = []
    for tile in tiles:
        rows.append(
            {
                "Measurement": measurement,
                "Tail": tile.tail,
                "Value": tile.value,
                "Condition": tile.condition,
                "Muscle": tile.muscle,
                "Block": tile.block,
                "Image": tile.image_name,
                "Id": tile.mito_id,
                "Figure_with_titles": str(labeled_figure_path),
                "Figure_no_titles": str(unlabeled_figure_path),
            }
        )

    slug = sanitize_name(measurement)
    manifest_path = output_dir / f"{slug}_examples_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def main() -> None:
    """Generate low-vs-high mitochondrial example grids for all measurements.

    Args:
        None: This function does not accept command-line arguments.

    Returns:
        None: Writes figures and manifests to disk and prints progress.
    """

    if not MEASUREMENTS_CSV.is_file():
        raise FileNotFoundError(f"Missing measurements file: {MEASUREMENTS_CSV}")
    if not STATISTICS_CSV.is_file():
        raise FileNotFoundError(f"Missing statistics file: {STATISTICS_CSV}")
    if not PROCESSED_ROOT.is_dir():
        raise FileNotFoundError(f"Missing processed root directory: {PROCESSED_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    measurements_df = pd.read_csv(MEASUREMENTS_CSV)
    statistics_df = pd.read_csv(STATISTICS_CSV)
    measurements = infer_measurements_from_statistics(statistics_df, measurements_df)
    if not measurements:
        raise ValueError("No non-count measurement columns found to process.")

    image_cache: dict[Path, np.ndarray] = {}
    for measurement in measurements:
        low_pool_df, high_pool_df, q20, q80 = compute_tail_pools(
            measurements_df, measurement
        )
        if low_pool_df.empty and high_pool_df.empty:
            print(f"[WARN] Skipping {measurement}: no numeric rows found.")
            continue

        seed_base = RANDOM_SEED + int(zlib.crc32(measurement.encode("utf-8")))
        low_tiles = select_valid_tiles_with_resampling(
            pool_df=low_pool_df,
            measurement=measurement,
            tail="low",
            image_cache=image_cache,
            target_count=SAMPLE_PER_TAIL,
            random_seed=seed_base,
        )
        high_tiles = select_valid_tiles_with_resampling(
            pool_df=high_pool_df,
            measurement=measurement,
            tail="high",
            image_cache=image_cache,
            target_count=SAMPLE_PER_TAIL,
            random_seed=seed_base + 1,
        )

        figure_paths = save_measurement_grids(
            measurement=measurement,
            low_tiles=low_tiles,
            high_tiles=high_tiles,
            q20=q20,
            q80=q80,
            output_dir=OUTPUT_DIR,
        )

        manifest_path = save_manifest(
            measurement=measurement,
            tiles=low_tiles + high_tiles,
            output_dir=OUTPUT_DIR,
            figure_paths=figure_paths,
        )

        print(
            f"[OK] {measurement}: low={len(low_tiles)} high={len(high_tiles)} "
            f"-> {figure_paths[0].name}, {figure_paths[1].name}, {manifest_path.name}"
        )


if __name__ == "__main__":
    main()
