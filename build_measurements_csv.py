#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

PIXEL_SIZE_UM = 0.0015396
INPUT_ROOT = Path("/workspaces/mito-counter/data/Calpaine_3/Processed")
OUTPUT_CSV = Path("/workspaces/mito-counter/data/Calpaine_3/results/measurments.csv")
OUTPUT_CLEANED_CSV = Path(
    "/workspaces/mito-counter/data/Calpaine_3/results/measurments_cleaned.csv"
)
MIN_MAJOR_AXIS_PX = 30.0
MIN_MINOR_AXIS_PX = 5.0

CONDITION_MAP = {
    "WT": "Wildtype",
    "KO-C3": "Calpain_3_Knockout",
}

MUSCLE_MAP = {
    "SOL": "Soleus",
    "TA": "Tibialis Anterior",
}

BLOCK_RE = re.compile(r"(?:^|[^A-Z0-9])(SOL|TA)[\s_-]*([0-9]+)", re.IGNORECASE)


def parse_block_number(path: Path) -> int:
    """Extract block number that follows SOL/TA in a filename.

    Args:
        path (Path): Metrics CSV path whose stem may contain tokens such as ``SOL_8`` or ``TA-1``.

    Returns:
        int: Parsed block number that directly follows ``SOL`` or ``TA``.
    """
    match = BLOCK_RE.search(path.stem)
    if not match:
        raise ValueError(f"Unable to parse block number from: {path.name}")
    return int(match.group(2))


def parse_image_label(path: Path) -> str:
    """Extract image label from a metrics filename.

    Args:
        path (Path): Metrics CSV path ending with ``_segmented_metrics.csv``.

    Returns:
        str: Filename stem without the ``_segmented_metrics`` suffix.
    """
    stem = path.stem
    suffix = "_segmented_metrics"
    if not stem.lower().endswith(suffix):
        raise ValueError(f"Unable to parse image label from: {path.name}")
    return stem[: -len(suffix)]


def parse_centroid(text: str) -> tuple[float, float]:
    """Parse centroid text into numeric coordinates.

    Args:
        text (str): Centroid text in the form ``"(x, y)"``.

    Returns:
        tuple[float, float]: Parsed x and y centroid coordinates.
    """
    cleaned = text.strip().lstrip("(").rstrip(")")
    x_str, y_str = [part.strip() for part in cleaned.split(",")]
    return float(x_str), float(y_str)


def get_first_value(row: dict, keys: list[str], required: bool = True) -> str | None:
    """Return the first non-empty value for candidate keys.

    Args:
        row (dict): Input record from the CSV reader.
        keys (list[str]): Ordered candidate column names to check.
        required (bool): Whether to raise if none of the keys are found with values.

    Returns:
        str | None: First matching non-empty value, or ``None`` when optional and missing.
    """
    for key in keys:
        if key in row and row[key] != "":
            return row[key]
    if required:
        raise KeyError(f"Missing required columns: {keys}")
    return None


def maybe_float(text: str | None) -> float | None:
    """Convert optional text to float.

    Args:
        text (str | None): String representation of a float, or ``None``.

    Returns:
        float | None: Parsed float when text is present, otherwise ``None``.
    """
    if text is None:
        return None
    return float(text)


def load_image_shapes(
    input_root: Path,
) -> dict[tuple[str, str, str], tuple[int, int]]:
    """Load image dimensions from per-image metadata JSON files.

    Args:
        input_root (Path): Root directory that contains processed condition/muscle folders.

    Returns:
        dict[tuple[str, str, str], tuple[int, int]]: Mapping from
        ``(Condition, Muscle, image_stem)`` to ``(width_px, height_px)``.
    """
    image_shapes: dict[tuple[str, str, str], tuple[int, int]] = {}
    metadata_paths = sorted(input_root.rglob("*.json"))
    for metadata_path in metadata_paths:
        parts = metadata_path.parts
        try:
            condition_raw = parts[parts.index("Processed") + 1]
            muscle_raw = parts[parts.index("Processed") + 2]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Unexpected path layout: {metadata_path}") from exc

        condition = CONDITION_MAP.get(condition_raw)
        muscle = MUSCLE_MAP.get(muscle_raw)
        if condition is None or muscle is None:
            continue

        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        shape = metadata.get("basic", {}).get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError(f"Missing or invalid shape in metadata: {metadata_path}")

        height_px = int(shape[0])
        width_px = int(shape[1])
        image_shapes[(condition, muscle, metadata_path.stem)] = (width_px, height_px)
    return image_shapes


def segmentation_touches_edge(
    *,
    centroid_x_px: float,
    centroid_y_px: float,
    major_axis_px: float,
    width_px: int,
    height_px: int,
) -> bool:
    """Estimate whether a segmented object touches image borders.

    Args:
        centroid_x_px (float): Object centroid x-coordinate in pixels.
        centroid_y_px (float): Object centroid y-coordinate in pixels.
        major_axis_px (float): Object major axis length in pixels.
        width_px (int): Image width in pixels.
        height_px (int): Image height in pixels.

    Returns:
        bool: ``True`` when centroid +/- major-axis half-width reaches or exceeds
        any image boundary; otherwise ``False``.
    """
    half_major = major_axis_px / 2.0
    left = centroid_x_px - half_major
    right = centroid_x_px + half_major
    top = centroid_y_px - half_major
    bottom = centroid_y_px + half_major
    return left <= 0.0 or right >= (width_px - 1) or top <= 0.0 or bottom >= (height_px - 1)


def clean_measurements_csv(
    measurements_csv: Path,
    cleaned_csv: Path,
    image_shapes: dict[tuple[str, str, str], tuple[int, int]],
    pixel_size_um: float,
    min_major_axis_px: float,
    min_minor_axis_px: float,
) -> tuple[int, int, int, int, int]:
    """Create a cleaned measurements CSV using axis-size, edge, and connectivity filters.

    Args:
        measurements_csv (Path): Source measurements CSV path.
        cleaned_csv (Path): Output cleaned CSV path.
        image_shapes (dict[tuple[str, str, str], tuple[int, int]]): Per-image shape
            mapping keyed by ``(Condition, Muscle, image)``.
        pixel_size_um (float): Pixel size in micrometers for unit conversions.
        min_major_axis_px (float): Minimum allowed major axis length in pixels.
        min_minor_axis_px (float): Minimum allowed minor axis length in pixels.

    Returns:
        tuple[int, int, int, int, int]: ``(kept_rows, removed_axis_size, removed_edge_touch, removed_disconnected_parts, total_rows)``.
    """
    kept_rows = 0
    removed_axis_size = 0
    removed_edge_touch = 0
    removed_disconnected_parts = 0
    total_rows = 0

    with open(measurements_csv, "r", newline="", encoding="utf-8") as in_handle:
        reader = csv.DictReader(in_handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in: {measurements_csv}")
        fieldnames = reader.fieldnames

        cleaned_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(cleaned_csv, "w", newline="", encoding="utf-8") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                total_rows += 1
                major_axis_um = float(get_first_value(row, ["Major_axis_length"]))
                minor_axis_um = float(get_first_value(row, ["Minor_axis_length"]))
                major_axis_px = major_axis_um / pixel_size_um
                minor_axis_px = minor_axis_um / pixel_size_um
                if (
                    major_axis_px <= min_major_axis_px
                    or minor_axis_px <= min_minor_axis_px
                ):
                    removed_axis_size += 1
                    continue

                condition = get_first_value(row, ["Condition"])
                muscle = get_first_value(row, ["Muscle"])
                image = get_first_value(row, ["image"])
                shape_key = (condition, muscle, image)
                if shape_key not in image_shapes:
                    raise KeyError(f"Missing shape metadata for: {shape_key}")

                width_px, height_px = image_shapes[shape_key]
                centroid_text = get_first_value(row, ["Centroid", "centroid"])
                centroid_x_um, centroid_y_um = parse_centroid(centroid_text)
                centroid_x_px = centroid_x_um / pixel_size_um
                centroid_y_px = centroid_y_um / pixel_size_um

                if segmentation_touches_edge(
                    centroid_x_px=centroid_x_px,
                    centroid_y_px=centroid_y_px,
                    major_axis_px=major_axis_px,
                    width_px=width_px,
                    height_px=height_px,
                ):
                    removed_edge_touch += 1
                    continue

                connected_parts = int(
                    get_first_value(
                        row, ["Connected_parts", "connected_parts"], required=False
                    )
                    or "1"
                )
                if connected_parts > 1:
                    removed_disconnected_parts += 1
                    continue

                writer.writerow(row)
                kept_rows += 1
    return (
        kept_rows,
        removed_axis_size,
        removed_edge_touch,
        removed_disconnected_parts,
        total_rows,
    )


def main() -> None:
    """Build the consolidated measurements CSV.

    Args:
        None: This function does not accept arguments.

    Returns:
        None: Writes the consolidated CSV to disk.
    """
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Input root not found: {INPUT_ROOT}")

    image_shapes = load_image_shapes(INPUT_ROOT)
    if not image_shapes:
        raise ValueError(f"No image metadata shape files found under: {INPUT_ROOT}")

    metrics_paths = sorted(INPUT_ROOT.rglob("*_segmented_metrics.csv"))
    if not metrics_paths:
        raise ValueError(f"No metrics files found under: {INPUT_ROOT}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    output_fields = [
        "Condition",
        "Muscle",
        "Block",
        "image",
        "Id",
        "Centroid",
        "Area",
        "Corrected_area",
        "Major_axis_length",
        "Minor_axis_length",
        "Minimum_Feret_Diameter",
        "Elongation",
        "Circularity",
        "Solidity",
        "NND",
        "Connected_parts",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=output_fields)
        writer.writeheader()

        for metrics_path in metrics_paths:
            parts = metrics_path.parts
            try:
                condition_raw = parts[parts.index("Processed") + 1]
                muscle_raw = parts[parts.index("Processed") + 2]
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Unexpected path layout: {metrics_path}") from exc

            condition = CONDITION_MAP.get(condition_raw)
            muscle = MUSCLE_MAP.get(muscle_raw)
            if condition is None or muscle is None:
                raise ValueError(f"Unknown condition/muscle in path: {metrics_path}")

            block_num = parse_block_number(metrics_path)
            image_label = parse_image_label(metrics_path)
            with open(metrics_path, "r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    centroid_text = get_first_value(row, ["Centroid", "centroid"])
                    cx, cy = parse_centroid(centroid_text)
                    cx_um = cx * PIXEL_SIZE_UM
                    cy_um = cy * PIXEL_SIZE_UM

                    area_text = get_first_value(row, ["Area", "area"])
                    area_um2 = float(area_text) * (PIXEL_SIZE_UM ** 2)
                    corrected_area_text = get_first_value(
                        row, ["Corrected_area", "corrected_area"], required=False
                    )

                    major_text = get_first_value(
                        row, ["Major_axis_length"], required=False
                    )
                    minor_text = get_first_value(
                        row, ["Minor_axis_length"], required=False
                    )
                    min_feret_text = get_first_value(
                        row,
                        [
                            "Minimum_Feret_Diameter",
                            "minimum_feret_diameter",
                            "Minimum Feret's Diameter",
                        ],
                        required=False,
                    )
                    nnd_text = get_first_value(
                        row, ["NND", "Nearest Neighbor Distance (NND)"], required=False
                    )

                    corrected_area_val = maybe_float(corrected_area_text)
                    major_val = maybe_float(major_text)
                    minor_val = maybe_float(minor_text)
                    min_feret_val = maybe_float(min_feret_text)
                    nnd_val = maybe_float(nnd_text)

                    corrected_area_um2 = (
                        corrected_area_val * (PIXEL_SIZE_UM ** 2)
                        if corrected_area_val is not None
                        else None
                    )
                    major_um = major_val * PIXEL_SIZE_UM if major_val is not None else None
                    minor_um = minor_val * PIXEL_SIZE_UM if minor_val is not None else None
                    min_feret_um = (
                        min_feret_val * PIXEL_SIZE_UM if min_feret_val is not None else None
                    )
                    nnd_um = nnd_val * PIXEL_SIZE_UM if nnd_val is not None else None

                    writer.writerow(
                        {
                            "Condition": condition,
                            "Muscle": muscle,
                            "Block": block_num,
                            "image": image_label,
                            "Id": get_first_value(row, ["Id", "id"]),
                            "Centroid": f"({cx_um:.6f}, {cy_um:.6f})",
                            "Area": f"{area_um2:.8f}",
                            "Corrected_area": ""
                            if corrected_area_um2 is None
                            else f"{corrected_area_um2:.8f}",
                            "Major_axis_length": ""
                            if major_um is None
                            else f"{major_um:.6f}",
                            "Minor_axis_length": ""
                            if minor_um is None
                            else f"{minor_um:.6f}",
                            "Minimum_Feret_Diameter": ""
                            if min_feret_um is None
                            else f"{min_feret_um:.6f}",
                            "Elongation": get_first_value(
                                row, ["Elongation", "Aspect Ratio (Elongation)"]
                            ),
                            "Circularity": get_first_value(
                                row, ["Circularity", "Circularity (Form Factor)"]
                            ),
                            "Solidity": get_first_value(
                                row, ["Solidity", "Solidity (Branching)"]
                            ),
                            "NND": "" if nnd_um is None else f"{nnd_um:.6f}",
                            "Connected_parts": get_first_value(
                                row, ["Connected_parts", "connected_parts"], required=False
                            )
                            or "1",
                        }
                    )

    (
        kept_rows,
        removed_axis_size,
        removed_edge_touch,
        removed_disconnected_parts,
        total_rows,
    ) = clean_measurements_csv(
        measurements_csv=OUTPUT_CSV,
        cleaned_csv=OUTPUT_CLEANED_CSV,
        image_shapes=image_shapes,
        pixel_size_um=PIXEL_SIZE_UM,
        min_major_axis_px=MIN_MAJOR_AXIS_PX,
        min_minor_axis_px=MIN_MINOR_AXIS_PX,
    )
    removed_total = removed_axis_size + removed_edge_touch + removed_disconnected_parts
    print(f"Wrote full measurements CSV: {OUTPUT_CSV}")
    print(f"Wrote cleaned measurements CSV: {OUTPUT_CLEANED_CSV}")
    print(
        "Cleaning summary: "
        f"total={total_rows}, kept={kept_rows}, removed={removed_total}, "
        f"axis_size={removed_axis_size}, edge_touch={removed_edge_touch}, "
        f"disconnected_parts={removed_disconnected_parts}"
    )


if __name__ == "__main__":
    main()
