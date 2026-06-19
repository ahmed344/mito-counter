#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dmd.build_measurements_csv import (  # noqa: E402
    CONDITION_MAP,
    IMF_LABEL,
    MIN_MAJOR_AXIS_PX,
    MIN_MINOR_AXIS_PX,
    MUSCLE_MAP,
    SS_LABEL,
    classify_image_region,
    coefficient_of_variation,
    format_optional_float,
    get_first_value,
    load_json_file,
    mean_or_none,
    maybe_float,
    numeric_values,
    pair_correlation_integral,
    parse_centroid,
    parse_image_label,
    resolve_pixel_size_nm,
    ripley_l_integral,
    segmentation_touches_image_edge,
    spatial_radii_nm,
    sum_or_none,
)

INPUT_ROOT = REPO_ROOT / "data" / "DMD_1X" / "Processed"
RESULTS_ROOT = REPO_ROOT / "data" / "DMD_1X" / "results"
OUTPUT_CSV = RESULTS_ROOT / "measurements.csv"
OUTPUT_CLEANED_CSV = RESULTS_ROOT / "measurements_cleaned.csv"
OUTPUT_SS_SUMMARY_CSV = RESULTS_ROOT / "measurements_cleaned_ss_summary.csv"
OUTPUT_IMF_SUMMARY_CSV = RESULTS_ROOT / "measurements_cleaned_imf_summary.csv"
OUTPUT_NO_COMPARTMENT_CLEANED_CSV = RESULTS_ROOT / "measurements_cleaned_no_compartment.csv"
OUTPUT_NO_COMPARTMENT_SUMMARY_CSV = RESULTS_ROOT / "measurements_cleaned_no_compartment_summary.csv"

GROUP_DIRECTORY_RE = re.compile(r"^(TA|EOM)_(WT|DMD)$", re.IGNORECASE)
COMPARTMENT_DIRECTORY_RE = re.compile(r"^(SS|IMF)_([0-9]+)$", re.IGNORECASE)
COMPARTMENT_MAP = {
    "SS": SS_LABEL,
    "IMF": IMF_LABEL,
}
ALL_COMPARTMENTS_LABEL = "All compartments"

ImageKey = tuple[str, str, str, str, str]


@dataclass(frozen=True)
class ImageRecord:
    condition: str
    muscle: str
    block_num: str
    compartment: str
    image_label: str
    width_px: int
    height_px: int
    image_path: Path
    metadata_path: Path
    metrics_path: Path
    pixel_size_nm: float
    pixel_size_source: str
    magnification: float | None


def parse_group_directory(group_directory: str) -> tuple[str, str]:
    """Parse a DMD_1X group directory into normalized condition and muscle labels.

    Args:
        group_directory (str): Directory name such as ``TA_WT`` or ``EOM_DMD``.

    Returns:
        tuple[str, str]: Normalized condition and muscle labels.
    """
    match = GROUP_DIRECTORY_RE.match(group_directory)
    if match is None:
        raise ValueError(
            "Unable to parse DMD_1X group directory. Expected '<muscle>_<condition>' "
            f"with values like 'TA_WT' or 'EOM_DMD': {group_directory}"
        )

    muscle_raw = match.group(1).upper()
    condition_raw = match.group(2).upper()
    muscle = MUSCLE_MAP.get(muscle_raw)
    condition = CONDITION_MAP.get(condition_raw)
    if muscle is None or condition is None:
        raise ValueError(f"Unknown DMD_1X group directory: {group_directory}")
    return condition, muscle


def parse_compartment_directory(compartment_directory: str) -> tuple[str, str]:
    """Parse a DMD_1X compartment directory into compartment label and block number.

    Args:
        compartment_directory (str): Directory name such as ``SS_3`` or ``IMF_2``.

    Returns:
        tuple[str, str]: Full compartment label and block number.
    """
    match = COMPARTMENT_DIRECTORY_RE.match(compartment_directory)
    if match is None:
        raise ValueError(
            "Unable to parse DMD_1X compartment directory. Expected 'SS_<block>' "
            f"or 'IMF_<block>': {compartment_directory}"
        )

    compartment_raw = match.group(1).upper()
    block_num = match.group(2)
    compartment = COMPARTMENT_MAP[compartment_raw]
    return compartment, block_num


def parse_record_path(path: Path) -> tuple[str, str, str, str]:
    """Parse DMD_1X condition, muscle, compartment, and block from a processed path.

    Args:
        path (Path): Path below the DMD_1X ``Processed`` directory.

    Returns:
        tuple[str, str, str, str]: Condition, muscle, block number, and compartment.
    """
    parts = path.parts
    try:
        processed_index = parts.index("Processed")
        group_directory = parts[processed_index + 1]
        compartment_directory = parts[processed_index + 2]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Unexpected DMD_1X processed path layout: {path}") from exc

    condition, muscle = parse_group_directory(group_directory)
    compartment, block_num = parse_compartment_directory(compartment_directory)
    return condition, muscle, block_num, compartment


def image_key(record: ImageRecord) -> ImageKey:
    """Build the stable lookup key for one DMD_1X image record.

    Args:
        record (ImageRecord): Image metadata record to key.

    Returns:
        ImageKey: Tuple containing condition, muscle, block, compartment, and image label.
    """
    return (
        record.condition,
        record.muscle,
        record.block_num,
        record.compartment,
        record.image_label,
    )


def load_image_records(input_root: Path) -> dict[ImageKey, ImageRecord]:
    """Load per-image metadata and calibration for the DMD_1X dataset.

    Args:
        input_root (Path): Root directory containing processed DMD_1X image outputs.

    Returns:
        dict[ImageKey, ImageRecord]: Mapping from image identity to metadata records.
    """
    records: dict[ImageKey, ImageRecord] = {}
    for metadata_path in sorted(input_root.rglob("*.json")):
        image_label = metadata_path.stem
        metrics_path = metadata_path.with_name(f"{image_label}_segmented_metrics.csv")
        if not metrics_path.is_file():
            continue

        condition, muscle, block_num, compartment = parse_record_path(metadata_path)
        metadata = load_json_file(metadata_path)
        shape = metadata.get("basic", {}).get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError(f"Missing or invalid shape in metadata: {metadata_path}")

        image_path = metadata_path.with_suffix(".tif")
        pixel_size_nm, pixel_size_source, magnification = resolve_pixel_size_nm(
            metadata=metadata,
            image_label=image_label,
            image_path=image_path,
        )

        record = ImageRecord(
            condition=condition,
            muscle=muscle,
            block_num=block_num,
            compartment=compartment,
            image_label=image_label,
            width_px=int(shape[1]),
            height_px=int(shape[0]),
            image_path=image_path,
            metadata_path=metadata_path,
            metrics_path=metrics_path,
            pixel_size_nm=pixel_size_nm,
            pixel_size_source=pixel_size_source,
            magnification=magnification,
        )
        key = image_key(record)
        if key in records:
            raise ValueError(f"Duplicate DMD_1X image record for key: {key}")
        records[key] = record
    return records


def key_from_output_row(row: dict[str, str]) -> ImageKey:
    """Build an image lookup key from a consolidated measurement row.

    Args:
        row (dict[str, str]): Row from ``measurements.csv`` or ``measurements_cleaned.csv``.

    Returns:
        ImageKey: Tuple containing condition, muscle, block, compartment, and image label.
    """
    condition = get_first_value(row, ["Condition"])
    muscle = get_first_value(row, ["Muscle"])
    block = get_first_value(row, ["Block"])
    compartment = get_first_value(row, ["Compartment"])
    image = get_first_value(row, ["image"])
    return condition, muscle, block, compartment, image


def build_output_row(row: dict[str, str], record: ImageRecord) -> dict[str, str]:
    """Convert one raw metrics row into the consolidated DMD_1X output schema.

    Args:
        row (dict[str, str]): Raw per-image metrics row from MitoNet output.
        record (ImageRecord): Metadata and calibration for the source image.

    Returns:
        dict[str, str]: CSV-ready output row with physical units.
    """
    centroid_text = get_first_value(row, ["Centroid", "centroid"])
    cx_px, cy_px = parse_centroid(centroid_text)
    cx_nm = cx_px * record.pixel_size_nm
    cy_nm = cy_px * record.pixel_size_nm

    area_text = get_first_value(row, ["Area", "area"])
    area_nm2 = float(area_text) * (record.pixel_size_nm**2)
    corrected_area_text = get_first_value(
        row, ["Corrected_area", "corrected_area"], required=False
    )

    major_text = get_first_value(row, ["Major_axis_length"], required=False)
    minor_text = get_first_value(row, ["Minor_axis_length"], required=False)
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
    third_nnd_text = get_first_value(row, ["3NND"], required=False)
    fifth_nnd_text = get_first_value(row, ["5NND"], required=False)
    voronoi_area_text = get_first_value(row, ["Voronoi_Cell_Area"], required=False)

    corrected_area_val = maybe_float(corrected_area_text)
    major_val = maybe_float(major_text)
    minor_val = maybe_float(minor_text)
    min_feret_val = maybe_float(min_feret_text)
    nnd_val = maybe_float(nnd_text)
    third_nnd_val = maybe_float(third_nnd_text)
    fifth_nnd_val = maybe_float(fifth_nnd_text)
    voronoi_area_val = maybe_float(voronoi_area_text)

    corrected_area_nm2 = (
        corrected_area_val * (record.pixel_size_nm**2)
        if corrected_area_val is not None
        else None
    )
    major_nm = major_val * record.pixel_size_nm if major_val is not None else None
    minor_nm = minor_val * record.pixel_size_nm if minor_val is not None else None
    min_feret_nm = (
        min_feret_val * record.pixel_size_nm if min_feret_val is not None else None
    )
    nnd_nm = nnd_val * record.pixel_size_nm if nnd_val is not None else None
    third_nnd_nm = (
        third_nnd_val * record.pixel_size_nm if third_nnd_val is not None else None
    )
    fifth_nnd_nm = (
        fifth_nnd_val * record.pixel_size_nm if fifth_nnd_val is not None else None
    )
    voronoi_area_nm2 = (
        voronoi_area_val * (record.pixel_size_nm**2)
        if voronoi_area_val is not None
        else None
    )

    return {
        "Condition": record.condition,
        "Muscle": record.muscle,
        "Block": record.block_num,
        "image": record.image_label,
        "Compartment": record.compartment,
        "Zoom": "" if record.magnification is None else f"{record.magnification:.6f}",
        "Pixel_size_nm": f"{record.pixel_size_nm:.8f}",
        "Id": get_first_value(row, ["Id", "id"]),
        "Centroid": f"({cx_nm:.6f}, {cy_nm:.6f})",
        "Image_Region": classify_image_region(
            centroid_x_px=cx_px,
            centroid_y_px=cy_px,
            width_px=record.width_px,
            height_px=record.height_px,
        ),
        "Area": f"{area_nm2:.8f}",
        "Corrected_area": ""
        if corrected_area_nm2 is None
        else f"{corrected_area_nm2:.8f}",
        "Major_axis_length": ""
        if major_nm is None
        else f"{major_nm:.6f}",
        "Minor_axis_length": ""
        if minor_nm is None
        else f"{minor_nm:.6f}",
        "Minimum_Feret_Diameter": ""
        if min_feret_nm is None
        else f"{min_feret_nm:.6f}",
        "Elongation": get_first_value(row, ["Elongation", "Aspect Ratio (Elongation)"]),
        "Circularity": get_first_value(row, ["Circularity", "Circularity (Form Factor)"]),
        "Solidity": get_first_value(row, ["Solidity", "Solidity (Branching)"]),
        "NND": "" if nnd_nm is None else f"{nnd_nm:.6f}",
        "3NND": "" if third_nnd_nm is None else f"{third_nnd_nm:.6f}",
        "5NND": "" if fifth_nnd_nm is None else f"{fifth_nnd_nm:.6f}",
        "Voronoi_Cell_Area": ""
        if voronoi_area_nm2 is None
        else f"{voronoi_area_nm2:.8f}",
        "Connected_parts": get_first_value(
            row, ["Connected_parts", "connected_parts"], required=False
        )
        or "1",
    }


def clean_measurements_csv(
    measurements_csv: Path,
    cleaned_csv: Path,
    image_records: dict[ImageKey, ImageRecord],
) -> tuple[int, int, int, int, int]:
    """Create a cleaned DMD_1X measurements CSV using image-level filters.

    Args:
        measurements_csv (Path): Source consolidated measurements CSV path.
        cleaned_csv (Path): Output cleaned CSV path.
        image_records (dict[ImageKey, ImageRecord]): Per-image metadata records.

    Returns:
        tuple[int, int, int, int, int]: Kept rows, axis-size removals, edge-touch
        removals, disconnected-part removals, and total rows.
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
                record = image_records.get(key_from_output_row(row))
                if record is None:
                    raise KeyError(f"Missing image metadata for row: {row}")

                major_axis_nm = float(get_first_value(row, ["Major_axis_length"]))
                minor_axis_nm = float(get_first_value(row, ["Minor_axis_length"]))
                major_axis_px = major_axis_nm / record.pixel_size_nm
                minor_axis_px = minor_axis_nm / record.pixel_size_nm
                if (
                    major_axis_px <= MIN_MAJOR_AXIS_PX
                    or minor_axis_px <= MIN_MINOR_AXIS_PX
                ):
                    removed_axis_size += 1
                    continue

                centroid_text = get_first_value(row, ["Centroid", "centroid"])
                centroid_x_nm, centroid_y_nm = parse_centroid(centroid_text)
                centroid_x_px = centroid_x_nm / record.pixel_size_nm
                centroid_y_px = centroid_y_nm / record.pixel_size_nm
                if segmentation_touches_image_edge(
                    centroid_x_px=centroid_x_px,
                    centroid_y_px=centroid_y_px,
                    major_axis_px=major_axis_px,
                    width_px=record.width_px,
                    height_px=record.height_px,
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


def summarize_image_rows(
    rows: list[dict[str, str]],
    record: ImageRecord,
    compartment_label: str | None = None,
) -> dict[str, str]:
    """Summarize cleaned DMD_1X instance rows for one image.

    Args:
        rows (list[dict[str, str]]): Cleaned measurement rows belonging to one image.
        record (ImageRecord): Metadata and calibration for the source image.
        compartment_label (str | None): Optional compartment label to write instead
            of the source image compartment.

    Returns:
        dict[str, str]: CSV-ready image summary values.
    """
    width_nm = float(record.width_px) * record.pixel_size_nm
    height_nm = float(record.height_px) * record.pixel_size_nm
    image_area_nm2 = width_nm * height_nm
    center_rows = [
        row for row in rows if row.get("Image_Region", "").strip().lower() == "center"
    ]
    coords_nm = [
        parse_centroid(get_first_value(row, ["Centroid", "centroid"])) for row in rows
    ]
    radii_nm = spatial_radii_nm(width_nm, height_nm)
    instance_count = len(rows)
    density = float(instance_count)

    return {
        "Compartment": compartment_label or record.compartment,
        "Density": format_optional_float(density, digits=12),
        "Instance_count": str(instance_count),
        "Zoom": format_optional_float(record.magnification, digits=6),
        "Image_width_px": str(record.width_px),
        "Image_height_px": str(record.height_px),
        "Pixel_size_nm": format_optional_float(record.pixel_size_nm, digits=8),
        "Image_area_nm2": format_optional_float(image_area_nm2, digits=8),
        "Area_sum": format_optional_float(sum_or_none(numeric_values(rows, "Area"))),
        "Corrected_area_sum": format_optional_float(
            sum_or_none(numeric_values(rows, "Corrected_area"))
        ),
        "Minimum_Feret_Diameter_sum": format_optional_float(
            sum_or_none(numeric_values(rows, "Minimum_Feret_Diameter")), digits=6
        ),
        "Minor_axis_length_sum": format_optional_float(
            sum_or_none(numeric_values(rows, "Minor_axis_length")), digits=6
        ),
        "Area_mean": format_optional_float(mean_or_none(numeric_values(rows, "Area"))),
        "Corrected_area_mean": format_optional_float(
            mean_or_none(numeric_values(rows, "Corrected_area"))
        ),
        "Minimum_Feret_Diameter_mean": format_optional_float(
            mean_or_none(numeric_values(rows, "Minimum_Feret_Diameter")), digits=6
        ),
        "Major_axis_length_mean": format_optional_float(
            mean_or_none(numeric_values(rows, "Major_axis_length")), digits=6
        ),
        "Minor_axis_length_mean": format_optional_float(
            mean_or_none(numeric_values(rows, "Minor_axis_length")), digits=6
        ),
        "Elongation_mean": format_optional_float(
            mean_or_none(numeric_values(rows, "Elongation")), digits=6
        ),
        "Circularity_mean": format_optional_float(
            mean_or_none(numeric_values(rows, "Circularity")), digits=6
        ),
        "Solidity_mean": format_optional_float(
            mean_or_none(numeric_values(rows, "Solidity")), digits=6
        ),
        "NND_center_mean": format_optional_float(
            mean_or_none(numeric_values(center_rows, "NND")), digits=6
        ),
        "3NND_center_mean": format_optional_float(
            mean_or_none(numeric_values(center_rows, "3NND")), digits=6
        ),
        "5NND_center_mean": format_optional_float(
            mean_or_none(numeric_values(center_rows, "5NND")), digits=6
        ),
        "Voronoi_Cell_Area_center_mean": format_optional_float(
            mean_or_none(numeric_values(center_rows, "Voronoi_Cell_Area"))
        ),
        "Voronoi_Cell_Area_center_cv": format_optional_float(
            coefficient_of_variation(numeric_values(center_rows, "Voronoi_Cell_Area")),
            digits=8,
        ),
        "Ripley_L_integral": format_optional_float(
            ripley_l_integral(coords_nm, image_area_nm2, radii_nm)
        ),
        "Pair_Correlation_integral": format_optional_float(
            pair_correlation_integral(coords_nm, image_area_nm2, radii_nm)
        ),
    }


def write_compartment_summary_csv(
    cleaned_csv: Path,
    summary_csv: Path,
    image_records: dict[ImageKey, ImageRecord],
    compartment: str,
) -> int:
    """Write one cleaned DMD_1X summary row per image for one compartment.

    Args:
        cleaned_csv (Path): Cleaned instance-level measurements CSV path.
        summary_csv (Path): Destination summary CSV path.
        image_records (dict[ImageKey, ImageRecord]): Per-image metadata records.
        compartment (str): Full compartment label to summarize.

    Returns:
        int: Number of image summary rows written.
    """
    grouped_rows: dict[ImageKey, list[dict[str, str]]] = {}
    with open(cleaned_csv, "r", newline="", encoding="utf-8") as in_handle:
        reader = csv.DictReader(in_handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in: {cleaned_csv}")
        for row in reader:
            key = key_from_output_row(row)
            if key[3] == compartment:
                grouped_rows.setdefault(key, []).append(row)

    fieldnames = [
        "Condition",
        "Muscle",
        "Block",
        "image",
        "Compartment",
        "Density",
        "Instance_count",
        "Zoom",
        "Image_width_px",
        "Image_height_px",
        "Pixel_size_nm",
        "Image_area_nm2",
        "Area_sum",
        "Corrected_area_sum",
        "Minimum_Feret_Diameter_sum",
        "Minor_axis_length_sum",
        "Area_mean",
        "Corrected_area_mean",
        "Minimum_Feret_Diameter_mean",
        "Major_axis_length_mean",
        "Minor_axis_length_mean",
        "Elongation_mean",
        "Circularity_mean",
        "Solidity_mean",
        "NND_center_mean",
        "3NND_center_mean",
        "5NND_center_mean",
        "Voronoi_Cell_Area_center_mean",
        "Voronoi_Cell_Area_center_cv",
        "Ripley_L_integral",
        "Pair_Correlation_integral",
    ]

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with open(summary_csv, "w", newline="", encoding="utf-8") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, record in sorted(image_records.items()):
            if record.compartment != compartment:
                continue
            rows = grouped_rows.get(key, [])
            summary = summarize_image_rows(rows=rows, record=record)
            writer.writerow(
                {
                    "Condition": record.condition,
                    "Muscle": record.muscle,
                    "Block": record.block_num,
                    "image": record.image_label,
                    **summary,
                }
            )
            rows_written += 1
    return rows_written


def write_no_compartment_measurements_csv(
    cleaned_csv: Path,
    no_compartment_csv: Path,
    compartment_label: str = ALL_COMPARTMENTS_LABEL,
) -> int:
    """Write cleaned instance rows with a pooled compartment label.

    Args:
        cleaned_csv (Path): Source cleaned instance-level measurements CSV path.
        no_compartment_csv (Path): Destination pooled measurements CSV path.
        compartment_label (str): Label to use for all rows in the ``Compartment`` column.

    Returns:
        int: Number of cleaned instance rows written.
    """
    no_compartment_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with open(cleaned_csv, "r", newline="", encoding="utf-8") as in_handle:
        reader = csv.DictReader(in_handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in: {cleaned_csv}")
        fieldnames = list(reader.fieldnames)
        if "Compartment" not in fieldnames:
            raise KeyError(f"Missing Compartment column in: {cleaned_csv}")
        with open(no_compartment_csv, "w", newline="", encoding="utf-8") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                row["Compartment"] = compartment_label
                writer.writerow(row)
                rows_written += 1
    return rows_written


def write_no_compartment_summary_csv(
    cleaned_csv: Path,
    summary_csv: Path,
    image_records: dict[ImageKey, ImageRecord],
    compartment_label: str = ALL_COMPARTMENTS_LABEL,
) -> int:
    """Write one pooled-compartment summary row per DMD_1X source image.

    Args:
        cleaned_csv (Path): Cleaned instance-level measurements CSV path.
        summary_csv (Path): Destination pooled image-summary CSV path.
        image_records (dict[ImageKey, ImageRecord]): Per-image metadata records.
        compartment_label (str): Label to use for all rows in the ``Compartment`` column.

    Returns:
        int: Number of image summary rows written.
    """
    grouped_rows: dict[ImageKey, list[dict[str, str]]] = {}
    with open(cleaned_csv, "r", newline="", encoding="utf-8") as in_handle:
        reader = csv.DictReader(in_handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in: {cleaned_csv}")
        for row in reader:
            grouped_rows.setdefault(key_from_output_row(row), []).append(row)

    fieldnames = [
        "Condition",
        "Muscle",
        "Block",
        "image",
        "Compartment",
        "Density",
        "Instance_count",
        "Zoom",
        "Image_width_px",
        "Image_height_px",
        "Pixel_size_nm",
        "Image_area_nm2",
        "Area_sum",
        "Corrected_area_sum",
        "Minimum_Feret_Diameter_sum",
        "Minor_axis_length_sum",
        "Area_mean",
        "Corrected_area_mean",
        "Minimum_Feret_Diameter_mean",
        "Major_axis_length_mean",
        "Minor_axis_length_mean",
        "Elongation_mean",
        "Circularity_mean",
        "Solidity_mean",
        "NND_center_mean",
        "3NND_center_mean",
        "5NND_center_mean",
        "Voronoi_Cell_Area_center_mean",
        "Voronoi_Cell_Area_center_cv",
        "Ripley_L_integral",
        "Pair_Correlation_integral",
    ]

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with open(summary_csv, "w", newline="", encoding="utf-8") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, record in sorted(image_records.items()):
            rows = grouped_rows.get(key, [])
            summary = summarize_image_rows(
                rows=rows,
                record=record,
                compartment_label=compartment_label,
            )
            writer.writerow(
                {
                    "Condition": record.condition,
                    "Muscle": record.muscle,
                    "Block": record.block_num,
                    "image": record.image_label,
                    **summary,
                }
            )
            rows_written += 1
    return rows_written


def write_measurements_csv(
    output_csv: Path,
    image_records: dict[ImageKey, ImageRecord],
) -> tuple[int, Counter[str]]:
    """Write the full DMD_1X measurements CSV from per-image metrics files.

    Args:
        output_csv (Path): Destination consolidated CSV path.
        image_records (dict[ImageKey, ImageRecord]): Per-image metadata records.

    Returns:
        tuple[int, Counter[str]]: Number of measurement rows written and pixel-size
        source counts by source label.
    """
    output_fields = [
        "Condition",
        "Muscle",
        "Block",
        "image",
        "Compartment",
        "Zoom",
        "Pixel_size_nm",
        "Id",
        "Centroid",
        "Image_Region",
        "Area",
        "Corrected_area",
        "Major_axis_length",
        "Minor_axis_length",
        "Minimum_Feret_Diameter",
        "Elongation",
        "Circularity",
        "Solidity",
        "NND",
        "3NND",
        "5NND",
        "Voronoi_Cell_Area",
        "Connected_parts",
    ]

    pixel_size_source_counts: Counter[str] = Counter()
    rows_written = 0
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=output_fields)
        writer.writeheader()
        for _, record in sorted(image_records.items()):
            if record.width_px <= 0 or record.height_px <= 0:
                raise ValueError(f"Invalid image shape for: {record.metadata_path}")
            pixel_size_source_counts[record.pixel_size_source] += 1

            with open(record.metrics_path, "r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise ValueError(f"Missing CSV header in: {record.metrics_path}")
                for row in reader:
                    writer.writerow(build_output_row(row=row, record=record))
                    rows_written += 1

    return rows_written, pixel_size_source_counts


def main() -> None:
    """Build consolidated, cleaned, and summary DMD_1X measurement CSV files.

    Args:
        None: This function does not accept arguments.

    Returns:
        None: Writes CSV files to ``data/DMD_1X/results`` and prints a summary.
    """
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Input root not found: {INPUT_ROOT}")

    image_records = load_image_records(INPUT_ROOT)
    if not image_records:
        raise ValueError(f"No DMD_1X image records found under: {INPUT_ROOT}")

    measurement_rows, pixel_size_source_counts = write_measurements_csv(
        output_csv=OUTPUT_CSV,
        image_records=image_records,
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
        image_records=image_records,
    )
    ss_summary_rows = write_compartment_summary_csv(
        cleaned_csv=OUTPUT_CLEANED_CSV,
        summary_csv=OUTPUT_SS_SUMMARY_CSV,
        image_records=image_records,
        compartment=SS_LABEL,
    )
    imf_summary_rows = write_compartment_summary_csv(
        cleaned_csv=OUTPUT_CLEANED_CSV,
        summary_csv=OUTPUT_IMF_SUMMARY_CSV,
        image_records=image_records,
        compartment=IMF_LABEL,
    )
    no_compartment_rows = write_no_compartment_measurements_csv(
        cleaned_csv=OUTPUT_CLEANED_CSV,
        no_compartment_csv=OUTPUT_NO_COMPARTMENT_CLEANED_CSV,
    )
    no_compartment_summary_rows = write_no_compartment_summary_csv(
        cleaned_csv=OUTPUT_CLEANED_CSV,
        summary_csv=OUTPUT_NO_COMPARTMENT_SUMMARY_CSV,
        image_records=image_records,
    )

    removed_total = removed_axis_size + removed_edge_touch + removed_disconnected_parts
    print(f"Loaded DMD_1X image records: {len(image_records)}")
    print(f"Wrote full measurements CSV: {OUTPUT_CSV}")
    print(f"Wrote full measurement rows: {measurement_rows}")
    print(f"Wrote cleaned measurements CSV: {OUTPUT_CLEANED_CSV}")
    print(
        "Cleaning summary: "
        f"total={total_rows}, kept={kept_rows}, removed={removed_total}, "
        f"axis_size={removed_axis_size}, image_edge={removed_edge_touch}, "
        f"disconnected_parts={removed_disconnected_parts}"
    )
    print(f"Wrote SS summary CSV: {OUTPUT_SS_SUMMARY_CSV}")
    print(f"Wrote IMF summary CSV: {OUTPUT_IMF_SUMMARY_CSV}")
    print(f"Wrote no-compartment cleaned CSV: {OUTPUT_NO_COMPARTMENT_CLEANED_CSV}")
    print(f"Wrote no-compartment summary CSV: {OUTPUT_NO_COMPARTMENT_SUMMARY_CSV}")
    print(f"SS summary rows: {ss_summary_rows}")
    print(f"IMF summary rows: {imf_summary_rows}")
    print(f"No-compartment cleaned rows: {no_compartment_rows}")
    print(f"No-compartment summary rows: {no_compartment_summary_rows}")
    print(
        "Pixel-size sources: "
        + ", ".join(
            f"{source}={count}" for source, count in sorted(pixel_size_source_counts.items())
        )
    )


if __name__ == "__main__":
    main()
