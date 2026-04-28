#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tifffile as tiff

REFERENCE_MAGNIFICATION = 6800.0
REFERENCE_PIXEL_SIZE_UM = 0.0015396
INPUT_ROOT = Path("/workspaces/mito-counter/data/Calpaine_3/Processed")
OUTPUT_CSV = Path("/workspaces/mito-counter/data/Calpaine_3/results/measurments.csv")
OUTPUT_CLEANED_CSV = Path(
    "/workspaces/mito-counter/data/Calpaine_3/results/measurments_cleaned.csv"
)
MIN_MAJOR_AXIS_PX = 30.0
MIN_MINOR_AXIS_PX = 5.0
PIXEL_SIZE_KEYWORDS = {
    "pixel_size",
    "pixel_size_um",
    "pixel_size_nm",
    "pixelsize",
    "sampling",
    "resolution",
    "scale",
    "x_scale",
    "y_scale",
}
UNIT_ALIASES_TO_NM = {
    "um": 1000.0,
    "micrometer": 1000.0,
    "micrometers": 1000.0,
    "micrometre": 1000.0,
    "micrometres": 1000.0,
    "micron": 1000.0,
    "microns": 1000.0,
    "nm": 1.0,
    "nanometer": 1.0,
    "nanometers": 1.0,
    "nanometre": 1.0,
    "nanometres": 1.0,
    "mm": 1_000_000.0,
    "millimeter": 1_000_000.0,
    "millimeters": 1_000_000.0,
    "millimetre": 1_000_000.0,
    "millimetres": 1_000_000.0,
    "pm": 1e-3,
    "picometer": 1e-3,
    "picometers": 1e-3,
    "picometre": 1e-3,
    "picometres": 1e-3,
    "a": 0.1,
    "angstrom": 0.1,
    "angstroms": 0.1,
}

CONDITION_MAP = {
    "WT": "Wildtype",
    "KO-C3": "Calpain_3_Knockout",
}

MUSCLE_MAP = {
    "SOL": "Soleus",
    "TA": "Tibialis Anterior",
}

BLOCK_RE = re.compile(r"(?:^|[^A-Z0-9])(SOL|TA)[\s_-]*([0-9]+)", re.IGNORECASE)
MAGNIFICATION_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*[xX]")


@dataclass(frozen=True)
class ImageRecord:
    """Resolved metadata needed to scale one image's measurements.

    Args:
        condition (str): Normalized condition label.
        muscle (str): Normalized muscle label.
        image_label (str): Image stem used in measurement CSVs.
        width_px (int): Image width in pixels.
        height_px (int): Image height in pixels.
        pixel_size_nm (float): Resolved physical pixel size in nanometers.
        pixel_size_source (str): Metadata source used to resolve `pixel_size_nm`.
        magnification (float | None): Magnification discovered in metadata or filename.

    Returns:
        None: Dataclass containers do not return values when initialized.
    """

    condition: str
    muscle: str
    image_label: str
    width_px: int
    height_px: int
    pixel_size_nm: float
    pixel_size_source: str
    magnification: float | None


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


def get_first_value(row: dict[str, str], keys: list[str], required: bool = True) -> str | None:
    """Return the first non-empty value for candidate keys.

    Args:
        row (dict[str, str]): Input record from the CSV reader.
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


def load_json_file(path: Path) -> dict[str, Any]:
    """Load JSON content from disk.

    Args:
        path (Path): JSON file path to read.

    Returns:
        dict[str, Any]: Parsed JSON object.
    """

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in: {path}")
    return data


def normalize_unit_name(unit: str | None) -> str | None:
    """Normalize a unit string to a canonical lookup key.

    Args:
        unit (str | None): Raw unit text from metadata.

    Returns:
        str | None: Lower-cased and symbol-normalized unit text, or ``None`` when missing.
    """

    if unit is None:
        return None
    normalized = unit.strip().lower()
    normalized = normalized.replace("µ", "u")
    normalized = normalized.replace("μ", "u")
    return normalized


def convert_to_nm(value: float, unit: str | None) -> float | None:
    """Convert a scalar value to nanometers when its unit is known.

    Args:
        value (float): Numeric value from metadata.
        unit (str | None): Unit associated with the value.

    Returns:
        float | None: Converted nanometer value, or ``None`` when the unit is unknown.
    """

    normalized = normalize_unit_name(unit)
    if normalized is None:
        return None
    factor = UNIT_ALIASES_TO_NM.get(normalized)
    if factor is None:
        return None
    return value * factor


def walk_nested_values(data: Any) -> list[Any]:
    """Flatten nested mappings and sequences into a list of values.

    Args:
        data (Any): Nested metadata structure composed of dicts, lists, and scalars.

    Returns:
        list[Any]: Depth-first list containing the current object and all nested values.
    """

    values = [data]
    if isinstance(data, dict):
        for child in data.values():
            values.extend(walk_nested_values(child))
    elif isinstance(data, list):
        for child in data:
            values.extend(walk_nested_values(child))
    return values


def extract_magnification_from_metadata(metadata: dict[str, Any]) -> float | None:
    """Extract microscope magnification from metadata when available.

    Args:
        metadata (dict[str, Any]): Parsed image sidecar metadata.

    Returns:
        float | None: Magnification value, or ``None`` when unavailable.
    """

    candidate_paths = [
        ("hyperspy_metadata", "Acquisition_instrument", "TEM", "magnification"),
        ("Acquisition_instrument", "TEM", "magnification"),
        ("basic", "magnification"),
    ]
    for path in candidate_paths:
        current: Any = metadata
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if isinstance(current, (int, float)) and float(current) > 0.0:
            return float(current)
    return None


def extract_magnification_from_filename(image_label: str) -> float | None:
    """Extract magnification from an image label.

    Args:
        image_label (str): Image stem that may contain a token such as ``6800X``.

    Returns:
        float | None: Parsed magnification value, or ``None`` when unavailable.
    """

    match = MAGNIFICATION_RE.search(image_label)
    if not match:
        return None
    return float(match.group(1))


def find_pixel_size_from_dict(node: dict[str, Any]) -> float | None:
    """Extract a nanometer-per-pixel value from a metadata mapping.

    Args:
        node (dict[str, Any]): Metadata mapping to inspect.

    Returns:
        float | None: Nanometer-per-pixel value when the mapping stores one, otherwise ``None``.
    """

    normalized_keys = {str(key).strip().lower().replace(" ", "_"): key for key in node}
    for normalized_key in PIXEL_SIZE_KEYWORDS:
        raw_key = normalized_keys.get(normalized_key)
        if raw_key is None:
            continue
        value = node[raw_key]
        if isinstance(value, (int, float)):
            value_float = float(value)
            if value_float <= 0.0:
                continue
            if normalized_key.endswith("_um"):
                return value_float * 1000.0
            if normalized_key.endswith("_nm"):
                return value_float
            if normalized_key in {"pixelsize", "scale"} and value_float < 1.0:
                return value_float * 1000.0
            unit_candidates = [
                node.get("units"),
                node.get("unit"),
                node.get("Units"),
                node.get("Unit"),
            ]
            for unit in unit_candidates:
                converted = convert_to_nm(value_float, unit if isinstance(unit, str) else None)
                if converted is not None and converted > 0.0:
                    return converted
    x_scale_key = normalized_keys.get("x_scale")
    y_scale_key = normalized_keys.get("y_scale")
    if x_scale_key is not None and y_scale_key is not None:
        x_scale = node[x_scale_key]
        y_scale = node[y_scale_key]
        if isinstance(x_scale, (int, float)) and isinstance(y_scale, (int, float)):
            if float(x_scale) > 0.0 and float(y_scale) > 0.0:
                unit_value = node.get("units") or node.get("unit")
                converted = convert_to_nm(float(x_scale), unit_value if isinstance(unit_value, str) else None)
                if converted is not None:
                    return converted
    return None


def extract_pixel_size_from_json_metadata(metadata: dict[str, Any]) -> float | None:
    """Try to resolve pixel size from nested JSON metadata.

    Args:
        metadata (dict[str, Any]): Parsed image sidecar metadata.

    Returns:
        float | None: Nanometer-per-pixel value when metadata encodes it, otherwise ``None``.
    """

    for value in walk_nested_values(metadata):
        if not isinstance(value, dict):
            continue
        pixel_size_nm = find_pixel_size_from_dict(value)
        if pixel_size_nm is not None and pixel_size_nm > 0.0:
            return pixel_size_nm
    return None


def extract_pixel_size_from_tiff_metadata(image_path: Path) -> float | None:
    """Try to resolve pixel size from TIFF metadata and tags.

    Args:
        image_path (Path): TIFF image path to inspect.

    Returns:
        float | None: Nanometer-per-pixel value when TIFF metadata encodes it, otherwise ``None``.
    """

    if not image_path.is_file():
        return None
    with tiff.TiffFile(str(image_path)) as tif:
        page = tif.pages[0]
        description = page.description or ""
        if description:
            try:
                parsed = json.loads(description)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                pixel_size_nm = extract_pixel_size_from_json_metadata(parsed)
                if pixel_size_nm is not None:
                    return pixel_size_nm

        imagej_metadata = tif.imagej_metadata
        if isinstance(imagej_metadata, dict):
            pixel_size_nm = extract_pixel_size_from_json_metadata(imagej_metadata)
            if pixel_size_nm is not None:
                return pixel_size_nm

        x_resolution_tag = page.tags.get("XResolution")
        resolution_unit_tag = page.tags.get("ResolutionUnit")
        if x_resolution_tag is not None and resolution_unit_tag is not None:
            numerator, denominator = x_resolution_tag.value
            if numerator and denominator:
                pixels_per_unit = float(numerator) / float(denominator)
                resolution_unit = int(resolution_unit_tag.value)
                if pixels_per_unit > 0.0:
                    if resolution_unit == 2:
                        return 25_400_000.0 / pixels_per_unit
                    if resolution_unit == 3:
                        return 10_000_000.0 / pixels_per_unit
    return None


def magnification_to_pixel_size_nm(magnification: float) -> float:
    """Estimate pixel size in nanometers from magnification using the existing 6800X reference.

    Args:
        magnification (float): Microscope magnification for the image.

    Returns:
        float: Estimated nanometers per pixel at the provided magnification.
    """

    if magnification <= 0.0:
        raise ValueError(f"Magnification must be positive, got {magnification}.")
    return REFERENCE_PIXEL_SIZE_UM * 1000.0 * (REFERENCE_MAGNIFICATION / magnification)


def resolve_pixel_size_nm(
    metadata: dict[str, Any],
    image_label: str,
    image_path: Path,
) -> tuple[float, str, float | None]:
    """Resolve a per-image pixel size using metadata before falling back to magnification.

    Args:
        metadata (dict[str, Any]): Parsed image sidecar metadata.
        image_label (str): Image filename stem.
        image_path (Path): TIFF image path used for TIFF-metadata fallback.

    Returns:
        tuple[float, str, float | None]: Resolved nanometers-per-pixel value, a source label,
        and the magnification used or discovered during resolution.
    """

    magnification = extract_magnification_from_metadata(metadata)
    pixel_size_nm = extract_pixel_size_from_json_metadata(metadata)
    if pixel_size_nm is not None:
        return pixel_size_nm, "json_metadata", magnification

    pixel_size_nm = extract_pixel_size_from_tiff_metadata(image_path)
    if pixel_size_nm is not None:
        return pixel_size_nm, "tiff_metadata", magnification

    if magnification is None:
        magnification = extract_magnification_from_filename(image_label)
    if magnification is None:
        raise ValueError(
            f"Unable to resolve pixel size or magnification for image '{image_label}'."
        )
    return magnification_to_pixel_size_nm(magnification), "magnification_fallback", magnification


def load_image_records(input_root: Path) -> dict[tuple[str, str, str], ImageRecord]:
    """Load per-image metadata, shapes, and nanometer calibration for the Calpaine_3 dataset.

    Args:
        input_root (Path): Root directory that contains processed condition and muscle folders.

    Returns:
        dict[tuple[str, str, str], ImageRecord]: Mapping from
        ``(Condition, Muscle, image_stem)`` to resolved image metadata.
    """

    records: dict[tuple[str, str, str], ImageRecord] = {}
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

        metadata = load_json_file(metadata_path)
        shape = metadata.get("basic", {}).get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError(f"Missing or invalid shape in metadata: {metadata_path}")

        image_label = metadata_path.stem
        pixel_size_nm, pixel_size_source, magnification = resolve_pixel_size_nm(
            metadata=metadata,
            image_label=image_label,
            image_path=metadata_path.with_suffix(".tif"),
        )
        records[(condition, muscle, image_label)] = ImageRecord(
            condition=condition,
            muscle=muscle,
            image_label=image_label,
            width_px=int(shape[1]),
            height_px=int(shape[0]),
            pixel_size_nm=pixel_size_nm,
            pixel_size_source=pixel_size_source,
            magnification=magnification,
        )
    return records


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
    image_records: dict[tuple[str, str, str], ImageRecord],
    min_major_axis_px: float,
    min_minor_axis_px: float,
) -> tuple[int, int, int, int, int]:
    """Create a cleaned measurements CSV using axis-size, edge, and connectivity filters.

    Args:
        measurements_csv (Path): Source measurements CSV path.
        cleaned_csv (Path): Output cleaned CSV path.
        image_records (dict[tuple[str, str, str], ImageRecord]): Per-image metadata
            keyed by ``(Condition, Muscle, image)``.
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
                condition = get_first_value(row, ["Condition"])
                muscle = get_first_value(row, ["Muscle"])
                image = get_first_value(row, ["image"])
                shape_key = (condition, muscle, image)
                image_record = image_records.get(shape_key)
                if image_record is None:
                    raise KeyError(f"Missing shape metadata for: {shape_key}")

                major_axis_nm = float(get_first_value(row, ["Major_axis_length"]))
                minor_axis_nm = float(get_first_value(row, ["Minor_axis_length"]))
                major_axis_px = major_axis_nm / image_record.pixel_size_nm
                minor_axis_px = minor_axis_nm / image_record.pixel_size_nm
                if major_axis_px <= min_major_axis_px or minor_axis_px <= min_minor_axis_px:
                    removed_axis_size += 1
                    continue

                centroid_text = get_first_value(row, ["Centroid", "centroid"])
                centroid_x_nm, centroid_y_nm = parse_centroid(centroid_text)
                centroid_x_px = centroid_x_nm / image_record.pixel_size_nm
                centroid_y_px = centroid_y_nm / image_record.pixel_size_nm

                if segmentation_touches_edge(
                    centroid_x_px=centroid_x_px,
                    centroid_y_px=centroid_y_px,
                    major_axis_px=major_axis_px,
                    width_px=image_record.width_px,
                    height_px=image_record.height_px,
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
    """Build the consolidated measurements CSV in nanometers.

    Args:
        None: This function does not accept arguments.

    Returns:
        None: Writes the consolidated and cleaned CSVs to disk.
    """

    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Input root not found: {INPUT_ROOT}")

    image_records = load_image_records(INPUT_ROOT)
    if not image_records:
        raise ValueError(f"No image metadata records found under: {INPUT_ROOT}")

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
            image_record = image_records.get((condition, muscle, image_label))
            if image_record is None:
                raise KeyError(
                    f"Missing metadata record for metrics file '{metrics_path.name}' "
                    f"with key {(condition, muscle, image_label)}."
                )

            with open(metrics_path, "r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    centroid_text = get_first_value(row, ["Centroid", "centroid"])
                    centroid_x_px, centroid_y_px = parse_centroid(centroid_text)
                    centroid_x_nm = centroid_x_px * image_record.pixel_size_nm
                    centroid_y_nm = centroid_y_px * image_record.pixel_size_nm

                    area_text = get_first_value(row, ["Area", "area"])
                    area_nm2 = float(area_text) * (image_record.pixel_size_nm**2)
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

                    corrected_area_nm2 = (
                        corrected_area_val * (image_record.pixel_size_nm**2)
                        if corrected_area_val is not None
                        else None
                    )
                    major_nm = (
                        major_val * image_record.pixel_size_nm if major_val is not None else None
                    )
                    minor_nm = (
                        minor_val * image_record.pixel_size_nm if minor_val is not None else None
                    )
                    min_feret_nm = (
                        min_feret_val * image_record.pixel_size_nm
                        if min_feret_val is not None
                        else None
                    )
                    nnd_nm = (
                        nnd_val * image_record.pixel_size_nm if nnd_val is not None else None
                    )

                    writer.writerow(
                        {
                            "Condition": condition,
                            "Muscle": muscle,
                            "Block": block_num,
                            "image": image_label,
                            "Id": get_first_value(row, ["Id", "id"]),
                            "Centroid": f"({centroid_x_nm:.6f}, {centroid_y_nm:.6f})",
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
                            "Elongation": get_first_value(
                                row, ["Elongation", "Aspect Ratio (Elongation)"]
                            ),
                            "Circularity": get_first_value(
                                row, ["Circularity", "Circularity (Form Factor)"]
                            ),
                            "Solidity": get_first_value(
                                row, ["Solidity", "Solidity (Branching)"]
                            ),
                            "NND": "" if nnd_nm is None else f"{nnd_nm:.6f}",
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
        image_records=image_records,
        min_major_axis_px=MIN_MAJOR_AXIS_PX,
        min_minor_axis_px=MIN_MINOR_AXIS_PX,
    )
    removed_total = removed_axis_size + removed_edge_touch + removed_disconnected_parts
    pixel_size_sources = Counter(record.pixel_size_source for record in image_records.values())
    print(f"Wrote full measurements CSV: {OUTPUT_CSV}")
    print(f"Wrote cleaned measurements CSV: {OUTPUT_CLEANED_CSV}")
    print(f"Resolved pixel size sources: {dict(pixel_size_sources)}")
    print(
        "Cleaning summary: "
        f"total={total_rows}, kept={kept_rows}, removed={removed_total}, "
        f"axis_size={removed_axis_size}, edge_touch={removed_edge_touch}, "
        f"disconnected_parts={removed_disconnected_parts}"
    )


if __name__ == "__main__":
    main()
