#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tifffile as tiff

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_ROOT = REPO_ROOT / "data" / "Calpaine_3" / "Processed"
OUTPUT_CSV = REPO_ROOT / "data" / "Calpaine_3" / "results" / "measurments.csv"
OUTPUT_CLEANED_CSV = REPO_ROOT / "data" / "Calpaine_3" / "results" / "measurments_cleaned.csv"
OUTPUT_IMAGE_SUMMARY_CSV = (
    REPO_ROOT / "data" / "Calpaine_3" / "results" / "measurments_cleaned_image_summary.csv"
)
REFERENCE_MAGNIFICATION = 6800.0
REFERENCE_PIXEL_SIZE_UM = 0.0015396
MIN_MAJOR_AXIS_PX = 30.0
MIN_MINOR_AXIS_PX = 5.0
SPATIAL_RADIUS_BIN_COUNT = 20
SPATIAL_MAX_RADIUS_FRACTION = 0.25
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


def classify_image_region(
    *,
    centroid_x_px: float,
    centroid_y_px: float,
    width_px: int,
    height_px: int,
) -> str:
    """Classify a centroid as inside the centered 50%-area square or outside it.

    Args:
        centroid_x_px (float): Object centroid x-coordinate in pixels.
        centroid_y_px (float): Object centroid y-coordinate in pixels.
        width_px (int): Image width in pixels from metadata.
        height_px (int): Image height in pixels from metadata.

    Returns:
        str: ``center`` when the centroid is inside the centered square, otherwise ``periphery``.
    """
    if width_px <= 0 or height_px <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width_px}x{height_px}.")

    target_side_px = (0.5 * float(width_px) * float(height_px)) ** 0.5
    square_side_px = min(target_side_px, float(width_px), float(height_px))
    half_side_px = square_side_px / 2.0
    center_x_px = (float(width_px) - 1.0) / 2.0
    center_y_px = (float(height_px) - 1.0) / 2.0
    inside_x = abs(centroid_x_px - center_x_px) <= half_side_px
    inside_y = abs(centroid_y_px - center_y_px) <= half_side_px
    return "center" if inside_x and inside_y else "periphery"


def parse_optional_float(text: str | None) -> float | None:
    """Parse optional CSV text into a float.

    Args:
        text (str | None): CSV value that may be blank or missing.

    Returns:
        float | None: Parsed numeric value, or ``None`` when the value is empty.
    """

    if text is None or text == "":
        return None
    return float(text)


def numeric_values(rows: list[dict[str, str]], column: str) -> list[float]:
    """Collect numeric values for one CSV column.

    Args:
        rows (list[dict[str, str]]): CSV rows from one image group.
        column (str): Column name to extract from each row.

    Returns:
        list[float]: Non-empty values parsed as floats.
    """

    values: list[float] = []
    for row in rows:
        value = parse_optional_float(row.get(column))
        if value is not None:
            values.append(value)
    return values


def mean_or_none(values: list[float]) -> float | None:
    """Calculate the arithmetic mean for a list of values.

    Args:
        values (list[float]): Numeric values to average.

    Returns:
        float | None: Mean value, or ``None`` when no values are provided.
    """

    if not values:
        return None
    return sum(values) / float(len(values))


def sum_or_none(values: list[float]) -> float | None:
    """Calculate the sum for a list of values.

    Args:
        values (list[float]): Numeric values to sum.

    Returns:
        float | None: Summed value, or ``None`` when no values are provided.
    """

    if not values:
        return None
    return sum(values)


def coefficient_of_variation(values: list[float]) -> float | None:
    """Calculate the population coefficient of variation.

    Args:
        values (list[float]): Numeric values whose variation should be summarized.

    Returns:
        float | None: Standard deviation divided by the mean, or ``None`` when
        fewer than two values are available or the mean is zero.
    """

    if len(values) < 2:
        return None
    mean_value = mean_or_none(values)
    if mean_value is None or mean_value == 0.0:
        return None
    variance = sum((value - mean_value) ** 2 for value in values) / float(len(values))
    return math.sqrt(variance) / mean_value


def format_optional_float(value: float | None, digits: int = 8) -> str:
    """Format optional floating-point output for CSV writing.

    Args:
        value (float | None): Numeric value to format.
        digits (int): Number of digits after the decimal point.

    Returns:
        str: Formatted decimal text, or an empty string when ``value`` is ``None``.
    """

    if value is None:
        return ""
    return f"{value:.{digits}f}"


def pairwise_distances_nm(coords_nm: list[tuple[float, float]]) -> list[float]:
    """Calculate unique pairwise centroid distances in nanometers.

    Args:
        coords_nm (list[tuple[float, float]]): Centroid coordinates as ``(x, y)``
            pairs in nanometers.

    Returns:
        list[float]: Distances for each unique unordered centroid pair.
    """

    distances: list[float] = []
    for i, (x_a, y_a) in enumerate(coords_nm):
        for x_b, y_b in coords_nm[i + 1 :]:
            distances.append(math.hypot(x_a - x_b, y_a - y_b))
    return distances


def spatial_radii_nm(width_nm: float, height_nm: float) -> list[float]:
    """Build evenly spaced radii for image-level spatial summaries.

    Args:
        width_nm (float): Physical image width in nanometers.
        height_nm (float): Physical image height in nanometers.

    Returns:
        list[float]: Positive radii up to a fixed fraction of the shorter image side.
    """

    max_radius = min(width_nm, height_nm) * SPATIAL_MAX_RADIUS_FRACTION
    if max_radius <= 0.0:
        return []
    step = max_radius / float(SPATIAL_RADIUS_BIN_COUNT)
    return [step * float(index) for index in range(1, SPATIAL_RADIUS_BIN_COUNT + 1)]


def trapezoid_integral(x_values: list[float], y_values: list[float]) -> float | None:
    """Integrate sampled values with the trapezoid rule.

    Args:
        x_values (list[float]): Monotonic x-axis sample positions.
        y_values (list[float]): Function values sampled at ``x_values``.

    Returns:
        float | None: Trapezoid-rule integral, or ``None`` when fewer than two
        samples are provided.
    """

    if len(x_values) < 2 or len(x_values) != len(y_values):
        return None
    total = 0.0
    for index in range(1, len(x_values)):
        width = x_values[index] - x_values[index - 1]
        total += width * (y_values[index] + y_values[index - 1]) / 2.0
    return total


def ripley_l_integral(
    coords_nm: list[tuple[float, float]],
    image_area_nm2: float,
    radii_nm: list[float],
) -> float | None:
    """Integrate signed Ripley L-function deviation from complete spatial randomness.

    Args:
        coords_nm (list[tuple[float, float]]): Cleaned centroid coordinates in nanometers.
        image_area_nm2 (float): Physical image area in square nanometers.
        radii_nm (list[float]): Radii at which to evaluate the L-function.

    Returns:
        float | None: Integral of ``L(r) - r`` across radii, or ``None`` when the
        estimate cannot be computed.
    """

    count = len(coords_nm)
    if count < 2 or image_area_nm2 <= 0.0 or not radii_nm:
        return None

    distances = pairwise_distances_nm(coords_nm)
    deviations = [0.0]
    for radius in radii_nm:
        unordered_pairs = sum(1 for distance in distances if distance <= radius)
        ordered_pairs = 2.0 * float(unordered_pairs)
        k_value = image_area_nm2 * ordered_pairs / float(count * (count - 1))
        l_value = math.sqrt(max(k_value, 0.0) / math.pi)
        deviations.append(l_value - radius)
    return trapezoid_integral([0.0, *radii_nm], deviations)


def pair_correlation_integral(
    coords_nm: list[tuple[float, float]],
    image_area_nm2: float,
    radii_nm: list[float],
) -> float | None:
    """Integrate signed pair-correlation deviation from complete spatial randomness.

    Args:
        coords_nm (list[tuple[float, float]]): Cleaned centroid coordinates in nanometers.
        image_area_nm2 (float): Physical image area in square nanometers.
        radii_nm (list[float]): Annular outer radii used to estimate ``g(r)``.

    Returns:
        float | None: Integral of ``g(r) - 1`` across annular radii, or ``None``
        when the estimate cannot be computed.
    """

    count = len(coords_nm)
    if count < 2 or image_area_nm2 <= 0.0 or not radii_nm:
        return None

    distances = pairwise_distances_nm(coords_nm)
    previous_radius = 0.0
    total = 0.0
    for radius in radii_nm:
        annulus_area = math.pi * (radius**2 - previous_radius**2)
        if annulus_area <= 0.0:
            previous_radius = radius
            continue
        unordered_pairs = sum(
            1 for distance in distances if previous_radius < distance <= radius
        )
        ordered_pairs = 2.0 * float(unordered_pairs)
        g_value = image_area_nm2 * ordered_pairs / (
            float(count * (count - 1)) * annulus_area
        )
        total += (g_value - 1.0) * (radius - previous_radius)
        previous_radius = radius
    return total


def summarize_image_rows(
    rows: list[dict[str, str]],
    image_record: ImageRecord,
) -> dict[str, str]:
    """Summarize cleaned instance rows for one image.

    Args:
        rows (list[dict[str, str]]): Cleaned measurement rows belonging to one image.
        image_record (ImageRecord): Metadata and calibration for the image.

    Returns:
        dict[str, str]: CSV-ready image-level summary values.
    """

    width_nm = float(image_record.width_px) * image_record.pixel_size_nm
    height_nm = float(image_record.height_px) * image_record.pixel_size_nm
    image_area_nm2 = width_nm * height_nm
    center_rows = [
        row for row in rows if row.get("Image_Region", "").strip().lower() == "center"
    ]
    coords_nm = [
        parse_centroid(get_first_value(row, ["Centroid", "centroid"])) for row in rows
    ]
    radii_nm = spatial_radii_nm(width_nm, height_nm)

    summary = {
        "Density": str(len(rows)),
        "Image_width_px": str(image_record.width_px),
        "Image_height_px": str(image_record.height_px),
        "Pixel_size_nm": format_optional_float(image_record.pixel_size_nm, digits=8),
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
    return summary


def write_image_summary_csv(
    cleaned_csv: Path,
    image_summary_csv: Path,
    image_records: dict[tuple[str, str, str], ImageRecord],
) -> int:
    """Write one cleaned-measurement summary row per image.

    Args:
        cleaned_csv (Path): Cleaned instance-level measurements CSV path.
        image_summary_csv (Path): Destination image-level summary CSV path.
        image_records (dict[tuple[str, str, str], ImageRecord]): Per-image metadata
            keyed by ``(Condition, Muscle, image)``.

    Returns:
        int: Number of image summary rows written.
    """

    grouped_rows: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    with open(cleaned_csv, "r", newline="", encoding="utf-8") as in_handle:
        reader = csv.DictReader(in_handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in: {cleaned_csv}")
        for row in reader:
            condition = get_first_value(row, ["Condition"])
            muscle = get_first_value(row, ["Muscle"])
            block = get_first_value(row, ["Block"])
            image = get_first_value(row, ["image"])
            grouped_rows.setdefault((condition, muscle, block, image), []).append(row)

    fieldnames = [
        "Condition",
        "Muscle",
        "Block",
        "image",
        "Density",
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

    image_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(image_summary_csv, "w", newline="", encoding="utf-8") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition, muscle, block, image in sorted(grouped_rows):
            image_record = image_records.get((condition, muscle, image))
            if image_record is None:
                raise KeyError(
                    f"Missing image metadata for: {(condition, muscle, image)}"
                )
            summary = summarize_image_rows(
                grouped_rows[(condition, muscle, block, image)], image_record
            )
            writer.writerow(
                {
                    "Condition": condition,
                    "Muscle": muscle,
                    "Block": block,
                    "image": image,
                    **summary,
                }
            )
    return len(grouped_rows)


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
                    third_nnd_text = get_first_value(row, ["3NND"], required=False)
                    fifth_nnd_text = get_first_value(row, ["5NND"], required=False)
                    voronoi_area_text = get_first_value(
                        row, ["Voronoi_Cell_Area"], required=False
                    )

                    corrected_area_val = maybe_float(corrected_area_text)
                    major_val = maybe_float(major_text)
                    minor_val = maybe_float(minor_text)
                    min_feret_val = maybe_float(min_feret_text)
                    nnd_val = maybe_float(nnd_text)
                    third_nnd_val = maybe_float(third_nnd_text)
                    fifth_nnd_val = maybe_float(fifth_nnd_text)
                    voronoi_area_val = maybe_float(voronoi_area_text)

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
                    third_nnd_nm = (
                        third_nnd_val * image_record.pixel_size_nm
                        if third_nnd_val is not None
                        else None
                    )
                    fifth_nnd_nm = (
                        fifth_nnd_val * image_record.pixel_size_nm
                        if fifth_nnd_val is not None
                        else None
                    )
                    voronoi_area_nm2 = (
                        voronoi_area_val * (image_record.pixel_size_nm**2)
                        if voronoi_area_val is not None
                        else None
                    )

                    writer.writerow(
                        {
                            "Condition": condition,
                            "Muscle": muscle,
                            "Block": block_num,
                            "image": image_label,
                            "Id": get_first_value(row, ["Id", "id"]),
                            "Centroid": f"({centroid_x_nm:.6f}, {centroid_y_nm:.6f})",
                            "Image_Region": classify_image_region(
                                centroid_x_px=centroid_x_px,
                                centroid_y_px=centroid_y_px,
                                width_px=image_record.width_px,
                                height_px=image_record.height_px,
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
                            "3NND": ""
                            if third_nnd_nm is None
                            else f"{third_nnd_nm:.6f}",
                            "5NND": ""
                            if fifth_nnd_nm is None
                            else f"{fifth_nnd_nm:.6f}",
                            "Voronoi_Cell_Area": ""
                            if voronoi_area_nm2 is None
                            else f"{voronoi_area_nm2:.8f}",
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
    image_summary_rows = write_image_summary_csv(
        cleaned_csv=OUTPUT_CLEANED_CSV,
        image_summary_csv=OUTPUT_IMAGE_SUMMARY_CSV,
        image_records=image_records,
    )
    removed_total = removed_axis_size + removed_edge_touch + removed_disconnected_parts
    pixel_size_sources = Counter(record.pixel_size_source for record in image_records.values())
    print(f"Wrote full measurements CSV: {OUTPUT_CSV}")
    print(f"Wrote cleaned measurements CSV: {OUTPUT_CLEANED_CSV}")
    print(f"Wrote image summary CSV: {OUTPUT_IMAGE_SUMMARY_CSV}")
    print(f"Resolved pixel size sources: {dict(pixel_size_sources)}")
    print(
        "Cleaning summary: "
        f"total={total_rows}, kept={kept_rows}, removed={removed_total}, "
        f"axis_size={removed_axis_size}, edge_touch={removed_edge_touch}, "
        f"disconnected_parts={removed_disconnected_parts}"
    )
    print(f"Image summary rows: {image_summary_rows}")


if __name__ == "__main__":
    main()
