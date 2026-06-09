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

import cv2
import numpy as np
import tifffile as tiff

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_ROOT = REPO_ROOT / "data" / "DMD" / "Processed"
RESULTS_ROOT = REPO_ROOT / "data" / "DMD" / "results"
OUTPUT_CSV = RESULTS_ROOT / "measurements.csv"
OUTPUT_CLEANED_CSV = RESULTS_ROOT / "measurements_cleaned.csv"
OUTPUT_SS_SUMMARY_CSV = RESULTS_ROOT / "measurments_cleaned_ss_summary.csv"
OUTPUT_IMF_SUMMARY_CSV = (
    RESULTS_ROOT / "measurments_cleaned_imf_summary.csv"
)
MIN_MAJOR_AXIS_PX = 30.0
MIN_MINOR_AXIS_PX = 5.0
REFERENCE_MAGNIFICATION = 6800.0
REFERENCE_PIXEL_SIZE_UM = 0.0015396
SS_THRESHOLD_NM = 1500.0
SS_LABEL = "Sub-sarcolemmal (SS)"
IMF_LABEL = "Intermyofibrillar (IMF)"

CONDITION_MAP = {
    "WT": "Wildtype",
    "DMD": "Duchenne_Muscular_Dystrophy",
}

MUSCLE_MAP = {
    "TA": "Tibialis Anterior",
    "EOM": "Extraocular Muscle",
}

DMD_BLOCK_PREFIX_RE = re.compile(r"^(TA|EOM)_(WT|DMD)_([0-9]+)_", re.IGNORECASE)
MAGNIFICATION_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*[xX]")
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


@dataclass(frozen=True)
class ImageRecord:
    condition: str
    muscle: str
    block_num: str
    image_label: str
    width_px: int
    height_px: int
    image_path: Path
    metadata_path: Path
    cell_mask_path: Path
    pixel_size_nm: float
    pixel_size_source: str
    magnification: float | None


def parse_dmd_block_prefix(image_label: str) -> str:
    """Extract the block number from a DMD processed image filename prefix.

    Args:
        image_label (str): Processed image stem beginning with a normalized raw
            block directory, such as ``TA_WT_1_`` or ``EOM_DMD_2_``.

    Returns:
        str: Block number parsed from the leading DMD block-directory prefix.
    """
    match = DMD_BLOCK_PREFIX_RE.match(image_label)
    if match is None:
        raise ValueError(
            "Unable to parse DMD block number from processed image name. "
            f"Expected prefix like 'TA_WT_1_' or 'EOM_DMD_2_': {image_label}"
        )
    return match.group(3)


def parse_image_label(path: Path) -> str:
    """Extract the image label from a metrics filename.

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
        text (str): Centroid text in the form ``\"(x, y)\"``.

    Returns:
        tuple[float, float]: Parsed x and y centroid coordinates in pixels.
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
        image_label (str): Image stem that may contain a token such as ``1200X``.

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
    """Estimate pixel size from magnification using the existing 6800X calibration reference.

    Args:
        magnification (float): Microscope magnification for the image.

    Returns:
        float: Estimated nanometers per pixel at the provided magnification.
    """
    if magnification <= 0.0:
        raise ValueError(f"Magnification must be positive, got {magnification}.")
    return REFERENCE_PIXEL_SIZE_UM * 1000.0 * (REFERENCE_MAGNIFICATION / magnification)


def resolve_pixel_size_nm(metadata: dict[str, Any], image_label: str, image_path: Path) -> tuple[float, str, float | None]:
    """Resolve a per-image pixel size using JSON metadata, TIFF metadata, then magnification fallback.

    Args:
        metadata (dict[str, Any]): Parsed image sidecar metadata.
        image_label (str): Image filename stem.
        image_path (Path): TIFF image path used for TIFF-metadata fallback.

    Returns:
        tuple[float, str, float | None]: Resolved nanometers-per-pixel value, a source label,
        and the magnification used or discovered during resolution.
    """
    magnification = extract_magnification_from_metadata(metadata)
    analysis_metadata_path = image_path.with_suffix("") / "data" / "metadata.json"
    metadata_sources: list[tuple[str, dict[str, Any]]] = [("json_metadata", metadata)]
    if analysis_metadata_path.is_file():
        metadata_sources.append(
            ("analysis_metadata", load_json_file(analysis_metadata_path))
        )
    for source_name, metadata_source in metadata_sources:
        pixel_size_nm = extract_pixel_size_from_json_metadata(metadata_source)
        if pixel_size_nm is not None:
            return pixel_size_nm, source_name, magnification

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
    """Load per-image metadata, shapes, paths, and calibration for the DMD dataset.

    Args:
        input_root (Path): Root directory that contains processed condition and muscle folders.

    Returns:
        dict[tuple[str, str, str], ImageRecord]: Mapping from
        ``(Condition, Muscle, image_stem)`` to resolved image metadata.
    """
    records: dict[tuple[str, str, str], ImageRecord] = {}
    metadata_paths = sorted(input_root.rglob("*.json"))
    for metadata_path in metadata_paths:
        image_path = metadata_path.with_suffix(".tif")
        if not image_path.is_file():
            continue

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
        cell_mask_path = image_path.with_name(f"{image_label}_cells.tif")
        block_num = parse_dmd_block_prefix(image_label)
        pixel_size_nm, pixel_size_source, magnification = resolve_pixel_size_nm(
            metadata=metadata,
            image_label=image_label,
            image_path=image_path,
        )

        record = ImageRecord(
            condition=condition,
            muscle=muscle,
            block_num=block_num,
            image_label=image_label,
            width_px=int(shape[1]),
            height_px=int(shape[0]),
            image_path=image_path,
            metadata_path=metadata_path,
            cell_mask_path=cell_mask_path,
            pixel_size_nm=pixel_size_nm,
            pixel_size_source=pixel_size_source,
            magnification=magnification,
        )
        records[(condition, muscle, image_label)] = record
    return records


def load_binary_mask(path: Path) -> np.ndarray:
    """Read a 2D TIFF mask and convert it to a boolean foreground mask.

    Args:
        path (Path): Cell-mask TIFF path.

    Returns:
        np.ndarray: Boolean array with ``True`` for cell pixels and ``False`` for background.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Cell mask not found: {path}")
    mask = tiff.imread(str(path))
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D cell mask, got shape {mask.shape} for {path}.")
    return mask > 0


def load_cell_labels(path: Path) -> np.ndarray:
    """Read a 2D TIFF cell mask as integer cell-instance labels.

    Args:
        path (Path): Cell-mask TIFF path.

    Returns:
        np.ndarray: Integer label image where background is ``0`` and positive
        values identify cell instances. Binary masks are normalized to cell ``1``.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Cell mask not found: {path}")
    mask = tiff.imread(str(path))
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D cell mask, got shape {mask.shape} for {path}.")

    positive_values = np.unique(mask[mask > 0])
    if positive_values.size <= 1:
        return (mask > 0).astype(np.int32)
    return mask.astype(np.int32)


def sample_cell_id(cell_labels: np.ndarray, x_px: float, y_px: float) -> int:
    """Sample the cell ID at a centroid, falling back to the nearest cell pixel.

    Args:
        cell_labels (np.ndarray): Integer cell-label image.
        x_px (float): X coordinate in image pixels.
        y_px (float): Y coordinate in image pixels.

    Returns:
        int: Positive cell ID assigned to the centroid.
    """
    row = int(np.clip(round(y_px), 0, cell_labels.shape[0] - 1))
    col = int(np.clip(round(x_px), 0, cell_labels.shape[1] - 1))
    cell_id = int(cell_labels[row, col])
    if cell_id > 0:
        return cell_id

    cell_pixels = np.argwhere(cell_labels > 0)
    if cell_pixels.size == 0:
        raise ValueError("Cell-label mask does not contain any positive cell IDs.")
    distances = np.sum((cell_pixels - np.array([row, col])) ** 2, axis=1)
    nearest_row, nearest_col = cell_pixels[int(np.argmin(distances))]
    return int(cell_labels[nearest_row, nearest_col])


def compute_external_background(mask: np.ndarray) -> np.ndarray:
    """Identify background pixels connected to the image border.

    Args:
        mask (np.ndarray): Boolean cell mask where ``True`` marks cell pixels.

    Returns:
        np.ndarray: Boolean array where ``True`` marks external black pixels outside the cell.
    """
    background = (~mask).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(background, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask, dtype=bool)
    border_labels = set(int(value) for value in labels[0, :])
    border_labels.update(int(value) for value in labels[-1, :])
    border_labels.update(int(value) for value in labels[:, 0])
    border_labels.update(int(value) for value in labels[:, -1])
    border_labels.discard(0)
    if not border_labels:
        return np.zeros_like(mask, dtype=bool)
    label_values = np.array(sorted(border_labels), dtype=labels.dtype)
    return np.isin(labels, label_values)


def build_cell_geometry_from_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a cell mask and per-pixel distance map to external background.

    Args:
        mask (np.ndarray): Boolean mask where ``True`` marks one cell or a union
            of cell pixels.

    Returns:
        tuple[np.ndarray, np.ndarray]: Boolean cell mask and Float32 distance map
        in pixels, ignoring enclosed internal black regions.
    """
    external_background = compute_external_background(mask)
    if not np.any(external_background):
        return mask, np.full(mask.shape, np.inf, dtype=np.float32)
    distance_input = np.where(external_background, 0, 1).astype(np.uint8)
    distance_map_px = cv2.distanceTransform(
        distance_input, distanceType=cv2.DIST_L2, maskSize=5
    )
    return mask, distance_map_px


def build_cell_geometry(cell_mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Build combined-cell geometry from a cell-mask TIFF path.

    Args:
        cell_mask_path (Path): TIFF mask where non-zero pixels mark cell regions.

    Returns:
        tuple[np.ndarray, np.ndarray]: Boolean foreground mask and Float32 distance map
        in pixels, ignoring enclosed internal black regions.
    """
    return build_cell_geometry_from_mask(load_binary_mask(cell_mask_path))


def build_distance_map_px(cell_mask_path: Path) -> np.ndarray:
    """Build a per-pixel distance map to the nearest external background pixel.

    Args:
        cell_mask_path (Path): TIFF mask where non-zero pixels mark cell regions.

    Returns:
        np.ndarray: Float32 distance map in pixels, ignoring enclosed internal black regions.
    """
    return build_cell_geometry(cell_mask_path)[1]


def sample_distance_map(distance_map_px: np.ndarray, x_px: float, y_px: float) -> float:
    """Sample a distance map at the nearest pixel to a centroid.

    Args:
        distance_map_px (np.ndarray): Distance map in pixels.
        x_px (float): X coordinate in image pixels.
        y_px (float): Y coordinate in image pixels.

    Returns:
        float: Distance in pixels at the nearest valid image coordinate.
    """
    row = int(np.clip(round(y_px), 0, distance_map_px.shape[0] - 1))
    col = int(np.clip(round(x_px), 0, distance_map_px.shape[1] - 1))
    return float(distance_map_px[row, col])


def segmentation_touches_image_edge(
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


def make_compartment(distance_to_membrane_nm: float) -> str:
    """Classify a mitochondrion as SS or IMF from its membrane distance.

    Args:
        distance_to_membrane_nm (float): Distance from the instance centroid to the
            cell membrane in nanometers.

    Returns:
        str: ``Sub-sarcolemmal (SS)`` when the distance is at most 1500 nm,
        otherwise ``Intermyofibrillar (IMF)``.
    """
    if distance_to_membrane_nm <= SS_THRESHOLD_NM:
        return SS_LABEL
    return IMF_LABEL


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
        rows (list[dict[str, str]]): CSV rows from one image-compartment group.
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
    """Build evenly spaced radii for compartment-level spatial summaries.

    Args:
        width_nm (float): Physical image width in nanometers.
        height_nm (float): Physical image height in nanometers.

    Returns:
        list[float]: Positive radii up to one quarter of the shorter image side.
    """
    max_radius = min(width_nm, height_nm) * 0.25
    if max_radius <= 0.0:
        return []
    step = max_radius / 20.0
    return [step * float(index) for index in range(1, 21)]


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
    compartment_area_nm2: float,
    radii_nm: list[float],
) -> float | None:
    """Integrate signed Ripley L-function deviation from random placement.

    Args:
        coords_nm (list[tuple[float, float]]): Cleaned centroid coordinates in nanometers.
        compartment_area_nm2 (float): Physical SS or IMF compartment area in square nanometers.
        radii_nm (list[float]): Radii at which to evaluate the L-function.

    Returns:
        float | None: Integral of ``L(r) - r`` across radii, or ``None`` when the
        estimate cannot be computed.
    """
    count = len(coords_nm)
    if count < 2 or compartment_area_nm2 <= 0.0 or not radii_nm:
        return None
    distances = pairwise_distances_nm(coords_nm)
    deviations = [0.0]
    for radius in radii_nm:
        unordered_pairs = sum(1 for distance in distances if distance <= radius)
        ordered_pairs = 2.0 * float(unordered_pairs)
        k_value = compartment_area_nm2 * ordered_pairs / float(count * (count - 1))
        l_value = math.sqrt(max(k_value, 0.0) / math.pi)
        deviations.append(l_value - radius)
    return trapezoid_integral([0.0, *radii_nm], deviations)


def pair_correlation_integral(
    coords_nm: list[tuple[float, float]],
    compartment_area_nm2: float,
    radii_nm: list[float],
) -> float | None:
    """Integrate signed pair-correlation deviation from random placement.

    Args:
        coords_nm (list[tuple[float, float]]): Cleaned centroid coordinates in nanometers.
        compartment_area_nm2 (float): Physical SS or IMF compartment area in square nanometers.
        radii_nm (list[float]): Annular outer radii used to estimate ``g(r)``.

    Returns:
        float | None: Integral of ``g(r) - 1`` across annular radii, or ``None``
        when the estimate cannot be computed.
    """
    count = len(coords_nm)
    if count < 2 or compartment_area_nm2 <= 0.0 or not radii_nm:
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
        g_value = compartment_area_nm2 * ordered_pairs / (
            float(count * (count - 1)) * annulus_area
        )
        total += (g_value - 1.0) * (radius - previous_radius)
        previous_radius = radius
    return total


def compartment_areas_nm2(
    image_record: ImageRecord,
    mask: np.ndarray,
    distance_map_px: np.ndarray,
) -> dict[str, float]:
    """Calculate SS and IMF cell-mask areas in square nanometers.

    Args:
        image_record (ImageRecord): Image metadata containing the pixel calibration.
        mask (np.ndarray): Boolean cell mask where ``True`` marks cell pixels.
        distance_map_px (np.ndarray): Distance to external background in pixels.

    Returns:
        dict[str, float]: Mapping from compartment label to metadata-calibrated area.
    """
    threshold_px = SS_THRESHOLD_NM / image_record.pixel_size_nm
    ss_pixels = mask & (distance_map_px <= threshold_px)
    imf_pixels = mask & (distance_map_px > threshold_px)
    pixel_area_nm2 = image_record.pixel_size_nm**2
    return {
        SS_LABEL: float(np.count_nonzero(ss_pixels)) * pixel_area_nm2,
        IMF_LABEL: float(np.count_nonzero(imf_pixels)) * pixel_area_nm2,
    }


def clean_measurements_csv(
    measurements_csv: Path,
    cleaned_csv: Path,
    image_records: dict[tuple[str, str, str], ImageRecord],
) -> tuple[int, int, int, int, int]:
    """Create a cleaned measurements CSV using size, image-edge, and connectivity filters.

    Args:
        measurements_csv (Path): Source measurements CSV path.
        cleaned_csv (Path): Output cleaned CSV path.
        image_records (dict[tuple[str, str, str], ImageRecord]): Per-image record mapping keyed by
            ``(Condition, Muscle, image)``.

    Returns:
        tuple[int, int, int, int, int]: ``(kept_rows, removed_axis_size, removed_edge_touch,
        removed_disconnected_parts, total_rows)``.
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
                image_key = (condition, muscle, image)
                record = image_records.get(image_key)
                if record is None:
                    raise KeyError(f"Missing image metadata for: {image_key}")

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


def summarize_compartment_rows(
    rows: list[dict[str, str]],
    image_record: ImageRecord,
    compartment: str,
    compartment_area_nm2: float,
    cell_area_nm2: float,
) -> dict[str, str]:
    """Summarize cleaned instance rows for one cell compartment.

    Args:
        rows (list[dict[str, str]]): Cleaned measurement rows belonging to one cell compartment.
        image_record (ImageRecord): Metadata and calibration for the source image.
        compartment (str): Compartment label being summarized.
        compartment_area_nm2 (float): Physical compartment area in square nanometers.
        cell_area_nm2 (float): Physical cell-mask area in square nanometers.

    Returns:
        dict[str, str]: CSV-ready compartment summary values.
    """
    width_nm = float(image_record.width_px) * image_record.pixel_size_nm
    height_nm = float(image_record.height_px) * image_record.pixel_size_nm
    center_rows = [
        row for row in rows if row.get("Image_Region", "").strip().lower() == "center"
    ]
    coords_nm = [
        parse_centroid(get_first_value(row, ["Centroid", "centroid"])) for row in rows
    ]
    radii_nm = spatial_radii_nm(width_nm, height_nm)
    instance_count = len(rows)
    density = (
        float(instance_count) / compartment_area_nm2
        if compartment_area_nm2 > 0.0
        else None
    )

    return {
        "Compartment": compartment,
        "Density": format_optional_float(density, digits=12),
        "Instance_count": str(instance_count),
        "Zoom": format_optional_float(image_record.magnification, digits=6),
        "Image_width_px": str(image_record.width_px),
        "Image_height_px": str(image_record.height_px),
        "Pixel_size_nm": format_optional_float(image_record.pixel_size_nm, digits=8),
        "Cell_area_nm2": format_optional_float(cell_area_nm2, digits=8),
        "Compartment_area_nm2": format_optional_float(compartment_area_nm2, digits=8),
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
            ripley_l_integral(coords_nm, compartment_area_nm2, radii_nm)
        ),
        "Pair_Correlation_integral": format_optional_float(
            pair_correlation_integral(coords_nm, compartment_area_nm2, radii_nm)
        ),
    }


def write_compartment_summary_csv(
    *,
    cleaned_csv: Path,
    summary_csv: Path,
    image_records: dict[tuple[str, str, str], ImageRecord],
    compartment: str,
) -> int:
    """Write one cleaned-measurement summary row per cell for one compartment.

    Args:
        cleaned_csv (Path): Cleaned instance-level measurements CSV path.
        summary_csv (Path): Destination compartment summary CSV path.
        image_records (dict[tuple[str, str, str], ImageRecord]): Per-image metadata
            keyed by ``(Condition, Muscle, image)``.
        compartment (str): Compartment label to summarize.

    Returns:
        int: Number of compartment summary rows written.
    """
    grouped_rows: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = {}
    with open(cleaned_csv, "r", newline="", encoding="utf-8") as in_handle:
        reader = csv.DictReader(in_handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in: {cleaned_csv}")
        for row in reader:
            condition = get_first_value(row, ["Condition"])
            muscle = get_first_value(row, ["Muscle"])
            block = row.get("Block", "")
            image = get_first_value(row, ["image"])
            cell_id = get_first_value(row, ["Cell_id"])
            row_compartment = get_first_value(row, ["Compartment"])
            if row_compartment == compartment:
                grouped_rows.setdefault(
                    (condition, muscle, block, image, cell_id, compartment), []
                ).append(row)

    fieldnames = [
        "Condition",
        "Muscle",
        "Block",
        "image",
        "Cell_id",
        "Compartment",
        "Density",
        "Instance_count",
        "Zoom",
        "Image_width_px",
        "Image_height_px",
        "Pixel_size_nm",
        "Cell_area_nm2",
        "Compartment_area_nm2",
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

    cell_label_cache: dict[Path, np.ndarray] = {}
    cell_geometry_cache: dict[tuple[Path, int], tuple[np.ndarray, np.ndarray]] = {}
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with open(summary_csv, "w", newline="", encoding="utf-8") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()
        for (
            condition,
            muscle,
            block,
            image,
            cell_id_text,
            _,
        ), rows in sorted(grouped_rows.items()):
            image_record = image_records.get((condition, muscle, image))
            if image_record is None:
                raise KeyError(
                    f"Missing image metadata for: {(condition, muscle, image)}"
                )
            if image_record.cell_mask_path not in cell_label_cache:
                cell_label_cache[image_record.cell_mask_path] = load_cell_labels(
                    image_record.cell_mask_path
                )
            cell_labels = cell_label_cache[image_record.cell_mask_path]
            if cell_labels.shape != (image_record.height_px, image_record.width_px):
                raise ValueError(
                    "Cell-mask shape does not match metadata shape for "
                    f"{image_record.cell_mask_path}: mask={cell_labels.shape}, "
                    f"metadata={(image_record.height_px, image_record.width_px)}"
                )
            cell_id = int(cell_id_text)
            geometry_key = (image_record.cell_mask_path, cell_id)
            if geometry_key not in cell_geometry_cache:
                cell_geometry_cache[geometry_key] = build_cell_geometry_from_mask(
                    cell_labels == cell_id
                )
            mask, distance_map_px = cell_geometry_cache[geometry_key]
            areas_nm2 = compartment_areas_nm2(image_record, mask, distance_map_px)
            cell_area_nm2 = float(np.count_nonzero(mask)) * (image_record.pixel_size_nm**2)
            summary = summarize_compartment_rows(
                rows=rows,
                image_record=image_record,
                compartment=compartment,
                compartment_area_nm2=areas_nm2[compartment],
                cell_area_nm2=cell_area_nm2,
            )
            writer.writerow(
                {
                    "Condition": condition,
                    "Muscle": muscle,
                    "Block": block,
                    "image": image,
                    "Cell_id": cell_id_text,
                    **summary,
                }
            )
            rows_written += 1
    return rows_written


def build_output_row(
    *,
    row: dict[str, str],
    condition: str,
    muscle: str,
    block_num: str,
    image_label: str,
    cell_id: int,
    pixel_size_nm: float,
    zoom: float | None,
    membrane_distance_px: float,
    image_width_px: int,
    image_height_px: int,
) -> dict[str, str]:
    """Convert one metrics row into the consolidated output schema.

    Args:
        row (dict[str, str]): Raw metrics row from a per-image CSV file.
        condition (str): Normalized condition label.
        muscle (str): Normalized muscle label.
        block_num (str): Block number parsed from the processed image filename prefix.
        image_label (str): Image stem without file extension.
        cell_id (int): Positive cell-instance label assigned to this mitochondrion.
        pixel_size_nm (float): Nanometers per pixel for this image.
        zoom (float | None): Microscope zoom/magnification for this image.
        membrane_distance_px (float): Centroid-to-membrane distance in pixels.
        image_width_px (int): Image width in pixels from metadata.
        image_height_px (int): Image height in pixels from metadata.

    Returns:
        dict[str, str]: Normalized output row ready for the consolidated CSV writer.
    """
    centroid_text = get_first_value(row, ["Centroid", "centroid"])
    cx_px, cy_px = parse_centroid(centroid_text)
    cx_nm = cx_px * pixel_size_nm
    cy_nm = cy_px * pixel_size_nm

    area_text = get_first_value(row, ["Area", "area"])
    area_nm2 = float(area_text) * (pixel_size_nm ** 2)
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
        corrected_area_val * (pixel_size_nm ** 2)
        if corrected_area_val is not None
        else None
    )
    major_nm = major_val * pixel_size_nm if major_val is not None else None
    minor_nm = minor_val * pixel_size_nm if minor_val is not None else None
    min_feret_nm = min_feret_val * pixel_size_nm if min_feret_val is not None else None
    nnd_nm = nnd_val * pixel_size_nm if nnd_val is not None else None
    third_nnd_nm = (
        third_nnd_val * pixel_size_nm if third_nnd_val is not None else None
    )
    fifth_nnd_nm = (
        fifth_nnd_val * pixel_size_nm if fifth_nnd_val is not None else None
    )
    voronoi_area_nm2 = (
        voronoi_area_val * (pixel_size_nm ** 2)
        if voronoi_area_val is not None
        else None
    )
    membrane_distance_nm = membrane_distance_px * pixel_size_nm

    return {
        "Condition": condition,
        "Muscle": muscle,
        "Block": block_num,
        "image": image_label,
        "Zoom": "" if zoom is None else f"{zoom:.6f}",
        "Id": get_first_value(row, ["Id", "id"]),
        "Cell_id": str(cell_id),
        "Centroid": f"({cx_nm:.6f}, {cy_nm:.6f})",
        "Image_Region": classify_image_region(
            centroid_x_px=cx_px,
            centroid_y_px=cy_px,
            width_px=image_width_px,
            height_px=image_height_px,
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
        "Distance_to_cell_membrane": f"{membrane_distance_nm:.6f}",
        "Compartment": make_compartment(membrane_distance_nm),
    }


def main() -> None:
    """Build the consolidated DMD measurements CSVs with cell-membrane distances.

    Args:
        None: This function does not accept arguments.

    Returns:
        None: Writes the consolidated and cleaned CSV files to disk.
    """
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Input root not found: {INPUT_ROOT}")

    image_records = load_image_records(INPUT_ROOT)
    if not image_records:
        raise ValueError(f"No image metadata files found under: {INPUT_ROOT}")

    metrics_paths = []
    for metrics_path in sorted(INPUT_ROOT.rglob("*_segmented_metrics.csv")):
        parts = metrics_path.parts
        try:
            condition_raw = parts[parts.index("Processed") + 1]
            muscle_raw = parts[parts.index("Processed") + 2]
        except (ValueError, IndexError):
            continue
        condition = CONDITION_MAP.get(condition_raw)
        muscle = MUSCLE_MAP.get(muscle_raw)
        if condition is None or muscle is None:
            continue
        image_label = parse_image_label(metrics_path)
        if (condition, muscle, image_label) in image_records:
            metrics_paths.append(metrics_path)
    if not metrics_paths:
        raise ValueError(f"No metrics files found under: {INPUT_ROOT}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output_fields = [
        "Condition",
        "Muscle",
        "Block",
        "image",
        "Zoom",
        "Id",
        "Cell_id",
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
        "Distance_to_cell_membrane",
        "Compartment",
    ]

    cell_label_cache: dict[Path, np.ndarray] = {}
    cell_geometry_cache: dict[tuple[Path, int], tuple[np.ndarray, np.ndarray]] = {}
    pixel_size_source_counts: Counter[str] = Counter()

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
                raise ValueError(f"Unknown condition or muscle in path: {metrics_path}")

            image_label = parse_image_label(metrics_path)
            image_key = (condition, muscle, image_label)
            record = image_records.get(image_key)
            if record is None:
                raise KeyError(f"Missing metadata record for: {image_key}")

            if record.width_px <= 0 or record.height_px <= 0:
                raise ValueError(f"Invalid image shape for: {record.metadata_path}")

            if record.cell_mask_path not in cell_label_cache:
                cell_label_cache[record.cell_mask_path] = load_cell_labels(
                    record.cell_mask_path
                )
            cell_labels = cell_label_cache[record.cell_mask_path]
            if cell_labels.shape != (record.height_px, record.width_px):
                raise ValueError(
                    "Cell-mask shape does not match metadata shape for "
                    f"{record.cell_mask_path}: mask={cell_labels.shape}, "
                    f"metadata={(record.height_px, record.width_px)}"
                )

            pixel_size_source_counts[record.pixel_size_source] += 1

            with open(metrics_path, "r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    centroid_text = get_first_value(row, ["Centroid", "centroid"])
                    cx_px, cy_px = parse_centroid(centroid_text)
                    cell_id = sample_cell_id(cell_labels, cx_px, cy_px)
                    geometry_key = (record.cell_mask_path, cell_id)
                    if geometry_key not in cell_geometry_cache:
                        cell_geometry_cache[geometry_key] = build_cell_geometry_from_mask(
                            cell_labels == cell_id
                        )
                    _, distance_map_px = cell_geometry_cache[geometry_key]
                    membrane_distance_px = sample_distance_map(distance_map_px, cx_px, cy_px)
                    writer.writerow(
                        build_output_row(
                            row=row,
                            condition=condition,
                            muscle=muscle,
                            block_num=record.block_num,
                            image_label=image_label,
                            cell_id=cell_id,
                            pixel_size_nm=record.pixel_size_nm,
                            zoom=record.magnification,
                            membrane_distance_px=membrane_distance_px,
                            image_width_px=record.width_px,
                            image_height_px=record.height_px,
                        )
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
    removed_total = removed_axis_size + removed_edge_touch + removed_disconnected_parts
    print(f"Wrote full measurements CSV: {OUTPUT_CSV}")
    print(f"Wrote cleaned measurements CSV: {OUTPUT_CLEANED_CSV}")
    print(f"Wrote SS summary CSV: {OUTPUT_SS_SUMMARY_CSV}")
    print(f"Wrote IMF summary CSV: {OUTPUT_IMF_SUMMARY_CSV}")
    print(
        "Cleaning summary: "
        f"total={total_rows}, kept={kept_rows}, removed={removed_total}, "
        f"axis_size={removed_axis_size}, image_edge={removed_edge_touch}, "
        f"disconnected_parts={removed_disconnected_parts}"
    )
    print(
        "Pixel-size sources: "
        + ", ".join(
            f"{source}={count}" for source, count in sorted(pixel_size_source_counts.items())
        )
    )
    print(f"SS summary rows: {ss_summary_rows}")
    print(f"IMF summary rows: {imf_summary_rows}")


if __name__ == "__main__":
    main()
