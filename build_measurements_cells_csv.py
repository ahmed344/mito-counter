#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile as tiff

INPUT_ROOT = Path("/workspaces/mito-counter/data/DMD/Processed")
OUTPUT_CSV = Path("/workspaces/mito-counter/data/DMD/results/measurements_cells.csv")
OUTPUT_CLEANED_CSV = Path(
    "/workspaces/mito-counter/data/DMD/results/measurements_cells_cleaned.csv"
)
MIN_MAJOR_AXIS_PX = 30.0
MIN_MINOR_AXIS_PX = 5.0
REFERENCE_MAGNIFICATION = 6800.0
REFERENCE_PIXEL_SIZE_UM = 0.0015396

CONDITION_MAP = {
    "WT": "Wildtype",
    "DMD": "Duchenne_Muscular_Dystrophy",
}

MUSCLE_MAP = {
    "TA": "Tibialis Anterior",
    "EOM": "Extraocular Muscle",
}

BLOCK_RE = re.compile(r"(?:^|[^A-Z0-9])(TA|EOM)[\s_-]*([0-9]+)(?=[\s_-]|$)", re.IGNORECASE)
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
UNIT_ALIASES = {
    "um": 1.0,
    "micrometer": 1.0,
    "micrometers": 1.0,
    "micrometre": 1.0,
    "micrometres": 1.0,
    "micron": 1.0,
    "microns": 1.0,
    "nm": 1e-3,
    "nanometer": 1e-3,
    "nanometers": 1e-3,
    "nanometre": 1e-3,
    "nanometres": 1e-3,
    "mm": 1e3,
    "millimeter": 1e3,
    "millimeters": 1e3,
    "millimetre": 1e3,
    "millimetres": 1e3,
    "pm": 1e-6,
    "picometer": 1e-6,
    "picometers": 1e-6,
    "picometre": 1e-6,
    "picometres": 1e-6,
    "a": 1e-4,
    "angstrom": 1e-4,
    "angstroms": 1e-4,
}


@dataclass(frozen=True)
class ImageRecord:
    condition: str
    muscle: str
    image_label: str
    width_px: int
    height_px: int
    image_path: Path
    metadata_path: Path
    cell_mask_path: Path
    pixel_size_um: float
    pixel_size_source: str
    magnification: float | None


def parse_block_number(path: Path) -> str:
    """Extract the optional block number from a DMD metrics filename.

    Args:
        path (Path): Metrics CSV path whose stem may contain tokens such as
            ``TA_DMD_3-1200X`` or ``EOM_WT_2-890X``.

    Returns:
        str: Parsed block number as text, or an empty string when the filename
        does not encode a block number.
    """
    match = BLOCK_RE.search(path.stem)
    if not match:
        return ""
    return match.group(2)


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


def convert_to_um(value: float, unit: str | None) -> float | None:
    """Convert a scalar value to micrometers when its unit is known.

    Args:
        value (float): Numeric value from metadata.
        unit (str | None): Unit associated with the value.

    Returns:
        float | None: Converted micrometer value, or ``None`` when the unit is unknown.
    """
    normalized = normalize_unit_name(unit)
    if normalized is None:
        return None
    factor = UNIT_ALIASES.get(normalized)
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
    """Extract a micrometer-per-pixel value from a metadata mapping.

    Args:
        node (dict[str, Any]): Metadata mapping to inspect.

    Returns:
        float | None: Micrometer-per-pixel value when the mapping stores one, otherwise ``None``.
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
                return value_float
            if normalized_key.endswith("_nm"):
                return value_float * 1e-3
            if normalized_key == "pixelsize" and value_float < 1.0:
                return value_float
            unit_candidates = [
                node.get("units"),
                node.get("unit"),
                node.get("Units"),
                node.get("Unit"),
            ]
            for unit in unit_candidates:
                converted = convert_to_um(value_float, unit if isinstance(unit, str) else None)
                if converted is not None and converted > 0.0:
                    return converted
            if normalized_key == "scale" and value_float < 1.0:
                return value_float
    x_scale_key = normalized_keys.get("x_scale")
    y_scale_key = normalized_keys.get("y_scale")
    if x_scale_key is not None and y_scale_key is not None:
        x_scale = node[x_scale_key]
        y_scale = node[y_scale_key]
        if isinstance(x_scale, (int, float)) and isinstance(y_scale, (int, float)):
            if float(x_scale) > 0.0 and float(y_scale) > 0.0:
                unit_value = node.get("units") or node.get("unit")
                converted = convert_to_um(float(x_scale), unit_value if isinstance(unit_value, str) else None)
                if converted is not None:
                    return converted
    return None


def extract_pixel_size_from_json_metadata(metadata: dict[str, Any]) -> float | None:
    """Try to resolve pixel size from nested JSON metadata.

    Args:
        metadata (dict[str, Any]): Parsed image sidecar metadata.

    Returns:
        float | None: Micrometer-per-pixel value when metadata encodes it, otherwise ``None``.
    """
    for value in walk_nested_values(metadata):
        if not isinstance(value, dict):
            continue
        pixel_size_um = find_pixel_size_from_dict(value)
        if pixel_size_um is not None and pixel_size_um > 0.0:
            return pixel_size_um
    return None


def extract_pixel_size_from_tiff_metadata(image_path: Path) -> float | None:
    """Try to resolve pixel size from TIFF metadata and tags.

    Args:
        image_path (Path): TIFF image path to inspect.

    Returns:
        float | None: Micrometer-per-pixel value when TIFF metadata encodes it, otherwise ``None``.
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
                pixel_size_um = extract_pixel_size_from_json_metadata(parsed)
                if pixel_size_um is not None:
                    return pixel_size_um

        imagej_metadata = tif.imagej_metadata
        if isinstance(imagej_metadata, dict):
            pixel_size_um = extract_pixel_size_from_json_metadata(imagej_metadata)
            if pixel_size_um is not None:
                return pixel_size_um

        x_resolution_tag = page.tags.get("XResolution")
        resolution_unit_tag = page.tags.get("ResolutionUnit")
        if x_resolution_tag is not None and resolution_unit_tag is not None:
            numerator, denominator = x_resolution_tag.value
            if numerator and denominator:
                pixels_per_unit = float(numerator) / float(denominator)
                resolution_unit = int(resolution_unit_tag.value)
                if pixels_per_unit > 0.0:
                    if resolution_unit == 2:
                        return 25400.0 / pixels_per_unit
                    if resolution_unit == 3:
                        return 10000.0 / pixels_per_unit
    return None


def magnification_to_pixel_size_um(magnification: float) -> float:
    """Estimate pixel size from magnification using the existing 6800X calibration reference.

    Args:
        magnification (float): Microscope magnification for the image.

    Returns:
        float: Estimated micrometers per pixel at the provided magnification.
    """
    if magnification <= 0.0:
        raise ValueError(f"Magnification must be positive, got {magnification}.")
    return REFERENCE_PIXEL_SIZE_UM * (REFERENCE_MAGNIFICATION / magnification)


def resolve_pixel_size_um(metadata: dict[str, Any], image_label: str, image_path: Path) -> tuple[float, str, float | None]:
    """Resolve a per-image pixel size using JSON metadata, TIFF metadata, then magnification fallback.

    Args:
        metadata (dict[str, Any]): Parsed image sidecar metadata.
        image_label (str): Image filename stem.
        image_path (Path): TIFF image path used for TIFF-metadata fallback.

    Returns:
        tuple[float, str, float | None]: Resolved micrometers-per-pixel value, a source label,
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
        pixel_size_um = extract_pixel_size_from_json_metadata(metadata_source)
        if pixel_size_um is not None:
            return pixel_size_um, source_name, magnification

    pixel_size_um = extract_pixel_size_from_tiff_metadata(image_path)
    if pixel_size_um is not None:
        return pixel_size_um, "tiff_metadata", magnification

    if magnification is None:
        magnification = extract_magnification_from_filename(image_label)
    if magnification is None:
        raise ValueError(
            f"Unable to resolve pixel size or magnification for image '{image_label}'."
        )
    return magnification_to_pixel_size_um(magnification), "magnification_fallback", magnification


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
        pixel_size_um, pixel_size_source, magnification = resolve_pixel_size_um(
            metadata=metadata,
            image_label=image_label,
            image_path=image_path,
        )

        record = ImageRecord(
            condition=condition,
            muscle=muscle,
            image_label=image_label,
            width_px=int(shape[1]),
            height_px=int(shape[0]),
            image_path=image_path,
            metadata_path=metadata_path,
            cell_mask_path=cell_mask_path,
            pixel_size_um=pixel_size_um,
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


def build_distance_map_px(cell_mask_path: Path) -> np.ndarray:
    """Build a per-pixel distance map to the nearest external background pixel.

    Args:
        cell_mask_path (Path): TIFF mask where non-zero pixels mark cell regions.

    Returns:
        np.ndarray: Float32 distance map in pixels, ignoring enclosed internal black regions.
    """
    mask = load_binary_mask(cell_mask_path)
    external_background = compute_external_background(mask)
    if not np.any(external_background):
        return np.full(mask.shape, np.inf, dtype=np.float32)
    distance_input = np.where(external_background, 0, 1).astype(np.uint8)
    return cv2.distanceTransform(distance_input, distanceType=cv2.DIST_L2, maskSize=5)


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


def segmentation_touches_cell_edge(
    *,
    centroid_to_membrane_px: float,
    major_axis_px: float,
) -> bool:
    """Estimate whether an instance reaches the cell membrane.

    Args:
        centroid_to_membrane_px (float): Centroid distance to the nearest external background pixel.
        major_axis_px (float): Major axis length of the instance in pixels.

    Returns:
        bool: ``True`` when the centroid-to-membrane distance is less than or equal to half
        the major axis length, otherwise ``False``.
    """
    if not np.isfinite(centroid_to_membrane_px):
        return False
    return centroid_to_membrane_px <= (major_axis_px / 2.0)


def clean_measurements_csv(
    measurements_csv: Path,
    cleaned_csv: Path,
    image_records: dict[tuple[str, str, str], ImageRecord],
) -> tuple[int, int, int, int, int]:
    """Create a cleaned measurements CSV using size, cell-edge, and connectivity filters.

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

                major_axis_um = float(get_first_value(row, ["Major_axis_length"]))
                minor_axis_um = float(get_first_value(row, ["Minor_axis_length"]))
                major_axis_px = major_axis_um / record.pixel_size_um
                minor_axis_px = minor_axis_um / record.pixel_size_um
                if (
                    major_axis_px <= MIN_MAJOR_AXIS_PX
                    or minor_axis_px <= MIN_MINOR_AXIS_PX
                ):
                    removed_axis_size += 1
                    continue

                membrane_distance_um = float(
                    get_first_value(row, ["Distance_to_cell_membrane"])
                )
                membrane_distance_px = membrane_distance_um / record.pixel_size_um
                if segmentation_touches_cell_edge(
                    centroid_to_membrane_px=membrane_distance_px,
                    major_axis_px=major_axis_px,
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


def build_output_row(
    *,
    row: dict[str, str],
    condition: str,
    muscle: str,
    block_num: str,
    image_label: str,
    pixel_size_um: float,
    membrane_distance_px: float,
) -> dict[str, str]:
    """Convert one metrics row into the consolidated output schema.

    Args:
        row (dict[str, str]): Raw metrics row from a per-image CSV file.
        condition (str): Normalized condition label.
        muscle (str): Normalized muscle label.
        block_num (str): Parsed block number or an empty string.
        image_label (str): Image stem without file extension.
        pixel_size_um (float): Micrometers per pixel for this image.
        membrane_distance_px (float): Centroid-to-membrane distance in pixels.

    Returns:
        dict[str, str]: Normalized output row ready for the consolidated CSV writer.
    """
    centroid_text = get_first_value(row, ["Centroid", "centroid"])
    cx_px, cy_px = parse_centroid(centroid_text)
    cx_um = cx_px * pixel_size_um
    cy_um = cy_px * pixel_size_um

    area_text = get_first_value(row, ["Area", "area"])
    area_um2 = float(area_text) * (pixel_size_um ** 2)
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

    corrected_area_val = maybe_float(corrected_area_text)
    major_val = maybe_float(major_text)
    minor_val = maybe_float(minor_text)
    min_feret_val = maybe_float(min_feret_text)
    nnd_val = maybe_float(nnd_text)

    corrected_area_um2 = (
        corrected_area_val * (pixel_size_um ** 2)
        if corrected_area_val is not None
        else None
    )
    major_um = major_val * pixel_size_um if major_val is not None else None
    minor_um = minor_val * pixel_size_um if minor_val is not None else None
    min_feret_um = min_feret_val * pixel_size_um if min_feret_val is not None else None
    nnd_um = nnd_val * pixel_size_um if nnd_val is not None else None
    membrane_distance_um = membrane_distance_px * pixel_size_um

    return {
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
        "Elongation": get_first_value(row, ["Elongation", "Aspect Ratio (Elongation)"]),
        "Circularity": get_first_value(row, ["Circularity", "Circularity (Form Factor)"]),
        "Solidity": get_first_value(row, ["Solidity", "Solidity (Branching)"]),
        "NND": "" if nnd_um is None else f"{nnd_um:.6f}",
        "Connected_parts": get_first_value(
            row, ["Connected_parts", "connected_parts"], required=False
        )
        or "1",
        "Distance_to_cell_membrane": f"{membrane_distance_um:.6f}",
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
        "Distance_to_cell_membrane",
    ]

    distance_cache: dict[Path, np.ndarray] = {}
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

            block_num = parse_block_number(metrics_path)
            image_label = parse_image_label(metrics_path)
            image_key = (condition, muscle, image_label)
            record = image_records.get(image_key)
            if record is None:
                raise KeyError(f"Missing metadata record for: {image_key}")

            if record.width_px <= 0 or record.height_px <= 0:
                raise ValueError(f"Invalid image shape for: {record.metadata_path}")

            if record.cell_mask_path not in distance_cache:
                distance_cache[record.cell_mask_path] = build_distance_map_px(
                    record.cell_mask_path
                )
            distance_map_px = distance_cache[record.cell_mask_path]
            if distance_map_px.shape != (record.height_px, record.width_px):
                raise ValueError(
                    "Cell-mask shape does not match metadata shape for "
                    f"{record.cell_mask_path}: mask={distance_map_px.shape}, "
                    f"metadata={(record.height_px, record.width_px)}"
                )

            pixel_size_source_counts[record.pixel_size_source] += 1

            with open(metrics_path, "r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    centroid_text = get_first_value(row, ["Centroid", "centroid"])
                    cx_px, cy_px = parse_centroid(centroid_text)
                    membrane_distance_px = sample_distance_map(distance_map_px, cx_px, cy_px)
                    writer.writerow(
                        build_output_row(
                            row=row,
                            condition=condition,
                            muscle=muscle,
                            block_num=block_num,
                            image_label=image_label,
                            pixel_size_um=record.pixel_size_um,
                            membrane_distance_px=membrane_distance_px,
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
    removed_total = removed_axis_size + removed_edge_touch + removed_disconnected_parts
    print(f"Wrote full measurements CSV: {OUTPUT_CSV}")
    print(f"Wrote cleaned measurements CSV: {OUTPUT_CLEANED_CSV}")
    print(
        "Cleaning summary: "
        f"total={total_rows}, kept={kept_rows}, removed={removed_total}, "
        f"axis_size={removed_axis_size}, cell_edge={removed_edge_touch}, "
        f"disconnected_parts={removed_disconnected_parts}"
    )
    print(
        "Pixel-size sources: "
        + ", ".join(
            f"{source}={count}" for source, count in sorted(pixel_size_source_counts.items())
        )
    )


if __name__ == "__main__":
    main()
