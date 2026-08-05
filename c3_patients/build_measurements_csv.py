#!/usr/bin/env python3
"""Build calibrated instance and image-summary CSVs for Calpain-3 patients."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import tifffile as tiff


REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_ROOT = REPO_ROOT / "data" / "Calpaine_3_patients" / "Processed"
RESULTS_ROOT = REPO_ROOT / "data" / "Calpaine_3_patients" / "results"
OUTPUT_CSV = RESULTS_ROOT / "measurements.csv"
OUTPUT_CLEANED_CSV = RESULTS_ROOT / "measurements_cleaned.csv"
OUTPUT_SS_SUMMARY_CSV = RESULTS_ROOT / "measurements_cleaned_ss_summary.csv"
OUTPUT_IMF_SUMMARY_CSV = RESULTS_ROOT / "measurements_cleaned_imf_summary.csv"
OUTPUT_NO_COMPARTMENT_CSV = RESULTS_ROOT / "measurements_cleaned_no_compartment.csv"
OUTPUT_NO_COMPARTMENT_SUMMARY_CSV = (
    RESULTS_ROOT / "measurements_cleaned_no_compartment_summary.csv"
)

REFERENCE_MAGNIFICATION = 4800.0
REFERENCE_PIXEL_SIZE_UM = 0.00218
MIN_MAJOR_AXIS_PX = 30.0
MIN_MINOR_AXIS_PX = 5.0
SPATIAL_RADIUS_BIN_COUNT = 20
SPATIAL_MAX_RADIUS_FRACTION = 0.25
SS_LABEL = "Sub-sarcolemmal (SS)"
IMF_LABEL = "Intermyofibrillar (IMF)"
ALL_COMPARTMENTS_LABEL = "All compartments"
MAGNIFICATION_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*[xX]")

PATIENT_FIELDS = [
    "Condition",
    "IDENTIFIER",
    "GENOTYPE",
    "DOB",
    "AGE",
    "GENDER",
    "SITE OF BIOPSY",
    "Notes",
    "Sample_ID",
]
COMPATIBILITY_FIELDS = ["Muscle", "Block", "image", "Compartment"]
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
    "pm": 0.001,
    "picometer": 0.001,
    "picometers": 0.001,
    "picometre": 0.001,
    "picometres": 0.001,
    "a": 0.1,
    "angstrom": 0.1,
    "angstroms": 0.1,
}

ImageKey = tuple[str, str, str, str]


@dataclass(frozen=True)
class ImageRecord:
    """Store clinical, path, shape, and calibration data for one source image.

    Args:
        condition (str): Condition directory label.
        identifier (str): Patient identifier directory label.
        genotype (str): Patient genotype from basic JSON metadata.
        dob (str): Patient date of birth from basic JSON metadata.
        age (str): Patient age from basic JSON metadata.
        gender (str): Patient gender from basic JSON metadata.
        biopsy_site (str): Biopsy site from basic JSON metadata.
        notes (str): Patient notes from basic JSON metadata.
        sample_id (str): Sample identifier from basic JSON metadata.
        compartment (str): Normalized source compartment label.
        image_label (str): Source image stem.
        width_px (int): Image width in pixels.
        height_px (int): Image height in pixels.
        image_path (Path): Source TIFF path.
        metadata_path (Path): Source JSON sidecar path.
        metrics_path (Path): Expected per-instance metrics CSV path.
        pixel_size_nm (float): Resolved nanometers per pixel.
        pixel_size_source (str): Source used to resolve the pixel size.
        magnification (float | None): Resolved or inferred magnification.

    Returns:
        None: Dataclass initialization creates a record and returns no value.
    """

    condition: str
    identifier: str
    genotype: str
    dob: str
    age: str
    gender: str
    biopsy_site: str
    notes: str
    sample_id: str
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


def normalize_key(value: str) -> str:
    """Normalize a metadata key for case- and separator-insensitive matching.

    Args:
        value (str): Metadata key to normalize.

    Returns:
        str: Uppercase alphanumeric-only key.
    """
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def stringify(value: Any) -> str:
    """Convert an optional metadata value to CSV text.

    Args:
        value (Any): Metadata value to convert.

    Returns:
        str: Empty text for ``None`` or a string representation otherwise.
    """
    return "" if value is None else str(value)


def load_json_file(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk.

    Args:
        path (Path): JSON file to load.

    Returns:
        dict[str, Any]: Parsed top-level JSON mapping.
    """
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in: {path}")
    return data


def basic_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return normalized basic metadata from a sidecar JSON object.

    Args:
        metadata (dict[str, Any]): Parsed image metadata.

    Returns:
        dict[str, Any]: Basic metadata keyed by normalized field names.
    """
    basic = metadata.get("basic", {})
    if not isinstance(basic, dict):
        basic = {}
    return {normalize_key(str(key)): value for key, value in basic.items()}


def basic_value(basic: dict[str, Any], *keys: str) -> str:
    """Read the first present clinical value from normalized basic metadata.

    Args:
        basic (dict[str, Any]): Normalized basic metadata.
        *keys (str): Candidate field names in priority order.

    Returns:
        str: CSV text for the first present value, or empty text.
    """
    for key in keys:
        normalized = normalize_key(key)
        if normalized in basic and basic[normalized] is not None:
            return stringify(basic[normalized])
    return ""


def normalize_compartment(value: str) -> str:
    """Normalize a source directory's compartment label.

    Args:
        value (str): Raw compartment directory name.

    Returns:
        str: DMD1X-compatible full compartment label.
    """
    token = normalize_key(value)
    if token.startswith("SS") or "SUBSARCOLEMMAL" in token:
        return SS_LABEL
    if token.startswith("IMF") or "INTERMYOFIBRILLAR" in token:
        return IMF_LABEL
    return value


def parse_record_path(path: Path, input_root: Path) -> tuple[str, str, str]:
    """Parse condition, patient identifier, and compartment from a JSON path.

    Args:
        path (Path): Image JSON sidecar path.
        input_root (Path): Processed dataset root.

    Returns:
        tuple[str, str, str]: Condition, identifier, and normalized compartment.
    """
    try:
        relative = path.relative_to(input_root)
    except ValueError as exc:
        raise ValueError(f"Metadata path is outside input root: {path}") from exc
    if len(relative.parts) != 4:
        raise ValueError(
            "Expected Processed/<condition>/<identifier>/<compartment>/<image>.json: "
            f"{path}"
        )
    condition, identifier, compartment, _ = relative.parts
    return condition, identifier, normalize_compartment(compartment)


def normalize_unit_name(unit: str | None) -> str | None:
    """Normalize a physical unit name.

    Args:
        unit (str | None): Raw unit text.

    Returns:
        str | None: Normalized unit text, or ``None`` when absent.
    """
    if unit is None:
        return None
    return unit.strip().lower().replace("µ", "u").replace("μ", "u")


def convert_to_nm(value: float, unit: str | None) -> float | None:
    """Convert a physical scale value to nanometers.

    Args:
        value (float): Positive physical scale.
        unit (str | None): Unit associated with the scale.

    Returns:
        float | None: Nanometer value, or ``None`` for an unknown unit.
    """
    normalized = normalize_unit_name(unit)
    factor = UNIT_ALIASES_TO_NM.get(normalized) if normalized is not None else None
    return None if factor is None else value * factor


def walk_nested_values(data: Any) -> Iterable[Any]:
    """Yield a nested metadata object and all descendants depth-first.

    Args:
        data (Any): Nested mapping, sequence, or scalar.

    Returns:
        Iterable[Any]: Iterator over the object and its descendants.
    """
    yield data
    if isinstance(data, dict):
        for child in data.values():
            yield from walk_nested_values(child)
    elif isinstance(data, (list, tuple)):
        for child in data:
            yield from walk_nested_values(child)


def find_pixel_size_from_dict(node: dict[str, Any]) -> float | None:
    """Extract a nanometers-per-pixel scale from one metadata mapping.

    Args:
        node (dict[str, Any]): Metadata mapping to inspect.

    Returns:
        float | None: Positive nanometers per pixel, or ``None`` when absent.
    """
    normalized_keys = {
        str(key).strip().lower().replace(" ", "_"): key for key in node
    }
    unit = next(
        (
            value
            for key in ("units", "unit", "Units", "Unit")
            if isinstance((value := node.get(key)), str)
        ),
        None,
    )
    for candidate in PIXEL_SIZE_KEYWORDS:
        raw_key = normalized_keys.get(candidate)
        if raw_key is None:
            continue
        raw_value = node[raw_key]
        if not isinstance(raw_value, (int, float)) or float(raw_value) <= 0.0:
            continue
        value = float(raw_value)
        if candidate.endswith("_um"):
            return value * 1000.0
        if candidate.endswith("_nm"):
            return value
        converted = convert_to_nm(value, unit)
        if converted is not None and converted > 0.0:
            return converted
        if candidate in {"pixelsize", "pixel_size", "scale"} and value < 1.0:
            return value * 1000.0
    return None


def extract_pixel_size_from_json(metadata: dict[str, Any]) -> float | None:
    """Find pixel size in nested HyperSpy or basic JSON metadata.

    Args:
        metadata (dict[str, Any]): Parsed image metadata.

    Returns:
        float | None: Nanometers per pixel, or ``None`` when unresolved.
    """
    for value in walk_nested_values(metadata):
        if isinstance(value, dict):
            pixel_size_nm = find_pixel_size_from_dict(value)
            if pixel_size_nm is not None:
                return pixel_size_nm
    return None


def extract_pixel_size_from_tiff(image_path: Path) -> float | None:
    """Find pixel size in TIFF description, ImageJ metadata, or resolution tags.

    Args:
        image_path (Path): TIFF image to inspect.

    Returns:
        float | None: Nanometers per pixel, or ``None`` when unresolved.
    """
    if not image_path.is_file():
        return None
    with tiff.TiffFile(str(image_path)) as tif:
        page = tif.pages[0]
        if page.description:
            try:
                description = json.loads(page.description)
            except json.JSONDecodeError:
                description = None
            if isinstance(description, dict):
                pixel_size_nm = extract_pixel_size_from_json(description)
                if pixel_size_nm is not None:
                    return pixel_size_nm
        if isinstance(tif.imagej_metadata, dict):
            pixel_size_nm = extract_pixel_size_from_json(tif.imagej_metadata)
            if pixel_size_nm is not None:
                return pixel_size_nm
        resolution = page.tags.get("XResolution")
        resolution_unit = page.tags.get("ResolutionUnit")
        if resolution is not None and resolution_unit is not None:
            numerator, denominator = resolution.value
            pixels_per_unit = (
                float(numerator) / float(denominator) if numerator and denominator else 0.0
            )
            if pixels_per_unit > 0.0:
                if int(resolution_unit.value) == 2:
                    return 25_400_000.0 / pixels_per_unit
                if int(resolution_unit.value) == 3:
                    return 10_000_000.0 / pixels_per_unit
    return None


def extract_magnification(metadata: dict[str, Any], image_label: str) -> float | None:
    """Extract magnification from metadata or the image filename.

    Args:
        metadata (dict[str, Any]): Parsed image metadata.
        image_label (str): Image stem that may contain an ``X`` token.

    Returns:
        float | None: Positive magnification, or ``None`` when absent.
    """
    for path in (
        ("hyperspy_metadata", "Acquisition_instrument", "TEM", "magnification"),
        ("Acquisition_instrument", "TEM", "magnification"),
        ("basic", "magnification"),
    ):
        current: Any = metadata
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if isinstance(current, (int, float)) and float(current) > 0.0:
            return float(current)
    match = MAGNIFICATION_RE.search(image_label)
    return float(match.group(1)) if match else None


def magnification_to_pixel_size_nm(magnification: float) -> float:
    """Scale the exact patient 4800X calibration to another magnification.

    Args:
        magnification (float): Positive image magnification.

    Returns:
        float: Nanometers per pixel based on 2.18 nm/px at 4800X.
    """
    if magnification <= 0.0:
        raise ValueError(f"Magnification must be positive, got {magnification}.")
    return REFERENCE_PIXEL_SIZE_UM * 1000.0 * (
        REFERENCE_MAGNIFICATION / magnification
    )


def resolve_pixel_size(
    metadata: dict[str, Any], image_label: str, image_path: Path
) -> tuple[float, str, float]:
    """Resolve pixel size from JSON, TIFF metadata, or calibrated magnification.

    Args:
        metadata (dict[str, Any]): Parsed image JSON metadata.
        image_label (str): Source image stem.
        image_path (Path): Source TIFF path.

    Returns:
        tuple[float, str, float]: Nanometers per pixel, source label, and
        magnification used for reporting.
    """
    magnification = extract_magnification(metadata, image_label)
    pixel_size_nm = extract_pixel_size_from_json(metadata)
    if pixel_size_nm is not None:
        return pixel_size_nm, "json_metadata", magnification or REFERENCE_MAGNIFICATION
    pixel_size_nm = extract_pixel_size_from_tiff(image_path)
    if pixel_size_nm is not None:
        return pixel_size_nm, "tiff_metadata", magnification or REFERENCE_MAGNIFICATION
    if magnification is not None:
        return (
            magnification_to_pixel_size_nm(magnification),
            "magnification_fallback",
            magnification,
        )
    return (
        REFERENCE_PIXEL_SIZE_UM * 1000.0,
        "4800x_reference_fallback",
        REFERENCE_MAGNIFICATION,
    )


def resolve_image_shape(
    metadata: dict[str, Any], image_path: Path
) -> tuple[int, int]:
    """Resolve image width and height from JSON with a TIFF fallback.

    Args:
        metadata (dict[str, Any]): Parsed image metadata.
        image_path (Path): Source TIFF path.

    Returns:
        tuple[int, int]: Width and height in pixels.
    """
    shape = metadata.get("basic", {}).get("shape")
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        height_px, width_px = int(shape[-2]), int(shape[-1])
    elif image_path.is_file():
        with tiff.TiffFile(str(image_path)) as tif:
            height_px, width_px = tif.pages[0].shape[-2:]
    else:
        raise ValueError(f"Missing image shape and TIFF file for: {image_path}")
    if width_px <= 0 or height_px <= 0:
        raise ValueError(f"Invalid image shape {width_px}x{height_px}: {image_path}")
    return width_px, height_px


def image_key(record: ImageRecord) -> ImageKey:
    """Build a source-image key that keeps compartment images distinct.

    Args:
        record (ImageRecord): Image record to identify.

    Returns:
        ImageKey: Condition, identifier, compartment, and image stem.
    """
    return (
        record.condition,
        record.identifier,
        record.compartment,
        record.image_label,
    )


def load_image_records(input_root: Path) -> dict[ImageKey, ImageRecord]:
    """Load all image records in the patient Processed tree.

    Args:
        input_root (Path): Root with condition/identifier/compartment images.

    Returns:
        dict[ImageKey, ImageRecord]: Records keyed by distinct source image.
    """
    records: dict[ImageKey, ImageRecord] = {}
    for metadata_path in sorted(input_root.rglob("*.json")):
        try:
            condition, identifier, compartment = parse_record_path(
                metadata_path, input_root
            )
        except ValueError:
            continue
        metadata = load_json_file(metadata_path)
        basic = basic_metadata(metadata)
        metadata_condition = basic_value(basic, "condition").upper()
        metadata_identifier = basic_value(basic, "IDENTIFIER", "patient_id").upper()
        metadata_compartment = normalize_compartment(
            basic_value(basic, "compartment")
        )
        if metadata_condition != condition:
            raise ValueError(
                f"Condition mismatch between path and metadata: {metadata_path}"
            )
        if metadata_identifier != identifier:
            raise ValueError(
                f"Identifier mismatch between path and metadata: {metadata_path}"
            )
        if metadata_compartment != compartment:
            raise ValueError(
                f"Compartment mismatch between path and metadata: {metadata_path}"
            )
        image_label = metadata_path.stem
        image_path = metadata_path.with_suffix(".tif")
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Source TIFF is missing for metadata sidecar: {metadata_path}"
            )
        width_px, height_px = resolve_image_shape(metadata, image_path)
        pixel_size_nm, pixel_size_source, magnification = resolve_pixel_size(
            metadata, image_label, image_path
        )
        record = ImageRecord(
            condition=condition,
            identifier=identifier,
            genotype=basic_value(basic, "GENOTYPE"),
            dob=basic_value(basic, "DOB"),
            age=basic_value(basic, "AGE"),
            gender=basic_value(basic, "GENDER", "SEX"),
            biopsy_site=basic_value(
                basic, "SITE OF BIOPSY", "SITE_OF_BIOPSY", "BIOPSY SITE"
            ),
            notes=basic_value(basic, "Notes"),
            sample_id=basic_value(basic, "Sample_ID", "SAMPLE ID"),
            compartment=compartment,
            image_label=image_label,
            width_px=width_px,
            height_px=height_px,
            image_path=image_path,
            metadata_path=metadata_path,
            metrics_path=metadata_path.with_name(
                f"{image_label}_segmented_metrics.csv"
            ),
            pixel_size_nm=pixel_size_nm,
            pixel_size_source=pixel_size_source,
            magnification=magnification,
        )
        key = image_key(record)
        if key in records:
            raise ValueError(f"Duplicate patient image record: {key}")
        records[key] = record
    return records


def patient_columns(record: ImageRecord, compartment: str | None = None) -> dict[str, str]:
    """Build patient and DMD1X compatibility columns for an output row.

    Args:
        record (ImageRecord): Source image record.
        compartment (str | None): Optional replacement compartment label.

    Returns:
        dict[str, str]: Patient and compatibility field values.
    """
    return {
        "Condition": record.condition,
        "IDENTIFIER": record.identifier,
        "GENOTYPE": record.genotype,
        "DOB": record.dob,
        "AGE": record.age,
        "GENDER": record.gender,
        "SITE OF BIOPSY": record.biopsy_site,
        "Notes": record.notes,
        "Sample_ID": record.sample_id,
        "Muscle": record.biopsy_site,
        "Block": record.identifier,
        "image": record.image_label,
        "Compartment": compartment or record.compartment,
    }


def get_first_value(
    row: dict[str, str], keys: list[str], required: bool = True
) -> str | None:
    """Return the first non-empty value among candidate CSV columns.

    Args:
        row (dict[str, str]): Input CSV row.
        keys (list[str]): Candidate column names.
        required (bool): Whether absence should raise an exception.

    Returns:
        str | None: First non-empty value or ``None`` when optional.
    """
    for key in keys:
        if key in row and row[key] not in ("", None):
            return row[key]
    if required:
        raise KeyError(f"Missing required columns {keys}; available={list(row)}")
    return None


def optional_float(value: str | None) -> float | None:
    """Parse an optional floating-point CSV value.

    Args:
        value (str | None): Text to parse.

    Returns:
        float | None: Parsed finite value, or ``None`` when blank.
    """
    if value is None or value.strip() == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def parse_centroid(text: str) -> tuple[float, float]:
    """Parse ``(x, y)`` centroid text.

    Args:
        text (str): Centroid text.

    Returns:
        tuple[float, float]: X and Y coordinates.
    """
    parts = text.strip().lstrip("(").rstrip(")").split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid centroid: {text}")
    return float(parts[0].strip()), float(parts[1].strip())


def classify_image_region(
    centroid_x_px: float,
    centroid_y_px: float,
    width_px: int,
    height_px: int,
) -> str:
    """Classify a centroid inside or outside the centered 50%-area square.

    Args:
        centroid_x_px (float): X coordinate in pixels.
        centroid_y_px (float): Y coordinate in pixels.
        width_px (int): Image width in pixels.
        height_px (int): Image height in pixels.

    Returns:
        str: ``center`` or ``periphery``.
    """
    side = min(math.sqrt(0.5 * width_px * height_px), width_px, height_px)
    center_x = (width_px - 1.0) / 2.0
    center_y = (height_px - 1.0) / 2.0
    return (
        "center"
        if abs(centroid_x_px - center_x) <= side / 2.0
        and abs(centroid_y_px - center_y) <= side / 2.0
        else "periphery"
    )


def eccentricity(major_axis: float | None, minor_axis: float | None) -> float | None:
    """Calculate ellipse eccentricity from major and minor axis lengths.

    Args:
        major_axis (float | None): Major axis length in any consistent unit.
        minor_axis (float | None): Minor axis length in the same unit.

    Returns:
        float | None: Eccentricity clamped to ``[0, 1]``, or ``None``.
    """
    if major_axis is None or minor_axis is None or major_axis <= 0.0:
        return None
    value = math.sqrt(max(0.0, 1.0 - (minor_axis / major_axis) ** 2))
    return min(1.0, max(0.0, value))


def scaled_text(value: float | None, factor: float, digits: int = 6) -> str:
    """Scale and format an optional numeric measurement.

    Args:
        value (float | None): Numeric value in pixels or square pixels.
        factor (float): Physical-unit scale factor.
        digits (int): Decimal places to write.

    Returns:
        str: Formatted scaled value, or empty text.
    """
    return "" if value is None else f"{value * factor:.{digits}f}"


def build_output_row(row: dict[str, str], record: ImageRecord) -> dict[str, str]:
    """Convert one raw instance row to calibrated patient output.

    Args:
        row (dict[str, str]): Raw per-image metrics row.
        record (ImageRecord): Source image metadata and calibration.

    Returns:
        dict[str, str]: Calibrated instance output row.
    """
    centroid = get_first_value(row, ["Centroid", "centroid"])
    if centroid is None:
        raise ValueError("Centroid unexpectedly missing.")
    cx_px, cy_px = parse_centroid(centroid)
    major_px = optional_float(
        get_first_value(row, ["Major_axis_length"], required=False)
    )
    minor_px = optional_float(
        get_first_value(row, ["Minor_axis_length"], required=False)
    )
    area_px2 = optional_float(get_first_value(row, ["Area", "area"]))
    if area_px2 is None:
        raise ValueError("Area unexpectedly missing.")
    corrected_area_px2 = optional_float(
        get_first_value(row, ["Corrected_area", "corrected_area"], required=False)
    )
    min_feret_px = optional_float(
        get_first_value(
            row,
            [
                "Minimum_Feret_Diameter",
                "minimum_feret_diameter",
                "Minimum Feret's Diameter",
            ],
            required=False,
        )
    )
    nnd_px = optional_float(
        get_first_value(
            row, ["NND", "Nearest Neighbor Distance (NND)"], required=False
        )
    )
    third_nnd_px = optional_float(get_first_value(row, ["3NND"], required=False))
    fifth_nnd_px = optional_float(get_first_value(row, ["5NND"], required=False))
    voronoi_px2 = optional_float(
        get_first_value(row, ["Voronoi_Cell_Area"], required=False)
    )
    eccentricity_value = eccentricity(major_px, minor_px)
    pixel_area_nm2 = record.pixel_size_nm**2
    return {
        **patient_columns(record),
        "Zoom": f"{record.magnification:.6f}"
        if record.magnification is not None
        else "",
        "Pixel_size_nm": f"{record.pixel_size_nm:.8f}",
        "Pixel_size_source": record.pixel_size_source,
        "Id": get_first_value(row, ["Id", "id"]) or "",
        "Centroid": (
            f"({cx_px * record.pixel_size_nm:.6f}, "
            f"{cy_px * record.pixel_size_nm:.6f})"
        ),
        "Image_Region": classify_image_region(
            cx_px, cy_px, record.width_px, record.height_px
        ),
        "Area": scaled_text(area_px2, pixel_area_nm2, 8),
        "Corrected_area": scaled_text(corrected_area_px2, pixel_area_nm2, 8),
        "Major_axis_length": scaled_text(major_px, record.pixel_size_nm),
        "Minor_axis_length": scaled_text(minor_px, record.pixel_size_nm),
        "Minimum_Feret_Diameter": scaled_text(
            min_feret_px, record.pixel_size_nm
        ),
        "Eccentricity": (
            "" if eccentricity_value is None else f"{eccentricity_value:.8f}"
        ),
        "Circularity": get_first_value(
            row, ["Circularity", "Circularity (Form Factor)"], required=False
        )
        or "",
        "Solidity": get_first_value(
            row, ["Solidity", "Solidity (Branching)"], required=False
        )
        or "",
        "NND": scaled_text(nnd_px, record.pixel_size_nm),
        "3NND": scaled_text(third_nnd_px, record.pixel_size_nm),
        "5NND": scaled_text(fifth_nnd_px, record.pixel_size_nm),
        "Voronoi_Cell_Area": scaled_text(voronoi_px2, pixel_area_nm2, 8),
        "Connected_parts": get_first_value(
            row, ["Connected_parts", "connected_parts"], required=False
        )
        or "1",
    }


INSTANCE_FIELDS = [
    *PATIENT_FIELDS,
    *COMPATIBILITY_FIELDS,
    "Zoom",
    "Pixel_size_nm",
    "Pixel_size_source",
    "Id",
    "Centroid",
    "Image_Region",
    "Area",
    "Corrected_area",
    "Major_axis_length",
    "Minor_axis_length",
    "Minimum_Feret_Diameter",
    "Eccentricity",
    "Circularity",
    "Solidity",
    "NND",
    "3NND",
    "5NND",
    "Voronoi_Cell_Area",
    "Connected_parts",
]


def write_measurements_csv(
    output_csv: Path, image_records: dict[ImageKey, ImageRecord]
) -> tuple[int, Counter[str], int]:
    """Write all calibrated patient instances.

    Args:
        output_csv (Path): Full measurements destination.
        image_records (dict[ImageKey, ImageRecord]): Source image records.

    Returns:
        tuple[int, Counter[str], int]: Written rows, pixel-source image counts,
        and images missing metrics files.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    source_counts: Counter[str] = Counter()
    rows_written = 0
    missing_metrics = 0
    with output_csv.open("w", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=INSTANCE_FIELDS)
        writer.writeheader()
        for record in sorted(image_records.values(), key=image_key):
            source_counts[record.pixel_size_source] += 1
            if not record.metrics_path.is_file():
                missing_metrics += 1
                continue
            with record.metrics_path.open(
                "r", newline="", encoding="utf-8"
            ) as metrics_handle:
                reader = csv.DictReader(metrics_handle)
                if reader.fieldnames is None:
                    raise ValueError(f"Missing CSV header: {record.metrics_path}")
                for row in reader:
                    writer.writerow(build_output_row(row, record))
                    rows_written += 1
    return rows_written, source_counts, missing_metrics


def key_from_row(row: dict[str, str]) -> ImageKey:
    """Build an image key from an output CSV row.

    Args:
        row (dict[str, str]): Consolidated output row.

    Returns:
        ImageKey: Condition, identifier, compartment, and image.
    """
    return (
        row["Condition"],
        row["IDENTIFIER"],
        row["Compartment"],
        row["image"],
    )


def segmentation_touches_edge(
    centroid_x_px: float,
    centroid_y_px: float,
    major_axis_px: float,
    width_px: int,
    height_px: int,
) -> bool:
    """Estimate whether an instance reaches an image boundary.

    Args:
        centroid_x_px (float): Centroid X coordinate in pixels.
        centroid_y_px (float): Centroid Y coordinate in pixels.
        major_axis_px (float): Major axis length in pixels.
        width_px (int): Image width in pixels.
        height_px (int): Image height in pixels.

    Returns:
        bool: Whether the major-axis bounding circle reaches an edge.
    """
    radius = major_axis_px / 2.0
    return (
        centroid_x_px - radius <= 0.0
        or centroid_x_px + radius >= width_px - 1
        or centroid_y_px - radius <= 0.0
        or centroid_y_px + radius >= height_px - 1
    )


def clean_measurements_csv(
    measurements_csv: Path,
    cleaned_csv: Path,
    image_records: dict[ImageKey, ImageRecord],
) -> tuple[int, int, int, int, int]:
    """Filter instances by axis size, image edge, and connected parts.

    Args:
        measurements_csv (Path): Unfiltered instance CSV.
        cleaned_csv (Path): Filtered instance CSV destination.
        image_records (dict[ImageKey, ImageRecord]): Source image records.

    Returns:
        tuple[int, int, int, int, int]: Kept, size-removed, edge-removed,
        disconnected-removed, and total row counts.
    """
    kept = size_removed = edge_removed = disconnected_removed = total = 0
    with measurements_csv.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {measurements_csv}")
        with cleaned_csv.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                total += 1
                record = image_records.get(key_from_row(row))
                if record is None:
                    raise KeyError(f"Missing image record for row: {key_from_row(row)}")
                major_px = float(row["Major_axis_length"]) / record.pixel_size_nm
                minor_px = float(row["Minor_axis_length"]) / record.pixel_size_nm
                if major_px <= MIN_MAJOR_AXIS_PX or minor_px <= MIN_MINOR_AXIS_PX:
                    size_removed += 1
                    continue
                cx_nm, cy_nm = parse_centroid(row["Centroid"])
                if segmentation_touches_edge(
                    cx_nm / record.pixel_size_nm,
                    cy_nm / record.pixel_size_nm,
                    major_px,
                    record.width_px,
                    record.height_px,
                ):
                    edge_removed += 1
                    continue
                if int(float(row.get("Connected_parts", "1") or "1")) > 1:
                    disconnected_removed += 1
                    continue
                writer.writerow(row)
                kept += 1
    return kept, size_removed, edge_removed, disconnected_removed, total


def numeric_values(rows: list[dict[str, str]], column: str) -> list[float]:
    """Collect finite numeric values from one column.

    Args:
        rows (list[dict[str, str]]): CSV rows to inspect.
        column (str): Numeric column name.

    Returns:
        list[float]: Parsed finite values.
    """
    values = []
    for row in rows:
        value = optional_float(row.get(column))
        if value is not None:
            values.append(value)
    return values


def mean_or_none(values: list[float]) -> float | None:
    """Calculate an arithmetic mean.

    Args:
        values (list[float]): Values to average.

    Returns:
        float | None: Mean, or ``None`` for an empty list.
    """
    return sum(values) / len(values) if values else None


def sum_or_none(values: list[float]) -> float | None:
    """Calculate a sum while preserving empty groups as missing.

    Args:
        values (list[float]): Values to sum.

    Returns:
        float | None: Sum, or ``None`` for an empty list.
    """
    return sum(values) if values else None


def coefficient_of_variation(values: list[float]) -> float | None:
    """Calculate the population coefficient of variation.

    Args:
        values (list[float]): Values to summarize.

    Returns:
        float | None: Population standard deviation divided by mean.
    """
    if len(values) < 2:
        return None
    mean = mean_or_none(values)
    if mean is None or mean == 0.0:
        return None
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


def format_optional(value: float | None, digits: int = 8) -> str:
    """Format an optional float for CSV output.

    Args:
        value (float | None): Value to format.
        digits (int): Decimal places.

    Returns:
        str: Formatted value or empty text.
    """
    return "" if value is None else f"{value:.{digits}f}"


def pairwise_distances(coords: list[tuple[float, float]]) -> list[float]:
    """Calculate unique pairwise coordinate distances.

    Args:
        coords (list[tuple[float, float]]): Coordinates in consistent units.

    Returns:
        list[float]: Unique unordered pair distances.
    """
    return [
        math.hypot(x_a - x_b, y_a - y_b)
        for index, (x_a, y_a) in enumerate(coords)
        for x_b, y_b in coords[index + 1 :]
    ]


def spatial_radii(width_nm: float, height_nm: float) -> list[float]:
    """Build radii for spatial-statistics integration.

    Args:
        width_nm (float): Physical image width.
        height_nm (float): Physical image height.

    Returns:
        list[float]: Twenty radii up to one quarter of the shorter side.
    """
    maximum = min(width_nm, height_nm) * SPATIAL_MAX_RADIUS_FRACTION
    if maximum <= 0.0:
        return []
    step = maximum / SPATIAL_RADIUS_BIN_COUNT
    return [step * index for index in range(1, SPATIAL_RADIUS_BIN_COUNT + 1)]


def trapezoid_integral(x_values: list[float], y_values: list[float]) -> float | None:
    """Integrate sampled values with the trapezoid rule.

    Args:
        x_values (list[float]): Ordered sample positions.
        y_values (list[float]): Values at each sample.

    Returns:
        float | None: Integral, or ``None`` for incompatible inputs.
    """
    if len(x_values) < 2 or len(x_values) != len(y_values):
        return None
    return sum(
        (x_values[index] - x_values[index - 1])
        * (y_values[index] + y_values[index - 1])
        / 2.0
        for index in range(1, len(x_values))
    )


def ripley_l_integral(
    coords_nm: list[tuple[float, float]], area_nm2: float, radii_nm: list[float]
) -> float | None:
    """Integrate Ripley L-function deviation from random placement.

    Args:
        coords_nm (list[tuple[float, float]]): Centroids in nanometers.
        area_nm2 (float): Source image area in square nanometers.
        radii_nm (list[float]): Evaluation radii.

    Returns:
        float | None: Integrated ``L(r) - r`` or ``None`` when unavailable.
    """
    count = len(coords_nm)
    if count < 2 or area_nm2 <= 0.0 or not radii_nm:
        return None
    distances = pairwise_distances(coords_nm)
    deviations = [0.0]
    for radius in radii_nm:
        pairs = 2.0 * sum(distance <= radius for distance in distances)
        k_value = area_nm2 * pairs / (count * (count - 1))
        deviations.append(math.sqrt(max(k_value, 0.0) / math.pi) - radius)
    return trapezoid_integral([0.0, *radii_nm], deviations)


def pair_correlation_integral(
    coords_nm: list[tuple[float, float]], area_nm2: float, radii_nm: list[float]
) -> float | None:
    """Integrate pair-correlation deviation from random placement.

    Args:
        coords_nm (list[tuple[float, float]]): Centroids in nanometers.
        area_nm2 (float): Source image area in square nanometers.
        radii_nm (list[float]): Annular outer radii.

    Returns:
        float | None: Integrated ``g(r) - 1`` or ``None`` when unavailable.
    """
    count = len(coords_nm)
    if count < 2 or area_nm2 <= 0.0 or not radii_nm:
        return None
    distances = pairwise_distances(coords_nm)
    previous = total = 0.0
    for radius in radii_nm:
        annulus_area = math.pi * (radius**2 - previous**2)
        pairs = 2.0 * sum(previous < distance <= radius for distance in distances)
        g_value = area_nm2 * pairs / (count * (count - 1) * annulus_area)
        total += (g_value - 1.0) * (radius - previous)
        previous = radius
    return total


SUMMARY_METRIC_FIELDS = [
    "Density",
    "Instance_count",
    "Zoom",
    "Image_width_px",
    "Image_height_px",
    "Pixel_size_nm",
    "Pixel_size_source",
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
    "Eccentricity_mean",
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
SUMMARY_FIELDS = [*PATIENT_FIELDS, *COMPATIBILITY_FIELDS, *SUMMARY_METRIC_FIELDS]


def summarize_rows(
    rows: list[dict[str, str]],
    record: ImageRecord,
    compartment: str | None = None,
) -> dict[str, str]:
    """Build one source-image summary, including empty-image summaries.

    Args:
        rows (list[dict[str, str]]): Cleaned instances for one source image.
        record (ImageRecord): Source metadata and calibration.
        compartment (str | None): Optional output compartment replacement.

    Returns:
        dict[str, str]: Complete CSV-ready image summary.
    """
    width_nm = record.width_px * record.pixel_size_nm
    height_nm = record.height_px * record.pixel_size_nm
    image_area_nm2 = width_nm * height_nm
    center_rows = [
        row for row in rows if row.get("Image_Region", "").lower() == "center"
    ]
    coords = [parse_centroid(row["Centroid"]) for row in rows]
    radii = spatial_radii(width_nm, height_nm)
    return {
        **patient_columns(record, compartment),
        "Density": str(len(rows)),
        "Instance_count": str(len(rows)),
        "Zoom": format_optional(record.magnification, 6),
        "Image_width_px": str(record.width_px),
        "Image_height_px": str(record.height_px),
        "Pixel_size_nm": format_optional(record.pixel_size_nm, 8),
        "Pixel_size_source": record.pixel_size_source,
        "Image_area_nm2": format_optional(image_area_nm2, 8),
        "Area_sum": format_optional(sum_or_none(numeric_values(rows, "Area"))),
        "Corrected_area_sum": format_optional(
            sum_or_none(numeric_values(rows, "Corrected_area"))
        ),
        "Minimum_Feret_Diameter_sum": format_optional(
            sum_or_none(numeric_values(rows, "Minimum_Feret_Diameter")), 6
        ),
        "Minor_axis_length_sum": format_optional(
            sum_or_none(numeric_values(rows, "Minor_axis_length")), 6
        ),
        "Area_mean": format_optional(mean_or_none(numeric_values(rows, "Area"))),
        "Corrected_area_mean": format_optional(
            mean_or_none(numeric_values(rows, "Corrected_area"))
        ),
        "Minimum_Feret_Diameter_mean": format_optional(
            mean_or_none(numeric_values(rows, "Minimum_Feret_Diameter")), 6
        ),
        "Major_axis_length_mean": format_optional(
            mean_or_none(numeric_values(rows, "Major_axis_length")), 6
        ),
        "Minor_axis_length_mean": format_optional(
            mean_or_none(numeric_values(rows, "Minor_axis_length")), 6
        ),
        "Eccentricity_mean": format_optional(
            mean_or_none(numeric_values(rows, "Eccentricity")), 8
        ),
        "Circularity_mean": format_optional(
            mean_or_none(numeric_values(rows, "Circularity")), 6
        ),
        "Solidity_mean": format_optional(
            mean_or_none(numeric_values(rows, "Solidity")), 6
        ),
        "NND_center_mean": format_optional(
            mean_or_none(numeric_values(center_rows, "NND")), 6
        ),
        "3NND_center_mean": format_optional(
            mean_or_none(numeric_values(center_rows, "3NND")), 6
        ),
        "5NND_center_mean": format_optional(
            mean_or_none(numeric_values(center_rows, "5NND")), 6
        ),
        "Voronoi_Cell_Area_center_mean": format_optional(
            mean_or_none(numeric_values(center_rows, "Voronoi_Cell_Area"))
        ),
        "Voronoi_Cell_Area_center_cv": format_optional(
            coefficient_of_variation(
                numeric_values(center_rows, "Voronoi_Cell_Area")
            )
        ),
        "Ripley_L_integral": format_optional(
            ripley_l_integral(coords, image_area_nm2, radii)
        ),
        "Pair_Correlation_integral": format_optional(
            pair_correlation_integral(coords, image_area_nm2, radii)
        ),
    }


def read_grouped_clean_rows(
    cleaned_csv: Path,
) -> dict[ImageKey, list[dict[str, str]]]:
    """Group cleaned rows by distinct source image.

    Args:
        cleaned_csv (Path): Cleaned instance CSV.

    Returns:
        dict[ImageKey, list[dict[str, str]]]: Rows grouped by image key.
    """
    grouped: dict[ImageKey, list[dict[str, str]]] = {}
    with cleaned_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {cleaned_csv}")
        for row in reader:
            grouped.setdefault(key_from_row(row), []).append(row)
    return grouped


def write_summary_csv(
    cleaned_csv: Path,
    summary_csv: Path,
    image_records: dict[ImageKey, ImageRecord],
    source_compartment: str | None = None,
    output_compartment: str | None = None,
) -> int:
    """Write one summary per source image, including images with zero instances.

    Args:
        cleaned_csv (Path): Cleaned instance CSV.
        summary_csv (Path): Summary destination.
        image_records (dict[ImageKey, ImageRecord]): All source image records.
        source_compartment (str | None): Optional source compartment filter.
        output_compartment (str | None): Optional output compartment label.

    Returns:
        int: Number of summary rows written.
    """
    grouped = read_grouped_clean_rows(cleaned_csv)
    rows_written = 0
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for record in sorted(image_records.values(), key=image_key):
            if (
                source_compartment is not None
                and record.compartment != source_compartment
            ):
                continue
            writer.writerow(
                summarize_rows(
                    grouped.get(image_key(record), []), record, output_compartment
                )
            )
            rows_written += 1
    return rows_written


def write_pooled_measurements(
    cleaned_csv: Path,
    pooled_csv: Path,
    compartment: str = ALL_COMPARTMENTS_LABEL,
) -> int:
    """Copy cleaned rows while relabeling, not merging, source compartments.

    Args:
        cleaned_csv (Path): Cleaned source instance CSV.
        pooled_csv (Path): Relabeled instance CSV destination.
        compartment (str): Pooled compartment label.

    Returns:
        int: Number of copied instance rows.
    """
    rows_written = 0
    with cleaned_csv.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {cleaned_csv}")
        with pooled_csv.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                row["Compartment"] = compartment
                writer.writerow(row)
                rows_written += 1
    return rows_written


def main() -> None:
    """Build all Calpain-3 patient measurement and summary CSV outputs.

    Args:
        None: This function accepts no arguments.

    Returns:
        None: Outputs are written under the patient results directory.
    """
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Input root not found: {INPUT_ROOT}")
    records = load_image_records(INPUT_ROOT)
    if not records:
        raise ValueError(f"No patient image JSON records found under: {INPUT_ROOT}")
    missing_metrics_paths = [
        record.metrics_path
        for record in records.values()
        if not record.metrics_path.is_file()
    ]
    if missing_metrics_paths:
        examples = ", ".join(str(path) for path in missing_metrics_paths[:5])
        raise FileNotFoundError(
            "Segmentation metrics are missing for "
            f"{len(missing_metrics_paths)} image(s). Examples: {examples}"
        )
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    measurement_rows, pixel_sources, missing_metrics = write_measurements_csv(
        OUTPUT_CSV, records
    )
    cleaning = clean_measurements_csv(OUTPUT_CSV, OUTPUT_CLEANED_CSV, records)
    ss_rows = write_summary_csv(
        OUTPUT_CLEANED_CSV,
        OUTPUT_SS_SUMMARY_CSV,
        records,
        source_compartment=SS_LABEL,
    )
    imf_rows = write_summary_csv(
        OUTPUT_CLEANED_CSV,
        OUTPUT_IMF_SUMMARY_CSV,
        records,
        source_compartment=IMF_LABEL,
    )
    pooled_rows = write_pooled_measurements(
        OUTPUT_CLEANED_CSV, OUTPUT_NO_COMPARTMENT_CSV
    )
    pooled_summary_rows = write_summary_csv(
        OUTPUT_CLEANED_CSV,
        OUTPUT_NO_COMPARTMENT_SUMMARY_CSV,
        records,
        output_compartment=ALL_COMPARTMENTS_LABEL,
    )
    kept, size_removed, edge_removed, disconnected_removed, total = cleaning
    print(f"Loaded patient image records: {len(records)}")
    print(f"Wrote measurements: {OUTPUT_CSV} ({measurement_rows} rows)")
    print(f"Wrote cleaned measurements: {OUTPUT_CLEANED_CSV} ({kept} rows)")
    print(
        "Cleaning summary: "
        f"total={total}, kept={kept}, axis_size={size_removed}, "
        f"image_edge={edge_removed}, disconnected_parts={disconnected_removed}"
    )
    print(f"Wrote SS summaries: {OUTPUT_SS_SUMMARY_CSV} ({ss_rows} rows)")
    print(f"Wrote IMF summaries: {OUTPUT_IMF_SUMMARY_CSV} ({imf_rows} rows)")
    print(
        f"Wrote pooled measurements: {OUTPUT_NO_COMPARTMENT_CSV} "
        f"({pooled_rows} rows)"
    )
    print(
        f"Wrote pooled summaries: {OUTPUT_NO_COMPARTMENT_SUMMARY_CSV} "
        f"({pooled_summary_rows} rows)"
    )
    print(f"Images without metrics CSVs: {missing_metrics}")
    print(
        "Pixel-size sources: "
        + ", ".join(
            f"{source}={count}" for source, count in sorted(pixel_sources.items())
        )
    )


if __name__ == "__main__":
    main()
