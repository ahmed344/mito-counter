#!/usr/bin/env python3
"""Convert Calpain-3 patient DM3 images to TIFF with clinical metadata."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import hyperspy.api as hs
import numpy as np
import tifffile as tif


DEFAULT_INPUT_ROOT = "/workspaces/mito-counter/data/Calpaine_3_patients/RAW"
DEFAULT_OUTPUT_ROOT = "/workspaces/mito-counter/data/Calpaine_3_patients/Processed"
DEFAULT_MAGNIFICATION = "4800X"
CONDITIONS = ("CAPN3", "CTRL")
QUANTIFICATION_DIRECTORIES = {
    "DATA QUANTIFICATION - IMF": "IMF",
    "DATA QUANTIFICATION - SS": "SS",
}
CLINICAL_FIELDS = (
    "IDENTIFIER",
    "GENOTYPE",
    "DOB",
    "AGE",
    "GENDER",
    "SITE OF BIOPSY",
    "Notes",
)
CSV_FIELDS = {
    "CAPN3": ("IDENTIFIER", "GENOTYPE", "DOB", "AGE", "GENDER", "SITE OF BIOPSY"),
    "CTRL": ("IDENTIFIER", "DOB", "AGE", "GENDER", "SITE OF BIOPSY", "Notes"),
}
IDENTIFIER_RE = re.compile(r"(?<![A-Za-z0-9])P\d{6}(?!\d)", re.IGNORECASE)
SAMPLE_ID_RE = re.compile(r"(?<![A-Za-z0-9])M\d{7}(?!\d)", re.IGNORECASE)
UNIT_TO_NM = {
    "nm": 1.0,
    "nanometer": 1.0,
    "nanometers": 1.0,
    "um": 1000.0,
    "micrometer": 1000.0,
    "micrometers": 1000.0,
    "pm": 0.001,
    "angstrom": 0.1,
    "angstroms": 0.1,
}


@dataclass(frozen=True)
class PatientLabels:
    """Labels parsed from a Calpain-3 patient DM3 path.

    Args:
        condition (str): Clinical group, either ``CAPN3`` or ``CTRL``.
        patient_folder (str): Source patient directory name.
        patient_id (str): Patient identifier in ``P######`` format.
        sample_id (str): Sample identifier in ``M#######`` format.
        compartment (str): Quantified compartment, ``IMF`` or ``SS``.
        quantification_directory (str): Exact source quantification directory name.

    Returns:
        None: Dataclass instances provide immutable structured labels.
    """

    condition: str
    patient_folder: str
    patient_id: str
    sample_id: str
    compartment: str
    quantification_directory: str


def _parse_patient_folder(patient_folder: str) -> tuple[str, str]:
    """Parse patient and sample identifiers from a patient folder name.

    Args:
        patient_folder (str): Folder containing one ``P######`` and one
            ``M#######`` token.

    Returns:
        tuple[str, str]: Uppercase ``(patient_id, sample_id)`` values.
    """
    patient_ids = IDENTIFIER_RE.findall(patient_folder)
    sample_ids = SAMPLE_ID_RE.findall(patient_folder)
    if len(patient_ids) != 1 or len(sample_ids) != 1:
        raise ValueError(
            "Expected exactly one P###### identifier and one M####### sample ID "
            f"in patient folder: {patient_folder!r}"
        )
    return patient_ids[0].upper(), sample_ids[0].upper()


def _is_excluded_path(relative_parts: Sequence[str]) -> bool:
    """Determine whether path segments identify ME data or preview content.

    Args:
        relative_parts (Sequence[str]): Path components relative to the RAW root.

    Returns:
        bool: ``True`` when any component is ME DATA or preview-related.
    """
    for part in relative_parts:
        normalized = re.sub(r"[\s_-]+", " ", part.strip()).casefold()
        if normalized == "me data" or "preview" in normalized:
            return True
    return False


def find_dm3_files(root: str, magnification: str | None) -> list[str]:
    """Discover eligible DM3 files in exact quantification directories.

    Args:
        root (str): Calpain-3 patient RAW root to scan recursively.
        magnification (str | None): Case-insensitive filename token required for
            inclusion, or ``None`` to disable magnification filtering.

    Returns:
        list[str]: Sorted absolute paths to matching DM3 files.
    """
    root = os.path.abspath(root)
    paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        relative_directory = os.path.relpath(dirpath, root)
        relative_parts = () if relative_directory == "." else tuple(relative_directory.split(os.sep))
        dirnames[:] = [
            name for name in dirnames if not _is_excluded_path((*relative_parts, name))
        ]
        if os.path.basename(dirpath) not in QUANTIFICATION_DIRECTORIES:
            continue
        if _is_excluded_path(relative_parts):
            continue
        for filename in filenames:
            if not filename.lower().endswith(".dm3"):
                continue
            if magnification and magnification.casefold() not in filename.casefold():
                continue
            paths.append(os.path.abspath(os.path.join(dirpath, filename)))
        dirnames[:] = []
    return sorted(paths)


def extract_labels(input_root: str, dm3_path: str) -> PatientLabels:
    """Extract patient labels from an eligible source path.

    Args:
        input_root (str): RAW root used to interpret relative path segments.
        dm3_path (str): Source DM3 image path.

    Returns:
        PatientLabels: Validated labels parsed from the source path.
    """
    relative_path = os.path.relpath(os.path.abspath(dm3_path), os.path.abspath(input_root))
    parts = relative_path.split(os.sep)
    if len(parts) < 4 or relative_path == os.pardir or relative_path.startswith(os.pardir + os.sep):
        raise ValueError(
            "Expected path layout <condition>/<patient folder>/.../"
            f"<quantification directory>/<image.dm3>: {dm3_path}"
        )

    condition = parts[0].upper()
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition directory {parts[0]!r}: {dm3_path}")
    if _is_excluded_path(parts):
        raise ValueError(f"Excluded ME DATA/preview path: {dm3_path}")

    quantification_directory = parts[-2]
    if quantification_directory not in QUANTIFICATION_DIRECTORIES:
        raise ValueError(
            "DM3 immediate parent is not an exact quantification directory: "
            f"{dm3_path}"
        )
    patient_id, sample_id = _parse_patient_folder(parts[1])
    return PatientLabels(
        condition=condition,
        patient_folder=parts[1],
        patient_id=patient_id,
        sample_id=sample_id,
        compartment=QUANTIFICATION_DIRECTORIES[quantification_directory],
        quantification_directory=quantification_directory,
    )


def _validate_csv_headers(condition: str, fieldnames: Sequence[str] | None, csv_path: str) -> None:
    """Validate that a clinical CSV has its exact group-specific schema.

    Args:
        condition (str): Clinical group associated with the CSV.
        fieldnames (Sequence[str] | None): Header values read by ``csv.DictReader``.
        csv_path (str): CSV path used in validation errors.

    Returns:
        None: The function returns after successful validation.
    """
    expected = CSV_FIELDS[condition]
    actual = tuple(fieldnames or ())
    if actual != expected:
        raise ValueError(
            f"Invalid headers in {csv_path}; expected {expected!r}, found {actual!r}"
        )


def load_clinical_metadata(input_root: str) -> dict[str, dict[str, str]]:
    """Load and strictly validate CAPN3 and CTRL clinical CSV files.

    Missing group-specific fields are represented by blank strings. Identifiers
    must be unique across both files.

    Args:
        input_root (str): RAW root containing ``CAPN3.csv`` and ``CTRL.csv``.

    Returns:
        dict[str, dict[str, str]]: Clinical records keyed by patient identifier.
    """
    records: dict[str, dict[str, str]] = {}
    for condition in CONDITIONS:
        csv_path = os.path.join(input_root, f"{condition}.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Missing required clinical CSV: {csv_path}")
        with open(csv_path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            _validate_csv_headers(condition, reader.fieldnames, csv_path)
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(f"Extra CSV values at {csv_path}:{line_number}")
                if any(value is None for value in row.values()):
                    raise ValueError(f"Missing CSV values at {csv_path}:{line_number}")
                if not any(value.strip() for value in row.values()):
                    raise ValueError(f"Blank CSV row at {csv_path}:{line_number}")
                blank_fields = [
                    field for field in CSV_FIELDS[condition] if not row[field].strip()
                ]
                if blank_fields:
                    raise ValueError(
                        f"Blank required fields {blank_fields!r} at "
                        f"{csv_path}:{line_number}"
                    )

                identifier = row["IDENTIFIER"].strip().upper()
                if IDENTIFIER_RE.fullmatch(identifier) is None:
                    raise ValueError(
                        f"Invalid IDENTIFIER {row['IDENTIFIER']!r} at "
                        f"{csv_path}:{line_number}"
                    )
                if identifier in records:
                    previous = records[identifier]["_csv_source"]
                    raise ValueError(
                        f"Duplicate identifier {identifier} in {csv_path}:{line_number}; "
                        f"first defined in {previous}"
                    )

                normalized = {field: "" for field in CLINICAL_FIELDS}
                for field in CSV_FIELDS[condition]:
                    normalized[field] = row[field].strip()
                normalized["IDENTIFIER"] = identifier
                normalized["_condition"] = condition
                normalized["_csv_source"] = f"{csv_path}:{line_number}"
                records[identifier] = normalized
    return records


def inventory_patient_folders(
    input_root: str,
) -> tuple[dict[tuple[str, str], str], list[str]]:
    """Inventory direct patient folders and collect malformed-folder failures.

    Args:
        input_root (str): RAW root containing condition directories.

    Returns:
        tuple[dict[tuple[str, str], str], list[str]]: Folder paths keyed by
            ``(condition, patient_id)`` and validation failure messages.
    """
    folders: dict[tuple[str, str], str] = {}
    failures: list[str] = []
    for condition in CONDITIONS:
        condition_root = os.path.join(input_root, condition)
        if not os.path.isdir(condition_root):
            failures.append(f"Missing condition directory: {condition_root}")
            continue
        with os.scandir(condition_root) as entries:
            for entry in sorted(entries, key=lambda item: item.name.casefold()):
                if not entry.is_dir():
                    continue
                try:
                    patient_id, _ = _parse_patient_folder(entry.name)
                except ValueError as exc:
                    failures.append(str(exc))
                    continue
                key = (condition, patient_id)
                if key in folders:
                    failures.append(
                        f"Duplicate patient folders for {condition}/{patient_id}: "
                        f"{folders[key]} and {entry.path}"
                    )
                    continue
                folders[key] = os.path.abspath(entry.path)
    return folders, failures


def compare_metadata_and_folders(
    records: dict[str, dict[str, str]],
    folders: dict[tuple[str, str], str],
) -> tuple[list[str], list[str]]:
    """Compare clinical CSV records with condition patient directories.

    Args:
        records (dict[str, dict[str, str]]): Clinical records keyed by identifier.
        folders (dict[tuple[str, str], str]): Patient folders keyed by condition
            and identifier.

    Returns:
        tuple[list[str], list[str]]: CSV patients lacking folders and folders
            lacking matching CSV records.
    """
    csv_keys = {(record["_condition"], identifier) for identifier, record in records.items()}
    folder_keys = set(folders)
    missing_folders = [
        f"{condition}/{identifier}" for condition, identifier in sorted(csv_keys - folder_keys)
    ]
    missing_records = [
        f"{condition}/{identifier}: {folders[(condition, identifier)]}"
        for condition, identifier in sorted(folder_keys - csv_keys)
    ]
    return missing_folders, missing_records


def _normalize_stem(value: str) -> str:
    """Normalize a source image stem for use in an output filename.

    Args:
        value (str): Original DM3 filename stem.

    Returns:
        str: Trimmed stem with whitespace collapsed to underscores.
    """
    normalized = re.sub(r"\s+", "_", value.strip())
    if not normalized:
        raise ValueError("Cannot normalize an empty DM3 filename stem")
    return normalized


def build_output_paths(
    output_root: str, labels: PatientLabels, dm3_path: str
) -> tuple[str, str]:
    """Build TIFF and JSON destinations for one patient image.

    Args:
        output_root (str): Processed output root.
        labels (PatientLabels): Labels parsed from the source path.
        dm3_path (str): Source DM3 path used to derive the filename stem.

    Returns:
        tuple[str, str]: TIFF and JSON output paths.
    """
    stem = _normalize_stem(os.path.splitext(os.path.basename(dm3_path))[0])
    output_directory = os.path.join(
        output_root, labels.condition, labels.patient_id, labels.compartment
    )
    return (
        os.path.join(output_directory, f"{stem}.tif"),
        os.path.join(output_directory, f"{stem}.json"),
    )


def _jsonify(value: Any) -> Any:
    """Convert nested HyperSpy metadata into JSON-compatible values.

    Args:
        value (Any): Arbitrary metadata value.

    Returns:
        Any: JSON-serializable representation of the input value.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return repr(value)


def _axis_calibrations(signal: Any) -> tuple[list[dict[str, Any]], float | None]:
    """Extract spatial-axis calibration and an isotropic pixel size.

    Args:
        signal (Any): HyperSpy signal containing an axes manager.

    Returns:
        tuple[list[dict[str, Any]], float | None]: JSON-ready signal-axis
            calibration records and nanometers per pixel when all spatial axes
            have a consistent recognized calibration.
    """
    calibrations: list[dict[str, Any]] = []
    pixel_sizes_nm: list[float] = []
    for axis in signal.axes_manager.signal_axes:
        units_text = "" if axis.units is None else str(axis.units)
        normalized_unit = (
            units_text.strip().lower().replace("µ", "u").replace("μ", "u")
        )
        scale = float(axis.scale)
        factor = UNIT_TO_NM.get(normalized_unit)
        pixel_size_nm = scale * factor if factor is not None and scale > 0.0 else None
        calibrations.append(
            {
                "name": str(axis.name),
                "size": int(axis.size),
                "scale": scale,
                "offset": float(axis.offset),
                "units": units_text,
                "pixel_size_nm": pixel_size_nm,
            }
        )
        if pixel_size_nm is not None:
            pixel_sizes_nm.append(pixel_size_nm)

    if len(pixel_sizes_nm) != len(calibrations) or not pixel_sizes_nm:
        return calibrations, None
    reference = pixel_sizes_nm[0]
    if not all(
        math.isclose(value, reference, rel_tol=1e-3, abs_tol=1e-9)
        for value in pixel_sizes_nm[1:]
    ):
        return calibrations, None
    return calibrations, reference


def _prepare_for_tiff(data: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert source pixels to 8-bit with per-image linear scaling.

    Args:
        data (np.ndarray): Source image array loaded by HyperSpy.

    Returns:
        tuple[np.ndarray, dict[str, Any]]: Eight-bit pixels and per-image
            conversion metadata.
    """
    array = np.asarray(data)
    if array.size == 0:
        raise ValueError("DM3 image data is empty")
    if array.dtype == np.dtype(np.uint8):
        saved = array
        method = "preserve_native_uint8"
        mapping_min = 0.0
        mapping_max = 255.0
    elif np.issubdtype(array.dtype, np.number):
        if not np.isfinite(array).all():
            raise ValueError("DM3 image contains non-finite pixel values")
        mapping_min = float(np.min(array))
        mapping_max = float(np.max(array))
        if mapping_max == mapping_min:
            saved = np.zeros(array.shape, dtype=np.uint8)
            method = "constant_image_to_zero_uint8"
        else:
            normalized = (array.astype(np.float64) - mapping_min) / (
                mapping_max - mapping_min
            )
            saved = np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
            method = "per_image_minmax_to_uint8"
    else:
        raise TypeError(f"Unsupported DM3 pixel dtype: {array.dtype}")

    conversion: dict[str, Any] = {
        "method": method,
        "original_dtype": str(array.dtype),
        "original_shape": list(array.shape),
        "mapping_min": mapping_min,
        "mapping_max": mapping_max,
        "saved_dtype": "uint8",
    }
    return saved, conversion


def export_dm3(
    dm3_path: str,
    input_root: str,
    output_root: str,
    records: dict[str, dict[str, str]],
    dry_run: bool = False,
) -> tuple[str, str]:
    """Export one patient DM3 image and its complete metadata.

    Args:
        dm3_path (str): Absolute source DM3 path.
        input_root (str): RAW root used for path label extraction.
        output_root (str): Processed output root.
        records (dict[str, dict[str, str]]): Validated clinical metadata records.
        dry_run (bool): If ``True``, validate and return paths without writing.

    Returns:
        tuple[str, str]: TIFF and JSON output paths.
    """
    labels = extract_labels(input_root, dm3_path)
    record = records.get(labels.patient_id)
    if record is None:
        raise ValueError(f"No clinical CSV record for {labels.condition}/{labels.patient_id}")
    if record["_condition"] != labels.condition:
        raise ValueError(
            f"Clinical condition mismatch for {labels.patient_id}: path is "
            f"{labels.condition}, CSV is {record['_condition']}"
        )
    out_tiff, out_json = build_output_paths(output_root, labels, dm3_path)
    if dry_run:
        return out_tiff, out_json

    signal = hs.load(dm3_path)
    if isinstance(signal, (list, tuple)):
        if len(signal) != 1:
            raise ValueError(f"Expected one HyperSpy signal, found {len(signal)}: {dm3_path}")
        signal = signal[0]
    source_data = np.asarray(signal.data)
    saved_data, conversion = _prepare_for_tiff(source_data)
    axis_calibrations, pixel_size_nm = _axis_calibrations(signal)
    clinical = {field: record[field] for field in CLINICAL_FIELDS}
    basic = {
        "source_path": os.path.abspath(dm3_path),
        "shape": list(source_data.shape),
        "dtype": str(source_data.dtype),
        "saved_shape": list(saved_data.shape),
        "saved_dtype": str(saved_data.dtype),
        "condition": labels.condition,
        "patient_folder": labels.patient_folder,
        "patient_id": labels.patient_id,
        "sample_id": labels.sample_id,
        "compartment": labels.compartment,
        "quantification_directory": labels.quantification_directory,
        "axis_calibrations": axis_calibrations,
        "pixel_size_nm": pixel_size_nm,
        **clinical,
    }
    metadata = {
        "basic": basic,
        "conversion": conversion,
        "hyperspy_metadata": _jsonify(signal.metadata.as_dictionary()),
        "hyperspy_original_metadata": _jsonify(signal.original_metadata.as_dictionary()),
    }

    os.makedirs(os.path.dirname(out_tiff), exist_ok=True)
    tif.imwrite(
        out_tiff,
        saved_data,
        description=json.dumps(basic, ensure_ascii=True),
        photometric="minisblack",
    )
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)
    return out_tiff, out_json


def _print_report(title: str, items: Sequence[str]) -> None:
    """Print a titled report section and its entries.

    Args:
        title (str): Human-readable report heading.
        items (Sequence[str]): Report entries to print.

    Returns:
        None: Output is written to standard output.
    """
    print(f"{title}: {len(items)}")
    for item in items:
        print(f"  - {item}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse converter command-line options.

    Args:
        argv (Sequence[str] | None): Arguments to parse, or ``None`` to use
            process command-line arguments.

    Returns:
        argparse.Namespace: Validated command-line options.
    """
    parser = argparse.ArgumentParser(
        description="Convert Calpain-3 patient quantification DM3 files to TIFF."
    )
    parser.add_argument(
        "--input",
        "--input-root",
        dest="input_root",
        default=DEFAULT_INPUT_ROOT,
        help="RAW input root containing CAPN3/ and CTRL/.",
    )
    parser.add_argument(
        "--output",
        "--output-root",
        dest="output_root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Processed output root.",
    )
    parser.add_argument(
        "--magnification",
        "--magnification-token",
        dest="magnification",
        default=DEFAULT_MAGNIFICATION,
        help="Case-insensitive filename token; pass an empty string to disable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate metadata and print destinations without loading or writing images.",
    )
    args = parser.parse_args(argv)
    args.input_root = os.path.abspath(os.path.expanduser(args.input_root))
    args.output_root = os.path.abspath(os.path.expanduser(args.output_root))
    args.magnification = args.magnification.strip() or None
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Calpain-3 patient DM3 conversion pipeline.

    Args:
        argv (Sequence[str] | None): CLI arguments, or ``None`` for process
            command-line arguments.

    Returns:
        int: Process exit code; zero indicates no validation or export failures.
    """
    args = parse_args(argv)
    try:
        records = load_clinical_metadata(args.input_root)
    except Exception as exc:
        print(f"[FATAL] Clinical metadata validation failed: {exc}")
        return 1

    folders, inventory_failures = inventory_patient_folders(args.input_root)
    missing_folders, missing_records = compare_metadata_and_folders(records, folders)
    dm3_files = find_dm3_files(args.input_root, args.magnification)

    _print_report("CSV patients lacking folders", missing_folders)
    _print_report("Folders lacking CSV records", missing_records)
    _print_report("Patient-folder validation failures", inventory_failures)
    print(f"Eligible DM3 files: {len(dm3_files)}")

    warning_filters = (
        (Warning, r".*ensure_directory has been moved.*"),
        (Warning, r".*overwrite has been moved.*"),
        (Warning, r".*get_file_handle has been moved.*"),
        (Warning, r".*append2pathname has been moved.*"),
        (Warning, r".*incremental_filename has been moved.*"),
        (Warning, r".*rgb_tools.*renamed.*"),
    )
    for category, pattern in warning_filters:
        warnings.filterwarnings("ignore", category=category, message=pattern)

    failures = list(inventory_failures)
    for dm3_path in dm3_files:
        try:
            out_tiff, out_json = export_dm3(
                dm3_path=dm3_path,
                input_root=args.input_root,
                output_root=args.output_root,
                records=records,
                dry_run=args.dry_run,
            )
        except Exception as exc:  # pragma: no cover - resilient batch operation
            message = f"{dm3_path} -> {exc}"
            failures.append(message)
            print(f"[ERROR] {message}")
            continue
        prefix = "[DRY RUN]" if args.dry_run else "[EXPORTED]"
        print(f"{prefix} {dm3_path} -> {out_tiff} (+ {out_json})")

    _print_report("Failures", failures)
    # Clinical rows without image folders are expected for this incomplete
    # cohort and are reported as warnings. Image folders without metadata are
    # unsafe to process and remain validation errors.
    has_validation_errors = bool(failures or missing_records)
    return 1 if has_validation_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
