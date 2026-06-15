#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any

import hyperspy.api as hs
import numpy as np
import tifffile as tif


DEFAULT_INPUT_ROOT = "/workspaces/mito-counter/data/DMD_1X/RAW"
DEFAULT_OUTPUT_ROOT = "/workspaces/mito-counter/data/DMD_1X/Processed"
DEFAULT_MAGNIFICATION_TOKEN: str | None = None
GROUP_DIR_RE = re.compile(r"\s*(TA|EOM)\s*[_-]\s*(WT|DMD)\s*", re.IGNORECASE)
COMPARTMENT_DIR_RE = re.compile(r"\s*(IMF|SS)\s*[_-]\s*([0-9]+)\s*", re.IGNORECASE)


@dataclass(frozen=True)
class Dmd1xLabels:
    """Labels parsed from a DMD_1X raw DM3 path.

    Args:
        muscle (str): Short muscle token, such as ``TA`` or ``EOM``.
        condition (str): Short condition token, such as ``WT`` or ``DMD``.
        group_directory (str): Raw group directory name, such as ``TA_WT``.
        compartment (str): Compartment token, such as ``IMF`` or ``SS``.
        replicate (str): Replicate number parsed from the compartment directory.
        compartment_directory (str): Raw compartment directory name, such as ``SS_3``.

    Returns:
        None: Dataclass instances are created for structured label storage.
    """

    muscle: str
    condition: str
    group_directory: str
    compartment: str
    replicate: str
    compartment_directory: str


def find_dm3_files(root: str, magnification_token: str | None) -> list[str]:
    """Return DM3 file paths under a DMD_1X raw root.

    Args:
        root (str): Root directory to recursively scan.
        magnification_token (str | None): Filename token required for inclusion,
            or ``None`` to include every DM3 file.

    Returns:
        list[str]: Absolute DM3 file paths that match the requested filter.
    """
    dm3_paths: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.lower().endswith(".dm3"):
                continue
            if magnification_token and magnification_token.lower() not in name.lower():
                continue
            dm3_paths.append(os.path.join(dirpath, name))
    return sorted(dm3_paths)


def _normalize_name(value: str) -> str:
    """Normalize spaces in a DMD_1X filename or directory token.

    Args:
        value (str): Raw filename or directory token.

    Returns:
        str: Token with whitespace collapsed to single underscores.
    """
    return re.sub(r"\s+", "_", value.strip())


def _parse_group_directory(group_directory: str) -> tuple[str, str]:
    """Parse muscle and condition from a DMD_1X group directory.

    Args:
        group_directory (str): Directory name such as ``TA_WT``.

    Returns:
        tuple[str, str]: Parsed ``(muscle, condition)`` tokens.
    """
    match = GROUP_DIR_RE.fullmatch(group_directory.strip())
    if match is None:
        raise ValueError(f"Unable to parse DMD_1X muscle/condition from: {group_directory}")
    muscle, condition = match.groups()
    return muscle.upper(), condition.upper()


def _parse_compartment_directory(compartment_directory: str) -> tuple[str, str]:
    """Parse compartment and replicate from a DMD_1X compartment directory.

    Args:
        compartment_directory (str): Directory name such as ``IMF_1`` or ``SS_3``.

    Returns:
        tuple[str, str]: Parsed ``(compartment, replicate)`` tokens.
    """
    match = COMPARTMENT_DIR_RE.fullmatch(compartment_directory.strip())
    if match is None:
        raise ValueError(f"Unable to parse DMD_1X compartment directory: {compartment_directory}")
    compartment, replicate = match.groups()
    return compartment.upper(), replicate


def extract_labels(input_root: str, dm3_path: str) -> Dmd1xLabels:
    """Extract DMD_1X labels from the raw path layout.

    Args:
        input_root (str): Dataset input root used to compute the relative path.
        dm3_path (str): Absolute DM3 file path.

    Returns:
        Dmd1xLabels: Parsed DMD_1X labels for the source image.
    """
    rel = os.path.relpath(dm3_path, input_root)
    parts = rel.split(os.sep)
    if len(parts) < 3:
        raise ValueError(
            "Expected DMD_1X raw path layout like "
            "<muscle_condition>/<compartment_replicate>/<image.dm3>: "
            f"{dm3_path}"
        )

    group_directory = parts[0]
    compartment_directory = parts[1]
    muscle, condition = _parse_group_directory(group_directory)
    compartment, replicate = _parse_compartment_directory(compartment_directory)
    return Dmd1xLabels(
        muscle=muscle,
        condition=condition,
        group_directory=group_directory,
        compartment=compartment,
        replicate=replicate,
        compartment_directory=compartment_directory,
    )


def build_output_paths(output_root: str, labels: Dmd1xLabels, dm3_path: str) -> tuple[str, str]:
    """Build destination TIFF and JSON paths for a DMD_1X DM3 image.

    Args:
        output_root (str): Root directory for exported files.
        labels (Dmd1xLabels): Parsed DMD_1X labels for the source image.
        dm3_path (str): Source DM3 path.

    Returns:
        tuple[str, str]: ``(tiff_output_path, json_output_path)``.
    """
    base = os.path.splitext(os.path.basename(dm3_path))[0]
    safe_group_directory = _normalize_name(labels.group_directory)
    safe_compartment_directory = _normalize_name(labels.compartment_directory)
    safe_base = _normalize_name(base)
    out_dir = os.path.join(output_root, safe_group_directory, safe_compartment_directory)
    out_tiff = os.path.join(out_dir, f"{safe_base}.tif")
    out_json = os.path.join(out_dir, f"{safe_base}.json")
    return out_tiff, out_json


def _jsonify(obj: Any) -> Any:
    """Convert nested metadata objects into JSON-serializable values.

    Args:
        obj (Any): Arbitrary metadata value returned by HyperSpy.

    Returns:
        Any: JSON-serializable representation of ``obj``.
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return repr(obj)


def _prepare_for_tiff(data: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Scale raw image data into an 8-bit TIFF-friendly array.

    Args:
        data (np.ndarray): Raw image array loaded from the DM3 file.

    Returns:
        tuple[np.ndarray, dict[str, Any]]: Converted image data and conversion metadata.
    """
    info: dict[str, Any] = {
        "original_dtype": str(data.dtype),
        "original_min": float(np.nanmin(data)),
        "original_max": float(np.nanmax(data)),
    }
    finite = np.isfinite(data)
    if not np.any(finite):
        scaled = np.zeros_like(data, dtype=np.uint8)
        info["conversion"] = "all_non_finite_to_zero_uint8"
        return scaled, info

    data_min = float(np.nanmin(data))
    data_max = float(np.nanmax(data))
    if data_max == data_min:
        scaled = np.zeros_like(data, dtype=np.uint8)
        info["conversion"] = "constant_to_zero_uint8"
        return scaled, info

    finite_data = data[finite]
    p1, p99 = np.percentile(finite_data, (1, 99.8))
    if p99 == p1:
        p1, p99 = data_min, data_max

    scaled = (data - p1) / (p99 - p1)
    scaled = np.clip(scaled, 0.0, 1.0)
    scaled = (scaled * np.iinfo(np.uint8).max).astype(np.uint8)
    info["conversion"] = "robust_rescale_to_uint8"
    return scaled, info


def export_dm3(
    dm3_path: str, output_root: str, input_root: str, dry_run: bool = False
) -> tuple[str, str]:
    """Export one DMD_1X DM3 file to TIFF plus JSON metadata.

    Args:
        dm3_path (str): Absolute source DM3 file path.
        output_root (str): Root directory for exported files.
        input_root (str): Dataset input root used for label extraction.
        dry_run (bool): Whether to skip file writes and only compute destination paths.

    Returns:
        tuple[str, str]: ``(tiff_output_path, json_output_path)``.
    """
    labels = extract_labels(input_root, dm3_path)
    out_tiff, out_json = build_output_paths(output_root, labels, dm3_path)

    if dry_run:
        return out_tiff, out_json

    os.makedirs(os.path.dirname(out_tiff), exist_ok=True)
    img = hs.load(dm3_path)
    data = img.data
    data_to_save, conversion_info = _prepare_for_tiff(data)

    basic_meta = {
        "source_path": dm3_path,
        "shape": tuple(data.shape),
        "dtype": str(data.dtype),
        "saved_dtype": str(data_to_save.dtype),
        "condition": labels.condition,
        "muscle": labels.muscle,
        "group_directory": labels.group_directory,
        "compartment": labels.compartment,
        "replicate": labels.replicate,
        "compartment_directory": labels.compartment_directory,
    }

    description = json.dumps(basic_meta, ensure_ascii=True)
    tif.imwrite(out_tiff, data_to_save, description=description, photometric="minisblack")

    metadata = {
        "basic": basic_meta,
        "conversion": conversion_info,
        "hyperspy_metadata": _jsonify(img.metadata.as_dictionary()),
    }
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)

    return out_tiff, out_json


def parse_args() -> argparse.Namespace:
    """Parse DMD_1X converter CLI arguments.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed command-line options.
    """
    parser = argparse.ArgumentParser(
        description="Convert DMD_1X DM3 files to TIFF with mirrored RAW-tree paths."
    )
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT, help="Input root with DM3 files.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Output root for TIFF exports.")
    parser.add_argument(
        "--magnification-token",
        default=DEFAULT_MAGNIFICATION_TOKEN,
        help="Optional filename token to filter magnifications. Disabled by default.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print planned outputs.")
    args = parser.parse_args()
    if isinstance(args.magnification_token, str):
        args.magnification_token = args.magnification_token.strip() or None
    return args


def main() -> None:
    """Run the DMD_1X DM3-to-TIFF conversion pipeline.

    Args:
        None

    Returns:
        None: This function is executed for side effects only.
    """
    args = parse_args()
    dm3_files = find_dm3_files(args.input_root, args.magnification_token)
    if not dm3_files:
        print(f"No .dm3 files found under: {args.input_root}")
        return

    warning_filters = [
        (Warning, r".*ensure_directory has been moved.*"),
        (Warning, r".*overwrite has been moved.*"),
        (Warning, r".*get_file_handle has been moved.*"),
        (Warning, r".*append2pathname has been moved.*"),
        (Warning, r".*incremental_filename has been moved.*"),
        (Warning, r".*rgb_tools.*renamed.*"),
    ]
    for category, pattern in warning_filters:
        warnings.filterwarnings("ignore", category=category, message=pattern)

    failed: list[str] = []
    for dm3_path in dm3_files:
        try:
            out_tiff, out_json = export_dm3(
                dm3_path, args.output_root, args.input_root, dry_run=args.dry_run
            )
        except Exception as exc:  # pragma: no cover - resilient batch handling
            failed.append(dm3_path)
            print(f"[ERROR] {dm3_path} -> {exc}")
            continue

        if args.dry_run:
            print(f"[DRY RUN] {dm3_path} -> {out_tiff} (+ {out_json})")
        else:
            print(f"Exported {dm3_path} -> {out_tiff} (+ {out_json})")

    if failed:
        print(f"Failed to export {len(failed)} files.")


if __name__ == "__main__":
    main()
