#!/usr/bin/env python3
import argparse
import json
import os
import re
import warnings
from typing import Any, Dict, Iterable, Optional, Tuple

import hyperspy.api as hs
import numpy as np
import tifffile as tif


DEFAULT_INPUT_ROOT = "/workspaces/mito-counter/data/Etude Calpaine 3/260121 - Ahmed"
DEFAULT_OUTPUT_ROOT = "/workspaces/mito-counter/data/Calpaine_3/Processed"

MAGNIFICATION_TOKEN = "6800X"


def find_dm3_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.lower().endswith(".dm3"):
                continue
            if MAGNIFICATION_TOKEN.lower() not in name.lower():
                continue
            yield os.path.join(dirpath, name)


def _normalize_whitespace(value: str) -> str:
    return value.replace(" ", "_")


def _extract_genotype(path_parts: Iterable[str]) -> Optional[str]:
    for part in path_parts:
        upper = part.upper()
        if re.search(r"\bKO\s*-?\s*C3\b", upper):
            return "KO-C3"
        if re.search(r"\bWT\b", upper):
            return "WT"
    return None


def _extract_muscle(path_parts: Iterable[str]) -> Optional[str]:
    for part in path_parts:
        upper = part.upper()
        match = re.search(r"\b(SOL|TA)\s*([0-9]+)\b", upper)
        if match:
            return f"{match.group(1)}_{match.group(2)}"
        match = re.search(r"\b(SOL|TA)\b", upper)
        if match:
            return match.group(1)
    return None


def extract_labels(input_root: str, dm3_path: str) -> Tuple[str, str]:
    rel = os.path.relpath(dm3_path, input_root)
    parts = rel.split(os.sep)
    genotype = _extract_genotype(parts) or "UNKNOWN"
    muscle = _extract_muscle(parts)
    filename = os.path.basename(dm3_path)
    if muscle is None or muscle in {"TA", "SOL"}:
        match = re.search(r"\b(SOL|TA)\s*([0-9]+)\b", filename.upper())
        if match:
            muscle = f"{match.group(1)}_{match.group(2)}"
        elif muscle is None:
            match = re.search(r"\b(SOL|TA)\b", filename.upper())
            muscle = match.group(1) if match else "UNKNOWN"
    return genotype, muscle


def _extract_muscle_group(muscle: str) -> str:
    match = re.match(r"^(TA|SOL)", muscle.upper())
    return match.group(1) if match else "UNKNOWN"


def build_output_paths(output_root: str, genotype: str, muscle: str, dm3_path: str) -> Tuple[str, str]:
    safe_genotype = _normalize_whitespace(genotype)
    muscle_group = _extract_muscle_group(muscle)
    safe_muscle = _normalize_whitespace(muscle_group)
    base = os.path.splitext(os.path.basename(dm3_path))[0]
    safe_base = _normalize_whitespace(base)
    out_dir = os.path.join(output_root, safe_genotype, safe_muscle)
    out_tiff = os.path.join(out_dir, f"{safe_base}.tif")
    out_json = os.path.join(out_dir, f"{safe_base}.json")
    return out_tiff, out_json


def _jsonify(obj: Any) -> Any:
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


def _prepare_for_tiff(data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {
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


def export_dm3(dm3_path: str, output_root: str, input_root: str, dry_run: bool = False) -> Tuple[str, str]:
    genotype, muscle = extract_labels(input_root, dm3_path)
    out_tiff, out_json = build_output_paths(output_root, genotype, muscle, dm3_path)
    os.makedirs(os.path.dirname(out_tiff), exist_ok=True)

    if dry_run:
        return out_tiff, out_json

    img = hs.load(dm3_path)
    data = img.data
    data_to_save, conversion_info = _prepare_for_tiff(data)

    basic_meta = {
        "source_path": dm3_path,
        "shape": tuple(data.shape),
        "dtype": str(data.dtype),
        "saved_dtype": str(data_to_save.dtype),
        "genotype": genotype,
        "muscle": muscle,
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
    parser = argparse.ArgumentParser(description="Convert DM3 files to TIFF with metadata.")
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT, help="Input root with DM3 files.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Output root for TIFF exports.")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dm3_files = list(find_dm3_files(args.input_root))
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
