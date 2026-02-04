#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path

PIXEL_SIZE_UM = 0.0015396
INPUT_ROOT = Path("/workspaces/mito-counter/data/Calpaine_3/Processed")
OUTPUT_CSV = Path("/workspaces/mito-counter/data/Calpaine_3/results/measurments.csv")

CONDITION_MAP = {
    "WT": "Wildtype",
    "KO-C3": "Calpain_3_Knockout",
}

MUSCLE_MAP = {
    "SOL": "Soleus",
    "TA": "Tibialis Anterior",
}

IMAGE_NUM_RE = re.compile(r"-(\d{4})_segmented_metrics\.csv$", re.IGNORECASE)


def parse_image_number(path: Path) -> int:
    """Extract trailing image number from a metrics filename."""
    match = IMAGE_NUM_RE.search(path.name)
    if not match:
        raise ValueError(f"Unable to parse image number from: {path.name}")
    return int(match.group(1))


def parse_centroid(text: str) -> tuple[float, float]:
    """Parse centroid string like '(x, y)' into floats."""
    cleaned = text.strip().lstrip("(").rstrip(")")
    x_str, y_str = [part.strip() for part in cleaned.split(",")]
    return float(x_str), float(y_str)


def get_first_value(row: dict, keys: list[str], required: bool = True) -> str | None:
    """Return the first present value in row for the given keys."""
    for key in keys:
        if key in row and row[key] != "":
            return row[key]
    if required:
        raise KeyError(f"Missing required columns: {keys}")
    return None


def maybe_float(text: str | None) -> float | None:
    """Convert string to float if present."""
    if text is None:
        return None
    return float(text)


def main() -> None:
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Input root not found: {INPUT_ROOT}")

    metrics_paths = sorted(INPUT_ROOT.rglob("*_segmented_metrics.csv"))
    if not metrics_paths:
        raise ValueError(f"No metrics files found under: {INPUT_ROOT}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    output_fields = [
        "Condition",
        "Muscle",
        "image",
        "Id",
        "Centroid",
        "Area",
        "Major_axis_length",
        "Minor_axis_length",
        "Elongation",
        "Circularity",
        "Solidity",
        "NND",
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

            image_num = parse_image_number(metrics_path)

            with open(metrics_path, "r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    centroid_text = get_first_value(row, ["Centroid", "centroid"])
                    cx, cy = parse_centroid(centroid_text)
                    cx_um = cx * PIXEL_SIZE_UM
                    cy_um = cy * PIXEL_SIZE_UM

                    area_text = get_first_value(row, ["Area", "area"])
                    area_um2 = float(area_text) * (PIXEL_SIZE_UM ** 2)

                    major_text = get_first_value(
                        row, ["Major_axis_length"], required=False
                    )
                    minor_text = get_first_value(
                        row, ["Minor_axis_length"], required=False
                    )
                    nnd_text = get_first_value(
                        row, ["NND", "Nearest Neighbor Distance (NND)"], required=False
                    )

                    major_val = maybe_float(major_text)
                    minor_val = maybe_float(minor_text)
                    nnd_val = maybe_float(nnd_text)

                    major_um = major_val * PIXEL_SIZE_UM if major_val is not None else None
                    minor_um = minor_val * PIXEL_SIZE_UM if minor_val is not None else None
                    nnd_um = nnd_val * PIXEL_SIZE_UM if nnd_val is not None else None

                    writer.writerow(
                        {
                            "Condition": condition,
                            "Muscle": muscle,
                            "image": image_num,
                            "Id": get_first_value(row, ["Id", "id"]),
                            "Centroid": f"({cx_um:.6f}, {cy_um:.6f})",
                            "Area": f"{area_um2:.8f}",
                            "Major_axis_length": ""
                            if major_um is None
                            else f"{major_um:.6f}",
                            "Minor_axis_length": ""
                            if minor_um is None
                            else f"{minor_um:.6f}",
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
                        }
                    )


if __name__ == "__main__":
    main()
