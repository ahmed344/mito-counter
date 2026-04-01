#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile as tiff

DEFAULT_INPUT_ROOT = Path("/workspaces/mito-counter/data/DMD/Processed")
DEFAULT_OUTPUT_ROOT = Path("/workspaces/mito-counter/data/DMD/results/overlay_qc")
SS_THRESHOLD_UM = 1.0
REFERENCE_MAGNIFICATION = 6800.0
REFERENCE_PIXEL_SIZE_UM = 0.0015396
EXCLUDED_SOURCE_SUFFIXES = (
    "_segmented",
    "_cells",
    "_center_cell_mask",
    "_sarcomere",
    "_sarcomere_corrected",
    "_sarcomere_segmented",
    "_cell_mask",
    "_sarcomere_mask",
    "_mbands",
    "_zbands",
)
EXCLUDED_SOURCE_STEMS = {
    "cell_mask",
    "sarcomere_mask",
    "mbands",
    "zbands",
}
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
class OverlayInput:
    """Container for one overlay rendering job.

    Args:
        corrected_path (Path): Path to the corrected grayscale TIFF image.
        segmented_path (Path): Path to the RGB mitochondria segmentation TIFF.
        cell_mask_path (Path): Path to the binary cell-mask TIFF.
        metadata_path (Path): Path to the per-image JSON metadata file.
        output_path (Path): Path where the rendered PNG overlay will be written.
        pixel_size_um (float): Micrometers per pixel for this image.
        pixel_size_source (str): Source label describing how pixel size was resolved.

    Returns:
        None: Dataclass field container.
    """

    corrected_path: Path
    segmented_path: Path
    cell_mask_path: Path
    metadata_path: Path
    output_path: Path
    pixel_size_um: float
    pixel_size_source: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the QC overlay batch script.

    Args:
        None: This function reads arguments from the process command line.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Render full-frame QC overlays for corrected DMD images using mito "
            "segmentation RGB, binary cell masks, and a 1.0 um membrane-adjacent SS band."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory scanned recursively for '*_corrected.tif' files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where PNG overlay outputs are written.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Optional single corrected TIFF to process instead of scanning --input-root.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of images processed after discovery.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite overlay files even when the PNG already exists.",
    )
    return parser.parse_args()


def load_json_file(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk.

    Args:
        path (Path): JSON file path to read.

    Returns:
        dict[str, Any]: Parsed JSON object.
    """

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in: {path}")
    return data


def normalize_unit_name(unit: str | None) -> str | None:
    """Normalize a unit string for lookup in the conversion table.

    Args:
        unit (str | None): Raw unit text from metadata.

    Returns:
        str | None: Lower-cased unit string with micro symbols normalized, or ``None``.
    """

    if unit is None:
        return None
    normalized = unit.strip().lower()
    normalized = normalized.replace("µ", "u")
    normalized = normalized.replace("μ", "u")
    return normalized


def convert_to_um(value: float, unit: str | None) -> float | None:
    """Convert a scalar value into micrometers.

    Args:
        value (float): Numeric value to convert.
        unit (str | None): Unit label associated with the numeric value.

    Returns:
        float | None: Converted micrometer value, or ``None`` when the unit is unsupported.
    """

    normalized = normalize_unit_name(unit)
    if normalized is None:
        return None
    factor = UNIT_ALIASES.get(normalized)
    if factor is None:
        return None
    return value * factor


def walk_nested_values(data: Any) -> list[Any]:
    """Flatten nested metadata into a depth-first value list.

    Args:
        data (Any): Nested metadata structure composed of dicts, lists, and scalars.

    Returns:
        list[Any]: List of the current object and all nested values.
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
    """Extract microscope magnification from a metadata mapping.

    Args:
        metadata (dict[str, Any]): Parsed image metadata.

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
    """Extract magnification from an image filename stem.

    Args:
        image_label (str): Image stem that may contain a token such as ``1200X``.

    Returns:
        float | None: Parsed magnification, or ``None`` when absent.
    """

    match = MAGNIFICATION_RE.search(image_label)
    if not match:
        return None
    return float(match.group(1))


def find_pixel_size_from_dict(node: dict[str, Any]) -> float | None:
    """Extract micrometers-per-pixel from one metadata mapping.

    Args:
        node (dict[str, Any]): Metadata mapping to inspect.

    Returns:
        float | None: Pixel size in micrometers, or ``None`` when not encoded here.
    """

    normalized_keys = {str(key).strip().lower().replace(" ", "_"): key for key in node}
    for normalized_key in PIXEL_SIZE_KEYWORDS:
        raw_key = normalized_keys.get(normalized_key)
        if raw_key is None:
            continue
        value = node[raw_key]
        if not isinstance(value, (int, float)):
            continue
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
    if x_scale_key is None or y_scale_key is None:
        return None
    x_scale = node[x_scale_key]
    y_scale = node[y_scale_key]
    if not isinstance(x_scale, (int, float)) or not isinstance(y_scale, (int, float)):
        return None
    if float(x_scale) <= 0.0 or float(y_scale) <= 0.0:
        return None
    unit_value = node.get("units") or node.get("unit")
    return convert_to_um(float(x_scale), unit_value if isinstance(unit_value, str) else None)


def extract_pixel_size_from_json_metadata(metadata: dict[str, Any]) -> float | None:
    """Resolve pixel size from nested JSON metadata.

    Args:
        metadata (dict[str, Any]): Parsed metadata dictionary.

    Returns:
        float | None: Pixel size in micrometers, or ``None`` when unavailable.
    """

    for value in walk_nested_values(metadata):
        if not isinstance(value, dict):
            continue
        pixel_size_um = find_pixel_size_from_dict(value)
        if pixel_size_um is not None and pixel_size_um > 0.0:
            return pixel_size_um
    return None


def extract_pixel_size_from_tiff_metadata(image_path: Path) -> float | None:
    """Resolve pixel size from TIFF metadata when available.

    Args:
        image_path (Path): TIFF image path to inspect.

    Returns:
        float | None: Pixel size in micrometers, or ``None`` when unavailable.
    """

    if not image_path.is_file():
        return None
    with tiff.TiffFile(str(image_path)) as tif_file:
        page = tif_file.pages[0]
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

        imagej_metadata = tif_file.imagej_metadata
        if isinstance(imagej_metadata, dict):
            pixel_size_um = extract_pixel_size_from_json_metadata(imagej_metadata)
            if pixel_size_um is not None:
                return pixel_size_um

        x_resolution_tag = page.tags.get("XResolution")
        resolution_unit_tag = page.tags.get("ResolutionUnit")
        if x_resolution_tag is None or resolution_unit_tag is None:
            return None
        numerator, denominator = x_resolution_tag.value
        if not numerator or not denominator:
            return None
        pixels_per_unit = float(numerator) / float(denominator)
        if pixels_per_unit <= 0.0:
            return None
        resolution_unit = int(resolution_unit_tag.value)
        if resolution_unit == 2:
            return 25400.0 / pixels_per_unit
        if resolution_unit == 3:
            return 10000.0 / pixels_per_unit
    return None


def magnification_to_pixel_size_um(magnification: float) -> float:
    """Estimate pixel size from magnification using the repo reference calibration.

    Args:
        magnification (float): Microscope magnification for the image.

    Returns:
        float: Estimated micrometers per pixel.
    """

    if magnification <= 0.0:
        raise ValueError(f"Magnification must be positive, got {magnification}.")
    return REFERENCE_PIXEL_SIZE_UM * (REFERENCE_MAGNIFICATION / magnification)


def resolve_pixel_size_um(
    metadata: dict[str, Any],
    image_label: str,
    image_path: Path,
) -> tuple[float, str]:
    """Resolve per-image pixel size from metadata or magnification fallback.

    Args:
        metadata (dict[str, Any]): Parsed sidecar metadata for the image.
        image_label (str): Image stem without suffixes.
        image_path (Path): Path to the base image TIFF.

    Returns:
        tuple[float, str]: Resolved pixel size in micrometers and a source label.
    """

    magnification = extract_magnification_from_metadata(metadata)
    analysis_metadata_path = image_path.with_suffix("") / "data" / "metadata.json"
    metadata_sources: list[tuple[str, dict[str, Any]]] = [("json_metadata", metadata)]
    if analysis_metadata_path.is_file():
        metadata_sources.append(("analysis_metadata", load_json_file(analysis_metadata_path)))

    for source_name, metadata_source in metadata_sources:
        pixel_size_um = extract_pixel_size_from_json_metadata(metadata_source)
        if pixel_size_um is not None:
            return pixel_size_um, source_name

    pixel_size_um = extract_pixel_size_from_tiff_metadata(image_path)
    if pixel_size_um is not None:
        return pixel_size_um, "tiff_metadata"

    if magnification is None:
        magnification = extract_magnification_from_filename(image_label)
    if magnification is None:
        raise ValueError(
            f"Unable to resolve pixel size or magnification for image '{image_label}'."
        )
    return magnification_to_pixel_size_um(magnification), "magnification_fallback"


def discover_corrected_images(
    input_root: Path,
    input_file: Path | None,
    limit: int | None,
) -> list[Path]:
    """Discover corrected TIFF images for overlay rendering.

    Args:
        input_root (Path): Root directory scanned recursively for corrected images.
        input_file (Path | None): Optional single corrected image to process.
        limit (int | None): Optional maximum number of corrected images to return.

    Returns:
        list[Path]: Sorted list of corrected TIFF paths.
    """

    def is_primary_corrected_image(path: Path) -> bool:
        """Check whether a corrected TIFF is a primary image rather than a derived output.

        Args:
            path (Path): Candidate corrected TIFF path.

        Returns:
            bool: ``True`` when the path looks like a primary corrected image.
        """

        if not path.name.lower().endswith("_corrected.tif"):
            return False
        stem_without_corrected = path.stem[: -len("_corrected")]
        stem_lower = stem_without_corrected.lower()
        if stem_lower in EXCLUDED_SOURCE_STEMS:
            return False
        return not stem_lower.endswith(EXCLUDED_SOURCE_SUFFIXES)

    if input_file is not None:
        if not input_file.is_file():
            raise FileNotFoundError(f"Corrected input file not found: {input_file}")
        corrected_paths = [input_file]
    else:
        corrected_paths = sorted(
            path for path in input_root.rglob("*_corrected.tif") if is_primary_corrected_image(path)
        )
    if limit is not None:
        return corrected_paths[:limit]
    return corrected_paths


def build_overlay_input(
    corrected_path: Path,
    input_root: Path,
    output_root: Path,
) -> OverlayInput:
    """Resolve all sibling files and output path for one corrected image.

    Args:
        corrected_path (Path): Path to the corrected TIFF image.
        input_root (Path): Root processed directory used to compute relative output paths.
        output_root (Path): Root output directory for saved overlays.

    Returns:
        OverlayInput: Resolved input bundle for rendering one overlay.
    """

    if corrected_path.stem.endswith("_corrected"):
        image_label = corrected_path.stem[: -len("_corrected")]
    else:
        raise ValueError(f"Expected corrected image stem to end with '_corrected': {corrected_path}")

    segmented_path = corrected_path.with_name(f"{image_label}_segmented.tif")
    cell_mask_path = corrected_path.with_name(f"{image_label}_cells.tif")
    metadata_path = corrected_path.with_name(f"{image_label}.json")
    if not segmented_path.is_file():
        raise FileNotFoundError(f"Segmented mito TIFF not found: {segmented_path}")
    if not cell_mask_path.is_file():
        raise FileNotFoundError(f"Cell mask TIFF not found: {cell_mask_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata JSON not found: {metadata_path}")

    metadata = load_json_file(metadata_path)
    pixel_size_um, pixel_size_source = resolve_pixel_size_um(
        metadata=metadata,
        image_label=image_label,
        image_path=corrected_path.with_name(f"{image_label}.tif"),
    )
    relative_parent = corrected_path.parent.relative_to(input_root)
    output_path = output_root / relative_parent / f"{image_label}_overlay.png"
    return OverlayInput(
        corrected_path=corrected_path,
        segmented_path=segmented_path,
        cell_mask_path=cell_mask_path,
        metadata_path=metadata_path,
        output_path=output_path,
        pixel_size_um=pixel_size_um,
        pixel_size_source=pixel_size_source,
    )


def load_grayscale_or_rgb(path: Path) -> np.ndarray:
    """Read a TIFF image from disk.

    Args:
        path (Path): TIFF path to read.

    Returns:
        np.ndarray: Loaded image array.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return tiff.imread(str(path))


def to_uint8_display(image: np.ndarray) -> np.ndarray:
    """Convert an image array into a display-ready uint8 representation.

    Args:
        image (np.ndarray): Input grayscale or RGB-like image array.

    Returns:
        np.ndarray: Display-ready uint8 image preserving dimensionality.
    """

    if image.dtype == np.uint8:
        return image.copy()
    arr = image.astype(np.float32)
    lo = float(np.percentile(arr, 1.0))
    hi = float(np.percentile(arr, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def ensure_rgb_display(image: np.ndarray) -> np.ndarray:
    """Ensure an image is represented as an RGB uint8 array.

    Args:
        image (np.ndarray): Input grayscale or RGB-like image array.

    Returns:
        np.ndarray: RGB uint8 image with shape ``(H, W, 3)``.
    """

    display = to_uint8_display(image)
    if display.ndim == 2:
        return np.repeat(display[:, :, None], 3, axis=2)
    if display.ndim == 3 and display.shape[2] >= 3:
        return display[:, :, :3].copy()
    raise ValueError(f"Unsupported image shape for RGB display: {display.shape}")


def load_binary_mask(path: Path) -> np.ndarray:
    """Load a binary mask from a TIFF file.

    Args:
        path (Path): Cell-mask TIFF path.

    Returns:
        np.ndarray: Boolean cell mask array with ``True`` for cell pixels.
    """

    mask = load_grayscale_or_rgb(path)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D cell mask, got shape {mask.shape} for {path}.")
    return mask > 0


def compute_external_background(mask: np.ndarray) -> np.ndarray:
    """Identify background pixels connected to the image border.

    Args:
        mask (np.ndarray): Boolean cell mask where ``True`` marks cell pixels.

    Returns:
        np.ndarray: Boolean mask where ``True`` marks external background pixels.
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


def build_distance_map_px(cell_mask: np.ndarray) -> np.ndarray:
    """Build a distance map from each pixel to the nearest external background pixel.

    Args:
        cell_mask (np.ndarray): Boolean cell mask with ``True`` for cell pixels.

    Returns:
        np.ndarray: Float32 distance map in pixels.
    """

    external_background = compute_external_background(cell_mask)
    if not np.any(external_background):
        return np.full(cell_mask.shape, np.inf, dtype=np.float32)
    distance_input = np.where(external_background, 0, 1).astype(np.uint8)
    return cv2.distanceTransform(distance_input, distanceType=cv2.DIST_L2, maskSize=5)


def label_cell_components(cell_mask: np.ndarray) -> np.ndarray:
    """Assign connected-component labels to cell regions.

    Args:
        cell_mask (np.ndarray): Boolean cell mask with ``True`` for cell pixels.

    Returns:
        np.ndarray: Integer label image where zero is background and positive values are cells.
    """

    _, labels = cv2.connectedComponents(cell_mask.astype(np.uint8), connectivity=8)
    return labels.astype(np.int32)


def component_color_rgb(component_id: int) -> np.ndarray:
    """Generate a deterministic RGB color for one cell component.

    Args:
        component_id (int): Connected-component identifier.

    Returns:
        np.ndarray: RGB uint8 color triplet for the component.
    """
    palette = np.array(
        [
            [214, 96, 77],
            [188, 126, 48],
            [175, 82, 142],
            [155, 103, 201],
            [196, 81, 111],
            [144, 116, 58],
            [205, 117, 86],
            [163, 92, 63],
            [182, 109, 154],
        ],
        dtype=np.uint8,
    )
    return palette[(component_id - 1) % len(palette)]


def blend_color_layer(
    canvas_rgb: np.ndarray,
    mask: np.ndarray,
    color_rgb: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend one color or one RGB layer into an image under a boolean mask.

    Args:
        canvas_rgb (np.ndarray): Base RGB float image array with shape ``(H, W, 3)``.
        mask (np.ndarray): Boolean mask specifying pixels to blend.
        color_rgb (np.ndarray): Either one RGB color triplet with shape ``(3,)`` or a full RGB
            image with shape ``(H, W, 3)``.
        alpha (float): Blend opacity in the range ``[0, 1]``.

    Returns:
        np.ndarray: Blended RGB float image array.
    """

    if alpha <= 0.0 or not np.any(mask):
        return canvas_rgb
    color_float = color_rgb.astype(np.float32)
    if color_float.ndim == 1:
        canvas_rgb[mask] = (
            (1.0 - alpha) * canvas_rgb[mask] + alpha * color_float
        )
        return canvas_rgb
    if color_float.shape != canvas_rgb.shape:
        raise ValueError(
            "Per-pixel color layer shape must match canvas shape: "
            f"canvas={canvas_rgb.shape}, color={color_float.shape}"
        )
    canvas_rgb[mask] = (
        (1.0 - alpha) * canvas_rgb[mask] + alpha * color_float[mask]
    )
    return canvas_rgb


def render_overlay(
    corrected_image: np.ndarray,
    segmented_rgb: np.ndarray,
    cell_mask: np.ndarray,
    pixel_size_um: float,
) -> np.ndarray:
    """Render one QC overlay image from corrected, mito, and cell inputs.

    Args:
        corrected_image (np.ndarray): Corrected grayscale or RGB image.
        segmented_rgb (np.ndarray): RGB mito-segmentation visualization image.
        cell_mask (np.ndarray): Boolean binary cell mask.
        pixel_size_um (float): Image pixel size in micrometers.

    Returns:
        np.ndarray: Rendered RGB uint8 overlay image.
    """

    base_rgb = ensure_rgb_display(corrected_image).astype(np.float32)
    mito_rgb = ensure_rgb_display(segmented_rgb)
    if base_rgb.shape[:2] != mito_rgb.shape[:2] or base_rgb.shape[:2] != cell_mask.shape:
        raise ValueError(
            "Corrected image, mito segmentation, and cell mask shapes must match: "
            f"corrected={base_rgb.shape}, segmented={mito_rgb.shape}, cell_mask={cell_mask.shape}"
        )

    distance_map_px = build_distance_map_px(cell_mask)
    ss_threshold_px = SS_THRESHOLD_UM / pixel_size_um
    ss_band_mask = cell_mask & np.isfinite(distance_map_px) & (distance_map_px <= ss_threshold_px)
    cell_labels = label_cell_components(cell_mask)
    component_ids = sorted(int(value) for value in np.unique(cell_labels) if int(value) > 0)

    overlay = base_rgb.copy()
    for component_id in component_ids:
        component_mask = cell_labels == component_id
        component_color = component_color_rgb(component_id)
        overlay = blend_color_layer(overlay, component_mask, component_color, alpha=0.15)
        overlay = blend_color_layer(
            overlay,
            component_mask & ss_band_mask,
            component_color,
            alpha=0.32,
        )

    mito_mask = np.any(mito_rgb > 0, axis=2)
    overlay = blend_color_layer(overlay, mito_mask, mito_rgb, alpha=0.28)
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def save_overlay_png(image_rgb: np.ndarray, output_path: Path) -> None:
    """Write one RGB overlay image to a PNG file.

    Args:
        image_rgb (np.ndarray): RGB uint8 image to save.
        output_path (Path): Output PNG path.

    Returns:
        None: Writes the image to disk.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(
        str(output_path),
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
    )
    if not success:
        raise RuntimeError(f"Failed to write overlay PNG: {output_path}")


def process_one_overlay(job: OverlayInput) -> None:
    """Render and save one overlay image.

    Args:
        job (OverlayInput): Input bundle describing one image to render.

    Returns:
        None: Writes one overlay PNG to disk.
    """

    corrected_image = load_grayscale_or_rgb(job.corrected_path)
    segmented_rgb = load_grayscale_or_rgb(job.segmented_path)
    cell_mask = load_binary_mask(job.cell_mask_path)
    overlay_rgb = render_overlay(
        corrected_image=corrected_image,
        segmented_rgb=segmented_rgb,
        cell_mask=cell_mask,
        pixel_size_um=job.pixel_size_um,
    )
    save_overlay_png(overlay_rgb, job.output_path)


def main() -> None:
    """Run the batch QC overlay rendering pipeline.

    Args:
        None: This function does not accept arguments directly.

    Returns:
        None: Discovers images, renders overlays, and writes PNG files to disk.
    """

    args = parse_args()
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")
    corrected_paths = discover_corrected_images(
        input_root=args.input_root,
        input_file=args.input_file,
        limit=args.limit,
    )
    if not corrected_paths:
        raise ValueError(f"No corrected TIFF files found under: {args.input_root}")

    written_count = 0
    skipped_count = 0
    failed_count = 0
    for corrected_path in corrected_paths:
        try:
            job = build_overlay_input(
                corrected_path=corrected_path,
                input_root=args.input_root,
                output_root=args.output_root,
            )
        except Exception as exc:
            failed_count += 1
            print(f"[ERROR] {corrected_path} -> {exc}")
            continue
        if job.output_path.is_file() and not args.overwrite:
            skipped_count += 1
            print(f"[SKIP] {job.output_path}")
            continue
        try:
            process_one_overlay(job)
        except Exception as exc:
            failed_count += 1
            print(f"[ERROR] {corrected_path} -> {exc}")
            continue
        written_count += 1
        print(
            f"[OK] {job.output_path} "
            f"(pixel_size_um={job.pixel_size_um:.6f}, source={job.pixel_size_source})"
        )

    print(
        f"Finished overlays: written={written_count}, skipped={skipped_count}, "
        f"failed={failed_count}, total={len(corrected_paths)}"
    )


if __name__ == "__main__":
    main()
