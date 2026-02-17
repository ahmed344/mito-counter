#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import csv

import cv2
import numpy as np
import torch
import yaml
from empanada.config_loaders import load_config
from empanada.inference.engines import PanopticDeepLabEngine

import tifffile as tiff

# === User-editable config file ===
INFERENCE_CONFIG = Path("/workspaces/mito-counter/mitonet_infenence.yaml")


def resize_array(image: np.ndarray, factor: float, is_mask: bool) -> np.ndarray:
    """Resize an image or mask by the given factor (from make_training_samples.py).

    Args:
        image: Input image or mask array.
        factor: Downsample factor (>= 1.0).
        is_mask: True when resizing label masks.

    Returns:
        Resized image or mask array.
    """
    if factor == 1.0:
        return image
    height, width = image.shape[:2]
    new_width = max(1, int(round(width / factor)))
    new_height = max(1, int(round(height / factor)))
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(image, (new_width, new_height), interpolation=interp)


def write_tiff(path: Path, data: np.ndarray) -> None:
    """Write an array to disk using TIFF (fallback to OpenCV).

    Args:
        path: Output file path.
        data: Image array to write.

    Returns:
        None
    """
    if tiff is not None:
        tiff.imwrite(str(path), data)
        return
    if data.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        data = data.astype(np.uint16)
    if data.ndim == 3 and data.shape[2] == 3:
        data = data[:, :, ::-1]
    if not cv2.imwrite(str(path), data):
        raise ValueError(f"Failed to write image: {path}")


def load_torchscript(model_path: Path) -> torch.jit.ScriptModule:
    """Load a TorchScript model from disk.

    Args:
        model_path: Path to the TorchScript model.

    Returns:
        Loaded TorchScript model.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return torch.jit.load(str(model_path), map_location="cpu")


def infer_input_channels(model: torch.jit.ScriptModule) -> int:
    """Infer expected input channels from model parameters.

    Args:
        model: TorchScript model to inspect.

    Returns:
        Number of input channels expected by the first conv layer.
    """
    for _, param in model.named_parameters():
        if param.ndim == 4:
            return int(param.shape[1])
    raise ValueError("Unable to infer input channels from model parameters.")


def normalize_image(image: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Normalize image to model input scale and stats.

    Args:
        image: Input image array.
        mean: Mean used for normalization.
        std: Standard deviation used for normalization.

    Returns:
        Normalized float32 image array.
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    else:
        image = image.astype(np.float32)
    return (image - mean) / (std + 1e-8)


def to_model_input(image: np.ndarray, in_channels: int, mean: float, std: float) -> torch.Tensor:
    """Convert image array into model-ready tensor.

    Args:
        image: Input image array (H, W) or (H, W, C).
        in_channels: Model expected channel count.
        mean: Normalization mean.
        std: Normalization standard deviation.

    Returns:
        Torch tensor with shape (1, C, H, W).
    """
    if image.ndim == 2:
        if in_channels == 1:
            image = image[:, :, None]
        elif in_channels == 3:
            image = np.repeat(image[:, :, None], 3, axis=2)
        else:
            raise ValueError(f"Unsupported in_channels for 2D input: {in_channels}")
    elif image.ndim == 3:
        channels = image.shape[2]
        if channels == in_channels:
            pass
        elif channels == 1 and in_channels == 3:
            image = np.repeat(image, 3, axis=2)
        elif channels == 3 and in_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, None]
        else:
            raise ValueError(f"Input channels {channels} do not match model {in_channels}")
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    image = normalize_image(image, mean, std)
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0)


def build_output_path(input_path: Path) -> Path:
    """Construct output path for a given input image.

    Args:
        input_path: Source image path.

    Returns:
        Output path with _segmented suffix.
    """
    stem = input_path.stem
    if stem.endswith("_corrected"):
        stem = stem[: -len("_corrected")]
    return input_path.parent / f"{stem}_segmented.tif"


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    """Convert label IDs into an RGB visualization.

    Args:
        labels: Integer label map.

    Returns:
        RGB visualization of labels.
    """
    labels = labels.astype(np.uint32)
    rgb = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    mask = labels > 0
    if not np.any(mask):
        return rgb
    vals = labels[mask]
    rgb_vals = np.stack(
        (
            (vals * 37 + 23) % 255,
            (vals * 17 + 91) % 255,
            (vals * 29 + 47) % 255,
        ),
        axis=1,
    ).astype(np.uint8)
    rgb[mask] = rgb_vals
    return rgb


def compute_instance_metrics(labels: np.ndarray) -> list[dict]:
    """Compute per-instance metrics from a label image.

    Args:
        labels: Integer label map where 0 is background.

    Returns:
        List of per-instance metrics dictionaries.
    """
    from skimage.measure import regionprops

    props = regionprops(labels)
    metrics: list[dict] = []
    centroids = []
    for prop in props:
        instance_id = int(prop.label)
        centroid_rc = prop.centroid
        area = float(prop.area)
        major_attr = (
            getattr(prop, "axis_major_length", None)
            if hasattr(prop, "axis_major_length")
            else getattr(prop, "major_axis_length", None)
        )
        minor_attr = (
            getattr(prop, "axis_minor_length", None)
            if hasattr(prop, "axis_minor_length")
            else getattr(prop, "minor_axis_length", None)
        )
        major = float(major_attr) if major_attr else 0.0
        minor = float(minor_attr) if minor_attr else 0.0
        elongation = (major / minor) if minor > 0 else 0.0
        perimeter = float(prop.perimeter) if prop.perimeter else 0.0
        circularity = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
        solidity = float(prop.solidity) if prop.solidity is not None else 0.0

        metrics.append(
            {
                "id": instance_id,
                "centroid": f"({centroid_rc[1]:.2f}, {centroid_rc[0]:.2f})",
                "area": area,
                "major_axis_length": major,
                "minor_axis_length": minor,
                "aspect_ratio_elongation": elongation,
                "circularity_form_factor": circularity,
                "solidity_branching": solidity,
                "nearest_neighbor_distance": 0.0,
                "centroid_x": float(centroid_rc[1]),
                "centroid_y": float(centroid_rc[0]),
            }
        )
        centroids.append((float(centroid_rc[1]), float(centroid_rc[0])))

    if centroids:
        coords = np.array(centroids, dtype=np.float64)
        for i in range(len(coords)):
            if len(coords) == 1:
                nnd = 0.0
            else:
                diff = coords - coords[i]
                dist = np.sqrt((diff ** 2).sum(axis=1))
                dist[i] = np.inf
                nnd = float(np.min(dist))
            metrics[i]["nearest_neighbor_distance"] = nnd

    return metrics


def write_metrics_csv(path: Path, metrics: list[dict]) -> None:
    """Write instance metrics to a CSV file.

    Args:
        path: Output CSV path.
        metrics: List of metric dictionaries.

    Returns:
        None
    """
    fieldnames = [
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
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(
                {
                    "Id": row["id"],
                    "Centroid": row["centroid"],
                    "Area": f"{row['area']:.2f}",
                    "Major_axis_length": f"{row['major_axis_length']:.2f}",
                    "Minor_axis_length": f"{row['minor_axis_length']:.2f}",
                    "Elongation": f"{row['aspect_ratio_elongation']:.3f}",
                    "Circularity": f"{row['circularity_form_factor']:.3f}",
                    "Solidity": f"{row['solidity_branching']:.3f}",
                    "NND": f"{row['nearest_neighbor_distance']:.3f}",
                }
            )


def main() -> None:
    """Run batch inference over the input directory.

    Returns:
        None
    """
    # Load inference configuration and validate required fields.
    if not INFERENCE_CONFIG.is_file():
        raise FileNotFoundError(f"Missing inference config: {INFERENCE_CONFIG}")
    with open(INFERENCE_CONFIG, "r", encoding="utf-8") as handle:
        inference_cfg = yaml.safe_load(handle)

    input_dir = Path(inference_cfg["paths"]["input_dir"])
    model_pth = Path(inference_cfg["paths"]["model_pth"])
    config_yaml = Path(inference_cfg["paths"]["config_yaml"])
    downsample_factor = float(inference_cfg["paths"]["downsample_factor"])
    device = str(inference_cfg["paths"]["device"])
    engine_params = inference_cfg["engine_params"]

    if device != "cuda":
        raise ValueError("device must be set to 'cuda' to run on GPU.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a GPU-enabled setup.")
    if downsample_factor < 1.0:
        raise ValueError("downsample_factor must be >= 1.0.")

    # Load model config and TorchScript model.
    cfg = load_config(str(config_yaml))

    model = load_torchscript(model_pth)
    model.to(device)
    model.eval()

    # Determine model input channels and normalization stats.
    in_channels = infer_input_channels(model)
    mean = float(cfg["norms"]["mean"])
    std = float(cfg["norms"]["std"])

    # Initialize the panoptic inference engine.
    engine = PanopticDeepLabEngine(
        model,
        thing_list=engine_params["thing_list"],
        label_divisor=engine_params["label_divisor"],
        stuff_area=engine_params["stuff_area"],
        void_label=engine_params["void_label"],
        nms_threshold=engine_params["nms_threshold"],
        nms_kernel=engine_params["nms_kernel"],
        confidence_thr=engine_params["confidence_thr"],
        coarse_boundaries=True,
    )

    # Find only corrected TIFFs recursively under the input directory.
    image_paths = sorted(
        [
            p
            for p in input_dir.rglob("*")
            if p.suffix.lower() in {".tif", ".tiff"}
            and p.name.endswith("_corrected.tif")
        ]
    )

    total = len(image_paths)

    for idx, image_path in enumerate(image_paths, start=1):
        output_path = build_output_path(image_path)

        print(f"Processing image {idx}/{total}: {image_path.name}")

        # Read and downsample the image for faster inference.
        image = tiff.imread(str(image_path))

        orig_h, orig_w = image.shape[:2]
        image = resize_array(image, downsample_factor, is_mask=False)

        # Convert to model tensor and run inference.
        input_tensor = to_model_input(image, in_channels, mean, std).to(device)
        with torch.no_grad():
            pan_pred = engine(input_tensor)

        # Upsample predictions and compute per-instance metrics.
        pan_np = pan_pred.squeeze().detach().cpu().numpy().astype(np.int32)
        if downsample_factor != 1.0:
            pan_np = cv2.resize(
                pan_np,
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )
        pan_np = pan_np.astype(np.uint32)
        metrics = compute_instance_metrics(pan_np)
        metrics_path = output_path.with_name(f"{output_path.stem}_metrics.csv")
        write_metrics_csv(metrics_path, metrics)

        # Convert to colorful visualization and annotate IDs at centroids.
        color = colorize_labels(pan_np)
        for row in metrics:
            cx = int(round(row["centroid_x"]))
            cy = int(round(row["centroid_y"]))
            cv2.putText(
                color,
                str(row["id"]),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Write output beside the input image.
        write_tiff(output_path, color)


if __name__ == "__main__":
    main()
