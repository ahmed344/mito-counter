#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import csv
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
import yaml
from empanada.config_loaders import load_config
from empanada.inference.engines import PanopticDeepLabEngine
from scipy.ndimage import find_objects
from scipy.spatial import QhullError, Voronoi

import tifffile as tiff

# === User-editable config file ===
INFERENCE_CONFIG = Path("/workspaces/mito-counter/mitonet_infenence.yaml")

# Defaults applied when a config omits the optional `fusion` block.
DEFAULT_FUSION_PARAMS = {
    "iou_threshold": 0.4,
    "containment_threshold": 0.75,
    "min_coverage_ratio": 0.7,
    "min_votes": 1,
}


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


def compute_minimum_feret_diameter(mask: np.ndarray) -> float:
    """Compute the minimum Feret's diameter from a binary instance mask.

    Args:
        mask (np.ndarray): Binary mask of a single labeled object, shape (H, W).

    Returns:
        float: Minimum Feret's diameter in pixels.
    """
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    contour = max(contours, key=cv2.contourArea)
    if contour.shape[0] < 3:
        return 0.0

    _, (width, height), _ = cv2.minAreaRect(contour)
    return float(min(width, height))


def count_connected_parts(mask: np.ndarray) -> int:
    """Count connected foreground components in a binary instance mask.

    Args:
        mask (np.ndarray): Binary mask of one instance, shape (H, W).

    Returns:
        int: Number of connected foreground components (8-connectivity).
    """
    mask_u8 = mask.astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(mask_u8, connectivity=8)
    return max(0, int(num_labels) - 1)


def compute_kth_nearest_distances(coords: np.ndarray, k_values: tuple[int, ...]) -> dict[int, np.ndarray]:
    """Compute k-th nearest centroid distances for each point.

    Args:
        coords (np.ndarray): Centroid coordinates with shape ``(n_points, 2)`` in pixels.
        k_values (tuple[int, ...]): One-based neighbor ranks to compute.

    Returns:
        dict[int, np.ndarray]: Mapping from each k value to per-point distances in pixels.
    """
    point_count = int(coords.shape[0])
    distances_by_k = {
        k_value: np.full(point_count, np.nan, dtype=np.float64) for k_value in k_values
    }
    if point_count <= 1:
        return distances_by_k

    diff = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(distances, np.inf)
    sorted_distances = np.sort(distances, axis=1)
    for k_value in k_values:
        if point_count > k_value:
            distances_by_k[k_value] = sorted_distances[:, k_value - 1]
    return distances_by_k


def polygon_area(points: np.ndarray) -> float:
    """Calculate polygon area using the shoelace formula.

    Args:
        points (np.ndarray): Ordered polygon vertices with shape ``(n_vertices, 2)``.

    Returns:
        float: Polygon area in square coordinate units.
    """
    if points.shape[0] < 3:
        return 0.0
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    return float(
        0.5
        * abs(
            np.dot(x_coords, np.roll(y_coords, -1))
            - np.dot(y_coords, np.roll(x_coords, -1))
        )
    )


def clip_polygon_to_half_plane(
    polygon: np.ndarray,
    *,
    axis: int,
    bound: float,
    keep_greater: bool,
) -> np.ndarray:
    """Clip a polygon to one axis-aligned half-plane.

    Args:
        polygon (np.ndarray): Ordered polygon vertices with shape ``(n_vertices, 2)``.
        axis (int): Coordinate axis to clip, where 0 is x and 1 is y.
        bound (float): Boundary coordinate for the half-plane.
        keep_greater (bool): Keep coordinates greater than or equal to ``bound`` when
            true, otherwise keep coordinates less than or equal to ``bound``.

    Returns:
        np.ndarray: Clipped polygon vertices.
    """
    if polygon.size == 0:
        return polygon.reshape(0, 2)

    clipped_points = []
    previous = polygon[-1]
    previous_inside = (
        previous[axis] >= bound if keep_greater else previous[axis] <= bound
    )
    for current in polygon:
        current_inside = (
            current[axis] >= bound if keep_greater else current[axis] <= bound
        )
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            if denominator != 0.0:
                fraction = (bound - previous[axis]) / denominator
                clipped_points.append(previous + fraction * (current - previous))
        if current_inside:
            clipped_points.append(current)
        previous = current
        previous_inside = current_inside

    if not clipped_points:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(clipped_points, dtype=np.float64)


def clip_polygon_to_image(
    polygon: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Clip a polygon to the image rectangle.

    Args:
        polygon (np.ndarray): Ordered polygon vertices with shape ``(n_vertices, 2)``.
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        np.ndarray: Polygon clipped to ``x=[0, width - 1]`` and ``y=[0, height - 1]``.
    """
    if width <= 1 or height <= 1:
        return np.empty((0, 2), dtype=np.float64)
    clipped = clip_polygon_to_half_plane(
        polygon, axis=0, bound=0.0, keep_greater=True
    )
    clipped = clip_polygon_to_half_plane(
        clipped, axis=0, bound=float(width - 1), keep_greater=False
    )
    clipped = clip_polygon_to_half_plane(
        clipped, axis=1, bound=0.0, keep_greater=True
    )
    clipped = clip_polygon_to_half_plane(
        clipped, axis=1, bound=float(height - 1), keep_greater=False
    )
    return clipped


def finite_voronoi_regions(
    voronoi: Voronoi,
    *,
    radius: float,
) -> tuple[list[list[int]], np.ndarray]:
    """Reconstruct finite Voronoi regions for a 2D diagram.

    Args:
        voronoi (Voronoi): SciPy Voronoi diagram built from centroid coordinates.
        radius (float): Distance used to close infinite ridges.

    Returns:
        tuple[list[list[int]], np.ndarray]: Region vertex indices per input point and
        the expanded vertex array.
    """
    if voronoi.points.shape[1] != 2:
        raise ValueError("Voronoi reconstruction only supports 2D coordinates.")

    new_regions: list[list[int]] = []
    new_vertices = voronoi.vertices.tolist()
    center = voronoi.points.mean(axis=0)
    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (point_a, point_b), (vertex_a, vertex_b) in zip(
        voronoi.ridge_points, voronoi.ridge_vertices
    ):
        all_ridges.setdefault(point_a, []).append((point_b, vertex_a, vertex_b))
        all_ridges.setdefault(point_b, []).append((point_a, vertex_a, vertex_b))

    for point_index, region_index in enumerate(voronoi.point_region):
        vertices = voronoi.regions[region_index]
        if all(vertex_index >= 0 for vertex_index in vertices):
            new_regions.append(vertices)
            continue

        new_region = [vertex_index for vertex_index in vertices if vertex_index >= 0]
        for neighbor_index, vertex_a, vertex_b in all_ridges.get(point_index, []):
            if vertex_a >= 0 and vertex_b >= 0:
                continue
            if vertex_a < 0:
                vertex_a, vertex_b = vertex_b, vertex_a
            tangent = voronoi.points[neighbor_index] - voronoi.points[point_index]
            tangent = tangent / np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = (
                voronoi.points[point_index] + voronoi.points[neighbor_index]
            ) / 2.0
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = voronoi.vertices[vertex_a] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        region_vertices = np.asarray([new_vertices[index] for index in new_region])
        angles = np.arctan2(
            region_vertices[:, 1] - region_vertices[:, 1].mean(),
            region_vertices[:, 0] - region_vertices[:, 0].mean(),
        )
        new_regions.append([new_region[index] for index in np.argsort(angles)])

    return new_regions, np.asarray(new_vertices, dtype=np.float64)


def compute_voronoi_cell_areas(coords: np.ndarray, width: int, height: int) -> np.ndarray:
    """Compute image-clipped Voronoi cell areas for centroid coordinates.

    Args:
        coords (np.ndarray): Centroid coordinates with shape ``(n_points, 2)`` in pixels.
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        np.ndarray: Per-centroid clipped Voronoi cell areas in square pixels.
    """
    point_count = int(coords.shape[0])
    areas = np.zeros(point_count, dtype=np.float64)
    if point_count < 3 or width <= 1 or height <= 1:
        return areas

    try:
        voronoi = Voronoi(coords)
    except QhullError:
        return areas

    radius = float(max(width, height) * 2)
    try:
        regions, vertices = finite_voronoi_regions(voronoi, radius=radius)
    except (ValueError, FloatingPointError):
        return areas

    for index, region in enumerate(regions):
        polygon = vertices[region]
        clipped_polygon = clip_polygon_to_image(polygon, width=width, height=height)
        areas[index] = polygon_area(clipped_polygon)
    return areas


def format_optional_float(value: float) -> str:
    """Format a finite float for CSV output, leaving missing values blank.

    Args:
        value (float): Numeric value to format.

    Returns:
        str: Fixed-width decimal text for finite values, otherwise an empty string.
    """
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        return ""
    return f"{numeric_value:.3f}"


def compute_instance_metrics(labels: np.ndarray) -> list[dict]:
    """Compute per-instance metrics from a label image.

    Args:
        labels (np.ndarray): Integer label map where 0 is background.

    Returns:
        list[dict]: Per-instance metric dictionaries that include
        connected-part counts and spatial neighborhood measurements for each
        segmentation.
    """
    from skimage.measure import regionprops

    props = regionprops(labels)
    metrics: list[dict] = []
    centroids = []
    for prop in props:
        connected_parts = count_connected_parts(prop.image)

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
        corrected_area = float(np.pi * ((minor / 2.0) ** 2))
        minimum_feret_diameter = compute_minimum_feret_diameter(prop.image)
        elongation = (major / minor) if minor > 0 else 1.0
        perimeter = float(prop.perimeter) if prop.perimeter else 0.0
        circularity = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
        solidity = float(prop.solidity) if prop.solidity is not None else 0.0

        metrics.append(
            {
                "id": instance_id,
                "centroid": f"({centroid_rc[1]:.2f}, {centroid_rc[0]:.2f})",
                "area": area,
                "corrected_area": corrected_area,
                "major_axis_length": major,
                "minor_axis_length": minor,
                "minimum_feret_diameter": minimum_feret_diameter,
                "aspect_ratio_elongation": elongation,
                "circularity_form_factor": circularity,
                "solidity_branching": solidity,
                "nearest_neighbor_distance": 0.0,
                "third_nearest_neighbor_distance": 0.0,
                "fifth_nearest_neighbor_distance": 0.0,
                "voronoi_cell_area": 0.0,
                "centroid_x": float(centroid_rc[1]),
                "centroid_y": float(centroid_rc[0]),
                "connected_parts": connected_parts,
            }
        )
        centroids.append((float(centroid_rc[1]), float(centroid_rc[0])))

    if centroids:
        coords = np.array(centroids, dtype=np.float64)
        distances_by_k = compute_kth_nearest_distances(coords, k_values=(1, 3, 5))
        voronoi_areas = compute_voronoi_cell_areas(
            coords, width=int(labels.shape[1]), height=int(labels.shape[0])
        )
        for i in range(len(coords)):
            metrics[i]["nearest_neighbor_distance"] = float(distances_by_k[1][i])
            metrics[i]["third_nearest_neighbor_distance"] = float(
                distances_by_k[3][i]
            )
            metrics[i]["fifth_nearest_neighbor_distance"] = float(
                distances_by_k[5][i]
            )
            metrics[i]["voronoi_cell_area"] = float(voronoi_areas[i])

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
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(
                {
                    "Id": row["id"],
                    "Centroid": row["centroid"],
                    "Area": f"{row['area']:.2f}",
                    "Corrected_area": f"{row['corrected_area']:.2f}",
                    "Major_axis_length": f"{row['major_axis_length']:.2f}",
                    "Minor_axis_length": f"{row['minor_axis_length']:.2f}",
                    "Minimum_Feret_Diameter": f"{row['minimum_feret_diameter']:.2f}",
                    "Elongation": f"{row['aspect_ratio_elongation']:.3f}",
                    "Circularity": f"{row['circularity_form_factor']:.3f}",
                    "Solidity": f"{row['solidity_branching']:.3f}",
                    "NND": format_optional_float(row["nearest_neighbor_distance"]),
                    "3NND": format_optional_float(row["third_nearest_neighbor_distance"]),
                    "5NND": format_optional_float(row["fifth_nearest_neighbor_distance"]),
                    "Voronoi_Cell_Area": f"{row['voronoi_cell_area']:.3f}",
                    "Connected_parts": int(row["connected_parts"]),
                }
            )


def resolve_downsample_factors(value) -> list[float]:
    """Normalize the configured downsample factor into an ordered scale list.

    Args:
        value: Either a single numeric downsample factor or a sequence of factors.

    Returns:
        list[float]: Unique factors sorted from finest (smallest) to coarsest.
    """
    if isinstance(value, (list, tuple)):
        factors = [float(item) for item in value]
    else:
        factors = [float(value)]
    if not factors:
        raise ValueError("downsample_factor must not be empty.")
    for factor in factors:
        if factor < 1.0:
            raise ValueError("downsample_factor values must be >= 1.0.")
    return sorted(set(factors))


def resolve_fusion_params(inference_cfg: dict) -> dict:
    """Merge the optional `fusion` config block over the built-in defaults.

    Args:
        inference_cfg (dict): Parsed inference YAML contents.

    Returns:
        dict: Fusion parameters with every expected key populated.
    """
    params = dict(DEFAULT_FUSION_PARAMS)
    user_params = inference_cfg.get("fusion") or {}
    unknown_keys = set(user_params) - set(DEFAULT_FUSION_PARAMS)
    if unknown_keys:
        raise ValueError(f"Unknown fusion parameters: {sorted(unknown_keys)}")
    params.update(user_params)
    params["iou_threshold"] = float(params["iou_threshold"])
    params["containment_threshold"] = float(params["containment_threshold"])
    params["min_coverage_ratio"] = float(params["min_coverage_ratio"])
    params["min_votes"] = int(params["min_votes"])
    return params


def segment_at_scale(
    image: np.ndarray,
    engine: PanopticDeepLabEngine,
    *,
    downsample_factor: float,
    in_channels: int,
    mean: float,
    std: float,
    device: str,
) -> np.ndarray:
    """Segment one image at a single downsample factor, restored to full resolution.

    Args:
        image (np.ndarray): Full-resolution source image.
        engine (PanopticDeepLabEngine): Initialized panoptic inference engine.
        downsample_factor (float): Factor applied before inference.
        in_channels (int): Channel count expected by the model.
        mean (float): Normalization mean.
        std (float): Normalization standard deviation.
        device (str): Torch device used for inference.

    Returns:
        np.ndarray: Instance label map at the original image resolution.
    """
    orig_h, orig_w = image.shape[:2]
    scaled_image = resize_array(image, downsample_factor, is_mask=False)

    input_tensor = to_model_input(scaled_image, in_channels, mean, std).to(device)
    with torch.no_grad():
        pan_pred = engine(input_tensor)

    labels = pan_pred.squeeze().detach().cpu().numpy().astype(np.int32)
    if labels.shape[:2] != (orig_h, orig_w):
        labels = cv2.resize(labels, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return labels.astype(np.uint32)


def compute_label_areas(labels: np.ndarray) -> np.ndarray:
    """Count pixels per label ID, including background at index 0.

    Args:
        labels (np.ndarray): Integer label map where 0 is background.

    Returns:
        np.ndarray: Pixel counts indexed by label ID.
    """
    return np.bincount(labels.ravel().astype(np.int64))


def compute_overlap_counts(labels_a: np.ndarray, labels_b: np.ndarray) -> np.ndarray:
    """Count overlapping pixels for every co-occurring instance pair.

    Args:
        labels_a (np.ndarray): First instance label map.
        labels_b (np.ndarray): Second instance label map with identical shape.

    Returns:
        np.ndarray: Array of shape ``(n_pairs, 3)`` holding label A, label B, and
        their shared pixel count.
    """
    overlap_mask = (labels_a > 0) & (labels_b > 0)
    if not np.any(overlap_mask):
        return np.empty((0, 3), dtype=np.int64)

    values_a = labels_a[overlap_mask].astype(np.int64)
    values_b = labels_b[overlap_mask].astype(np.int64)
    stride = int(values_b.max()) + 1
    keys, counts = np.unique(values_a * stride + values_b, return_counts=True)
    return np.stack([keys // stride, keys % stride, counts], axis=1)


def find_group_root(parents: dict, node: tuple[int, int]) -> tuple[int, int]:
    """Resolve the union-find root of a node with path compression.

    Args:
        parents (dict): Mapping from node to its current parent node.
        node (tuple[int, int]): Node identified by scale index and label ID.

    Returns:
        tuple[int, int]: Root node of the set containing ``node``.
    """
    root = node
    while parents[root] != root:
        root = parents[root]
    while parents[node] != root:
        parents[node], node = root, parents[node]
    return root


def group_instances_across_scales(
    label_maps: list[np.ndarray],
    areas_by_scale: list[np.ndarray],
    *,
    iou_threshold: float,
    containment_threshold: float,
) -> list[list[tuple[int, int]]]:
    """Group instances from every scale that describe the same object.

    Instances from different scales are linked when they overlap strongly, either
    by intersection-over-union or because one is largely contained in the other.
    Containment links absorb the fragments a finer scale produces for a single
    large object. Linked instances are merged transitively, so every group is one
    fused object and instances detected by only one scale form their own group,
    which keeps the result a union over all scales.

    Args:
        label_maps (list[np.ndarray]): Full-resolution label map per scale.
        areas_by_scale (list[np.ndarray]): Per-scale pixel counts indexed by label ID.
        iou_threshold (float): Minimum intersection-over-union that links two instances.
        containment_threshold (float): Minimum fraction of the smaller instance that
            must fall inside the larger one to link them.

    Returns:
        list[list[tuple[int, int]]]: Groups of ``(scale_index, label_id)`` members.
    """
    parents: dict[tuple[int, int], tuple[int, int]] = {}
    for scale_index, areas in enumerate(areas_by_scale):
        for label_id in np.nonzero(areas)[0]:
            if label_id == 0:
                continue
            node = (scale_index, int(label_id))
            parents[node] = node

    for index_a in range(len(label_maps)):
        for index_b in range(index_a + 1, len(label_maps)):
            pairs = compute_overlap_counts(label_maps[index_a], label_maps[index_b])
            for label_a, label_b, intersection in pairs:
                area_a = float(areas_by_scale[index_a][label_a])
                area_b = float(areas_by_scale[index_b][label_b])
                union = area_a + area_b - float(intersection)
                iou = (float(intersection) / union) if union > 0 else 0.0
                smaller_area = min(area_a, area_b)
                containment = (
                    float(intersection) / smaller_area if smaller_area > 0 else 0.0
                )
                if iou < iou_threshold and containment < containment_threshold:
                    continue
                root_a = find_group_root(parents, (index_a, int(label_a)))
                root_b = find_group_root(parents, (index_b, int(label_b)))
                if root_a != root_b:
                    parents[root_b] = root_a

    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for node in parents:
        groups.setdefault(find_group_root(parents, node), []).append(node)
    return list(groups.values())


def select_group_representative(
    group: Sequence[tuple[int, int]],
    areas_by_scale: list[np.ndarray],
    *,
    min_coverage_ratio: float,
) -> tuple[int, int]:
    """Pick the scale that best represents one fused object.

    The finest scale is preferred, because it resolves boundaries most sharply,
    but only when it segmented the object as a single instance covering enough of
    the object's largest extent across scales. A scale that fragmented the object
    contributes several members and is skipped, which is what pushes large objects
    onto coarser scales. When no scale qualifies, the single instance with the
    largest area wins.

    Args:
        group (Sequence[tuple[int, int]]): ``(scale_index, label_id)`` members of one object.
        areas_by_scale (list[np.ndarray]): Per-scale pixel counts indexed by label ID.
        min_coverage_ratio (float): Minimum area fraction, relative to the best-covering
            scale, that a scale must reach to be accepted.

    Returns:
        tuple[int, int]: The ``(scale_index, label_id)`` chosen to represent the object.
    """
    members_by_scale: dict[int, list[int]] = {}
    for scale_index, label_id in group:
        members_by_scale.setdefault(scale_index, []).append(label_id)

    coverage_by_scale = {
        scale_index: sum(float(areas_by_scale[scale_index][label]) for label in labels)
        for scale_index, labels in members_by_scale.items()
    }
    best_coverage = max(coverage_by_scale.values())

    for scale_index in sorted(members_by_scale):
        labels = members_by_scale[scale_index]
        if len(labels) != 1:
            continue
        if coverage_by_scale[scale_index] < min_coverage_ratio * best_coverage:
            continue
        return (scale_index, labels[0])

    return max(group, key=lambda node: float(areas_by_scale[node[0]][node[1]]))


def fuse_label_maps(
    label_maps: list[np.ndarray],
    *,
    iou_threshold: float,
    containment_threshold: float,
    min_coverage_ratio: float,
    min_votes: int,
) -> np.ndarray:
    """Fuse per-scale label maps into a single instance segmentation.

    Args:
        label_maps (list[np.ndarray]): Full-resolution label map per scale, ordered
            from finest to coarsest.
        iou_threshold (float): Minimum intersection-over-union that links two instances.
        containment_threshold (float): Minimum containment fraction that links two instances.
        min_coverage_ratio (float): Minimum relative coverage for a scale to represent
            a fused object.
        min_votes (int): Minimum number of scales that must detect an object for it to
            be kept. A value of 1 keeps the full union.

    Returns:
        np.ndarray: Fused instance label map relabeled from 1 to N.
    """
    if len(label_maps) == 1:
        return label_maps[0]

    areas_by_scale = [compute_label_areas(labels) for labels in label_maps]
    slices_by_scale = [find_objects(labels.astype(np.int32)) for labels in label_maps]

    groups = group_instances_across_scales(
        label_maps,
        areas_by_scale,
        iou_threshold=iou_threshold,
        containment_threshold=containment_threshold,
    )

    selected: list[tuple[int, int]] = []
    for group in groups:
        if len({scale_index for scale_index, _ in group}) < min_votes:
            continue
        selected.append(
            select_group_representative(
                group, areas_by_scale, min_coverage_ratio=min_coverage_ratio
            )
        )

    # Paint the largest objects first so smaller neighbors keep contested pixels
    # instead of being swallowed by an overlapping coarse-scale instance.
    selected.sort(
        key=lambda node: float(areas_by_scale[node[0]][node[1]]), reverse=True
    )

    fused = np.zeros(label_maps[0].shape[:2], dtype=np.uint32)
    for next_label, (scale_index, label_id) in enumerate(selected, start=1):
        bounds = slices_by_scale[scale_index][label_id - 1]
        if bounds is None:
            continue
        window = label_maps[scale_index][bounds]
        fused[bounds][window == label_id] = next_label

    # Objects fully overwritten during painting leave gaps, so relabel densely.
    remaining = np.unique(fused)
    remaining = remaining[remaining > 0]
    if remaining.size == 0:
        return fused
    lookup = np.zeros(int(fused.max()) + 1, dtype=np.uint32)
    lookup[remaining] = np.arange(1, remaining.size + 1, dtype=np.uint32)
    return lookup[fused]


def resolve_inference_images(input_dir: Path, input_file: Optional[str]) -> list[Path]:
    """Resolve image paths for inference from config settings.

    Args:
        input_dir (Path): Root directory used for recursive image discovery.
        input_file (Optional[str]): Optional single TIFF path to process. Relative paths are
            resolved against ``input_dir``.

    Returns:
        list[Path]: List of TIFF image paths to process.
    """
    if input_file:
        single_path = Path(input_file)
        if not single_path.is_absolute():
            single_path = input_dir / single_path
        if not single_path.is_file():
            raise FileNotFoundError(f"Configured input_file not found: {single_path}")
        if single_path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"Configured input_file must be a TIFF: {single_path}")
        return [single_path]

    return sorted(
        [
            p
            for p in input_dir.rglob("*")
            if p.suffix.lower() in {".tif", ".tiff"} and p.name.endswith("_corrected.tif")
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse MitoNet inference command-line arguments.

    Args:
        None: This function reads arguments from the command line.

    Returns:
        argparse.Namespace: Parsed arguments containing the inference config path.
    """
    parser = argparse.ArgumentParser(
        description="Run MitoNet inference using a selected YAML configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=INFERENCE_CONFIG,
        help=f"Inference YAML path (default: {INFERENCE_CONFIG}).",
    )
    return parser.parse_args()


def main(config_path: Path = INFERENCE_CONFIG) -> None:
    """Run batch inference using a YAML configuration.

    Args:
        config_path (Path): Path to the inference YAML configuration.

    Returns:
        None: This function writes segmentation images and metric CSV files.
    """
    # Load inference configuration and validate required fields.
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing inference config: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        inference_cfg = yaml.safe_load(handle)

    paths_cfg = inference_cfg["paths"]
    input_dir = Path(paths_cfg["input_dir"])
    input_file = paths_cfg.get("input_file")
    model_pth = Path(paths_cfg["model_pth"])
    config_yaml = Path(paths_cfg["config_yaml"])
    downsample_factors = resolve_downsample_factors(paths_cfg["downsample_factor"])
    device = str(paths_cfg["device"])
    engine_params = inference_cfg["engine_params"]
    fusion_params = resolve_fusion_params(inference_cfg)

    if device != "cuda":
        raise ValueError("device must be set to 'cuda' to run on GPU.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a GPU-enabled setup.")

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

    # Resolve either one configured image or all corrected TIFFs under the input directory.
    image_paths = resolve_inference_images(input_dir, input_file)
    if not image_paths:
        raise FileNotFoundError(
            "No inference images found. Set paths.input_file or ensure input_dir "
            "contains '*_corrected.tif' files."
        )

    total = len(image_paths)
    scales_text = ", ".join(f"{factor:g}" for factor in downsample_factors)
    print(f"Running inference at downsample factor(s): {scales_text}")

    for idx, image_path in enumerate(image_paths, start=1):
        output_path = build_output_path(image_path)

        print(f"Processing image {idx}/{total}: {image_path.name}")

        image = tiff.imread(str(image_path))

        # Segment at every configured scale, then fuse into one instance map.
        label_maps = [
            segment_at_scale(
                image,
                engine,
                downsample_factor=factor,
                in_channels=in_channels,
                mean=mean,
                std=std,
                device=device,
            )
            for factor in downsample_factors
        ]
        pan_np = fuse_label_maps(label_maps, **fusion_params)

        metrics = compute_instance_metrics(pan_np)
        metrics_path = output_path.with_name(f"{output_path.stem}_metrics.csv")
        write_metrics_csv(metrics_path, metrics)

        # Convert to colorful visualization.
        color = colorize_labels(pan_np)

        # Write output beside the input image.
        write_tiff(output_path, color)


if __name__ == "__main__":
    main(parse_args().config)
