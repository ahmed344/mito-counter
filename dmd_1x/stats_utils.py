"""DMD_1X-local plotting helpers for boxplots and Bayesian superplots."""

from __future__ import annotations

from pathlib import Path
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

SUPERPLOT_GROUP_WIDTH = 1.0
SUPERPLOT_HALF_WIDTH = 0.42
SUPERPLOT_GRID_SIZE = 256
SUPERPLOT_POINT_SIZE_MIN = 5
SUPERPLOT_POINT_SIZE_MAX = 28
SUPERPLOT_POINT_ALPHA = 0.4
ROBUST_Y_LOWER_QUANTILE = 0.01
ROBUST_Y_UPPER_QUANTILE = 0.99
SUPERPLOT_ANNOTATION_BASE_Y = 1.0
SUPERPLOT_ANNOTATION_BRACKET_HEIGHT = 0.025
SUPERPLOT_ANNOTATION_TEXT_OFFSET = 0.008
SUPERPLOT_ANNOTATION_TEXT_LINE_GAP = 0.055
SUPERPLOT_ANNOTATION_STACK_GAP = 0.065
SUPERPLOT_ANNOTATION_FONT_SIZE = 10
SUPERPLOT_ANNOTATION_HDI_FONT_SIZE = 9
SUPERPLOT_ANNOTATION_HDI_COLOR = "0.18"
MUSCLE_CONDITION_SHORT_LABELS = ("EOM", "TA")
MUSCLE_LABEL_ALIASES = {
    "extraocular muscle": "EOM",
    "eom": "EOM",
    "tibialis anterior": "TA",
    "ta": "TA",
}


def build_output_path(
    y: str,
    x: str,
    hue: str,
    save_dir: str | Path | None,
    suffix: str,
    filename_prefix: str | None = None,
    output_dir_suffix: str = "",
) -> Path | None:
    """Construct a superplot output path under a plot-type directory.

    Args:
        y (str): Numeric metric column name.
        x (str): X-axis grouping column name.
        hue (str): Hue grouping column name.
        save_dir (str | Path | None): Base output directory.
        suffix (str): Plot-type suffix, e.g. ``superviolin``.
        filename_prefix (str | None): Optional filename prefix before suffix.
        output_dir_suffix (str): Optional suffix appended to the output directory
            name, e.g. ``_muscle`` to write into ``super_violins_muscle``.

    Returns:
        Path | None: Full output path when ``save_dir`` is provided, otherwise ``None``.
    """

    if save_dir is None:
        return None
    directory_by_suffix = {
        "superviolin": "super_violins",
        "superbeeswarm": "super_beeswarms",
    }
    directory_name = directory_by_suffix.get(suffix, "")
    output_dir = Path(save_dir) / f"{directory_name}{str(output_dir_suffix).strip()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if filename_prefix is not None:
        safe_prefix = str(filename_prefix).strip().replace("/", "_").replace(" ", "_")
        return output_dir / f"{safe_prefix}{suffix}.png"
    safe_parts = [str(part).replace("/", "_").replace(" ", "_") for part in (y, x, hue, suffix)]
    return output_dir / f"{safe_parts[0]}_by_{safe_parts[1]}_and_{safe_parts[2]}_{safe_parts[3]}.png"


def sort_condition_values(values: list[str]) -> list[str]:
    """Sort condition labels with WT-like labels before DMD/KO-like labels.

    Args:
        values (list[str]): Condition labels extracted from a dataframe.

    Returns:
        list[str]: Sorted labels with Wildtype-like values first.
    """

    if is_muscle_condition_labels(values=values):
        return sort_muscle_condition_values(values=values)

    return sorted(
        values,
        key=lambda value: (
            0
            if (
                "wildtype" in str(value).lower()
                or str(value).strip().lower() == "wt"
                or str(value).strip().lower().endswith("_wt")
            )
            else 1
            if (
                "dystrophy" in str(value).lower()
                or str(value).strip().lower() == "dmd"
                or str(value).strip().lower().endswith("_dmd")
                or "knockout" in str(value).lower()
                or str(value).strip().lower() == "ko"
                or str(value).strip().lower().endswith("_ko")
            )
            else 2,
            str(value).lower(),
        ),
    )


def normalize_muscle_condition_label(condition_value: str) -> str | None:
    """Map raw condition labels to muscle short labels when possible.

    Args:
        condition_value (str): Raw condition label from data or config.

    Returns:
        str | None: ``EOM`` or ``TA`` when the input is a known muscle label,
        otherwise ``None``.
    """

    normalized = str(condition_value).strip().lower()
    return MUSCLE_LABEL_ALIASES.get(normalized)


def is_muscle_condition_labels(values: list[str]) -> bool:
    """Return whether all provided labels are recognized muscle condition labels.

    Args:
        values (list[str]): Condition labels extracted from plotting data.

    Returns:
        bool: ``True`` when every label is recognized as EOM/TA and at least one
        label is present.
    """

    if not values:
        return False
    normalized_values = [normalize_muscle_condition_label(condition_value=value) for value in values]
    return all(label is not None for label in normalized_values)


def sort_muscle_condition_values(values: list[str]) -> list[str]:
    """Sort muscle condition labels deterministically with EOM before TA.

    Args:
        values (list[str]): Raw muscle condition labels.

    Returns:
        list[str]: Sorted labels preserving original string forms.
    """

    order_map = {label: index for index, label in enumerate(MUSCLE_CONDITION_SHORT_LABELS)}
    return sorted(
        values,
        key=lambda value: (
            order_map.get(
                normalize_muscle_condition_label(condition_value=value) or "",
                len(order_map),
            ),
            str(value).lower(),
        ),
    )


def sort_block_values(values: list[object]) -> list[str]:
    """Sort block labels numerically when possible.

    Args:
        values (list[object]): Raw block labels.

    Returns:
        list[str]: Sorted string labels.
    """

    normalized = [str(value) for value in values]
    try:
        return sorted(normalized, key=lambda value: (0, int(value)))
    except ValueError:
        return sorted(normalized, key=lambda value: (1, value))


def prepare_plot_data(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    block: str | None = None,
) -> pd.DataFrame:
    """Coerce numeric columns and drop incomplete rows for plotting.

    Args:
        data (pd.DataFrame): Source dataframe.
        x (str): X-axis grouping column.
        y (str): Numeric metric column.
        hue (str): Hue grouping column.
        block (str | None): Optional replicate-block column.

    Returns:
        pd.DataFrame: Clean plotting dataframe.
    """

    plot_data = data.copy()
    plot_data[y] = pd.to_numeric(plot_data[y], errors="coerce")
    required_columns = [x, y, hue]
    if block is not None:
        required_columns.append(block)
    plot_data = plot_data.dropna(subset=required_columns)
    if block is not None and block in plot_data.columns:
        plot_data[block] = plot_data[block].astype(str)
    return plot_data


def format_unit_label(metric_name: str, unit_dict: dict[str, str] | None) -> str:
    """Build a formatted unit suffix for axis labels.

    Args:
        metric_name (str): Metric key shown on y-axis.
        unit_dict (dict[str, str] | None): Optional metric-to-unit mapping.

    Returns:
        str: Unit suffix in parentheses, or an empty string.
    """

    if not unit_dict or metric_name not in unit_dict or not unit_dict[metric_name]:
        return ""
    unit_text = unit_dict[metric_name]
    if unit_text == "um":
        formatted_unit = r"$\mu m$"
    elif unit_text == "um^2":
        formatted_unit = r"$\mu m^2$"
    elif unit_text == "nm":
        formatted_unit = r"$nm$"
    elif unit_text == "nm^2":
        formatted_unit = r"$nm^2$"
    else:
        formatted_unit = unit_text.replace("^2", r"$^2$")
    return f" ({formatted_unit})"


def get_category_orders(
    plot_data: pd.DataFrame,
    x: str,
    hue: str,
    block: str | None = None,
    x_order_override: list[str] | None = None,
    hue_order_override: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Determine stable x/hue/block plotting order.

    Args:
        plot_data (pd.DataFrame): Plot-ready dataframe.
        x (str): X-axis grouping column.
        hue (str): Hue grouping column.
        block (str | None): Optional replicate-block column.
        x_order_override (list[str] | None): Explicit x order when provided.
        hue_order_override (list[str] | None): Explicit hue order when provided.

    Returns:
        tuple[list[str], list[str], list[str]]: Orders for x, hue, and block.
    """

    if x_order_override is None:
        x_order = sorted(plot_data[x].astype(str).unique().tolist())
    else:
        data_values = set(plot_data[x].astype(str).unique().tolist())
        x_order = [value for value in x_order_override if str(value) in data_values]

    if hue_order_override is None:
        if hue == "Condition":
            hue_order = sort_condition_values(plot_data[hue].astype(str).unique().tolist())
        else:
            hue_order = sorted(plot_data[hue].astype(str).unique().tolist())
    else:
        data_values = set(plot_data[hue].astype(str).unique().tolist())
        hue_order = [value for value in hue_order_override if str(value) in data_values]

    block_order: list[str] = []
    if block is not None and block in plot_data.columns:
        block_order = sort_block_values(plot_data[block].astype(str).unique().tolist())
    return x_order, hue_order, block_order


def create_condition_block_palette(
    block_order: list[str],
    hue_order: list[str],
) -> dict[tuple[str, str], tuple[float, float, float]]:
    """Create colors keyed by both condition and block number.

    Args:
        block_order (list[str]): Ordered block labels.
        hue_order (list[str]): Ordered condition labels.

    Returns:
        dict[tuple[str, str], tuple[float, float, float]]: Mapping from ``(hue, block)``
        to RGB color, with WT-like hues using blue shades and DMD/KO-like hues using
        orange shades.
    """

    if not block_order or not hue_order:
        return {}

    condition_palette_map: dict[str, list[tuple[float, float, float]]] = {}
    for hue_value in hue_order:
        condition_key = format_condition_legend_label(hue_value)
        if condition_key in {"WT", "EOM"}:
            condition_palette_map[hue_value] = list(
                sns.blend_palette(
                    ["#041B4D", "#08519C", "#2171B5", "#6BAED6", "#C6DBEF"],
                    n_colors=len(block_order),
                )
            )
        elif condition_key in {"DMD", "KO", "TA"}:
            condition_palette_map[hue_value] = list(
                sns.blend_palette(
                    ["#5A1A02", "#A63603", "#D94801", "#F16913", "#FDBE85"],
                    n_colors=len(block_order),
                )
            )
        else:
            condition_palette_map[hue_value] = list(
                sns.color_palette("Greys", len(block_order) + 2)[2:]
            )

    return {
        (str(hue_value), str(block_label)): condition_palette_map[hue_value][block_index]
        for hue_value in hue_order
        for block_index, block_label in enumerate(block_order)
    }


def format_condition_legend_label(condition_value: str) -> str:
    """Convert raw condition names into compact legend labels.

    Args:
        condition_value (str): Raw condition label.

    Returns:
        str: Short legend-friendly label.
    """

    muscle_label = normalize_muscle_condition_label(condition_value=condition_value)
    if muscle_label is not None:
        return muscle_label
    normalized = str(condition_value).strip()
    lower_value = normalized.lower()
    if "wildtype" in lower_value or lower_value == "wt" or lower_value.endswith("_wt"):
        return "WT"
    if "dystrophy" in lower_value or lower_value == "dmd" or lower_value.endswith("_dmd"):
        return "DMD"
    if "knockout" in lower_value or lower_value == "ko" or lower_value.endswith("_ko"):
        return "KO"
    return normalized.replace("_", " ")


def format_condition_display_label(condition_value: str) -> str:
    """Format condition labels for axis tick display.

    Args:
        condition_value (str): Raw condition label from plotting data.

    Returns:
        str: Compact condition label for plot text.
    """

    muscle_label = normalize_muscle_condition_label(condition_value=condition_value)
    if muscle_label is not None:
        return muscle_label
    normalized = str(condition_value).strip()
    lower_value = normalized.lower()
    if "wildtype" in lower_value or lower_value == "wt" or lower_value.endswith("_wt"):
        return "WT"
    if "dystrophy" in lower_value or lower_value == "dmd" or lower_value.endswith("_dmd"):
        return "DMD"
    if "knockout" in lower_value or lower_value == "ko" or lower_value.endswith("_ko"):
        return "KO"
    return normalized.replace("_", " ")


def build_block_legend_labels(
    plot_data: pd.DataFrame,
    block: str,
    hue: str,
    block_order: list[str],
) -> dict[tuple[str, str], str]:
    """Create readable legend labels for colored replicate blocks.

    Args:
        plot_data (pd.DataFrame): Plot dataframe containing block and hue columns.
        block (str): Block column name.
        hue (str): Hue column name.
        block_order (list[str]): Ordered block labels.

    Returns:
        dict[tuple[str, str], str]: Mapping from ``(hue, block)`` to legend text.
    """

    labels: dict[tuple[str, str], str] = {}
    for block_label in block_order:
        rows = plot_data.loc[plot_data[block].astype(str) == block_label].copy()
        if rows.empty:
            continue
        present_hues = sorted(rows[hue].astype(str).unique().tolist(), key=lambda value: value)
        for hue_value in present_hues:
            labels[(str(hue_value), str(block_label))] = (
                f"{format_condition_legend_label(hue_value)} {block_label}"
            )
    return labels


def get_group_centers(x_order: list[str], hue_order: list[str]) -> dict[tuple[str, str], float]:
    """Compute numeric x-centers for each x/hue pair.

    Args:
        x_order (list[str]): X categories in plotting order.
        hue_order (list[str]): Hue categories in plotting order.

    Returns:
        dict[tuple[str, str], float]: Mapping from ``(x, hue)`` to x-axis center.
    """

    base_spacing = max(1.6, len(hue_order) * SUPERPLOT_GROUP_WIDTH)
    if len(hue_order) <= 1:
        offsets = [0.0]
    else:
        offsets = np.linspace(-SUPERPLOT_GROUP_WIDTH / 2, SUPERPLOT_GROUP_WIDTH / 2, len(hue_order))
    return {
        (str(x_value), str(hue_value)): x_index * base_spacing + float(offset)
        for x_index, x_value in enumerate(x_order)
        for hue_value, offset in zip(hue_order, offsets)
    }


def build_density_grid(values: np.ndarray) -> np.ndarray:
    """Build a KDE evaluation grid around observed values.

    Args:
        values (np.ndarray): Numeric values from one plotted group.

    Returns:
        np.ndarray: Dense y-grid spanning the observed range with padding.
    """

    values = np.asarray(values, dtype=float)
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    spread = value_max - value_min
    if spread == 0:
        padding = max(abs(value_min) * 0.1, 1.0)
    else:
        padding = max(spread * 0.1, np.std(values) * 0.3, 1e-3)
    return np.linspace(value_min - padding, value_max + padding, SUPERPLOT_GRID_SIZE)


def estimate_density(values: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """Estimate non-negative density values on a y-grid.

    Args:
        values (np.ndarray): Numeric observations.
        y_grid (np.ndarray): Shared y-grid.

    Returns:
        np.ndarray: Density values evaluated on ``y_grid``.
    """

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros_like(y_grid, dtype=float)
    if np.unique(values).size == 1:
        bandwidth = max(np.ptp(y_grid) * 0.03, 1e-3)
        return np.exp(-0.5 * ((y_grid - values[0]) / bandwidth) ** 2)
    try:
        density = gaussian_kde(values)(y_grid)
    except (np.linalg.LinAlgError, ValueError):
        bandwidth = max(np.std(values) * 0.3, np.ptp(y_grid) * 0.03, 1e-3)
        density = np.zeros_like(y_grid, dtype=float)
        for value in values:
            density += np.exp(-0.5 * ((y_grid - value) / bandwidth) ** 2)
    return np.clip(density, a_min=0, a_max=None)


def apply_robust_y_limits(ax: plt.Axes, values: np.ndarray | pd.Series | list[float]) -> None:
    """Set robust y-limits using quantiles to reduce outlier dominance.

    Args:
        ax (plt.Axes): Target axes for y-limit updates.
        values (np.ndarray | pd.Series | list[float]): Numeric values used to compute limits.

    Returns:
        None: This function mutates axis limits in place.
    """

    value_array = np.asarray(values, dtype=float)
    value_array = value_array[np.isfinite(value_array)]
    if value_array.size == 0:
        return
    if value_array.size < 20:
        value_min = float(np.min(value_array))
        value_max = float(np.max(value_array))
    else:
        value_min = float(np.quantile(value_array, ROBUST_Y_LOWER_QUANTILE))
        value_max = float(np.quantile(value_array, ROBUST_Y_UPPER_QUANTILE))
    spread = max(value_max - value_min, max(abs(value_min) * 0.1, 1.0))
    padding = max(spread * 0.08, 1e-6)
    ax.set_ylim(value_min - padding, value_max + padding)


def get_dynamic_beeswarm_point_size(group_count: int, max_group_count: int) -> float:
    """Scale beeswarm marker size inversely with group density.

    Args:
        group_count (int): Number of points in the current group.
        max_group_count (int): Maximum group size across the subplot.

    Returns:
        float: Marker area for Matplotlib scatter.
    """

    if group_count <= 1 or max_group_count <= 1:
        return float(SUPERPLOT_POINT_SIZE_MAX)
    sparse_count_scale = np.clip((np.log10(max(group_count, 1)) - 1.0) / 2.0, 0.0, 1.0)
    relative_density = np.sqrt(group_count / max_group_count)
    density_scale = max(sparse_count_scale, 0.35 * relative_density)
    marker_size = SUPERPLOT_POINT_SIZE_MAX - ((SUPERPLOT_POINT_SIZE_MAX - SUPERPLOT_POINT_SIZE_MIN) * density_scale)
    return float(np.clip(marker_size, SUPERPLOT_POINT_SIZE_MIN, SUPERPLOT_POINT_SIZE_MAX))


def build_violin_like_swarm_positions(
    group_data: pd.DataFrame,
    y: str,
    y_grid: np.ndarray,
    half_width: np.ndarray,
    center: float,
    seed: int,
) -> np.ndarray:
    """Compute beeswarm x positions that follow a violin-like profile.

    Args:
        group_data (pd.DataFrame): Rows in one x/hue group.
        y (str): Numeric metric column name.
        y_grid (np.ndarray): Shared y-grid for density interpolation.
        half_width (np.ndarray): Group half-width per y-grid value.
        center (float): X-axis center of the current group.
        seed (int): Random seed for stable jittering.

    Returns:
        np.ndarray: X coordinates for each row in ``group_data``.
    """

    values = group_data[y].to_numpy(dtype=float)
    if values.size == 0:
        return np.array([], dtype=float)
    bin_count = max(12, min(45, int(np.sqrt(values.size) * 2)))
    bin_edges = np.linspace(float(np.min(y_grid)), float(np.max(y_grid)), bin_count + 1)
    bin_ids = np.clip(np.digitize(values, bin_edges) - 1, 0, bin_count - 1)
    point_x = np.full(values.shape, center, dtype=float)
    rng = np.random.default_rng(seed)
    for bin_id in np.unique(bin_ids):
        bin_indices = np.where(bin_ids == bin_id)[0]
        if bin_indices.size == 0:
            continue
        local_half_width = float(np.interp(np.mean(values[bin_indices]), y_grid, half_width))
        if local_half_width <= 0:
            continue
        if bin_indices.size == 1:
            offsets = np.array([0.0], dtype=float)
        else:
            offsets = np.linspace(-local_half_width, local_half_width, bin_indices.size)
            offsets += rng.uniform(-local_half_width * 0.08, local_half_width * 0.08, size=bin_indices.size)
        shuffled_indices = bin_indices[rng.permutation(bin_indices.size)]
        point_x[shuffled_indices] = center + offsets
    return point_x


def add_superplot_summary(
    ax: plt.Axes,
    group_data: pd.DataFrame,
    y: str,
    block: str,
    hue_value: str,
    center: float,
    condition_block_palette: dict[tuple[str, str], tuple[float, float, float]],
) -> None:
    """Overlay per-block means and overall mean +/- SEM markers.

    Args:
        ax (plt.Axes): Axes receiving summary markers.
        group_data (pd.DataFrame): Group rows for a single x/hue pair.
        y (str): Numeric metric column.
        block (str): Replicate block column.
        hue_value (str): Condition label for the current group.
        center (float): X-axis center of the group.
        condition_block_palette (dict[tuple[str, str], tuple[float, float, float]]):
            Colors keyed by ``(hue, block)``.

    Returns:
        None: This function draws artists on ``ax``.
    """

    block_means: list[float] = []
    available_blocks = sorted(
        group_data[block].astype(str).unique(),
        key=lambda value: ((str(hue_value), value) not in condition_block_palette, value),
    )
    offsets = [0.0] if len(available_blocks) == 1 else np.linspace(-0.09, 0.09, len(available_blocks))
    for block_label, offset in zip(available_blocks, offsets):
        block_values = group_data.loc[group_data[block].astype(str) == block_label, y].dropna().to_numpy()
        if block_values.size == 0:
            continue
        block_mean = float(np.mean(block_values))
        block_means.append(block_mean)
        ax.scatter(
            center + float(offset),
            block_mean,
            s=30,
            marker="D",
            color=condition_block_palette.get((str(hue_value), str(block_label)), "0.4"),
            edgecolor="black",
            linewidth=0.5,
            zorder=6,
        )
    if not block_means:
        return
    overall_mean = float(np.mean(block_means))
    sem = float(np.std(block_means, ddof=1) / np.sqrt(len(block_means))) if len(block_means) > 1 else 0.0
    ax.errorbar(
        center,
        overall_mean,
        yerr=sem if sem > 0 else None,
        fmt="o",
        color="black",
        markersize=5,
        capsize=3,
        linewidth=1.3,
        zorder=7,
    )


def normalize_superplot_annotations(
    superplot_annotations: dict[str, str] | list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    """Normalize optional annotation inputs into structured records.

    Args:
        superplot_annotations (dict[str, str] | list[dict[str, str]] | None): Input annotations.

    Returns:
        list[dict[str, str]]: Normalized annotation records.
    """

    if superplot_annotations is None:
        return []
    if isinstance(superplot_annotations, dict):
        return [
            {"x": str(x_value), "label": str(label)}
            for x_value, label in superplot_annotations.items()
            if str(label).strip()
        ]
    return [
        {str(key): str(value) for key, value in record.items()}
        for record in superplot_annotations
        if (
            str(record.get("label", "")).strip()
            or str(record.get("mean_label", "")).strip()
            or str(record.get("median_label", "")).strip()
        )
    ]


def superplot_annotation_text_items(record: dict[str, str]) -> list[tuple[str, str]]:
    """Extract annotation text lines and colors in drawing order.

    Args:
        record (dict[str, str]): Annotation record with label or mean/median fields.

    Returns:
        list[tuple[str, str]]: Sequence of ``(text, color)`` entries.
    """

    median_label = str(record.get("median_label", "")).strip()
    if median_label:
        return [
            (
                median_label,
                str(record.get("median_color", record.get("color", "black"))).strip() or "black",
            )
        ]
    label = str(record.get("label", "")).strip()
    if not label:
        return []
    return [(label, str(record.get("color", "black")).strip() or "black")]


def split_effect_summary_annotation(label: str) -> tuple[str, str, str] | None:
    """Split effect-summary text into estimate, interval, and probability spans.

    Args:
        label (str): Annotation text, e.g. ``-0.2 [-0.4, -0.1] 97.2%``.

    Returns:
        tuple[str, str, str] | None: Prefix, bracket interval, and suffix if parsable.
    """

    text = str(label)
    left_bracket = text.find("[")
    right_bracket = text.find("]", left_bracket + 1)
    if left_bracket < 0 or right_bracket < 0:
        return None
    prefix = text[:left_bracket].rstrip()
    hdi_text = text[left_bracket : right_bracket + 1]
    suffix = text[right_bracket + 1 :].lstrip()
    if not prefix or not suffix:
        return None
    return prefix, hdi_text, suffix


def draw_superplot_annotation_text(
    ax: plt.Axes,
    x_center: float,
    y_position: float,
    label: str,
    text_color: str,
    transform,
) -> None:
    """Draw one annotation line with optional HDI styling.

    Args:
        ax (plt.Axes): Destination axes.
        x_center (float): Text center x-coordinate.
        y_position (float): Text baseline y-coordinate in transform space.
        label (str): Annotation line.
        text_color (str): Main text color.
        transform: Matplotlib transform used for x-axis-relative placement.

    Returns:
        None: This function adds text artists to ``ax``.
    """

    effect_parts = split_effect_summary_annotation(label)
    if effect_parts is None:
        ax.text(
            x_center,
            y_position,
            label,
            ha="center",
            va="bottom",
            fontsize=SUPERPLOT_ANNOTATION_FONT_SIZE,
            color=text_color,
            transform=transform,
            clip_on=False,
            zorder=11,
        )
        return
    prefix, hdi_text, suffix = effect_parts
    text_box = HPacker(
        children=[
            TextArea(f"{prefix} ", textprops={"fontsize": SUPERPLOT_ANNOTATION_FONT_SIZE, "color": text_color}),
            TextArea(hdi_text, textprops={"fontsize": SUPERPLOT_ANNOTATION_HDI_FONT_SIZE, "color": SUPERPLOT_ANNOTATION_HDI_COLOR}),
            TextArea(f" {suffix}", textprops={"fontsize": SUPERPLOT_ANNOTATION_FONT_SIZE, "color": text_color}),
        ],
        align="baseline",
        pad=0,
        sep=0,
    )
    annotation_box = AnnotationBbox(
        text_box,
        (x_center, y_position),
        xycoords=transform,
        box_alignment=(0.5, 0.0),
        frameon=False,
        pad=0,
        annotation_clip=False,
        zorder=11,
    )
    ax.add_artist(annotation_box)


def add_superplot_annotations(
    ax: plt.Axes,
    annotation_records: list[dict[str, str]],
    group_centers: dict[tuple[str, str], float],
    hue_order: list[str],
) -> float:
    """Draw bracket annotations and avoid label collisions between nearby groups.

    Args:
        ax (plt.Axes): Axes receiving annotations.
        annotation_records (list[dict[str, str]]): Bracket/text records.
        group_centers (dict[tuple[str, str], float]): X-center map for plotted groups.
        hue_order (list[str]): Hue ordering used in the subplot.

    Returns:
        float: Highest y-position used in axis-transform coordinates.
    """

    if len(hue_order) < 2 or not annotation_records:
        return 1.0
    xaxis_transform = ax.get_xaxis_transform()
    placed_boxes: list[tuple[float, float, float, float]] = []
    annotation_counts_by_x: dict[str, int] = {}
    x_position_by_value: dict[str, float] = {}
    for x_value, _ in group_centers:
        centers = [
            center
            for (candidate_x, _), center in group_centers.items()
            if str(candidate_x) == str(x_value)
        ]
        if centers:
            x_position_by_value[str(x_value)] = float(np.mean(centers))
    x_rank_by_value = {
        x_value: rank
        for rank, x_value in enumerate(
            sorted(x_position_by_value, key=lambda value: x_position_by_value[value])
        )
    }
    max_top = 1.0
    for record in annotation_records:
        x_value = str(record.get("x", ""))
        text_items = superplot_annotation_text_items(record)
        if not x_value or not text_items:
            continue
        bracket_color = str(record.get("bracket_color", "black")).strip() or "black"
        hue_start = str(record.get("hue_start", hue_order[0]))
        hue_end = str(record.get("hue_end", hue_order[1]))
        start_key = (x_value, hue_start)
        end_key = (x_value, hue_end)
        if start_key not in group_centers or end_key not in group_centers:
            continue
        x_start = min(group_centers[start_key], group_centers[end_key])
        x_end = max(group_centers[start_key], group_centers[end_key])
        annotation_index = annotation_counts_by_x.get(x_value, 0)
        annotation_counts_by_x[x_value] = annotation_index + 1
        stagger_index = x_rank_by_value.get(x_value, 0) % 3
        y_line = SUPERPLOT_ANNOTATION_BASE_Y + SUPERPLOT_ANNOTATION_STACK_GAP * (
            annotation_index + stagger_index
        )
        line_count = len(text_items)
        estimated_height = (
            SUPERPLOT_ANNOTATION_BRACKET_HEIGHT
            + SUPERPLOT_ANNOTATION_TEXT_OFFSET
            + SUPERPLOT_ANNOTATION_TEXT_LINE_GAP * max(line_count - 1, 0)
            + 0.04
        )
        while any(
            (
                x_start <= other_x1
                and x_end >= other_x0
                and y_line < other_y1
                and (y_line + estimated_height) > other_y0
            )
            for other_x0, other_x1, other_y0, other_y1 in placed_boxes
        ):
            y_line += SUPERPLOT_ANNOTATION_STACK_GAP
        y_bracket_top = y_line + SUPERPLOT_ANNOTATION_BRACKET_HEIGHT
        ax.plot(
            [x_start, x_start, x_end, x_end],
            [y_line, y_bracket_top, y_bracket_top, y_line],
            color=bracket_color,
            linewidth=1.1,
            transform=xaxis_transform,
            clip_on=False,
            zorder=10,
        )
        y_text = y_bracket_top + SUPERPLOT_ANNOTATION_TEXT_OFFSET
        for text_index, (text_label, text_color) in enumerate(text_items):
            draw_superplot_annotation_text(
                ax=ax,
                x_center=(x_start + x_end) / 2.0,
                y_position=y_text + SUPERPLOT_ANNOTATION_TEXT_LINE_GAP * text_index,
                label=text_label,
                text_color=text_color,
                transform=xaxis_transform,
            )
        top_y = y_text + SUPERPLOT_ANNOTATION_TEXT_LINE_GAP * max(line_count - 1, 0) + 0.035
        placed_boxes.append((x_start, x_end, y_line, top_y))
        max_top = max(max_top, top_y)
    return max_top


def superplot_top_margin(annotation_top: float) -> float:
    """Return a tight layout top margin that preserves annotation headroom.

    Args:
        annotation_top (float): Highest annotation y-position in axis-transform coordinates.

    Returns:
        float: ``tight_layout`` top rect value.
    """

    overflow = max(0.0, float(annotation_top) - 1.0)
    return max(0.94, 0.997 - 0.035 * overflow)


def style_superplot_axis(
    ax: plt.Axes,
    x: str,
    y: str,
    tick_positions: list[float],
    tick_labels: list[str],
    unit_dict: dict[str, str] | None,
    y_values: np.ndarray,
    title: str | None,
) -> None:
    """Apply shared axis styling for one superplot panel.

    Args:
        ax (plt.Axes): Target axis.
        x (str): X-axis variable name.
        y (str): Y-axis variable name.
        tick_positions (list[float]): Group center positions.
        tick_labels (list[str]): Tick labels per center.
        unit_dict (dict[str, str] | None): Optional unit mapping.
        y_values (np.ndarray): Numeric values used for robust y-limits.
        title (str | None): Optional panel title.

    Returns:
        None: This function mutates ``ax``.
    """

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=11)
    ax.set_xlabel(x, fontsize=13)
    ax.set_ylabel(f"{y}{format_unit_label(y, unit_dict)}", fontsize=13)
    if title:
        ax.set_title(title, fontsize=10, pad=26)
    apply_robust_y_limits(ax=ax, values=y_values)
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    sns.despine(ax=ax)


def add_block_legend(
    ax: plt.Axes,
    condition_block_palette: dict[tuple[str, str], tuple[float, float, float]],
    block_legend_labels: dict[tuple[str, str], str],
) -> None:
    """Draw a block-color legend on one axis.

    Args:
        ax (plt.Axes): Axis where legend will be drawn.
        condition_block_palette (dict[tuple[str, str], tuple[float, float, float]]):
            ``(hue, block)`` color mapping.
        block_legend_labels (dict[tuple[str, str], str]): ``(hue, block)`` label mapping.

    Returns:
        None: This function adds legend artists to ``ax``.
    """

    if not condition_block_palette:
        return
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=6,
            label=block_legend_labels.get((hue_value, block_label), f"{hue_value} {block_label}"),
        )
        for (hue_value, block_label), color in condition_block_palette.items()
    ]
    ax.legend(handles=handles, frameon=True, fontsize=8, title="Block", title_fontsize=9, loc="upper right")


def render_super_violin_on_ax(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    block: str,
    unit_dict: dict[str, str] | None = None,
    title_override: str | None = None,
    superplot_annotations: dict[str, str] | list[dict[str, str]] | None = None,
    x_order_override: list[str] | None = None,
    hue_order_override: list[str] | None = None,
    show_legend: bool = False,
) -> float:
    """Render one superviolin panel on a provided axis.

    Args:
        ax (plt.Axes): Destination axis.
        data (pd.DataFrame): Source dataframe.
        x (str): X-axis grouping column.
        y (str): Numeric metric column.
        hue (str): Hue grouping column.
        block (str): Replicate block column.
        unit_dict (dict[str, str] | None): Optional y-unit mapping.
        title_override (str | None): Optional custom panel title.
        superplot_annotations (dict[str, str] | list[dict[str, str]] | None): Optional bracket labels.
        x_order_override (list[str] | None): Optional explicit x order.
        hue_order_override (list[str] | None): Optional explicit hue order.
        show_legend (bool): Whether to draw block legend on this panel.

    Returns:
        float: Highest annotation y-position in axis-transform coordinates.
    """

    plot_data = prepare_plot_data(data=data, x=x, y=y, hue=hue, block=block)
    if plot_data.empty:
        ax.set_visible(False)
        return 1.0
    x_order, hue_order, block_order = get_category_orders(
        plot_data=plot_data,
        x=x,
        hue=hue,
        block=block,
        x_order_override=x_order_override,
        hue_order_override=hue_order_override,
    )
    if not x_order or not hue_order or not block_order:
        ax.set_visible(False)
        return 1.0
    annotation_records = normalize_superplot_annotations(superplot_annotations)
    condition_block_palette = create_condition_block_palette(
        block_order=block_order,
        hue_order=hue_order,
    )
    block_legend_labels = build_block_legend_labels(plot_data=plot_data, block=block, hue=hue, block_order=block_order)
    group_centers = get_group_centers(x_order=x_order, hue_order=hue_order)
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    group_profiles: list[dict[str, object]] = []
    global_density_max = 0.0
    for (x_value, hue_value), center in group_centers.items():
        group_data = plot_data[(plot_data[x].astype(str) == x_value) & (plot_data[hue].astype(str) == hue_value)]
        if group_data.empty:
            continue
        values = group_data[y].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        y_grid = build_density_grid(values)
        density_rows: list[np.ndarray] = []
        stripe_blocks: list[str] = []
        for block_label in block_order:
            block_values = group_data.loc[group_data[block].astype(str) == block_label, y].dropna().to_numpy(dtype=float)
            if block_values.size == 0:
                continue
            density_rows.append(estimate_density(block_values, y_grid) * block_values.size)
            stripe_blocks.append(block_label)
        if not density_rows:
            continue
        density_matrix = np.vstack(density_rows)
        total_density = density_matrix.sum(axis=0)
        if np.allclose(total_density.max(), 0):
            continue
        global_density_max = max(global_density_max, float(total_density.max()))
        group_profiles.append(
            {
                "center": center,
                "x_value": x_value,
                "hue_value": hue_value,
                "group_data": group_data,
                "y_grid": y_grid,
                "density_matrix": density_matrix,
                "total_density": total_density,
                "stripe_blocks": stripe_blocks,
            }
        )
    if global_density_max == 0.0:
        ax.set_visible(False)
        return 1.0
    for profile in group_profiles:
        center = float(profile["center"])
        x_value = str(profile["x_value"])
        hue_value = str(profile["hue_value"])
        group_data = profile["group_data"]
        y_grid = np.asarray(profile["y_grid"], dtype=float)
        density_matrix = np.asarray(profile["density_matrix"], dtype=float)
        total_density = np.asarray(profile["total_density"], dtype=float)
        stripe_blocks = list(profile["stripe_blocks"])
        half_width = SUPERPLOT_HALF_WIDTH * (total_density / global_density_max)
        left_edge = center - half_width
        right_edge = center + half_width
        current_left = left_edge.copy()
        full_width = right_edge - left_edge
        for block_label, block_density in zip(stripe_blocks, density_matrix):
            density_fraction = np.divide(block_density, total_density, out=np.zeros_like(block_density), where=total_density > 0)
            stripe_right = current_left + full_width * density_fraction
            ax.fill_betweenx(
                y_grid,
                current_left,
                stripe_right,
                color=condition_block_palette[(str(hue_value), str(block_label))],
                alpha=0.85,
                linewidth=0,
                zorder=2,
            )
            if block_label != stripe_blocks[-1]:
                ax.plot(stripe_right, y_grid, color="black", linewidth=0.8, zorder=3)
            current_left = stripe_right
        ax.plot(left_edge, y_grid, color="black", linewidth=1.0, zorder=3)
        ax.plot(right_edge, y_grid, color="black", linewidth=1.0, zorder=3)
        add_superplot_summary(
            ax=ax,
            group_data=group_data,
            y=y,
            block=block,
            hue_value=hue_value,
            center=center,
            condition_block_palette=condition_block_palette,
        )
        tick_positions.append(center)
        tick_labels.append(f"{x_value}\n{format_condition_display_label(hue_value)}")
    annotation_top = add_superplot_annotations(
        ax=ax,
        annotation_records=annotation_records,
        group_centers=group_centers,
        hue_order=hue_order,
    )
    style_superplot_axis(
        ax=ax,
        x=x,
        y=y,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        unit_dict=unit_dict,
        y_values=plot_data[y].to_numpy(dtype=float),
        title=title_override,
    )
    if show_legend:
        add_block_legend(
            ax=ax,
            condition_block_palette=condition_block_palette,
            block_legend_labels=block_legend_labels,
        )
    return annotation_top


def render_super_beeswarm_on_ax(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    block: str,
    unit_dict: dict[str, str] | None = None,
    title_override: str | None = None,
    superplot_annotations: dict[str, str] | list[dict[str, str]] | None = None,
    x_order_override: list[str] | None = None,
    hue_order_override: list[str] | None = None,
    show_legend: bool = False,
) -> float:
    """Render one superbeeswarm panel on a provided axis.

    Args:
        ax (plt.Axes): Destination axis.
        data (pd.DataFrame): Source dataframe.
        x (str): X-axis grouping column.
        y (str): Numeric metric column.
        hue (str): Hue grouping column.
        block (str): Replicate block column.
        unit_dict (dict[str, str] | None): Optional y-unit mapping.
        title_override (str | None): Optional custom panel title.
        superplot_annotations (dict[str, str] | list[dict[str, str]] | None): Optional bracket labels.
        x_order_override (list[str] | None): Optional explicit x order.
        hue_order_override (list[str] | None): Optional explicit hue order.
        show_legend (bool): Whether to draw block legend on this panel.

    Returns:
        float: Highest annotation y-position in axis-transform coordinates.
    """

    plot_data = prepare_plot_data(data=data, x=x, y=y, hue=hue, block=block)
    if plot_data.empty:
        ax.set_visible(False)
        return 1.0
    x_order, hue_order, block_order = get_category_orders(
        plot_data=plot_data,
        x=x,
        hue=hue,
        block=block,
        x_order_override=x_order_override,
        hue_order_override=hue_order_override,
    )
    if not x_order or not hue_order or not block_order:
        ax.set_visible(False)
        return 1.0
    annotation_records = normalize_superplot_annotations(superplot_annotations)
    condition_block_palette = create_condition_block_palette(
        block_order=block_order,
        hue_order=hue_order,
    )
    block_legend_labels = build_block_legend_labels(plot_data=plot_data, block=block, hue=hue, block_order=block_order)
    group_centers = get_group_centers(x_order=x_order, hue_order=hue_order)
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    group_profiles: list[dict[str, object]] = []
    global_density_max = 0.0
    for group_index, ((x_value, hue_value), center) in enumerate(group_centers.items()):
        group_data = plot_data[(plot_data[x].astype(str) == x_value) & (plot_data[hue].astype(str) == hue_value)]
        if group_data.empty:
            continue
        values = group_data[y].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        y_grid = build_density_grid(values)
        scaled_density = estimate_density(values, y_grid) * values.size
        if np.allclose(scaled_density.max(), 0):
            continue
        global_density_max = max(global_density_max, float(scaled_density.max()))
        group_profiles.append(
            {
                "group_index": group_index,
                "center": center,
                "x_value": x_value,
                "hue_value": hue_value,
                "group_data": group_data.reset_index(drop=True),
                "y_grid": y_grid,
                "scaled_density": scaled_density,
            }
        )
    if global_density_max == 0.0:
        ax.set_visible(False)
        return 1.0
    max_group_count = max(len(profile["group_data"]) for profile in group_profiles)
    for profile in group_profiles:
        group_index = int(profile["group_index"])
        center = float(profile["center"])
        x_value = str(profile["x_value"])
        hue_value = str(profile["hue_value"])
        group_data = profile["group_data"]
        y_grid = np.asarray(profile["y_grid"], dtype=float)
        scaled_density = np.asarray(profile["scaled_density"], dtype=float)
        point_size = get_dynamic_beeswarm_point_size(group_count=len(group_data), max_group_count=max_group_count)
        half_width = SUPERPLOT_HALF_WIDTH * (scaled_density / global_density_max)
        point_x = build_violin_like_swarm_positions(
            group_data=group_data,
            y=y,
            y_grid=y_grid,
            half_width=half_width,
            center=center,
            seed=group_index * 10_000,
        )
        for block_label in block_order:
            block_mask = group_data[block].astype(str) == block_label
            if not block_mask.any():
                continue
            ax.scatter(
                point_x[block_mask.to_numpy()],
                group_data.loc[block_mask, y].to_numpy(dtype=float),
                s=point_size,
                color=condition_block_palette[(str(hue_value), str(block_label))],
                edgecolor="white",
                linewidth=0.2,
                alpha=SUPERPLOT_POINT_ALPHA,
                zorder=3,
            )
        add_superplot_summary(
            ax=ax,
            group_data=group_data,
            y=y,
            block=block,
            hue_value=hue_value,
            center=center,
            condition_block_palette=condition_block_palette,
        )
        tick_positions.append(center)
        tick_labels.append(f"{x_value}\n{format_condition_display_label(hue_value)}")
    annotation_top = add_superplot_annotations(
        ax=ax,
        annotation_records=annotation_records,
        group_centers=group_centers,
        hue_order=hue_order,
    )
    style_superplot_axis(
        ax=ax,
        x=x,
        y=y,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        unit_dict=unit_dict,
        y_values=plot_data[y].to_numpy(dtype=float),
        title=title_override,
    )
    if show_legend:
        add_block_legend(
            ax=ax,
            condition_block_palette=condition_block_palette,
            block_legend_labels=block_legend_labels,
        )
    return annotation_top


def plot_super_violin(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    block: str,
    unit_dict: dict[str, str] | None = None,
    save_dir: str | Path | None = None,
    title_override: str | None = None,
    filename_prefix: str | None = None,
    superplot_annotations: dict[str, str] | list[dict[str, str]] | None = None,
    x_order_override: list[str] | None = None,
    hue_order_override: list[str] | None = None,
    output_dir_suffix: str = "",
) -> None:
    """Render and save one standalone superviolin figure.

    Args:
        data (pd.DataFrame): Source dataframe.
        x (str): X-axis grouping column.
        y (str): Numeric metric column.
        hue (str): Hue grouping column.
        block (str): Replicate block column.
        unit_dict (dict[str, str] | None): Optional y-unit mapping.
        save_dir (str | Path | None): Base output directory.
        title_override (str | None): Optional title override.
        filename_prefix (str | None): Optional filename prefix.
        superplot_annotations (dict[str, str] | list[dict[str, str]] | None): Optional bracket labels.
        x_order_override (list[str] | None): Optional explicit x order.
        hue_order_override (list[str] | None): Optional explicit hue order.
        output_dir_suffix (str): Optional suffix appended to generated figure
            directories.

    Returns:
        None: Saves a PNG when ``save_dir`` is provided.
    """

    fig, ax = plt.subplots(figsize=(14, 6))
    annotation_top = render_super_violin_on_ax(
        ax=ax,
        data=data,
        x=x,
        y=y,
        hue=hue,
        block=block,
        unit_dict=unit_dict,
        title_override=title_override,
        superplot_annotations=superplot_annotations,
        x_order_override=x_order_override,
        hue_order_override=hue_order_override,
        show_legend=False,
    )
    output_path = build_output_path(
        y=y,
        x=x,
        hue=hue,
        save_dir=save_dir,
        suffix="superviolin",
        filename_prefix=filename_prefix,
        output_dir_suffix=output_dir_suffix,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, superplot_top_margin(annotation_top)))
    if output_path is not None:
        fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_super_beeswarm(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    block: str,
    unit_dict: dict[str, str] | None = None,
    save_dir: str | Path | None = None,
    title_override: str | None = None,
    filename_prefix: str | None = None,
    superplot_annotations: dict[str, str] | list[dict[str, str]] | None = None,
    x_order_override: list[str] | None = None,
    hue_order_override: list[str] | None = None,
    output_dir_suffix: str = "",
) -> None:
    """Render and save one standalone superbeeswarm figure.

    Args:
        data (pd.DataFrame): Source dataframe.
        x (str): X-axis grouping column.
        y (str): Numeric metric column.
        hue (str): Hue grouping column.
        block (str): Replicate block column.
        unit_dict (dict[str, str] | None): Optional y-unit mapping.
        save_dir (str | Path | None): Base output directory.
        title_override (str | None): Optional title override.
        filename_prefix (str | None): Optional filename prefix.
        superplot_annotations (dict[str, str] | list[dict[str, str]] | None): Optional bracket labels.
        x_order_override (list[str] | None): Optional explicit x order.
        hue_order_override (list[str] | None): Optional explicit hue order.
        output_dir_suffix (str): Optional suffix appended to generated figure
            directories.

    Returns:
        None: Saves a PNG when ``save_dir`` is provided.
    """

    fig, ax = plt.subplots(figsize=(14, 6))
    annotation_top = render_super_beeswarm_on_ax(
        ax=ax,
        data=data,
        x=x,
        y=y,
        hue=hue,
        block=block,
        unit_dict=unit_dict,
        title_override=title_override,
        superplot_annotations=superplot_annotations,
        x_order_override=x_order_override,
        hue_order_override=hue_order_override,
        show_legend=False,
    )
    output_path = build_output_path(
        y=y,
        x=x,
        hue=hue,
        save_dir=save_dir,
        suffix="superbeeswarm",
        filename_prefix=filename_prefix,
        output_dir_suffix=output_dir_suffix,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, superplot_top_margin(annotation_top)))
    if output_path is not None:
        fig.savefig(output_path, dpi=300)
    plt.close(fig)
