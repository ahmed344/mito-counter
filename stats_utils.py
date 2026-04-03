from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from statannotations.Annotator import Annotator

BOX_PLOT_PALETTE = ["tab:blue", "tab:orange"]
SUPERPLOT_GROUP_WIDTH = 1.0
SUPERPLOT_HALF_WIDTH = 0.42
SUPERPLOT_GRID_SIZE = 256
SUPERPLOT_POINT_SIZE_MIN = 4
SUPERPLOT_POINT_SIZE_MAX = 11
SUPERPLOT_POINT_ALPHA = 0.4
ROBUST_Y_LOWER_QUANTILE = 0.01
ROBUST_Y_UPPER_QUANTILE = 0.99


def maybe_show_current_figure():
    """Display the current figure only when an interactive backend is available.

    Args:
        None: This helper inspects the active Matplotlib backend instead of receiving inputs.

    Returns:
        None: The function optionally displays the current figure and does not return a value.
    """
    if "agg" not in plt.get_backend().lower():
        plt.show()
    return None


def sort_condition_values(values):
    """Sort condition labels with WT-like labels before KO-like labels.

    Args:
        values (list[str]): Condition labels extracted from the dataset.

    Returns:
        list[str]: Sorted condition labels using a WT/KO-aware ordering.
    """
    return sorted(
        values,
        key=lambda v: (
            0
            if (
                "wildtype" in str(v).lower()
                or str(v).strip().lower() == "wt"
                or str(v).strip().lower().endswith("_wt")
            )
            else 1
            if (
                "knockout" in str(v).lower()
                or str(v).strip().lower() == "ko"
                or str(v).strip().lower().endswith("_ko")
            )
            else 2,
            str(v).lower(),
        ),
    )


def sort_block_values(values):
    """Sort block labels numerically when possible and lexically otherwise.

    Args:
        values (list[object]): Block labels extracted from the dataset.

    Returns:
        list[str]: Sorted block labels converted to strings.
    """
    normalized_values = [str(value) for value in values]
    try:
        return sorted(normalized_values, key=lambda value: (0, int(value)))
    except ValueError:
        return sorted(normalized_values, key=lambda value: (1, value))


def prepare_plot_data(data, x, y, hue, block=None):
    """Prepare a plotting dataframe with numeric coercion and NA filtering.

    Args:
        data (pd.DataFrame): Source dataframe containing plotting columns.
        x (str): Column name used on the x-axis.
        y (str): Column name used as the numeric response variable.
        hue (str): Column name used for grouping within each x category.
        block (str | None): Optional column name used for block-aware plots.

    Returns:
        pd.DataFrame: Filtered dataframe ready for plotting.
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


def get_category_orders(plot_data, x, hue, block=None):
    """Build stable category orders for the requested plotting columns.

    Args:
        plot_data (pd.DataFrame): Prepared plotting dataframe.
        x (str): Column name used on the x-axis.
        hue (str): Column name used for grouped categories.
        block (str | None): Optional column name used for block-aware plots.

    Returns:
        tuple[list[str], list[str], list[str]]: Orders for x, hue, and block.
    """
    x_order = sorted(plot_data[x].astype(str).unique())
    if hue == "Condition":
        hue_order = sort_condition_values(plot_data[hue].dropna().astype(str).unique().tolist())
    else:
        hue_order = sorted(plot_data[hue].dropna().astype(str).unique().tolist())

    block_order = []
    if block is not None and block in plot_data.columns:
        block_order = sort_block_values(plot_data[block].dropna().unique().tolist())

    return x_order, hue_order, block_order


def format_unit_label(metric_name, unit_dict):
    """Format a metric unit label for display on the y-axis.

    Args:
        metric_name (str): Name of the plotted metric.
        unit_dict (dict[str, str] | None): Optional mapping from metric names to units.

    Returns:
        str: Formatted unit suffix including parentheses when available.
    """
    if not unit_dict or metric_name not in unit_dict or not unit_dict[metric_name]:
        return ""

    formatted_unit = unit_dict[metric_name].replace("um", r"$\mu m$").replace("^2", r"$^2$")
    return f" ({formatted_unit})"


def build_output_path(y, x, hue, save_dir, suffix):
    """Construct a figure output path for a plot type.

    Args:
        y (str): Column name used as the numeric response variable.
        x (str): Column name used on the x-axis.
        hue (str): Column name used for grouped categories.
        save_dir (str | Path | None): Directory where figures are written.
        suffix (str): Plot-type suffix appended to the filename.

    Returns:
        Path | None: Output path when ``save_dir`` is provided, otherwise ``None``.
    """
    if save_dir is None:
        return None

    directory_by_suffix = {
        "boxplot": "box_plots",
        "superviolin": "super_violin",
        "superbeeswarm": "super_beeswarm",
    }
    output_dir = Path(save_dir) / directory_by_suffix.get(suffix, "")
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_parts = [str(part).replace("/", "_").replace(" ", "_") for part in (y, x, hue, suffix)]
    return output_dir / f"{safe_parts[0]}_by_{safe_parts[1]}_and_{safe_parts[2]}_{safe_parts[3]}.png"


def create_block_palette(block_order):
    """Create a consistent color palette for block labels.

    Args:
        block_order (list[str]): Ordered block labels used in the plots.

    Returns:
        dict[str, tuple[float, float, float]]: Mapping from block label to RGB color.
    """
    if not block_order:
        return {}
    base_palette = list(sns.color_palette("Set1", min(len(block_order), 9)))
    if len(base_palette) > 5:
        base_palette[2], base_palette[5] = base_palette[5], base_palette[2]
    if len(block_order) > len(base_palette):
        extra_count = len(block_order) - len(base_palette)
        base_palette.extend(sns.color_palette("Dark2", min(extra_count, 8)))
    if len(block_order) > len(base_palette):
        base_palette.extend(sns.color_palette("tab20", len(block_order) - len(base_palette)))
    palette = base_palette[:len(block_order)]
    return dict(zip(block_order, palette))


def format_condition_legend_label(condition_value):
    """Convert a raw condition label into a compact legend-friendly name.

    Args:
        condition_value (str): Raw condition name from the dataset.

    Returns:
        str: Short condition label suitable for block legends.
    """
    normalized_value = str(condition_value).strip()
    lower_value = normalized_value.lower()
    if "wildtype" in lower_value or lower_value == "wt" or lower_value.endswith("_wt"):
        return "WT"
    if "calpain_3" in lower_value and "knockout" in lower_value:
        return "KO-C3"
    if "knockout" in lower_value or lower_value == "ko" or lower_value.endswith("_ko"):
        return "KO"
    return normalized_value.replace("_", " ")


def build_block_legend_labels(plot_data, block, hue, block_order):
    """Build human-readable legend labels for each block color.

    Args:
        plot_data (pd.DataFrame): Prepared plotting dataframe containing block and hue columns.
        block (str): Column name identifying replicate blocks.
        hue (str): Column name identifying the condition for each block.
        block_order (list[str]): Ordered block labels used in the plots.

    Returns:
        dict[str, str]: Mapping from block label to legend text.
    """
    block_legend_labels = {}
    for block_label in block_order:
        block_rows = plot_data.loc[plot_data[block].astype(str) == block_label, hue].dropna()
        if block_rows.empty:
            block_legend_labels[block_label] = str(block_label)
            continue
        condition_label = format_condition_legend_label(block_rows.iloc[0])
        block_legend_labels[block_label] = f"{condition_label} {block_label}"
    return block_legend_labels


def get_group_centers(x_order, hue_order):
    """Compute x-axis centers for grouped categorical plots.

    Args:
        x_order (list[str]): Ordered x-axis categories.
        hue_order (list[str]): Ordered hue categories within each x group.

    Returns:
        dict[tuple[str, str], float]: Mapping from ``(x, hue)`` to plot center.
    """
    base_spacing = max(1.6, len(hue_order) * SUPERPLOT_GROUP_WIDTH)
    if len(hue_order) == 1:
        offsets = [0.0]
    else:
        offsets = np.linspace(
            -SUPERPLOT_GROUP_WIDTH / 2,
            SUPERPLOT_GROUP_WIDTH / 2,
            len(hue_order),
        )

    return {
        (str(x_value), str(hue_value)): index * base_spacing + float(offset)
        for index, x_value in enumerate(x_order)
        for hue_value, offset in zip(hue_order, offsets)
    }


def build_density_grid(values):
    """Build a y-axis grid spanning the provided values for KDE-based plots.

    Args:
        values (np.ndarray): Numeric values belonging to one x/hue group.

    Returns:
        np.ndarray: Evenly spaced y-axis grid covering the data range.
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


def estimate_density(values, y_grid):
    """Estimate a smooth density curve for one set of observations.

    Args:
        values (np.ndarray): Numeric observations used to compute a density curve.
        y_grid (np.ndarray): Shared y-axis grid where the density is evaluated.

    Returns:
        np.ndarray: Non-negative density values evaluated on ``y_grid``.
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


def build_violin_like_swarm_positions(group_data, y, y_grid, half_width, center, seed):
    """Compute point-only beeswarm x positions that form a violin-like silhouette.

    Args:
        group_data (pd.DataFrame): Data for a single x/hue category.
        y (str): Column name used as the numeric response variable.
        y_grid (np.ndarray): Shared y-axis grid where the density was evaluated.
        half_width (np.ndarray): Local half-width of the swarm silhouette on ``y_grid``.
        center (float): Horizontal center of the current group.
        seed (int): Random seed used for deterministic within-bin shuffling.

    Returns:
        np.ndarray: X positions for every row in ``group_data``.
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
            offsets += rng.uniform(
                -local_half_width * 0.08,
                local_half_width * 0.08,
                size=bin_indices.size,
            )

        shuffled_indices = bin_indices[rng.permutation(bin_indices.size)]
        point_x[shuffled_indices] = center + offsets

    return point_x


def apply_robust_y_limits(ax, values):
    """Set y-axis limits from the central distribution instead of extreme outliers.

    Args:
        ax (plt.Axes): Matplotlib axes whose y-limits should be updated.
        values (np.ndarray | pd.Series | list[float]): Numeric values used to estimate
            a robust display range.

    Returns:
        None: The function updates ``ax`` in place and does not return a value.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None

    if values.size < 20:
        value_min = float(np.min(values))
        value_max = float(np.max(values))
    else:
        value_min = float(np.quantile(values, ROBUST_Y_LOWER_QUANTILE))
        value_max = float(np.quantile(values, ROBUST_Y_UPPER_QUANTILE))

    if np.isclose(value_min, value_max):
        spread = max(abs(value_min) * 0.1, 1.0)
    else:
        spread = value_max - value_min

    padding = max(spread * 0.08, 1e-6)
    ax.set_ylim(value_min - padding, value_max + padding)
    return None


def get_dynamic_beeswarm_point_size(group_count, max_group_count, metric_name):
    """Compute a beeswarm marker size that shrinks as the group gets denser.

    Args:
        group_count (int): Number of observations in the current x/hue group.
        max_group_count (int): Largest observation count among all x/hue groups in the plot.
        metric_name (str): Name of the metric currently being rendered.

    Returns:
        float: Marker area passed to Matplotlib ``scatter``.
    """
    if metric_name == "Count":
        point_size_min = 10
        point_size_max = 22
    else:
        point_size_min = SUPERPLOT_POINT_SIZE_MIN
        point_size_max = SUPERPLOT_POINT_SIZE_MAX

    if group_count <= 1 or max_group_count <= 1:
        return float(point_size_max)

    relative_density = np.sqrt(group_count / max_group_count)
    marker_size = point_size_max - (
        (point_size_max - point_size_min) * relative_density
    )
    return float(np.clip(marker_size, point_size_min, point_size_max))


def add_superplot_summary(ax, group_data, y, block, center, block_palette):
    """Overlay block means and mean-plus-SEM summary markers on a superplot.

    Args:
        ax (plt.Axes): Matplotlib axes receiving the summary overlay.
        group_data (pd.DataFrame): Data for a single x/hue category.
        y (str): Column name used as the numeric response variable.
        block (str): Column name identifying replicate blocks.
        center (float): Horizontal center of the current group.
        block_palette (dict[str, tuple[float, float, float]]): Mapping from block label to color.

    Returns:
        None: The function adds artists directly to ``ax``.
    """
    block_means = []
    available_blocks = sorted(group_data[block].astype(str).unique(), key=lambda value: (value not in block_palette, value))
    if len(available_blocks) == 1:
        offsets = [0.0]
    else:
        offsets = np.linspace(-0.09, 0.09, len(available_blocks))

    for block_label, offset in zip(available_blocks, offsets):
        block_values = group_data.loc[group_data[block].astype(str) == block_label, y].dropna().to_numpy()
        if block_values.size == 0:
            continue
        block_mean = float(np.mean(block_values))
        block_means.append(block_mean)
        ax.scatter(
            center + float(offset),
            block_mean,
            s=36,
            marker="D",
            color=block_palette.get(block_label, "0.4"),
            edgecolor="black",
            linewidth=0.5,
            zorder=6,
        )

    if not block_means:
        return None

    overall_mean = float(np.mean(block_means))
    sem = float(np.std(block_means, ddof=1) / np.sqrt(len(block_means))) if len(block_means) > 1 else 0.0
    ax.errorbar(
        center,
        overall_mean,
        yerr=sem if sem > 0 else None,
        fmt="o",
        color="black",
        markersize=6,
        capsize=4,
        linewidth=1.4,
        zorder=7,
    )
    return None


def finalize_superplot(ax, y, x, tick_positions, tick_labels, unit_dict, block_palette, block_legend_labels, title, output_path, y_values):
    """Apply shared formatting, legend handling, and saving for custom superplots.

    Args:
        ax (plt.Axes): Matplotlib axes containing the finished plot.
        y (str): Column name used as the numeric response variable.
        x (str): Column name used on the x-axis.
        tick_positions (list[float]): X-axis tick positions for each group.
        tick_labels (list[str]): X-axis labels for each group center.
        unit_dict (dict[str, str] | None): Optional mapping from metric names to units.
        block_palette (dict[str, tuple[float, float, float]]): Mapping from block label to color.
        block_legend_labels (dict[str, str]): Mapping from block label to display text in the legend.
        title (str): Plot title.
        output_path (Path | None): Optional destination where the figure is saved.
        y_values (np.ndarray | pd.Series | list[float]): Numeric values used to set a
            robust y-axis display range.

    Returns:
        None: The function formats and optionally saves the current figure.
    """
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel(f"{y}{format_unit_label(y, unit_dict)}", fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    apply_robust_y_limits(ax=ax, values=y_values)
    if block_palette:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.4,
                markersize=7,
                label=block_legend_labels.get(block_label, str(block_label)),
            )
            for block_label, color in block_palette.items()
        ]
        ax.legend(handles=handles, frameon=True)

    sns.despine(ax=ax)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=300)
    maybe_show_current_figure()
    plt.close()
    return None


def plot_stat_boxplot(data, x, y, hue, unit_dict=None, test="Mann-Whitney", text_format="star", save_dir=None):
    """Plot a grouped boxplot with statistical annotations.

    Args:
        data (pd.DataFrame): Source dataframe containing plotting columns.
        x (str): Column name used on the x-axis.
        y (str): Column name used as the numeric response variable.
        hue (str): Column name used for grouping within each x category.
        unit_dict (dict[str, str] | None): Optional mapping from metric name to display unit.
        test (str): Statistical test name consumed by ``statannotations``.
        text_format (str): Annotation style passed to ``statannotations``.
        save_dir (str | Path | None): Optional output directory where a PNG is saved.

    Returns:
        None: The function renders/saves the plot and does not return a value.
    """
    plot_data = prepare_plot_data(data=data, x=x, y=y, hue=hue)
    if plot_data.empty:
        print(f"Skipping {y}: no numeric data available after coercion.")
        return None

    plt.figure(figsize=(8, 6))
    x_order, hue_order, _ = get_category_orders(plot_data=plot_data, x=x, hue=hue)

    ax = sns.boxplot(
        data=plot_data,
        x=x,
        y=y,
        hue=hue,
        order=x_order,
        hue_order=hue_order,
        linewidth=1.5,
        palette=BOX_PLOT_PALETTE,
        showfliers=False,
        gap=0.2,
        width=0.6,
    )

    box_pairs = []
    hue_combinations = list(combinations(hue_order, 2))
    for x_val in x_order:
        for hue_pair in hue_combinations:
            box_pairs.append(((x_val, hue_pair[0]), (x_val, hue_pair[1])))

    annotator = Annotator(
        ax,
        box_pairs,
        data=plot_data,
        x=x,
        y=y,
        hue=hue,
        order=x_order,
        hue_order=hue_order,
    )
    annotator.configure(test=test, text_format=text_format, loc="inside", verbose=0)
    annotator.apply_and_annotate()

    sns.despine()
    apply_robust_y_limits(ax=ax, values=plot_data[y].to_numpy(dtype=float))
    plt.ylabel(f"{y}{format_unit_label(y, unit_dict)}", fontsize=12)
    plt.xlabel(x, fontsize=12)
    plt.title(f"{y} by {x} and {hue}", fontsize=14, pad=20)
    plt.legend(title=hue)
    plt.tight_layout()

    output_path = build_output_path(y=y, x=x, hue=hue, save_dir=save_dir, suffix="boxplot")
    if output_path is not None:
        plt.savefig(output_path, dpi=300)
    maybe_show_current_figure()
    plt.close()
    return None


def plot_super_violin(data, x, y, hue, block, unit_dict=None, save_dir=None):
    """Plot a block-striped violin superplot with replicate summaries.

    Args:
        data (pd.DataFrame): Source dataframe containing plotting columns.
        x (str): Column name used on the x-axis.
        y (str): Column name used as the numeric response variable.
        hue (str): Column name used for grouped categories within each x group.
        block (str): Column name identifying replicate blocks.
        unit_dict (dict[str, str] | None): Optional mapping from metric name to display unit.
        save_dir (str | Path | None): Optional output directory where a PNG is saved.

    Returns:
        None: The function renders/saves the plot and does not return a value.
    """
    plot_data = prepare_plot_data(data=data, x=x, y=y, hue=hue, block=block)
    if plot_data.empty:
        print(f"Skipping {y} superviolin: no block-aware numeric data available.")
        return None

    x_order, hue_order, block_order = get_category_orders(plot_data=plot_data, x=x, hue=hue, block=block)
    if not block_order:
        print(f"Skipping {y} superviolin: no block values available.")
        return None

    block_palette = create_block_palette(block_order)
    block_legend_labels = build_block_legend_labels(
        plot_data=plot_data,
        block=block,
        hue=hue,
        block_order=block_order,
    )
    group_centers = get_group_centers(x_order=x_order, hue_order=hue_order)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    tick_positions = []
    tick_labels = []
    group_profiles = []
    global_density_max = 0.0

    for (x_value, hue_value), center in group_centers.items():
        group_data = plot_data[
            (plot_data[x].astype(str) == x_value) & (plot_data[hue].astype(str) == hue_value)
        ]
        if group_data.empty:
            continue

        values = group_data[y].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue

        y_grid = build_density_grid(values)
        density_rows = []
        stripe_blocks = []
        for block_label in block_order:
            block_values = group_data.loc[
                group_data[block].astype(str) == block_label,
                y,
            ].dropna().to_numpy(dtype=float)
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
        print(f"Skipping {y} superviolin: unable to estimate density widths.")
        plt.close()
        return None

    for profile in group_profiles:
        center = profile["center"]
        x_value = profile["x_value"]
        hue_value = profile["hue_value"]
        group_data = profile["group_data"]
        y_grid = profile["y_grid"]
        density_matrix = profile["density_matrix"]
        total_density = profile["total_density"]
        stripe_blocks = profile["stripe_blocks"]

        half_width = SUPERPLOT_HALF_WIDTH * (total_density / global_density_max)
        left_edge = center - half_width
        right_edge = center + half_width
        current_left = left_edge.copy()
        full_width = right_edge - left_edge

        for block_label, block_density in zip(stripe_blocks, density_matrix):
            density_fraction = np.divide(
                block_density,
                total_density,
                out=np.zeros_like(block_density),
                where=total_density > 0,
            )
            stripe_right = current_left + full_width * density_fraction
            ax.fill_betweenx(
                y_grid,
                current_left,
                stripe_right,
                color=block_palette[block_label],
                alpha=0.85,
                linewidth=0,
                zorder=2,
            )
            if block_label != stripe_blocks[-1]:
                ax.plot(
                    stripe_right,
                    y_grid,
                    color="black",
                    linewidth=0.8,
                    zorder=3,
                )
            current_left = stripe_right

        ax.plot(left_edge, y_grid, color="black", linewidth=1.0, zorder=3)
        ax.plot(right_edge, y_grid, color="black", linewidth=1.0, zorder=3)
        add_superplot_summary(
            ax=ax,
            group_data=group_data,
            y=y,
            block=block,
            center=center,
            block_palette=block_palette,
        )
        tick_positions.append(center)
        tick_labels.append(f"{x_value}\n{hue_value}")

    output_path = build_output_path(y=y, x=x, hue=hue, save_dir=save_dir, suffix="superviolin")
    finalize_superplot(
        ax=ax,
        y=y,
        x=x,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        unit_dict=unit_dict,
        block_palette=block_palette,
        block_legend_labels=block_legend_labels,
        title=f"{y} superviolin by {x} and {hue}",
        output_path=output_path,
        y_values=plot_data[y].to_numpy(dtype=float),
    )
    return None


def plot_super_beeswarm(data, x, y, hue, block, unit_dict=None, save_dir=None):
    """Plot a block-colored beeswarm superplot shaped by pooled density.

    Args:
        data (pd.DataFrame): Source dataframe containing plotting columns.
        x (str): Column name used on the x-axis.
        y (str): Column name used as the numeric response variable.
        hue (str): Column name used for grouped categories within each x group.
        block (str): Column name identifying replicate blocks.
        unit_dict (dict[str, str] | None): Optional mapping from metric name to display unit.
        save_dir (str | Path | None): Optional output directory where a PNG is saved.

    Returns:
        None: The function renders/saves the plot and does not return a value.
    """
    plot_data = prepare_plot_data(data=data, x=x, y=y, hue=hue, block=block)
    if plot_data.empty:
        print(f"Skipping {y} superbeeswarm: no block-aware numeric data available.")
        return None

    x_order, hue_order, block_order = get_category_orders(plot_data=plot_data, x=x, hue=hue, block=block)
    if not block_order:
        print(f"Skipping {y} superbeeswarm: no block values available.")
        return None

    block_palette = create_block_palette(block_order)
    block_legend_labels = build_block_legend_labels(
        plot_data=plot_data,
        block=block,
        hue=hue,
        block_order=block_order,
    )
    group_centers = get_group_centers(x_order=x_order, hue_order=hue_order)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    tick_positions = []
    tick_labels = []
    group_profiles = []
    global_density_max = 0.0

    for group_index, ((x_value, hue_value), center) in enumerate(group_centers.items()):
        group_data = plot_data[
            (plot_data[x].astype(str) == x_value) & (plot_data[hue].astype(str) == hue_value)
        ]
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
        print(f"Skipping {y} superbeeswarm: unable to estimate density widths.")
        plt.close()
        return None

    max_group_count = max(len(profile["group_data"]) for profile in group_profiles)

    for profile in group_profiles:
        group_index = profile["group_index"]
        center = profile["center"]
        x_value = profile["x_value"]
        hue_value = profile["hue_value"]
        group_data = profile["group_data"]
        y_grid = profile["y_grid"]
        scaled_density = profile["scaled_density"]
        point_size = get_dynamic_beeswarm_point_size(
            group_count=len(group_data),
            max_group_count=max_group_count,
            metric_name=y,
        )

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
                color=block_palette[block_label],
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
            center=center,
            block_palette=block_palette,
        )
        tick_positions.append(center)
        tick_labels.append(f"{x_value}\n{hue_value}")

    output_path = build_output_path(y=y, x=x, hue=hue, save_dir=save_dir, suffix="superbeeswarm")
    finalize_superplot(
        ax=ax,
        y=y,
        x=x,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        unit_dict=unit_dict,
        block_palette=block_palette,
        block_legend_labels=block_legend_labels,
        title=f"{y} superbeeswarm by {x} and {hue}",
        output_path=output_path,
        y_values=plot_data[y].to_numpy(dtype=float),
    )
    return None


def plot_metric_variants(data, x, y, hue, block, unit_dict=None, save_dir=None):
    """Generate boxplot, superviolin, and superbeeswarm outputs for one metric.

    Args:
        data (pd.DataFrame): Source dataframe containing plotting columns.
        x (str): Column name used on the x-axis.
        y (str): Column name used as the numeric response variable.
        hue (str): Column name used for grouped categories within each x group.
        block (str): Column name identifying replicate blocks.
        unit_dict (dict[str, str] | None): Optional mapping from metric name to display unit.
        save_dir (str | Path | None): Optional output directory where PNG files are saved.

    Returns:
        None: The function renders/saves plots and does not return a value.
    """
    plot_stat_boxplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        unit_dict=unit_dict,
        save_dir=save_dir,
    )
    plot_super_violin(
        data=data,
        x=x,
        y=y,
        hue=hue,
        block=block,
        unit_dict=unit_dict,
        save_dir=save_dir,
    )
    plot_super_beeswarm(
        data=data,
        x=x,
        y=y,
        hue=hue,
        block=block,
        unit_dict=unit_dict,
        save_dir=save_dir,
    )
    return None
