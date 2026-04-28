"""Generate Bayesian diagnostic, PPC, and biology-facing figures for all fits."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import arviz as az
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from hierarchical_bayes_config import (
    BayesFitConfig,
    DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH,
    fit_config_for_pair,
    load_hierarchical_bayes_config,
    repeated_fit_configs,
    validate_config_muscles,
)
from hierarchical_bayes_metrics import (
    BOUNDED_LIKELIHOODS,
    BOUNDED_METRICS,
    POSITIVE_LIKELIHOODS,
    POSITIVE_METRICS,
    PreparedMetricData,
    attach_response_scale_posterior,
    biology_posterior_var_names,
    diagnostic_var_names,
    fit_stem,
    prepare_metric_data,
    trace_path_for_fit,
)
from stats_utils import plot_super_beeswarm, plot_super_violin


BAYES_SCRIPT_PATH = Path("/workspaces/mito-counter/hierarchical_bayes_metrics.py")
QUANTILE_LEVELS = (0.10, 0.50, 0.90)
GRID_ALPHA = 0.22
GRID_LINESTYLE = "--"
XTICK_ROTATION = 25
YTICK_ROTATION = 0
TITLE_TOP_DEFAULT = 0.94
TITLE_TOP_POSTERIOR = 0.91
POSTERIOR_XTICK_ROTATION = 20
TRACE_RANK_HSPACE = 0.62
TRACE_HSPACE = 0.78
TRACE_RANK_WSPACE = 0.28
DEFAULT_HSPACE = 0.35
DEFAULT_WSPACE = 0.22
METRIC_UNITS = {
    "Area": "nm^2",
    "Corrected_area": "nm^2",
    "Major_axis_length": "nm",
    "Minor_axis_length": "nm",
    "Minimum_Feret_Diameter": "nm",
    "Elongation": "",
    "Circularity": "",
    "Solidity": "",
    "NND": "nm",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the Bayesian visualization workflow.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Generate Bayesian diagnostics, PPCs, and biology-facing figures."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH)
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    """Load the Bayesian summary CSV produced by the fitting workflow.

    Args:
        path (Path): Summary CSV path produced by the Bayesian fitting script.

    Returns:
        pd.DataFrame: Summary table containing one row per muscle-metric fit.
    """

    return pd.read_csv(path)


def filter_summary_to_fit_pairs(
    summary_df: pd.DataFrame,
    fit_configs: list[BayesFitConfig],
) -> pd.DataFrame:
    """Filter the summary table to the configured rerun fits.

    Args:
        summary_df (pd.DataFrame): Full Bayesian summary table.
        fit_configs (list[BayesFitConfig]): Configured fit entries to keep.

    Returns:
        pd.DataFrame: Filtered summary table in original row order.
    """

    fit_pairs = {(fit_config.muscle, fit_config.metric) for fit_config in fit_configs}
    filtered = summary_df.loc[
        summary_df.apply(lambda row: (row["muscle"], row["metric"]) in fit_pairs, axis=1)
    ].copy()
    return filtered.sort_values(["muscle", "metric"]).reset_index(drop=True)


def resolve_trace_path(row: pd.Series, trace_dir: Path) -> Path:
    """Resolve the NetCDF path for a summary row.

    Args:
        row (pd.Series): Summary row describing one muscle-metric fit.
        trace_dir (Path): Default trace directory used when the CSV lacks a path.

    Returns:
        Path: NetCDF path for the requested fit.
    """

    trace_path = str(row.get("trace_path", "")).strip()
    if trace_path:
        return Path(trace_path)
    return trace_path_for_fit(trace_dir=trace_dir, muscle=str(row["muscle"]), metric=str(row["metric"]))


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for a file path if needed.

    Args:
        path (Path): File path whose parent directory should exist.

    Returns:
        None
    """

    path.parent.mkdir(parents=True, exist_ok=True)


def axes_to_figure(axes: Any) -> plt.Figure:
    """Recover the owning figure from an ArviZ or Matplotlib axes object.

    Args:
        axes (Any): Axes object, list of axes, or ndarray returned by a plotting call.

    Returns:
        plt.Figure: Figure containing the provided axes.
    """

    if hasattr(axes, "figure"):
        return axes.figure
    axes_array = np.asarray(axes, dtype=object)
    return axes_array.flat[0].figure


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save and close a Matplotlib figure.

    Args:
        fig (plt.Figure): Figure to be written to disk.
        output_path (Path): Destination PNG path.

    Returns:
        None
    """

    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def style_figure_axes(
    fig: plt.Figure,
    grid_axis: str = "y",
    xtick_rotation: float = XTICK_ROTATION,
    ytick_rotation: float = YTICK_ROTATION,
) -> None:
    """Apply shared grid and tick styling across all axes in a figure.

    Args:
        fig (plt.Figure): Figure whose axes should be styled.
        grid_axis (str): Axis selection passed to ``Axes.grid``.
        xtick_rotation (float): Rotation angle applied to x tick labels.
        ytick_rotation (float): Rotation angle applied to y tick labels.

    Returns:
        None
    """

    for ax in fig.axes:
        if ax is None:
            continue
        ax.grid(axis=grid_axis, alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)
        ax.tick_params(axis="x", labelrotation=xtick_rotation)
        ax.tick_params(axis="y", labelrotation=ytick_rotation)


def finalize_figure_layout(
    fig: plt.Figure,
    title_top: float = TITLE_TOP_DEFAULT,
    hspace: float = DEFAULT_HSPACE,
    wspace: float = DEFAULT_WSPACE,
) -> None:
    """Reserve room for titles and increase subplot separation before saving.

    Args:
        fig (plt.Figure): Figure whose layout should be adjusted.
        title_top (float): Top boundary reserved for subplot layout below the title.
        hspace (float): Vertical spacing passed to ``subplots_adjust``.
        wspace (float): Horizontal spacing passed to ``subplots_adjust``.

    Returns:
        None
    """

    fig.subplots_adjust(top=title_top, hspace=hspace, wspace=wspace)


def rerun_bayesian_fits(config_path: Path) -> None:
    """Invoke the Bayesian fitting workflow to refresh traces and summary rows.

    Args:
        config_path (Path): YAML configuration path shared with the metrics workflow.

    Returns:
        None
    """

    command = [
        sys.executable,
        str(BAYES_SCRIPT_PATH),
        "--config",
        str(config_path),
    ]
    subprocess.run(command, check=True)


def traces_missing(summary_df: pd.DataFrame, trace_dir: Path) -> bool:
    """Check whether any requested summary rows lack a corresponding trace file.

    Args:
        summary_df (pd.DataFrame): Summary table to validate.
        trace_dir (Path): Default trace directory used when the CSV lacks a path.

    Returns:
        bool: ``True`` when at least one trace file is missing.
    """

    for _, row in summary_df.iterrows():
        if not resolve_trace_path(row=row, trace_dir=trace_dir).exists():
            return True
    return False


def load_or_refresh_summary(
    summary_path: Path,
    trace_dir: Path,
    refresh_mode: str,
    config_path: Path,
    fit_configs: list[BayesFitConfig],
) -> pd.DataFrame:
    """Load the configured summary table and optionally refresh fits first.

    Args:
        summary_path (Path): Summary CSV path produced by the metrics workflow.
        trace_dir (Path): Default trace directory used when the CSV lacks a path.
        refresh_mode (str): Visualization refresh mode from the YAML config.
        config_path (Path): YAML configuration path shared with the metrics workflow.
        fit_configs (list[BayesFitConfig]): Configured fit entries that should be visualized.

    Returns:
        pd.DataFrame: Filtered summary table after any requested rerun step.
    """

    if refresh_mode == "refit_first":
        rerun_bayesian_fits(config_path=config_path)
    elif refresh_mode == "rerun_missing_traces":
        if not summary_path.exists():
            rerun_bayesian_fits(config_path=config_path)
        else:
            current_summary = filter_summary_to_fit_pairs(
                summary_df=load_summary(summary_path),
                fit_configs=fit_configs,
            )
            if traces_missing(summary_df=current_summary, trace_dir=trace_dir):
                rerun_bayesian_fits(config_path=config_path)
    elif refresh_mode != "never":
        raise ValueError(f"Unsupported visualization refresh mode: {refresh_mode}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
    refreshed_summary = load_summary(summary_path)
    filtered_summary = filter_summary_to_fit_pairs(summary_df=refreshed_summary, fit_configs=fit_configs)
    if filtered_summary.empty:
        raise ValueError("Configured fits were not found in the summary CSV.")
    return filtered_summary


def resolved_likelihood_name(row: pd.Series, fit_config: BayesFitConfig) -> str:
    """Resolve the concrete likelihood name stored for one summary row.

    Args:
        row (pd.Series): Summary row describing one muscle-metric fit.
        fit_config (BayesFitConfig): Configured fit entry for the same muscle-metric pair.

    Returns:
        str: Concrete likelihood name used to fit the row.
    """

    likelihood_name = str(row.get("likelihood_name", "")).strip()
    if likelihood_name:
        return likelihood_name
    return fit_config.likelihood


def prepare_data_for_row(
    row: pd.Series,
    measurements_df: pd.DataFrame,
    fit_config: BayesFitConfig,
) -> PreparedMetricData:
    """Rebuild prepared arrays for one summary row using its stored likelihood choice.

    Args:
        row (pd.Series): Summary row describing one muscle-metric fit.
        measurements_df (pd.DataFrame): Full cleaned measurements table.
        fit_config (BayesFitConfig): Configured fit entry for the same muscle-metric pair.

    Returns:
        PreparedMetricData: Prepared arrays matching the stored fit configuration.
    """

    likelihood_name = resolved_likelihood_name(row=row, fit_config=fit_config)
    positive_likelihood = likelihood_name if str(row["family"]) == "positive" else POSITIVE_LIKELIHOODS[0]
    bounded_likelihood = likelihood_name if str(row["family"]) == "bounded" else BOUNDED_LIKELIHOODS[1]
    return prepare_metric_data(
        df=measurements_df,
        muscle=str(row["muscle"]),
        metric=str(row["metric"]),
        positive_likelihood=positive_likelihood,
        bounded_likelihood=bounded_likelihood,
    )


def ensure_augmented_idata(
    prepared: PreparedMetricData,
    idata: az.InferenceData,
) -> az.InferenceData:
    """Ensure a loaded inference dataset contains response-scale derived variables.

    Args:
        prepared (PreparedMetricData): Prepared metric-specific arrays matching the fitted model.
        idata (az.InferenceData): Loaded inference data for the current fit.

    Returns:
        az.InferenceData: Inference data guaranteed to include response-scale variables.
    """

    required_variables = biology_posterior_var_names(family=prepared.family)
    if all(variable in idata.posterior.data_vars for variable in required_variables):
        return idata
    return attach_response_scale_posterior(data=prepared, idata=idata)


def metric_axis_label(metric: str) -> str:
    """Build a human-readable axis label for one metric.

    Args:
        metric (str): Metric name used in the measurements table.

    Returns:
        str: Axis label including units when they are known.
    """

    unit = str(METRIC_UNITS.get(metric, "")).strip()
    return f"{metric} ({unit})" if unit else metric


def ppc_arrays_on_response_scale(
    idata: az.InferenceData,
    prepared: PreparedMetricData,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observed and predictive PPC arrays on the original response scale.

    Args:
        idata (az.InferenceData): Inference data containing posterior predictive draws.
        prepared (PreparedMetricData): Prepared metric-specific arrays matching the fit.

    Returns:
        tuple[np.ndarray, np.ndarray]: Observed vector and predictive matrix on the response scale.
    """

    observed = np.asarray(prepared.observed_y, dtype=float)
    predictive = np.asarray(idata.posterior_predictive["observed_metric"], dtype=float)
    predictive_samples = predictive.reshape(predictive.shape[0] * predictive.shape[1], predictive.shape[2])
    if prepared.family == "positive":
        predictive_samples = prepared.positive_offset + predictive_samples * prepared.positive_scale
    return observed, predictive_samples


def histogram_bin_edges(observed: np.ndarray, predictive_samples: np.ndarray) -> np.ndarray:
    """Compute stable histogram edges shared by observed and predictive PPC arrays.

    Bin width and axis span are driven by a central band of the pooled data (default
    0.5th--99.5th percentiles, with a small pad) so isolated extreme values do not stretch
    the x-axis and flatten the main density peaks.

    Args:
        observed (np.ndarray): Observed response-scale values.
        predictive_samples (np.ndarray): Predictive draws on the response scale.

    Returns:
        np.ndarray: Monotonic bin edges suitable for density overlays.
    """

    predictive_subset = predictive_samples[: min(25, predictive_samples.shape[0])].reshape(-1)
    combined = np.concatenate([observed.reshape(-1), predictive_subset])
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        return np.linspace(0.0, 1.0, 39)

    finite_for_edges = finite
    if finite.size >= 10:
        lo = float(np.percentile(finite, 0.5))
        hi = float(np.percentile(finite, 99.5))
        if hi > lo:
            span = hi - lo
            pad = max(0.02 * span, np.finfo(float).tiny)
            central = finite[(finite >= lo - pad) & (finite <= hi + pad)]
            if central.size >= 5:
                finite_for_edges = central

    lower = float(np.min(finite_for_edges))
    upper = float(np.max(finite_for_edges))
    if np.isclose(lower, upper):
        padding = max(abs(lower) * 0.05, 0.05)
        return np.linspace(lower - padding, upper + padding, 39)
    edges_fd = np.histogram_bin_edges(finite_for_edges, bins="fd")
    n_fd_bins = max(1, edges_fd.size - 1)
    edges = np.histogram_bin_edges(finite_for_edges, bins=2 * n_fd_bins)
    if edges.size < 12:
        edges = np.linspace(lower, upper, 39)
    if edges.size > 120:
        edges = np.linspace(lower, upper, 120)
    return np.asarray(edges, dtype=float)


def animal_display_label(animal_label: str) -> str:
    """Shorten a condition-prefixed animal label for figure x-axis display.

    Args:
        animal_label (str): Prepared animal label in `Condition__Block` form.

    Returns:
        str: Display label emphasizing the block identifier.
    """

    parts = str(animal_label).split("__", maxsplit=1)
    return parts[-1] if len(parts) == 2 else str(animal_label)


def plot_density_overlay(
    ax: plt.Axes,
    observed: np.ndarray,
    predictive_samples: np.ndarray,
    title: str,
    rng: np.random.Generator,
    edges: np.ndarray | None = None,
) -> None:
    """Draw one observed-vs-predictive density-style overlay on a Matplotlib axis.

    Args:
        ax (plt.Axes): Matplotlib axis receiving the overlay.
        observed (np.ndarray): Observed response-scale values for one subset.
        predictive_samples (np.ndarray): Predictive draws on the response scale.
        title (str): Axis title for the current subset.
        rng (np.random.Generator): Random number generator used to subsample predictive draws.
        edges (np.ndarray | None): Optional precomputed bin edges; when None, edges are
            chosen from the pooled observed and predictive values for this subplot.

    Returns:
        None
    """

    if observed.size == 0:
        ax.set_title(f"{title} (no observations)")
        if edges is not None:
            ax.set_xlim(float(edges[0]), float(edges[-1]))
        return
    if edges is None:
        edges = histogram_bin_edges(observed=observed, predictive_samples=predictive_samples)
    centers = 0.5 * (edges[1:] + edges[:-1])
    observed_density, _ = np.histogram(observed, bins=edges, density=True)
    predictive_count = min(40, predictive_samples.shape[0])
    draw_indices = rng.choice(predictive_samples.shape[0], size=predictive_count, replace=False)
    predictive_densities = np.asarray(
        [
            np.histogram(predictive_samples[index], bins=edges, density=True)[0]
            for index in draw_indices
        ],
        dtype=float,
    )
    for predictive_density in predictive_densities:
        ax.step(centers, predictive_density, where="mid", color="tab:blue", alpha=0.08, linewidth=1.0)
    ax.step(centers, np.median(predictive_densities, axis=0), where="mid", color="tab:blue", linewidth=2.0)
    ax.step(
        centers,
        observed_density,
        where="mid",
        color="red",
        alpha=0.7,
        linewidth=2.0,
    )
    ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.set_title(title, pad=10)


def fit_output_dir(root: Path, category: str, row: pd.Series) -> Path:
    """Build an output directory for a single fit and figure category.

    Args:
        root (Path): Root figure directory for the visualization workflow.
        category (str): Figure category such as ``diagnostics`` or ``ppc``.
        row (pd.Series): Summary row describing one fit.

    Returns:
        Path: Directory where figures for this fit and category should be written.
    """

    return root / category / fit_stem(muscle=str(row["muscle"]), metric=str(row["metric"]))


def available_var_names(idata: az.InferenceData, requested: list[str]) -> list[str]:
    """Filter a requested variable list to only those present in the posterior group.

    Args:
        idata (az.InferenceData): Inference data containing posterior samples.
        requested (list[str]): Desired posterior variable names.

    Returns:
        list[str]: Variable names that exist in ``idata.posterior``.
    """

    return [variable for variable in requested if variable in idata.posterior.data_vars]


def fit_title(row: pd.Series) -> str:
    """Build a short title prefix for figures describing one fit.

    Args:
        row (pd.Series): Summary row describing one muscle-metric fit.

    Returns:
        str: Human-readable title prefix for the current fit.
    """

    return f"{row['metric']} - {row['muscle']}"


def summary_text_or_empty(value: Any) -> str:
    """Normalize an optional summary field into a clean title-safe string.

    Args:
        value (Any): Raw dataframe cell value that may be empty or NaN.

    Returns:
        str: Stripped text, or an empty string when no usable text is present.
    """

    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def superplot_title_text(row: pd.Series) -> str:
    """Build the superplot title including posterior and Bayes-factor summaries.

    Args:
        row (pd.Series): Summary row describing one muscle-metric fit.

    Returns:
        str: Multi-line title text for the superplot figure.
    """

    title_lines = [fit_title(row)]
    delta_summary = summary_text_or_empty(row.get("delta_mean_summary", ""))
    bf_summary = summary_text_or_empty(row.get("delta_mean_bf_summary", ""))
    if delta_summary:
        title_lines.append(delta_summary)
    if bf_summary:
        title_lines.append(bf_summary)
    return "\n".join(title_lines)


def figure_filename_prefix(row: pd.Series) -> str:
    """Build a flat filename prefix for figure outputs from one fit.

    Args:
        row (pd.Series): Summary row describing one muscle-metric fit.

    Returns:
        str: Prefix such as ``soleus_area_`` used before each figure filename.
    """

    metric_text = str(row["metric"]).strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")
    muscle_text = str(row["muscle"]).strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")
    return f"{metric_text}_{muscle_text}_"


def compute_rhat_xlim(rhat_values: np.ndarray) -> tuple[float, float]:
    """Compute symmetric R-hat x-limits centered exactly at one.

    Args:
        rhat_values (np.ndarray): Flattened array of R-hat values for plotted parameters.

    Returns:
        tuple[float, float]: Symmetric lower and upper bounds around ``1.0``.
    """

    finite_values = np.asarray(rhat_values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return (0.95, 1.05)

    max_deviation = float(np.max(np.abs(finite_values - 1.0)))
    half_width = max(0.02, 1.12 * max_deviation)
    half_width = min(0.5, half_width)
    return (1.0 - half_width, 1.0 + half_width)


def clean_forest_ytick_label(label: str) -> str:
    """Convert raw ArviZ forest labels into cleaner parameter/group labels.

    Args:
        label (str): Raw y-axis tick label from a forest plot.

    Returns:
        str: Compact label using short genotype names when available.
    """

    compact = " ".join(str(label).split())
    if not compact:
        return compact

    bracket_groups = re.findall(r"\[([^\]]+)\]", compact)
    parameter_name = compact.split("[", maxsplit=1)[0].strip()
    replacements = {
        "Wildtype": "WT",
        "Calpain_3_Knockout": "KO",
        "Calpain 3 Knockout": "KO",
    }
    cleaned_groups = [replacements.get(group.strip(), group.strip()) for group in bracket_groups]
    if parameter_name and cleaned_groups:
        return "\n".join([parameter_name, *cleaned_groups])
    if cleaned_groups:
        return "\n".join(cleaned_groups)
    return replacements.get(compact, compact)


def compute_posterior_xlim(values: np.ndarray) -> tuple[float, float]:
    """Compute padded x-limits for posterior summaries in one parameter panel.

    Args:
        values (np.ndarray): Posterior summary values (HDI bounds and median).

    Returns:
        tuple[float, float]: Lower and upper x-limits with small data-driven padding.
    """

    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return (-1.0, 1.0)
    lower = float(np.min(finite_values))
    upper = float(np.max(finite_values))
    span = max(upper - lower, 1e-6)
    padding = max(0.05 * span, 1e-3)
    return (lower - padding, upper + padding)


def summarize_forest_parameters(
    idata: az.InferenceData,
    var_names: list[str],
    hdi_prob: float = 0.95,
) -> list[dict[str, Any]]:
    """Collect per-parameter summaries with separated indexed subgroup rows.

    Args:
        idata (az.InferenceData): Inference data containing posterior samples.
        var_names (list[str]): Top-level parameter names to summarize.
        hdi_prob (float): HDI probability mass for uncertainty intervals.

    Returns:
        list[dict[str, Any]]: Panel summaries containing top-level names and subgroup row summaries.
    """

    summaries: list[dict[str, Any]] = []
    rhat_dataset = az.rhat(idata, var_names=var_names)
    for variable_name in var_names:
        if variable_name not in idata.posterior.data_vars:
            continue
        variable_da = idata.posterior[variable_name]
        value_array = np.asarray(variable_da, dtype=float)
        non_sample_dims = [dim for dim in variable_da.dims if dim not in {"chain", "draw"}]
        non_sample_shape = tuple(variable_da.sizes[dim] for dim in non_sample_dims)

        if variable_name in rhat_dataset.data_vars:
            rhat_da = rhat_dataset[variable_name]
            rhat_array = np.asarray(rhat_da, dtype=float)
        else:
            rhat_da = None
            rhat_array = np.asarray([], dtype=float)

        rows: list[dict[str, float | str]] = []
        index_iter = np.ndindex(non_sample_shape) if non_sample_shape else [()]
        for index_tuple in index_iter:
            sample_index = (slice(None), slice(None), *index_tuple)
            posterior_values = value_array[sample_index].ravel()
            posterior_values = posterior_values[np.isfinite(posterior_values)]
            if posterior_values.size == 0:
                continue

            hdi_bounds = np.asarray(az.hdi(posterior_values, hdi_prob=hdi_prob), dtype=float)
            inner_bounds = np.asarray(np.quantile(posterior_values, [0.25, 0.75]), dtype=float)
            median_value = float(np.median(posterior_values))

            if rhat_da is not None:
                if index_tuple:
                    rhat_value = float(rhat_array[index_tuple])
                else:
                    rhat_value = float(rhat_array)
            else:
                rhat_value = float("nan")

            if index_tuple:
                subgroup_values = []
                for dim_name, dim_index in zip(non_sample_dims, index_tuple):
                    coord = variable_da.coords[dim_name].values[dim_index]
                    subgroup_values.append(str(coord))
                subgroup_label = clean_forest_ytick_label("[" + "][".join(subgroup_values) + "]")
            else:
                subgroup_label = clean_forest_ytick_label(variable_name)

            rows.append(
                {
                    "label": subgroup_label,
                    "hdi_low": float(hdi_bounds[0]),
                    "hdi_high": float(hdi_bounds[1]),
                    "inner_low": float(inner_bounds[0]),
                    "inner_high": float(inner_bounds[1]),
                    "median": median_value,
                    "rhat": rhat_value,
                }
            )

        if rows:
            summaries.append({"name": clean_forest_ytick_label(variable_name), "rows": rows})
    return summaries


def plot_trace_figure(
    idata: az.InferenceData,
    row: pd.Series,
    output_dir: Path,
    output_prefix: str,
) -> None:
    """Generate and save an ArviZ trace plot for one fit.

    Args:
        idata (az.InferenceData): Inference data for the current fit.
        row (pd.Series): Summary row describing the fit.
        output_dir (Path): Directory where the trace plot should be saved.
        output_prefix (str): Prefix prepended to the saved filename.

    Returns:
        None
    """

    var_names = available_var_names(
        idata=idata,
        requested=diagnostic_var_names(
            str(row["family"]),
            str(row.get("likelihood_name", "")),
        ),
    )
    if not var_names:
        return
    axes = az.plot_trace(
        idata,
        var_names=var_names,
        compact=False,
        legend=False,
        figsize=(16, max(8, 3.3 * len(var_names))),
        show=False,
    )
    fig = axes_to_figure(axes)
    style_figure_axes(fig=fig, grid_axis="both", xtick_rotation=XTICK_ROTATION)
    for ax in fig.axes:
        title = str(ax.get_title())
        if title:
            ax.set_title(" ".join(title.splitlines()))
    fig.suptitle(f"{fit_title(row)} - Trace plot", y=0.98, fontsize=14)
    finalize_figure_layout(
        fig=fig,
        title_top=0.94,
        hspace=TRACE_HSPACE,
        wspace=TRACE_RANK_WSPACE,
    )
    save_figure(fig=fig, output_path=output_dir / f"{output_prefix}trace.png")


def plot_rank_figure(
    idata: az.InferenceData,
    row: pd.Series,
    output_dir: Path,
    output_prefix: str,
) -> None:
    """Generate and save an ArviZ rank plot for one fit.

    Args:
        idata (az.InferenceData): Inference data for the current fit.
        row (pd.Series): Summary row describing the fit.
        output_dir (Path): Directory where the rank plot should be saved.
        output_prefix (str): Prefix prepended to the saved filename.

    Returns:
        None
    """

    var_names = available_var_names(
        idata=idata,
        requested=diagnostic_var_names(
            str(row["family"]),
            str(row.get("likelihood_name", "")),
        ),
    )
    if not var_names:
        return
    axes = az.plot_rank(
        idata,
        var_names=var_names,
        kind="bars",
        figsize=(16, max(7, 2.2 * len(var_names))),
        show=False,
    )
    fig = axes_to_figure(axes)
    style_figure_axes(fig=fig, grid_axis="y", xtick_rotation=XTICK_ROTATION)
    fig.suptitle(f"{fit_title(row)} - Rank plot", y=0.98, fontsize=14)
    finalize_figure_layout(
        fig=fig,
        title_top=0.87,
        hspace=TRACE_RANK_HSPACE,
        wspace=TRACE_RANK_WSPACE,
    )
    save_figure(fig=fig, output_path=output_dir / f"{output_prefix}rank.png")


def plot_forest_figure(
    idata: az.InferenceData,
    row: pd.Series,
    output_dir: Path,
    output_prefix: str,
) -> None:
    """Generate and save a forest plot with R-hat for one fit.

    Args:
        idata (az.InferenceData): Inference data for the current fit.
        row (pd.Series): Summary row describing the fit.
        output_dir (Path): Directory where the forest plot should be saved.
        output_prefix (str): Prefix prepended to the saved filename.

    Returns:
        None
    """

    var_names = available_var_names(
        idata=idata,
        requested=diagnostic_var_names(
            str(row["family"]),
            str(row.get("likelihood_name", "")),
        ),
    )
    parameter_summaries = summarize_forest_parameters(idata=idata, var_names=var_names, hdi_prob=0.95)
    if not parameter_summaries:
        return

    fig_height = max(6.0, 2.4 * len(parameter_summaries))
    fig, axes = plt.subplots(
        nrows=len(parameter_summaries),
        ncols=2,
        figsize=(12, fig_height),
        squeeze=False,
        gridspec_kw={"width_ratios": [4.4, 1.8]},
    )

    for index, summary in enumerate(parameter_summaries):
        posterior_axis = axes[index, 0]
        rhat_axis = axes[index, 1]
        row_summaries = list(summary["rows"])
        y_positions = np.arange(len(row_summaries), dtype=float)

        x_values: list[float] = []
        for y_position, row_summary in zip(y_positions, row_summaries):
            posterior_axis.hlines(
                y=y_position,
                xmin=float(row_summary["hdi_low"]),
                xmax=float(row_summary["hdi_high"]),
                color="tab:blue",
                linewidth=2.0,
                alpha=0.95,
            )
            posterior_axis.hlines(
                y=y_position,
                xmin=float(row_summary["inner_low"]),
                xmax=float(row_summary["inner_high"]),
                color="tab:blue",
                linewidth=6.0,
                alpha=0.45,
            )
            posterior_axis.plot(
                float(row_summary["median"]),
                y_position,
                marker="o",
                color="black",
                markersize=5.5,
                zorder=3,
            )
            x_values.extend(
                [
                    float(row_summary["hdi_low"]),
                    float(row_summary["hdi_high"]),
                    float(row_summary["median"]),
                ]
            )

        posterior_xlim = compute_posterior_xlim(values=np.asarray(x_values, dtype=float))
        posterior_axis.set_xlim(posterior_xlim)
        if len(row_summaries) == 1:
            posterior_axis.set_yticks([])
        else:
            posterior_axis.set_yticks(y_positions, labels=[str(row_summary["label"]) for row_summary in row_summaries])
        y_padding = 0.7
        posterior_axis.set_ylim(-y_padding, len(row_summaries) - 1 + y_padding)
        posterior_axis.invert_yaxis()
        posterior_axis.grid(axis="x", alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)
        posterior_axis.spines["top"].set_visible(False)
        posterior_axis.spines["right"].set_visible(False)
        posterior_axis.set_title(str(summary["name"]), loc="left", fontsize=11, pad=8)
        posterior_axis.set_xlabel("Posterior value" if index == len(parameter_summaries) - 1 else "")

        finite_rhats = [
            float(row_summary["rhat"])
            for row_summary in row_summaries
            if np.isfinite(float(row_summary["rhat"]))
        ]
        if finite_rhats:
            rhat_xlim = compute_rhat_xlim(rhat_values=np.asarray(finite_rhats, dtype=float))
            for y_position, row_summary in zip(y_positions, row_summaries):
                rhat_value = float(row_summary["rhat"])
                if np.isfinite(rhat_value):
                    rhat_axis.plot(
                        rhat_value,
                        y_position,
                        marker="o",
                        color="tab:orange",
                        markersize=6.5,
                        zorder=3,
                    )
                else:
                    rhat_axis.text(
                        x=1.0,
                        y=y_position,
                        s="N/A",
                        ha="center",
                        va="center",
                        fontsize=10,
                    )
        else:
            rhat_xlim = (0.99, 1.01)
            for y_position in y_positions:
                rhat_axis.text(
                    x=1.0,
                    y=y_position,
                    s="N/A",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
        rhat_axis.set_xlim(rhat_xlim)
        rhat_axis.set_xticks(np.asarray([0.99, 1.01], dtype=float))
        rhat_axis.axvline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        rhat_axis.set_yticks(y_positions, labels=[""] * len(y_positions))
        rhat_axis.set_ylim(-y_padding, len(row_summaries) - 1 + y_padding)
        rhat_axis.invert_yaxis()
        rhat_axis.grid(axis="x", alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)
        rhat_axis.spines["top"].set_visible(False)
        rhat_axis.spines["left"].set_visible(False)
        rhat_axis.spines["right"].set_visible(False)
        rhat_axis.set_xlabel("R-hat" if index == len(parameter_summaries) - 1 else "")

    fig.suptitle(f"{fit_title(row)} - Forest plot with R-hat", y=0.985, fontsize=14)
    finalize_figure_layout(fig=fig, title_top=0.9, hspace=1.05, wspace=0.3)
    save_figure(fig=fig, output_path=output_dir / f"{output_prefix}forest_rhat.png")


def plot_energy_figure(
    idata: az.InferenceData,
    row: pd.Series,
    output_dir: Path,
    output_prefix: str,
) -> None:
    """Generate and save an energy/BFMI plot for a warning-prone fit.

    Args:
        idata (az.InferenceData): Inference data for the current fit.
        row (pd.Series): Summary row describing the fit.
        output_dir (Path): Directory where the energy plot should be saved.
        output_prefix (str): Prefix prepended to the saved filename.

    Returns:
        None
    """

    axes = az.plot_energy(idata, figsize=(8, 5), show=False)
    fig = axes_to_figure(axes)
    style_figure_axes(fig=fig, grid_axis="both", xtick_rotation=0)
    for ax in fig.axes:
        if not str(ax.get_xlabel()).strip():
            ax.set_xlabel("Energy")
        if not str(ax.get_ylabel()).strip():
            ax.set_ylabel("Density")
    fig.suptitle(f"{fit_title(row)} - Energy / BFMI", y=0.985, fontsize=14)
    finalize_figure_layout(fig=fig, title_top=0.86)
    save_figure(fig=fig, output_path=output_dir / f"{output_prefix}energy.png")


def plot_ppc_density(
    idata: az.InferenceData,
    row: pd.Series,
    prepared: PreparedMetricData,
    output_dir: Path,
    output_prefix: str,
) -> None:
    """Generate and save a posterior predictive density overlay for one fit.

    Args:
        idata (az.InferenceData): Inference data containing posterior predictive draws.
        row (pd.Series): Summary row describing the fit.
        prepared (PreparedMetricData): Prepared metric-specific arrays matching the fit.
        output_dir (Path): Directory where the PPC density plot should be saved.
        output_prefix (str): Prefix prepended to the saved filename.

    Returns:
        None
    """

    observed, predictive_samples = ppc_arrays_on_response_scale(idata=idata, prepared=prepared)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_density_overlay(
        ax=ax,
        observed=observed,
        predictive_samples=predictive_samples,
        title="All observations",
        rng=np.random.default_rng(20260415),
    )
    ax.set_xlabel(metric_axis_label(str(row["metric"])))
    ax.set_ylabel("Density")
    ax.legend(
        handles=[
            matplotlib.lines.Line2D(
                [0], [0], color="red", alpha=0.7, linewidth=2.0, label="Observed data"
            ),
            matplotlib.lines.Line2D(
                [0],
                [0],
                color="tab:blue",
                linewidth=2.0,
                label="Posterior predictive median density",
            ),
        ],
        frameon=True,
    )
    style_figure_axes(fig=fig, grid_axis="both", xtick_rotation=0)
    fig.suptitle(f"{fit_title(row)} - Posterior predictive density", y=0.99, fontsize=14)
    finalize_figure_layout(fig=fig, title_top=0.9)
    save_figure(fig=fig, output_path=output_dir / f"{output_prefix}ppc_density.png")


def plot_ppc_density_by_condition(
    idata: az.InferenceData,
    row: pd.Series,
    prepared: PreparedMetricData,
    output_dir: Path,
    output_prefix: str,
) -> None:
    """Generate stratified PPC density overlays split by genotype condition.

    Args:
        idata (az.InferenceData): Inference data containing posterior predictive draws.
        row (pd.Series): Summary row describing the fit.
        prepared (PreparedMetricData): Prepared metric-specific arrays matching the fit.
        output_dir (Path): Directory where the stratified PPC plot should be saved.
        output_prefix (str): Prefix prepended to the saved filename.

    Returns:
        None
    """

    observed, predictive_samples = ppc_arrays_on_response_scale(idata=idata, prepared=prepared)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)
    rng = np.random.default_rng(20260416)
    shared_edges = histogram_bin_edges(observed=observed, predictive_samples=predictive_samples)
    genotype_labels = ((0, prepared.wt_label), (1, prepared.ko_label))
    for ax, (genotype_value, label) in zip(np.atleast_1d(axes), genotype_labels):
        mask = prepared.genotype_idx_obs == genotype_value
        plot_density_overlay(
            ax=ax,
            observed=observed[mask],
            predictive_samples=predictive_samples[:, mask],
            title=label,
            rng=rng,
            edges=shared_edges,
        )
        ax.set_xlabel(metric_axis_label(str(row["metric"])))
    np.atleast_1d(axes)[0].set_ylabel("Density")
    axes_array = np.atleast_1d(axes)
    axes_array[0].legend(
        handles=[
            matplotlib.lines.Line2D(
                [0], [0], color="red", alpha=0.7, linewidth=2.0, label="Observed data"
            ),
            matplotlib.lines.Line2D(
                [0],
                [0],
                color="tab:blue",
                linewidth=2.0,
                label="Posterior predictive median density",
            ),
        ],
        frameon=True,
    )
    style_figure_axes(fig=fig, grid_axis="both", xtick_rotation=0)
    fig.suptitle(f"{fit_title(row)} - PPC density by condition", y=0.99, fontsize=14)
    finalize_figure_layout(fig=fig, title_top=0.88)
    save_figure(fig=fig, output_path=output_dir / f"{output_prefix}ppc_density_by_condition.png")


def plot_ppc_quantiles(
    idata: az.InferenceData,
    row: pd.Series,
    prepared: PreparedMetricData,
    output_dir: Path,
    output_prefix: str,
) -> None:
    """Generate and save observed-vs-predictive quantile comparisons for one fit.

    Args:
        idata (az.InferenceData): Inference data containing posterior predictive draws.
        row (pd.Series): Summary row describing the fit.
        prepared (PreparedMetricData): Prepared metric-specific arrays matching the fit.
        output_dir (Path): Directory where the quantile comparison should be saved.
        output_prefix (str): Prefix prepended to the saved filename.

    Returns:
        None
    """

    observed, predictive_samples = ppc_arrays_on_response_scale(idata=idata, prepared=prepared)

    predictive_quantiles = np.quantile(predictive_samples, QUANTILE_LEVELS, axis=1)
    observed_quantiles = np.quantile(observed, QUANTILE_LEVELS)
    predictive_medians = np.median(predictive_quantiles, axis=1)
    predictive_hdis = np.asarray(
        [az.hdi(predictive_quantiles[index], hdi_prob=0.95) for index in range(len(QUANTILE_LEVELS))]
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = np.arange(len(QUANTILE_LEVELS), dtype=float)
    ax.errorbar(
        x_positions,
        predictive_medians,
        yerr=np.vstack(
            [
                predictive_medians - predictive_hdis[:, 0],
                predictive_hdis[:, 1] - predictive_medians,
            ]
        ),
        fmt="o",
        color="tab:blue",
        capsize=4,
        linewidth=1.5,
        label="Posterior predictive quantile 95% HDI",
    )
    ax.scatter(
        x_positions,
        observed_quantiles,
        color="black",
        marker="D",
        s=40,
        label="Observed quantiles",
        zorder=3,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"q={level:.2f}" for level in QUANTILE_LEVELS], rotation=0)
    ax.set_ylabel(metric_axis_label(str(row["metric"])))
    ax.set_title(f"{fit_title(row)} - Observed vs predictive quantiles", pad=16)
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)
    fig.subplots_adjust(top=0.9, bottom=0.14)
    save_figure(fig=fig, output_path=output_dir / f"{output_prefix}ppc_quantiles.png")


def plot_ppc_animal_summary(
    idata: az.InferenceData,
    row: pd.Series,
    prepared: PreparedMetricData,
    output_dir: Path,
    output_prefix: str,
) -> None:
    """Generate per-animal PPC summaries to separate pooled and conditional misfit.

    Args:
        idata (az.InferenceData): Inference data containing posterior predictive draws.
        row (pd.Series): Summary row describing the fit.
        prepared (PreparedMetricData): Prepared metric-specific arrays matching the fit.
        output_dir (Path): Directory where the animal-level PPC summary should be saved.
        output_prefix (str): Prefix prepended to the saved filename.

    Returns:
        None
    """

    observed, predictive_samples = ppc_arrays_on_response_scale(idata=idata, prepared=prepared)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    axes_array = np.atleast_1d(axes)
    genotype_panels = ((0, prepared.wt_label), (1, prepared.ko_label))
    for ax, (genotype_value, condition_label) in zip(axes_array, genotype_panels):
        animal_ids = [
            animal_index
            for animal_index, animal_label in enumerate(prepared.animal_labels)
            if animal_label.startswith(f"{condition_label}__")
        ]
        if not animal_ids:
            ax.set_title(f"{condition_label} (no animals)")
            continue
        positions = np.arange(len(animal_ids), dtype=float)
        observed_means: list[float] = []
        predictive_medians: list[float] = []
        predictive_lows: list[float] = []
        predictive_highs: list[float] = []
        labels: list[str] = []
        for animal_id in animal_ids:
            mask = prepared.animal_idx_obs == animal_id
            observed_values = observed[mask]
            predictive_group = predictive_samples[:, mask]
            predictive_means = np.mean(predictive_group, axis=1)
            hdi_low, hdi_high = np.asarray(az.hdi(predictive_means, hdi_prob=0.95), dtype=float)
            observed_means.append(float(np.mean(observed_values)))
            predictive_medians.append(float(np.median(predictive_means)))
            predictive_lows.append(float(hdi_low))
            predictive_highs.append(float(hdi_high))
            labels.append(animal_display_label(prepared.animal_labels[animal_id]))
        predictive_medians_array = np.asarray(predictive_medians, dtype=float)
        predictive_lows_array = np.asarray(predictive_lows, dtype=float)
        predictive_highs_array = np.asarray(predictive_highs, dtype=float)
        ax.errorbar(
            positions,
            predictive_medians_array,
            yerr=np.vstack(
                [
                    predictive_medians_array - predictive_lows_array,
                    predictive_highs_array - predictive_medians_array,
                ]
            ),
            fmt="o",
            color="tab:blue",
            capsize=4,
            linewidth=1.5,
            label="Posterior predictive mean 95% HDI",
        )
        ax.scatter(
            positions,
            observed_means,
            color="black",
            marker="D",
            s=36,
            label="Observed mean",
            zorder=3,
        )
        ax.set_title(condition_label, pad=10)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Animal / block")
    axes_array[0].set_ylabel(metric_axis_label(str(row["metric"])))
    axes_array[0].legend(frameon=True)
    style_figure_axes(fig=fig, grid_axis="y", xtick_rotation=45)
    fig.suptitle(f"{fit_title(row)} - PPC mean by animal", y=0.99, fontsize=14)
    finalize_figure_layout(fig=fig, title_top=0.84, hspace=0.4, wspace=0.18)
    save_figure(fig=fig, output_path=output_dir / f"{output_prefix}ppc_animal_mean.png")


def posterior_plot_specs(row: pd.Series) -> list[tuple[str, str, str, str]]:
    """Describe the biology-facing posterior plots to generate for one fit.

    Args:
        row (pd.Series): Summary row describing one muscle-metric fit.

    Returns:
        list[tuple[str, str, str, str]]: Posterior var name, summary column, filename stem, and label.
    """

    return [
        ("delta_mean_response", "delta_mean_summary", "delta_mean", "Genotype delta in mean"),
        (
            "delta_image_variance_response",
            "delta_image_variance_summary",
            "delta_image_variance",
            "Genotype delta in aggregated image variance",
        ),
        (
            "delta_mito_variance_response",
            "delta_mito_variance_summary",
            "delta_mito_variance",
            "Genotype delta in aggregated mito variance",
        ),
    ]


def plot_biology_posteriors(
    idata: az.InferenceData,
    row: pd.Series,
    output_dir: Path,
    output_prefix: str,
) -> None:
    """Generate and save posterior plots for biological effect summaries.

    Args:
        idata (az.InferenceData): Inference data for the current fit.
        row (pd.Series): Summary row describing the fit.
        output_dir (Path): Directory where posterior plots should be saved.
        output_prefix (str): Prefix prepended to the saved filename.

    Returns:
        None
    """

    available = set(biology_posterior_var_names(family=str(row["family"])))
    for variable_name, summary_column, filename_stem, quantity_label in posterior_plot_specs(row):
        if variable_name not in available or variable_name not in idata.posterior.data_vars:
            continue
        axes = az.plot_posterior(
            idata,
            var_names=[variable_name],
            hdi_prob=0.95,
            point_estimate="median",
            ref_val=0.0,
            figsize=(8, 4),
            show=False,
        )
        fig = axes_to_figure(axes)
        style_figure_axes(fig=fig, grid_axis="x", xtick_rotation=POSTERIOR_XTICK_ROTATION)
        figure_summary = str(row.get(summary_column, "")).strip()
        fig.suptitle(fit_title(row), y=0.99, fontsize=13)
        if fig.axes:
            title_text = quantity_label if not figure_summary else f"{quantity_label}\n{figure_summary}"
            fig.axes[0].set_title(title_text, fontsize=11, pad=18)
        finalize_figure_layout(fig=fig, title_top=0.8)
        save_figure(fig=fig, output_path=output_dir / f"{output_prefix}{filename_stem}.png")


def plot_bayesian_superplots(
    measurements_df: pd.DataFrame,
    row: pd.Series,
    output_dir: Path,
    output_prefix: str,
) -> None:
    """Generate observed-data superplots for one fit using the existing project style.

    Args:
        measurements_df (pd.DataFrame): Full cleaned measurements table.
        row (pd.Series): Summary row describing the fit.
        output_dir (Path): Root directory where superplot outputs should be saved.
        output_prefix (str): Prefix prepended to generated superplot filenames.

    Returns:
        None
    """

    subset = measurements_df.loc[measurements_df["Muscle"] == row["muscle"]].copy()
    title_text = superplot_title_text(row)
    fit_dir = output_dir
    plot_super_violin(
        data=subset,
        x="Muscle",
        y=str(row["metric"]),
        hue="Condition",
        block="Block",
        unit_dict=METRIC_UNITS,
        save_dir=fit_dir,
        title_override=title_text,
        filename_prefix=output_prefix,
    )
    plot_super_beeswarm(
        data=subset,
        x="Muscle",
        y=str(row["metric"]),
        hue="Condition",
        block="Block",
        unit_dict=METRIC_UNITS,
        save_dir=fit_dir,
        title_override=title_text,
        filename_prefix=output_prefix,
    )


def generate_fit_visualizations(
    row: pd.Series,
    idata: az.InferenceData,
    prepared: PreparedMetricData,
    measurements_df: pd.DataFrame,
    figure_root: Path,
) -> None:
    """Generate every requested visualization for one muscle-metric fit.

    Args:
        row (pd.Series): Summary row describing one muscle-metric fit.
        idata (az.InferenceData): Inference data for the current fit.
        prepared (PreparedMetricData): Prepared metric-specific arrays matching the fit.
        measurements_df (pd.DataFrame): Full cleaned measurements table.
        figure_root (Path): Root figure directory for the visualization workflow.

    Returns:
        None
    """

    output_prefix = figure_filename_prefix(row=row)
    output_dir = figure_root
    superplot_dir = figure_root

    plot_trace_figure(idata=idata, row=row, output_dir=output_dir, output_prefix=output_prefix)
    plot_rank_figure(idata=idata, row=row, output_dir=output_dir, output_prefix=output_prefix)
    plot_forest_figure(idata=idata, row=row, output_dir=output_dir, output_prefix=output_prefix)
    if str(row.get("engine_fit_status", row.get("fit_status", ""))) == "warn":
        plot_energy_figure(idata=idata, row=row, output_dir=output_dir, output_prefix=output_prefix)

    plot_ppc_density(
        idata=idata,
        row=row,
        prepared=prepared,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )
    plot_ppc_density_by_condition(
        idata=idata,
        row=row,
        prepared=prepared,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )
    plot_ppc_quantiles(
        idata=idata,
        row=row,
        prepared=prepared,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )
    plot_ppc_animal_summary(
        idata=idata,
        row=row,
        prepared=prepared,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )
    plot_biology_posteriors(idata=idata, row=row, output_dir=output_dir, output_prefix=output_prefix)
    plot_bayesian_superplots(
        measurements_df=measurements_df,
        row=row,
        output_dir=superplot_dir,
        output_prefix=output_prefix,
    )


def main() -> None:
    """Run the full Bayesian visualization workflow.

    Args:
        None

    Returns:
        None
    """

    args = parse_args()
    config = load_hierarchical_bayes_config(
        path=args.config,
        positive_metrics=tuple(POSITIVE_METRICS),
        bounded_metrics=tuple(BOUNDED_METRICS),
    )
    fit_configs = repeated_fit_configs(config)
    if not fit_configs:
        print(f"No fits have repeat=true in {config.config_path}; nothing to do.")
        return
    measurements_df = pd.read_csv(config.paths.input_csv)
    validate_config_muscles(
        config=config,
        valid_muscles=set(measurements_df["Muscle"].dropna().unique().tolist()),
    )
    summary_df = load_or_refresh_summary(
        summary_path=config.paths.summary_csv,
        trace_dir=config.paths.trace_dir,
        refresh_mode=config.runtime.visualization_refresh_mode,
        config_path=args.config,
        fit_configs=fit_configs,
    )

    for _, row in summary_df.iterrows():
        fit_config = fit_config_for_pair(
            config=config,
            muscle=str(row["muscle"]),
            metric=str(row["metric"]),
        )
        trace_path = resolve_trace_path(row=row, trace_dir=config.paths.trace_dir)
        if not trace_path.exists():
            raise FileNotFoundError(f"Missing trace file for {row['muscle']} / {row['metric']}: {trace_path}")
        print(f"Generating figures for {row['metric']} ({row['muscle']})...")
        idata = az.from_netcdf(trace_path)
        prepared = prepare_data_for_row(
            row=row,
            measurements_df=measurements_df,
            fit_config=fit_config,
        )
        idata = ensure_augmented_idata(prepared=prepared, idata=idata)
        generate_fit_visualizations(
            row=row,
            idata=idata,
            prepared=prepared,
            measurements_df=measurements_df,
            figure_root=config.paths.figure_root,
        )


if __name__ == "__main__":
    main()
