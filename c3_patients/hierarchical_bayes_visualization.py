#!/usr/bin/env python3
"""Visualize CAPN3 patient hierarchical Bayesian fits independently."""

from __future__ import annotations

import argparse
import math
import zlib
from pathlib import Path
from typing import Any, Callable

import arviz as az
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from hierarchical_bayes_config import (
    DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH,
    PatientBayesAnalysisConfig,
    PatientBayesFitConfig,
    enabled_fit_configs,
    load_hierarchical_bayes_config,
)
from hierarchical_bayes_metrics import (
    INSTANCE_CLUSTERING_METRICS,
    PreparedPatientMetricData,
    SiteScenario,
    apply_site_scenario,
    build_site_scenarios,
    flatten_posterior,
    load_measurements,
    prepare_metric_data,
    simulate_predictive_subset_indexed,
)


PRIMARY_SCENARIO = "primary_exclude_unknown"
SCENARIO_ORDER = (
    PRIMARY_SCENARIO,
    "sensitivity_DD",
    "sensitivity_DQ",
    "sensitivity_QD",
    "sensitivity_QQ",
)
CONDITION_LABELS = ("CTRL", "CAPN3")
FIGURE_DPI = 300
HDI_PROBABILITY = 0.95
DIAGNOSTIC_LAYOUT_RECT: tuple[float, float, float, float] = (0.0, 0.05, 1.0, 0.94)
TRACE_HSPACE = 0.78
TRACE_RANK_HSPACE = 0.62
RHAT_CENTER = 1.0
RHAT_THRESHOLD_LOW = 0.99
RHAT_THRESHOLD_HIGH = 1.01
HDI_TYPICAL_WIDTH_MULTIPLIER = 3.0
METRIC_UNITS: dict[str, str] = {
    "Minimum_Feret_Diameter": "nm",
    "Eccentricity": "unitless",
    "Circularity": "unitless",
    "Solidity": "unitless",
    "NND": "nm",
    "3NND": "nm",
    "5NND": "nm",
    "Voronoi_Cell_Area": "nm²",
    "Instance_count": "mitochondria / image",
    "Minimum_Feret_Diameter_mean": "nm",
    "Eccentricity_mean": "unitless",
    "Circularity_mean": "unitless",
    "Solidity_mean": "unitless",
    "NND_center_mean": "nm",
    "3NND_center_mean": "nm",
    "5NND_center_mean": "nm",
    "Voronoi_Cell_Area_center_mean": "nm²",
    "Voronoi_Cell_Area_center_cv": "unitless",
    "Ripley_L_integral": "integrated nm²",
    "Pair_Correlation_integral": "integrated unitless",
}


def stable_seed(base_seed: int, *parts: str) -> int:
    """Derive a deterministic 32-bit seed from labels.

    Args:
        base_seed (int): User-provided base seed.
        *parts (str): Stable task labels.

    Returns:
        int: Deterministic NumPy-compatible seed.
    """

    token = "::".join((str(base_seed), *parts)).encode("utf-8")
    return int(zlib.crc32(token) & 0xFFFFFFFF)


def figure_path(
    root: Path,
    category: str,
    fit_id: str,
    suffix: str,
) -> Path:
    """Build a categorized figure path.

    Args:
        root (Path): Configured figure root.
        category (str): Figure family.
        fit_id (str): Stable fit identifier.
        suffix (str): Figure suffix without extension.

    Returns:
        Path: PNG output path.
    """

    return root / category / f"{fit_id}__{suffix}.png"


def save_figure(
    figure: plt.Figure,
    output_path: Path,
    overwrite: bool,
) -> bool:
    """Save and close a figure unless an existing file should be retained.

    Args:
        figure (plt.Figure): Matplotlib figure.
        output_path (Path): PNG destination.
        overwrite (bool): Whether to replace an existing figure.

    Returns:
        bool: Whether a new file was written.
    """

    if output_path.exists() and not overwrite:
        plt.close(figure)
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    return True


def quality_text(row: pd.Series) -> str:
    """Format sampler and PPC quality information.

    Args:
        row (pd.Series): One summary row.

    Returns:
        str: Compact warning-aware quality text.
    """

    parts = [
        f"divergences={int(row.get('divergences', 0))}",
        f"max R-hat={float(row.get('rhat_max', np.nan)):.3f}",
        f"min bulk ESS={float(row.get('ess_bulk_min', np.nan)):.0f}",
        f"PPC={row.get('ppc_fit_status', 'unknown')}",
    ]
    if bool(row.get("exploratory", False)):
        parts.append("EXPLORATORY spatial estimator")
    return " | ".join(parts)


def add_quality_banner(figure: plt.Figure, row: pd.Series) -> None:
    """Add an inferential-quality banner to a figure.

    Args:
        figure (plt.Figure): Target figure.
        row (pd.Series): Summary row.

    Returns:
        None: Mutates the figure text.
    """

    warning = int(row.get("divergences", 0)) > 0 or row.get("fit_status") != "ok"
    figure.text(
        0.5,
        0.005,
        quality_text(row),
        ha="center",
        va="bottom",
        fontsize=8,
        color="darkred" if warning else "0.3",
        weight="bold" if warning else "normal",
    )


def apply_tight_layout(
    figure: plt.Figure,
    rect: tuple[float, float, float, float] = DIAGNOSTIC_LAYOUT_RECT,
) -> None:
    """Apply tight layout while reserving room for the title and quality banner.

    Args:
        figure (plt.Figure): Figure whose subplot spacing should be tightened.
        rect (tuple[float, float, float, float]): ``tight_layout`` rectangle
            ``(left, bottom, right, top)`` in figure coordinates.

    Returns:
        None: Mutates the figure layout.
    """

    figure.tight_layout(rect=rect)


def increase_row_spacing(figure: plt.Figure, hspace: float) -> None:
    """Increase vertical space between subplot rows after tight layout.

    Args:
        figure (plt.Figure): Figure whose row spacing should be increased.
        hspace (float): Matplotlib ``hspace`` value passed to ``subplots_adjust``.

    Returns:
        None: Mutates the figure layout without changing outer margins.
    """

    figure.subplots_adjust(hspace=hspace)


def hide_inner_tick_labels(axes: Any) -> None:
    """Keep x labels on the bottom row and y labels on the left column only.

    Args:
        axes (Any): Axes, sequence of axes, or ndarray returned by an ArviZ plot.

    Returns:
        None: Mutates tick and axis labels on the provided axes.
    """

    axes_list = [
        axis
        for axis in np.asarray(axes, dtype=object).ravel()
        if axis is not None
    ]
    if not axes_list:
        return
    row_positions = sorted({round(axis.get_position().y0, 3) for axis in axes_list})
    column_positions = sorted({round(axis.get_position().x0, 3) for axis in axes_list})
    bottom_row = row_positions[0]
    left_column = column_positions[0]
    for axis in axes_list:
        if round(axis.get_position().y0, 3) != bottom_row:
            axis.set_xlabel("")
            axis.tick_params(axis="x", labelbottom=False)
        if round(axis.get_position().x0, 3) != left_column:
            axis.set_ylabel("")
            axis.tick_params(axis="y", labelleft=False)


def robust_hdi_xlim(lows: np.ndarray, highs: np.ndarray) -> tuple[float, float]:
    """Compute a shared HDI x-limit from typical interval widths.

    Args:
        lows (np.ndarray): Lower HDI endpoints.
        highs (np.ndarray): Upper HDI endpoints.

    Returns:
        tuple[float, float]: Padded lower and upper x-limits that ignore
            intervals wider than ``HDI_TYPICAL_WIDTH_MULTIPLIER`` times the
            median width, falling back to the IQR of all endpoints.
    """

    lower_bounds = np.asarray(lows, dtype=float)
    upper_bounds = np.asarray(highs, dtype=float)
    finite = np.isfinite(lower_bounds) & np.isfinite(upper_bounds)
    lower_bounds = lower_bounds[finite]
    upper_bounds = upper_bounds[finite]
    if lower_bounds.size == 0:
        return (-1.0, 1.0)
    widths = np.maximum(upper_bounds - lower_bounds, 0.0)
    median_width = float(np.median(widths))
    typical = widths <= max(HDI_TYPICAL_WIDTH_MULTIPLIER * median_width, 1e-12)
    if np.any(typical):
        selected_low = lower_bounds[typical]
        selected_high = upper_bounds[typical]
        lower = float(np.min(selected_low))
        upper = float(np.max(selected_high))
    else:
        bounds = np.concatenate([lower_bounds, upper_bounds])
        quartile_low, quartile_high = np.quantile(bounds, [0.25, 0.75])
        lower = float(quartile_low)
        upper = float(quartile_high)
    span = max(upper - lower, 1e-6)
    padding = 0.05 * span
    return (lower - padding, upper + padding)


def symmetric_rhat_xlim(rhat_values: np.ndarray) -> tuple[float, float]:
    """Compute symmetric R-hat x-limits centered at one.

    Args:
        rhat_values (np.ndarray): R-hat values for the plotted parameters.

    Returns:
        tuple[float, float]: Lower and upper bounds centered at ``RHAT_CENTER``
            with half-width of at least ``0.02`` so the 0.99 and 1.01
            thresholds remain visible.
    """

    finite_values = np.asarray(rhat_values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return (RHAT_CENTER - 0.02, RHAT_CENTER + 0.02)
    max_deviation = float(np.max(np.abs(finite_values - RHAT_CENTER)))
    half_width = max(0.02, 1.12 * max_deviation)
    half_width = min(0.5, half_width)
    return (RHAT_CENTER - half_width, RHAT_CENTER + half_width)


def hdi_column_names(summary: pd.DataFrame) -> tuple[str, str]:
    """Return the lower and upper HDI column names from an ArviZ summary.

    Args:
        summary (pd.DataFrame): Output of ``az.summary``.

    Returns:
        tuple[str, str]: Lower and upper HDI column names.

    Raises:
        KeyError: If the summary does not contain two HDI columns.
    """

    columns = [str(column) for column in summary.columns if str(column).startswith("hdi_")]
    if len(columns) < 2:
        raise KeyError(f"Expected two HDI columns in summary; found {columns}.")

    ordered = sorted(columns, key=hdi_column_percent)
    return ordered[0], ordered[-1]


def hdi_column_percent(column: str) -> float:
    """Parse the percentage encoded in an HDI column name.

    Args:
        column (str): Column name such as ``hdi_2.5%``.

    Returns:
        float: Numeric percentage used for sorting.
    """

    return float(column.replace("hdi_", "").replace("%", ""))


def order_summary_rows(summary: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Reorder summary rows to follow the requested variable-name sequence.

    Args:
        summary (pd.DataFrame): ArviZ summary indexed by parameter labels.
        names (list[str]): Top-level posterior variable names in display order.

    Returns:
        pd.DataFrame: Rows matching ``names``, including indexed coordinates.
    """

    labels = summary.index.astype(str)
    selected: list[int] = []
    used: set[int] = set()
    for name in names:
        for position, label in enumerate(labels):
            if position in used:
                continue
            if label == name or label.startswith(f"{name}["):
                selected.append(position)
                used.add(position)
    if not selected:
        return summary.iloc[0:0].copy()
    return summary.iloc[selected].copy()


def forest_parameter_rows(
    idata: az.InferenceData,
    names: list[str],
) -> pd.DataFrame:
    """Build one forest-plot row per posterior parameter coordinate.

    Args:
        idata (az.InferenceData): Saved posterior.
        names (list[str]): Top-level posterior variable names to include.

    Returns:
        pd.DataFrame: Median, 95% HDI, R-hat, and ESS columns ordered by
            ``names``.
    """

    empty = pd.DataFrame(
        columns=["median", "hdi_low", "hdi_high", "r_hat", "ess_bulk", "ess_tail"]
    )
    if not names:
        return empty
    summary = az.summary(
        idata,
        var_names=names,
        hdi_prob=HDI_PROBABILITY,
        kind="all",
        round_to=None,
    )
    summary = order_summary_rows(summary, names)
    if summary.empty:
        return empty
    low_column, high_column = hdi_column_names(summary)
    median_summary = az.summary(
        idata,
        var_names=names,
        kind="stats",
        stat_focus="median",
        round_to=None,
    )
    median_summary = order_summary_rows(median_summary, names)
    aligned_median = (
        median_summary["median"].reindex(summary.index)
        if "median" in median_summary.columns
        else summary["mean"]
    )
    n_rows = len(summary)
    rows = pd.DataFrame(
        {
            "median": pd.to_numeric(aligned_median, errors="coerce").to_numpy(dtype=float),
            "hdi_low": pd.to_numeric(summary[low_column], errors="coerce").to_numpy(dtype=float),
            "hdi_high": pd.to_numeric(summary[high_column], errors="coerce").to_numpy(dtype=float),
            "r_hat": pd.to_numeric(summary["r_hat"], errors="coerce").to_numpy(dtype=float)
            if "r_hat" in summary.columns
            else np.full(n_rows, np.nan),
            "ess_bulk": pd.to_numeric(summary["ess_bulk"], errors="coerce").to_numpy(dtype=float)
            if "ess_bulk" in summary.columns
            else np.full(n_rows, np.nan),
            "ess_tail": pd.to_numeric(summary["ess_tail"], errors="coerce").to_numpy(dtype=float)
            if "ess_tail" in summary.columns
            else np.full(n_rows, np.nan),
        },
        index=summary.index.astype(str),
    )
    return rows


def draw_clipped_hdi(
    axis: plt.Axes,
    y_value: float,
    median: float,
    hdi_low: float,
    hdi_high: float,
    xlim: tuple[float, float],
) -> None:
    """Draw one HDI bar, clipping outliers and annotating truncated intervals.

    Args:
        axis (plt.Axes): Forest-plot axis.
        y_value (float): Vertical position of the interval.
        median (float): Posterior median used as the point marker.
        hdi_low (float): Lower 95% HDI endpoint.
        hdi_high (float): Upper 95% HDI endpoint.
        xlim (tuple[float, float]): Shared robust x-limits.

    Returns:
        None: Mutates ``axis``.
    """

    x_min, x_max = xlim
    span = max(x_max - x_min, 1e-12)
    arrow_len = 0.045 * span
    visible_low = min(max(hdi_low, x_min), x_max)
    visible_high = max(min(hdi_high, x_max), x_min)
    left_clipped = bool(np.isfinite(hdi_low) and hdi_low < x_min)
    right_clipped = bool(np.isfinite(hdi_high) and hdi_high > x_max)
    if visible_high > visible_low:
        axis.hlines(
            y_value,
            visible_low,
            visible_high,
            color="midnightblue",
            linewidth=2.2,
            zorder=2,
        )
    if left_clipped:
        axis.annotate(
            "",
            xy=(x_min, y_value),
            xytext=(x_min + arrow_len, y_value),
            arrowprops={"arrowstyle": "-|>", "color": "midnightblue", "lw": 1.4},
            annotation_clip=False,
        )
    if right_clipped:
        axis.annotate(
            "",
            xy=(x_max, y_value),
            xytext=(x_max - arrow_len, y_value),
            arrowprops={"arrowstyle": "-|>", "color": "midnightblue", "lw": 1.4},
            annotation_clip=False,
        )
    if np.isfinite(median) and x_min <= median <= x_max:
        axis.plot(median, y_value, "o", color="black", markersize=5.0, zorder=3)
    if left_clipped or right_clipped:
        text_x = float(np.clip(median, x_min, x_max)) if np.isfinite(median) else 0.5 * (x_min + x_max)
        axis.annotate(
            f"[{hdi_low:.3g}, {hdi_high:.3g}]",
            xy=(text_x, y_value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.25",
        )


def load_summary(config: PatientBayesAnalysisConfig) -> pd.DataFrame:
    """Load and validate the configured Bayesian summary.

    Args:
        config (PatientBayesAnalysisConfig): Analysis configuration.

    Returns:
        pd.DataFrame: Summary with unique fit/scenario rows.
    """

    if not config.paths.summary_csv.exists():
        raise FileNotFoundError(config.paths.summary_csv)
    summary = pd.read_csv(config.paths.summary_csv, low_memory=False)
    required = {
        "analysis_id",
        "fit_id",
        "metric",
        "site_scenario",
        "trace_path",
        "capn3_effect_response",
        "capn3_effect_response_hdi_low",
        "capn3_effect_response_hdi_high",
        "divergences",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise KeyError(f"Summary CSV is missing visualization columns: {missing}")
    summary = summary.loc[summary["analysis_id"] == config.analysis_id].copy()
    duplicated = summary.duplicated(["analysis_id", "fit_id", "site_scenario"])
    if duplicated.any():
        raise ValueError("Summary contains duplicate fit/site-scenario rows.")
    return summary


def selected_fits(
    config: PatientBayesAnalysisConfig,
    requested_ids: list[str] | None,
) -> list[PatientBayesFitConfig]:
    """Return enabled visualization fits in YAML order.

    Args:
        config (PatientBayesAnalysisConfig): Analysis configuration.
        requested_ids (list[str] | None): Optional targeted IDs.

    Returns:
        list[PatientBayesFitConfig]: Selected fits.
    """

    fits = enabled_fit_configs(config)
    if not requested_ids:
        return fits
    unknown = sorted(set(requested_ids) - {fit.fit_id for fit in fits})
    if unknown:
        raise ValueError(f"Unknown or disabled fit IDs: {unknown}")
    requested = set(requested_ids)
    return [fit for fit in fits if fit.fit_id in requested]


def summary_row(
    summary: pd.DataFrame,
    fit_id: str,
    scenario_name: str,
) -> pd.Series:
    """Select exactly one fit/scenario summary row.

    Args:
        summary (pd.DataFrame): Loaded summary.
        fit_id (str): Fit identifier.
        scenario_name (str): Site scenario.

    Returns:
        pd.Series: Matching summary row.
    """

    rows = summary.loc[
        (summary["fit_id"] == fit_id)
        & (summary["site_scenario"] == scenario_name)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one summary row for {fit_id}/{scenario_name}; found {len(rows)}."
        )
    return rows.iloc[0]


def validate_fit_inventory(
    summary: pd.DataFrame,
    fits: list[PatientBayesFitConfig],
) -> None:
    """Require all five summary rows and trace files for selected fits.

    Args:
        summary (pd.DataFrame): Loaded analysis summary.
        fits (list[PatientBayesFitConfig]): Selected fits.

    Returns:
        None: Raises for incomplete or missing artifacts.
    """

    for fit in fits:
        rows = summary.loc[
            (summary["fit_id"] == fit.fit_id)
            & (summary["site_scenario"].isin(SCENARIO_ORDER))
        ]
        scenarios = set(rows["site_scenario"].astype(str))
        if scenarios != set(SCENARIO_ORDER):
            missing = sorted(set(SCENARIO_ORDER) - scenarios)
            raise ValueError(f"Fit '{fit.fit_id}' is missing scenarios: {missing}")
        for trace_value in rows["trace_path"]:
            if not isinstance(trace_value, str) or not Path(trace_value).exists():
                raise FileNotFoundError(f"Missing trace for fit '{fit.fit_id}': {trace_value}")


def load_trace(row: pd.Series, config: PatientBayesAnalysisConfig) -> az.InferenceData:
    """Load a trace and validate its identifying attributes.

    Args:
        row (pd.Series): Summary row.
        config (PatientBayesAnalysisConfig): Analysis configuration.

    Returns:
        az.InferenceData: Validated saved posterior.
    """

    trace_value = str(row.get("trace_path", "")).strip()
    path = Path(trace_value) if trace_value else (
        config.paths.trace_dir
        / f"{config.analysis_id}__{row['fit_id']}__{str(row['site_scenario']).lower()}.nc"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    idata = az.from_netcdf(path)
    for key in ("analysis_id", "fit_id", "metric", "site_scenario"):
        expected = str(row[key])
        actual = str(idata.attrs.get(key, ""))
        if actual and actual != expected:
            raise ValueError(
                f"Trace attribute mismatch for {key}: {actual!r} != {expected!r}"
            )
    return idata


def diagnostic_variable_names(
    idata: az.InferenceData,
    analysis_id: str,
) -> list[str]:
    """Select biological and scale parameters for diagnostics.

    Args:
        idata (az.InferenceData): Saved posterior.
        analysis_id (str): Instance or image-summary level.

    Returns:
        list[str]: Existing posterior variable names.
    """

    candidates = [
        "beta_disease",
        "beta_site",
        "beta_male",
        "beta_maturation",
        "beta_aging",
        "beta_compartment",
        "interaction_disease_site",
        "interaction_disease_compartment",
        "subtype_offset",
        "subtype_deviation",
        "sigma_subtype",
        "sigma_patient",
        "sigma_observation",
        "kappa",
        "negative_binomial_alpha",
        "nu_minus_two",
    ]
    if analysis_id == "instance":
        candidates.append("sigma_image")
    return [name for name in candidates if name in idata.posterior]


def plot_trace(
    idata: az.InferenceData,
    names: list[str],
    row: pd.Series,
) -> plt.Figure:
    """Create an ArviZ trace figure.

    Args:
        idata (az.InferenceData): Saved posterior.
        names (list[str]): Variables to include.
        row (pd.Series): Summary row.

    Returns:
        plt.Figure: Trace figure.
    """

    axes = az.plot_trace(
        idata,
        var_names=names,
        compact=True,
        figsize=(14, max(8, 3.0 * len(names))),
    )
    figure = np.asarray(axes).ravel()[0].figure
    figure.suptitle(f"{row['metric']} — primary trace diagnostics")
    apply_tight_layout(figure)
    increase_row_spacing(figure, TRACE_HSPACE)
    add_quality_banner(figure, row)
    return figure


def plot_rank(
    idata: az.InferenceData,
    names: list[str],
    row: pd.Series,
) -> plt.Figure:
    """Create chain-rank diagnostics.

    Args:
        idata (az.InferenceData): Saved posterior.
        names (list[str]): Variables to include.
        row (pd.Series): Summary row.

    Returns:
        plt.Figure: Rank figure.
    """

    axes = az.plot_rank(
        idata,
        var_names=names,
        kind="bars",
        figsize=(13, max(7, 2.5 * math.ceil(len(names) / 2))),
    )
    figure = np.asarray(axes).ravel()[0].figure
    hide_inner_tick_labels(axes)
    figure.suptitle(f"{row['metric']} — chain rank diagnostics")
    apply_tight_layout(figure)
    increase_row_spacing(figure, TRACE_RANK_HSPACE)
    add_quality_banner(figure, row)
    return figure


def plot_forest(
    idata: az.InferenceData,
    names: list[str],
    row: pd.Series,
) -> plt.Figure:
    """Create posterior forest, R-hat, and ESS panels.

    Args:
        idata (az.InferenceData): Saved posterior.
        names (list[str]): Variables to include.
        row (pd.Series): Summary row.

    Returns:
        plt.Figure: Forest diagnostic figure.
    """

    parameters = forest_parameter_rows(idata, names)
    n_rows = max(len(parameters), 1)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13, max(7, 0.45 * n_rows + 3.5)),
        sharey=True,
        gridspec_kw={"width_ratios": [3.2, 1.1, 1.1]},
    )
    forest_axis, rhat_axis, ess_axis = axes
    if parameters.empty:
        forest_axis.set_xlabel("Posterior (95% HDI)")
        rhat_axis.set_xlabel("R-hat")
        ess_axis.set_xlabel("ESS")
        figure.suptitle(f"{row['metric']} — posterior and convergence")
        apply_tight_layout(figure)
        add_quality_banner(figure, row)
        return figure

    positions = np.arange(len(parameters), dtype=float)
    hdi_xlim = robust_hdi_xlim(parameters["hdi_low"].to_numpy(), parameters["hdi_high"].to_numpy())
    rhat_xlim = symmetric_rhat_xlim(parameters["r_hat"].to_numpy())
    for y_value, (_, record) in zip(positions, parameters.iterrows(), strict=True):
        draw_clipped_hdi(
            axis=forest_axis,
            y_value=float(y_value),
            median=float(record["median"]),
            hdi_low=float(record["hdi_low"]),
            hdi_high=float(record["hdi_high"]),
            xlim=hdi_xlim,
        )
        rhat_value = float(record["r_hat"])
        if np.isfinite(rhat_value):
            rhat_axis.plot(
                rhat_value,
                y_value,
                marker="o",
                color="tab:orange",
                markersize=6.0,
                zorder=3,
            )
        ess_bulk = float(record["ess_bulk"])
        ess_tail = float(record["ess_tail"])
        if np.isfinite(ess_bulk):
            ess_axis.plot(
                ess_bulk,
                y_value,
                marker="o",
                color="midnightblue",
                markersize=6.0,
                zorder=3,
                label="bulk" if y_value == positions[0] else None,
            )
        if np.isfinite(ess_tail):
            ess_axis.plot(
                ess_tail,
                y_value,
                marker="s",
                color="steelblue",
                markersize=5.5,
                zorder=3,
                label="tail" if y_value == positions[0] else None,
            )

    forest_axis.set_yticks(positions, parameters.index.astype(str))
    forest_axis.set_xlim(hdi_xlim)
    forest_axis.set_xlabel("Posterior (95% HDI)")
    forest_axis.grid(axis="x", alpha=0.2)
    forest_axis.invert_yaxis()

    rhat_axis.set_xlim(rhat_xlim)
    rhat_axis.axvline(RHAT_CENTER, color="0.75", linestyle="--", linewidth=1.0)
    rhat_axis.axvline(RHAT_THRESHOLD_LOW, color="red", linestyle="--", linewidth=1.0)
    rhat_axis.axvline(RHAT_THRESHOLD_HIGH, color="red", linestyle="--", linewidth=1.0)
    rhat_axis.set_xlabel("R-hat")
    rhat_axis.grid(axis="x", alpha=0.2)

    ess_axis.set_xlabel("ESS")
    ess_axis.grid(axis="x", alpha=0.2)
    handles, labels = ess_axis.get_legend_handles_labels()
    if handles:
        ess_axis.legend(handles, labels, frameon=False, fontsize=8)

    figure.suptitle(f"{row['metric']} — posterior and convergence")
    apply_tight_layout(figure)
    add_quality_banner(figure, row)
    return figure


def plot_energy(idata: az.InferenceData, row: pd.Series) -> plt.Figure:
    """Create an energy/BFMI diagnostic figure.

    Args:
        idata (az.InferenceData): Saved posterior.
        row (pd.Series): Summary row.

    Returns:
        plt.Figure: Energy figure.
    """

    axes = az.plot_energy(idata, figsize=(9, 5))
    figure = np.asarray(axes).ravel()[0].figure
    bfmi = np.asarray(az.bfmi(idata), dtype=float)
    figure.suptitle(
        f"{row['metric']} — energy diagnostic (BFMI min={np.min(bfmi):.3f})"
    )
    apply_tight_layout(figure)
    add_quality_banner(figure, row)
    return figure


def plot_divergence_locations(
    idata: az.InferenceData,
    row: pd.Series,
) -> plt.Figure:
    """Plot divergent transitions against key hierarchy scales.

    Args:
        idata (az.InferenceData): Saved posterior.
        row (pd.Series): Summary row.

    Returns:
        plt.Figure: Divergence-location scatter panels.
    """

    divergent = np.asarray(idata.sample_stats["diverging"]).reshape(-1).astype(bool)
    disease = flatten_posterior(idata, "beta_disease")
    scales = [
        name
        for name in ("sigma_subtype", "sigma_patient", "sigma_image")
        if name in idata.posterior
    ]
    figure, axes = plt.subplots(
        1,
        len(scales),
        figsize=(5.2 * len(scales), 4.5),
        squeeze=False,
    )
    for axis, name in zip(axes.ravel(), scales, strict=True):
        scale = flatten_posterior(idata, name)
        axis.scatter(
            disease[~divergent],
            scale[~divergent],
            s=7,
            alpha=0.18,
            color="0.45",
            label="non-divergent",
        )
        if divergent.any():
            axis.scatter(
                disease[divergent],
                scale[divergent],
                s=18,
                alpha=0.8,
                color="firebrick",
                label="divergent",
            )
        axis.set_xlabel("CAPN3 effect (link scale)")
        axis.set_ylabel(name.replace("_", " "))
        axis.grid(alpha=0.2)
    axes.ravel()[0].legend(frameon=False)
    figure.suptitle(f"{row['metric']} — divergent-transition locations")
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    add_quality_banner(figure, row)
    return figure


def robust_histogram_edges(values: np.ndarray, bins: int = 35) -> np.ndarray:
    """Create robust shared histogram edges.

    Args:
        values (np.ndarray): Combined observed and predictive values.
        bins (int): Number of bins.

    Returns:
        np.ndarray: Monotonic bin edges.
    """

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    low, high = np.quantile(finite, [0.005, 0.995])
    if not high > low:
        low, high = float(np.min(finite)), float(np.max(finite) + 1.0)
    return np.linspace(low, high, bins + 1)


def plot_ppc_density(
    data: PreparedPatientMetricData,
    sample: Any,
    row: pd.Series,
) -> plt.Figure:
    """Plot pooled observed and predictive densities.

    Args:
        data (PreparedPatientMetricData): Prepared fitted data.
        sample (Any): Indexed posterior-predictive sample.
        row (pd.Series): Summary row.

    Returns:
        plt.Figure: Pooled PPC density figure.
    """

    del data
    predictive_flat = sample.predictive.reshape(-1)
    edges = robust_histogram_edges(
        np.concatenate([sample.observed, predictive_flat])
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(
        predictive_flat,
        bins=edges,
        density=True,
        alpha=0.45,
        color="steelblue",
        label="posterior predictive",
    )
    axis.hist(
        sample.observed,
        bins=edges,
        density=True,
        histtype="step",
        linewidth=2,
        color="firebrick",
        label="observed",
    )
    axis.set_xlabel(f"{row['metric']} ({METRIC_UNITS.get(str(row['metric']), 'value')})")
    axis.set_ylabel("Density")
    axis.set_title(f"{row['metric']} — pooled posterior predictive check")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    add_quality_banner(figure, row)
    return figure


def plot_ppc_by_condition(
    data: PreparedPatientMetricData,
    sample: Any,
    row: pd.Series,
) -> plt.Figure:
    """Plot observed and predictive densities by condition.

    Args:
        data (PreparedPatientMetricData): Prepared fitted data.
        sample (Any): Indexed posterior-predictive sample.
        row (pd.Series): Summary row.

    Returns:
        plt.Figure: Condition-stratified PPC.
    """

    diseases = data.disease_obs[sample.observation_indices]
    edges = robust_histogram_edges(
        np.concatenate([sample.observed, sample.predictive.reshape(-1)])
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True, sharey=True)
    for condition_index, (axis, label) in enumerate(
        zip(axes, CONDITION_LABELS, strict=True)
    ):
        mask = diseases == condition_index
        axis.hist(
            sample.predictive[:, mask].reshape(-1),
            bins=edges,
            density=True,
            alpha=0.45,
            color="steelblue",
            label="posterior predictive",
        )
        axis.hist(
            sample.observed[mask],
            bins=edges,
            density=True,
            histtype="step",
            linewidth=2,
            color="firebrick",
            label="observed",
        )
        axis.set_title(label)
        axis.set_xlabel(METRIC_UNITS.get(str(row["metric"]), "value"))
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)
    figure.suptitle(f"{row['metric']} — posterior predictive check by condition")
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    add_quality_banner(figure, row)
    return figure


def plot_ppc_quantiles(sample: Any, row: pd.Series) -> plt.Figure:
    """Compare observed and predictive distribution quantiles.

    Args:
        sample (Any): Indexed posterior-predictive sample.
        row (pd.Series): Summary row.

    Returns:
        plt.Figure: Quantile PPC.
    """

    probabilities = np.array([0.1, 0.5, 0.9])
    predictive_quantiles = np.quantile(sample.predictive, probabilities, axis=1)
    observed_quantiles = np.quantile(sample.observed, probabilities)
    medians = np.median(predictive_quantiles, axis=1)
    intervals = np.array(
        [az.hdi(values, hdi_prob=HDI_PROBABILITY) for values in predictive_quantiles]
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(
        probabilities,
        medians,
        yerr=np.vstack([medians - intervals[:, 0], intervals[:, 1] - medians]),
        fmt="o",
        capsize=4,
        color="steelblue",
        label="predictive median and 95% HDI",
    )
    axis.scatter(
        probabilities,
        observed_quantiles,
        marker="D",
        color="firebrick",
        label="observed",
        zorder=3,
    )
    axis.set_xticks(probabilities, ["10th", "median", "90th"])
    axis.set_ylabel(f"{row['metric']} ({METRIC_UNITS.get(str(row['metric']), 'value')})")
    axis.set_xlabel("Distribution quantile")
    axis.set_title(f"{row['metric']} — posterior predictive quantiles")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    add_quality_banner(figure, row)
    return figure


def plot_ppc_patient_means(
    data: PreparedPatientMetricData,
    sample: Any,
    row: pd.Series,
) -> plt.Figure:
    """Compare observed and predictive means for sampled patients.

    Args:
        data (PreparedPatientMetricData): Prepared fitted data.
        sample (Any): Indexed posterior-predictive sample.
        row (pd.Series): Summary row.

    Returns:
        plt.Figure: Per-patient PPC.
    """

    patient_indices = data.patient_idx_obs[sample.observation_indices]
    disease = data.disease_obs[sample.observation_indices]
    records: list[dict[str, Any]] = []
    for patient_index in np.unique(patient_indices):
        mask = patient_indices == patient_index
        if int(mask.sum()) < 2:
            continue
        predictive_mean = np.mean(sample.predictive[:, mask], axis=1)
        low, high = az.hdi(predictive_mean, hdi_prob=HDI_PROBABILITY)
        records.append(
            {
                "patient": data.patient_labels[int(patient_index)],
                "condition": CONDITION_LABELS[int(disease[mask][0])],
                "observed": float(np.mean(sample.observed[mask])),
                "predicted": float(np.median(predictive_mean)),
                "low": float(low),
                "high": float(high),
            }
        )
    frame = pd.DataFrame(records).sort_values(["condition", "patient"])
    positions = np.arange(len(frame))
    figure, axis = plt.subplots(figsize=(max(10, 0.45 * len(frame)), 5.5))
    axis.errorbar(
        positions,
        frame["predicted"],
        yerr=np.vstack(
            [frame["predicted"] - frame["low"], frame["high"] - frame["predicted"]]
        ),
        fmt="o",
        capsize=3,
        color="steelblue",
        label="predictive median and 95% HDI",
    )
    axis.scatter(
        positions,
        frame["observed"],
        marker="D",
        color="firebrick",
        label="observed",
        zorder=3,
    )
    axis.set_xticks(
        positions,
        [f"{patient}\n{condition}" for patient, condition in zip(
            frame["patient"], frame["condition"], strict=True
        )],
        rotation=60,
        ha="right",
    )
    axis.set_ylabel(f"Patient mean {row['metric']}")
    axis.set_title(f"{row['metric']} — sampled per-patient predictive check")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    add_quality_banner(figure, row)
    return figure


def effect_forest(
    row: pd.Series,
    specs: list[tuple[str, str]],
    title: str,
) -> plt.Figure:
    """Plot summary-CSV posterior contrasts as an HDI forest.

    Args:
        row (pd.Series): Primary summary row.
        specs (list[tuple[str, str]]): Column prefixes and display labels.
        title (str): Figure title.

    Returns:
        plt.Figure: Effect forest.
    """

    available = [
        (prefix, label)
        for prefix, label in specs
        if prefix in row.index and pd.notna(row[prefix])
    ]
    positions = np.arange(len(available))
    values = np.array([float(row[prefix]) for prefix, _ in available])
    lows = np.array([float(row[f"{prefix}_hdi_low"]) for prefix, _ in available])
    highs = np.array([float(row[f"{prefix}_hdi_high"]) for prefix, _ in available])
    pds = np.array([float(row[f"{prefix}_pd"]) for prefix, _ in available])
    figure, axis = plt.subplots(figsize=(10, max(4.5, 0.55 * len(available) + 2)))
    axis.errorbar(
        values,
        positions,
        xerr=np.vstack([values - lows, highs - values]),
        fmt="o",
        capsize=3,
        color="midnightblue",
    )
    axis.axvline(0.0, color="0.3", linestyle="--", linewidth=1)
    for x_value, y_value, pd_value in zip(values, positions, pds, strict=True):
        axis.annotate(
            f"pd={pd_value:.1f}%",
            (x_value, y_value),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_yticks(positions, [label for _, label in available])
    axis.invert_yaxis()
    axis.set_xlabel(
        "% change" if row.get("contrast_kind") == "percent_change" else "Difference"
    )
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    add_quality_banner(figure, row)
    return figure


def biology_effect_specs() -> list[tuple[str, str]]:
    """Return overall and site×compartment response-scale contrast specs.

    Args:
        None: Uses fixed patient-model contrast names.

    Returns:
        list[tuple[str, str]]: Summary prefixes and labels.
    """

    return [
        ("capn3_effect_response", "Overall CAPN3 − CTRL"),
        ("capn3_effect_deltoid_imf_response", "Deltoid | IMF"),
        ("capn3_effect_deltoid_ss_response", "Deltoid | SS"),
        ("capn3_effect_quadriceps_imf_response", "Quadriceps | IMF"),
        ("capn3_effect_quadriceps_ss_response", "Quadriceps | SS"),
    ]


def subtype_effect_specs() -> list[tuple[str, str]]:
    """Return CAPN3 subtype and pairwise response-scale contrast specs.

    Args:
        None: Uses configured patient genotype slugs.

    Returns:
        list[tuple[str, str]]: Summary prefixes and labels.
    """

    return [
        ("subtype_non_null_non_null_effect_response", "non-null/non-null vs CTRL"),
        ("subtype_null_non_null_effect_response", "null/non-null vs CTRL"),
        ("subtype_null_null_effect_response", "null/null vs CTRL"),
        (
            "subtype_non_null_non_null_vs_null_non_null_contrast_response",
            "non-null/non-null vs null/non-null",
        ),
        (
            "subtype_non_null_non_null_vs_null_null_contrast_response",
            "non-null/non-null vs null/null",
        ),
        (
            "subtype_null_non_null_vs_null_null_contrast_response",
            "null/non-null vs null/null",
        ),
    ]


def covariate_effect_specs() -> list[tuple[str, str]]:
    """Return link-scale covariate effect specs.

    Args:
        None: Uses fixed patient model coefficient names.

    Returns:
        list[tuple[str, str]]: Summary prefixes and labels.
    """

    return [
        ("site_effect_link", "Quadriceps vs Deltoid"),
        ("male_effect_link", "Male vs Female"),
        ("maturation_effect_link", "Maturation basis"),
        ("aging_effect_link", "Aging per 10 years after 50"),
        ("compartment_effect_link", "SS vs IMF"),
        ("disease_site_interaction_link", "CAPN3 × site"),
        ("disease_compartment_interaction_link", "CAPN3 × compartment"),
    ]


def scenario_forest(
    rows: pd.DataFrame,
    fit: PatientBayesFitConfig,
) -> plt.Figure:
    """Compare the global CAPN3 effect across all site scenarios.

    Args:
        rows (pd.DataFrame): Five scenario summary rows.
        fit (PatientBayesFitConfig): Fit metadata.

    Returns:
        plt.Figure: Site-sensitivity forest.
    """

    ordered = rows.copy()
    ordered["scenario_order"] = ordered["site_scenario"].map(
        {name: index for index, name in enumerate(SCENARIO_ORDER)}
    )
    ordered = ordered.sort_values("scenario_order")
    values = ordered["capn3_effect_response"].to_numpy(dtype=float)
    lows = ordered["capn3_effect_response_hdi_low"].to_numpy(dtype=float)
    highs = ordered["capn3_effect_response_hdi_high"].to_numpy(dtype=float)
    positions = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.errorbar(
        values,
        positions,
        xerr=np.vstack([values - lows, highs - values]),
        fmt="o",
        capsize=4,
        color="midnightblue",
    )
    axis.axvline(0.0, color="0.3", linestyle="--")
    labels = []
    for _, row in ordered.iterrows():
        assignment = str(row.get("site_assignments", "")).strip()
        labels.append(
            str(row["site_scenario"])
            + (f"\n{assignment}" if assignment and assignment != "nan" else "")
        )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel(
        "% change" if ordered.iloc[0]["contrast_kind"] == "percent_change"
        else "Difference"
    )
    axis.set_title(f"{fit.metric} — unknown-site assignment sensitivity")
    axis.grid(axis="x", alpha=0.2)
    max_divergences = int(ordered["divergences"].max())
    figure.text(
        0.5,
        0.01,
        f"All scenarios shown; maximum divergences={max_divergences}. "
        "Interpret sensitivity only after sampler remediation.",
        ha="center",
        color="darkred",
        fontsize=8,
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return figure


def primary_effect_overview(summary: pd.DataFrame) -> plt.Figure:
    """Plot all primary CAPN3 effects separated by contrast scale.

    Args:
        summary (pd.DataFrame): Primary rows for selected fits.

    Returns:
        plt.Figure: Multi-metric effect overview.
    """

    kinds = [
        kind
        for kind in ("percent_change", "difference")
        if kind in set(summary["contrast_kind"])
    ]
    figure, axes = plt.subplots(
        1,
        len(kinds),
        figsize=(8 * len(kinds), max(6, 0.45 * len(summary) + 2)),
        squeeze=False,
    )
    for axis, kind in zip(axes.ravel(), kinds, strict=True):
        rows = summary.loc[summary["contrast_kind"] == kind].sort_values(
            "capn3_effect_response"
        )
        positions = np.arange(len(rows))
        values = rows["capn3_effect_response"].to_numpy(dtype=float)
        lows = rows["capn3_effect_response_hdi_low"].to_numpy(dtype=float)
        highs = rows["capn3_effect_response_hdi_high"].to_numpy(dtype=float)
        axis.errorbar(
            values,
            positions,
            xerr=np.vstack([values - lows, highs - values]),
            fmt="o",
            capsize=3,
            color="midnightblue",
        )
        axis.axvline(0.0, color="0.3", linestyle="--")
        axis.set_yticks(positions, rows["metric"])
        axis.set_xlabel("% change" if kind == "percent_change" else "Difference")
        axis.set_title(kind.replace("_", " ").title())
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Primary adjusted CAPN3 effects across metrics")
    figure.text(
        0.5,
        0.005,
        "Every displayed fit has divergent transitions; estimates are provisional.",
        ha="center",
        color="darkred",
        fontsize=9,
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    return figure


def diagnostic_overview(summary: pd.DataFrame) -> plt.Figure:
    """Plot divergences, R-hat, and ESS across primary fits.

    Args:
        summary (pd.DataFrame): Primary rows.

    Returns:
        plt.Figure: Diagnostic overview.
    """

    rows = summary.sort_values("divergences")
    positions = np.arange(len(rows))
    figure, axes = plt.subplots(1, 3, figsize=(17, max(6, 0.42 * len(rows) + 2)))
    axes[0].barh(positions, rows["divergences"], color="firebrick", alpha=0.8)
    axes[0].set_xlabel("Divergent transitions")
    axes[1].scatter(rows["rhat_max"], positions, color="midnightblue")
    axes[1].axvline(1.01, color="firebrick", linestyle="--")
    axes[1].set_xlabel("Maximum R-hat")
    axes[2].scatter(rows["ess_bulk_min"], positions, color="midnightblue")
    axes[2].axvline(400, color="firebrick", linestyle="--")
    axes[2].set_xlabel("Minimum bulk ESS")
    for axis in axes:
        axis.set_yticks(positions, rows["metric"] if axis is axes[0] else [])
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Primary fit quality overview")
    figure.tight_layout()
    return figure


def observed_frame_for_fit(
    frame: pd.DataFrame,
    config: PatientBayesAnalysisConfig,
    fit: PatientBayesFitConfig,
    scenario: SiteScenario,
) -> pd.DataFrame:
    """Reconstruct the observed dataframe scope used by a fit.

    Args:
        frame (pd.DataFrame): Zoom-filtered measurements.
        config (PatientBayesAnalysisConfig): Analysis configuration.
        fit (PatientBayesFitConfig): Fit metadata.
        scenario (SiteScenario): Primary site scenario.

    Returns:
        pd.DataFrame: Filtered response rows for descriptive plotting.
    """

    result = apply_site_scenario(frame, config.model, scenario)
    if config.analysis_id == "instance" and fit.metric in INSTANCE_CLUSTERING_METRICS:
        region = result[config.model.image_region_column].astype(str).str.lower()
        result = result.loc[region == "center"].copy()
    result[fit.metric] = pd.to_numeric(result[fit.metric], errors="coerce")
    result = result.dropna(
        subset=[
            fit.metric,
            config.model.condition_column,
            config.model.patient_column,
            config.model.site_column,
            config.model.compartment_column,
        ]
    ).copy()
    if fit.likelihood in {"lognormal", "gamma"}:
        result = result.loc[result[fit.metric] > 0.0].copy()
    return result


def observed_annotation(row: pd.Series) -> str:
    """Format site×compartment adjusted effects for observed superplots.

    Args:
        row (pd.Series): Primary summary row.

    Returns:
        str: Multiline adjusted-effect annotation.
    """

    labels = (
        ("Deltoid | IMF", "capn3_effect_deltoid_imf_response"),
        ("Deltoid | SS", "capn3_effect_deltoid_ss_response"),
        ("Quadriceps | IMF", "capn3_effect_quadriceps_imf_response"),
        ("Quadriceps | SS", "capn3_effect_quadriceps_ss_response"),
    )
    lines = []
    for label, prefix in labels:
        lines.append(f"{label}: {row.get(f'{prefix}_summary', 'unavailable')}")
    return "\n".join(lines)


def add_manifest_record(
    records: list[dict[str, Any]],
    path: Path,
    row: pd.Series,
    category: str,
    written: bool,
) -> None:
    """Append one figure record to the visualization manifest.

    Args:
        records (list[dict[str, Any]]): Mutable manifest records.
        path (Path): Figure path.
        row (pd.Series): Summary row.
        category (str): Figure family.
        written (bool): Whether this run wrote the file.

    Returns:
        None: Appends one record.
    """

    records.append(
        {
            "analysis_id": row["analysis_id"],
            "fit_id": row["fit_id"],
            "metric": row["metric"],
            "site_scenario": row["site_scenario"],
            "category": category,
            "path": str(path),
            "written": written,
            "fit_status": row.get("fit_status"),
            "divergences": row.get("divergences"),
            "rhat_max": row.get("rhat_max"),
            "ess_bulk_min": row.get("ess_bulk_min"),
            "ppc_fit_status": row.get("ppc_fit_status"),
            "exploratory": row.get("exploratory"),
        }
    )


def save_named_figure(
    figure: plt.Figure,
    path: Path,
    overwrite: bool,
    row: pd.Series,
    category: str,
    records: list[dict[str, Any]],
) -> None:
    """Save a named figure and record its manifest entry.

    Args:
        figure (plt.Figure): Figure to save.
        path (Path): Output path.
        overwrite (bool): Whether to replace existing output.
        row (pd.Series): Summary metadata.
        category (str): Figure family.
        records (list[dict[str, Any]]): Mutable manifest.

    Returns:
        None: Saves and records the figure.
    """

    written = save_figure(figure, path, overwrite)
    add_manifest_record(records, path, row, category, written)


def generate_primary_fit_figures(
    config: PatientBayesAnalysisConfig,
    fit: PatientBayesFitConfig,
    row: pd.Series,
    data: PreparedPatientMetricData,
    idata: az.InferenceData,
    observed_frame: pd.DataFrame,
    overwrite: bool,
    ppc_seed: int,
    ppc_draws: int,
    ppc_observations: int,
    records: list[dict[str, Any]],
) -> None:
    """Generate the full primary figure suite for one fit.

    Args:
        config (PatientBayesAnalysisConfig): Analysis configuration.
        fit (PatientBayesFitConfig): Fit metadata.
        row (pd.Series): Primary summary row.
        data (PreparedPatientMetricData): Prepared fitted data.
        idata (az.InferenceData): Saved posterior.
        observed_frame (pd.DataFrame): Matched descriptive rows.
        overwrite (bool): Whether to replace figures.
        ppc_seed (int): Base PPC seed.
        ppc_draws (int): PPC posterior draw limit.
        ppc_observations (int): PPC observation limit.
        records (list[dict[str, Any]]): Mutable manifest.

    Returns:
        None: Writes all primary figures.
    """

    root = config.paths.figure_root
    names = diagnostic_variable_names(idata, config.analysis_id)
    diagnostic_plots: list[tuple[str, Callable[[], plt.Figure]]] = [
        ("trace", lambda: plot_trace(idata, names, row)),
        ("rank", lambda: plot_rank(idata, names, row)),
        ("forest_rhat_ess", lambda: plot_forest(idata, names, row)),
        ("energy", lambda: plot_energy(idata, row)),
        ("divergence_locations", lambda: plot_divergence_locations(idata, row)),
    ]
    for suffix, builder in diagnostic_plots:
        save_named_figure(
            builder(),
            figure_path(root, "diagnostics", fit.fit_id, suffix),
            overwrite,
            row,
            "diagnostics",
            records,
        )
    sample = simulate_predictive_subset_indexed(
        data=data,
        idata=idata,
        random_seed=stable_seed(ppc_seed, config.analysis_id, fit.fit_id),
        draw_limit=ppc_draws,
        observation_limit=ppc_observations,
    )
    ppc_plots: list[tuple[str, Callable[[], plt.Figure]]] = [
        ("density", lambda: plot_ppc_density(data, sample, row)),
        ("density_by_condition", lambda: plot_ppc_by_condition(data, sample, row)),
        ("quantiles", lambda: plot_ppc_quantiles(sample, row)),
        ("patient_means", lambda: plot_ppc_patient_means(data, sample, row)),
    ]
    for suffix, builder in ppc_plots:
        save_named_figure(
            builder(),
            figure_path(root, "ppc", fit.fit_id, suffix),
            overwrite,
            row,
            "ppc",
            records,
        )
    posterior_plots = [
        (
            "capn3_effects",
            effect_forest(
                row,
                biology_effect_specs(),
                f"{fit.metric} — adjusted CAPN3 effects",
            ),
        ),
        (
            "subtype_effects",
            effect_forest(
                row,
                subtype_effect_specs(),
                f"{fit.metric} — CAPN3 subtype contrasts",
            ),
        ),
        (
            "covariate_effects",
            effect_forest(
                row,
                covariate_effect_specs(),
                f"{fit.metric} — covariates and interactions (link scale)",
            ),
        ),
    ]
    for suffix, figure in posterior_plots:
        save_named_figure(
            figure,
            figure_path(root, "posteriors", fit.fit_id, suffix),
            overwrite,
            row,
            "posteriors",
            records,
        )
    generate_observed_figures(
        config=config,
        fit=fit,
        row=row,
        frame=observed_frame,
        overwrite=overwrite,
        records=records,
    )


def generate_observed_figures(
    config: PatientBayesAnalysisConfig,
    fit: PatientBayesFitConfig,
    row: pd.Series,
    frame: pd.DataFrame,
    overwrite: bool,
    records: list[dict[str, Any]],
) -> None:
    """Generate patient-aware observed superplots using local utilities.

    Args:
        config (PatientBayesAnalysisConfig): Analysis configuration.
        fit (PatientBayesFitConfig): Fit metadata.
        row (pd.Series): Primary summary row.
        frame (pd.DataFrame): Matched observed rows.
        overwrite (bool): Whether to replace figures.
        records (list[dict[str, Any]]): Mutable manifest.

    Returns:
        None: Writes observed superplots.
    """

    from stats_utils import (
        CONDITION_ORDER,
        SITE_COMPARTMENT_COLUMN,
        add_site_compartment_column,
        bayesian_superplot_annotations,
        build_output_path,
        metric_unit_mapping,
        plot_super_beeswarm,
        plot_super_violin,
        site_compartment_labels,
    )

    model = config.model
    plot_data = add_site_compartment_column(
        frame,
        site_column=model.site_column,
        compartment_column=model.compartment_column,
    )
    annotations = bayesian_superplot_annotations(row)
    unit_dict = metric_unit_mapping([fit.metric])
    save_dir = config.paths.figure_root / "observed"
    filename_prefix = f"{fit.fit_id}__"
    common = {
        "data": plot_data,
        "x": SITE_COMPARTMENT_COLUMN,
        "y": fit.metric,
        "hue": model.condition_column,
        "block": model.patient_column,
        "unit_dict": unit_dict,
        "save_dir": save_dir,
        "title_override": f"{fit.metric} — observed patient distributions",
        "filename_prefix": filename_prefix,
        "superplot_annotations": annotations,
        "x_order_override": site_compartment_labels(),
        "hue_order_override": list(CONDITION_ORDER),
    }
    for plot_function, suffix in (
        (plot_super_violin, "superviolin"),
        (plot_super_beeswarm, "superbeeswarm"),
    ):
        output_path = build_output_path(
            y=fit.metric,
            x=SITE_COMPARTMENT_COLUMN,
            hue=model.condition_column,
            save_dir=save_dir,
            suffix=suffix,
            filename_prefix=filename_prefix,
        )
        if output_path is not None and output_path.exists() and not overwrite:
            add_manifest_record(records, output_path, row, "observed", False)
            continue
        written_path = plot_function(**common)
        if written_path is not None:
            add_manifest_record(records, written_path, row, "observed", True)


def run_visualization(
    config: PatientBayesAnalysisConfig,
    requested_fit_ids: list[str] | None,
    overwrite: bool,
    ppc_seed: int,
    ppc_draws: int,
    ppc_observations: int,
) -> None:
    """Generate all selected patient Bayesian visualizations.

    Args:
        config (PatientBayesAnalysisConfig): Analysis configuration.
        requested_fit_ids (list[str] | None): Optional targeted fits.
        overwrite (bool): Whether to replace existing figures.
        ppc_seed (int): Deterministic PPC seed.
        ppc_draws (int): PPC posterior draw limit.
        ppc_observations (int): PPC observation limit.

    Returns:
        None: Writes figures and a manifest.
    """

    summary = load_summary(config)
    fits = selected_fits(config, requested_fit_ids)
    validate_fit_inventory(summary, fits)
    frame = load_measurements(config)
    scenarios = build_site_scenarios(frame, config.model)
    scenarios_by_name = {scenario.name: scenario for scenario in scenarios}
    if PRIMARY_SCENARIO not in scenarios_by_name:
        raise ValueError("Primary exclusion scenario is unavailable.")
    records: list[dict[str, Any]] = []
    primary_rows: list[pd.Series] = []
    for fit in fits:
        primary_row = summary_row(summary, fit.fit_id, PRIMARY_SCENARIO)
        primary_rows.append(primary_row)
        idata = load_trace(primary_row, config)
        data = prepare_metric_data(
            frame,
            config,
            fit,
            scenarios_by_name[PRIMARY_SCENARIO],
        )
        observed_frame = observed_frame_for_fit(
            frame,
            config,
            fit,
            scenarios_by_name[PRIMARY_SCENARIO],
        )
        generate_primary_fit_figures(
            config=config,
            fit=fit,
            row=primary_row,
            data=data,
            idata=idata,
            observed_frame=observed_frame,
            overwrite=overwrite,
            ppc_seed=ppc_seed,
            ppc_draws=ppc_draws,
            ppc_observations=ppc_observations,
            records=records,
        )
        sensitivity_rows = summary.loc[
            (summary["fit_id"] == fit.fit_id)
            & (summary["site_scenario"].isin(SCENARIO_ORDER))
        ].copy()
        if set(sensitivity_rows["site_scenario"]) != set(SCENARIO_ORDER):
            raise ValueError(f"Incomplete site sensitivity rows for {fit.fit_id}.")
        path = figure_path(
            config.paths.figure_root,
            "sensitivity",
            fit.fit_id,
            "site_scenarios",
        )
        written = save_figure(
            scenario_forest(sensitivity_rows, fit),
            path,
            overwrite,
        )
        add_manifest_record(
            records,
            path,
            primary_row,
            "sensitivity",
            written,
        )
        print(f"Visualized {config.analysis_id} fit: {fit.fit_id}")
    primary_frame = pd.DataFrame(primary_rows)
    overview_row = primary_rows[0]
    for suffix, builder in (
        ("primary_effects", lambda: primary_effect_overview(primary_frame)),
        ("fit_quality", lambda: diagnostic_overview(primary_frame)),
    ):
        path = config.paths.figure_root / "overview" / f"{suffix}.png"
        written = save_figure(builder(), path, overwrite)
        add_manifest_record(
            records,
            path,
            overview_row,
            "overview",
            written,
        )
    manifest_path = config.paths.figure_root / "visualization_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    print(f"Wrote visualization manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    """Parse visualization CLI arguments.

    Args:
        None: Reads command-line arguments.

    Returns:
        argparse.Namespace: Parsed options.
    """

    parser = argparse.ArgumentParser(
        description="Visualize CAPN3 patient Bayesian fits without refitting."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH,
    )
    parser.add_argument("--fit-id", action="append", dest="fit_ids")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ppc-seed", type=int, default=20260812)
    parser.add_argument("--ppc-draws", type=int, default=200)
    parser.add_argument("--ppc-observations", type=int, default=500)
    args = parser.parse_args()
    if args.ppc_draws < 1 or args.ppc_observations < 1:
        parser.error("PPC draw and observation limits must be positive.")
    return args


def main() -> None:
    """Run patient Bayesian visualization generation.

    Args:
        None: Reads command-line arguments.

    Returns:
        None: Writes figures and manifests.
    """

    args = parse_args()
    config = load_hierarchical_bayes_config(args.config)
    run_visualization(
        config=config,
        requested_fit_ids=args.fit_ids,
        overwrite=args.overwrite,
        ppc_seed=args.ppc_seed,
        ppc_draws=args.ppc_draws,
        ppc_observations=args.ppc_observations,
    )


if __name__ == "__main__":
    main()
