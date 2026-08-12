"""Patient-specific plotting utilities for CTRL and CAPN3 measurements."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


CONDITION_ORDER = ("CTRL", "CAPN3")
SITE_ORDER = ("Deltoid", "Quadriceps")
COMPARTMENT_ORDER = ("Intermyofibrillar (IMF)", "Sub-sarcolemmal (SS)")
CAPN3_SUBTYPE_ORDER = ("non_null/non_null", "null/non_null", "null/null")

CONDITION_PALETTE = {"CTRL": "#4C78A8", "CAPN3": "#E45756"}
CAPN3_SUBTYPE_PALETTE = {
    "non_null/non_null": "#59A14F",
    "null/non_null": "#F28E2B",
    "null/null": "#B279A2",
}
_PATIENT_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "*")
_COMPARTMENT_SHORT = {
    "Intermyofibrillar (IMF)": "IMF",
    "Sub-sarcolemmal (SS)": "SS",
}
_METRIC_LABELS = {
    "Area": "Mitochondrial area",
    "Corrected_area": "Corrected mitochondrial area",
    "Major_axis_length": "Major-axis length",
    "Minor_axis_length": "Minor-axis length",
    "Minimum_Feret_Diameter": "Minimum Feret diameter",
    "Eccentricity": "Eccentricity",
    "Circularity": "Circularity",
    "Solidity": "Solidity",
    "NND": "Nearest-neighbor distance",
    "3NND": "Third-nearest-neighbor distance",
    "5NND": "Fifth-nearest-neighbor distance",
    "Voronoi_Cell_Area": "Voronoi cell area",
    "Instance_count": "Mitochondria count",
    "Ripley_L_integral": "Ripley's L integral",
    "Pair_Correlation_integral": "Pair-correlation integral",
}


def save_figure(
    figure: Figure,
    output_path: str | Path,
    dpi: int = 300,
    close: bool = True,
) -> Path:
    """Save a figure atomically after creating its parent directory.

    Args:
        figure (Figure): Matplotlib figure to save.
        output_path (str | Path): Destination filename.
        dpi (int): Raster resolution.
        close (bool): Whether to close the figure after saving.

    Returns:
        Path: Resolved output path.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        figure.savefig(
            temporary_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            metadata={"Creator": "c3_patients.stats_utils"},
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
        if close:
            plt.close(figure)
    return path.resolve()


def metric_label(metric: str) -> str:
    """Return a readable patient-analysis metric label.

    Args:
        metric (str): Measurement column name.

    Returns:
        str: Human-readable label.
    """

    if metric in _METRIC_LABELS:
        return _METRIC_LABELS[metric]
    suffixes = ("_center_mean", "_center_cv", "_mean", "_sum")
    base = metric
    suffix = ""
    for candidate in suffixes:
        if metric.endswith(candidate):
            base = metric[: -len(candidate)]
            suffix = candidate
            break
    base_label = _METRIC_LABELS.get(base, base.replace("_", " ").strip().capitalize())
    suffix_label = {
        "_center_mean": " (center mean)",
        "_center_cv": " (center CV)",
        "_mean": " (image mean)",
        "_sum": " (image sum)",
    }.get(suffix, "")
    return f"{base_label}{suffix_label}"


def metric_unit(metric: str) -> str:
    """Infer the display unit for a configured patient metric.

    Args:
        metric (str): Measurement column name.

    Returns:
        str: Unit text, or an empty string for unitless metrics.
    """

    normalized = metric.lower()
    if normalized.startswith(("eccentricity", "circularity", "solidity")):
        return ""
    if normalized.endswith("_cv"):
        return ""
    if normalized == "instance_count":
        return "count"
    if normalized == "ripley_l_integral":
        return "integrated nm²"
    if normalized == "pair_correlation_integral":
        return "integrated unitless"
    if "area" in normalized:
        return "nm²"
    if normalized in {"nnd", "3nnd", "5nnd"} or "diameter" in normalized:
        return "nm"
    if "axis_length" in normalized or "_nnd" in normalized:
        return "nm"
    return ""


def metric_axis_label(metric: str) -> str:
    """Build a y-axis label with a unit when one is meaningful.

    Args:
        metric (str): Measurement column name.

    Returns:
        str: Complete y-axis label.
    """

    label = metric_label(metric)
    unit = metric_unit(metric)
    return f"{label} ({unit})" if unit else label


def condition_sort_key(condition: str) -> tuple[int, str]:
    """Return a stable sort key with CTRL before CAPN3.

    Args:
        condition (str): Condition label.

    Returns:
        tuple[int, str]: Preferred-order index and normalized label.
    """

    text = str(condition)
    try:
        index = CONDITION_ORDER.index(text)
    except ValueError:
        index = len(CONDITION_ORDER)
    return index, text.casefold()


def sort_conditions(conditions: Sequence[str]) -> list[str]:
    """Sort unique condition labels in patient-study order.

    Args:
        conditions (Sequence[str]): Condition labels.

    Returns:
        list[str]: Unique sorted labels.
    """

    return sorted({str(value) for value in conditions}, key=condition_sort_key)


def site_compartment_label(site: str, compartment: str) -> str:
    """Combine biopsy site and compartment into a compact axis label.

    Args:
        site (str): Biopsy site.
        compartment (str): Mitochondrial compartment.

    Returns:
        str: Two-line site and compartment label.
    """

    compartment_text = _COMPARTMENT_SHORT.get(str(compartment), str(compartment))
    return f"{site}\n{compartment_text}"


def site_compartment_labels() -> list[str]:
    """Return all site-by-compartment labels in canonical order.

    Args:
        None: This function takes no arguments.

    Returns:
        list[str]: Ordered site-by-compartment labels.
    """

    return [
        site_compartment_label(site, compartment)
        for site in SITE_ORDER
        for compartment in COMPARTMENT_ORDER
    ]


def bayesian_adjusted_effect_text(
    summary_row: Mapping[str, Any] | pd.Series | None,
) -> str:
    """Format the primary Bayesian adjusted CTRL-to-CAPN3 effect.

    Args:
        summary_row (Mapping[str, Any] | pd.Series | None): Primary fit summary.

    Returns:
        str: Concise adjusted-effect annotation, or an unavailable message.
    """

    if summary_row is None:
        return "Bayesian adjusted effect: summary unavailable"
    row = dict(summary_row)
    cell_effects = (
        ("Deltoid | IMF", "capn3_effect_deltoid_imf_response_summary"),
        ("Deltoid | SS", "capn3_effect_deltoid_ss_response_summary"),
        ("Quadriceps | IMF", "capn3_effect_quadriceps_imf_response_summary"),
        ("Quadriceps | SS", "capn3_effect_quadriceps_ss_response_summary"),
    )
    if all(pd.notna(row.get(column)) for _, column in cell_effects):
        lines = ["Bayesian adjusted CAPN3 effects:"]
        lines.extend(f"{label}: {row[column]}" for label, column in cell_effects)
        return "\n".join(lines)
    ready_summary = row.get("capn3_effect_response_summary")
    if pd.notna(ready_summary) and str(ready_summary).strip():
        return f"Bayesian adjusted CAPN3 effect: {ready_summary}"
    value = pd.to_numeric(
        pd.Series([row.get("capn3_effect_response")]), errors="coerce"
    ).iloc[0]
    low = pd.to_numeric(
        pd.Series([row.get("capn3_effect_response_hdi_low")]), errors="coerce"
    ).iloc[0]
    high = pd.to_numeric(
        pd.Series([row.get("capn3_effect_response_hdi_high")]), errors="coerce"
    ).iloc[0]
    if all(np.isfinite(number) for number in (value, low, high)):
        return (
            "Bayesian adjusted CAPN3 effect: "
            f"{value:.3g} (95% HDI {low:.3g} to {high:.3g})"
        )
    return "Bayesian adjusted effect: not available"


def quality_warning_text(
    summary_row: Mapping[str, Any] | pd.Series | None,
) -> str:
    """Format fit-quality warnings from a primary Bayesian summary row.

    Args:
        summary_row (Mapping[str, Any] | pd.Series | None): Primary fit summary.

    Returns:
        str: Warning annotation, or an empty string for a clean fit.
    """

    if summary_row is None:
        return "Quality note: Bayesian summary not found"
    row = dict(summary_row)
    status = str(row.get("fit_status", "")).strip().lower()
    warning = str(row.get("warning_message", "")).strip()
    error = str(row.get("error_message", "")).strip()
    if status in {"ok", "success"} and not warning:
        return ""
    detail = error if error and error.lower() != "nan" else warning
    if detail and detail.lower() != "nan":
        return f"Quality warning ({status or 'unknown'}): {detail}"
    if status and status not in {"ok", "success"}:
        return f"Quality warning: fit status is {status}"
    return ""


def _stable_patient_marker(patient: str) -> str:
    """Choose a deterministic marker without exposing a patient legend.

    Args:
        patient (str): Patient identifier.

    Returns:
        str: Matplotlib marker symbol.
    """

    digest = hashlib.sha256(str(patient).encode("utf-8")).digest()
    return _PATIENT_MARKERS[int.from_bytes(digest[:2], "big") % len(_PATIENT_MARKERS)]


def _prepare_plot_data(
    data: pd.DataFrame,
    metric: str,
    patient_column: str,
    condition_column: str,
    site_column: str,
    compartment_column: str,
    aggregate: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate plotting columns and calculate equal-weight patient values.

    Args:
        data (pd.DataFrame): Raw observation or image-level measurements.
        metric (str): Numeric metric column.
        patient_column (str): Patient identifier column.
        condition_column (str): Plot hue column.
        site_column (str): Biopsy-site column.
        compartment_column (str): Compartment column.
        aggregate (str): ``median`` or ``mean`` patient aggregation.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Clean raw rows and patient aggregates.
    """

    required = {
        metric,
        patient_column,
        condition_column,
        site_column,
        compartment_column,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise KeyError(f"Missing plot columns: {missing}")
    if aggregate not in {"median", "mean"}:
        raise ValueError("aggregate must be 'median' or 'mean'.")
    frame = data[list(required)].copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.dropna(subset=list(required))
    frame = frame.loc[np.isfinite(frame[metric])].copy()
    frame["site_compartment"] = [
        site_compartment_label(site, compartment)
        for site, compartment in zip(
            frame[site_column], frame[compartment_column], strict=True
        )
    ]
    group_columns = [
        patient_column,
        condition_column,
        site_column,
        compartment_column,
        "site_compartment",
    ]
    grouped = frame.groupby(group_columns, observed=True, as_index=False)[metric]
    patient_values = grouped.median() if aggregate == "median" else grouped.mean()
    return frame, patient_values


def _annotation_text(
    summary_row: Mapping[str, Any] | pd.Series | None,
    extra_warning: str | None,
) -> str:
    """Combine adjusted-effect and quality annotations.

    Args:
        summary_row (Mapping[str, Any] | pd.Series | None): Primary fit summary.
        extra_warning (str | None): Additional descriptive-data warning.

    Returns:
        str: Multi-line plot annotation.
    """

    parts = [bayesian_adjusted_effect_text(summary_row)]
    warning = quality_warning_text(summary_row)
    if warning:
        parts.append(warning)
    if extra_warning:
        parts.append(str(extra_warning))
    return "\n".join(parts)


def _finish_patient_plot(
    ax: Axes,
    metric: str,
    title: str | None,
    hue_order: Sequence[str],
    palette: Mapping[str, str],
    annotation: str,
) -> None:
    """Apply shared labels, legend, and annotation to a patient plot.

    Args:
        ax (Axes): Plot axes.
        metric (str): Numeric metric column.
        title (str | None): Optional title.
        hue_order (Sequence[str]): Ordered hue labels.
        palette (Mapping[str, str]): Hue colors.
        annotation (str): Adjusted-effect and warning text.

    Returns:
        None: The axes are modified in place.
    """

    ax.set_xlabel("Biopsy site × compartment")
    ax.set_ylabel(metric_axis_label(metric))
    ax.set_title(title or metric_label(metric), loc="left", weight="bold")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=palette[label],
            markeredgecolor="black",
            markersize=7,
            label=label,
        )
        for label in hue_order
        if label in palette
    ]
    if handles:
        ax.legend(handles=handles, title=None, frameon=False, loc="best")
    ax.text(
        0.01,
        0.99,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
    )
    sns.despine(ax=ax)


def _draw_patient_points(
    ax: Axes,
    patient_values: pd.DataFrame,
    metric: str,
    patient_column: str,
    hue_column: str,
    x_order: Sequence[str],
    hue_order: Sequence[str],
    palette: Mapping[str, str],
) -> None:
    """Draw equal-weight patient aggregates with deterministic identities.

    Args:
        ax (Axes): Plot axes.
        patient_values (pd.DataFrame): One value per patient and x/hue group.
        metric (str): Numeric metric column.
        patient_column (str): Patient identifier column.
        hue_column (str): Condition or subtype hue column.
        x_order (Sequence[str]): Ordered site-compartment labels.
        hue_order (Sequence[str]): Ordered hue labels.
        palette (Mapping[str, str]): Hue colors.

    Returns:
        None: Patient points are added in place.
    """

    hue_count = max(len(hue_order), 1)
    offsets = np.linspace(-0.20, 0.20, hue_count) if hue_count > 1 else np.array([0.0])
    x_lookup = {label: index for index, label in enumerate(x_order)}
    hue_lookup = {label: index for index, label in enumerate(hue_order)}
    for _, values in patient_values.iterrows():
        x_label = str(values["site_compartment"])
        hue = str(values[hue_column])
        if x_label not in x_lookup or hue not in hue_lookup:
            continue
        patient = str(values[patient_column])
        ax.scatter(
            x_lookup[x_label] + offsets[hue_lookup[hue]],
            float(values[metric]),
            s=48,
            marker=_stable_patient_marker(patient),
            facecolor=palette.get(hue, "#777777"),
            edgecolor="black",
            linewidth=0.65,
            alpha=0.98,
            zorder=5,
        )


def patient_superviolin(
    data: pd.DataFrame,
    metric: str,
    patient_column: str = "IDENTIFIER",
    condition_column: str = "Condition",
    site_column: str = "SITE OF BIOPSY",
    compartment_column: str = "Compartment",
    aggregate: str = "median",
    summary_row: Mapping[str, Any] | pd.Series | None = None,
    output_path: str | Path | None = None,
    title: str | None = None,
    hue_order: Sequence[str] = CONDITION_ORDER,
    palette: Mapping[str, str] = CONDITION_PALETTE,
    quality_warning: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot distributions plus prominent equal-weight patient aggregates.

    Args:
        data (pd.DataFrame): Raw observation or image-level rows.
        metric (str): Numeric metric column.
        patient_column (str): Patient identifier column.
        condition_column (str): Condition or subtype hue column.
        site_column (str): Biopsy-site column.
        compartment_column (str): Compartment column.
        aggregate (str): ``median`` or ``mean`` patient aggregation.
        summary_row (Mapping[str, Any] | pd.Series | None): Bayesian summary row.
        output_path (str | Path | None): Optional figure destination.
        title (str | None): Optional plot title.
        hue_order (Sequence[str]): Ordered hue labels.
        palette (Mapping[str, str]): Hue colors.
        quality_warning (str | None): Additional warning annotation.

    Returns:
        tuple[Figure, Axes]: Created Matplotlib figure and axes.
    """

    raw, patients = _prepare_plot_data(
        data,
        metric,
        patient_column,
        condition_column,
        site_column,
        compartment_column,
        aggregate,
    )
    x_order = site_compartment_labels()
    figure, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    sns.violinplot(
        data=raw,
        x="site_compartment",
        y=metric,
        hue=condition_column,
        order=x_order,
        hue_order=list(hue_order),
        palette=dict(palette),
        cut=0,
        inner=None,
        linewidth=0.7,
        density_norm="width",
        saturation=0.75,
        ax=ax,
    )
    sns.stripplot(
        data=raw,
        x="site_compartment",
        y=metric,
        hue=condition_column,
        order=x_order,
        hue_order=list(hue_order),
        dodge=True,
        palette=dict(palette),
        size=1.8,
        alpha=0.15,
        jitter=0.18,
        linewidth=0,
        legend=False,
        ax=ax,
    )
    _draw_patient_points(
        ax,
        patients,
        metric,
        patient_column,
        condition_column,
        x_order,
        hue_order,
        palette,
    )
    _finish_patient_plot(
        ax,
        metric,
        title,
        hue_order,
        palette,
        _annotation_text(summary_row, quality_warning),
    )
    if output_path is not None:
        save_figure(figure, output_path, close=False)
    return figure, ax


def patient_superbeeswarm(
    data: pd.DataFrame,
    metric: str,
    patient_column: str = "IDENTIFIER",
    condition_column: str = "Condition",
    site_column: str = "SITE OF BIOPSY",
    compartment_column: str = "Compartment",
    aggregate: str = "median",
    summary_row: Mapping[str, Any] | pd.Series | None = None,
    output_path: str | Path | None = None,
    title: str | None = None,
    hue_order: Sequence[str] = CONDITION_ORDER,
    palette: Mapping[str, str] = CONDITION_PALETTE,
    quality_warning: str | None = None,
    raw_point_limit: int = 4000,
) -> tuple[Figure, Axes]:
    """Plot a light raw beeswarm plus equal-weight patient aggregates.

    Args:
        data (pd.DataFrame): Raw observation or image-level rows.
        metric (str): Numeric metric column.
        patient_column (str): Patient identifier column.
        condition_column (str): Condition or subtype hue column.
        site_column (str): Biopsy-site column.
        compartment_column (str): Compartment column.
        aggregate (str): ``median`` or ``mean`` patient aggregation.
        summary_row (Mapping[str, Any] | pd.Series | None): Bayesian summary row.
        output_path (str | Path | None): Optional figure destination.
        title (str | None): Optional plot title.
        hue_order (Sequence[str]): Ordered hue labels.
        palette (Mapping[str, str]): Hue colors.
        quality_warning (str | None): Additional warning annotation.
        raw_point_limit (int): Maximum raw rows drawn for responsiveness.

    Returns:
        tuple[Figure, Axes]: Created Matplotlib figure and axes.
    """

    raw, patients = _prepare_plot_data(
        data,
        metric,
        patient_column,
        condition_column,
        site_column,
        compartment_column,
        aggregate,
    )
    if raw_point_limit < 1:
        raise ValueError("raw_point_limit must be positive.")
    if len(raw) > raw_point_limit:
        raw = raw.sample(raw_point_limit, random_state=20260812)
        sample_warning = (
            f"Display note: raw points sampled ({raw_point_limit:,} of {len(data):,})"
        )
        quality_warning = (
            f"{quality_warning}; {sample_warning}" if quality_warning else sample_warning
        )
    x_order = site_compartment_labels()
    figure, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    point_size = max(1.2, min(2.5, 20.0 / math.sqrt(max(len(raw), 1))))
    sns.stripplot(
        data=raw,
        x="site_compartment",
        y=metric,
        hue=condition_column,
        order=x_order,
        hue_order=list(hue_order),
        dodge=True,
        jitter=0.20,
        palette=dict(palette),
        size=point_size,
        alpha=0.20,
        linewidth=0,
        legend=False,
        ax=ax,
    )
    _draw_patient_points(
        ax,
        patients,
        metric,
        patient_column,
        condition_column,
        x_order,
        hue_order,
        palette,
    )
    _finish_patient_plot(
        ax,
        metric,
        title,
        hue_order,
        palette,
        _annotation_text(summary_row, quality_warning),
    )
    if output_path is not None:
        save_figure(figure, output_path, close=False)
    return figure, ax
