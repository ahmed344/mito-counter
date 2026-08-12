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
        figsize=(14, max(8, 2.6 * len(names))),
    )
    figure = np.asarray(axes).ravel()[0].figure
    figure.suptitle(f"{row['metric']} — primary trace diagnostics", y=1.002)
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
        figsize=(13, max(7, 2.2 * math.ceil(len(names) / 2))),
    )
    figure = np.asarray(axes).ravel()[0].figure
    figure.suptitle(f"{row['metric']} — chain rank diagnostics", y=1.002)
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

    axes = az.plot_forest(
        idata,
        var_names=names,
        combined=True,
        hdi_prob=HDI_PROBABILITY,
        r_hat=True,
        ess=True,
        figsize=(13, max(7, 0.7 * len(names) + 4)),
    )
    figure = np.asarray(axes).ravel()[0].figure
    figure.suptitle(f"{row['metric']} — posterior and convergence", y=1.002)
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

    from stats_utils import patient_superbeeswarm, patient_superviolin

    plot_summary = {
        "capn3_effect_response_summary": observed_annotation(row),
        "fit_status": "ok",
    }
    common = {
        "data": frame,
        "metric": fit.metric,
        "condition_column": config.model.condition_column,
        "patient_column": config.model.patient_column,
        "site_column": config.model.site_column,
        "compartment_column": config.model.compartment_column,
        "aggregate": "median" if config.analysis_id == "instance" else "mean",
        "summary_row": plot_summary,
        "quality_warning": quality_text(row),
        "title": f"{fit.metric} — observed patient distributions",
    }
    for suffix, plotter in (
        ("superviolin", patient_superviolin),
        ("superbeeswarm", patient_superbeeswarm),
    ):
        figure, _ = plotter(**common)
        save_named_figure(
            figure,
            figure_path(config.paths.figure_root, "observed", fit.fit_id, suffix),
            overwrite,
            row,
            "observed",
            records,
        )


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
