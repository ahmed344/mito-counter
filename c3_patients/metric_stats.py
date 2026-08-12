# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
"""Descriptive and exploratory patient-level statistics for CAPN3 measurements."""

# %% [markdown]
# ## Setup and helpers
#
# This notebook uses only enabled Bayesian fit definitions. Its tests are
# exploratory and use one equal-weight value per patient, site, and compartment.

# %%
from __future__ import annotations

import argparse
import itertools
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
if not sys.path or sys.path[0] != str(SCRIPT_DIR):
    sys.path.insert(0, str(SCRIPT_DIR))

from hierarchical_bayes_config import (
    DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH,
    PatientBayesAnalysisConfig,
    PatientBayesFitConfig,
    enabled_fit_configs,
    load_hierarchical_bayes_config,
)
from stats_utils import (
    CAPN3_SUBTYPE_ORDER,
    CAPN3_SUBTYPE_PALETTE,
    COMPARTMENT_ORDER,
    CONDITION_ORDER,
    SITE_ORDER,
    patient_superbeeswarm,
    patient_superviolin,
)


IMAGE_SUMMARY_CONFIG_PATH = (
    SCRIPT_DIR / "hierarchical_bayes_image_summary_config.yaml"
)
CENTER_IMAGE_REGION = "center"
INSTANCE_CLUSTERING_METRICS = frozenset(
    {"NND", "3NND", "5NND", "Voronoi_Cell_Area"}
)
PRIMARY_SITE_SCENARIO = "primary_exclude_unknown"
MINIMUM_GROUP_PATIENTS = 2


def slugify(value: str) -> str:
    """Convert text to a stable lowercase filename component.

    Args:
        value (str): Raw text.

    Returns:
        str: Filesystem-safe slug.
    """

    slug = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return slug or "metric"


def write_csv_atomic(data: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a CSV atomically after creating its parent directory.

    Args:
        data (pd.DataFrame): Table to persist.
        output_path (str | Path): Destination CSV.

    Returns:
        Path: Resolved output path.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        data.to_csv(temporary_path, index=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path.resolve()


def load_measurements(config: PatientBayesAnalysisConfig) -> pd.DataFrame:
    """Load configured measurement files and apply the configured zoom filter.

    Args:
        config (PatientBayesAnalysisConfig): Analysis configuration.

    Returns:
        pd.DataFrame: Concatenated and zoom-filtered measurements.
    """

    paths: list[Path] = []
    if config.paths.input_csv is not None:
        paths.append(config.paths.input_csv)
    paths.extend(config.paths.input_csvs)
    if not paths:
        raise ValueError(f"No input CSV is configured for {config.analysis_id}.")
    missing_paths = [str(path) for path in paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Configured input CSVs do not exist: {missing_paths}")
    frames = [pd.read_csv(path, low_memory=False) for path in paths]
    measurements = pd.concat(frames, ignore_index=True, sort=False)
    if config.filters is not None:
        if "Zoom" not in measurements.columns:
            raise KeyError("Configured zoom filtering requires the 'Zoom' column.")
        zoom = pd.to_numeric(measurements["Zoom"], errors="coerce")
        measurements = measurements.loc[
            zoom.between(
                config.filters.minimum_zoom,
                config.filters.maximum_zoom,
                inclusive="both",
            )
        ].copy()
        if measurements.empty:
            raise ValueError(
                f"Zoom filtering removed all {config.analysis_id} measurements."
            )
    return measurements


def unknown_site_mask(
    data: pd.DataFrame,
    config: PatientBayesAnalysisConfig,
) -> pd.Series:
    """Identify rows with missing or configured unknown biopsy sites.

    Args:
        data (pd.DataFrame): Measurement rows.
        config (PatientBayesAnalysisConfig): Analysis configuration.

    Returns:
        pd.Series: Boolean mask aligned to ``data``.
    """

    site_column = config.model.site_column
    if site_column not in data.columns:
        raise KeyError(f"Missing biopsy-site column: {site_column}")
    site = data[site_column]
    normalized = site.astype("string").str.strip().str.upper()
    unknown_values = {
        value.strip().upper()
        for value in config.model.site_sensitivity.unknown_values
    }
    return site.isna() | normalized.isin(unknown_values)


def apply_primary_filters(
    data: pd.DataFrame,
    config: PatientBayesAnalysisConfig,
    fit: PatientBayesFitConfig,
) -> pd.DataFrame:
    """Apply primary site exclusion and metric-specific center filtering.

    Args:
        data (pd.DataFrame): Zoom-filtered measurements.
        config (PatientBayesAnalysisConfig): Analysis configuration.
        fit (PatientBayesFitConfig): Enabled metric fit.

    Returns:
        pd.DataFrame: Rows matching the primary Bayesian analysis.
    """

    model = config.model
    required = {
        fit.metric,
        model.condition_column,
        model.patient_column,
        model.genotype_column,
        model.site_column,
        model.compartment_column,
    }
    if config.analysis_id == "instance" and fit.metric in INSTANCE_CLUSTERING_METRICS:
        required.add(model.image_region_column)
    missing = sorted(required - set(data.columns))
    if missing:
        raise KeyError(f"Missing columns for fit '{fit.fit_id}': {missing}")
    filtered = data.copy()
    if model.site_sensitivity.primary_strategy == "exclude":
        filtered = filtered.loc[~unknown_site_mask(filtered, config)].copy()
    if config.analysis_id == "instance" and fit.metric in INSTANCE_CLUSTERING_METRICS:
        region = (
            filtered[model.image_region_column]
            .astype("string")
            .str.strip()
            .str.casefold()
        )
        filtered = filtered.loc[region == CENTER_IMAGE_REGION].copy()
    filtered[fit.metric] = pd.to_numeric(filtered[fit.metric], errors="coerce")
    filtered = filtered.dropna(subset=list(required - {model.image_region_column}))
    filtered = filtered.loc[np.isfinite(filtered[fit.metric])].copy()
    if fit.likelihood in {"lognormal", "gamma"}:
        filtered = filtered.loc[filtered[fit.metric] > 0.0].copy()
    elif fit.likelihood == "beta":
        filtered = filtered.loc[
            filtered[fit.metric].between(0.0, 1.0, inclusive="neither")
        ].copy()
    elif fit.likelihood == "negative_binomial":
        integer_like = np.isclose(filtered[fit.metric], np.round(filtered[fit.metric]))
        filtered = filtered.loc[
            (filtered[fit.metric] >= 0.0) & integer_like
        ].copy()
    allowed_sites = set(SITE_ORDER)
    allowed_compartments = set(COMPARTMENT_ORDER)
    filtered = filtered.loc[
        filtered[model.site_column].astype(str).isin(allowed_sites)
        & filtered[model.compartment_column].astype(str).isin(allowed_compartments)
        & filtered[model.condition_column].astype(str).isin(CONDITION_ORDER)
    ].copy()
    if filtered.empty:
        raise ValueError(f"No primary rows remain for fit '{fit.fit_id}'.")
    return filtered


def load_primary_summary_row(
    config: PatientBayesAnalysisConfig,
    fit_id: str,
) -> pd.Series | None:
    """Load the primary Bayesian summary row for one configured fit.

    Args:
        config (PatientBayesAnalysisConfig): Analysis configuration.
        fit_id (str): Stable fit identifier.

    Returns:
        pd.Series | None: Matching primary row, or ``None`` when unavailable.
    """

    path = config.paths.summary_csv
    if not path.exists():
        return None
    summary = pd.read_csv(path, low_memory=False)
    if "fit_id" not in summary.columns:
        raise KeyError(f"Bayesian summary lacks 'fit_id': {path}")
    selected = summary.loc[summary["fit_id"].astype(str) == str(fit_id)].copy()
    if "site_scenario" in selected.columns:
        selected = selected.loc[
            selected["site_scenario"].astype(str) == PRIMARY_SITE_SCENARIO
        ]
    if selected.empty:
        return None
    return selected.iloc[-1]


def aggregate_patient_values(
    data: pd.DataFrame,
    config: PatientBayesAnalysisConfig,
    fit: PatientBayesFitConfig,
) -> pd.DataFrame:
    """Aggregate to one equal-weight value per patient, site, and compartment.

    Args:
        data (pd.DataFrame): Primary-filtered metric rows.
        config (PatientBayesAnalysisConfig): Analysis configuration.
        fit (PatientBayesFitConfig): Enabled metric fit.

    Returns:
        pd.DataFrame: Equal-weight patient values.
    """

    model = config.model
    group_columns = [
        model.patient_column,
        model.condition_column,
        model.genotype_column,
        model.site_column,
        model.compartment_column,
    ]
    grouped = data.groupby(group_columns, observed=True, as_index=False)[fit.metric]
    if config.analysis_id == "instance":
        return grouped.median()
    return grouped.mean()


def rank_biserial_from_u(
    u_statistic: float,
    n_control: int,
    n_capn3: int,
) -> float:
    """Calculate rank-biserial correlation in the CTRL-minus-CAPN3 direction.

    Args:
        u_statistic (float): Mann-Whitney U statistic for CTRL as the first group.
        n_control (int): Number of CTRL patients.
        n_capn3 (int): Number of CAPN3 patients.

    Returns:
        float: Rank-biserial effect from -1 to 1.
    """

    denominator = n_control * n_capn3
    if denominator <= 0:
        return float("nan")
    return float((2.0 * u_statistic / denominator) - 1.0)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Adjust finite p-values with the Benjamini-Hochberg procedure.

    Args:
        p_values (Sequence[float]): Raw p-values, possibly including NaN.

    Returns:
        np.ndarray: Adjusted q-values aligned to the input.
    """

    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return adjusted
    finite_values = values[finite_indices]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    count = len(ranked)
    raw_adjusted = ranked * count / np.arange(1, count + 1)
    monotone = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    restored = np.empty(count, dtype=float)
    restored[order] = np.clip(monotone, 0.0, 1.0)
    adjusted[finite_indices] = restored
    return adjusted


def condition_mann_whitney_table(
    patient_values: pd.DataFrame,
    config: PatientBayesAnalysisConfig,
    fit: PatientBayesFitConfig,
) -> pd.DataFrame:
    """Run exploratory CTRL-versus-CAPN3 tests within each anatomical group.

    Args:
        patient_values (pd.DataFrame): Equal-weight patient aggregates.
        config (PatientBayesAnalysisConfig): Analysis configuration.
        fit (PatientBayesFitConfig): Enabled metric fit.

    Returns:
        pd.DataFrame: Mann-Whitney results with rank-biserial effects.
    """

    model = config.model
    rows: list[dict[str, Any]] = []
    for site in SITE_ORDER:
        for compartment in COMPARTMENT_ORDER:
            group = patient_values.loc[
                (patient_values[model.site_column].astype(str) == site)
                & (
                    patient_values[model.compartment_column].astype(str)
                    == compartment
                )
            ]
            control = group.loc[
                group[model.condition_column].astype(str) == CONDITION_ORDER[0],
                fit.metric,
            ].to_numpy(dtype=float)
            capn3 = group.loc[
                group[model.condition_column].astype(str) == CONDITION_ORDER[1],
                fit.metric,
            ].to_numpy(dtype=float)
            adequate = (
                len(control) >= MINIMUM_GROUP_PATIENTS
                and len(capn3) >= MINIMUM_GROUP_PATIENTS
            )
            u_statistic = float("nan")
            p_value = float("nan")
            effect = float("nan")
            if adequate:
                test = stats.mannwhitneyu(
                    control, capn3, alternative="two-sided", method="auto"
                )
                u_statistic = float(test.statistic)
                p_value = float(test.pvalue)
                effect = rank_biserial_from_u(
                    u_statistic, len(control), len(capn3)
                )
            rows.append(
                {
                    "analysis_id": config.analysis_id,
                    "fit_id": fit.fit_id,
                    "metric": fit.metric,
                    "site": site,
                    "compartment": compartment,
                    "contrast": "CTRL_vs_CAPN3",
                    "n_ctrl_patients": len(control),
                    "n_capn3_patients": len(capn3),
                    "ctrl_median": (
                        float(np.median(control)) if len(control) else np.nan
                    ),
                    "capn3_median": (
                        float(np.median(capn3)) if len(capn3) else np.nan
                    ),
                    "mann_whitney_u_ctrl": u_statistic,
                    "rank_biserial_ctrl_minus_capn3": effect,
                    "p_value": p_value,
                    "status": "tested" if adequate else "insufficient_patients",
                    "exploratory": True,
                }
            )
    result = pd.DataFrame(rows)
    result["q_value_bh"] = benjamini_hochberg(result["p_value"])
    return result


def capn3_subtype_tables(
    patient_values: pd.DataFrame,
    config: PatientBayesAnalysisConfig,
    fit: PatientBayesFitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run adequate CAPN3 subtype Kruskal-Wallis and pairwise tests.

    Args:
        patient_values (pd.DataFrame): Equal-weight patient aggregates.
        config (PatientBayesAnalysisConfig): Analysis configuration.
        fit (PatientBayesFitConfig): Enabled metric fit.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Omnibus and pairwise result tables.
    """

    model = config.model
    capn3 = patient_values.loc[
        patient_values[model.condition_column].astype(str) == CONDITION_ORDER[1]
    ].copy()
    omnibus_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    for site in SITE_ORDER:
        for compartment in COMPARTMENT_ORDER:
            group = capn3.loc[
                (capn3[model.site_column].astype(str) == site)
                & (capn3[model.compartment_column].astype(str) == compartment)
            ]
            subtype_values = {
                subtype: group.loc[
                    group[model.genotype_column].astype(str) == subtype,
                    fit.metric,
                ].to_numpy(dtype=float)
                for subtype in CAPN3_SUBTYPE_ORDER
            }
            adequate = {
                subtype: values
                for subtype, values in subtype_values.items()
                if len(values) >= MINIMUM_GROUP_PATIENTS
            }
            if len(adequate) < 2:
                continue
            omnibus = stats.kruskal(*adequate.values(), nan_policy="omit")
            omnibus_rows.append(
                {
                    "analysis_id": config.analysis_id,
                    "fit_id": fit.fit_id,
                    "metric": fit.metric,
                    "site": site,
                    "compartment": compartment,
                    "subtypes_tested": "|".join(adequate),
                    "n_subtypes": len(adequate),
                    "sample_sizes": "|".join(
                        f"{subtype}:{len(values)}"
                        for subtype, values in adequate.items()
                    ),
                    "kruskal_h": float(omnibus.statistic),
                    "p_value": float(omnibus.pvalue),
                    "exploratory": True,
                }
            )
            for first, second in itertools.combinations(adequate, 2):
                first_values = adequate[first]
                second_values = adequate[second]
                test = stats.mannwhitneyu(
                    first_values,
                    second_values,
                    alternative="two-sided",
                    method="auto",
                )
                pairwise_rows.append(
                    {
                        "analysis_id": config.analysis_id,
                        "fit_id": fit.fit_id,
                        "metric": fit.metric,
                        "site": site,
                        "compartment": compartment,
                        "subtype_1": first,
                        "subtype_2": second,
                        "n_subtype_1": len(first_values),
                        "n_subtype_2": len(second_values),
                        "mann_whitney_u_subtype_1": float(test.statistic),
                        "rank_biserial_subtype_1_minus_subtype_2": (
                            rank_biserial_from_u(
                                float(test.statistic),
                                len(first_values),
                                len(second_values),
                            )
                        ),
                        "p_value": float(test.pvalue),
                        "exploratory": True,
                    }
                )
    omnibus_columns = [
        "analysis_id",
        "fit_id",
        "metric",
        "site",
        "compartment",
        "subtypes_tested",
        "n_subtypes",
        "sample_sizes",
        "kruskal_h",
        "p_value",
        "q_value_bh",
        "exploratory",
    ]
    pairwise_columns = [
        "analysis_id",
        "fit_id",
        "metric",
        "site",
        "compartment",
        "subtype_1",
        "subtype_2",
        "n_subtype_1",
        "n_subtype_2",
        "mann_whitney_u_subtype_1",
        "rank_biserial_subtype_1_minus_subtype_2",
        "p_value",
        "q_value_bh",
        "exploratory",
    ]
    omnibus_table = pd.DataFrame(omnibus_rows)
    pairwise_table = pd.DataFrame(pairwise_rows)
    if not omnibus_table.empty:
        omnibus_table["q_value_bh"] = benjamini_hochberg(
            omnibus_table["p_value"]
        )
        omnibus_table = omnibus_table[omnibus_columns]
    else:
        omnibus_table = pd.DataFrame(columns=omnibus_columns)
    if not pairwise_table.empty:
        pairwise_table["q_value_bh"] = benjamini_hochberg(
            pairwise_table["p_value"]
        )
        pairwise_table = pairwise_table[pairwise_columns]
    else:
        pairwise_table = pd.DataFrame(columns=pairwise_columns)
    return omnibus_table, pairwise_table


def subtype_figure_is_feasible(
    patient_values: pd.DataFrame,
    config: PatientBayesAnalysisConfig,
) -> bool:
    """Check whether at least two CAPN3 subtypes have patient observations.

    Args:
        patient_values (pd.DataFrame): Equal-weight patient aggregates.
        config (PatientBayesAnalysisConfig): Analysis configuration.

    Returns:
        bool: Whether a subtype descriptive plot is informative.
    """

    model = config.model
    capn3 = patient_values.loc[
        patient_values[model.condition_column].astype(str) == CONDITION_ORDER[1]
    ]
    represented = capn3[model.genotype_column].dropna().astype(str).unique()
    return len(set(represented) & set(CAPN3_SUBTYPE_ORDER)) >= 2


def make_metric_figures(
    data: pd.DataFrame,
    patient_values: pd.DataFrame,
    config: PatientBayesAnalysisConfig,
    fit: PatientBayesFitConfig,
    summary_row: pd.Series | None,
    figure_dir: Path,
) -> list[Path]:
    """Generate condition and feasible CAPN3-subtype descriptive figures.

    Args:
        data (pd.DataFrame): Primary-filtered raw or image-summary rows.
        patient_values (pd.DataFrame): Equal-weight patient aggregates.
        config (PatientBayesAnalysisConfig): Analysis configuration.
        fit (PatientBayesFitConfig): Enabled metric fit.
        summary_row (pd.Series | None): Primary Bayesian fit summary.
        figure_dir (Path): Analysis-level output directory.

    Returns:
        list[Path]: Figure paths written for the metric.
    """

    model = config.model
    aggregate = "median" if config.analysis_id == "instance" else "mean"
    metric_slug = slugify(fit.fit_id)
    outputs: list[Path] = []
    condition_specs = (
        ("superviolin", patient_superviolin),
        ("superbeeswarm", patient_superbeeswarm),
    )
    for plot_name, plot_function in condition_specs:
        output = figure_dir / f"{metric_slug}_{plot_name}.png"
        figure, _ = plot_function(
            data=data,
            metric=fit.metric,
            patient_column=model.patient_column,
            condition_column=model.condition_column,
            site_column=model.site_column,
            compartment_column=model.compartment_column,
            aggregate=aggregate,
            summary_row=summary_row,
            output_path=output,
            title=f"{fit.metric}: CTRL and CAPN3",
        )
        plt.close(figure)
        outputs.append(output)
    if subtype_figure_is_feasible(patient_values, config):
        capn3_data = data.loc[
            data[model.condition_column].astype(str) == CONDITION_ORDER[1]
        ].copy()
        subtype_summary: dict[str, Any] = {
            "capn3_effect_response_summary": (
                "not applicable; CAPN3 subtype description"
            ),
            "fit_status": "ok",
        }
        for plot_name, plot_function in condition_specs:
            output = figure_dir / f"{metric_slug}_capn3_subtype_{plot_name}.png"
            figure, _ = plot_function(
                data=capn3_data,
                metric=fit.metric,
                patient_column=model.patient_column,
                condition_column=model.genotype_column,
                site_column=model.site_column,
                compartment_column=model.compartment_column,
                aggregate=aggregate,
                summary_row=subtype_summary,
                output_path=output,
                title=f"{fit.metric}: CAPN3 subtype (exploratory)",
                hue_order=CAPN3_SUBTYPE_ORDER,
                palette=CAPN3_SUBTYPE_PALETTE,
                quality_warning="Exploratory descriptive comparison",
            )
            plt.close(figure)
            outputs.append(output)
    return outputs


def run_analysis(config_path: str | Path) -> dict[str, Path]:
    """Run one configured measurement-level descriptive analysis.

    Args:
        config_path (str | Path): Instance or image-summary YAML path.

    Returns:
        dict[str, Path]: Written exploratory table paths keyed by table type.
    """

    config = load_hierarchical_bayes_config(Path(config_path))
    fits = enabled_fit_configs(config)
    if not fits:
        print(f"No enabled fits for {config.analysis_id}; skipping.")
        return {}
    measurements = load_measurements(config)
    results_dir = config.paths.summary_csv.parent
    figure_dir = (
        results_dir
        / "figures"
        / f"metric_stats_{config.analysis_id}"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    condition_tables: list[pd.DataFrame] = []
    omnibus_tables: list[pd.DataFrame] = []
    pairwise_tables: list[pd.DataFrame] = []
    for fit in fits:
        primary_data = apply_primary_filters(measurements, config, fit)
        patient_values = aggregate_patient_values(primary_data, config, fit)
        summary_row = load_primary_summary_row(config, fit.fit_id)
        make_metric_figures(
            primary_data,
            patient_values,
            config,
            fit,
            summary_row,
            figure_dir,
        )
        condition_tables.append(
            condition_mann_whitney_table(patient_values, config, fit)
        )
        omnibus, pairwise = capn3_subtype_tables(patient_values, config, fit)
        omnibus_tables.append(omnibus)
        pairwise_tables.append(pairwise)
        print(
            f"Completed descriptive {config.analysis_id} metric "
            f"{fit.fit_id} ({fit.metric})."
        )
    output_paths = {
        "condition": (
            results_dir
            / f"exploratory_mann_whitney_{config.analysis_id}.csv"
        ),
        "subtype_omnibus": (
            results_dir
            / f"exploratory_capn3_subtype_kruskal_{config.analysis_id}.csv"
        ),
        "subtype_pairwise": (
            results_dir
            / f"exploratory_capn3_subtype_pairwise_{config.analysis_id}.csv"
        ),
    }
    condition_result = pd.concat(condition_tables, ignore_index=True)
    omnibus_result = pd.concat(omnibus_tables, ignore_index=True)
    pairwise_result = pd.concat(pairwise_tables, ignore_index=True)
    condition_result["q_value_bh"] = benjamini_hochberg(
        condition_result["p_value"]
    )
    if not omnibus_result.empty:
        omnibus_result["q_value_bh"] = benjamini_hochberg(
            omnibus_result["p_value"]
        )
    if not pairwise_result.empty:
        pairwise_result["q_value_bh"] = benjamini_hochberg(
            pairwise_result["p_value"]
        )
    write_csv_atomic(condition_result, output_paths["condition"])
    write_csv_atomic(omnibus_result, output_paths["subtype_omnibus"])
    write_csv_atomic(pairwise_result, output_paths["subtype_pairwise"])
    return output_paths


def main(argv: Sequence[str] | None = None) -> int:
    """Run instance, image-summary, or both descriptive workflows.

    Args:
        argv (Sequence[str] | None): Optional command-line arguments.

    Returns:
        int: Process exit status.
    """

    parser = argparse.ArgumentParser(
        description="Generate CAPN3 patient descriptive figures and exploratory tests."
    )
    parser.add_argument(
        "--analysis",
        choices=("all", "instance", "image_summary"),
        default="all",
        help="Measurement level to process.",
    )
    parser.add_argument(
        "--instance-config",
        type=Path,
        default=DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH,
        help="Instance-level Bayesian YAML used for metric selection.",
    )
    parser.add_argument(
        "--image-summary-config",
        type=Path,
        default=IMAGE_SUMMARY_CONFIG_PATH,
        help="Image-summary Bayesian YAML used for metric selection.",
    )
    args = parser.parse_args(argv)
    if args.analysis in {"all", "instance"}:
        run_analysis(args.instance_config)
    if args.analysis in {"all", "image_summary"}:
        run_analysis(args.image_summary_config)
    return 0


# %% [markdown]
# ## Instance execution
#
# Uncomment the next line in a notebook to generate instance-level outputs.

# %%
# run_analysis(DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH)


# %% [markdown]
# ## Image-summary execution
#
# Uncomment the next line in a notebook to generate image-summary outputs.

# %%
# run_analysis(IMAGE_SUMMARY_CONFIG_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
