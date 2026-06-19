#!/usr/bin/env python3
"""Generate DMD_1X no-compartment statistics CSVs and superplots."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
import yaml

sns.set_style("whitegrid")
plt.switch_backend("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dmd_1x import stats_utils as dmd_stats_utils

RESULTS_DIR = REPO_ROOT / "data" / "DMD_1X" / "results"
INSTANCE_INPUT_CSV = RESULTS_DIR / "measurements_cleaned_no_compartment.csv"
IMAGE_SUMMARY_INPUT_CSV = RESULTS_DIR / "measurements_cleaned_no_compartment_summary.csv"
INSTANCE_FIGURES_DIR = RESULTS_DIR / "figures_instance_no_compartment"
IMAGE_SUMMARY_FIGURES_DIR = RESULTS_DIR / "figures_image_summary_no_compartment"
STATISTICS_CSV = RESULTS_DIR / "statistics_image_summary_no_compartment.csv"
STATISTICS_MUSCLE_CSV = RESULTS_DIR / "statistics_image_summary_muscle_no_compartment.csv"
BAYESIAN_INSTANCE_SUMMARY_CSV = RESULTS_DIR / "hierarchical_bayes_statistics_no_compartment.csv"
BAYESIAN_IMAGE_SUMMARY_CSV = RESULTS_DIR / "hierarchical_bayes_statistics_no_compartment_image_summary.csv"
BAYESIAN_INSTANCE_MUSCLE_SUMMARY_CSV = RESULTS_DIR / "hierarchical_bayes_muscle_no_compartment_statistics.csv"
BAYESIAN_IMAGE_SUMMARY_MUSCLE_SUMMARY_CSV = (
    RESULTS_DIR / "hierarchical_bayes_muscle_no_compartment_statistics_image_summary.csv"
)
BAYESIAN_CONFIG_YAML = REPO_ROOT / "dmd_1x" / "hierarchical_bayes_no_compartment_config.yaml"
BAYESIAN_IMAGE_SUMMARY_CONFIG_YAML = (
    REPO_ROOT / "dmd_1x" / "hierarchical_bayes_no_compartment_image_summary_config.yaml"
)
BAYESIAN_MUSCLE_CONFIG_YAML = REPO_ROOT / "dmd_1x" / "hierarchical_bayes_muscle_no_compartment_config.yaml"
BAYESIAN_MUSCLE_IMAGE_SUMMARY_CONFIG_YAML = (
    REPO_ROOT / "dmd_1x" / "hierarchical_bayes_muscle_no_compartment_image_summary_config.yaml"
)
ALL_COMPARTMENTS_LABEL = "All compartments"
MUSCLE_ORDER = ["Extraocular Muscle", "Tibialis Anterior"]
BAYES_FACTOR_ANNOTATION_MODE = "bayes_factor"
EFFECT_SUMMARY_ANNOTATION_MODE = "effect_summary"
DARK_MEDIAN_ANNOTATION_COLOR = "#08306b"
EXCLUDED_IMAGE_SUMMARY_METRICS = {
    "Zoom",
    "Image_width_px",
    "Image_height_px",
    "Pixel_size_nm",
    "Image_area_nm2",
}


def sort_conditions(values: list[str]) -> list[str]:
    """Sort condition labels with Wildtype-like labels first.

    Args:
        values (list[str]): Raw condition labels.

    Returns:
        list[str]: Sorted condition labels.
    """

    return dmd_stats_utils.sort_condition_values(values)


def build_muscle_label(muscle: str, compartment: str) -> str:
    """Create the no-compartment x-axis label for genotype comparisons.

    Args:
        muscle (str): Muscle label.
        compartment (str): Pooled compartment label, accepted for annotation compatibility.

    Returns:
        str: Muscle label used as the x-axis group.
    """

    _ = compartment
    return str(muscle)


def build_genotype_label(condition: str, compartment: str) -> str:
    """Create the no-compartment x-axis label for muscle comparisons.

    Args:
        condition (str): Genotype/condition label.
        compartment (str): Pooled compartment label, accepted for annotation compatibility.

    Returns:
        str: Short genotype label used as the x-axis group.
    """

    _ = compartment
    return dmd_stats_utils.format_condition_display_label(condition)


def load_single_dataframe(input_csv: Path) -> pd.DataFrame:
    """Load one DMD_1X no-compartment measurements CSV.

    Args:
        input_csv (Path): CSV path to load.

    Returns:
        pd.DataFrame: Loaded dataframe with categorical group columns.
    """

    if not input_csv.exists():
        raise FileNotFoundError(
            f"No DMD_1X no-compartment CSV was found at {input_csv}. "
            "Run dmd_1x/build_measurements_csv.py first."
        )
    df = pd.read_csv(input_csv)
    required_columns = {"Condition", "Muscle", "Compartment", "Block", "image"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise KeyError(f"Missing required no-compartment columns in {input_csv}: {missing_columns}")
    df["Condition"] = df["Condition"].astype("category")
    df["Muscle"] = df["Muscle"].astype("category")
    df = df.dropna(subset=["Compartment"]).copy()
    df["Compartment"] = df["Compartment"].astype("category")
    df["Muscle_Group"] = df["Muscle"].astype(str)
    df["Genotype_Group"] = df["Condition"].astype(str).map(
        dmd_stats_utils.format_condition_display_label
    )
    return df


def load_superplot_annotation_mode(config_yaml: Path) -> str:
    """Load the configured Bayesian superplot annotation mode.

    Args:
        config_yaml (Path): Bayesian config YAML path.

    Returns:
        str: Annotation mode, defaulting to ``effect_summary`` when unavailable.
    """

    if not config_yaml.exists():
        return EFFECT_SUMMARY_ANNOTATION_MODE
    with config_yaml.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    runtime = config.get("runtime", {})
    mode = runtime.get("superplot_annotation_mode", EFFECT_SUMMARY_ANNOTATION_MODE)
    if mode not in {BAYES_FACTOR_ANNOTATION_MODE, EFFECT_SUMMARY_ANNOTATION_MODE}:
        return EFFECT_SUMMARY_ANNOTATION_MODE
    return str(mode)


def load_bayesian_superplot_annotations(summary_csv: Path) -> pd.DataFrame:
    """Load Bayesian summary rows used for superplot bracket annotations.

    Args:
        summary_csv (Path): CSV containing hierarchical Bayesian fit summaries.

    Returns:
        pd.DataFrame: Summary dataframe, or an empty compatible dataframe.
    """

    required_columns = {"metric", "muscle", "compartment", "wt_label", "ko_label"}
    if not summary_csv.exists():
        return pd.DataFrame(columns=sorted(required_columns))
    summary_df = pd.read_csv(summary_csv)
    if not required_columns.issubset(summary_df.columns):
        missing_columns = sorted(required_columns - set(summary_df.columns))
        print(
            "Skipping Bayesian superplot annotations: missing columns "
            f"{missing_columns} in {summary_csv}."
        )
        return pd.DataFrame(columns=sorted(required_columns))
    return summary_df


def format_delta_effect_annotation(row: pd.Series, label: str) -> str:
    """Format one posterior delta effect summary for superplot annotation text.

    Args:
        row (pd.Series): Bayesian summary row with delta estimate fields.
        label (str): Column prefix such as ``delta_mean`` or ``delta_median``.

    Returns:
        str: Formatted annotation text, or an empty string when unavailable.
    """

    estimate = row.get(f"{label}_response", row.get(label))
    low = row.get(f"{label}_hdi_low_response", row.get(f"{label}_hdi_low"))
    high = row.get(f"{label}_hdi_high_response", row.get(f"{label}_hdi_high"))
    probability = row.get(f"{label}_pd")
    if pd.isna(estimate) or pd.isna(low) or pd.isna(high):
        return ""
    annotation = f"{float(estimate):.3g} [{float(low):.3g}, {float(high):.3g}]"
    if probability is not None and not pd.isna(probability):
        annotation = f"{annotation} {float(probability):.1f}%"
    return annotation


def bayesian_annotations_for_metric(
    summary_df: pd.DataFrame,
    metric: str,
    annotation_mode: str,
    x_label_builder: Callable[[str, str], str],
) -> list[dict[str, str]]:
    """Build Bayesian superplot annotations for one metric.

    Args:
        summary_df (pd.DataFrame): Bayesian summary dataframe.
        metric (str): Metric currently being plotted.
        annotation_mode (str): Either ``bayes_factor`` or ``effect_summary``.
        x_label_builder (Callable[[str, str], str]): Builder for x-axis group labels
            using summary ``muscle`` and ``compartment`` values.

    Returns:
        list[dict[str, str]]: Annotation records for ``stats_utils``.
    """

    if summary_df.empty:
        return []
    metric_rows = summary_df.loc[summary_df["metric"].astype(str) == str(metric)]
    annotations: list[dict[str, str]] = []
    for _, row in metric_rows.iterrows():
        x_label = x_label_builder(str(row["muscle"]), str(row["compartment"]))
        if annotation_mode == EFFECT_SUMMARY_ANNOTATION_MODE:
            mean_label = format_delta_effect_annotation(row=row, label="delta_mean")
            median_label = format_delta_effect_annotation(row=row, label="delta_median")
        else:
            mean_label = str(row.get("delta_mean_bf_annotation", "")).strip()
            median_label = str(row.get("delta_median_bf_annotation", "")).strip()
            mean_label = f"mean {mean_label}" if mean_label and mean_label != "nan" else ""
            median_label = f"median {median_label}" if median_label and median_label != "nan" else ""
        if not mean_label and not median_label:
            continue
        annotations.append(
            {
                "x": x_label,
                "hue_start": str(row["wt_label"]),
                "hue_end": str(row["ko_label"]),
                "mean_label": mean_label,
                "mean_color": "purple",
                "median_label": median_label,
                "median_color": DARK_MEDIAN_ANNOTATION_COLOR,
            }
        )
    return annotations


def numeric_metric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric metric columns for plotting and testing.

    Args:
        df (pd.DataFrame): DMD_1X no-compartment dataframe.

    Returns:
        list[str]: Metric column names sorted in dataframe order.
    """

    excluded_columns = {
        "Condition",
        "Muscle",
        "Block",
        "image",
        "Id",
        "Centroid",
        "Image_Region",
        "Compartment",
        "Muscle_Group",
        "Genotype_Group",
        "Zoom",
        "Pixel_size_nm",
        "Connected_parts",
    }
    metrics: list[str] = []
    for column in df.columns:
        if column in excluded_columns:
            continue
        numeric_values = pd.to_numeric(df[column], errors="coerce")
        if numeric_values.notna().any():
            metrics.append(column)
    return metrics


def ordered_muscles(data: pd.DataFrame) -> list[str]:
    """Return a stable two-level muscle order for no-compartment plots.

    Args:
        data (pd.DataFrame): Dataframe containing a ``Muscle`` column.

    Returns:
        list[str]: Ordered muscle labels.
    """

    muscle_order = [value for value in MUSCLE_ORDER if value in set(data["Muscle"].dropna().astype(str))]
    remaining_muscles = sorted(set(data["Muscle"].dropna().astype(str)) - set(muscle_order))
    muscle_order.extend(remaining_muscles)
    if len(muscle_order) != 2:
        raise ValueError(f"Expected exactly 2 muscle levels, found {len(muscle_order)} ({muscle_order}).")
    return muscle_order


def ordered_conditions(data: pd.DataFrame) -> list[str]:
    """Return a stable two-level condition order for no-compartment plots.

    Args:
        data (pd.DataFrame): Dataframe containing a ``Condition`` column.

    Returns:
        list[str]: Ordered condition labels.
    """

    conditions = sort_conditions(list(data["Condition"].dropna().astype(str).unique()))
    if len(conditions) != 2:
        raise ValueError(f"Expected exactly 2 conditions, found {len(conditions)} ({conditions}).")
    return conditions


def ordered_genotype_groups(conditions: list[str]) -> list[str]:
    """Return display labels for genotype x-axis groups.

    Args:
        conditions (list[str]): Ordered raw condition labels.

    Returns:
        list[str]: Formatted genotype labels.
    """

    return [dmd_stats_utils.format_condition_display_label(condition) for condition in conditions]


def generate_superplots_for_metrics(
    data: pd.DataFrame,
    metrics: list[str],
    save_dir: Path,
    x_order: list[str],
    hue_order: list[str],
    unit_dict: dict[str, str],
    summary_df: pd.DataFrame,
    annotation_mode: str,
    x_column: str,
    hue_column: str,
    output_dir_suffix: str,
    x_label_builder: Callable[[str, str], str],
) -> None:
    """Generate no-compartment superviolin and superbeeswarm plots.

    Args:
        data (pd.DataFrame): DMD_1X no-compartment dataframe.
        metrics (list[str]): Metrics to plot.
        save_dir (Path): Directory where figures are saved.
        x_order (list[str]): Explicit x-axis order.
        hue_order (list[str]): Explicit hue order.
        unit_dict (dict[str, str]): Metric unit mapping.
        summary_df (pd.DataFrame): Bayesian summary dataframe for annotations.
        annotation_mode (str): Superplot annotation mode.
        x_column (str): Data column used for the x-axis grouping.
        hue_column (str): Data column used for group hues.
        output_dir_suffix (str): Suffix appended to generated superplot directories.
        x_label_builder (Callable[[str, str], str]): Builder used to align Bayesian
            annotation x labels with plotting groups.

    Returns:
        None: Saves figures to disk.
    """

    for metric in metrics:
        annotations = bayesian_annotations_for_metric(
            summary_df=summary_df,
            metric=metric,
            annotation_mode=annotation_mode,
            x_label_builder=x_label_builder,
        )
        dmd_stats_utils.plot_super_violin(
            data=data,
            x=x_column,
            y=metric,
            hue=hue_column,
            block="Block",
            unit_dict=unit_dict,
            save_dir=save_dir,
            superplot_annotations=annotations,
            x_order_override=x_order,
            hue_order_override=hue_order,
            output_dir_suffix=output_dir_suffix,
        )
        dmd_stats_utils.plot_super_beeswarm(
            data=data,
            x=x_column,
            y=metric,
            hue=hue_column,
            block="Block",
            unit_dict=unit_dict,
            save_dir=save_dir,
            superplot_annotations=annotations,
            x_order_override=x_order,
            hue_order_override=hue_order,
            output_dir_suffix=output_dir_suffix,
        )


def compute_genotype_statistics(
    data: pd.DataFrame,
    metrics: list[str],
    conditions: list[str],
    muscles: list[str],
) -> pd.DataFrame:
    """Compute image-summary genotype statistics within each muscle.

    Args:
        data (pd.DataFrame): Image-summary no-compartment dataframe.
        metrics (list[str]): Metrics to compare.
        conditions (list[str]): Ordered genotype labels.
        muscles (list[str]): Ordered muscle labels.

    Returns:
        pd.DataFrame: Mann-Whitney summary rows for genotype comparisons.
    """

    results: list[dict[str, object]] = []
    for metric in metrics:
        for muscle in muscles:
            group_slice = data.loc[data["Muscle"].astype(str) == muscle].copy()
            if group_slice.empty:
                continue
            values_a = pd.to_numeric(
                group_slice.loc[group_slice["Condition"].astype(str) == conditions[0], metric],
                errors="coerce",
            ).dropna()
            values_b = pd.to_numeric(
                group_slice.loc[group_slice["Condition"].astype(str) == conditions[1], metric],
                errors="coerce",
            ).dropna()
            if values_a.empty or values_b.empty:
                continue
            statistic, p_value = mannwhitneyu(values_a, values_b, alternative="two-sided")
            results.append(
                {
                    "Measurement": metric,
                    "Muscle": muscle,
                    "Compartment": ALL_COMPARTMENTS_LABEL,
                    "Condition_A": conditions[0],
                    "Condition_B": conditions[1],
                    "N_A": len(values_a),
                    "N_B": len(values_b),
                    "Mean_A": values_a.mean(),
                    "Mean_B": values_b.mean(),
                    "Median_A": values_a.median(),
                    "Median_B": values_b.median(),
                    "Mann_Whitney_U": statistic,
                    "p_value": p_value,
                }
            )
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values(["Measurement", "Muscle", "Compartment"])


def compute_muscle_statistics(
    data: pd.DataFrame,
    metrics: list[str],
    conditions: list[str],
    muscles: list[str],
) -> pd.DataFrame:
    """Compute image-summary muscle statistics within each genotype.

    Args:
        data (pd.DataFrame): Image-summary no-compartment dataframe.
        metrics (list[str]): Metrics to compare.
        conditions (list[str]): Ordered genotype labels.
        muscles (list[str]): Ordered muscle labels.

    Returns:
        pd.DataFrame: Mann-Whitney summary rows for muscle comparisons.
    """

    results: list[dict[str, object]] = []
    for metric in metrics:
        for condition in conditions:
            group_slice = data.loc[data["Condition"].astype(str) == condition].copy()
            if group_slice.empty:
                continue
            values_a = pd.to_numeric(
                group_slice.loc[group_slice["Muscle"].astype(str) == muscles[0], metric],
                errors="coerce",
            ).dropna()
            values_b = pd.to_numeric(
                group_slice.loc[group_slice["Muscle"].astype(str) == muscles[1], metric],
                errors="coerce",
            ).dropna()
            if values_a.empty or values_b.empty:
                continue
            statistic, p_value = mannwhitneyu(values_a, values_b, alternative="two-sided")
            results.append(
                {
                    "Measurement": metric,
                    "Genotype": condition,
                    "Compartment": ALL_COMPARTMENTS_LABEL,
                    "Muscle_A": muscles[0],
                    "Muscle_B": muscles[1],
                    "N_A": len(values_a),
                    "N_B": len(values_b),
                    "Mean_A": values_a.mean(),
                    "Mean_B": values_b.mean(),
                    "Median_A": values_a.median(),
                    "Median_B": values_b.median(),
                    "Mann_Whitney_U": statistic,
                    "p_value": p_value,
                }
            )
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values(["Measurement", "Genotype", "Compartment"])


def metric_units() -> dict[str, str]:
    """Return display units for DMD_1X instance and image-summary metrics.

    Args:
        None: This function does not accept arguments.

    Returns:
        dict[str, str]: Metric-to-unit mapping.
    """

    return {
        "Area": "nm^2",
        "Corrected_area": "nm^2",
        "Major_axis_length": "nm",
        "Minor_axis_length": "nm",
        "Minimum_Feret_Diameter": "nm",
        "Elongation": "",
        "Circularity": "",
        "Solidity": "",
        "NND": "nm",
        "3NND": "nm",
        "5NND": "nm",
        "Voronoi_Cell_Area": "nm^2",
        "Density": "count/image",
        "Instance_count": "count",
        "Image_width_px": "px",
        "Image_height_px": "px",
        "Pixel_size_nm": "nm",
        "Image_area_nm2": "nm^2",
        "Area_sum": "nm^2",
        "Corrected_area_sum": "nm^2",
        "Minimum_Feret_Diameter_sum": "nm",
        "Minor_axis_length_sum": "nm",
        "Area_mean": "nm^2",
        "Corrected_area_mean": "nm^2",
        "Minimum_Feret_Diameter_mean": "nm",
        "Major_axis_length_mean": "nm",
        "Minor_axis_length_mean": "nm",
        "NND_center_mean": "nm",
        "3NND_center_mean": "nm",
        "5NND_center_mean": "nm",
        "Voronoi_Cell_Area_center_mean": "nm^2",
    }


def generate_instance_outputs(df_instance: pd.DataFrame, units: dict[str, str]) -> None:
    """Generate no-compartment instance-level genotype and muscle superplots.

    Args:
        df_instance (pd.DataFrame): Instance-level no-compartment dataframe.
        units (dict[str, str]): Metric-to-unit mapping.

    Returns:
        None: Saves instance-level superplot files.
    """

    conditions = ordered_conditions(df_instance)
    muscles = ordered_muscles(df_instance)
    instance_metrics = numeric_metric_columns(df_instance)
    genotype_summary_df = load_bayesian_superplot_annotations(summary_csv=BAYESIAN_INSTANCE_SUMMARY_CSV)
    genotype_annotation_mode = load_superplot_annotation_mode(config_yaml=BAYESIAN_CONFIG_YAML)
    generate_superplots_for_metrics(
        data=df_instance,
        metrics=instance_metrics,
        save_dir=INSTANCE_FIGURES_DIR,
        x_order=muscles,
        hue_order=conditions,
        unit_dict=units,
        summary_df=genotype_summary_df,
        annotation_mode=genotype_annotation_mode,
        x_column="Muscle_Group",
        hue_column="Condition",
        output_dir_suffix="",
        x_label_builder=build_muscle_label,
    )

    muscle_summary_df = load_bayesian_superplot_annotations(summary_csv=BAYESIAN_INSTANCE_MUSCLE_SUMMARY_CSV)
    muscle_annotation_mode = load_superplot_annotation_mode(config_yaml=BAYESIAN_MUSCLE_CONFIG_YAML)
    generate_superplots_for_metrics(
        data=df_instance,
        metrics=instance_metrics,
        save_dir=INSTANCE_FIGURES_DIR,
        x_order=ordered_genotype_groups(conditions),
        hue_order=muscles,
        unit_dict=units,
        summary_df=muscle_summary_df,
        annotation_mode=muscle_annotation_mode,
        x_column="Genotype_Group",
        hue_column="Muscle",
        output_dir_suffix="_muscle",
        x_label_builder=build_genotype_label,
    )


def generate_image_summary_outputs(df_image_summary: pd.DataFrame, units: dict[str, str]) -> None:
    """Generate no-compartment image-summary plots and statistics CSVs.

    Args:
        df_image_summary (pd.DataFrame): Image-summary no-compartment dataframe.
        units (dict[str, str]): Metric-to-unit mapping.

    Returns:
        None: Saves image-summary figures and Mann-Whitney statistics CSVs.
    """

    conditions = ordered_conditions(df_image_summary)
    muscles = ordered_muscles(df_image_summary)
    metrics = [
        metric
        for metric in numeric_metric_columns(df_image_summary)
        if metric not in EXCLUDED_IMAGE_SUMMARY_METRICS
    ]

    genotype_summary_df = load_bayesian_superplot_annotations(summary_csv=BAYESIAN_IMAGE_SUMMARY_CSV)
    genotype_annotation_mode = load_superplot_annotation_mode(config_yaml=BAYESIAN_IMAGE_SUMMARY_CONFIG_YAML)
    generate_superplots_for_metrics(
        data=df_image_summary,
        metrics=metrics,
        save_dir=IMAGE_SUMMARY_FIGURES_DIR,
        x_order=muscles,
        hue_order=conditions,
        unit_dict=units,
        summary_df=genotype_summary_df,
        annotation_mode=genotype_annotation_mode,
        x_column="Muscle_Group",
        hue_column="Condition",
        output_dir_suffix="",
        x_label_builder=build_muscle_label,
    )

    muscle_summary_df = load_bayesian_superplot_annotations(
        summary_csv=BAYESIAN_IMAGE_SUMMARY_MUSCLE_SUMMARY_CSV
    )
    muscle_annotation_mode = load_superplot_annotation_mode(config_yaml=BAYESIAN_MUSCLE_IMAGE_SUMMARY_CONFIG_YAML)
    generate_superplots_for_metrics(
        data=df_image_summary,
        metrics=metrics,
        save_dir=IMAGE_SUMMARY_FIGURES_DIR,
        x_order=ordered_genotype_groups(conditions),
        hue_order=muscles,
        unit_dict=units,
        summary_df=muscle_summary_df,
        annotation_mode=muscle_annotation_mode,
        x_column="Genotype_Group",
        hue_column="Muscle",
        output_dir_suffix="_muscle",
        x_label_builder=build_genotype_label,
    )

    genotype_statistics = compute_genotype_statistics(
        data=df_image_summary,
        metrics=metrics,
        conditions=conditions,
        muscles=muscles,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    genotype_statistics.to_csv(STATISTICS_CSV, index=False)
    print(f"Saved DMD_1X no-compartment image-summary statistics to {STATISTICS_CSV}")

    muscle_statistics = compute_muscle_statistics(
        data=df_image_summary,
        metrics=metrics,
        conditions=conditions,
        muscles=muscles,
    )
    muscle_statistics.to_csv(STATISTICS_MUSCLE_CSV, index=False)
    print(f"Saved DMD_1X no-compartment muscle image-summary statistics to {STATISTICS_MUSCLE_CSV}")


def main() -> None:
    """Run no-compartment DMD_1X metric statistics and superplot generation.

    Args:
        None: Reads fixed DMD_1X result paths from module constants.

    Returns:
        None: Writes CSV and figure outputs under ``data/DMD_1X/results``.
    """

    units = metric_units()
    df_instance = load_single_dataframe(input_csv=INSTANCE_INPUT_CSV)
    generate_instance_outputs(df_instance=df_instance, units=units)

    df_image_summary = load_single_dataframe(input_csv=IMAGE_SUMMARY_INPUT_CSV)
    generate_image_summary_outputs(df_image_summary=df_image_summary, units=units)


if __name__ == "__main__":
    main()
