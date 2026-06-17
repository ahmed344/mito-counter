# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% Imports and plotting configuration
from __future__ import annotations

from pathlib import Path
import sys
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

INPUT_CSV = REPO_ROOT / "data" / "DMD_1X" / "results" / "measurements_cleaned.csv"
IMAGE_SUMMARY_INPUT_CSVS = [
    REPO_ROOT / "data" / "DMD_1X" / "results" / "measurements_cleaned_ss_summary.csv",
    REPO_ROOT / "data" / "DMD_1X" / "results" / "measurements_cleaned_imf_summary.csv",
]
RESULTS_DIR = REPO_ROOT / "data" / "DMD_1X" / "results"
INSTANCE_FIGURES_DIR = RESULTS_DIR / "figures_instance"
IMAGE_SUMMARY_FIGURES_DIR = RESULTS_DIR / "figures_image_summary"
STATISTICS_CSV = RESULTS_DIR / "statistics_image_summary.csv"
STATISTICS_MUSCLE_CSV = RESULTS_DIR / "statistics_image_summary_muscle.csv"
BAYESIAN_INSTANCE_SUMMARY_CSV = RESULTS_DIR / "hierarchical_bayes_statistics.csv"
BAYESIAN_IMAGE_SUMMARY_CSV = RESULTS_DIR / "hierarchical_bayes_statistics_image_summary.csv"
BAYESIAN_INSTANCE_MUSCLE_SUMMARY_CSV = RESULTS_DIR / "hierarchical_bayes_muscle_statistics.csv"
BAYESIAN_IMAGE_SUMMARY_MUSCLE_SUMMARY_CSV = (
    RESULTS_DIR / "hierarchical_bayes_muscle_statistics_image_summary.csv"
)
BAYESIAN_CONFIG_YAML = REPO_ROOT / "dmd_1x" / "hierarchical_bayes_config.yaml"
BAYESIAN_IMAGE_SUMMARY_CONFIG_YAML = REPO_ROOT / "dmd_1x" / "hierarchical_bayes_image_summary_config.yaml"
BAYESIAN_MUSCLE_CONFIG_YAML = REPO_ROOT / "dmd_1x" / "hierarchical_bayes_muscle_config.yaml"
BAYESIAN_MUSCLE_IMAGE_SUMMARY_CONFIG_YAML = (
    REPO_ROOT / "dmd_1x" / "hierarchical_bayes_muscle_image_summary_config.yaml"
)
SS_LABEL = "Sub-sarcolemmal (SS)"
IMF_LABEL = "Intermyofibrillar (IMF)"
COMPARTMENT_ORDER = [SS_LABEL, IMF_LABEL]
MUSCLE_ORDER = ["Extraocular Muscle", "Tibialis Anterior"]
BAYES_FACTOR_ANNOTATION_MODE = "bayes_factor"
EFFECT_SUMMARY_ANNOTATION_MODE = "effect_summary"
DARK_MEDIAN_ANNOTATION_COLOR = "#08306b"


# %% Helper functions
def sort_conditions(values: list[str]) -> list[str]:
    """Sort condition labels with Wildtype-like labels first.

    Args:
        values (list[str]): Raw condition labels.

    Returns:
        list[str]: Sorted condition labels.
    """

    return dmd_stats_utils.sort_condition_values(values)


def build_muscle_compartment_label(muscle: str, compartment: str) -> str:
    """Create a compact combined plotting label for muscle and compartment.

    Args:
        muscle (str): Muscle label.
        compartment (str): Compartment label.

    Returns:
        str: Combined muscle-compartment label.
    """

    muscle_text = str(muscle).strip()
    if muscle_text == "Extraocular Muscle":
        short_muscle = "EOM"
    elif muscle_text == "Tibialis Anterior":
        short_muscle = "TA"
    else:
        short_muscle = muscle_text
    short_compartment = "SS" if compartment == SS_LABEL else "IMF"
    return f"{short_muscle} | {short_compartment}"


def build_genotype_compartment_label(condition: str, compartment: str) -> str:
    """Create a compact combined plotting label for genotype and compartment.

    Args:
        condition (str): Genotype/condition label.
        compartment (str): Compartment label.

    Returns:
        str: Combined genotype-compartment label.
    """

    short_condition = dmd_stats_utils.format_condition_display_label(condition)
    short_compartment = "SS" if compartment == SS_LABEL else "IMF"
    return f"{short_condition} | {short_compartment}"


def load_image_summary_dataframe(input_csvs: list[Path]) -> pd.DataFrame:
    """Load and combine DMD_1X image-summary CSV files.

    Args:
        input_csvs (list[Path]): CSV paths containing one row per image.

    Returns:
        pd.DataFrame: Concatenated image-summary table with categorical group columns.
    """

    frames = [pd.read_csv(path) for path in input_csvs if path.exists()]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["Condition"] = combined["Condition"].astype("category")
    combined["Muscle"] = combined["Muscle"].astype("category")
    combined = combined.dropna(subset=["Compartment"]).copy()
    combined["Compartment"] = combined["Compartment"].astype("category")
    combined["Muscle_Compartment"] = combined.apply(
        lambda row: build_muscle_compartment_label(row["Muscle"], row["Compartment"]),
        axis=1,
    )
    return combined


def load_instance_dataframe(input_csv: Path) -> pd.DataFrame:
    """Load the DMD_1X cleaned instance-level measurements CSV.

    Args:
        input_csv (Path): CSV path containing one row per mitochondrion instance.

    Returns:
        pd.DataFrame: Instance-level table with categorical group columns.
    """

    if not input_csv.exists():
        raise FileNotFoundError(
            f"No DMD_1X instance CSV was found at {input_csv}. Run dmd_1x/build_measurements_csv.py first."
        )
    df = pd.read_csv(input_csv)
    df["Condition"] = df["Condition"].astype("category")
    df["Muscle"] = df["Muscle"].astype("category")
    df = df.dropna(subset=["Compartment"]).copy()
    df["Compartment"] = df["Compartment"].astype("category")
    df["Muscle_Compartment"] = df.apply(
        lambda row: build_muscle_compartment_label(row["Muscle"], row["Compartment"]),
        axis=1,
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
    x_label_builder: Callable[[str, str], str] = build_muscle_compartment_label,
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
    """Return numeric image-summary metric columns for plotting and testing.

    Args:
        df (pd.DataFrame): DMD_1X image-summary dataframe.

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
        "Muscle_Compartment",
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


def generate_superplots_for_metrics(
    data: pd.DataFrame,
    metrics: list[str],
    save_dir: Path,
    x_order: list[str],
    hue_order: list[str],
    unit_dict: dict[str, str],
    summary_df: pd.DataFrame,
    annotation_mode: str,
    x_column: str = "Muscle_Compartment",
    hue_column: str = "Condition",
    output_dir_suffix: str = "",
    x_label_builder: Callable[[str, str], str] = build_muscle_compartment_label,
) -> None:
    """Generate image-level superviolin and superbeeswarm plots.

    Args:
        data (pd.DataFrame): DMD_1X image-summary dataframe.
        metrics (list[str]): Metrics to plot.
        save_dir (Path): Directory where figures are saved.
        x_order (list[str]): Explicit x-axis order.
        hue_order (list[str]): Explicit hue order.
        unit_dict (dict[str, str]): Metric unit mapping.
        summary_df (pd.DataFrame): Bayesian summary dataframe for annotations.
        annotation_mode (str): Superplot annotation mode.
        x_column (str): Data column used for the x-axis grouping.
        hue_column (str): Data column used for group hues.
        output_dir_suffix (str): Optional suffix appended to generated superplot
            directories.
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


def prepare_muscle_contrast_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare a dataframe for muscle contrast plots within genotype strata.

    Args:
        data (pd.DataFrame): Source dataframe with ``Condition``, ``Muscle``, and
            ``Compartment`` columns.

    Returns:
        pd.DataFrame: Copy with ``Genotype_Compartment`` helper labels.
    """

    prepared = data.copy()
    prepared["Genotype_Compartment"] = prepared.apply(
        lambda row: build_genotype_compartment_label(
            condition=str(row["Condition"]),
            compartment=str(row["Compartment"]),
        ),
        axis=1,
    )
    return prepared


# %% Load and prepare instance-level input data
df_instance = load_instance_dataframe(input_csv=INPUT_CSV)
conditions = sort_conditions(list(df_instance["Condition"].dropna().astype(str).unique()))
if len(conditions) != 2:
    raise ValueError(f"Expected exactly 2 conditions, found {len(conditions)} ({conditions}).")

muscle_order = [value for value in MUSCLE_ORDER if value in set(df_instance["Muscle"].dropna().astype(str))]
remaining_muscles = sorted(set(df_instance["Muscle"].dropna().astype(str)) - set(muscle_order))
muscle_order.extend(remaining_muscles)
muscle_compartment_order = [
    build_muscle_compartment_label(muscle, compartment)
    for muscle in muscle_order
    for compartment in COMPARTMENT_ORDER
    if (
        (df_instance["Muscle"].astype(str) == muscle)
        & (df_instance["Compartment"].astype(str) == compartment)
    ).any()
]

# %% Generate instance-level superplots with optional Bayesian annotations
units = {
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

instance_superplot_annotation_mode = load_superplot_annotation_mode(config_yaml=BAYESIAN_CONFIG_YAML)
bayesian_instance_summary_df = load_bayesian_superplot_annotations(
    summary_csv=BAYESIAN_INSTANCE_SUMMARY_CSV
)
instance_metrics = numeric_metric_columns(df_instance)
generate_superplots_for_metrics(
    data=df_instance,
    metrics=instance_metrics,
    save_dir=INSTANCE_FIGURES_DIR,
    x_order=muscle_compartment_order,
    hue_order=conditions,
    unit_dict=units,
    summary_df=bayesian_instance_summary_df,
    annotation_mode=instance_superplot_annotation_mode,
)

# %% Generate instance-level muscle-contrast superplots
df_instance_muscle = prepare_muscle_contrast_dataframe(data=df_instance)
instance_muscle_order = [
    value for value in MUSCLE_ORDER if value in set(df_instance_muscle["Muscle"].dropna().astype(str))
]
instance_remaining_muscles = sorted(set(df_instance_muscle["Muscle"].dropna().astype(str)) - set(instance_muscle_order))
instance_muscle_order.extend(instance_remaining_muscles)
if len(instance_muscle_order) != 2:
    raise ValueError(
        f"Expected exactly 2 instance muscle levels for muscle superplots, found "
        f"{len(instance_muscle_order)} ({instance_muscle_order})."
    )
instance_genotype_compartment_order = [
    build_genotype_compartment_label(condition=condition, compartment=compartment)
    for condition in conditions
    for compartment in COMPARTMENT_ORDER
    if (
        (df_instance_muscle["Condition"].astype(str) == condition)
        & (df_instance_muscle["Compartment"].astype(str) == compartment)
    ).any()
]
instance_muscle_annotation_mode = load_superplot_annotation_mode(
    config_yaml=BAYESIAN_MUSCLE_CONFIG_YAML
)
bayesian_instance_muscle_summary_df = load_bayesian_superplot_annotations(
    summary_csv=BAYESIAN_INSTANCE_MUSCLE_SUMMARY_CSV
)
generate_superplots_for_metrics(
    data=df_instance_muscle,
    metrics=instance_metrics,
    save_dir=INSTANCE_FIGURES_DIR,
    x_order=instance_genotype_compartment_order,
    hue_order=instance_muscle_order,
    unit_dict=units,
    summary_df=bayesian_instance_muscle_summary_df,
    annotation_mode=instance_muscle_annotation_mode,
    x_column="Genotype_Compartment",
    hue_column="Muscle",
    output_dir_suffix="_muscle",
    x_label_builder=build_genotype_compartment_label,
)


# %% Load and prepare image-summary input data
df_image_summary = load_image_summary_dataframe(input_csvs=IMAGE_SUMMARY_INPUT_CSVS)
if df_image_summary.empty:
    raise FileNotFoundError(
        "No DMD_1X image-summary CSVs were found. Run dmd_1x/build_measurements_csv.py first."
    )

image_summary_conditions = sort_conditions(list(df_image_summary["Condition"].dropna().astype(str).unique()))
if len(image_summary_conditions) != 2:
    raise ValueError(
        f"Expected exactly 2 image-summary conditions, found "
        f"{len(image_summary_conditions)} ({image_summary_conditions})."
    )

image_summary_muscle_compartment_order = [
    build_muscle_compartment_label(muscle, compartment)
    for muscle in muscle_order
    for compartment in COMPARTMENT_ORDER
    if (
        (df_image_summary["Muscle"] == muscle)
        & (df_image_summary["Compartment"] == compartment)
    ).any()
]


# %% Generate image-summary superplots with optional Bayesian annotations
superplot_annotation_mode = load_superplot_annotation_mode(config_yaml=BAYESIAN_IMAGE_SUMMARY_CONFIG_YAML)
bayesian_summary_df = load_bayesian_superplot_annotations(summary_csv=BAYESIAN_IMAGE_SUMMARY_CSV)
metrics = [
    metric
    for metric in numeric_metric_columns(df_image_summary)
    if metric not in {"Zoom", "Image_width_px", "Image_height_px", "Pixel_size_nm", "Image_area_nm2"}
]
generate_superplots_for_metrics(
    data=df_image_summary,
    metrics=metrics,
    save_dir=IMAGE_SUMMARY_FIGURES_DIR,
    x_order=image_summary_muscle_compartment_order,
    hue_order=image_summary_conditions,
    unit_dict=units,
    summary_df=bayesian_summary_df,
    annotation_mode=superplot_annotation_mode,
)

# %% Generate image-summary muscle-contrast superplots
df_image_summary_muscle = prepare_muscle_contrast_dataframe(data=df_image_summary)
image_summary_muscle_order = [
    value
    for value in MUSCLE_ORDER
    if value in set(df_image_summary_muscle["Muscle"].dropna().astype(str))
]
image_summary_remaining_muscles = sorted(
    set(df_image_summary_muscle["Muscle"].dropna().astype(str)) - set(image_summary_muscle_order)
)
image_summary_muscle_order.extend(image_summary_remaining_muscles)
if len(image_summary_muscle_order) != 2:
    raise ValueError(
        f"Expected exactly 2 image-summary muscle levels for muscle analyses, found "
        f"{len(image_summary_muscle_order)} ({image_summary_muscle_order})."
    )
image_summary_genotype_compartment_order = [
    build_genotype_compartment_label(condition=condition, compartment=compartment)
    for condition in image_summary_conditions
    for compartment in COMPARTMENT_ORDER
    if (
        (df_image_summary_muscle["Condition"].astype(str) == condition)
        & (df_image_summary_muscle["Compartment"].astype(str) == compartment)
    ).any()
]
muscle_superplot_annotation_mode = load_superplot_annotation_mode(
    config_yaml=BAYESIAN_MUSCLE_IMAGE_SUMMARY_CONFIG_YAML
)
bayesian_muscle_summary_df = load_bayesian_superplot_annotations(
    summary_csv=BAYESIAN_IMAGE_SUMMARY_MUSCLE_SUMMARY_CSV
)
generate_superplots_for_metrics(
    data=df_image_summary_muscle,
    metrics=metrics,
    save_dir=IMAGE_SUMMARY_FIGURES_DIR,
    x_order=image_summary_genotype_compartment_order,
    hue_order=image_summary_muscle_order,
    unit_dict=units,
    summary_df=bayesian_muscle_summary_df,
    annotation_mode=muscle_superplot_annotation_mode,
    x_column="Genotype_Compartment",
    hue_column="Muscle",
    output_dir_suffix="_muscle",
    x_label_builder=build_genotype_compartment_label,
)


# %% Compute Mann-Whitney statistics by image-summary metric, muscle, and compartment
results: list[dict[str, object]] = []
for metric in metrics:
    for muscle in MUSCLE_ORDER:
        for compartment in COMPARTMENT_ORDER:
            group_slice = df_image_summary[
                (df_image_summary["Muscle"] == muscle)
                & (df_image_summary["Compartment"] == compartment)
            ].copy()
            if group_slice.empty:
                continue
            values_a = pd.to_numeric(
                group_slice.loc[
                    group_slice["Condition"].astype(str) == image_summary_conditions[0],
                    metric,
                ],
                errors="coerce",
            ).dropna()
            values_b = pd.to_numeric(
                group_slice.loc[
                    group_slice["Condition"].astype(str) == image_summary_conditions[1],
                    metric,
                ],
                errors="coerce",
            ).dropna()
            if values_a.empty or values_b.empty:
                continue
            statistic, p_value = mannwhitneyu(values_a, values_b, alternative="two-sided")
            results.append(
                {
                    "Measurement": metric,
                    "Muscle": muscle,
                    "Compartment": compartment,
                    "Condition_A": image_summary_conditions[0],
                    "Condition_B": image_summary_conditions[1],
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

results_df = pd.DataFrame(results).sort_values(["Measurement", "Muscle", "Compartment"])
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
results_df.to_csv(STATISTICS_CSV, index=False)
print(f"Saved DMD_1X image-summary statistics to {STATISTICS_CSV}")


# %% Compute Mann-Whitney muscle statistics by image-summary metric, genotype, and compartment
muscle_results: list[dict[str, object]] = []
for metric in metrics:
    for condition in image_summary_conditions:
        for compartment in COMPARTMENT_ORDER:
            group_slice = df_image_summary_muscle[
                (df_image_summary_muscle["Condition"].astype(str) == condition)
                & (df_image_summary_muscle["Compartment"].astype(str) == compartment)
            ].copy()
            if group_slice.empty:
                continue
            values_a = pd.to_numeric(
                group_slice.loc[
                    group_slice["Muscle"].astype(str) == image_summary_muscle_order[0],
                    metric,
                ],
                errors="coerce",
            ).dropna()
            values_b = pd.to_numeric(
                group_slice.loc[
                    group_slice["Muscle"].astype(str) == image_summary_muscle_order[1],
                    metric,
                ],
                errors="coerce",
            ).dropna()
            if values_a.empty or values_b.empty:
                continue
            statistic, p_value = mannwhitneyu(values_a, values_b, alternative="two-sided")
            muscle_results.append(
                {
                    "Measurement": metric,
                    "Genotype": condition,
                    "Compartment": compartment,
                    "Muscle_A": image_summary_muscle_order[0],
                    "Muscle_B": image_summary_muscle_order[1],
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

muscle_results_df = pd.DataFrame(muscle_results).sort_values(
    ["Measurement", "Genotype", "Compartment"]
)
muscle_results_df.to_csv(STATISTICS_MUSCLE_CSV, index=False)
print(f"Saved DMD_1X muscle image-summary statistics to {STATISTICS_MUSCLE_CSV}")
