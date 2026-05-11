# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% Imports and plotting configuration
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import mannwhitneyu

from stats_utils import filter_metric_region, maybe_show_current_figure, plot_metric_variants

# Set the style for the plots
sns.set_style("whitegrid")
# %% Load and prepare input data
# Read per-object morphological measurements exported by the pipeline.
_DATA_RESULTS = _REPO_ROOT / "data" / "Calpaine_3" / "results"
df = pd.read_csv(_DATA_RESULTS / "measurments_cleaned.csv")
df_image_summary = pd.read_csv(_DATA_RESULTS / "measurments_cleaned_image_summary.csv")
excluded_measurements = {"Connected_parts", "Image_Region"}

# Treat grouping columns as categorical to keep consistent ordering/group handling.
df["Condition"] = df["Condition"].astype("category")
df["Muscle"] = df["Muscle"].astype("category")
df_image_summary["Condition"] = df_image_summary["Condition"].astype("category")
df_image_summary["Muscle"] = df_image_summary["Muscle"].astype("category")

# Display the first few rows of the dataframe
df.head()

# %% Quick distribution checks
# Metric columns start after metadata columns.
metrics = [
    column
    for column in df.columns[6:]
    if column not in excluded_measurements
]
metric_values_df = pd.DataFrame(
    {
        metric: pd.to_numeric(
            filter_metric_region(data=df, metric_name=metric)[metric],
            errors="coerce",
        ).reset_index(drop=True)
        for metric in metrics
    }
)

# Summarize zero values in each column.
metric_values_df.isin([0]).sum()

# %% Plot histograms for all numeric metrics
# This gives a first-pass view of skew/outliers for each measurement.
metric_values_df.hist(figsize=(25, 8), bins=70, layout=(2, np.ceil(len(metrics) / 2).astype(int)))
plt.savefig(_DATA_RESULTS / "figures" / "histograms.png", dpi=900)
maybe_show_current_figure()
# %% Human-readable units for y-axis labels

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
    "Voronoi_Cell_Area_center_cv": "",
    "Ripley_L_integral": "nm^2",
    "Pair_Correlation_integral": "",
    "Corrected_area_sum": "nm^2",
    "Minimum_Feret_Diameter_sum": "nm",
    "Minimum_Feret_Diameter_mean": "nm",
    "Elongation_mean": "",
    "Circularity_mean": "",
    "Solidity_mean": "",
    "3NND_center_mean": "nm",
    "Voronoi_Cell_Area_center_mean": "nm^2",
}
# %% Generate annotated comparison boxplots
# All plots are saved to the same figures directory.
save_dir = _DATA_RESULTS / "figures"
image_summary_save_dir = save_dir / "image_summary"
bayesian_summary_csv = _DATA_RESULTS / "hierarchical_bayes_statistics.csv"
bayesian_image_summary_csv = _DATA_RESULTS / "hierarchical_bayes_statistics_image_summary.csv"
_c3_dir = Path(__file__).resolve().parent
bayesian_config_yaml = _c3_dir / "hierarchical_bayes_config.yaml"
bayesian_image_summary_config_yaml = _c3_dir / "hierarchical_bayes_image_summary_config.yaml"
BAYES_MEAN_ANNOTATION_COLOR = "purple"
BAYES_MEDIAN_ANNOTATION_COLOR = "tab:blue"
BAYES_FACTOR_ANNOTATION_MODE = "bayes_factor"
EFFECT_SUMMARY_ANNOTATION_MODE = "effect_summary"
image_summary_metrics = [
    "Density",
    "Voronoi_Cell_Area_center_cv",
    "Ripley_L_integral",
    "Pair_Correlation_integral",
    "Corrected_area_sum",
    "Minimum_Feret_Diameter_sum",
    "Minimum_Feret_Diameter_mean",
    "Elongation_mean",
    "Circularity_mean",
    "Solidity_mean",
    "3NND_center_mean",
    "Voronoi_Cell_Area_center_mean",
]


def load_superplot_annotation_mode(config_yaml):
    """Load the configured superplot annotation mode from a Bayesian YAML file.

    Args:
        config_yaml (Path): Path to a hierarchical Bayesian YAML configuration file.

    Returns:
        str: Annotation mode, defaulting to ``bayes_factor`` when unavailable.
    """

    if not config_yaml.exists():
        return BAYES_FACTOR_ANNOTATION_MODE
    with config_yaml.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = dict(config.get("runtime", {})) if isinstance(config, dict) else {}
    annotation_mode = str(runtime.get("superplot_annotation_mode", BAYES_FACTOR_ANNOTATION_MODE)).strip()
    if annotation_mode not in {BAYES_FACTOR_ANNOTATION_MODE, EFFECT_SUMMARY_ANNOTATION_MODE}:
        return BAYES_FACTOR_ANNOTATION_MODE
    return annotation_mode


def load_bayesian_superplot_annotations(summary_csv):
    """Load compact Bayesian BF labels for superplot WT-vs-KO annotations.

    Args:
        summary_csv (Path): CSV path containing hierarchical Bayesian summary rows.

    Returns:
        pd.DataFrame: Summary rows with the columns needed to annotate superplots.
    """

    required_columns = {
        "metric",
        "muscle",
        "wt_label",
        "ko_label",
    }
    if not summary_csv.exists():
        return pd.DataFrame(columns=sorted(required_columns))
    summary_df = pd.read_csv(summary_csv)
    if not required_columns.issubset(summary_df.columns):
        missing_columns = sorted(required_columns - set(summary_df.columns))
        print(f"Skipping Bayesian BF superplot annotations: missing columns {missing_columns}.")
        return pd.DataFrame(columns=sorted(required_columns))
    return summary_df


def format_delta_effect_annotation(row, label):
    """Format a Bayesian effect estimate, HDI, and probability of direction.

    Args:
        row (pd.Series): Bayesian summary row containing delta columns.
        label (str): Delta column prefix, such as ``delta_mean`` or ``delta_median``.

    Returns:
        str: Compact effect annotation text, or an empty string when unavailable.
    """

    try:
        estimate = float(row[label])
        hdi_low = float(row[f"{label}_hdi_low"])
        hdi_high = float(row[f"{label}_hdi_high"])
        pd_value = float(row[f"{label}_pd"])
    except (KeyError, TypeError, ValueError):
        return ""
    if not all(np.isfinite(value) for value in (estimate, hdi_low, hdi_high, pd_value)):
        return ""
    return f"{estimate:.4g} [{hdi_low:.4g}, {hdi_high:.4g}] {pd_value:.1f}%"


def bayesian_annotations_for_metric(summary_df, metric, annotation_mode=BAYES_FACTOR_ANNOTATION_MODE):
    """Build per-muscle BF annotation records for one metric.

    Args:
        summary_df (pd.DataFrame): Bayesian summary table loaded from disk.
        metric (str): Metric name currently being plotted.
        annotation_mode (str): Annotation system to use for superplots.

    Returns:
        list[dict[str, str]]: Annotation records consumed by the superplot helpers.
    """

    if summary_df.empty:
        return []
    metric_rows = summary_df.loc[summary_df["metric"].astype(str) == str(metric)]
    annotations = []
    for _, row in metric_rows.iterrows():
        if annotation_mode == EFFECT_SUMMARY_ANNOTATION_MODE:
            mean_label = format_delta_effect_annotation(row=row, label="delta_mean")
            median_label = format_delta_effect_annotation(row=row, label="delta_median")
        else:
            mean_label = str(row.get("delta_mean_bf_annotation", "")).strip()
            if mean_label.lower() == "nan":
                mean_label = ""
            median_label = str(row.get("delta_median_bf_annotation", "")).strip()
            if median_label.lower() == "nan":
                median_label = ""
        if not mean_label and not median_label:
            continue
        annotations.append(
            {
                "x": str(row["muscle"]),
                "hue_start": str(row["wt_label"]),
                "hue_end": str(row["ko_label"]),
                "mean_label": (
                    mean_label
                    if annotation_mode == EFFECT_SUMMARY_ANNOTATION_MODE
                    else f"mean {mean_label}"
                    if mean_label
                    else ""
                ),
                "mean_color": BAYES_MEAN_ANNOTATION_COLOR,
                "median_label": (
                    median_label
                    if annotation_mode == EFFECT_SUMMARY_ANNOTATION_MODE
                    else f"median {median_label}"
                    if median_label
                    else ""
                ),
                "median_color": BAYES_MEDIAN_ANNOTATION_COLOR,
            }
        )
    return annotations


bayesian_summary_df = load_bayesian_superplot_annotations(summary_csv=bayesian_summary_csv)
bayesian_image_summary_df = load_bayesian_superplot_annotations(
    summary_csv=bayesian_image_summary_csv
)
bayesian_annotation_mode = load_superplot_annotation_mode(config_yaml=bayesian_config_yaml)
bayesian_image_summary_annotation_mode = load_superplot_annotation_mode(
    config_yaml=bayesian_image_summary_config_yaml
)

# Plot Morphological measurments with Units
for measurment in metrics:
    if measurment in df.columns:
        plot_df = filter_metric_region(data=df, metric_name=measurment)
        plot_metric_variants(
            data=plot_df,
            x='Muscle',
            y=measurment,
            hue='Condition',
            block='Block',
            unit_dict=units,
            save_dir=save_dir,
            superplot_annotations=bayesian_annotations_for_metric(
                summary_df=bayesian_summary_df,
                metric=measurment,
                annotation_mode=bayesian_annotation_mode,
            ),
        )

# Plot image-summary metrics with one observation per image.
for measurment in image_summary_metrics:
    if measurment in df_image_summary.columns:
        plot_metric_variants(
            data=df_image_summary,
            x='Muscle',
            y=measurment,
            hue='Condition',
            block='Block',
            unit_dict=units,
            save_dir=image_summary_save_dir,
            superplot_annotations=bayesian_annotations_for_metric(
                summary_df=bayesian_image_summary_df,
                metric=measurment,
                annotation_mode=bayesian_image_summary_annotation_mode,
            ),
        )

# %% Run statistical tests and export results
# For each muscle, compare the two conditions for every measurement.

# Determine the two conditions to compare
if isinstance(df["Condition"].dtype, pd.CategoricalDtype):
    condition_values = list(df["Condition"].cat.categories)
else:
    condition_values = list(df["Condition"].dropna().unique())
conditions = sorted(
    condition_values,
    key=lambda v: (0 if ("wildtype" in str(v).lower() or str(v).strip().lower() == "wt" or str(v).strip().lower().endswith("_wt"))
                   else 1 if ("knockout" in str(v).lower() or str(v).strip().lower() == "ko" or str(v).strip().lower().endswith("_ko"))
                   else 2, str(v).lower())
)

if len(conditions) != 2:
    raise ValueError(f"Expected exactly 2 conditions, found {len(conditions)}: {conditions}")

# Use the same measurment set as the plotting loop, filtering to numeric columns
measurment_cols = [
    c for c in df.columns[5:]
    if pd.api.types.is_numeric_dtype(df[c]) and c not in excluded_measurements
]

results = []
for muscle in sorted(df["Muscle"].dropna().unique()):
    # Restrict testing to one muscle at a time.
    df_muscle = df[df["Muscle"] == muscle]
    group_a = df_muscle[df_muscle["Condition"] == conditions[0]]
    group_b = df_muscle[df_muscle["Condition"] == conditions[1]]

    for measurment in measurment_cols:
        metric_group_a = filter_metric_region(data=group_a, metric_name=measurment)
        metric_group_b = filter_metric_region(data=group_b, metric_name=measurment)
        values_a = metric_group_a[measurment].dropna()
        values_b = metric_group_b[measurment].dropna()

        if len(values_a) == 0 or len(values_b) == 0:
            continue

        # Two-sided non-parametric comparison between conditions.
        u_stat, p_value = mannwhitneyu(values_a, values_b, alternative="two-sided")
        results.append({
            "Measurment": measurment,
            "Muscle": muscle,
            "Condition_A": conditions[0],
            "Condition_B": conditions[1],
            "N_A": len(values_a),
            "N_B": len(values_b),
            "U": u_stat,
            "p_value": p_value,
        })

# Save one combined table containing measurement test results.
results_df = pd.DataFrame(results).sort_values(["Measurment", "Muscle"])
results_df.to_csv(_DATA_RESULTS / "statistics.csv", index=False)
results_df
