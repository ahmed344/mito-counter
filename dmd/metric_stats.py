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
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator
import yaml

sns.set_style("whitegrid")
plt.switch_backend("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dmd import stats_utils as dmd_stats_utils

INPUT_CSV = REPO_ROOT / "data" / "DMD" / "results" / "measurements_cleaned.csv"
IMAGE_SUMMARY_INPUT_CSVS = [
    REPO_ROOT / "data" / "DMD" / "results" / "measurments_cleaned_ss_summary.csv",
    REPO_ROOT / "data" / "DMD" / "results" / "measurments_cleaned_imf_summary.csv",
]
RESULTS_DIR = REPO_ROOT / "data" / "DMD" / "results"
FIGURES_DIR = RESULTS_DIR / "figures_cells"
IMAGE_SUMMARY_FIGURES_DIR = FIGURES_DIR / "image_summary"
STATISTICS_CSV = RESULTS_DIR / "statistics_cells.csv"
BAYESIAN_SUMMARY_CSV = RESULTS_DIR / "hierarchical_bayes_statistics.csv"
BAYESIAN_CONFIG_YAML = REPO_ROOT / "dmd" / "hierarchical_bayes_config.yaml"
BAYESIAN_IMAGE_SUMMARY_CSV = RESULTS_DIR / "hierarchical_bayes_statistics_cell_summary.csv"
BAYESIAN_IMAGE_SUMMARY_CONFIG_YAML = (
    REPO_ROOT / "dmd" / "hierarchical_bayes_cell_summary_config.yaml"
)
SS_LABEL = "Sub-sarcolemmal (SS)"
IMF_LABEL = "Intermyofibrillar (IMF)"
COMPARTMENT_ORDER = [SS_LABEL, IMF_LABEL]
MUSCLE_ORDER = ["Extraocular Muscle", "Tibialis Anterior"]
EXCLUDED_MEASUREMENTS = {
    "Connected_parts",
    "Distance_to_cell_membrane",
    "Image_Region",
    "Cell_id",
}
BAYES_MEAN_ANNOTATION_COLOR = "purple"
BAYES_MEDIAN_ANNOTATION_COLOR = "tab:blue"
BAYES_FACTOR_ANNOTATION_MODE = "bayes_factor"
EFFECT_SUMMARY_ANNOTATION_MODE = "effect_summary"


# %% Helper functions
def sort_conditions(values: list[str]) -> list[str]:
    """Sort condition labels with Wildtype-like labels first.

    Args:
        values (list[str]): Raw condition labels.

    Returns:
        list[str]: Sorted condition labels.
    """
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
            )
            else 2,
            str(value).lower(),
        ),
    )


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


def plot_stat_boxplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    x_order: list[str],
    hue_order: list[str],
    unit_dict: dict[str, str] | None = None,
    test: str = "Mann-Whitney",
    text_format: str = "star",
    save_dir: Path | None = None,
) -> None:
    """Plot a grouped boxplot with statistical annotations.

    Args:
        data (pd.DataFrame): Source dataframe containing plotting columns.
        x (str): Column name used on the x-axis.
        y (str): Column name used as the numeric response variable.
        hue (str): Column name used for grouping within each x category.
        x_order (list[str]): Explicit x-axis category order.
        hue_order (list[str]): Explicit hue category order.
        unit_dict (dict[str, str] | None): Optional mapping from metric name to display unit.
        test (str): Statistical test name consumed by ``statannotations``.
        text_format (str): Annotation style passed to ``statannotations``.
        save_dir (Path | None): Optional output directory where a PNG is saved.

    Returns:
        None: The function renders and optionally saves the plot.
    """
    plot_data = data.copy()
    plot_data[y] = pd.to_numeric(plot_data[y], errors="coerce")
    plot_data = plot_data.dropna(subset=[x, y, hue])
    if plot_data.empty:
        print(f"Skipping {y}: no numeric data available after coercion.")
        return

    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(
        data=plot_data,
        x=x,
        y=y,
        hue=hue,
        order=x_order,
        hue_order=hue_order,
        linewidth=1.5,
        palette=["tab:blue", "tab:orange"],
        showfliers=False,
        gap=0.2,
        width=0.6,
    )

    box_pairs = []
    hue_combinations = list(combinations(hue_order, 2))
    for x_value in x_order:
        for hue_pair in hue_combinations:
            box_pairs.append(((x_value, hue_pair[0]), (x_value, hue_pair[1])))

    if box_pairs:
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

    unit_label = ""
    if unit_dict and y in unit_dict and unit_dict[y]:
        formatted_unit = unit_dict[y].replace("um", r"$\mu m$").replace("^2", r"$^2$")
        unit_label = f" ({formatted_unit})"

    sns.despine()
    plt.ylabel(f"{y}{unit_label}", fontsize=13)
    plt.xlabel(x, fontsize=13)
    plt.title(f"{y} by {x} and {hue}", fontsize=14, pad=20)
    plt.legend(title=hue)
    plt.xticks(rotation=0, ha="center", fontsize=11)
    plt.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{y}_by_{x}_and_{hue}.png", dpi=300)
    plt.close()


def get_numeric_measurement_columns(data: pd.DataFrame) -> list[str]:
    """Identify numeric measurement columns while excluding metadata.

    Args:
        data (pd.DataFrame): Measurement dataframe.

    Returns:
        list[str]: Numeric measurement column names.
    """
    excluded_columns = {
        "Condition",
        "Muscle",
        "Block",
        "image",
        "Id",
        "Centroid",
        "Compartment",
        "Muscle_Compartment",
    }
    numeric_columns: list[str] = []
    for column in data.columns:
        if column in excluded_columns or column in EXCLUDED_MEASUREMENTS:
            continue
        coerced = pd.to_numeric(data[column], errors="coerce")
        if coerced.notna().any():
            data[column] = coerced
            numeric_columns.append(column)
    return numeric_columns


def load_image_summary_dataframe(input_csvs: list[Path]) -> pd.DataFrame:
    """Load and combine DMD image-summary CSV files into one dataframe.

    Args:
        input_csvs (list[Path]): CSV paths containing per-image summary measurements.

    Returns:
        pd.DataFrame: Concatenated image-summary table with categorical group columns.
    """

    available_paths = [path for path in input_csvs if path.exists()]
    if not available_paths:
        return pd.DataFrame()
    combined = pd.concat((pd.read_csv(path) for path in available_paths), ignore_index=True)
    combined["Condition"] = combined["Condition"].astype("category")
    combined["Muscle"] = combined["Muscle"].astype("category")
    combined = combined.dropna(subset=["Compartment"]).copy()
    combined["Compartment"] = combined["Compartment"].astype("category")
    combined["Muscle_Compartment"] = combined.apply(
        lambda row: build_muscle_compartment_label(row["Muscle"], row["Compartment"]),
        axis=1,
    )
    return combined


def load_superplot_annotation_mode(config_yaml: Path) -> str:
    """Load superplot annotation mode from hierarchical Bayesian YAML config.

    Args:
        config_yaml (Path): Path to DMD hierarchical Bayesian config YAML.

    Returns:
        str: Annotation mode; defaults to ``bayes_factor`` when absent/invalid.
    """

    if not config_yaml.exists():
        return BAYES_FACTOR_ANNOTATION_MODE
    with config_yaml.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = dict(config.get("runtime", {})) if isinstance(config, dict) else {}
    annotation_mode = str(
        runtime.get("superplot_annotation_mode", BAYES_FACTOR_ANNOTATION_MODE)
    ).strip()
    if annotation_mode not in {
        BAYES_FACTOR_ANNOTATION_MODE,
        EFFECT_SUMMARY_ANNOTATION_MODE,
    }:
        return BAYES_FACTOR_ANNOTATION_MODE
    return annotation_mode


def load_bayesian_superplot_annotations(summary_csv: Path) -> pd.DataFrame:
    """Load Bayesian summary rows used for superplot bracket annotations.

    Args:
        summary_csv (Path): CSV containing hierarchical Bayesian fit summaries.

    Returns:
        pd.DataFrame: Summary dataframe; empty when the file is unavailable/incomplete.
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
        str: Formatted text like ``estimate [hdi_low, hdi_high] pd%`` or empty string.
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


def bayesian_annotations_for_metric(
    summary_df: pd.DataFrame,
    metric: str,
    annotation_mode: str,
) -> list[dict[str, str]]:
    """Build DMD muscle-compartment annotation records for one metric.

    Args:
        summary_df (pd.DataFrame): Bayesian summary dataframe.
        metric (str): Metric currently being plotted.
        annotation_mode (str): Either ``bayes_factor`` or ``effect_summary``.

    Returns:
        list[dict[str, str]]: Records consumed by DMD superplot renderers.
    """

    if summary_df.empty:
        return []
    metric_rows = summary_df.loc[summary_df["metric"].astype(str) == str(metric)]
    annotations: list[dict[str, str]] = []
    for _, row in metric_rows.iterrows():
        if annotation_mode == EFFECT_SUMMARY_ANNOTATION_MODE:
            mean_label = format_delta_effect_annotation(row=row, label="delta_mean")
            median_label = format_delta_effect_annotation(row=row, label="delta_median")
        else:
            mean_label = str(row.get("delta_mean_bf_annotation", "")).strip()
            median_label = str(row.get("delta_median_bf_annotation", "")).strip()
            if mean_label.lower() == "nan":
                mean_label = ""
            if median_label.lower() == "nan":
                median_label = ""
        if not mean_label and not median_label:
            continue
        x_label = build_muscle_compartment_label(
            muscle=str(row["muscle"]),
            compartment=str(row["compartment"]),
        )
        annotations.append(
            {
                "x": x_label,
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


def generate_superplots_for_metrics(
    metric_specs: list[tuple[str, pd.DataFrame]],
    save_dir: Path,
    x: str,
    hue: str,
    block: str,
    x_order: list[str],
    hue_order: list[str],
    unit_dict: dict[str, str],
    summary_df: pd.DataFrame,
    annotation_mode: str,
) -> None:
    """Render and save one superviolin/superbeeswarm file per metric.

    Args:
        metric_specs (list[tuple[str, pd.DataFrame]]): Ordered ``(metric, dataframe)`` pairs.
        save_dir (Path): Directory where superplot folders are written.
        x (str): X-axis column.
        hue (str): Hue column.
        block (str): Block column.
        x_order (list[str]): Explicit x-axis order.
        hue_order (list[str]): Explicit hue order.
        unit_dict (dict[str, str]): Metric unit mapping.
        summary_df (pd.DataFrame): Bayesian summary dataframe for annotations.
        annotation_mode (str): Superplot annotation mode.

    Returns:
        None: Saves per-metric superplot PNG figures.
    """
    for metric, source_df in metric_specs:
        annotations = bayesian_annotations_for_metric(
            summary_df=summary_df,
            metric=metric,
            annotation_mode=annotation_mode,
        )
        dmd_stats_utils.plot_super_violin(
            data=source_df,
            x=x,
            y=metric,
            hue=hue,
            block=block,
            unit_dict=unit_dict,
            save_dir=save_dir,
            superplot_annotations=annotations,
            x_order_override=x_order,
            hue_order_override=hue_order,
        )
        dmd_stats_utils.plot_super_beeswarm(
            data=source_df,
            x=x,
            y=metric,
            hue=hue,
            block=block,
            unit_dict=unit_dict,
            save_dir=save_dir,
            superplot_annotations=annotations,
            x_order_override=x_order,
            hue_order_override=hue_order,
        )


# %% Load and prepare input data
df = pd.read_csv(INPUT_CSV)
if "Compartment" not in df.columns:
    raise KeyError(
        "Missing required 'Compartment' column. Run dmd/build_measurements_csv.py "
        "to regenerate measurements_cleaned.csv."
    )
df["Condition"] = df["Condition"].astype("category")
df["Muscle"] = df["Muscle"].astype("category")
df = df.dropna(subset=["Compartment"]).copy()
df["Compartment"] = df["Compartment"].astype("category")
df["Muscle_Compartment"] = df.apply(
    lambda row: build_muscle_compartment_label(row["Muscle"], row["Compartment"]),
    axis=1,
)
df.head()


# %% Define plotting and testing orders
conditions = sort_conditions(list(df["Condition"].dropna().unique()))
if len(conditions) != 2:
    raise ValueError(f"Expected exactly 2 conditions, found {len(conditions)}: {conditions}")

muscle_order = [value for value in MUSCLE_ORDER if value in set(df["Muscle"].dropna().unique())]
remaining_muscles = sorted(
    set(df["Muscle"].dropna().unique()) - set(muscle_order)
)
muscle_order.extend(remaining_muscles)
muscle_compartment_order = [
    build_muscle_compartment_label(muscle, compartment)
    for muscle in muscle_order
    for compartment in COMPARTMENT_ORDER
    if ((df["Muscle"] == muscle) & (df["Compartment"] == compartment)).any()
]


# %% Select metrics and inspect distributions
metrics = get_numeric_measurement_columns(df)
df[metrics].isin([0]).sum()


# %% Plot histograms for all numeric metrics
hist_axes = df[metrics].hist(
    figsize=(25, 10),
    bins=70,
    layout=(2, int(pd.Series(metrics).shape[0] / 2 + 0.999)),
)
plt.tight_layout()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "histograms_cells.png", dpi=300)
plt.close()


# %% Build count table per image and compartment
df_counts = (
    df.groupby(
        ["Condition", "Muscle", "Compartment", "Block", "image"],
        observed=True,
        dropna=False,
    )
    .size()
    .reset_index(name="Count")
)
df_counts["Muscle_Compartment"] = df_counts.apply(
    lambda row: build_muscle_compartment_label(row["Muscle"], row["Compartment"]),
    axis=1,
)
df_counts


# %% Explore count variation across blocks and compartments
if df_counts["Block"].notna().any():
    plt.figure(figsize=(20, 6))
    sns.boxplot(
        data=df_counts,
        x="Block",
        y="Count",
        hue="Muscle_Compartment",
        showfliers=False,
        gap=0.1,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "counts_by_block_and_compartment.png", dpi=300)
    plt.close()
else:
    print("Skipping counts_by_block_and_compartment: Block column is empty.")


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
    "Count": "",
    "Density": "count/image",
    "Area_sum": "nm^2",
    "Corrected_area_sum": "nm^2",
    "Minimum_Feret_Diameter_sum": "nm",
    "Minor_axis_length_sum": "nm",
    "Area_mean": "nm^2",
    "Corrected_area_mean": "nm^2",
    "Minimum_Feret_Diameter_mean": "nm",
    "Major_axis_length_mean": "nm",
    "Minor_axis_length_mean": "nm",
    "Elongation_mean": "",
    "Circularity_mean": "",
    "Solidity_mean": "",
    "NND_center_mean": "nm",
    "3NND_center_mean": "nm",
    "5NND_center_mean": "nm",
    "Voronoi_Cell_Area_center_mean": "nm^2",
    "Voronoi_Cell_Area_center_cv": "",
    "Ripley_L_integral": "nm^2",
    "Pair_Correlation_integral": "",
}


# %% Generate annotated comparison boxplots
plot_stat_boxplot(
    data=df_counts,
    x="Muscle_Compartment",
    y="Count",
    hue="Condition",
    x_order=muscle_compartment_order,
    hue_order=conditions,
    unit_dict=units,
    save_dir=FIGURES_DIR,
)

for measurement in metrics:
    plot_stat_boxplot(
        data=df,
        x="Muscle_Compartment",
        y=measurement,
        hue="Condition",
        x_order=muscle_compartment_order,
        hue_order=conditions,
        unit_dict=units,
        save_dir=FIGURES_DIR,
    )


# %% Generate superviolin and superbeeswarm comparison plots (one file per metric)
superplot_annotation_mode = load_superplot_annotation_mode(
    config_yaml=BAYESIAN_CONFIG_YAML,
)
bayesian_summary_df = load_bayesian_superplot_annotations(
    summary_csv=BAYESIAN_SUMMARY_CSV,
)
metric_specs: list[tuple[str, pd.DataFrame]] = [("Count", df_counts)]
metric_specs.extend((measurement, df) for measurement in metrics)
generate_superplots_for_metrics(
    metric_specs=metric_specs,
    save_dir=FIGURES_DIR,
    x="Muscle_Compartment",
    hue="Condition",
    block="Block",
    x_order=muscle_compartment_order,
    hue_order=conditions,
    unit_dict=units,
    summary_df=bayesian_summary_df,
    annotation_mode=superplot_annotation_mode,
)


# %% Generate image-summary superviolin and superbeeswarm comparison plots
df_image_summary = load_image_summary_dataframe(input_csvs=IMAGE_SUMMARY_INPUT_CSVS)
if not df_image_summary.empty:
    image_summary_conditions = sort_conditions(
        list(df_image_summary["Condition"].dropna().unique())
    )
    if len(image_summary_conditions) == 2:
        image_summary_muscle_compartment_order = [
            build_muscle_compartment_label(muscle, compartment)
            for muscle in muscle_order
            for compartment in COMPARTMENT_ORDER
            if (
                (df_image_summary["Muscle"] == muscle)
                & (df_image_summary["Compartment"] == compartment)
            ).any()
        ]
        bayesian_image_summary_mode = load_superplot_annotation_mode(
            config_yaml=BAYESIAN_IMAGE_SUMMARY_CONFIG_YAML,
        )
        bayesian_image_summary_df = load_bayesian_superplot_annotations(
            summary_csv=BAYESIAN_IMAGE_SUMMARY_CSV,
        )
        image_summary_metrics = sorted(
            bayesian_image_summary_df["metric"].astype(str).unique().tolist()
        )
        image_summary_metric_specs = [
            (metric, df_image_summary)
            for metric in image_summary_metrics
            if metric in df_image_summary.columns
        ]
        if image_summary_metric_specs:
            generate_superplots_for_metrics(
                metric_specs=image_summary_metric_specs,
                save_dir=IMAGE_SUMMARY_FIGURES_DIR,
                x="Muscle_Compartment",
                hue="Condition",
                block="Block",
                x_order=image_summary_muscle_compartment_order,
                hue_order=image_summary_conditions,
                unit_dict=units,
                summary_df=bayesian_image_summary_df,
                annotation_mode=bayesian_image_summary_mode,
            )
    else:
        print(
            "Skipping image-summary superplots: expected exactly 2 conditions, found "
            f"{len(image_summary_conditions)} ({image_summary_conditions})."
        )
else:
    print("Skipping image-summary superplots: no image-summary CSVs were found.")


# %% Run WT-vs-DMD statistical tests for each muscle-compartment group
results: list[dict[str, object]] = []
for muscle in muscle_order:
    for compartment in COMPARTMENT_ORDER:
        group_slice = df[
            (df["Muscle"] == muscle) & (df["Compartment"] == compartment)
        ]
        if group_slice.empty:
            continue

        group_a = group_slice[group_slice["Condition"] == conditions[0]]
        group_b = group_slice[group_slice["Condition"] == conditions[1]]
        if group_a.empty or group_b.empty:
            continue

        for measurement in metrics:
            values_a = pd.to_numeric(group_a[measurement], errors="coerce").dropna()
            values_b = pd.to_numeric(group_b[measurement], errors="coerce").dropna()
            if len(values_a) == 0 or len(values_b) == 0:
                continue

            u_stat, p_value = mannwhitneyu(
                values_a,
                values_b,
                alternative="two-sided",
            )
            results.append(
                {
                    "Measurement": measurement,
                    "Muscle": muscle,
                    "Compartment": compartment,
                    "Condition_A": conditions[0],
                    "Condition_B": conditions[1],
                    "N_A": len(values_a),
                    "N_B": len(values_b),
                    "U": u_stat,
                    "p_value": p_value,
                }
            )

for muscle in muscle_order:
    for compartment in COMPARTMENT_ORDER:
        counts_slice = df_counts[
            (df_counts["Muscle"] == muscle) & (df_counts["Compartment"] == compartment)
        ]
        if counts_slice.empty:
            continue

        count_group_a = counts_slice[counts_slice["Condition"] == conditions[0]]["Count"].dropna()
        count_group_b = counts_slice[counts_slice["Condition"] == conditions[1]]["Count"].dropna()
        if len(count_group_a) == 0 or len(count_group_b) == 0:
            continue

        count_u_stat, count_p_value = mannwhitneyu(
            count_group_a,
            count_group_b,
            alternative="two-sided",
        )
        results.append(
            {
                "Measurement": "Count",
                "Muscle": muscle,
                "Compartment": compartment,
                "Condition_A": conditions[0],
                "Condition_B": conditions[1],
                "N_A": len(count_group_a),
                "N_B": len(count_group_b),
                "U": count_u_stat,
                "p_value": count_p_value,
            }
        )

results_df = pd.DataFrame(results).sort_values(["Measurement", "Muscle", "Compartment"])
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
results_df.to_csv(STATISTICS_CSV, index=False)
results_df
