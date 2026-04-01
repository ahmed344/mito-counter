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

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator

sns.set_style("whitegrid")
plt.switch_backend("Agg")

INPUT_CSV = Path("/workspaces/mito-counter/data/DMD/results/measurements_cells_cleaned.csv")
RESULTS_DIR = Path("/workspaces/mito-counter/data/DMD/results")
FIGURES_DIR = RESULTS_DIR / "figures_cells"
STATISTICS_CSV = RESULTS_DIR / "statistics_cells.csv"
SS_THRESHOLD_UM = 1.0
SS_LABEL = "Sub-sarcolemmal (SS)"
IMF_LABEL = "Intermyofibrillar (IMF)"
COMPARTMENT_ORDER = [SS_LABEL, IMF_LABEL]
MUSCLE_ORDER = ["Extraocular Muscle", "Tibialis Anterior"]
EXCLUDED_MEASUREMENTS = {"Connected_parts", "Distance_to_cell_membrane"}


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


def make_compartment(distance_to_membrane_um: float) -> str:
    """Classify a mitochondrion as SS or IMF.

    Args:
        distance_to_membrane_um (float): Distance from the instance to the cell membrane in micrometers.

    Returns:
        str: ``Sub-sarcolemmal (SS)`` when the distance is at most 1.0 um, otherwise
        ``Intermyofibrillar (IMF)``.
    """
    if distance_to_membrane_um <= SS_THRESHOLD_UM:
        return SS_LABEL
    return IMF_LABEL


def build_muscle_compartment_label(muscle: str, compartment: str) -> str:
    """Create a compact combined plotting label for muscle and compartment.

    Args:
        muscle (str): Muscle label.
        compartment (str): Compartment label.

    Returns:
        str: Combined muscle-compartment label.
    """
    short_compartment = "SS" if compartment == SS_LABEL else "IMF"
    return f"{muscle} | {short_compartment}"


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
    plt.ylabel(f"{y}{unit_label}", fontsize=12)
    plt.xlabel(x, fontsize=12)
    plt.title(f"{y} by {x} and {hue}", fontsize=14, pad=20)
    plt.legend(title=hue)
    plt.xticks(rotation=15, ha="right")
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


# %% Load and prepare input data
df = pd.read_csv(INPUT_CSV)
df["Condition"] = df["Condition"].astype("category")
df["Muscle"] = df["Muscle"].astype("category")
df["Distance_to_cell_membrane"] = pd.to_numeric(
    df["Distance_to_cell_membrane"], errors="coerce"
)
df = df.dropna(subset=["Distance_to_cell_membrane"]).copy()
df["Compartment"] = df["Distance_to_cell_membrane"].apply(make_compartment)
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
    "Area": "um^2",
    "Corrected_area": "um^2",
    "Major_axis_length": "um",
    "Minor_axis_length": "um",
    "Minimum_Feret_Diameter": "um",
    "Elongation": "",
    "Circularity": "",
    "Solidity": "",
    "NND": "um",
    "Count": "",
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
