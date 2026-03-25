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
import numpy as np
import pandas as pd
from itertools import combinations
from statannotations.Annotator import Annotator
from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# Set the style for the plots
sns.set_style("whitegrid")

# %% Load and prepare input data
# Read per-object morphological measurements exported by the pipeline.
df = pd.read_csv("/workspaces/mito-counter/data/Calpaine_3/results/measurments_cleaned.csv")

# Treat grouping columns as categorical to keep consistent ordering/group handling.
df["Condition"] = df["Condition"].astype("category")
df["Muscle"] = df["Muscle"].astype("category")

# Display the first few rows of the dataframe
df.head()

# %% Quick distribution checks
# Metric columns start after metadata columns.
metrics = df.columns[6:-1]

# Count the zero values in each column
df[metrics].isin([0]).sum()

# %% Plot histograms for all numeric metrics
# This gives a first-pass view of skew/outliers for each measurement.
df[metrics].hist(figsize=(25, 8), bins=70, layout=(2, np.ceil(metrics.shape[0] / 2).astype(int)))
plt.savefig("/workspaces/mito-counter/data/Calpaine_3/results/figures/histograms.png", dpi=900)
plt.show()
# Count the zero values in each column

# %% Plot helper with statistical annotations
def plot_stat_boxplot(data, x, y, hue, unit_dict=None, test='Mann-Whitney', text_format='star', save_dir=None):
    """Plot a grouped boxplot with statistical annotations.

    Args:
        data (pd.DataFrame): Source dataframe containing plotting columns.
        x (str): Column name used on the x-axis.
        y (str): Column name used as the numeric response variable.
        hue (str): Column name used for grouping within each x category.
        unit_dict (dict[str, str] | None): Optional mapping from metric name to display unit.
        test (str): Statistical test name consumed by ``statannotations``.
        text_format (str): Annotation style passed to ``statannotations``.
        save_dir (str | None): Optional output directory where a PNG is saved.

    Returns:
        None: The function renders/saves the plot and does not return a value.
    """
    plot_data = data.copy()
    plot_data[y] = pd.to_numeric(plot_data[y], errors="coerce")
    plot_data = plot_data.dropna(subset=[x, y, hue])
    if plot_data.empty:
        print(f"Skipping {y}: no numeric data available after coercion.")
        return None

    plt.figure(figsize=(8, 6))
    
    x_order = sorted(plot_data[x].unique())
    if hue == "Condition":
        hue_order = sorted(
            plot_data[hue].dropna().unique(),
            key=lambda v: (0 if ("wildtype" in str(v).lower() or str(v).strip().lower() == "wt" or str(v).strip().lower().endswith("_wt"))
                           else 1 if ("knockout" in str(v).lower() or str(v).strip().lower() == "ko" or str(v).strip().lower().endswith("_ko"))
                           else 2, str(v).lower())
        )
    else:
        hue_order = sorted(plot_data[hue].unique())
    
    # 1. Create Boxplot
    ax = sns.boxplot(
        data=plot_data, x=x, y=y, hue=hue,
        order=x_order, hue_order=hue_order,
        linewidth=1.5, 
        palette=["tab:blue", "tab:orange"],
        showfliers=False,
        gap=0.2,
        width=0.6
    )
    
    # 2. Add Stats
    box_pairs = []
    hue_combinations = list(combinations(hue_order, 2))
    for x_val in x_order:
        for hue_pair in hue_combinations:
            box_pairs.append(((x_val, hue_pair[0]), (x_val, hue_pair[1])))
            
    annotator = Annotator(ax, box_pairs, data=plot_data, x=x, y=y, hue=hue, 
                          order=x_order, hue_order=hue_order)
    annotator.configure(test=test, text_format=text_format, loc='inside', verbose=0)
    annotator.apply_and_annotate()
    
    # 3. Label Formatting with Units
    # Determine the unit string
    unit_label = ""
    if unit_dict and y in unit_dict:
        raw_unit = unit_dict[y]
        if raw_unit:
            # Optional: Convert "um" to Greek mu for scientific notation
            # This makes "um^2" look like actual superscript
            formatted_unit = raw_unit.replace("um", r"$\mu m$").replace("^2", r"$^2$")
            unit_label = f" ({formatted_unit})"
    
    sns.despine()
    # Apply the formatted Y-label
    plt.ylabel(f"{y}{unit_label}", fontsize=12)
    plt.xlabel(x, fontsize=12)
    
    # Title and Legend
    plt.title(f"{y} by {x} and {hue}", fontsize=14, pad=20)
    plt.legend(title=hue)
    
    plt.tight_layout()

    if save_dir:
        plt.savefig(f'{save_dir}/{y}_by_{x}_and_{hue}.png')
    plt.show()


# %% Build count table (objects per image)
# Each row in df_counts is one image-level count used for count comparisons.
df_counts = df.groupby(['Condition', 'Muscle', 'Block', 'image'], observed=True).size().reset_index(name='Count')
df_counts

# %% Sanity check the count table schema/types
df_counts.info()

# %% Explore count variation across blocks
plt.figure(figsize=(20, 6))
sns.boxplot(y="Count", x="Block", data=df_counts, hue="Muscle", gap=0.1)
plt.savefig("/workspaces/mito-counter/data/Calpaine_3/results/figures/counts_by_block.png", dpi=900)
plt.show()

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
    "NND": "um"
}

# %% Generate annotated comparison boxplots
# All plots are saved to the same figures directory.
save_dir = '/workspaces/mito-counter/data/Calpaine_3/results/figures'

# Plot Counts (No units needed, or you can add "objects" if you like)
plot_stat_boxplot(df_counts, x='Muscle', y='Count', hue='Condition', save_dir=save_dir)

# Plot Morphological measurments with Units
for measurment in metrics:
    if measurment in df.columns:
        # Reuse one plotting helper so all outputs have identical style/stat format.
        plot_stat_boxplot(
            data=df, 
            x='Muscle', 
            y=measurment, 
            hue='Condition',
            unit_dict=units,
            save_dir=save_dir
        )

# %% Run statistical tests and export results
# For each muscle, compare the two conditions for every measurement and for Count.

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
    if pd.api.types.is_numeric_dtype(df[c])
]

results = []
for muscle in sorted(df["Muscle"].dropna().unique()):
    # Restrict testing to one muscle at a time.
    df_muscle = df[df["Muscle"] == muscle]
    group_a = df_muscle[df_muscle["Condition"] == conditions[0]]
    group_b = df_muscle[df_muscle["Condition"] == conditions[1]]

    for measurment in measurment_cols:
        values_a = group_a[measurment].dropna()
        values_b = group_b[measurment].dropna()

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

# Add count-based tests (per image object counts), matching the same output schema.
for muscle in sorted(df_counts["Muscle"].dropna().unique()):
    df_counts_muscle = df_counts[df_counts["Muscle"] == muscle]
    count_group_a = df_counts_muscle[df_counts_muscle["Condition"] == conditions[0]]["Count"].dropna()
    count_group_b = df_counts_muscle[df_counts_muscle["Condition"] == conditions[1]]["Count"].dropna()

    if len(count_group_a) == 0 or len(count_group_b) == 0:
        continue

    count_u_stat, count_p_value = mannwhitneyu(count_group_a, count_group_b, alternative="two-sided")
    results.append({
        "Measurment": "Count",
        "Muscle": muscle,
        "Condition_A": conditions[0],
        "Condition_B": conditions[1],
        "N_A": len(count_group_a),
        "N_B": len(count_group_b),
        "U": count_u_stat,
        "p_value": count_p_value,
    })

# Save one combined table containing both measurement and count test results.
results_df = pd.DataFrame(results).sort_values(["Measurment", "Muscle"])
results_df.to_csv("/workspaces/mito-counter/data/Calpaine_3/results/statistics.csv", index=False)
results_df
