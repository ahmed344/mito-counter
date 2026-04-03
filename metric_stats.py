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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from stats_utils import maybe_show_current_figure, plot_metric_variants

# Set the style for the plots
sns.set_style("whitegrid")
# %% Load and prepare input data
# Read per-object morphological measurements exported by the pipeline.
df = pd.read_csv("/workspaces/mito-counter/data/Calpaine_3/results/measurments_cleaned.csv")
excluded_measurements = {"Connected_parts"}

# Treat grouping columns as categorical to keep consistent ordering/group handling.
df["Condition"] = df["Condition"].astype("category")
df["Muscle"] = df["Muscle"].astype("category")

# Display the first few rows of the dataframe
df.head()

# %% Quick distribution checks
# Metric columns start after metadata columns.
metrics = [
    column
    for column in df.columns[6:]
    if column not in excluded_measurements
]

# Count the zero values in each column
df[metrics].isin([0]).sum()

# %% Plot histograms for all numeric metrics
# This gives a first-pass view of skew/outliers for each measurement.
df[metrics].hist(figsize=(25, 8), bins=70, layout=(2, np.ceil(len(metrics) / 2).astype(int)))
plt.savefig("/workspaces/mito-counter/data/Calpaine_3/results/figures/histograms.png", dpi=900)
maybe_show_current_figure()
# Count the zero values in each column

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
maybe_show_current_figure()

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
save_dir = Path('/workspaces/mito-counter/data/Calpaine_3/results/figures')

# Plot Counts (No units needed, or you can add "objects" if you like)
plot_metric_variants(
    data=df_counts,
    x='Muscle',
    y='Count',
    hue='Condition',
    block='Block',
    save_dir=save_dir,
)

# Plot Morphological measurments with Units
for measurment in metrics:
    if measurment in df.columns:
        plot_metric_variants(
            data=df,
            x='Muscle',
            y=measurment,
            hue='Condition',
            block='Block',
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
    if pd.api.types.is_numeric_dtype(df[c]) and c not in excluded_measurements
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
