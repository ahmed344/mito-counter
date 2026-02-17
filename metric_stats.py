# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
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

# %%
# Load the data
df = pd.read_csv("/workspaces/mito-counter/data/Calpaine_3/results/measurments.csv")

# transform the Condition and Muscle columns to a categorical variable
df["Condition"] = df["Condition"].astype("category")
df["Muscle"] = df["Muscle"].astype("category")

# Display the first few rows of the dataframe
df.head()


# %%
def plot_stat_boxplot(data, x, y, hue, unit_dict=None, test='Mann-Whitney', text_format='star', save_dir=None):
    plt.figure(figsize=(8, 6))
    
    x_order = sorted(data[x].unique())
    hue_order = sorted(data[hue].unique())
    
    # 1. Create Boxplot
    ax = sns.boxplot(
        data=data, x=x, y=y, hue=hue,
        order=x_order, hue_order=hue_order,
        linewidth=1.5, 
        palette="Set2",
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
            
    annotator = Annotator(ax, box_pairs, data=data, x=x, y=y, hue=hue, 
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


# %%
# Count the number of rows (objects) per image, preserving Condition and Muscle info
df_counts = df.groupby(['Condition', 'Muscle', 'image'], observed=True).size().reset_index(name='Count')
df_counts

# %%
# Units Dictionary
units = {
    "Area": "um^2",
    "Major_axis_length": "um",
    "Minor_axis_length": "um",
    "Elongation": "",
    "Circularity": "",
    "Solidity": "",
    "NND": "um",
}

# %%
# Set the save directory
save_dir = '/workspaces/mito-counter/data/Calpaine_3/results/figures'

# Plot Counts (No units needed, or you can add "objects" if you like)
plot_stat_boxplot(df_counts, x='Muscle', y='Count', hue='Condition', save_dir=save_dir)

# Plot Morphological measurments with Units
for measurment in df.columns[5:]:
    if measurment in df.columns:
        plot_stat_boxplot(
            data=df, 
            x='Muscle', 
            y=measurment, 
            hue='Condition',
            unit_dict=units,
            save_dir=save_dir
        )

# %%
# Now perform a Mann-Whitney U test for each measurment for each muscle between the two conditions

# Determine the two conditions to compare
if isinstance(df["Condition"].dtype, pd.CategoricalDtype):
    conditions = list(df["Condition"].cat.categories)
else:
    conditions = sorted(df["Condition"].dropna().unique())

if len(conditions) != 2:
    raise ValueError(f"Expected exactly 2 conditions, found {len(conditions)}: {conditions}")

# Use the same measurment set as the plotting loop, filtering to numeric columns
measurment_cols = [
    c for c in df.columns[5:]
    if pd.api.types.is_numeric_dtype(df[c])
]

results = []
for muscle in sorted(df["Muscle"].dropna().unique()):
    df_muscle = df[df["Muscle"] == muscle]
    group_a = df_muscle[df_muscle["Condition"] == conditions[0]]
    group_b = df_muscle[df_muscle["Condition"] == conditions[1]]

    for measurment in measurment_cols:
        values_a = group_a[measurment].dropna()
        values_b = group_b[measurment].dropna()

        if len(values_a) == 0 or len(values_b) == 0:
            continue

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

results_df = pd.DataFrame(results).sort_values(["Measurment", "Muscle"])
results_df

# %%
