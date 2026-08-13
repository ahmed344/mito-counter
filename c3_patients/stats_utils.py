"""Patient-specific plotting utilities for CTRL and CAPN3 measurements."""

from __future__ import annotations

import colorsys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from scipy.stats import gaussian_kde


CONDITION_ORDER = ("CTRL", "CAPN3")
SITE_ORDER = ("Deltoid", "Quadriceps")
COMPARTMENT_ORDER = ("Intermyofibrillar (IMF)", "Sub-sarcolemmal (SS)")
CAPN3_SUBTYPE_ORDER = ("non_null/non_null", "null/non_null", "null/null")
SITE_COMPARTMENT_COLUMN = "Site_Compartment"

CONDITION_PALETTE = {"CTRL": "#4C78A8", "CAPN3": "#E45756"}
CAPN3_SUBTYPE_PALETTE = {
    "non_null/non_null": "#59A14F",
    "null/non_null": "#F28E2B",
    "null/null": "#B279A2",
}
_COMPARTMENT_SHORT = {
    "Intermyofibrillar (IMF)": "IMF",
    "Sub-sarcolemmal (SS)": "SS",
}
_METRIC_LABELS = {
    "Area": "Mitochondrial area",
    "Corrected_area": "Corrected mitochondrial area",
    "Major_axis_length": "Major-axis length",
    "Minor_axis_length": "Minor-axis length",
    "Minimum_Feret_Diameter": "Minimum Feret diameter",
    "Eccentricity": "Eccentricity",
    "Circularity": "Circularity",
    "Solidity": "Solidity",
    "NND": "Nearest-neighbor distance",
    "3NND": "Third-nearest-neighbor distance",
    "5NND": "Fifth-nearest-neighbor distance",
    "Voronoi_Cell_Area": "Voronoi cell area",
    "Instance_count": "Mitochondria count",
    "Ripley_L_integral": "Ripley's L integral",
    "Pair_Correlation_integral": "Pair-correlation integral",
}

SUPERPLOT_HALF_WIDTH = 0.42
SUPERPLOT_BODY_WIDTH_SCALE = 1.3
SUPERPLOT_HUE_GAP = 0.28
SUPERPLOT_HUE_GAP_TRIPLE = 1.05
SUPERPLOT_X_CLUSTER_GAP = 1.6
SUPERPLOT_X_CLUSTER_GAP_CROWDED = 2.2
SUPERPLOT_CROWDED_HUE_COUNT = 3
SUPERPLOT_CROWDED_FIGURE_HEIGHT = 12.0
SUPERPLOT_FIGURE_HEIGHT = 7.2
SUPERPLOT_GRID_SIZE = 512
SUPERPLOT_DPI = 600
SUPERVIOLIN_BLOCK_SEPARATOR_LINEWIDTH = 0.35
SUPERVIOLIN_SUBTYPE_SEPARATOR_LINEWIDTH = 1.4
SUPERPLOT_POINT_SIZE_MIN = 5
SUPERPLOT_POINT_SIZE_MAX = 28
SUPERPLOT_POINT_ALPHA = 0.4
SUPERPLOT_SUMMARY_SPREAD_FRACTION = 0.55
ROBUST_Y_LOWER_QUANTILE = 0.01
ROBUST_Y_UPPER_QUANTILE = 0.99
SUPERPLOT_ANNOTATION_BASE_Y = 1.06
SUPERPLOT_ANNOTATION_BRACKET_HEIGHT = 0.025
SUPERPLOT_ANNOTATION_TEXT_OFFSET = 0.012
SUPERPLOT_ANNOTATION_TEXT_LINE_GAP = 0.055
SUPERPLOT_ANNOTATION_STACK_GAP = 0.16
SUPERPLOT_ANNOTATION_BOX_PADDING = 0.04
SUPERPLOT_TITLE_ANNOTATION_GAP = 0.018
SUPERPLOT_ANNOTATION_FONT_SIZE = 10
SUPERPLOT_ANNOTATION_HDI_FONT_SIZE = 9
SUPERPLOT_ANNOTATION_HDI_COLOR = "0.18"
SUPERPLOT_TITLE_FONT_SIZE = 16
SUPERPLOT_TITLE_Y = 0.985
SUPERPLOT_ANNOTATION_SINGLE_OVERFLOW = 0.16
SUPERPLOT_X_TICK_FONT_SIZE = 13
SUPERPLOT_Y_TICK_FONT_SIZE = 13
DARK_MEDIAN_ANNOTATION_COLOR = "#08306b"
OVERALL_EFFECT_PREFIX = "capn3_effect_response"
CONDITION_OVERVIEW_HUE_COLUMN = "_condition_overview_hue"
CONDITION_OVERVIEW_HUE_VALUE = "_overview"
CONDITION_OVERVIEW_HALF_WIDTH = 1.05
CONDITION_OVERVIEW_X_GAP = 0.55
CONDITION_OVERVIEW_RIGHT_MARGIN = 0.70
CONDITION_OVERVIEW_OUTPUT_SUFFIX = "_condition"
CONDITION_OVERVIEW_NO_PATIENT_STEM_SUFFIX = "_no_patient"
CONDITION_OVERVIEW_LEGEND_FONT_SIZE = 6.5
CONDITION_OVERVIEW_LEGEND_TITLE_FONT_SIZE = 8
CONDITION_OVERVIEW_LEGEND_BBOX = (1.02, 1.18)

CONDITION_SITE_COMPARTMENT_COLOR_STOPS: dict[tuple[str, str, str], list[str]] = {
    ("CTRL", "Deltoid", "Intermyofibrillar (IMF)"): [
        "#041B4D",
        "#08519C",
        "#2171B5",
        "#6BAED6",
        "#C6DBEF",
    ],
    ("CAPN3", "Deltoid", "Intermyofibrillar (IMF)"): [
        "#014636",
        "#01665E",
        "#238B8F",
        "#66C2A4",
        "#CCECE6",
    ],
    ("CTRL", "Deltoid", "Sub-sarcolemmal (SS)"): [
        "#00441B",
        "#006D2C",
        "#238B45",
        "#74C476",
        "#C7E9C0",
    ],
    ("CAPN3", "Deltoid", "Sub-sarcolemmal (SS)"): [
        "#3F007D",
        "#6A51A3",
        "#9E9AC8",
        "#CBC9E2",
        "#F2F0F7",
    ],
    ("CTRL", "Quadriceps", "Intermyofibrillar (IMF)"): [
        "#5A6200",
        "#9AAA00",
        "#D0DE00",
        "#F2F84A",
        "#FFFFB8",
    ],
    ("CAPN3", "Quadriceps", "Intermyofibrillar (IMF)"): [
        "#6B2410",
        "#B85A32",
        "#E0895C",
        "#F5C4A1",
        "#FDE8D8",
    ],
    ("CTRL", "Quadriceps", "Sub-sarcolemmal (SS)"): [
        "#67000D",
        "#A50F15",
        "#CB181D",
        "#FB6A4A",
        "#FCBBA1",
    ],
    ("CAPN3", "Quadriceps", "Sub-sarcolemmal (SS)"): [
        "#3E1F00",
        "#8C510A",
        "#BF812D",
        "#DFC27D",
        "#F6E8C3",
    ],
}
GREY_BLOCK_COLOR_STOPS = ["#252525", "#525252", "#737373", "#BDBDBD", "#F0F0F0"]
SUBTYPE_SITE_COLOR_STOPS: dict[tuple[str, str], list[str]] = {
    ("non_null/non_null", "Deltoid"): [
        "#00441B",
        "#006D2C",
        "#238B45",
        "#74C476",
        "#C7E9C0",
    ],
    ("non_null/non_null", "Quadriceps"): [
        "#1B4D00",
        "#3D8B00",
        "#59A14F",
        "#A1D99B",
        "#E5F5E0",
    ],
    ("null/non_null", "Deltoid"): [
        "#7F2704",
        "#D94801",
        "#F28E2B",
        "#FDAE6B",
        "#FDD0A2",
    ],
    ("null/non_null", "Quadriceps"): [
        "#5C3A00",
        "#B36B00",
        "#E08A00",
        "#F0B429",
        "#FFE08A",
    ],
    ("null/null", "Deltoid"): [
        "#4A0050",
        "#7A0177",
        "#B279A2",
        "#D4B9D9",
        "#F2E6F2",
    ],
    ("null/null", "Quadriceps"): [
        "#3F007D",
        "#6A51A3",
        "#9E9AC8",
        "#CBC9E2",
        "#F2F0F7",
    ],
}
CELL_EFFECT_COLUMNS = (
    ("Deltoid", "Intermyofibrillar (IMF)", "capn3_effect_deltoid_imf_response"),
    ("Deltoid", "Sub-sarcolemmal (SS)", "capn3_effect_deltoid_ss_response"),
    ("Quadriceps", "Intermyofibrillar (IMF)", "capn3_effect_quadriceps_imf_response"),
    ("Quadriceps", "Sub-sarcolemmal (SS)", "capn3_effect_quadriceps_ss_response"),
)
SUBTYPE_PAIR_CONTRASTS: tuple[tuple[str, str], ...] = (
    (CAPN3_SUBTYPE_ORDER[0], CAPN3_SUBTYPE_ORDER[1]),
    (CAPN3_SUBTYPE_ORDER[1], CAPN3_SUBTYPE_ORDER[2]),
    (CAPN3_SUBTYPE_ORDER[0], CAPN3_SUBTYPE_ORDER[2]),
)


def save_figure(
    figure: Figure,
    output_path: str | Path,
    dpi: int = SUPERPLOT_DPI,
    close: bool = True,
) -> Path:
    """Save a figure atomically after creating its parent directory.

    Args:
        figure (Figure): Matplotlib figure to save.
        output_path (str | Path): Destination filename.
        dpi (int): Raster resolution.
        close (bool): Whether to close the figure after saving.

    Returns:
        Path: Resolved output path.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        figure.savefig(
            temporary_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            metadata={"Creator": "c3_patients.stats_utils"},
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
        if close:
            plt.close(figure)
    return path.resolve()


def metric_label(metric: str) -> str:
    """Return a readable patient-analysis metric label.

    Args:
        metric (str): Measurement column name.

    Returns:
        str: Human-readable label.
    """

    if metric in _METRIC_LABELS:
        return _METRIC_LABELS[metric]
    suffixes = ("_center_mean", "_center_cv", "_mean", "_sum")
    base = metric
    suffix = ""
    for candidate in suffixes:
        if metric.endswith(candidate):
            base = metric[: -len(candidate)]
            suffix = candidate
            break
    base_label = _METRIC_LABELS.get(base, base.replace("_", " ").strip().capitalize())
    suffix_label = {
        "_center_mean": " (center mean)",
        "_center_cv": " (center CV)",
        "_mean": " (image mean)",
        "_sum": " (image sum)",
    }.get(suffix, "")
    return f"{base_label}{suffix_label}"


def metric_unit(metric: str) -> str:
    """Infer the display unit for a configured patient metric.

    Args:
        metric (str): Measurement column name.

    Returns:
        str: Unit text, or an empty string for unitless metrics.
    """

    normalized = metric.lower()
    if normalized.startswith(("eccentricity", "circularity", "solidity")):
        return ""
    if normalized.endswith("_cv"):
        return ""
    if normalized == "instance_count":
        return "count"
    if normalized == "ripley_l_integral":
        return "integrated nm²"
    if normalized == "pair_correlation_integral":
        return "integrated unitless"
    if "area" in normalized:
        return "nm²"
    if normalized in {"nnd", "3nnd", "5nnd"} or "diameter" in normalized:
        return "nm"
    if "axis_length" in normalized or "_nnd" in normalized:
        return "nm"
    return ""


def metric_axis_label(metric: str) -> str:
    """Build a y-axis label with a unit when one is meaningful.

    Args:
        metric (str): Measurement column name.

    Returns:
        str: Complete y-axis label.
    """

    label = metric_label(metric)
    unit = metric_unit(metric)
    return f"{label} ({unit})" if unit else label


def metric_unit_mapping(metrics: Sequence[str]) -> dict[str, str]:
    """Build a metric-to-unit mapping for superplot axis labels.

    Args:
        metrics (Sequence[str]): Metric column names.

    Returns:
        dict[str, str]: Units keyed by metric, omitting unitless metrics.
    """

    mapping: dict[str, str] = {}
    for metric in metrics:
        unit = metric_unit(metric)
        if unit:
            mapping[str(metric)] = unit
    return mapping


def condition_sort_key(condition: str) -> tuple[int, str]:
    """Return a stable sort key with CTRL before CAPN3.

    Args:
        condition (str): Condition label.

    Returns:
        tuple[int, str]: Preferred-order index and normalized label.
    """

    text = str(condition)
    try:
        index = CONDITION_ORDER.index(text)
    except ValueError:
        try:
            index = len(CONDITION_ORDER) + CAPN3_SUBTYPE_ORDER.index(text)
        except ValueError:
            index = len(CONDITION_ORDER) + len(CAPN3_SUBTYPE_ORDER)
    return index, text.casefold()


def sort_conditions(conditions: Sequence[str]) -> list[str]:
    """Sort unique condition labels in patient-study order.

    Args:
        conditions (Sequence[str]): Condition labels.

    Returns:
        list[str]: Unique sorted labels.
    """

    return sorted({str(value) for value in conditions}, key=condition_sort_key)


def site_compartment_label(site: str, compartment: str) -> str:
    """Combine biopsy site and compartment into a compact axis label.

    Args:
        site (str): Biopsy site.
        compartment (str): Mitochondrial compartment.

    Returns:
        str: ``Site | IMF`` or ``Site | SS`` label.
    """

    compartment_text = _COMPARTMENT_SHORT.get(str(compartment), str(compartment))
    return f"{site} | {compartment_text}"


def site_compartment_labels() -> list[str]:
    """Return all site-by-compartment labels in canonical order.

    Args:
        None: This function takes no arguments.

    Returns:
        list[str]: Ordered site-by-compartment labels.
    """

    return [
        site_compartment_label(site, compartment)
        for site in SITE_ORDER
        for compartment in COMPARTMENT_ORDER
    ]


def add_site_compartment_column(
    data: pd.DataFrame,
    site_column: str,
    compartment_column: str,
    output_column: str = SITE_COMPARTMENT_COLUMN,
) -> pd.DataFrame:
    """Copy a dataframe and add combined site-compartment labels.

    Args:
        data (pd.DataFrame): Measurement rows.
        site_column (str): Biopsy-site column.
        compartment_column (str): Compartment column.
        output_column (str): Name of the combined label column.

    Returns:
        pd.DataFrame: Copy with ``output_column`` populated.
    """

    frame = data.copy()
    frame[output_column] = [
        site_compartment_label(site, compartment)
        for site, compartment in zip(
            frame[site_column], frame[compartment_column], strict=True
        )
    ]
    return frame


def format_effect_annotation(
    row: Mapping[str, Any] | pd.Series,
    prefix: str,
) -> str:
    """Format one posterior contrast as superplot annotation text.

    Args:
        row (Mapping[str, Any] | pd.Series): Bayesian summary fields.
        prefix (str): Column prefix such as ``capn3_effect_deltoid_imf_response``.

    Returns:
        str: ``estimate [low, high] pd%`` text, or an empty string.
    """

    estimate = pd.to_numeric(pd.Series([row.get(prefix)]), errors="coerce").iloc[0]
    low = pd.to_numeric(
        pd.Series([row.get(f"{prefix}_hdi_low")]), errors="coerce"
    ).iloc[0]
    high = pd.to_numeric(
        pd.Series([row.get(f"{prefix}_hdi_high")]), errors="coerce"
    ).iloc[0]
    if pd.isna(estimate) or pd.isna(low) or pd.isna(high):
        return ""
    annotation = f"{float(estimate):.3g} [{float(low):.3g}, {float(high):.3g}]"
    probability = pd.to_numeric(
        pd.Series([row.get(f"{prefix}_pd")]), errors="coerce"
    ).iloc[0]
    if not pd.isna(probability):
        annotation = f"{annotation} {float(probability):.1f}%"
    return annotation


def bayesian_superplot_annotations(
    summary_row: Mapping[str, Any] | pd.Series | None,
    hue_start: str = CONDITION_ORDER[0],
    hue_end: str = CONDITION_ORDER[1],
) -> list[dict[str, str]]:
    """Build per-cell Bayesian bracket annotations for CTRL versus CAPN3.

    Args:
        summary_row (Mapping[str, Any] | pd.Series | None): Primary fit summary.
        hue_start (str): Left hue of each annotated pair.
        hue_end (str): Right hue of each annotated pair.

    Returns:
        list[dict[str, str]]: Annotation records for ``plot_super_violin``.
    """

    if summary_row is None:
        return []
    annotations: list[dict[str, str]] = []
    for site, compartment, prefix in CELL_EFFECT_COLUMNS:
        label = format_effect_annotation(summary_row, prefix)
        if not label:
            continue
        annotations.append(
            {
                "x": site_compartment_label(site, compartment),
                "hue_start": hue_start,
                "hue_end": hue_end,
                "median_label": label,
                "median_color": DARK_MEDIAN_ANNOTATION_COLOR,
            }
        )
    return annotations


def bayesian_overall_superplot_annotation(
    summary_row: Mapping[str, Any] | pd.Series | None,
    hue_value: str = CONDITION_OVERVIEW_HUE_VALUE,
) -> list[dict[str, str]]:
    """Build one CTRL-versus-CAPN3 bracket from the overall posterior contrast.

    Args:
        summary_row (Mapping[str, Any] | pd.Series | None): Primary fit summary.
        hue_value (str): Dummy hue used by two-tick condition overview plots.

    Returns:
        list[dict[str, str]]: A single annotation record spanning CTRL and CAPN3,
        or an empty list when the overall contrast is missing.
    """

    if summary_row is None:
        return []
    label = format_effect_annotation(summary_row, OVERALL_EFFECT_PREFIX)
    if not label:
        return []
    return [
        {
            "x": CONDITION_ORDER[0],
            "x_end": CONDITION_ORDER[1],
            "hue_start": str(hue_value),
            "hue_end": str(hue_value),
            "median_label": label,
            "median_color": DARK_MEDIAN_ANNOTATION_COLOR,
        }
    ]


def subtype_contrast_slug(label: str) -> str:
    """Convert a CAPN3 subtype label into a Bayesian summary slug.

    Args:
        label (str): Subtype label such as ``non_null/non_null``.

    Returns:
        str: Underscore-separated slug used in summary column names.
    """

    slug = "".join(
        character if character.isalnum() else "_" for character in str(label).lower()
    )
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def subtype_pair_contrast_prefix(hue_start: str, hue_end: str) -> str:
    """Return the response-scale summary prefix for one subtype pair.

    Args:
        hue_start (str): First subtype label.
        hue_end (str): Second subtype label.

    Returns:
        str: Column prefix such as
        ``subtype_non_null_non_null_vs_null_null_contrast_response``.
    """

    start_slug = subtype_contrast_slug(hue_start)
    end_slug = subtype_contrast_slug(hue_end)
    return f"subtype_{start_slug}_vs_{end_slug}_contrast_response"


def bayesian_subtype_superplot_annotations(
    summary_row: Mapping[str, Any] | pd.Series | None,
) -> list[dict[str, str]]:
    """Build pairwise Bayesian brackets for CAPN3 subtype superplots.

    Adjacent pairs are emitted before the spanning pair so stacked brackets
    draw the two neighboring comparisons first and the full three-hue span
    on top.

    Args:
        summary_row (Mapping[str, Any] | pd.Series | None): Primary fit summary.

    Returns:
        list[dict[str, str]]: Annotation records for ``plot_super_violin``.
    """

    if summary_row is None:
        return []
    annotations: list[dict[str, str]] = []
    for site, compartment, _prefix in CELL_EFFECT_COLUMNS:
        x_label = site_compartment_label(site, compartment)
        for hue_start, hue_end in SUBTYPE_PAIR_CONTRASTS:
            label = format_effect_annotation(
                summary_row,
                subtype_pair_contrast_prefix(hue_start, hue_end),
            )
            if not label:
                continue
            annotations.append(
                {
                    "x": x_label,
                    "hue_start": hue_start,
                    "hue_end": hue_end,
                    "median_label": label,
                    "median_color": DARK_MEDIAN_ANNOTATION_COLOR,
                }
            )
    return annotations


def build_output_path(
    y: str,
    x: str,
    hue: str,
    save_dir: str | Path | None,
    suffix: str,
    filename_prefix: str | None = None,
    output_dir_suffix: str = "",
    filename_stem_suffix: str = "",
) -> Path | None:
    """Construct a superplot output path under a plot-type directory.

    Args:
        y (str): Numeric metric column name.
        x (str): X-axis grouping column name.
        hue (str): Hue grouping column name.
        save_dir (str | Path | None): Base output directory.
        suffix (str): Plot-type suffix, e.g. ``superviolin``.
        filename_prefix (str | None): Optional filename prefix before suffix.
        output_dir_suffix (str): Optional suffix appended to the output directory
            name, e.g. ``_capn3_subtype``.
        filename_stem_suffix (str): Optional suffix inserted before ``.png``,
            e.g. ``_no_patient``.

    Returns:
        Path | None: Full output path when ``save_dir`` is provided, otherwise ``None``.
    """

    if save_dir is None:
        return None
    directory_by_suffix = {
        "superviolin": "super_violins",
        "superbeeswarm": "super_beeswarms",
    }
    directory_name = directory_by_suffix.get(suffix, "")
    output_dir = Path(save_dir) / f"{directory_name}{str(output_dir_suffix).strip()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem_suffix = str(filename_stem_suffix).strip()
    if filename_prefix is not None:
        safe_prefix = str(filename_prefix).strip().replace("/", "_").replace(" ", "_")
        return output_dir / f"{safe_prefix}{suffix}{stem_suffix}.png"
    safe_parts = [
        str(part).replace("/", "_").replace(" ", "_") for part in (y, x, hue, suffix)
    ]
    return output_dir / (
        f"{safe_parts[0]}_by_{safe_parts[1]}_and_{safe_parts[2]}_{safe_parts[3]}{stem_suffix}.png"
    )


def sort_block_values(values: Sequence[object]) -> list[str]:
    """Sort patient or block labels numerically when possible.

    Args:
        values (Sequence[object]): Raw patient identifiers.

    Returns:
        list[str]: Sorted string labels.
    """

    normalized = [str(value) for value in values]
    try:
        return sorted(normalized, key=lambda value: (0, int(value)))
    except ValueError:
        return sorted(normalized, key=lambda value: (1, value))


def capn3_subtype_index(genotype: str) -> int:
    """Return the canonical CAPN3 subtype rank for one genotype label.

    Args:
        genotype (str): Genotype or subtype label.

    Returns:
        int: Index in ``CAPN3_SUBTYPE_ORDER``, or one past the last subtype.
    """

    try:
        return CAPN3_SUBTYPE_ORDER.index(str(genotype))
    except ValueError:
        return len(CAPN3_SUBTYPE_ORDER)


def patient_genotype_map(
    plot_data: pd.DataFrame,
    block: str,
    genotype_column: str,
) -> dict[str, str]:
    """Map each patient identifier to a single genotype label.

    Args:
        plot_data (pd.DataFrame): Plot dataframe containing patient and genotype
            columns.
        block (str): Patient-identity column name.
        genotype_column (str): Genotype column name.

    Returns:
        dict[str, str]: Mapping from patient identifier to genotype string.
    """

    if genotype_column not in plot_data.columns:
        return {}
    pairs = (
        plot_data[[block, genotype_column]]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )
    mapping: dict[str, str] = {}
    for patient, genotype in zip(
        pairs[block].tolist(), pairs[genotype_column].tolist(), strict=True
    ):
        mapping.setdefault(patient, genotype)
    return mapping


def order_group_blocks(
    plot_data: pd.DataFrame | None,
    x: str | None,
    hue: str | None,
    block: str | None,
    x_value: str,
    hue_value: str,
    block_order: list[str],
    genotype_column: str | None = None,
) -> list[str]:
    """Return patients for one violin, optionally ordered by CAPN3 subtype.

    CAPN3 violins are ordered ``non_null/non_null``, ``null/non_null``, then
    ``null/null``, with the usual identifier sort within each subtype. CTRL
    and all other hues keep ``block_order``.

    Args:
        plot_data (pd.DataFrame | None): Plot dataframe, or ``None`` to keep
            ``block_order``.
        x (str | None): X-axis grouping column name.
        hue (str | None): Hue grouping column name.
        block (str | None): Patient-identity column name.
        x_value (str): X-axis category label for the current group.
        hue_value (str): Hue category label for the current group.
        block_order (list[str]): Default patient order.
        genotype_column (str | None): Optional genotype column used to order
            CAPN3 patients.

    Returns:
        list[str]: Ordered patient labels for this group.
    """

    present = group_block_labels(
        plot_data=plot_data,
        x=x,
        hue=hue,
        block=block,
        x_value=x_value,
        hue_value=hue_value,
        block_order=block_order,
    )
    if (
        not present
        or plot_data is None
        or block is None
        or genotype_column is None
        or genotype_column not in plot_data.columns
        or str(hue_value) != CONDITION_ORDER[1]
    ):
        return present
    subtype_map = patient_genotype_map(
        plot_data=plot_data,
        block=block,
        genotype_column=genotype_column,
    )
    grouped: dict[int, list[str]] = {}
    for label in present:
        index = capn3_subtype_index(subtype_map.get(label, ""))
        grouped.setdefault(index, []).append(label)
    ordered: list[str] = []
    for index in sorted(grouped):
        ordered.extend(sort_block_values(grouped[index]))
    return ordered


def superviolin_separator_linewidth(
    current_block: str,
    next_block: str,
    hue_value: str,
    genotype_map: Mapping[str, str],
) -> float:
    """Choose the stripe-separator width between two adjacent patients.

    Args:
        current_block (str): Left-hand patient identifier.
        next_block (str): Right-hand patient identifier.
        hue_value (str): Violin hue label.
        genotype_map (Mapping[str, str]): Patient-to-genotype mapping.

    Returns:
        float: Linewidth for the separator.
    """

    if str(hue_value) != CONDITION_ORDER[1]:
        return SUPERVIOLIN_BLOCK_SEPARATOR_LINEWIDTH
    current_genotype = genotype_map.get(str(current_block))
    next_genotype = genotype_map.get(str(next_block))
    if (
        current_genotype is None
        or next_genotype is None
        or current_genotype == next_genotype
    ):
        return SUPERVIOLIN_BLOCK_SEPARATOR_LINEWIDTH
    return SUPERVIOLIN_SUBTYPE_SEPARATOR_LINEWIDTH


def prepare_plot_data(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    block: str | None = None,
) -> pd.DataFrame:
    """Coerce numeric columns and drop incomplete rows for plotting.

    Args:
        data (pd.DataFrame): Source dataframe.
        x (str): X-axis grouping column.
        y (str): Numeric metric column.
        hue (str): Hue grouping column.
        block (str | None): Optional patient-identity column.

    Returns:
        pd.DataFrame: Clean plotting dataframe.
    """

    plot_data = data.copy()
    plot_data[y] = pd.to_numeric(plot_data[y], errors="coerce")
    required_columns = [x, y, hue]
    if block is not None:
        required_columns.append(block)
    plot_data = plot_data.dropna(subset=required_columns)
    if block is not None and block in plot_data.columns:
        plot_data[block] = plot_data[block].astype(str)
    return plot_data


def format_unit_label(metric_name: str, unit_dict: dict[str, str] | None) -> str:
    """Build a formatted unit suffix for axis labels.

    Args:
        metric_name (str): Metric key shown on y-axis.
        unit_dict (dict[str, str] | None): Optional metric-to-unit mapping.

    Returns:
        str: Unit suffix in parentheses, or an empty string.
    """

    if not unit_dict or metric_name not in unit_dict or not unit_dict[metric_name]:
        return ""
    unit_text = unit_dict[metric_name]
    if unit_text in {"um", "µm"}:
        formatted_unit = r"$\mu m$"
    elif unit_text in {"um^2", "µm^2", "µm²"}:
        formatted_unit = r"$\mu m^2$"
    elif unit_text == "nm":
        formatted_unit = r"$nm$"
    elif unit_text in {"nm^2", "nm²"}:
        formatted_unit = r"$nm^2$"
    else:
        formatted_unit = unit_text.replace("^2", r"$^2$")
    return f" ({formatted_unit})"


def get_category_orders(
    plot_data: pd.DataFrame,
    x: str,
    hue: str,
    block: str | None = None,
    x_order_override: list[str] | None = None,
    hue_order_override: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Determine stable x/hue/patient plotting order.

    Args:
        plot_data (pd.DataFrame): Plot-ready dataframe.
        x (str): X-axis grouping column.
        hue (str): Hue grouping column.
        block (str | None): Optional patient-identity column.
        x_order_override (list[str] | None): Explicit x order when provided.
        hue_order_override (list[str] | None): Explicit hue order when provided.

    Returns:
        tuple[list[str], list[str], list[str]]: Orders for x, hue, and patients.
    """

    if x_order_override is None:
        data_values = plot_data[x].astype(str).unique().tolist()
        canonical = site_compartment_labels()
        x_order = [value for value in canonical if value in set(data_values)]
        remaining = sorted(set(data_values) - set(x_order))
        x_order.extend(remaining)
    else:
        data_values = set(plot_data[x].astype(str).unique().tolist())
        x_order = [value for value in x_order_override if str(value) in data_values]

    if hue_order_override is None:
        hue_order = sort_conditions(plot_data[hue].astype(str).unique().tolist())
    else:
        data_values = set(plot_data[hue].astype(str).unique().tolist())
        hue_order = [value for value in hue_order_override if str(value) in data_values]

    block_order: list[str] = []
    if block is not None and block in plot_data.columns:
        block_order = sort_block_values(plot_data[block].astype(str).unique().tolist())
    return x_order, hue_order, block_order


def primary_category_token(label: str) -> str:
    """Extract the primary token from a raw or combined category label.

    Args:
        label (str): Raw label, optionally combined as ``TOKEN | SS``.

    Returns:
        str: Left-side token before ``|``, stripped of whitespace.
    """

    normalized = str(label).strip()
    if "|" in normalized:
        return normalized.split("|", maxsplit=1)[0].strip()
    return normalized


def parse_site_label(label: str) -> str | None:
    """Parse a biopsy-site name from a category label.

    Args:
        label (str): Raw site, condition, or combined plot label.

    Returns:
        str | None: ``Deltoid`` or ``Quadriceps`` when recognized, otherwise ``None``.
    """

    token = primary_category_token(label=label)
    if token in SITE_ORDER:
        return token
    lowered = token.casefold()
    for site in SITE_ORDER:
        if site.casefold() == lowered:
            return site
    return None


def parse_case_label(label: str) -> str | None:
    """Parse a condition or CAPN3-subtype label.

    Args:
        label (str): Raw hue or combined plot label.

    Returns:
        str | None: Condition or subtype string when recognized, otherwise ``None``.
    """

    token = str(label).strip()
    if token in CONDITION_ORDER or token in CAPN3_SUBTYPE_ORDER:
        return token
    if token == "ref/ref":
        return CONDITION_ORDER[0]
    return None


def parse_compartment_label(label: str) -> str | None:
    """Parse a mitochondrial compartment from a category label.

    Args:
        label (str): Raw or combined plot label, e.g. ``Deltoid | IMF``.

    Returns:
        str | None: Canonical compartment name when recognized, otherwise ``None``.
    """

    normalized = str(label).strip()
    token = normalized
    if "|" in normalized:
        token = normalized.split("|", maxsplit=1)[1].strip()
    if token in COMPARTMENT_ORDER:
        return token
    lowered = token.casefold()
    for full_name, short_name in _COMPARTMENT_SHORT.items():
        if short_name.casefold() == lowered or full_name.casefold() == lowered:
            return full_name
    return None


def resolve_case_site(x_value: str, hue_value: str) -> tuple[str, str] | None:
    """Resolve case and biopsy-site codes from an x/hue label pair.

    Args:
        x_value (str): X-axis category label.
        hue_value (str): Hue category label.

    Returns:
        tuple[str, str] | None: ``(case, site)`` when both can be inferred.
    """

    case = parse_case_label(label=x_value) or parse_case_label(label=hue_value)
    site = parse_site_label(label=x_value) or parse_site_label(label=hue_value)
    if case is None or site is None:
        return None
    return case, site


def resolve_palette_group(
    x_value: str,
    hue_value: str,
) -> tuple[str, str, str] | tuple[str, str] | None:
    """Resolve palette-group codes from an x/hue label pair.

    Args:
        x_value (str): X-axis category label.
        hue_value (str): Hue category label.

    Returns:
        tuple[str, str, str] | tuple[str, str] | None: ``(case, site, compartment)``
        when compartment is known, ``(case, site)`` when only those can be inferred,
        otherwise ``None``.
    """

    case_site = resolve_case_site(x_value=x_value, hue_value=hue_value)
    if case_site is None:
        return None
    compartment = parse_compartment_label(label=x_value) or parse_compartment_label(
        label=hue_value
    )
    if compartment is None:
        return case_site
    return case_site[0], case_site[1], compartment


def lookup_color_stops(
    group_key: tuple[str, str, str] | tuple[str, str] | None,
) -> list[str] | None:
    """Return sequential color stops for one palette group.

    Args:
        group_key (tuple[str, str, str] | tuple[str, str] | None): Palette lookup
            key from ``resolve_palette_group``.

    Returns:
        list[str] | None: Hex stops when a family is defined, otherwise ``None``.
    """

    if group_key is None:
        return None
    if len(group_key) == 3:
        color_stops = CONDITION_SITE_COMPARTMENT_COLOR_STOPS.get(group_key)
        if color_stops is not None:
            return color_stops
        return SUBTYPE_SITE_COLOR_STOPS.get(group_key[:2])
    return SUBTYPE_SITE_COLOR_STOPS.get(group_key)


def format_palette_group_text(
    group_key: tuple[str, str, str] | tuple[str, str] | None,
    hue_value: str,
) -> str:
    """Build a compact legend prefix for one palette group.

    Args:
        group_key (tuple[str, str, str] | tuple[str, str] | None): Resolved palette
            group, or ``None`` when case and site cannot be inferred.
        hue_value (str): Hue category label used as a fallback.

    Returns:
        str: Legend prefix such as ``CTRL Deltoid IMF``.
    """

    if group_key is None:
        return format_condition_display_label(hue_value)
    if len(group_key) >= 3:
        compartment_short = _COMPARTMENT_SHORT.get(group_key[2], group_key[2])
        return f"{group_key[0]} {group_key[1]} {compartment_short}"
    return f"{group_key[0]} {group_key[1]}"


def group_block_labels(
    plot_data: pd.DataFrame | None,
    x: str | None,
    hue: str | None,
    block: str | None,
    x_value: str,
    hue_value: str,
    block_order: list[str],
) -> list[str]:
    """Return patients present in one x/hue group, preserving ``block_order``.

    Args:
        plot_data (pd.DataFrame | None): Plot dataframe, or ``None`` to keep all
            patients.
        x (str | None): X-axis grouping column name.
        hue (str | None): Hue grouping column name.
        block (str | None): Patient-identity column name.
        x_value (str): X-axis category label for the current group.
        hue_value (str): Hue category label for the current group.
        block_order (list[str]): Ordered patient labels.

    Returns:
        list[str]: Patients to color in this group.
    """

    if plot_data is None or x is None or hue is None or block is None:
        return list(block_order)
    present = set(
        plot_data.loc[
            (plot_data[x].astype(str) == str(x_value))
            & (plot_data[hue].astype(str) == str(hue_value)),
            block,
        ]
        .astype(str)
        .unique()
        .tolist()
    )
    return [label for label in block_order if label in present]


def blend_block_palette(
    color_stops: list[str],
    n_colors: int,
) -> list[tuple[float, float, float]]:
    """Build a dark-to-light RGB palette with even HLS lightness steps.

    Args:
        color_stops (list[str]): Hex color stops for the hue family.
        n_colors (int): Number of patient colors to generate.

    Returns:
        list[tuple[float, float, float]]: RGB colors of length ``n_colors``.
    """

    if n_colors <= 0:
        return []
    start_lightness = colorsys.rgb_to_hls(*to_rgb(color_stops[0]))[1]
    end_lightness = colorsys.rgb_to_hls(*to_rgb(color_stops[-1]))[1]
    sample_count = max(n_colors, 2)
    rgb_ramp = list(sns.blend_palette(color_stops, n_colors=sample_count))
    if n_colors == 1:
        hue, _lightness, saturation = colorsys.rgb_to_hls(*rgb_ramp[len(rgb_ramp) // 2])
        mid_lightness = (start_lightness + end_lightness) / 2.0
        return [colorsys.hls_to_rgb(hue, mid_lightness, saturation)]
    lightnesses = np.linspace(start_lightness, end_lightness, n_colors)
    colors: list[tuple[float, float, float]] = []
    for index, lightness in enumerate(lightnesses):
        hue, _lightness, saturation = colorsys.rgb_to_hls(*rgb_ramp[index])
        colors.append(colorsys.hls_to_rgb(hue, float(lightness), saturation))
    return colors


def create_condition_block_palette(
    block_order: list[str],
    x_order: list[str],
    hue_order: list[str],
    plot_data: pd.DataFrame | None = None,
    x: str | None = None,
    hue: str | None = None,
    block: str | None = None,
    genotype_column: str | None = None,
) -> dict[tuple[str, str, str], tuple[float, float, float]]:
    """Create colors keyed by case×site×compartment group and patient identity.

    Args:
        block_order (list[str]): Ordered patient labels.
        x_order (list[str]): Ordered x-axis category labels.
        hue_order (list[str]): Ordered hue category labels.
        plot_data (pd.DataFrame | None): Plot dataframe used to find patients in
            each x/hue group. When omitted, every patient shares the full ramp.
        x (str | None): X-axis grouping column name.
        hue (str | None): Hue grouping column name.
        block (str | None): Patient-identity column name.
        genotype_column (str | None): Optional genotype column used to order
            CAPN3 patients left to right.

    Returns:
        dict[tuple[str, str, str], tuple[float, float, float]]: Mapping from
        ``(x, hue, patient)`` to RGB color.
    """

    if not block_order or not x_order or not hue_order:
        return {}

    palette: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    for x_value in x_order:
        for hue_value in hue_order:
            group_blocks = order_group_blocks(
                plot_data=plot_data,
                x=x,
                hue=hue,
                block=block,
                x_value=str(x_value),
                hue_value=str(hue_value),
                block_order=block_order,
                genotype_column=genotype_column,
            )
            if not group_blocks:
                continue
            group_key = resolve_palette_group(
                x_value=str(x_value), hue_value=str(hue_value)
            )
            color_stops = lookup_color_stops(group_key)
            if color_stops is None:
                color_stops = GREY_BLOCK_COLOR_STOPS
            block_colors = blend_block_palette(
                color_stops=color_stops,
                n_colors=len(group_blocks),
            )
            for block_label, color in zip(group_blocks, block_colors, strict=True):
                palette[(str(x_value), str(hue_value), str(block_label))] = color
    return palette


def format_condition_display_label(condition_value: str) -> str:
    """Format condition or subtype labels for axis tick display.

    Args:
        condition_value (str): Raw hue label from plotting data.

    Returns:
        str: Compact label for plot text.
    """

    return str(condition_value).strip().replace("_", " ")


def build_block_legend_labels(
    plot_data: pd.DataFrame,
    x: str,
    block: str,
    hue: str,
    x_order: list[str],
    hue_order: list[str],
    block_order: list[str],
    genotype_column: str | None = None,
    include_patient_id: bool = True,
) -> dict[tuple[str, str, str], str]:
    """Create readable legend labels for colored patients.

    Args:
        plot_data (pd.DataFrame): Plot dataframe containing x, patient, and hue columns.
        x (str): X-axis grouping column name.
        block (str): Patient column name.
        hue (str): Hue column name.
        x_order (list[str]): Ordered x-axis category labels.
        hue_order (list[str]): Ordered hue category labels.
        block_order (list[str]): Ordered patient labels.
        genotype_column (str | None): Optional genotype column. When provided,
            CAPN3 labels include the patient subtype.
        include_patient_id (bool): When ``False``, omit the patient identifier
            from each legend line.

    Returns:
        dict[tuple[str, str, str], str]: Mapping from ``(x, hue, patient)`` to legend text.
    """

    labels: dict[tuple[str, str, str], str] = {}
    for x_value in x_order:
        for hue_value in hue_order:
            for block_label in block_order:
                rows = plot_data.loc[
                    (plot_data[x].astype(str) == str(x_value))
                    & (plot_data[hue].astype(str) == str(hue_value))
                    & (plot_data[block].astype(str) == str(block_label))
                ]
                if rows.empty:
                    continue
                group_key = resolve_palette_group(
                    x_value=str(x_value), hue_value=str(hue_value)
                )
                group_text = format_palette_group_text(
                    group_key=group_key, hue_value=str(hue_value)
                )
                parts = [group_text]
                if (
                    genotype_column is not None
                    and genotype_column in rows.columns
                    and str(hue_value) == CONDITION_ORDER[1]
                ):
                    genotype = rows[genotype_column].dropna().astype(str)
                    if not genotype.empty:
                        subtype_text = format_condition_display_label(genotype.iloc[0])
                        parts.append(f"({subtype_text})")
                if include_patient_id:
                    parts.append(str(block_label))
                labels[(str(x_value), str(hue_value), str(block_label))] = " ".join(
                    parts
                )
    return labels


def superplot_is_crowded(hue_order: Sequence[str]) -> bool:
    """Return whether hue groups need extra spacing and rotated ticks.

    Args:
        hue_order (Sequence[str]): Hue categories in plotting order.

    Returns:
        bool: ``True`` when three or more hue levels are plotted.
    """

    return len(hue_order) >= SUPERPLOT_CROWDED_HUE_COUNT


def superplot_hue_gap_for_count(hue_count: int) -> float:
    """Return the within-cluster gap for a given number of hues.

    Args:
        hue_count (int): Number of hues present in one x-cluster.

    Returns:
        float: Gap between neighboring violin centers, excluding body width.
    """

    if hue_count >= SUPERPLOT_CROWDED_HUE_COUNT:
        return SUPERPLOT_HUE_GAP_TRIPLE
    return SUPERPLOT_HUE_GAP


def cluster_hue_offsets(hue_count: int) -> tuple[np.ndarray, float]:
    """Compute centered hue offsets and width for one x-cluster.

    Args:
        hue_count (int): Number of hues to place in the cluster.

    Returns:
        tuple[np.ndarray, float]: Offsets from the cluster center and the
        full cluster width.
    """

    count = max(int(hue_count), 1)
    hue_gap = superplot_hue_gap_for_count(count)
    hue_spacing = 2.0 * SUPERPLOT_HALF_WIDTH + hue_gap
    if count <= 1:
        return np.array([0.0], dtype=float), 0.0
    offsets = np.linspace(
        -(count - 1) * hue_spacing / 2.0,
        (count - 1) * hue_spacing / 2.0,
        count,
    )
    cluster_width = float((count - 1) * hue_spacing)
    return offsets, cluster_width


def present_cluster_hues(
    plot_data: pd.DataFrame,
    x: str,
    hue: str,
    x_value: str,
    hue_order: Sequence[str],
) -> list[str]:
    """Return hues that have observations in one x-cluster, in plotting order.

    Args:
        plot_data (pd.DataFrame): Plot-ready dataframe.
        x (str): X-axis grouping column.
        hue (str): Hue grouping column.
        x_value (str): Cluster label.
        hue_order (Sequence[str]): Global hue order.

    Returns:
        list[str]: Present hue labels for this cluster.
    """

    present = set(
        plot_data.loc[plot_data[x].astype(str) == str(x_value), hue].astype(str)
    )
    return [str(hue_value) for hue_value in hue_order if str(hue_value) in present]


def superplot_cluster_geometry(
    hue_order: Sequence[str],
) -> tuple[np.ndarray, float]:
    """Compute within-cluster hue offsets and spacing between x clusters.

    Args:
        hue_order (Sequence[str]): Hue categories in plotting order.

    Returns:
        tuple[np.ndarray, float]: Hue offsets from the cluster center and the
        distance between neighboring cluster centers.
    """

    hue_count = max(len(hue_order), 1)
    crowded = superplot_is_crowded(hue_order)
    cluster_gap = (
        SUPERPLOT_X_CLUSTER_GAP_CROWDED if crowded else SUPERPLOT_X_CLUSTER_GAP
    )
    offsets, cluster_width = cluster_hue_offsets(hue_count)
    base_spacing = cluster_width + cluster_gap
    return offsets, base_spacing


def get_group_centers(
    x_order: list[str],
    hue_order: list[str],
    plot_data: pd.DataFrame | None = None,
    x: str | None = None,
    hue: str | None = None,
) -> dict[tuple[str, str], float]:
    """Compute numeric x-centers for each observed x/hue pair.

    When ``plot_data`` is provided, missing hues are omitted per cluster and
    clusters are packed sequentially. Three-hue clusters use a wider gap so
    pairwise brackets have room; two-hue clusters stay compact.

    Args:
        x_order (list[str]): X categories in plotting order.
        hue_order (list[str]): Hue categories in plotting order.
        plot_data (pd.DataFrame | None): Plot dataframe used to drop empty hues.
        x (str | None): X-axis grouping column name.
        hue (str | None): Hue grouping column name.

    Returns:
        dict[tuple[str, str], float]: Mapping from ``(x, hue)`` to x-axis center.
    """

    if plot_data is None or x is None or hue is None:
        offsets, base_spacing = superplot_cluster_geometry(hue_order)
        return {
            (str(x_value), str(hue_value)): x_index * base_spacing + float(offset)
            for x_index, x_value in enumerate(x_order)
            for hue_value, offset in zip(hue_order, offsets)
        }

    crowded = superplot_is_crowded(hue_order)
    cluster_gap = (
        SUPERPLOT_X_CLUSTER_GAP_CROWDED if crowded else SUPERPLOT_X_CLUSTER_GAP
    )
    centers: dict[tuple[str, str], float] = {}
    cursor = 0.0
    started = False
    for x_value in x_order:
        present_hues = present_cluster_hues(
            plot_data=plot_data,
            x=x,
            hue=hue,
            x_value=str(x_value),
            hue_order=hue_order,
        )
        if not present_hues:
            continue
        offsets, cluster_width = cluster_hue_offsets(len(present_hues))
        if started:
            cursor += cluster_gap
        cluster_center = cursor + cluster_width / 2.0
        for hue_value, offset in zip(present_hues, offsets, strict=True):
            centers[(str(x_value), str(hue_value))] = cluster_center + float(offset)
        cursor += cluster_width
        started = True
    return centers


def superplot_layout_params(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    block: str,
    x_order_override: list[str] | None = None,
    hue_order_override: list[str] | None = None,
) -> tuple[tuple[float, float], bool, float]:
    """Choose figure size, tick rotation, and bottom margin from group counts.

    Args:
        data (pd.DataFrame): Source dataframe.
        x (str): X-axis grouping column.
        y (str): Numeric metric column.
        hue (str): Hue grouping column.
        block (str): Patient-identity column.
        x_order_override (list[str] | None): Optional explicit x order.
        hue_order_override (list[str] | None): Optional explicit hue order.

    Returns:
        tuple[tuple[float, float], bool, float]: Figure size, whether to rotate
        ticks, and the ``tight_layout`` bottom rect value.
    """

    plot_data = prepare_plot_data(data=data, x=x, y=y, hue=hue, block=block)
    x_order, hue_order, _ = get_category_orders(
        plot_data=plot_data,
        x=x,
        hue=hue,
        block=block,
        x_order_override=x_order_override,
        hue_order_override=hue_order_override,
    )
    crowded = superplot_is_crowded(hue_order)
    group_centers = get_group_centers(
        x_order=x_order,
        hue_order=hue_order,
        plot_data=plot_data,
        x=x,
        hue=hue,
    )
    if group_centers:
        positions = np.array(list(group_centers.values()), dtype=float)
        axis_span = float(np.max(positions) - np.min(positions)) + 1.0
    else:
        axis_span = 1.0
    inches_per_unit = 0.68 if crowded else 1.45
    maximum_width = 16.5 if crowded else 22.0
    width = float(np.clip(axis_span * inches_per_unit + 2.0, 14.0, maximum_width))
    height = SUPERPLOT_CROWDED_FIGURE_HEIGHT if crowded else SUPERPLOT_FIGURE_HEIGHT
    bottom = 0.24 if crowded else 0.06
    return (width, height), crowded, bottom


def condition_overview_group_centers(
    condition_order: Sequence[str],
) -> dict[tuple[str, str], float]:
    """Place CTRL and CAPN3 centers with wide overview violin bodies.

    Args:
        condition_order (Sequence[str]): Condition labels in tick order.

    Returns:
        dict[tuple[str, str], float]: Mapping from ``(condition, dummy hue)``
        to x-axis center.
    """

    max_half_width = CONDITION_OVERVIEW_HALF_WIDTH * SUPERPLOT_BODY_WIDTH_SCALE
    spacing = 2.0 * max_half_width + CONDITION_OVERVIEW_X_GAP
    return {
        (str(condition), CONDITION_OVERVIEW_HUE_VALUE): float(index) * spacing
        for index, condition in enumerate(condition_order)
    }


def prepare_condition_overview_plot_data(
    data: pd.DataFrame,
    y: str,
    condition_column: str,
    block: str,
    site_compartment_column: str,
) -> pd.DataFrame:
    """Prepare pooled CTRL-versus-CAPN3 rows with a dummy hue column.

    Args:
        data (pd.DataFrame): Source dataframe that already includes site×compartment
            labels.
        y (str): Numeric metric column.
        condition_column (str): CTRL/CAPN3 column name.
        block (str): Patient-identity column name.
        site_compartment_column (str): Combined site×compartment column name.

    Returns:
        pd.DataFrame: Plot-ready dataframe with ``CONDITION_OVERVIEW_HUE_COLUMN``.
    """

    plot_data = data.copy()
    plot_data[CONDITION_OVERVIEW_HUE_COLUMN] = CONDITION_OVERVIEW_HUE_VALUE
    plot_data = prepare_plot_data(
        data=plot_data,
        x=condition_column,
        y=y,
        hue=CONDITION_OVERVIEW_HUE_COLUMN,
        block=block,
    )
    if site_compartment_column not in plot_data.columns:
        raise KeyError(
            f"Condition overview plots require the '{site_compartment_column}' column."
        )
    plot_data = plot_data.dropna(subset=[site_compartment_column])
    plot_data[site_compartment_column] = plot_data[site_compartment_column].astype(str)
    plot_data[condition_column] = plot_data[condition_column].astype(str)
    return plot_data


def order_overview_stripes(
    plot_data: pd.DataFrame,
    condition_value: str,
    condition_column: str,
    site_compartment_column: str,
    block: str,
    genotype_column: str | None,
    site_compartment_order: Sequence[str],
    block_order: Sequence[str],
) -> list[tuple[str, str]]:
    """Order ``(site×compartment, patient)`` stripes inside one condition body.

    CAPN3 stripes are grouped by subtype, then site×compartment, then patient
    identifier. CTRL stripes skip subtype and use site×compartment then patient.

    Args:
        plot_data (pd.DataFrame): Plot-ready overview dataframe.
        condition_value (str): ``CTRL`` or ``CAPN3``.
        condition_column (str): Condition column name.
        site_compartment_column (str): Combined site×compartment column name.
        block (str): Patient-identity column name.
        genotype_column (str | None): Optional genotype column used for CAPN3
            subtype order.
        site_compartment_order (Sequence[str]): Canonical site×compartment order.
        block_order (Sequence[str]): Default patient identifier order.

    Returns:
        list[tuple[str, str]]: Ordered ``(site_compartment, patient)`` pairs.
    """

    group = plot_data.loc[
        plot_data[condition_column].astype(str) == str(condition_value)
    ]
    if group.empty:
        return []
    pairs = (
        group[[site_compartment_column, block]]
        .astype(str)
        .drop_duplicates()
    )
    present = {
        (str(site_label), str(patient_label))
        for site_label, patient_label in zip(
            pairs[site_compartment_column], pairs[block], strict=True
        )
    }
    site_rank = {
        str(label): index for index, label in enumerate(site_compartment_order)
    }
    patient_rank = {str(label): index for index, label in enumerate(block_order)}
    genotype_map: dict[str, str] = {}
    if genotype_column is not None:
        genotype_map = patient_genotype_map(
            plot_data=plot_data,
            block=block,
            genotype_column=genotype_column,
        )

    def stripe_sort_key(pair: tuple[str, str]) -> tuple[int, int, int, str, str]:
        """Return a sort key for one overview stripe.

        Args:
            pair (tuple[str, str]): ``(site_compartment, patient)`` identity.

        Returns:
            tuple[int, int, int, str, str]: Subtype, site, patient ranks and labels.
        """

        site_label, patient_label = pair
        subtype_rank = 0
        if str(condition_value) == CONDITION_ORDER[1]:
            subtype_rank = capn3_subtype_index(genotype_map.get(patient_label, ""))
        return (
            subtype_rank,
            site_rank.get(site_label, len(site_rank)),
            patient_rank.get(patient_label, len(patient_rank)),
            site_label,
            patient_label,
        )

    return sorted(present, key=stripe_sort_key)


def overview_legend_key_order(
    condition_order: Sequence[str],
    stripes_by_condition: Mapping[str, Sequence[tuple[str, str]]],
) -> list[tuple[str, str, str]]:
    """Return legend keys in superviolin stripe order.

    All CTRL stripes come first, then CAPN3 stripes in subtype, site×compartment,
    and patient order, matching the bodies drawn on the overview violins.

    Args:
        condition_order (Sequence[str]): Condition labels in tick order.
        stripes_by_condition (Mapping[str, Sequence[tuple[str, str]]]): Ordered
            ``(site_compartment, patient)`` stripes per condition.

    Returns:
        list[tuple[str, str, str]]: Keys as ``(site_compartment, condition, patient)``.
    """

    keys: list[tuple[str, str, str]] = []
    for condition_value in condition_order:
        for site_label, patient_label in stripes_by_condition.get(
            str(condition_value), ()
        ):
            keys.append((str(site_label), str(condition_value), str(patient_label)))
    return keys


def build_density_grid(values: np.ndarray) -> np.ndarray:
    """Build a KDE evaluation grid around observed values.

    The lower edge is clipped at the observed minimum, and at zero when all
    values are non-negative, so violin tails do not cross the x-axis.

    Args:
        values (np.ndarray): Numeric values from one plotted group.

    Returns:
        np.ndarray: Dense y-grid spanning the observed range with upper padding.
    """

    values = np.asarray(values, dtype=float)
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    spread = value_max - value_min
    if spread == 0:
        padding = max(abs(value_min) * 0.1, 1.0)
    else:
        padding = max(spread * 0.1, np.std(values) * 0.3, 1e-3)
    lower = value_min
    if value_min >= 0.0:
        lower = max(lower, 0.0)
    return np.linspace(lower, value_max + padding, SUPERPLOT_GRID_SIZE)


def estimate_density(values: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """Estimate non-negative density values on a y-grid.

    Density below zero is forced to zero when all observations are non-negative
    so violin bodies stay above the x-axis.

    Args:
        values (np.ndarray): Numeric observations.
        y_grid (np.ndarray): Shared y-grid.

    Returns:
        np.ndarray: Density values evaluated on ``y_grid``.
    """

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros_like(y_grid, dtype=float)
    if np.unique(values).size == 1:
        bandwidth = max(np.ptp(y_grid) * 0.03, 1e-3)
        density = np.exp(-0.5 * ((y_grid - values[0]) / bandwidth) ** 2)
    else:
        try:
            density = gaussian_kde(values)(y_grid)
        except (np.linalg.LinAlgError, ValueError):
            bandwidth = max(np.std(values) * 0.3, np.ptp(y_grid) * 0.03, 1e-3)
            density = np.zeros_like(y_grid, dtype=float)
            for value in values:
                density += np.exp(-0.5 * ((y_grid - value) / bandwidth) ** 2)
    density = np.clip(density, a_min=0, a_max=None)
    if float(np.min(values)) >= 0.0:
        density = np.where(y_grid >= 0.0, density, 0.0)
    return density


def clip_violin_support_at_zero(
    y_grid: np.ndarray,
    density_matrix: np.ndarray,
    total_density: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop KDE samples below zero so violin bodies stay above the x-axis.

    Args:
        y_grid (np.ndarray): Density evaluation coordinates.
        density_matrix (np.ndarray): Per-patient densities on ``y_grid``.
        total_density (np.ndarray): Combined density on ``y_grid``.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Clipped grid and densities.
    """

    keep = y_grid >= 0.0
    if np.all(keep):
        return y_grid, density_matrix, total_density
    if not np.any(keep):
        return y_grid[-1:], density_matrix[:, -1:], total_density[-1:]
    return y_grid[keep], density_matrix[:, keep], total_density[keep]


def apply_robust_y_limits(
    ax: plt.Axes,
    values: np.ndarray | pd.Series | list[float],
) -> None:
    """Set robust y-limits using quantiles to reduce outlier dominance.

    The lower limit is expanded when needed so the drawn violin body (clipped
    at zero for non-negative metrics) sits above the x-axis instead of
    crossing it.

    Args:
        ax (plt.Axes): Target axes for y-limit updates.
        values (np.ndarray | pd.Series | list[float]): Numeric values used to compute limits.

    Returns:
        None: This function mutates axis limits in place.
    """

    value_array = np.asarray(values, dtype=float)
    value_array = value_array[np.isfinite(value_array)]
    if value_array.size == 0:
        return
    observed_min = float(np.min(value_array))
    if value_array.size < 20:
        value_min = observed_min
        value_max = float(np.max(value_array))
    else:
        value_min = float(np.quantile(value_array, ROBUST_Y_LOWER_QUANTILE))
        value_max = float(np.quantile(value_array, ROBUST_Y_UPPER_QUANTILE))
    spread = max(value_max - value_min, max(abs(value_min) * 0.1, 1.0))
    padding = max(spread * 0.08, 1e-6)
    lower = value_min - padding
    upper = value_max + padding
    violin_floor = max(observed_min, 0.0) if observed_min >= 0.0 else observed_min
    lower = min(lower, violin_floor - padding)
    ax.set_ylim(lower, upper)


def get_dynamic_beeswarm_point_size(group_count: int, max_group_count: int) -> float:
    """Scale beeswarm marker size inversely with group density.

    Args:
        group_count (int): Number of points in the current group.
        max_group_count (int): Maximum group size across the subplot.

    Returns:
        float: Marker area for Matplotlib scatter.
    """

    if group_count <= 1 or max_group_count <= 1:
        return float(SUPERPLOT_POINT_SIZE_MAX)
    sparse_count_scale = np.clip((np.log10(max(group_count, 1)) - 1.0) / 2.0, 0.0, 1.0)
    relative_density = np.sqrt(group_count / max_group_count)
    density_scale = max(sparse_count_scale, 0.35 * relative_density)
    marker_size = SUPERPLOT_POINT_SIZE_MAX - (
        (SUPERPLOT_POINT_SIZE_MAX - SUPERPLOT_POINT_SIZE_MIN) * density_scale
    )
    return float(np.clip(marker_size, SUPERPLOT_POINT_SIZE_MIN, SUPERPLOT_POINT_SIZE_MAX))


def build_violin_like_swarm_positions(
    group_data: pd.DataFrame,
    y: str,
    y_grid: np.ndarray,
    half_width: np.ndarray,
    center: float,
    seed: int,
) -> np.ndarray:
    """Compute beeswarm x positions that follow a violin-like profile.

    Args:
        group_data (pd.DataFrame): Rows in one x/hue group.
        y (str): Numeric metric column name.
        y_grid (np.ndarray): Shared y-grid for density interpolation.
        half_width (np.ndarray): Group half-width per y-grid value.
        center (float): X-axis center of the current group.
        seed (int): Random seed for stable jittering.

    Returns:
        np.ndarray: X coordinates for each row in ``group_data``.
    """

    values = group_data[y].to_numpy(dtype=float)
    if values.size == 0:
        return np.array([], dtype=float)
    bin_count = max(12, min(45, int(np.sqrt(values.size) * 2)))
    bin_edges = np.linspace(float(np.min(y_grid)), float(np.max(y_grid)), bin_count + 1)
    bin_ids = np.clip(np.digitize(values, bin_edges) - 1, 0, bin_count - 1)
    point_x = np.full(values.shape, center, dtype=float)
    rng = np.random.default_rng(seed)
    for bin_id in np.unique(bin_ids):
        bin_indices = np.where(bin_ids == bin_id)[0]
        if bin_indices.size == 0:
            continue
        local_half_width = float(
            np.interp(np.mean(values[bin_indices]), y_grid, half_width)
        )
        if local_half_width <= 0:
            continue
        if bin_indices.size == 1:
            offsets = np.array([0.0], dtype=float)
        else:
            offsets = np.linspace(-local_half_width, local_half_width, bin_indices.size)
            offsets += rng.uniform(
                -local_half_width * 0.08,
                local_half_width * 0.08,
                size=bin_indices.size,
            )
        shuffled_indices = bin_indices[rng.permutation(bin_indices.size)]
        point_x[shuffled_indices] = center + offsets
    return point_x


def interpolate_stripe_center_x(
    y_grid: np.ndarray,
    left_edge: np.ndarray,
    right_edge: np.ndarray,
    y_value: float,
) -> float:
    """Return the horizontal midpoint of one violin stripe at a y-value.

    Args:
        y_grid (np.ndarray): Density evaluation coordinates.
        left_edge (np.ndarray): Left x-coordinates of the stripe on ``y_grid``.
        right_edge (np.ndarray): Right x-coordinates of the stripe on ``y_grid``.
        y_value (float): Vertical position of the mean marker.

    Returns:
        float: X-coordinate at the stripe center.
    """

    left_x = float(np.interp(y_value, y_grid, left_edge))
    right_x = float(np.interp(y_value, y_grid, right_edge))
    return (left_x + right_x) / 2.0


def add_superplot_mean_markers(
    ax: plt.Axes,
    center: float,
    means: Sequence[float],
    colors: Sequence[Any],
    x_positions: Sequence[float],
) -> None:
    """Draw ordered mean diamonds and the overall mean +/- SEM.

    Args:
        ax (plt.Axes): Axes receiving markers.
        center (float): X-position of the overall mean marker.
        means (Sequence[float]): Per-patient or per-stripe means.
        colors (Sequence[Any]): Diamond colors aligned with ``means``.
        x_positions (Sequence[float]): Diamond x-positions aligned with ``means``.

    Returns:
        None: This function draws artists on ``ax``.
    """

    plotted_means: list[float] = []
    for mean_value, color, x_position in zip(means, colors, x_positions, strict=True):
        plotted_means.append(float(mean_value))
        ax.scatter(
            float(x_position),
            float(mean_value),
            s=30,
            marker="D",
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=6,
        )
    if not plotted_means:
        return
    overall_mean = float(np.mean(plotted_means))
    sem = (
        float(np.std(plotted_means, ddof=1) / np.sqrt(len(plotted_means)))
        if len(plotted_means) > 1
        else 0.0
    )
    ax.errorbar(
        center,
        overall_mean,
        yerr=sem if sem > 0 else None,
        fmt="o",
        color="black",
        markersize=5,
        capsize=3,
        linewidth=1.3,
        zorder=7,
    )


def add_superplot_summary(
    ax: plt.Axes,
    group_data: pd.DataFrame,
    y: str,
    block: str,
    x_value: str,
    hue_value: str,
    center: float,
    condition_block_palette: dict[tuple[str, str, str], tuple[float, float, float]],
    block_order: Sequence[str] | None = None,
    spread: float = 0.09,
) -> None:
    """Overlay per-patient means and overall mean +/- SEM markers.

    Diamonds are placed left to right in ``block_order`` when provided, otherwise
    in palette-then-identifier order.

    Args:
        ax (plt.Axes): Axes receiving summary markers.
        group_data (pd.DataFrame): Group rows for a single x/hue pair.
        y (str): Numeric metric column.
        block (str): Patient-identity column.
        x_value (str): X-axis category label for the current group.
        hue_value (str): Hue label for the current group.
        center (float): X-axis center of the group.
        condition_block_palette (dict[tuple[str, str, str], tuple[float, float, float]]):
            Colors keyed by ``(x, hue, patient)``.
        block_order (Sequence[str] | None): Optional patient order matching the
            plotted stripes or legend.
        spread (float): Half-width used to space diamonds around ``center``.

    Returns:
        None: This function draws artists on ``ax``.
    """

    present = set(group_data[block].astype(str).unique())
    if block_order is None:
        available_blocks = sorted(
            present,
            key=lambda value: (
                (str(x_value), str(hue_value), value) not in condition_block_palette,
                value,
            ),
        )
    else:
        available_blocks = [
            str(label) for label in block_order if str(label) in present
        ]
        seen = set(available_blocks)
        available_blocks.extend(label for label in sorted(present) if label not in seen)
    means: list[float] = []
    colors: list[Any] = []
    for block_label in available_blocks:
        block_values = group_data.loc[
            group_data[block].astype(str) == block_label, y
        ].dropna().to_numpy()
        if block_values.size == 0:
            continue
        means.append(float(np.mean(block_values)))
        colors.append(
            condition_block_palette.get(
                (str(x_value), str(hue_value), str(block_label)),
                "0.4",
            )
        )
    if not means:
        return
    offsets = (
        [0.0]
        if len(means) == 1
        else np.linspace(-float(spread), float(spread), len(means))
    )
    x_positions = [center + float(offset) for offset in offsets]
    add_superplot_mean_markers(
        ax=ax,
        center=center,
        means=means,
        colors=colors,
        x_positions=x_positions,
    )


def normalize_superplot_annotations(
    superplot_annotations: dict[str, str] | list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    """Normalize optional annotation inputs into structured records.

    Args:
        superplot_annotations (dict[str, str] | list[dict[str, str]] | None): Input annotations.

    Returns:
        list[dict[str, str]]: Normalized annotation records.
    """

    if superplot_annotations is None:
        return []
    if isinstance(superplot_annotations, dict):
        return [
            {"x": str(x_value), "label": str(label)}
            for x_value, label in superplot_annotations.items()
            if str(label).strip()
        ]
    return [
        {str(key): str(value) for key, value in record.items()}
        for record in superplot_annotations
        if (
            str(record.get("label", "")).strip()
            or str(record.get("mean_label", "")).strip()
            or str(record.get("median_label", "")).strip()
        )
    ]


def superplot_annotation_text_items(record: dict[str, str]) -> list[tuple[str, str]]:
    """Extract annotation text lines and colors in drawing order.

    Args:
        record (dict[str, str]): Annotation record with label or mean/median fields.

    Returns:
        list[tuple[str, str]]: Sequence of ``(text, color)`` entries.
    """

    items: list[tuple[str, str]] = []
    mean_label = str(record.get("mean_label", "")).strip()
    if mean_label:
        items.append(
            (
                mean_label,
                str(record.get("mean_color", record.get("color", "purple"))).strip()
                or "purple",
            )
        )
    median_label = str(record.get("median_label", "")).strip()
    if median_label:
        items.append(
            (
                median_label,
                str(
                    record.get("median_color", record.get("color", "black"))
                ).strip()
                or "black",
            )
        )
    if items:
        return items
    label = str(record.get("label", "")).strip()
    if not label:
        return []
    return [(label, str(record.get("color", "black")).strip() or "black")]


def split_effect_summary_annotation(label: str) -> tuple[str, str, str] | None:
    """Split effect-summary text into estimate, interval, and probability spans.

    Args:
        label (str): Annotation text, e.g. ``-0.2 [-0.4, -0.1] 97.2%``.

    Returns:
        tuple[str, str, str] | None: Prefix, bracket interval, and suffix if parsable.
    """

    text = str(label)
    left_bracket = text.find("[")
    right_bracket = text.find("]", left_bracket + 1)
    if left_bracket < 0 or right_bracket < 0:
        return None
    prefix = text[:left_bracket].rstrip()
    hdi_text = text[left_bracket : right_bracket + 1]
    suffix = text[right_bracket + 1 :].lstrip()
    if not prefix or not suffix:
        return None
    return prefix, hdi_text, suffix


def draw_superplot_annotation_text(
    ax: plt.Axes,
    x_center: float,
    y_position: float,
    label: str,
    text_color: str,
    transform: Any,
) -> None:
    """Draw one annotation line with optional HDI styling.

    Args:
        ax (plt.Axes): Destination axes.
        x_center (float): Text center x-coordinate.
        y_position (float): Text baseline y-coordinate in transform space.
        label (str): Annotation line.
        text_color (str): Main text color.
        transform (Any): Matplotlib transform used for x-axis-relative placement.

    Returns:
        None: This function adds text artists to ``ax``.
    """

    effect_parts = split_effect_summary_annotation(label)
    if effect_parts is None:
        ax.text(
            x_center,
            y_position,
            label,
            ha="center",
            va="bottom",
            fontsize=SUPERPLOT_ANNOTATION_FONT_SIZE,
            color=text_color,
            transform=transform,
            clip_on=False,
            zorder=11,
        )
        return
    prefix, hdi_text, suffix = effect_parts
    text_box = HPacker(
        children=[
            TextArea(
                f"{prefix} ",
                textprops={
                    "fontsize": SUPERPLOT_ANNOTATION_FONT_SIZE,
                    "color": text_color,
                },
            ),
            TextArea(
                hdi_text,
                textprops={
                    "fontsize": SUPERPLOT_ANNOTATION_HDI_FONT_SIZE,
                    "color": SUPERPLOT_ANNOTATION_HDI_COLOR,
                },
            ),
            TextArea(
                f" {suffix}",
                textprops={
                    "fontsize": SUPERPLOT_ANNOTATION_FONT_SIZE,
                    "color": text_color,
                },
            ),
        ],
        align="baseline",
        pad=0,
        sep=0,
    )
    annotation_box = AnnotationBbox(
        text_box,
        (x_center, y_position),
        xycoords=transform,
        box_alignment=(0.5, 0.0),
        frameon=False,
        pad=0,
        annotation_clip=False,
        zorder=11,
    )
    ax.add_artist(annotation_box)


def add_superplot_annotations(
    ax: plt.Axes,
    annotation_records: list[dict[str, str]],
    group_centers: dict[tuple[str, str], float],
    hue_order: list[str],
) -> float:
    """Draw bracket annotations and avoid label collisions between nearby groups.

    Args:
        ax (plt.Axes): Axes receiving annotations.
        annotation_records (list[dict[str, str]]): Bracket/text records.
        group_centers (dict[tuple[str, str], float]): X-center map for plotted groups.
        hue_order (list[str]): Hue ordering used in the subplot.

    Returns:
        float: Highest y-position used in axis-transform coordinates.
    """

    if not annotation_records:
        return 1.0
    default_hue_start = str(hue_order[0]) if hue_order else ""
    default_hue_end = (
        str(hue_order[1]) if len(hue_order) >= 2 else default_hue_start
    )
    xaxis_transform = ax.get_xaxis_transform()
    placed_boxes: list[tuple[float, float, float, float]] = []
    annotation_counts_by_x: dict[str, int] = {}
    max_top = 1.0
    for record in annotation_records:
        x_value = str(record.get("x", ""))
        text_items = superplot_annotation_text_items(record)
        if not x_value or not text_items:
            continue
        bracket_color = str(record.get("bracket_color", "black")).strip() or "black"
        hue_start = str(record.get("hue_start", default_hue_start))
        hue_end = str(record.get("hue_end", default_hue_end))
        x_end_value = str(record.get("x_end", x_value))
        start_key = (x_value, hue_start)
        end_key = (x_end_value, hue_end)
        if start_key not in group_centers or end_key not in group_centers:
            continue
        x_start = min(group_centers[start_key], group_centers[end_key])
        x_end = max(group_centers[start_key], group_centers[end_key])
        if x_start == x_end:
            continue
        annotation_index = annotation_counts_by_x.get(x_value, 0)
        annotation_counts_by_x[x_value] = annotation_index + 1
        y_line = SUPERPLOT_ANNOTATION_BASE_Y + SUPERPLOT_ANNOTATION_STACK_GAP * (
            annotation_index
        )
        line_count = len(text_items)
        estimated_height = (
            SUPERPLOT_ANNOTATION_BRACKET_HEIGHT
            + SUPERPLOT_ANNOTATION_TEXT_OFFSET
            + SUPERPLOT_ANNOTATION_TEXT_LINE_GAP * max(line_count - 1, 0)
            + SUPERPLOT_ANNOTATION_BOX_PADDING
        )
        while any(
            (
                x_start <= other_x1
                and x_end >= other_x0
                and y_line < other_y1
                and (y_line + estimated_height) > other_y0
            )
            for other_x0, other_x1, other_y0, other_y1 in placed_boxes
        ):
            y_line += SUPERPLOT_ANNOTATION_STACK_GAP
        y_bracket_top = y_line + SUPERPLOT_ANNOTATION_BRACKET_HEIGHT
        ax.plot(
            [x_start, x_start, x_end, x_end],
            [y_line, y_bracket_top, y_bracket_top, y_line],
            color=bracket_color,
            linewidth=1.1,
            transform=xaxis_transform,
            clip_on=False,
            zorder=10,
        )
        y_text = y_bracket_top + SUPERPLOT_ANNOTATION_TEXT_OFFSET
        for text_index, (text_label, text_color) in enumerate(text_items):
            draw_superplot_annotation_text(
                ax=ax,
                x_center=(x_start + x_end) / 2.0,
                y_position=y_text + SUPERPLOT_ANNOTATION_TEXT_LINE_GAP * text_index,
                label=text_label,
                text_color=text_color,
                transform=xaxis_transform,
            )
        top_y = (
            y_text
            + SUPERPLOT_ANNOTATION_TEXT_LINE_GAP * max(line_count - 1, 0)
            + SUPERPLOT_ANNOTATION_BOX_PADDING
        )
        placed_boxes.append((x_start, x_end, y_line, top_y))
        max_top = max(max_top, top_y)
    return max_top


def superplot_top_margin(annotation_top: float) -> float:
    """Return a tight layout top margin that preserves annotation headroom.

    One-row CTRL-versus-CAPN3 annotations keep the original margin. Stacked
    subtype brackets keep most of the data area; the title is placed from the
    actual axes position after layout rather than by shrinking this rect.

    Args:
        annotation_top (float): Highest annotation y-position in axis-transform coordinates.

    Returns:
        float: ``tight_layout`` top rect value.
    """

    overflow = max(0.0, float(annotation_top) - 1.0)
    extra = max(0.0, overflow - SUPERPLOT_ANNOTATION_SINGLE_OVERFLOW)
    if extra <= 0.0:
        return max(0.90, 0.96 - 0.07 * overflow)
    return max(0.88, 0.96 - 0.04 * extra)


def superplot_title_y(
    annotation_top: float,
    bottom_margin: float = 0.06,
    axes: plt.Axes | None = None,
) -> float:
    """Return a figure-level title y-position just above stacked annotations.

    The previous mapping used the ``tight_layout`` rect as if it were the data
    axes. Rotated tick labels make the real axes much shorter and lower, so
    that formula placed the title far above the rendered brackets. When
    ``axes`` is provided after layout, the title sits a small pad above the
    true annotation top.

    Args:
        annotation_top (float): Highest annotation y-position in axis-transform coordinates.
        bottom_margin (float): Unused fallback when ``axes`` is omitted.
        axes (plt.Axes | None): Laid-out axes used to map axis coordinates
            into figure coordinates.

    Returns:
        float: ``suptitle`` y-value in figure coordinates.
    """

    if axes is not None:
        position = axes.get_position()
        annotation_fig_y = float(position.y0) + float(annotation_top) * float(
            position.height
        )
        return annotation_fig_y + SUPERPLOT_TITLE_ANNOTATION_GAP
    overflow = max(0.0, float(annotation_top) - 1.0)
    extra = max(0.0, overflow - SUPERPLOT_ANNOTATION_SINGLE_OVERFLOW)
    if extra <= 0.0:
        return SUPERPLOT_TITLE_Y
    top = superplot_top_margin(annotation_top)
    axes_height = max(top - float(bottom_margin), 0.01)
    return top + 0.10 * overflow * axes_height + SUPERPLOT_TITLE_ANNOTATION_GAP


def style_superplot_axis(
    ax: plt.Axes,
    x: str,
    y: str,
    tick_positions: list[float],
    tick_labels: list[str],
    unit_dict: dict[str, str] | None,
    y_values: np.ndarray,
    title: str | None,
    rotate_ticks: bool = False,
) -> None:
    """Apply shared axis styling for one superplot panel.

    Args:
        ax (plt.Axes): Target axis.
        x (str): X-axis variable name, retained for call-site compatibility.
        y (str): Y-axis variable name, retained for call-site compatibility.
        tick_positions (list[float]): Group center positions.
        tick_labels (list[str]): Tick labels per center.
        unit_dict (dict[str, str] | None): Unused; y-axis labels are omitted.
        y_values (np.ndarray): Numeric values used for robust y-limits.
        title (str | None): Unused; titles are drawn as figure-level ``suptitle``.
        rotate_ticks (bool): Whether to rotate x-tick labels to avoid overlap.

    Returns:
        None: This function mutates ``ax``.
    """

    ax.set_xticks(tick_positions)
    if rotate_ticks:
        ax.set_xticklabels(
            tick_labels,
            rotation=40,
            ha="right",
            fontsize=SUPERPLOT_X_TICK_FONT_SIZE,
        )
    else:
        ax.set_xticklabels(
            tick_labels,
            rotation=0,
            ha="center",
            fontsize=SUPERPLOT_X_TICK_FONT_SIZE,
        )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=SUPERPLOT_Y_TICK_FONT_SIZE)
    apply_robust_y_limits(ax=ax, values=y_values)
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    sns.despine(ax=ax)


def add_block_legend(
    ax: plt.Axes,
    condition_block_palette: dict[tuple[str, str, str], tuple[float, float, float]],
    block_legend_labels: dict[tuple[str, str, str], str],
    outside: bool = False,
    key_order: Sequence[tuple[str, str, str]] | None = None,
) -> None:
    """Draw a patient-color legend on one axis.

    Args:
        ax (plt.Axes): Axis where legend will be drawn.
        condition_block_palette (dict[tuple[str, str, str], tuple[float, float, float]]):
            ``(x, hue, patient)`` color mapping.
        block_legend_labels (dict[tuple[str, str, str], str]): ``(x, hue, patient)``
            label mapping.
        outside (bool): When ``True``, place the legend to the right of the axes
            instead of overlaying the data.
        key_order (Sequence[tuple[str, str, str]] | None): Optional explicit
            ``(x, hue, patient)`` order. When omitted, palette insertion order
            is used.

    Returns:
        None: This function adds legend artists to ``ax``.
    """

    if not condition_block_palette:
        return
    ordered_keys = (
        list(key_order)
        if key_order is not None
        else list(condition_block_palette.keys())
    )
    handles = []
    for x_value, hue_value, block_label in ordered_keys:
        key = (str(x_value), str(hue_value), str(block_label))
        if key not in block_legend_labels or key not in condition_block_palette:
            continue
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=condition_block_palette[key],
                markeredgecolor="black",
                markeredgewidth=0.4,
                markersize=6,
                label=block_legend_labels.get(
                    key,
                    f"{x_value} {hue_value} {block_label}",
                ),
            )
        )
    if not handles:
        return
    if outside:
        legend = ax.legend(
            handles=handles,
            frameon=True,
            fontsize=CONDITION_OVERVIEW_LEGEND_FONT_SIZE,
            title="Patient",
            title_fontsize=CONDITION_OVERVIEW_LEGEND_TITLE_FONT_SIZE,
            loc="upper left",
            bbox_to_anchor=CONDITION_OVERVIEW_LEGEND_BBOX,
            borderaxespad=0.0,
            ncol=1,
        )
        legend._legend_box.align = "center"
        return
    ax.legend(
        handles=handles,
        frameon=True,
        fontsize=8,
        title="Patient",
        title_fontsize=9,
        loc="upper right",
    )


def render_super_violin_on_ax(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    block: str,
    unit_dict: dict[str, str] | None = None,
    title_override: str | None = None,
    superplot_annotations: dict[str, str] | list[dict[str, str]] | None = None,
    x_order_override: list[str] | None = None,
    hue_order_override: list[str] | None = None,
    show_legend: bool = False,
    genotype_column: str | None = None,
) -> float:
    """Render one superviolin panel on a provided axis.

    Args:
        ax (plt.Axes): Destination axis.
        data (pd.DataFrame): Source dataframe.
        x (str): X-axis grouping column.
        y (str): Numeric metric column.
        hue (str): Hue grouping column.
        block (str): Patient-identity column.
        unit_dict (dict[str, str] | None): Optional y-unit mapping.
        title_override (str | None): Optional custom panel title.
        superplot_annotations (dict[str, str] | list[dict[str, str]] | None): Optional
            bracket labels.
        x_order_override (list[str] | None): Optional explicit x order.
        hue_order_override (list[str] | None): Optional explicit hue order.
        show_legend (bool): Whether to draw a patient legend on this panel.
        genotype_column (str | None): Optional genotype column used to order
            CAPN3 patients and thicken between-subtype separators.

    Returns:
        float: Highest annotation y-position in axis-transform coordinates.
    """

    plot_data = prepare_plot_data(data=data, x=x, y=y, hue=hue, block=block)
    if plot_data.empty:
        ax.set_visible(False)
        return 1.0
    x_order, hue_order, block_order = get_category_orders(
        plot_data=plot_data,
        x=x,
        hue=hue,
        block=block,
        x_order_override=x_order_override,
        hue_order_override=hue_order_override,
    )
    if not x_order or not hue_order or not block_order:
        ax.set_visible(False)
        return 1.0
    annotation_records = normalize_superplot_annotations(superplot_annotations)
    genotype_map: dict[str, str] = {}
    if genotype_column is not None:
        genotype_map = patient_genotype_map(
            plot_data=plot_data,
            block=block,
            genotype_column=genotype_column,
        )
    condition_block_palette = create_condition_block_palette(
        block_order=block_order,
        x_order=x_order,
        hue_order=hue_order,
        plot_data=plot_data,
        x=x,
        hue=hue,
        block=block,
        genotype_column=genotype_column,
    )
    block_legend_labels = build_block_legend_labels(
        plot_data=plot_data,
        x=x,
        block=block,
        hue=hue,
        x_order=x_order,
        hue_order=hue_order,
        block_order=block_order,
    )
    group_centers = get_group_centers(
        x_order=x_order,
        hue_order=hue_order,
        plot_data=plot_data,
        x=x,
        hue=hue,
    )
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    group_profiles: list[dict[str, object]] = []
    global_density_max = 0.0
    for (x_value, hue_value), center in group_centers.items():
        group_data = plot_data[
            (plot_data[x].astype(str) == x_value)
            & (plot_data[hue].astype(str) == hue_value)
        ]
        if group_data.empty:
            continue
        values = group_data[y].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        y_grid = build_density_grid(values)
        density_rows: list[np.ndarray] = []
        stripe_blocks: list[str] = []
        ordered_blocks = order_group_blocks(
            plot_data=plot_data,
            x=x,
            hue=hue,
            block=block,
            x_value=str(x_value),
            hue_value=str(hue_value),
            block_order=block_order,
            genotype_column=genotype_column,
        )
        for block_label in ordered_blocks:
            block_values = group_data.loc[
                group_data[block].astype(str) == block_label, y
            ].dropna().to_numpy(dtype=float)
            if block_values.size == 0:
                continue
            density_rows.append(estimate_density(block_values, y_grid) * block_values.size)
            stripe_blocks.append(block_label)
        if not density_rows:
            continue
        density_matrix = np.vstack(density_rows)
        total_density = density_matrix.sum(axis=0)
        if np.allclose(total_density.max(), 0):
            continue
        global_density_max = max(global_density_max, float(total_density.max()))
        group_profiles.append(
            {
                "center": center,
                "x_value": x_value,
                "hue_value": hue_value,
                "group_data": group_data,
                "y_grid": y_grid,
                "density_matrix": density_matrix,
                "total_density": total_density,
                "stripe_blocks": stripe_blocks,
            }
        )
    if global_density_max == 0.0:
        ax.set_visible(False)
        return 1.0
    for profile in group_profiles:
        center = float(profile["center"])
        x_value = str(profile["x_value"])
        hue_value = str(profile["hue_value"])
        group_data = profile["group_data"]
        y_grid = np.asarray(profile["y_grid"], dtype=float)
        density_matrix = np.asarray(profile["density_matrix"], dtype=float)
        total_density = np.asarray(profile["total_density"], dtype=float)
        group_values = group_data[y].to_numpy(dtype=float)
        group_values = group_values[np.isfinite(group_values)]
        if group_values.size > 0 and float(np.min(group_values)) >= 0.0:
            y_grid, density_matrix, total_density = clip_violin_support_at_zero(
                y_grid,
                density_matrix,
                total_density,
            )
        stripe_blocks = list(profile["stripe_blocks"])
        half_width = (
            SUPERPLOT_HALF_WIDTH
            * SUPERPLOT_BODY_WIDTH_SCALE
            * (total_density / global_density_max)
        )
        left_edge = center - half_width
        right_edge = center + half_width
        current_left = left_edge.copy()
        full_width = right_edge - left_edge
        stripe_means: list[float] = []
        stripe_colors: list[Any] = []
        stripe_x_positions: list[float] = []
        for index, (block_label, block_density) in enumerate(
            zip(stripe_blocks, density_matrix, strict=True)
        ):
            density_fraction = np.divide(
                block_density,
                total_density,
                out=np.zeros_like(block_density),
                where=total_density > 0,
            )
            stripe_right = current_left + full_width * density_fraction
            stripe_color = condition_block_palette[
                (str(x_value), str(hue_value), str(block_label))
            ]
            ax.fill_betweenx(
                y_grid,
                current_left,
                stripe_right,
                color=stripe_color,
                alpha=0.85,
                linewidth=0,
                zorder=2,
                clip_on=True,
            )
            block_values = group_data.loc[
                group_data[block].astype(str) == str(block_label), y
            ].dropna().to_numpy(dtype=float)
            if block_values.size > 0:
                mean_y = float(np.mean(block_values))
                stripe_means.append(mean_y)
                stripe_colors.append(stripe_color)
                stripe_x_positions.append(
                    interpolate_stripe_center_x(
                        y_grid=y_grid,
                        left_edge=current_left,
                        right_edge=stripe_right,
                        y_value=mean_y,
                    )
                )
            if index < len(stripe_blocks) - 1:
                ax.plot(
                    stripe_right,
                    y_grid,
                    color="black",
                    linewidth=superviolin_separator_linewidth(
                        current_block=str(block_label),
                        next_block=str(stripe_blocks[index + 1]),
                        hue_value=hue_value,
                        genotype_map=genotype_map,
                    ),
                    zorder=3,
                )
            current_left = stripe_right
        ax.plot(left_edge, y_grid, color="black", linewidth=1.0, zorder=3)
        ax.plot(right_edge, y_grid, color="black", linewidth=1.0, zorder=3)
        add_superplot_mean_markers(
            ax=ax,
            center=center,
            means=stripe_means,
            colors=stripe_colors,
            x_positions=stripe_x_positions,
        )
        tick_positions.append(center)
        tick_labels.append(f"{x_value}\n{format_condition_display_label(hue_value)}")
    annotation_top = add_superplot_annotations(
        ax=ax,
        annotation_records=annotation_records,
        group_centers=group_centers,
        hue_order=hue_order,
    )
    style_superplot_axis(
        ax=ax,
        x=x,
        y=y,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        unit_dict=unit_dict,
        y_values=plot_data[y].to_numpy(dtype=float),
        title=title_override,
        rotate_ticks=superplot_is_crowded(hue_order),
    )
    if show_legend:
        add_block_legend(
            ax=ax,
            condition_block_palette=condition_block_palette,
            block_legend_labels=block_legend_labels,
        )
    return annotation_top


def render_super_beeswarm_on_ax(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    block: str,
    unit_dict: dict[str, str] | None = None,
    title_override: str | None = None,
    superplot_annotations: dict[str, str] | list[dict[str, str]] | None = None,
    x_order_override: list[str] | None = None,
    hue_order_override: list[str] | None = None,
    show_legend: bool = False,
) -> float:
    """Render one superbeeswarm panel on a provided axis.

    Args:
        ax (plt.Axes): Destination axis.
        data (pd.DataFrame): Source dataframe.
        x (str): X-axis grouping column.
        y (str): Numeric metric column.
        hue (str): Hue grouping column.
        block (str): Patient-identity column.
        unit_dict (dict[str, str] | None): Optional y-unit mapping.
        title_override (str | None): Optional custom panel title.
        superplot_annotations (dict[str, str] | list[dict[str, str]] | None): Optional
            bracket labels.
        x_order_override (list[str] | None): Optional explicit x order.
        hue_order_override (list[str] | None): Optional explicit hue order.
        show_legend (bool): Whether to draw a patient legend on this panel.

    Returns:
        float: Highest annotation y-position in axis-transform coordinates.
    """

    plot_data = prepare_plot_data(data=data, x=x, y=y, hue=hue, block=block)
    if plot_data.empty:
        ax.set_visible(False)
        return 1.0
    x_order, hue_order, block_order = get_category_orders(
        plot_data=plot_data,
        x=x,
        hue=hue,
        block=block,
        x_order_override=x_order_override,
        hue_order_override=hue_order_override,
    )
    if not x_order or not hue_order or not block_order:
        ax.set_visible(False)
        return 1.0
    annotation_records = normalize_superplot_annotations(superplot_annotations)
    condition_block_palette = create_condition_block_palette(
        block_order=block_order,
        x_order=x_order,
        hue_order=hue_order,
        plot_data=plot_data,
        x=x,
        hue=hue,
        block=block,
    )
    block_legend_labels = build_block_legend_labels(
        plot_data=plot_data,
        x=x,
        block=block,
        hue=hue,
        x_order=x_order,
        hue_order=hue_order,
        block_order=block_order,
    )
    group_centers = get_group_centers(
        x_order=x_order,
        hue_order=hue_order,
        plot_data=plot_data,
        x=x,
        hue=hue,
    )
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    group_profiles: list[dict[str, object]] = []
    global_density_max = 0.0
    for group_index, ((x_value, hue_value), center) in enumerate(group_centers.items()):
        group_data = plot_data[
            (plot_data[x].astype(str) == x_value)
            & (plot_data[hue].astype(str) == hue_value)
        ]
        if group_data.empty:
            continue
        values = group_data[y].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        y_grid = build_density_grid(values)
        scaled_density = estimate_density(values, y_grid) * values.size
        if np.allclose(scaled_density.max(), 0):
            continue
        global_density_max = max(global_density_max, float(scaled_density.max()))
        group_profiles.append(
            {
                "group_index": group_index,
                "center": center,
                "x_value": x_value,
                "hue_value": hue_value,
                "group_data": group_data.reset_index(drop=True),
                "y_grid": y_grid,
                "scaled_density": scaled_density,
            }
        )
    if global_density_max == 0.0:
        ax.set_visible(False)
        return 1.0
    max_group_count = max(len(profile["group_data"]) for profile in group_profiles)
    for profile in group_profiles:
        group_index = int(profile["group_index"])
        center = float(profile["center"])
        x_value = str(profile["x_value"])
        hue_value = str(profile["hue_value"])
        group_data = profile["group_data"]
        y_grid = np.asarray(profile["y_grid"], dtype=float)
        scaled_density = np.asarray(profile["scaled_density"], dtype=float)
        point_size = get_dynamic_beeswarm_point_size(
            group_count=len(group_data), max_group_count=max_group_count
        )
        half_width = (
            SUPERPLOT_HALF_WIDTH
            * SUPERPLOT_BODY_WIDTH_SCALE
            * (scaled_density / global_density_max)
        )
        point_x = build_violin_like_swarm_positions(
            group_data=group_data,
            y=y,
            y_grid=y_grid,
            half_width=half_width,
            center=center,
            seed=group_index * 10_000,
        )
        for block_label in block_order:
            block_mask = group_data[block].astype(str) == block_label
            if not block_mask.any():
                continue
            ax.scatter(
                point_x[block_mask.to_numpy()],
                group_data.loc[block_mask, y].to_numpy(dtype=float),
                s=point_size,
                color=condition_block_palette[
                    (str(x_value), str(hue_value), str(block_label))
                ],
                edgecolor="white",
                linewidth=0.2,
                alpha=SUPERPLOT_POINT_ALPHA,
                zorder=3,
            )
        add_superplot_summary(
            ax=ax,
            group_data=group_data,
            y=y,
            block=block,
            x_value=x_value,
            hue_value=hue_value,
            center=center,
            condition_block_palette=condition_block_palette,
            block_order=block_order,
            spread=float(np.max(half_width)) * SUPERPLOT_SUMMARY_SPREAD_FRACTION,
        )
        tick_positions.append(center)
        tick_labels.append(f"{x_value}\n{format_condition_display_label(hue_value)}")
    annotation_top = add_superplot_annotations(
        ax=ax,
        annotation_records=annotation_records,
        group_centers=group_centers,
        hue_order=hue_order,
    )
    style_superplot_axis(
        ax=ax,
        x=x,
        y=y,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        unit_dict=unit_dict,
        y_values=plot_data[y].to_numpy(dtype=float),
        title=title_override,
        rotate_ticks=superplot_is_crowded(hue_order),
    )
    if show_legend:
        add_block_legend(
            ax=ax,
            condition_block_palette=condition_block_palette,
            block_legend_labels=block_legend_labels,
        )
    return annotation_top


def finalize_superplot_figure(
    figure: Figure,
    annotation_top: float,
    title: str | None,
    output_path: Path | None,
    bottom_margin: float,
    right_margin: float = 1.0,
) -> Path | None:
    """Apply title, margins, and optional save for one superplot figure.

    Args:
        figure (Figure): Matplotlib figure containing the superplot.
        annotation_top (float): Highest annotation y-position in axis coordinates.
        title (str | None): Optional figure-level title drawn above annotations.
        output_path (Path | None): Destination PNG, or ``None`` to skip saving.
        bottom_margin (float): ``tight_layout`` bottom rect value.
        right_margin (float): ``tight_layout`` right rect value. Values below
            ``1.0`` leave space for an outside legend.

    Returns:
        Path | None: Saved PNG path when ``output_path`` is provided.
    """

    figure.tight_layout(
        rect=(
            0.0,
            bottom_margin,
            float(right_margin),
            superplot_top_margin(annotation_top),
        )
    )
    if title:
        axes = figure.axes[0] if figure.axes else None
        title_x = 0.5
        if axes is not None and float(right_margin) < 1.0:
            position = axes.get_position()
            title_x = float(position.x0 + position.x1) / 2.0
        figure.suptitle(
            title,
            x=title_x,
            ha="center",
            fontsize=SUPERPLOT_TITLE_FONT_SIZE,
            y=superplot_title_y(
                annotation_top,
                bottom_margin=bottom_margin,
                axes=axes,
            ),
            verticalalignment="bottom",
        )
    if output_path is not None:
        return save_figure(
            figure=figure,
            output_path=output_path,
            dpi=SUPERPLOT_DPI,
            close=True,
        )
    plt.close(figure)
    return None


def plot_super_violin(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    block: str,
    unit_dict: dict[str, str] | None = None,
    save_dir: str | Path | None = None,
    title_override: str | None = None,
    filename_prefix: str | None = None,
    superplot_annotations: dict[str, str] | list[dict[str, str]] | None = None,
    x_order_override: list[str] | None = None,
    hue_order_override: list[str] | None = None,
    output_dir_suffix: str = "",
    genotype_column: str | None = None,
) -> Path | None:
    """Render and save one standalone superviolin figure.

    Args:
        data (pd.DataFrame): Source dataframe.
        x (str): X-axis grouping column.
        y (str): Numeric metric column.
        hue (str): Hue grouping column.
        block (str): Patient-identity column.
        unit_dict (dict[str, str] | None): Optional y-unit mapping.
        save_dir (str | Path | None): Base output directory.
        title_override (str | None): Optional title override.
        filename_prefix (str | None): Optional filename prefix.
        superplot_annotations (dict[str, str] | list[dict[str, str]] | None): Optional
            bracket labels.
        x_order_override (list[str] | None): Optional explicit x order.
        hue_order_override (list[str] | None): Optional explicit hue order.
        output_dir_suffix (str): Optional suffix appended to generated figure
            directories.
        genotype_column (str | None): Optional genotype column used to order
            CAPN3 patients and thicken between-subtype separators.

    Returns:
        Path | None: Saved PNG path when ``save_dir`` is provided.
    """

    figsize, _rotate_ticks, bottom_margin = superplot_layout_params(
        data=data,
        x=x,
        y=y,
        hue=hue,
        block=block,
        x_order_override=x_order_override,
        hue_order_override=hue_order_override,
    )
    fig, ax = plt.subplots(figsize=figsize)
    annotation_top = render_super_violin_on_ax(
        ax=ax,
        data=data,
        x=x,
        y=y,
        hue=hue,
        block=block,
        unit_dict=unit_dict,
        title_override=title_override,
        superplot_annotations=superplot_annotations,
        x_order_override=x_order_override,
        hue_order_override=hue_order_override,
        show_legend=False,
        genotype_column=genotype_column,
    )
    output_path = build_output_path(
        y=y,
        x=x,
        hue=hue,
        save_dir=save_dir,
        suffix="superviolin",
        filename_prefix=filename_prefix,
        output_dir_suffix=output_dir_suffix,
    )
    return finalize_superplot_figure(
        figure=fig,
        annotation_top=annotation_top,
        title=title_override,
        output_path=output_path,
        bottom_margin=bottom_margin,
    )


def plot_super_beeswarm(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    block: str,
    unit_dict: dict[str, str] | None = None,
    save_dir: str | Path | None = None,
    title_override: str | None = None,
    filename_prefix: str | None = None,
    superplot_annotations: dict[str, str] | list[dict[str, str]] | None = None,
    x_order_override: list[str] | None = None,
    hue_order_override: list[str] | None = None,
    output_dir_suffix: str = "",
) -> Path | None:
    """Render and save one standalone superbeeswarm figure.

    Args:
        data (pd.DataFrame): Source dataframe.
        x (str): X-axis grouping column.
        y (str): Numeric metric column.
        hue (str): Hue grouping column.
        block (str): Patient-identity column.
        unit_dict (dict[str, str] | None): Optional y-unit mapping.
        save_dir (str | Path | None): Base output directory.
        title_override (str | None): Optional title override.
        filename_prefix (str | None): Optional filename prefix.
        superplot_annotations (dict[str, str] | list[dict[str, str]] | None): Optional
            bracket labels.
        x_order_override (list[str] | None): Optional explicit x order.
        hue_order_override (list[str] | None): Optional explicit hue order.
        output_dir_suffix (str): Optional suffix appended to generated figure
            directories.

    Returns:
        Path | None: Saved PNG path when ``save_dir`` is provided.
    """

    figsize, _rotate_ticks, bottom_margin = superplot_layout_params(
        data=data,
        x=x,
        y=y,
        hue=hue,
        block=block,
        x_order_override=x_order_override,
        hue_order_override=hue_order_override,
    )
    fig, ax = plt.subplots(figsize=figsize)
    annotation_top = render_super_beeswarm_on_ax(
        ax=ax,
        data=data,
        x=x,
        y=y,
        hue=hue,
        block=block,
        unit_dict=unit_dict,
        title_override=title_override,
        superplot_annotations=superplot_annotations,
        x_order_override=x_order_override,
        hue_order_override=hue_order_override,
        show_legend=False,
    )
    output_path = build_output_path(
        y=y,
        x=x,
        hue=hue,
        save_dir=save_dir,
        suffix="superbeeswarm",
        filename_prefix=filename_prefix,
        output_dir_suffix=output_dir_suffix,
    )
    return finalize_superplot_figure(
        figure=fig,
        annotation_top=annotation_top,
        title=title_override,
        output_path=output_path,
        bottom_margin=bottom_margin,
    )


def build_condition_overview_context(
    data: pd.DataFrame,
    y: str,
    condition_column: str,
    block: str,
    site_compartment_column: str,
    genotype_column: str | None = None,
    include_patient_id: bool = True,
) -> dict[str, Any] | None:
    """Build shared palette, stripe order, and geometry for overview superplots.

    Args:
        data (pd.DataFrame): Source dataframe with site×compartment labels.
        y (str): Numeric metric column.
        condition_column (str): CTRL/CAPN3 column name.
        block (str): Patient-identity column name.
        site_compartment_column (str): Combined site×compartment column name.
        genotype_column (str | None): Optional genotype column used for CAPN3
            stripe order and palette assignment.
        include_patient_id (bool): When ``False``, omit patient identifiers from
            legend labels.

    Returns:
        dict[str, Any] | None: Prepared overview context, or ``None`` when there
        are no plottable rows.
    """

    plot_data = prepare_condition_overview_plot_data(
        data=data,
        y=y,
        condition_column=condition_column,
        block=block,
        site_compartment_column=site_compartment_column,
    )
    if plot_data.empty:
        return None
    site_compartment_order, condition_order, block_order = get_category_orders(
        plot_data=plot_data,
        x=site_compartment_column,
        hue=condition_column,
        block=block,
        x_order_override=site_compartment_labels(),
        hue_order_override=list(CONDITION_ORDER),
    )
    present_conditions = set(plot_data[condition_column].astype(str))
    condition_order = [
        str(condition)
        for condition in CONDITION_ORDER
        if str(condition) in present_conditions
    ]
    if not site_compartment_order or not condition_order or not block_order:
        return None
    stripe_palette = create_condition_block_palette(
        block_order=block_order,
        x_order=site_compartment_order,
        hue_order=condition_order,
        plot_data=plot_data,
        x=site_compartment_column,
        hue=condition_column,
        block=block,
        genotype_column=genotype_column,
    )
    legend_labels = build_block_legend_labels(
        plot_data=plot_data,
        x=site_compartment_column,
        block=block,
        hue=condition_column,
        x_order=site_compartment_order,
        hue_order=condition_order,
        block_order=block_order,
        genotype_column=genotype_column,
        include_patient_id=include_patient_id,
    )
    stripes_by_condition = {
        condition_value: order_overview_stripes(
            plot_data=plot_data,
            condition_value=condition_value,
            condition_column=condition_column,
            site_compartment_column=site_compartment_column,
            block=block,
            genotype_column=genotype_column,
            site_compartment_order=site_compartment_order,
            block_order=block_order,
        )
        for condition_value in condition_order
    }
    genotype_map: dict[str, str] = {}
    if genotype_column is not None:
        genotype_map = patient_genotype_map(
            plot_data=plot_data,
            block=block,
            genotype_column=genotype_column,
        )
    return {
        "plot_data": plot_data,
        "condition_order": condition_order,
        "block_order": block_order,
        "stripe_palette": stripe_palette,
        "legend_labels": legend_labels,
        "stripes_by_condition": stripes_by_condition,
        "legend_key_order": overview_legend_key_order(
            condition_order=condition_order,
            stripes_by_condition=stripes_by_condition,
        ),
        "group_centers": condition_overview_group_centers(condition_order),
        "genotype_map": genotype_map,
    }


def render_condition_overview_super_violin_on_ax(
    ax: plt.Axes,
    data: pd.DataFrame,
    y: str,
    condition_column: str,
    block: str,
    unit_dict: dict[str, str] | None = None,
    superplot_annotations: dict[str, str] | list[dict[str, str]] | None = None,
    genotype_column: str | None = None,
    site_compartment_column: str = SITE_COMPARTMENT_COLUMN,
    include_patient_id: bool = True,
) -> float:
    """Render a two-tick CTRL-versus-CAPN3 superviolin on a provided axis.

    Args:
        ax (plt.Axes): Destination axis.
        data (pd.DataFrame): Source dataframe with site×compartment labels.
        y (str): Numeric metric column.
        condition_column (str): CTRL/CAPN3 column name.
        block (str): Patient-identity column name.
        unit_dict (dict[str, str] | None): Optional y-unit mapping.
        superplot_annotations (dict[str, str] | list[dict[str, str]] | None):
            Optional bracket labels spanning the two condition ticks.
        genotype_column (str | None): Optional genotype column used to order
            CAPN3 stripes and thicken between-subtype separators.
        site_compartment_column (str): Combined site×compartment column name.
        include_patient_id (bool): When ``False``, omit patient identifiers from
            legend labels.

    Returns:
        float: Highest annotation y-position in axis-transform coordinates.
    """

    context = build_condition_overview_context(
        data=data,
        y=y,
        condition_column=condition_column,
        block=block,
        site_compartment_column=site_compartment_column,
        genotype_column=genotype_column,
        include_patient_id=include_patient_id,
    )
    if context is None:
        ax.set_visible(False)
        return 1.0
    plot_data = context["plot_data"]
    condition_order = context["condition_order"]
    stripe_palette = context["stripe_palette"]
    legend_labels = context["legend_labels"]
    stripes_by_condition = context["stripes_by_condition"]
    legend_key_order = context["legend_key_order"]
    group_centers = context["group_centers"]
    genotype_map = context["genotype_map"]
    annotation_records = normalize_superplot_annotations(superplot_annotations)
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    group_profiles: list[dict[str, object]] = []
    global_density_max = 0.0
    for condition_value in condition_order:
        center = group_centers[(str(condition_value), CONDITION_OVERVIEW_HUE_VALUE)]
        group_data = plot_data.loc[
            plot_data[condition_column].astype(str) == str(condition_value)
        ]
        if group_data.empty:
            continue
        values = group_data[y].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        y_grid = build_density_grid(values)
        density_rows: list[np.ndarray] = []
        stripe_blocks: list[tuple[str, str]] = []
        for site_label, patient_label in stripes_by_condition[str(condition_value)]:
            block_values = group_data.loc[
                (group_data[site_compartment_column].astype(str) == str(site_label))
                & (group_data[block].astype(str) == str(patient_label)),
                y,
            ].dropna().to_numpy(dtype=float)
            if block_values.size == 0:
                continue
            density_rows.append(
                estimate_density(block_values, y_grid) * block_values.size
            )
            stripe_blocks.append((str(site_label), str(patient_label)))
        if not density_rows:
            continue
        density_matrix = np.vstack(density_rows)
        total_density = density_matrix.sum(axis=0)
        if np.allclose(total_density.max(), 0):
            continue
        global_density_max = max(global_density_max, float(total_density.max()))
        group_profiles.append(
            {
                "center": center,
                "condition_value": str(condition_value),
                "group_data": group_data,
                "y_grid": y_grid,
                "density_matrix": density_matrix,
                "total_density": total_density,
                "stripe_blocks": stripe_blocks,
            }
        )
    if global_density_max == 0.0:
        ax.set_visible(False)
        return 1.0
    for profile in group_profiles:
        center = float(profile["center"])
        condition_value = str(profile["condition_value"])
        group_data = profile["group_data"]
        y_grid = np.asarray(profile["y_grid"], dtype=float)
        density_matrix = np.asarray(profile["density_matrix"], dtype=float)
        total_density = np.asarray(profile["total_density"], dtype=float)
        group_values = group_data[y].to_numpy(dtype=float)
        group_values = group_values[np.isfinite(group_values)]
        if group_values.size > 0 and float(np.min(group_values)) >= 0.0:
            y_grid, density_matrix, total_density = clip_violin_support_at_zero(
                y_grid,
                density_matrix,
                total_density,
            )
        stripe_blocks = list(profile["stripe_blocks"])
        half_width = (
            CONDITION_OVERVIEW_HALF_WIDTH
            * SUPERPLOT_BODY_WIDTH_SCALE
            * (total_density / global_density_max)
        )
        left_edge = center - half_width
        right_edge = center + half_width
        current_left = left_edge.copy()
        full_width = right_edge - left_edge
        stripe_means: list[float] = []
        stripe_colors: list[Any] = []
        stripe_x_positions: list[float] = []
        for index, (block_density, (site_label, patient_label)) in enumerate(
            zip(density_matrix, stripe_blocks, strict=True)
        ):
            density_fraction = np.divide(
                block_density,
                total_density,
                out=np.zeros_like(block_density),
                where=total_density > 0,
            )
            stripe_right = current_left + full_width * density_fraction
            stripe_color = stripe_palette.get(
                (str(site_label), str(condition_value), str(patient_label)),
                (0.4, 0.4, 0.4),
            )
            ax.fill_betweenx(
                y_grid,
                current_left,
                stripe_right,
                color=stripe_color,
                alpha=0.85,
                linewidth=0,
                zorder=2,
                clip_on=True,
            )
            block_values = group_data.loc[
                (group_data[site_compartment_column].astype(str) == str(site_label))
                & (group_data[block].astype(str) == str(patient_label)),
                y,
            ].dropna().to_numpy(dtype=float)
            if block_values.size > 0:
                mean_y = float(np.mean(block_values))
                stripe_means.append(mean_y)
                stripe_colors.append(stripe_color)
                stripe_x_positions.append(
                    interpolate_stripe_center_x(
                        y_grid=y_grid,
                        left_edge=current_left,
                        right_edge=stripe_right,
                        y_value=mean_y,
                    )
                )
            if index < len(stripe_blocks) - 1:
                next_patient = stripe_blocks[index + 1][1]
                ax.plot(
                    stripe_right,
                    y_grid,
                    color="black",
                    linewidth=superviolin_separator_linewidth(
                        current_block=str(patient_label),
                        next_block=str(next_patient),
                        hue_value=condition_value,
                        genotype_map=genotype_map,
                    ),
                    zorder=3,
                )
            current_left = stripe_right
        ax.plot(left_edge, y_grid, color="black", linewidth=1.0, zorder=3)
        ax.plot(right_edge, y_grid, color="black", linewidth=1.0, zorder=3)
        add_superplot_mean_markers(
            ax=ax,
            center=center,
            means=stripe_means,
            colors=stripe_colors,
            x_positions=stripe_x_positions,
        )
        tick_positions.append(center)
        tick_labels.append(format_condition_display_label(condition_value))
    annotation_top = add_superplot_annotations(
        ax=ax,
        annotation_records=annotation_records,
        group_centers=group_centers,
        hue_order=[CONDITION_OVERVIEW_HUE_VALUE],
    )
    style_superplot_axis(
        ax=ax,
        x=condition_column,
        y=y,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        unit_dict=unit_dict,
        y_values=plot_data[y].to_numpy(dtype=float),
        title=None,
        rotate_ticks=False,
    )
    add_block_legend(
        ax=ax,
        condition_block_palette=stripe_palette,
        block_legend_labels=legend_labels,
        outside=True,
        key_order=legend_key_order,
    )
    return annotation_top


def render_condition_overview_super_beeswarm_on_ax(
    ax: plt.Axes,
    data: pd.DataFrame,
    y: str,
    condition_column: str,
    block: str,
    unit_dict: dict[str, str] | None = None,
    superplot_annotations: dict[str, str] | list[dict[str, str]] | None = None,
    genotype_column: str | None = None,
    site_compartment_column: str = SITE_COMPARTMENT_COLUMN,
    include_patient_id: bool = True,
) -> float:
    """Render a two-tick CTRL-versus-CAPN3 superbeeswarm on a provided axis.

    Args:
        ax (plt.Axes): Destination axis.
        data (pd.DataFrame): Source dataframe with site×compartment labels.
        y (str): Numeric metric column.
        condition_column (str): CTRL/CAPN3 column name.
        block (str): Patient-identity column name.
        unit_dict (dict[str, str] | None): Optional y-unit mapping.
        superplot_annotations (dict[str, str] | list[dict[str, str]] | None):
            Optional bracket labels spanning the two condition ticks.
        genotype_column (str | None): Optional genotype column used to match
            violin patient colors.
        site_compartment_column (str): Combined site×compartment column name.
        include_patient_id (bool): When ``False``, omit patient identifiers from
            legend labels.

    Returns:
        float: Highest annotation y-position in axis-transform coordinates.
    """

    context = build_condition_overview_context(
        data=data,
        y=y,
        condition_column=condition_column,
        block=block,
        site_compartment_column=site_compartment_column,
        genotype_column=genotype_column,
        include_patient_id=include_patient_id,
    )
    if context is None:
        ax.set_visible(False)
        return 1.0
    plot_data = context["plot_data"]
    condition_order = context["condition_order"]
    stripe_palette = context["stripe_palette"]
    legend_labels = context["legend_labels"]
    stripes_by_condition = context["stripes_by_condition"]
    legend_key_order = context["legend_key_order"]
    group_centers = context["group_centers"]
    annotation_records = normalize_superplot_annotations(superplot_annotations)
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    group_profiles: list[dict[str, object]] = []
    global_density_max = 0.0
    for group_index, condition_value in enumerate(condition_order):
        center = group_centers[(str(condition_value), CONDITION_OVERVIEW_HUE_VALUE)]
        group_data = plot_data.loc[
            plot_data[condition_column].astype(str) == str(condition_value)
        ].reset_index(drop=True)
        if group_data.empty:
            continue
        values = group_data[y].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        y_grid = build_density_grid(values)
        scaled_density = estimate_density(values, y_grid) * values.size
        if np.allclose(scaled_density.max(), 0):
            continue
        global_density_max = max(global_density_max, float(scaled_density.max()))
        group_profiles.append(
            {
                "group_index": group_index,
                "center": center,
                "condition_value": str(condition_value),
                "group_data": group_data,
                "y_grid": y_grid,
                "scaled_density": scaled_density,
            }
        )
    if global_density_max == 0.0:
        ax.set_visible(False)
        return 1.0
    max_group_count = max(len(profile["group_data"]) for profile in group_profiles)
    for profile in group_profiles:
        group_index = int(profile["group_index"])
        center = float(profile["center"])
        condition_value = str(profile["condition_value"])
        group_data = profile["group_data"]
        y_grid = np.asarray(profile["y_grid"], dtype=float)
        scaled_density = np.asarray(profile["scaled_density"], dtype=float)
        point_size = get_dynamic_beeswarm_point_size(
            group_count=len(group_data), max_group_count=max_group_count
        )
        half_width = (
            CONDITION_OVERVIEW_HALF_WIDTH
            * SUPERPLOT_BODY_WIDTH_SCALE
            * (scaled_density / global_density_max)
        )
        point_x = build_violin_like_swarm_positions(
            group_data=group_data,
            y=y,
            y_grid=y_grid,
            half_width=half_width,
            center=center,
            seed=group_index * 10_000,
        )
        point_colors = [
            stripe_palette.get(
                (str(site_label), str(condition_value), str(patient_label)),
                (0.4, 0.4, 0.4),
            )
            for site_label, patient_label in zip(
                group_data[site_compartment_column].astype(str),
                group_data[block].astype(str),
                strict=True,
            )
        ]
        ax.scatter(
            point_x,
            group_data[y].to_numpy(dtype=float),
            s=point_size,
            c=point_colors,
            edgecolor="white",
            linewidth=0.2,
            alpha=SUPERPLOT_POINT_ALPHA,
            zorder=3,
        )
        stripe_means: list[float] = []
        stripe_colors: list[Any] = []
        for site_label, patient_label in stripes_by_condition.get(
            str(condition_value), ()
        ):
            block_values = group_data.loc[
                (group_data[site_compartment_column].astype(str) == str(site_label))
                & (group_data[block].astype(str) == str(patient_label)),
                y,
            ].dropna().to_numpy(dtype=float)
            if block_values.size == 0:
                continue
            stripe_means.append(float(np.mean(block_values)))
            stripe_colors.append(
                stripe_palette.get(
                    (str(site_label), str(condition_value), str(patient_label)),
                    (0.4, 0.4, 0.4),
                )
            )
        if stripe_means:
            spread = float(np.max(half_width)) * SUPERPLOT_SUMMARY_SPREAD_FRACTION
            offsets = (
                [0.0]
                if len(stripe_means) == 1
                else np.linspace(-spread, spread, len(stripe_means))
            )
            add_superplot_mean_markers(
                ax=ax,
                center=center,
                means=stripe_means,
                colors=stripe_colors,
                x_positions=[center + float(offset) for offset in offsets],
            )
        tick_positions.append(center)
        tick_labels.append(format_condition_display_label(condition_value))
    annotation_top = add_superplot_annotations(
        ax=ax,
        annotation_records=annotation_records,
        group_centers=group_centers,
        hue_order=[CONDITION_OVERVIEW_HUE_VALUE],
    )
    style_superplot_axis(
        ax=ax,
        x=condition_column,
        y=y,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        unit_dict=unit_dict,
        y_values=plot_data[y].to_numpy(dtype=float),
        title=None,
        rotate_ticks=False,
    )
    add_block_legend(
        ax=ax,
        condition_block_palette=stripe_palette,
        block_legend_labels=legend_labels,
        outside=True,
        key_order=legend_key_order,
    )
    return annotation_top


def save_condition_overview_superplot_pair(
    render_on_ax: Callable[..., float],
    data: pd.DataFrame,
    y: str,
    condition_column: str,
    block: str,
    plot_suffix: str,
    unit_dict: dict[str, str] | None = None,
    save_dir: str | Path | None = None,
    title_override: str | None = None,
    filename_prefix: str | None = None,
    superplot_annotations: dict[str, str] | list[dict[str, str]] | None = None,
    genotype_column: str | None = None,
    site_compartment_column: str = SITE_COMPARTMENT_COLUMN,
) -> list[Path]:
    """Save overview superplots with and without patient IDs in the legend.

    Args:
        render_on_ax (Callable[..., float]): Overview renderer for one axis.
        data (pd.DataFrame): Source dataframe with site×compartment labels.
        y (str): Numeric metric column.
        condition_column (str): CTRL/CAPN3 column name.
        block (str): Patient-identity column name.
        plot_suffix (str): Plot-type suffix, ``superviolin`` or ``superbeeswarm``.
        unit_dict (dict[str, str] | None): Optional y-unit mapping.
        save_dir (str | Path | None): Base output directory.
        title_override (str | None): Optional title override.
        filename_prefix (str | None): Optional filename prefix.
        superplot_annotations (dict[str, str] | list[dict[str, str]] | None):
            Optional bracket labels spanning the two condition ticks.
        genotype_column (str | None): Optional genotype column used for CAPN3
            stripe order and legend subtypes.
        site_compartment_column (str): Combined site×compartment column name.

    Returns:
        list[Path]: Written PNG paths, patient-ID copy first.
    """

    figsize, _rotate_ticks, bottom_margin = superplot_layout_params(
        data=data,
        x=site_compartment_column,
        y=y,
        hue=condition_column,
        block=block,
        x_order_override=site_compartment_labels(),
        hue_order_override=list(CONDITION_ORDER),
    )
    outputs: list[Path] = []
    for include_patient_id, stem_suffix in (
        (True, ""),
        (False, CONDITION_OVERVIEW_NO_PATIENT_STEM_SUFFIX),
    ):
        fig, ax = plt.subplots(figsize=figsize)
        annotation_top = render_on_ax(
            ax=ax,
            data=data,
            y=y,
            condition_column=condition_column,
            block=block,
            unit_dict=unit_dict,
            superplot_annotations=superplot_annotations,
            genotype_column=genotype_column,
            site_compartment_column=site_compartment_column,
            include_patient_id=include_patient_id,
        )
        output_path = build_output_path(
            y=y,
            x=condition_column,
            hue=condition_column,
            save_dir=save_dir,
            suffix=plot_suffix,
            filename_prefix=filename_prefix,
            output_dir_suffix=CONDITION_OVERVIEW_OUTPUT_SUFFIX,
            filename_stem_suffix=stem_suffix,
        )
        written = finalize_superplot_figure(
            figure=fig,
            annotation_top=annotation_top,
            title=title_override,
            output_path=output_path,
            bottom_margin=bottom_margin,
            right_margin=CONDITION_OVERVIEW_RIGHT_MARGIN,
        )
        if written is not None:
            outputs.append(written)
    return outputs


def plot_condition_overview_super_violin(
    data: pd.DataFrame,
    y: str,
    condition_column: str,
    block: str,
    unit_dict: dict[str, str] | None = None,
    save_dir: str | Path | None = None,
    title_override: str | None = None,
    filename_prefix: str | None = None,
    superplot_annotations: dict[str, str] | list[dict[str, str]] | None = None,
    genotype_column: str | None = None,
    site_compartment_column: str = SITE_COMPARTMENT_COLUMN,
) -> list[Path]:
    """Render and save two-tick CTRL-versus-CAPN3 superviolins.

    Writes two copies: one legend with patient IDs and one without. Figure size
    matches the four-cluster CAPN3-versus-CTRL superplots.

    Args:
        data (pd.DataFrame): Source dataframe with site×compartment labels.
        y (str): Numeric metric column.
        condition_column (str): CTRL/CAPN3 column name.
        block (str): Patient-identity column name.
        unit_dict (dict[str, str] | None): Optional y-unit mapping.
        save_dir (str | Path | None): Base output directory.
        title_override (str | None): Optional title override.
        filename_prefix (str | None): Optional filename prefix.
        superplot_annotations (dict[str, str] | list[dict[str, str]] | None):
            Optional bracket labels spanning the two condition ticks.
        genotype_column (str | None): Optional genotype column used to order
            CAPN3 stripes and thicken between-subtype separators.
        site_compartment_column (str): Combined site×compartment column name.

    Returns:
        list[Path]: Written PNG paths, patient-ID copy first.
    """

    return save_condition_overview_superplot_pair(
        render_on_ax=render_condition_overview_super_violin_on_ax,
        data=data,
        y=y,
        condition_column=condition_column,
        block=block,
        plot_suffix="superviolin",
        unit_dict=unit_dict,
        save_dir=save_dir,
        title_override=title_override,
        filename_prefix=filename_prefix,
        superplot_annotations=superplot_annotations,
        genotype_column=genotype_column,
        site_compartment_column=site_compartment_column,
    )


def plot_condition_overview_super_beeswarm(
    data: pd.DataFrame,
    y: str,
    condition_column: str,
    block: str,
    unit_dict: dict[str, str] | None = None,
    save_dir: str | Path | None = None,
    title_override: str | None = None,
    filename_prefix: str | None = None,
    superplot_annotations: dict[str, str] | list[dict[str, str]] | None = None,
    genotype_column: str | None = None,
    site_compartment_column: str = SITE_COMPARTMENT_COLUMN,
) -> list[Path]:
    """Render and save two-tick CTRL-versus-CAPN3 superbeeswarms.

    Writes two copies: one legend with patient IDs and one without. Figure size
    matches the four-cluster CAPN3-versus-CTRL superplots.

    Args:
        data (pd.DataFrame): Source dataframe with site×compartment labels.
        y (str): Numeric metric column.
        condition_column (str): CTRL/CAPN3 column name.
        block (str): Patient-identity column name.
        unit_dict (dict[str, str] | None): Optional y-unit mapping.
        save_dir (str | Path | None): Base output directory.
        title_override (str | None): Optional title override.
        filename_prefix (str | None): Optional filename prefix.
        superplot_annotations (dict[str, str] | list[dict[str, str]] | None):
            Optional bracket labels spanning the two condition ticks.
        genotype_column (str | None): Optional genotype column used to match
            violin patient colors.
        site_compartment_column (str): Combined site×compartment column name.

    Returns:
        list[Path]: Written PNG paths, patient-ID copy first.
    """

    return save_condition_overview_superplot_pair(
        render_on_ax=render_condition_overview_super_beeswarm_on_ax,
        data=data,
        y=y,
        condition_column=condition_column,
        block=block,
        plot_suffix="superbeeswarm",
        unit_dict=unit_dict,
        save_dir=save_dir,
        title_override=title_override,
        filename_prefix=filename_prefix,
        superplot_annotations=superplot_annotations,
        genotype_column=genotype_column,
        site_compartment_column=site_compartment_column,
    )

