"""Generate DMD Bayesian diagnostics, PPCs, and cell-aware superplots."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
C3_DIR = REPO_ROOT / "c3"
DMD_DIR = Path(__file__).resolve().parent
if str(C3_DIR) not in sys.path:
    sys.path.insert(0, str(C3_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stats_utils import plot_super_beeswarm, plot_super_violin


def load_module_from_path(module_name: str, path: Path) -> Any:
    """Load a Python module from a specific file path.

    Args:
        module_name (str): Import name assigned to the loaded module.
        path (Path): Python source file to load.

    Returns:
        Any: Loaded module object.
    """

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


c3_vis = load_module_from_path("c3_hierarchical_bayes_visualization", C3_DIR / "hierarchical_bayes_visualization.py")
dmd_metrics = load_module_from_path("dmd_hierarchical_bayes_metrics_for_visuals", DMD_DIR / "hierarchical_bayes_metrics.py")
dmd_config = dmd_metrics.dmd_config

BAYES_SCRIPT_PATH = DMD_DIR / "hierarchical_bayes_metrics.py"
DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH = dmd_metrics.DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH
BAYES_FACTOR_ANNOTATION_MODE = "bayes_factor"
EFFECT_SUMMARY_ANNOTATION_MODE = "effect_summary"
METRIC_UNITS = {
    **c3_vis.METRIC_UNITS,
    "Density": "count/nm^2",
    "Area_sum": "nm^2",
    "Minor_axis_length_sum": "nm",
    "Area_mean": "nm^2",
    "Corrected_area_mean": "nm^2",
    "Major_axis_length_mean": "nm",
    "Minor_axis_length_mean": "nm",
    "NND_center_mean": "nm",
    "5NND_center_mean": "nm",
}


def short_compartment_label(compartment: str) -> str:
    """Return a compact display label for a DMD compartment.

    Args:
        compartment (str): Full compartment label.

    Returns:
        str: ``SS``, ``IMF``, or the original label.
    """

    if str(compartment) == dmd_metrics.SS_LABEL:
        return "SS"
    if str(compartment) == dmd_metrics.IMF_LABEL:
        return "IMF"
    return str(compartment)


def comparison_label(muscle: str, compartment: str) -> str:
    """Build a combined muscle-compartment label for plotting.

    Args:
        muscle (str): Muscle label.
        compartment (str): Compartment label.

    Returns:
        str: Compact comparison label.
    """

    return f"{muscle} | {short_compartment_label(compartment)}"


def dmd_fit_title(row: pd.Series) -> str:
    """Build a DMD figure title for one fit.

    Args:
        row (pd.Series): Summary row describing one fit.

    Returns:
        str: Human-readable title.
    """

    metric = str(row["metric"])
    muscle = str(row["muscle"])
    compartment = str(row.get("compartment", ""))
    model_level = str(row.get("model_level", "instance"))
    scope = " center instances" if model_level == "instance" and metric in dmd_metrics.CLUSTERING_METRICS else ""
    return f"{metric} - {muscle} | {short_compartment_label(compartment)}{scope}"


def dmd_posterior_plot_specs(row: pd.Series) -> list[tuple[str, str, str, str]]:
    """Describe biology-facing posterior plots for DMD fits.

    Args:
        row (pd.Series): Summary row describing one fit.

    Returns:
        list[tuple[str, str, str, str]]: Posterior variable, summary column, filename stem, and label.
    """

    specs = [
        ("delta_mean_response", "delta_mean_summary", "delta_mean", "DMD-WT delta in mean"),
        ("delta_median_response", "delta_median_summary", "delta_median", "DMD-WT delta in median"),
        (
            "delta_image_variance_response",
            "delta_image_variance_summary",
            "delta_cell_variance",
            "DMD-WT delta in aggregated cell variance",
        ),
    ]
    if str(row.get("model_level", "instance")) != "image_summary":
        specs.append(
            (
                "delta_mito_variance_response",
                "delta_mito_variance_summary",
                "delta_mito_variance",
                "DMD-WT delta in mito variance",
            )
        )
    return specs


def figure_filename_prefix(row: pd.Series) -> str:
    """Build a flat filename prefix for one DMD fit.

    Args:
        row (pd.Series): Summary row describing one fit.

    Returns:
        str: Filename prefix ending in an underscore.
    """

    return (
        dmd_metrics.fit_stem(
            muscle=str(row["muscle"]),
            compartment=str(row["compartment"]),
            metric=str(row["metric"]),
        )
        + "_"
    )


c3_vis.METRIC_UNITS.update(METRIC_UNITS)
c3_vis.fit_title = dmd_fit_title
c3_vis.posterior_plot_specs = dmd_posterior_plot_specs
c3_vis.figure_filename_prefix = figure_filename_prefix


def parse_args() -> argparse.Namespace:
    """Parse command-line options for DMD Bayesian visualization.

    Args:
        None: Reads arguments from ``sys.argv``.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Generate DMD Bayesian diagnostics, PPCs, and cell-aware superplots."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH)
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    """Load a DMD Bayesian summary CSV.

    Args:
        path (Path): Summary CSV path.

    Returns:
        pd.DataFrame: Loaded summary dataframe.
    """

    return pd.read_csv(path)


def filter_summary_to_fit_configs(summary_df: pd.DataFrame, fit_configs: list[Any]) -> pd.DataFrame:
    """Filter summary rows to configured repeated DMD fits.

    Args:
        summary_df (pd.DataFrame): Full summary table.
        fit_configs (list[Any]): Fit configs selected for visualization.

    Returns:
        pd.DataFrame: Filtered and sorted summary table.
    """

    fit_keys = {(fit.muscle, fit.compartment, fit.metric) for fit in fit_configs}
    filtered = summary_df.loc[
        summary_df.apply(
            lambda row: (row["muscle"], row["compartment"], row["metric"]) in fit_keys,
            axis=1,
        )
    ].copy()
    return filtered.sort_values(["muscle", "compartment", "metric"]).reset_index(drop=True)


def resolve_trace_path(row: pd.Series, trace_dir: Path) -> Path:
    """Resolve the trace path for one DMD summary row.

    Args:
        row (pd.Series): Summary row describing one fit.
        trace_dir (Path): Default trace directory.

    Returns:
        Path: NetCDF trace path.
    """

    trace_path = str(row.get("trace_path", "")).strip()
    if trace_path:
        return Path(trace_path)
    return dmd_metrics.trace_path_for_fit(
        trace_dir=trace_dir,
        muscle=str(row["muscle"]),
        compartment=str(row["compartment"]),
        metric=str(row["metric"]),
    )


def traces_missing(summary_df: pd.DataFrame, trace_dir: Path) -> bool:
    """Check whether any summary row is missing its trace file.

    Args:
        summary_df (pd.DataFrame): Summary rows to inspect.
        trace_dir (Path): Default trace directory.

    Returns:
        bool: ``True`` when at least one trace file is absent.
    """

    for _, row in summary_df.iterrows():
        if not resolve_trace_path(row=row, trace_dir=trace_dir).exists():
            return True
    return False


def rerun_bayesian_fits(config_path: Path) -> None:
    """Invoke the DMD fitting script for missing visualization inputs.

    Args:
        config_path (Path): YAML config path.

    Returns:
        None: Runs the fitting script as a subprocess.
    """

    subprocess.run(
        [sys.executable, str(BAYES_SCRIPT_PATH), "--config", str(config_path)],
        check=True,
    )


def load_or_refresh_summary(
    summary_path: Path,
    trace_dir: Path,
    refresh_mode: str,
    config_path: Path,
    fit_configs: list[Any],
) -> pd.DataFrame:
    """Load the DMD summary CSV, optionally refreshing fits first.

    Args:
        summary_path (Path): Summary CSV path.
        trace_dir (Path): Trace directory.
        refresh_mode (str): Refresh mode from the config.
        config_path (Path): YAML config path.
        fit_configs (list[Any]): Fit configs to keep.

    Returns:
        pd.DataFrame: Summary rows for configured fits.
    """

    if refresh_mode == "refit_first":
        rerun_bayesian_fits(config_path=config_path)
    elif refresh_mode == "rerun_missing_traces":
        if not summary_path.exists():
            rerun_bayesian_fits(config_path=config_path)
        else:
            current_summary = filter_summary_to_fit_configs(
                summary_df=load_summary(summary_path),
                fit_configs=fit_configs,
            )
            if traces_missing(summary_df=current_summary, trace_dir=trace_dir):
                rerun_bayesian_fits(config_path=config_path)
    elif refresh_mode != "never":
        raise ValueError(f"Unsupported visualization refresh mode: {refresh_mode}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Bayesian summary CSV: {summary_path}")
    return filter_summary_to_fit_configs(
        summary_df=load_summary(summary_path),
        fit_configs=fit_configs,
    )


def resolved_likelihood_name(row: pd.Series, fit_config: Any) -> str:
    """Resolve the concrete likelihood name for a summary row.

    Args:
        row (pd.Series): Summary row describing one fit.
        fit_config (Any): Configured fit entry.

    Returns:
        str: Likelihood name used to prepare matching data arrays.
    """

    likelihood_name = str(row.get("likelihood_name", "")).strip()
    if likelihood_name:
        return likelihood_name
    return str(fit_config.likelihood)


def prepare_data_for_row(
    row: pd.Series,
    measurements_df: pd.DataFrame,
    fit_config: Any,
    model_level: str,
) -> Any:
    """Rebuild prepared DMD arrays for one visualization row.

    Args:
        row (pd.Series): Summary row describing one fit.
        measurements_df (pd.DataFrame): Full input dataframe.
        fit_config (Any): Configured fit entry.
        model_level (str): C3-compatible model level.

    Returns:
        Any: Prepared metric data matching the saved trace.
    """

    likelihood_name = resolved_likelihood_name(row=row, fit_config=fit_config)
    family = str(row["family"])
    positive_likelihood = likelihood_name if family == "positive" else dmd_metrics.POSITIVE_LIKELIHOODS[0]
    bounded_likelihood = likelihood_name if family == "bounded" else dmd_metrics.BOUNDED_LIKELIHOODS[1]
    real_likelihood = likelihood_name if family == "real" else dmd_metrics.REAL_LIKELIHOODS[0]
    if model_level == "image_summary":
        return dmd_metrics.prepare_cell_summary_metric_data(
            df=measurements_df,
            muscle=str(row["muscle"]),
            compartment=str(row["compartment"]),
            metric=str(row["metric"]),
            positive_likelihood=positive_likelihood,
            bounded_likelihood=bounded_likelihood,
            real_likelihood=real_likelihood,
        )
    return dmd_metrics.prepare_metric_data(
        df=measurements_df,
        muscle=str(row["muscle"]),
        compartment=str(row["compartment"]),
        metric=str(row["metric"]),
        positive_likelihood=positive_likelihood,
        bounded_likelihood=bounded_likelihood,
    )


def format_delta_effect_annotation(row: pd.Series, label: str) -> str:
    """Format a posterior delta estimate for a superplot annotation.

    Args:
        row (pd.Series): Summary row with posterior delta columns.
        label (str): Delta column prefix.

    Returns:
        str: Compact annotation text.
    """

    return c3_vis.format_delta_effect_annotation(row=row, label=label)


def bayesian_superplot_annotations(
    row: pd.Series,
    annotation_mode: str = BAYES_FACTOR_ANNOTATION_MODE,
) -> list[dict[str, str]]:
    """Build DMD superplot annotation records for one comparison.

    Args:
        row (pd.Series): Summary row describing one fit.
        annotation_mode (str): Annotation style, Bayes factor or effect summary.

    Returns:
        list[dict[str, str]]: Annotation records for stats_utils superplots.
    """

    base_record = {
        "x": comparison_label(str(row["muscle"]), str(row["compartment"])),
        "hue_start": str(row["wt_label"]),
        "hue_end": str(row["ko_label"]),
    }
    if annotation_mode == EFFECT_SUMMARY_ANNOTATION_MODE:
        mean_label = format_delta_effect_annotation(row=row, label="delta_mean")
        median_label = format_delta_effect_annotation(row=row, label="delta_median")
    else:
        mean_label = c3_vis.summary_text_or_empty(row.get("delta_mean_bf_annotation", ""))
        median_label = c3_vis.summary_text_or_empty(row.get("delta_median_bf_annotation", ""))
    if not mean_label and not median_label:
        return []
    return [
        {
            **base_record,
            "mean_label": (
                mean_label
                if annotation_mode == EFFECT_SUMMARY_ANNOTATION_MODE
                else f"mean {mean_label}"
                if mean_label
                else ""
            ),
            "mean_color": c3_vis.BAYES_MEAN_ANNOTATION_COLOR,
            "median_label": (
                median_label
                if annotation_mode == EFFECT_SUMMARY_ANNOTATION_MODE
                else f"median {median_label}"
                if median_label
                else ""
            ),
            "median_color": c3_vis.BAYES_MEDIAN_ANNOTATION_COLOR,
        }
    ]


def plot_bayesian_superplots(
    measurements_df: pd.DataFrame,
    row: pd.Series,
    output_dir: Path,
    output_prefix: str,
    annotation_mode: str,
) -> None:
    """Generate DMD cell-aware Bayesian superplots for one fit.

    Args:
        measurements_df (pd.DataFrame): Full input dataframe.
        row (pd.Series): Summary row describing one fit.
        output_dir (Path): Output root for superplots.
        output_prefix (str): Filename prefix.
        annotation_mode (str): Annotation style from the config.

    Returns:
        None: Saves superplot figures.
    """

    subset = measurements_df.loc[
        (measurements_df["Muscle"].astype(str) == str(row["muscle"]))
        & (measurements_df["Compartment"].astype(str) == str(row["compartment"]))
    ].copy()
    if subset.empty:
        print(f"Skipping superplots for {dmd_fit_title(row)}: no matching rows.")
        return
    subset["Comparison"] = comparison_label(str(row["muscle"]), str(row["compartment"]))
    annotation_records = bayesian_superplot_annotations(row=row, annotation_mode=annotation_mode)
    plot_super_violin(
        data=subset,
        x="Comparison",
        y=str(row["metric"]),
        hue="Condition",
        block="Block",
        unit_dict=METRIC_UNITS,
        save_dir=output_dir,
        title_override=c3_vis.superplot_title_text(row),
        filename_prefix=output_prefix,
        superplot_annotations=annotation_records,
    )
    plot_super_beeswarm(
        data=subset,
        x="Comparison",
        y=str(row["metric"]),
        hue="Condition",
        block="Block",
        unit_dict=METRIC_UNITS,
        save_dir=output_dir,
        title_override=c3_vis.superplot_title_text(row),
        filename_prefix=output_prefix,
        superplot_annotations=annotation_records,
    )


def generate_fit_visualizations(
    row: pd.Series,
    idata: az.InferenceData,
    prepared: Any,
    measurements_df: pd.DataFrame,
    figure_root: Path,
    annotation_mode: str,
) -> None:
    """Generate all requested DMD visualizations for one fit.

    Args:
        row (pd.Series): Summary row describing one fit.
        idata (az.InferenceData): Loaded inference data.
        prepared (Any): Prepared metric arrays matching the trace.
        measurements_df (pd.DataFrame): Full input dataframe.
        figure_root (Path): Root figure directory.
        annotation_mode (str): Superplot annotation mode.

    Returns:
        None: Saves figures under ``figure_root``.
    """

    output_prefix = figure_filename_prefix(row=row)
    output_dir = figure_root
    c3_vis.plot_trace_figure(idata=idata, row=row, output_dir=output_dir, output_prefix=output_prefix)
    c3_vis.plot_rank_figure(idata=idata, row=row, output_dir=output_dir, output_prefix=output_prefix)
    c3_vis.plot_forest_figure(idata=idata, row=row, output_dir=output_dir, output_prefix=output_prefix)
    if str(row.get("engine_fit_status", row.get("fit_status", ""))) == "warn":
        c3_vis.plot_energy_figure(idata=idata, row=row, output_dir=output_dir, output_prefix=output_prefix)
    c3_vis.plot_ppc_density(
        idata=idata,
        row=row,
        prepared=prepared,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )
    c3_vis.plot_ppc_density_by_condition(
        idata=idata,
        row=row,
        prepared=prepared,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )
    c3_vis.plot_ppc_quantiles(
        idata=idata,
        row=row,
        prepared=prepared,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )
    c3_vis.plot_ppc_animal_summary(
        idata=idata,
        row=row,
        prepared=prepared,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )
    c3_vis.plot_biology_posteriors(idata=idata, row=row, output_dir=output_dir, output_prefix=output_prefix)
    plot_bayesian_superplots(
        measurements_df=measurements_df,
        row=row,
        output_dir=figure_root,
        output_prefix=output_prefix,
        annotation_mode=annotation_mode,
    )


def run_visualization_analysis(config_path: Path, config: Any, model_level: str) -> None:
    """Generate figures for one configured DMD Bayesian analysis.

    Args:
        config_path (Path): YAML config path.
        config (Any): Parsed DMD analysis config.
        model_level (str): C3-compatible model level.

    Returns:
        None: Writes figure files.
    """

    if not config.enabled:
        print(f"Analysis {config.analysis_id} is disabled in {config_path}; nothing to do.")
        return
    fit_configs = dmd_config.repeated_fit_configs(config)
    if not fit_configs:
        print(f"No {config.analysis_id} fits have repeat=true in {config_path}; nothing to do.")
        return
    measurements_df = dmd_metrics.load_measurements(config.paths, config.filters)
    dmd_config.validate_config_groups(
        config=config,
        valid_muscles=set(measurements_df["Muscle"].dropna().astype(str).unique().tolist()),
        valid_compartments=set(measurements_df["Compartment"].dropna().astype(str).unique().tolist()),
    )
    summary_df = load_or_refresh_summary(
        summary_path=config.paths.summary_csv,
        trace_dir=config.paths.trace_dir,
        refresh_mode=config.runtime.visualization_refresh_mode,
        config_path=config_path,
        fit_configs=fit_configs,
    )
    for _, row in summary_df.iterrows():
        fit_config = dmd_config.fit_config_for_key(
            config=config,
            muscle=str(row["muscle"]),
            compartment=str(row["compartment"]),
            metric=str(row["metric"]),
        )
        trace_path = resolve_trace_path(row=row, trace_dir=config.paths.trace_dir)
        if not trace_path.exists():
            raise FileNotFoundError(
                "Missing trace file for "
                f"{row['muscle']} / {row['compartment']} / {row['metric']}: {trace_path}"
            )
        print(f"Generating figures for {row['metric']} ({row['muscle']} / {row['compartment']})...")
        idata = az.from_netcdf(trace_path)
        prepared = prepare_data_for_row(
            row=row,
            measurements_df=measurements_df,
            fit_config=fit_config,
            model_level=model_level,
        )
        idata = c3_vis.ensure_augmented_idata(prepared=prepared, idata=idata)
        generate_fit_visualizations(
            row=row,
            idata=idata,
            prepared=prepared,
            measurements_df=measurements_df,
            figure_root=config.paths.figure_root,
            annotation_mode=config.runtime.superplot_annotation_mode,
        )


def main() -> None:
    """Run the DMD Bayesian visualization workflow.

    Args:
        None: Reads command-line arguments.

    Returns:
        None: Saves visualization files.
    """

    args = parse_args()
    config = dmd_metrics.load_config(args.config)
    for analysis_id, analysis in config.analyses_by_id.items():
        model_level = "image_summary" if analysis_id == "cell_summary" else "instance"
        run_visualization_analysis(
            config_path=args.config,
            config=analysis,
            model_level=model_level,
        )


if __name__ == "__main__":
    main()
