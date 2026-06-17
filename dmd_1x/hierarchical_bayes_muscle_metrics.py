"""Fit DMD_1X hierarchical Bayesian models for muscle contrasts."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DMD_1X_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


dmd_1x_metrics = load_module_from_path(
    "dmd_1x_hierarchical_bayes_metrics_for_muscle",
    DMD_1X_DIR / "hierarchical_bayes_metrics.py",
)
dmd_1x_config = dmd_1x_metrics.dmd_1x_config

POSITIVE_METRICS = tuple(dmd_1x_metrics.POSITIVE_METRICS)
BOUNDED_METRICS = tuple(dmd_1x_metrics.BOUNDED_METRICS)
DMD_1X_IMAGE_SUMMARY_POSITIVE_METRICS = tuple(dmd_1x_metrics.DMD_1X_IMAGE_SUMMARY_POSITIVE_METRICS)
DMD_1X_IMAGE_SUMMARY_BOUNDED_METRICS = tuple(dmd_1x_metrics.DMD_1X_IMAGE_SUMMARY_BOUNDED_METRICS)
DMD_1X_IMAGE_SUMMARY_REAL_METRICS = tuple(dmd_1x_metrics.DMD_1X_IMAGE_SUMMARY_REAL_METRICS)
POSITIVE_LIKELIHOODS = tuple(dmd_1x_metrics.POSITIVE_LIKELIHOODS)
BOUNDED_LIKELIHOODS = tuple(dmd_1x_metrics.BOUNDED_LIKELIHOODS)
REAL_LIKELIHOODS = tuple(dmd_1x_metrics.REAL_LIKELIHOODS)
DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH = (
    DMD_1X_DIR / "hierarchical_bayes_muscle_config.yaml"
)
MUSCLE_CONTRAST_FLAG_COLUMN = "__muscle_contrast_remapped__"


def remap_dataframe_for_muscle_contrast(df: pd.DataFrame) -> pd.DataFrame:
    """Swap grouping columns to model muscle contrast within each genotype.

    Args:
        df (pd.DataFrame): Raw DMD_1X measurements table.

    Returns:
        pd.DataFrame: Dataframe where ``Muscle`` stores genotype and ``Condition``
            stores muscle labels for EOM-vs-TA contrasts.
    """

    if MUSCLE_CONTRAST_FLAG_COLUMN in df.columns:
        return df.copy()
    missing = [column for column in ("Condition", "Muscle") if column not in df.columns]
    if missing:
        raise KeyError(
            "Muscle-contrast analysis requires the following columns in the input CSV: "
            f"{missing}"
        )
    remapped = df.copy()
    original_condition = remapped["Condition"].astype(str)
    original_muscle = remapped["Muscle"].astype(str)
    remapped["Condition"] = original_muscle
    remapped["Muscle"] = original_condition
    remapped[MUSCLE_CONTRAST_FLAG_COLUMN] = True
    return remapped


def load_measurements(paths: Any, zoom_filter: Any | None = None) -> pd.DataFrame:
    """Load configured measurements and remap them for muscle contrast fits.

    Args:
        paths (Any): Path config with ``input_csv`` and ``input_csvs`` fields.
        zoom_filter (Any | None): Optional inclusive zoom filter config.

    Returns:
        pd.DataFrame: Loaded and remapped measurements dataframe.
    """

    loaded = dmd_1x_metrics.load_measurements(paths=paths, zoom_filter=zoom_filter)
    return remap_dataframe_for_muscle_contrast(df=loaded)


def prepare_instance_metric_data(
    df: pd.DataFrame,
    genotype: str,
    compartment: str,
    metric: str,
    positive_likelihood: str = dmd_1x_metrics.c3_metrics.DEFAULT_POSITIVE_LIKELIHOOD,
    bounded_likelihood: str = dmd_1x_metrics.c3_metrics.DEFAULT_BOUNDED_LIKELIHOOD,
) -> Any:
    """Prepare one instance-level genotype/compartment metric for muscle contrast.

    Args:
        df (pd.DataFrame): Raw or remapped DMD_1X instance dataframe.
        genotype (str): Genotype label used to stratify the fit.
        compartment (str): Compartment label to analyze.
        metric (str): Metric column name.
        positive_likelihood (str): Likelihood for positive metrics.
        bounded_likelihood (str): Likelihood strategy for bounded metrics.

    Returns:
        Any: Prepared arrays with muscles as the modeled two-level comparison.
    """

    remapped = remap_dataframe_for_muscle_contrast(df=df)
    return dmd_1x_metrics.prepare_instance_metric_data(
        df=remapped,
        muscle=genotype,
        compartment=compartment,
        metric=metric,
        positive_likelihood=positive_likelihood,
        bounded_likelihood=bounded_likelihood,
    )


def prepare_image_summary_metric_data(
    df: pd.DataFrame,
    genotype: str,
    compartment: str,
    metric: str,
    positive_likelihood: str = dmd_1x_metrics.c3_metrics.DEFAULT_POSITIVE_LIKELIHOOD,
    bounded_likelihood: str = dmd_1x_metrics.c3_metrics.DEFAULT_BOUNDED_LIKELIHOOD,
    real_likelihood: str = REAL_LIKELIHOODS[0],
) -> Any:
    """Prepare one image-summary genotype/compartment metric for muscle contrast.

    Args:
        df (pd.DataFrame): Raw or remapped DMD_1X image-summary dataframe.
        genotype (str): Genotype label used to stratify the fit.
        compartment (str): Compartment label to analyze.
        metric (str): Metric column name.
        positive_likelihood (str): Likelihood for positive metrics.
        bounded_likelihood (str): Likelihood strategy for bounded metrics.
        real_likelihood (str): Likelihood for real-valued metrics.

    Returns:
        Any: Prepared arrays with one observation per image and muscle contrast labels.
    """

    remapped = remap_dataframe_for_muscle_contrast(df=df)
    return dmd_1x_metrics.prepare_image_summary_metric_data(
        df=remapped,
        muscle=genotype,
        compartment=compartment,
        metric=metric,
        positive_likelihood=positive_likelihood,
        bounded_likelihood=bounded_likelihood,
        real_likelihood=real_likelihood,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for muscle-contrast Bayesian fitting.

    Args:
        None: Reads arguments from ``sys.argv``.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Fit DMD_1X hierarchical Bayesian muscle contrasts by genotype and compartment."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH)
    return parser.parse_args()


def load_config(path: Path) -> Any:
    """Load a DMD_1X muscle-contrast Bayesian config.

    Args:
        path (Path): YAML config path.

    Returns:
        Any: Parsed DMD_1X hierarchical Bayes config.
    """

    return dmd_1x_config.load_hierarchical_bayes_config(
        path=path,
        positive_metrics=POSITIVE_METRICS,
        bounded_metrics=BOUNDED_METRICS,
        image_summary_positive_metrics=DMD_1X_IMAGE_SUMMARY_POSITIVE_METRICS,
        image_summary_bounded_metrics=DMD_1X_IMAGE_SUMMARY_BOUNDED_METRICS,
        image_summary_real_metrics=DMD_1X_IMAGE_SUMMARY_REAL_METRICS,
    )


def run_analysis(config_path: Path, analysis: Any) -> None:
    """Run one configured DMD_1X muscle-contrast analysis and save summary CSV.

    Args:
        config_path (Path): YAML configuration path used for status messages.
        analysis (Any): Normalized DMD_1X analysis configuration.

    Returns:
        None: Writes summary CSV rows and optional trace NetCDF files.
    """

    if not analysis.enabled:
        print(f"Analysis {analysis.analysis_id} is disabled in {config_path}; nothing to do.")
        return
    df = load_measurements(paths=analysis.paths, zoom_filter=analysis.filters)
    fit_configs = dmd_1x_config.repeated_fit_configs(analysis)
    if not fit_configs:
        print(f"No {analysis.analysis_id} fits have repeat=true in {config_path}; nothing to do.")
        return
    rows: list[dict[str, Any]] = []
    for fit_index, fit_config in enumerate(fit_configs):
        seed = analysis.runtime.seed + fit_index
        positive_likelihood, bounded_likelihood, real_likelihood = dmd_1x_metrics.fit_likelihood_arguments(
            fit_config=fit_config,
            model_level=analysis.analysis_id,
        )
        print(
            "Fitting "
            f"{analysis.analysis_id} {fit_config.metric} for "
            f"{fit_config.muscle} / {fit_config.compartment} (TA vs EOM)..."
        )
        start = time.perf_counter()
        result = dmd_1x_metrics.analyze_metric(
            df=df,
            muscle=fit_config.muscle,
            compartment=fit_config.compartment,
            metric=fit_config.metric,
            positive_likelihood=positive_likelihood,
            bounded_likelihood=bounded_likelihood,
            real_likelihood=real_likelihood,
            draws=fit_config.draws,
            tune=fit_config.tune,
            chains=fit_config.chains,
            cores=analysis.runtime.cores,
            target_accept=fit_config.target_accept,
            random_seed=seed,
            retry_on_warnings=analysis.runtime.retry_on_warnings,
            retry_max_draws=analysis.runtime.retry_max_draws,
            retry_max_tune=analysis.runtime.retry_max_tune,
            retry_max_target_accept=analysis.runtime.retry_max_target_accept,
            model_level=analysis.analysis_id,
        )
        finalized_result = dmd_1x_metrics.finalize_fit_artifacts(
            result=result,
            trace_dir=None if not analysis.runtime.save_idata else analysis.paths.trace_dir,
            random_seed=seed + 10_000,
            compartment=fit_config.compartment,
        )
        finalized_result.row["contrast_axis"] = "muscle"
        finalized_result.row["stratified_genotype"] = fit_config.muscle
        finalized_result.row["wall_seconds_total"] = time.perf_counter() - start
        rows.append(finalized_result.row)
    result_df = dmd_1x_metrics.merge_result_rows(
        output_path=analysis.paths.summary_csv,
        new_rows_df=pd.DataFrame(rows),
        update_mode=analysis.runtime.summary_update_mode,
    )
    analysis.paths.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(analysis.paths.summary_csv, index=False)
    print(f"Saved {analysis.analysis_id} muscle-contrast results to {analysis.paths.summary_csv}")


def main() -> None:
    """Run all configured DMD_1X hierarchical Bayesian muscle analyses.

    Args:
        None: Reads command-line arguments.

    Returns:
        None: Saves configured analysis outputs.
    """

    args = parse_args()
    config = load_config(path=args.config)
    for analysis in config.analyses_by_id.values():
        run_analysis(config_path=args.config, analysis=analysis)


if __name__ == "__main__":
    main()
