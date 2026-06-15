"""Fit DMD_1X hierarchical Bayesian models by compartment."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
C3_DIR = REPO_ROOT / "c3"
DMD_1X_DIR = Path(__file__).resolve().parent
if str(C3_DIR) not in sys.path:
    sys.path.insert(0, str(C3_DIR))
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


c3_metrics = load_module_from_path("c3_hierarchical_bayes_metrics_for_dmd_1x", C3_DIR / "hierarchical_bayes_metrics.py")
dmd_1x_config = load_module_from_path(
    "dmd_1x_hierarchical_bayes_config",
    DMD_1X_DIR / "hierarchical_bayes_config.py",
)

PreparedMetricData = c3_metrics.PreparedMetricData
MetricAnalysisResult = c3_metrics.MetricAnalysisResult
POSITIVE_LIKELIHOODS = tuple(c3_metrics.POSITIVE_LIKELIHOODS)
BOUNDED_LIKELIHOODS = tuple(c3_metrics.BOUNDED_LIKELIHOODS)
REAL_LIKELIHOODS = tuple(c3_metrics.REAL_LIKELIHOODS)
POSITIVE_METRICS = tuple(c3_metrics.POSITIVE_METRICS)
BOUNDED_METRICS = tuple(c3_metrics.BOUNDED_METRICS)
CLUSTERING_METRICS = set(c3_metrics.CLUSTERING_METRICS)
CENTER_IMAGE_REGION = str(c3_metrics.CENTER_IMAGE_REGION)
SMALL_VALUE = float(c3_metrics.SMALL_VALUE)
DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH = dmd_1x_config.DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH

DMD_LABEL = "Duchenne_Muscular_Dystrophy"
WT_LABEL = "Wildtype"
SS_LABEL = "Sub-sarcolemmal (SS)"
IMF_LABEL = "Intermyofibrillar (IMF)"
DMD_1X_IMAGE_SUMMARY_POSITIVE_METRICS = (
    "Density",
    "Instance_count",
    "Area_sum",
    "Corrected_area_sum",
    "Minimum_Feret_Diameter_sum",
    "Minor_axis_length_sum",
    "Area_mean",
    "Corrected_area_mean",
    "Minimum_Feret_Diameter_mean",
    "Major_axis_length_mean",
    "Minor_axis_length_mean",
    "Elongation_mean",
    "NND_center_mean",
    "3NND_center_mean",
    "5NND_center_mean",
    "Voronoi_Cell_Area_center_mean",
    "Voronoi_Cell_Area_center_cv",
)
DMD_1X_IMAGE_SUMMARY_BOUNDED_METRICS = (
    "Circularity_mean",
    "Solidity_mean",
)
DMD_1X_IMAGE_SUMMARY_REAL_METRICS = (
    "Ripley_L_integral",
    "Pair_Correlation_integral",
)


def slugify_value(value: str) -> str:
    """Convert a label into a filesystem-friendly slug.

    Args:
        value (str): Raw label.

    Returns:
        str: Lowercase slug containing only alphanumerics and underscores.
    """

    return c3_metrics.slugify_value(value)


def fit_stem(muscle: str, compartment: str, metric: str) -> str:
    """Build the filename stem for one DMD_1X fit.

    Args:
        muscle (str): Muscle label.
        compartment (str): Compartment label.
        metric (str): Metric column name.

    Returns:
        str: Stable stem containing muscle, compartment, and metric.
    """

    return "__".join(
        [
            slugify_value(muscle),
            slugify_value(compartment),
            slugify_value(metric),
        ]
    )


def trace_path_for_fit(trace_dir: Path, muscle: str, compartment: str, metric: str) -> Path:
    """Return the NetCDF path used to store one DMD_1X fit.

    Args:
        trace_dir (Path): Directory where traces are saved.
        muscle (str): Muscle label.
        compartment (str): Compartment label.
        metric (str): Metric column name.

    Returns:
        Path: Trace path for the requested fit.
    """

    return trace_dir / f"{fit_stem(muscle=muscle, compartment=compartment, metric=metric)}.nc"


def metric_family_for_level(metric: str, model_level: str) -> str:
    """Return the C3-compatible model family for a DMD_1X metric.

    Args:
        metric (str): Metric column name.
        model_level (str): Model level, either ``instance`` or ``image_summary``.

    Returns:
        str: Family label: ``positive``, ``bounded``, or ``real``.
    """

    if model_level == "instance":
        return c3_metrics.metric_family_for_metric(
            metric=metric,
            positive_metrics=POSITIVE_METRICS,
            bounded_metrics=BOUNDED_METRICS,
        )
    return c3_metrics.metric_family_for_metric(
        metric=metric,
        positive_metrics=DMD_1X_IMAGE_SUMMARY_POSITIVE_METRICS,
        bounded_metrics=DMD_1X_IMAGE_SUMMARY_BOUNDED_METRICS,
        real_metrics=DMD_1X_IMAGE_SUMMARY_REAL_METRICS,
    )

def filter_metric_dataframe(
    df: pd.DataFrame,
    muscle: str,
    compartment: str,
    metric: str,
    required_columns: list[str],
) -> pd.DataFrame:
    """Filter the input dataframe to one DMD_1X muscle, compartment, and metric.

    Args:
        df (pd.DataFrame): Full DMD_1X image-summary dataframe.
        muscle (str): Muscle label to keep.
        compartment (str): Compartment label to keep.
        metric (str): Metric column name.
        required_columns (list[str]): Columns needed downstream.

    Returns:
        pd.DataFrame: Filtered dataframe with the metric renamed to ``value``.
    """

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required DMD_1X columns for {metric}: {missing_columns}")
    df_metric = df.loc[
        (df["Muscle"].astype(str) == muscle)
        & (df["Compartment"].astype(str) == compartment),
        required_columns,
    ].copy()
    if metric in CLUSTERING_METRICS:
        df_metric = df_metric.loc[
            df_metric["Image_Region"].astype(str).str.lower() == CENTER_IMAGE_REGION
        ].copy()
    df_metric = df_metric.rename(columns={metric: "value"})
    df_metric["value"] = pd.to_numeric(df_metric["value"], errors="coerce")
    return df_metric


def apply_metric_transform(
    df_metric: pd.DataFrame,
    metric: str,
    family: str,
    positive_likelihood: str,
    bounded_likelihood: str,
    real_likelihood: str,
) -> tuple[pd.DataFrame, np.ndarray, str, bool, float, float, float, float]:
    """Transform image-level metric values into the model scale.

    Args:
        df_metric (pd.DataFrame): Filtered dataframe containing a numeric ``value`` column.
        metric (str): Metric column name being modeled.
        family (str): Model family for the metric.
        positive_likelihood (str): Positive-family likelihood name.
        bounded_likelihood (str): Bounded-family likelihood name.
        real_likelihood (str): Real-valued likelihood name.

    Returns:
        tuple[pd.DataFrame, np.ndarray, str, bool, float, float, float, float]:
            Transformed dataframe, raw response values, likelihood name, boundary-adjusted
            flag, positive scale, positive offset, pooled median, and pooled standard deviation.
    """

    transformed = df_metric.copy()
    raw_values = transformed["value"].to_numpy(dtype=float)
    values = raw_values.copy()
    boundary_adjusted = False
    if family == "positive":
        if positive_likelihood not in POSITIVE_LIKELIHOODS:
            raise ValueError(f"Unsupported positive likelihood: {positive_likelihood}")
        likelihood_name = positive_likelihood
    elif family == "bounded":
        likelihood_name = c3_metrics.resolve_bounded_likelihood(
            values=raw_values,
            bounded_likelihood=bounded_likelihood,
        )
        if likelihood_name not in BOUNDED_LIKELIHOODS[1:]:
            raise ValueError(f"Unsupported bounded likelihood: {likelihood_name}")
        if likelihood_name in ("beta", "logitnormal", "logit_skew_normal"):
            values, boundary_adjusted = c3_metrics.squeeze_open_interval_values(values)
    elif family == "real":
        if real_likelihood not in REAL_LIKELIHOODS:
            raise ValueError(f"Unsupported real-valued likelihood: {real_likelihood}")
        likelihood_name = real_likelihood
    else:
        raise ValueError(f"Unsupported DMD_1X metric family: {family}")

    positive_scale = 1.0
    positive_offset = 0.0
    if family == "positive":
        positive_offset = c3_metrics.positive_metric_offset(metric=metric)
        values = values - positive_offset
        if np.any(values <= 0.0):
            min_raw_value = float(np.min(raw_values))
            raise ValueError(
                f"{metric} must be strictly greater than {positive_offset} for positive modeling; "
                f"minimum observed value is {min_raw_value}."
            )
        positive_scale = max(float(np.median(values)), SMALL_VALUE)
        values = values / positive_scale
    elif family == "real":
        positive_offset = float(np.median(values))
        positive_scale = max(float(np.std(values, ddof=1)), SMALL_VALUE) if values.size > 1 else 1.0
        values = (values - positive_offset) / positive_scale

    transformed["value"] = values
    pooled_median = float(np.median(values))
    pooled_std = float(np.std(values, ddof=1)) if len(values) > 1 else SMALL_VALUE
    pooled_std = max(pooled_std, SMALL_VALUE)
    return (
        transformed,
        raw_values,
        likelihood_name,
        boundary_adjusted,
        positive_scale,
        positive_offset,
        pooled_median,
        pooled_std,
    )


def build_prepared_image_data(
    *,
    df_metric: pd.DataFrame,
    raw_values: np.ndarray,
    muscle: str,
    compartment: str,
    metric: str,
    family: str,
    likelihood_name: str,
    boundary_adjusted: bool,
    positive_scale: float,
    positive_offset: float,
    pooled_median: float,
    pooled_std: float,
) -> PreparedMetricData:
    """Build the C3-compatible prepared data container for DMD_1X image data.

    Args:
        df_metric (pd.DataFrame): Metric dataframe with one row per image.
        raw_values (np.ndarray): Response-scale observations before model scaling.
        muscle (str): Muscle label.
        compartment (str): Compartment label.
        metric (str): Metric column name.
        family (str): Metric family.
        likelihood_name (str): Concrete likelihood used by the model.
        boundary_adjusted (bool): Whether bounded values were moved from 0/1.
        positive_scale (float): Response scaling factor.
        positive_offset (float): Response offset.
        pooled_median (float): Median on the model scale.
        pooled_std (float): Standard deviation on the model scale.

    Returns:
        PreparedMetricData: Data container consumed by the shared PyMC builders.
    """

    wt_label, dmd_label = c3_metrics.determine_condition_labels(
        [str(value) for value in df_metric["Condition"].dropna().unique().tolist()]
    )
    indexed = df_metric.copy()
    genotype_map = {wt_label: 0, dmd_label: 1}
    indexed["genotype_idx"] = indexed["Condition"].map(genotype_map).astype(int)
    indexed["animal_key"] = indexed["Condition"].astype(str) + "__" + indexed["Block"].astype(str)
    animal_table = (
        indexed[["animal_key", "Condition", "Block"]]
        .drop_duplicates()
        .sort_values(["Condition", "Block"])
        .reset_index(drop=True)
    )
    animal_table["animal_idx"] = np.arange(len(animal_table), dtype=int)
    animal_idx_map = animal_table.set_index("animal_key")["animal_idx"].to_dict()
    animal_table["genotype_idx"] = animal_table["Condition"].map(genotype_map).astype(int)
    indexed["animal_idx"] = indexed["animal_key"].map(animal_idx_map).astype(int)
    indexed["image_idx"] = np.arange(len(indexed), dtype=int)
    image_labels = (
        indexed["animal_key"].astype(str)
        + "__"
        + indexed["Compartment"].astype(str)
        + "__"
        + indexed["image"].astype(str)
    )
    return PreparedMetricData(
        muscle=muscle,
        metric=metric,
        family=family,
        likelihood_name=likelihood_name,
        wt_label=wt_label,
        ko_label=dmd_label,
        y=indexed["value"].to_numpy(dtype=float),
        observed_y=raw_values,
        genotype_idx_obs=indexed["genotype_idx"].to_numpy(dtype=int),
        image_idx_obs=indexed["image_idx"].to_numpy(dtype=int),
        animal_idx_obs=indexed["animal_idx"].to_numpy(dtype=int),
        animal_idx_image=indexed["animal_idx"].to_numpy(dtype=int),
        genotype_idx_animal=animal_table["genotype_idx"].to_numpy(dtype=int),
        genotype_idx_image=indexed["genotype_idx"].to_numpy(dtype=int),
        animal_labels=animal_table["animal_key"].astype(str).tolist(),
        image_labels=image_labels.astype(str).tolist(),
        boundary_adjusted=boundary_adjusted,
        positive_scale=positive_scale,
        positive_offset=positive_offset,
        pooled_median=max(pooled_median, SMALL_VALUE),
        pooled_std=pooled_std,
        n_obs_wt=int(np.sum(indexed["genotype_idx"] == 0)),
        n_obs_ko=int(np.sum(indexed["genotype_idx"] == 1)),
        n_animals_wt=int(np.sum(animal_table["genotype_idx"] == 0)),
        n_animals_ko=int(np.sum(animal_table["genotype_idx"] == 1)),
        n_images_wt=int(np.sum(indexed["genotype_idx"] == 0)),
        n_images_ko=int(np.sum(indexed["genotype_idx"] == 1)),
        analysis_id="image_summary",
        model_level="image_summary",
    )


def build_prepared_instance_data(
    *,
    df_metric: pd.DataFrame,
    raw_values: np.ndarray,
    muscle: str,
    compartment: str,
    metric: str,
    family: str,
    likelihood_name: str,
    boundary_adjusted: bool,
    positive_scale: float,
    positive_offset: float,
    pooled_median: float,
    pooled_std: float,
) -> PreparedMetricData:
    """Build the C3-compatible prepared data container for DMD_1X instance data.

    Args:
        df_metric (pd.DataFrame): Metric dataframe with one row per mitochondrion.
        raw_values (np.ndarray): Response-scale observations before model scaling.
        muscle (str): Muscle label.
        compartment (str): Compartment label.
        metric (str): Metric column name.
        family (str): Metric family.
        likelihood_name (str): Concrete likelihood used by the model.
        boundary_adjusted (bool): Whether bounded values were moved from 0/1.
        positive_scale (float): Response scaling factor.
        positive_offset (float): Response offset.
        pooled_median (float): Median on the model scale.
        pooled_std (float): Standard deviation on the model scale.

    Returns:
        PreparedMetricData: Data container consumed by the shared PyMC builders.
    """

    wt_label, dmd_label = c3_metrics.determine_condition_labels(
        [str(value) for value in df_metric["Condition"].dropna().unique().tolist()]
    )
    indexed = df_metric.copy()
    genotype_map = {wt_label: 0, dmd_label: 1}
    indexed["genotype_idx"] = indexed["Condition"].map(genotype_map).astype(int)
    indexed["animal_key"] = indexed["Condition"].astype(str) + "__" + indexed["Block"].astype(str)
    indexed["image_key"] = (
        indexed["animal_key"]
        + "__"
        + indexed["Compartment"].astype(str)
        + "__"
        + indexed["image"].astype(str)
    )
    animal_table = (
        indexed[["animal_key", "Condition", "Block"]]
        .drop_duplicates()
        .sort_values(["Condition", "Block"])
        .reset_index(drop=True)
    )
    animal_table["animal_idx"] = np.arange(len(animal_table), dtype=int)
    image_table = (
        indexed[["image_key", "animal_key", "Condition", "Block", "image", "Compartment"]]
        .drop_duplicates()
        .sort_values(["Condition", "animal_key", "Compartment", "image"])
        .reset_index(drop=True)
    )
    image_table["image_idx"] = np.arange(len(image_table), dtype=int)
    animal_idx_map = animal_table.set_index("animal_key")["animal_idx"].to_dict()
    image_idx_map = image_table.set_index("image_key")["image_idx"].to_dict()
    indexed["animal_idx"] = indexed["animal_key"].map(animal_idx_map).astype(int)
    indexed["image_idx"] = indexed["image_key"].map(image_idx_map).astype(int)
    animal_table["genotype_idx"] = animal_table["Condition"].map(genotype_map).astype(int)
    image_table["animal_idx"] = image_table["animal_key"].map(animal_idx_map).astype(int)
    image_table["genotype_idx"] = image_table["Condition"].map(genotype_map).astype(int)
    return PreparedMetricData(
        muscle=muscle,
        metric=metric,
        family=family,
        likelihood_name=likelihood_name,
        wt_label=wt_label,
        ko_label=dmd_label,
        y=indexed["value"].to_numpy(dtype=float),
        observed_y=raw_values,
        genotype_idx_obs=indexed["genotype_idx"].to_numpy(dtype=int),
        image_idx_obs=indexed["image_idx"].to_numpy(dtype=int),
        animal_idx_obs=indexed["animal_idx"].to_numpy(dtype=int),
        animal_idx_image=image_table["animal_idx"].to_numpy(dtype=int),
        genotype_idx_animal=animal_table["genotype_idx"].to_numpy(dtype=int),
        genotype_idx_image=image_table["genotype_idx"].to_numpy(dtype=int),
        animal_labels=animal_table["animal_key"].astype(str).tolist(),
        image_labels=image_table["image_key"].astype(str).tolist(),
        boundary_adjusted=boundary_adjusted,
        positive_scale=positive_scale,
        positive_offset=positive_offset,
        pooled_median=max(pooled_median, SMALL_VALUE),
        pooled_std=pooled_std,
        n_obs_wt=int(np.sum(indexed["genotype_idx"] == 0)),
        n_obs_ko=int(np.sum(indexed["genotype_idx"] == 1)),
        n_animals_wt=int(np.sum(animal_table["genotype_idx"] == 0)),
        n_animals_ko=int(np.sum(animal_table["genotype_idx"] == 1)),
        n_images_wt=int(np.sum(image_table["genotype_idx"] == 0)),
        n_images_ko=int(np.sum(image_table["genotype_idx"] == 1)),
        analysis_id="instance",
        model_level="instance",
    )


def prepare_instance_metric_data(
    df: pd.DataFrame,
    muscle: str,
    compartment: str,
    metric: str,
    positive_likelihood: str = c3_metrics.DEFAULT_POSITIVE_LIKELIHOOD,
    bounded_likelihood: str = c3_metrics.DEFAULT_BOUNDED_LIKELIHOOD,
) -> PreparedMetricData:
    """Prepare one DMD_1X instance-level muscle/compartment/metric subset.

    Args:
        df (pd.DataFrame): Full DMD_1X cleaned instance table.
        muscle (str): Muscle label to analyze.
        compartment (str): Compartment label to analyze.
        metric (str): Metric column name.
        positive_likelihood (str): Likelihood for positive metrics.
        bounded_likelihood (str): Likelihood strategy for bounded metrics.

    Returns:
        PreparedMetricData: Prepared arrays with mitochondria nested in images.
    """

    family = metric_family_for_level(metric=metric, model_level="instance")
    required_columns = ["Condition", "Muscle", "Compartment", "Block", "image", "Id", metric]
    if metric in CLUSTERING_METRICS:
        required_columns.append("Image_Region")
    df_metric = filter_metric_dataframe(
        df=df,
        muscle=muscle,
        compartment=compartment,
        metric=metric,
        required_columns=required_columns,
    )
    if family == "positive":
        df_metric = df_metric.loc[df_metric["value"] > 0.0].copy()
    df_metric = df_metric.dropna(subset=["Condition", "Block", "image", "Id", "value"])
    df_metric = df_metric.sort_values(["Condition", "Block", "image", "Id"]).reset_index(drop=True)
    if df_metric.empty:
        region_text = " center-region" if metric in CLUSTERING_METRICS else ""
        raise ValueError(f"No{region_text} observations available for {muscle} / {compartment} / {metric}.")
    transformed = apply_metric_transform(
        df_metric=df_metric,
        metric=metric,
        family=family,
        positive_likelihood=positive_likelihood,
        bounded_likelihood=bounded_likelihood,
        real_likelihood=REAL_LIKELIHOODS[0],
    )
    (
        df_metric,
        raw_values,
        likelihood_name,
        boundary_adjusted,
        positive_scale,
        positive_offset,
        pooled_median,
        pooled_std,
    ) = transformed
    return build_prepared_instance_data(
        df_metric=df_metric,
        raw_values=raw_values,
        muscle=muscle,
        compartment=compartment,
        metric=metric,
        family=family,
        likelihood_name=likelihood_name,
        boundary_adjusted=boundary_adjusted,
        positive_scale=positive_scale,
        positive_offset=positive_offset,
        pooled_median=pooled_median,
        pooled_std=pooled_std,
    )


def prepare_image_summary_metric_data(
    df: pd.DataFrame,
    muscle: str,
    compartment: str,
    metric: str,
    positive_likelihood: str = c3_metrics.DEFAULT_POSITIVE_LIKELIHOOD,
    bounded_likelihood: str = c3_metrics.DEFAULT_BOUNDED_LIKELIHOOD,
    real_likelihood: str = REAL_LIKELIHOODS[0],
) -> PreparedMetricData:
    """Prepare one DMD_1X image-summary muscle/compartment/metric subset.

    Args:
        df (pd.DataFrame): Full DMD_1X image-summary table.
        muscle (str): Muscle label to analyze.
        compartment (str): Compartment label to analyze.
        metric (str): Metric column name.
        positive_likelihood (str): Likelihood for positive metrics.
        bounded_likelihood (str): Likelihood strategy for bounded metrics.
        real_likelihood (str): Likelihood for real-valued metrics.

    Returns:
        PreparedMetricData: Prepared arrays with one observation per image.
    """

    family = metric_family_for_level(metric=metric, model_level="image_summary")
    required_columns = [
        "Condition",
        "Muscle",
        "Compartment",
        "Block",
        "image",
        metric,
    ]
    df_metric = filter_metric_dataframe(
        df=df,
        muscle=muscle,
        compartment=compartment,
        metric=metric,
        required_columns=required_columns,
    )
    if family == "positive":
        df_metric = df_metric.loc[df_metric["value"] > 0.0].copy()
    df_metric = df_metric.dropna(subset=["Condition", "Block", "image", "value"])
    df_metric = df_metric.sort_values(["Condition", "Block", "image"]).reset_index(drop=True)
    if df_metric.empty:
        raise ValueError(f"No image-summary observations available for {muscle} / {compartment} / {metric}.")
    transformed = apply_metric_transform(
        df_metric=df_metric,
        metric=metric,
        family=family,
        positive_likelihood=positive_likelihood,
        bounded_likelihood=bounded_likelihood,
        real_likelihood=real_likelihood,
    )
    (
        df_metric,
        raw_values,
        likelihood_name,
        boundary_adjusted,
        positive_scale,
        positive_offset,
        pooled_median,
        pooled_std,
    ) = transformed
    return build_prepared_image_data(
        df_metric=df_metric,
        raw_values=raw_values,
        muscle=muscle,
        compartment=compartment,
        metric=metric,
        family=family,
        likelihood_name=likelihood_name,
        boundary_adjusted=boundary_adjusted,
        positive_scale=positive_scale,
        positive_offset=positive_offset,
        pooled_median=pooled_median,
        pooled_std=pooled_std,
    )


def fit_likelihood_arguments(fit_config: Any, model_level: str) -> tuple[str, str, str]:
    """Resolve likelihood arguments for one configured DMD_1X fit.

    Args:
        fit_config (Any): DMD_1X fit configuration.
        model_level (str): Model level, either ``instance`` or ``image_summary``.

    Returns:
        tuple[str, str, str]: Positive, bounded, and real likelihood names.
    """

    family = metric_family_for_level(metric=fit_config.metric, model_level=model_level)
    if family == "positive":
        return fit_config.likelihood, "beta", REAL_LIKELIHOODS[0]
    if family == "bounded":
        return "gamma", fit_config.likelihood, REAL_LIKELIHOODS[0]
    if family == "real":
        return "gamma", "beta", fit_config.likelihood
    raise ValueError(f"Unsupported configured metric: {fit_config.metric}")


def summarize_dmd_1x_model(
    data: PreparedMetricData,
    idata: az.InferenceData,
    sampling_seconds: float,
    random_seed: int,
    compartment: str,
) -> dict[str, Any]:
    """Summarize one fitted DMD_1X model and add compartment metadata.

    Args:
        data (PreparedMetricData): Prepared metric arrays.
        idata (az.InferenceData): Posterior samples.
        sampling_seconds (float): Sampler wall time in seconds.
        random_seed (int): Random seed used for the winning fit.
        compartment (str): Compartment label for this fit.

    Returns:
        dict[str, Any]: Summary row for the result CSV.
    """

    row = c3_metrics.summarize_model(
        data=data,
        idata=idata,
        sampling_seconds=sampling_seconds,
        random_seed=random_seed,
    )
    row["compartment"] = compartment
    return row


def analyze_metric(
    df: pd.DataFrame,
    muscle: str,
    compartment: str,
    metric: str,
    positive_likelihood: str,
    bounded_likelihood: str,
    real_likelihood: str,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
    random_seed: int,
    retry_on_warnings: bool,
    retry_max_draws: int,
    retry_max_tune: int,
    retry_max_target_accept: float,
    model_level: str,
) -> MetricAnalysisResult:
    """Fit one DMD_1X metric and return a summary row.

    Args:
        df (pd.DataFrame): Full DMD_1X measurements dataframe.
        muscle (str): Muscle label to analyze.
        compartment (str): Compartment label to analyze.
        metric (str): Metric column to fit.
        positive_likelihood (str): Positive-family likelihood name.
        bounded_likelihood (str): Bounded-family likelihood name.
        real_likelihood (str): Real-valued likelihood name.
        draws (int): Number of posterior draws per chain.
        tune (int): Number of warmup draws per chain.
        chains (int): Number of MCMC chains.
        cores (int): Number of CPU cores used by PyMC.
        target_accept (float): NUTS target acceptance rate.
        random_seed (int): Random seed for reproducible sampling.
        retry_on_warnings (bool): Whether to retry fits after sampler warnings.
        retry_max_draws (int): Upper cap for adaptive retry posterior draws per chain.
        retry_max_tune (int): Upper cap for adaptive retry warmup draws per chain.
        retry_max_target_accept (float): Upper cap for adaptive retry target acceptance.
        model_level (str): Model level, either ``instance`` or ``image_summary``.

    Returns:
        MetricAnalysisResult: Winning summary row plus prepared data and posterior samples.
    """

    if model_level == "image_summary":
        data = prepare_image_summary_metric_data(
            df=df,
            muscle=muscle,
            compartment=compartment,
            metric=metric,
            positive_likelihood=positive_likelihood,
            bounded_likelihood=bounded_likelihood,
            real_likelihood=real_likelihood,
        )
    elif model_level == "instance":
        data = prepare_instance_metric_data(
            df=df,
            muscle=muscle,
            compartment=compartment,
            metric=metric,
            positive_likelihood=positive_likelihood,
            bounded_likelihood=bounded_likelihood,
        )
    else:
        raise ValueError(f"Unsupported DMD_1X model level: {model_level}")
    attempt_results: list[MetricAnalysisResult] = []
    engine_warning_message = ""
    max_attempts = 3 if retry_on_warnings else 1
    for attempt_index in range(max_attempts):
        attempt_draws, attempt_tune, attempt_target_accept = c3_metrics.sampling_plan(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            attempt_index=attempt_index,
            warning_message=engine_warning_message,
            retry_max_draws=retry_max_draws,
            retry_max_tune=retry_max_tune,
            retry_max_target_accept=retry_max_target_accept,
        )
        try:
            model = c3_metrics.build_model(data)
            idata, sampling_seconds = c3_metrics.sample_model(
                model=model,
                draws=attempt_draws,
                tune=attempt_tune,
                chains=chains,
                cores=cores,
                target_accept=attempt_target_accept,
                random_seed=random_seed + attempt_index,
            )
            row = summarize_dmd_1x_model(
                data=data,
                idata=idata,
                sampling_seconds=sampling_seconds,
                random_seed=random_seed + attempt_index,
                compartment=compartment,
            )
            row["attempt"] = attempt_index + 1
            row["sampling_draws"] = attempt_draws
            row["sampling_tune"] = attempt_tune
            row["sampling_target_accept"] = attempt_target_accept
            attempt_result = MetricAnalysisResult(row=row, data=data, idata=idata)
            attempt_results.append(attempt_result)
            engine_warning_message = str(row.get("engine_warning_message", ""))
            if row.get("engine_fit_status") == "ok":
                return attempt_result
        except Exception as exc:  # pragma: no cover - defensive reporting for long batch runs.
            error_row = {
                "analysis_id": data.analysis_id,
                "model_level": data.model_level,
                "muscle": muscle,
                "compartment": compartment,
                "metric": metric,
                "family": data.family,
                "likelihood_name": data.likelihood_name,
                "wt_label": data.wt_label,
                "ko_label": data.ko_label,
                "n_obs_wt": data.n_obs_wt,
                "n_obs_ko": data.n_obs_ko,
                "n_animals_wt": data.n_animals_wt,
                "n_animals_ko": data.n_animals_ko,
                "n_images_wt": data.n_images_wt,
                "n_images_ko": data.n_images_ko,
                "boundary_adjusted": data.boundary_adjusted,
                "fit_status": "error",
                "engine_fit_status": "error",
                "ppc_fit_status": "error",
                "engine_warning_message": "error",
                "ppc_warning_message": "error",
                "warning_message": "error",
                "error_message": str(exc),
                "attempt": attempt_index + 1,
                "sampling_draws": attempt_draws,
                "sampling_tune": attempt_tune,
                "sampling_target_accept": attempt_target_accept,
            }
            error_row.update(c3_metrics.empty_bayes_factor_summary())
            attempt_results.append(MetricAnalysisResult(row=error_row, data=data, idata=None))
            engine_warning_message = "error"
    return min(attempt_results, key=lambda result: c3_metrics.fit_quality_key(result.row))


def apply_zoom_filter(df: pd.DataFrame, zoom_filter: Any | None) -> pd.DataFrame:
    """Apply an optional inclusive DMD_1X zoom filter.

    Args:
        df (pd.DataFrame): Loaded DMD_1X measurements dataframe.
        zoom_filter (Any | None): Filter config with minimum and maximum zoom values.

    Returns:
        pd.DataFrame: Filtered measurements dataframe.
    """

    if zoom_filter is None:
        return df
    if "Zoom" not in df.columns:
        raise KeyError("Zoom filter requires a 'Zoom' column in the DMD_1X measurements CSV.")
    zoom_values = pd.to_numeric(df["Zoom"], errors="coerce")
    filtered = df.loc[
        zoom_values.notna()
        & (zoom_values >= zoom_filter.minimum_zoom)
        & (zoom_values <= zoom_filter.maximum_zoom)
    ].copy()
    if filtered.empty:
        raise ValueError(
            "Zoom filter removed every DMD_1X row: "
            f"{zoom_filter.minimum_zoom} <= Zoom <= {zoom_filter.maximum_zoom}."
        )
    return filtered


def load_measurements(paths: Any, zoom_filter: Any | None = None) -> pd.DataFrame:
    """Load one or more configured DMD_1X image-summary CSVs.

    Args:
        paths (Any): Path config with ``input_csv`` and ``input_csvs`` fields.
        zoom_filter (Any | None): Optional inclusive zoom filter config.

    Returns:
        pd.DataFrame: Concatenated and optionally zoom-filtered measurements dataframe.
    """

    frames: list[pd.DataFrame] = []
    if paths.input_csv is not None:
        frames.append(pd.read_csv(paths.input_csv))
    for input_csv in paths.input_csvs:
        frames.append(pd.read_csv(input_csv))
    if not frames:
        raise ValueError("No input CSV paths were configured.")
    return apply_zoom_filter(
        df=pd.concat(frames, ignore_index=True, sort=False),
        zoom_filter=zoom_filter,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for DMD_1X Bayesian fitting.

    Args:
        None: Reads arguments from ``sys.argv``.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Fit DMD_1X hierarchical Bayesian models by compartment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH)
    return parser.parse_args()


def merge_result_rows(
    output_path: Path,
    new_rows_df: pd.DataFrame,
    update_mode: str,
) -> pd.DataFrame:
    """Merge refreshed rows into an existing DMD_1X summary CSV.

    Args:
        output_path (Path): Summary CSV path.
        new_rows_df (pd.DataFrame): Newly computed rows.
        update_mode (str): Merge strategy, either ``merge`` or ``replace``.

    Returns:
        pd.DataFrame: Final summary dataframe to save.
    """

    sort_columns = ["muscle", "compartment", "metric"]
    if update_mode == "replace" or not output_path.exists():
        return new_rows_df.sort_values(sort_columns).reset_index(drop=True)
    existing_df = pd.read_csv(output_path)
    rerun_keys = set(
        zip(
            new_rows_df["muscle"],
            new_rows_df["compartment"],
            new_rows_df["metric"],
            strict=False,
        )
    )
    preserved_df = existing_df.loc[
        ~existing_df.apply(
            lambda row: (row["muscle"], row["compartment"], row["metric"]) in rerun_keys,
            axis=1,
        )
    ].copy()
    merged_df = pd.concat([preserved_df, new_rows_df], ignore_index=True, sort=False)
    return merged_df.sort_values(sort_columns).reset_index(drop=True)


def finalize_fit_artifacts(
    result: MetricAnalysisResult,
    trace_dir: Path | None,
    random_seed: int,
    compartment: str,
) -> MetricAnalysisResult:
    """Attach PPC draws, response-scale variables, and optional DMD_1X trace output.

    Args:
        result (MetricAnalysisResult): Winning fit result.
        trace_dir (Path | None): Directory where traces are saved, or ``None``.
        random_seed (int): Posterior predictive seed.
        compartment (str): Compartment label for trace metadata.

    Returns:
        MetricAnalysisResult: Updated result after finalization.
    """

    row = result.row
    row["trace_path"] = ""
    if result.data is None or result.idata is None:
        return result
    model = c3_metrics.build_model(result.data)
    idata = c3_metrics.attach_posterior_predictive(
        model=model,
        idata=result.idata,
        random_seed=random_seed,
    )
    idata = c3_metrics.attach_response_scale_posterior(data=result.data, idata=idata)
    idata.attrs["compartment"] = compartment
    idata.attrs["fit_stem"] = fit_stem(
        muscle=result.data.muscle,
        compartment=compartment,
        metric=result.data.metric,
    )
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        output_path = trace_path_for_fit(
            trace_dir=trace_dir,
            muscle=result.data.muscle,
            compartment=compartment,
            metric=result.data.metric,
        )
        az.to_netcdf(idata, output_path)
        row["trace_path"] = str(output_path)
    return MetricAnalysisResult(row=row, data=result.data, idata=idata)


def run_analysis(config_path: Path, analysis: Any) -> None:
    """Run one configured DMD_1X Bayesian analysis and save its summary CSV.

    Args:
        config_path (Path): YAML configuration path used for status messages.
        analysis (Any): Normalized DMD_1X analysis configuration.

    Returns:
        None: Writes the summary CSV and optional NetCDF traces.
    """

    if not analysis.enabled:
        print(f"Analysis {analysis.analysis_id} is disabled in {config_path}; nothing to do.")
        return
    df = load_measurements(analysis.paths, analysis.filters)
    dmd_1x_config.validate_config_groups(
        config=analysis,
        valid_muscles=set(df["Muscle"].dropna().astype(str).unique().tolist()),
        valid_compartments=set(df["Compartment"].dropna().astype(str).unique().tolist()),
    )
    fit_configs = dmd_1x_config.repeated_fit_configs(analysis)
    if not fit_configs:
        print(f"No {analysis.analysis_id} fits have repeat=true in {config_path}; nothing to do.")
        return
    rows: list[dict[str, Any]] = []
    for fit_index, fit_config in enumerate(fit_configs):
        seed = analysis.runtime.seed + fit_index
        positive_likelihood, bounded_likelihood, real_likelihood = fit_likelihood_arguments(
            fit_config=fit_config,
            model_level=analysis.analysis_id,
        )
        print(
            "Fitting "
            f"{analysis.analysis_id} {fit_config.metric} for "
            f"{fit_config.muscle} / {fit_config.compartment}..."
        )
        start = time.perf_counter()
        result = analyze_metric(
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
        finalized_result = finalize_fit_artifacts(
            result=result,
            trace_dir=None if not analysis.runtime.save_idata else analysis.paths.trace_dir,
            random_seed=seed + 10_000,
            compartment=fit_config.compartment,
        )
        finalized_result.row["wall_seconds_total"] = time.perf_counter() - start
        rows.append(finalized_result.row)
    result_df = merge_result_rows(
        output_path=analysis.paths.summary_csv,
        new_rows_df=pd.DataFrame(rows),
        update_mode=analysis.runtime.summary_update_mode,
    )
    analysis.paths.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(analysis.paths.summary_csv, index=False)
    print(f"Saved {analysis.analysis_id} results to {analysis.paths.summary_csv}")


def load_config(path: Path) -> Any:
    """Load a DMD_1X Bayesian config with the metric sets from this module.

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


def main() -> None:
    """Run all configured DMD_1X hierarchical Bayesian analyses.

    Args:
        None: Reads command-line arguments.

    Returns:
        None: Saves configured analysis outputs.
    """

    args = parse_args()
    config = load_config(args.config)
    for analysis in config.analyses_by_id.values():
        run_analysis(config_path=args.config, analysis=analysis)


if __name__ == "__main__":
    main()
