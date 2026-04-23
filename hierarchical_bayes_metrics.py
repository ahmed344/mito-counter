"""Hierarchical Bayesian analysis for per-mitochondrion metrics."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from scipy.special import expit
from scipy.stats import skewnorm, wasserstein_distance

from hierarchical_bayes_config import (
    BayesFitConfig,
    DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH,
    load_hierarchical_bayes_config,
    repeated_fit_configs,
    validate_config_muscles,
)


INPUT_CSV = Path("/workspaces/mito-counter/data/Calpaine_3/results/measurments_cleaned.csv")
OUTPUT_CSV = Path(
    "/workspaces/mito-counter/data/Calpaine_3/results/hierarchical_bayes_statistics.csv"
)
TRACE_DIR = Path("/workspaces/mito-counter/data/Calpaine_3/results/bayes_traces")
DEFAULT_POSITIVE_LIKELIHOOD = "gamma"
DEFAULT_BOUNDED_LIKELIHOOD = "auto"
POSITIVE_LIKELIHOODS = ("gamma", "lognormal")
BOUNDED_LIKELIHOODS = ("auto", "beta", "zero_one_inflated_beta", "logitnormal", "logit_skew_normal")

POSITIVE_METRICS = (
    "Area",
    "Corrected_area",
    "Major_axis_length",
    "Minor_axis_length",
    "Minimum_Feret_Diameter",
    "Elongation",
    "NND",
)
BOUNDED_METRICS = (
    "Circularity",
    "Solidity",
)
GENOTYPE_ORDER = ("Wildtype", "Calpain_3_Knockout")
SMALL_VALUE = 1e-9
PPC_POSTERIOR_DRAWS = 200
PPC_OBSERVATION_LIMIT = 500
REFIT_ESS_THRESHOLD = 400.0
PPC_RELATIVE_ERROR_THRESHOLD = 0.35
PPC_DENSITY_BIN_LIMIT = 60
LOGITNORMAL_QUADRATURE_POINTS = 20
LOGIT_SKEW_NORMAL_QUANTILE_POINTS = 24
LOGIT_SKEW_NORMAL_MOMENT_CHUNK_SIZE = 4096


@dataclass(frozen=True)
class PreparedMetricData:
    """Container for one metric within one muscle."""

    muscle: str
    metric: str
    family: str
    likelihood_name: str
    wt_label: str
    ko_label: str
    y: np.ndarray
    observed_y: np.ndarray
    genotype_idx_obs: np.ndarray
    image_idx_obs: np.ndarray
    animal_idx_obs: np.ndarray
    animal_idx_image: np.ndarray
    genotype_idx_animal: np.ndarray
    genotype_idx_image: np.ndarray
    animal_labels: list[str]
    image_labels: list[str]
    boundary_adjusted: bool
    positive_scale: float
    pooled_median: float
    pooled_std: float
    n_obs_wt: int
    n_obs_ko: int
    n_animals_wt: int
    n_animals_ko: int
    n_images_wt: int
    n_images_ko: int


@dataclass
class MetricAnalysisResult:
    """Container for the winning fit artifacts of one muscle-metric analysis."""

    row: dict[str, Any]
    data: PreparedMetricData | None
    idata: az.InferenceData | None


def slugify_value(value: str) -> str:
    """Convert a string into a filesystem-friendly lowercase slug.

    Args:
        value (str): Raw label that will become part of a filename.

    Returns:
        str: Lowercase slug containing only alphanumerics and underscores.
    """

    raw = str(value).strip().lower()
    slug_chars = [character if character.isalnum() else "_" for character in raw]
    slug = "".join(slug_chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def fit_stem(muscle: str, metric: str) -> str:
    """Build a stable filename stem for one muscle-metric fit.

    Args:
        muscle (str): Muscle label for the current fit.
        metric (str): Metric name for the current fit.

    Returns:
        str: Predictable stem combining muscle and metric slugs.
    """

    return f"{slugify_value(muscle)}__{slugify_value(metric)}"


def trace_path_for_fit(trace_dir: Path, muscle: str, metric: str) -> Path:
    """Return the NetCDF path used to store one fit's inference data.

    Args:
        trace_dir (Path): Root directory where Bayesian trace files are saved.
        muscle (str): Muscle label for the current fit.
        metric (str): Metric name for the current fit.

    Returns:
        Path: NetCDF path for the requested fit.
    """

    return trace_dir / f"{fit_stem(muscle=muscle, metric=metric)}.nc"


def diagnostic_var_names(family: str, likelihood_name: str = "") -> list[str]:
    """Return the posterior variable names used for engine-check plots.

    Args:
        family (str): Metric family name, either ``positive`` or ``bounded``.
        likelihood_name (str): Concrete likelihood name used within the metric family.

    Returns:
        list[str]: Variable names suitable for trace, rank, and forest plots.
    """

    if family == "positive":
        return [
            "genotype_mean",
            "animal_variance",
            "image_variance",
            "mito_variance",
            "sigma_log_image_variance",
            "sigma_log_mito_variance",
            "delta_mean",
            "delta_image_variance",
            "delta_mito_variance",
        ]
    if likelihood_name == "logitnormal":
        return [
            "genotype_mean",
            "animal_logit_sigma",
            "image_sigma",
            "mito_sigma",
            "sigma_log_image_sigma",
            "sigma_log_mito_sigma",
            "delta_mean",
        ]
    if likelihood_name == "logit_skew_normal":
        return [
            "genotype_mean",
            "animal_logit_sigma",
            "image_sigma",
            "mito_sigma",
            "sigma_log_image_sigma",
            "sigma_log_mito_sigma",
            "skew_alpha",
            "delta_mean",
        ]
    return [
        "genotype_mean",
        "kappa_animal",
        "kappa_image",
        "kappa_mito",
        "sigma_log_kappa_image",
        "sigma_log_kappa_mito",
        "boundary_mass",
        "one_given_boundary",
        "delta_mean",
    ]


def biology_posterior_var_names(family: str) -> list[str]:
    """Return response-scale effect variables used for biology-facing posterior plots.

    Args:
        family (str): Metric family name, either ``positive`` or ``bounded``.

    Returns:
        list[str]: Response-scale posterior effect variable names.
    """

    if family == "positive":
        return [
            "delta_mean_response",
            "delta_image_variance_response",
            "delta_mito_variance_response",
        ]
    return [
        "delta_mean_response",
        "delta_image_variance_response",
        "delta_mito_variance_response",
    ]


def condition_sort_key(value: str) -> tuple[int, str]:
    """Return a stable sort key that keeps WT-like labels before KO-like labels.

    Args:
        value (str): Raw condition label from the input dataframe.

    Returns:
        tuple[int, str]: Sort key used to order condition labels consistently.
    """

    normalized = str(value).strip().lower()
    if "wildtype" in normalized or normalized == "wt" or normalized.endswith("_wt"):
        return (0, normalized)
    if "knockout" in normalized or normalized == "ko" or normalized.endswith("_ko"):
        return (1, normalized)
    return (2, normalized)


def determine_condition_labels(condition_values: list[str]) -> tuple[str, str]:
    """Determine the WT and KO labels present in the dataset.

    Args:
        condition_values (list[str]): Unique non-null condition labels.

    Returns:
        tuple[str, str]: The ordered `(wt_label, ko_label)` pair.
    """

    ordered = sorted(condition_values, key=condition_sort_key)
    if len(ordered) != 2:
        raise ValueError(f"Expected exactly 2 conditions, found {ordered}")
    return ordered[0], ordered[1]


def squeeze_open_interval_values(values: np.ndarray, epsilon: float = 1e-6) -> tuple[np.ndarray, bool]:
    """Move exact boundary values into the open interval required by open-interval models.

    Args:
        values (np.ndarray): Observed bounded metric values.
        epsilon (float): Small offset applied only when a value equals 0 or 1.

    Returns:
        tuple[np.ndarray, bool]: Adjusted values and whether any adjustment was applied.
    """

    adjusted = values.copy()
    touched = False
    zero_mask = adjusted <= 0.0
    one_mask = adjusted >= 1.0
    if np.any(zero_mask):
        adjusted[zero_mask] = epsilon
        touched = True
    if np.any(one_mask):
        adjusted[one_mask] = 1.0 - epsilon
        touched = True
    return adjusted, touched


def squeeze_beta_values(values: np.ndarray, epsilon: float = 1e-6) -> tuple[np.ndarray, bool]:
    """Move exact boundary values into the open interval required by the Beta distribution.

    Args:
        values (np.ndarray): Observed bounded metric values.
        epsilon (float): Small offset applied only when a value equals 0 or 1.

    Returns:
        tuple[np.ndarray, bool]: Adjusted values and whether any adjustment was applied.
    """

    return squeeze_open_interval_values(values=values, epsilon=epsilon)


def logit_transform(values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Compute a numerically stable logit transform for bounded values.

    Args:
        values (np.ndarray): Response-scale values expected to lie in ``[0, 1]``.
        epsilon (float): Small clipping margin used to keep the transform finite.

    Returns:
        np.ndarray: Logit-transformed values on the real line.
    """

    clipped = np.clip(np.asarray(values, dtype=float), epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


def sigmoid_tensor(values: pt.TensorVariable) -> pt.TensorVariable:
    """Apply the logistic inverse transform to a PyTensor variable.

    Args:
        values (pt.TensorVariable): Real-valued PyTensor variable on the logit scale.

    Returns:
        pt.TensorVariable: Variable transformed back to the open interval ``(0, 1)``.
    """

    return pt.sigmoid(values)


def logit_tensor(values: pt.TensorVariable, epsilon: float = 1e-6) -> pt.TensorVariable:
    """Compute a numerically stable logit transform for a PyTensor variable.

    Args:
        values (pt.TensorVariable): PyTensor variable expected on the bounded response scale.
        epsilon (float): Small clipping margin used to keep the transform finite.

    Returns:
        pt.TensorVariable: Variable transformed to the real line.
    """

    clipped = pt.clip(values, epsilon, 1.0 - epsilon)
    return pt.log(clipped) - pt.log1p(-clipped)


def logitnormal_response_moments(
    logit_location: np.ndarray,
    sigma: np.ndarray,
    quadrature_points: int = LOGITNORMAL_QUADRATURE_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate response-scale mean and variance for a logit-normal distribution.

    Args:
        logit_location (np.ndarray): Logit-scale location parameter values.
        sigma (np.ndarray): Positive logit-scale standard deviation values.
        quadrature_points (int): Number of Gauss-Hermite quadrature points to use.

    Returns:
        tuple[np.ndarray, np.ndarray]: Response-scale mean and variance arrays.
    """

    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_points)
    location_array = np.asarray(logit_location, dtype=float)
    sigma_array = np.asarray(sigma, dtype=float)
    expand_shape = (quadrature_points,) + (1,) * location_array.ndim
    scaled_nodes = np.sqrt(2.0) * nodes.reshape(expand_shape)
    weights_array = weights.reshape(expand_shape)
    response_samples = expit(location_array[None, ...] + sigma_array[None, ...] * scaled_nodes)
    response_mean = np.sum(weights_array * response_samples, axis=0) / np.sqrt(np.pi)
    response_second_moment = np.sum(weights_array * np.square(response_samples), axis=0) / np.sqrt(np.pi)
    response_variance = np.clip(response_second_moment - np.square(response_mean), 0.0, np.inf)
    return response_mean, response_variance


def logit_skew_normal_logp(
    value: pt.TensorVariable,
    mu: pt.TensorVariable,
    sigma: pt.TensorVariable,
    alpha: pt.TensorVariable,
) -> pt.TensorVariable:
    """Compute the bounded log-probability for a logit-skew-normal observation.

    Args:
        value (pt.TensorVariable): Observed response-scale values expected in ``(0, 1)``.
        mu (pt.TensorVariable): Location parameter on the logit scale.
        sigma (pt.TensorVariable): Positive scale parameter on the logit scale.
        alpha (pt.TensorVariable): Skewness parameter on the logit scale.

    Returns:
        pt.TensorVariable: Elementwise log-probability on the bounded response scale.
    """

    logit_value = logit_tensor(value)
    base_logp = pm.logp(pm.SkewNormal.dist(mu=mu, sigma=sigma, alpha=alpha), logit_value)
    log_jacobian = -pt.log(value) - pt.log1p(-value)
    in_support = pt.and_(pt.gt(value, 0.0), pt.lt(value, 1.0))
    return pt.switch(in_support, base_logp + log_jacobian, -np.inf)


def logit_skew_normal_random(
    mu: np.ndarray,
    sigma: np.ndarray,
    alpha: np.ndarray,
    rng: np.random.Generator | None = None,
    size: tuple[int, ...] | int | None = None,
) -> np.ndarray:
    """Draw random samples from a logit-skew-normal distribution.

    Args:
        mu (np.ndarray): Location parameter on the logit scale.
        sigma (np.ndarray): Positive scale parameter on the logit scale.
        alpha (np.ndarray): Skewness parameter on the logit scale.
        rng (np.random.Generator | None): Random number generator supplied by PyMC.
        size (tuple[int, ...] | int | None): Optional output size requested by PyMC.

    Returns:
        np.ndarray: Random draws transformed back onto the open interval ``(0, 1)``.
    """

    random_generator = np.random.default_rng() if rng is None else rng
    logit_draws = skewnorm.rvs(
        a=np.asarray(alpha, dtype=float),
        loc=np.asarray(mu, dtype=float),
        scale=np.asarray(sigma, dtype=float),
        size=size,
        random_state=random_generator,
    )
    return expit(logit_draws)


def logit_skew_normal_response_moments(
    logit_location: np.ndarray,
    sigma: np.ndarray,
    alpha: np.ndarray,
    quantile_points: int = LOGIT_SKEW_NORMAL_QUANTILE_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate response-scale mean and variance for a logit-skew-normal distribution.

    Args:
        logit_location (np.ndarray): Logit-scale location parameter values.
        sigma (np.ndarray): Positive logit-scale standard deviation values.
        alpha (np.ndarray): Logit-scale skewness parameter values.
        quantile_points (int): Number of midpoint quantile nodes used for approximation.

    Returns:
        tuple[np.ndarray, np.ndarray]: Response-scale mean and variance arrays.
    """

    probabilities = (np.arange(quantile_points, dtype=float) + 0.5) / float(quantile_points)
    location_array = np.asarray(logit_location, dtype=float)
    sigma_array = np.asarray(sigma, dtype=float)
    alpha_array = np.asarray(alpha, dtype=float)
    while sigma_array.ndim < location_array.ndim:
        sigma_array = sigma_array[..., None]
    while alpha_array.ndim < location_array.ndim:
        alpha_array = alpha_array[..., None]
    sigma_array = np.broadcast_to(sigma_array, location_array.shape)
    alpha_array = np.broadcast_to(alpha_array, location_array.shape)

    flat_location = location_array.reshape(-1)
    flat_sigma = sigma_array.reshape(-1)
    flat_alpha = alpha_array.reshape(-1)
    flat_mean = np.empty_like(flat_location)
    flat_variance = np.empty_like(flat_location)

    for start_index in range(0, flat_location.size, LOGIT_SKEW_NORMAL_MOMENT_CHUNK_SIZE):
        stop_index = min(start_index + LOGIT_SKEW_NORMAL_MOMENT_CHUNK_SIZE, flat_location.size)
        chunk_location = flat_location[start_index:stop_index]
        chunk_sigma = flat_sigma[start_index:stop_index]
        chunk_alpha = flat_alpha[start_index:stop_index]
        logit_samples = skewnorm.ppf(
            probabilities[:, None],
            a=chunk_alpha[None, :],
            loc=chunk_location[None, :],
            scale=chunk_sigma[None, :],
        )
        response_samples = expit(logit_samples)
        flat_mean[start_index:stop_index] = np.mean(response_samples, axis=0)
        flat_variance[start_index:stop_index] = np.var(response_samples, axis=0)

    response_mean = flat_mean.reshape(location_array.shape)
    response_variance = np.clip(flat_variance.reshape(location_array.shape), 0.0, np.inf)
    return response_mean, response_variance


def resolve_bounded_likelihood(values: np.ndarray, bounded_likelihood: str) -> str:
    """Resolve the bounded-data likelihood choice for one metric subset.

    Args:
        values (np.ndarray): Raw bounded observations before any preprocessing.
        bounded_likelihood (str): Requested bounded likelihood strategy.

    Returns:
        str: Concrete bounded likelihood name used to fit the current subset.
    """

    if bounded_likelihood == "auto":
        has_boundary_mass = bool(np.any((values <= 0.0) | (values >= 1.0)))
        return "zero_one_inflated_beta" if has_boundary_mass else "beta"
    return bounded_likelihood


def lognormal_mu_sigma_from_mean_variance(
    mean: np.ndarray,
    variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert response-scale mean and variance into LogNormal parameters.

    Args:
        mean (np.ndarray): Positive response-scale means.
        variance (np.ndarray): Positive response-scale variances.

    Returns:
        tuple[np.ndarray, np.ndarray]: Underlying Normal `mu` and `sigma` arrays.
    """

    clipped_mean = np.clip(mean, SMALL_VALUE, None)
    clipped_variance = np.clip(variance, SMALL_VALUE, None)
    sigma_squared = np.log1p(clipped_variance / np.square(clipped_mean))
    sigma = np.sqrt(np.clip(sigma_squared, SMALL_VALUE, None))
    mu = np.log(clipped_mean) - 0.5 * sigma_squared
    return mu, sigma


def combine_warning_messages(messages: list[str]) -> str:
    """Merge semicolon-separated warning fragments into one normalized string.

    Args:
        messages (list[str]): Individual warning strings that may contain `;` separators.

    Returns:
        str: Deduplicated warning labels joined by `;`.
    """

    merged: list[str] = []
    for message in messages:
        for fragment in str(message).split(";"):
            cleaned = fragment.strip()
            if cleaned and cleaned not in merged:
                merged.append(cleaned)
    return ";".join(merged)


def prepare_metric_data(
    df: pd.DataFrame,
    muscle: str,
    metric: str,
    positive_likelihood: str = DEFAULT_POSITIVE_LIKELIHOOD,
    bounded_likelihood: str = DEFAULT_BOUNDED_LIKELIHOOD,
) -> PreparedMetricData:
    """Prepare one muscle/metric subset with integer indices for the hierarchy.

    Args:
        df (pd.DataFrame): Full cleaned measurements table.
        muscle (str): Muscle name to analyze separately.
        metric (str): Metric column to model.
        positive_likelihood (str): Positive-family likelihood name to use when applicable.
        bounded_likelihood (str): Bounded-family likelihood strategy to use when applicable.

    Returns:
        PreparedMetricData: Structured arrays and metadata for model construction.
    """

    family = "positive" if metric in POSITIVE_METRICS else "bounded"
    wt_label, ko_label = determine_condition_labels(
        [str(value) for value in df["Condition"].dropna().unique().tolist()]
    )
    df_metric = (
        df.loc[df["Muscle"] == muscle, ["Condition", "Block", "image", "Id", metric]]
        .copy()
        .rename(columns={metric: "value"})
    )
    df_metric["value"] = pd.to_numeric(df_metric["value"], errors="coerce")
    df_metric = df_metric.dropna(subset=["Condition", "Block", "image", "value"])
    df_metric = df_metric.sort_values(["Condition", "Block", "image", "Id"]).reset_index(drop=True)

    raw_values = df_metric["value"].to_numpy(dtype=float)
    values = raw_values.copy()
    boundary_adjusted = False
    if family == "positive" and positive_likelihood not in POSITIVE_LIKELIHOODS:
        raise ValueError(f"Unsupported positive likelihood: {positive_likelihood}")
    likelihood_name = (
        positive_likelihood
        if family == "positive"
        else resolve_bounded_likelihood(values=raw_values, bounded_likelihood=bounded_likelihood)
    )
    if family == "bounded" and likelihood_name not in BOUNDED_LIKELIHOODS[1:]:
        raise ValueError(f"Unsupported bounded likelihood: {likelihood_name}")
    if family == "bounded" and likelihood_name in ("beta", "logitnormal", "logit_skew_normal"):
        values, boundary_adjusted = squeeze_open_interval_values(values)
    observed_y = raw_values.copy()

    positive_scale = 1.0
    if family == "positive":
        positive_scale = max(float(np.median(values)), SMALL_VALUE)
        values = values / positive_scale

    df_metric["value"] = values

    pooled_median = float(np.median(values))
    pooled_std = float(np.std(values, ddof=1)) if len(values) > 1 else SMALL_VALUE
    pooled_std = max(pooled_std, SMALL_VALUE)

    genotype_map = {wt_label: 0, ko_label: 1}
    df_metric["genotype_idx"] = df_metric["Condition"].map(genotype_map).astype(int)
    df_metric["animal_key"] = df_metric["Condition"].astype(str) + "__" + df_metric["Block"].astype(str)

    animal_table = (
        df_metric[["animal_key", "Condition", "Block"]]
        .drop_duplicates()
        .sort_values(["Condition", "Block"])
        .reset_index(drop=True)
    )
    animal_table["animal_idx"] = np.arange(len(animal_table), dtype=int)

    image_table = (
        df_metric[["image", "animal_key", "Condition"]]
        .drop_duplicates()
        .sort_values(["Condition", "animal_key", "image"])
        .reset_index(drop=True)
    )
    image_table["image_idx"] = np.arange(len(image_table), dtype=int)

    animal_idx_map = animal_table.set_index("animal_key")["animal_idx"].to_dict()
    image_idx_map = image_table.set_index("image")["image_idx"].to_dict()

    df_metric["animal_idx"] = df_metric["animal_key"].map(animal_idx_map).astype(int)
    df_metric["image_idx"] = df_metric["image"].map(image_idx_map).astype(int)

    image_table["animal_idx"] = image_table["animal_key"].map(animal_idx_map).astype(int)
    animal_table["genotype_idx"] = animal_table["Condition"].map(genotype_map).astype(int)
    image_table["genotype_idx"] = image_table["Condition"].map(genotype_map).astype(int)

    return PreparedMetricData(
        muscle=muscle,
        metric=metric,
        family=family,
        likelihood_name=likelihood_name,
        wt_label=wt_label,
        ko_label=ko_label,
        y=df_metric["value"].to_numpy(dtype=float),
        observed_y=observed_y,
        genotype_idx_obs=df_metric["genotype_idx"].to_numpy(dtype=int),
        image_idx_obs=df_metric["image_idx"].to_numpy(dtype=int),
        animal_idx_obs=df_metric["animal_idx"].to_numpy(dtype=int),
        animal_idx_image=image_table["animal_idx"].to_numpy(dtype=int),
        genotype_idx_animal=animal_table["genotype_idx"].to_numpy(dtype=int),
        genotype_idx_image=image_table["genotype_idx"].to_numpy(dtype=int),
        animal_labels=animal_table["animal_key"].astype(str).tolist(),
        image_labels=image_table["image"].astype(str).tolist(),
        boundary_adjusted=boundary_adjusted,
        positive_scale=positive_scale,
        pooled_median=max(pooled_median, SMALL_VALUE),
        pooled_std=pooled_std,
        n_obs_wt=int(np.sum(df_metric["genotype_idx"] == 0)),
        n_obs_ko=int(np.sum(df_metric["genotype_idx"] == 1)),
        n_animals_wt=int(np.sum(animal_table["genotype_idx"] == 0)),
        n_animals_ko=int(np.sum(animal_table["genotype_idx"] == 1)),
        n_images_wt=int(np.sum(image_table["genotype_idx"] == 0)),
        n_images_ko=int(np.sum(image_table["genotype_idx"] == 1)),
    )


def model_coords(data: PreparedMetricData) -> dict[str, list[Any] | np.ndarray]:
    """Build coordinate labels for a PyMC model.

    Args:
        data (PreparedMetricData): Prepared metric-specific arrays and labels.

    Returns:
        dict[str, list[Any] | np.ndarray]: Coordinate mapping used by PyMC dims.
    """

    return {
        "genotype": [data.wt_label, data.ko_label],
        "animal": data.animal_labels,
        "image": data.image_labels,
        "obs_id": np.arange(data.y.shape[0]),
    }


def mean_over_selected_units(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Average a draw-aligned array over selected units on its last axis.

    Args:
        values (np.ndarray): Array whose last axis indexes units such as animals or images.
        mask (np.ndarray): Boolean mask selecting the units to average on the last axis.

    Returns:
        np.ndarray: Input draws averaged over the selected units.
    """

    selected_mask = np.asarray(mask, dtype=bool)
    if not np.any(selected_mask):
        raise ValueError("Expected at least one selected unit when averaging posterior draws.")
    return np.mean(np.asarray(values, dtype=float)[..., selected_mask], axis=-1)


def positive_response_scale_variance_components(
    data: PreparedMetricData,
    posterior: xr.Dataset,
) -> dict[str, np.ndarray]:
    """Compute response-scale positive-model variance components by genotype.

    Args:
        data (PreparedMetricData): Prepared metric-specific arrays and labels.
        posterior (xr.Dataset): Posterior dataset containing positive-model variance draws.

    Returns:
        dict[str, np.ndarray]: Draw-aligned WT, KO, and shared variance components.
    """

    variance_scale = data.positive_scale**2
    image_variance_by_image = variance_scale * np.asarray(posterior["image_variance_by_image"], dtype=float)
    mito_variance_by_image = variance_scale * np.asarray(posterior["mito_variance_by_image"], dtype=float)
    animal_variance_shared = variance_scale * np.asarray(posterior["animal_variance"], dtype=float)
    wt_images = data.genotype_idx_image == 0
    ko_images = data.genotype_idx_image == 1
    wt_image_variance = mean_over_selected_units(image_variance_by_image, wt_images)
    ko_image_variance = mean_over_selected_units(image_variance_by_image, ko_images)
    wt_mito_variance = mean_over_selected_units(mito_variance_by_image, wt_images)
    ko_mito_variance = mean_over_selected_units(mito_variance_by_image, ko_images)
    return {
        "wt_image_variance": wt_image_variance,
        "ko_image_variance": ko_image_variance,
        "delta_image_variance": ko_image_variance - wt_image_variance,
        "wt_mito_variance": wt_mito_variance,
        "ko_mito_variance": ko_mito_variance,
        "delta_mito_variance": ko_mito_variance - wt_mito_variance,
        "animal_variance_shared": animal_variance_shared,
    }


def beta_response_scale_variance_components(
    data: PreparedMetricData,
    posterior: xr.Dataset,
) -> dict[str, np.ndarray]:
    """Compute response-scale Beta-family variance components by genotype.

    Args:
        data (PreparedMetricData): Prepared metric-specific arrays and labels.
        posterior (xr.Dataset): Posterior dataset containing bounded-model mean and kappa draws.

    Returns:
        dict[str, np.ndarray]: Draw-aligned WT, KO, and shared variance components.
    """

    animal_mean = np.asarray(posterior["animal_mean"], dtype=float)
    image_mean = np.asarray(posterior["image_mean"], dtype=float)
    kappa_animal = np.asarray(posterior["kappa_animal"], dtype=float)
    kappa_image_by_image = np.asarray(posterior["kappa_image_by_image"], dtype=float)
    kappa_mito_by_image = np.asarray(posterior["kappa_mito_by_image"], dtype=float)
    wt_animals = data.genotype_idx_animal == 0
    ko_animals = data.genotype_idx_animal == 1
    wt_images = data.genotype_idx_image == 0
    ko_images = data.genotype_idx_image == 1

    animal_var_wt = mean_over_selected_units(
        animal_mean[..., wt_animals] * (1.0 - animal_mean[..., wt_animals]),
        np.ones(int(np.sum(wt_animals)), dtype=bool),
    ) / (kappa_animal + 1.0)
    animal_var_ko = mean_over_selected_units(
        animal_mean[..., ko_animals] * (1.0 - animal_mean[..., ko_animals]),
        np.ones(int(np.sum(ko_animals)), dtype=bool),
    ) / (kappa_animal + 1.0)

    animal_parent_mean_by_image = animal_mean[..., data.animal_idx_image]
    image_variance_by_image = animal_parent_mean_by_image * (1.0 - animal_parent_mean_by_image)
    image_variance_by_image = image_variance_by_image / (kappa_image_by_image + 1.0)
    mito_variance_by_image = image_mean * (1.0 - image_mean)
    mito_variance_by_image = mito_variance_by_image / (kappa_mito_by_image + 1.0)
    wt_image_variance = mean_over_selected_units(image_variance_by_image, wt_images)
    ko_image_variance = mean_over_selected_units(image_variance_by_image, ko_images)
    wt_mito_variance = mean_over_selected_units(mito_variance_by_image, wt_images)
    ko_mito_variance = mean_over_selected_units(mito_variance_by_image, ko_images)
    return {
        "wt_image_variance": wt_image_variance,
        "ko_image_variance": ko_image_variance,
        "delta_image_variance": ko_image_variance - wt_image_variance,
        "wt_mito_variance": wt_mito_variance,
        "ko_mito_variance": ko_mito_variance,
        "delta_mito_variance": ko_mito_variance - wt_mito_variance,
        "animal_variance_shared": 0.5 * (animal_var_wt + animal_var_ko),
    }


def logitnormal_response_scale_variance_components(
    data: PreparedMetricData,
    posterior: xr.Dataset,
) -> dict[str, np.ndarray]:
    """Compute response-scale logit-normal variance components by genotype.

    Args:
        data (PreparedMetricData): Prepared metric-specific arrays and labels.
        posterior (xr.Dataset): Posterior dataset containing logit-normal latent draws.

    Returns:
        dict[str, np.ndarray]: Draw-aligned WT, KO, and shared variance components.
    """

    animal_mean = np.asarray(posterior["animal_mean"], dtype=float)
    image_mean = np.asarray(posterior["image_mean"], dtype=float)
    image_logit_mean = np.asarray(posterior["image_logit_mean"], dtype=float)
    mito_sigma_by_image = np.asarray(posterior["mito_sigma_by_image"], dtype=float)
    wt_animals = data.genotype_idx_animal == 0
    ko_animals = data.genotype_idx_animal == 1
    wt_images = data.genotype_idx_image == 0
    ko_images = data.genotype_idx_image == 1

    animal_var_wt = np.var(animal_mean[..., wt_animals], axis=-1)
    animal_var_ko = np.var(animal_mean[..., ko_animals], axis=-1)
    image_var_wt = np.var(image_mean[..., wt_images], axis=-1)
    image_var_ko = np.var(image_mean[..., ko_images], axis=-1)
    _, mito_variance_by_image = logitnormal_response_moments(
        logit_location=image_logit_mean,
        sigma=mito_sigma_by_image,
    )
    wt_mito_variance = mean_over_selected_units(mito_variance_by_image, wt_images)
    ko_mito_variance = mean_over_selected_units(mito_variance_by_image, ko_images)
    return {
        "wt_image_variance": image_var_wt,
        "ko_image_variance": image_var_ko,
        "delta_image_variance": image_var_ko - image_var_wt,
        "wt_mito_variance": wt_mito_variance,
        "ko_mito_variance": ko_mito_variance,
        "delta_mito_variance": ko_mito_variance - wt_mito_variance,
        "animal_variance_shared": 0.5 * (animal_var_wt + animal_var_ko),
    }


def logit_skew_normal_response_scale_variance_components(
    data: PreparedMetricData,
    posterior: xr.Dataset,
) -> dict[str, np.ndarray]:
    """Compute response-scale logit-skew-normal variance components by genotype.

    Args:
        data (PreparedMetricData): Prepared metric-specific arrays and labels.
        posterior (xr.Dataset): Posterior dataset containing logit-skew-normal latent draws.

    Returns:
        dict[str, np.ndarray]: Draw-aligned WT, KO, and shared variance components.
    """

    animal_mean = np.asarray(posterior["animal_mean"], dtype=float)
    image_mean = np.asarray(posterior["image_mean"], dtype=float)
    image_logit_mean = np.asarray(posterior["image_logit_mean"], dtype=float)
    mito_sigma_by_image = np.asarray(posterior["mito_sigma_by_image"], dtype=float)
    skew_alpha = np.asarray(posterior["skew_alpha"], dtype=float)
    wt_animals = data.genotype_idx_animal == 0
    ko_animals = data.genotype_idx_animal == 1
    wt_images = data.genotype_idx_image == 0
    ko_images = data.genotype_idx_image == 1

    animal_var_wt = np.var(animal_mean[..., wt_animals], axis=-1)
    animal_var_ko = np.var(animal_mean[..., ko_animals], axis=-1)
    image_var_wt = np.var(image_mean[..., wt_images], axis=-1)
    image_var_ko = np.var(image_mean[..., ko_images], axis=-1)
    _, mito_variance_by_image = logit_skew_normal_response_moments(
        logit_location=image_logit_mean,
        sigma=mito_sigma_by_image,
        alpha=skew_alpha,
    )
    wt_mito_variance = mean_over_selected_units(mito_variance_by_image, wt_images)
    ko_mito_variance = mean_over_selected_units(mito_variance_by_image, ko_images)
    return {
        "wt_image_variance": image_var_wt,
        "ko_image_variance": image_var_ko,
        "delta_image_variance": image_var_ko - image_var_wt,
        "wt_mito_variance": wt_mito_variance,
        "ko_mito_variance": ko_mito_variance,
        "delta_mito_variance": ko_mito_variance - wt_mito_variance,
        "animal_variance_shared": 0.5 * (animal_var_wt + animal_var_ko),
    }


def bounded_response_scale_variance_components(
    data: PreparedMetricData,
    posterior: xr.Dataset,
) -> dict[str, np.ndarray]:
    """Compute response-scale bounded-model variance components by genotype.

    Args:
        data (PreparedMetricData): Prepared metric-specific arrays and labels.
        posterior (xr.Dataset): Posterior dataset containing bounded-model latent draws.

    Returns:
        dict[str, np.ndarray]: Draw-aligned WT, KO, and shared variance components.
    """

    if data.likelihood_name == "logitnormal":
        return logitnormal_response_scale_variance_components(data=data, posterior=posterior)
    if data.likelihood_name == "logit_skew_normal":
        return logit_skew_normal_response_scale_variance_components(data=data, posterior=posterior)
    return beta_response_scale_variance_components(data=data, posterior=posterior)


def build_positive_model(data: PreparedMetricData) -> pm.Model:
    """Construct the Gamma hierarchy for positive continuous metrics.

    Args:
        data (PreparedMetricData): Prepared metric-specific arrays and labels.

    Returns:
        pm.Model: Fully specified PyMC model on the natural response scale.
    """

    coords = model_coords(data)
    variance_scale = max(data.pooled_std**2, SMALL_VALUE)
    prior_sigma = max(2.0 * data.pooled_std, SMALL_VALUE)

    with pm.Model(coords=coords) as model:
        genotype_idx_obs = pm.Data("genotype_idx_obs", data.genotype_idx_obs, dims="obs_id")
        image_idx_obs = pm.Data("image_idx_obs", data.image_idx_obs, dims="obs_id")
        animal_idx_image = pm.Data("animal_idx_image", data.animal_idx_image, dims="image")
        genotype_idx_animal = pm.Data(
            "genotype_idx_animal", data.genotype_idx_animal, dims="animal"
        )
        genotype_idx_image = pm.Data("genotype_idx_image", data.genotype_idx_image, dims="image")

        genotype_mean = pm.Gamma(
            "genotype_mean",
            mu=data.pooled_median,
            sigma=prior_sigma,
            dims="genotype",
        )
        animal_variance = pm.HalfNormal("animal_variance", sigma=variance_scale)
        log_image_variance_genotype = pm.Normal(
            "log_image_variance_genotype",
            mu=np.log(variance_scale),
            sigma=1.5,
            dims="genotype",
        )
        sigma_log_image_variance = pm.HalfNormal("sigma_log_image_variance", sigma=1.0)
        image_variance_offset = pm.Normal("image_variance_offset", mu=0.0, sigma=1.0, dims="image")
        log_mito_variance_genotype = pm.Normal(
            "log_mito_variance_genotype",
            mu=np.log(variance_scale),
            sigma=1.5,
            dims="genotype",
        )
        sigma_log_mito_variance = pm.HalfNormal("sigma_log_mito_variance", sigma=1.0)
        mito_variance_offset = pm.Normal("mito_variance_offset", mu=0.0, sigma=1.0, dims="image")

        image_variance = pm.Deterministic(
            "image_variance",
            pt.exp(log_image_variance_genotype),
            dims="genotype",
        )
        image_variance_by_image = pm.Deterministic(
            "image_variance_by_image",
            pt.exp(log_image_variance_genotype[genotype_idx_image] + sigma_log_image_variance * image_variance_offset),
            dims="image",
        )
        mito_variance = pm.Deterministic(
            "mito_variance",
            pt.exp(log_mito_variance_genotype),
            dims="genotype",
        )
        mito_variance_by_image = pm.Deterministic(
            "mito_variance_by_image",
            pt.exp(log_mito_variance_genotype[genotype_idx_image] + sigma_log_mito_variance * mito_variance_offset),
            dims="image",
        )

        animal_sigma = pm.Deterministic("animal_sigma", pt.sqrt(animal_variance))
        image_sigma_by_image = pm.Deterministic(
            "image_sigma_by_image",
            pt.sqrt(image_variance_by_image),
            dims="image",
        )
        mito_sigma_by_image = pm.Deterministic(
            "mito_sigma_by_image",
            pt.sqrt(mito_variance_by_image),
            dims="image",
        )

        animal_mean = pm.Gamma(
            "animal_mean",
            mu=genotype_mean[genotype_idx_animal],
            sigma=animal_sigma,
            dims="animal",
        )
        image_mean = pm.Gamma(
            "image_mean",
            mu=animal_mean[animal_idx_image],
            sigma=image_sigma_by_image,
            dims="image",
        )
        if data.likelihood_name == "lognormal":
            observed_mean = image_mean[image_idx_obs]
            observed_variance = mito_variance_by_image[image_idx_obs]
            log_sigma_squared = pt.log1p(
                observed_variance / pt.clip(pt.square(observed_mean), SMALL_VALUE, np.inf)
            )
            log_sigma = pt.sqrt(pt.clip(log_sigma_squared, SMALL_VALUE, np.inf))
            log_mu = pt.log(pt.clip(observed_mean, SMALL_VALUE, np.inf)) - 0.5 * log_sigma_squared
            pm.LogNormal(
                "observed_metric",
                mu=log_mu,
                sigma=log_sigma,
                observed=data.y,
                dims="obs_id",
            )
        else:
            pm.Gamma(
                "observed_metric",
                mu=image_mean[image_idx_obs],
                sigma=mito_sigma_by_image[image_idx_obs],
                observed=data.y,
                dims="obs_id",
            )

        pm.Deterministic("delta_mean", genotype_mean[1] - genotype_mean[0])
        pm.Deterministic("delta_image_variance", image_variance[1] - image_variance[0])
        pm.Deterministic("delta_mito_variance", mito_variance[1] - mito_variance[0])

    return model


def build_bounded_model(data: PreparedMetricData) -> pm.Model:
    """Construct the bounded hierarchy for shape metrics on the original response scale.

    Args:
        data (PreparedMetricData): Prepared metric-specific arrays and labels.

    Returns:
        pm.Model: Fully specified PyMC model on the original bounded scale.
    """

    coords = model_coords(data)

    with pm.Model(coords=coords) as model:
        genotype_idx_obs = pm.Data("genotype_idx_obs", data.genotype_idx_obs, dims="obs_id")
        image_idx_obs = pm.Data("image_idx_obs", data.image_idx_obs, dims="obs_id")
        animal_idx_image = pm.Data("animal_idx_image", data.animal_idx_image, dims="image")
        genotype_idx_animal = pm.Data(
            "genotype_idx_animal", data.genotype_idx_animal, dims="animal"
        )
        genotype_idx_image = pm.Data("genotype_idx_image", data.genotype_idx_image, dims="image")

        genotype_mean = pm.Beta("genotype_mean", alpha=2.0, beta=2.0, dims="genotype")
        if data.likelihood_name in ("logitnormal", "logit_skew_normal"):
            genotype_logit_mean = pm.Deterministic(
                "genotype_logit_mean",
                logit_tensor(genotype_mean),
                dims="genotype",
            )
            animal_logit_sigma = pm.HalfNormal("animal_logit_sigma", sigma=0.75)
            log_image_sigma_genotype = pm.Normal(
                "log_image_sigma_genotype",
                mu=np.log(0.35),
                sigma=1.0,
                dims="genotype",
            )
            sigma_log_image_sigma = pm.HalfNormal("sigma_log_image_sigma", sigma=0.5)
            image_sigma_offset = pm.Normal("image_sigma_offset", mu=0.0, sigma=1.0, dims="image")
            log_mito_sigma_genotype = pm.Normal(
                "log_mito_sigma_genotype",
                mu=np.log(0.35),
                sigma=1.0,
                dims="genotype",
            )
            sigma_log_mito_sigma = pm.HalfNormal("sigma_log_mito_sigma", sigma=0.5)
            mito_sigma_offset = pm.Normal("mito_sigma_offset", mu=0.0, sigma=1.0, dims="image")

            image_sigma = pm.Deterministic(
                "image_sigma",
                pt.exp(log_image_sigma_genotype),
                dims="genotype",
            )
            image_sigma_by_image = pm.Deterministic(
                "image_sigma_by_image",
                pt.exp(log_image_sigma_genotype[genotype_idx_image] + sigma_log_image_sigma * image_sigma_offset),
                dims="image",
            )
            mito_sigma = pm.Deterministic(
                "mito_sigma",
                pt.exp(log_mito_sigma_genotype),
                dims="genotype",
            )
            mito_sigma_by_image = pm.Deterministic(
                "mito_sigma_by_image",
                pt.exp(log_mito_sigma_genotype[genotype_idx_image] + sigma_log_mito_sigma * mito_sigma_offset),
                dims="image",
            )

            animal_logit_mean = pm.Normal(
                "animal_logit_mean",
                mu=genotype_logit_mean[genotype_idx_animal],
                sigma=animal_logit_sigma,
                dims="animal",
            )
            animal_mean = pm.Deterministic(
                "animal_mean",
                sigmoid_tensor(animal_logit_mean),
                dims="animal",
            )
            image_logit_mean = pm.Normal(
                "image_logit_mean",
                mu=animal_logit_mean[animal_idx_image],
                sigma=image_sigma_by_image,
                dims="image",
            )
            image_mean = pm.Deterministic(
                "image_mean",
                sigmoid_tensor(image_logit_mean),
                dims="image",
            )
            if data.likelihood_name == "logit_skew_normal":
                skew_alpha = pm.Normal("skew_alpha", mu=0.0, sigma=2.0)
                pm.CustomDist(
                    "observed_metric",
                    image_logit_mean[image_idx_obs],
                    mito_sigma_by_image[image_idx_obs],
                    skew_alpha,
                    logp=logit_skew_normal_logp,
                    random=logit_skew_normal_random,
                    observed=data.y,
                    dims="obs_id",
                )
            else:
                pm.LogitNormal(
                    "observed_metric",
                    mu=image_logit_mean[image_idx_obs],
                    sigma=mito_sigma_by_image[image_idx_obs],
                    observed=data.y,
                    dims="obs_id",
                )
            pm.Deterministic("delta_mean", genotype_mean[1] - genotype_mean[0])
            return model

        log_kappa_animal = pm.Normal("log_kappa_animal", mu=np.log(20.0), sigma=1.5)
        log_kappa_image_genotype = pm.Normal(
            "log_kappa_image_genotype",
            mu=np.log(20.0),
            sigma=1.5,
            dims="genotype",
        )
        sigma_log_kappa_image = pm.HalfNormal("sigma_log_kappa_image", sigma=1.0)
        kappa_image_offset = pm.Normal("kappa_image_offset", mu=0.0, sigma=1.0, dims="image")
        log_kappa_mito_genotype = pm.Normal(
            "log_kappa_mito_genotype",
            mu=np.log(20.0),
            sigma=1.5,
            dims="genotype",
        )
        sigma_log_kappa_mito = pm.HalfNormal("sigma_log_kappa_mito", sigma=1.0)
        kappa_mito_offset = pm.Normal("kappa_mito_offset", mu=0.0, sigma=1.0, dims="image")

        kappa_animal = pm.Deterministic("kappa_animal", pt.exp(log_kappa_animal))
        kappa_image = pm.Deterministic(
            "kappa_image",
            pt.exp(log_kappa_image_genotype),
            dims="genotype",
        )
        kappa_image_by_image = pm.Deterministic(
            "kappa_image_by_image",
            pt.exp(log_kappa_image_genotype[genotype_idx_image] + sigma_log_kappa_image * kappa_image_offset),
            dims="image",
        )
        kappa_mito = pm.Deterministic(
            "kappa_mito",
            pt.exp(log_kappa_mito_genotype),
            dims="genotype",
        )
        kappa_mito_by_image = pm.Deterministic(
            "kappa_mito_by_image",
            pt.exp(log_kappa_mito_genotype[genotype_idx_image] + sigma_log_kappa_mito * kappa_mito_offset),
            dims="image",
        )

        animal_alpha = pt.clip(genotype_mean[genotype_idx_animal] * kappa_animal, SMALL_VALUE, np.inf)
        animal_beta = pt.clip(
            (1.0 - genotype_mean[genotype_idx_animal]) * kappa_animal, SMALL_VALUE, np.inf
        )
        animal_mean = pm.Beta("animal_mean", alpha=animal_alpha, beta=animal_beta, dims="animal")

        image_parent_mean = animal_mean[animal_idx_image]
        image_alpha = pt.clip(image_parent_mean * kappa_image_by_image, SMALL_VALUE, np.inf)
        image_beta = pt.clip((1.0 - image_parent_mean) * kappa_image_by_image, SMALL_VALUE, np.inf)
        image_mean = pm.Beta("image_mean", alpha=image_alpha, beta=image_beta, dims="image")

        mito_parent_mean = image_mean[image_idx_obs]
        mito_kappa = kappa_mito_by_image[image_idx_obs]
        mito_alpha = pt.clip(mito_parent_mean * mito_kappa, SMALL_VALUE, np.inf)
        mito_beta = pt.clip((1.0 - mito_parent_mean) * mito_kappa, SMALL_VALUE, np.inf)
        if data.likelihood_name == "zero_one_inflated_beta":
            boundary_mass = pm.Beta("boundary_mass", alpha=1.5, beta=8.5, dims="genotype")
            one_given_boundary = pm.Beta("one_given_boundary", alpha=1.5, beta=1.5, dims="genotype")
            zero_weight = boundary_mass[genotype_idx_obs] * (1.0 - one_given_boundary[genotype_idx_obs])
            one_weight = boundary_mass[genotype_idx_obs] * one_given_boundary[genotype_idx_obs]
            beta_weight = 1.0 - boundary_mass[genotype_idx_obs]
            weights = pt.stack([zero_weight, one_weight, beta_weight], axis=1)
            zero_component = pm.DiracDelta.dist(0.0, shape=data.y.shape[0])
            one_component = pm.DiracDelta.dist(1.0, shape=data.y.shape[0])
            beta_component = pm.Beta.dist(alpha=mito_alpha, beta=mito_beta, shape=data.y.shape[0])
            pm.Mixture(
                "observed_metric",
                w=weights,
                comp_dists=[zero_component, one_component, beta_component],
                observed=data.y,
                dims="obs_id",
            )
        else:
            pm.Beta(
                "observed_metric",
                alpha=mito_alpha,
                beta=mito_beta,
                observed=data.y,
                dims="obs_id",
            )

        pm.Deterministic("delta_mean", genotype_mean[1] - genotype_mean[0])

    return model


def build_model(data: PreparedMetricData) -> pm.Model:
    """Dispatch to the appropriate model builder for one prepared metric dataset.

    Args:
        data (PreparedMetricData): Prepared metric-specific arrays and metadata.

    Returns:
        pm.Model: PyMC model matching the family and likelihood configured for `data`.
    """

    if data.family == "positive":
        return build_positive_model(data)
    return build_bounded_model(data)


def sample_model(
    model: pm.Model,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
    random_seed: int,
) -> tuple[az.InferenceData, float]:
    """Run NUTS sampling for one fitted model.

    Args:
        model (pm.Model): PyMC model to sample.
        draws (int): Number of posterior draws per chain.
        tune (int): Number of warmup draws per chain.
        chains (int): Number of MCMC chains.
        cores (int): Number of CPU cores used by PyMC.
        target_accept (float): NUTS target acceptance rate.
        random_seed (int): Random seed for reproducible sampling.

    Returns:
        tuple[az.InferenceData, float]: Posterior samples and wall-clock runtime in seconds.
    """

    start = time.perf_counter()
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
        )
    elapsed = time.perf_counter() - start
    return idata, elapsed


def attach_posterior_predictive(
    model: pm.Model,
    idata: az.InferenceData,
    random_seed: int,
) -> az.InferenceData:
    """Sample posterior predictive draws and attach them to an inference dataset.

    Args:
        model (pm.Model): Reconstructed PyMC model matching the saved posterior draws.
        idata (az.InferenceData): Posterior samples returned by ``pm.sample``.
        random_seed (int): Random seed for reproducible posterior predictive simulation.

    Returns:
        az.InferenceData: Updated inference data containing a ``posterior_predictive`` group.
    """

    with model:
        updated_idata = pm.sample_posterior_predictive(
            idata,
            var_names=["observed_metric"],
            extend_inferencedata=True,
            random_seed=random_seed,
            progressbar=True,
        )
    return updated_idata


def attach_response_scale_posterior(
    data: PreparedMetricData,
    idata: az.InferenceData,
) -> az.InferenceData:
    """Add response-scale derived posterior variables to an inference dataset.

    Args:
        data (PreparedMetricData): Prepared metric-specific arrays and labels.
        idata (az.InferenceData): Inference data containing posterior samples.

    Returns:
        az.InferenceData: Updated inference data with response-scale derived variables.
    """

    posterior = idata.posterior
    genotype_coords = posterior.coords["genotype"]
    chain_coords = posterior.coords["chain"]
    draw_coords = posterior.coords["draw"]

    response_variables: dict[str, xr.DataArray] = {}
    if data.family == "positive":
        mean_scale = data.positive_scale
        variance_draws = positive_response_scale_variance_components(data=data, posterior=posterior)
        response_variables = {
            "genotype_mean_response": posterior["genotype_mean"] * mean_scale,
            "delta_mean_response": posterior["delta_mean"] * mean_scale,
            "animal_variance_shared_response": xr.DataArray(
                variance_draws["animal_variance_shared"],
                dims=("chain", "draw"),
                coords={"chain": chain_coords, "draw": draw_coords},
            ),
            "image_variance_response": xr.DataArray(
                np.stack([variance_draws["wt_image_variance"], variance_draws["ko_image_variance"]], axis=-1),
                dims=("chain", "draw", "genotype"),
                coords={"chain": chain_coords, "draw": draw_coords, "genotype": genotype_coords},
            ),
            "mito_variance_response": xr.DataArray(
                np.stack([variance_draws["wt_mito_variance"], variance_draws["ko_mito_variance"]], axis=-1),
                dims=("chain", "draw", "genotype"),
                coords={"chain": chain_coords, "draw": draw_coords, "genotype": genotype_coords},
            ),
            "delta_image_variance_response": xr.DataArray(
                variance_draws["delta_image_variance"],
                dims=("chain", "draw"),
                coords={"chain": chain_coords, "draw": draw_coords},
            ),
            "delta_mito_variance_response": xr.DataArray(
                variance_draws["delta_mito_variance"],
                dims=("chain", "draw"),
                coords={"chain": chain_coords, "draw": draw_coords},
            ),
        }
    else:
        variance_draws = bounded_response_scale_variance_components(data=data, posterior=posterior)

        genotype_mean_response = xr.DataArray(
            np.asarray(posterior["genotype_mean"]),
            dims=("chain", "draw", "genotype"),
            coords={"chain": chain_coords, "draw": draw_coords, "genotype": genotype_coords},
        )
        image_variance_response = xr.DataArray(
            np.stack([variance_draws["wt_image_variance"], variance_draws["ko_image_variance"]], axis=-1),
            dims=("chain", "draw", "genotype"),
            coords={"chain": chain_coords, "draw": draw_coords, "genotype": genotype_coords},
        )
        mito_variance_response = xr.DataArray(
            np.stack([variance_draws["wt_mito_variance"], variance_draws["ko_mito_variance"]], axis=-1),
            dims=("chain", "draw", "genotype"),
            coords={"chain": chain_coords, "draw": draw_coords, "genotype": genotype_coords},
        )
        response_variables = {
            "genotype_mean_response": genotype_mean_response,
            "delta_mean_response": posterior["delta_mean"],
            "animal_variance_shared_response": xr.DataArray(
                variance_draws["animal_variance_shared"],
                dims=("chain", "draw"),
                coords={"chain": chain_coords, "draw": draw_coords},
            ),
            "image_variance_response": image_variance_response,
            "mito_variance_response": mito_variance_response,
            "delta_image_variance_response": xr.DataArray(
                variance_draws["delta_image_variance"],
                dims=("chain", "draw"),
                coords={"chain": chain_coords, "draw": draw_coords},
            ),
            "delta_mito_variance_response": xr.DataArray(
                variance_draws["delta_mito_variance"],
                dims=("chain", "draw"),
                coords={"chain": chain_coords, "draw": draw_coords},
            ),
        }

    idata.posterior = posterior.assign(response_variables)
    idata.attrs["muscle"] = data.muscle
    idata.attrs["metric"] = data.metric
    idata.attrs["family"] = data.family
    idata.attrs["likelihood_name"] = data.likelihood_name
    idata.attrs["fit_stem"] = fit_stem(muscle=data.muscle, metric=data.metric)
    return idata


def save_inference_data(
    data: PreparedMetricData,
    idata: az.InferenceData,
    trace_dir: Path,
) -> Path:
    """Persist one fit's inference data to NetCDF.

    Args:
        data (PreparedMetricData): Prepared metric-specific arrays and labels.
        idata (az.InferenceData): Inference data to be saved.
        trace_dir (Path): Directory where trace files are stored.

    Returns:
        Path: Path to the saved NetCDF file.
    """

    trace_dir.mkdir(parents=True, exist_ok=True)
    output_path = trace_path_for_fit(trace_dir=trace_dir, muscle=data.muscle, metric=data.metric)
    az.to_netcdf(idata, output_path)
    return output_path


def flatten_draws(array: Any) -> np.ndarray:
    """Flatten chain and draw dimensions into a single vector or matrix.

    Args:
        array (Any): Posterior array-like object from ArviZ/xarray.

    Returns:
        np.ndarray: Flattened array with samples on the first axis.
    """

    values = np.asarray(array)
    if values.ndim == 0:
        return values.reshape(1)
    if values.ndim == 1:
        return values
    sample_count = values.shape[0] * values.shape[1]
    return values.reshape(sample_count, *values.shape[2:])


def probability_of_direction(draws: np.ndarray) -> float:
    """Compute the posterior probability of direction in percent.

    Args:
        draws (np.ndarray): Posterior draws for a scalar quantity.

    Returns:
        float: `pd` expressed on the 0-100 scale.
    """

    flat = np.asarray(draws).reshape(-1)
    return 100.0 * max(float(np.mean(flat > 0.0)), float(np.mean(flat < 0.0)))


def summarize_scalar(draws: np.ndarray, label: str) -> dict[str, float | str]:
    """Create numeric and formatted summaries for one posterior scalar.

    Args:
        draws (np.ndarray): Posterior draws for a scalar quantity.
        label (str): Column prefix used in the output dataframe.

    Returns:
        dict[str, float | str]: Summary columns including estimate, HDI, pd, and text.
    """

    flat = np.asarray(draws).reshape(-1)
    estimate = float(np.median(flat))
    hdi_low, hdi_high = np.asarray(az.hdi(flat, hdi_prob=0.95), dtype=float)
    pd_value = probability_of_direction(flat)
    return {
        label: estimate,
        f"{label}_hdi_low": float(hdi_low),
        f"{label}_hdi_high": float(hdi_high),
        f"{label}_pd": pd_value,
        f"{label}_summary": (
            f"delta={estimate:.4g}, 95% HDI [{hdi_low:.4g},{hdi_high:.4g}], pd={pd_value:.1f}%"
        ),
    }


def gamma_variance_summaries(data: PreparedMetricData, idata: az.InferenceData) -> dict[str, np.ndarray]:
    """Extract Gamma variance-component draws from an inference dataset.

    Args:
        data (PreparedMetricData): Prepared metric-specific arrays and labels.
        idata (az.InferenceData): Posterior samples for a positive-metric model.

    Returns:
        dict[str, np.ndarray]: Flattened variance draws for WT, KO, and deltas.
    """

    variance_draws = positive_response_scale_variance_components(data=data, posterior=idata.posterior)
    return {
        "wt_image_variance": flatten_draws(variance_draws["wt_image_variance"]),
        "ko_image_variance": flatten_draws(variance_draws["ko_image_variance"]),
        "delta_image_variance": flatten_draws(variance_draws["delta_image_variance"]),
        "wt_mito_variance": flatten_draws(variance_draws["wt_mito_variance"]),
        "ko_mito_variance": flatten_draws(variance_draws["ko_mito_variance"]),
        "delta_mito_variance": flatten_draws(variance_draws["delta_mito_variance"]),
        "animal_variance_shared": flatten_draws(variance_draws["animal_variance_shared"]),
    }


def beta_variance_summaries(data: PreparedMetricData, idata: az.InferenceData) -> dict[str, np.ndarray]:
    """Compute response-scale bounded variance summaries averaged within genotype.

    Args:
        data (PreparedMetricData): Prepared metric-specific arrays and labels.
        idata (az.InferenceData): Posterior samples for a bounded-metric model.

    Returns:
        dict[str, np.ndarray]: Flattened variance draws for WT, KO, and deltas.
    """

    variance_draws = bounded_response_scale_variance_components(data=data, posterior=idata.posterior)
    return {
        "wt_image_variance": flatten_draws(variance_draws["wt_image_variance"]),
        "ko_image_variance": flatten_draws(variance_draws["ko_image_variance"]),
        "delta_image_variance": flatten_draws(variance_draws["delta_image_variance"]),
        "wt_mito_variance": flatten_draws(variance_draws["wt_mito_variance"]),
        "ko_mito_variance": flatten_draws(variance_draws["ko_mito_variance"]),
        "delta_mito_variance": flatten_draws(variance_draws["delta_mito_variance"]),
        "animal_variance_shared": flatten_draws(variance_draws["animal_variance_shared"]),
    }


def posterior_diagnostics(idata: az.InferenceData, var_names: list[str]) -> dict[str, float]:
    """Compute scalar convergence diagnostics from selected posterior variables.

    Args:
        idata (az.InferenceData): Posterior samples for one fitted model.
        var_names (list[str]): Variable names included in the diagnostic summary.

    Returns:
        dict[str, float]: Maximum R-hat, minimum ESS values, and divergence count.
    """

    available_var_names = [var_name for var_name in var_names if var_name in idata.posterior.data_vars]
    summary = az.summary(idata, var_names=available_var_names, round_to=None)
    sample_stats = idata.sample_stats
    return {
        "rhat_max": float(summary["r_hat"].max()),
        "ess_bulk_min": float(summary["ess_bulk"].min()),
        "ess_tail_min": float(summary["ess_tail"].min()),
        "divergences": int(np.asarray(sample_stats["diverging"]).sum()),
    }


def diagnostic_status(diagnostics: dict[str, float]) -> tuple[str, str]:
    """Classify whether a fitted model passes basic convergence heuristics.

    Args:
        diagnostics (dict[str, float]): Scalar diagnostic summary for one fitted model.

    Returns:
        tuple[str, str]: Fit status label and a semicolon-separated warning message.
    """

    warnings: list[str] = []
    if diagnostics["divergences"] > 0:
        warnings.append("divergences")
    if diagnostics["rhat_max"] > 1.01:
        warnings.append("rhat_gt_1.01")
    if diagnostics["ess_bulk_min"] < REFIT_ESS_THRESHOLD:
        warnings.append("ess_bulk_lt_400")
    if diagnostics["ess_tail_min"] < REFIT_ESS_THRESHOLD:
        warnings.append("ess_tail_lt_400")
    if warnings:
        return ("warn", ";".join(warnings))
    return ("ok", "")


def fit_quality_key(row: dict[str, Any]) -> tuple[int, int, float, float]:
    """Create an ordering key where smaller values correspond to better fits.

    Args:
        row (dict[str, Any]): Flat result row produced by `summarize_model`.

    Returns:
        tuple[int, int, float, float]: Ordered quality key used to compare attempts.
    """

    if row.get("fit_status") == "error":
        return (3, int(1e9), float(1e9), float(1e9))
    status_penalty = 0 if row.get("engine_fit_status") == "ok" else 1
    divergences = int(row.get("divergences", 0))
    predictive_penalty = 0 if row.get("ppc_fit_status") == "ok" else 1
    rhat_penalty = max(float(row.get("rhat_max", 99.0)) - 1.0, 0.0) + float(
        row.get("ppc_max_abs_rel", 99.0)
    )
    ess_penalty = -float(row.get("ess_bulk_min", 0.0))
    return (status_penalty + predictive_penalty, divergences, rhat_penalty, ess_penalty)


def sampling_plan(
    draws: int,
    tune: int,
    target_accept: float,
    attempt_index: int,
    warning_message: str,
) -> tuple[int, int, float]:
    """Return adaptive sampling settings for a retry attempt.

    Args:
        draws (int): Current posterior draws per chain.
        tune (int): Current warmup draws per chain.
        target_accept (float): Current NUTS target acceptance rate.
        attempt_index (int): Zero-based retry attempt index.
        warning_message (str): Warning labels from the previous fit.

    Returns:
        tuple[int, int, float]: Updated `(draws, tune, target_accept)` values.
    """

    if attempt_index == 0:
        return draws, tune, target_accept

    next_draws = max(draws, 1000)
    next_tune = max(tune, 1500)
    next_target_accept = max(target_accept, 0.99)
    if "divergences" in warning_message:
        next_tune = max(next_tune, 2000)
        next_target_accept = max(next_target_accept, 0.995)
    if attempt_index >= 2:
        next_draws = max(next_draws, 1500)
        next_tune = max(next_tune, 2500)
        next_target_accept = max(next_target_accept, 0.995)
    return next_draws, next_tune, next_target_accept


def simulate_predictive_subset(
    data: PreparedMetricData,
    idata: az.InferenceData,
    draw_indices: np.ndarray,
    obs_indices: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate posterior predictive draws for a subset of observations.

    Args:
        data (PreparedMetricData): Prepared metric-specific arrays and metadata.
        idata (az.InferenceData): Posterior samples for one fitted model.
        draw_indices (np.ndarray): Flattened posterior draw indices to sample from.
        obs_indices (np.ndarray): Observation indices to simulate.
        rng (np.random.Generator): Random number generator used for simulation.

    Returns:
        np.ndarray: Predictive draws with shape `(n_draws, n_observations)`.
    """

    image_mean = flatten_draws(idata.posterior["image_mean"])[draw_indices]
    obs_image_idx = data.image_idx_obs[obs_indices]
    obs_genotype_idx = data.genotype_idx_obs[obs_indices]
    parent_mean = image_mean[:, obs_image_idx]

    if data.family == "positive":
        mito_variance_by_image = flatten_draws(idata.posterior["mito_variance_by_image"])[draw_indices]
        obs_variance = mito_variance_by_image[:, obs_image_idx]
        if data.likelihood_name == "lognormal":
            log_mu, log_sigma = lognormal_mu_sigma_from_mean_variance(
                mean=parent_mean,
                variance=obs_variance,
            )
            return data.positive_scale * rng.lognormal(mean=log_mu, sigma=log_sigma)
        gamma_shape = np.square(parent_mean) / np.clip(obs_variance, SMALL_VALUE, None)
        gamma_scale = np.clip(obs_variance, SMALL_VALUE, None) / np.clip(parent_mean, SMALL_VALUE, None)
        return data.positive_scale * rng.gamma(shape=gamma_shape, scale=gamma_scale)

    if data.likelihood_name == "logitnormal":
        image_logit_mean = flatten_draws(idata.posterior["image_logit_mean"])[draw_indices]
        mito_sigma_by_image = flatten_draws(idata.posterior["mito_sigma_by_image"])[draw_indices]
        obs_logit_mean = image_logit_mean[:, obs_image_idx]
        obs_sigma = mito_sigma_by_image[:, obs_image_idx]
        return expit(rng.normal(loc=obs_logit_mean, scale=obs_sigma))
    if data.likelihood_name == "logit_skew_normal":
        image_logit_mean = flatten_draws(idata.posterior["image_logit_mean"])[draw_indices]
        mito_sigma_by_image = flatten_draws(idata.posterior["mito_sigma_by_image"])[draw_indices]
        skew_alpha = flatten_draws(idata.posterior["skew_alpha"])[draw_indices]
        obs_logit_mean = image_logit_mean[:, obs_image_idx]
        obs_sigma = mito_sigma_by_image[:, obs_image_idx]
        return expit(
            skewnorm.rvs(
                a=skew_alpha[:, None],
                loc=obs_logit_mean,
                scale=obs_sigma,
                random_state=rng,
            )
        )

    kappa_mito_by_image = flatten_draws(idata.posterior["kappa_mito_by_image"])[draw_indices]
    obs_kappa = kappa_mito_by_image[:, obs_image_idx]
    alpha = np.clip(parent_mean * obs_kappa, SMALL_VALUE, None)
    beta = np.clip((1.0 - parent_mean) * obs_kappa, SMALL_VALUE, None)
    beta_predictive = rng.beta(alpha, beta)
    if data.likelihood_name != "zero_one_inflated_beta":
        return beta_predictive

    boundary_mass = flatten_draws(idata.posterior["boundary_mass"])[draw_indices][:, obs_genotype_idx]
    one_given_boundary = flatten_draws(idata.posterior["one_given_boundary"])[draw_indices][
        :, obs_genotype_idx
    ]
    uniforms = rng.random(size=beta_predictive.shape)
    zero_cutoff = boundary_mass * (1.0 - one_given_boundary)
    boundary_cutoff = boundary_mass
    return np.where(
        uniforms < zero_cutoff,
        0.0,
        np.where(uniforms < boundary_cutoff, 1.0, beta_predictive),
    )


def histogram_bin_edges_from_samples(
    observed: np.ndarray,
    predictive: np.ndarray,
) -> np.ndarray:
    """Build shared histogram edges for observed and predictive PPC samples.

    Args:
        observed (np.ndarray): Observed response-scale sample values.
        predictive (np.ndarray): Predictive response-scale sample values.

    Returns:
        np.ndarray: Histogram bin edges covering both samples.
    """

    combined = np.concatenate([np.asarray(observed, dtype=float), np.asarray(predictive, dtype=float)])
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        return np.linspace(0.0, 1.0, 20)
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if np.isclose(lower, upper):
        padding = max(abs(lower) * 0.05, 0.05)
        return np.linspace(lower - padding, upper + padding, 20)
    edges = np.histogram_bin_edges(finite, bins="fd")
    if edges.size < 12:
        edges = np.linspace(lower, upper, 20)
    if edges.size > PPC_DENSITY_BIN_LIMIT:
        edges = np.linspace(lower, upper, PPC_DENSITY_BIN_LIMIT)
    return np.asarray(edges, dtype=float)


def integrated_density_error(
    observed: np.ndarray,
    predictive: np.ndarray,
) -> float:
    """Compute integrated absolute histogram-density error between two samples.

    Args:
        observed (np.ndarray): Observed response-scale sample values.
        predictive (np.ndarray): Predictive response-scale sample values.

    Returns:
        float: L1 distance between observed and predictive histogram densities.
    """

    edges = histogram_bin_edges_from_samples(observed=observed, predictive=predictive)
    observed_density, _ = np.histogram(observed, bins=edges, density=True)
    predictive_density, _ = np.histogram(predictive, bins=edges, density=True)
    widths = np.diff(edges)
    return float(np.sum(np.abs(observed_density - predictive_density) * widths))


def compute_ppc_summary(
    data: PreparedMetricData,
    idata: az.InferenceData,
    random_seed: int,
) -> dict[str, float]:
    """Run a lightweight posterior predictive check on a subset of observations.

    Args:
        data (PreparedMetricData): Prepared metric-specific arrays and labels.
        idata (az.InferenceData): Posterior samples for one fitted model.
        random_seed (int): Random seed used for reproducible PPC subsampling.

    Returns:
        dict[str, float]: Simple predictive discrepancies for mean, sd, and quantiles.
    """

    rng = np.random.default_rng(random_seed)
    posterior_size = idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
    draw_count = min(PPC_POSTERIOR_DRAWS, posterior_size)
    obs_count = min(PPC_OBSERVATION_LIMIT, data.y.shape[0])

    draw_indices = rng.choice(posterior_size, size=draw_count, replace=False)
    obs_indices = rng.choice(data.y.shape[0], size=obs_count, replace=False)
    observed_subset = data.observed_y[obs_indices]

    predictive = simulate_predictive_subset(
        data=data,
        idata=idata,
        draw_indices=draw_indices,
        obs_indices=obs_indices,
        rng=rng,
    )
    predictive_flat = predictive.reshape(-1)
    observed_mean = float(np.mean(observed_subset))
    observed_sd = float(np.std(observed_subset, ddof=1)) if observed_subset.size > 1 else 0.0
    observed_q10 = float(np.quantile(observed_subset, 0.10))
    observed_q90 = float(np.quantile(observed_subset, 0.90))
    robust_scale = max(
        observed_sd,
        float(np.quantile(observed_subset, 0.90) - np.quantile(observed_subset, 0.10)),
        float(np.abs(np.median(observed_subset))),
        SMALL_VALUE,
    )
    mean_error = float(np.mean(predictive_flat) - observed_mean)
    sd_error = float(np.std(predictive_flat, ddof=1) - observed_sd)
    q10_error = float(np.quantile(predictive_flat, 0.10) - observed_q10)
    q90_error = float(np.quantile(predictive_flat, 0.90) - observed_q90)
    wasserstein = float(wasserstein_distance(observed_subset, predictive_flat))
    density_l1 = integrated_density_error(observed=observed_subset, predictive=predictive_flat)
    return {
        "ppc_observed_mean": observed_mean,
        "ppc_observed_sd": observed_sd,
        "ppc_mean_error": mean_error,
        "ppc_sd_error": sd_error,
        "ppc_q10_error": q10_error,
        "ppc_q90_error": q90_error,
        "ppc_scale": robust_scale,
        "ppc_mean_abs_rel": float(abs(mean_error) / robust_scale),
        "ppc_sd_abs_rel": float(abs(sd_error) / robust_scale),
        "ppc_q10_abs_rel": float(abs(q10_error) / robust_scale),
        "ppc_q90_abs_rel": float(abs(q90_error) / robust_scale),
        "ppc_max_abs_rel": float(max(abs(mean_error), abs(sd_error), abs(q10_error), abs(q90_error)) / robust_scale),
        "ppc_wasserstein": wasserstein,
        "ppc_wasserstein_rel": float(wasserstein / robust_scale),
        "ppc_density_l1": density_l1,
    }


def ppc_status(ppc_summary: dict[str, float]) -> tuple[str, str]:
    """Classify whether PPC discrepancies are small enough for an `ok` fit label.

    Args:
        ppc_summary (dict[str, float]): Scalar posterior predictive discrepancy summary.

    Returns:
        tuple[str, str]: PPC status label and a semicolon-separated warning message.
    """

    warnings: list[str] = []
    if float(ppc_summary["ppc_mean_abs_rel"]) > PPC_RELATIVE_ERROR_THRESHOLD:
        warnings.append("ppc_mean_mismatch")
    if float(ppc_summary["ppc_sd_abs_rel"]) > PPC_RELATIVE_ERROR_THRESHOLD:
        warnings.append("ppc_sd_mismatch")
    if max(
        float(ppc_summary["ppc_q10_abs_rel"]),
        float(ppc_summary["ppc_q90_abs_rel"]),
    ) > PPC_RELATIVE_ERROR_THRESHOLD:
        warnings.append("ppc_quantile_mismatch")
    if warnings:
        return ("warn", ";".join(warnings))
    return ("ok", "")


def summarize_model(
    data: PreparedMetricData,
    idata: az.InferenceData,
    sampling_seconds: float,
    random_seed: int,
) -> dict[str, Any]:
    """Convert one fitted model into a flat row for the output CSV.

    Args:
        data (PreparedMetricData): Prepared metric-specific arrays and labels.
        idata (az.InferenceData): Posterior samples for one fitted model.
        sampling_seconds (float): Wall-clock runtime for posterior sampling.
        random_seed (int): Random seed used for reproducible PPC summaries.

    Returns:
        dict[str, Any]: One tidy result row with effects, HDIs, pd, and diagnostics.
    """

    posterior = idata.posterior
    genotype_mean = flatten_draws(posterior["genotype_mean"])
    wt_mean = genotype_mean[:, 0]
    ko_mean = genotype_mean[:, 1]

    if data.family == "positive":
        variance_draws = gamma_variance_summaries(data, idata)
        diagnostic_names = diagnostic_var_names(data.family, data.likelihood_name)
    else:
        variance_draws = beta_variance_summaries(data, idata)
        diagnostic_names = diagnostic_var_names(data.family, data.likelihood_name)

    mean_scale = data.positive_scale if data.family == "positive" else 1.0

    row: dict[str, Any] = {
        "muscle": data.muscle,
        "metric": data.metric,
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
        "positive_scale": data.positive_scale,
        "sampling_seconds": sampling_seconds,
        "wt_mean": float(np.median(wt_mean) * mean_scale),
        "ko_mean": float(np.median(ko_mean) * mean_scale),
        "animal_variance_shared": float(np.median(variance_draws["animal_variance_shared"])),
    }
    row.update(summarize_scalar((ko_mean - wt_mean) * mean_scale, "delta_mean"))
    row.update(summarize_scalar(variance_draws["delta_image_variance"], "delta_image_variance"))
    row.update(summarize_scalar(variance_draws["delta_mito_variance"], "delta_mito_variance"))
    row["wt_image_variance"] = float(np.median(variance_draws["wt_image_variance"]))
    row["ko_image_variance"] = float(np.median(variance_draws["ko_image_variance"]))
    row["wt_mito_variance"] = float(np.median(variance_draws["wt_mito_variance"]))
    row["ko_mito_variance"] = float(np.median(variance_draws["ko_mito_variance"]))
    diagnostics = posterior_diagnostics(idata, diagnostic_names)
    row.update(diagnostics)
    ppc_summary = compute_ppc_summary(data, idata, random_seed=random_seed)
    row.update(ppc_summary)
    engine_fit_status, engine_warning_message = diagnostic_status(diagnostics)
    ppc_fit_status, ppc_warning_message = ppc_status(ppc_summary)
    row["engine_fit_status"] = engine_fit_status
    row["engine_warning_message"] = engine_warning_message
    row["ppc_fit_status"] = ppc_fit_status
    row["ppc_warning_message"] = ppc_warning_message
    row["fit_status"] = "ok" if engine_fit_status == "ok" and ppc_fit_status == "ok" else "warn"
    row["warning_message"] = combine_warning_messages([engine_warning_message, ppc_warning_message])
    return row


def analyze_metric(
    df: pd.DataFrame,
    muscle: str,
    metric: str,
    positive_likelihood: str,
    bounded_likelihood: str,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
    random_seed: int,
) -> MetricAnalysisResult:
    """Fit one metric within one muscle and return a summary row.

    Args:
        df (pd.DataFrame): Full cleaned measurements table.
        muscle (str): Muscle name to analyze.
        metric (str): Metric column to fit.
        positive_likelihood (str): Positive-family likelihood name used when applicable.
        bounded_likelihood (str): Bounded-family likelihood strategy used when applicable.
        draws (int): Number of posterior draws per chain.
        tune (int): Number of warmup draws per chain.
        chains (int): Number of MCMC chains.
        cores (int): Number of CPU cores used by PyMC.
        target_accept (float): NUTS target acceptance rate.
        random_seed (int): Random seed for reproducible sampling and PPC.

    Returns:
        MetricAnalysisResult: Winning summary row plus prepared data and posterior samples.
    """

    data = prepare_metric_data(
        df=df,
        muscle=muscle,
        metric=metric,
        positive_likelihood=positive_likelihood,
        bounded_likelihood=bounded_likelihood,
    )
    attempt_results: list[MetricAnalysisResult] = []
    engine_warning_message = ""

    for attempt_index in range(3):
        attempt_draws, attempt_tune, attempt_target_accept = sampling_plan(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            attempt_index=attempt_index,
            warning_message=engine_warning_message,
        )
        try:
            model = build_model(data)
            idata, sampling_seconds = sample_model(
                model=model,
                draws=attempt_draws,
                tune=attempt_tune,
                chains=chains,
                cores=cores,
                target_accept=attempt_target_accept,
                random_seed=random_seed + attempt_index,
            )
            row = summarize_model(
                data=data,
                idata=idata,
                sampling_seconds=sampling_seconds,
                random_seed=random_seed + attempt_index,
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
                "muscle": muscle,
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
            attempt_results.append(
                MetricAnalysisResult(
                    row=error_row,
                    data=data,
                    idata=None,
                )
            )
            engine_warning_message = "error"

    return min(attempt_results, key=lambda result: fit_quality_key(result.row))


def load_measurements(path: Path) -> pd.DataFrame:
    """Load the cleaned measurements CSV used by both workflows.

    Args:
        path (Path): CSV path containing one mitochondrion per row.

    Returns:
        pd.DataFrame: Loaded measurements dataframe.
    """

    return pd.read_csv(path)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the hierarchical analysis workflow.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Fit hierarchical Bayesian models for per-mitochondrion metrics."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH)
    return parser.parse_args()


def fit_likelihood_arguments(fit_config: BayesFitConfig) -> tuple[str, str]:
    """Resolve positive and bounded likelihood arguments for one fit config.

    Args:
        fit_config (BayesFitConfig): Per-fit configuration from the YAML file.

    Returns:
        tuple[str, str]: Positive and bounded likelihood names for `analyze_metric()`.
    """

    if fit_config.metric in POSITIVE_METRICS:
        return fit_config.likelihood, "beta"
    return "gamma", fit_config.likelihood


def merge_result_rows(
    output_path: Path,
    new_rows_df: pd.DataFrame,
    update_mode: str,
) -> pd.DataFrame:
    """Merge refreshed summary rows into an existing summary CSV when requested.

    Args:
        output_path (Path): Summary CSV path configured for the workflow.
        new_rows_df (pd.DataFrame): Freshly computed rows for the current rerun.
        update_mode (str): Summary update mode, either `merge` or `replace`.

    Returns:
        pd.DataFrame: Final summary dataframe that should be written to disk.
    """

    if update_mode == "replace" or not output_path.exists():
        return new_rows_df.sort_values(["muscle", "metric"]).reset_index(drop=True)
    existing_df = pd.read_csv(output_path)
    rerun_pairs = set(zip(new_rows_df["muscle"], new_rows_df["metric"], strict=False))
    preserved_df = existing_df.loc[
        ~existing_df.apply(lambda row: (row["muscle"], row["metric"]) in rerun_pairs, axis=1)
    ].copy()
    merged_df = pd.concat([preserved_df, new_rows_df], ignore_index=True, sort=False)
    return merged_df.sort_values(["muscle", "metric"]).reset_index(drop=True)


def finalize_fit_artifacts(
    result: MetricAnalysisResult,
    trace_dir: Path | None,
    random_seed: int,
) -> MetricAnalysisResult:
    """Attach PPC draws, add response-scale variables, and optionally save traces.

    Args:
        result (MetricAnalysisResult): Winning fit result for one muscle-metric pair.
        trace_dir (Path | None): Optional directory where NetCDF traces are written.
        random_seed (int): Random seed for reproducible posterior predictive simulation.

    Returns:
        MetricAnalysisResult: Updated result row and inference data after finalization.
    """

    row = result.row
    row["trace_path"] = ""
    if result.data is None or result.idata is None:
        return result

    model = build_model(result.data)
    idata = attach_posterior_predictive(model=model, idata=result.idata, random_seed=random_seed)
    idata = attach_response_scale_posterior(data=result.data, idata=idata)

    if trace_dir is not None:
        output_path = save_inference_data(data=result.data, idata=idata, trace_dir=trace_dir)
        row["trace_path"] = str(output_path)

    return MetricAnalysisResult(row=row, data=result.data, idata=idata)


def main() -> None:
    """Run the full hierarchical Bayesian analysis and save a summary CSV.

    Args:
        None

    Returns:
        None
    """

    args = parse_args()
    config = load_hierarchical_bayes_config(
        path=args.config,
        positive_metrics=tuple(POSITIVE_METRICS),
        bounded_metrics=tuple(BOUNDED_METRICS),
    )
    df = load_measurements(config.paths.input_csv)
    validate_config_muscles(config=config, valid_muscles=set(df["Muscle"].dropna().unique().tolist()))
    fit_configs = repeated_fit_configs(config)
    if not fit_configs:
        print(f"No fits have repeat=true in {config.config_path}; nothing to do.")
        return
    rows: list[dict[str, Any]] = []
    for fit_index, fit_config in enumerate(fit_configs):
        seed = config.runtime.seed + fit_index
        positive_likelihood, bounded_likelihood = fit_likelihood_arguments(fit_config=fit_config)
        print(f"Fitting {fit_config.metric} for {fit_config.muscle}...")
        result = analyze_metric(
            df=df,
            muscle=fit_config.muscle,
            metric=fit_config.metric,
            positive_likelihood=positive_likelihood,
            bounded_likelihood=bounded_likelihood,
            draws=fit_config.draws,
            tune=fit_config.tune,
            chains=fit_config.chains,
            cores=config.runtime.cores,
            target_accept=fit_config.target_accept,
            random_seed=seed,
        )
        finalized_result = finalize_fit_artifacts(
            result=result,
            trace_dir=None if not config.runtime.save_idata else config.paths.trace_dir,
            random_seed=seed + 10_000,
        )
        rows.append(finalized_result.row)

    result_df = pd.DataFrame(rows)
    result_df = merge_result_rows(
        output_path=config.paths.summary_csv,
        new_rows_df=result_df,
        update_mode=config.runtime.summary_update_mode,
    )
    config.paths.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(config.paths.summary_csv, index=False)
    print(f"Saved results to {config.paths.summary_csv}")


if __name__ == "__main__":
    main()
