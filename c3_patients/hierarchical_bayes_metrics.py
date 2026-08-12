#!/usr/bin/env python3
"""Fit covariate-adjusted hierarchical Bayesian models for CAPN3 patients."""

from __future__ import annotations

import argparse
import itertools
import os
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from hierarchical_bayes_config import (
    DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH,
    PatientBayesAnalysisConfig,
    PatientBayesFitConfig,
    PatientBayesModelConfig,
    PatientBayesPriorConfig,
    enabled_fit_configs,
    load_hierarchical_bayes_config,
)


SMALL_VALUE = 1e-9
PPC_DRAW_LIMIT = 200
PPC_OBSERVATION_LIMIT = 500
ESS_WARNING_THRESHOLD = 400.0
PPC_RELATIVE_ERROR_THRESHOLD = 0.35
CENTER_IMAGE_REGION = "center"
INSTANCE_CLUSTERING_METRICS = frozenset(
    {"NND", "3NND", "5NND", "Voronoi_Cell_Area"}
)
POSITIVE_LIKELIHOODS = frozenset({"lognormal", "gamma"})
BOUNDED_LIKELIHOODS = frozenset({"beta"})
COUNT_LIKELIHOODS = frozenset({"negative_binomial"})
REAL_LIKELIHOODS = frozenset({"student_t"})
THREAD_LIMIT_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class SiteScenario:
    """One primary or deterministic unknown-site sensitivity scenario."""

    name: str
    exclude_unknown: bool
    assignments: dict[str, str]


@dataclass(frozen=True)
class PreparedPatientMetricData:
    """Arrays and metadata needed to fit one metric under one site scenario."""

    analysis_id: str
    fit_id: str
    metric: str
    family: str
    likelihood_name: str
    exploratory: bool
    scenario_name: str
    scenario_assignments: str
    y: np.ndarray
    observed_y: np.ndarray
    patient_idx_obs: np.ndarray
    image_idx_obs: np.ndarray
    disease_obs: np.ndarray
    site_obs: np.ndarray
    male_obs: np.ndarray
    maturation_obs: np.ndarray
    aging_obs: np.ndarray
    compartment_obs: np.ndarray
    subtype_idx_obs: np.ndarray
    log_area_offset: np.ndarray
    patient_labels: list[str]
    image_labels: list[str]
    subtype_labels: list[str]
    subtype_weights: np.ndarray
    standard_site: np.ndarray
    standard_male: np.ndarray
    standard_maturation: np.ndarray
    standard_aging: np.ndarray
    standard_compartment: np.ndarray
    response_center: float
    response_scale: float
    area_reference_nm2: float
    intercept_location: float
    priors: PatientBayesPriorConfig
    n_invalid_response_dropped: int
    n_patients_control: int
    n_patients_capn3: int
    n_images_control: int
    n_images_capn3: int


@dataclass(frozen=True)
class PosteriorPredictiveSubset:
    """Indexed response-scale posterior-predictive sample."""

    observed: np.ndarray
    predictive: np.ndarray
    observation_indices: np.ndarray
    posterior_draw_indices: np.ndarray


@dataclass
class MetricFitResult:
    """Artifacts returned by one model-fitting attempt."""

    row: dict[str, Any]
    data: PreparedPatientMetricData
    idata: az.InferenceData | None


@dataclass(frozen=True)
class FitTask:
    """One deterministic metric and site-scenario sampling task."""

    operation_index: int
    fit: PatientBayesFitConfig
    data: PreparedPatientMetricData
    random_seed: int


@contextmanager
def limited_thread_environment(thread_count: int) -> Iterator[None]:
    """Temporarily limit numerical-library threads inherited by workers.

    Args:
        thread_count (int): Threads allowed per sampling process.

    Returns:
        Iterator[None]: Context that restores prior environment values.
    """

    previous_values = {
        name: os.environ.get(name) for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES
    }
    thread_text = str(thread_count)
    for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES:
        os.environ[name] = thread_text
    try:
        yield
    finally:
        for name, previous_value in previous_values.items():
            if previous_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous_value


def slugify_value(value: str) -> str:
    """Convert a label into a stable lowercase slug.

    Args:
        value (str): Raw label.

    Returns:
        str: Filesystem-friendly slug.
    """

    slug = "".join(
        character if character.isalnum() else "_" for character in str(value).lower()
    )
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def metric_family(likelihood_name: str) -> str:
    """Return the broad outcome family for a likelihood.

    Args:
        likelihood_name (str): Configured likelihood.

    Returns:
        str: ``positive``, ``bounded``, ``count``, or ``real``.
    """

    if likelihood_name in POSITIVE_LIKELIHOODS:
        return "positive"
    if likelihood_name in BOUNDED_LIKELIHOODS:
        return "bounded"
    if likelihood_name in COUNT_LIKELIHOODS:
        return "count"
    if likelihood_name in REAL_LIKELIHOODS:
        return "real"
    raise ValueError(f"Unsupported likelihood: {likelihood_name}")


def unknown_site_mask(df: pd.DataFrame, model: PatientBayesModelConfig) -> pd.Series:
    """Identify rows whose biopsy site is unavailable.

    Args:
        df (pd.DataFrame): Measurements dataframe.
        model (PatientBayesModelConfig): Column and missing-value settings.

    Returns:
        pd.Series: Boolean row mask.
    """

    site = df[model.site_column]
    normalized = site.astype("string").str.strip().str.upper()
    unknown_tokens = {
        value.strip().upper() for value in model.site_sensitivity.unknown_values
    }
    return site.isna() | normalized.isin(unknown_tokens)


def build_site_scenarios(
    df: pd.DataFrame,
    model: PatientBayesModelConfig,
) -> list[SiteScenario]:
    """Build the primary exclusion and all deterministic site assignments.

    Args:
        df (pd.DataFrame): Loaded measurements.
        model (PatientBayesModelConfig): Model configuration.

    Returns:
        list[SiteScenario]: Primary scenario followed by sensitivity scenarios.
    """

    mask = unknown_site_mask(df=df, model=model)
    unknown_patients = sorted(
        df.loc[mask, model.patient_column].dropna().astype(str).unique().tolist()
    )
    scenarios = [
        SiteScenario(name="primary_exclude_unknown", exclude_unknown=True, assignments={})
    ]
    if not model.site_sensitivity.enabled or not unknown_patients:
        return scenarios
    if len(unknown_patients) > 8:
        raise ValueError(
            "Refusing to enumerate more than 2^8 unknown-site assignments."
        )
    reference = model.reference.reference_site
    comparison = model.reference.comparison_site
    abbreviations = {reference: "D", comparison: "Q"}
    for values in itertools.product((reference, comparison), repeat=len(unknown_patients)):
        assignments = dict(zip(unknown_patients, values, strict=True))
        suffix = "".join(abbreviations[value] for value in values)
        scenarios.append(
            SiteScenario(
                name=f"sensitivity_{suffix}",
                exclude_unknown=False,
                assignments=assignments,
            )
        )
    return scenarios


def apply_site_scenario(
    df: pd.DataFrame,
    model: PatientBayesModelConfig,
    scenario: SiteScenario,
) -> pd.DataFrame:
    """Apply one missing-site scenario to a dataframe.

    Args:
        df (pd.DataFrame): Measurements dataframe.
        model (PatientBayesModelConfig): Model configuration.
        scenario (SiteScenario): Scenario to apply.

    Returns:
        pd.DataFrame: Filtered or site-assigned copy.
    """

    result = df.copy()
    mask = unknown_site_mask(df=result, model=model)
    if scenario.exclude_unknown:
        return result.loc[~mask].copy()
    for patient, site in scenario.assignments.items():
        patient_mask = result[model.patient_column].astype(str) == patient
        result.loc[patient_mask & mask, model.site_column] = site
    remaining = unknown_site_mask(df=result, model=model)
    if remaining.any():
        raise ValueError(
            f"Site scenario '{scenario.name}' left unknown biopsy-site rows."
        )
    return result


def load_measurements(config: PatientBayesAnalysisConfig) -> pd.DataFrame:
    """Load configured input CSV files and apply the zoom filter.

    Args:
        config (PatientBayesAnalysisConfig): Analysis configuration.

    Returns:
        pd.DataFrame: Concatenated, filtered measurements.
    """

    frames: list[pd.DataFrame] = []
    if config.paths.input_csv is not None:
        frames.append(pd.read_csv(config.paths.input_csv, low_memory=False))
    for path in config.paths.input_csvs:
        frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        raise ValueError("No input CSVs were configured.")
    df = pd.concat(frames, ignore_index=True, sort=False)
    if config.filters is not None:
        if "Zoom" not in df.columns:
            raise KeyError("Configured zoom filtering requires the 'Zoom' column.")
        zoom = pd.to_numeric(df["Zoom"], errors="coerce")
        df = df.loc[
            zoom.between(
                config.filters.minimum_zoom,
                config.filters.maximum_zoom,
                inclusive="both",
            )
        ].copy()
        if df.empty:
            raise ValueError("Zoom filtering removed all observations.")
    return df


def validate_required_columns(
    df: pd.DataFrame,
    config: PatientBayesAnalysisConfig,
    fit: PatientBayesFitConfig,
) -> None:
    """Validate columns needed for one fit.

    Args:
        df (pd.DataFrame): Measurements dataframe.
        config (PatientBayesAnalysisConfig): Analysis configuration.
        fit (PatientBayesFitConfig): Metric fit configuration.

    Returns:
        None: Raises ``KeyError`` if columns are missing.
    """

    model = config.model
    required = {
        model.condition_column,
        model.patient_column,
        model.genotype_column,
        model.age_column,
        model.gender_column,
        model.site_column,
        model.compartment_column,
        model.image_column,
        fit.metric,
    }
    if config.analysis_id == "instance" and fit.metric in INSTANCE_CLUSTERING_METRICS:
        required.add(model.image_region_column)
    if metric_family(fit.likelihood) == "count":
        required.add(model.image_area_column)
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing columns for fit '{fit.fit_id}': {missing}")


def validate_patient_table(
    patient_table: pd.DataFrame,
    model: PatientBayesModelConfig,
) -> None:
    """Validate patient-level condition, genotype, sex, site, and age labels.

    Args:
        patient_table (pd.DataFrame): One row per patient.
        model (PatientBayesModelConfig): Model configuration.

    Returns:
        None: Raises ``ValueError`` for unsupported labels.
    """

    reference = model.reference
    allowed_conditions = {
        reference.control_condition,
        reference.disease_condition,
    }
    conditions = set(patient_table[model.condition_column].astype(str))
    if conditions != allowed_conditions:
        raise ValueError(
            f"Expected conditions {sorted(allowed_conditions)}, found {sorted(conditions)}."
        )
    allowed_sites = {reference.reference_site, reference.comparison_site}
    sites = set(patient_table[model.site_column].astype(str))
    if not sites.issubset(allowed_sites):
        raise ValueError(f"Unsupported biopsy sites: {sorted(sites - allowed_sites)}")
    allowed_genders = {reference.reference_gender, reference.comparison_gender}
    genders = set(patient_table[model.gender_column].astype(str))
    if not genders.issubset(allowed_genders):
        raise ValueError(f"Unsupported genders: {sorted(genders - allowed_genders)}")
    control_rows = patient_table[
        patient_table[model.condition_column].astype(str)
        == reference.control_condition
    ]
    control_genotypes = set(control_rows[model.genotype_column].astype(str))
    if control_genotypes != {reference.control_genotype}:
        raise ValueError(
            f"Control genotypes must be '{reference.control_genotype}', "
            f"found {sorted(control_genotypes)}."
        )
    disease_rows = patient_table[
        patient_table[model.condition_column].astype(str)
        == reference.disease_condition
    ]
    disease_genotypes = set(disease_rows[model.genotype_column].astype(str))
    allowed_disease_genotypes = set(reference.disease_genotypes)
    if not disease_genotypes.issubset(allowed_disease_genotypes):
        raise ValueError(
            "Unsupported CAPN3 genotypes: "
            f"{sorted(disease_genotypes - allowed_disease_genotypes)}"
        )
    ages = pd.to_numeric(patient_table[model.age_column], errors="coerce")
    if ages.isna().any() or (ages <= 0.0).any():
        raise ValueError("Every included patient must have a positive numeric age.")


def patient_metadata_table(
    df: pd.DataFrame,
    model: PatientBayesModelConfig,
) -> pd.DataFrame:
    """Create and validate one metadata row per patient.

    Args:
        df (pd.DataFrame): Scenario-specific metric rows.
        model (PatientBayesModelConfig): Model configuration.

    Returns:
        pd.DataFrame: Stable patient table.
    """

    columns = [
        model.patient_column,
        model.condition_column,
        model.genotype_column,
        model.age_column,
        model.gender_column,
        model.site_column,
    ]
    unique = df[columns].drop_duplicates()
    counts = unique.groupby(model.patient_column, dropna=False).size()
    inconsistent = counts[counts != 1]
    if not inconsistent.empty:
        raise ValueError(
            "Patient metadata are inconsistent for: "
            f"{inconsistent.index.astype(str).tolist()}"
        )
    table = unique.sort_values(model.patient_column).reset_index(drop=True)
    validate_patient_table(patient_table=table, model=model)
    return table


def prepare_metric_data(
    df: pd.DataFrame,
    config: PatientBayesAnalysisConfig,
    fit: PatientBayesFitConfig,
    scenario: SiteScenario,
) -> PreparedPatientMetricData:
    """Prepare one joint metric dataset for PyMC.

    Args:
        df (pd.DataFrame): Full loaded measurements.
        config (PatientBayesAnalysisConfig): Analysis configuration.
        fit (PatientBayesFitConfig): Metric fit configuration.
        scenario (SiteScenario): Missing-site scenario.

    Returns:
        PreparedPatientMetricData: Indexed and scaled model arrays.
    """

    validate_required_columns(df=df, config=config, fit=fit)
    model = config.model
    family = metric_family(fit.likelihood)
    metric_df = apply_site_scenario(df=df, model=model, scenario=scenario)
    if config.analysis_id == "instance" and fit.metric in INSTANCE_CLUSTERING_METRICS:
        region = metric_df[model.image_region_column].astype(str).str.lower()
        metric_df = metric_df.loc[region == CENTER_IMAGE_REGION].copy()
    required_nonmissing = [
        model.condition_column,
        model.patient_column,
        model.genotype_column,
        model.age_column,
        model.gender_column,
        model.site_column,
        model.compartment_column,
        model.image_column,
        fit.metric,
    ]
    if family == "count":
        required_nonmissing.append(model.image_area_column)
    metric_df[fit.metric] = pd.to_numeric(metric_df[fit.metric], errors="coerce")
    metric_df = metric_df.dropna(subset=required_nonmissing).copy()
    finite_response = np.isfinite(metric_df[fit.metric].to_numpy(dtype=float))
    metric_df = metric_df.loc[finite_response].copy()
    n_invalid_response_dropped = 0
    if family == "positive":
        positive_response = metric_df[fit.metric] > 0.0
        n_invalid_response_dropped = int((~positive_response).sum())
        metric_df = metric_df.loc[positive_response].copy()
    if metric_df.empty:
        raise ValueError(
            f"No observations remain for {fit.fit_id} / {scenario.name}."
        )
    reference = model.reference
    compartments = set(metric_df[model.compartment_column].astype(str))
    allowed_compartments = {
        reference.reference_compartment,
        reference.comparison_compartment,
    }
    if not compartments.issubset(allowed_compartments):
        raise ValueError(
            f"Unsupported compartments: {sorted(compartments - allowed_compartments)}"
        )
    patient_table = patient_metadata_table(df=metric_df, model=model)
    patient_table["patient_idx"] = np.arange(len(patient_table), dtype=int)
    patient_map = patient_table.set_index(model.patient_column)["patient_idx"].to_dict()
    metric_df["patient_idx"] = (
        metric_df[model.patient_column].map(patient_map).astype(int)
    )
    metric_df["image_key"] = (
        metric_df[model.patient_column].astype(str)
        + "__"
        + metric_df[model.compartment_column].astype(str)
        + "__"
        + metric_df[model.image_column].astype(str)
    )
    image_table = (
        metric_df[
            ["image_key", model.patient_column, model.condition_column]
        ]
        .drop_duplicates()
        .sort_values("image_key")
        .reset_index(drop=True)
    )
    image_table["image_idx"] = np.arange(len(image_table), dtype=int)
    image_map = image_table.set_index("image_key")["image_idx"].to_dict()
    metric_df["image_idx"] = metric_df["image_key"].map(image_map).astype(int)

    raw_y = metric_df[fit.metric].to_numpy(dtype=float)
    response_center = 0.0
    response_scale = 1.0
    area_reference = 1.0
    log_area_offset = np.zeros(len(metric_df), dtype=float)
    if family == "positive":
        if np.any(raw_y <= 0.0):
            raise ValueError(f"{fit.metric} must be strictly positive.")
        response_scale = max(float(np.median(raw_y)), SMALL_VALUE)
        y = raw_y / response_scale
        intercept_location = 0.0
    elif family == "bounded":
        if np.any(raw_y <= 0.0) or np.any(raw_y >= 1.0):
            raise ValueError(
                f"{fit.metric} contains exact/out-of-range boundaries; "
                "the configured Beta model requires values strictly within (0,1)."
            )
        y = raw_y
        median = float(np.median(raw_y))
        intercept_location = float(np.log(median / (1.0 - median)))
    elif family == "count":
        if np.any(raw_y < 0.0) or not np.allclose(raw_y, np.round(raw_y)):
            raise ValueError(f"{fit.metric} must contain non-negative integer counts.")
        area = pd.to_numeric(
            metric_df[model.image_area_column], errors="coerce"
        ).to_numpy(dtype=float)
        if np.any(~np.isfinite(area)) or np.any(area <= 0.0):
            raise ValueError("Count models require positive finite image areas.")
        area_reference = float(np.median(area))
        log_area_offset = np.log(area / area_reference)
        y = np.round(raw_y).astype(int)
        intercept_location = float(np.log(max(float(np.mean(raw_y)), SMALL_VALUE)))
    else:
        response_center = float(np.median(raw_y))
        response_scale = max(float(np.std(raw_y, ddof=1)), SMALL_VALUE)
        y = (raw_y - response_center) / response_scale
        intercept_location = float(np.median(y))

    disease_patient = (
        patient_table[model.condition_column].astype(str)
        == reference.disease_condition
    ).astype(int)
    site_patient = (
        patient_table[model.site_column].astype(str)
        == reference.comparison_site
    ).astype(int)
    male_patient = (
        patient_table[model.gender_column].astype(str)
        == reference.comparison_gender
    ).astype(int)
    ages = pd.to_numeric(
        patient_table[model.age_column], errors="raise"
    ).to_numpy(dtype=float)
    maturation_patient = np.minimum(
        ages, model.age.maturation_age
    ) / model.age.maturation_age
    aging_patient = np.maximum(
        ages - model.age.aging_start_age, 0.0
    ) / model.age.aging_scale_years
    subtype_map = {
        label: index for index, label in enumerate(reference.disease_genotypes)
    }
    subtype_patient = (
        patient_table[model.genotype_column]
        .astype(str)
        .map(subtype_map)
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    disease_patient_array = disease_patient.to_numpy(dtype=int)
    disease_subtypes = subtype_patient[disease_patient_array == 1]
    subtype_counts = np.bincount(
        disease_subtypes, minlength=len(reference.disease_genotypes)
    ).astype(float)
    if np.any(subtype_counts == 0.0):
        missing_subtypes = [
            reference.disease_genotypes[index]
            for index, count in enumerate(subtype_counts)
            if count == 0.0
        ]
        raise ValueError(
            f"Scenario lacks configured CAPN3 subtypes: {missing_subtypes}"
        )
    subtype_weights = subtype_counts / subtype_counts.sum()

    patient_idx_obs = metric_df["patient_idx"].to_numpy(dtype=int)
    disease_obs = disease_patient_array[patient_idx_obs]
    site_obs = site_patient.to_numpy(dtype=int)[patient_idx_obs]
    male_obs = male_patient.to_numpy(dtype=int)[patient_idx_obs]
    maturation_obs = maturation_patient[patient_idx_obs]
    aging_obs = aging_patient[patient_idx_obs]
    subtype_idx_obs = subtype_patient[patient_idx_obs]
    compartment_obs = (
        metric_df[model.compartment_column].astype(str)
        == reference.comparison_compartment
    ).astype(int).to_numpy()

    standard_site = np.repeat(site_patient.to_numpy(dtype=float), 2)
    standard_male = np.repeat(male_patient.to_numpy(dtype=float), 2)
    standard_maturation = np.repeat(maturation_patient, 2)
    standard_aging = np.repeat(aging_patient, 2)
    standard_compartment = np.tile(np.array([0.0, 1.0]), len(patient_table))
    condition_by_image = image_table[model.condition_column].astype(str)
    assignments_text = ";".join(
        f"{patient}={site}" for patient, site in sorted(scenario.assignments.items())
    )
    return PreparedPatientMetricData(
        analysis_id=config.analysis_id,
        fit_id=fit.fit_id,
        metric=fit.metric,
        family=family,
        likelihood_name=fit.likelihood,
        exploratory=fit.exploratory,
        scenario_name=scenario.name,
        scenario_assignments=assignments_text,
        y=np.asarray(y),
        observed_y=raw_y,
        patient_idx_obs=patient_idx_obs,
        image_idx_obs=metric_df["image_idx"].to_numpy(dtype=int),
        disease_obs=disease_obs,
        site_obs=site_obs,
        male_obs=male_obs,
        maturation_obs=maturation_obs,
        aging_obs=aging_obs,
        compartment_obs=compartment_obs,
        subtype_idx_obs=subtype_idx_obs,
        log_area_offset=log_area_offset,
        patient_labels=patient_table[model.patient_column].astype(str).tolist(),
        image_labels=image_table["image_key"].astype(str).tolist(),
        subtype_labels=list(reference.disease_genotypes),
        subtype_weights=subtype_weights,
        standard_site=standard_site,
        standard_male=standard_male,
        standard_maturation=standard_maturation,
        standard_aging=standard_aging,
        standard_compartment=standard_compartment,
        response_center=response_center,
        response_scale=response_scale,
        area_reference_nm2=area_reference,
        intercept_location=intercept_location,
        priors=model.priors,
        n_invalid_response_dropped=n_invalid_response_dropped,
        n_patients_control=int(np.sum(disease_patient_array == 0)),
        n_patients_capn3=int(np.sum(disease_patient_array == 1)),
        n_images_control=int(
            np.sum(condition_by_image == reference.control_condition)
        ),
        n_images_capn3=int(
            np.sum(condition_by_image == reference.disease_condition)
        ),
    )


def build_linear_predictor(
    data: PreparedPatientMetricData,
) -> tuple[pt.TensorVariable, dict[str, Any]]:
    """Construct shared covariate and random-effect terms.

    Args:
        data (PreparedPatientMetricData): Prepared model arrays.

    Returns:
        tuple[pt.TensorVariable, dict[str, Any]]: Observation predictor and parameters.
    """

    priors = data.priors
    patient_idx = pm.Data("patient_idx_obs", data.patient_idx_obs, dims="obs_id")
    disease = pm.Data("disease_obs", data.disease_obs, dims="obs_id")
    site = pm.Data("site_obs", data.site_obs, dims="obs_id")
    male = pm.Data("male_obs", data.male_obs, dims="obs_id")
    maturation = pm.Data(
        "maturation_obs", data.maturation_obs, dims="obs_id"
    )
    aging = pm.Data("aging_obs", data.aging_obs, dims="obs_id")
    compartment = pm.Data(
        "compartment_obs", data.compartment_obs, dims="obs_id"
    )
    subtype_idx = pm.Data(
        "subtype_idx_obs", data.subtype_idx_obs, dims="obs_id"
    )
    alpha = pm.Normal(
        "alpha",
        mu=data.intercept_location,
        sigma=priors.intercept_sigma,
    )
    beta_disease = pm.Normal(
        "beta_disease", mu=0.0, sigma=priors.coefficient_sigma
    )
    beta_site = pm.Normal(
        "beta_site", mu=0.0, sigma=priors.coefficient_sigma
    )
    beta_male = pm.Normal(
        "beta_male", mu=0.0, sigma=priors.coefficient_sigma
    )
    beta_maturation = pm.Normal(
        "beta_maturation", mu=0.0, sigma=priors.coefficient_sigma
    )
    beta_aging = pm.Normal(
        "beta_aging", mu=0.0, sigma=priors.coefficient_sigma
    )
    beta_compartment = pm.Normal(
        "beta_compartment", mu=0.0, sigma=priors.coefficient_sigma
    )
    interaction_site = pm.Normal(
        "interaction_disease_site",
        mu=0.0,
        sigma=priors.interaction_sigma,
    )
    interaction_compartment = pm.Normal(
        "interaction_disease_compartment",
        mu=0.0,
        sigma=priors.interaction_sigma,
    )
    sigma_subtype = pm.HalfNormal(
        "sigma_subtype", sigma=priors.subtype_sigma
    )
    subtype_raw = pm.Normal(
        "subtype_raw",
        mu=0.0,
        sigma=sigma_subtype,
        dims="subtype",
    )
    weights = pt.as_tensor_variable(data.subtype_weights)
    subtype_deviation = pm.Deterministic(
        "subtype_deviation",
        subtype_raw - pt.sum(subtype_raw * weights),
        dims="subtype",
    )
    sigma_patient = pm.HalfNormal(
        "sigma_patient", sigma=priors.random_effect_sigma
    )
    patient_offset = pm.Normal(
        "patient_offset", mu=0.0, sigma=1.0, dims="patient"
    )
    patient_effect = patient_offset * sigma_patient
    eta = (
        alpha
        + beta_disease * disease
        + disease * subtype_deviation[subtype_idx]
        + beta_site * site
        + interaction_site * disease * site
        + beta_male * male
        + beta_maturation * maturation
        + beta_aging * aging
        + beta_compartment * compartment
        + interaction_compartment * disease * compartment
        + patient_effect[patient_idx]
    )
    parameters: dict[str, Any] = {
        "alpha": alpha,
        "beta_disease": beta_disease,
        "beta_site": beta_site,
        "beta_male": beta_male,
        "beta_maturation": beta_maturation,
        "beta_aging": beta_aging,
        "beta_compartment": beta_compartment,
        "interaction_site": interaction_site,
        "interaction_compartment": interaction_compartment,
        "subtype_deviation": subtype_deviation,
    }
    if data.analysis_id == "instance":
        image_idx = pm.Data(
            "image_idx_obs", data.image_idx_obs, dims="obs_id"
        )
        sigma_image = pm.HalfNormal(
            "sigma_image", sigma=priors.random_effect_sigma
        )
        image_offset = pm.Normal(
            "image_offset", mu=0.0, sigma=1.0, dims="image"
        )
        image_effect = image_offset * sigma_image
        eta = eta + image_effect[image_idx]
        parameters["sigma_image"] = sigma_image
    pm.Deterministic(
        "effect_disease_deltoid_imf",
        beta_disease,
    )
    pm.Deterministic(
        "effect_disease_quadriceps_imf",
        beta_disease + interaction_site,
    )
    pm.Deterministic(
        "effect_disease_deltoid_ss",
        beta_disease + interaction_compartment,
    )
    pm.Deterministic(
        "effect_disease_quadriceps_ss",
        beta_disease + interaction_site + interaction_compartment,
    )
    pm.Deterministic(
        "effect_disease_by_subtype",
        beta_disease + subtype_deviation,
        dims="subtype",
    )
    return eta, parameters


def build_model(data: PreparedPatientMetricData) -> pm.Model:
    """Construct the likelihood-specific PyMC model.

    Args:
        data (PreparedPatientMetricData): Prepared metric data.

    Returns:
        pm.Model: Fully specified hierarchical model.
    """

    coords = {
        "obs_id": np.arange(len(data.y)),
        "patient": data.patient_labels,
        "image": data.image_labels,
        "subtype": data.subtype_labels,
    }
    with pm.Model(coords=coords) as model:
        eta, _ = build_linear_predictor(data=data)
        if data.likelihood_name == "lognormal":
            sigma = pm.HalfNormal(
                "sigma_observation", sigma=data.priors.residual_sigma
            )
            pm.LogNormal(
                "observed_metric",
                mu=eta,
                sigma=sigma,
                observed=data.y,
                dims="obs_id",
            )
        elif data.likelihood_name == "gamma":
            sigma = pm.HalfNormal(
                "sigma_observation", sigma=data.priors.residual_sigma
            )
            pm.Gamma(
                "observed_metric",
                mu=pt.exp(eta),
                sigma=sigma,
                observed=data.y,
                dims="obs_id",
            )
        elif data.likelihood_name == "beta":
            log_kappa = pm.Normal("log_kappa", mu=np.log(20.0), sigma=1.0)
            kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
            mean = pm.math.sigmoid(eta)
            pm.Beta(
                "observed_metric",
                alpha=pt.clip(mean * kappa, SMALL_VALUE, np.inf),
                beta=pt.clip((1.0 - mean) * kappa, SMALL_VALUE, np.inf),
                observed=data.y,
                dims="obs_id",
            )
        elif data.likelihood_name == "negative_binomial":
            log_area = pm.Data(
                "log_area_offset", data.log_area_offset, dims="obs_id"
            )
            alpha_nb = pm.Exponential(
                "negative_binomial_alpha",
                lam=data.priors.negative_binomial_alpha_rate,
            )
            pm.NegativeBinomial(
                "observed_metric",
                mu=pt.exp(eta + log_area),
                alpha=alpha_nb,
                observed=data.y,
                dims="obs_id",
            )
        elif data.likelihood_name == "student_t":
            sigma = pm.HalfNormal(
                "sigma_observation", sigma=data.priors.residual_sigma
            )
            nu_minus_two = pm.Exponential("nu_minus_two", lam=0.1)
            pm.StudentT(
                "observed_metric",
                nu=nu_minus_two + 2.0,
                mu=eta,
                sigma=sigma,
                observed=data.y,
                dims="obs_id",
            )
        else:
            raise ValueError(f"Unsupported likelihood: {data.likelihood_name}")
    return model


def flatten_posterior(idata: az.InferenceData, name: str) -> np.ndarray:
    """Flatten chain and draw dimensions for one posterior variable.

    Args:
        idata (az.InferenceData): Posterior inference data.
        name (str): Variable name.

    Returns:
        np.ndarray: Samples in the first dimension.
    """

    values = np.asarray(idata.posterior[name])
    return values.reshape(values.shape[0] * values.shape[1], *values.shape[2:])


def expected_response(
    data: PreparedPatientMetricData,
    eta: np.ndarray,
) -> np.ndarray:
    """Convert a linear predictor to its response-scale expectation.

    Args:
        data (PreparedPatientMetricData): Metric scaling metadata.
        eta (np.ndarray): Link-scale values.

    Returns:
        np.ndarray: Response-scale expected values.
    """

    if data.family == "positive":
        return data.response_scale * np.exp(eta)
    if data.family == "count":
        return np.exp(eta)
    if data.family == "bounded":
        return 1.0 / (1.0 + np.exp(-eta))
    return data.response_center + data.response_scale * eta


def standardized_response_contrast(
    data: PreparedPatientMetricData,
    idata: az.InferenceData,
    site_override: float | None = None,
    compartment_override: float | None = None,
    subtype_index: int | None = None,
) -> np.ndarray:
    """Compute patient-standardized disease contrasts on the response scale.

    Args:
        data (PreparedPatientMetricData): Prepared metric data.
        idata (az.InferenceData): Posterior draws.
        site_override (float | None): Optional fixed site indicator.
        compartment_override (float | None): Optional fixed compartment indicator.
        subtype_index (int | None): Optional CAPN3 subtype index.

    Returns:
        np.ndarray: Draw-wise percent changes or absolute differences.
    """

    alpha = flatten_posterior(idata, "alpha")[:, None]
    beta_disease = flatten_posterior(idata, "beta_disease")[:, None]
    beta_site = flatten_posterior(idata, "beta_site")[:, None]
    beta_male = flatten_posterior(idata, "beta_male")[:, None]
    beta_maturation = flatten_posterior(idata, "beta_maturation")[:, None]
    beta_aging = flatten_posterior(idata, "beta_aging")[:, None]
    beta_compartment = flatten_posterior(idata, "beta_compartment")[:, None]
    interaction_site = flatten_posterior(
        idata, "interaction_disease_site"
    )[:, None]
    interaction_compartment = flatten_posterior(
        idata, "interaction_disease_compartment"
    )[:, None]
    subtype_deviation = flatten_posterior(idata, "subtype_deviation")
    site = (
        np.full_like(data.standard_site, site_override, dtype=float)
        if site_override is not None
        else data.standard_site
    )[None, :]
    compartment = (
        np.full_like(
            data.standard_compartment, compartment_override, dtype=float
        )
        if compartment_override is not None
        else data.standard_compartment
    )[None, :]
    base_eta = (
        alpha
        + beta_site * site
        + beta_male * data.standard_male[None, :]
        + beta_maturation * data.standard_maturation[None, :]
        + beta_aging * data.standard_aging[None, :]
        + beta_compartment * compartment
    )
    control = expected_response(data=data, eta=base_eta)
    common_disease = (
        beta_disease
        + interaction_site * site
        + interaction_compartment * compartment
    )
    if subtype_index is None:
        disease = np.zeros_like(control)
        for index, weight in enumerate(data.subtype_weights):
            disease += weight * expected_response(
                data=data,
                eta=base_eta
                + common_disease
                + subtype_deviation[:, index][:, None],
            )
    else:
        disease = expected_response(
            data=data,
            eta=base_eta
            + common_disease
            + subtype_deviation[:, subtype_index][:, None],
        )
    control_mean = np.mean(control, axis=1)
    disease_mean = np.mean(disease, axis=1)
    if data.family in {"positive", "count"}:
        return 100.0 * (disease_mean / control_mean - 1.0)
    return disease_mean - control_mean


def standardized_subtype_pair_contrast(
    data: PreparedPatientMetricData,
    idata: az.InferenceData,
    numerator_index: int,
    denominator_index: int,
) -> np.ndarray:
    """Compare two CAPN3 subtypes on the standardized response scale.

    Args:
        data (PreparedPatientMetricData): Prepared metric data.
        idata (az.InferenceData): Posterior draws.
        numerator_index (int): Numerator or first subtype index.
        denominator_index (int): Denominator or second subtype index.

    Returns:
        np.ndarray: Draw-wise percent changes or absolute differences.
    """

    numerator_vs_control = standardized_response_contrast(
        data=data,
        idata=idata,
        subtype_index=numerator_index,
    )
    denominator_vs_control = standardized_response_contrast(
        data=data,
        idata=idata,
        subtype_index=denominator_index,
    )
    if data.family in {"positive", "count"}:
        numerator_ratio = 1.0 + numerator_vs_control / 100.0
        denominator_ratio = 1.0 + denominator_vs_control / 100.0
        return 100.0 * (numerator_ratio / denominator_ratio - 1.0)
    return numerator_vs_control - denominator_vs_control


def probability_of_direction(draws: np.ndarray) -> float:
    """Calculate posterior probability of direction in percent.

    Args:
        draws (np.ndarray): Scalar posterior draws.

    Returns:
        float: Probability of the more likely sign.
    """

    flat = np.asarray(draws, dtype=float).reshape(-1)
    return 100.0 * max(float(np.mean(flat > 0.0)), float(np.mean(flat < 0.0)))


def summarize_scalar(draws: np.ndarray, prefix: str) -> dict[str, Any]:
    """Summarize one scalar posterior quantity.

    Args:
        draws (np.ndarray): Posterior draws.
        prefix (str): Output-column prefix.

    Returns:
        dict[str, Any]: Median, HDI, probability, and formatted text.
    """

    flat = np.asarray(draws, dtype=float).reshape(-1)
    median = float(np.median(flat))
    low, high = np.asarray(az.hdi(flat, hdi_prob=0.95), dtype=float)
    pd_value = probability_of_direction(flat)
    return {
        prefix: median,
        f"{prefix}_hdi_low": float(low),
        f"{prefix}_hdi_high": float(high),
        f"{prefix}_pd": pd_value,
        f"{prefix}_summary": (
            f"{median:.4g} [95% HDI {low:.4g}, {high:.4g}], pd={pd_value:.1f}%"
        ),
    }


def posterior_diagnostics(idata: az.InferenceData) -> dict[str, Any]:
    """Compute convergence diagnostics for biological parameters.

    Args:
        idata (az.InferenceData): Posterior inference data.

    Returns:
        dict[str, Any]: R-hat, ESS, divergences, and status.
    """

    names = [
        "beta_disease",
        "beta_site",
        "beta_male",
        "beta_maturation",
        "beta_aging",
        "beta_compartment",
        "interaction_disease_site",
        "interaction_disease_compartment",
        "subtype_deviation",
        "sigma_patient",
    ]
    if "sigma_image" in idata.posterior:
        names.append("sigma_image")
    summary = az.summary(idata, var_names=names, round_to=None)
    rhat_max = float(summary["r_hat"].max())
    ess_bulk_min = float(summary["ess_bulk"].min())
    ess_tail_min = float(summary["ess_tail"].min())
    divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
    warnings: list[str] = []
    if divergences > 0:
        warnings.append("divergences")
    if rhat_max > 1.01:
        warnings.append("rhat_gt_1.01")
    if ess_bulk_min < ESS_WARNING_THRESHOLD:
        warnings.append("ess_bulk_lt_400")
    if ess_tail_min < ESS_WARNING_THRESHOLD:
        warnings.append("ess_tail_lt_400")
    return {
        "rhat_max": rhat_max,
        "ess_bulk_min": ess_bulk_min,
        "ess_tail_min": ess_tail_min,
        "divergences": divergences,
        "engine_fit_status": "warn" if warnings else "ok",
        "engine_warning_message": ";".join(warnings),
    }


def posterior_eta_subset(
    data: PreparedPatientMetricData,
    idata: az.InferenceData,
    draw_indices: np.ndarray,
    obs_indices: np.ndarray,
) -> np.ndarray:
    """Reconstruct link predictors for selected draws and observations.

    Args:
        data (PreparedPatientMetricData): Prepared metric data.
        idata (az.InferenceData): Posterior inference data.
        draw_indices (np.ndarray): Flattened posterior draw indices.
        obs_indices (np.ndarray): Observation indices.

    Returns:
        np.ndarray: Predictor matrix with draw and observation dimensions.
    """

    def scalar(name: str) -> np.ndarray:
        """Extract one selected scalar variable.

        Args:
            name (str): Posterior variable name.

        Returns:
            np.ndarray: Selected draws as one column.
        """

        return flatten_posterior(idata, name)[draw_indices][:, None]

    eta = (
        scalar("alpha")
        + scalar("beta_disease") * data.disease_obs[obs_indices][None, :]
        + scalar("beta_site") * data.site_obs[obs_indices][None, :]
        + scalar("beta_male") * data.male_obs[obs_indices][None, :]
        + scalar("beta_maturation")
        * data.maturation_obs[obs_indices][None, :]
        + scalar("beta_aging") * data.aging_obs[obs_indices][None, :]
        + scalar("beta_compartment")
        * data.compartment_obs[obs_indices][None, :]
        + scalar("interaction_disease_site")
        * data.disease_obs[obs_indices][None, :]
        * data.site_obs[obs_indices][None, :]
        + scalar("interaction_disease_compartment")
        * data.disease_obs[obs_indices][None, :]
        * data.compartment_obs[obs_indices][None, :]
    )
    subtype = flatten_posterior(idata, "subtype_deviation")[draw_indices]
    eta += (
        data.disease_obs[obs_indices][None, :]
        * subtype[:, data.subtype_idx_obs[obs_indices]]
    )
    patient_effect = (
        flatten_posterior(idata, "patient_offset")[draw_indices]
        * flatten_posterior(idata, "sigma_patient")[draw_indices][:, None]
    )
    eta += patient_effect[:, data.patient_idx_obs[obs_indices]]
    if data.analysis_id == "instance":
        image_effect = (
            flatten_posterior(idata, "image_offset")[draw_indices]
            * flatten_posterior(idata, "sigma_image")[draw_indices][:, None]
        )
        eta += image_effect[:, data.image_idx_obs[obs_indices]]
    return eta


def simulate_predictive_subset_indexed(
    data: PreparedPatientMetricData,
    idata: az.InferenceData,
    random_seed: int,
    draw_limit: int = PPC_DRAW_LIMIT,
    observation_limit: int = PPC_OBSERVATION_LIMIT,
) -> PosteriorPredictiveSubset:
    """Simulate an indexed posterior-predictive subset on the response scale.

    Args:
        data (PreparedPatientMetricData): Prepared metric data.
        idata (az.InferenceData): Posterior inference data.
        random_seed (int): Reproducible random seed.
        draw_limit (int): Maximum flattened posterior draws to simulate.
        observation_limit (int): Maximum observations to subsample.

    Returns:
        PosteriorPredictiveSubset: Observed values, predictions, and source indices.
    """

    rng = np.random.default_rng(random_seed)
    posterior_count = (
        idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
    )
    if draw_limit < 1 or observation_limit < 1:
        raise ValueError("PPC draw and observation limits must be positive.")
    draw_count = min(draw_limit, posterior_count)
    observation_count = min(observation_limit, len(data.y))
    draw_indices = rng.choice(posterior_count, draw_count, replace=False)
    obs_indices = rng.choice(len(data.y), observation_count, replace=False)
    eta = posterior_eta_subset(
        data=data,
        idata=idata,
        draw_indices=draw_indices,
        obs_indices=obs_indices,
    )
    if data.likelihood_name == "lognormal":
        sigma = flatten_posterior(
            idata, "sigma_observation"
        )[draw_indices][:, None]
        model_predictive = rng.lognormal(mean=eta, sigma=sigma)
    elif data.likelihood_name == "gamma":
        sigma = flatten_posterior(
            idata, "sigma_observation"
        )[draw_indices][:, None]
        mean = np.exp(eta)
        shape = np.square(mean / np.clip(sigma, SMALL_VALUE, None))
        scale = np.square(sigma) / np.clip(mean, SMALL_VALUE, None)
        model_predictive = rng.gamma(shape=shape, scale=scale)
    elif data.likelihood_name == "beta":
        kappa = flatten_posterior(idata, "kappa")[draw_indices][:, None]
        mean = 1.0 / (1.0 + np.exp(-eta))
        model_predictive = rng.beta(
            np.clip(mean * kappa, SMALL_VALUE, None),
            np.clip((1.0 - mean) * kappa, SMALL_VALUE, None),
        )
    elif data.likelihood_name == "negative_binomial":
        alpha_nb = flatten_posterior(
            idata, "negative_binomial_alpha"
        )[draw_indices][:, None]
        mean = np.exp(
            eta + data.log_area_offset[obs_indices][None, :]
        )
        probability = alpha_nb / (alpha_nb + mean)
        model_predictive = rng.negative_binomial(alpha_nb, probability)
    else:
        sigma = flatten_posterior(
            idata, "sigma_observation"
        )[draw_indices][:, None]
        nu = (
            flatten_posterior(idata, "nu_minus_two")[draw_indices][:, None]
            + 2.0
        )
        model_predictive = eta + sigma * rng.standard_t(df=nu, size=eta.shape)
    if data.family == "positive":
        predictive = model_predictive * data.response_scale
    elif data.family == "real":
        predictive = (
            data.response_center + data.response_scale * model_predictive
        )
    else:
        predictive = model_predictive
    return PosteriorPredictiveSubset(
        observed=data.observed_y[obs_indices],
        predictive=predictive,
        observation_indices=obs_indices,
        posterior_draw_indices=draw_indices,
    )


def simulate_predictive_subset(
    data: PreparedPatientMetricData,
    idata: az.InferenceData,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a bounded posterior-predictive subset without storing huge arrays.

    Args:
        data (PreparedPatientMetricData): Prepared metric data.
        idata (az.InferenceData): Posterior inference data.
        random_seed (int): Reproducible random seed.

    Returns:
        tuple[np.ndarray, np.ndarray]: Raw observed subset and predictive draws.
    """

    sample = simulate_predictive_subset_indexed(
        data=data,
        idata=idata,
        random_seed=random_seed,
    )
    return sample.observed, sample.predictive


def posterior_predictive_summary(
    data: PreparedPatientMetricData,
    idata: az.InferenceData,
    random_seed: int,
) -> dict[str, Any]:
    """Compute lightweight posterior-predictive discrepancy summaries.

    Args:
        data (PreparedPatientMetricData): Prepared metric data.
        idata (az.InferenceData): Posterior inference data.
        random_seed (int): Reproducible random seed.

    Returns:
        dict[str, Any]: Predictive errors and status.
    """

    observed, predictive = simulate_predictive_subset(
        data=data, idata=idata, random_seed=random_seed
    )
    predictive_flat = predictive.reshape(-1)
    observed_sd = float(np.std(observed, ddof=1)) if observed.size > 1 else 0.0
    robust_scale = max(
        observed_sd,
        float(np.quantile(observed, 0.9) - np.quantile(observed, 0.1)),
        abs(float(np.median(observed))),
        SMALL_VALUE,
    )
    errors = {
        "mean": float(np.mean(predictive_flat) - np.mean(observed)),
        "sd": float(np.std(predictive_flat, ddof=1) - observed_sd),
        "q10": float(
            np.quantile(predictive_flat, 0.1) - np.quantile(observed, 0.1)
        ),
        "q90": float(
            np.quantile(predictive_flat, 0.9) - np.quantile(observed, 0.9)
        ),
    }
    relative = {key: abs(value) / robust_scale for key, value in errors.items()}
    warning_labels = [
        f"ppc_{key}_mismatch"
        for key, value in relative.items()
        if value > PPC_RELATIVE_ERROR_THRESHOLD
    ]
    return {
        "ppc_observed_mean": float(np.mean(observed)),
        "ppc_observed_sd": observed_sd,
        "ppc_mean_error": errors["mean"],
        "ppc_sd_error": errors["sd"],
        "ppc_q10_error": errors["q10"],
        "ppc_q90_error": errors["q90"],
        "ppc_max_abs_rel": float(max(relative.values())),
        "ppc_fit_status": "warn" if warning_labels else "ok",
        "ppc_warning_message": ";".join(warning_labels),
    }


def summarize_model(
    data: PreparedPatientMetricData,
    idata: az.InferenceData,
    sampling_seconds: float,
    random_seed: int,
) -> dict[str, Any]:
    """Create one flat summary row for a fitted patient model.

    Args:
        data (PreparedPatientMetricData): Prepared metric data.
        idata (az.InferenceData): Posterior inference data.
        sampling_seconds (float): NUTS wall-clock time.
        random_seed (int): Seed used for PPC subsampling.

    Returns:
        dict[str, Any]: Effect summaries, counts, and diagnostics.
    """

    contrast_kind = (
        "percent_change" if data.family in {"positive", "count"} else "difference"
    )
    row: dict[str, Any] = {
        "analysis_id": data.analysis_id,
        "fit_id": data.fit_id,
        "metric": data.metric,
        "family": data.family,
        "likelihood_name": data.likelihood_name,
        "exploratory": data.exploratory,
        "site_scenario": data.scenario_name,
        "site_assignments": data.scenario_assignments,
        "contrast_kind": contrast_kind,
        "n_observations": len(data.y),
        "n_invalid_response_dropped": data.n_invalid_response_dropped,
        "n_patients_control": data.n_patients_control,
        "n_patients_capn3": data.n_patients_capn3,
        "n_images_control": data.n_images_control,
        "n_images_capn3": data.n_images_capn3,
        "response_center": data.response_center,
        "response_scale": data.response_scale,
        "area_reference_nm2": data.area_reference_nm2,
        "sampling_seconds": sampling_seconds,
    }
    row.update(
        summarize_scalar(
            flatten_posterior(idata, "beta_disease"),
            "capn3_effect_link",
        )
    )
    row.update(
        summarize_scalar(
            standardized_response_contrast(data=data, idata=idata),
            "capn3_effect_response",
        )
    )
    row.update(
        summarize_scalar(
            standardized_response_contrast(
                data=data, idata=idata, site_override=0.0
            ),
            "capn3_effect_deltoid_response",
        )
    )
    row.update(
        summarize_scalar(
            standardized_response_contrast(
                data=data, idata=idata, site_override=1.0
            ),
            "capn3_effect_quadriceps_response",
        )
    )
    row.update(
        summarize_scalar(
            standardized_response_contrast(
                data=data, idata=idata, compartment_override=0.0
            ),
            "capn3_effect_imf_response",
        )
    )
    row.update(
        summarize_scalar(
            standardized_response_contrast(
                data=data, idata=idata, compartment_override=1.0
            ),
            "capn3_effect_ss_response",
        )
    )
    for variable, prefix, site_value, compartment_value in (
        (
            "effect_disease_deltoid_imf",
            "capn3_effect_deltoid_imf",
            0.0,
            0.0,
        ),
        (
            "effect_disease_quadriceps_imf",
            "capn3_effect_quadriceps_imf",
            1.0,
            0.0,
        ),
        (
            "effect_disease_deltoid_ss",
            "capn3_effect_deltoid_ss",
            0.0,
            1.0,
        ),
        (
            "effect_disease_quadriceps_ss",
            "capn3_effect_quadriceps_ss",
            1.0,
            1.0,
        ),
    ):
        row.update(
            summarize_scalar(
                flatten_posterior(idata, variable),
                f"{prefix}_link",
            )
        )
        row.update(
            summarize_scalar(
                standardized_response_contrast(
                    data=data,
                    idata=idata,
                    site_override=site_value,
                    compartment_override=compartment_value,
                ),
                f"{prefix}_response",
            )
        )
    subtype_deviation = flatten_posterior(idata, "subtype_deviation")
    subtype_effect = flatten_posterior(idata, "effect_disease_by_subtype")
    for index, label in enumerate(data.subtype_labels):
        slug = slugify_value(label)
        row.update(
            summarize_scalar(
                subtype_deviation[:, index],
                f"subtype_{slug}_deviation_link",
            )
        )
        row.update(
            summarize_scalar(
                subtype_effect[:, index],
                f"subtype_{slug}_effect_link",
            )
        )
        row.update(
            summarize_scalar(
                standardized_response_contrast(
                    data=data, idata=idata, subtype_index=index
                ),
                f"subtype_{slug}_effect_response",
            )
        )
    for numerator_index, denominator_index in itertools.combinations(
        range(len(data.subtype_labels)), 2
    ):
        numerator_slug = slugify_value(data.subtype_labels[numerator_index])
        denominator_slug = slugify_value(data.subtype_labels[denominator_index])
        prefix = f"subtype_{numerator_slug}_vs_{denominator_slug}_contrast"
        row.update(
            summarize_scalar(
                subtype_effect[:, numerator_index]
                - subtype_effect[:, denominator_index],
                f"{prefix}_link",
            )
        )
        row.update(
            summarize_scalar(
                standardized_subtype_pair_contrast(
                    data=data,
                    idata=idata,
                    numerator_index=numerator_index,
                    denominator_index=denominator_index,
                ),
                f"{prefix}_response",
            )
        )
    for variable, prefix in (
        ("beta_site", "site_effect_link"),
        ("beta_male", "male_effect_link"),
        ("beta_maturation", "maturation_effect_link"),
        ("beta_aging", "aging_effect_link"),
        ("beta_compartment", "compartment_effect_link"),
        ("interaction_disease_site", "disease_site_interaction_link"),
        (
            "interaction_disease_compartment",
            "disease_compartment_interaction_link",
        ),
    ):
        row.update(summarize_scalar(flatten_posterior(idata, variable), prefix))
    diagnostics = posterior_diagnostics(idata=idata)
    ppc = posterior_predictive_summary(
        data=data, idata=idata, random_seed=random_seed
    )
    row.update(diagnostics)
    row.update(ppc)
    warning_parts = [
        diagnostics["engine_warning_message"],
        ppc["ppc_warning_message"],
    ]
    row["fit_status"] = (
        "ok"
        if diagnostics["engine_fit_status"] == "ok"
        and ppc["ppc_fit_status"] == "ok"
        else "warn"
    )
    row["warning_message"] = ";".join(part for part in warning_parts if part)
    return row


def sampling_plan(
    fit: PatientBayesFitConfig,
    runtime: Any,
    attempt_index: int,
    warning_message: str,
) -> tuple[int, int, float]:
    """Calculate retry-adjusted NUTS settings.

    Args:
        fit (PatientBayesFitConfig): Base fit settings.
        runtime (Any): Runtime retry limits.
        attempt_index (int): Zero-based attempt number.
        warning_message (str): Previous diagnostic warning.

    Returns:
        tuple[int, int, float]: Draws, tune, and target acceptance.
    """

    if attempt_index == 0:
        return fit.draws, fit.tune, fit.target_accept
    mixing = any(
        label in warning_message
        for label in ("rhat_gt_1.01", "ess_bulk_lt_400", "ess_tail_lt_400")
    )
    divergences = "divergences" in warning_message
    draw_multiplier = 1.5 if mixing else 1.0
    tune_multiplier = 1.5 if divergences else 1.25
    if attempt_index >= 2:
        draw_multiplier = max(draw_multiplier, 2.0)
        tune_multiplier = max(tune_multiplier, 2.0)
    draws = min(int(np.ceil(fit.draws * draw_multiplier)), runtime.retry_max_draws)
    tune = min(int(np.ceil(fit.tune * tune_multiplier)), runtime.retry_max_tune)
    target_accept = min(
        max(fit.target_accept, 0.995 if divergences else 0.99),
        runtime.retry_max_target_accept,
    )
    return draws, tune, target_accept


def fit_metric(
    data: PreparedPatientMetricData,
    fit: PatientBayesFitConfig,
    config: PatientBayesAnalysisConfig,
    random_seed: int,
    progressbar: bool = True,
    multiprocessing_context: str | None = None,
) -> MetricFitResult:
    """Sample one metric with optional diagnostic retries.

    Args:
        data (PreparedPatientMetricData): Prepared metric data.
        fit (PatientBayesFitConfig): Sampling settings.
        config (PatientBayesAnalysisConfig): Runtime settings.
        random_seed (int): Base seed.
        progressbar (bool): Whether PyMC should render a progress bar.
        multiprocessing_context (str | None): PyMC chain-process start method.

    Returns:
        MetricFitResult: Best available fitted result.
    """

    attempts = 3 if config.runtime.retry_on_warnings else 1
    results: list[MetricFitResult] = []
    warning_message = ""
    for attempt_index in range(attempts):
        draws, tune, target_accept = sampling_plan(
            fit=fit,
            runtime=config.runtime,
            attempt_index=attempt_index,
            warning_message=warning_message,
        )
        try:
            model = build_model(data=data)
            chain_process_context = (
                get_context(multiprocessing_context)
                if multiprocessing_context is not None
                else None
            )
            start = time.perf_counter()
            with model:
                idata = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=fit.chains,
                    cores=min(config.runtime.cores, fit.chains),
                    target_accept=target_accept,
                    random_seed=random_seed + attempt_index,
                    return_inferencedata=True,
                    progressbar=progressbar,
                    mp_ctx=chain_process_context,
                )
            sampling_seconds = time.perf_counter() - start
            row = summarize_model(
                data=data,
                idata=idata,
                sampling_seconds=sampling_seconds,
                random_seed=random_seed + 10_000 + attempt_index,
            )
            row.update(
                {
                    "attempt": attempt_index + 1,
                    "sampling_draws": draws,
                    "sampling_tune": tune,
                    "sampling_target_accept": target_accept,
                }
            )
            result = MetricFitResult(row=row, data=data, idata=idata)
            results.append(result)
            warning_message = str(row.get("engine_warning_message", ""))
            if row["engine_fit_status"] == "ok":
                return result
        except Exception as exc:
            row = error_result_row(data=data, error=exc)
            row.update(
                {
                    "attempt": attempt_index + 1,
                    "sampling_draws": draws,
                    "sampling_tune": tune,
                    "sampling_target_accept": target_accept,
                }
            )
            results.append(MetricFitResult(row=row, data=data, idata=None))
            warning_message = "error"
    return min(results, key=fit_quality_key)


def error_result_row(
    data: PreparedPatientMetricData,
    error: Exception,
) -> dict[str, Any]:
    """Build a summary row for a failed fit.

    Args:
        data (PreparedPatientMetricData): Prepared metric data.
        error (Exception): Raised fitting exception.

    Returns:
        dict[str, Any]: Error summary row.
    """

    return {
        "analysis_id": data.analysis_id,
        "fit_id": data.fit_id,
        "metric": data.metric,
        "family": data.family,
        "likelihood_name": data.likelihood_name,
        "exploratory": data.exploratory,
        "site_scenario": data.scenario_name,
        "site_assignments": data.scenario_assignments,
        "n_observations": len(data.y),
        "n_invalid_response_dropped": data.n_invalid_response_dropped,
        "n_patients_control": data.n_patients_control,
        "n_patients_capn3": data.n_patients_capn3,
        "n_images_control": data.n_images_control,
        "n_images_capn3": data.n_images_capn3,
        "fit_status": "error",
        "engine_fit_status": "error",
        "ppc_fit_status": "error",
        "warning_message": "error",
        "error_message": str(error),
    }


def fit_quality_key(result: MetricFitResult) -> tuple[int, int, float, float]:
    """Rank attempts so lower keys represent better fits.

    Args:
        result (MetricFitResult): Candidate fitting result.

    Returns:
        tuple[int, int, float, float]: Status, divergences, R-hat, and ESS key.
    """

    row = result.row
    if row.get("fit_status") == "error":
        return (2, int(1e9), float(1e9), float(1e9))
    status = 0 if row.get("engine_fit_status") == "ok" else 1
    return (
        status,
        int(row.get("divergences", 0)),
        float(row.get("rhat_max", 99.0)),
        -float(row.get("ess_bulk_min", 0.0)),
    )


def trace_path(
    config: PatientBayesAnalysisConfig,
    data: PreparedPatientMetricData,
) -> Path:
    """Return the NetCDF path for one metric and scenario.

    Args:
        config (PatientBayesAnalysisConfig): Analysis configuration.
        data (PreparedPatientMetricData): Prepared metric metadata.

    Returns:
        Path: Trace output path.
    """

    stem = "__".join(
        [
            slugify_value(data.analysis_id),
            slugify_value(data.fit_id),
            slugify_value(data.scenario_name),
        ]
    )
    return config.paths.trace_dir / f"{stem}.nc"


def save_trace(
    config: PatientBayesAnalysisConfig,
    result: MetricFitResult,
) -> str:
    """Save a successful inference dataset when configured.

    Args:
        config (PatientBayesAnalysisConfig): Analysis configuration.
        result (MetricFitResult): Fitted result.

    Returns:
        str: Saved path or empty text.
    """

    if not config.runtime.save_idata or result.idata is None:
        return ""
    output_path = trace_path(config=config, data=result.data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.idata.attrs.update(
        {
            "analysis_id": result.data.analysis_id,
            "fit_id": result.data.fit_id,
            "metric": result.data.metric,
            "family": result.data.family,
            "likelihood_name": result.data.likelihood_name,
            "site_scenario": result.data.scenario_name,
            "site_assignments": result.data.scenario_assignments,
            "response_center": result.data.response_center,
            "response_scale": result.data.response_scale,
            "area_reference_nm2": result.data.area_reference_nm2,
        }
    )
    az.to_netcdf(result.idata, output_path)
    return str(output_path)


def run_fit_task(
    task: FitTask,
    config: PatientBayesAnalysisConfig,
    progressbar: bool,
    multiprocessing_context: str | None,
) -> dict[str, Any]:
    """Fit one task, save its unique trace, and return only its summary row.

    Args:
        task (FitTask): Prepared deterministic sampling task.
        config (PatientBayesAnalysisConfig): Analysis and runtime configuration.
        progressbar (bool): Whether PyMC should render chain progress.
        multiprocessing_context (str | None): PyMC chain-process start method.

    Returns:
        dict[str, Any]: Small serializable summary row.
    """

    try:
        result = fit_metric(
            data=task.data,
            fit=task.fit,
            config=config,
            random_seed=task.random_seed,
            progressbar=progressbar,
            multiprocessing_context=multiprocessing_context,
        )
        result.row["trace_path"] = save_trace(config=config, result=result)
        row = result.row
    except Exception as exc:
        row = error_result_row(data=task.data, error=exc)
        row["trace_path"] = ""
    row["operation_index"] = task.operation_index
    row["sampling_cores_per_fit"] = min(config.runtime.cores, task.fit.chains)
    return row


def merge_result_row(
    output_path: Path,
    row: dict[str, Any],
    update_mode: str,
) -> pd.DataFrame:
    """Merge one result row into the analysis summary.

    Args:
        output_path (Path): Summary CSV path.
        row (dict[str, Any]): New result row.
        update_mode (str): ``merge`` or ``replace``.

    Returns:
        pd.DataFrame: Updated summary table.
    """

    new_df = pd.DataFrame([row])
    keys = ["analysis_id", "fit_id", "site_scenario"]
    if update_mode == "replace" or not output_path.exists():
        return new_df.sort_values(keys).reset_index(drop=True)
    existing = pd.read_csv(output_path, low_memory=False)
    key = tuple(row[column] for column in keys)
    if all(column in existing.columns for column in keys):
        keep = ~existing.apply(
            lambda existing_row: tuple(existing_row[column] for column in keys)
            == key,
            axis=1,
        )
        existing = existing.loc[keep].copy()
    merged = pd.concat([existing, new_df], ignore_index=True, sort=False)
    return merged.sort_values(keys).reset_index(drop=True)


def persist_result_row(
    config: PatientBayesAnalysisConfig,
    row: dict[str, Any],
    update_mode: str,
) -> None:
    """Serially merge one completed worker row into the summary CSV.

    Args:
        config (PatientBayesAnalysisConfig): Analysis configuration.
        row (dict[str, Any]): Completed fit summary.
        update_mode (str): Summary merge mode for this write.

    Returns:
        None: Writes the configured summary CSV.
    """

    output_path = config.paths.summary_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged = merge_result_row(
        output_path=output_path,
        row=row,
        update_mode=update_mode,
    )
    merged.to_csv(output_path, index=False)


def validate_prepared_data(data: PreparedPatientMetricData) -> None:
    """Run inexpensive internal consistency checks.

    Args:
        data (PreparedPatientMetricData): Prepared metric data.

    Returns:
        None: Raises ``ValueError`` for inconsistent arrays.
    """

    observation_count = len(data.y)
    arrays = [
        data.observed_y,
        data.patient_idx_obs,
        data.image_idx_obs,
        data.disease_obs,
        data.site_obs,
        data.male_obs,
        data.maturation_obs,
        data.aging_obs,
        data.compartment_obs,
        data.subtype_idx_obs,
        data.log_area_offset,
    ]
    if any(len(array) != observation_count for array in arrays):
        raise ValueError("Prepared observation arrays have inconsistent lengths.")
    if data.n_patients_control < 1 or data.n_patients_capn3 < 1:
        raise ValueError("Both CTRL and CAPN3 patients are required.")
    if not np.isclose(np.sum(data.subtype_weights), 1.0):
        raise ValueError("CAPN3 subtype weights must sum to one.")


def prior_predictive_smoke(
    data: PreparedPatientMetricData,
    draws: int,
    random_seed: int,
) -> None:
    """Build a model and sample a small prior predictive smoke test.

    Args:
        data (PreparedPatientMetricData): Prepared metric data.
        draws (int): Number of prior draws.
        random_seed (int): Reproducible seed.

    Returns:
        None: Raises if model construction or simulation fails.
    """

    model = build_model(data=data)
    with model:
        pm.sample_prior_predictive(
            samples=draws,
            var_names=["observed_metric", "beta_disease"],
            random_seed=random_seed,
            return_inferencedata=False,
        )


def select_scenarios(
    scenarios: list[SiteScenario],
    requested_name: str | None,
) -> list[SiteScenario]:
    """Filter site scenarios by an optional CLI name.

    Args:
        scenarios (list[SiteScenario]): Available scenarios.
        requested_name (str | None): Requested scenario.

    Returns:
        list[SiteScenario]: Selected scenarios.
    """

    if requested_name is None:
        return scenarios
    selected = [
        scenario for scenario in scenarios if scenario.name == requested_name
    ]
    if not selected:
        raise ValueError(
            f"Unknown site scenario '{requested_name}'. Available: "
            f"{[scenario.name for scenario in scenarios]}"
        )
    return selected


def select_fits(
    config: PatientBayesAnalysisConfig,
    requested_ids: list[str] | None,
) -> list[PatientBayesFitConfig]:
    """Select enabled fits, optionally restricted by CLI IDs.

    Args:
        config (PatientBayesAnalysisConfig): Analysis configuration.
        requested_ids (list[str] | None): Optional fit IDs.

    Returns:
        list[PatientBayesFitConfig]: Selected enabled fits.
    """

    fits = enabled_fit_configs(config)
    if not requested_ids:
        return fits
    unknown = sorted(set(requested_ids) - set(config.fits_by_id))
    if unknown:
        raise ValueError(f"Unknown fit IDs: {unknown}")
    disabled = [
        fit_id
        for fit_id in requested_ids
        if not config.fits_by_id[fit_id].enabled
    ]
    if disabled:
        raise ValueError(f"Requested fits are disabled in the YAML: {disabled}")
    requested = set(requested_ids)
    return [fit for fit in fits if fit.fit_id in requested]


def execute_fit_tasks(
    tasks: list[FitTask],
    config: PatientBayesAnalysisConfig,
    concurrent_fits: int,
    update_mode: str,
) -> None:
    """Sample prepared tasks serially or with an outer process pool.

    Args:
        tasks (list[FitTask]): Deterministically ordered prepared tasks.
        config (PatientBayesAnalysisConfig): Analysis configuration.
        concurrent_fits (int): Maximum simultaneous fit/scenario jobs.
        update_mode (str): Parent-only summary merge mode.

    Returns:
        None: Saves traces in workers and summary rows in the parent.
    """

    worker_count = min(concurrent_fits, len(tasks))
    if worker_count <= 1:
        for completed_count, task in enumerate(tasks, start=1):
            row = run_fit_task(
                task=task,
                config=config,
                progressbar=True,
                multiprocessing_context=None,
            )
            persist_result_row(
                config=config,
                row=row,
                update_mode=update_mode,
            )
            print(
                f"Completed {completed_count}/{len(tasks)} "
                f"{task.fit.fit_id} / {task.data.scenario_name}: "
                f"status={row.get('fit_status')}."
            )
        return

    print(
        f"Sampling {len(tasks)} fit/scenario jobs with {worker_count} "
        f"concurrent fits x {config.runtime.cores} cores per fit "
        f"(up to {worker_count * config.runtime.cores} cores)."
    )
    spawn_context = get_context("spawn")
    with limited_thread_environment(thread_count=1):
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=spawn_context,
        ) as executor:
            future_to_task = {
                executor.submit(
                    run_fit_task,
                    task,
                    config,
                    False,
                    "spawn",
                ): task
                for task in tasks
            }
            completed_count = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = error_result_row(data=task.data, error=exc)
                    row["trace_path"] = ""
                    row["operation_index"] = task.operation_index
                    row["sampling_cores_per_fit"] = min(
                        config.runtime.cores,
                        task.fit.chains,
                    )
                persist_result_row(
                    config=config,
                    row=row,
                    update_mode=update_mode,
                )
                completed_count += 1
                print(
                    f"Completed {completed_count}/{len(tasks)} "
                    f"{task.fit.fit_id} / {task.data.scenario_name}: "
                    f"status={row.get('fit_status')}."
                )


def run_analysis(
    config: PatientBayesAnalysisConfig,
    requested_fit_ids: list[str] | None,
    requested_scenario: str | None,
    validate_only: bool,
    prior_predictive_draws: int,
    concurrent_fits_override: int | None,
) -> None:
    """Validate or fit all selected metrics and site scenarios.

    Args:
        config (PatientBayesAnalysisConfig): Analysis configuration.
        requested_fit_ids (list[str] | None): Optional targeted fits.
        requested_scenario (str | None): Optional targeted scenario.
        validate_only (bool): Whether to stop after data preparation.
        prior_predictive_draws (int): Optional smoke-test prior draws.
        concurrent_fits_override (int | None): Optional outer-worker override.

    Returns:
        None: Prints validation status and writes fit artifacts.
    """

    if not config.enabled:
        print(f"Analysis {config.analysis_id} is disabled; nothing to do.")
        return
    df = load_measurements(config=config)
    fits = select_fits(config=config, requested_ids=requested_fit_ids)
    if not fits:
        print(f"No {config.analysis_id} fits have enabled=true; nothing to do.")
        return
    scenarios = select_scenarios(
        scenarios=build_site_scenarios(df=df, model=config.model),
        requested_name=requested_scenario,
    )
    tasks: list[FitTask] = []
    operation_index = 0
    for fit in fits:
        for scenario in scenarios:
            data = prepare_metric_data(
                df=df,
                config=config,
                fit=fit,
                scenario=scenario,
            )
            validate_prepared_data(data=data)
            print(
                f"Prepared {config.analysis_id} {fit.fit_id} / "
                f"{scenario.name}: observations={len(data.y)}, "
                f"patients={len(data.patient_labels)}, images={len(data.image_labels)}"
            )
            if prior_predictive_draws > 0:
                prior_predictive_smoke(
                    data=data,
                    draws=prior_predictive_draws,
                    random_seed=config.runtime.seed + operation_index,
                )
                print(
                    f"Prior predictive smoke passed for {fit.fit_id} / "
                    f"{scenario.name}."
                )
            if validate_only or prior_predictive_draws > 0:
                operation_index += 1
                continue
            tasks.append(
                FitTask(
                    operation_index=operation_index,
                    fit=fit,
                    data=data,
                    random_seed=config.runtime.seed + operation_index,
                )
            )
            operation_index += 1
    if validate_only or prior_predictive_draws > 0:
        return
    result_update_mode = config.runtime.summary_update_mode
    if result_update_mode == "replace":
        config.paths.summary_csv.unlink(missing_ok=True)
        result_update_mode = "merge"
    concurrent_fits = (
        concurrent_fits_override
        if concurrent_fits_override is not None
        else config.runtime.concurrent_fits
    )
    execute_fit_tasks(
        tasks=tasks,
        config=config,
        concurrent_fits=concurrent_fits,
        update_mode=result_update_mode,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    Args:
        None: Reads ``sys.argv``.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(
        description="Fit CAPN3 patient hierarchical Bayesian models."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH,
    )
    parser.add_argument(
        "--fit-id",
        action="append",
        dest="fit_ids",
        help="Run one enabled fit ID; repeat the option for multiple fits.",
    )
    parser.add_argument(
        "--scenario",
        help="Run one site scenario, for example primary_exclude_unknown.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config and prepare data without constructing or sampling models.",
    )
    parser.add_argument(
        "--prior-predictive-draws",
        type=int,
        default=0,
        help="Run model-build/prior-predictive smoke checks instead of NUTS.",
    )
    parser.add_argument(
        "--concurrent-fits",
        type=int,
        help=(
            "Override runtime.concurrent_fits; use 1 for serial sampling and "
            "progress bars."
        ),
    )
    args = parser.parse_args()
    if args.prior_predictive_draws < 0:
        parser.error("--prior-predictive-draws must be >= 0.")
    if args.concurrent_fits is not None and args.concurrent_fits < 1:
        parser.error("--concurrent-fits must be >= 1.")
    return args


def main() -> None:
    """Run the configured patient Bayesian analysis.

    Args:
        None: Reads command-line arguments.

    Returns:
        None: Validates data or writes summaries and traces.
    """

    args = parse_args()
    config = load_hierarchical_bayes_config(args.config)
    run_analysis(
        config=config,
        requested_fit_ids=args.fit_ids,
        requested_scenario=args.scenario,
        validate_only=args.validate_only,
        prior_predictive_draws=args.prior_predictive_draws,
        concurrent_fits_override=args.concurrent_fits,
    )


if __name__ == "__main__":
    main()
