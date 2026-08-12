"""Strict configuration loading for CAPN3 patient Bayesian analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH = (
    Path(__file__).resolve().parent / "hierarchical_bayes_config.yaml"
)
ALLOWED_ANALYSIS_IDS = ("instance", "image_summary")
ALLOWED_SUMMARY_UPDATE_MODES = ("merge", "replace")
ALLOWED_LIKELIHOODS = (
    "lognormal",
    "gamma",
    "beta",
    "negative_binomial",
    "student_t",
)
ALLOWED_SITE_PRIMARY_STRATEGIES = ("exclude",)

INSTANCE_METRIC_LIKELIHOODS: dict[str, tuple[str, ...]] = {
    "Area": ("lognormal", "gamma"),
    "Corrected_area": ("lognormal", "gamma"),
    "Major_axis_length": ("lognormal", "gamma"),
    "Minor_axis_length": ("lognormal", "gamma"),
    "Minimum_Feret_Diameter": ("lognormal", "gamma"),
    "Eccentricity": ("beta",),
    "Circularity": ("beta",),
    "Solidity": ("beta",),
    "NND": ("lognormal", "gamma"),
    "3NND": ("lognormal", "gamma"),
    "5NND": ("lognormal", "gamma"),
    "Voronoi_Cell_Area": ("lognormal", "gamma"),
}
IMAGE_SUMMARY_METRIC_LIKELIHOODS: dict[str, tuple[str, ...]] = {
    "Density": ("negative_binomial",),
    "Instance_count": ("negative_binomial",),
    "Area_sum": ("lognormal", "gamma"),
    "Corrected_area_sum": ("lognormal", "gamma"),
    "Minimum_Feret_Diameter_sum": ("lognormal", "gamma"),
    "Minor_axis_length_sum": ("lognormal", "gamma"),
    "Area_mean": ("lognormal", "gamma"),
    "Corrected_area_mean": ("lognormal", "gamma"),
    "Minimum_Feret_Diameter_mean": ("lognormal", "gamma"),
    "Major_axis_length_mean": ("lognormal", "gamma"),
    "Minor_axis_length_mean": ("lognormal", "gamma"),
    "Eccentricity_mean": ("beta",),
    "Circularity_mean": ("beta",),
    "Solidity_mean": ("beta",),
    "NND_center_mean": ("lognormal", "gamma"),
    "3NND_center_mean": ("lognormal", "gamma"),
    "5NND_center_mean": ("lognormal", "gamma"),
    "Voronoi_Cell_Area_center_mean": ("lognormal", "gamma"),
    "Voronoi_Cell_Area_center_cv": ("lognormal", "gamma"),
    "Ripley_L_integral": ("student_t",),
    "Pair_Correlation_integral": ("student_t",),
}


@dataclass(frozen=True)
class PatientBayesPathConfig:
    """Filesystem paths used by one CAPN3 patient analysis."""

    input_csv: Path | None
    input_csvs: tuple[Path, ...]
    summary_csv: Path
    trace_dir: Path
    figure_root: Path


@dataclass(frozen=True)
class PatientBayesRuntimeConfig:
    """Runtime and persistence settings for one analysis."""

    cores: int
    concurrent_fits: int
    seed: int
    save_idata: bool
    summary_update_mode: str
    retry_on_warnings: bool
    retry_max_draws: int
    retry_max_tune: int
    retry_max_target_accept: float


@dataclass(frozen=True)
class PatientBayesFilterConfig:
    """Optional image-magnification filter."""

    minimum_zoom: float
    maximum_zoom: float


@dataclass(frozen=True)
class PatientBayesReferenceConfig:
    """Reference labels and allowed patient metadata values."""

    control_condition: str
    disease_condition: str
    reference_site: str
    comparison_site: str
    reference_gender: str
    comparison_gender: str
    reference_compartment: str
    comparison_compartment: str
    control_genotype: str
    disease_genotypes: tuple[str, ...]


@dataclass(frozen=True)
class PatientBayesAgeConfig:
    """Biologically constrained age-basis settings."""

    maturation_age: float
    aging_start_age: float
    aging_scale_years: float


@dataclass(frozen=True)
class PatientBayesPriorConfig:
    """Weakly informative priors on the common link scale."""

    intercept_sigma: float
    coefficient_sigma: float
    interaction_sigma: float
    subtype_sigma: float
    random_effect_sigma: float
    residual_sigma: float
    negative_binomial_alpha_rate: float


@dataclass(frozen=True)
class PatientBayesSiteSensitivityConfig:
    """Handling of unknown biopsy-site values."""

    enabled: bool
    primary_strategy: str
    unknown_values: tuple[str, ...]


@dataclass(frozen=True)
class PatientBayesModelConfig:
    """Column names and statistical-model settings."""

    condition_column: str
    patient_column: str
    genotype_column: str
    age_column: str
    gender_column: str
    site_column: str
    compartment_column: str
    image_column: str
    image_region_column: str
    image_area_column: str
    reference: PatientBayesReferenceConfig
    age: PatientBayesAgeConfig
    priors: PatientBayesPriorConfig
    site_sensitivity: PatientBayesSiteSensitivityConfig


@dataclass(frozen=True)
class PatientBayesFitConfig:
    """Per-metric likelihood and sampling configuration."""

    fit_id: str
    metric: str
    enabled: bool
    likelihood: str
    draws: int
    tune: int
    chains: int
    target_accept: float
    exploratory: bool


@dataclass(frozen=True)
class PatientBayesAnalysisConfig:
    """Normalized configuration for one patient analysis level."""

    config_path: Path
    enabled: bool
    analysis_id: str
    paths: PatientBayesPathConfig
    runtime: PatientBayesRuntimeConfig
    filters: PatientBayesFilterConfig | None
    model: PatientBayesModelConfig
    fit_order: tuple[str, ...]
    fits_by_id: dict[str, PatientBayesFitConfig]


def require_mapping(section_name: str, value: Any) -> dict[str, Any]:
    """Require and return a plain mapping.

    Args:
        section_name (str): Human-readable section name.
        value (Any): Value loaded from YAML.

    Returns:
        dict[str, Any]: Validated mapping.
    """

    if not isinstance(value, dict):
        raise ValueError(f"Config section '{section_name}' must be a mapping.")
    return dict(value)


def validate_section_keys(
    section_name: str,
    section: dict[str, Any],
    required_keys: set[str],
    optional_keys: set[str] | None = None,
) -> None:
    """Require exactly the documented keys in a mapping.

    Args:
        section_name (str): Human-readable section name.
        section (dict[str, Any]): Mapping to validate.
        required_keys (set[str]): Keys that must be present.
        optional_keys (set[str] | None): Additional accepted keys.

    Returns:
        None: Raises ``ValueError`` for missing or unexpected keys.
    """

    allowed = set(required_keys)
    if optional_keys:
        allowed.update(optional_keys)
    missing = sorted(required_keys - set(section))
    unexpected = sorted(set(section) - allowed)
    if missing:
        raise ValueError(f"Config section '{section_name}' is missing keys: {missing}")
    if unexpected:
        raise ValueError(
            f"Config section '{section_name}' has unexpected keys: {unexpected}"
        )


def require_string(section_name: str, key_name: str, value: Any) -> str:
    """Require a non-empty string.

    Args:
        section_name (str): Human-readable section name.
        key_name (str): Configuration key.
        value (Any): Raw value.

    Returns:
        str: Stripped string.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Config value '{section_name}.{key_name}' must be a non-empty string."
        )
    return value.strip()


def require_bool(section_name: str, key_name: str, value: Any) -> bool:
    """Require a boolean.

    Args:
        section_name (str): Human-readable section name.
        key_name (str): Configuration key.
        value (Any): Raw value.

    Returns:
        bool: Validated boolean.
    """

    if not isinstance(value, bool):
        raise ValueError(
            f"Config value '{section_name}.{key_name}' must be a boolean."
        )
    return value


def require_int(
    section_name: str,
    key_name: str,
    value: Any,
    minimum: int = 1,
) -> int:
    """Require an integer at or above a lower bound.

    Args:
        section_name (str): Human-readable section name.
        key_name (str): Configuration key.
        value (Any): Raw value.
        minimum (int): Inclusive lower bound.

    Returns:
        int: Validated integer.
    """

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"Config value '{section_name}.{key_name}' must be an integer >= {minimum}."
        )
    return value


def require_float(
    section_name: str,
    key_name: str,
    value: Any,
    minimum: float,
    maximum: float = float("inf"),
) -> float:
    """Require a numeric value within inclusive bounds.

    Args:
        section_name (str): Human-readable section name.
        key_name (str): Configuration key.
        value (Any): Raw value.
        minimum (float): Inclusive lower bound.
        maximum (float): Inclusive upper bound.

    Returns:
        float: Validated floating-point value.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Config value '{section_name}.{key_name}' must be numeric."
        )
    numeric = float(value)
    if numeric < minimum or numeric > maximum:
        raise ValueError(
            f"Config value '{section_name}.{key_name}' must be between "
            f"{minimum} and {maximum}."
        )
    return numeric


def require_string_list(section_name: str, key_name: str, value: Any) -> tuple[str, ...]:
    """Require a non-empty list of unique strings.

    Args:
        section_name (str): Human-readable section name.
        key_name (str): Configuration key.
        value (Any): Raw value.

    Returns:
        tuple[str, ...]: Validated strings.
    """

    if not isinstance(value, list) or not value:
        raise ValueError(
            f"Config value '{section_name}.{key_name}' must be a non-empty list."
        )
    values = tuple(require_string(section_name, key_name, item) for item in value)
    if len(values) != len(set(values)):
        raise ValueError(
            f"Config value '{section_name}.{key_name}' contains duplicates."
        )
    return values


def require_allowed_string(
    section_name: str,
    key_name: str,
    value: Any,
    allowed_values: tuple[str, ...],
) -> str:
    """Require a string selected from an allowed set.

    Args:
        section_name (str): Human-readable section name.
        key_name (str): Configuration key.
        value (Any): Raw value.
        allowed_values (tuple[str, ...]): Accepted strings.

    Returns:
        str: Validated string.
    """

    string_value = require_string(section_name, key_name, value)
    if string_value not in allowed_values:
        raise ValueError(
            f"Config value '{section_name}.{key_name}' must be one of "
            f"{list(allowed_values)}."
        )
    return string_value


def parse_paths(section: dict[str, Any]) -> PatientBayesPathConfig:
    """Parse analysis input and output paths.

    Args:
        section (dict[str, Any]): Raw paths mapping.

    Returns:
        PatientBayesPathConfig: Parsed paths.
    """

    validate_section_keys(
        "paths",
        section,
        {"summary_csv", "trace_dir", "figure_root"},
        {"input_csv", "input_csvs"},
    )
    input_csv = (
        Path(require_string("paths", "input_csv", section["input_csv"]))
        if "input_csv" in section
        else None
    )
    input_csvs = (
        tuple(
            Path(value)
            for value in require_string_list(
                "paths", "input_csvs", section["input_csvs"]
            )
        )
        if "input_csvs" in section
        else ()
    )
    if input_csv is None and not input_csvs:
        raise ValueError("Config paths require 'input_csv' or 'input_csvs'.")
    if input_csv is not None and input_csvs:
        raise ValueError("Configure only one of 'input_csv' and 'input_csvs'.")
    return PatientBayesPathConfig(
        input_csv=input_csv,
        input_csvs=input_csvs,
        summary_csv=Path(
            require_string("paths", "summary_csv", section["summary_csv"])
        ),
        trace_dir=Path(require_string("paths", "trace_dir", section["trace_dir"])),
        figure_root=Path(
            require_string("paths", "figure_root", section["figure_root"])
        ),
    )


def parse_runtime(section: dict[str, Any]) -> PatientBayesRuntimeConfig:
    """Parse runtime and persistence settings.

    Args:
        section (dict[str, Any]): Raw runtime mapping.

    Returns:
        PatientBayesRuntimeConfig: Parsed runtime settings.
    """

    validate_section_keys(
        "runtime",
        section,
        {
            "cores",
            "seed",
            "save_idata",
            "summary_update_mode",
            "retry_on_warnings",
            "retry_max_draws",
            "retry_max_tune",
            "retry_max_target_accept",
        },
        {"concurrent_fits"},
    )
    return PatientBayesRuntimeConfig(
        cores=require_int("runtime", "cores", section["cores"]),
        concurrent_fits=require_int(
            "runtime",
            "concurrent_fits",
            section.get("concurrent_fits", 1),
        ),
        seed=require_int("runtime", "seed", section["seed"], minimum=0),
        save_idata=require_bool(
            "runtime", "save_idata", section["save_idata"]
        ),
        summary_update_mode=require_allowed_string(
            "runtime",
            "summary_update_mode",
            section["summary_update_mode"],
            ALLOWED_SUMMARY_UPDATE_MODES,
        ),
        retry_on_warnings=require_bool(
            "runtime", "retry_on_warnings", section["retry_on_warnings"]
        ),
        retry_max_draws=require_int(
            "runtime", "retry_max_draws", section["retry_max_draws"]
        ),
        retry_max_tune=require_int(
            "runtime", "retry_max_tune", section["retry_max_tune"]
        ),
        retry_max_target_accept=require_float(
            "runtime",
            "retry_max_target_accept",
            section["retry_max_target_accept"],
            0.8,
            0.999,
        ),
    )


def parse_filters(section: dict[str, Any]) -> PatientBayesFilterConfig:
    """Parse an optional zoom filter.

    Args:
        section (dict[str, Any]): Raw filter mapping.

    Returns:
        PatientBayesFilterConfig: Parsed zoom limits.
    """

    validate_section_keys(
        "filters", section, {"minimum_zoom", "maximum_zoom"}
    )
    minimum_zoom = require_float(
        "filters", "minimum_zoom", section["minimum_zoom"], 0.0
    )
    maximum_zoom = require_float(
        "filters", "maximum_zoom", section["maximum_zoom"], 0.0
    )
    if maximum_zoom < minimum_zoom:
        raise ValueError("filters.maximum_zoom must be >= filters.minimum_zoom.")
    return PatientBayesFilterConfig(minimum_zoom, maximum_zoom)


def parse_reference(section: dict[str, Any]) -> PatientBayesReferenceConfig:
    """Parse reference levels and allowed disease genotypes.

    Args:
        section (dict[str, Any]): Raw reference mapping.

    Returns:
        PatientBayesReferenceConfig: Parsed reference settings.
    """

    validate_section_keys(
        "model.reference",
        section,
        {
            "control_condition",
            "disease_condition",
            "reference_site",
            "comparison_site",
            "reference_gender",
            "comparison_gender",
            "reference_compartment",
            "comparison_compartment",
            "control_genotype",
            "disease_genotypes",
        },
    )
    return PatientBayesReferenceConfig(
        control_condition=require_string(
            "model.reference", "control_condition", section["control_condition"]
        ),
        disease_condition=require_string(
            "model.reference", "disease_condition", section["disease_condition"]
        ),
        reference_site=require_string(
            "model.reference", "reference_site", section["reference_site"]
        ),
        comparison_site=require_string(
            "model.reference", "comparison_site", section["comparison_site"]
        ),
        reference_gender=require_string(
            "model.reference", "reference_gender", section["reference_gender"]
        ),
        comparison_gender=require_string(
            "model.reference", "comparison_gender", section["comparison_gender"]
        ),
        reference_compartment=require_string(
            "model.reference",
            "reference_compartment",
            section["reference_compartment"],
        ),
        comparison_compartment=require_string(
            "model.reference",
            "comparison_compartment",
            section["comparison_compartment"],
        ),
        control_genotype=require_string(
            "model.reference", "control_genotype", section["control_genotype"]
        ),
        disease_genotypes=require_string_list(
            "model.reference", "disease_genotypes", section["disease_genotypes"]
        ),
    )


def parse_age(section: dict[str, Any]) -> PatientBayesAgeConfig:
    """Parse the biological age-basis settings.

    Args:
        section (dict[str, Any]): Raw age mapping.

    Returns:
        PatientBayesAgeConfig: Parsed age settings.
    """

    validate_section_keys(
        "model.age",
        section,
        {"maturation_age", "aging_start_age", "aging_scale_years"},
    )
    maturation_age = require_float(
        "model.age", "maturation_age", section["maturation_age"], 1.0
    )
    aging_start_age = require_float(
        "model.age", "aging_start_age", section["aging_start_age"], 1.0
    )
    if aging_start_age <= maturation_age:
        raise ValueError("model.age.aging_start_age must exceed maturation_age.")
    return PatientBayesAgeConfig(
        maturation_age=maturation_age,
        aging_start_age=aging_start_age,
        aging_scale_years=require_float(
            "model.age", "aging_scale_years", section["aging_scale_years"], 1.0
        ),
    )


def parse_priors(section: dict[str, Any]) -> PatientBayesPriorConfig:
    """Parse prior scales.

    Args:
        section (dict[str, Any]): Raw prior mapping.

    Returns:
        PatientBayesPriorConfig: Parsed prior scales.
    """

    keys = {
        "intercept_sigma",
        "coefficient_sigma",
        "interaction_sigma",
        "subtype_sigma",
        "random_effect_sigma",
        "residual_sigma",
        "negative_binomial_alpha_rate",
    }
    validate_section_keys("model.priors", section, keys)
    values = {
        key: require_float("model.priors", key, section[key], 1e-9)
        for key in keys
    }
    return PatientBayesPriorConfig(**values)


def parse_site_sensitivity(
    section: dict[str, Any],
) -> PatientBayesSiteSensitivityConfig:
    """Parse unknown-site sensitivity settings.

    Args:
        section (dict[str, Any]): Raw site-sensitivity mapping.

    Returns:
        PatientBayesSiteSensitivityConfig: Parsed sensitivity settings.
    """

    validate_section_keys(
        "model.site_sensitivity",
        section,
        {"enabled", "primary_strategy", "unknown_values"},
    )
    return PatientBayesSiteSensitivityConfig(
        enabled=require_bool(
            "model.site_sensitivity", "enabled", section["enabled"]
        ),
        primary_strategy=require_allowed_string(
            "model.site_sensitivity",
            "primary_strategy",
            section["primary_strategy"],
            ALLOWED_SITE_PRIMARY_STRATEGIES,
        ),
        unknown_values=require_string_list(
            "model.site_sensitivity", "unknown_values", section["unknown_values"]
        ),
    )


def parse_model(section: dict[str, Any]) -> PatientBayesModelConfig:
    """Parse column names and statistical-model settings.

    Args:
        section (dict[str, Any]): Raw model mapping.

    Returns:
        PatientBayesModelConfig: Parsed model settings.
    """

    column_keys = {
        "condition_column",
        "patient_column",
        "genotype_column",
        "age_column",
        "gender_column",
        "site_column",
        "compartment_column",
        "image_column",
        "image_region_column",
        "image_area_column",
    }
    validate_section_keys(
        "model",
        section,
        column_keys | {"reference", "age", "priors", "site_sensitivity"},
    )
    columns = {
        key: require_string("model", key, section[key]) for key in column_keys
    }
    return PatientBayesModelConfig(
        **columns,
        reference=parse_reference(require_mapping("model.reference", section["reference"])),
        age=parse_age(require_mapping("model.age", section["age"])),
        priors=parse_priors(require_mapping("model.priors", section["priors"])),
        site_sensitivity=parse_site_sensitivity(
            require_mapping("model.site_sensitivity", section["site_sensitivity"])
        ),
    )


def validate_available_options(section: dict[str, Any]) -> None:
    """Validate the documentation-only available-options block.

    Args:
        section (dict[str, Any]): Raw available-options mapping.

    Returns:
        None: Raises ``ValueError`` if options differ from parser support.
    """

    validate_section_keys(
        "available_options",
        section,
        {"likelihoods", "summary_update_modes", "enabled_options"},
    )
    if tuple(section["likelihoods"]) != ALLOWED_LIKELIHOODS:
        raise ValueError(
            f"available_options.likelihoods must be {list(ALLOWED_LIKELIHOODS)}."
        )
    if tuple(section["summary_update_modes"]) != ALLOWED_SUMMARY_UPDATE_MODES:
        raise ValueError(
            "available_options.summary_update_modes must be "
            f"{list(ALLOWED_SUMMARY_UPDATE_MODES)}."
        )
    if tuple(section["enabled_options"]) != (True, False):
        raise ValueError(
            "available_options.enabled_options must be [true, false]."
        )


def parse_fit(
    fit_id: str,
    section: dict[str, Any],
    analysis_id: str,
) -> PatientBayesFitConfig:
    """Parse and validate one metric fit.

    Args:
        fit_id (str): YAML mapping key.
        section (dict[str, Any]): Raw fit mapping.
        analysis_id (str): Analysis level.

    Returns:
        PatientBayesFitConfig: Parsed fit.
    """

    section_name = f"fits.{fit_id}"
    validate_section_keys(
        section_name,
        section,
        {
            "metric",
            "enabled",
            "likelihood",
            "draws",
            "tune",
            "chains",
            "target_accept",
        },
        {"exploratory"},
    )
    metric = require_string(section_name, "metric", section["metric"])
    metric_registry = (
        INSTANCE_METRIC_LIKELIHOODS
        if analysis_id == "instance"
        else IMAGE_SUMMARY_METRIC_LIKELIHOODS
    )
    if metric not in metric_registry:
        raise ValueError(
            f"Unsupported {analysis_id} metric in '{section_name}': {metric}"
        )
    likelihood = require_string(
        section_name, "likelihood", section["likelihood"]
    )
    if likelihood not in metric_registry[metric]:
        raise ValueError(
            f"Likelihood '{likelihood}' is invalid for metric '{metric}'. "
            f"Allowed: {list(metric_registry[metric])}."
        )
    return PatientBayesFitConfig(
        fit_id=fit_id,
        metric=metric,
        enabled=require_bool(section_name, "enabled", section["enabled"]),
        likelihood=likelihood,
        draws=require_int(section_name, "draws", section["draws"]),
        tune=require_int(section_name, "tune", section["tune"]),
        chains=require_int(section_name, "chains", section["chains"]),
        target_accept=require_float(
            section_name, "target_accept", section["target_accept"], 0.8, 0.999
        ),
        exploratory=require_bool(
            section_name,
            "exploratory",
            section.get("exploratory", False),
        ),
    )


def validate_parallel_sampling(
    runtime: PatientBayesRuntimeConfig,
    fits: dict[str, PatientBayesFitConfig],
) -> None:
    """Require enough per-fit cores to sample all enabled chains concurrently.

    Args:
        runtime (PatientBayesRuntimeConfig): Runtime parallelism settings.
        fits (dict[str, PatientBayesFitConfig]): Parsed fit configurations.

    Returns:
        None: Raises ``ValueError`` for enabled fits exceeding the core limit.
    """

    invalid_parallel_fits = [
        fit.fit_id
        for fit in fits.values()
        if fit.enabled and fit.chains > runtime.cores
    ]
    if invalid_parallel_fits:
        raise ValueError(
            "Enabled fits cannot use more chains than runtime.cores. "
            f"Invalid fit IDs: {invalid_parallel_fits}"
        )


def load_hierarchical_bayes_config(path: Path) -> PatientBayesAnalysisConfig:
    """Load and strictly validate one patient Bayesian YAML file.

    Args:
        path (Path): YAML path.

    Returns:
        PatientBayesAnalysisConfig: Normalized configuration.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    root = require_mapping("root", raw)
    validate_section_keys(
        "root",
        root,
        {
            "enabled",
            "analysis_id",
            "paths",
            "runtime",
            "model",
            "available_options",
            "fits",
        },
        {"filters"},
    )
    analysis_id = require_allowed_string(
        "root", "analysis_id", root["analysis_id"], ALLOWED_ANALYSIS_IDS
    )
    validate_available_options(
        require_mapping("available_options", root["available_options"])
    )
    fits_mapping = require_mapping("fits", root["fits"])
    if not fits_mapping:
        raise ValueError("Config section 'fits' cannot be empty.")
    fits: dict[str, PatientBayesFitConfig] = {}
    metric_ids: set[str] = set()
    for fit_id, raw_fit in fits_mapping.items():
        normalized_fit_id = require_string("fits", "fit_id", fit_id)
        fit = parse_fit(
            fit_id=normalized_fit_id,
            section=require_mapping(f"fits.{normalized_fit_id}", raw_fit),
            analysis_id=analysis_id,
        )
        if fit.metric in metric_ids:
            raise ValueError(f"Metric '{fit.metric}' is configured more than once.")
        fits[fit.fit_id] = fit
        metric_ids.add(fit.metric)
    runtime = parse_runtime(require_mapping("runtime", root["runtime"]))
    validate_parallel_sampling(runtime=runtime, fits=fits)
    return PatientBayesAnalysisConfig(
        config_path=Path(path),
        enabled=require_bool("root", "enabled", root["enabled"]),
        analysis_id=analysis_id,
        paths=parse_paths(require_mapping("paths", root["paths"])),
        runtime=runtime,
        filters=(
            parse_filters(require_mapping("filters", root["filters"]))
            if "filters" in root
            else None
        ),
        model=parse_model(require_mapping("model", root["model"])),
        fit_order=tuple(fits),
        fits_by_id=fits,
    )


def enabled_fit_configs(
    config: PatientBayesAnalysisConfig,
) -> list[PatientBayesFitConfig]:
    """Return enabled fits in stable YAML order.

    Args:
        config (PatientBayesAnalysisConfig): Loaded analysis config.

    Returns:
        list[PatientBayesFitConfig]: Enabled per-metric fits.
    """

    if not config.enabled:
        return []
    return [
        config.fits_by_id[fit_id]
        for fit_id in config.fit_order
        if config.fits_by_id[fit_id].enabled
    ]
