"""Strict DMD configuration parsing for hierarchical Bayesian workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH = (
    Path(__file__).resolve().parent / "hierarchical_bayes_config.yaml"
)
ALLOWED_ANALYSIS_IDS = ("instance", "cell_summary")
ALLOWED_SUMMARY_UPDATE_MODES = ("merge", "replace")
ALLOWED_VISUALIZATION_REFRESH_MODES = ("never", "refit_first", "rerun_missing_traces")
ALLOWED_SUPERPLOT_ANNOTATION_MODES = ("bayes_factor", "effect_summary")
ALLOWED_POSITIVE_LIKELIHOODS = ("gamma", "lognormal")
ALLOWED_BOUNDED_LIKELIHOODS = ("beta", "zero_one_inflated_beta", "logitnormal", "logit_skew_normal")
ALLOWED_REAL_LIKELIHOODS = ("normal", "student_t", "skew_normal", "skew_student_t")


@dataclass(frozen=True)
class DmdBayesPathConfig:
    """Filesystem paths used by one DMD Bayesian analysis."""

    input_csv: Path | None
    input_csvs: tuple[Path, ...]
    summary_csv: Path
    trace_dir: Path
    figure_root: Path


@dataclass(frozen=True)
class DmdBayesRuntimeConfig:
    """Runtime switches shared by DMD fitting and visualization scripts."""

    cores: int
    seed: int
    save_idata: bool
    summary_update_mode: str
    visualization_refresh_mode: str
    superplot_annotation_mode: str
    retry_on_warnings: bool
    retry_max_draws: int
    retry_max_tune: int
    retry_max_target_accept: float


@dataclass(frozen=True)
class DmdBayesFilterConfig:
    """Data filters shared by one DMD Bayesian analysis."""

    minimum_zoom: float
    maximum_zoom: float


@dataclass(frozen=True)
class DmdBayesFitConfig:
    """Per-fit DMD model and sampling configuration."""

    fit_id: str
    muscle: str
    compartment: str
    metric: str
    repeat: bool
    likelihood: str
    draws: int
    tune: int
    chains: int
    target_accept: float


@dataclass(frozen=True)
class DmdBayesAnalysisConfig:
    """Normalized configuration for one DMD Bayesian analysis dataset."""

    analysis_id: str
    enabled: bool
    paths: DmdBayesPathConfig
    runtime: DmdBayesRuntimeConfig
    filters: DmdBayesFilterConfig | None
    fit_order: tuple[str, ...]
    fits_by_id: dict[str, DmdBayesFitConfig]
    fits_by_key: dict[tuple[str, str, str], DmdBayesFitConfig]


@dataclass(frozen=True)
class DmdHierarchicalBayesConfig:
    """Normalized DMD hierarchical Bayes workflow configuration."""

    config_path: Path
    paths: DmdBayesPathConfig
    runtime: DmdBayesRuntimeConfig
    filters: DmdBayesFilterConfig | None
    fit_order: tuple[str, ...]
    fits_by_id: dict[str, DmdBayesFitConfig]
    fits_by_key: dict[tuple[str, str, str], DmdBayesFitConfig]
    enabled: bool
    analyses_by_id: dict[str, DmdBayesAnalysisConfig]


def require_mapping(section_name: str, value: Any) -> dict[str, Any]:
    """Require that a config section is a mapping.

    Args:
        section_name (str): Human-readable config section name.
        value (Any): Raw value loaded from YAML.

    Returns:
        dict[str, Any]: The mapping converted to a plain dictionary.
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
    """Validate that a config mapping contains only expected keys.

    Args:
        section_name (str): Human-readable config section name.
        section (dict[str, Any]): Mapping to validate.
        required_keys (set[str]): Keys that must be present.
        optional_keys (set[str] | None): Additional accepted keys.

    Returns:
        None: Raises ``ValueError`` when validation fails.
    """

    allowed_keys = set(required_keys)
    if optional_keys is not None:
        allowed_keys.update(optional_keys)
    missing_keys = sorted(required_keys - set(section))
    if missing_keys:
        raise ValueError(f"Config section '{section_name}' is missing keys: {missing_keys}")
    unexpected_keys = sorted(set(section) - allowed_keys)
    if unexpected_keys:
        raise ValueError(f"Config section '{section_name}' has unexpected keys: {unexpected_keys}")


def require_string(section_name: str, key_name: str, value: Any) -> str:
    """Require that a config value is a non-empty string.

    Args:
        section_name (str): Human-readable config section name.
        key_name (str): Config key name.
        value (Any): Raw value loaded from YAML.

    Returns:
        str: Stripped string value.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config value '{section_name}.{key_name}' must be a non-empty string.")
    return value.strip()


def require_bool(section_name: str, key_name: str, value: Any) -> bool:
    """Require that a config value is boolean.

    Args:
        section_name (str): Human-readable config section name.
        key_name (str): Config key name.
        value (Any): Raw value loaded from YAML.

    Returns:
        bool: Validated boolean value.
    """

    if not isinstance(value, bool):
        raise ValueError(f"Config value '{section_name}.{key_name}' must be a boolean.")
    return value


def require_int(section_name: str, key_name: str, value: Any, minimum: int = 1) -> int:
    """Require that a config value is an integer above a minimum.

    Args:
        section_name (str): Human-readable config section name.
        key_name (str): Config key name.
        value (Any): Raw value loaded from YAML.
        minimum (int): Smallest accepted integer.

    Returns:
        int: Validated integer value.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Config value '{section_name}.{key_name}' must be an integer.")
    if value < minimum:
        raise ValueError(
            f"Config value '{section_name}.{key_name}' must be greater than or equal to {minimum}."
        )
    return value


def require_float(
    section_name: str,
    key_name: str,
    value: Any,
    minimum: float,
    maximum: float,
) -> float:
    """Require that a config value is numeric and within bounds.

    Args:
        section_name (str): Human-readable config section name.
        key_name (str): Config key name.
        value (Any): Raw value loaded from YAML.
        minimum (float): Inclusive lower bound.
        maximum (float): Inclusive upper bound.

    Returns:
        float: Validated floating-point value.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Config value '{section_name}.{key_name}' must be numeric.")
    numeric_value = float(value)
    if numeric_value < minimum or numeric_value > maximum:
        raise ValueError(
            f"Config value '{section_name}.{key_name}' must be between {minimum} and {maximum}."
        )
    return numeric_value


def require_string_list(section_name: str, key_name: str, value: Any) -> tuple[str, ...]:
    """Require that a config value is a non-empty list of strings.

    Args:
        section_name (str): Human-readable config section name.
        key_name (str): Config key name.
        value (Any): Raw value loaded from YAML.

    Returns:
        tuple[str, ...]: Validated string values.
    """

    if not isinstance(value, list) or not value:
        raise ValueError(f"Config value '{section_name}.{key_name}' must be a non-empty list.")
    values = tuple(require_string(section_name, key_name, item) for item in value)
    if len(set(values)) != len(values):
        raise ValueError(f"Config value '{section_name}.{key_name}' contains duplicate entries.")
    return values


def parse_paths_section(section: dict[str, Any]) -> DmdBayesPathConfig:
    """Parse filesystem paths for one DMD analysis.

    Args:
        section (dict[str, Any]): Raw ``paths`` mapping from YAML.

    Returns:
        DmdBayesPathConfig: Validated path configuration.
    """

    validate_section_keys(
        section_name="paths",
        section=section,
        required_keys={"summary_csv", "trace_dir", "figure_root"},
        optional_keys={"input_csv", "input_csvs"},
    )
    input_csv = None
    input_csvs: tuple[Path, ...] = ()
    if "input_csv" in section:
        input_csv = Path(require_string("paths", "input_csv", section["input_csv"]))
    if "input_csvs" in section:
        raw_paths = require_string_list("paths", "input_csvs", section["input_csvs"])
        input_csvs = tuple(Path(raw_path) for raw_path in raw_paths)
    if input_csv is None and not input_csvs:
        raise ValueError("Config section 'paths' must contain either 'input_csv' or 'input_csvs'.")
    return DmdBayesPathConfig(
        input_csv=input_csv,
        input_csvs=input_csvs,
        summary_csv=Path(require_string("paths", "summary_csv", section["summary_csv"])),
        trace_dir=Path(require_string("paths", "trace_dir", section["trace_dir"])),
        figure_root=Path(require_string("paths", "figure_root", section["figure_root"])),
    )


def require_allowed_string(
    section_name: str,
    key_name: str,
    value: Any,
    allowed_values: tuple[str, ...],
) -> str:
    """Require that a string config value is one of a fixed set.

    Args:
        section_name (str): Human-readable config section name.
        key_name (str): Config key name.
        value (Any): Raw value loaded from YAML.
        allowed_values (tuple[str, ...]): Accepted string values.

    Returns:
        str: Validated string value.
    """

    string_value = require_string(section_name, key_name, value)
    if string_value not in allowed_values:
        raise ValueError(
            f"Config value '{section_name}.{key_name}' must be one of {list(allowed_values)}."
        )
    return string_value


def parse_runtime_section(section: dict[str, Any]) -> DmdBayesRuntimeConfig:
    """Parse runtime options for one DMD analysis.

    Args:
        section (dict[str, Any]): Raw ``runtime`` mapping from YAML.

    Returns:
        DmdBayesRuntimeConfig: Validated runtime configuration.
    """

    validate_section_keys(
        section_name="runtime",
        section=section,
        required_keys={
            "cores",
            "seed",
            "save_idata",
            "summary_update_mode",
            "visualization_refresh_mode",
            "superplot_annotation_mode",
            "retry_on_warnings",
            "retry_max_draws",
            "retry_max_tune",
            "retry_max_target_accept",
        },
    )
    return DmdBayesRuntimeConfig(
        cores=require_int("runtime", "cores", section["cores"]),
        seed=require_int("runtime", "seed", section["seed"], minimum=0),
        save_idata=require_bool("runtime", "save_idata", section["save_idata"]),
        summary_update_mode=require_allowed_string(
            "runtime",
            "summary_update_mode",
            section["summary_update_mode"],
            ALLOWED_SUMMARY_UPDATE_MODES,
        ),
        visualization_refresh_mode=require_allowed_string(
            "runtime",
            "visualization_refresh_mode",
            section["visualization_refresh_mode"],
            ALLOWED_VISUALIZATION_REFRESH_MODES,
        ),
        superplot_annotation_mode=require_allowed_string(
            "runtime",
            "superplot_annotation_mode",
            section["superplot_annotation_mode"],
            ALLOWED_SUPERPLOT_ANNOTATION_MODES,
        ),
        retry_on_warnings=require_bool("runtime", "retry_on_warnings", section["retry_on_warnings"]),
        retry_max_draws=require_int("runtime", "retry_max_draws", section["retry_max_draws"]),
        retry_max_tune=require_int("runtime", "retry_max_tune", section["retry_max_tune"]),
        retry_max_target_accept=require_float(
            "runtime",
            "retry_max_target_accept",
            section["retry_max_target_accept"],
            minimum=0.8,
            maximum=0.999,
        ),
    )


def parse_filters_section(section: dict[str, Any]) -> DmdBayesFilterConfig:
    """Parse data filters for one DMD analysis.

    Args:
        section (dict[str, Any]): Raw ``filters`` mapping from YAML.

    Returns:
        DmdBayesFilterConfig: Validated zoom filter configuration.
    """

    validate_section_keys(
        section_name="filters",
        section=section,
        required_keys={"minimum_zoom", "maximum_zoom"},
    )
    minimum_zoom = require_float(
        "filters",
        "minimum_zoom",
        section["minimum_zoom"],
        minimum=0.0,
        maximum=float("inf"),
    )
    maximum_zoom = require_float(
        "filters",
        "maximum_zoom",
        section["maximum_zoom"],
        minimum=0.0,
        maximum=float("inf"),
    )
    if maximum_zoom < minimum_zoom:
        raise ValueError("Config value 'filters.maximum_zoom' must be greater than or equal to minimum_zoom.")
    return DmdBayesFilterConfig(
        minimum_zoom=minimum_zoom,
        maximum_zoom=maximum_zoom,
    )


def validate_available_options(section: dict[str, Any], allow_real_likelihoods: bool) -> None:
    """Validate the documentation-only ``available_options`` config block.

    Args:
        section (dict[str, Any]): Raw ``available_options`` mapping from YAML.
        allow_real_likelihoods (bool): Whether real-valued likelihoods are valid for this analysis.

    Returns:
        None: Raises ``ValueError`` when the options do not match the parser.
    """

    optional_keys = {"real_likelihoods"} if allow_real_likelihoods else set()
    validate_section_keys(
        section_name="available_options",
        section=section,
        required_keys={
            "positive_likelihoods",
            "bounded_likelihoods",
            "summary_update_modes",
            "visualization_refresh_modes",
            "superplot_annotation_modes",
            "repeat_options",
        },
        optional_keys=optional_keys,
    )
    expected_pairs = {
        "positive_likelihoods": ALLOWED_POSITIVE_LIKELIHOODS,
        "bounded_likelihoods": ALLOWED_BOUNDED_LIKELIHOODS,
        "summary_update_modes": ALLOWED_SUMMARY_UPDATE_MODES,
        "visualization_refresh_modes": ALLOWED_VISUALIZATION_REFRESH_MODES,
        "superplot_annotation_modes": ALLOWED_SUPERPLOT_ANNOTATION_MODES,
    }
    for key_name, expected_values in expected_pairs.items():
        if tuple(section[key_name]) != expected_values:
            raise ValueError(f"Config section 'available_options.{key_name}' must be {list(expected_values)}.")
    if tuple(section["repeat_options"]) != (True, False):
        raise ValueError("Config section 'available_options.repeat_options' must be [true, false].")
    if allow_real_likelihoods and tuple(section.get("real_likelihoods", ())) != ALLOWED_REAL_LIKELIHOODS:
        raise ValueError(
            "Config section 'available_options.real_likelihoods' must be "
            f"{list(ALLOWED_REAL_LIKELIHOODS)}."
        )


def likelihoods_for_metric(
    metric: str,
    positive_metrics: tuple[str, ...],
    bounded_metrics: tuple[str, ...],
    real_metrics: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return valid likelihoods for a configured metric.

    Args:
        metric (str): Metric column name from the input CSV.
        positive_metrics (tuple[str, ...]): Metrics modeled as positive values.
        bounded_metrics (tuple[str, ...]): Metrics modeled on the unit interval.
        real_metrics (tuple[str, ...]): Metrics modeled on the real line.

    Returns:
        tuple[str, ...]: Allowed likelihood names for the metric.
    """

    if metric in positive_metrics:
        return ALLOWED_POSITIVE_LIKELIHOODS
    if metric in bounded_metrics:
        return ALLOWED_BOUNDED_LIKELIHOODS
    if metric in real_metrics:
        return ALLOWED_REAL_LIKELIHOODS
    raise ValueError(f"Unsupported metric in config: {metric}")


def parse_sampling_fields(section_name: str, section: dict[str, Any]) -> dict[str, Any]:
    """Parse fields shared by explicit and grid-generated fits.

    Args:
        section_name (str): Human-readable config section name.
        section (dict[str, Any]): Raw fit or metric mapping from YAML.

    Returns:
        dict[str, Any]: Parsed likelihood and sampling values.
    """

    validate_section_keys(
        section_name=section_name,
        section=section,
        required_keys={"repeat", "likelihood", "draws", "tune", "chains", "target_accept"},
    )
    return {
        "repeat": require_bool(section_name, "repeat", section["repeat"]),
        "likelihood": require_string(section_name, "likelihood", section["likelihood"]),
        "draws": require_int(section_name, "draws", section["draws"]),
        "tune": require_int(section_name, "tune", section["tune"]),
        "chains": require_int(section_name, "chains", section["chains"]),
        "target_accept": require_float(
            section_name,
            "target_accept",
            section["target_accept"],
            minimum=0.8,
            maximum=0.999,
        ),
    }


def slugify_value(value: str) -> str:
    """Convert a label into a stable lowercase identifier.

    Args:
        value (str): Raw label.

    Returns:
        str: Filesystem- and YAML-key-friendly slug.
    """

    slug = "".join(character if character.isalnum() else "_" for character in str(value).lower())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def build_fit_id(muscle: str, compartment: str, metric: str) -> str:
    """Build a deterministic fit identifier.

    Args:
        muscle (str): Muscle label.
        compartment (str): Compartment label.
        metric (str): Metric column name.

    Returns:
        str: Identifier combining muscle, compartment, and metric.
    """

    return "__".join(
        [
            slugify_value(muscle),
            slugify_value(compartment),
            slugify_value(metric),
        ]
    )


def parse_explicit_fit(
    fit_id: str,
    section: dict[str, Any],
    positive_metrics: tuple[str, ...],
    bounded_metrics: tuple[str, ...],
    real_metrics: tuple[str, ...] = (),
) -> DmdBayesFitConfig:
    """Parse one explicit ``fits`` entry.

    Args:
        fit_id (str): Mapping key used for the fit.
        section (dict[str, Any]): Raw fit mapping from YAML.
        positive_metrics (tuple[str, ...]): Metrics modeled as positive values.
        bounded_metrics (tuple[str, ...]): Metrics modeled on the unit interval.
        real_metrics (tuple[str, ...]): Metrics modeled on the real line.

    Returns:
        DmdBayesFitConfig: Validated fit configuration.
    """

    section_name = f"fits.{fit_id}"
    validate_section_keys(
        section_name=section_name,
        section=section,
        required_keys={
            "muscle",
            "compartment",
            "metric",
            "repeat",
            "likelihood",
            "draws",
            "tune",
            "chains",
            "target_accept",
        },
    )
    muscle = require_string(section_name, "muscle", section["muscle"])
    compartment = require_string(section_name, "compartment", section["compartment"])
    metric = require_string(section_name, "metric", section["metric"])
    sampling_section = {
        key: section[key]
        for key in ("repeat", "likelihood", "draws", "tune", "chains", "target_accept")
    }
    parsed = parse_sampling_fields(section_name=section_name, section=sampling_section)
    if parsed["likelihood"] not in likelihoods_for_metric(metric, positive_metrics, bounded_metrics, real_metrics):
        raise ValueError(f"Invalid likelihood for configured metric '{metric}' in '{section_name}'.")
    return DmdBayesFitConfig(
        fit_id=fit_id,
        muscle=muscle,
        compartment=compartment,
        metric=metric,
        **parsed,
    )


def parse_fit_grid(
    section: dict[str, Any],
    positive_metrics: tuple[str, ...],
    bounded_metrics: tuple[str, ...],
    real_metrics: tuple[str, ...] = (),
) -> list[DmdBayesFitConfig]:
    """Expand a compact fit grid into per-fit configurations.

    Args:
        section (dict[str, Any]): Raw ``fit_grid`` mapping from YAML.
        positive_metrics (tuple[str, ...]): Metrics modeled as positive values.
        bounded_metrics (tuple[str, ...]): Metrics modeled on the unit interval.
        real_metrics (tuple[str, ...]): Metrics modeled on the real line.

    Returns:
        list[DmdBayesFitConfig]: Fit configurations in deterministic grid order.
    """

    validate_section_keys(
        section_name="fit_grid",
        section=section,
        required_keys={"muscles", "compartments", "metrics"},
    )
    muscles = require_string_list("fit_grid", "muscles", section["muscles"])
    compartments = require_string_list("fit_grid", "compartments", section["compartments"])
    metric_sections = require_mapping("fit_grid.metrics", section["metrics"])
    fits: list[DmdBayesFitConfig] = []
    for muscle in muscles:
        for compartment in compartments:
            for metric, raw_metric_section in metric_sections.items():
                metric_name = require_string("fit_grid.metrics", str(metric), str(metric))
                metric_section = require_mapping(f"fit_grid.metrics.{metric_name}", raw_metric_section)
                parsed = parse_sampling_fields(
                    section_name=f"fit_grid.metrics.{metric_name}",
                    section=metric_section,
                )
                if parsed["likelihood"] not in likelihoods_for_metric(
                    metric_name,
                    positive_metrics,
                    bounded_metrics,
                    real_metrics,
                ):
                    raise ValueError(f"Invalid likelihood for configured metric '{metric_name}'.")
                fit_id = build_fit_id(muscle=muscle, compartment=compartment, metric=metric_name)
                fits.append(
                    DmdBayesFitConfig(
                        fit_id=fit_id,
                        muscle=muscle,
                        compartment=compartment,
                        metric=metric_name,
                        **parsed,
                    )
                )
    return fits


def normalize_fit_configs(fits: list[DmdBayesFitConfig]) -> DmdBayesAnalysisConfig:
    """Placeholder to satisfy type checkers during incremental parsing.

    Args:
        fits (list[DmdBayesFitConfig]): Parsed fit configurations.

    Returns:
        DmdBayesAnalysisConfig: This helper is not called and exists only to keep
        the parser implementation compact.
    """

    raise NotImplementedError("normalize_fit_configs is not used directly.")


def parse_fit_configs(
    config_mapping: dict[str, Any],
    positive_metrics: tuple[str, ...],
    bounded_metrics: tuple[str, ...],
    real_metrics: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], dict[str, DmdBayesFitConfig], dict[tuple[str, str, str], DmdBayesFitConfig]]:
    """Parse explicit or grid-based fit configuration.

    Args:
        config_mapping (dict[str, Any]): Analysis config mapping.
        positive_metrics (tuple[str, ...]): Metrics modeled as positive values.
        bounded_metrics (tuple[str, ...]): Metrics modeled on the unit interval.
        real_metrics (tuple[str, ...]): Metrics modeled on the real line.

    Returns:
        tuple[tuple[str, ...], dict[str, DmdBayesFitConfig], dict[tuple[str, str, str], DmdBayesFitConfig]]:
            Fit order, fits keyed by ID, and fits keyed by muscle/compartment/metric.
    """

    has_fits = "fits" in config_mapping
    has_fit_grid = "fit_grid" in config_mapping
    if has_fits == has_fit_grid:
        raise ValueError("Config must contain exactly one of 'fits' or 'fit_grid'.")
    if has_fit_grid:
        fits = parse_fit_grid(
            section=require_mapping("fit_grid", config_mapping["fit_grid"]),
            positive_metrics=positive_metrics,
            bounded_metrics=bounded_metrics,
            real_metrics=real_metrics,
        )
    else:
        fits_section = require_mapping("fits", config_mapping["fits"])
        fits = [
            parse_explicit_fit(
                fit_id=fit_id,
                section=require_mapping(f"fits.{fit_id}", fit_section),
                positive_metrics=positive_metrics,
                bounded_metrics=bounded_metrics,
                real_metrics=real_metrics,
            )
            for fit_id, fit_section in fits_section.items()
        ]
    if not fits:
        raise ValueError("Config must define at least one fit.")
    fit_order = tuple(fit.fit_id for fit in fits)
    fits_by_id = {fit.fit_id: fit for fit in fits}
    if len(fits_by_id) != len(fits):
        raise ValueError("Config contains duplicate fit identifiers.")
    fits_by_key: dict[tuple[str, str, str], DmdBayesFitConfig] = {}
    for fit in fits:
        fit_key = (fit.muscle, fit.compartment, fit.metric)
        if fit_key in fits_by_key:
            raise ValueError(f"Config contains duplicate fit for {fit_key}.")
        fits_by_key[fit_key] = fit
    return fit_order, fits_by_id, fits_by_key


def parse_analysis_config(
    analysis_id: str,
    config_mapping: dict[str, Any],
    positive_metrics: tuple[str, ...],
    bounded_metrics: tuple[str, ...],
    real_metrics: tuple[str, ...] = (),
) -> DmdBayesAnalysisConfig:
    """Parse one DMD analysis section.

    Args:
        analysis_id (str): Analysis identifier from the YAML file.
        config_mapping (dict[str, Any]): Raw analysis mapping.
        positive_metrics (tuple[str, ...]): Metrics modeled as positive values.
        bounded_metrics (tuple[str, ...]): Metrics modeled on the unit interval.
        real_metrics (tuple[str, ...]): Metrics modeled on the real line.

    Returns:
        DmdBayesAnalysisConfig: Validated analysis configuration.
    """

    validate_section_keys(
        section_name=analysis_id,
        section=config_mapping,
        required_keys={"paths", "runtime", "available_options"},
        optional_keys={"enabled", "filters", "fits", "fit_grid"},
    )
    validate_available_options(
        section=require_mapping(f"{analysis_id}.available_options", config_mapping["available_options"]),
        allow_real_likelihoods=bool(real_metrics),
    )
    fit_order, fits_by_id, fits_by_key = parse_fit_configs(
        config_mapping=config_mapping,
        positive_metrics=positive_metrics,
        bounded_metrics=bounded_metrics,
        real_metrics=real_metrics,
    )
    return DmdBayesAnalysisConfig(
        analysis_id=analysis_id,
        enabled=require_bool(analysis_id, "enabled", config_mapping.get("enabled", True)),
        paths=parse_paths_section(require_mapping(f"{analysis_id}.paths", config_mapping["paths"])),
        runtime=parse_runtime_section(require_mapping(f"{analysis_id}.runtime", config_mapping["runtime"])),
        filters=(
            parse_filters_section(require_mapping(f"{analysis_id}.filters", config_mapping["filters"]))
            if "filters" in config_mapping
            else None
        ),
        fit_order=fit_order,
        fits_by_id=fits_by_id,
        fits_by_key=fits_by_key,
    )


def load_hierarchical_bayes_config(
    path: Path,
    positive_metrics: tuple[str, ...],
    bounded_metrics: tuple[str, ...],
    cell_summary_positive_metrics: tuple[str, ...] = (),
    cell_summary_bounded_metrics: tuple[str, ...] = (),
    cell_summary_real_metrics: tuple[str, ...] = (),
) -> DmdHierarchicalBayesConfig:
    """Load and validate a DMD hierarchical Bayes YAML config.

    Args:
        path (Path): YAML config path.
        positive_metrics (tuple[str, ...]): Instance positive metrics.
        bounded_metrics (tuple[str, ...]): Instance bounded metrics.
        cell_summary_positive_metrics (tuple[str, ...]): Cell-summary positive metrics.
        cell_summary_bounded_metrics (tuple[str, ...]): Cell-summary bounded metrics.
        cell_summary_real_metrics (tuple[str, ...]): Cell-summary real-valued metrics.

    Returns:
        DmdHierarchicalBayesConfig: Normalized workflow configuration.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    config_mapping = require_mapping("root", raw_config)
    validate_section_keys(
        section_name="root",
        section=config_mapping,
        required_keys={"paths", "runtime", "available_options"},
        optional_keys={"enabled", "analysis_id", "filters", "fits", "fit_grid", "cell_summary"},
    )
    root_analysis_id = require_allowed_string(
        "root",
        "analysis_id",
        config_mapping.get("analysis_id", "instance"),
        ALLOWED_ANALYSIS_IDS,
    )
    if root_analysis_id == "cell_summary":
        root_positive_metrics = cell_summary_positive_metrics
        root_bounded_metrics = cell_summary_bounded_metrics
        root_real_metrics = cell_summary_real_metrics
    else:
        root_positive_metrics = positive_metrics
        root_bounded_metrics = bounded_metrics
        root_real_metrics = ()
    root_analysis = parse_analysis_config(
        analysis_id=root_analysis_id,
        config_mapping={
            "enabled": config_mapping.get("enabled", True),
            "paths": config_mapping["paths"],
            "runtime": config_mapping["runtime"],
            "available_options": config_mapping["available_options"],
            **({"filters": config_mapping["filters"]} if "filters" in config_mapping else {}),
            **({"fits": config_mapping["fits"]} if "fits" in config_mapping else {}),
            **({"fit_grid": config_mapping["fit_grid"]} if "fit_grid" in config_mapping else {}),
        },
        positive_metrics=root_positive_metrics,
        bounded_metrics=root_bounded_metrics,
        real_metrics=root_real_metrics,
    )
    analyses_by_id = {root_analysis.analysis_id: root_analysis}
    if "cell_summary" in config_mapping:
        analyses_by_id["cell_summary"] = parse_analysis_config(
            analysis_id="cell_summary",
            config_mapping=require_mapping("cell_summary", config_mapping["cell_summary"]),
            positive_metrics=cell_summary_positive_metrics,
            bounded_metrics=cell_summary_bounded_metrics,
            real_metrics=cell_summary_real_metrics,
        )
    return DmdHierarchicalBayesConfig(
        config_path=Path(path),
        paths=root_analysis.paths,
        runtime=root_analysis.runtime,
        filters=root_analysis.filters,
        fit_order=root_analysis.fit_order,
        fits_by_id=root_analysis.fits_by_id,
        fits_by_key=root_analysis.fits_by_key,
        enabled=root_analysis.enabled,
        analyses_by_id=analyses_by_id,
    )


def repeated_fit_configs(config: DmdBayesAnalysisConfig) -> list[DmdBayesFitConfig]:
    """Return fit entries whose ``repeat`` flag is enabled.

    Args:
        config (DmdBayesAnalysisConfig): Analysis configuration.

    Returns:
        list[DmdBayesFitConfig]: Enabled fit entries in YAML order.
    """

    if not config.enabled:
        return []
    return [
        config.fits_by_id[fit_id]
        for fit_id in config.fit_order
        if config.fits_by_id[fit_id].repeat
    ]


def fit_config_for_key(
    config: DmdBayesAnalysisConfig,
    muscle: str,
    compartment: str,
    metric: str,
) -> DmdBayesFitConfig:
    """Return the config entry for one muscle/compartment/metric fit.

    Args:
        config (DmdBayesAnalysisConfig): Analysis configuration.
        muscle (str): Muscle label.
        compartment (str): Compartment label.
        metric (str): Metric column name.

    Returns:
        DmdBayesFitConfig: Matching fit configuration.
    """

    fit_key = (str(muscle), str(compartment), str(metric))
    if fit_key not in config.fits_by_key:
        raise KeyError(f"Config does not define fit {fit_key}.")
    return config.fits_by_key[fit_key]


def validate_config_groups(
    config: DmdBayesAnalysisConfig,
    valid_muscles: set[str],
    valid_compartments: set[str],
) -> None:
    """Require configured muscles and compartments to exist in the dataset.

    Args:
        config (DmdBayesAnalysisConfig): Analysis configuration.
        valid_muscles (set[str]): Muscle labels present in the data.
        valid_compartments (set[str]): Compartment labels present in the data.

    Returns:
        None: Raises ``ValueError`` when any configured group is absent.
    """

    configured_muscles = sorted({fit.muscle for fit in config.fits_by_id.values()})
    configured_compartments = sorted({fit.compartment for fit in config.fits_by_id.values()})
    unknown_muscles = [muscle for muscle in configured_muscles if muscle not in valid_muscles]
    unknown_compartments = [
        compartment for compartment in configured_compartments if compartment not in valid_compartments
    ]
    if unknown_muscles:
        raise ValueError(f"Config contains muscles absent from the data: {unknown_muscles}.")
    if unknown_compartments:
        raise ValueError(f"Config contains compartments absent from the data: {unknown_compartments}.")
