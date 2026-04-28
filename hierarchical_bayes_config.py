"""Strict YAML configuration for the hierarchical Bayesian workflow.

Positive-metric units are inherited from the configured measurement CSV rather
than hardcoded here, so the workflow remains valid after rebuilding the inputs
in nanometers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH = Path(
    "/workspaces/mito-counter/hierarchical_bayes_config.yaml"
)
ALLOWED_SUMMARY_UPDATE_MODES = ("merge", "replace")
ALLOWED_VISUALIZATION_REFRESH_MODES = ("never", "refit_first", "rerun_missing_traces")
ALLOWED_POSITIVE_LIKELIHOODS = ("gamma", "lognormal")
ALLOWED_BOUNDED_LIKELIHOODS = ("beta", "zero_one_inflated_beta", "logitnormal", "logit_skew_normal")


@dataclass(frozen=True)
class BayesPathConfig:
    """Filesystem paths used by the hierarchical Bayesian workflow."""

    input_csv: Path
    summary_csv: Path
    trace_dir: Path
    figure_root: Path


@dataclass(frozen=True)
class BayesRuntimeConfig:
    """Global runtime switches shared by the fit and visualization scripts."""

    cores: int
    seed: int
    save_idata: bool
    summary_update_mode: str
    visualization_refresh_mode: str


@dataclass(frozen=True)
class BayesFitConfig:
    """Per-fit model and sampling configuration."""

    fit_id: str
    muscle: str
    metric: str
    repeat: bool
    likelihood: str
    draws: int
    tune: int
    chains: int
    target_accept: float


@dataclass(frozen=True)
class HierarchicalBayesConfig:
    """Normalized hierarchical Bayes workflow configuration."""

    config_path: Path
    paths: BayesPathConfig
    runtime: BayesRuntimeConfig
    fit_order: tuple[str, ...]
    fits_by_id: dict[str, BayesFitConfig]
    fits_by_pair: dict[tuple[str, str], BayesFitConfig]


def require_mapping(
    section_name: str,
    value: Any,
) -> dict[str, Any]:
    """Require that a config section is a dictionary-like mapping.

    Args:
        section_name (str): Human-readable config section name for error messages.
        value (Any): Raw loaded YAML value for the section.

    Returns:
        dict[str, Any]: Mapping value converted to a plain dictionary.
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
    """Validate that a mapping contains exactly the expected keys.

    Args:
        section_name (str): Human-readable config section name for error messages.
        section (dict[str, Any]): Mapping to validate.
        required_keys (set[str]): Keys that must be present.
        optional_keys (set[str] | None): Additional accepted keys.

    Returns:
        None
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


def require_string(
    section_name: str,
    key_name: str,
    value: Any,
) -> str:
    """Require that a config value is a non-empty string.

    Args:
        section_name (str): Human-readable config section name for error messages.
        key_name (str): Config key name for error messages.
        value (Any): Raw loaded YAML value.

    Returns:
        str: Validated string value.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config value '{section_name}.{key_name}' must be a non-empty string.")
    return value.strip()


def require_bool(
    section_name: str,
    key_name: str,
    value: Any,
) -> bool:
    """Require that a config value is boolean.

    Args:
        section_name (str): Human-readable config section name for error messages.
        key_name (str): Config key name for error messages.
        value (Any): Raw loaded YAML value.

    Returns:
        bool: Validated boolean value.
    """

    if not isinstance(value, bool):
        raise ValueError(f"Config value '{section_name}.{key_name}' must be a boolean.")
    return value


def require_int(
    section_name: str,
    key_name: str,
    value: Any,
    minimum: int = 1,
) -> int:
    """Require that a config value is an integer above a minimum.

    Args:
        section_name (str): Human-readable config section name for error messages.
        key_name (str): Config key name for error messages.
        value (Any): Raw loaded YAML value.
        minimum (int): Smallest allowed integer value.

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
    """Require that a config value is numeric and within a closed interval.

    Args:
        section_name (str): Human-readable config section name for error messages.
        key_name (str): Config key name for error messages.
        value (Any): Raw loaded YAML value.
        minimum (float): Lower inclusive bound.
        maximum (float): Upper inclusive bound.

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


def parse_paths_section(section: dict[str, Any]) -> BayesPathConfig:
    """Parse and validate the `paths` config section.

    Args:
        section (dict[str, Any]): Raw `paths` mapping from the YAML config.

    Returns:
        BayesPathConfig: Validated path configuration.
    """

    validate_section_keys(
        section_name="paths",
        section=section,
        required_keys={"input_csv", "summary_csv", "trace_dir", "figure_root"},
    )
    return BayesPathConfig(
        input_csv=Path(require_string("paths", "input_csv", section["input_csv"])),
        summary_csv=Path(require_string("paths", "summary_csv", section["summary_csv"])),
        trace_dir=Path(require_string("paths", "trace_dir", section["trace_dir"])),
        figure_root=Path(require_string("paths", "figure_root", section["figure_root"])),
    )


def parse_runtime_section(section: dict[str, Any]) -> BayesRuntimeConfig:
    """Parse and validate the `runtime` config section.

    Args:
        section (dict[str, Any]): Raw `runtime` mapping from the YAML config.

    Returns:
        BayesRuntimeConfig: Validated runtime configuration.
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
        },
    )
    summary_update_mode = require_string(
        "runtime",
        "summary_update_mode",
        section["summary_update_mode"],
    )
    if summary_update_mode not in ALLOWED_SUMMARY_UPDATE_MODES:
        raise ValueError(
            "Config value 'runtime.summary_update_mode' must be one of "
            f"{list(ALLOWED_SUMMARY_UPDATE_MODES)}."
        )
    visualization_refresh_mode = require_string(
        "runtime",
        "visualization_refresh_mode",
        section["visualization_refresh_mode"],
    )
    if visualization_refresh_mode not in ALLOWED_VISUALIZATION_REFRESH_MODES:
        raise ValueError(
            "Config value 'runtime.visualization_refresh_mode' must be one of "
            f"{list(ALLOWED_VISUALIZATION_REFRESH_MODES)}."
        )
    return BayesRuntimeConfig(
        cores=require_int("runtime", "cores", section["cores"]),
        seed=require_int("runtime", "seed", section["seed"], minimum=0),
        save_idata=require_bool("runtime", "save_idata", section["save_idata"]),
        summary_update_mode=summary_update_mode,
        visualization_refresh_mode=visualization_refresh_mode,
    )


def validate_available_options(section: dict[str, Any]) -> None:
    """Validate the `available_options` documentation block for consistency.

    Args:
        section (dict[str, Any]): Raw `available_options` mapping from the YAML config.

    Returns:
        None
    """

    validate_section_keys(
        section_name="available_options",
        section=section,
        required_keys={
            "positive_likelihoods",
            "bounded_likelihoods",
            "summary_update_modes",
            "visualization_refresh_modes",
            "repeat_options",
        },
    )
    positive_likelihoods = tuple(section["positive_likelihoods"])
    bounded_likelihoods = tuple(section["bounded_likelihoods"])
    summary_update_modes = tuple(section["summary_update_modes"])
    visualization_refresh_modes = tuple(section["visualization_refresh_modes"])
    repeat_options = tuple(section["repeat_options"])
    if positive_likelihoods != ALLOWED_POSITIVE_LIKELIHOODS:
        raise ValueError(
            "Config section 'available_options.positive_likelihoods' must be "
            f"{list(ALLOWED_POSITIVE_LIKELIHOODS)}."
        )
    if bounded_likelihoods != ALLOWED_BOUNDED_LIKELIHOODS:
        raise ValueError(
            "Config section 'available_options.bounded_likelihoods' must be "
            f"{list(ALLOWED_BOUNDED_LIKELIHOODS)}."
        )
    if summary_update_modes != ALLOWED_SUMMARY_UPDATE_MODES:
        raise ValueError(
            "Config section 'available_options.summary_update_modes' must be "
            f"{list(ALLOWED_SUMMARY_UPDATE_MODES)}."
        )
    if visualization_refresh_modes != ALLOWED_VISUALIZATION_REFRESH_MODES:
        raise ValueError(
            "Config section 'available_options.visualization_refresh_modes' must be "
            f"{list(ALLOWED_VISUALIZATION_REFRESH_MODES)}."
        )
    if repeat_options != (True, False):
        raise ValueError(
            "Config section 'available_options.repeat_options' must be [true, false]."
        )


def parse_fit_entry(
    fit_id: str,
    section: dict[str, Any],
    positive_metrics: tuple[str, ...],
    bounded_metrics: tuple[str, ...],
) -> BayesFitConfig:
    """Parse and validate one `fits` entry from the YAML config.

    Args:
        fit_id (str): Mapping key used for the fit entry.
        section (dict[str, Any]): Raw fit mapping from the YAML config.
        positive_metrics (tuple[str, ...]): Metric names allowed to use positive likelihoods.
        bounded_metrics (tuple[str, ...]): Metric names allowed to use bounded likelihoods.

    Returns:
        BayesFitConfig: Validated per-fit configuration.
    """

    section_name = f"fits.{fit_id}"
    validate_section_keys(
        section_name=section_name,
        section=section,
        required_keys={"muscle", "metric", "repeat", "likelihood", "draws", "tune", "chains", "target_accept"},
    )
    muscle = require_string(section_name, "muscle", section["muscle"])
    metric = require_string(section_name, "metric", section["metric"])
    likelihood = require_string(section_name, "likelihood", section["likelihood"])
    metric_family = "unknown"
    if metric in positive_metrics:
        metric_family = "positive"
        if likelihood not in ALLOWED_POSITIVE_LIKELIHOODS:
            raise ValueError(
                "Config value "
                f"'{section_name}.likelihood' must be one of {list(ALLOWED_POSITIVE_LIKELIHOODS)} for metric '{metric}'."
            )
    elif metric in bounded_metrics:
        metric_family = "bounded"
        if likelihood not in ALLOWED_BOUNDED_LIKELIHOODS:
            raise ValueError(
                "Config value "
                f"'{section_name}.likelihood' must be one of {list(ALLOWED_BOUNDED_LIKELIHOODS)} for metric '{metric}'."
            )
    if metric_family == "unknown":
        raise ValueError(f"Config value '{section_name}.metric' is not a supported metric: {metric}.")
    return BayesFitConfig(
        fit_id=fit_id,
        muscle=muscle,
        metric=metric,
        repeat=require_bool(section_name, "repeat", section["repeat"]),
        likelihood=likelihood,
        draws=require_int(section_name, "draws", section["draws"]),
        tune=require_int(section_name, "tune", section["tune"]),
        chains=require_int(section_name, "chains", section["chains"]),
        target_accept=require_float(section_name, "target_accept", section["target_accept"], minimum=0.8, maximum=0.999),
    )


def validate_fit_coverage(
    fits_by_pair: dict[tuple[str, str], BayesFitConfig],
    positive_metrics: tuple[str, ...],
    bounded_metrics: tuple[str, ...],
) -> None:
    """Require explicit per-metric coverage for each muscle present in the config.

    Args:
        fits_by_pair (dict[tuple[str, str], BayesFitConfig]): Fit configs keyed by `(muscle, metric)`.
        positive_metrics (tuple[str, ...]): Supported positive metric names.
        bounded_metrics (tuple[str, ...]): Supported bounded metric names.

    Returns:
        None
    """

    all_metrics = tuple(positive_metrics) + tuple(bounded_metrics)
    muscles = sorted({muscle for muscle, _ in fits_by_pair})
    missing_pairs = [
        (muscle, metric)
        for muscle in muscles
        for metric in all_metrics
        if (muscle, metric) not in fits_by_pair
    ]
    if missing_pairs:
        raise ValueError(
            "Config section 'fits' must contain an explicit entry for every metric in each configured muscle. "
            f"Missing pairs: {missing_pairs}"
        )


def load_hierarchical_bayes_config(
    path: Path,
    positive_metrics: tuple[str, ...],
    bounded_metrics: tuple[str, ...],
) -> HierarchicalBayesConfig:
    """Load, validate, and normalize the hierarchical Bayes YAML config.

    Args:
        path (Path): YAML configuration path.
        positive_metrics (tuple[str, ...]): Supported positive metric names.
        bounded_metrics (tuple[str, ...]): Supported bounded metric names.

    Returns:
        HierarchicalBayesConfig: Validated workflow configuration.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    config_mapping = require_mapping("root", raw_config)
    validate_section_keys(
        section_name="root",
        section=config_mapping,
        required_keys={"paths", "runtime", "available_options", "fits"},
    )
    paths = parse_paths_section(require_mapping("paths", config_mapping["paths"]))
    runtime = parse_runtime_section(require_mapping("runtime", config_mapping["runtime"]))
    validate_available_options(require_mapping("available_options", config_mapping["available_options"]))
    fits_section = require_mapping("fits", config_mapping["fits"])
    if not fits_section:
        raise ValueError("Config section 'fits' must contain at least one fit entry.")
    fit_order = tuple(fits_section.keys())
    fits_by_id: dict[str, BayesFitConfig] = {}
    fits_by_pair: dict[tuple[str, str], BayesFitConfig] = {}
    for fit_id in fit_order:
        fit_config = parse_fit_entry(
            fit_id=fit_id,
            section=require_mapping(f"fits.{fit_id}", fits_section[fit_id]),
            positive_metrics=positive_metrics,
            bounded_metrics=bounded_metrics,
        )
        fit_pair = (fit_config.muscle, fit_config.metric)
        if fit_pair in fits_by_pair:
            raise ValueError(
                "Config section 'fits' contains duplicate muscle/metric entries for "
                f"{fit_pair}."
            )
        fits_by_id[fit_id] = fit_config
        fits_by_pair[fit_pair] = fit_config
    validate_fit_coverage(
        fits_by_pair=fits_by_pair,
        positive_metrics=positive_metrics,
        bounded_metrics=bounded_metrics,
    )
    return HierarchicalBayesConfig(
        config_path=Path(path),
        paths=paths,
        runtime=runtime,
        fit_order=fit_order,
        fits_by_id=fits_by_id,
        fits_by_pair=fits_by_pair,
    )


def repeated_fit_configs(config: HierarchicalBayesConfig) -> list[BayesFitConfig]:
    """Return fit entries whose `repeat` flag is enabled.

    Args:
        config (HierarchicalBayesConfig): Validated workflow configuration.

    Returns:
        list[BayesFitConfig]: Enabled fit entries in YAML order.
    """

    return [
        config.fits_by_id[fit_id]
        for fit_id in config.fit_order
        if config.fits_by_id[fit_id].repeat
    ]


def fit_config_for_pair(
    config: HierarchicalBayesConfig,
    muscle: str,
    metric: str,
) -> BayesFitConfig:
    """Return the config entry for one `(muscle, metric)` pair.

    Args:
        config (HierarchicalBayesConfig): Validated workflow configuration.
        muscle (str): Muscle name for the requested fit.
        metric (str): Metric name for the requested fit.

    Returns:
        BayesFitConfig: Matching fit configuration.
    """

    fit_pair = (str(muscle), str(metric))
    if fit_pair not in config.fits_by_pair:
        raise KeyError(f"Config does not define fit {fit_pair}.")
    return config.fits_by_pair[fit_pair]


def validate_config_muscles(
    config: HierarchicalBayesConfig,
    valid_muscles: set[str],
) -> None:
    """Require that every configured muscle exists in the current dataset.

    Args:
        config (HierarchicalBayesConfig): Validated workflow configuration.
        valid_muscles (set[str]): Muscle names available in the current dataset.

    Returns:
        None
    """

    configured_muscles = sorted({fit_config.muscle for fit_config in config.fits_by_id.values()})
    unknown_muscles = [muscle for muscle in configured_muscles if muscle not in valid_muscles]
    if unknown_muscles:
        raise ValueError(
            "Config section 'fits' contains muscles that are not present in the measurements data: "
            f"{unknown_muscles}."
        )
