"""Strict DMD_1X configuration parsing for Bayesian workflows."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DMD_DIR = REPO_ROOT / "dmd"
DEFAULT_HIERARCHICAL_BAYES_CONFIG_PATH = (
    Path(__file__).resolve().parent / "hierarchical_bayes_config.yaml"
)
ALLOWED_ANALYSIS_IDS = ("instance", "image_summary")


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


dmd_config = load_module_from_path("dmd_hierarchical_bayes_config_for_dmd_1x", DMD_DIR / "hierarchical_bayes_config.py")

DmdBayesPathConfig = dmd_config.DmdBayesPathConfig
DmdBayesRuntimeConfig = dmd_config.DmdBayesRuntimeConfig
DmdBayesFilterConfig = dmd_config.DmdBayesFilterConfig
DmdBayesFitConfig = dmd_config.DmdBayesFitConfig
DmdBayesAnalysisConfig = dmd_config.DmdBayesAnalysisConfig
DmdHierarchicalBayesConfig = dmd_config.DmdHierarchicalBayesConfig


def load_hierarchical_bayes_config(
    path: Path,
    positive_metrics: tuple[str, ...],
    bounded_metrics: tuple[str, ...],
    image_summary_positive_metrics: tuple[str, ...],
    image_summary_bounded_metrics: tuple[str, ...],
    image_summary_real_metrics: tuple[str, ...],
) -> Any:
    """Load and validate a DMD_1X Bayes YAML config.

    Args:
        path (Path): YAML config path.
        positive_metrics (tuple[str, ...]): Positive instance-level metrics.
        bounded_metrics (tuple[str, ...]): Bounded instance-level metrics.
        image_summary_positive_metrics (tuple[str, ...]): Positive image-level metrics.
        image_summary_bounded_metrics (tuple[str, ...]): Bounded image-level metrics.
        image_summary_real_metrics (tuple[str, ...]): Real-valued image-level metrics.

    Returns:
        Any: Normalized DMD-compatible hierarchical Bayes config.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    config_mapping = dmd_config.require_mapping("root", raw_config)
    dmd_config.validate_section_keys(
        section_name="root",
        section=config_mapping,
        required_keys={"paths", "runtime", "available_options"},
        optional_keys={"enabled", "analysis_id", "filters", "fits", "fit_grid"},
    )
    if "image_summary" in config_mapping:
        raise ValueError(
            "Nested 'image_summary' sections are no longer supported in this file. "
            "Use dmd_1x/hierarchical_bayes_image_summary_config.yaml for image-summary analysis."
        )
    analysis_id = dmd_config.require_allowed_string(
        "root",
        "analysis_id",
        config_mapping.get("analysis_id", "instance"),
        ALLOWED_ANALYSIS_IDS,
    )
    if analysis_id == "image_summary":
        root_positive_metrics = image_summary_positive_metrics
        root_bounded_metrics = image_summary_bounded_metrics
        root_real_metrics = image_summary_real_metrics
    else:
        root_positive_metrics = positive_metrics
        root_bounded_metrics = bounded_metrics
        root_real_metrics = ()
    analysis = dmd_config.parse_analysis_config(
        analysis_id=analysis_id,
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
    analyses_by_id = {analysis.analysis_id: analysis}
    return DmdHierarchicalBayesConfig(
        config_path=Path(path),
        paths=analysis.paths,
        runtime=analysis.runtime,
        filters=analysis.filters,
        fit_order=analysis.fit_order,
        fits_by_id=analysis.fits_by_id,
        fits_by_key=analysis.fits_by_key,
        enabled=analysis.enabled,
        analyses_by_id=analyses_by_id,
    )


def repeated_fit_configs(config: Any) -> list[Any]:
    """Return DMD_1X fit entries whose ``repeat`` flag is enabled.

    Args:
        config (Any): Analysis configuration.

    Returns:
        list[Any]: Fit configurations selected for execution.
    """

    return dmd_config.repeated_fit_configs(config)


def fit_config_for_key(
    config: Any,
    muscle: str,
    compartment: str,
    metric: str,
) -> Any:
    """Return the config entry for one DMD_1X muscle/compartment/metric fit.

    Args:
        config (Any): Analysis configuration.
        muscle (str): Muscle label.
        compartment (str): Compartment label.
        metric (str): Metric column name.

    Returns:
        Any: Matching fit configuration.
    """

    return dmd_config.fit_config_for_key(
        config=config,
        muscle=muscle,
        compartment=compartment,
        metric=metric,
    )


def validate_config_groups(
    config: Any,
    valid_muscles: set[str],
    valid_compartments: set[str],
) -> None:
    """Require configured DMD_1X muscles and compartments to exist in the dataset.

    Args:
        config (Any): Analysis configuration.
        valid_muscles (set[str]): Muscle labels present in the data.
        valid_compartments (set[str]): Compartment labels present in the data.

    Returns:
        None: Raises ``ValueError`` when any configured group is absent.
    """

    dmd_config.validate_config_groups(
        config=config,
        valid_muscles=valid_muscles,
        valid_compartments=valid_compartments,
    )
