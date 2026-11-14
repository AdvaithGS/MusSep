"""YAML configuration helpers shared by the training and inference entry points."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class _ConfigLoader(yaml.SafeLoader):
    """SafeLoader that also understands ``!!python/tuple``.

    Community BS-RoFormer configs (ZFTurbo's Music-Source-Separation-Training)
    tag ``freqs_per_bands`` with ``!!python/tuple``, which ``yaml.safe_load``
    rejects. Building a tuple is harmless, so it is allowed here; every other
    Python-specific tag is still refused.
    """


def _construct_tuple(loader: yaml.SafeLoader, node: yaml.Node) -> tuple:
    return tuple(loader.construct_sequence(node, deep=True))


_ConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_tuple)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=_ConfigLoader)  # noqa: S506 - restricted loader
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(data).__name__}")
    return data


def resolve_tier(config: Dict[str, Any], tier: Optional[str] = None) -> Dict[str, Any]:
    """Merge ``config['model']`` with the selected entry of ``config['tiers']``.

    The tier name comes from ``tier`` (CLI override) or ``config['model']['tier']``.
    Keys of the tier override keys of ``model``. ``type`` and ``tier`` are removed
    from the result so that it can be passed straight to a model constructor.
    """
    model_cfg = copy.deepcopy(config.get("model", {}))
    tier_name = tier or model_cfg.get("tier")
    tiers = config.get("tiers", {})
    if tier_name is not None:
        if tier_name not in tiers:
            raise KeyError(f"unknown tier '{tier_name}'; available: {sorted(tiers)}")
        model_cfg.update(copy.deepcopy(tiers[tier_name]))
    model_cfg.pop("tier", None)
    model_cfg.pop("type", None)
    return model_cfg


def tier_name(config: Dict[str, Any], tier: Optional[str] = None) -> Optional[str]:
    return tier or config.get("model", {}).get("tier")


def get(config: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """``get(cfg, 'training.batch_size', 4)`` with graceful fallback."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return default if node is None else node
