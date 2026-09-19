"""Configuration loading and path resolution.

Values are resolved with this precedence, highest first:

1. explicit CLI flags
2. ``SPARSEST_*`` environment variables (plus the legacy ``IPAD_*`` aliases)
3. ``configs/local.yaml``, if present -- machine-specific, not committed
4. the dataset config passed with ``--config``

Data and output locations are always expressed as a root plus a relative path, so a
different machine only needs to override the root.
"""

import copy
import os
from pathlib import Path

import yaml

# Environment variable -> dotted config key.
ENV_OVERRIDES = {
    "SPARSEST_DATA_ROOT": "data.root",
    "SPARSEST_OUTPUT_ROOT": "output.root",
    "SPARSEST_TEST_NPY": "data.test",
    "SPARSEST_CHECKPOINT": "eval.checkpoint",
}

# Kept so existing sweep scripts keep working unchanged.
LEGACY_ENV_OVERRIDES = {
    "IPAD_TEST_NPY": "data.test",
    "IPAD_TEST_CHECKPOINT": "eval.checkpoint",
}

REPO_ROOT = Path(__file__).resolve().parents[2]


class Config:
    """Attribute and item access over a nested config dict."""

    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        # Private and dunder names must never be looked up in the config. Pickle probes
        # for __setstate__ and friends before __init__ has run, and resolving those
        # through self._data would recurse forever -- which is exactly what happens when
        # a Config is handed to torch.multiprocessing.spawn.
        if name.startswith("_"):
            raise AttributeError(name)

        data = object.__getattribute__(self, "_data")
        try:
            value = data[name]
        except KeyError as exc:
            raise AttributeError(
                f"no config key {name!r} (available: {sorted(data)})"
            ) from exc
        return Config(value) if isinstance(value, dict) else value

    def __getstate__(self):
        return {"data": object.__getattribute__(self, "_data")}

    def __setstate__(self, state):
        object.__setattr__(self, "_data", state["data"])

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return name in self._data

    def get(self, name, default=None):
        value = self._data.get(name, default)
        return Config(value) if isinstance(value, dict) else value

    def to_dict(self):
        return copy.deepcopy(self._data)

    # -- path helpers ----------------------------------------------------------

    def data_path(self, key):
        """Resolve ``data.<key>`` against ``data.root``."""
        return _join(self._data["data"]["root"], self._data["data"][key])

    def output_path(self, key, *parts, create=True):
        """Resolve ``output.<key>`` against ``output.root``.

        Creates the directory by default, since most callers are about to write. Pass
        ``create=False`` when only reading, so a lookup does not leave empty directories
        behind.
        """
        base = _join(self._data["output"]["root"], self._data["output"][key])
        path = base.joinpath(*parts) if parts else base
        if create:
            target = path if not path.suffix else path.parent
            target.mkdir(parents=True, exist_ok=True)
        return path

    def __repr__(self):
        return f"Config({self._data!r})"


def _join(root, value):
    """Join ``value`` onto ``root`` unless it is already absolute."""
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    root_path = Path(str(root)).expanduser()
    if not root_path.is_absolute():
        root_path = (REPO_ROOT / root_path).resolve()
    return root_path / path


def _deep_merge(base, override):
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_dotted(data, dotted, value):
    keys = dotted.split(".")
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def _apply_env(data):
    """Apply ``SPARSEST_*`` overrides, then the legacy ``IPAD_*`` aliases."""
    for env_name, dotted in ENV_OVERRIDES.items():
        if os.environ.get(env_name):
            _set_dotted(data, dotted, os.environ[env_name])

    for env_name, dotted in LEGACY_ENV_OVERRIDES.items():
        if os.environ.get(env_name) and not os.environ.get(
            _preferred_env_for(dotted), ""
        ):
            _set_dotted(data, dotted, os.environ[env_name])


def _preferred_env_for(dotted):
    for env_name, target in ENV_OVERRIDES.items():
        if target == dotted:
            return env_name
    return ""


def load_config(config_path, overrides=None):
    """Load a dataset config, layering local overrides, env vars and CLI values.

    Args:
        config_path: path to a YAML config, e.g. ``configs/ipad.yaml``.
        overrides: optional dotted-key mapping applied last, from the CLI.

    Returns:
        A :class:`Config`.
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        candidate = REPO_ROOT / config_path
        if candidate.exists():
            config_path = candidate
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    with open(config_path) as handle:
        data = yaml.safe_load(handle) or {}

    # A shared local.yaml applies to every dataset; local.<name>.yaml refines one of
    # them and wins, which is what lets the two datasets sit under different roots.
    for local_path in (
        config_path.parent / "local.yaml",
        config_path.parent / f"local.{config_path.stem}.yaml",
    ):
        if local_path.exists():
            with open(local_path) as handle:
                data = _deep_merge(data, yaml.safe_load(handle) or {})

    _apply_env(data)

    for dotted, value in (overrides or {}).items():
        if value is not None:
            _set_dotted(data, dotted, value)

    return Config(data)
