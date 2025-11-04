"""Helpers for loading legacy .keras archives produced by TensorFlow Keras 2.x."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Union

import tensorflow as tf

PathLike = Union[str, os.PathLike[str]]


def _patch_config(obj: Any) -> None:
    """Recursively fix fields that are incompatible with modern Keras."""
    if isinstance(obj, dict):
        config = obj.get("config")
        if isinstance(config, dict):
            if "batch_shape" in config and "batch_input_shape" not in config:
                config["batch_input_shape"] = config.pop("batch_shape")
            dtype = config.get("dtype")
            if isinstance(dtype, dict):
                if dtype.get("class_name") == "DTypePolicy":
                    config["dtype"] = dtype.get("config", {}).get("name", "float32")
        for value in obj.values():
            _patch_config(value)
    elif isinstance(obj, list):
        for item in obj:
            _patch_config(item)


def load_legacy_keras_model(path: PathLike) -> tf.keras.Model:
    """Load a `.keras` model that still stores `batch_shape` in its config.

    TensorFlow 2.15 + Keras 3 removed support for the legacy `batch_shape`
    attribute that was serialized by tf.keras 2.x.  The models checked into the
    repository were saved before that change, so attempting to load them with the
    stock API raises `ValueError: Unrecognized keyword arguments: ['batch_shape']`.

    This helper extracts the archive to a temporary directory, rewrites the
    stored `config.json` so that each layer uses the modern `batch_input_shape`
    field, normalises dtype declarations, and strips the embedded compile
    configuration (which references deprecated optimizer kwargs).  The patched
    archive is then re-zipped and loaded with `tf.keras.models.load_model`.
    """

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        with zipfile.ZipFile(model_path) as archive:
            archive.extractall(tmpdir_path)

        config_path = tmpdir_path / "config.json"
        config = json.loads(config_path.read_text())
        _patch_config(config)
        # Avoid attempting to deserialize the legacy optimizer configuration.
        config["compile_config"] = None
        config_path.write_text(json.dumps(config))

        patched_archive = tmpdir_path / "patched.keras"
        with zipfile.ZipFile(patched_archive, "w") as archive:
            for item in tmpdir_path.iterdir():
                if item.name == patched_archive.name:
                    continue
                archive.write(item, item.name)

        model = tf.keras.models.load_model(patched_archive)

    return model


__all__ = ["load_legacy_keras_model"]