import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import SRC_DIR

CLASS_NAMES = ["class0", "class1", "class2"]


def write_config(config: dict, path) -> str:
    with open(path, "w") as f:
        json.dump(config, f)
    return str(path)


def run_optieeg_subprocess(*args, timeout=120):
    return subprocess.run(
        [sys.executable, "run_optieeg.py", *args],
        cwd=str(SRC_DIR),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def create_synthetic_data(tmp_path):
    rng = np.random.default_rng(42)
    n_epochs, n_channels, n_timestamps = 60, 4, 100

    data = rng.standard_normal((n_epochs, n_channels, n_timestamps)).astype(np.float32)
    labels = np.array([CLASS_NAMES[i % len(CLASS_NAMES)] for i in range(n_epochs)])

    data_path = tmp_path / "data.npy"
    labels_path = tmp_path / "labels.npy"
    np.save(data_path, data)
    np.save(labels_path, labels)

    return str(data_path), str(labels_path), n_channels


def create_config(data_path, labels_path, n_channels, checkpoint_dir):
    return {
        "project_name": "test-project",
        "path": {
            "checkpoint_path": str(checkpoint_dir),
            "data_path": data_path,
            "labels_path": labels_path,
            "test_data_path": "",
            "test_labels_path": "",
            "fixed_config_path": "",
        },
        "data": {
            "sample_rate": 200,
            "n_classes": len(CLASS_NAMES),
            "class_names": CLASS_NAMES,
            "n_channels": n_channels,
            "channel_names": [f"ch{i}" for i in range(n_channels)],
            "omit_channels": [],
            "start_sample": 0,
            "end_sample": -1,
            "highpass": None,
            "lowpass": None,
        },
        "train": {
            "max_n_epochs": 3,
            "n_cv": 2,
            "loss_func_name": "CrossEntropyLoss",
            "eval_metric_name": "macro-f1",
            "eval_increases": True,
            "patience": 2,
            "min_delta": 0.001,
            "overfit_thresh": 0.25,
            "overfit_epoch": 1,
            "data_seed": 42,
            "model_seed": 42,
        },
    }


# Scope=class means runs once for the whole TestFixedParamRun class
@pytest.fixture(scope="class")
def fixed_param_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("fixed_param_run")

    data_path, labels_path, n_channels = create_synthetic_data(tmp_path)

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    config = create_config(data_path, labels_path, n_channels, checkpoint_dir)

    config["path"]["fixed_config_path"] = str(Path(__file__).parent / "fixed_params_eegnet.json")
    config_path = write_config(config, tmp_path / "config.json")

    result = run_optieeg_subprocess(config_path, "--noopt", "--nowandb")

    return {
        "result": result,
        "checkpoint_dir": checkpoint_dir,
        "n_cv": config["train"]["n_cv"],
    }


class TestFixedParamRun:

    def test_completes_successfully(self, fixed_param_run):
        """Subprocess exited with success."""
        result = fixed_param_run["result"]
        assert result.returncode == 0, (
            f"run_optieeg.py failed (exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout[-2000:]}\n"
            f"--- stderr ---\n{result.stderr[-2000:]}"
        )

    def test_checkpoints_created(self, fixed_param_run):
        """A checkpoint file saved for each CV fold."""
        checkpoints = list(fixed_param_run["checkpoint_dir"].rglob("*.ckpt"))
        assert len(checkpoints) == fixed_param_run["n_cv"], (
            f"Expected {fixed_param_run['n_cv']} checkpoints, found {len(checkpoints)}: {checkpoints}"
        )

    def test_checkpoints_loadable(self, fixed_param_run):
        """Saved checkpoints loadable and contain expected keys."""
        for ckpt in fixed_param_run["checkpoint_dir"].rglob("*.ckpt"):
            checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint
            assert "hyper_parameters" in checkpoint
            assert "random_states" in checkpoint

    def test_output_logs_metrics(self, fixed_param_run):
        """Stdout/stderr contain average fold metrics."""
        combined = fixed_param_run["result"].stdout + fixed_param_run["result"].stderr
        assert "Average metrics" in combined, (
            f"Expected 'Average metrics' in output, got:\n{combined[-2000:]}"
        )
        assert "Fixed parameter run completed" in combined
