"""Configuration, factory routing and checkpointing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tearfilm_persistence.comparators.adapter import ComparatorAdapter
from tearfilm_persistence.models.teardyn import ModelConfig, TearDynModel
from tearfilm_persistence.runtime.checkpoint import (
    FORMAT_VERSION,
    load_checkpoint,
    save_checkpoint,
    sha256,
)
from tearfilm_persistence.runtime.config import deep_merge, flatten, load_experiment, read_yaml
from tearfilm_persistence.runtime.factory import build_model, model_config, train_config
from tearfilm_persistence.runtime.seeding import set_seed
from tearfilm_persistence.streams.acquisition import stack_clips
from tearfilm_persistence.verification.probe_inputs import probe_clip as make_clip

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "configs" / "experiments"


def test_main_experiment_resolves_every_section() -> None:
    payload = load_experiment(EXPERIMENTS / "main.yaml")
    assert payload["model"]["fast_branch"] == "td"
    assert payload["acquisition"]["adequacy"]["sampling_floor_hz"] == 30.0
    assert payload["optimisation"]["epochs"] == 40


def test_float_yaml_values_are_not_strings() -> None:
    payload = load_experiment(EXPERIMENTS / "main.yaml")
    assert isinstance(payload["acquisition"]["adequacy"]["sampling_floor_hz"], float)


def test_ablation_files_toggle_one_component() -> None:
    baseline = load_experiment(EXPERIMENTS / "main.yaml")
    ablation = load_experiment(EXPERIMENTS / "ablation_minus_fast.yaml")
    assert baseline["model"]["use_fast"] is True
    assert ablation["model"]["use_fast"] is False


def test_deep_merge_is_recursive() -> None:
    merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}})
    assert merged == {"a": {"b": 1, "c": 3}}


def test_read_yaml_rejects_unknown_sections(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("nonsense: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_yaml(path)


def test_flatten_names_nested_keys() -> None:
    flat = flatten({"model": {"head": {"hidden": 64}}})
    assert flat["model.head.hidden"] == 64


def test_model_config_reads_the_paper_window() -> None:
    config = model_config(load_experiment(EXPERIMENTS / "main.yaml"))
    assert config.persistence.tau_seconds == pytest.approx(0.17)
    assert config.derivative.window == 5
    assert config.area.rows * config.area.cols == 64


def test_train_config_reads_the_release_defaults() -> None:
    config = train_config(load_experiment(EXPERIMENTS / "main.yaml"))
    assert config.batch_size == 8
    assert config.epochs == 40
    assert config.weights.repeatability > 0


def test_factory_builds_the_default_model() -> None:
    model = build_model(load_experiment(EXPERIMENTS / "main.yaml"))
    assert isinstance(model, TearDynModel)


def test_factory_routes_baseline_files_to_comparators() -> None:
    model = build_model(load_experiment(EXPERIMENTS / "baseline_siamese_pair.yaml"))
    assert isinstance(model, ComparatorAdapter)
    assert model.name == "siamese_pair"


def test_factory_rejects_an_unknown_comparator() -> None:
    with pytest.raises(KeyError):
        build_model({"model": {"comparator": "not_a_model"}})


def test_smoke_config_stays_inside_the_locus_range() -> None:
    payload = load_experiment(EXPERIMENTS / "_smoke.yaml")
    config = model_config(payload)
    assert 25 <= config.area.rows * config.area.cols <= 100


def test_set_seed_is_reproducible() -> None:
    set_seed(11)
    first = torch.rand(3)
    set_seed(11)
    second = torch.rand(3)
    assert torch.equal(first, second)


def test_set_seed_rejects_a_negative_value() -> None:
    with pytest.raises(ValueError):
        set_seed(-1)


def test_checkpoint_roundtrip(tmp_path, batch) -> None:
    model = TearDynModel(ModelConfig()).eval()
    with torch.no_grad():
        before = model(batch).logit.clone()
    path = tmp_path / "model.pt"
    state = save_checkpoint(
        path, model, None, epoch=3, seed=7, config={"a": 1}, metrics={"loss": 0.5}
    )
    assert state.digest == sha256(path)
    restored = TearDynModel(ModelConfig()).eval()
    loaded = load_checkpoint(path, restored)
    with torch.no_grad():
        after = restored(batch).logit
    assert torch.allclose(before, after, atol=1e-6)
    assert loaded.epoch == 3
    assert loaded.seed == 7
    assert loaded.metrics["loss"] == pytest.approx(0.5)


def test_checkpoint_write_leaves_no_temporary_file(tmp_path) -> None:
    model = TearDynModel(ModelConfig())
    save_checkpoint(tmp_path / "model.pt", model, None, epoch=1, seed=1)
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_refuses_an_unknown_format(tmp_path) -> None:
    path = tmp_path / "old.pt"
    torch.save({"format_version": FORMAT_VERSION + 1, "model": {}, "epoch": 0, "seed": 0}, path)
    with pytest.raises(ValueError):
        load_checkpoint(path, TearDynModel(ModelConfig()))


def test_checkpoint_metadata_is_json_serialisable(tmp_path) -> None:
    model = TearDynModel(ModelConfig())
    path = tmp_path / "model.pt"
    state = save_checkpoint(
        path, model, None, epoch=2, seed=4, config={"model": {"fast_branch": "td"}}
    )
    json.dumps(state.config)


def test_single_clip_inference_returns_a_descriptor() -> None:
    model = TearDynModel(ModelConfig()).eval()
    result = model.infer(make_clip())
    assert len(result.descriptor.as_dict()) == 6
    assert 0.0 <= result.probability <= 1.0


def test_single_clip_inference_batches_two_clips() -> None:
    model = TearDynModel(ModelConfig()).eval()
    stacked = stack_clips([make_clip(seed=0), make_clip(seed=1, eye_id="other")])
    with torch.no_grad():
        output = model(stacked)
    assert output.logit.shape == (2,)
