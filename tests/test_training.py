"""Training closure: two steps, a checkpoint, and a scored split."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from tearfilm_persistence.cli.evaluate import main as evaluate_main
from tearfilm_persistence.cli.train import main as train_main
from tearfilm_persistence.models.teardyn import ModelConfig, TearDynModel
from tearfilm_persistence.runtime.config import load_experiment
from tearfilm_persistence.runtime.loop import TrainConfig, score_loader, train_model
from tearfilm_persistence.runtime.seeding import set_seed
from tearfilm_persistence.streams.acquisition import save_clip, stack_clips
from tearfilm_persistence.verification.probe_inputs import probe_clip as make_clip

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "experiments" / "_smoke.yaml"


class _Clips(torch.utils.data.Dataset):
    def __init__(self, clips):
        self.clips = clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        return self.clips[index]


def _tiny_loader() -> DataLoader:
    clips = [
        make_clip(
            seed=index, n_frames=10, eye_id=f"e{index}", reference_onset=4.0 if index < 2 else None
        )
        for index in range(4)
    ]
    return DataLoader(_Clips(clips), batch_size=2, collate_fn=stack_clips)


def test_two_steps_reduce_the_loss(tmp_path) -> None:
    set_seed(3)
    model = TearDynModel(ModelConfig())
    history = train_model(
        model,
        _tiny_loader(),
        TrainConfig(epochs=2, batch_size=2, warmup_steps=1, precision="fp32", learning_rate=3e-3),
        device=torch.device("cpu"),
        seed=3,
        checkpoint_path=tmp_path / "model.pt",
    )
    assert history.epochs[-1].total < history.epochs[0].total
    assert Path(history.checkpoint).exists()


def test_every_loss_term_is_finite(tmp_path) -> None:
    set_seed(4)
    model = TearDynModel(ModelConfig())
    history = train_model(
        model,
        _tiny_loader(),
        TrainConfig(epochs=1, batch_size=2, warmup_steps=1, precision="fp32"),
        device=torch.device("cpu"),
        seed=4,
        checkpoint_path=tmp_path / "model.pt",
    )
    record = history.epochs[0]
    assert all(torch.isfinite(torch.tensor([record.decision, record.onset, record.repeatability])))


def test_accumulated_step_matches_one_step(tmp_path) -> None:
    set_seed(5)
    model = TearDynModel(ModelConfig())
    loader = _tiny_loader()
    history = train_model(
        model,
        loader,
        TrainConfig(
            epochs=1,
            batch_size=2,
            grad_accumulation=2,
            warmup_steps=1,
            precision="fp32",
            learning_rate=1e-3,
        ),
        device=torch.device("cpu"),
        seed=5,
        checkpoint_path=tmp_path / "model.pt",
    )
    assert history.epochs[0].steps == 1


def test_score_loader_returns_one_row_per_clip() -> None:
    set_seed(6)
    model = TearDynModel(ModelConfig()).eval()
    rows = score_loader(
        model, _loader_of_clips([make_clip(seed=0), make_clip(seed=1)]), torch.device("cpu")
    )
    assert len(rows) == 2
    assert rows[0].patient_id == "patient-eye-0"


def _loader_of_clips(clips) -> DataLoader:
    return DataLoader(_Clips(clips), batch_size=2, collate_fn=stack_clips)


def _write_cohort(tmp_path: Path) -> Path:
    manifest = tmp_path / "cohort.jsonl"
    rows = []
    for index in range(6):
        eye = f"e{index}"
        clip = make_clip(
            seed=index, n_frames=10, eye_id=eye, reference_onset=5.0 if index % 2 == 0 else None
        )
        clip_path = tmp_path / f"{eye}.npz"
        save_clip(clip, clip_path)
        rows.append(
            {
                "patient_id": f"p{index}",
                "eye_id": eye,
                "site": "Site A",
                "region": "Region I",
                "session": "s1",
                "repeat": 1,
                "arm": "retrospective",
                "split": "train" if index < 4 else "test",
                "path": str(clip_path),
                "reference_onset": 5.0 if index % 2 == 0 else None,
            }
        )
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return manifest


def test_cli_train_writes_a_checkpoint_and_history(tmp_path) -> None:
    manifest = _write_cohort(tmp_path)
    exit_code = train_main(
        [
            "--experiment",
            str(SMOKE),
            "--cohort",
            str(manifest),
            "--output-dir",
            str(tmp_path / "run"),
            "--seed",
            "0",
            "--device",
            "cpu",
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "run" / "model.pt").exists()
    assert (tmp_path / "run" / "training_history.json").exists()


def test_cli_evaluate_writes_a_report(tmp_path) -> None:
    manifest = _write_cohort(tmp_path)
    exit_code = train_main(
        [
            "--experiment",
            str(SMOKE),
            "--cohort",
            str(manifest),
            "--output-dir",
            str(tmp_path / "run"),
            "--seed",
            "0",
        ]
    )
    assert exit_code == 0
    exit_code = evaluate_main(
        [
            "--experiment",
            str(SMOKE),
            "--cohort",
            str(manifest),
            "--checkpoint",
            str(tmp_path / "run" / "model.pt"),
            "--output-dir",
            str(tmp_path / "run"),
            "--split",
            "test",
        ]
    )
    assert exit_code == 0
    report = json.loads((tmp_path / "run" / "evaluation.json").read_text(encoding="utf-8"))
    assert report["split"] == "test"
    assert report["n_eyes"] == 2
    assert "auc" in report["arm"]


def test_evaluate_cli_refuses_a_split_the_table_does_not_carry(tmp_path) -> None:
    manifest = _write_cohort(tmp_path)
    train_main(
        [
            "--experiment",
            str(SMOKE),
            "--cohort",
            str(manifest),
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    with pytest.raises(ValueError):
        evaluate_main(
            [
                "--experiment",
                str(SMOKE),
                "--cohort",
                str(manifest),
                "--checkpoint",
                str(tmp_path / "run" / "model.pt"),
                "--output-dir",
                str(tmp_path / "run"),
                "--split",
                "prospective",
            ]
        )


def test_smoke_config_carries_the_test_warning() -> None:
    text = SMOKE.read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("# for unit-test smoke only")


def test_experiment_seed_list_matches_the_reported_initialisations() -> None:
    payload = load_experiment(ROOT / "configs" / "experiments" / "main.yaml")
    assert len(payload["analysis"]["seeds"]) == 5
