"""Layer-two checks: the released code actually runs.

Each check exercises one mechanism (an operator, a loss, a metric, a loop) and
records what it measured. A check that cannot be exercised on the released
material is reported ``NOT_RUN`` with the reason; a check whose measured value
falls outside its tolerance is ``FAIL``.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tearfilm_persistence.cohort.manifest import ClipRecord, CohortTable, ensure_patient_level
from tearfilm_persistence.deformation.curve import locus_entropy, ols_slope
from tearfilm_persistence.deformation.derivative import TemporalDerivative
from tearfilm_persistence.deformation.persistence import (
    area_fraction,
    first_passage,
    hard_occupancy,
    soft_occupancy,
)
from tearfilm_persistence.descriptors.calibration import (
    LogisticCalibrator,
    VarianceComponentCalibrator,
)
from tearfilm_persistence.descriptors.repeatability import (
    bland_altman,
    coefficient_of_repeatability,
    icc_one_way,
    summarise_repeats,
    variance_shares,
    within_subject_sd,
)
from tearfilm_persistence.descriptors.vector import COMPONENTS
from tearfilm_persistence.evaluation.ablation import interaction_ratio
from tearfilm_persistence.evaluation.aggregate import ScoredEye, summarise_arm
from tearfilm_persistence.evaluation.agreement import weighted_kappa
from tearfilm_persistence.evaluation.budget import count_flops, profile_checks
from tearfilm_persistence.evaluation.criteria import evaluate
from tearfilm_persistence.evaluation.discrimination import (
    cluster_bootstrap,
    operating_point,
    roc_auc,
)
from tearfilm_persistence.evaluation.mechanistic import percent_change
from tearfilm_persistence.evaluation.multiplicity import holm_stepdown
from tearfilm_persistence.evaluation.properties import (
    cross_site_change,
    expected_calibration_error,
    false_early_rate,
)
from tearfilm_persistence.evaluation.reader_study import orc_difference, reliance_summary
from tearfilm_persistence.models.teardyn import ModelConfig, TearDynModel
from tearfilm_persistence.objectives.composite import (
    decision_term,
    onset_term,
    repeatability_term,
)
from tearfilm_persistence.runtime.checkpoint import load_checkpoint, save_checkpoint
from tearfilm_persistence.runtime.config import load_experiment
from tearfilm_persistence.runtime.loop import TrainConfig, train_model
from tearfilm_persistence.runtime.seeding import set_seed
from tearfilm_persistence.streams.acquisition import AcquisitionClip, stack_clips
from tearfilm_persistence.streams.adequacy import AdequacyConfig, assess
from tearfilm_persistence.verification.probe_inputs import (
    alternating_derivative,
    flipping_derivative,
    partial_mask,
    probe_batch,
    probe_clip,
    ramp_derivative,
    step_derivative,
    uniform_mask,
)

Status = Literal["PASS", "FAIL", "NOT_RUN", "BLOCKED"]
Check = Callable[[], tuple[Status, str, float | None]]


@dataclass(frozen=True)
class ExecutionCheck:
    id: str
    name: str
    statement: str
    status: Status
    detail: str
    quantity: float | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ok(
    condition: bool, detail: str, quantity: float | None = None
) -> tuple[Status, str, float | None]:
    return ("PASS" if condition else "FAIL", detail, quantity)


def check_import_surface() -> tuple[Status, str, float | None]:
    modules = [
        "tearfilm_persistence.streams",
        "tearfilm_persistence.deformation",
        "tearfilm_persistence.structure",
        "tearfilm_persistence.descriptors",
        "tearfilm_persistence.models",
        "tearfilm_persistence.comparators",
        "tearfilm_persistence.objectives",
        "tearfilm_persistence.cohort",
        "tearfilm_persistence.evaluation",
        "tearfilm_persistence.runtime",
        "tearfilm_persistence.verification",
    ]
    import importlib

    for name in modules:
        importlib.import_module(name)
    return _ok(True, f"{len(modules)} public packages import", float(len(modules)))


def check_config_loads() -> tuple[Status, str, float | None]:
    path = repo_root() / "configs" / "experiments" / "main.yaml"
    payload = load_experiment(path)
    required = {"model", "acquisition", "optimisation"}
    present = required & set(payload)
    return _ok(present == required, f"sections present: {sorted(present)}", float(len(present)))


def check_derivative_on_ramp() -> tuple[Status, str, float | None]:
    steps = 12
    field = torch.arange(steps, dtype=torch.float32).view(1, 1, steps) * 0.5
    derivative = TemporalDerivative()(field, 30.0)
    interior = float(derivative[0, 0, 5:].mean().item())
    expected = 0.5 * 30.0
    return _ok(
        abs(interior - expected) < 1e-3,
        f"interior mean {interior:.4f} against {expected}",
        interior,
    )


def check_derivative_window() -> tuple[Status, str, float | None]:
    module = TemporalDerivative()
    taps = int(module.kernel.numel())
    return _ok(taps <= 5, f"{taps} taps", float(taps))


def check_persistence_maturity() -> tuple[Status, str, float | None]:
    field = step_derivative((1, 4, 12), start=0)
    occupancy = hard_occupancy(field, span=3)
    before = bool(occupancy[0, 0, 2].item())
    at = bool(occupancy[0, 0, 3].item())
    return _ok((not before) and at, f"t=2 occupied={before}, t=3 occupied={at}", 3.0)


def check_persistence_sign_flip() -> tuple[Status, str, float | None]:
    field = flipping_derivative((1, 4, 12), flip_at=6)
    occupancy = hard_occupancy(field, span=3)
    kept_before = bool(occupancy[0, 0, 5].item())
    dropped_after = bool(occupancy[0, 0, 6].item())
    return _ok(
        kept_before and not dropped_after,
        f"flip frame occupied={dropped_after}",
        0.0 if dropped_after else 1.0,
    )


def check_persistence_alternating() -> tuple[Status, str, float | None]:
    field = alternating_derivative((1, 4, 16), period=1)
    occupancy = hard_occupancy(field, span=2)
    any_occupied = bool(occupancy.any().item())
    return _ok(not any_occupied, f"alternating field occupied={any_occupied}", 0.0)


def check_soft_hard_agreement() -> tuple[Status, str, float | None]:
    field = step_derivative((1, 8, 12), start=0)
    hard = hard_occupancy(field, span=3).float()
    soft = soft_occupancy(field, span=3, gamma=20.0)
    gap = float((hard[:, :, 4:] - soft[:, :, 4:]).abs().max().item())
    return _ok(gap < 0.05, f"maximum gap after maturity {gap:.4f}", gap)


def check_first_passage() -> tuple[Status, str, float | None]:
    curve = torch.tensor([[0.0, 0.02, 0.11, 0.4, 0.5]])
    index, found = first_passage(curve, 0.10)
    return _ok(
        bool(found.item()) and int(index.item()) == 2,
        f"index {int(index.item())}",
        float(index.item()),
    )


def check_area_fraction() -> tuple[Status, str, float | None]:
    occupancy = torch.ones((1, 16, 5))
    mask = partial_mask(16, 10)
    fraction = float(area_fraction(occupancy, mask)[0, 0].item())
    half = occupancy.clone()
    half[:, 10:] = 0.0
    half_fraction = float(area_fraction(half, mask)[0, 0].item())
    return _ok(
        abs(fraction - 1.0) < 1e-6 and abs(half_fraction - 1.0) < 1e-6,
        f"full {fraction:.4f}, masked-out {half_fraction:.4f}",
        fraction,
    )


def check_locus_entropy() -> tuple[Status, str, float | None]:
    mask = uniform_mask(4)
    single = torch.zeros(4)
    single[0] = 1.0
    even = torch.full((4,), 0.5)
    low = locus_entropy(single, mask)
    high = locus_entropy(even, mask)
    return _ok(low < 1e-6 and abs(high - 1.0) < 1e-6, f"single {low:.4f}, even {high:.4f}", high)


def check_ols_slope() -> tuple[Status, str, float | None]:
    curve = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    slope = ols_slope(curve, fps=10.0)
    return _ok(abs(slope - 1.0) < 1e-5, f"slope per second {slope:.4f}", slope)


def check_descriptor_order() -> tuple[Status, str, float | None]:
    expected = ("onset", "peak_instability", "curve_slope", "locus_entropy", "tmh", "tma")
    return _ok(expected == COMPONENTS, f"{COMPONENTS}", float(len(COMPONENTS)))


def check_repeatability_arithmetic() -> tuple[Status, str, float | None]:
    groups = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    summary = summarise_repeats(groups)
    sw = within_subject_sd(groups)
    cor = coefficient_of_repeatability(sw)
    ok = abs(summary["cor"] - cor) < 1e-9 and abs(cor - 2.77) < 1e-9
    return _ok(ok, f"Sw {sw:.4f}, CoR {cor:.4f}", cor)


def check_icc_ceiling() -> tuple[Status, str, float | None]:
    value = icc_one_way([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    return _ok(abs(value - 1.0) < 1e-9, f"ICC {value:.4f}", value)


def check_bland_altman() -> tuple[Status, str, float | None]:
    summary = bland_altman([2.0, 3.0, 4.0], [1.0, 2.0, 3.0])
    return _ok(abs(summary.bias - 1.0) < 1e-9, f"bias {summary.bias:.4f}", summary.bias)


def check_variance_shares() -> tuple[Status, str, float | None]:
    values = [1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1]
    sites = ["A"] * 4 + ["B"] * 4
    sessions = ["s1", "s1", "s2", "s2"] * 2
    eyes = ["e1", "e1", "e2", "e2", "e3", "e3", "e4", "e4"]
    shares = variance_shares(values, sites, sessions, eyes)
    total = sum(shares.values())
    return _ok(abs(total - 1.0) < 1e-9 and shares["site"] > 0, f"shares {shares}", total)


def check_calibrator_shrinkage() -> tuple[Status, str, float | None]:
    matrix = np.asarray(
        [
            [1.0, 0.5, 0.2, 0.1, 0.0, 0.0],
            [1.1, 0.6, 0.3, 0.2, 0.0, 0.1],
            [3.0, 2.5, 2.2, 2.1, 2.0, 2.0],
            [3.1, 2.6, 2.3, 2.2, 2.0, 2.1],
        ],
        dtype=np.float64,
    )
    eyes = ["e1", "e1", "e2", "e2"]
    calibrator = VarianceComponentCalibrator().fit(matrix, eyes, ["A"] * 4)
    calibrated = calibrator.transform(matrix, ["A", "A"])
    raw_distance = float(np.abs(matrix[0] - matrix.mean(axis=0)).sum())
    calibrated_distance = float(np.abs(calibrated[0] - matrix.mean(axis=0)).sum())
    mean_reliability = float(np.mean(calibrator.reliability))
    rho = calibrator.reliability
    detail = (
        f"reliability {mean_reliability:.3f}, "
        f"distance {calibrated_distance:.4f} of {raw_distance:.4f}"
    )
    return _ok(
        all(0.0 <= value <= 1.0 for value in rho)
        and mean_reliability > 0.5
        and calibrated_distance < raw_distance,
        detail,
        mean_reliability,
    )


def check_logistic_calibrator() -> tuple[Status, str, float | None]:
    rng = np.random.default_rng(3)
    logits = rng.normal(size=4000) * 1.5
    truth = (rng.random(4000) < 1.0 / (1.0 + np.exp(-(1.8 * logits - 0.7)))).astype(np.float64)
    fitted = LogisticCalibrator().fit(logits, truth)
    return _ok(
        abs(fitted.slope - 1.8) < 0.25 and abs(fitted.intercept + 0.7) < 0.2,
        f"slope {fitted.slope:.3f}, intercept {fitted.intercept:.3f}",
        fitted.slope,
    )


def _small_model() -> TearDynModel:
    return TearDynModel(ModelConfig())


def check_model_shapes() -> tuple[Status, str, float | None]:
    model = _small_model().eval()
    batch = probe_batch(count=3)
    with torch.no_grad():
        output = model(batch)
    ok = (
        output.logit.shape == (3,)
        and output.descriptor_matrix.shape == (3, len(COMPONENTS))
        and output.probability.shape == (3,)
        and output.abstain.shape == (3,)
    )
    return _ok(
        ok,
        f"logit {tuple(output.logit.shape)}, descriptors {tuple(output.descriptor_matrix.shape)}",
        3.0,
    )


def check_gate_rate_floor() -> tuple[Status, str, float | None]:
    clip = probe_clip(fps=15.0)
    decision = assess(clip, AdequacyConfig())
    return _ok(
        decision.abstained and "sampling_rate_below_floor" in decision.reasons,
        ";".join(decision.reasons),
        15.0,
    )


def check_gate_fixation() -> tuple[Status, str, float | None]:
    clip = probe_clip(fixation_spread=0.25)
    decision = assess(clip, AdequacyConfig())
    return _ok(
        decision.abstained and "fixation_stability_outside_band" in decision.reasons,
        ";".join(decision.reasons),
        decision.fixation_drift,
    )


def check_gate_accepts() -> tuple[Status, str, float | None]:
    clip = probe_clip()
    decision = assess(clip, AdequacyConfig())
    return _ok(
        decision.accepted,
        "accepted" if decision.accepted else ";".join(decision.reasons),
        decision.fixation_drift,
    )


def check_abstention_propagates() -> tuple[Status, str, float | None]:
    good = probe_clip(seed=0, eye_id="a")
    slow = probe_clip(seed=1, eye_id="b", fps=12.0)
    batch = stack_clips([good, slow])
    model = _small_model().eval()
    with torch.no_grad():
        output = model(batch)
    return _ok(
        bool(output.abstain[1].item()) and not bool(output.abstain[0].item()),
        f"abstain {output.abstain.tolist()}",
        float(output.abstain.sum().item()),
    )


def check_comparator_shapes() -> tuple[Status, str, float | None]:
    from tearfilm_persistence.comparators import COMPARATORS, build_comparator

    batch = probe_batch(count=2, n_frames=10, size=32)
    sizes = []
    for name in COMPARATORS:
        comparator = build_comparator(name).eval()
        with torch.no_grad():
            logit, onset = comparator(batch.frames, batch.frame_mask)
        if logit.shape != (2,) or onset.shape != (2,):
            return _ok(False, f"{name} returned {tuple(logit.shape)}", None)
        sizes.append(logit.numel())
    return _ok(len(sizes) == len(COMPARATORS), f"{len(sizes)} comparators", float(len(sizes)))


def check_gradients_reach_components() -> tuple[Status, str, float | None]:
    set_seed(0)
    model = _small_model()
    model.train()
    batch = probe_batch(count=4)
    output = model(batch)
    loss = output.logit.sum() + output.soft_descriptor_matrix.sum()
    loss.backward()
    adapter_module: nn.Module = getattr(model.backbone, "adapter", model.backbone)
    backbones = len(
        [
            parameter
            for parameter in adapter_module.parameters()
            if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        ]
    )
    structural = len(
        [
            parameter
            for parameter in model.structural.parameters()
            if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        ]
    )
    head = len(
        [
            parameter
            for parameter in model.head.parameters()
            if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        ]
    )
    ok = backbones > 0 and structural > 0 and head > 0
    return _ok(
        ok,
        f"adapter {backbones}, structural {structural}, head {head}",
        float(backbones + structural + head),
    )


def check_single_batch_overfit() -> tuple[Status, str, float | None]:
    set_seed(1)
    model = _small_model()
    model.train()
    batch = probe_batch(count=4, n_frames=10, size=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    first = 0.0
    last = 0.0
    for step in range(30):
        output = model(batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output.logit, labels)
        if step == 0:
            first = float(loss.item())
        last = float(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    return _ok(last < 0.7 * first, f"loss {first:.4f} to {last:.4f}", last / max(first, 1e-9))


def check_training_loop() -> tuple[Status, str, float | None]:
    class _Clips(torch.utils.data.Dataset[AcquisitionClip]):
        def __init__(self, clips: list[AcquisitionClip]) -> None:
            self.clips = clips

        def __len__(self) -> int:
            return len(self.clips)

        def __getitem__(self, index: int) -> AcquisitionClip:
            return self.clips[index]

    set_seed(2)
    clips = [
        probe_clip(
            seed=index,
            n_frames=10,
            size=32,
            eye_id=f"e{index}",
            reference_onset=5.0 if index < 2 else None,
        )
        for index in range(4)
    ]
    loader = DataLoader(_Clips(clips), batch_size=2, collate_fn=stack_clips)
    model = _small_model()
    with tempfile.TemporaryDirectory() as directory:
        history = train_model(
            model,
            loader,
            TrainConfig(epochs=2, batch_size=2, warmup_steps=1, precision="fp32"),
            device=torch.device("cpu"),
            seed=2,
            checkpoint_path=Path(directory) / "model.pt",
        )
        improved = history.epochs[-1].total < history.epochs[0].total
        exists = Path(history.checkpoint).exists()
    return _ok(
        improved and exists,
        f"epochs {[round(e.total, 4) for e in history.epochs]}",
        history.epochs[-1].total,
    )


def check_checkpoint_roundtrip() -> tuple[Status, str, float | None]:
    set_seed(3)
    model = _small_model().eval()
    batch = probe_batch(count=2, n_frames=8, size=32)
    with torch.no_grad():
        before = model(batch).logit.clone()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pt"
        save_checkpoint(path, model, None, epoch=1, seed=3)
        restored = _small_model().eval()
        state = load_checkpoint(path, restored)
        with torch.no_grad():
            after = restored(batch).logit
        leftovers = list(Path(directory).glob("*.tmp"))
    gap = float((before - after).abs().max().item())
    return _ok(
        gap < 1e-6 and not leftovers and state.seed == 3, f"maximum logit gap {gap:.2e}", gap
    )


def check_repeatability_penalty() -> tuple[Status, str, float | None]:
    clean = torch.tensor([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]])
    noisy = torch.tensor([[1.0, 1.0], [1.9, 1.9], [2.0, 2.0], [2.9, 2.9]])
    groups = torch.tensor([0, 0, 1, 1])
    clean_value = float(repeatability_term(clean, groups).item())
    noisy_value = float(repeatability_term(noisy, groups).item())
    return _ok(
        clean_value < 1e-6 and noisy_value > 0.2,
        f"clean {clean_value:.4f}, noisy {noisy_value:.4f}",
        noisy_value,
    )


def check_decision_loss() -> tuple[Status, str, float | None]:
    logit = torch.tensor([0.5, -1.5, 2.0])
    label = torch.tensor([1.0, 0.0, 1.0])
    valid = torch.tensor([True, True, False])
    computed = float(decision_term(logit, label, valid).item())
    expected = float(
        torch.nn.functional.binary_cross_entropy_with_logits(logit[:2], label[:2]).item()
    )
    return _ok(abs(computed - expected) < 1e-6, f"{computed:.4f} against {expected:.4f}", computed)


def check_onset_loss_zero() -> tuple[Status, str, float | None]:
    prediction = torch.tensor([2.0, 3.0])
    value = float(onset_term(prediction, prediction.clone(), torch.tensor([True, True])).item())
    return _ok(value < 1e-9, f"loss {value:.2e}", value)


def check_auc_identity() -> tuple[Status, str, float | None]:
    perfect = roc_auc(np.asarray([1.0, 2.0, 3.0, 4.0]), np.asarray([0.0, 0.0, 1.0, 1.0]))
    tied = roc_auc(np.asarray([1.0, 1.0, 1.0, 1.0]), np.asarray([0.0, 0.0, 1.0, 1.0]))
    ok = abs(perfect - 1.0) < 1e-9 and abs(tied - 0.5) < 1e-9
    return _ok(ok, f"perfect {perfect:.4f}, tied {tied:.4f}", perfect)


def check_bootstrap_interval() -> tuple[Status, str, float | None]:
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(1.0, 1.0, 60), rng.normal(0.0, 1.0, 60)])
    labels = np.concatenate([np.ones(60), np.zeros(60)])
    clusters = np.asarray([f"p{index // 2}" for index in range(120)])
    result = cluster_bootstrap(scores, labels, clusters, n_boot=200, seed=0)
    return _ok(
        result.lower <= result.auc <= result.upper,
        f"{result.lower:.3f} to {result.upper:.3f}",
        result.auc,
    )


def check_operating_point_floor() -> tuple[Status, str, float | None]:
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.normal(1.0, 1.0, 100), rng.normal(0.0, 1.0, 100)])
    labels = np.concatenate([np.ones(100), np.zeros(100)])
    point = operating_point(scores, labels, 0.90)
    return _ok(
        point.specificity >= 0.90 - 1e-9, f"specificity {point.specificity:.3f}", point.specificity
    )


def check_false_early_rate() -> tuple[Status, str, float | None]:
    rate = false_early_rate(np.asarray([1.0, 5.0, np.nan]), np.asarray([5.0, 5.0, 8.0]))
    return _ok(abs(rate - 0.5) < 1e-9, f"rate {rate:.4f}", rate)


def check_cross_site_change() -> tuple[Status, str, float | None]:
    change = cross_site_change({"A": 0.90, "B": 0.80}, {"A": 0.95, "B": 0.80})
    return _ok(abs(change - 0.025) < 1e-12, f"change {change:.4f}", change)


def check_calibration_error() -> tuple[Status, str, float | None]:
    probabilities = np.asarray([0.05, 0.05, 0.95, 0.95])
    labels = np.asarray([0.0, 0.0, 1.0, 1.0])
    value = expected_calibration_error(probabilities, labels, bins=10)
    return _ok(value < 0.051, f"ECE {value:.4f}", value)


def check_weighted_kappa() -> tuple[Status, str, float | None]:
    identical = weighted_kappa(["mild", "moderate", "severe"], ["mild", "moderate", "severe"])
    shifted = weighted_kappa(["mild", "mild", "mild"], ["severe", "severe", "severe"])
    return _ok(
        abs(identical - 1.0) < 1e-9 and shifted < 1.0,
        f"identical {identical:.3f}, shifted {shifted:.3f}",
        identical,
    )


def check_holm() -> tuple[Status, str, float | None]:
    result = holm_stepdown({"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04}, alpha=0.05)
    adjusted = dict(zip(result.names, result.adjusted, strict=True))
    ok = (
        abs(adjusted["a"] - 0.04) < 1e-9 and abs(adjusted["b"] - 0.06) < 1e-9 and result.rejected[0]
    )
    return _ok(ok, f"adjusted {adjusted}", adjusted["a"])


def check_orc() -> tuple[Status, str, float | None]:
    rng = np.random.default_rng(2)
    base = rng.normal(size=(8, 30))
    test = orc_difference(base + 0.05 + rng.normal(scale=0.01, size=base.shape), base)
    return _ok(
        abs(test.difference - 0.05) < 0.005 and test.standard_error > 0,
        f"difference {test.difference:.4f}, se {test.standard_error:.4f}",
        test.difference,
    )


def check_reliance() -> tuple[Status, str, float | None]:
    summary = reliance_summary(20, 10, 318)
    return _ok(
        abs(summary.followed_rate - 20 / 318) < 1e-9
        and summary.followed_interval[0] < summary.followed_rate,
        f"rate {summary.followed_rate:.4f}",
        summary.followed_rate,
    )


def check_interaction_ratio() -> tuple[Status, str, float | None]:
    value = interaction_ratio(0.944, 0.897, 0.912, 0.838)
    return _ok(abs(value - 1.34) < 0.005, f"ratio {value:.4f}", value)


def check_budget_profile() -> tuple[Status, str, float | None]:
    checks = profile_checks()
    ok = abs(checks["gflops_residual"]) < 1e-9 and abs(checks["share_residual"]) < 2e-3
    return _ok(
        ok, f"sum {checks['gflops_sum']}, share {checks['share_sum']:.4f}", checks["gflops_sum"]
    )


def check_flop_counter() -> tuple[Status, str, float | None]:
    module = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1),
    )
    value = count_flops(module, (torch.zeros(2, 3, 16, 16),))
    return _ok(value > 0, f"{value:.6f} GFLOPs", value)


def check_percent_change() -> tuple[Status, str, float | None]:
    estimate = percent_change([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], n_boot=200)
    return _ok(
        abs(estimate.percent_change - 100.0) < 1e-9,
        f"{estimate.percent_change:.2f}%",
        estimate.percent_change,
    )


def check_manifest_leakage() -> tuple[Status, str, float | None]:
    table = CohortTable(
        records=[
            ClipRecord("p1", "e1", "A", "I", "s1", 1, "retrospective", "train", "a.npz"),
            ClipRecord("p1", "e2", "A", "I", "s1", 1, "retrospective", "test", "b.npz"),
        ]
    )
    try:
        ensure_patient_level(table)
    except ValueError:
        return _ok(True, "leakage refused", 1.0)
    return _ok(False, "leakage accepted", 0.0)


def check_padding_mask() -> tuple[Status, str, float | None]:
    first = probe_clip(seed=0, n_frames=10)
    second = probe_clip(seed=1, n_frames=6)
    batch = stack_clips([first, second])
    ok = int(batch.frame_mask[0].sum().item()) == 10 and int(batch.frame_mask[1].sum().item()) == 6
    return _ok(
        ok,
        f"mask counts {batch.frame_mask.sum(dim=1).tolist()}",
        float(batch.frame_mask.sum().item()),
    )


def check_determinism() -> tuple[Status, str, float | None]:
    batch = probe_batch(count=2, n_frames=8, size=32)
    set_seed(7)
    first = _small_model().eval()
    with torch.no_grad():
        a = first(batch).logit.clone()
    set_seed(7)
    second = _small_model().eval()
    with torch.no_grad():
        b = second(batch).logit.clone()
    gap = float((a - b).abs().max().item())
    return _ok(gap < 1e-6, f"gap {gap:.2e}", gap)


def check_cli_parser() -> tuple[Status, str, float | None]:
    from tearfilm_persistence.cli.evaluate import build_parser
    from tearfilm_persistence.cli.train import build_parser as train_parser

    evaluate_args = build_parser().parse_args(
        [
            "--experiment",
            "configs/experiments/main.yaml",
            "--cohort",
            "c.jsonl",
            "--checkpoint",
            "m.pt",
        ]
    )
    train_args = train_parser().parse_args(
        ["--experiment", "configs/experiments/main.yaml", "--cohort", "c.jsonl"]
    )
    ok = evaluate_args.split == "test" and train_args.checkpoint.endswith(".pt")
    return _ok(ok, f"evaluate split {evaluate_args.split}", 2.0)


def check_single_clip_inference() -> tuple[Status, str, float | None]:
    set_seed(4)
    model = _small_model().eval()
    result = model.infer(probe_clip())
    ok = len(result.descriptor.as_dict()) == len(COMPONENTS) and result.n_subtests >= 1
    return _ok(
        ok,
        f"sub-tests {result.n_subtests}, declared {result.descriptor.declared}",
        float(result.n_subtests),
    )


def check_relaxation_gap() -> tuple[Status, str, float | None]:
    field = ramp_derivative((1, 8, 24), slope=0.4)
    mask = uniform_mask(8)
    hard = hard_occupancy(field, span=3).float()
    soft = soft_occupancy(field, span=3, gamma=20.0)
    hard_curve = area_fraction(hard, mask)[0]
    soft_curve = area_fraction(soft, mask)[0]
    index, found = first_passage(hard_curve.unsqueeze(0), 0.1)
    if not bool(found.item()):
        return _ok(False, "the hard curve never declared", None)
    hard_onset = float(index.item()) / 30.0
    soft_index, soft_found = first_passage(soft_curve.unsqueeze(0), 0.5)
    if not bool(soft_found.item()):
        return _ok(False, "the relaxed curve never reached one half", None)
    soft_onset = float(soft_index.item()) / 30.0
    gap = abs(hard_onset - soft_onset)
    return _ok(gap <= 0.5, f"onset gap {gap:.3f} s", gap)


def check_stratified_report() -> tuple[Status, str, float | None]:
    rows = [
        ScoredEye(
            f"e{index}",
            f"p{index}",
            "A",
            "I",
            "retrospective",
            "mild" if index % 2 == 0 else "severe",
            5.0 if index < 10 else None,
            5.5 if index < 10 else None,
            0.85 - 0.035 * index,
            False,
        )
        for index in range(20)
    ]
    report = summarise_arm(rows, specificity_floor=0.5, n_boot=50, seed=0)
    return _ok(report.n_eyes == 20, f"{report.n_eyes} eyes, AUC {report.auc:.3f}", report.auc)


def check_criteria_thresholds() -> tuple[Status, str, float | None]:
    results = evaluate(0.93, 0.06, 0.9, 40.0)
    return _ok(
        all(result.passed for result in results),
        f"{[r.name for r in results]}",
        float(len(results)),
    )


CHECKS: tuple[tuple[str, str, str, Check], ...] = (
    ("E01", "import_surface", "Every public package imports", check_import_surface),
    (
        "E02",
        "config_load",
        "The main experiment file resolves its component files",
        check_config_loads,
    ),
    (
        "E03",
        "derivative_ramp",
        "The derivative of a linear ramp is its slope per second",
        check_derivative_on_ramp,
    ),
    (
        "E04",
        "derivative_window",
        "The derivative stencil is at most five frames wide",
        check_derivative_window,
    ),
    (
        "E05",
        "persistence_maturity",
        "A locus is occupied only after the window expires",
        check_persistence_maturity,
    ),
    (
        "E06",
        "persistence_flip",
        "A sign reversal inside the window clears the locus",
        check_persistence_sign_flip,
    ),
    (
        "E07",
        "persistence_alternating",
        "An alternating derivative never occupies a locus",
        check_persistence_alternating,
    ),
    (
        "E08",
        "soft_hard_agreement",
        "The relaxed indicator tracks the hard one on a clean run",
        check_soft_hard_agreement,
    ),
    (
        "E09",
        "first_passage",
        "The declared onset is the first crossing of alpha",
        check_first_passage,
    ),
    (
        "E10",
        "area_fraction",
        "The occupied fraction is taken over the masked area",
        check_area_fraction,
    ),
    (
        "E11",
        "locus_entropy",
        "Entropy is zero for one locus and one for an even spread",
        check_locus_entropy,
    ),
    (
        "E12",
        "ols_slope",
        "The post-declaration slope is a least-squares slope per second",
        check_ols_slope,
    ),
    (
        "E13",
        "descriptor_order",
        "The descriptor keeps the six-component order",
        check_descriptor_order,
    ),
    (
        "E14",
        "repeatability",
        "CoR equals 2.77 Sw and follows the pooled Sw",
        check_repeatability_arithmetic,
    ),
    ("E15", "icc", "ICC reaches one when repeats are identical", check_icc_ceiling),
    ("E16", "bland_altman", "The agreement bias is the mean paired difference", check_bland_altman),
    ("E17", "variance_shares", "The nested decomposition closes to one", check_variance_shares),
    (
        "E18",
        "calibrator",
        "Shrinkage moves a measurement toward its site mean",
        check_calibrator_shrinkage,
    ),
    ("E19", "logistic", "The calibration map recovers a known slope", check_logistic_calibrator),
    (
        "E20",
        "model_shapes",
        "The model returns one row per clip and six descriptors",
        check_model_shapes,
    ),
    ("E21", "gate_floor", "The gate abstains below the sampling floor", check_gate_rate_floor),
    ("E22", "gate_fixation", "The gate abstains outside the stability band", check_gate_fixation),
    ("E23", "gate_accept", "The gate accepts an adequate acquisition", check_gate_accepts),
    ("E24", "abstention", "Abstention reaches the model output", check_abstention_propagates),
    ("E25", "comparators", "Every comparator returns a clip-level pair", check_comparator_shapes),
    (
        "E26",
        "gradients",
        "One backward pass reaches the adapter, the structural branch and the head",
        check_gradients_reach_components,
    ),
    ("E27", "overfit", "A single batch is overfitted by thirty steps", check_single_batch_overfit),
    (
        "E28",
        "training_loop",
        "Two epochs on a small loader reduce the loss and write a checkpoint",
        check_training_loop,
    ),
    (
        "E29",
        "checkpoint",
        "A checkpoint round-trip reproduces the logits",
        check_checkpoint_roundtrip,
    ),
    (
        "E30",
        "repeatability_penalty",
        "The penalty separates clean from noisy repeats",
        check_repeatability_penalty,
    ),
    (
        "E31",
        "decision_loss",
        "The decision term matches cross-entropy on the valid rows",
        check_decision_loss,
    ),
    ("E32", "onset_loss", "The onset term is zero at a perfect prediction", check_onset_loss_zero),
    ("E33", "auc", "The area matches the Mann-Whitney identity", check_auc_identity),
    (
        "E34",
        "bootstrap",
        "The cluster interval brackets its point estimate",
        check_bootstrap_interval,
    ),
    (
        "E35",
        "operating_point",
        "The operating point respects the specificity floor",
        check_operating_point_floor,
    ),
    (
        "E36",
        "false_early",
        "False early counts onsets at least three seconds early",
        check_false_early_rate,
    ),
    (
        "E37",
        "cross_site",
        "Cross-site change is the mean absolute AUC change",
        check_cross_site_change,
    ),
    (
        "E38",
        "calibration_error",
        "A calibrated sample has near-zero calibration error",
        check_calibration_error,
    ),
    ("E39", "kappa", "Weighted kappa is one for identical ratings", check_weighted_kappa),
    ("E40", "holm", "Holm's step-down adjusts in ascending order", check_holm),
    ("E41", "orc", "The reader-study variance is positive and the shift is recovered", check_orc),
    ("E42", "reliance", "Reliance counts produce the reported rate", check_reliance),
    (
        "E43",
        "interaction",
        "The interaction ratio reproduces the table's arithmetic",
        check_interaction_ratio,
    ),
    ("E44", "budget", "The declared stage profile closes to its total", check_budget_profile),
    ("E45", "flops", "The analytical counter returns a positive cost", check_flop_counter),
    (
        "E46",
        "mechanistic",
        "A doubling reads as a one-hundred-percent change",
        check_percent_change,
    ),
    (
        "E47",
        "manifest_leakage",
        "A patient shared across splits is refused",
        check_manifest_leakage,
    ),
    ("E48", "padding", "Padding is excluded by the frame mask", check_padding_mask),
    ("E49", "determinism", "The same seed reproduces the logits", check_determinism),
    ("E50", "cli", "The documented command lines parse", check_cli_parser),
    (
        "E51",
        "single_clip",
        "One acquisition returns a complete descriptor",
        check_single_clip_inference,
    ),
    (
        "E52",
        "relaxation",
        "The relaxed and hard onsets agree within half a second",
        check_relaxation_gap,
    ),
    (
        "E53",
        "stratum_report",
        "A scored table summarises to an arm report",
        check_stratified_report,
    ),
    ("E54", "criteria", "The criteria evaluator accepts a passing set", check_criteria_thresholds),
)


def run_execution_checks() -> list[ExecutionCheck]:
    results: list[ExecutionCheck] = []
    for identifier, name, statement, function in CHECKS:
        try:
            status, detail, quantity = function()
        except Exception as error:  # a failing check is recorded, not raised
            status = "FAIL"
            detail = f"{type(error).__name__}: {error}"
            quantity = None
        results.append(
            ExecutionCheck(
                id=identifier,
                name=name,
                statement=statement,
                status=status,
                detail=detail,
                quantity=quantity,
            )
        )
    return results
