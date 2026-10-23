"""Computational profile of the inference path.

Ref: Table 8, cost per clip on a single accelerator, excluding acquisition time.

The stage shares are arithmetic on the declared stage costs, and the profile is
checked for the two invariants the table asserts: the stages sum to the reported
total and the shares sum to one. An architectural count over a different clip
geometry is a separate quantity and is never substituted for the declared one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class StageCost:
    stage: str
    operation: str
    gflops: float


DECLARED_PROFILE: tuple[StageCost, ...] = (
    StageCost("adequacy_gate", "sampling-rate and fixation-stability test", 1.2),
    StageCost("fast_branch", "derivative field over at most five frames", 47.5),
    StageCost("persistence", "sign-consistency test over tau frames", 10.4),
    StageCost("structural_branch", "meniscus height and area from one frame", 9.0),
    StageCost("descriptor_head", "assembly, calibration, uncertainty", 7.3),
    StageCost("post_processing", "artefact channel, repeatability lookup", 2.6),
)

DECLARED_TOTAL_GFLOPS: float = 78.0
DECLARED_TOTAL_MILLISECONDS: float = 164.0
UNCONSTRAINED_GFLOPS: float = 214.0
UNCONSTRAINED_MILLISECONDS: float = 268.0


def total_gflops(stages: Sequence[StageCost]) -> float:
    return float(sum(stage.gflops for stage in stages))


def shares(stages: Sequence[StageCost] = DECLARED_PROFILE) -> list[float]:
    total = total_gflops(stages)
    if total <= 0:
        raise ValueError("a cost profile needs a positive total")
    return [stage.gflops / total for stage in stages]


def profile_checks(stages: Sequence[StageCost] = DECLARED_PROFILE) -> dict[str, float]:
    share_values = shares(stages)
    return {
        "gflops_sum": total_gflops(stages),
        "gflops_declared": DECLARED_TOTAL_GFLOPS,
        "gflops_residual": total_gflops(stages) - DECLARED_TOTAL_GFLOPS,
        "share_sum": float(sum(share_values)),
        "share_residual": float(sum(share_values) - 1.0),
    }


def count_flops(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> float:
    """Analytical GFLOPs for one forward pass of convolutions and linear layers."""
    total = [0.0]

    def conv_hook(
        module: nn.Conv2d | nn.Conv3d, args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        kernel = 1
        for size in module.kernel_size:
            kernel *= int(size)
        per_output = kernel * (module.in_channels // module.groups)
        total[0] += 2.0 * float(output.numel()) * per_output / 1e9

    def linear_hook(
        module: nn.Linear, args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        total[0] += 2.0 * float(output.numel()) * float(module.in_features) / 1e9

    handles = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(*inputs)
    model.train(was_training)
    for handle in handles:
        handle.remove()
    return total[0]
