"""Component ablation arithmetic.

Ref: Table 3 and Sec. 2.3. Removing the two temporal branches costs more than
the sum of the two individual losses; the interaction ratio is the joint loss
over that sum.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationDelta:
    name: str
    full: float
    variant: float
    delta: float


def delta(full: float, variant: float) -> float:
    return float(full - variant)


def deltas(full: float, variants: dict[str, float]) -> list[AblationDelta]:
    return [
        AblationDelta(name=name, full=full, variant=value, delta=delta(full, value))
        for name, value in variants.items()
    ]


def interaction_ratio(
    full: float, minus_one: float, minus_two: float, both_removed: float
) -> float:
    """Joint loss over the sum of the individual losses.

    A ratio above one says the two components are not additive: neither alone
    recovers what the pair supplies.
    """
    individual = (full - minus_one) + (full - minus_two)
    joint = full - both_removed
    if individual <= 0:
        raise ValueError("the individual losses must be positive for a ratio")
    return float(joint / individual)
