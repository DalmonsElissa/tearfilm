"""Assembly of the three release reports.

The claim map, the verification report and the integrity manifest are written
from one pass over the same measurements, so a status cannot differ between
them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tearfilm_persistence.verification.claims import CLAIMS, ClaimResult, evaluate_claims
from tearfilm_persistence.verification.conformance import ConformanceCheck, run_conformance_checks
from tearfilm_persistence.verification.execution import ExecutionCheck, run_execution_checks
from tearfilm_persistence.verification.hygiene import HygieneReport, scan
from tearfilm_persistence.verification.hygiene import as_dict as hygiene_dict
from tearfilm_persistence.verification.manuscript import ManuscriptCheck, run_manuscript_checks

PAPER: dict[str, Any] = {
    "title": (
        "Repurposing Robotic Anterior Segment Imaging for Tear Film "
        "Assessment in Dry Eye with Temporal Models"
    ),
    "venue": "npj Digital Medicine",
    "package": "tearfilm_persistence",
    "cohort_distributed": False,
    "cohort_note": (
        "The clinical cohort is private and de-identified at patient and "
        "institution level; no record, derived descriptor or re-linkage key "
        "is shipped or reconstructed."
    ),
    "manuscript_type": (
        "clinical validation with a completed prospective arm and a three-arm reader study"
    ),
}

DEVIATIONS: tuple[dict[str, str], ...] = (
    {
        "id": "D01",
        "area": "persistence operator",
        "manuscript_location": "Sec. 5.4, Algorithm 1 line 9",
        "released_behaviour": (
            "Two evaluators are released. The hard evaluator applies the sign "
            "test as written and produces every reported descriptor. The soft "
            "evaluator replaces the sign-equality indicator with a sigmoid of "
            "the consecutive derivative product so that the integrated "
            "objective has a gradient; training uses it and inference does not."
        ),
        "justification": (
            "The sign test is piecewise constant in the derivative and "
            "therefore has zero gradient almost everywhere. A relaxation is the "
            "only way to train the three-term objective end to end, and the gap "
            "between the two evaluators is measured in execution check E52."
        ),
    },
    {
        "id": "D02",
        "area": "affected-area fraction",
        "manuscript_location": "Sec. 5.4, the threshold alpha",
        "released_behaviour": "alpha defaults to 0.10 of the examined area.",
        "justification": (
            "The article names alpha but does not print a value. The default is "
            "bounded above by the smallest peak affected fraction the case table "
            "reports (0.36), because a declaration cannot follow its own peak."
        ),
    },
    {
        "id": "D03",
        "area": "onset regression",
        "manuscript_location": "Sec. 5.4, the second objective term",
        "released_behaviour": (
            "Smooth L1 with a half-second transition, applied to the sub-tests "
            "that carry a reference onset."
        ),
        "justification": (
            "The term is named without a norm. Smooth L1 keeps a linear tail for "
            "the late declarations that the persistence requirement can produce."
        ),
    },
    {
        "id": "D04",
        "area": "objective weights",
        "manuscript_location": "Sec. 5.4",
        "released_behaviour": "Decision 1.0, onset 1.0, repeatability 0.1.",
        "justification": (
            "The article states that the three terms are combined and does not "
            "print the weights."
        ),
    },
    {
        "id": "D05",
        "area": "optimisation",
        "manuscript_location": "Sec. 5.4",
        "released_behaviour": (
            "AdamW, cosine schedule with linear warm-up, 3e-4, weight decay 1e-2, "
            "40 epochs, batch 8, bf16, gradient clipping at 1.0."
        ),
        "justification": (
            "No schedule, optimiser, epoch count, batch size or precision is "
            "printed. These are released values and are recorded as engineering "
            "defaults rather than as manuscript settings."
        ),
    },
    {
        "id": "D06",
        "area": "operating point",
        "manuscript_location": "Sec. 2.7",
        "released_behaviour": (
            "The threshold maximises sensitivity subject to a specificity floor of 0.90."
        ),
        "justification": (
            "The article states that the threshold is placed at the weighted "
            "specificity limit derived from the two error costs and does not "
            "print the costs or the floor."
        ),
    },
    {
        "id": "D07",
        "area": "structural read-out",
        "manuscript_location": "Sec. 5.2",
        "released_behaviour": (
            "Lateral pixel pitch equals the printed axial resolution of 15 micrometres."
        ),
        "justification": "Only the axial resolution is printed; the area needs both pitches.",
    },
    {
        "id": "D08",
        "area": "blink segmentation",
        "manuscript_location": "Sec. 5.6",
        "released_behaviour": (
            "Blinks are detected from a joint contrast collapse and motion spike "
            "in the acquisition."
        ),
        "justification": (
            "The formal statement takes blink events as given. A released "
            "pipeline has to obtain them, and the detector is reported so that a "
            "reader can replace it."
        ),
    },
    {
        "id": "D09",
        "area": "sub-test boundary",
        "manuscript_location": "Algorithm 1, lines 9 to 12",
        "released_behaviour": (
            "A sub-test whose loop variable passes the epoch end is recorded as "
            "censored, not as a declaration at the boundary frame."
        ),
        "justification": (
            "Written literally, line 12 assigns the loop variable even when the "
            "while condition failed, which would place a declaration one frame "
            "beyond the epoch it belongs to."
        ),
    },
    {
        "id": "D10",
        "area": "decision target",
        "manuscript_location": "Sec. 5.4, the first objective term",
        "released_behaviour": (
            "The decision is formed once per acquisition from the presence of a "
            "reference break-up."
        ),
        "justification": (
            "The term is described over sub-tests. The model reports one decision "
            "per acquisition because the clinical endpoint and the comparator "
            "table are both clip-level; the per-sub-test labels are used for the "
            "onset term."
        ),
    },
    {
        "id": "D11",
        "area": "fast-branch weights",
        "manuscript_location": "Sec. 5.4",
        "released_behaviour": (
            "No pretrained clip checkpoint is shipped. When one is not supplied "
            "the trunk trains jointly with the adapter; when one is supplied the "
            "trunk is frozen and only the adapter is fitted."
        ),
        "justification": (
            "The pretrained prior the article describes is a third-party "
            "checkpoint that cannot be redistributed here. The ablation row that "
            "depends on it is reported BLOCKED rather than approximated."
        ),
    },
)

ENGINEERING_DEFAULTS: tuple[dict[str, str], ...] = (
    {"name": "batch_size", "value": "8 clips", "status": "not printed in the article"},
    {"name": "grad_accumulation", "value": "1", "status": "not printed in the article"},
    {"name": "epochs", "value": "40", "status": "not printed in the article"},
    {"name": "learning_rate", "value": "3e-4", "status": "not printed in the article"},
    {"name": "weight_decay", "value": "1e-2", "status": "not printed in the article"},
    {"name": "warmup_steps", "value": "500", "status": "not printed in the article"},
    {
        "name": "scheduler",
        "value": "cosine with linear warm-up",
        "status": "not printed in the article",
    },
    {"name": "precision", "value": "bf16", "status": "not printed in the article"},
    {"name": "grad_clip", "value": "1.0", "status": "not printed in the article"},
    {
        "name": "affected_area_fraction",
        "value": "0.10",
        "status": "named but not valued in the article",
    },
    {
        "name": "relaxation_sharpness",
        "value": "20.0",
        "status": "engineering value for the differentiable surrogate",
    },
    {"name": "onset_beta", "value": "0.5 s", "status": "norm not printed in the article"},
    {"name": "repeatability_weight", "value": "0.1", "status": "not printed in the article"},
    {"name": "specificity_floor", "value": "0.90", "status": "cost weighting not printed"},
    {"name": "lateral_pitch", "value": "15 um", "status": "only the axial resolution is printed"},
    {
        "name": "max_fixation_drift",
        "value": "0.055 of frame width",
        "status": "band width not printed in the article",
    },
    {
        "name": "locus_grid",
        "value": "8 x 8",
        "status": "the article prints a 25 to 100 element range",
    },
    {
        "name": "clip_trunk_widths",
        "value": "24, 48, 64",
        "status": "architecture width not printed",
    },
)

_ACQ = "cohort/manifest.py, cohort/sampling.py"
_FRAME = "deformation/adapter.py, deformation/derivative.py"
_CURVE = "deformation/persistence.py, deformation/curve.py"
_DESC = "descriptors/vector.py, descriptors/calibration.py"
_REPORT = "evaluation/aggregate.py, evaluation/discrimination.py"

IMPLEMENTATION_MAP: tuple[dict[str, str], ...] = (
    {"section": "Sec. 2.1, Sec. 5.1", "item": "cohorts, arms, partition", "files": _ACQ},
    {
        "section": "Sec. 5.2",
        "item": "acquisition and adequacy gate",
        "files": "streams/adequacy.py, streams/acquisition.py",
    },
    {
        "section": "Sec. 5.6",
        "item": "sub-test partition",
        "files": "streams/blinks.py, deformation/persistence.py",
    },
    {
        "section": "Sec. 5.4, D_t = d_t phi(x_t)",
        "item": "representation and derivative",
        "files": _FRAME,
    },
    {"section": "Sec. 5.4, Pi_tau", "item": "persistence operator and curve", "files": _CURVE},
    {
        "section": "Algorithm 1, lines 9 to 15",
        "item": "onset, peak, slope, entropy, artefact channel",
        "files": "deformation/curve.py, models/teardyn.py",
    },
    {"section": "Sec. 5.4, z_k", "item": "descriptor vector and calibration", "files": _DESC},
    {
        "section": "Sec. 5.2, 15 um",
        "item": "meniscus height and area",
        "files": "structure/meniscus.py",
    },
    {
        "section": "Sec. 5.4, three terms",
        "item": "integrated objective",
        "files": "objectives/composite.py",
    },
    {"section": "Table 2", "item": "six comparators", "files": "comparators/"},
    {
        "section": "Table 3",
        "item": "component ablations and interaction",
        "files": "models/teardyn.py, evaluation/ablation.py",
    },
    {"section": "Table 4, Table 5", "item": "site and stratum reporting", "files": _REPORT},
    {
        "section": "Sec. 2.6, Table 6",
        "item": "three-arm reader study",
        "files": "evaluation/reader_study.py",
    },
    {
        "section": "Sec. 5.8",
        "item": "multiplicity and agreement",
        "files": "evaluation/multiplicity.py, evaluation/agreement.py",
    },
    {"section": "Sec. 5.9", "item": "pre-specified criteria", "files": "evaluation/criteria.py"},
    {"section": "Table 8", "item": "computational profile", "files": "evaluation/budget.py"},
    {
        "section": "Sec. 2.9, Sec. 5.10",
        "item": "mechanistic arm",
        "files": "evaluation/mechanistic.py",
    },
    {
        "section": "Sec. 5.4",
        "item": "training loop and checkpointing",
        "files": "runtime/loop.py, runtime/checkpoint.py",
    },
)


@dataclass
class VerificationBundle:
    claims: list[ClaimResult]
    execution: list[ExecutionCheck]
    manuscript: list[ManuscriptCheck]
    conformance: list[ConformanceCheck]
    hygiene: HygieneReport


def collect(root: Path | None = None) -> VerificationBundle:
    return VerificationBundle(
        claims=evaluate_claims(CLAIMS),
        execution=run_execution_checks(),
        manuscript=run_manuscript_checks(),
        conformance=run_conformance_checks(root),
        hygiene=scan(root),
    )


def _counts(statuses: Iterable[str]) -> dict[str, int]:
    counts = {"PASS": 0, "FAIL": 0, "NOT_RUN": 0, "BLOCKED": 0}
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    return counts


def _claim_entries(bundle: VerificationBundle) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for result in bundle.claims:
        entries.append(
            {
                "id": result.id,
                "location": result.location,
                "statement": result.statement,
                "family": result.family,
                "symbols": result.symbols,
                "missing_symbols": result.missing,
                "status": result.status,
                "details": result.details,
                "code_present": not result.missing,
                "verification": "mapping" if result.family == "code" else "record",
            }
        )
    return entries


def claim_report(bundle: VerificationBundle) -> dict[str, Any]:
    counts = _counts(result.status for result in bundle.claims)
    return {
        "paper": PAPER,
        "summary": {
            "claims_checked": len(bundle.claims),
            "claims_passed": counts["PASS"],
            "claims_blocked": counts["BLOCKED"],
            "claims_failed": counts["FAIL"],
            "code_status": "PASS" if counts["FAIL"] == 0 else "FAIL",
            "note": (
                "A claim is BLOCKED when its statement is a recorded measurement "
                "that needs the private cohort."
            ),
        },
        "claims": _claim_entries(bundle),
        "implementation_map": list(IMPLEMENTATION_MAP),
        "deviations": list(DEVIATIONS),
        "engineering_defaults": list(ENGINEERING_DEFAULTS),
    }


def verification_report(bundle: VerificationBundle) -> dict[str, Any]:
    claim_counts = _counts(result.status for result in bundle.claims)
    execution_counts = _counts(check.status for check in bundle.execution)
    manuscript_counts = _counts(check.status for check in bundle.manuscript)
    conformance_counts = _counts(check.status for check in bundle.conformance)
    hygiene_errors = len(bundle.hygiene.errors)

    code_status = "PASS" if claim_counts["FAIL"] == 0 else "FAIL"
    execution_status = "PASS" if execution_counts["FAIL"] == 0 else "FAIL"
    manuscript_status = "PASS" if manuscript_counts["FAIL"] == 0 else "FAIL"
    conformance_status = "PASS" if conformance_counts["FAIL"] == 0 else "FAIL"
    hygiene_status = "PASS" if hygiene_errors == 0 else "FAIL"

    blocking = (
        code_status == "PASS"
        and execution_status == "PASS"
        and conformance_status == "PASS"
        and hygiene_status == "PASS"
    )
    if blocking and manuscript_status == "PASS" and claim_counts["BLOCKED"] == 0:
        overall = "VERIFIED"
    elif code_status == "PASS" and execution_status == "PASS":
        overall = "PARTIALLY_VERIFIED"
    else:
        overall = "UNVERIFIED"

    return {
        "paper": PAPER,
        "summary": {
            "overall_status": overall,
            "code_status": code_status,
            "execution_status": execution_status,
            "manuscript_status": manuscript_status,
            "conformance_status": conformance_status,
            "hygiene_status": hygiene_status,
            "claims_checked": len(bundle.claims),
            "claims_blocked": claim_counts["BLOCKED"],
            "claims_failed": claim_counts["FAIL"],
            "execution_checks": len(bundle.execution),
            "execution_failed": execution_counts["FAIL"],
            "manuscript_checks": len(bundle.manuscript),
            "manuscript_findings": manuscript_counts["FAIL"],
            "conformance_checks": len(bundle.conformance),
            "conformance_failed": conformance_counts["FAIL"],
            "hygiene_errors": hygiene_errors,
        },
        "layer_one_paper_to_code": {
            "checks": _claim_entries(bundle),
            "counts": claim_counts,
        },
        "layer_two_execution": {
            "checks": [check.__dict__ for check in bundle.execution],
            "counts": execution_counts,
        },
        "manuscript_checks": [check.as_dict() for check in bundle.manuscript],
        "manuscript_inconsistencies": [
            check.as_dict() for check in bundle.manuscript if check.status == "FAIL"
        ],
        "conformance": [check.as_dict() for check in bundle.conformance],
        "deviations": list(DEVIATIONS),
        "engineering_defaults": list(ENGINEERING_DEFAULTS),
        "hygiene": hygiene_dict(bundle.hygiene),
        "notes": [
            (
                "Record-level values of Table 2, Table 4, Table 5 and Table 6 are "
                "BLOCKED: the cohort is private and is neither shipped nor "
                "reconstructed."
            ),
            (
                "Execution checks run on tensors built inside the check process; "
                "no record is read and nothing is written outside a temporary "
                "directory."
            ),
            (
                "The manuscript checks test arithmetic internal to the article and "
                "are reported separately from the code checks, because a "
                "discrepancy inside the article is not a defect of this release."
            ),
            (
                "The mechanistic arm is reported as partial in the article; its "
                "replicate table is not distributed and no read-out is reproduced."
            ),
        ],
    }


def integrity_manifest(
    root: Path, exclude: Iterable[str] = ("integrity_manifest.json",)
) -> dict[str, Any]:
    excluded = set(exclude)
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(
            part in {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".venv"}
            for part in relative.parts
        ):
            continue
        if str(relative) in excluded or relative.name in excluded:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({"path": str(relative), "sha256": digest, "bytes": path.stat().st_size})
    combined = hashlib.sha256()
    for entry in entries:
        combined.update(entry["path"].encode("utf-8"))
        combined.update(entry["sha256"].encode("utf-8"))
    return {
        "algorithm": "sha256",
        "file_count": len(entries),
        "excludes": sorted(excluded),
        "entries": entries,
        "manifest_digest": combined.hexdigest(),
    }


def write_reports(
    root: Path | None = None, bundle: VerificationBundle | None = None
) -> dict[str, Path]:
    base = root or Path(__file__).resolve().parents[3]
    data = bundle or collect(base)
    written: dict[str, Path] = {}
    written["claim_to_code"] = _write(base / "claim_to_code.json", claim_report(data))
    written["verification_report"] = _write(
        base / "verification_report.json", verification_report(data)
    )
    written["summary"] = _write_text(base / "verification_summary.txt", summary_text(data))
    written["manifest"] = _write(base / "integrity_manifest.json", integrity_manifest(base))
    return written


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def summary_text(bundle: VerificationBundle) -> str:
    report = verification_report(bundle)
    summary = report["summary"]
    lines = [
        f"overall            {summary['overall_status']}",
        f"code               {summary['code_status']}",
        f"  claims failed    {summary['claims_failed']} of {summary['claims_checked']}",
        f"  claims blocked   {summary['claims_blocked']}",
        f"execution          {summary['execution_status']}",
        f"  checks failed    {summary['execution_failed']} of {summary['execution_checks']}",
        f"manuscript         {summary['manuscript_status']}",
        f"  findings         {summary['manuscript_findings']} of {summary['manuscript_checks']}",
        f"conformance        {summary['conformance_status']}",
        f"  checks failed    {summary['conformance_failed']} of {summary['conformance_checks']}",
        f"hygiene            {summary['hygiene_status']}",
        f"  errors           {summary['hygiene_errors']}",
        "",
        "manuscript discrepancies",
    ]
    for check in report["manuscript_inconsistencies"]:
        lines.append(f"  {check['id']}  {check['location']}")
        lines.append(f"      {check['detail']}")
    failed = [
        check for check in report["layer_two_execution"]["checks"] if check["status"] != "PASS"
    ]
    if failed:
        lines.append("")
        lines.append("execution checks not passing")
        for check in failed:
            lines.append(f"  {check['id']}  {check['name']}: {check['detail']}")
    lines.append("")
    lines.append("blocked record-level claims carry their staging reason in claim_to_code.json")
    return "\n".join(lines) + "\n"
