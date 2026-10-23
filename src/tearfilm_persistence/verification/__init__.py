"""Verification layer."""

from tearfilm_persistence.verification.claims import CLAIMS, Claim, ClaimResult, evaluate_claims
from tearfilm_persistence.verification.conformance import ConformanceCheck, run_conformance_checks
from tearfilm_persistence.verification.execution import ExecutionCheck, run_execution_checks
from tearfilm_persistence.verification.hygiene import HygieneReport, scan
from tearfilm_persistence.verification.manuscript import ManuscriptCheck, run_manuscript_checks
from tearfilm_persistence.verification.report import (
    DEVIATIONS,
    ENGINEERING_DEFAULTS,
    IMPLEMENTATION_MAP,
    PAPER,
    VerificationBundle,
    collect,
    integrity_manifest,
    verification_report,
    write_reports,
)

__all__ = [
    "CLAIMS",
    "DEVIATIONS",
    "ENGINEERING_DEFAULTS",
    "IMPLEMENTATION_MAP",
    "PAPER",
    "Claim",
    "ClaimResult",
    "ConformanceCheck",
    "ExecutionCheck",
    "HygieneReport",
    "ManuscriptCheck",
    "VerificationBundle",
    "collect",
    "evaluate_claims",
    "integrity_manifest",
    "run_conformance_checks",
    "run_execution_checks",
    "run_manuscript_checks",
    "scan",
    "verification_report",
    "write_reports",
]
