"""Report assembly and hygiene rules."""

from __future__ import annotations

import json
from pathlib import Path

from tearfilm_persistence.verification.claims import CLAIMS, evaluate_claims, resolve_symbol
from tearfilm_persistence.verification.hygiene import scan
from tearfilm_persistence.verification.manuscript import run_manuscript_checks
from tearfilm_persistence.verification.report import (
    claim_report,
    collect,
    integrity_manifest,
    verification_report,
)

ROOT = Path(__file__).resolve().parents[1]


def test_every_claim_symbol_resolves() -> None:
    results = evaluate_claims()
    assert not [result for result in results if result.missing]
    assert len(results) == len(CLAIMS) == 80


def test_claim_families_are_declared() -> None:
    assert {claim.family for claim in CLAIMS} <= {"code", "value", "design"}


def test_resolve_symbol_names_the_module_and_attribute() -> None:
    assert callable(resolve_symbol("tearfilm_persistence.deformation.persistence:hard_occupancy"))


def test_manuscript_checks_are_arithmetic_only() -> None:
    checks = run_manuscript_checks()
    assert all(check.computed is not None or check.printed is not None for check in checks)
    assert len(checks) >= 50


def test_manuscript_findings_name_their_location() -> None:
    findings = [check for check in run_manuscript_checks() if check.status == "FAIL"]
    assert findings
    assert all(check.location for check in findings)
    assert all(check.detail for check in findings)


def test_hygiene_scan_reports_no_absolute_paths() -> None:
    report = scan(ROOT)
    absolute = [finding for finding in report.findings if finding.rule == "absolute_path"]
    assert not absolute


def test_hygiene_scan_reports_one_markdown_file() -> None:
    report = scan(ROOT)
    markdown = [finding for finding in report.findings if finding.rule == "markdown_count"]
    assert not markdown


def test_hygiene_scan_finds_no_workflow_directory() -> None:
    report = scan(ROOT)
    assert not [finding for finding in report.findings if finding.rule == "workflow_directory"]


def test_claim_report_summarises_statuses() -> None:
    bundle = collect(ROOT)
    report = claim_report(bundle)
    assert report["summary"]["claims_checked"] == 80
    assert report["summary"]["code_status"] in {"PASS", "FAIL"}
    assert len(report["claims"]) == 80


def test_verification_report_keeps_manuscript_and_code_apart() -> None:
    report = verification_report(collect(ROOT))
    assert report["summary"]["code_status"] == "PASS"
    assert report["summary"]["manuscript_status"] in {"PASS", "FAIL"}
    assert "overall_status" in report["summary"]


def test_integrity_manifest_excludes_itself() -> None:
    manifest = integrity_manifest(ROOT)
    paths = {entry["path"] for entry in manifest["entries"]}
    assert "integrity_manifest.json" not in paths
    assert "README.md" in paths
    assert manifest["file_count"] == len(manifest["entries"])


def test_reports_are_json_serialisable(tmp_path) -> None:
    bundle = collect(ROOT)
    for payload in (claim_report(bundle), verification_report(bundle)):
        json.dumps(payload)
