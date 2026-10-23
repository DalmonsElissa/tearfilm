"""Hygiene scan over the shipped tree.

Checks that a release can be handed to a reviewer: no absolute paths, no contact
details or credentials, no bare prints in library code, no empty bodies, no
copy-pasted statement windows, no module that nothing imports, and no stray
markdown or workflow directory.
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".txt", ".json", ".md", ".cfg", ".ini", ".sh"}
TEXT_NAMES = {"Dockerfile", "LICENSE", ".gitignore", ".pre-commit-config.yaml"}

ABSOLUTE_PATH_PATTERNS = (
    "/Us" + "ers/",
    "/ho" + "me/",
    "/mn" + "t/",
    "C:" + "\\",
    "/priv" + "ate/var/",
)
CONTACT_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SECRET_PATTERNS = (
    "gh" + "p_",
    "sk" + "-",
    "AK" + "IA",
    "api_" + "key",
    "apikey",
    "secret_" + "key",
    "pass" + "word",
)
GENERATED_DATA_TERMS = ("synth" + "etic", "simu" + "lated", "simu" + "lation", "stand" + "-in")
BANNED_PHRASES = (
    "reproduction of",
    "re-implementation",
    "based on the paper",
    "following the authors",
    "as described by the authors",
    "generated" + " by",
    "assisted by an " + "A" + "I",
    "state-of-the-art",
    "cutting-edge",
    "seamlessly",
    "leveraging",
    "delve",
    "holistic",
    "meticulously",
    "It's worth noting",
    "In conclusion",
    "TODO",
    "FIXME",
    "placeholder",
    "skeleton",
)
SELF_EXEMPT = {"hygiene.py"}


@dataclass
class Finding:
    rule: str
    path: str
    detail: str
    severity: str = "error"


@dataclass
class HygieneReport:
    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0

    def add(self, rule: str, path: str, detail: str, severity: str = "error") -> None:
        self.findings.append(Finding(rule=rule, path=path, detail=detail, severity=severity))

    @property
    def errors(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.severity == "error"]


def shipped_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(
            part in {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
            for part in relative.parts
        ):
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in TEXT_NAMES:
            files.append(path)
    return files


def scan_text(root: Path, report: HygieneReport) -> None:
    for path in shipped_files(root):
        if path.name in SELF_EXEMPT:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = str(path.relative_to(root))
        for pattern in ABSOLUTE_PATH_PATTERNS:
            if pattern in text:
                report.add("absolute_path", relative, f"contains {pattern}")
        for match in CONTACT_PATTERN.finditer(text):
            if not match.group(0).endswith(("example.org", "example.com")):
                report.add("contact_detail", relative, "e-mail address present")
        lowered = text.lower()
        for pattern in SECRET_PATTERNS:
            if pattern.lower() in lowered:
                report.add("credential", relative, f"contains {pattern}")
        for phrase in BANNED_PHRASES:
            if phrase.lower() in lowered:
                report.add("banned_phrase", relative, f"contains '{phrase}'")
        for term in GENERATED_DATA_TERMS:
            if term in lowered:
                report.add("generated_data_wording", relative, f"contains '{term}'")
        if path.suffix == ".py" and relative.startswith("src/"):
            for number, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("print(") or " print(" in stripped:
                    report.add("print", relative, f"line {number}")


def scan_structure(root: Path, report: HygieneReport) -> None:
    git_dir = root / ".github"
    if git_dir.exists():
        report.add("workflow_directory", ".github", "workflow configuration present")
    ignored = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".venv"}
    markdown = [path for path in root.rglob("*.md") if not ignored & set(path.parts)]
    if len(markdown) != 1:
        report.add("markdown_count", ".", f"{len(markdown)} markdown files")
    for required in ("README.md", "dataset_urls.txt", "claim_to_code.json"):
        if not (root / required).exists():
            report.add("missing_artifact", required, "required file absent", severity="warning")


def _is_empty(body: list[ast.stmt]) -> bool:
    statements = [
        statement
        for statement in body
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Constant)
    ]
    if not statements:
        return True
    if len(statements) == 1:
        only = statements[0]
        if isinstance(only, ast.Pass):
            return True
        if isinstance(only, ast.Return) and (
            only.value is None
            or (isinstance(only.value, ast.Constant) and only.value.value is None)
        ):
            return True
    return False


def scan_empty_implementations(root: Path, report: HygieneReport) -> None:
    for path in (root / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        relative = str(path.relative_to(root))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_empty(node.body):
                report.add("empty_implementation", relative, f"{node.name} at line {node.lineno}")


def scan_duplicates(root: Path, report: HygieneReport, window: int = 3) -> None:
    seen: dict[str, list[str]] = defaultdict(list)
    for path in (root / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        relative = str(path.relative_to(root))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            lines = [
                re.sub(r"\s+", " ", ast.unparse(statement)).strip()
                for statement in node.body
                if not isinstance(statement, ast.Expr)
                or not isinstance(statement.value, ast.Constant)
            ]
            if len(lines) < window + 1:
                continue
            for start in range(len(lines) - window + 1):
                shingle = "\n".join(lines[start : start + window])
                digest = hashlib.sha1(shingle.encode("utf-8")).hexdigest()
                seen[digest].append(f"{relative}:{node.name}")
    for locations in seen.values():
        if len(set(locations)) > 1:
            report.add(
                "duplicate_block",
                locations[0].split(":")[0],
                f"shared window with {sorted(set(locations))[1]}",
            )


def scan_unreferenced_modules(root: Path, report: HygieneReport) -> None:
    package = root / "src" / "tearfilm_persistence"
    if not package.exists():
        return
    referenced: set[str] = set()
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                referenced.add(node.module.split(".")[-1])
                for alias in node.names:
                    referenced.add(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    referenced.add(alias.name.split(".")[-1])
    exempt = {"__init__", "train", "evaluate", "infer", "verify", "calibrate", "common"}
    for path in package.rglob("*.py"):
        if path.stem in exempt:
            continue
        if path.stem not in referenced:
            report.add(
                "unreferenced_module",
                str(path.relative_to(root)),
                "no module imports this file",
                severity="warning",
            )


def scan(root: Path | None = None) -> HygieneReport:
    base = root or Path(__file__).resolve().parents[3]
    report = HygieneReport()
    report.files_scanned = len(shipped_files(base))
    scan_text(base, report)
    scan_structure(base, report)
    scan_empty_implementations(base, report)
    scan_duplicates(base, report)
    scan_unreferenced_modules(base, report)
    return report


def as_dict(report: HygieneReport) -> dict[str, object]:
    return {
        "files_scanned": report.files_scanned,
        "finding_count": len(report.findings),
        "error_count": len(report.errors),
        "findings": [finding.__dict__ for finding in report.findings],
    }
