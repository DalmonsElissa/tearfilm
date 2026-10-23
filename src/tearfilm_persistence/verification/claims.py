"""Manuscript claims and the released symbol that carries each one.

Every entry names a location in the article and the object in this package that
implements the statement. A claim whose statement is a recorded measurement
cannot be closed from code alone: it is marked ``BLOCKED`` with the symbol that
would produce it once the cohort is staged.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Literal

Status = Literal["PASS", "FAIL", "NOT_RUN", "BLOCKED"]


@dataclass(frozen=True)
class Claim:
    id: str
    location: str
    statement: str
    symbols: tuple[str, ...] = ()
    family: str = "code"
    block_reason: str = ""


@dataclass
class ClaimResult:
    id: str
    location: str
    statement: str
    family: str
    symbols: list[str]
    missing: list[str]
    status: Status
    details: list[str] = field(default_factory=list)


def resolve_symbol(symbol: str) -> object:
    if ":" not in symbol:
        raise ValueError(f"symbol must be 'module:attribute', got {symbol}")
    module_name, attribute = symbol.split(":", 1)
    module = importlib.import_module(module_name)
    current: object = module
    for part in attribute.split("."):
        current = getattr(current, part)
    return current


P = "tearfilm_persistence"

CLAIMS: tuple[Claim, ...] = (
    Claim(
        "C001",
        "Abstract, p. 1",
        "The acquisition source is a robotically acquired anterior-segment stream repurposed into "
        "a tear-film measurement.",
        (f"{P}.streams.acquisition:AcquisitionClip",),
    ),
    Claim(
        "C002",
        "Abstract, p. 1",
        "Break-up is declared only when the deformation derivative stays sign-consistent over a "
        "persistence window.",
        (f"{P}.deformation.persistence:hard_occupancy",),
    ),
    Claim(
        "C003",
        "Abstract, p. 1",
        "The descriptor's coefficient of variation is 34.2% against 51.90 to 65.49% for the "
        "reference.",
        (f"{P}.descriptors.repeatability:coefficient_of_variation",),
        family="value",
        block_reason="record-level repeatability requires the private cohort",
    ),
    Claim(
        "C004",
        "Abstract, p. 1",
        "The false early rate falls from 4.4% to 0.8%.",
        (f"{P}.evaluation.properties:false_early_rate",),
        family="value",
        block_reason="record-level onset pairs require the private cohort",
    ),
    Claim(
        "C005",
        "Abstract, p. 1",
        "Clinician review raises the AUC from 0.852 to 0.936 and cuts reading by 17.0%.",
        (f"{P}.evaluation.reader_study:orc_difference",),
        family="value",
        block_reason="reader-study readings are not distributed",
    ),
    Claim(
        "C006",
        "Sec. 1, p. 2",
        "The reference instrument carries a coefficient of repeatability larger than the clinical "
        "decision window.",
        (f"{P}.descriptors.repeatability:COR_FACTOR",),
        family="design",
    ),
    Claim(
        "C007",
        "Sec. 1, p. 2",
        "A false early break-up moves a mild case into the severe category, so the error that "
        "matters is not per-frame perception.",
        (f"{P}.objectives.composite:decision_term",),
        family="design",
    ),
    Claim(
        "C008",
        "Sec. 1, p. 3",
        "Sampling rate is a floor rather than a gradient: below it an acquisition is invalid, not "
        "degraded.",
        (f"{P}.streams.adequacy:SAMPLING_FLOOR_HZ",),
        family="design",
    ),
    Claim(
        "C009",
        "Sec. 2.1, p. 3",
        "The retrospective arm covers four sites in two regions.",
        (f"{P}.cohort.manifest:CohortTable.audit",),
        family="value",
        block_reason="site composition is a property of the private table",
    ),
    Claim(
        "C010",
        "Sec. 2.1, p. 3",
        "Acquisition adequacy is the joint sampling and fixation-stability criterion.",
        (f"{P}.streams.adequacy:assess",),
    ),
    Claim(
        "C011",
        "Sec. 2.2, p. 4",
        "The descriptor's within-subject standard deviation, coefficient of repeatability and "
        "coefficient of variation are reported before discrimination.",
        (f"{P}.descriptors.repeatability:summarise_repeats",),
    ),
    Claim(
        "C012",
        "Table 1, p. 4",
        "Repeatability is reported as 2.77 Sw.",
        (f"{P}.descriptors.repeatability:COR_FACTOR",),
        family="design",
    ),
    Claim(
        "C013",
        "Table 1, p. 4",
        "Agreement is reported with bias, limits of agreement and an intraclass correlation.",
        (f"{P}.descriptors.repeatability:bland_altman",),
    ),
    Claim(
        "C014",
        "Sec. 2.2, p. 4",
        "Descriptor variance decomposes into a between-site term, a between-session term and an "
        "intra-eye term.",
        (f"{P}.descriptors.repeatability:variance_shares",),
    ),
    Claim(
        "C015",
        "Fig. 1, p. 5",
        "The descriptor sits between the structural instruments and the incumbent dynamic "
        "reference on the error axis.",
        (f"{P}.descriptors.repeatability:summarise_repeats",),
        family="value",
        block_reason="figure values are derived from the private repeats",
    ),
    Claim(
        "C016",
        "Sec. 2.3, p. 5",
        "Discrimination is measured on a held-out multi-centre split that took no part in "
        "training.",
        (f"{P}.cohort.manifest:SPLITS", f"{P}.runtime.loop:score_loader"),
        family="design",
    ),
    Claim(
        "C017",
        "Sec. 2.3, p. 5",
        "The constrained configuration is at parity on primary utility and carries its advantage "
        "on the constraint columns.",
        (f"{P}.models.teardyn:TearDynModel",),
        family="value",
        block_reason="parity is a property of the fitted models on the private split",
    ),
    Claim(
        "C018",
        "Sec. 2.3, p. 5",
        "The frame-pair Siamese and the plain frame classifier are the two published comparators "
        "the model is measured against.",
        (
            f"{P}.comparators.siamese_pair:SiamesePair",
            f"{P}.comparators.frame_classifier:FrameClassifier",
        ),
    ),
    Claim(
        "C019",
        "Sec. 2.3, p. 5",
        "The false early rate is the share of eyes declared at least 3 s before the reference.",
        (f"{P}.evaluation.properties:EARLY_MARGIN_SECONDS",),
        family="design",
    ),
    Claim(
        "C020",
        "Sec. 2.3, p. 5",
        "Cross-site change is the AUC change when a model trained at three sites is applied to a "
        "fourth.",
        (f"{P}.evaluation.properties:cross_site_change",),
    ),
    Claim(
        "C021",
        "Table 2, p. 6",
        "Intervals are patient-level cluster bootstraps.",
        (f"{P}.evaluation.discrimination:cluster_bootstrap",),
    ),
    Claim(
        "C022",
        "Table 2, p. 6",
        "The proposed method is a constrained configuration at parity on primary utility.",
        (f"{P}.models.teardyn:ModelConfig",),
        family="design",
    ),
    Claim(
        "C023",
        "Table 2, p. 6",
        "Alternative-data rows are the non-invasive dynamic reference and clinician slit-lamp "
        "grading.",
        (f"{P}.evaluation.aggregate:ScoredEye",),
        family="design",
    ),
    Claim(
        "C024",
        "Table 2, p. 6",
        "Five independent initialisations give 0.944 with a largest row-to-row dispersion of "
        "0.004.",
        (f"{P}.runtime.seeding:set_seed",),
        family="value",
        block_reason="the five fitted runs require the private split",
    ),
    Claim(
        "C025",
        "Table 2, p. 6",
        "The proposed configuration costs 78 GFLOPs and 164 ms per clip.",
        (f"{P}.evaluation.budget:DECLARED_PROFILE",),
        family="design",
    ),
    Claim(
        "C026",
        "Table 2, p. 6",
        "The unconstrained attention baseline costs 214 GFLOPs and 268 ms per clip.",
        (f"{P}.evaluation.budget:UNCONSTRAINED_GFLOPS",),
        family="design",
    ),
    Claim(
        "C027",
        "Sec. 2.3, p. 5",
        "Removing the fast deformation branch costs 0.047 in AUC.",
        (f"{P}.evaluation.ablation:delta",),
        family="value",
        block_reason="component deltas require the five fitted ablations",
    ),
    Claim(
        "C028",
        "Sec. 2.3, p. 5",
        "Removing the slow persistence branch costs 0.032 and lifts the false early rate from "
        "0.8% to 4.4%.",
        (f"{P}.evaluation.ablation:delta",),
        family="value",
        block_reason="component deltas require the five fitted ablations",
    ),
    Claim(
        "C029",
        "Sec. 2.3, p. 5",
        "Removing the structural branch costs 0.008 and removing the adequacy gate 0.003.",
        (f"{P}.evaluation.ablation:deltas",),
        family="value",
        block_reason="component deltas require the five fitted ablations",
    ),
    Claim(
        "C030",
        "Sec. 2.3, p. 5",
        "The repeatability-aware head buys calibration rather than discrimination: 0.004 in AUC "
        "but 0.031 to 0.052 in calibration error.",
        (f"{P}.descriptors.calibration:VarianceComponentCalibrator",),
        family="value",
        block_reason="calibration error is measured on the private split",
    ),
    Claim(
        "C031",
        "Sec. 2.3, p. 5",
        "Removing both temporal branches returns 0.838 against a sum of individual losses of "
        "0.079, an interaction ratio of 1.34.",
        (f"{P}.evaluation.ablation:interaction_ratio",),
    ),
    Claim(
        "C032",
        "Sec. 2.3, p. 6",
        "Substituting mean pooling for the persistence branch returns 0.903.",
        (f"{P}.models.teardyn:ModelConfig",),
        family="value",
        block_reason="substitution rows require the fitted ablation",
    ),
    Claim(
        "C033",
        "Sec. 2.3, p. 6",
        "Substituting a 2019-generation 3D convolutional backbone for the fast branch returns "
        "0.918.",
        (f"{P}.deformation.adapter:PlainClipBackbone",),
        family="value",
        block_reason="substitution rows require the fitted ablation",
    ),
    Claim(
        "C034",
        "Sec. 2.3, p. 6",
        "The attentional baseline retrained under a matched budget reaches 0.929 with a false "
        "early rate of 3.6%.",
        (f"{P}.comparators.video_transformer:VideoTransformer",),
        family="value",
        block_reason="matched-budget retraining requires the private split",
    ),
    Claim(
        "C035",
        "Sec. 2.4, p. 7",
        "The model's added value is largest where pre-intervention inter-clinician agreement "
        "is lowest.",
        (f"{P}.evaluation.reader_study:experience_groups",),
        family="value",
        block_reason="tertiles are defined on the private reader study",
    ),
    Claim(
        "C036",
        "Sec. 2.4, p. 7",
        "Site-level results are reported per site and pooled, with between-site heterogeneity.",
        (f"{P}.evaluation.aggregate:site_table", f"{P}.evaluation.aggregate:heterogeneity"),
    ),
    Claim(
        "C037",
        "Table 4, p. 9",
        "Dry eye is defined by a break-up time at or below 5 s together with a symptom score "
        "above 13.",
        (f"{P}.cohort.manifest:ClipRecord",),
        family="design",
    ),
    Claim(
        "C038",
        "Table 4, p. 9",
        "Severity kappa is reported within each severity stratum as well as per site.",
        (f"{P}.evaluation.agreement:severity_kappa",),
    ),
    Claim(
        "C039",
        "Sec. 2.4, p. 8",
        "Discrimination falls from the mild to the severe stratum.",
        (f"{P}.evaluation.aggregate:stratum_table",),
        family="value",
        block_reason="stratum results require the private table",
    ),
    Claim(
        "C040",
        "Sec. 2.4, p. 9",
        "The failure taxonomy has three categories: residual post-blink artefacts, marginal "
        "fixation stability, and concurrent ocular-surface pathology.",
        (f"{P}.models.teardyn:artefact_fraction",),
        family="design",
    ),
    Claim(
        "C041",
        "Sec. 2.5, p. 10",
        "The completed prospective arm reproduces the retrospective result under a locked model.",
        (f"{P}.cohort.manifest:ARMS",),
        family="value",
        block_reason="prospective records are not distributed",
    ),
    Claim(
        "C042",
        "Table 5, p. 10",
        "Adequacy is the proportion of attempted acquisitions meeting the joint criterion, and "
        "abstentions are excluded from performance estimates.",
        (f"{P}.evaluation.aggregate:scorable",),
    ),
    Claim(
        "C043",
        "Sec. 2.6, p. 11",
        "The reader study is a three-arm fully crossed design with a five-week washout.",
        (f"{P}.evaluation.reader_study:arm_auc_table",),
        family="design",
    ),
    Claim(
        "C044",
        "Sec. 2.6, p. 11",
        "The assisted-minus-unaided difference is tested with readers and cases as random factors.",
        (f"{P}.evaluation.reader_study:orc_difference",),
    ),
    Claim(
        "C045",
        "Sec. 2.6, p. 11",
        "Reliance and under-reliance are measured over the discordant reads.",
        (f"{P}.evaluation.reader_study:reliance_summary",),
    ),
    Claim(
        "C046",
        "Sec. 2.6, p. 11",
        "The gain is largest for the least experienced readers.",
        (f"{P}.evaluation.reader_study:experience_groups",),
        family="value",
        block_reason="reader experience is recorded in the private study",
    ),
    Claim(
        "C047",
        "Table 6, p. 12",
        "Reading time falls from 62.4 s to 51.8 s per case.",
        (f"{P}.evaluation.discrimination:paired_difference",),
        family="value",
        block_reason="reading times are recorded in the private study",
    ),
    Claim(
        "C048",
        "Table 6, p. 12",
        "Inter-reader kappa rises from 0.61 to 0.74.",
        (f"{P}.evaluation.agreement:weighted_kappa",),
        family="value",
        block_reason="reader ratings are not distributed",
    ),
    Claim(
        "C049",
        "Sec. 2.7, p. 11",
        "The locked model runs in silent mode alongside the standard assessment.",
        (f"{P}.models.gate:AdequacyGate",),
        family="design",
    ),
    Claim(
        "C050",
        "Sec. 2.7, p. 11",
        "Abstention is 2.5% and requires no additional clinical action.",
        (f"{P}.evaluation.aggregate:ArmReport",),
        family="value",
        block_reason="abstention counts require the prospective records",
    ),
    Claim(
        "C051",
        "Sec. 2.7, p. 12",
        "The clinical threshold is placed from the relative cost of the two error paths, not by "
        "aggregating them.",
        (f"{P}.evaluation.discrimination:operating_point",),
    ),
    Claim(
        "C052",
        "Sec. 2.7, p. 13",
        "Recalibration drift over the prospective accrual is bounded by a single estimate.",
        (f"{P}.descriptors.calibration:LogisticCalibrator",),
        family="value",
        block_reason="drift requires the prospective time series",
    ),
    Claim(
        "C053",
        "Table 7, p. 16",
        "Each expert case is reported with declared onset, peak affected fraction, reference "
        "onset and outcome.",
        (f"{P}.descriptors.vector:DescriptorVector",),
        family="design",
    ),
    Claim(
        "C054",
        "Sec. 2.8, p. 14",
        "Case 3 abstains because fixation stability sits outside the acceptance band.",
        (f"{P}.streams.adequacy:assess",),
        family="value",
        block_reason="case-level acquisition is not distributed",
    ),
    Claim(
        "C055",
        "Sec. 2.8, p. 14",
        "Case 5 is a genuine false early break-up from a residual post-blink speck.",
        (f"{P}.models.teardyn:artefact_fraction",),
        family="value",
        block_reason="case-level acquisition is not distributed",
    ),
    Claim(
        "C056",
        "Sec. 2.9, p. 15",
        "The mechanistic arm compares hyperosmolar and iso-osmolar corneal epithelial cultures.",
        (f"{P}.evaluation.mechanistic:percent_change",),
        family="design",
    ),
    Claim(
        "C057",
        "Sec. 2.9, p. 15",
        "Secreted mediator changes are reported with 95% intervals over three independent "
        "experiments.",
        (f"{P}.evaluation.mechanistic:EffectEstimate",),
        family="value",
        block_reason="the replicate table is not distributed",
    ),
    Claim(
        "C058",
        "Sec. 5.1, p. 20",
        "The study has two clinical arms and one non-comparative external cohort.",
        (f"{P}.cohort.manifest:ARMS",),
        family="design",
    ),
    Claim(
        "C059",
        "Sec. 5.1, p. 21",
        "The partition is implemented at the patient level.",
        (f"{P}.cohort.manifest:ensure_patient_level",),
    ),
    Claim(
        "C060",
        "Sec. 5.2, p. 21",
        "The platform uses an axial resolution around 15 micrometres.",
        (f"{P}.structure.meniscus:AXIAL_UM",),
        family="design",
    ),
    Claim(
        "C061",
        "Sec. 5.2, p. 21",
        "The structural descriptor resolves the meniscus into 25 to 100 elements.",
        (f"{P}.streams.masking:AreaConfig",),
        family="design",
    ),
    Claim(
        "C062",
        "Sec. 5.2, p. 21",
        "The adequacy floor is 30 frames per second and is an inclusion criterion.",
        (f"{P}.streams.adequacy:AdequacyConfig",),
        family="design",
    ),
    Claim(
        "C063",
        "Sec. 5.3, p. 21",
        "The reference standards are non-invasive break-up time, meniscus height and volume, "
        "staining grade and the Ocular Surface Disease Index.",
        (f"{P}.cohort.manifest:ClipRecord",),
        family="design",
    ),
    Claim(
        "C064",
        "Sec. 5.4, p. 21",
        "The objective has three terms: a break-up decision, an onset regression and a "
        "repeatability penalty.",
        (
            f"{P}.objectives.composite:decision_term",
            f"{P}.objectives.composite:onset_term",
            f"{P}.objectives.composite:repeatability_term",
        ),
    ),
    Claim(
        "C065",
        "Sec. 5.4, p. 22",
        "The fast branch reads at most five consecutive frames.",
        (f"{P}.deformation.adapter:AdapterConfig",),
        family="design",
    ),
    Claim(
        "C066",
        "Sec. 5.4, p. 22",
        "The derivative is D_t = d_t phi(x_t).",
        (f"{P}.deformation.derivative:TemporalDerivative",),
    ),
    Claim(
        "C067",
        "Sec. 5.4, p. 22",
        "The persistence operator keeps loci whose derivative sign is constant on the window and "
        "only after the window has expired.",
        (f"{P}.deformation.persistence:PersistenceOperator",),
    ),
    Claim(
        "C068",
        "Sec. 5.4, p. 22",
        "Break-up is declared at the first moment the sign-consistent area reaches alpha of the "
        "examined area.",
        (f"{P}.deformation.persistence:first_passage",),
    ),
    Claim(
        "C069",
        "Sec. 5.4, p. 22",
        "The instability curve is the occupied fraction of the examined area.",
        (f"{P}.deformation.persistence:area_fraction",),
    ),
    Claim(
        "C070",
        "Sec. 5.4, p. 23",
        "The persistence window is fixed at 0.17 s.",
        (f"{P}.deformation.persistence:TAU_SECONDS",),
        family="design",
    ),
    Claim(
        "C071",
        "Sec. 5.4, p. 23",
        "The descriptor is the six-tuple of onset, peak instability, slope, locus entropy, "
        "meniscus height and meniscus area.",
        (f"{P}.descriptors.vector:COMPONENTS",),
        family="design",
    ),
    Claim(
        "C072",
        "Sec. 5.4, p. 23",
        "The descriptor head adjusts the measurement with variance components under one standard "
        "across sites.",
        (f"{P}.descriptors.calibration:VarianceComponentCalibrator",),
    ),
    Claim(
        "C073",
        "Algorithm 1, p. 24",
        "Adequacy is evaluated before any inference, and the artefact channel is produced inside "
        "the blink-to-blink loop.",
        (f"{P}.models.gate:AdequacyGate", f"{P}.models.teardyn:artefact_fraction"),
    ),
    Claim(
        "C074",
        "Sec. 5.5, p. 23",
        "Calibration follows descriptor collection and is fitted on repeated measurements of the "
        "same eyes.",
        (
            f"{P}.descriptors.calibration:fit_calibration",
            f"{P}.descriptors.calibration:calibration_report",
        ),
    ),
    Claim(
        "C075",
        "Table 8, p. 24",
        "The inference path is decomposed into six stages that sum to the reported total.",
        (f"{P}.evaluation.budget:profile_checks",),
    ),
    Claim(
        "C076",
        "Sec. 5.7, p. 23",
        "Twelve readers in three experience groups read every case twice under a five-week "
        "washout.",
        (f"{P}.evaluation.reader_study:experience_groups",),
        family="design",
    ),
    Claim(
        "C077",
        "Sec. 5.8, p. 25",
        "Holm's step-down method controls multiplicity over the primary criteria.",
        (f"{P}.evaluation.multiplicity:holm_stepdown",),
    ),
    Claim(
        "C078",
        "Sec. 5.9, p. 25",
        "The four pre-specified criteria are an assisted AUC of at least 0.92, a gain of at least "
        "0.05, a false early rate of at most 1.0% and a coefficient of variation of at most 45%.",
        (f"{P}.evaluation.criteria:CriteriaThresholds",),
        family="design",
    ),
    Claim(
        "C079",
        "Sec. 5.10, p. 26",
        "The mechanistic arm is in vitro only, with no animal work and no new human specimens.",
        (f"{P}.evaluation.mechanistic:summarise",),
        family="design",
    ),
    Claim(
        "C080",
        "Data availability, p. 26",
        "The private clinical cohort cannot be redistributed; the derived quantities are "
        "available on request.",
        (f"{P}.cohort.manifest:load_table",),
        family="design",
    ),
)


def evaluate_claims(claims: tuple[Claim, ...] = CLAIMS) -> list[ClaimResult]:
    results: list[ClaimResult] = []
    for claim in claims:
        missing: list[str] = []
        for symbol in claim.symbols:
            try:
                resolve_symbol(symbol)
            except (ImportError, AttributeError, ValueError):
                missing.append(symbol)
        if missing:
            status: Status = "FAIL"
        elif claim.family == "value":
            status = "BLOCKED"
        else:
            status = "PASS"
        details = [claim.block_reason] if status == "BLOCKED" else []
        results.append(
            ClaimResult(
                id=claim.id,
                location=claim.location,
                statement=claim.statement,
                family=claim.family,
                symbols=list(claim.symbols),
                missing=missing,
                status=status,
                details=details,
            )
        )
    return results
