# Repurposing Robotic Anterior Segment Imaging for Tear Film Assessment in Dry Eye with Temporal Models

## Overview

The measurement this release implements is a persistence test. A frame is mapped to a signed
tear-film deformation field, the field is differentiated in time over at most five frames,
and a locus counts as breaking up once its derivative has held one sign for longer than the
persistence window. The occupied share of the examined area is the instability curve, and its
first passage through a threshold is the declared onset.

The rest of the package supports that measurement: the acquisition gate that decides whether
a stream may be measured, the structural branch that separates a shallow tear reservoir from a
rapidly destabilising film, the calibration that shrinks a reading by how well the same eye
repeats, the six comparators the detection table reports, the reader-study statistics, and two
verification layers that check the code against the article and then check that the code runs.

The code follows the measurement chain rather than a training script. `streams` holds what
comes off the device, `deformation` holds the operator, `structure` holds the single-frame
geometry, `descriptors` holds the vector and its calibration, `models` assembles the chain, and
`evaluation` turns scored eyes into the rows the article prints.

No clinical record is shipped, rebuilt or inferred. The multi-centre cohort, the external
cohort, the prospective arm and the reader study are private and de-identified at patient and
institution level. Every command that reads records takes a table from the caller and stops
without one. No record-level number in the article is claimed here: each is reported `BLOCKED`
with its staging condition in `verification_report.json`. The execution checks use tensors built
inside the check process.

## What is in the tree

| path | contents |
| --- | --- |
| `claim_to_code.json` | eighty article claims, each with its location, the released symbol that carries it, and its status |
| `verification_report.json` | both verification layers, the manuscript arithmetic, the configuration conformance, the hygiene scan and every blocked record-level check |
| `verification_summary.txt` | the same verdicts as plain text |
| `dataset_urls.txt` | the adjudicated data endpoint, its licence, and what the probe returned |
| `integrity_manifest.json` | one SHA-256 per shipped file, written last |
| `THIRD_PARTY_NOTICES.txt` | the dataset attribution and the publications the comparator designs follow |

## Installation

```bash
# pip
python3 -m pip install -r requirements.txt -r requirements-dev.txt
export PYTHONPATH="$PWD/src"

# conda
conda env create -f environment.yml
conda activate tearfilm-persistence

# container
docker build -t tearfilm-persistence .
```

The package is not installed into the environment by default. `PYTHONPATH=src` is what the
scripts use and what `pyproject.toml` sets for the test runner.

## Data

One open resource is used, for representation pretraining and to initialise the structural
branch:

| source | version | licence | size | role |
| --- | --- | --- | --- | --- |
| Multicentre pixel-level tear-meniscus segmentation dataset, `10.6084/m9.figshare.28650536.v2` | Figshare v2 | CC BY 4.0 | 4.57 GB | 1,693 colour and 1,739 infrared images with meniscus masks |

The clinical streams are not distributed. To run anything that touches records, stage them
yourself and describe them in a JSONL table, one object per acquisition:

```bash
bash scripts/stage_data.sh /path/to/clips /path/to/cohort.jsonl
```

Each line carries `patient_id`, `eye_id`, `site`, `region`, `session`, `repeat`, `arm`,
`split`, `path` and, where the reference recorded one, `reference_onset`, `reference_breakup`,
`symptom_score`, `severity` and `device_generation`. `path` points at an `.npz` holding
`frames` (`[T, 3, H, W]` in the unit interval), `fps` and `fixation` (`[T, 2]` normalised
centroids), with an optional `.json` sidecar of the same fields. `split` is one of `train`,
`validation`, `test`, `external`, `prospective`.

The table is partitioned by patient. `tearfilm_persistence.cohort.manifest.ensure_patient_level`
refuses a table that puts one patient on both sides of a split, and the split audit is recorded
with every run.

## Running an experiment

There is one file per reported configuration under `configs/experiments/`. A file names the
component files it builds on and overrides only what it changes.

```bash
# the constrained configuration
bash scripts/train.sh configs/experiments/main.yaml /path/to/cohort.jsonl

# the five component ablations and the two substitutions
bash scripts/train.sh configs/experiments/ablation_minus_fast.yaml /path/to/cohort.jsonl
bash scripts/train.sh configs/experiments/ablation_minus_persistence.yaml /path/to/cohort.jsonl
bash scripts/train.sh configs/experiments/ablation_minus_structural.yaml /path/to/cohort.jsonl
bash scripts/train.sh configs/experiments/ablation_minus_repeatability_head.yaml /path/to/cohort.jsonl
bash scripts/train.sh configs/experiments/ablation_minus_adequacy_gate.yaml /path/to/cohort.jsonl
bash scripts/train.sh configs/experiments/ablation_persistence_mean_pool.yaml /path/to/cohort.jsonl
bash scripts/train.sh configs/experiments/ablation_fast_conv3d.yaml /path/to/cohort.jsonl
bash scripts/train.sh configs/experiments/ablation_both_temporal_removed.yaml /path/to/cohort.jsonl

# two of the six comparison rows, trained under the same partition and budget
bash scripts/train.sh configs/experiments/baseline_siamese_pair.yaml /path/to/cohort.jsonl
bash scripts/train.sh configs/experiments/baseline_video_transformer.yaml /path/to/cohort.jsonl
```

The four other baseline files follow the same pattern. A baseline file routes the factory to
the comparator named under `model.comparator`, wrapped into the same batch interface so that
partition, preprocessing and scoring are shared.

## Reading a result

```bash
bash scripts/evaluate.sh configs/experiments/main.yaml /path/to/cohort.jsonl \
    artifacts/model.pt test evaluation
```

The report holds the arm summary, the per-site rows, the severity strata and the abstention
count. It does not hold an expected value, because no expected value can be stated without the
cohort. Each number the article prints is mapped to the symbol that would produce it and marked
`BLOCKED` in `claim_to_code.json`; once the table is staged, the command above produces that
number and the comparison becomes available.

The descriptor calibration is a separate, post-collection step, because Sec. 5.5 fits it on
repeated acquisitions and applies it afterwards. `evaluate` can dump the per-eye descriptor
table, and `calibrate` fits the variance-component map and the logistic map on one table and
reports the calibration error on another:

```bash
python -m tearfilm_persistence.cli.evaluate \
    --experiment configs/experiments/main.yaml --cohort /path/to/cohort.jsonl \
    --checkpoint artifacts/model.pt --split validation --name validation \
    --dump-descriptors artifacts/validation.jsonl

python -m tearfilm_persistence.cli.calibrate \
    --fit artifacts/validation.jsonl --apply artifacts/test.jsonl
```

Single acquisitions are measured directly:

```bash
python -m tearfilm_persistence.cli.infer \
    --experiment configs/experiments/main.yaml \
    --clip /path/to/clip.npz --checkpoint artifacts/model.pt
```

The output carries the six-component descriptor, the declared onset, the artefact rate, the
sub-test count and, when the gate refuses the acquisition, the reason codes
(`sampling_rate_below_floor`, `fixation_stability_outside_band`, `clip_too_short`).

## Cost

The article prints the inference cost and nothing about the training machine: 78 GFLOPs and
164 ms per clip for the constrained configuration, 214 GFLOPs and 268 ms for the unconstrained
attention baseline, decomposed over six stages in Table 8. It does not print the accelerator
model or count, the clip resolution, wall-clock time or storage.

The released defaults are therefore defaults, not manuscript settings: batch 8 clips, gradient
accumulation 1, 40 epochs, AdamW at 3e-4 with weight decay 1e-2, cosine schedule after 500
warm-up steps, bf16, gradient clipping at 1.0, five initialisations. Both reports list them
under `engineering_defaults`. `scripts/train.sh` accepts overrides on the command line and
nothing in the release reduces them; the only file with smaller values is
`configs/experiments/_smoke.yaml`, marked test-only in its first line.

A full run needs the private cohort and cannot be timed here. On one laptop CPU, without a GPU
and without any data, `pytest -q` finishes 193 tests in well under a minute and
`scripts/verify.sh` writes all four artefacts in about 8 s. The inference cost above is the article's, not a
measurement of this machine.

## Where the code sits

| module | responsibility |
| --- | --- |
| `streams` | clip container, photometric blink segmentation, adequacy gate, locus grid |
| `deformation` | frame representation, temporal-difference adapter, causal derivative, persistence operator, curve statistics |
| `structure` | meniscus segmentation and the millimetre read-out at 15 um axial pitch |
| `descriptors` | six-component vector, variance-component calibration, repeatability statistics |
| `models` | the chain, the adequacy gate inside it, and the descriptor head |
| `comparators` | frame classifier, mean-pooled CNN, (2+1)D, frame-pair Siamese, frozen probe, divided space-time transformer |
| `objectives` | decision, onset and repeatability terms |
| `cohort` | manifest, patient-level partition audit, repeat grouping, batching |
| `evaluation` | discrimination, properties, agreement, reader study, multiplicity, criteria, budget, mechanistic arm |
| `runtime` | seeding, configuration, weight files, training and scoring loops |
| `verification` | claim map, execution layer, manuscript arithmetic, conformance, hygiene, report assembly |

## Verification

```bash
bash scripts/verify.sh
```

The pass writes the three reports and exits non-zero if a code check fails. It writes the
manifest twice, because the manifest digests the shipped tree and the first write changes it.

The two layers are separate:

- Layer one resolves every claim in `claim_to_code.json` against a live symbol. A claim whose
  statement is a recorded measurement is `BLOCKED`, never `PASS`.
- Layer two runs the mechanism: the derivative against an analytic ramp, the persistence
  operator against constructed sign patterns, the relaxed indicator against the hard one, every
  comparator's forward pass, one backward pass that must reach the adapter, the structural
  branch and the head, a single-batch overfit, a two-epoch loop that must lower the loss, a
  weight-file round-trip that must reproduce the logits bit for bit, and each statistic against
  a hand-computed case.

The manuscript arithmetic is reported beside them and kept out of the code verdict. Five
relationships inside the article's own tables do not close, and they are listed as findings
rather than repaired: the prose of Sec. 2.3 transposes the model and the attention baseline;
the adequacy rate is printed as 97.3% in Sec. 2.1 and 97.5% elsewhere; 97.3% plus the 2.3%
removed does not reach 100%; case 1's reference break-up time is 6 s in the prose and 3.8 s in
the case table; case 5's instability value is 0.48 in the prose and 0.38 in the case table.

## Limits

The release implements the measurement, not the cohort. It cannot reproduce a table without the
records and does not claim to. It provides a chain in which each step the article describes is
a named function with a paper location in its docstring, a claim map that says which function
carries which sentence, and execution checks that fail when the chain stops working.
