# Paper 2 Revised Experiments (v2)

This directory contains the corrected, preservation-safe implementation of the
Paper 2 within-corpus experiment protocol.

The historical implementation in `Multimodal/modeling/`, its checkpoints,
features, manifests, and result files are treated as **Exploratory v1**. The v2
pipeline must not overwrite them.

## Design rules

1. All generated runs use unique IDs and immutable output directories.
2. Every run validates against the schemas in `registry/schemas/`.
3. Every prediction is saved at segment and call level.
4. The historical 142-call test is burned and is not used for v2 model,
   threshold, or narrative selection.
5. Correctness tests must pass before training.
6. Paper 2 is currently scoped as a source-aware within-corpus methodological
   study; see `../paper2/WITHIN_CORPUS_SCOPE_DECISION_2026-08-09.md`.

## Layout

```text
scripts/              reproducibility and audit utilities
registry/schemas/     run and prediction JSON Schemas
registry/templates/   preregistration/experiment-card templates
src/paper2_v2/        corrected data, model, training, and evaluation code
tests/                correctness and protocol tests
configs/              frozen experiment configurations
artifacts/            generated v2 audit/fold/run artifacts (never v1 data)
```

Commands are run from `Multimodal/` with the project virtual environment:

```bash
source /home/mbmk92/.virtualenvs/vishing/bin/activate
```

## Work 5 data audit

The frozen Work 5 pipeline recovers sanitized source provenance, verifies the
FSS sequential-rename mapping, hashes exact audio/text duplicates, creates a
reviewed relation graph, marks components connected to the burned historical
test, and records descriptive shortcut signals:

```bash
python revised_experiments/scripts/run_work5_audit.py
```

It writes a new immutable directory at
`revised_experiments/artifacts/work5_data_audit_v1/`. It refuses to overwrite
an existing artifact. Original FSS filenames, external record paths, and
per-call AIHub demographics are withheld from the public JSON/JSONL files.

The automatic audit intentionally leaves one borderline template candidate
pending. Its redacted human review is sealed separately in
`artifacts/work5_manual_review_v1/`. Work 6 consumes the base hard edges plus
addendum rows having `include_in_grouping=true`; neither artifact is rewritten
after review.

After Work 6 is frozen, run the learned shortcut diagnostics on its canonical
grouped partitions:

```bash
python revised_experiments/scripts/run_learned_shortcut_probes.py
```

The text vectorizer, metadata scaler, and logistic models fit on each outer
training partition only. Thresholds come from that fold's validation
partition; each `outer_eval` call is scored once. Results are sealed as
diagnostics at
`artifacts/work5_learned_shortcut_probes_grouped_v1/` and are forbidden from
changing Work 6 or Work 8. The checksum-bound, read-only interpretation is in
`artifacts/work5_learned_shortcut_probe_report_v1/`.

## Work 9 Gate-1 post-run analysis

After all 200 primary-matrix selections and outer evaluations are complete,
freeze the canonical Work 9 outer-ledger index. Gate-1 then consumes that JSON
index and its bound selection index directly; it never trains, selects, or
opens an unindexed prediction file:

```bash
python revised_experiments/scripts/run_work9_gate1_analysis.py \
  --outer-ledger-index revised_experiments/artifacts/work9_runs_v1/_matrix/outer_ledger_index.json \
  --outer-ledger-index-sha256 <precommitted-lowercase-sha256> \
  --output-dir revised_experiments/artifacts/work9_gate1_v1
```

The analyzer reconstructs every validation choice and complete-call mean,
verifies every immutable reference and frozen manifest identity, checks the
random-train pairing ledgers against the Work 8 receipt, recomputes outer
metrics, and applies the preregistered relation-group clustered bootstrap,
permutation, Holm, and continuation rules. Its report is statistical and
within-corpus only; no practical-utility claim is authorized without a frozen
cost policy. The exact percentile/linear confidence interval, two-sided
permutation, finite-sample correction, seeds, and 10,000-replicate choices are
frozen before outcomes in
`configs/work9_gate1_analysis_protocol_v1.json` and checksum-bound into every
analysis output.
