# Work 6 grouped-fold contract

This directory documents the code contract only. No real folds are generated
or frozen until the Work 5 provenance, duplicate, and shortcut audit has been
reviewed.

## Why all historical calls are inputs

The graph is constructed over historical train, validation, **and burned test**
calls before eligibility is decided. If any call is connected to a burned-test
call by an accepted relation, the entire connected component is quarantined.
The burned partition is never emitted as `test` or `confirmatory`; the only
outer development role is `outer_eval`.

## Canonical call-provenance JSONL

One row per call is required:

```json
{"call_id":"vishing_001","label":1,"historical_split":"train","source":"FSS","audio_sha256":"64-lowercase-hex-or-unknown"}
```

Required fields are `call_id`, binary `label`, and `historical_split`. The
canonical split values are `train`, `val`, and `test`; `test` means the burned
historical test. `source` and `audio_sha256` are strongly recommended.
Unknown provenance must be written explicitly (for example, `"source":
"unknown"`), not reconstructed from memory. For transition from Work 5, the
parser also accepts documented aliases such as `original_split`, `split`,
`waveform_sha256`, and `exact_audio_sha256`.

The actual Work 5 schema stores reviewed fields at the top level
(`is_burned`, `relation_component_id`, `canonical_call_id`, and
`provisional_primary_status`) and audio equivalence hashes under
`audio.byte_sha256` and `audio.pcm_sha256`. Work 6 validates those fields and
uses both file-byte and decoded-PCM equivalence. The only accepted final Work 5
statuses are `historical_burned`, `quarantine_burned_related_component`,
`exclude_redundant_exact_audio`, and `eligible_for_work6`; a pending or unknown
status stops fold generation.

## Canonical relation JSONL

The preferred Work 5 output is one accepted hard component per row:

```json
{"component_id":"template_004","members":["call_a","call_b"],"accepted_relation_types":["exact_transcript"],"status":"accepted"}
```

Pairwise edges are also accepted using `call_id_a`/`call_id_b`,
`left_call_id`/`right_call_id`, or `source_call_id`/`target_call_id`. Rows with
`include_in_grouping: false` or an explicitly rejected/excluded status are
ignored. All other rows are conservatively treated as grouping relations.
Every member must exist in call provenance; an unknown member is a hard error
so a burned connection cannot be silently lost.

The reviewed Work 5 `group_components.jsonl` may include singleton components.
They are validated as identity components and add no graph edge. Aggregate
fields (`member_count`, labels, sources, historical splits, burned members,
`touches_burned`, and development quarantine members) are cross-checked against
call provenance. Once aggregate component rows are detected, they must cover
every provenance call exactly once. Edge-form relation rows still require two
members.

Exact-audio components should contain only an exact type (`exact_audio`,
`byte_identical_audio`, `audio_sha256`, or an accepted alias). Do not combine
exact-audio and softer relation types in one aggregate row: exact equivalence
classes are deduplicated, while all accepted relations are used for grouping.
The provenance `audio_sha256` is independently used to recover exact classes.

## Segment-manifest input

Supply the historical train, validation, and burned-test segment JSONL files.
Every provenance call must occur at least once, every `segment_id` must be
globally unique, and the segment label must match call provenance. Rows are
copied to the v2 manifests and augmented with `group_id`, partition identifiers,
and an inverse-segments-per-call `sample_weight`.

## Deterministic command (run from `Multimodal/`)

```bash
PYTHONPATH=revised_experiments/src \
/home/mbmk92/.virtualenvs/vishing/bin/python \
  revised_experiments/scripts/build_grouped_folds.py \
  --call-provenance revised_experiments/artifacts/work5/call_provenance.jsonl \
  --relations revised_experiments/artifacts/work5/relation_components.jsonl \
  --segments data/train_segment_manifest_merged.jsonl \
  --segments data/val_segment_manifest_merged.jsonl \
  --segments data/test_segment_manifest_merged.jsonl \
  --output-dir data/paper2_v2/folds/grouped_nested_v1
```

The destination must not already exist. This protects a reviewed fold set from
silent replacement. The generator uses seed `20260809`, repeat `00`, five
stratified group-disjoint outer partitions, and a deterministic group-disjoint
one-fifth validation partition inside each outer training pool.

## Outputs

```text
grouped_nested_v1/
  protocol.json
  assignments.jsonl
  SHA256SUMS.txt
  repeat_00/
    outer_00/ ... outer_04/
      train_call_manifest.jsonl
      train_segment_manifest.jsonl
      val_call_manifest.jsonl
      val_segment_manifest.jsonl
      outer_eval_call_manifest.jsonl
      outer_eval_segment_manifest.jsonl
```

`assignments.jsonl` is the exclusion inventory consumed by Work 8. It records
eligible, exact-duplicate-excluded, and quarantined calls. `protocol.json`
contains immutable input hashes, algorithm identifiers, class/group/segment
counts, and fold paths. `SHA256SUMS.txt` covers all other output files.

The initial status is `generated_unfrozen`. Generate a second tree at a
temporary destination and compare the complete tree digest/bytes. After human
review verifies Work 5 decisions and the Work 6 completion report, regenerate
the canonical path with `--freeze` (or call `freeze_fold_tree` on the already
verified tree). Freezing sets the protocol status, adds a deterministic notice,
regenerates and verifies `SHA256SUMS.txt`, and changes the tree to read-only.
Only frozen manifest hashes may be inserted into the locked Work 8
configuration.
