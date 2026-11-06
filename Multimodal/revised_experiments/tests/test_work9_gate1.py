from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from paper2_v2.conditions import CORE_CONDITIONS
from paper2_v2.config import PRIMARY_COMPARISONS, REPORTED_CALL_METRICS, THRESHOLD_TIE_BREAK
from paper2_v2.evaluation import compute_call_metrics
from paper2_v2.pairing import PairingItem, build_random_train_pairing
from paper2_v2.work9_acceptance import environment_lock
from paper2_v2.work9_types import SUPERSEDED_ACCEPTANCE_GENERATIONS
from paper2_v2.work9_gate1 import (
    CALL_SCHEMA_VERSION,
    CALL_SCORE_SCHEMA_VERSION,
    EPOCH_TRACE_SCHEMA_VERSION,
    GATE1_ANALYSIS_PROTOCOL_SHA256,
    GATE1_DECISION_POLICY_SHA256,
    INDEX_SCHEMA_VERSION,
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_SCORE_SCHEMA_VERSION,
    THRESHOLD_TRACE_SCHEMA_VERSION,
    Gate1AnalysisError,
    Gate1Call,
    ImmutableRef,
    ManifestSegment,
    _bootstrap_effects,
    _call_nll,
    _cluster_confusion_contributions,
    _permutation_effects,
    _validate_epoch_trace,
    _validate_acceptance_environment,
    _validate_outer_metrics_artifact,
    _validate_pairing_plan_artifact,
    _validate_acceptance_supersession,
    _validate_prediction_ledgers,
    _validate_probability_score_ledgers,
    _validate_report_schema,
    _validate_threshold_trace,
    analyze_primary_comparisons,
    build_condition_metrics,
    classify_gate1_continuation,
    compute_declared_call_metrics,
    holm_adjust,
    load_gate1_decision_policy,
    load_gate1_analysis_protocol,
    recompute_threshold_candidates,
    validate_gate1_inputs,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref_mapping(path: Path, root: Path, *, ledger: bool) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "row_count": len(path.read_text(encoding="utf-8").splitlines()) if ledger else None,
    }


def immutable_ref(path: Path, root: Path, *, ledger: bool) -> ImmutableRef:
    return ImmutableRef.from_mapping(
        ref_mapping(path, root, ledger=ledger),
        project_root=root,
        context=path.name,
        ledger=ledger,
    )


def fake_selection() -> SimpleNamespace:
    return SimpleNamespace(
        threshold_start=0.01,
        threshold_end=0.99,
        threshold_step=0.01,
        threshold_tie_break=THRESHOLD_TIE_BREAK,
    )


def fake_statistics() -> SimpleNamespace:
    return SimpleNamespace(
        primary_comparisons=PRIMARY_COMPARISONS,
        primary_estimand="mean_across_seeds_of_paired_call_macro_f1_difference",
        reported_call_metrics=REPORTED_CALL_METRICS,
        bootstrap_method="paired_relation_group_cluster_keep_all_calls_segments_seeds",
        bootstrap_replicates=10000,
        bootstrap_seed=20260809,
        confidence_level=0.95,
        permutation_method="paired_relation_group_cluster_swap_all_calls_seeds",
        permutation_replicates=10000,
        permutation_seed=20260810,
        multiplicity_method="holm",
        familywise_alpha=0.05,
        segment_metrics_role="secondary_weak_label",
        practical_claim_rule="no_practical_utility_claim_without_pre_work9_cost_policy",
    )


class FakeFold:
    def __init__(self, outer_fold: int) -> None:
        self.partition_repeat = 0
        self.outer_fold = outer_fold


def fake_analysis_config() -> SimpleNamespace:
    return SimpleNamespace(
        protocol_id="paper2-v2-test",
        conditions=CORE_CONDITIONS,
        seeds=(17, 29, 43, 71, 101),
        folds=tuple(FakeFold(index) for index in range(5)),
        selection=fake_selection(),
        statistics=fake_statistics(),
    )


def call(
    condition: str,
    fold: int,
    seed: int,
    suffix: str,
    label: int,
    prediction: int,
    *,
    group: str,
) -> Gate1Call:
    probability = 0.9 if prediction else 0.1
    return Gate1Call(
        condition=condition,
        partition_repeat=0,
        outer_fold=fold,
        seed=seed,
        call_id=f"fold{fold}_{suffix}",
        group_id=group,
        label=label,
        probability=probability,
        hard_prediction=prediction,
        threshold=0.5,
        n_segments=1,
        source="source",
        start=0.0,
        end=1.0,
    )


def synthetic_matrix_calls(config: SimpleNamespace) -> dict[tuple[str, int, int, int], tuple[Gate1Call, ...]]:
    result: dict[tuple[str, int, int, int], tuple[Gate1Call, ...]] = {}
    for condition in config.conditions:
        for fold in config.folds:
            for seed in config.seeds:
                negative_prediction = 0
                positive_prediction = 1
                if condition == "A-query" and fold.outer_fold == 0:
                    positive_prediction = 0
                if condition == "AT-concat" and fold.outer_fold == 1:
                    negative_prediction = 1
                if condition == "AT-gate" and fold.outer_fold == 2:
                    positive_prediction = 0
                result[(condition, 0, fold.outer_fold, seed)] = (
                    call(
                        condition,
                        fold.outer_fold,
                        seed,
                        "negative",
                        0,
                        negative_prediction,
                        group=f"group_{fold.outer_fold}",
                    ),
                    call(
                        condition,
                        fold.outer_fold,
                        seed,
                        "positive",
                        1,
                        positive_prediction,
                        group=f"group_{fold.outer_fold}",
                    ),
                )
    return result


def expected_segments() -> dict[str, ManifestSegment]:
    return {
        "s0": ManifestSegment("s0", "c0", "g0", 0, "source", 0.0, 1.0),
        "s1": ManifestSegment("s1", "c0", "g0", 0, "source", 1.0, 2.0),
        "s2": ManifestSegment("s2", "c1", "g1", 1, "source", 0.0, 1.0),
        "s3": ManifestSegment("s3", "c1", "g1", 1, "source", 1.0, 2.0),
    }


def coordinate_fields(*, partition: str) -> dict[str, object]:
    return {
        "run_id": "run-test",
        "protocol_id": "paper2-v2-test",
        "run_family": "primary_matrix",
        "partition": partition,
        "partition_repeat": 0,
        "outer_fold": 0,
        "seed": 17,
        "condition": "AT-cross",
    }


def make_prediction_ledgers(
    root: Path,
    *,
    partition: str,
    probabilities: tuple[float, float, float, float],
    threshold: float,
    checkpoint_sha256: str,
    manifest_sha256: str,
) -> tuple[Path, Path]:
    segment_rows = []
    manifest = expected_segments()
    for segment_id, probability in zip(sorted(manifest), probabilities, strict=True):
        expected = manifest[segment_id]
        segment_rows.append(
            {
                "schema_version": SEGMENT_SCHEMA_VERSION,
                **coordinate_fields(partition=partition),
                "segment_id": segment_id,
                "call_id": expected.call_id,
                "group_id": expected.group_id,
                "source": expected.source,
                "label": expected.label,
                "probability": probability,
                "hard_prediction": int(probability >= threshold),
                "start": expected.start,
                "end": expected.end,
                "duration_seconds": expected.end - expected.start,
                "threshold": threshold,
                "checkpoint_sha256": checkpoint_sha256,
                "manifest_sha256": manifest_sha256,
            }
        )
    call_rows = []
    for call_id, segment_ids in (("c0", ("s0", "s1")), ("c1", ("s2", "s3"))):
        rows = [row for row in segment_rows if row["segment_id"] in segment_ids]
        probability = sum(float(row["probability"]) for row in rows) / len(rows)
        call_rows.append(
            {
                "schema_version": CALL_SCHEMA_VERSION,
                **coordinate_fields(partition=partition),
                "call_id": call_id,
                "group_id": rows[0]["group_id"],
                "source": rows[0]["source"],
                "label": rows[0]["label"],
                "probability": probability,
                "hard_prediction": int(probability >= threshold),
                "n_segments": len(rows),
                "aggregation_rule": "mean_probability",
                "start": 0.0,
                "end": 2.0,
                "duration_seconds": 2.0,
                "threshold": threshold,
                "checkpoint_sha256": checkpoint_sha256,
                "manifest_sha256": manifest_sha256,
            }
        )
    segment_path = root / f"{partition}_segments.jsonl"
    call_path = root / f"{partition}_calls.jsonl"
    write_jsonl(segment_path, segment_rows)
    write_jsonl(call_path, call_rows)
    return segment_path, call_path


def make_score_ledgers(
    root: Path,
    *,
    epoch: int,
    probabilities: tuple[float, float, float, float],
    checkpoint_sha256: str,
    manifest_sha256: str,
) -> tuple[Path, Path]:
    segment_rows = []
    manifest = expected_segments()
    for segment_id, probability in zip(sorted(manifest), probabilities, strict=True):
        expected = manifest[segment_id]
        segment_rows.append(
            {
                "schema_version": SEGMENT_SCORE_SCHEMA_VERSION,
                **coordinate_fields(partition="validation"),
                "epoch": epoch,
                "segment_id": segment_id,
                "call_id": expected.call_id,
                "group_id": expected.group_id,
                "source": expected.source,
                "label": expected.label,
                "probability": probability,
                "start": expected.start,
                "end": expected.end,
                "duration_seconds": expected.end - expected.start,
                "checkpoint_sha256": checkpoint_sha256,
                "manifest_sha256": manifest_sha256,
            }
        )
    call_rows = []
    for call_id, segment_ids in (("c0", ("s0", "s1")), ("c1", ("s2", "s3"))):
        rows = [row for row in segment_rows if row["segment_id"] in segment_ids]
        probability = sum(float(row["probability"]) for row in rows) / len(rows)
        call_rows.append(
            {
                "schema_version": CALL_SCORE_SCHEMA_VERSION,
                **coordinate_fields(partition="validation"),
                "epoch": epoch,
                "call_id": call_id,
                "group_id": rows[0]["group_id"],
                "source": rows[0]["source"],
                "label": rows[0]["label"],
                "probability": probability,
                "n_segments": len(rows),
                "aggregation_rule": "mean_probability",
                "start": 0.0,
                "end": 2.0,
                "duration_seconds": 2.0,
                "checkpoint_sha256": checkpoint_sha256,
                "manifest_sha256": manifest_sha256,
            }
        )
    segment_path = root / f"epoch_{epoch}_segment_scores.jsonl"
    call_path = root / f"epoch_{epoch}_call_scores.jsonl"
    write_jsonl(segment_path, segment_rows)
    write_jsonl(call_path, call_rows)
    return segment_path, call_path


class Work9Gate1Tests(unittest.TestCase):
    def test_runner_acceptance_reopens_all_superseded_evidence(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        generations = []
        for generation, draft, _, failure, _, reason in SUPERSEDED_ACCEPTANCE_GENERATIONS:
            generations.append(
                {
                    "acceptance_generation": generation,
                    "draft": ref_mapping(
                        project_root / draft, project_root, ledger=False
                    ),
                    "failure": ref_mapping(
                        project_root / failure, project_root, ledger=False
                    ),
                    "reason": reason,
                }
            )
        acceptance = {
            "supersedes": {
                "current_acceptance_generation": "work9_runner_acceptance_v3",
                "generations": generations,
            }
        }
        _validate_acceptance_supersession(
            acceptance, project_root=project_root
        )
        mutated = json.loads(json.dumps(acceptance))
        mutated["supersedes"]["generations"][1]["failure"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(Gate1AnalysisError, "hash mismatch"):
            _validate_acceptance_supersession(
                mutated, project_root=project_root
            )
        reordered = json.loads(json.dumps(acceptance))
        reordered["supersedes"]["generations"].reverse()
        with self.assertRaisesRegex(Gate1AnalysisError, "identity changed"):
            _validate_acceptance_supersession(
                reordered, project_root=project_root
            )

    def test_runner_acceptance_environment_binds_smokes_and_unit_tests(self) -> None:
        lock = environment_lock()
        acceptance = {
            "environment_lock": lock,
            "cuda_smokes": [
                {"environment_lock_sha256": lock["environment_lock_sha256"]},
                {"environment_lock_sha256": lock["environment_lock_sha256"]},
            ],
            "unit_tests": {
                "environment_lock_sha256": lock["environment_lock_sha256"]
            },
        }
        self.assertEqual(
            _validate_acceptance_environment(acceptance),
            lock["environment_lock_sha256"],
        )
        acceptance["unit_tests"]["environment_lock_sha256"] = "0" * 64
        with self.assertRaisesRegex(Gate1AnalysisError, "unit-test"):
            _validate_acceptance_environment(acceptance)

    def test_frozen_analysis_protocol_hash_and_mutation_fail_closed(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        path = (
            project_root
            / "revised_experiments/configs/work9_gate1_analysis_protocol_v1.json"
        )
        self.assertEqual(digest(path), GATE1_ANALYSIS_PROTOCOL_SHA256)
        self.assertEqual(path.stat().st_mode & 0o777, 0o444)
        protocol = load_gate1_analysis_protocol(path)
        self.assertEqual(protocol["bootstrap"]["replicates"], 10000)
        with tempfile.TemporaryDirectory() as temporary:
            mutated = Path(temporary) / path.name
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["bootstrap"]["quantile_method"] = "nearest"
            write_json(mutated, payload)
            with self.assertRaisesRegex(Gate1AnalysisError, "analysis protocol hash mismatch"):
                load_gate1_analysis_protocol(mutated)

    def test_immutable_reference_rejects_symlink_components(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real_dir = root / "real"
            real_dir.mkdir()
            artifact = real_dir / "artifact.json"
            write_json(artifact, {"value": 1})
            (root / "linked").symlink_to(real_dir, target_is_directory=True)
            raw = {
                "path": "linked/artifact.json",
                "sha256": digest(artifact),
                "bytes": artifact.stat().st_size,
                "row_count": None,
            }
            with self.assertRaisesRegex(Gate1AnalysisError, "traverses a symlink"):
                ImmutableRef.from_mapping(
                    raw,
                    project_root=root,
                    context="linked artifact",
                    ledger=False,
                )

    def test_pairing_artifact_reopens_against_train_universe_and_semantic_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = expected_segments()
            plan = build_random_train_pairing(
                [
                    PairingItem(segment_id, row.call_id, row.group_id)
                    for segment_id, row in manifest.items()
                ],
                seed=17,
            )
            good_path = root / "good" / "pairing_plan.jsonl"
            write_jsonl(good_path, plan.as_records())
            _validate_pairing_plan_artifact(
                immutable_ref(good_path, root, ledger=True),
                seed=17,
                expected_segments=manifest,
                expected_semantic_sha256=plan.sha256,
            )

            bad_rows = plan.as_records()
            bad_rows[0] = {**bad_rows[0], "donor_segment_id": bad_rows[0]["receiver_segment_id"]}
            bad_path = root / "bad" / "pairing_plan.jsonl"
            write_jsonl(bad_path, bad_rows)
            with self.assertRaisesRegex(Gate1AnalysisError, "identity mismatch|invariants fail"):
                _validate_pairing_plan_artifact(
                    immutable_ref(bad_path, root, ledger=True),
                    seed=17,
                    expected_segments=manifest,
                    expected_semantic_sha256=plan.sha256,
                )

    def test_declared_metrics_and_complete_call_mean_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            checkpoint_sha = digest(checkpoint)
            manifest_sha = "a" * 64
            segment_path, call_path = make_prediction_ledgers(
                root,
                partition="outer_eval",
                probabilities=(0.1, 0.3, 0.7, 0.9),
                threshold=0.5,
                checkpoint_sha256=checkpoint_sha,
                manifest_sha256=manifest_sha,
            )
            refs = {
                "checkpoint": immutable_ref(checkpoint, root, ledger=False),
                "outer_segment_ledger": immutable_ref(segment_path, root, ledger=True),
                "outer_call_ledger": immutable_ref(call_path, root, ledger=True),
            }
            entry = {
                **coordinate_fields(partition="outer_eval"),
                "threshold": 0.5,
            }
            pair = _validate_prediction_ledgers(
                entry=entry,
                refs=refs,
                config=SimpleNamespace(protocol_id="paper2-v2-test"),
                expected_segments=expected_segments(),
                partition="outer_eval",
                manifest_sha256=manifest_sha,
                segment_ref_name="outer_segment_ledger",
                call_ref_name="outer_call_ledger",
                checkpoint_sha256=checkpoint_sha,
                expected_threshold=0.5,
            )
            self.assertEqual([item.probability for item in pair.calls], [0.2, 0.8])
            metrics = compute_declared_call_metrics(pair.calls)
            self.assertEqual(tuple(key for key in metrics if key in REPORTED_CALL_METRICS), REPORTED_CALL_METRICS)
            self.assertEqual(metrics["macro_f1"], 1.0)
            metrics_path = root / "outer_metrics.json"
            write_json(
                metrics_path,
                {
                    "schema_version": "paper2-v2-work9-outer-metrics-1",
                    **coordinate_fields(partition="outer_eval"),
                    "aggregation_rule": "mean_probability",
                    "threshold": 0.5,
                    "checkpoint_sha256": checkpoint_sha,
                    "outer_eval_manifest_sha256": manifest_sha,
                    "metrics": compute_call_metrics(pair.calls, threshold=0.5),
                    "interpretation": (
                        "within_corpus_source_confounded_outer_development_evaluation"
                    ),
                },
            )
            _validate_outer_metrics_artifact(
                immutable_ref(metrics_path, root, ledger=False),
                entry=entry,
                config=SimpleNamespace(protocol_id="paper2-v2-test"),
                calls=pair.calls,
                checkpoint_sha256=checkpoint_sha,
                outer_manifest_sha256=manifest_sha,
            )
            with self.assertRaisesRegex(Gate1AnalysisError, "selected epoch score"):
                _validate_prediction_ledgers(
                    entry=entry,
                    refs=refs,
                    config=SimpleNamespace(protocol_id="paper2-v2-test"),
                    expected_segments=expected_segments(),
                    partition="outer_eval",
                    manifest_sha256=manifest_sha,
                    segment_ref_name="outer_segment_ledger",
                    call_ref_name="outer_call_ledger",
                    checkpoint_sha256=checkpoint_sha,
                    expected_threshold=0.5,
                    expected_segment_probabilities={key: 0.5 for key in expected_segments()},
                )

    def test_score_ledgers_epoch_selection_and_threshold_trace_are_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_sha = "a" * 64
            config = SimpleNamespace(protocol_id="paper2-v2-test", selection=fake_selection())
            epoch_rows = []
            selected_checkpoint_ref = None
            selected_score_pair = None
            for epoch, probabilities in (
                (1, (0.8, 0.8, 0.2, 0.2)),
                (2, (0.1, 0.1, 0.9, 0.9)),
            ):
                checkpoint = root / f"epoch_{epoch}.pt"
                checkpoint.write_bytes(f"checkpoint-{epoch}".encode())
                checkpoint_ref = immutable_ref(checkpoint, root, ledger=False)
                segment_path, call_path = make_score_ledgers(
                    root,
                    epoch=epoch,
                    probabilities=probabilities,
                    checkpoint_sha256=checkpoint_ref.sha256,
                    manifest_sha256=manifest_sha,
                )
                score_refs = {
                    "checkpoint": checkpoint_ref,
                    "validation_segment_scores": immutable_ref(segment_path, root, ledger=True),
                    "validation_call_scores": immutable_ref(call_path, root, ledger=True),
                }
                pair = _validate_probability_score_ledgers(
                    entry=coordinate_fields(partition="validation"),
                    refs=score_refs,
                    config=config,
                    expected_segments=expected_segments(),
                    manifest_sha256=manifest_sha,
                    checkpoint_sha256=checkpoint_ref.sha256,
                    expected_epoch=epoch,
                )
                _, diagnostic_threshold = recompute_threshold_candidates(pair.calls, config)
                epoch_rows.append(
                    {
                        "epoch": epoch,
                        "validation_call_nll": _call_nll(pair.calls),
                        "train_reported_loss": 0.5 / epoch,
                        "diagnostic_threshold": diagnostic_threshold,
                        "checkpoint": checkpoint_ref.as_mapping(),
                        "validation_segment_scores": score_refs[
                            "validation_segment_scores"
                        ].as_mapping(),
                        "validation_call_scores": score_refs[
                            "validation_call_scores"
                        ].as_mapping(),
                    }
                )
                if epoch == 2:
                    selected_checkpoint_ref = checkpoint_ref
                    selected_score_pair = pair
            assert selected_checkpoint_ref is not None and selected_score_pair is not None
            trace_payload = {
                "schema_version": EPOCH_TRACE_SCHEMA_VERSION,
                **{
                    key: value
                    for key, value in coordinate_fields(partition="validation").items()
                    if key != "partition"
                },
                "monitor": "validation_call_nll",
                "tie_break": "earlier_epoch",
                "epochs": epoch_rows,
                "selected_epoch": 2,
            }
            trace_path = root / "epoch_trace.json"
            write_json(trace_path, trace_payload)
            entry = {
                **coordinate_fields(partition="validation"),
                "selected_epoch": 2,
                "validation_call_nll": _call_nll(selected_score_pair.calls),
                "threshold": recompute_threshold_candidates(selected_score_pair.calls, config)[1],
            }
            selected_pair, audit = _validate_epoch_trace(
                project_root=root,
                trace_ref=immutable_ref(trace_path, root, ledger=False),
                entry=entry,
                refs={"checkpoint": selected_checkpoint_ref},
                config=config,
                expected_segments=expected_segments(),
                validation_manifest_sha256=manifest_sha,
            )
            self.assertEqual(audit["selected_epoch"], 2)
            self.assertEqual(selected_pair.segment_probabilities, selected_score_pair.segment_probabilities)

            selected_threshold = float(entry["threshold"])
            validation_segment, validation_call = make_prediction_ledgers(
                root,
                partition="validation",
                probabilities=(0.1, 0.1, 0.9, 0.9),
                threshold=selected_threshold,
                checkpoint_sha256=selected_checkpoint_ref.sha256,
                manifest_sha256=manifest_sha,
            )
            validation_refs = {
                "checkpoint": selected_checkpoint_ref,
                "validation_segment_ledger": immutable_ref(validation_segment, root, ledger=True),
                "validation_call_ledger": immutable_ref(validation_call, root, ledger=True),
            }
            final_pair = _validate_prediction_ledgers(
                entry=entry,
                refs=validation_refs,
                config=config,
                expected_segments=expected_segments(),
                partition="validation",
                manifest_sha256=manifest_sha,
                segment_ref_name="validation_segment_ledger",
                call_ref_name="validation_call_ledger",
                checkpoint_sha256=selected_checkpoint_ref.sha256,
                expected_threshold=selected_threshold,
                expected_segment_probabilities=selected_pair.segment_probabilities,
            )
            candidates, recomputed_threshold = recompute_threshold_candidates(
                final_pair.calls, config
            )
            threshold_payload = {
                "schema_version": THRESHOLD_TRACE_SCHEMA_VERSION,
                **coordinate_fields(partition="validation"),
                "validation_call_ledger": validation_refs[
                    "validation_call_ledger"
                ].as_mapping(),
                "threshold_grid": {
                    "start": 0.01,
                    "end": 0.99,
                    "step": 0.01,
                    "count": 99,
                },
                "threshold_tie_break": list(THRESHOLD_TIE_BREAK),
                "candidates": candidates,
                "selected_threshold": recomputed_threshold,
            }
            threshold_path = root / "threshold_trace.json"
            write_json(threshold_path, threshold_payload)
            threshold_audit = _validate_threshold_trace(
                trace_ref=immutable_ref(threshold_path, root, ledger=False),
                entry=entry,
                refs=validation_refs,
                config=config,
                selected_validation_calls=final_pair.calls,
            )
            self.assertEqual(threshold_audit["threshold_candidates_recomputed"], 99)

    def test_clustered_inference_holm_and_stability_are_deterministic(self) -> None:
        config = fake_analysis_config()
        calls = synthetic_matrix_calls(config)
        records, summaries, lookup = build_condition_metrics(calls, config)
        self.assertEqual(len(records), len(CORE_CONDITIONS) * 5 * 6)
        self.assertEqual(set(summaries), set(CORE_CONDITIONS))
        first = analyze_primary_comparisons(
            calls_by_coordinate=calls,
            config=config,
            metric_lookup=lookup,
            bootstrap_replicates=200,
            permutation_replicates=200,
        )
        second = analyze_primary_comparisons(
            calls_by_coordinate=calls,
            config=config,
            metric_lookup=lookup,
            bootstrap_replicates=200,
            permutation_replicates=200,
        )
        self.assertEqual(first, second)
        self.assertEqual([row["comparison_id"] for row in first], [
            "AT-cross_minus_A-query",
            "AT-cross_minus_AT-concat",
            "AT-cross_minus_AT-gate",
        ])
        self.assertAlmostEqual(first[0]["observed_effect"], 10.0 / 99.0)
        self.assertAlmostEqual(
            first[0]["effect_sizes"]["absolute_macro_f1_difference_magnitude"],
            10.0 / 99.0,
        )
        contribution = _cluster_confusion_contributions(
            calls,
            condition="AT-cross",
            seeds=config.seeds,
            folds=config.folds,
            group_ids=[f"group_{index}" for index in range(5)],
        )
        self.assertEqual(contribution[0, 0].sum(), 2)
        self.assertEqual(holm_adjust([0.01, 0.03, 0.04]), [0.03, 0.06, 0.06])

        primary = np.asarray(
            [
                [[1, 0, 0, 1]],
                [[1, 0, 0, 1]],
            ],
            dtype=np.int64,
        )
        comparator = np.asarray(
            [
                [[1, 0, 1, 0]],
                [[0, 1, 0, 1]],
            ],
            dtype=np.int64,
        )
        expected_effect = 2.0 / 3.0
        boot = _bootstrap_effects(primary, comparator, replicates=25, seed=7)
        self.assertTrue(np.allclose(boot, expected_effect))
        perm = _permutation_effects(primary, comparator, replicates=25, seed=7)
        self.assertTrue(np.allclose(np.abs(perm), expected_effect))

    def test_frozen_continuation_classification_is_not_practical_utility(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        policy_path = project_root / "revised_experiments/configs/work9_gate1_decision_policy_v1.json"
        self.assertEqual(digest(policy_path), GATE1_DECISION_POLICY_SHA256)
        policy = load_gate1_decision_policy(policy_path)

        def comparison(name: str, observed: float, lower: float, adjusted: float) -> dict[str, object]:
            return {
                "comparison_id": name,
                "observed_effect": observed,
                "bootstrap": {"ci_lower": lower},
                "multiplicity": {"adjusted_p_value": adjusted},
            }

        classification = classify_gate1_continuation(
            [
                comparison("AT-cross_minus_A-query", 0.02, 0.001, 0.049),
                comparison("AT-cross_minus_AT-concat", 0.01, 0.0001, 0.01),
                comparison("AT-cross_minus_AT-gate", 0.01, 0.0001, 0.02),
            ],
            policy,
        )
        self.assertEqual(
            classification["statistical_incremental_text_conditioning"]["status"],
            "supported",
        )
        self.assertEqual(classification["cross_mechanism_superiority"]["status"], "supported")
        self.assertEqual(classification["gate2_cross_specific_pairing_controls"], "proceed")
        self.assertEqual(classification["practical_utility_claim"], "forbidden_not_made")
        boundary = classify_gate1_continuation(
            [
                comparison("AT-cross_minus_A-query", 0.02, 0.0, 0.049),
                comparison("AT-cross_minus_AT-concat", -0.01, -0.02, 0.9),
                comparison("AT-cross_minus_AT-gate", 0.01, 0.001, 0.049),
            ],
            policy,
        )
        self.assertEqual(
            boundary["statistical_incremental_text_conditioning"]["status"],
            "inconclusive",
        )
        self.assertEqual(boundary["cross_mechanism_superiority"]["status"], "unsupported")
        self.assertEqual(boundary["gate2_cross_specific_pairing_controls"], "do_not_proceed")
        strict_p_boundary = classify_gate1_continuation(
            [
                comparison("AT-cross_minus_A-query", 0.02, 0.001, 0.05),
                comparison("AT-cross_minus_AT-concat", 0.01, 0.001, 0.01),
                comparison("AT-cross_minus_AT-gate", 0.01, 0.001, 0.01),
            ],
            policy,
        )
        self.assertEqual(
            strict_p_boundary["statistical_incremental_text_conditioning"]["status"],
            "inconclusive",
        )

    def test_generated_report_shape_is_strict_and_rejects_empty_statistics(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        policy = load_gate1_decision_policy(
            project_root
            / "revised_experiments/configs/work9_gate1_decision_policy_v1.json"
        )
        config = fake_analysis_config()
        calls = synthetic_matrix_calls(config)
        _, summaries, lookup = build_condition_metrics(calls, config)
        comparisons = analyze_primary_comparisons(
            calls_by_coordinate=calls,
            config=config,
            metric_lookup=lookup,
        )
        report = {
            "schema_version": "paper2-v2-work9-gate1-report-1",
            "status": "pass",
            "analysis_role": "post_run_gate1_immutable_ledger_statistics",
            "protocol_id": config.protocol_id,
            "within_corpus_only": True,
            "inputs": {
                name: {"path": f"/{name}.json", "sha256": "a" * 64}
                for name in (
                    "matrix_config",
                    "outer_ledger_index",
                    "selection_index",
                    "gate1_decision_policy",
                    "gate1_analysis_protocol",
                )
            },
            "validation": {"status": "pass"},
            "call_primary_policy": {
                "independent_unit": "call",
                "complete_call_aggregation": "mean_probability_over_all_manifest_segments",
                "outer_folds_combined_before_per_seed_metric": 5,
                "seeds_combined_by": "arithmetic_mean_of_five_per_seed_metrics",
                "reported_call_metrics": list(REPORTED_CALL_METRICS),
                "segment_metrics_role": "secondary_weak_label",
            },
            "condition_summaries": summaries,
            "primary_comparisons": comparisons,
            "gate1_continuation_classification": classify_gate1_continuation(
                comparisons, policy
            ),
            "multiplicity": {
                "method": "holm",
                "familywise_alpha": 0.05,
                "family": [list(pair) for pair in PRIMARY_COMPARISONS],
            },
            "practical_utility": {
                "frozen_rule": "no_practical_utility_claim_without_pre_work9_cost_policy",
                "cost_policy_available": False,
                "smallest_practically_meaningful_effect": None,
                "claim_permitted": False,
                "claim_made": False,
            },
        }
        _validate_report_schema(report, project_root=project_root)
        malformed = json.loads(json.dumps(report))
        malformed["primary_comparisons"][0]["bootstrap"] = {}
        with self.assertRaisesRegex(Gate1AnalysisError, "report schema validation failed"):
            _validate_report_schema(malformed, project_root=project_root)

    def test_run_index_hash_and_coordinate_completeness_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            project_root = Path(__file__).resolve().parents[2]
            matrix = (
                project_root
                / "revised_experiments/configs/work8_mfcc_matched_within_corpus_v1.json"
            )
            index = temporary_root / "outer_ledger_index.json"
            dummy_ref = {
                "path": "selection_index.json",
                "sha256": "a" * 64,
                "bytes": 1,
                "row_count": None,
            }
            write_json(
                index,
                {
                    "schema_version": INDEX_SCHEMA_VERSION,
                    "protocol_id": "paper2-v2-mfcc-matched-within-corpus-v1",
                    "run_family": "primary_matrix",
                    "status": "frozen_complete",
                    "selection_index": dummy_ref,
                    "gate1_analysis_protocol": dummy_ref,
                    "expected_coordinate_count": 200,
                    "entries": [{}],
                },
            )
            with self.assertRaisesRegex(Gate1AnalysisError, "exactly 200 entries"):
                validate_gate1_inputs(
                    project_root=project_root,
                    matrix_config_path=matrix,
                    outer_ledger_index_path=index,
                    expected_outer_ledger_index_sha256=digest(index),
                )
            with self.assertRaisesRegex(Gate1AnalysisError, "hash mismatch"):
                validate_gate1_inputs(
                    project_root=project_root,
                    matrix_config_path=matrix,
                    outer_ledger_index_path=index,
                    expected_outer_ledger_index_sha256="0" * 64,
                )

            altered_matrix = temporary_root / "matrix.json"
            write_json(altered_matrix, {"locked": True})
            fake_config = SimpleNamespace(seeds=(17, 29, 43, 71, 101))
            with self.assertRaisesRegex(Gate1AnalysisError, "canonical frozen Work 8"):
                validate_gate1_inputs(
                    project_root=temporary_root,
                    matrix_config_path=altered_matrix,
                    outer_ledger_index_path=index,
                    expected_outer_ledger_index_sha256=digest(index),
                    config=fake_config,
                )

    def test_schema_documents_are_valid_draft_2020_12(self) -> None:
        import jsonschema

        project_root = Path(__file__).resolve().parents[2]
        schema_paths = [
            project_root
            / "revised_experiments"
            / "registry"
            / "schemas"
            / "work9_gate1_report.schema.json"
        ]
        schema_paths.extend(
            sorted(
                (
                    project_root
                    / "revised_experiments"
                    / "registry"
                    / "schemas"
                    / "work9"
                ).glob("*.schema.json")
            )
        )
        for schema_path in schema_paths:
            schema = json.loads(
                schema_path.read_text(encoding="utf-8")
            )
            jsonschema.Draft202012Validator.check_schema(schema)


if __name__ == "__main__":
    unittest.main()
