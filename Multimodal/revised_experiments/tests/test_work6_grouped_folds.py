from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from paper2_v2.folds import (
    FoldBuildError,
    build_grouped_folds,
    tree_digest,
    validate_fold_tree,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class SyntheticWork5Artifacts:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.provenance = root / "call_provenance.jsonl"
        self.relations = root / "relation_components.jsonl"
        self.segments = root / "segments.jsonl"
        calls: list[dict[str, object]] = []
        segments: list[dict[str, object]] = []
        for label, prefix in ((0, "neg"), (1, "pos")):
            for index in range(20):
                call_id = f"{prefix}_{index:02d}"
                calls.append(
                    {
                        "call_id": call_id,
                        "label": label,
                        "historical_split": "train" if index % 4 else "val",
                        "source": "AIHub" if label == 0 else "FSS",
                        "audio_sha256": (
                            "a" * 64 if label == 1 and index in {2, 3} else "unknown"
                        ),
                    }
                )
                for segment_index in range(2):
                    segments.append(
                        {
                            "call_id": call_id,
                            "segment_id": f"{call_id}_{segment_index}",
                            "start": float(segment_index),
                            "end": float(segment_index + 1),
                            "label": label,
                            "mfcc_path": f"features/mfcc/{call_id}_{segment_index}.npy",
                            "text": f"text {call_id} {segment_index}",
                        }
                    )
        for label, call_id in ((0, "burned_neg"), (1, "burned_pos")):
            calls.append(
                {
                    "call_id": call_id,
                    "label": label,
                    "historical_split": "test",
                    "source": "AIHub" if label == 0 else "FSS",
                    "audio_sha256": "unknown",
                }
            )
            segments.append(
                {
                    "call_id": call_id,
                    "segment_id": f"{call_id}_0",
                    "start": 0.0,
                    "end": 1.0,
                    "label": label,
                    "mfcc_path": f"features/mfcc/{call_id}_0.npy",
                    "text": "burned",
                }
            )
        relations = [
            # Transitivity is essential: pos_00 is not directly linked to the
            # burned call but must be quarantined through pos_01.
            {
                "component_id": "burn_bridge_1",
                "call_ids": ["pos_00", "pos_01"],
                "relation_types": ["speaker_cluster"],
                "status": "accepted",
            },
            {
                "component_id": "burn_bridge_2",
                "call_ids": ["pos_01", "burned_pos"],
                "relation_types": ["template_cluster"],
                "status": "accepted",
            },
            # This known transcript family must remain one group, not be
            # deduplicated because it is not exact audio.
            {
                "component_id": "template_pair",
                "call_ids": ["neg_00", "neg_01"],
                "relation_types": ["exact_transcript"],
                "status": "accepted",
            },
            # Explicit rejects are not grouping edges.
            {
                "component_id": "rejected_candidate",
                "call_ids": ["neg_18", "pos_18"],
                "relation_types": ["near_audio"],
                "status": "rejected",
            },
        ]
        _write_jsonl(self.provenance, calls)
        _write_jsonl(self.relations, relations)
        _write_jsonl(self.segments, segments)


class Work6FoldTests(unittest.TestCase):
    def _build(self, root: Path, name: str = "folds") -> Path:
        inputs = SyntheticWork5Artifacts(root / "inputs")
        output = root / name
        build_grouped_folds(
            call_provenance_path=inputs.provenance,
            relation_component_paths=[inputs.relations],
            segment_manifest_paths=[inputs.segments],
            output_dir=output,
        )
        return output

    def test_transitive_burned_quarantine_and_exact_audio_dedup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._build(Path(temporary))
            rows = {row["call_id"]: row for row in _read_jsonl(output / "assignments.jsonl")}
            self.assertEqual(rows["burned_pos"]["role"], "quarantine")
            self.assertEqual(rows["burned_neg"]["role"], "quarantine")
            self.assertEqual(rows["pos_00"]["role"], "quarantine")
            self.assertEqual(rows["pos_01"]["role"], "quarantine")
            self.assertEqual(rows["pos_03"]["role"], "excluded")
            self.assertEqual(rows["pos_03"]["exact_audio_representative_call_id"], "pos_02")
            self.assertEqual(rows["neg_00"]["group_id"], rows["neg_01"]["group_id"])
            self.assertNotEqual(rows["neg_18"]["group_id"], rows["pos_18"]["group_id"])

    def test_fold_coverage_stratification_names_and_leakage_guards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._build(Path(temporary))
            validate_fold_tree(output)
            protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
            self.assertEqual(protocol["terminology"]["outer_partition_name"], "outer_eval")
            self.assertEqual(len(protocol["folds"]), 5)
            assignments = _read_jsonl(output / "assignments.jsonl")
            eligible = {row["call_id"] for row in assignments if row["role"] == "eligible"}
            outer_counts: Counter[str] = Counter()
            for fold_index in range(5):
                fold_root = output / "repeat_00" / f"outer_{fold_index:02d}"
                self.assertEqual(
                    {path.name for path in fold_root.iterdir()},
                    {
                        "train_call_manifest.jsonl",
                        "train_segment_manifest.jsonl",
                        "val_call_manifest.jsonl",
                        "val_segment_manifest.jsonl",
                        "outer_eval_call_manifest.jsonl",
                        "outer_eval_segment_manifest.jsonl",
                    },
                )
                role_calls: dict[str, set[str]] = {}
                role_groups: dict[str, set[str]] = {}
                for role in ("train", "val", "outer_eval"):
                    rows = _read_jsonl(fold_root / f"{role}_call_manifest.jsonl")
                    role_calls[role] = {row["call_id"] for row in rows}
                    role_groups[role] = {row["group_id"] for row in rows}
                    self.assertEqual({row["partition"] for row in rows}, {role})
                    self.assertEqual({row["label"] for row in rows}, {0, 1})
                for left, right in (
                    ("train", "val"),
                    ("train", "outer_eval"),
                    ("val", "outer_eval"),
                ):
                    self.assertFalse(role_calls[left] & role_calls[right])
                    self.assertFalse(role_groups[left] & role_groups[right])
                self.assertEqual(set().union(*role_calls.values()), eligible)
                outer_counts.update(role_calls["outer_eval"])
            self.assertEqual(set(outer_counts), eligible)
            self.assertEqual(set(outer_counts.values()), {1})

    def test_identical_inputs_produce_byte_identical_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = SyntheticWork5Artifacts(root / "inputs")
            outputs = []
            for name in ("first", "second"):
                output = root / name
                build_grouped_folds(
                    call_provenance_path=inputs.provenance,
                    relation_component_paths=[inputs.relations],
                    segment_manifest_paths=[inputs.segments],
                    output_dir=output,
                )
                outputs.append(output)
            self.assertEqual(tree_digest(outputs[0]), tree_digest(outputs[1]))
            first_files = {
                path.relative_to(outputs[0]).as_posix(): path.read_bytes()
                for path in outputs[0].rglob("*")
                if path.is_file()
            }
            second_files = {
                path.relative_to(outputs[1]).as_posix(): path.read_bytes()
                for path in outputs[1].rglob("*")
                if path.is_file()
            }
            self.assertEqual(first_files, second_files)

    def test_refuses_to_overwrite_existing_fold_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self._build(root)
            inputs = SyntheticWork5Artifacts(root / "second_inputs")
            with self.assertRaises(FileExistsError):
                build_grouped_folds(
                    call_provenance_path=inputs.provenance,
                    relation_component_paths=[inputs.relations],
                    segment_manifest_paths=[inputs.segments],
                    output_dir=output,
                )

    def test_unknown_relation_member_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = SyntheticWork5Artifacts(root / "inputs")
            _write_jsonl(
                inputs.relations,
                [
                    {
                        "component_id": "unsafe",
                        "call_ids": ["pos_02", "absent_burned_call"],
                        "relation_types": ["exact_audio"],
                    }
                ],
            )
            with self.assertRaisesRegex(FoldBuildError, "absent from provenance"):
                build_grouped_folds(
                    call_provenance_path=inputs.provenance,
                    relation_component_paths=[inputs.relations],
                    segment_manifest_paths=[inputs.segments],
                    output_dir=root / "folds",
                )

    def test_exact_audio_label_conflict_is_hard_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = SyntheticWork5Artifacts(root / "inputs")
            _write_jsonl(
                inputs.relations,
                [
                    {
                        "component_id": "conflict",
                        "call_ids": ["neg_02", "pos_02"],
                        "relation_types": ["exact_audio"],
                    }
                ],
            )
            with self.assertRaisesRegex(FoldBuildError, "conflicting labels"):
                build_grouped_folds(
                    call_provenance_path=inputs.provenance,
                    relation_component_paths=[inputs.relations],
                    segment_manifest_paths=[inputs.segments],
                    output_dir=root / "folds",
                )

    def test_audio_hash_relation_to_burned_call_quarantines_without_edge_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = SyntheticWork5Artifacts(root / "inputs")
            provenance = _read_jsonl(inputs.provenance)
            for row in provenance:
                if row["call_id"] in {"neg_02", "burned_neg"}:
                    row["audio_sha256"] = "b" * 64
            _write_jsonl(inputs.provenance, provenance)
            output = root / "folds"
            build_grouped_folds(
                call_provenance_path=inputs.provenance,
                relation_component_paths=[inputs.relations],
                segment_manifest_paths=[inputs.segments],
                output_dir=output,
            )
            rows = {row["call_id"]: row for row in _read_jsonl(output / "assignments.jsonl")}
            self.assertEqual(rows["neg_02"]["role"], "quarantine")
            self.assertEqual(rows["burned_neg"]["role"], "quarantine")

    def test_actual_work5_nested_schema_and_singleton_components(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = SyntheticWork5Artifacts(root / "inputs")
            provenance = _read_jsonl(inputs.provenance)
            component_members = [
                ["pos_00", "pos_01", "burned_pos"],
                ["neg_00", "neg_01"],
                ["pos_02", "pos_03"],
            ]
            already_grouped = set().union(*(set(members) for members in component_members))
            component_members.extend(
                [[row["call_id"]] for row in provenance if row["call_id"] not in already_grouped]
            )
            component_for = {
                call_id: f"component_{index:03d}"
                for index, members in enumerate(component_members)
                for call_id in members
            }
            for row in provenance:
                call_id = row["call_id"]
                byte_hash = row.pop("audio_sha256", "unknown")
                row["audio"] = {
                    "byte_sha256": byte_hash,
                    "pcm_sha256": "c" * 64 if call_id in {"pos_02", "pos_03"} else "unknown",
                }
                row["is_burned"] = row["historical_split"] == "test"
                row["relation_component_id"] = component_for[call_id]
                row["canonical_call_id"] = "pos_02" if call_id == "pos_03" else call_id
                if row["historical_split"] == "test":
                    status = "historical_burned"
                elif call_id in {"pos_00", "pos_01"}:
                    status = "quarantine_burned_related_component"
                elif call_id == "pos_03":
                    status = "exclude_redundant_exact_audio"
                else:
                    status = "eligible_for_work6"
                row["provisional_primary_status"] = status
            _write_jsonl(inputs.provenance, provenance)
            calls = {row["call_id"]: row for row in provenance}
            components = []
            for index, members in enumerate(component_members):
                burned = sorted(
                    call_id for call_id in members if calls[call_id]["historical_split"] == "test"
                )
                development_to_quarantine = sorted(
                    call_id
                    for call_id in members
                    if burned and calls[call_id]["historical_split"] != "test"
                )
                relation_types: list[str]
                if set(members) == {"pos_02", "pos_03"}:
                    relation_types = ["exact_audio_pcm_sha256"]
                elif len(members) > 1:
                    relation_types = ["reviewed_hard_relation"]
                else:
                    relation_types = []
                components.append(
                    {
                        "component_id": f"component_{index:03d}",
                        "members": members,
                        "member_count": len(members),
                        "labels": sorted({calls[call_id]["label"] for call_id in members}),
                        "sources": sorted({calls[call_id]["source"] for call_id in members}),
                        "historical_splits": sorted(
                            {calls[call_id]["historical_split"] for call_id in members}
                        ),
                        "accepted_relation_types": relation_types,
                        "burned_call_ids": burned,
                        "touches_burned": bool(burned),
                        "development_members_to_quarantine": development_to_quarantine,
                        "redundant_exact_audio_members": (
                            ["pos_03"] if set(members) == {"pos_02", "pos_03"} else []
                        ),
                    }
                )
            _write_jsonl(inputs.relations, components)
            output = root / "folds"
            build_grouped_folds(
                call_provenance_path=inputs.provenance,
                relation_component_paths=[inputs.relations],
                segment_manifest_paths=[inputs.segments],
                output_dir=output,
            )
            validate_fold_tree(output)
            assignments = {
                row["call_id"]: row for row in _read_jsonl(output / "assignments.jsonl")
            }
            self.assertEqual(assignments["pos_03"]["role"], "excluded")
            self.assertEqual(
                assignments["pos_03"]["work5_primary_status"],
                "exclude_redundant_exact_audio",
            )
            self.assertEqual(assignments["pos_00"]["role"], "quarantine")

    def test_pending_manual_relation_decision_stops_fold_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = SyntheticWork5Artifacts(root / "inputs")
            _write_jsonl(
                inputs.relations,
                [
                    {
                        "call_id_a": "neg_02",
                        "call_id_b": "neg_03",
                        "relation_type": "near_audio",
                        "decision": "pending",
                        "hard_group_edge": False,
                    }
                ],
            )
            with self.assertRaisesRegex(FoldBuildError, "must be resolved"):
                build_grouped_folds(
                    call_provenance_path=inputs.provenance,
                    relation_component_paths=[inputs.relations],
                    segment_manifest_paths=[inputs.segments],
                    output_dir=root / "folds",
                )


if __name__ == "__main__":
    unittest.main()
