from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from paper2_v2.data import InputSpec
from paper2_v2.pairing import build_random_train_pairing
from paper2_v2.work9_data import Work9FeatureDataset


class FakeTextStore:
    def __init__(self, values: dict[str, torch.Tensor]) -> None:
        self.values = values
        self.get_calls: list[str] = []

    def __contains__(self, segment_id: str) -> bool:
        return segment_id in self.values

    def get(self, segment_id: str) -> torch.Tensor:
        self.get_calls.append(segment_id)
        return self.values[segment_id].clone()


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class Work9DataTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, list[dict[str, object]]]:
        rows: list[dict[str, object]] = []
        calls = (("c0", "g0", 0, 1), ("c1", "g1", 1, 1), ("c2", "g2", 0, 2))
        index = 0
        for call_id, group_id, label, count in calls:
            for local in range(count):
                feature = np.full((2 + local, 13), index + 0.25, dtype=np.float32)
                path = root / f"mfcc-{index}.npy"
                np.save(path, feature, allow_pickle=False)
                rows.append(
                    {
                        "segment_id": f"s{index}",
                        "call_id": call_id,
                        "group_id": group_id,
                        "source": "source",
                        "label": label,
                        "sample_weight": 1.0 / count,
                        "start": local + 0.1234567890123,
                        "end": local + 0.9876543210987,
                        "mfcc_path": path.name,
                    }
                )
                index += 1
        manifest = root / "manifest.jsonl"
        write_manifest(manifest, rows)
        return manifest, rows

    def test_cache_is_semantics_preserving_and_weights_total_one_per_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self._fixture(root)
            spec = InputSpec(sequence_features=("mfcc",), global_features=())
            cached = Work9FeatureDataset(
                manifest,
                input_spec=spec,
                project_root=root,
                text_store=None,
                cache_mfcc=True,
            )
            uncached = Work9FeatureDataset(
                manifest,
                input_spec=spec,
                project_root=root,
                text_store=None,
                cache_mfcc=False,
            )
            self.assertEqual(cached.segment_count, 4)
            self.assertEqual(cached.call_count, 3)
            self.assertAlmostEqual(cached.loss_scale, 4.0 / 3.0)
            self.assertAlmostEqual(sum(row["sample_weight"] for row in cached.rows), 3.0)
            self.assertIsNotNone(cached.cache_content_sha256)
            for index in range(len(cached)):
                self.assertTrue(
                    torch.equal(
                        cached[index].sequences["mfcc"],
                        uncached[index].sequences["mfcc"],
                    )
                )

    def test_audio_only_dataset_does_not_require_or_touch_text_store(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self._fixture(root)
            dataset = Work9FeatureDataset(
                manifest,
                input_spec=InputSpec(
                    sequence_features=("mfcc",), global_features=()
                ),
                project_root=root,
                text_store=None,
                cache_mfcc=False,
            )
            self.assertEqual(set(dataset[0].sequences), {"mfcc"})
            self.assertEqual(dict(dataset[0].globals), {})

    def test_random_pairing_changes_only_text_and_stays_in_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, rows = self._fixture(root)
            plan = build_random_train_pairing(rows, seed=17)
            store = FakeTextStore(
                {
                    str(row["segment_id"]): torch.full((768,), index, dtype=torch.float32)
                    for index, row in enumerate(rows)
                }
            )
            dataset = Work9FeatureDataset(
                manifest,
                input_spec=InputSpec(
                    sequence_features=("mfcc",),
                    global_features=("text",),
                    expected_text_dim=768,
                ),
                project_root=root,
                text_store=store,  # type: ignore[arg-type]
                pairing_plan=plan,
                cache_mfcc=False,
            )
            sample = dataset[0]
            receiver = rows[0]
            donor = plan.donor_segment_id(str(receiver["segment_id"]))
            self.assertEqual(store.get_calls, [donor])
            self.assertEqual(sample.segment_id, receiver["segment_id"])
            self.assertEqual(sample.call_id, receiver["call_id"])
            self.assertEqual(sample.label, receiver["label"])
            self.assertNotEqual(donor, sample.segment_id)

    def test_pairing_donor_outside_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, rows = self._fixture(root)
            plan = build_random_train_pairing(rows, seed=17)
            # PairingPlan itself is immutable and valid; a different manifest
            # must fail before any donor lookup.
            write_manifest(manifest, rows[:-1])
            with self.assertRaisesRegex(ValueError, "universe differs"):
                Work9FeatureDataset(
                    manifest,
                    input_spec=InputSpec(global_features=("text",)),
                    project_root=root,
                    text_store=FakeTextStore(  # type: ignore[arg-type]
                        {
                            str(row["segment_id"]): torch.zeros(768)
                            for row in rows
                        }
                    ),
                    pairing_plan=plan,
                )


if __name__ == "__main__":
    unittest.main()
