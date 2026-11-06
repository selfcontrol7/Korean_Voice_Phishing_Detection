from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from paper2_v2.batch import SequenceBatch
from paper2_v2.data import (
    FeatureDataset,
    FeatureSample,
    FeatureValidationError,
    InputSpec,
    collate_samples,
)


class BatchAndLoaderTests(unittest.TestCase):
    def test_masks_lengths_and_metadata_survive_device_transfer(self) -> None:
        samples = [
            FeatureSample(
                sequences={"mfcc": torch.ones(length, 13)},
                globals={"text": torch.full((768,), float(index))},
                label=float(index % 2), sample_weight=1.0,
                call_id=f"c{index}", segment_id=f"s{index}", group_id=f"g{index}",
                source="source", start=0.0, end=float(length),
            )
            for index, length in enumerate((3, 7))
        ]
        spec = InputSpec(sequence_features=("mfcc",), global_features=("text",))
        batch = collate_samples(samples, spec)
        sequence = batch.sequences["mfcc"]
        self.assertEqual(sequence.lengths.tolist(), [3, 7])
        self.assertTrue(torch.equal(sequence.valid_mask, ~sequence.key_padding_mask))
        self.assertEqual(sequence.valid_mask.sum(dim=1).tolist(), [3, 7])
        moved = batch.to("cpu")
        self.assertEqual(moved.segment_ids, ("s0", "s1"))
        self.assertEqual(moved.call_ids, ("c0", "c1"))
        self.assertTrue(torch.equal(moved.sequences["mfcc"].lengths, sequence.lengths))

    def test_sequence_batch_rejects_wrong_mask_polarity(self) -> None:
        with self.assertRaises(ValueError):
            SequenceBatch(
                values=torch.zeros(1, 2, 3),
                lengths=torch.tensor([1], dtype=torch.int64),
                valid_mask=torch.tensor([[True, False]]),
                key_padding_mask=torch.tensor([[True, False]]),
            )

    def test_branch_selective_loader_never_touches_disabled_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "features").mkdir()
            np.save(root / "features/mfcc.npy", np.ones((4, 13), dtype=np.float32))
            np.save(root / "features/text.npy", np.ones(768, dtype=np.float32))
            manifest = root / "manifest.jsonl"
            row = {
                "call_id": "call", "segment_id": "segment", "label": 1,
                "start": 0.0, "end": 1.0, "source": "test",
                "mfcc_path": "features/mfcc.npy",
                "krsbert_text_path": "features/text.npy",
                "egemaps_path": "features/DOES_NOT_EXIST.npy",
                "wav2vec2_path": "features/DOES_NOT_EXIST_EITHER.npy",
            }
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            spec = InputSpec(sequence_features=("mfcc",), global_features=("text",))
            dataset = FeatureDataset(manifest, input_spec=spec, feature_base_dir=root)
            with patch("numpy.load", wraps=np.load) as load_spy:
                sample = dataset[0]
            loaded = {Path(call.args[0]).name for call in load_spy.call_args_list}
            self.assertEqual(loaded, {"mfcc.npy", "text.npy"})
            self.assertEqual(set(sample.sequences), {"mfcc"})
            self.assertEqual(set(sample.globals), {"text"})

    def test_text_only_loader_requires_and_loads_no_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(root / "text.npy", np.ones(768, dtype=np.float32))
            manifest = root / "manifest.jsonl"
            row = {
                "call_id": "call", "segment_id": "segment", "label": 0,
                "start": 0.0, "end": 1.0, "source": "test",
                "krsbert_text_path": "text.npy",
            }
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = FeatureDataset(
                manifest,
                input_spec=InputSpec(global_features=("text",)),
                feature_base_dir=root,
            )
            with patch("numpy.load", wraps=np.load) as load_spy:
                sample = dataset[0]
            self.assertEqual(len(load_spy.call_args_list), 1)
            self.assertEqual(sample.sequences, {})

    def test_strict_loader_rejects_empty_and_nonfinite_sequences(self) -> None:
        for array in (
            np.empty((0, 13), dtype=np.float32),
            np.full((2, 13), np.nan, dtype=np.float32),
        ):
            with self.subTest(shape=array.shape):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    np.save(root / "mfcc.npy", array)
                    manifest = root / "manifest.jsonl"
                    row = {
                        "call_id": "c", "segment_id": "s", "label": 1,
                        "start": 0, "end": 1, "source": "x",
                        "mfcc_path": "mfcc.npy",
                    }
                    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
                    dataset = FeatureDataset(
                        manifest,
                        input_spec=InputSpec(sequence_features=("mfcc",)),
                        feature_base_dir=root,
                    )
                    with self.assertRaises(FeatureValidationError):
                        _ = dataset[0]

    def test_egemaps_is_global_not_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(root / "egemaps.npy", np.ones((1, 88), dtype=np.float32))
            manifest = root / "manifest.jsonl"
            row = {
                "call_id": "c", "segment_id": "s", "label": 1,
                "start": 0, "end": 1, "source": "x", "egemaps_path": "egemaps.npy",
            }
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            spec = InputSpec(global_features=("egemaps",))
            dataset = FeatureDataset(manifest, input_spec=spec, feature_base_dir=root)
            batch = collate_samples([dataset[0]], spec)
            self.assertEqual(batch.globals["egemaps"].shape, (1, 88))
            self.assertNotIn("egemaps", batch.sequences)


if __name__ == "__main__":
    unittest.main()
