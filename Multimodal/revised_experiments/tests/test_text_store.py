from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from paper2_v2.data import FeatureDataset, InputSpec
from paper2_v2.text_features import IndexedTextStore, sha256_file


class IndexedTextStoreTests(unittest.TestCase):
    def make_store(self, root: Path) -> Path:
        store = root / "store"
        store.mkdir()
        vectors = np.arange(12, dtype=np.float32).reshape(3, 4)
        np.save(store / "vectors.npy", vectors)
        with (store / "index.jsonl").open("w", encoding="utf-8") as handle:
            for index in range(3):
                handle.write(json.dumps({"row_index": index, "segment_id": f"s{index}"}) + "\n")
        metadata = {
            "status": "sealed",
            "feature_set_id": "test-store",
            "shape": [3, 4],
            "index_sha256": sha256_file(store / "index.jsonl"),
            "vectors_sha256": sha256_file(store / "vectors.npy"),
        }
        (store / "feature_store.json").write_text(json.dumps(metadata), encoding="utf-8")
        return store

    def test_lookup_and_hash_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_store(Path(directory))
            store = IndexedTextStore(path)
            torch.testing.assert_close(store.get("s1"), torch.tensor([4.0, 5.0, 6.0, 7.0]))
            with self.assertRaises(KeyError):
                store.get("missing")
            with (path / "index.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(" ")
            with self.assertRaises(ValueError):
                IndexedTextStore(path)

    def test_dataset_resolves_by_segment_not_legacy_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = IndexedTextStore(self.make_store(root))
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "call_id": "c0",
                        "segment_id": "s0",
                        "label": 0,
                        "start": 0,
                        "end": 1,
                        "krsbert_text_path": "deliberately/missing.npy",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            spec = InputSpec(global_features=("text",), expected_text_dim=4)
            dataset = FeatureDataset(
                manifest, input_spec=spec, feature_base_dir=root, text_store=store
            )
            torch.testing.assert_close(dataset[0].globals["text"], torch.arange(4).float())


if __name__ == "__main__":
    unittest.main()
