from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from paper2_v2.shortcut_probes import (
    CallFeatures,
    ShortcutProbeError,
    _aggregate_prediction_metrics,
    _numeric_matrix,
    load_frozen_folds,
    select_validation_threshold,
    verify_checksum_manifest,
)
from paper2_v2.work5_audit import sha256_file


class ShortcutProbeTests(unittest.TestCase):
    def test_threshold_selection_uses_frozen_tie_break(self) -> None:
        labels = [0, 0, 1, 1]
        probabilities = [0.1, 0.4, 0.6, 0.9]
        threshold, metrics = select_validation_threshold(
            labels,
            probabilities,
            {"grid_start": 0.3, "grid_stop": 0.7, "grid_step": 0.1},
        )
        self.assertEqual(threshold, 0.5)
        self.assertEqual(metrics["macro_f1"], 1.0)

    def test_numeric_probe_rejects_identifier_or_source_input(self) -> None:
        features = {
            "c": CallFeatures(
                call_id="c", label=0, text="text", numeric={"call_duration_seconds": 1.0}
            )
        }
        rows = [{"call_id": "c"}]
        with self.assertRaises(ShortcutProbeError):
            _numeric_matrix(rows, features, ("source",))

    def test_aggregate_metrics_respect_fold_specific_predictions(self) -> None:
        rows = [
            {"label": 0, "probability": 0.4, "prediction": 0, "threshold": 0.5},
            {"label": 1, "probability": 0.4, "prediction": 1, "threshold": 0.3},
        ]
        metrics = _aggregate_prediction_metrics(rows)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["thresholds_used"], [0.3, 0.5])

    def test_checksum_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "artifact.txt").write_text("actual", encoding="utf-8")
            manifest = root / "SHA256SUMS.txt"
            manifest.write_text(f"{'0' * 64}  artifact.txt\n", encoding="utf-8")
            with self.assertRaises(ShortcutProbeError):
                verify_checksum_manifest(root, manifest)

    def test_unfrozen_work6_tree_is_rejected_before_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = root / "protocol.json"
            assignments = root / "assignments.jsonl"
            protocol.write_text(
                json.dumps(
                    {
                        "artifact_status": "generated_unfrozen",
                        "terminology": {"outer_partition_name": "outer_eval"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            assignments.write_text('{"call_id":"c","role":"eligible"}\n', encoding="utf-8")
            checksum = root / "SHA256SUMS.txt"
            checksum.write_text(
                f"{sha256_file(protocol)}  protocol.json\n"
                f"{sha256_file(assignments)}  assignments.jsonl\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ShortcutProbeError, "require a frozen Work 6"):
                load_frozen_folds(root)


if __name__ == "__main__":
    unittest.main()
