from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper2_v2.work9_gate1_figures import (
    COMPARISONS,
    CONDITIONS,
    FIGURE_SPEC_RELATIVE,
    SEEDS,
    CallPrediction,
    Gate1FigureData,
    Gate1FigureError,
    _validate_condition_records,
    compute_call_metrics,
    expected_figure_ids,
    load_figure_spec,
    render_figures,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]


def _synthetic_data() -> Gate1FigureData:
    calls: list[CallPrediction] = []
    records = []
    for condition_index, condition in enumerate(CONDITIONS):
        for seed_index, seed in enumerate(SEEDS):
            pooled: list[CallPrediction] = []
            for fold in range(5):
                coordinate: list[CallPrediction] = []
                for call_index in range(10):
                    label = call_index % 2
                    base = 0.22 + 0.055 * condition_index
                    signed = (0.24 + 0.008 * seed_index - 0.006 * fold) * (1 if label else -1)
                    probability = min(0.97, max(0.03, base + signed))
                    call = CallPrediction(
                        condition=condition,
                        outer_fold=fold,
                        seed=seed,
                        call_id=f"f{fold}-c{call_index}",
                        group_id=f"g{fold}-{call_index // 2}",
                        label=label,
                        probability=probability,
                        hard_prediction=int(probability >= 0.5),
                        threshold=0.5,
                    )
                    coordinate.append(call)
                    calls.append(call)
                pooled.extend(coordinate)
                records.append(
                    {
                        "schema_version": "paper2-v2-work9-gate1-condition-metrics-1",
                        "scope": "outer_fold_per_seed",
                        "condition": condition,
                        "seed": seed,
                        "outer_fold": fold,
                        "metrics": compute_call_metrics(coordinate),
                    }
                )
            records.append(
                {
                    "schema_version": "paper2-v2-work9-gate1-condition-metrics-1",
                    "scope": "pooled_five_outer_folds_per_seed",
                    "condition": condition,
                    "seed": seed,
                    "outer_fold": None,
                    "metrics": compute_call_metrics(pooled),
                }
            )
    records.sort(
        key=lambda row: (
            CONDITIONS.index(row["condition"]),
            row["seed"],
            -1 if row["outer_fold"] is None else row["outer_fold"],
        )
    )
    comparison_rows = []
    comparators = ("A-query", "AT-concat", "AT-gate")
    effects = (0.035, 0.022, 0.015)
    for index, (comparison, comparator, effect) in enumerate(
        zip(COMPARISONS, comparators, effects, strict=True)
    ):
        seed_values = [effect + offset for offset in (-0.010, -0.004, 0.0, 0.005, 0.009)]
        fold_values = {
            str(fold): [
                effect + 0.002 * (fold - 2) + offset
                for offset in (-0.010, -0.004, 0.0, 0.005, 0.009)
            ]
            for fold in range(5)
        }
        comparison_rows.append(
            {
                "schema_version": "paper2-v2-work9-gate1-primary-comparison-1",
                "comparison_index": index,
                "comparison_id": comparison,
                "observed_effect": effect,
                "bootstrap": {
                    "method": "paired_relation_group_cluster_keep_all_calls_segments_seeds",
                    "replicates": 10000,
                    "confidence_level": 0.95,
                    "ci_lower": effect - 0.012,
                    "ci_upper": effect + 0.014,
                },
                "multiplicity": {
                    "method": "holm",
                    "family_size": 3,
                    "familywise_alpha": 0.05,
                    "adjusted_p_value": 0.01 * (index + 1),
                },
                "seed_stability": {
                    "per_seed_differences": {
                        str(seed): value
                        for seed, value in zip(SEEDS, seed_values, strict=True)
                    }
                },
                "fold_stability": {
                    "per_fold_per_seed_differences": fold_values,
                },
                "primary_condition": "AT-cross",
                "comparator_condition": comparator,
            }
        )
    return Gate1FigureData(tuple(records), tuple(comparison_rows), tuple(calls))


class Work9Gate1FigureTests(unittest.TestCase):
    def test_frozen_preoutcome_spec_binds_current_upstream_and_implementation(self) -> None:
        spec = load_figure_spec(ROOT)
        self.assertEqual(
            [row["figure_id"] for row in spec["figures"]],
            list(expected_figure_ids()),
        )
        self.assertFalse(spec["data_access"]["checkpoints_permitted"])
        self.assertFalse(spec["data_access"]["outer_segment_ledgers_permitted"])
        self.assertTrue(spec["data_access"]["outer_complete_call_ledgers_permitted"])
        self.assertFalse(spec["claim_boundary"]["practical_utility_claim_permitted"])
        for binding in spec["upstream_bindings"] + spec["implementation_bindings"]:
            self.assertEqual(
                binding["sha256"], sha256_file(ROOT / binding["path"])
            )

    def test_synthetic_fixture_renders_every_precommitted_figure(self) -> None:
        data = _synthetic_data()
        with tempfile.TemporaryDirectory() as temporary:
            results = render_figures(data, Path(temporary), formats=("png",))
            self.assertEqual(tuple(results), expected_figure_ids())
            self.assertEqual(len(list(Path(temporary).glob("*.png"))), 11)
            self.assertTrue(
                all(path.stat().st_size > 1000 for paths in results.values() for path in paths)
            )

    def test_condition_grid_rejects_missing_or_duplicate_coordinates(self) -> None:
        rows = list(_synthetic_data().condition_records)
        _validate_condition_records(rows)
        rows[-1] = rows[0]
        with self.assertRaisesRegex(Gate1FigureError, "unexpected/duplicate"):
            _validate_condition_records(rows)

    def test_metric_recomputation_uses_complete_calls(self) -> None:
        calls = (
            CallPrediction("T", 0, 17, "a", "g1", 0, 0.1, 0, 0.5),
            CallPrediction("T", 0, 17, "b", "g2", 0, 0.9, 1, 0.5),
            CallPrediction("T", 0, 17, "c", "g3", 1, 0.2, 0, 0.5),
            CallPrediction("T", 0, 17, "d", "g4", 1, 0.8, 1, 0.5),
        )
        metrics = compute_call_metrics(calls)
        self.assertEqual(metrics["confusion"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1})
        self.assertAlmostEqual(metrics["macro_f1"], 0.5)
        self.assertEqual(metrics["n_calls"], 4)

    def test_spec_explicitly_excludes_posthoc_error_and_utility_figures(self) -> None:
        spec = json.loads((ROOT / FIGURE_SPEC_RELATIVE).read_text(encoding="utf-8"))
        excluded = {row["figure_id"]: row["reason"] for row in spec["excluded_figures"]}
        self.assertIn("named_or_ranked_error_examples", excluded)
        self.assertIn("source_stratified_error_or_generalization_claims", excluded)
        self.assertIn("decision_curve_or_operational_utility", excluded)


if __name__ == "__main__":
    unittest.main()
