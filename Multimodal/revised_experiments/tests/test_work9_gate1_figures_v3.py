from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

from paper2_v2.work9_gate1_figures_v3 import (
    COMPARISONS,
    CONDITIONS,
    FIGURE_SPEC_RELATIVE,
    SEEDS,
    VISUAL_CORRECTION,
    CallPrediction,
    Gate1FigureData,
    _metric_profile,
    _primary_effects,
    _probability_distributions,
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
                    margin = 0.20 + 0.02 * condition_index + 0.004 * seed_index
                    probability = 0.5 + margin * (1 if label else -1)
                    probability += 0.002 * (fold - 2)
                    probability = min(0.98, max(0.02, probability))
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
                    calls.append(call)
                    coordinate.append(call)
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

    comparisons = []
    comparators = ("A-query", "AT-concat", "AT-gate")
    effects = (0.001, -0.002, -0.003)
    intervals = ((-0.010, 0.012), (-0.013, 0.006), (-0.015, 0.004))
    for index, (comparison, comparator, effect, interval) in enumerate(
        zip(COMPARISONS, comparators, effects, intervals, strict=True)
    ):
        per_seed = {
            str(seed): effect + offset
            for seed, offset in zip(SEEDS, (-0.002, -0.001, 0.0, 0.001, 0.002), strict=True)
        }
        per_fold = {
            str(fold): [value + 0.0005 * (fold - 2) for value in per_seed.values()]
            for fold in range(5)
        }
        comparisons.append(
            {
                "schema_version": "paper2-v2-work9-gate1-primary-comparison-1",
                "comparison_index": index,
                "comparison_id": comparison,
                "observed_effect": effect,
                "bootstrap": {
                    "method": "paired_relation_group_cluster_keep_all_calls_segments_seeds",
                    "replicates": 10000,
                    "confidence_level": 0.95,
                    "ci_lower": interval[0],
                    "ci_upper": interval[1],
                },
                "multiplicity": {
                    "method": "holm",
                    "family_size": 3,
                    "familywise_alpha": 0.05,
                    "adjusted_p_value": 1.0,
                },
                "seed_stability": {"per_seed_differences": per_seed},
                "fold_stability": {"per_fold_per_seed_differences": per_fold},
                "primary_condition": "AT-cross",
                "comparator_condition": comparator,
            }
        )
    return Gate1FigureData(tuple(records), tuple(comparisons), tuple(calls))


class Work9Gate1FigureV3Tests(unittest.TestCase):
    def test_v3_spec_binds_implementation_and_preserves_inventory(self) -> None:
        spec = load_figure_spec(ROOT)
        self.assertEqual(spec["visual_correction"], VISUAL_CORRECTION)
        self.assertEqual(
            [row["figure_id"] for row in spec["figures"]],
            list(expected_figure_ids()),
        )
        self.assertFalse(spec["visual_correction"]["scientific_values_changed"])
        self.assertFalse(spec["visual_correction"]["figure_inventory_changed"])
        for binding in spec["upstream_bindings"] + spec["implementation_bindings"]:
            self.assertEqual(binding["sha256"], sha256_file(ROOT / binding["path"]))
        self.assertTrue((ROOT / FIGURE_SPEC_RELATIVE).is_file())

    def test_metric_profile_has_visible_ceiling_headroom(self) -> None:
        figure = _metric_profile(_synthetic_data())
        try:
            for axis in figure.axes[:5]:
                self.assertGreaterEqual(axis.get_ylim()[1], 1.03)
        finally:
            plt.close(figure)

    def test_primary_effects_reserves_space_beyond_all_intervals(self) -> None:
        data = _synthetic_data()
        figure = _primary_effects(data)
        try:
            axis = figure.axes[0]
            maximum_upper = max(float(row["bootstrap"]["ci_upper"]) for row in data.comparisons)
            maximum_absolute = max(
                abs(float(row["bootstrap"][key]))
                for row in data.comparisons
                for key in ("ci_lower", "ci_upper")
            )
            self.assertGreater(axis.get_xlim()[1] - maximum_upper, 0.20 * maximum_absolute)
            p_texts = [text for text in axis.texts if text.get_text().startswith("Holm p=")]
            self.assertEqual(len(p_texts), 3)
            self.assertTrue(all(text.get_position()[0] >= 0.98 for text in p_texts))
        finally:
            plt.close(figure)

    def test_probability_figure_discloses_panel_specific_y_scales(self) -> None:
        figure = _probability_distributions(_synthetic_data())
        try:
            text = " ".join(item.get_text() for item in figure.texts)
            self.assertIn("Panel-specific y-axis limits", text)
            self.assertIn("not peak height", text)
        finally:
            plt.close(figure)

    def test_synthetic_fixture_renders_the_same_eleven_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = render_figures(
                _synthetic_data(), Path(temporary), formats=("png",)
            )
            self.assertEqual(tuple(results), expected_figure_ids())
            files = list(Path(temporary).glob("*.png"))
            self.assertEqual(len(files), 11)
            self.assertTrue(all(path.stat().st_size > 1000 for path in files))


if __name__ == "__main__":
    unittest.main()
