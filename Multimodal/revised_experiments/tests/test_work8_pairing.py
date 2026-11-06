from __future__ import annotations

import unittest

from paper2_v2.pairing import (
    PairingItem,
    build_random_train_pairing,
    pairing_plan_for_split,
)


def items(group_count: int = 8, segments_per_group: int = 3) -> list[PairingItem]:
    return [
        PairingItem(
            segment_id=f"segment_{group}_{segment}",
            call_id=f"call_{group}_{segment}",
            group_id=f"group_{group}",
        )
        for group in range(group_count)
        for segment in range(segments_per_group)
    ]


class Work8PairingTests(unittest.TestCase):
    def test_deterministic_one_to_one_group_disjoint_plan(self) -> None:
        rows = items()
        first = build_random_train_pairing(rows, seed=17)
        second = build_random_train_pairing(reversed(rows), seed=17)
        self.assertEqual(dict(first.donor_by_receiver), dict(second.donor_by_receiver))
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(set(first.donor_by_receiver), {row.segment_id for row in rows})
        self.assertEqual(
            set(first.donor_by_receiver.values()), {row.segment_id for row in rows}
        )
        for receiver_id, donor_id in first.donor_by_receiver.items():
            receiver = first.item_by_segment[receiver_id]
            donor = first.item_by_segment[donor_id]
            self.assertNotEqual(receiver.segment_id, donor.segment_id)
            self.assertNotEqual(receiver.call_id, donor.call_id)
            self.assertNotEqual(receiver.group_id, donor.group_id)

    def test_seed_changes_saved_pairing_plan(self) -> None:
        rows = items(group_count=10, segments_per_group=2)
        first = build_random_train_pairing(rows, seed=17)
        second = build_random_train_pairing(rows, seed=29)
        self.assertNotEqual(first.sha256, second.sha256)

    def test_random_pairing_is_train_only_and_only_for_randtrain(self) -> None:
        rows = items()
        self.assertIsNotNone(
            pairing_plan_for_split(
                "AT-cross-randtrain", split_role="train", rows=rows, seed=17
            )
        )
        self.assertIsNone(
            pairing_plan_for_split(
                "AT-cross-randtrain", split_role="val", rows=rows, seed=17
            )
        )
        self.assertIsNone(
            pairing_plan_for_split(
                "AT-cross-randtrain", split_role="outer_eval", rows=rows, seed=17
            )
        )
        self.assertIsNone(
            pairing_plan_for_split("AT-cross", split_role="train", rows=rows, seed=17)
        )
        with self.assertRaises(ValueError):
            build_random_train_pairing(rows, seed=17, split_role="val")

    def test_impossible_dominant_group_is_rejected(self) -> None:
        rows = [
            PairingItem(f"s{index}", f"c{index}", "dominant" if index < 4 else f"g{index}")
            for index in range(6)
        ]
        with self.assertRaisesRegex(ValueError, "more than half"):
            build_random_train_pairing(rows, seed=17)

    def test_call_cannot_have_inconsistent_group_identity(self) -> None:
        rows = [
            PairingItem("s1", "same_call", "g1"),
            PairingItem("s2", "same_call", "g2"),
            PairingItem("s3", "other", "g3"),
            PairingItem("s4", "another", "g4"),
        ]
        with self.assertRaisesRegex(ValueError, "stable group_id"):
            build_random_train_pairing(rows, seed=17)


if __name__ == "__main__":
    unittest.main()

