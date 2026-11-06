from __future__ import annotations

import unittest

from paper2_v2.work5_audit import (
    UnionFind,
    Work5Audit,
    natural_key,
    normalize_text_canonical,
    normalize_text_strict,
    normalized_stem,
    pseudonym,
    stable_id,
)


class Work5AuditUnitTests(unittest.TestCase):
    def test_text_normalization_is_explicit_and_stable(self) -> None:
        self.assertEqual(normalize_text_strict(" Ａ\n한글  "), "A 한글")
        self.assertEqual(normalize_text_canonical(" Ａ, 한글! 12 "), "A한글12")

    def test_nfkc_stem_join_handles_fss_compatibility_characters(self) -> None:
        self.assertEqual(normalized_stem("(대출사기 ①).mp3"), "(대출사기 1)")
        self.assertEqual(normalized_stem("NEW-ⓐ.json"), "NEW-a")

    def test_aihub_member_mapping_preserves_three_digit_job_code(self) -> None:
        self.assertEqual(
            Work5Audit._aihub_member("d01j020s020193"),
            ("TL_D01.zip", "D01/J020/S020193/S020193.json"),
        )
        self.assertEqual(
            Work5Audit._aihub_member("d02j12s004943"),
            ("TL_D02.zip", "D02/J12/S004943/S004943.json"),
        )

    def test_component_ids_do_not_depend_on_union_order(self) -> None:
        first = UnionFind(["vishing_2", "vishing_10", "vishing_3"])
        second = UnionFind(["vishing_2", "vishing_10", "vishing_3"])
        first.union("vishing_10", "vishing_3")
        first.union("vishing_2", "vishing_10")
        second.union("vishing_2", "vishing_3")
        second.union("vishing_10", "vishing_2")
        self.assertEqual(first.components(), second.components())
        members = first.components()[0]
        self.assertEqual(members, ["vishing_2", "vishing_3", "vishing_10"])
        self.assertEqual(
            stable_id("relation_component", *members),
            stable_id("relation_component", *second.components()[0]),
        )

    def test_pseudonym_never_contains_raw_value(self) -> None:
        raw = "sensitive original filename with a person name"
        value = pseudonym("fss_record", raw)
        self.assertNotIn(raw, value)
        self.assertRegex(value, r"^fss_record_[0-9a-f]{24}$")

    def test_natural_call_sort_is_numeric(self) -> None:
        values = ["vishing_10", "vishing_2", "vishing_1"]
        self.assertEqual(sorted(values, key=natural_key), ["vishing_1", "vishing_2", "vishing_10"])


if __name__ == "__main__":
    unittest.main()
