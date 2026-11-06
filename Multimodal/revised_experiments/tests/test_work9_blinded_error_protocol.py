from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import jsonschema

from paper2_v2.work9_blinded_error_protocol import (
    ANNOTATION_SCHEMA_RELATIVE_PATH,
    BLINDED_SCHEMA_FORBIDDEN_PROPERTY_NAMES,
    DRAFT_CONFIG_RELATIVE_PATH,
    DRAFT_SCHEMA_RELATIVE_PATH,
    POST_UNBLIND_SCHEMA_RELATIVE_PATH,
    BlindedErrorProtocolError,
    build_draft_readiness_receipt,
    validate_annotation_payload,
)


class Work9BlindedErrorProtocolDraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.protocol = json.loads(
            (cls.root / DRAFT_CONFIG_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        cls.protocol_schema = json.loads(
            (cls.root / DRAFT_SCHEMA_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        cls.annotation_schema = json.loads(
            (cls.root / ANNOTATION_SCHEMA_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        cls.post_unblind_schema = json.loads(
            (cls.root / POST_UNBLIND_SCHEMA_RELATIVE_PATH).read_text(encoding="utf-8")
        )

    @staticmethod
    def _all_property_names(value: object) -> set[str]:
        names: set[str] = set()
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict):
                names.update(str(name) for name in properties)
            for child in value.values():
                names.update(Work9BlindedErrorProtocolDraftTests._all_property_names(child))
        elif isinstance(value, list):
            for child in value:
                names.update(Work9BlindedErrorProtocolDraftTests._all_property_names(child))
        return names

    @staticmethod
    def _audio_annotation() -> dict[str, object]:
        return {
            "schema_version": "paper2-v2-work9-blinded-reviewer-annotation-1",
            "protocol_id": "paper2-v2-work9-blinded-call-error-analysis-v1",
            "case_id": "EA-ABCDEFGHIJKLMNOP",
            "reviewer_id": "R-01",
            "annotation_round": "independent",
            "review_mode": "redacted_audio_and_redacted_transcript",
            "content_accessibility": {
                "audio": "usable",
                "transcript": "usable",
                "redaction_blocks_judgment": False,
                "overall_assessability": "assessable",
            },
            "transcript_asr": {
                "correctness": "accurate",
                "issue_types": ["none_observed"],
                "comprehension_confidence": 4,
            },
            "speakers": {
                "perceived_count": 2,
                "local_roles": ["unknown"],
                "role_confidence": 3,
            },
            "scam_cues": ["none_observed"],
            "legitimate_cues": ["routine_service_or_support"],
            "perceived_class": {"label": "benign", "confidence": 4},
            "topic": {"primary": "customer_service_or_complaint", "secondary": []},
            "channel_acquisition": {
                "artifacts": ["none_observed"],
                "artifact_severity": "none",
            },
            "shortcut_template": {
                "possible_cues": ["none_observed"],
                "template_likelihood": "unlikely",
            },
            "ambiguity": {"is_ambiguous": False, "reasons": []},
            "privacy_redaction": {
                "possible_unredacted_identifier": False,
                "identifier_types": [],
                "redaction_quality": "adequate",
            },
            "rationale": {
                "summary_nonverbatim": "Routine service interaction with no sensitive request.",
                "contains_verbatim_quote": False,
                "contains_known_identifier": False,
            },
            "form_status": "locked",
        }

    def test_draft_is_schema_valid_and_fail_closed(self) -> None:
        jsonschema.Draft202012Validator.check_schema(self.protocol_schema)
        jsonschema.Draft202012Validator(
            self.protocol_schema, format_checker=jsonschema.FormatChecker()
        ).validate(self.protocol)
        receipt = build_draft_readiness_receipt(self.root)
        self.assertEqual(receipt["status"], "draft_valid_but_not_ready_for_freeze")
        self.assertEqual(receipt["unresolved_pi_decision_count"], 3)
        self.assertEqual(
            [item["decision_id"] for item in receipt["unresolved_pi_decisions"]],
            ["review_team", "governance_authorization", "review_mode"],
        )
        self.assertEqual(receipt["quota_case_maximum_excluding_exhaustive_errors"], 74)
        self.assertTrue(all(value is False for value in receipt["safety"].values()))
        self.assertEqual(
            {item["role"] for item in receipt["opened_files"]},
            {
                "draft_protocol",
                "draft_protocol_schema",
                "blinded_annotation_schema",
                "post_unblind_schema",
            },
        )

    def test_cli_runs_from_canonical_cwd_without_pythonpath_and_fail_closes(self) -> None:
        script = Path("revised_experiments/scripts/check_work9_blinded_error_protocol_draft.py")
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        base_command = [
            sys.executable,
            str(script),
            "--project-root",
            ".",
        ]
        check = subprocess.run(
            base_command,
            cwd=self.root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(check.returncode, 0, check.stderr)
        receipt = json.loads(check.stdout)
        self.assertEqual(receipt["status"], "draft_valid_but_not_ready_for_freeze")
        self.assertTrue(all(value is False for value in receipt["safety"].values()))

        freeze_check = subprocess.run(
            [*base_command, "--require-freeze-ready"],
            cwd=self.root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(freeze_check.returncode, 2, freeze_check.stderr)
        blocked_receipt = json.loads(freeze_check.stdout)
        self.assertEqual(blocked_receipt["unresolved_pi_decision_count"], 3)
        self.assertTrue(
            all(value is False for value in blocked_receipt["safety"].values())
        )

    def test_draft_schema_rejects_any_authorization_or_freeze(self) -> None:
        for field in (
            "scientific_protocol_frozen",
            "case_selection_authorized",
            "prediction_ledger_access_authorized",
            "reviewer_packet_creation_authorized",
        ):
            changed = copy.deepcopy(self.protocol)
            changed[field] = True
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.Draft202012Validator(self.protocol_schema).validate(changed)

    def test_evidence_bindings_are_metadata_only(self) -> None:
        self.assertEqual(len(self.protocol["evidence_bindings"]), 5)
        self.assertTrue(
            all(
                binding["draft_validator_may_open"] is False
                for binding in self.protocol["evidence_bindings"]
            )
        )
        opened = build_draft_readiness_receipt(self.root)["opened_files"]
        self.assertFalse(any("outer_ledger" in item["path"] for item in opened))
        self.assertFalse(any("call_provenance" in item["path"] for item in opened))

    def test_blinded_annotation_schema_contains_no_outcome_or_identity_fields(self) -> None:
        jsonschema.Draft202012Validator.check_schema(self.annotation_schema)
        found = (
            self._all_property_names(self.annotation_schema)
            & BLINDED_SCHEMA_FORBIDDEN_PROPERTY_NAMES
        )
        self.assertEqual(found, set())
        self.assertNotIn("call_id", json.dumps(self.annotation_schema))
        self.assertNotIn("probability", json.dumps(self.annotation_schema))

    def test_audio_annotation_example_is_valid(self) -> None:
        validate_annotation_payload(self.root, self._audio_annotation())

    def test_transcript_only_mode_forces_asr_and_channel_not_assessable(self) -> None:
        payload = self._audio_annotation()
        payload["review_mode"] = "redacted_transcript_only"
        with self.assertRaises(BlindedErrorProtocolError):
            validate_annotation_payload(self.root, payload)

        payload["content_accessibility"]["audio"] = "not_provided"
        payload["transcript_asr"]["correctness"] = "not_assessable_without_audio"
        payload["transcript_asr"]["issue_types"] = ["not_assessable"]
        payload["channel_acquisition"]["artifacts"] = ["not_assessable"]
        payload["channel_acquisition"]["artifact_severity"] = "not_assessable"
        validate_annotation_payload(self.root, payload)

    def test_privacy_exposure_forces_quarantine(self) -> None:
        payload = self._audio_annotation()
        payload["privacy_redaction"]["possible_unredacted_identifier"] = True
        payload["privacy_redaction"]["identifier_types"] = ["person"]
        payload["privacy_redaction"]["redaction_quality"] = "failed"
        with self.assertRaises(BlindedErrorProtocolError):
            validate_annotation_payload(self.root, payload)
        payload["form_status"] = "quarantined_privacy"
        validate_annotation_payload(self.root, payload)

    def test_exclusive_annotation_tags_cannot_be_combined(self) -> None:
        payload = self._audio_annotation()
        payload["scam_cues"] = ["none_observed", "urgency_threat_or_fear"]
        with self.assertRaisesRegex(BlindedErrorProtocolError, "may not combine"):
            validate_annotation_payload(self.root, payload)

    def test_post_unblind_schema_forbids_raw_identity_and_public_case_release(self) -> None:
        jsonschema.Draft202012Validator.check_schema(self.post_unblind_schema)
        top_level = self.post_unblind_schema["properties"]
        for forbidden in (
            "call_id",
            "relation_group_id",
            "original_filename",
            "original_path",
            "audio",
            "transcript",
            "reviewer_name",
        ):
            self.assertNotIn(forbidden, top_level)
        self.assertEqual(top_level["public_release_eligible"]["const"], False)


if __name__ == "__main__":
    unittest.main()
