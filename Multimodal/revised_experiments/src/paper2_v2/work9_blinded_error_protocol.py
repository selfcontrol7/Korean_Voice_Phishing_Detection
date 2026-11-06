"""Outcome-free validation for the draft Work 9 blinded error protocol.

This module deliberately has no case-selection or reviewer-packet functions.
It opens only the draft protocol and its JSON schemas.  In particular, it
does not resolve, hash, parse, or otherwise touch any prediction ledger,
transcript, audio file, call manifest, or call-provenance record.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


DRAFT_CONFIG_RELATIVE_PATH = Path(
    "revised_experiments/configs/"
    "work9_blinded_error_analysis_protocol_draft_v1.json"
)
DRAFT_SCHEMA_RELATIVE_PATH = Path(
    "revised_experiments/registry/schemas/work9_error_analysis/"
    "work9_blinded_error_analysis_protocol_draft.schema.json"
)
ANNOTATION_SCHEMA_RELATIVE_PATH = Path(
    "revised_experiments/registry/schemas/work9_error_analysis/"
    "work9_blinded_reviewer_annotation.schema.json"
)
POST_UNBLIND_SCHEMA_RELATIVE_PATH = Path(
    "revised_experiments/registry/schemas/work9_error_analysis/"
    "work9_unblinded_case_merge.schema.json"
)

EXPECTED_EVIDENCE_BINDINGS = {
    "primary_outer_ledger_index": (
        "revised_experiments/artifacts/work9_runs_v1/_matrix/"
        "outer_ledger_index.json",
        "640ce46c8f4717b29dc97ee7936d3b3331f92878185a1510833ac6dab5c1684e",
    ),
    "capacity_outer_ledger_index": (
        "revised_experiments/artifacts/work9_capacity_matched_runs_v1/_matrix/"
        "outer_ledger_index.json",
        "a4757a44a7c8d38e7194c26ffd0af6e9da3cf94ce5a5f47db39896b6a759db3a",
    ),
    "completed_gate1_report": (
        "revised_experiments/artifacts/work9_gate1_v2/report.json",
        "e52c8d4fa07d9cbd77d36c1901220c5e7c286251f95ec75911209e1dcb77ed72",
    ),
    "work5_call_provenance": (
        "revised_experiments/artifacts/work5_data_audit_v1/call_provenance.jsonl",
        "ec74b5750d33dbdffe6e8cd263b3b26cc4e331a8fbd011b0803b33871ccce8b8",
    ),
    "work6_relation_aware_assignments": (
        "data/paper2_v2/folds/grouped_nested_v1/assignments.jsonl",
        "8ea63672a758b3fbb8cf3125fbcb46bafd4795ae900f0873f8c1142239440332",
    ),
}

EXPECTED_STRATA = (
    ("E_consensus_errors", None, None),
    ("N_near_threshold", 2, 20),
    ("H_high_confidence_correct", 1, 10),
    ("F_high_confidence_flips", 3, 24),
    ("C_random_correct_controls", 2, 20),
)

EXPECTED_PI_DECISIONS = (
    "review_team",
    "governance_authorization",
    "review_mode",
)

BLINDED_SCHEMA_FORBIDDEN_PROPERTY_NAMES = frozenset(
    {
        "call_id",
        "relation_group_id",
        "source",
        "frozen_label",
        "outer_fold",
        "condition",
        "prediction",
        "probability",
        "threshold",
        "error_status",
        "sampling_stratum",
        "capacity_status",
        "original_filename",
        "original_path",
    }
)


class BlindedErrorProtocolError(RuntimeError):
    """Raised when the draft violates a scientific or safety invariant."""


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise BlindedErrorProtocolError(f"expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_json_schema(schema: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError as error:  # pragma: no cover - environment failure
        raise BlindedErrorProtocolError("jsonschema is required") from error
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(
            schema, format_checker=jsonschema.FormatChecker()
        )
        validator.validate(payload)
    except jsonschema.SchemaError as error:
        raise BlindedErrorProtocolError(
            f"invalid JSON schema at {list(error.absolute_schema_path)}: {error.message}"
        ) from error
    except jsonschema.ValidationError as error:
        raise BlindedErrorProtocolError(
            f"draft protocol schema violation at {list(error.absolute_path)}: "
            f"{error.message}"
        ) from error


def _validate_schema_only(schema: Mapping[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError as error:  # pragma: no cover - environment failure
        raise BlindedErrorProtocolError("jsonschema is required") from error
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as error:
        raise BlindedErrorProtocolError(
            f"invalid JSON schema at {list(error.absolute_schema_path)}: {error.message}"
        ) from error


def _all_property_names(schema: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            properties = value.get("properties")
            if isinstance(properties, Mapping):
                names.update(str(key) for key in properties)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(schema)
    return names


def _validate_semantics(
    protocol: Mapping[str, Any], annotation_schema: Mapping[str, Any]
) -> None:
    safety_flags = (
        "scientific_protocol_frozen",
        "case_selection_authorized",
        "prediction_ledger_access_authorized",
        "reviewer_packet_creation_authorized",
    )
    if protocol.get("status") != "draft_not_frozen":
        raise BlindedErrorProtocolError("this validator accepts only the mutable draft")
    if any(protocol.get(key) is not False for key in safety_flags):
        raise BlindedErrorProtocolError(
            "a draft may not freeze science, authorize outcome access, select cases, "
            "or create reviewer packets"
        )

    bindings = protocol.get("evidence_bindings")
    if not isinstance(bindings, list):
        raise BlindedErrorProtocolError("evidence_bindings must be a list")
    actual_bindings: dict[str, tuple[str, str]] = {}
    for item in bindings:
        if not isinstance(item, Mapping):
            raise BlindedErrorProtocolError("evidence binding is not an object")
        if item.get("draft_validator_may_open") is not False:
            raise BlindedErrorProtocolError("draft validator may not open evidence artifacts")
        role = str(item.get("artifact_role"))
        if role in actual_bindings:
            raise BlindedErrorProtocolError(f"duplicate evidence role: {role}")
        actual_bindings[role] = (str(item.get("path")), str(item.get("sha256")))
    if actual_bindings != EXPECTED_EVIDENCE_BINDINGS:
        raise BlindedErrorProtocolError("evidence bindings differ from reviewed metadata")

    construction = protocol.get("case_construction")
    if not isinstance(construction, Mapping):
        raise BlindedErrorProtocolError("case_construction is not an object")
    strata = construction.get("strata")
    if not isinstance(strata, list):
        raise BlindedErrorProtocolError("strata must be a list")
    actual_strata = tuple(
        (item.get("stratum_id"), item.get("quota_per_cell"), item.get("maximum_cases"))
        for item in strata
        if isinstance(item, Mapping)
    )
    if actual_strata != EXPECTED_STRATA:
        raise BlindedErrorProtocolError("stratum order or quotas changed")
    if tuple(construction.get("stratum_priority", ())) != tuple(
        item[0] for item in EXPECTED_STRATA
    ):
        raise BlindedErrorProtocolError("stratum priority differs from the strata")
    tie_breaking = construction.get("tie_breaking")
    if not isinstance(tie_breaking, Mapping) or tie_breaking.get("secret_commitment") is not None:
        raise BlindedErrorProtocolError(
            "the draft draw-key commitment must remain unresolved"
        )

    decisions = protocol.get("unresolved_pi_decisions")
    if not isinstance(decisions, list):
        raise BlindedErrorProtocolError("unresolved_pi_decisions must be a list")
    decision_ids = tuple(
        str(item.get("decision_id")) for item in decisions if isinstance(item, Mapping)
    )
    if decision_ids != EXPECTED_PI_DECISIONS:
        raise BlindedErrorProtocolError("the draft must expose the exact three PI decisions")
    if any(
        not isinstance(item, Mapping)
        or item.get("blocking") is not True
        or item.get("current_value") is not None
        for item in decisions
    ):
        raise BlindedErrorProtocolError("all draft PI decisions must remain unresolved blockers")

    forbidden_found = sorted(
        BLINDED_SCHEMA_FORBIDDEN_PROPERTY_NAMES & _all_property_names(annotation_schema)
    )
    if forbidden_found:
        raise BlindedErrorProtocolError(
            f"blinded annotation schema exposes forbidden properties: {forbidden_found}"
        )

    analysis_rules = protocol.get("analysis_rules")
    claim_boundaries = protocol.get("claim_boundaries")
    if not isinstance(analysis_rules, Mapping) or analysis_rules.get("descriptive_only") is not True:
        raise BlindedErrorProtocolError("the analysis must remain descriptive")
    if analysis_rules.get("hypothesis_tests_authorized") is not False:
        raise BlindedErrorProtocolError("hypothesis tests are not authorized")
    if not isinstance(claim_boundaries, Mapping):
        raise BlindedErrorProtocolError("claim boundaries are missing")
    forbidden_claims = set(claim_boundaries.get("forbidden", ()))
    required_forbidden = {
        "source_independent_fraud_detection",
        "reopening_or_overturning_Gate1_or_Gate2",
        "real_time_mobile_deployable_robust_or_privacy_preserving_system_claims",
    }
    if not required_forbidden <= forbidden_claims:
        raise BlindedErrorProtocolError("critical claim boundaries are missing")


def build_draft_readiness_receipt(project_root: str | Path) -> dict[str, Any]:
    """Validate draft-only files and return a non-authorizing readiness receipt."""

    root = Path(project_root).resolve()
    paths = {
        "draft_protocol": root / DRAFT_CONFIG_RELATIVE_PATH,
        "draft_protocol_schema": root / DRAFT_SCHEMA_RELATIVE_PATH,
        "blinded_annotation_schema": root / ANNOTATION_SCHEMA_RELATIVE_PATH,
        "post_unblind_schema": root / POST_UNBLIND_SCHEMA_RELATIVE_PATH,
    }
    protocol = _load_json_object(paths["draft_protocol"])
    protocol_schema = _load_json_object(paths["draft_protocol_schema"])
    annotation_schema = _load_json_object(paths["blinded_annotation_schema"])
    post_unblind_schema = _load_json_object(paths["post_unblind_schema"])

    _validate_json_schema(protocol_schema, protocol)
    # Check both data schemas themselves without creating or opening any data.
    for schema in (annotation_schema, post_unblind_schema):
        _validate_schema_only(schema)
    _validate_semantics(protocol, annotation_schema)

    blockers = [
        {
            "decision_id": item["decision_id"],
            "question": item["question"],
            "required_choice": item["required_choice"],
        }
        for item in protocol["unresolved_pi_decisions"]
    ]
    return {
        "schema_version": "paper2-v2-work9-blinded-error-draft-readiness-1",
        "status": "draft_valid_but_not_ready_for_freeze",
        "protocol_id": protocol["protocol_id"],
        "opened_files": [
            {
                "role": role,
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
            }
            for role, path in paths.items()
        ],
        "unresolved_pi_decisions": blockers,
        "unresolved_pi_decision_count": len(blockers),
        "safety": {
            "scientific_protocol_frozen": False,
            "case_selection_authorized": False,
            "prediction_ledgers_opened": False,
            "prediction_values_opened": False,
            "call_names_opened": False,
            "transcripts_opened": False,
            "audio_opened": False,
            "named_errors_opened": False,
            "reviewer_packets_created": False,
            "outcome_analysis_launched": False,
        },
        "quota_case_maximum_excluding_exhaustive_errors": 74,
        "next_state": (
            "PI resolves review_team, governance_authorization, and review_mode; "
            "then create and independently audit a new freeze candidate"
        ),
    }


def validate_annotation_payload(
    project_root: str | Path, payload: Mapping[str, Any]
) -> None:
    """Validate one synthetic or future blinded form against the draft schema."""

    root = Path(project_root).resolve()
    schema = _load_json_object(root / ANNOTATION_SCHEMA_RELATIVE_PATH)
    _validate_json_schema(schema, payload)
    exclusive_tokens = {"none_observed", "not_assessable"}
    array_paths = (
        ("transcript_asr", "issue_types"),
        (None, "scam_cues"),
        (None, "legitimate_cues"),
        ("channel_acquisition", "artifacts"),
        ("shortcut_template", "possible_cues"),
    )
    for parent, name in array_paths:
        value = payload[name] if parent is None else payload[parent][name]
        if len(value) > 1 and exclusive_tokens.intersection(value):
            raise BlindedErrorProtocolError(
                f"{parent + '.' if parent else ''}{name} may not combine "
                "none_observed/not_assessable with another tag"
            )
    ambiguity = payload["ambiguity"]
    if bool(ambiguity["is_ambiguous"]) != bool(ambiguity["reasons"]):
        raise BlindedErrorProtocolError(
            "ambiguity reasons must be nonempty exactly when is_ambiguous is true"
        )
    privacy = payload["privacy_redaction"]
    if bool(privacy["possible_unredacted_identifier"]) != bool(
        privacy["identifier_types"]
    ):
        raise BlindedErrorProtocolError(
            "identifier_types must be nonempty exactly when exposure is possible"
        )
