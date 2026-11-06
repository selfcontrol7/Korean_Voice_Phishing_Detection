"""Deterministic, relation-aware Work 6 fold construction.

The historical test partition is burned.  It is read only to construct the
relationship graph and quarantine every connected component that touches it;
it can never become an output evaluation partition.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class FoldBuildError(ValueError):
    """Raised when an input cannot support a leakage-safe fold build."""


SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "development": "val",
    "test": "test",
    "burned_test": "test",
    "historical_test": "test",
}
EXACT_AUDIO_TYPES = {
    "audio_sha256",
    "byte_identical_audio",
    "exact_audio",
    "exact_audio_byte_sha256",
    "exact_audio_duplicate",
    "exact_audio_pcm_sha256",
    "exact_audio_sha256",
    "exact_byte_audio",
    "exact_pcm_audio",
    "pcm_sha256",
    "waveform_sha256",
}
REJECTED_RELATION_STATUSES = {"exclude", "excluded", "false", "rejected"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FoldBuildConfig:
    """Frozen partitioning policy for the initial Paper 2 v2 protocol."""

    protocol_id: str = "paper2-v2-grouped-nested-v1"
    grouping_policy_id: str = "hard-components-exact-audio-dedup-v1"
    partition_repeat: int = 0
    outer_folds: int = 5
    validation_parts: int = 5
    partition_seed: int = 20260809

    def __post_init__(self) -> None:
        if self.partition_repeat != 0:
            raise FoldBuildError("initial Work 6 protocol permits repeat 00 only")
        if self.outer_folds != 5:
            raise FoldBuildError("initial Work 6 protocol requires five outer folds")
        if self.validation_parts < 2:
            raise FoldBuildError("validation_parts must be at least two")
        if not self.protocol_id or not self.grouping_policy_id:
            raise FoldBuildError("protocol and grouping policy IDs cannot be empty")


@dataclass(frozen=True)
class CallRecord:
    call_id: str
    label: int
    historical_split: str
    source: str
    audio_byte_sha256: str | None
    audio_pcm_sha256: str | None
    canonical_call_id: str | None
    provisional_primary_status: str | None
    relation_component_id: str | None
    is_burned: bool
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class RelationRecord:
    relation_id: str
    members: tuple[str, ...]
    relation_types: tuple[str, ...]
    source_path: str
    source_row: int

    @property
    def is_exact_audio(self) -> bool:
        return bool(self.relation_types) and set(self.relation_types) <= EXACT_AUDIO_TYPES


@dataclass(frozen=True)
class GroupRecord:
    group_id: str
    call_ids: tuple[str, ...]
    label_counts: tuple[int, int]
    segment_count: int

    @property
    def call_count(self) -> int:
        return len(self.call_ids)


@dataclass(frozen=True)
class FoldBuildResult:
    output_dir: Path
    protocol_path: Path
    assignments_path: Path
    checksums_path: Path
    retained_call_count: int
    quarantined_call_count: int
    exact_duplicate_exclusion_count: int


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        if value not in self.parent:
            raise FoldBuildError(f"relationship references unknown call_id {value!r}")
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1

    def components(self) -> list[tuple[str, ...]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for value in sorted(self.parent):
            grouped[self.find(value)].append(value)
        return sorted(tuple(members) for members in grouped.values())


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(_canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(*parts: object) -> str:
    value = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise FoldBuildError(f"invalid JSON at {path}:{row_number}: {error}") from error
            if not isinstance(row, dict):
                raise FoldBuildError(f"{path}:{row_number} must contain a JSON object")
            rows.append(row)
    if not rows:
        raise FoldBuildError(f"JSONL artifact is empty: {path}")
    return rows


def _first_present(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _parse_binary_label(raw: Any, context: str) -> int:
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int) and raw in (0, 1):
        return raw
    if isinstance(raw, float) and raw in (0.0, 1.0):
        return int(raw)
    normalized = str(raw).strip().lower()
    aliases = {
        "0": 0,
        "benign": 0,
        "non_vishing": 0,
        "non-vishing": 0,
        "negative": 0,
        "1": 1,
        "vishing": 1,
        "positive": 1,
    }
    if normalized not in aliases:
        raise FoldBuildError(f"{context} has non-binary label {raw!r}")
    return aliases[normalized]


def _normalize_split(raw: Any, context: str) -> str:
    normalized = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return SPLIT_ALIASES[normalized]
    except KeyError as error:
        raise FoldBuildError(
            f"{context} has unsupported historical split {raw!r}; expected train/val/test"
        ) from error


def _parse_call_provenance(path: Path) -> dict[str, CallRecord]:
    calls: dict[str, CallRecord] = {}
    for row_number, row in enumerate(_load_jsonl(path), start=1):
        call_id = str(_first_present(row, ("call_id", "id")) or "").strip()
        if not call_id:
            raise FoldBuildError(f"{path}:{row_number} has no call_id")
        if call_id in calls:
            raise FoldBuildError(f"duplicate call_id {call_id!r} in {path}")
        if "label" not in row:
            raise FoldBuildError(f"{path}:{row_number} has no label")
        split_raw = _first_present(
            row, ("historical_split", "original_split", "legacy_split", "split")
        )
        if split_raw is None:
            raise FoldBuildError(f"{path}:{row_number} has no historical_split")
        audio = row.get("audio", {})
        if audio is None:
            audio = {}
        if not isinstance(audio, Mapping):
            raise FoldBuildError(f"{path}:{row_number} audio must be an object")

        def parse_hash(raw: Any, name: str) -> str | None:
            if raw in (None, "", "unknown"):
                return None
            value = str(raw).strip().lower()
            if not SHA256_RE.fullmatch(value):
                raise FoldBuildError(
                    f"{path}:{row_number} {name} must be lowercase SHA-256 or unknown"
                )
            return value

        byte_hash_raw = _first_present(
            audio, ("byte_sha256", "file_sha256", "audio_sha256")
        )
        if byte_hash_raw is None:
            byte_hash_raw = _first_present(
                row, ("audio_sha256", "waveform_sha256", "exact_audio_sha256")
            )
        pcm_hash_raw = _first_present(
            audio, ("pcm_sha256", "decoded_pcm_sha256", "waveform_pcm_sha256")
        )
        canonical_call_id_raw = row.get("canonical_call_id")
        canonical_call_id = (
            str(canonical_call_id_raw).strip()
            if canonical_call_id_raw not in (None, "", "unknown")
            else None
        )
        status_raw = row.get("provisional_primary_status")
        status = (
            str(status_raw).strip().lower()
            if status_raw not in (None, "", "unknown")
            else None
        )
        component_raw = row.get("relation_component_id")
        relation_component_id = (
            str(component_raw).strip()
            if component_raw not in (None, "", "unknown")
            else None
        )
        historical_split = _normalize_split(split_raw, f"{path}:{row_number}")
        is_burned_raw = row.get("is_burned", historical_split == "test")
        if not isinstance(is_burned_raw, bool):
            raise FoldBuildError(f"{path}:{row_number} is_burned must be Boolean")
        if is_burned_raw != (historical_split == "test"):
            raise FoldBuildError(
                f"{path}:{row_number} is_burned disagrees with historical_split"
            )
        calls[call_id] = CallRecord(
            call_id=call_id,
            label=_parse_binary_label(row["label"], f"{path}:{row_number}"),
            historical_split=historical_split,
            source=str(row.get("source", "unknown")),
            audio_byte_sha256=parse_hash(byte_hash_raw, "audio.byte_sha256"),
            audio_pcm_sha256=parse_hash(pcm_hash_raw, "audio.pcm_sha256"),
            canonical_call_id=canonical_call_id,
            provisional_primary_status=status,
            relation_component_id=relation_component_id,
            is_burned=is_burned_raw,
            raw=dict(row),
        )
    for call in calls.values():
        if call.canonical_call_id and call.canonical_call_id not in calls:
            raise FoldBuildError(
                f"call {call.call_id!r} names absent canonical_call_id "
                f"{call.canonical_call_id!r}"
            )
    return calls


def _member_id(raw: Any) -> str:
    if isinstance(raw, Mapping):
        raw = _first_present(raw, ("call_id", "id"))
    return str(raw or "").strip()


def _parse_relation_members(row: Mapping[str, Any]) -> tuple[str, ...]:
    for key in ("call_ids", "member_call_ids", "members"):
        if key in row:
            raw_members = row[key]
            if not isinstance(raw_members, list):
                raise FoldBuildError(f"relation field {key!r} must be a list")
            return tuple(sorted({_member_id(member) for member in raw_members if _member_id(member)}))
    edge_aliases = (
        ("call_id_a", "call_id_b"),
        ("left_call_id", "right_call_id"),
        ("source_call_id", "target_call_id"),
    )
    for left_key, right_key in edge_aliases:
        if left_key in row or right_key in row:
            return tuple(sorted({_member_id(row.get(left_key)), _member_id(row.get(right_key))} - {""}))
    return ()


def _normalize_relation_type(raw: Any) -> str:
    return str(raw).strip().lower().replace("-", "_").replace(" ", "_")


def _parse_relation_types(row: Mapping[str, Any]) -> tuple[str, ...]:
    raw = _first_present(
        row,
        (
            "accepted_relation_types",
            "relation_types",
            "types",
            "relation_type",
            "type",
            "kind",
        ),
    )
    if raw is None:
        return ("unspecified_hard_relation",)
    values = raw if isinstance(raw, list) else [raw]
    normalized = sorted({_normalize_relation_type(value) for value in values if str(value).strip()})
    return tuple(normalized or ["unspecified_hard_relation"])


def _relation_is_included(row: Mapping[str, Any]) -> bool:
    if "hard_group_edge" in row:
        hard_edge = row["hard_group_edge"]
        if not isinstance(hard_edge, bool):
            raise FoldBuildError("hard_group_edge must be Boolean")
        decision = str(row.get("decision", "")).strip().lower()
        if decision in {"pending", "review", "review_required", "unresolved"}:
            raise FoldBuildError("pending relation decision must be resolved before Work 6")
        return hard_edge
    explicit = _first_present(row, ("include_in_grouping", "included_in_grouping", "accepted"))
    if explicit is not None:
        if isinstance(explicit, bool):
            return explicit
        normalized = str(explicit).strip().lower()
        if normalized in {"false", "0", "no"}:
            return False
        if normalized in {"true", "1", "yes"}:
            return True
        raise FoldBuildError(f"invalid include_in_grouping value {explicit!r}")
    status = str(row.get("status", "")).strip().lower()
    return status not in REJECTED_RELATION_STATUSES


def _normalized_string_set(raw: Any, context: str) -> set[str]:
    if not isinstance(raw, list):
        raise FoldBuildError(f"{context} must be a list")
    return {str(value) for value in raw}


def _validate_work5_component_summary(
    row: Mapping[str, Any],
    *,
    members: tuple[str, ...],
    calls: Mapping[str, CallRecord],
    context: str,
) -> None:
    if "member_count" in row and int(row["member_count"]) != len(members):
        raise FoldBuildError(f"{context} member_count disagrees with members")
    expected_labels = {str(calls[call_id].label) for call_id in members}
    if "labels" in row:
        actual_labels = _normalized_string_set(row["labels"], f"{context} labels")
        if actual_labels != expected_labels:
            raise FoldBuildError(
                f"{context} labels disagree with provenance: {actual_labels} != {expected_labels}"
            )
    expected_sources = {calls[call_id].source for call_id in members}
    if "sources" in row:
        actual_sources = _normalized_string_set(row["sources"], f"{context} sources")
        if actual_sources != expected_sources:
            raise FoldBuildError(f"{context} sources disagree with provenance")
    expected_splits = {calls[call_id].historical_split for call_id in members}
    if "historical_splits" in row:
        actual_splits = _normalized_string_set(
            row["historical_splits"], f"{context} historical_splits"
        )
        if actual_splits != expected_splits:
            raise FoldBuildError(f"{context} historical_splits disagree with provenance")
    burned = {call_id for call_id in members if calls[call_id].historical_split == "test"}
    if "burned_call_ids" in row:
        actual_burned = _normalized_string_set(
            row["burned_call_ids"], f"{context} burned_call_ids"
        )
        if actual_burned != burned:
            raise FoldBuildError(f"{context} burned_call_ids disagree with provenance")
    if "touches_burned" in row:
        if not isinstance(row["touches_burned"], bool):
            raise FoldBuildError(f"{context} touches_burned must be Boolean")
        if row["touches_burned"] != bool(burned):
            raise FoldBuildError(f"{context} touches_burned disagrees with provenance")
    if "development_members_to_quarantine" in row:
        expected_quarantine = (
            {call_id for call_id in members if calls[call_id].historical_split != "test"}
            if burned
            else set()
        )
        actual_quarantine = _normalized_string_set(
            row["development_members_to_quarantine"],
            f"{context} development_members_to_quarantine",
        )
        if actual_quarantine != expected_quarantine:
            raise FoldBuildError(
                f"{context} development_members_to_quarantine disagrees with graph policy"
            )
    if "redundant_exact_audio_members" in row:
        expected_redundant = {
            call_id
            for call_id in members
            if calls[call_id].provisional_primary_status
            == "exclude_redundant_exact_audio"
        }
        actual_redundant = _normalized_string_set(
            row["redundant_exact_audio_members"],
            f"{context} redundant_exact_audio_members",
        )
        if actual_redundant != expected_redundant:
            raise FoldBuildError(
                f"{context} redundant_exact_audio_members disagree with call provenance"
            )


def _parse_relations(
    paths: Sequence[Path], calls: Mapping[str, CallRecord]
) -> list[RelationRecord]:
    relations: list[RelationRecord] = []
    seen_ids: set[str] = set()
    work5_component_membership: dict[str, str] = {}
    saw_work5_component_inventory = False
    for path in sorted(paths, key=lambda item: str(item)):
        for row_number, row in enumerate(_load_jsonl(path), start=1):
            if not _relation_is_included(row):
                continue
            members = _parse_relation_members(row)
            component_form = any(key in row for key in ("call_ids", "member_call_ids", "members"))
            minimum_members = 1 if component_form else 2
            if len(members) < minimum_members:
                raise FoldBuildError(
                    f"{path}:{row_number} included "
                    f"{'component' if component_form else 'edge'} requires at least "
                    f"{minimum_members} member(s)"
                )
            unknown = sorted(set(members) - set(calls))
            if unknown:
                raise FoldBuildError(
                    f"{path}:{row_number} references calls absent from provenance: {unknown}"
                )
            relation_id = str(
                _first_present(row, ("relation_id", "component_id", "id"))
                or f"{path.name}:{row_number}"
            )
            globally_unique_id = f"{path.name}:{relation_id}"
            if globally_unique_id in seen_ids:
                raise FoldBuildError(f"duplicate relation ID {globally_unique_id!r}")
            seen_ids.add(globally_unique_id)
            is_work5_component = any(
                key in row
                for key in (
                    "member_count",
                    "accepted_relation_types",
                    "touches_burned",
                    "development_members_to_quarantine",
                )
            )
            if is_work5_component:
                saw_work5_component_inventory = True
                _validate_work5_component_summary(
                    row,
                    members=members,
                    calls=calls,
                    context=f"{path}:{row_number}",
                )
                component_id = str(row.get("component_id", relation_id))
                for call_id in members:
                    previous = work5_component_membership.get(call_id)
                    if previous is not None and previous != component_id:
                        raise FoldBuildError(
                            f"call {call_id!r} occurs in multiple Work 5 components: "
                            f"{previous!r}, {component_id!r}"
                        )
                    work5_component_membership[call_id] = component_id
                    declared = calls[call_id].relation_component_id
                    if declared is not None and declared != component_id:
                        raise FoldBuildError(
                            f"call {call_id!r} provenance component {declared!r} disagrees "
                            f"with {component_id!r}"
                        )
            relations.append(
                RelationRecord(
                    relation_id=globally_unique_id,
                    members=members,
                    relation_types=_parse_relation_types(row),
                    source_path=str(path),
                    source_row=row_number,
                )
            )
    if saw_work5_component_inventory:
        missing = sorted(set(calls) - set(work5_component_membership))
        if missing:
            raise FoldBuildError(
                "Work 5 group_components must cover every provenance call exactly once; "
                f"missing={missing[:5]} (total {len(missing)})"
            )
    return relations


def _load_segments(
    paths: Sequence[Path], calls: Mapping[str, CallRecord]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    by_call: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_artifacts: dict[str, str] = {}
    segment_ids: set[str] = set()
    for path in sorted(paths, key=lambda item: str(item)):
        source_artifacts[str(path)] = sha256_file(path)
        for row_number, row in enumerate(_load_jsonl(path), start=1):
            call_id = str(row.get("call_id", "")).strip()
            segment_id = str(row.get("segment_id", "")).strip()
            if not call_id or not segment_id:
                raise FoldBuildError(f"{path}:{row_number} requires call_id and segment_id")
            if call_id not in calls:
                raise FoldBuildError(
                    f"{path}:{row_number} call_id {call_id!r} is absent from provenance"
                )
            if segment_id in segment_ids:
                raise FoldBuildError(f"duplicate segment_id {segment_id!r} across manifests")
            segment_ids.add(segment_id)
            if "label" not in row:
                raise FoldBuildError(f"{path}:{row_number} has no label")
            label = _parse_binary_label(row["label"], f"{path}:{row_number}")
            if label != calls[call_id].label:
                raise FoldBuildError(
                    f"{path}:{row_number} label disagrees with call provenance for {call_id}"
                )
            by_call[call_id].append(dict(row))
    missing_calls = sorted(set(calls) - set(by_call))
    if missing_calls:
        raise FoldBuildError(
            "every provenance call must appear in the supplied segment manifests; "
            f"missing {len(missing_calls)} calls, first={missing_calls[:5]}"
        )
    for rows in by_call.values():
        rows.sort(key=lambda row: str(row["segment_id"]))
    return dict(by_call), source_artifacts


def _union_members(union_find: _UnionFind, members: Sequence[str]) -> None:
    anchor = members[0]
    for member in members[1:]:
        union_find.union(anchor, member)


def _components_by_audio_hash(calls: Mapping[str, CallRecord]) -> list[tuple[str, ...]]:
    components: set[tuple[str, ...]] = set()
    for field in ("audio_pcm_sha256", "audio_byte_sha256"):
        hashes: dict[str, list[str]] = defaultdict(list)
        for call in calls.values():
            value = getattr(call, field)
            if value:
                hashes[value].append(call.call_id)
        components.update(
            tuple(sorted(members)) for members in hashes.values() if len(members) > 1
        )
    return sorted(components)


def _component_id(prefix: str, members: Sequence[str]) -> str:
    digest = hashlib.sha256("\x00".join(sorted(members)).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:20]}"


def _build_groups(
    calls: Mapping[str, CallRecord],
    relations: Sequence[RelationRecord],
    segments_by_call: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[
    list[GroupRecord],
    list[dict[str, Any]],
    dict[str, str],
    dict[str, str],
]:
    relation_graph = _UnionFind(calls)
    exact_graph = _UnionFind(calls)
    for relation in relations:
        _union_members(relation_graph, relation.members)
        if relation.is_exact_audio:
            _union_members(exact_graph, relation.members)
    audio_hash_components = _components_by_audio_hash(calls)
    for members in audio_hash_components:
        _union_members(relation_graph, members)
        _union_members(exact_graph, members)
    # Work 5 records the reviewed canonical representative explicitly.  This
    # is also an exact-equivalence edge, even if the original byte hash was
    # unavailable after historical preprocessing.
    for call in calls.values():
        if call.canonical_call_id and call.canonical_call_id != call.call_id:
            relation_graph.union(call.call_id, call.canonical_call_id)
            exact_graph.union(call.call_id, call.canonical_call_id)

    relation_components = relation_graph.components()
    relation_component_by_call: dict[str, tuple[str, ...]] = {}
    quarantined: set[str] = set()
    for members in relation_components:
        for call_id in members:
            relation_component_by_call[call_id] = members
        if any(calls[call_id].historical_split == "test" for call_id in members):
            quarantined.update(members)

    exact_representative: dict[str, str] = {}
    exact_excluded: dict[str, str] = {}
    for members in exact_graph.components():
        labels = {calls[call_id].label for call_id in members}
        if len(members) > 1 and len(labels) != 1:
            raise FoldBuildError(
                "exact-audio equivalence class has conflicting labels: "
                f"{list(members)}"
            )
        eligible = sorted(
            call_id
            for call_id in members
            if calls[call_id].historical_split in {"train", "val"}
            and call_id not in quarantined
        )
        if not eligible:
            continue
        declared_representatives = {
            calls[call_id].canonical_call_id
            for call_id in members
            if calls[call_id].canonical_call_id is not None
        }
        if len(declared_representatives) > 1:
            raise FoldBuildError(
                "exact-audio equivalence class has inconsistent canonical_call_id values: "
                f"members={list(members)}, canonical={sorted(declared_representatives)}"
            )
        if declared_representatives:
            representative = next(iter(declared_representatives))
            if representative not in eligible:
                raise FoldBuildError(
                    "reviewed exact-audio canonical representative is not eligible: "
                    f"members={list(members)}, canonical={representative!r}"
                )
        else:
            representative = eligible[0]
        for call_id in eligible:
            exact_representative[call_id] = representative
            if call_id != representative:
                exact_excluded[call_id] = representative

    retained = sorted(
        call_id
        for call_id, call in calls.items()
        if call.historical_split in {"train", "val"}
        and call_id not in quarantined
        and call_id not in exact_excluded
    )
    if not retained:
        raise FoldBuildError("no eligible calls remain after quarantine/deduplication")

    group_calls: dict[str, list[str]] = defaultdict(list)
    group_id_by_call: dict[str, str] = {}
    for call_id in retained:
        component = relation_component_by_call[call_id]
        group_id = _component_id("grp", component)
        group_calls[group_id].append(call_id)
        group_id_by_call[call_id] = group_id

    groups: list[GroupRecord] = []
    for group_id in sorted(group_calls):
        member_calls = tuple(sorted(group_calls[group_id]))
        labels = Counter(calls[call_id].label for call_id in member_calls)
        groups.append(
            GroupRecord(
                group_id=group_id,
                call_ids=member_calls,
                label_counts=(labels[0], labels[1]),
                segment_count=sum(len(segments_by_call[call_id]) for call_id in member_calls),
            )
        )

    assignments: list[dict[str, Any]] = []
    allowed_work5_statuses = {
        "historical_burned",
        "quarantine_burned_related_component",
        "exclude_redundant_exact_audio",
        "eligible_for_work6",
    }
    for call_id in sorted(calls):
        call = calls[call_id]
        component = relation_component_by_call[call_id]
        row: dict[str, Any] = {
            "call_id": call_id,
            "historical_split": call.historical_split,
            "label": call.label,
            "relation_component_id": _component_id("component", component),
            "relation_component_members": list(component),
            "source": call.source,
        }
        if call.historical_split == "test":
            row.update(
                role="quarantine",
                exclusion_reason="burned_historical_test",
            )
            computed_work5_status = "historical_burned"
        elif call_id in quarantined:
            row.update(
                role="quarantine",
                exclusion_reason="relation_component_touches_burned_historical_test",
            )
            computed_work5_status = "quarantine_burned_related_component"
        elif call_id in exact_excluded:
            row.update(
                role="excluded",
                exclusion_reason="exact_audio_duplicate_noncanonical",
                exact_audio_representative_call_id=exact_excluded[call_id],
            )
            computed_work5_status = "exclude_redundant_exact_audio"
        elif call_id in group_id_by_call:
            row.update(
                role="eligible",
                group_id=group_id_by_call[call_id],
                exact_audio_representative_call_id=exact_representative.get(call_id, call_id),
            )
            computed_work5_status = "eligible_for_work6"
        else:
            raise AssertionError(f"unclassified call {call_id}")
        if call.provisional_primary_status is not None:
            if call.provisional_primary_status not in allowed_work5_statuses:
                raise FoldBuildError(
                    f"call {call_id!r} has unresolved/unknown Work 5 status "
                    f"{call.provisional_primary_status!r}"
                )
            if call.provisional_primary_status != computed_work5_status:
                raise FoldBuildError(
                    f"call {call_id!r} Work 5 status {call.provisional_primary_status!r} "
                    f"disagrees with recomputed status {computed_work5_status!r}"
                )
        row["work5_primary_status"] = computed_work5_status
        assignments.append(row)
    return groups, assignments, group_id_by_call, exact_excluded


def _assignment_score(
    bins: Sequence[Sequence[GroupRecord]], totals: tuple[int, int]
) -> tuple[float, tuple[tuple[str, ...], ...]]:
    n_parts = len(bins)
    target_labels = tuple(total / n_parts for total in totals)
    total_calls = sum(totals)
    total_segments = sum(group.segment_count for bin_ in bins for group in bin_)
    target_calls = total_calls / n_parts
    target_segments = total_segments / n_parts
    score = 0.0
    for bin_ in bins:
        label_counts = [sum(group.label_counts[label] for group in bin_) for label in (0, 1)]
        # Empty class cells make a fold scientifically unusable.  This term is
        # intentionally dominant and also spreads both labels early in greed.
        score += 1_000_000.0 * sum(
            1 for label in (0, 1) if totals[label] > 0 and label_counts[label] == 0
        )
        for label in (0, 1):
            denominator = max(1.0, target_labels[label])
            score += ((label_counts[label] - target_labels[label]) / denominator) ** 2
        call_count = sum(group.call_count for group in bin_)
        segment_count = sum(group.segment_count for group in bin_)
        score += 0.15 * ((call_count - target_calls) / max(1.0, target_calls)) ** 2
        score += 0.02 * (
            (segment_count - target_segments) / max(1.0, target_segments)
        ) ** 2
    serialized = tuple(tuple(sorted(group.group_id for group in bin_)) for bin_ in bins)
    return score, serialized


def _balanced_group_assignment(
    groups: Sequence[GroupRecord], *, n_parts: int, seed: int, purpose: str
) -> tuple[tuple[GroupRecord, ...], ...]:
    if len(groups) < n_parts:
        raise FoldBuildError(
            f"{purpose} needs at least {n_parts} independent groups; found {len(groups)}"
        )
    totals = tuple(sum(group.label_counts[label] for group in groups) for label in (0, 1))
    total_calls = sum(group.call_count for group in groups)
    total_segments = sum(group.segment_count for group in groups)
    target_labels = tuple(total / n_parts for total in totals)
    target_calls = total_calls / n_parts
    target_segments = total_segments / n_parts
    for label in (0, 1):
        supporting_groups = sum(group.label_counts[label] > 0 for group in groups)
        if supporting_groups < n_parts:
            raise FoldBuildError(
                f"{purpose} cannot place label {label} in every partition: "
                f"only {supporting_groups} independent groups for {n_parts} partitions"
            )

    best: tuple[float, tuple[tuple[str, ...], ...], tuple[tuple[GroupRecord, ...], ...]] | None = None
    # Multiple deterministic starts avoid an arbitrary poor greedy solution
    # while retaining byte-identical output for identical inputs and seed.
    for attempt in range(32):
        ordered = sorted(
            groups,
            key=lambda group: (
                -max(
                    group.label_counts[label] / max(1, totals[label])
                    for label in (0, 1)
                ),
                -group.call_count,
                -group.segment_count,
                _stable_hash(seed, purpose, attempt, group.group_id),
            ),
        )
        bins: list[list[GroupRecord]] = [[] for _ in range(n_parts)]
        label_loads = [[0, 0] for _ in range(n_parts)]
        call_loads = [0 for _ in range(n_parts)]
        segment_loads = [0 for _ in range(n_parts)]

        def projected_score(candidate_part: int, group: GroupRecord) -> float:
            score = 0.0
            for part in range(n_parts):
                projected_labels = [
                    label_loads[part][label]
                    + (group.label_counts[label] if part == candidate_part else 0)
                    for label in (0, 1)
                ]
                score += 1_000_000.0 * sum(
                    1
                    for label in (0, 1)
                    if totals[label] > 0 and projected_labels[label] == 0
                )
                for label in (0, 1):
                    denominator = max(1.0, target_labels[label])
                    score += (
                        (projected_labels[label] - target_labels[label]) / denominator
                    ) ** 2
                projected_calls = call_loads[part] + (
                    group.call_count if part == candidate_part else 0
                )
                projected_segments = segment_loads[part] + (
                    group.segment_count if part == candidate_part else 0
                )
                score += 0.15 * (
                    (projected_calls - target_calls) / max(1.0, target_calls)
                ) ** 2
                score += 0.02 * (
                    (projected_segments - target_segments) / max(1.0, target_segments)
                ) ** 2
            return score

        for group in ordered:
            candidates = []
            for part in range(n_parts):
                candidates.append(
                    (
                        projected_score(part, group),
                        _stable_hash(seed, purpose, attempt, group.group_id, part),
                        part,
                    )
                )
            chosen = min(candidates)[2]
            bins[chosen].append(group)
            for label in (0, 1):
                label_loads[chosen][label] += group.label_counts[label]
            call_loads[chosen] += group.call_count
            segment_loads[chosen] += group.segment_count
        final_score, serialized = _assignment_score(bins, totals)
        candidate_bins = tuple(
            tuple(sorted(bin_, key=lambda group: group.group_id)) for bin_ in bins
        )
        candidate = (final_score, serialized, candidate_bins)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    result = best[2]
    for part, bin_ in enumerate(result):
        label_counts = tuple(
            sum(group.label_counts[label] for group in bin_) for label in (0, 1)
        )
        if 0 in label_counts:
            raise FoldBuildError(
                f"{purpose} partition {part} lacks a class after deterministic allocation: "
                f"labels={label_counts}"
            )
    return result


def _call_output_row(
    call: CallRecord,
    *,
    group_id: str,
    segment_count: int,
    partition: str,
    outer_fold: int,
    repeat: int,
) -> dict[str, Any]:
    return {
        "call_id": call.call_id,
        "group_id": group_id,
        "historical_split": call.historical_split,
        "label": call.label,
        "outer_fold": outer_fold,
        "partition": partition,
        "partition_repeat": repeat,
        "segment_count": segment_count,
        "source": call.source,
    }


def _segment_output_rows(
    call_id: str,
    segments_by_call: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    group_id: str,
    partition: str,
    outer_fold: int,
    repeat: int,
) -> list[dict[str, Any]]:
    source_rows = segments_by_call[call_id]
    sample_weight = 1.0 / len(source_rows)
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        row = dict(source)
        row.update(
            group_id=group_id,
            historical_split=row.get("historical_split"),
            outer_fold=outer_fold,
            partition=partition,
            partition_repeat=repeat,
            sample_weight=sample_weight,
        )
        # Do not emit a misleading null historical split on segment rows.
        if row["historical_split"] is None:
            del row["historical_split"]
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["segment_id"]))


def _artifact_summary(
    calls: Sequence[str], calls_by_id: Mapping[str, CallRecord], groups_by_call: Mapping[str, str]
) -> dict[str, Any]:
    label_counts = Counter(calls_by_id[call_id].label for call_id in calls)
    return {
        "call_count": len(calls),
        "group_count": len({groups_by_call[call_id] for call_id in calls}),
        "label_call_counts": {"0": label_counts[0], "1": label_counts[1]},
    }


def _write_checksums(output_dir: Path) -> Path:
    checksum_path = output_dir / "SHA256SUMS.txt"
    files = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    return checksum_path


def verify_checksums(output_dir: str | Path) -> None:
    """Verify every entry in the deterministic fold checksum manifest."""

    root = Path(output_dir).resolve()
    checksum_path = root / "SHA256SUMS.txt"
    if not checksum_path.is_file():
        raise FoldBuildError(f"missing checksum manifest: {checksum_path}")
    declared: set[str] = set()
    for row_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as error:
            raise FoldBuildError(
                f"invalid checksum row {checksum_path}:{row_number}"
            ) from error
        if not SHA256_RE.fullmatch(expected):
            raise FoldBuildError(f"invalid SHA-256 at {checksum_path}:{row_number}")
        if relative in declared:
            raise FoldBuildError(f"duplicate checksum path {relative!r}")
        declared.add(relative)
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise FoldBuildError(f"checksum mismatch for {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if actual != declared:
        raise FoldBuildError(
            f"checksum coverage mismatch; missing={sorted(actual-declared)}, "
            f"stale={sorted(declared-actual)}"
        )


def freeze_fold_tree(output_dir: str | Path) -> None:
    """Seal a validated tree and make its contents read-only.

    This is deliberately separate from generation so the first tree can be
    inspected and independently reproduced before it becomes canonical.
    """

    root = Path(output_dir).resolve()
    validate_fold_tree(root)
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    status = protocol.get("artifact_status")
    if status not in {"generated_unfrozen", "frozen"}:
        raise FoldBuildError(f"cannot freeze artifact with status {status!r}")
    if status == "generated_unfrozen":
        protocol["artifact_status"] = "frozen"
        _write_json(protocol_path, protocol)
    notice_path = root / "READ_ONLY_NOTICE.txt"
    if not notice_path.exists():
        notice_path.write_text(
            "Paper 2 Work 6 grouped folds. Frozen after validation and byte-reproducibility "
            "check. Do not edit in place; build a new protocol ID.\n",
            encoding="utf-8",
        )
    _write_checksums(root)
    verify_checksums(root)
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


def build_grouped_folds(
    *,
    call_provenance_path: str | Path,
    relation_component_paths: Sequence[str | Path],
    segment_manifest_paths: Sequence[str | Path],
    output_dir: str | Path,
    config: FoldBuildConfig = FoldBuildConfig(),
) -> FoldBuildResult:
    """Build the one-repeat/five-fold Work 6 artifacts.

    The output directory must not exist.  This fail-closed behavior prevents a
    later invocation from silently replacing a frozen split.
    """

    provenance_path = Path(call_provenance_path).resolve()
    relation_paths = tuple(Path(path).resolve() for path in relation_component_paths)
    segment_paths = tuple(Path(path).resolve() for path in segment_manifest_paths)
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite fold artifact directory: {destination}")
    if not relation_paths:
        raise FoldBuildError(
            "at least one Work 5 relation artifact is required; use an explicit empty relation "
            "inventory only after Work 5 review"
        )
    if not segment_paths:
        raise FoldBuildError("at least one segment manifest is required")
    for path in (provenance_path, *relation_paths, *segment_paths):
        if not path.is_file():
            raise FileNotFoundError(path)

    calls = _parse_call_provenance(provenance_path)
    relations = _parse_relations(relation_paths, calls)
    segments_by_call, segment_source_hashes = _load_segments(segment_paths, calls)
    groups, assignments, groups_by_call, exact_excluded = _build_groups(
        calls, relations, segments_by_call
    )
    outer_bins = _balanced_group_assignment(
        groups,
        n_parts=config.outer_folds,
        seed=config.partition_seed,
        purpose="outer_eval",
    )

    destination.mkdir(parents=True)
    assignments_path = destination / "assignments.jsonl"
    _write_jsonl(assignments_path, assignments)

    fold_records: list[dict[str, Any]] = []
    eligible_calls = sorted(groups_by_call)
    for outer_fold in range(config.outer_folds):
        outer_eval_groups = set(group.group_id for group in outer_bins[outer_fold])
        inner_candidates = [group for group in groups if group.group_id not in outer_eval_groups]
        validation_bins = _balanced_group_assignment(
            inner_candidates,
            n_parts=config.validation_parts,
            seed=config.partition_seed + 10_000 + outer_fold,
            purpose=f"outer_{outer_fold:02d}_validation",
        )
        val_groups = {group.group_id for group in validation_bins[0]}
        partition_calls = {
            "outer_eval": sorted(
                call_id for call_id in eligible_calls if groups_by_call[call_id] in outer_eval_groups
            ),
            "val": sorted(call_id for call_id in eligible_calls if groups_by_call[call_id] in val_groups),
        }
        partition_calls["train"] = sorted(
            set(eligible_calls) - set(partition_calls["outer_eval"]) - set(partition_calls["val"])
        )
        fold_dir = (
            destination
            / f"repeat_{config.partition_repeat:02d}"
            / f"outer_{outer_fold:02d}"
        )
        manifests: dict[str, dict[str, str]] = {}
        summaries: dict[str, Any] = {}
        for partition in ("train", "val", "outer_eval"):
            call_ids = partition_calls[partition]
            call_rows = [
                _call_output_row(
                    calls[call_id],
                    group_id=groups_by_call[call_id],
                    segment_count=len(segments_by_call[call_id]),
                    partition=partition,
                    outer_fold=outer_fold,
                    repeat=config.partition_repeat,
                )
                for call_id in call_ids
            ]
            segment_rows = [
                row
                for call_id in call_ids
                for row in _segment_output_rows(
                    call_id,
                    segments_by_call,
                    group_id=groups_by_call[call_id],
                    partition=partition,
                    outer_fold=outer_fold,
                    repeat=config.partition_repeat,
                )
            ]
            call_path = fold_dir / f"{partition}_call_manifest.jsonl"
            segment_path = fold_dir / f"{partition}_segment_manifest.jsonl"
            _write_jsonl(call_path, call_rows)
            _write_jsonl(segment_path, sorted(segment_rows, key=lambda row: str(row["segment_id"])))
            manifests[partition] = {
                "call_manifest": call_path.relative_to(destination).as_posix(),
                "segment_manifest": segment_path.relative_to(destination).as_posix(),
            }
            summaries[partition] = {
                **_artifact_summary(call_ids, calls, groups_by_call),
                "segment_count": len(segment_rows),
            }
        fold_records.append(
            {
                "outer_fold": outer_fold,
                "partition_repeat": config.partition_repeat,
                "manifests": manifests,
                "summary": summaries,
            }
        )

    quarantine_count = sum(row["role"] == "quarantine" for row in assignments)
    protocol_path = destination / "protocol.json"
    protocol = {
        "algorithm": {
            "exact_audio_policy": (
                "work5_declared_canonical_else_lexicographically_first_call_retained"
            ),
            "graph_policy": "connected_components_over_all_historical_calls_before_filtering",
            "inner_validation": f"deterministic_1_of_{config.validation_parts}_group_parts",
            "outer_assignment": "deterministic_greedy_multistart_stratified_group_balance",
            "quarantine_policy": "entire_component_if_any_member_is_historical_test",
        },
        "artifact_status": "generated_unfrozen",
        "folds": fold_records,
        "grouping_policy_id": config.grouping_policy_id,
        "input_artifacts": {
            "call_provenance": {
                "path": str(provenance_path),
                "sha256": sha256_file(provenance_path),
            },
            "relation_components": [
                {"path": str(path), "sha256": sha256_file(path)} for path in relation_paths
            ],
            "segment_manifests": [
                {"path": path, "sha256": digest}
                for path, digest in sorted(segment_source_hashes.items())
            ],
        },
        "partition_repeat": config.partition_repeat,
        "partition_seed": config.partition_seed,
        "protocol_id": config.protocol_id,
        "schema_version": "1.0.0",
        "summary": {
            "all_provenance_calls": len(calls),
            "eligible_retained_calls": len(eligible_calls),
            "eligible_groups": len(groups),
            "exact_audio_duplicate_exclusions": len(exact_excluded),
            "historical_split_counts": dict(
                sorted(Counter(call.historical_split for call in calls.values()).items())
            ),
            "quarantined_calls": quarantine_count,
            "relation_records_used": len(relations),
        },
        "terminology": {
            "forbidden_partition_names": ["test", "confirmatory"],
            "outer_partition_name": "outer_eval",
            "outer_eval_interpretation": "within_corpus_outer_development_evaluation",
        },
    }
    _write_json(protocol_path, protocol)
    checksums_path = _write_checksums(destination)
    return FoldBuildResult(
        output_dir=destination,
        protocol_path=protocol_path,
        assignments_path=assignments_path,
        checksums_path=checksums_path,
        retained_call_count=len(eligible_calls),
        quarantined_call_count=quarantine_count,
        exact_duplicate_exclusion_count=len(exact_excluded),
    )


def validate_fold_tree(output_dir: str | Path) -> None:
    """Independently re-check coverage and call/group isolation of a built tree."""

    root = Path(output_dir).resolve()
    protocol_path = root / "protocol.json"
    assignments_path = root / "assignments.jsonl"
    if not protocol_path.is_file() or not assignments_path.is_file():
        raise FoldBuildError("fold tree requires protocol.json and assignments.jsonl")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("terminology", {}).get("outer_partition_name") != "outer_eval":
        raise FoldBuildError("fold protocol does not use the required outer_eval name")
    assignments = _load_jsonl(assignments_path)
    eligible = {str(row["call_id"]) for row in assignments if row.get("role") == "eligible"}
    excluded = {str(row["call_id"]) for row in assignments if row.get("role") != "eligible"}
    group_by_call = {
        str(row["call_id"]): str(row["group_id"])
        for row in assignments
        if row.get("role") == "eligible"
    }
    outer_eval_seen: Counter[str] = Counter()
    folds = protocol.get("folds", [])
    if len(folds) != 5:
        raise FoldBuildError(f"expected five outer folds, found {len(folds)}")
    for fold in folds:
        role_calls: dict[str, set[str]] = {}
        role_groups: dict[str, set[str]] = {}
        manifest_roles = set(fold.get("manifests", {}))
        if manifest_roles != {"train", "val", "outer_eval"}:
            raise FoldBuildError(f"unexpected partition roles: {sorted(manifest_roles)}")
        for role in ("train", "val", "outer_eval"):
            relative = fold["manifests"][role]["segment_manifest"]
            path = root / relative
            rows = _load_jsonl(path)
            calls = {str(row.get("call_id", "")) for row in rows}
            groups = {str(row.get("group_id", "")) for row in rows}
            if "" in calls or "" in groups:
                raise FoldBuildError(f"{path} has blank call_id or group_id")
            if calls & excluded:
                raise FoldBuildError(f"{path} contains quarantined/excluded calls")
            if any(group_by_call.get(call_id) not in groups for call_id in calls):
                raise FoldBuildError(f"{path} group IDs disagree with assignments")
            role_calls[role] = calls
            role_groups[role] = groups
        for left, right in (("train", "val"), ("train", "outer_eval"), ("val", "outer_eval")):
            if role_calls[left] & role_calls[right]:
                raise FoldBuildError(f"call leakage between {left} and {right}")
            if role_groups[left] & role_groups[right]:
                raise FoldBuildError(f"group leakage between {left} and {right}")
        covered = set().union(*role_calls.values())
        if covered != eligible:
            raise FoldBuildError("fold does not cover every eligible call exactly once")
        outer_eval_seen.update(role_calls["outer_eval"])
    if set(outer_eval_seen) != eligible or set(outer_eval_seen.values()) != {1}:
        raise FoldBuildError("eligible calls must occur in outer_eval exactly once")


def tree_digest(output_dir: str | Path) -> str:
    """Return a path-and-bytes digest used by byte-reproducibility tests."""

    root = Path(output_dir).resolve()
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()
