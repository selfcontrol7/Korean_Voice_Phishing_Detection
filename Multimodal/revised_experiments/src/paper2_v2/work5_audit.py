"""Deterministic Work 5 provenance, duplicate, relation, and shortcut audit.

This module is deliberately independent of the historical model code.  It
reads v1 inputs, but writes only a new immutable artifact directory.  Raw FSS
filenames and per-speaker demographics are never written to public artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import tempfile
import unicodedata
import wave
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


AUDIT_SCHEMA = "paper2-v2-work5-audit-1"
CALL_SCHEMA = "paper2-v2-call-provenance-1"
RELATION_SCHEMA = "paper2-v2-relation-1"
COMPONENT_SCHEMA = "paper2-v2-relation-component-1"
_AIHUB_ID = re.compile(
    r"^d(?P<domain>\d{2})j(?P<job>\d{2,3})s(?P<session>\d{6})$"
)
_FSS_SPLIT_SUFFIX = re.compile(r"^(?P<parent>.+)-(?P<part>\d{2})$")
_MEDIA_SUFFIXES = {".wav", ".mp3", ".mp4", ".mpeg", ".m4a", ".flac", ".ogg"}


class AuditError(RuntimeError):
    """A reproducibility or evidence invariant failed."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: str, length: int = 24) -> str:
    payload = "\0".join((prefix, *parts)).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:length]}"


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(piece) if piece.isdigit() else piece for piece in re.split(r"(\d+)", value))


def normalize_text_strict(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def normalize_text_canonical(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(character for character in normalized if character.isalnum())


def normalized_stem(filename: str) -> str:
    return unicodedata.normalize("NFKC", os.path.splitext(filename)[0])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )


def pseudonym(namespace: str, raw_value: str) -> str:
    return stable_id(namespace, unicodedata.normalize("NFKC", raw_value))


def _strip_public_asr_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if key != "userdata"}


@dataclass(frozen=True)
class WavAudit:
    byte_sha256: str
    pcm_sha256: str
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int
    frame_count: int
    duration_seconds: float


def inspect_wav(path: Path) -> WavAudit:
    byte_hash = sha256_file(path)
    pcm_hash = hashlib.sha256()
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        sample_rate = audio.getframerate()
        frame_count = audio.getnframes()
        pcm_hash.update(struct.pack("<IIII", channels, sample_width, sample_rate, frame_count))
        while True:
            frames = audio.readframes(1024 * 1024)
            if not frames:
                break
            pcm_hash.update(frames)
    return WavAudit(
        byte_sha256=byte_hash,
        pcm_sha256=pcm_hash.hexdigest(),
        sample_rate_hz=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
        frame_count=frame_count,
        duration_seconds=frame_count / sample_rate,
    )


class UnionFind:
    def __init__(self, members: Iterable[str]) -> None:
        self.parent = {member: member for member in members}

    def find(self, member: str) -> str:
        parent = self.parent[member]
        if parent != member:
            self.parent[member] = self.find(parent)
        return self.parent[member]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if natural_key(left_root) <= natural_key(right_root):
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root

    def components(self) -> list[list[str]]:
        grouped: defaultdict[str, list[str]] = defaultdict(list)
        for member in self.parent:
            grouped[self.find(member)].append(member)
        return sorted(
            (sorted(members, key=natural_key) for members in grouped.values()),
            key=lambda members: natural_key(members[0]),
        )


def _windows_dir_b(*, cmd_path: Path, windows_directory: str) -> list[str]:
    process = subprocess.run(
        [str(cmd_path), "/u", "/c", "dir", "/b", windows_directory],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        stderr = process.stderr.decode("utf-16le", errors="replace").strip()
        raise AuditError(f"Windows enumeration failed for configured FSS directory: {stderr}")
    decoded = process.stdout.decode("utf-16le", errors="strict").replace("\r", "")
    return [line for line in decoded.split("\n") if line]


def _powershell_sha256(*, powershell_path: Path, windows_path: str) -> str:
    quoted = windows_path.replace("'", "''")
    script = (
        f"(Get-FileHash -LiteralPath '{quoted}' -Algorithm SHA256).Hash.ToLowerInvariant()"
    )
    process = subprocess.run(
        [str(powershell_path), "-NoProfile", "-NonInteractive", "-Command", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    value = process.stdout.decode("utf-8", errors="replace").strip().lower()
    if process.returncode != 0 or not re.fullmatch(r"[0-9a-f]{64}", value):
        message = process.stderr.decode("utf-8", errors="replace").strip()
        raise AuditError(f"PowerShell SHA-256 failed: {message}")
    return value


def _validate_expected(actual: Any, expected: Any, description: str) -> None:
    if actual != expected:
        raise AuditError(f"{description}: expected {expected!r}, found {actual!r}")


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


class Work5Audit:
    """Execute the frozen Work 5 audit contract."""

    def __init__(self, *, project_root: Path, config_path: Path) -> None:
        self.root = project_root.resolve()
        self.config_path = config_path.resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.inputs = self.config["inputs"]
        self.expected = self.config["expected"]
        self.calls: list[dict[str, Any]] = []
        self.by_id: dict[str, dict[str, Any]] = {}
        self.relations: list[dict[str, Any]] = []
        self.aihub_rows: list[dict[str, Any]] = []
        self.fss_rows: list[dict[str, Any]] = []
        self.external_inventory: dict[str, Any] = {}
        self.coverage_gaps: list[dict[str, Any]] = []

    def resolve(self, raw_path: str) -> Path:
        path = Path(raw_path)
        return path if path.is_absolute() else self.root / path

    def load_inputs(self) -> None:
        master_path = self.resolve(self.inputs["master_manifest"])
        master = read_jsonl(master_path)
        _validate_expected(len(master), self.expected["calls"], "master call count")
        labels = Counter(str(int(row["label"])) for row in master)
        _validate_expected(dict(sorted(labels.items())), self.expected["labels"], "label counts")

        split_by_call: dict[str, str] = {}
        input_hashes: dict[str, str] = {str(master_path.relative_to(self.root)): sha256_file(master_path)}
        for split, raw_path in self.inputs["split_manifests"].items():
            path = self.resolve(raw_path)
            rows = read_jsonl(path)
            _validate_expected(len(rows), self.expected["splits"][split], f"{split} call count")
            input_hashes[str(path.relative_to(self.root))] = sha256_file(path)
            for row in rows:
                call_id = str(row["call_id"])
                if call_id in split_by_call:
                    raise AuditError(f"call appears in multiple historical splits: {call_id}")
                split_by_call[call_id] = split
        _validate_expected(set(split_by_call), {str(row["call_id"]) for row in master}, "split coverage")

        for row in sorted(master, key=lambda item: natural_key(str(item["call_id"]))):
            call_id = str(row["call_id"])
            audio_path = self.resolve(str(row["audio_filepath"]))
            transcript_path = self.resolve(str(row["transcript_filepath"]))
            if not audio_path.is_file() or not transcript_path.is_file():
                raise AuditError(f"missing current call input: {call_id}")
            transcript_bytes = transcript_path.read_bytes()
            transcript = json.loads(transcript_bytes)
            full_text = transcript.get("text")
            if not isinstance(full_text, str):
                raise AuditError(f"transcript lacks top-level text: {call_id}")
            strict_text = normalize_text_strict(full_text)
            canonical_text = normalize_text_canonical(full_text)
            segments = transcript.get("segments") or []
            if not isinstance(segments, list):
                raise AuditError(f"transcript segments are not a list: {call_id}")
            segment_confidences = [
                value
                for value in (_safe_float(segment.get("confidence")) for segment in segments)
                if value is not None
            ]
            segment_durations = [
                max(0.0, (float(segment.get("end", 0)) - float(segment.get("start", 0))) / 1000.0)
                for segment in segments
            ]
            word_count = sum(
                len(segment.get("words") or [])
                if isinstance(segment.get("words"), list)
                else len(str(segment.get("text", "")).split())
                for segment in segments
            )
            speaker_labels = sorted(
                {
                    str((segment.get("speaker") or {}).get("label"))
                    for segment in segments
                    if (segment.get("speaker") or {}).get("label") is not None
                }
            )
            wav = inspect_wav(audio_path)
            params = transcript.get("params") or {}
            userdata = params.get("userdata") or {}
            record: dict[str, Any] = {
                "schema_version": CALL_SCHEMA,
                "call_id": call_id,
                "label": int(row["label"]),
                "source": str(row.get("source", "unknown")),
                "historical_split": split_by_call[call_id],
                "is_burned": split_by_call[call_id] == "test",
                "audio": {
                    "relative_path": str(audio_path.relative_to(self.root)),
                    "byte_sha256": wav.byte_sha256,
                    "pcm_sha256": wav.pcm_sha256,
                    "sample_rate_hz": wav.sample_rate_hz,
                    "channels": wav.channels,
                    "sample_width_bytes": wav.sample_width_bytes,
                    "frame_count": wav.frame_count,
                    "duration_seconds": round(wav.duration_seconds, 9),
                },
                "transcript": {
                    "relative_path": str(transcript_path.relative_to(self.root)),
                    "byte_sha256": sha256_bytes(transcript_bytes),
                    "strict_text_sha256": sha256_bytes(strict_text.encode("utf-8")),
                    "canonical_text_sha256": sha256_bytes(canonical_text.encode("utf-8")),
                    "strict_character_count": len(strict_text),
                    "canonical_character_count": len(canonical_text),
                    "segment_count": len(segments),
                    "word_count": int(word_count),
                    "mean_segment_duration_seconds": (
                        round(_mean(segment_durations), 9) if segment_durations else None
                    ),
                    "mean_segment_confidence": (
                        round(_mean(segment_confidences), 9) if segment_confidences else None
                    ),
                    "call_confidence": _safe_float(transcript.get("confidence")),
                    "local_speaker_labels": speaker_labels,
                    "local_speaker_count": len(speaker_labels),
                    "asr_engine_version": transcript.get("version"),
                    "asr_public_params_sha256": sha256_bytes(
                        canonical_json_bytes(_strip_public_asr_params(params))
                    ),
                    "asr_domain_code": userdata.get("_ncp_DomainCode"),
                    "asr_domain_id": userdata.get("_ncp_DomainId"),
                },
                "source_provenance": {
                    "status": "unknown",
                    "evidence_method": None,
                    "source_record_id": None,
                    "parent_recording_id": None,
                    "category": None,
                    "subcategory": None,
                    "collection_date": None,
                    "input_type": None,
                    "speaker_roles": [],
                    "telephone_networks": [],
                },
                "privacy": {
                    "raw_external_path_published": False,
                    "original_filename_published": False,
                    "per_call_demographics_published": False,
                },
            }
            record["_strict_text"] = strict_text
            record["_canonical_text"] = canonical_text
            self.calls.append(record)
            self.by_id[call_id] = record
        self.external_inventory["input_manifest_sha256"] = dict(sorted(input_hashes.items()))

    @staticmethod
    def _aihub_member(call_id: str) -> tuple[str, str]:
        match = _AIHUB_ID.fullmatch(call_id)
        if not match:
            raise AuditError(f"invalid AIHub-derived call ID: {call_id}")
        domain, job, session = match.group("domain", "job", "session")
        archive = f"TL_D{domain}.zip"
        member = f"D{domain}/J{job}/S{session}/S{session}.json"
        return archive, member

    def recover_aihub(self) -> dict[str, Any]:
        archive_dir = self.resolve(self.inputs["aihub_archive_dir"])
        benign = [call for call in self.calls if call["label"] == 0]
        by_archive: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for call in benign:
            archive, _ = self._aihub_member(call["call_id"])
            by_archive[archive].append(call)

        sensitive_aggregate = {
            "age": Counter(),
            "gender": Counter(),
            "residence": Counter(),
        }
        duration_deltas: list[float] = []
        archive_inventory: list[dict[str, Any]] = []
        for archive_name in sorted(by_archive):
            archive_path = archive_dir / archive_name
            if not archive_path.is_file():
                raise AuditError(f"required AIHub label archive is unavailable: {archive_name}")
            archive_inventory.append(
                {"archive_name": archive_name, "size_bytes": archive_path.stat().st_size}
            )
            with zipfile.ZipFile(archive_path) as archive:
                for call in sorted(by_archive[archive_name], key=lambda item: natural_key(item["call_id"])):
                    _, member = self._aihub_member(call["call_id"])
                    try:
                        raw = archive.read(member)
                    except KeyError as error:
                        raise AuditError(f"AIHub annotation missing for {call['call_id']}") from error
                    annotation = json.loads(raw)["dataSet"]
                    type_info = annotation["typeInfo"]
                    speakers = type_info.get("speakers") or []
                    dialogs = annotation.get("dialogs") or []
                    original_duration = sum(float(dialog["duration"]) for dialog in dialogs)
                    current_duration = float(call["audio"]["duration_seconds"])
                    duration_delta = current_duration - original_duration
                    duration_deltas.append(duration_delta)
                    for speaker in speakers:
                        for field in sensitive_aggregate:
                            sensitive_aggregate[field][str(speaker.get(field, "unknown"))] += 1
                    source_record_id = pseudonym("aihub_record", member)
                    roles = sorted({str(speaker.get("type")) for speaker in speakers if speaker.get("type")})
                    networks = sorted(
                        {str(speaker.get("telephone_network")) for speaker in speakers if speaker.get("telephone_network")}
                    )
                    human_text = " ".join(str(dialog.get("text", "")) for dialog in dialogs)
                    call["source_provenance"].update(
                        {
                            "status": "verified_exact_archive_identifier",
                            "evidence_method": "call-id-to-AIHub-member exact mapping",
                            "source_record_id": source_record_id,
                            "parent_recording_id": source_record_id,
                            "category": type_info.get("category"),
                            "subcategory": type_info.get("subcategory"),
                            "collection_date": None if annotation.get("date") in {None, "null"} else annotation.get("date"),
                            "input_type": type_info.get("inputType"),
                            "speaker_roles": roles,
                            "telephone_networks": networks,
                            "original_dialog_count": len(dialogs),
                            "original_dialog_duration_seconds": round(original_duration, 9),
                            "human_transcript_strict_sha256": sha256_bytes(
                                normalize_text_strict(human_text).encode("utf-8")
                            ),
                        }
                    )
                    self.aihub_rows.append(
                        {
                            "schema_version": "paper2-v2-aihub-mapping-1",
                            "call_id": call["call_id"],
                            "status": "verified",
                            "archive_name": archive_name,
                            "archive_member_id": pseudonym("aihub_member", member),
                            "annotation_sha256": sha256_bytes(raw),
                            "source_record_id": source_record_id,
                            "category": type_info.get("category"),
                            "subcategory": type_info.get("subcategory"),
                            "speaker_count": len(speakers),
                            "speaker_roles": roles,
                            "telephone_networks": networks,
                            "duration_delta_seconds": round(duration_delta, 9),
                            "duration_validation_within_2_seconds": abs(duration_delta) <= 2.0,
                            "per_call_demographics_withheld": True,
                        }
                    )
        _validate_expected(
            len(self.aihub_rows), self.expected["aihub_exact_mappings"], "AIHub exact mappings"
        )
        return {
            "mapping_count": len(self.aihub_rows),
            "archives": archive_inventory,
            "duration_delta_seconds": {
                "minimum": round(min(duration_deltas), 9),
                "maximum": round(max(duration_deltas), 9),
                "mean": round(float(np.mean(duration_deltas)), 9),
                "within_0_01_seconds": sum(abs(value) <= 0.01 for value in duration_deltas),
                "within_2_seconds": sum(abs(value) <= 2.0 for value in duration_deltas),
            },
            "sensitive_metadata_aggregate_only": {
                field: dict(sorted(counter.items())) for field, counter in sensitive_aggregate.items()
            },
        }

    @staticmethod
    def _strip_all_media_suffixes(filename: str) -> str:
        value = filename
        while Path(value).suffix.lower() in _MEDIA_SUFFIXES | {".txt", ".json"}:
            value = value[: -len(Path(value).suffix)]
        return unicodedata.normalize("NFKC", value)

    def _load_fss_taxonomy(self, fss_wsl_root: Path) -> dict[str, dict[str, Any]]:
        csv_path = (
            fss_wsl_root.parent
            / "All transcripts_new"
            / "FSS_final_lenght_2022_04_01.csv"
        )
        if not csv_path.is_file():
            self.coverage_gaps.append(
                {
                    "area": "fss_taxonomy",
                    "status": "unavailable",
                    "reason": "taxonomy CSV was not available at the configured archive release",
                }
            )
            return {}
        candidates: defaultdict[str, set[tuple[str | None, str | None, str | None]]] = defaultdict(set)
        with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                windows_path = PureWindowsPath(str(row.get("file_name", "")))
                parts = list(windows_path.parts)
                lowered = [part.lower() for part in parts]
                try:
                    marker = lowered.index("by_folder")
                except ValueError:
                    continue
                after = parts[marker + 1 : -1]
                category = after[0] if after else None
                script_status = next(
                    (part for part in after if part.lower() in {"with_scripts", "no_scripts"}),
                    None,
                )
                subcategory = next(
                    (
                        part
                        for part in after[1:]
                        if part.lower() not in {"with_scripts", "no_scripts", "new"}
                    ),
                    None,
                )
                key = self._strip_all_media_suffixes(windows_path.name)
                candidates[key].add((category, subcategory, script_status))
        result: dict[str, dict[str, Any]] = {}
        for key, values in candidates.items():
            if len(values) == 1:
                category, subcategory, script_status = next(iter(values))
                result[key] = {
                    "category": category,
                    "subcategory": subcategory,
                    "script_status": script_status,
                    "evidence": "FSS_final_lenght_2022_04_01.csv unique normalized basename",
                }
        return result

    @staticmethod
    def _decode_audio_ffmpeg(path: Path) -> np.ndarray:
        process = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
                "-acodec", "pcm_f32le", "-ac", "1", "-ar", "16000", "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if process.returncode != 0:
            raise AuditError(
                "ffmpeg sample validation failed: "
                + process.stderr.decode("utf-8", errors="replace")[-500:]
            )
        return np.frombuffer(process.stdout, dtype="<f4").astype(np.float64)

    def _validate_fss_audio_sample(
        self, *, index: int, raw_audio_path: Path, source_record_id: str
    ) -> dict[str, Any]:
        current_path = self.resolve(self.by_id[f"vishing_{index}"]["audio"]["relative_path"])
        raw = self._decode_audio_ffmpeg(raw_audio_path)
        current = self._decode_audio_ffmpeg(current_path)
        length = min(len(raw), len(current))
        if length == 0:
            raise AuditError(f"empty decoded audio in FSS validation sample {index}")
        left, right = raw[:length], current[:length]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        correlation = float(np.dot(left, right) / denominator) if denominator else 0.0
        return {
            "call_id": f"vishing_{index}",
            "source_record_id": source_record_id,
            "decoded_sample_rate_hz": 16000,
            "raw_frame_count": len(raw),
            "current_frame_count": len(current),
            "zero_lag_cosine": round(correlation, 12),
            "frame_count_delta": len(current) - len(raw),
            "status": "verified" if correlation >= 0.999 else "failed",
            "raw_filename_withheld": True,
        }

    def recover_fss(self) -> dict[str, Any]:
        cmd_path = self.resolve(self.inputs["windows_cmd"])
        powershell_path = self.resolve(self.inputs["windows_powershell"])
        fss_wsl_root = self.resolve(self.inputs["fss_wsl_root"])
        windows_audio_dir = str(self.inputs["fss_windows_audio_dir"])
        windows_transcript_dir = str(self.inputs["fss_windows_transcript_dir"])

        audio_names = [
            name
            for name in _windows_dir_b(cmd_path=cmd_path, windows_directory=windows_audio_dir)
            if Path(name).suffix.lower() in _MEDIA_SUFFIXES
        ]
        transcript_names = [
            name
            for name in _windows_dir_b(cmd_path=cmd_path, windows_directory=windows_transcript_dir)
            if name.lower().endswith(".json")
        ]
        audio_names = sorted(audio_names)
        _validate_expected(
            len(audio_names), self.expected["fss_sorted_source_pairs"], "FSS source audio count"
        )
        _validate_expected(
            len(transcript_names), self.expected["fss_sorted_source_pairs"], "FSS source JSON count"
        )
        transcript_by_stem: dict[str, str] = {}
        for name in transcript_names:
            key = normalized_stem(name)
            if key in transcript_by_stem:
                raise AuditError("FSS transcript normalized-stem collision prevents one-to-one mapping")
            transcript_by_stem[key] = name
        audio_stems = [normalized_stem(name) for name in audio_names]
        if len(set(audio_stems)) != len(audio_stems) or set(audio_stems) != set(transcript_by_stem):
            raise AuditError("FSS audio/JSON normalized stems are not a one-to-one bijection")

        split_groups: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        raw_parent_by_index: dict[int, str] = {}
        raw_part_by_index: dict[int, int | None] = {}
        for index, audio_name in enumerate(audio_names, start=1):
            stem = normalized_stem(audio_name)
            match = _FSS_SPLIT_SUFFIX.fullmatch(stem)
            if match:
                split_groups[match.group("parent")].append((index, int(match.group("part"))))
        accepted_split_groups = {
            parent: sorted(members)
            for parent, members in split_groups.items()
            if len(members) >= 2
        }
        for index, audio_name in enumerate(audio_names, start=1):
            stem = normalized_stem(audio_name)
            match = _FSS_SPLIT_SUFFIX.fullmatch(stem)
            if match and match.group("parent") in accepted_split_groups:
                raw_parent_by_index[index] = match.group("parent")
                raw_part_by_index[index] = int(match.group("part"))
            else:
                raw_parent_by_index[index] = stem
                raw_part_by_index[index] = None

        taxonomy_by_stem = self._load_fss_taxonomy(fss_wsl_root)
        current_hash_counts = Counter(
            self.by_id[f"vishing_{index}"]["transcript"]["byte_sha256"]
            for index in range(1, len(audio_names) + 1)
        )
        direct_reads = 0
        powershell_reads = 0
        adjusted_stem_joins = 0
        audio_samples: list[dict[str, Any]] = []
        sample_indices = set(int(value) for value in self.config["fss_mapping"]["audio_validation_sample_indices"])

        for index, audio_name in enumerate(audio_names, start=1):
            call_id = f"vishing_{index}"
            call = self.by_id.get(call_id)
            if call is None:
                raise AuditError(f"sorted FSS mapping produced absent current call: {call_id}")
            audio_stem = normalized_stem(audio_name)
            transcript_name = transcript_by_stem[audio_stem]
            if os.path.splitext(audio_name)[0] != os.path.splitext(transcript_name)[0]:
                adjusted_stem_joins += 1
            transcript_wsl_path = fss_wsl_root / "All transcripts" / transcript_name
            try:
                original_json_hash = sha256_file(transcript_wsl_path)
                read_method = "WSL direct file read"
                direct_reads += 1
            except OSError:
                windows_path = windows_transcript_dir + "\\" + transcript_name
                original_json_hash = _powershell_sha256(
                    powershell_path=powershell_path, windows_path=windows_path
                )
                read_method = "Windows PowerShell Get-FileHash fallback"
                powershell_reads += 1
            current_json_hash = str(call["transcript"]["byte_sha256"])
            if original_json_hash != current_json_hash:
                raise AuditError(
                    f"FSS sort replay JSON mismatch at {call_id}; mapping is not safe to publish"
                )
            source_record_id = pseudonym("fss_record", audio_stem)
            parent_recording_id = pseudonym("fss_parent", raw_parent_by_index[index])
            taxonomy = taxonomy_by_stem.get(self._strip_all_media_suffixes(audio_name), {})
            call["source_provenance"].update(
                {
                    "status": "verified_sorted_rename_replay_and_exact_json",
                    "evidence_method": (
                        "Windows UTF-16 enumeration; Python code-point audio-name sort replay; "
                        "NFKC stem bijection; per-index exact JSON SHA-256"
                    ),
                    "source_record_id": source_record_id,
                    "parent_recording_id": parent_recording_id,
                    "parent_split_part": raw_part_by_index[index],
                    "category": taxonomy.get("category"),
                    "subcategory": taxonomy.get("subcategory"),
                    "script_status": taxonomy.get("script_status"),
                    "collection_date": None,
                    "input_type": "telephone recording",
                }
            )
            self.fss_rows.append(
                {
                    "schema_version": "paper2-v2-fss-mapping-1",
                    "call_id": call_id,
                    "status": "verified",
                    "source_record_id": source_record_id,
                    "parent_recording_id": parent_recording_id,
                    "parent_split_part": raw_part_by_index[index],
                    "audio_extension": Path(audio_name).suffix.lower(),
                    "current_json_sha256": current_json_hash,
                    "original_json_sha256": original_json_hash,
                    "json_sha256_match": True,
                    "json_hash_collision_size": current_hash_counts[current_json_hash],
                    "source_json_read_method": read_method,
                    "nfkc_stem_join_required": (
                        os.path.splitext(audio_name)[0] != os.path.splitext(transcript_name)[0]
                    ),
                    "category": taxonomy.get("category"),
                    "subcategory": taxonomy.get("subcategory"),
                    "script_status": taxonomy.get("script_status"),
                    "original_filename_withheld": True,
                }
            )
            if index in sample_indices:
                raw_audio_path = fss_wsl_root / "All audios" / audio_name
                audio_samples.append(
                    self._validate_fss_audio_sample(
                        index=index,
                        raw_audio_path=raw_audio_path,
                        source_record_id=source_record_id,
                    )
                )

        _validate_expected(
            len(self.fss_rows), self.expected["fss_sorted_source_pairs"], "verified FSS mappings"
        )
        if any(sample["status"] != "verified" for sample in audio_samples):
            raise AuditError("FSS decoded-audio sample validation failed")
        self.coverage_gaps.extend(
            [
                {
                    "area": "FSS filesystem access",
                    "status": "partially mitigated",
                    "reason": (
                        "WSL directory enumeration returns EIO for parts of the mounted archive; "
                        "the frozen mapping uses cmd.exe UTF-16 enumeration and PowerShell hash fallback"
                    ),
                },
                {
                    "area": "global speaker identity",
                    "status": "unknown",
                    "reason": "CLova diarization speaker labels are call-local; no validated global speaker map exists",
                },
                {
                    "area": "robust cropped/transcoded acoustic search",
                    "status": "pending optional enrichment",
                    "reason": "minimum safe graph uses exact current audio, exact/near text, and recovered source parents",
                },
            ]
        )
        return {
            "mapping_count": len(self.fss_rows),
            "enumeration_method": self.config["fss_mapping"]["enumeration"],
            "sort_method": self.config["fss_mapping"]["sort"],
            "stem_join_method": self.config["fss_mapping"]["stem_join"],
            "audio_name_list_sha256": sha256_bytes("\n".join(audio_names).encode("utf-8")),
            "transcript_name_list_sha256": sha256_bytes(
                "\n".join(sorted(transcript_names)).encode("utf-8")
            ),
            "normalized_stem_join_adjustments": adjusted_stem_joins,
            "json_hash_validation": {
                "matches": len(self.fss_rows),
                "mismatches": 0,
                "wsl_direct_reads": direct_reads,
                "powershell_hash_fallbacks": powershell_reads,
            },
            "recovered_parent_groups": len(accepted_split_groups),
            "calls_in_recovered_parent_groups": sum(len(value) for value in accepted_split_groups.values()),
            "parent_group_size_distribution": dict(
                sorted(Counter(str(len(value)) for value in accepted_split_groups.values()).items())
            ),
            "taxonomy_recovered_calls": sum(bool(row.get("category")) for row in self.fss_rows),
            "decoded_audio_validation_samples": audio_samples,
            "raw_names_withheld": True,
        }

    def _add_relation(
        self,
        *,
        call_a: str,
        call_b: str,
        relation_type: str,
        method_id: str,
        score: float | None,
        decision: str,
        hard_group_edge: bool,
        reason: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        left, right = sorted((call_a, call_b), key=natural_key)
        if left == right:
            return
        edge_id = stable_id("edge", relation_type, method_id, left, right)
        self.relations.append(
            {
                "schema_version": RELATION_SCHEMA,
                "edge_id": edge_id,
                "call_id_a": left,
                "call_id_b": right,
                "relation_type": relation_type,
                "method_id": method_id,
                "score": round(float(score), 12) if score is not None else None,
                "decision": decision,
                "hard_group_edge": bool(hard_group_edge),
                "decision_reason": reason,
                "evidence": dict(evidence or {}),
            }
        )

    def _add_group_star_edges(
        self,
        groups: Mapping[str, Sequence[str]],
        *,
        relation_type: str,
        method_id: str,
        reason: str,
    ) -> None:
        for evidence_value, raw_members in sorted(groups.items()):
            members = sorted(set(raw_members), key=natural_key)
            if len(members) < 2:
                continue
            anchor = members[0]
            evidence_id = pseudonym(f"{relation_type}_evidence", evidence_value)
            for member in members[1:]:
                self._add_relation(
                    call_a=anchor,
                    call_b=member,
                    relation_type=relation_type,
                    method_id=method_id,
                    score=1.0,
                    decision="accepted",
                    hard_group_edge=True,
                    reason=reason,
                    evidence={"equivalence_class_id": evidence_id},
                )

    def discover_relations(self) -> dict[str, Any]:
        audio_byte_groups: defaultdict[str, list[str]] = defaultdict(list)
        audio_pcm_groups: defaultdict[str, list[str]] = defaultdict(list)
        strict_text_groups: defaultdict[str, list[str]] = defaultdict(list)
        canonical_text_groups: defaultdict[str, list[str]] = defaultdict(list)
        parent_groups: defaultdict[str, list[str]] = defaultdict(list)
        for call in self.calls:
            call_id = call["call_id"]
            audio_byte_groups[call["audio"]["byte_sha256"]].append(call_id)
            audio_pcm_groups[call["audio"]["pcm_sha256"]].append(call_id)
            strict_text_groups[call["transcript"]["strict_text_sha256"]].append(call_id)
            canonical_text_groups[call["transcript"]["canonical_text_sha256"]].append(call_id)
            parent = call["source_provenance"].get("parent_recording_id")
            if parent:
                parent_groups[str(parent)].append(call_id)

        self._add_group_star_edges(
            audio_byte_groups,
            relation_type="exact_audio_bytes",
            method_id="sha256-current-wav-bytes-v1",
            reason="byte-identical current WAV files are the same observation",
        )
        self._add_group_star_edges(
            audio_pcm_groups,
            relation_type="exact_audio_pcm",
            method_id="sha256-wave-parameters-plus-pcm-v1",
            reason="identical decoded PCM and format parameters are the same observation",
        )
        self._add_group_star_edges(
            parent_groups,
            relation_type="recovered_parent_recording",
            method_id="verified-source-parent-pseudonym-v1",
            reason="verified source filenames identify split children of one parent recording",
        )

        exact_minimum = int(self.config["exact_text"]["auto_accept_min_characters"])
        for relation_type, method_id, groups, length_field in (
            (
                "exact_transcript_strict",
                "sha256-full-call-asr-nfkc-whitespace-v1",
                strict_text_groups,
                "strict_character_count",
            ),
            (
                "exact_transcript_canonical",
                "sha256-full-call-asr-nfkc-alphanumeric-v1",
                canonical_text_groups,
                "canonical_character_count",
            ),
        ):
            for digest, raw_members in sorted(groups.items()):
                members = sorted(set(raw_members), key=natural_key)
                if len(members) < 2:
                    continue
                anchor = members[0]
                for member in members[1:]:
                    minimum_length = min(
                        int(self.by_id[anchor]["transcript"][length_field]),
                        int(self.by_id[member]["transcript"][length_field]),
                    )
                    accepted = minimum_length >= exact_minimum
                    self._add_relation(
                        call_a=anchor,
                        call_b=member,
                        relation_type=relation_type,
                        method_id=method_id,
                        score=1.0,
                        decision="accepted" if accepted else "rejected_short_generic_text",
                        hard_group_edge=accepted,
                        reason=(
                            "full-call exact normalized transcript exceeds frozen minimum length"
                            if accepted
                            else "exact normalized text is too short to establish call relatedness"
                        ),
                        evidence={
                            "text_digest": pseudonym("text_digest", digest),
                            "minimum_character_count": minimum_length,
                            "minimum_required": exact_minimum,
                        },
                    )

        text_spec = self.config["text_similarity"]
        from sklearn import __version__ as sklearn_version
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            analyzer=str(text_spec["analyzer"]),
            ngram_range=tuple(int(value) for value in text_spec["ngram_range"]),
            lowercase=bool(text_spec["lowercase"]),
            min_df=int(text_spec["min_df"]),
            max_df=float(text_spec["max_df"]),
            norm=str(text_spec["norm"]),
            use_idf=bool(text_spec["use_idf"]),
            smooth_idf=bool(text_spec["smooth_idf"]),
            sublinear_tf=bool(text_spec["sublinear_tf"]),
            dtype=np.float64,
        )
        texts = [str(call["_strict_text"]) for call in self.calls]
        matrix = vectorizer.fit_transform(texts)
        threshold = float(text_spec["candidate_threshold_inclusive"])
        minimum_characters = int(text_spec["auto_accept_min_characters"])
        minimum_ratio = float(text_spec["auto_accept_min_length_ratio"])
        block_rows = int(text_spec["pairwise_block_rows"])
        near_pair_count = 0
        for start in range(0, len(self.calls), block_rows):
            stop = min(len(self.calls), start + block_rows)
            similarities = (matrix[start:stop] @ matrix.T).toarray()
            for local_index, row_scores in enumerate(similarities):
                left_index = start + local_index
                for right_index in np.flatnonzero(row_scores >= threshold):
                    if right_index <= left_index:
                        continue
                    left = self.calls[left_index]
                    right = self.calls[int(right_index)]
                    left_length = len(str(left["_strict_text"]))
                    right_length = len(str(right["_strict_text"]))
                    length_ratio = min(left_length, right_length) / max(left_length, right_length)
                    same_label = left["label"] == right["label"]
                    accepted = (
                        same_label
                        and min(left_length, right_length) >= minimum_characters
                        and length_ratio >= minimum_ratio
                    )
                    self._add_relation(
                        call_a=left["call_id"],
                        call_b=right["call_id"],
                        relation_type="near_transcript_tfidf",
                        method_id=str(text_spec["method_id"]),
                        score=float(row_scores[right_index]),
                        decision="accepted_conservative" if accepted else "pending_manual_review",
                        hard_group_edge=accepted,
                        reason=(
                            "same-label full-call similarity passes frozen length and ratio rules"
                            if accepted
                            else "candidate does not satisfy every conservative automatic grouping rule"
                        ),
                        evidence={
                            "threshold_inclusive": threshold,
                            "length_ratio": round(length_ratio, 12),
                            "minimum_character_count": min(left_length, right_length),
                            "same_label": same_label,
                        },
                    )
                    near_pair_count += 1

        self.relations.sort(
            key=lambda row: (
                row["relation_type"],
                natural_key(row["call_id_a"]),
                natural_key(row["call_id_b"]),
                row["edge_id"],
            )
        )
        seen_edges: set[str] = set()
        for relation in self.relations:
            if relation["edge_id"] in seen_edges:
                raise AuditError(f"duplicate relation edge ID: {relation['edge_id']}")
            seen_edges.add(relation["edge_id"])

        mandatory_pairs = [
            {"vishing_554", "vishing_555"},
            {"vishing_135", "vishing_200"},
            {"vishing_58", "vishing_61"},
            {"vishing_60", "vishing_63"},
            {"vishing_126", "vishing_188"},
        ]
        accepted_pairs = [
            {row["call_id_a"], row["call_id_b"]}
            for row in self.relations
            if row["hard_group_edge"]
        ]
        missing_mandatory = [sorted(pair, key=natural_key) for pair in mandatory_pairs if pair not in accepted_pairs]
        if missing_mandatory:
            raise AuditError(f"mandatory known relations were not recovered: {missing_mandatory}")

        return {
            "relation_count": len(self.relations),
            "accepted_hard_edge_count": sum(row["hard_group_edge"] for row in self.relations),
            "pending_review_count": sum(row["decision"] == "pending_manual_review" for row in self.relations),
            "counts_by_type": dict(sorted(Counter(row["relation_type"] for row in self.relations).items())),
            "near_text": {
                "method_id": text_spec["method_id"],
                "candidate_count": near_pair_count,
                "vocabulary_size": len(vectorizer.vocabulary_),
                "scikit_learn_version": sklearn_version,
                "frozen_parameters": text_spec,
            },
            "mandatory_relations_recovered": True,
        }

    def build_components(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        all_ids = [call["call_id"] for call in self.calls]
        exact_audio = UnionFind(all_ids)
        all_relations = UnionFind(all_ids)
        for relation in self.relations:
            if not relation["hard_group_edge"]:
                continue
            left, right = relation["call_id_a"], relation["call_id_b"]
            all_relations.union(left, right)
            if relation["relation_type"] in {"exact_audio_bytes", "exact_audio_pcm"}:
                exact_audio.union(left, right)

        duplicate_clusters: list[dict[str, Any]] = []
        redundant_calls: set[str] = set()
        canonical_by_call = {call_id: call_id for call_id in all_ids}
        for members in exact_audio.components():
            if len(members) < 2:
                continue
            canonical = members[0]
            redundant = members[1:]
            redundant_calls.update(redundant)
            for member in members:
                canonical_by_call[member] = canonical
            duplicate_clusters.append(
                {
                    "schema_version": "paper2-v2-exact-duplicate-cluster-1",
                    "cluster_id": stable_id("exact_duplicate", *members),
                    "members": members,
                    "canonical_call_id": canonical,
                    "excluded_redundant_call_ids": redundant,
                    "canonical_rule": "natural-sort-lowest call_id",
                    "primary_metric_policy": "retain canonical only",
                }
            )

        components: list[dict[str, Any]] = []
        component_by_call: dict[str, str] = {}
        burned_related_dev: set[str] = set()
        burned_calls: set[str] = {
            call["call_id"] for call in self.calls if call["is_burned"]
        }
        for members in all_relations.components():
            component_id = stable_id("relation_component", *members)
            member_burned = sorted(set(members) & burned_calls, key=natural_key)
            touches_burned = bool(member_burned)
            if touches_burned:
                burned_related_dev.update(set(members) - burned_calls)
            accepted_types = sorted(
                {
                    relation["relation_type"]
                    for relation in self.relations
                    if relation["hard_group_edge"]
                    and relation["call_id_a"] in members
                    and relation["call_id_b"] in members
                }
            )
            for member in members:
                component_by_call[member] = component_id
            components.append(
                {
                    "schema_version": COMPONENT_SCHEMA,
                    "component_id": component_id,
                    "members": members,
                    "member_count": len(members),
                    "labels": sorted({self.by_id[member]["label"] for member in members}),
                    "sources": sorted({self.by_id[member]["source"] for member in members}),
                    "historical_splits": sorted(
                        {self.by_id[member]["historical_split"] for member in members}
                    ),
                    "accepted_relation_types": accepted_types,
                    "burned_call_ids": member_burned,
                    "touches_burned": touches_burned,
                    "development_members_to_quarantine": sorted(
                        [member for member in members if member not in burned_calls],
                        key=natural_key,
                    )
                    if touches_burned
                    else [],
                    "redundant_exact_audio_members": sorted(
                        set(members) & redundant_calls, key=natural_key
                    ),
                }
            )

        eligible: list[str] = []
        status_counts: Counter[str] = Counter()
        for call in self.calls:
            call_id = call["call_id"]
            if call_id in burned_calls:
                status = "historical_burned"
            elif call_id in burned_related_dev:
                status = "quarantine_burned_related_component"
            elif call_id in redundant_calls:
                status = "exclude_redundant_exact_audio"
            else:
                status = "eligible_for_work6"
                eligible.append(call_id)
            call["relation_component_id"] = component_by_call[call_id]
            call["canonical_call_id"] = canonical_by_call[call_id]
            call["provisional_primary_status"] = status
            status_counts[status] += 1

        eligible_segments = sum(
            int(self.by_id[call_id]["transcript"]["segment_count"]) for call_id in eligible
        )
        components.sort(key=lambda row: natural_key(row["members"][0]))
        duplicate_clusters.sort(key=lambda row: natural_key(row["members"][0]))
        summary = {
            "component_count": len(components),
            "non_singleton_component_count": sum(row["member_count"] > 1 for row in components),
            "largest_component_size": max(row["member_count"] for row in components),
            "exact_duplicate_cluster_count": len(duplicate_clusters),
            "redundant_exact_audio_call_count": len(redundant_calls),
            "burned_call_count": len(burned_calls),
            "burned_related_development_call_count": len(burned_related_dev),
            "provisional_work6_eligible_calls": len(eligible),
            "provisional_work6_eligible_segments": eligible_segments,
            "status_counts": dict(sorted(status_counts.items())),
            "pending_relation_candidates_not_grouped": sum(
                relation["decision"] == "pending_manual_review" for relation in self.relations
            ),
        }
        return components, duplicate_clusters, summary

    @staticmethod
    def _binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

        return {
            "accuracy": round(float(accuracy_score(labels, predictions)), 12),
            "positive_f1": round(float(f1_score(labels, predictions, zero_division=0)), 12),
            "macro_f1": round(
                float(f1_score(labels, predictions, average="macro", zero_division=0)), 12
            ),
            "mcc": round(float(matthews_corrcoef(labels, predictions)), 12),
        }

    def shortcut_probes(self) -> dict[str, Any]:
        from sklearn.metrics import roc_auc_score

        scopes = {
            "full_corpus_discovery_only": self.calls,
            "historical_development_discovery_only": [
                call for call in self.calls if call["historical_split"] in {"train", "val"}
            ],
            "provisional_work6_pool_discovery_only": [
                call for call in self.calls if call["provisional_primary_status"] == "eligible_for_work6"
            ],
        }
        numeric_fields = {
            "call_duration_seconds": lambda call: call["audio"]["duration_seconds"],
            "segment_count": lambda call: call["transcript"]["segment_count"],
            "word_count": lambda call: call["transcript"]["word_count"],
            "mean_segment_duration_seconds": lambda call: call["transcript"]["mean_segment_duration_seconds"],
            "mean_segment_confidence": lambda call: call["transcript"]["mean_segment_confidence"],
            "local_speaker_count": lambda call: call["transcript"]["local_speaker_count"],
            "words_per_second": lambda call: (
                call["transcript"]["word_count"] / call["audio"]["duration_seconds"]
                if call["audio"]["duration_seconds"] > 0
                else None
            ),
        }
        scope_results: dict[str, Any] = {}
        for scope_name, calls in scopes.items():
            labels = np.asarray([call["label"] for call in calls], dtype=np.int64)
            baselines = {
                "always_positive": self._binary_metrics(labels, np.ones_like(labels)),
                "majority_class": self._binary_metrics(
                    labels, np.full_like(labels, int(labels.mean() >= 0.5))
                ),
            }
            univariate: dict[str, Any] = {}
            for field, extractor in numeric_fields.items():
                pairs = [
                    (float(value), int(call["label"]))
                    for call in calls
                    if (value := extractor(call)) is not None and math.isfinite(float(value))
                ]
                values = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
                field_labels = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
                auc = float(roc_auc_score(field_labels, values))
                by_label = {
                    str(label): values[field_labels == label] for label in (0, 1)
                }
                univariate[field] = {
                    "n": len(values),
                    "roc_auc_label1_direction": round(auc, 12),
                    "direction_agnostic_auc": round(max(auc, 1.0 - auc), 12),
                    "label_summary": {
                        label: {
                            "mean": round(float(data.mean()), 12),
                            "standard_deviation": round(float(data.std(ddof=0)), 12),
                        }
                        for label, data in by_label.items()
                    },
                    "interpretation": "descriptive separability only; no held-out predictive claim",
                }

            proxy_fields = {
                "collection_source": lambda call: call["source"],
                "call_id_family": lambda call: "vishing" if call["call_id"].startswith("vishing_") else "aihub",
                "asr_domain_code": lambda call: call["transcript"]["asr_domain_code"],
                "asr_domain_id": lambda call: str(call["transcript"]["asr_domain_id"]),
            }
            proxies: dict[str, Any] = {}
            for field, extractor in proxy_fields.items():
                contingency: defaultdict[str, Counter[int]] = defaultdict(Counter)
                for call in calls:
                    contingency[str(extractor(call))][int(call["label"])] += 1
                best_correct = sum(max(counter.values()) for counter in contingency.values())
                proxies[field] = {
                    "contingency": {
                        value: {str(label): count for label, count in sorted(counter.items())}
                        for value, counter in sorted(contingency.items())
                    },
                    "deterministic_lookup_accuracy": round(best_correct / len(calls), 12),
                    "must_never_enter_learned_inputs": True,
                }
            scope_results[scope_name] = {
                "call_count": len(calls),
                "label_counts": dict(
                    sorted(Counter(str(call["label"]) for call in calls).items())
                ),
                "baselines": baselines,
                "univariate_metadata": univariate,
                "direct_label_proxies": proxies,
            }

        return {
            "schema_version": "paper2-v2-shortcut-probe-audit-1",
            "evidence_status": "descriptive discovery audit, not corrected model performance",
            "scopes": scope_results,
            "deferred_until_work6_folds_are_frozen": [
                "character n-gram text classifier",
                "multivariate ASR metadata logistic regression",
                "global acoustic-feature logistic regression",
                "single acoustic-feature threshold probes",
                "channel-statistic classifier",
                "leave-one-benign-domain-out false-positive analysis",
                "all train-fitted normalization and threshold selection",
            ],
            "defer_reason": (
                "Learned shortcut probes require the same burned-excluded, relation-grouped folds as "
                "the corrected comparison. Running them on old call-ID-only folds would create invalid v2 evidence."
            ),
        }

    def run(self, output_dir: Path) -> dict[str, Any]:
        target = output_dir.resolve()
        if target.exists():
            raise AuditError(f"immutable Work 5 output already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging.", dir=target.parent))

        self.load_inputs()
        aihub_summary = self.recover_aihub()
        fss_summary = self.recover_fss()
        relation_summary = self.discover_relations()
        components, duplicate_clusters, component_summary = self.build_components()
        shortcut_results = self.shortcut_probes()

        public_calls: list[dict[str, Any]] = []
        for call in self.calls:
            public = {key: value for key, value in call.items() if not key.startswith("_")}
            public_calls.append(public)

        relation_decisions = [
            {
                "edge_id": relation["edge_id"],
                "call_id_a": relation["call_id_a"],
                "call_id_b": relation["call_id_b"],
                "relation_type": relation["relation_type"],
                "decision": relation["decision"],
                "hard_group_edge": relation["hard_group_edge"],
                "decision_reason": relation["decision_reason"],
            }
            for relation in self.relations
        ]
        source_inventory = {
            "schema_version": "paper2-v2-source-inventory-1",
            "input_manifest_sha256": self.external_inventory["input_manifest_sha256"],
            "current_corpus": {
                "calls": len(self.calls),
                "labels": dict(sorted(Counter(str(call["label"]) for call in self.calls).items())),
                "historical_splits": dict(
                    sorted(Counter(call["historical_split"] for call in self.calls).items())
                ),
                "current_wav_format_counts": {},
            },
            "aihub": aihub_summary,
            "fss": fss_summary,
            "privacy": self.config["privacy"],
        }
        format_counts = Counter(
            f"{call['audio']['sample_rate_hz']}Hz/{call['audio']['channels']}ch/"
            f"{8 * call['audio']['sample_width_bytes']}bit"
            for call in self.calls
        )
        source_inventory["current_corpus"]["current_wav_format_counts"] = dict(
            sorted(format_counts.items())
        )

        audit_summary = {
            "schema_version": AUDIT_SCHEMA,
            "artifact_id": self.config["artifact_id"],
            "audit_date": self.config["audit_date"],
            "status": "pass",
            "calls": len(self.calls),
            "aihub": {
                "verified_mappings": len(self.aihub_rows),
                "mapping_coverage": 1.0,
            },
            "fss": {
                "verified_mappings": len(self.fss_rows),
                "mapping_coverage": 1.0,
                "recovered_parent_groups": fss_summary["recovered_parent_groups"],
                "calls_in_parent_groups": fss_summary["calls_in_recovered_parent_groups"],
                "taxonomy_recovered_calls": fss_summary["taxonomy_recovered_calls"],
            },
            "relations": relation_summary,
            "components": component_summary,
            "shortcut_probe_status": shortcut_results["evidence_status"],
            "coverage_gap_count": len(self.coverage_gaps),
            "work6_ready": relation_summary["pending_review_count"] == 0,
        }
        protocol = {
            "schema_version": "paper2-v2-work5-protocol-1",
            "artifact_id": self.config["artifact_id"],
            "audit_date": self.config["audit_date"],
            "config_sha256": sha256_file(self.config_path),
            "config": self.config,
            "input_manifest_sha256": self.external_inventory["input_manifest_sha256"],
            "evidence_policy": {
                "exact_audio": "accepted and deduplicated for primary metrics",
                "verified_source_parent": "accepted hard grouping edge",
                "exact_full_call_text": "accepted only at or above frozen character minimum",
                "near_full_call_text": "accepted only under frozen same-label, length, and ratio rules",
                "pending_candidates": "never enter connected components",
                "burned_policy": "all development members of a component touching historical test are quarantined",
            },
            "privacy_policy": self.config["privacy"],
        }

        write_json(staging / "protocol.json", protocol)
        write_json(staging / "source_inventory.json", source_inventory)
        write_jsonl(staging / "call_provenance.jsonl", public_calls)
        write_jsonl(staging / "aihub_mapping_validation.jsonl", self.aihub_rows)
        write_jsonl(staging / "fss_mapping_validation.jsonl", self.fss_rows)
        write_jsonl(staging / "relation_candidates.jsonl", self.relations)
        write_jsonl(staging / "relation_decisions.jsonl", relation_decisions)
        write_jsonl(staging / "exact_duplicate_clusters.jsonl", duplicate_clusters)
        write_jsonl(staging / "group_components.jsonl", components)
        write_json(staging / "shortcut_probe_results.json", shortcut_results)
        write_json(staging / "coverage_gaps.json", {"coverage_gaps": self.coverage_gaps})
        write_json(staging / "audit_summary.json", audit_summary)

        files = sorted(path for path in staging.iterdir() if path.is_file())
        hash_lines = [f"{sha256_file(path)}  {path.name}" for path in files]
        (staging / "hashes.sha256").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
        os.replace(staging, target)
        return audit_summary


def run_work5_audit(*, project_root: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    return Work5Audit(project_root=project_root, config_path=config_path).run(output_dir)
