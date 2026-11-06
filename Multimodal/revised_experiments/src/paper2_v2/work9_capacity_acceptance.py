"""Build the source/environment acceptance for capacity sensitivity."""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
import os
import platform
import re
import sys
import sysconfig
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import sha256_file
from .work9_capacity_registry import atomic_write_once, _reject_symlink_components
from .work9_capacity_types import (
    ACCEPTANCE_DRAFT_RELATIVE,
    ACCEPTANCE_RECEIPT_RELATIVE,
    ACCEPTANCE_SCHEMA_VERSION,
    CUDA_SMOKE_RECEIPT_SCHEMA_VERSION,
    DRY_RECEIPT_RELATIVE,
    EXECUTION_ADDENDUM_RELATIVE,
    EXECUTION_ADDENDUM_SHA256,
    EXPECTED_COMPUTE_CAPABILITY,
    EXPECTED_CUDA_VERSION,
    EXPECTED_CUDNN_VERSION,
    EXPECTED_DEVICE_NAME,
    EXPECTED_TORCH_VERSION,
    EXPERIMENT_CARD_RELATIVE,
    EXPERIMENT_CARD_SHA256,
    FIGURE_SPEC_SHA256,
    GATE1_HASH_MANIFEST_SHA256,
    PRIMARY_OUTER_INDEX_SHA256,
    PRIMARY_ENVIRONMENT_LOCK_SHA256,
    PRIMARY_RUNNER_ACCEPTANCE_RELATIVE,
    PRIMARY_RUNNER_ACCEPTANCE_SHA256,
    PRIMARY_SELECTION_INDEX_SHA256,
    SMOKE_EVIDENCE_ROOT_RELATIVE,
    CAPACITY_ANALYSIS_PROTOCOL_RELATIVE,
    CAPACITY_ANALYSIS_PROTOCOL_SHA256,
    ANCHOR_POLICY_SHA256,
    UNIT_TEST_RECEIPT_SCHEMA_VERSION,
    UNIT_TEST_EVIDENCE_ROOT_RELATIVE,
    WORK8_CONFIG_SHA256,
    WORK8_RECEIPT_SHA256,
    canonical_json_bytes,
)


RUNNER_SOURCE_PATHS = (
    "revised_experiments/src/paper2_v2/conditions.py",
    "revised_experiments/src/paper2_v2/config.py",
    "revised_experiments/src/paper2_v2/data.py",
    "revised_experiments/src/paper2_v2/evaluation.py",
    "revised_experiments/src/paper2_v2/models.py",
    "revised_experiments/src/paper2_v2/pairing.py",
    "revised_experiments/src/paper2_v2/text_features.py",
    "revised_experiments/src/paper2_v2/work9_capacity_types.py",
    "revised_experiments/src/paper2_v2/work9_capacity_registry.py",
    "revised_experiments/src/paper2_v2/work9_data.py",
    "revised_experiments/src/paper2_v2/work9_capacity_training.py",
    "revised_experiments/src/paper2_v2/work9_capacity_acceptance.py",
    "revised_experiments/src/paper2_v2/work9_capacity_analysis.py",
    "revised_experiments/src/paper2_v2/work9_capacity_figures.py",
    "revised_experiments/scripts/run_work9_capacity.py",
    "revised_experiments/scripts/build_work9_capacity_runner_acceptance.py",
    "revised_experiments/scripts/run_work9_capacity_tests.py",
    "revised_experiments/scripts/build_work9_capacity_dry_receipt.py",
    "revised_experiments/scripts/run_work9_capacity_analysis.py",
    "revised_experiments/scripts/generate_work9_capacity_figures.py",
    "revised_experiments/tests/test_work9_capacity_registry.py",
    "revised_experiments/tests/test_work9_capacity_training.py",
    "revised_experiments/tests/test_work9_capacity_lineage.py",
    "revised_experiments/tests/test_work9_capacity_analysis_figures.py",
)


class Work9AcceptanceError(RuntimeError):
    pass


def _ref(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "row_count": None,
    }


def source_contracts(project_root: Path) -> list[dict[str, Any]]:
    paths = [project_root / value for value in RUNNER_SOURCE_PATHS]
    paths.extend(
        sorted(
            (project_root / "revised_experiments/registry/schemas/work9_capacity").glob(
                "*.schema.json"
            )
        )
    )
    if len(paths) != len({path.resolve() for path in paths}):
        raise Work9AcceptanceError("runner source contract contains duplicate paths")
    return [_ref(path, project_root) for path in paths]


def bundle_digests(project_root: Path) -> dict[str, str]:
    contracts = source_contracts(project_root)
    buckets: dict[str, list[dict[str, Any]]] = {
        "runner_source": [],
        "schemas": [],
        "tests": [],
    }
    for reference in contracts:
        path = str(reference["path"])
        if "/registry/schemas/" in path:
            buckets["schemas"].append(reference)
        elif "/tests/" in path:
            buckets["tests"].append(reference)
        else:
            buckets["runner_source"].append(reference)
    return {
        f"{name}_bundle_sha256": hashlib.sha256(
            canonical_json_bytes(sorted(rows, key=lambda row: row["path"]))
        ).hexdigest()
        for name, rows in buckets.items()
    }


def _environment_file_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise Work9AcceptanceError(f"environment file is unavailable or symlinked: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def verify_environment_lock_mapping(raw: Any) -> str:
    """Validate a serialized fixed-root environment lock and return its digest."""

    if not isinstance(raw, Mapping):
        raise Work9AcceptanceError("environment lock must be an object")
    expected_keys = {
        "python",
        "executable",
        "environment_prefix",
        "platform",
        "inventory_method",
        "distribution_roots",
        "site_configuration_files",
        "interpreter_configuration",
        "critical_module_files",
        "packages",
        "packages_sha256",
        "environment_lock_sha256",
    }
    if set(raw) != expected_keys:
        raise Work9AcceptanceError("environment lock keys changed")
    if raw["inventory_method"] != (
        "importlib_metadata_explicit_sysconfig_purelib_platlib_record_hashes_v1"
    ):
        raise Work9AcceptanceError("environment inventory method changed")
    for key in ("python", "executable", "environment_prefix", "platform"):
        if not isinstance(raw[key], str) or not raw[key]:
            raise Work9AcceptanceError(f"environment {key} is invalid")
    roots = raw["distribution_roots"]
    if not isinstance(roots, list) or not roots:
        raise Work9AcceptanceError("environment distribution roots are missing")
    for root in roots:
        if not isinstance(root, Mapping) or set(root) != {
            "configured_paths",
            "resolved_path",
            "roles",
        }:
            raise Work9AcceptanceError("environment distribution root is malformed")
        if not isinstance(root["roles"], list) or root["roles"] != sorted(
            set(root["roles"])
        ):
            raise Work9AcceptanceError("environment distribution root roles are invalid")
        if not isinstance(root["configured_paths"], list) or root[
            "configured_paths"
        ] != sorted(set(root["configured_paths"])):
            raise Work9AcceptanceError("environment configured paths are invalid")
        if not isinstance(root["resolved_path"], str) or not Path(
            root["resolved_path"]
        ).is_absolute():
            raise Work9AcceptanceError("environment resolved root is invalid")
    packages = raw["packages"]
    if not isinstance(packages, list) or not packages:
        raise Work9AcceptanceError("environment package inventory is empty")
    package_keys = {
        "name",
        "metadata_name",
        "version",
        "dist_info_path",
        "metadata_sha256",
        "record_sha256",
        "direct_url_sha256",
    }
    names: list[str] = []
    for package in packages:
        if not isinstance(package, Mapping) or set(package) != package_keys:
            raise Work9AcceptanceError("environment package record is malformed")
        name = package["name"]
        if not isinstance(name, str) or not re.fullmatch(
            r"[a-z0-9]+(?:-[a-z0-9]+)*", name
        ):
            raise Work9AcceptanceError("environment normalized package name is invalid")
        names.append(name)
        if (
            not isinstance(package["metadata_name"], str)
            or not package["metadata_name"]
            or not isinstance(package["version"], str)
            or not package["version"]
            or not isinstance(package["dist_info_path"], str)
            or not Path(package["dist_info_path"]).is_absolute()
        ):
            raise Work9AcceptanceError("environment package identity is invalid")
        for key in ("metadata_sha256", "record_sha256"):
            if not isinstance(package[key], str) or not re.fullmatch(
                r"[0-9a-f]{64}", package[key]
            ):
                raise Work9AcceptanceError(f"environment package {key} is invalid")
        direct = package["direct_url_sha256"]
        if direct is not None and (
            not isinstance(direct, str) or not re.fullmatch(r"[0-9a-f]{64}", direct)
        ):
            raise Work9AcceptanceError("environment direct_url hash is invalid")
    if names != sorted(names) or len(names) != len(set(names)):
        raise Work9AcceptanceError("environment package names are not unique and sorted")
    expected_packages_sha = hashlib.sha256(
        canonical_json_bytes({"packages": packages})
    ).hexdigest()
    if raw["packages_sha256"] != expected_packages_sha:
        raise Work9AcceptanceError("environment package inventory hash mismatch")
    for collection_name in (
        "site_configuration_files",
        "interpreter_configuration",
        "critical_module_files",
    ):
        collection = raw[collection_name]
        if not isinstance(collection, list):
            raise Work9AcceptanceError(f"environment {collection_name} must be a list")
        paths: list[str] = []
        for item in collection:
            if not isinstance(item, Mapping) or set(item) != {
                "path",
                "sha256",
                "bytes",
            }:
                raise Work9AcceptanceError(
                    f"environment {collection_name} record is malformed"
                )
            paths.append(str(item["path"]))
            if not re.fullmatch(r"[0-9a-f]{64}", str(item["sha256"])):
                raise Work9AcceptanceError(
                    f"environment {collection_name} hash is invalid"
                )
            if not isinstance(item["bytes"], int) or isinstance(item["bytes"], bool):
                raise Work9AcceptanceError(
                    f"environment {collection_name} byte count is invalid"
                )
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise Work9AcceptanceError(
                f"environment {collection_name} paths are not unique and sorted"
            )
    base = dict(raw)
    declared_lock_sha = base.pop("environment_lock_sha256")
    actual_lock_sha = hashlib.sha256(canonical_json_bytes(base)).hexdigest()
    if declared_lock_sha != actual_lock_sha:
        raise Work9AcceptanceError("environment lock self-hash mismatch")
    return actual_lock_sha


def _distribution_package_records(
    distributions: Sequence[Any], resolved_roots: Sequence[str]
) -> list[dict[str, Any]]:
    """Build canonical package records from an explicitly rooted discovery."""

    package_records: dict[str, dict[str, Any]] = {}
    for distribution in distributions:
        declared = distribution.metadata.get("Name")
        if not isinstance(declared, str) or not declared.strip():
            raise Work9AcceptanceError("distribution is missing a valid Name")
        name = re.sub(r"[-_.]+", "-", declared).lower()
        version = distribution.version
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
            raise Work9AcceptanceError(f"distribution name is invalid: {declared!r}")
        if not isinstance(version, str) or not version.strip():
            raise Work9AcceptanceError(f"distribution version is invalid: {declared}")
        if name in package_records:
            raise Work9AcceptanceError(
                f"duplicate distribution name in fixed environment roots: {name}"
            )
        dist_info = Path(str(getattr(distribution, "_path", "")))
        if not dist_info.is_dir() or dist_info.is_symlink():
            raise Work9AcceptanceError(f"distribution metadata path is invalid: {name}")
        resolved_dist_info = dist_info.resolve()
        if not any(
            resolved_dist_info == Path(root)
            or Path(root) in resolved_dist_info.parents
            for root in resolved_roots
        ):
            raise Work9AcceptanceError(f"distribution metadata escaped fixed roots: {name}")
        metadata_path = dist_info / "METADATA"
        record_path = dist_info / "RECORD"
        if not metadata_path.is_file() or not record_path.is_file():
            raise Work9AcceptanceError(f"distribution lacks METADATA/RECORD: {name}")
        direct_url_path = dist_info / "direct_url.json"
        package_records[name] = {
            "name": name,
            "metadata_name": declared,
            "version": version,
            "dist_info_path": str(resolved_dist_info),
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
            "direct_url_sha256": (
                hashlib.sha256(direct_url_path.read_bytes()).hexdigest()
                if direct_url_path.is_file()
                else None
            ),
        }
    if not package_records:
        raise Work9AcceptanceError("fixed environment package inventory is empty")
    return [package_records[name] for name in sorted(package_records)]


def environment_lock() -> dict[str, Any]:
    """Inventory only distributions installed in this interpreter's site roots.

    ``importlib.metadata.distributions()`` without an explicit path follows
    mutable ``sys.path``.  Runtime imports can append setuptools' vendored
    packages or temporary build directories, which are not environment
    changes.  The sysconfig purelib/platlib roots are fixed for the running
    interpreter and exclude that contamination.
    """

    configured = sysconfig.get_paths()
    roots_by_resolved: dict[str, dict[str, set[str]]] = {}
    for role in ("purelib", "platlib"):
        configured_path = configured.get(role)
        if not configured_path:
            continue
        resolved = str(Path(configured_path).resolve())
        record = roots_by_resolved.setdefault(
            resolved, {"roles": set(), "configured_paths": set()}
        )
        record["roles"].add(role)
        record["configured_paths"].add(str(configured_path))
    distribution_roots = [
        {
            "resolved_path": resolved,
            "roles": sorted(values["roles"]),
            "configured_paths": sorted(values["configured_paths"]),
        }
        for resolved, values in sorted(roots_by_resolved.items())
    ]
    resolved_roots = [item["resolved_path"] for item in distribution_roots]
    if not resolved_roots or any(not Path(root).is_dir() for root in resolved_roots):
        raise Work9AcceptanceError("interpreter distribution roots are unavailable")
    environment_prefix = str(Path(sys.prefix).resolve())
    if any(
        Path(root) != Path(environment_prefix)
        and Path(environment_prefix) not in Path(root).parents
        for root in resolved_roots
    ):
        raise Work9AcceptanceError("distribution root is outside interpreter prefix")
    packages = _distribution_package_records(
        list(importlib.metadata.distributions(path=resolved_roots)), resolved_roots
    )
    package_payload = {"packages": packages}
    site_configuration_files = sorted(
        (
            _environment_file_record(path)
            for root in resolved_roots
            for path in Path(root).glob("*.pth")
        ),
        key=lambda item: item["path"],
    )
    pyvenv = Path(environment_prefix) / "pyvenv.cfg"
    interpreter_configuration = (
        [_environment_file_record(pyvenv)] if pyvenv.is_file() else []
    )
    critical_relatives = (
        "jsonschema/__init__.py",
        "numpy/__init__.py",
        "safetensors/__init__.py",
        "safetensors/torch.py",
        "sklearn/__init__.py",
        "torch/__init__.py",
        "torch/version.py",
    )
    critical_module_files: list[dict[str, Any]] = []
    for relative in critical_relatives:
        matches = [Path(root) / relative for root in resolved_roots if (Path(root) / relative).is_file()]
        if len(matches) != 1:
            raise Work9AcceptanceError(
                f"critical environment module does not resolve uniquely: {relative}"
            )
        critical_module_files.append(_environment_file_record(matches[0]))
    critical_module_files.sort(key=lambda item: item["path"])
    base = {
        "python": sys.version,
        "executable": sys.executable,
        "environment_prefix": environment_prefix,
        "platform": platform.platform(),
        "inventory_method": (
            "importlib_metadata_explicit_sysconfig_purelib_platlib_record_hashes_v1"
        ),
        "distribution_roots": distribution_roots,
        "site_configuration_files": site_configuration_files,
        "interpreter_configuration": interpreter_configuration,
        "critical_module_files": critical_module_files,
        "packages": packages,
        "packages_sha256": hashlib.sha256(
            canonical_json_bytes(package_payload)
        ).hexdigest(),
    }
    result = {
        **base,
        "environment_lock_sha256": hashlib.sha256(
            canonical_json_bytes(base)
        ).hexdigest(),
    }
    verify_environment_lock_mapping(result)
    return result


def verify_dry_receipt_payload_against_current(
    project_root: Path, dry: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Bind no-training evidence to the exact current source/schema/test code."""

    current_sources = source_contracts(project_root)
    current_bundles = bundle_digests(project_root)
    expected_dry_lineage = {
        "work8_config_sha256": WORK8_CONFIG_SHA256,
        "work8_receipt_sha256": WORK8_RECEIPT_SHA256,
        "experiment_card_sha256": EXPERIMENT_CARD_SHA256,
        "anchor_policy_sha256": ANCHOR_POLICY_SHA256,
        "capacity_analysis_protocol_sha256": CAPACITY_ANALYSIS_PROTOCOL_SHA256,
        "figure_spec_sha256": FIGURE_SPEC_SHA256,
        "execution_addendum_sha256": EXECUTION_ADDENDUM_SHA256,
        "primary_runner_acceptance_sha256": PRIMARY_RUNNER_ACCEPTANCE_SHA256,
        "primary_environment_lock_sha256": PRIMARY_ENVIRONMENT_LOCK_SHA256,
        "primary_selection_index_sha256": PRIMARY_SELECTION_INDEX_SHA256,
        "expected_primary_outer_index_sha256_after_selection_freeze": (
            PRIMARY_OUTER_INDEX_SHA256
        ),
        "gate1_hash_manifest_sha256": GATE1_HASH_MANIFEST_SHA256,
    }
    if dry.get("status") != "pass" or dry.get("lineage") != expected_dry_lineage:
        raise Work9AcceptanceError("dry receipt core lineage differs from frozen protocol")
    if dry.get("source_contracts") != current_sources:
        raise Work9AcceptanceError("dry receipt source inventory differs from current code")
    if dry.get("bundle_digests") != current_bundles:
        raise Work9AcceptanceError("dry receipt bundle digests differ from current code")
    expected_safety = {
        "training_launched": False,
        "optimizer_steps": 0,
        "checkpoints_written": 0,
        "new_capacity_outer_outcomes_accessed": False,
        "primary_outer_index_opened_or_hashed": False,
        "primary_outer_artifacts_opened_or_hashed": False,
        "primary_selection_registry_read_only_verified": True,
        "production_capacity_registry_accessed": False,
        "primary_registry_modified": False,
        "gate1_reopened": False,
        "gate2_reopened": False,
    }
    if dry.get("safety") != expected_safety:
        raise Work9AcceptanceError("dry receipt safety declaration differs")
    return current_sources, current_bundles


def _base_payload(project_root: Path) -> dict[str, Any]:
    card_path = project_root / EXPERIMENT_CARD_RELATIVE
    if card_path.stat().st_mode & 0o222:
        raise Work9AcceptanceError("experiment card must be read-only before smoke")
    if sha256_file(card_path) != EXPERIMENT_CARD_SHA256:
        raise Work9AcceptanceError("capacity experiment card hash changed")
    for relative, digest, label in (
        (
            CAPACITY_ANALYSIS_PROTOCOL_RELATIVE,
            CAPACITY_ANALYSIS_PROTOCOL_SHA256,
            "capacity analysis protocol",
        ),
        (EXECUTION_ADDENDUM_RELATIVE, EXECUTION_ADDENDUM_SHA256, "execution addendum"),
        (
            PRIMARY_RUNNER_ACCEPTANCE_RELATIVE,
            PRIMARY_RUNNER_ACCEPTANCE_SHA256,
            "primary runner acceptance",
        ),
    ):
        path = project_root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_mode & 0o222
            or sha256_file(path) != digest
        ):
            raise Work9AcceptanceError(f"immutable {label} differs")
    dry_receipt_path = project_root / DRY_RECEIPT_RELATIVE
    if not dry_receipt_path.is_file() or dry_receipt_path.stat().st_mode & 0o222:
        raise Work9AcceptanceError("immutable capacity dry-run receipt is missing")
    dry = json.loads(dry_receipt_path.read_text(encoding="utf-8"))
    current_sources, current_bundles = verify_dry_receipt_payload_against_current(
        project_root, dry
    )
    current_environment = environment_lock()
    if current_environment.get("environment_lock_sha256") != PRIMARY_ENVIRONMENT_LOCK_SHA256:
        raise Work9AcceptanceError("capacity environment differs from primary acceptance")
    primary_acceptance = json.loads(
        (project_root / PRIMARY_RUNNER_ACCEPTANCE_RELATIVE).read_text(encoding="utf-8")
    )
    if (
        verify_environment_lock_mapping(primary_acceptance.get("environment_lock"))
        != PRIMARY_ENVIRONMENT_LOCK_SHA256
        or primary_acceptance.get("environment_lock") != current_environment
    ):
        raise Work9AcceptanceError("current and primary accepted environments differ")
    return {
        "schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "acceptance_generation": "work9_capacity_runner_acceptance_v1",
        "work8_config_sha256": WORK8_CONFIG_SHA256,
        "work8_receipt_sha256": WORK8_RECEIPT_SHA256,
        "experiment_card_sha256": EXPERIMENT_CARD_SHA256,
        "anchor_policy_sha256": ANCHOR_POLICY_SHA256,
        "capacity_analysis_protocol_sha256": CAPACITY_ANALYSIS_PROTOCOL_SHA256,
        "capacity_analysis_protocol": _ref(
            project_root / CAPACITY_ANALYSIS_PROTOCOL_RELATIVE, project_root
        ),
        "figure_spec_sha256": FIGURE_SPEC_SHA256,
        "execution_addendum_sha256": EXECUTION_ADDENDUM_SHA256,
        "execution_addendum": _ref(
            project_root / EXECUTION_ADDENDUM_RELATIVE, project_root
        ),
        "primary_runner_acceptance_sha256": PRIMARY_RUNNER_ACCEPTANCE_SHA256,
        "primary_runner_acceptance": _ref(
            project_root / PRIMARY_RUNNER_ACCEPTANCE_RELATIVE, project_root
        ),
        "primary_environment_lock_sha256": PRIMARY_ENVIRONMENT_LOCK_SHA256,
        "primary_selection_index_sha256": PRIMARY_SELECTION_INDEX_SHA256,
        "primary_outer_index_sha256": PRIMARY_OUTER_INDEX_SHA256,
        "gate1_hash_manifest_sha256": GATE1_HASH_MANIFEST_SHA256,
        "dry_run_receipt": _ref(dry_receipt_path, project_root),
        "hybrid_coordinate_contract": {
            "new_fit_coordinates": 150,
            "immutable_primary_anchors": 50,
            "logical_coordinates": 200,
        },
        "claim_boundary": {
            "secondary_sensitivity": True,
            "can_reopen_gate1": False,
            "can_reopen_gate2": False,
        },
        "source_contracts": current_sources,
        "bundle_digests": current_bundles,
        "environment_lock": current_environment,
        "device_contract": {
            "device_name": EXPECTED_DEVICE_NAME,
            "compute_capability": list(EXPECTED_COMPUTE_CAPABILITY),
            "torch_version": EXPECTED_TORCH_VERSION,
            "cuda_version": EXPECTED_CUDA_VERSION,
            "cudnn_version": EXPECTED_CUDNN_VERSION,
            "must_match_primary_anchors_smokes_and_production": True,
        },
        "determinism": {
            "pythonhashseed": "coordinate_seed_before_process_start",
            "cublas_workspace_config": ":4096:8",
            "torch_deterministic_algorithms": True,
            "allow_tf32": False,
            "mixed_precision": False,
            "dataloader_workers": 0,
        },
        "commands": {
            "validate": "python revised_experiments/scripts/run_work9_capacity.py validate",
            "cuda_smoke": (
                "PYTHONHASHSEED=17 CUBLAS_WORKSPACE_CONFIG=:4096:8 python "
                "revised_experiments/scripts/run_work9_capacity.py smoke --device cuda:0 "
                "--condition <A-mean|AT-cross-randtrain> "
                "--output-directory "
                "revised_experiments/artifacts/work9_capacity_cuda_smokes_v1/<NEW_ID>"
            ),
            "fit_select_matrix": (
                "python revised_experiments/scripts/run_work9_capacity.py run "
                "--phase fit-select --device cuda:0"
            ),
            "freeze_selection": (
                "python revised_experiments/scripts/run_work9_capacity.py run "
                "--phase freeze-selection-index"
            ),
            "outer_eval_matrix": (
                "python revised_experiments/scripts/run_work9_capacity.py run "
                "--phase outer-eval --device cuda:0"
            ),
            "unit_test_receipt": (
                "python revised_experiments/scripts/run_work9_capacity_tests.py "
                "--output-directory "
                "revised_experiments/artifacts/work9_capacity_unit_tests_v1/<NEW_ID>"
            ),
            "finalize_acceptance": (
                "python revised_experiments/scripts/build_work9_capacity_runner_acceptance.py "
                "--finalize --smoke-receipt <SMOKE_A> --smoke-receipt <SMOKE_B> "
                "--unit-test-receipt <TEST_RECEIPT>"
            ),
        },
    }


def build_draft(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    payload = {
        **_base_payload(root),
        "status": "awaiting_two_independent_cuda_smokes",
        "cuda_smokes": [],
        "unit_tests": {"status": "must_pass_before_finalization"},
        "production_training_authorized": False,
    }
    target = root / ACCEPTANCE_DRAFT_RELATIVE
    _reject_symlink_components(target, stop=root)
    target.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write_once(target, canonical_json_bytes(payload)).path


def verify_draft_against_current(project_root: str | Path) -> tuple[Mapping[str, Any], str]:
    root = Path(project_root).resolve()
    draft_path = root / ACCEPTANCE_DRAFT_RELATIVE
    if not draft_path.is_file():
        raise Work9AcceptanceError("immutable runner-acceptance DRAFT must predate smoke/tests")
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    if draft.get("schema_version") != ACCEPTANCE_SCHEMA_VERSION:
        raise Work9AcceptanceError("runner-acceptance DRAFT schema is unsupported")
    if draft.get("status") != "awaiting_two_independent_cuda_smokes":
        raise Work9AcceptanceError("runner-acceptance draft has unexpected status")
    current_base = _base_payload(root)
    expected_keys = set(current_base) | {
        "status",
        "cuda_smokes",
        "unit_tests",
        "production_training_authorized",
    }
    if set(draft) != expected_keys:
        raise Work9AcceptanceError("runner-acceptance DRAFT keys changed")
    changed = [key for key, value in current_base.items() if draft.get(key) != value]
    if changed:
        raise Work9AcceptanceError(
            f"runner-acceptance immutable base changed after DRAFT: {changed}"
        )
    return draft, sha256_file(draft_path)


def _load_smoke(
    path: Path,
    project_root: Path,
    *,
    draft_sha256: str,
    bundles: Mapping[str, str],
    environment_lock_sha256: str,
) -> dict[str, Any]:
    if path == Path(".") or ".." in path.parts:
        raise Work9AcceptanceError("CUDA smoke receipt path is lexically unsafe")
    lexical = path if path.is_absolute() else project_root / path
    _reject_symlink_components(lexical, stop=project_root)
    allowed = (project_root / SMOKE_EVIDENCE_ROOT_RELATIVE).resolve()
    resolved = lexical.resolve()
    if (
        lexical.is_symlink()
        or not resolved.is_file()
        or allowed not in resolved.parents
        or resolved.name != "smoke_receipt.json"
    ):
        raise Work9AcceptanceError("CUDA smoke receipt is outside dedicated evidence root")
    path = resolved
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": CUDA_SMOKE_RECEIPT_SCHEMA_VERSION,
        "status": "pass",
        "device_type": "cuda",
        "seed": 17,
        "pythonhashseed": "17",
        "cublas_workspace_config": ":4096:8",
    }
    mismatch = {
        key: (wanted, raw.get(key))
        for key, wanted in required.items()
        if raw.get(key) != wanted
    }
    if mismatch:
        raise Work9AcceptanceError(f"CUDA smoke receipt mismatch: {mismatch}")
    if raw.get("condition") not in {"A-mean", "AT-cross-randtrain"}:
        raise Work9AcceptanceError("CUDA smoke condition is outside frozen branch contract")
    safety = raw.get("safety", {})
    if safety != {
        "outer_eval_accessed": False,
        "primary_selection_registry_read_only_verified": True,
        "primary_outer_registry_accessed": False,
        "primary_registry_modified": False,
        "scientific_selection_claimed": False,
    }:
        raise Work9AcceptanceError("CUDA smoke safety declaration changed")
    semantic = path.parent / "semantic_output.json"
    _reject_symlink_components(semantic, stop=project_root)
    if (
        semantic.is_symlink()
        or not semantic.is_file()
        or sha256_file(semantic) != raw.get("semantic_output_sha256")
        or semantic.stat().st_size != raw.get("semantic_output_bytes")
    ):
        raise Work9AcceptanceError("CUDA smoke semantic output hash mismatch")
    if raw.get("runner_acceptance_draft_sha256") != draft_sha256:
        raise Work9AcceptanceError("CUDA smoke did not bind current acceptance DRAFT")
    if raw.get("bundle_digests") != bundles:
        raise Work9AcceptanceError("CUDA smoke source/schema/test bundle differs")
    if raw.get("environment_lock_sha256") != environment_lock_sha256:
        raise Work9AcceptanceError("CUDA smoke environment lock differs")
    expected_device = {
        "device_name": EXPECTED_DEVICE_NAME,
        "compute_capability": list(EXPECTED_COMPUTE_CAPABILITY),
        "torch_version": EXPECTED_TORCH_VERSION,
        "cuda_version": EXPECTED_CUDA_VERSION,
        "cudnn_version": EXPECTED_CUDNN_VERSION,
    }
    mismatch = {
        key: (wanted, raw.get(key))
        for key, wanted in expected_device.items()
        if raw.get(key) != wanted
    }
    if mismatch:
        raise Work9AcceptanceError(f"CUDA smoke primary-device mismatch: {mismatch}")
    return {**raw, "receipt": _ref(path, project_root), "semantic_output": _ref(semantic, project_root)}


def _load_test_receipt(
    path: Path,
    project_root: Path,
    *,
    draft_sha256: str,
    bundles: Mapping[str, str],
    environment_lock_sha256: str,
) -> dict[str, Any]:
    if path == Path(".") or ".." in path.parts:
        raise Work9AcceptanceError("unit-test receipt path is lexically unsafe")
    lexical = path if path.is_absolute() else project_root / path
    _reject_symlink_components(lexical, stop=project_root)
    allowed = (project_root / UNIT_TEST_EVIDENCE_ROOT_RELATIVE).resolve()
    resolved = lexical.resolve()
    if (
        lexical.is_symlink()
        or not resolved.is_file()
        or allowed not in resolved.parents
        or resolved.name != "unit_test_receipt.json"
    ):
        raise Work9AcceptanceError("unit-test receipt is outside dedicated evidence root")
    path = resolved
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != UNIT_TEST_RECEIPT_SCHEMA_VERSION:
        raise Work9AcceptanceError("unit-test receipt schema mismatch")
    if raw.get("status") != "pass" or int(raw.get("tests_passed", 0)) <= 0:
        raise Work9AcceptanceError("unit-test receipt is not passing")
    if raw.get("runner_acceptance_draft_sha256") != draft_sha256:
        raise Work9AcceptanceError("unit tests did not bind current acceptance DRAFT")
    if raw.get("bundle_digests") != bundles:
        raise Work9AcceptanceError("unit-test source/schema/test bundle differs")
    if raw.get("environment_lock_sha256") != environment_lock_sha256:
        raise Work9AcceptanceError("unit-test environment lock differs")
    log_reference = raw.get("log")
    if not isinstance(log_reference, Mapping):
        raise Work9AcceptanceError("unit-test log reference is invalid")
    raw_log_path = Path(str(log_reference.get("path", "")))
    if (
        raw_log_path.is_absolute()
        or raw_log_path == Path(".")
        or ".." in raw_log_path.parts
    ):
        raise Work9AcceptanceError("unit-test log path is lexically unsafe")
    lexical_log_path = project_root / raw_log_path
    _reject_symlink_components(lexical_log_path, stop=project_root)
    log_path = lexical_log_path.resolve()
    expected_log_path = (path.parent / "unittest.log").resolve()
    if (
        lexical_log_path.is_symlink()
        or not log_path.is_file()
        or log_path != expected_log_path
        or sha256_file(log_path) != log_reference.get("sha256")
        or log_path.stat().st_size != log_reference.get("bytes")
        or log_reference.get("row_count") is not None
    ):
        raise Work9AcceptanceError("unit-test log reference is invalid")
    return {**raw, "receipt": _ref(path, project_root)}


def finalize_acceptance(
    project_root: str | Path,
    *,
    smoke_receipts: Sequence[str | Path],
    unit_test_receipt: str | Path,
) -> Path:
    root = Path(project_root).resolve()
    draft, draft_sha = verify_draft_against_current(root)
    bundles = dict(draft["bundle_digests"])
    environment_lock_sha256 = str(
        draft["environment_lock"]["environment_lock_sha256"]
    )
    if len(smoke_receipts) != 2:
        raise Work9AcceptanceError("exactly two independent CUDA smoke receipts required")
    smokes = [
        _load_smoke(
            Path(path),
            root,
            draft_sha256=draft_sha,
            bundles=bundles,
            environment_lock_sha256=environment_lock_sha256,
        )
        for path in smoke_receipts
    ]
    if smokes[0]["receipt"]["sha256"] == smokes[1]["receipt"]["sha256"]:
        raise Work9AcceptanceError("CUDA smoke receipts must be separate invocations")
    if {smoke["condition"] for smoke in smokes} != {
        "A-mean", "AT-cross-randtrain"
    }:
        raise Work9AcceptanceError("CUDA smokes must cover both frozen branch contracts")
    device_fields = (
        "device_name", "compute_capability", "torch_version", "cuda_version", "cudnn_version"
    )
    if any(smokes[0].get(key) != smokes[1].get(key) for key in device_fields):
        raise Work9AcceptanceError("CUDA smoke device identities differ")
    tests = _load_test_receipt(
        Path(unit_test_receipt),
        root,
        draft_sha256=draft_sha,
        bundles=bundles,
        environment_lock_sha256=environment_lock_sha256,
    )
    _, publication_draft_sha = verify_draft_against_current(root)
    if publication_draft_sha != draft_sha:
        raise Work9AcceptanceError("runner-acceptance DRAFT changed during finalization")
    draft_only_keys = {
        "status",
        "cuda_smokes",
        "unit_tests",
        "production_training_authorized",
    }
    accepted_base = {
        key: value for key, value in draft.items() if key not in draft_only_keys
    }
    payload = {
        **accepted_base,
        "status": "accepted_after_two_cuda_smokes",
        "runner_acceptance_draft": _ref(
            root / ACCEPTANCE_DRAFT_RELATIVE, root
        ),
        "cuda_smokes": smokes,
        "unit_tests": tests,
        "production_training_authorized": True,
    }
    target = root / ACCEPTANCE_RECEIPT_RELATIVE
    _reject_symlink_components(target, stop=root)
    target.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write_once(target, canonical_json_bytes(payload)).path
