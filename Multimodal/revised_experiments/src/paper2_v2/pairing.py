"""Immutable, group-safe text pairing plans for Work 8 controls."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

from .conditions import CONDITION_DEFINITIONS, ConditionId, PairingMode


@dataclass(frozen=True, order=True)
class PairingItem:
    segment_id: str
    call_id: str
    group_id: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "PairingItem":
        values = tuple(str(row.get(key, "")) for key in ("segment_id", "call_id", "group_id"))
        if any(not value for value in values):
            raise ValueError("pairing rows require nonempty segment_id, call_id, and group_id")
        return cls(*values)


@dataclass(frozen=True)
class PairingPlan:
    """Receiver segment -> donor-text segment mapping.

    The mapping is a one-to-one permutation and never crosses into validation
    or outer evaluation. Donors must come from a different hard-relation group,
    which also guarantees a different call under the Work 6 group contract.
    """

    seed: int
    split_role: str
    donor_by_receiver: Mapping[str, str]
    item_by_segment: Mapping[str, PairingItem]

    def __post_init__(self) -> None:
        if self.split_role != "train":
            raise ValueError("randomized pairing plans are permitted only for train")
        donor_by_receiver = dict(self.donor_by_receiver)
        item_by_segment = dict(self.item_by_segment)
        if set(donor_by_receiver) != set(item_by_segment):
            raise ValueError("pairing receivers must equal the training segment universe")
        if set(donor_by_receiver.values()) != set(item_by_segment):
            raise ValueError("pairing donors must be a one-to-one permutation")
        for receiver_id, donor_id in donor_by_receiver.items():
            receiver = item_by_segment[receiver_id]
            donor = item_by_segment[donor_id]
            if receiver.segment_id == donor.segment_id:
                raise ValueError("random pairing cannot retain the same segment")
            if receiver.call_id == donor.call_id:
                raise ValueError("random pairing donor must come from a different call")
            if receiver.group_id == donor.group_id:
                raise ValueError("random pairing donor must come from a different group")
        object.__setattr__(
            self, "donor_by_receiver", MappingProxyType(donor_by_receiver)
        )
        object.__setattr__(self, "item_by_segment", MappingProxyType(item_by_segment))

    @property
    def sha256(self) -> str:
        payload = {
            "seed": self.seed,
            "split_role": self.split_role,
            "pairs": [
                {"receiver_segment_id": receiver, "donor_segment_id": donor}
                for receiver, donor in sorted(self.donor_by_receiver.items())
            ],
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def donor_segment_id(self, receiver_segment_id: str) -> str:
        try:
            return self.donor_by_receiver[receiver_segment_id]
        except KeyError as error:
            raise KeyError(
                f"segment {receiver_segment_id!r} is not in this train pairing plan"
            ) from error

    def as_records(self) -> list[dict[str, object]]:
        records = []
        for receiver_id, donor_id in sorted(self.donor_by_receiver.items()):
            receiver = self.item_by_segment[receiver_id]
            donor = self.item_by_segment[donor_id]
            records.append(
                {
                    "seed": self.seed,
                    "split_role": self.split_role,
                    "receiver_segment_id": receiver_id,
                    "receiver_call_id": receiver.call_id,
                    "receiver_group_id": receiver.group_id,
                    "donor_segment_id": donor_id,
                    "donor_call_id": donor.call_id,
                    "donor_group_id": donor.group_id,
                }
            )
        return records


def _coerce_items(
    rows: Iterable[PairingItem | Mapping[str, object]],
) -> list[PairingItem]:
    items = [
        row if isinstance(row, PairingItem) else PairingItem.from_mapping(row)
        for row in rows
    ]
    if len(items) < 2:
        raise ValueError("at least two training segments are required for random pairing")
    segment_ids = [item.segment_id for item in items]
    if len(set(segment_ids)) != len(segment_ids):
        raise ValueError("pairing input contains duplicate segment_id")
    call_groups: dict[str, set[str]] = defaultdict(set)
    for item in items:
        call_groups[item.call_id].add(item.group_id)
    inconsistent = sorted(call for call, groups in call_groups.items() if len(groups) != 1)
    if inconsistent:
        raise ValueError(
            "each call must have exactly one stable group_id; inconsistent calls: "
            f"{inconsistent[:5]}"
        )
    return items


def build_random_train_pairing(
    rows: Iterable[PairingItem | Mapping[str, object]],
    *,
    seed: int,
    split_role: str = "train",
) -> PairingPlan:
    """Build a deterministic group-disjoint permutation in O(N log N).

    Items are arranged in contiguous group blocks and circularly shifted by
    the largest block size. Such a derangement exists exactly when no group
    contains more than half the segment universe. This avoids an unreliable
    retry-until-lucky shuffle on the 35k-segment development pool.
    """

    if split_role != "train":
        raise ValueError("randomized pairing may be constructed only for train")
    items = _coerce_items(rows)
    by_group: dict[str, list[PairingItem]] = defaultdict(list)
    for item in items:
        by_group[item.group_id].append(item)
    max_group_size = max(len(group_items) for group_items in by_group.values())
    if max_group_size * 2 > len(items):
        raise ValueError(
            "no group-disjoint permutation exists: one group contains more than half "
            "of the training segments"
        )

    rng = random.Random(seed)
    group_ids = sorted(by_group)
    rng.shuffle(group_ids)
    ordered: list[PairingItem] = []
    for group_id in group_ids:
        group_items = sorted(by_group[group_id])
        rng.shuffle(group_items)
        ordered.extend(group_items)

    donors = ordered[max_group_size:] + ordered[:max_group_size]
    mapping = {
        receiver.segment_id: donor.segment_id
        for receiver, donor in zip(ordered, donors, strict=True)
    }
    return PairingPlan(
        seed=seed,
        split_role="train",
        donor_by_receiver=mapping,
        item_by_segment={item.segment_id: item for item in items},
    )


def pairing_plan_for_split(
    condition: str | ConditionId,
    *,
    split_role: str,
    rows: Iterable[PairingItem | Mapping[str, object]],
    seed: int,
) -> PairingPlan | None:
    """Return a plan only for AT-cross-randtrain's training view.

    Validation and outer evaluation always return ``None``, meaning the exact
    paired text referenced by the feature store is used.
    """

    condition_id = condition if isinstance(condition, ConditionId) else ConditionId(condition)
    definition = CONDITION_DEFINITIONS[condition_id]
    if definition.pairing_mode is PairingMode.CORRECT or split_role != "train":
        return None
    return build_random_train_pairing(rows, seed=seed, split_role=split_role)

