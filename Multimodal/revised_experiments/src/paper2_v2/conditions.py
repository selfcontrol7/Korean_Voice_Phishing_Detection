"""Declarative factory for the locked Work 8 MFCC comparison matrix.

Condition names, input requirements, and randomized-pairing behavior are
defined in one place.  The factory deliberately replaces the historical
negative ``--no_*`` flags, which allowed the loader and model to disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from torch import nn

from .data import InputSpec
from .models import (
    AudioOnlyModel,
    AudioTextConcatModel,
    AudioTextCrossModel,
    AudioTextGateModel,
    TextOnlyModel,
)


class ConditionId(str, Enum):
    TEXT = "T"
    AUDIO_MEAN = "A-mean"
    AUDIO_QUERY = "A-query"
    AUDIO_TEMPORAL = "A-temp"
    AUDIO_TEXT_CONCAT = "AT-concat"
    AUDIO_TEXT_GATE = "AT-gate"
    AUDIO_TEXT_CROSS = "AT-cross"
    AUDIO_TEXT_CROSS_RANDOM_TRAIN = "AT-cross-randtrain"


CORE_CONDITIONS: tuple[str, ...] = tuple(condition.value for condition in ConditionId)


class PairingMode(str, Enum):
    CORRECT = "correct"
    RANDOM_TRAIN_ONLY = "random_train_only"


@dataclass(frozen=True)
class ConditionDefinition:
    condition_id: ConditionId
    requires_mfcc: bool
    requires_text: bool
    pairing_mode: PairingMode
    mechanism: str

    def input_spec(self, *, text_path_key: str, text_dim: int) -> InputSpec:
        return InputSpec(
            sequence_features=("mfcc",) if self.requires_mfcc else (),
            global_features=("text",) if self.requires_text else (),
            text_path_key=text_path_key,
            expected_text_dim=text_dim,
        )


CONDITION_DEFINITIONS: dict[ConditionId, ConditionDefinition] = {
    ConditionId.TEXT: ConditionDefinition(
        ConditionId.TEXT, False, True, PairingMode.CORRECT, "text_only_mlp"
    ),
    ConditionId.AUDIO_MEAN: ConditionDefinition(
        ConditionId.AUDIO_MEAN, True, False, PairingMode.CORRECT, "masked_mean"
    ),
    ConditionId.AUDIO_QUERY: ConditionDefinition(
        ConditionId.AUDIO_QUERY, True, False, PairingMode.CORRECT, "learned_query_set"
    ),
    ConditionId.AUDIO_TEMPORAL: ConditionDefinition(
        ConditionId.AUDIO_TEMPORAL,
        True,
        False,
        PairingMode.CORRECT,
        "learned_query_absolute_position",
    ),
    ConditionId.AUDIO_TEXT_CONCAT: ConditionDefinition(
        ConditionId.AUDIO_TEXT_CONCAT,
        True,
        True,
        PairingMode.CORRECT,
        "learned_query_set_plus_text_concat",
    ),
    ConditionId.AUDIO_TEXT_GATE: ConditionDefinition(
        ConditionId.AUDIO_TEXT_GATE,
        True,
        True,
        PairingMode.CORRECT,
        "learned_query_set_plus_text_film",
    ),
    ConditionId.AUDIO_TEXT_CROSS: ConditionDefinition(
        ConditionId.AUDIO_TEXT_CROSS,
        True,
        True,
        PairingMode.CORRECT,
        "text_conditioned_acoustic_set_pool",
    ),
    ConditionId.AUDIO_TEXT_CROSS_RANDOM_TRAIN: ConditionDefinition(
        ConditionId.AUDIO_TEXT_CROSS_RANDOM_TRAIN,
        True,
        True,
        PairingMode.RANDOM_TRAIN_ONLY,
        "text_conditioned_acoustic_set_pool",
    ),
}


@dataclass(frozen=True)
class BuiltCondition:
    definition: ConditionDefinition
    input_spec: InputSpec
    model: nn.Module
    architecture: dict[str, Any]

    @property
    def total_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    @property
    def trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )


def _parse_condition(condition: str | ConditionId) -> ConditionId:
    if isinstance(condition, ConditionId):
        return condition
    try:
        return ConditionId(condition)
    except ValueError as error:
        raise ValueError(
            f"unknown Work 8 condition {condition!r}; expected one of {CORE_CONDITIONS}"
        ) from error


def build_condition(
    condition: str | ConditionId,
    *,
    hidden_dim: int = 256,
    num_heads: int = 8,
    dropout: float = 0.2,
    text_dim: int = 768,
    text_path_key: str = "__indexed_text_store__",
) -> BuiltCondition:
    """Construct one matrix condition and its exact loader requirements."""

    condition_id = _parse_condition(condition)
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if num_heads <= 0 or hidden_dim % num_heads:
        raise ValueError("num_heads must be positive and divide hidden_dim")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must satisfy 0 <= dropout < 1")
    # The current corrected multimodal implementations intentionally have one
    # verified global sentence-vector contract. A new dimensionality must be a
    # separately implemented and named representation, not an implicit concat.
    if text_dim != 768:
        raise ValueError("Work 8 requires one declared 768-D sentence representation")

    definition = CONDITION_DEFINITIONS[condition_id]
    common = {"hidden_dim": hidden_dim, "dropout": dropout}
    if condition_id is ConditionId.TEXT:
        model: nn.Module = TextOnlyModel(text_dim=text_dim, **common)
    elif condition_id is ConditionId.AUDIO_MEAN:
        model = AudioOnlyModel(pooling="mean", num_heads=num_heads, **common)
    elif condition_id is ConditionId.AUDIO_QUERY:
        model = AudioOnlyModel(pooling="set", num_heads=num_heads, **common)
    elif condition_id is ConditionId.AUDIO_TEMPORAL:
        model = AudioOnlyModel(pooling="temporal", num_heads=num_heads, **common)
    elif condition_id is ConditionId.AUDIO_TEXT_CONCAT:
        model = AudioTextConcatModel(num_heads=num_heads, **common)
    elif condition_id is ConditionId.AUDIO_TEXT_GATE:
        model = AudioTextGateModel(num_heads=num_heads, **common)
    elif condition_id in {
        ConditionId.AUDIO_TEXT_CROSS,
        ConditionId.AUDIO_TEXT_CROSS_RANDOM_TRAIN,
    }:
        # The two conditions must remain exactly architecture-identical. Only
        # their training pairing view differs.
        model = AudioTextCrossModel(num_heads=num_heads, temporal=False, **common)
    else:  # pragma: no cover - Enum exhaustiveness guard
        raise AssertionError(condition_id)

    input_spec = definition.input_spec(
        text_path_key=text_path_key,
        text_dim=text_dim,
    )
    architecture = {
        "condition_id": condition_id.value,
        "mechanism": definition.mechanism,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "dropout": dropout,
        "text_dim": text_dim,
        "sequence_features": list(input_spec.sequence_features),
        "global_features": list(input_spec.global_features),
        "pairing_mode": definition.pairing_mode.value,
    }
    return BuiltCondition(definition, input_spec, model, architecture)
