from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from typing import Any


PROVIDER_STATUSES = frozenset(
    {
        "MODEL_OK",
        "EMPTY_OUTPUT",
        "ERROR",
    }
)


@dataclass(frozen=True, slots=True)
class NormalizedProviderResult:
    """
    Provider-agnostic result contract.

    This is the only result shape exposed by the unified
    provider bridge.
    """

    provider: str
    model: str
    provider_status: str

    input_tokens: int
    reasoning_tokens: int
    output_tokens: int
    total_tokens: int

    latency_ms: float

    raw_text: str
    response_id: str | None

    error: str | None

    def __post_init__(self) -> None:
        if self.provider_status not in PROVIDER_STATUSES:
            raise ValueError(
                "invalid provider_status: "
                + repr(self.provider_status)
            )

        for field_name in (
            "input_tokens",
            "reasoning_tokens",
            "output_tokens",
            "total_tokens",
        ):
            value = getattr(self, field_name)

            if type(value) is not int:
                raise TypeError(
                    f"{field_name} must be int (not bool)"
                )

            if value < 0:
                raise ValueError(
                    f"{field_name} must be >= 0"
                )

        if not isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError(
                "latency_ms must be finite and >= 0"
            )

        if not isinstance(self.raw_text, str):
            raise TypeError(
                "raw_text must be str"
            )

        if self.provider_status == "MODEL_OK":
            if not self.raw_text.strip():
                raise ValueError(
                    "MODEL_OK requires non-empty raw_text"
                )

        if self.provider_status == "EMPTY_OUTPUT":
            if self.raw_text.strip():
                raise ValueError(
                    "EMPTY_OUTPUT requires empty raw_text"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "provider_status": self.provider_status,
            "input_tokens": self.input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "raw_text": self.raw_text,
            "response_id": self.response_id,
            "error": self.error,
        }


class ProviderAdapter(ABC):
    """
    Stable abstraction consumed by future provider-aware runners.

    Current V0.7 execution does not import this class.
    """

    provider_name: str = ""

    @abstractmethod
    def generate(self, prompt: str) -> NormalizedProviderResult:
        raise NotImplementedError


def normalize_status(
    *,
    raw_status: Any,
    raw_text: str,
    error: str | None,
) -> str:
    """Provider failures never become successes merely because text exists.

    Adapters must explicitly assert MODEL_OK; unknown / timeout / error
    statuses fail closed even if an SDK provides partial or stale output.
    """
    text = raw_text.strip()
    if error is not None:
        return "ERROR"
    if raw_status == "MODEL_OK":
        return "MODEL_OK" if text else "EMPTY_OUTPUT"
    if raw_status in ("EMPTY_OUTPUT", "OUTPUT_EMPTY"):
        return "EMPTY_OUTPUT" if not text else "ERROR"
    return "ERROR"

def normalize_error(
    error: Any,
) -> str | None:

    if error is None:
        return None

    text = str(error).strip()

    return text if text else None


def normalize_int(
    value: Any,
) -> int:

    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def normalize_float(
    value: Any,
) -> float:

    try:
        parsed = float(value or 0.0)
        return parsed if isfinite(parsed) and parsed >= 0.0 else 0.0
    except (TypeError, ValueError, OverflowError):
        return 0.0


def build_result(
    *,
    provider: str,
    model: Any,
    raw_status: Any,
    input_tokens: Any = 0,
    reasoning_tokens: Any = 0,
    output_tokens: Any = 0,
    total_tokens: Any = 0,
    latency_ms: Any = 0.0,
    raw_text: Any = "",
    response_id: Any = None,
    error: Any = None,
) -> NormalizedProviderResult:

    model_text = str(model or "").strip()

    text = (
        raw_text
        if isinstance(raw_text, str)
        else str(raw_text or "")
    )

    normalized_error = normalize_error(error)

    status = normalize_status(
        raw_status=raw_status,
        raw_text=text,
        error=normalized_error,
    )

    return NormalizedProviderResult(
        provider=str(provider),
        model=model_text,
        provider_status=status,
        input_tokens=normalize_int(input_tokens),
        reasoning_tokens=normalize_int(reasoning_tokens),
        output_tokens=normalize_int(output_tokens),
        total_tokens=normalize_int(total_tokens),
        latency_ms=normalize_float(latency_ms),
        raw_text=text,
        response_id=(
            str(response_id)
            if response_id is not None
            else None
        ),
        error=normalized_error,
    )
