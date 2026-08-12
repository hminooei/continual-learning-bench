"""Shared usage and cost tracking helpers for system compute."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional

import litellm

# Long-context tier rate keys in litellm.model_cost look like
# `input_cost_per_token_above_200k_tokens`. Anthropic, Gemini, and GPT-5
# (non-mini) all publish such rates. litellm.cost_per_token applies the tier
# correctly for Anthropic/Gemini but not for OpenAI gpt-5; we apply it
# ourselves whenever a tiered rate exists so behavior is consistent across
# providers.
_TIER_KEY_PATTERN = re.compile(r"^input_cost_per_token_above_(\d+)k_tokens$")


@dataclass
class UsageEvent:
    """Normalized record of one billable model call."""

    call_type: str
    model: str
    provider: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    pricing_source: Optional[str] = None
    pricing_error: Optional[str] = None
    response_id: Optional[str] = None
    raw_usage: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.cost_usd is not None:
            payload["cost_usd"] = round(float(self.cost_usd), 10)
        return payload


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage_value(usage: Any, *names: str) -> Any:
    for name in names:
        if usage is None:
            continue
        if hasattr(usage, name):
            return getattr(usage, name)
        if isinstance(usage, dict) and name in usage:
            return usage[name]
    return None


def _resolve_usage_counts(
    response: Any,
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    input_tokens = _coerce_optional_int(
        _usage_value(
            usage,
            "prompt_tokens",
            "input_tokens",
            "prompt_token_count",
        )
    )
    output_tokens = _coerce_optional_int(
        _usage_value(
            usage,
            "completion_tokens",
            "output_tokens",
            "completion_token_count",
        )
    )
    total_tokens = _coerce_optional_int(_usage_value(usage, "total_tokens"))
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return input_tokens, output_tokens, total_tokens


def _resolve_usage_details(
    response: Any,
) -> tuple[Optional[int], Optional[int], Optional[int], Optional[dict[str, Any]]]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    raw_usage = _plain_usage(usage)
    reasoning_tokens = _coerce_optional_int(
        _usage_nested_value(
            usage,
            ("output_tokens_details", "reasoning_tokens"),
            ("completion_tokens_details", "reasoning_tokens"),
            ("reasoning_tokens",),
            ("reasoning_output_tokens",),
        )
    )
    cached_input_tokens = _coerce_optional_int(
        _usage_nested_value(
            usage,
            ("input_tokens_details", "cached_tokens"),
            ("prompt_tokens_details", "cached_tokens"),
            ("cached_input_tokens",),
            ("cache_read_input_tokens",),
        )
    )
    cache_creation_input_tokens = _coerce_optional_int(
        _usage_nested_value(usage, ("cache_creation_input_tokens",))
    )
    return (
        reasoning_tokens,
        cached_input_tokens,
        cache_creation_input_tokens,
        raw_usage,
    )


def _usage_nested_value(usage: Any, *paths: tuple[str, ...]) -> Any:
    for path in paths:
        current = usage
        for name in path:
            current = _usage_value(current, name)
            if current is None:
                break
        if current is not None:
            return current
    return None


def _plain_usage(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _plain_usage_value(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return {str(k): _plain_usage_value(v) for k, v in dumped.items()}
    if hasattr(value, "__dict__"):
        return {
            str(k): _plain_usage_value(v)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
    return None


def _plain_usage_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_plain_usage_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _plain_usage_value(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        return _plain_usage_value(value.model_dump())
    if hasattr(value, "__dict__"):
        return {
            str(k): _plain_usage_value(v)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
    return str(value)


def _lookup_model_rates(
    model: str, provider: Optional[str]
) -> Optional[dict[str, Any]]:
    """Return the litellm.model_cost entry for a model, trying common key forms."""
    clean = (
        model.removeprefix("vertex_ai/")
        .removeprefix("vertex/")
        .removeprefix("anthropic/")
    )
    candidates = [
        model,
        model.split("/", 1)[-1],
        clean,
        f"vertex_ai/{clean}",
        f"anthropic/{clean}",
    ]
    if "@" in clean:
        base = clean.split("@", 1)[0]
        base_clean = base.removesuffix("-v2").removesuffix("-v1")
        candidates.extend(
            [
                base,
                base_clean,
                f"vertex_ai/{base}",
                f"vertex_ai/{base_clean}",
                f"anthropic/{base}",
                f"anthropic/{base_clean}",
            ]
        )
    elif "-v2" in clean or "-v1" in clean:
        base_clean = clean.removesuffix("-v2").removesuffix("-v1")
        candidates.extend(
            [
                base_clean,
                f"vertex_ai/{base_clean}",
                f"anthropic/{base_clean}",
            ]
        )
    if provider:
        candidates.append(f"{provider}/{model}")
        candidates.append(f"{provider}/{clean}")
    seen: set[str] = set()
    for key in candidates:
        if not key or key in seen:
            continue
        seen.add(key)
        rates = getattr(litellm, "model_cost", {}).get(key)
        if rates:
            return rates
    return None


def _detect_tier_threshold(rates: dict[str, Any]) -> Optional[int]:
    """Return the long-context threshold in tokens, if the rate table has one."""
    for key in rates:
        match = _TIER_KEY_PATTERN.match(key)
        if match:
            return int(match.group(1)) * 1000
    return None


def _compute_tiered_cost(
    rates: dict[str, Any],
    threshold: int,
    *,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int,
    cache_creation_input_tokens: int,
) -> float:
    """Apply long-context tier rates when input_tokens exceeds threshold.

    For all known providers (Anthropic, Gemini, OpenAI), once the prompt
    crosses the threshold every input and output token is billed at the
    tiered rate — the rate change is not piecewise across the boundary.
    Cache_read fallback (10% of input rate) matches litellm's convention.
    """
    suffix = f"_above_{threshold // 1000}k_tokens"
    use_tier = input_tokens > threshold

    def rate(base_key: str) -> float:
        if use_tier:
            tier_value = rates.get(f"{base_key}{suffix}")
            if tier_value is not None:
                return float(tier_value)
        value = rates.get(base_key)
        return float(value) if value is not None else 0.0

    input_rate = rate("input_cost_per_token")
    cache_read_rate_value = rates.get(
        f"cache_read_input_token_cost{suffix}"
        if use_tier
        else "cache_read_input_token_cost"
    )
    if cache_read_rate_value is None:
        cache_read_rate_value = rates.get("cache_read_input_token_cost")
    cache_read_rate = (
        float(cache_read_rate_value)
        if cache_read_rate_value is not None
        else input_rate * 0.1
    )
    cache_creation_rate_value = rates.get(
        f"cache_creation_input_token_cost{suffix}"
        if use_tier
        else "cache_creation_input_token_cost"
    )
    if cache_creation_rate_value is None:
        cache_creation_rate_value = rates.get("cache_creation_input_token_cost")
    cache_creation_rate = (
        float(cache_creation_rate_value)
        if cache_creation_rate_value is not None
        else input_rate
    )
    output_rate = rate("output_cost_per_token")

    uncached_input = max(
        0, input_tokens - cached_input_tokens - cache_creation_input_tokens
    )
    return (
        uncached_input * input_rate
        + cached_input_tokens * cache_read_rate
        + cache_creation_input_tokens * cache_creation_rate
        + output_tokens * output_rate
    )


def _compute_cost(
    *,
    model: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    cached_input_tokens: Optional[int] = None,
    cache_creation_input_tokens: Optional[int] = None,
    call_type: str,
    provider: Optional[str],
    response: Any = None,
) -> tuple[Optional[float], Optional[str], Optional[str]]:
    last_error: Optional[str] = None

    rates = _lookup_model_rates(model, provider)
    threshold = _detect_tier_threshold(rates) if rates else None
    if threshold is not None and rates is not None and (input_tokens or 0) > threshold:
        try:
            cost = _compute_tiered_cost(
                rates,
                threshold,
                input_tokens=int(input_tokens or 0),
                output_tokens=int(output_tokens or 0),
                cached_input_tokens=int(cached_input_tokens or 0),
                cache_creation_input_tokens=int(cache_creation_input_tokens or 0),
            )
            return cost, "cl_benchmark.tiered", None
        except Exception as exc:  # pragma: no cover - guard against rate-table drift
            last_error = str(exc)

    if response is not None:
        try:
            return (
                float(
                    litellm.completion_cost(
                        completion_response=response,
                        model=model,
                        call_type=call_type,
                        custom_llm_provider=provider,
                    )
                ),
                "litellm.completion_cost",
                None,
            )
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = str(exc)

    for custom_provider in (provider, None):
        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=model,
                prompt_tokens=input_tokens or 0,
                completion_tokens=output_tokens or 0,
                cache_read_input_tokens=cached_input_tokens or 0,
                cache_creation_input_tokens=cache_creation_input_tokens or 0,
                call_type=call_type,
                custom_llm_provider=custom_provider,
            )
            return (
                float(prompt_cost) + float(completion_cost),
                "litellm.cost_per_token",
                None,
            )
        except Exception as exc:
            last_error = str(exc)

    return None, None, last_error


def build_usage_event(
    *,
    model: str,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    call_type: str = "completion",
    provider: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    response: Any = None,
    reasoning_tokens: Optional[int] = None,
    cached_input_tokens: Optional[int] = None,
    cache_creation_input_tokens: Optional[int] = None,
    response_id: Optional[str] = None,
    raw_usage: Optional[dict[str, Any]] = None,
) -> UsageEvent:
    """Create a usage event from token counts and optional raw response."""

    resolved_total_tokens = (
        _coerce_optional_int(total_tokens)
        if total_tokens is not None
        else (
            (_coerce_optional_int(input_tokens) or 0)
            + (_coerce_optional_int(output_tokens) or 0)
            if input_tokens is not None or output_tokens is not None
            else None
        )
    )
    resolved_input_tokens = _coerce_optional_int(input_tokens)
    resolved_output_tokens = _coerce_optional_int(output_tokens)
    resolved_reasoning_tokens = _coerce_optional_int(reasoning_tokens)
    resolved_cached_input_tokens = _coerce_optional_int(cached_input_tokens)
    resolved_cache_creation_input_tokens = _coerce_optional_int(
        cache_creation_input_tokens
    )
    cost_usd, pricing_source, pricing_error = _compute_cost(
        model=model,
        input_tokens=resolved_input_tokens,
        output_tokens=resolved_output_tokens,
        cached_input_tokens=resolved_cached_input_tokens,
        cache_creation_input_tokens=resolved_cache_creation_input_tokens,
        call_type=call_type,
        provider=provider,
        response=response,
    )
    return UsageEvent(
        call_type=call_type,
        model=model,
        provider=provider,
        input_tokens=resolved_input_tokens,
        output_tokens=resolved_output_tokens,
        total_tokens=resolved_total_tokens,
        reasoning_tokens=resolved_reasoning_tokens,
        cached_input_tokens=resolved_cached_input_tokens,
        cache_creation_input_tokens=resolved_cache_creation_input_tokens,
        cost_usd=cost_usd,
        pricing_source=pricing_source,
        pricing_error=pricing_error,
        response_id=response_id,
        raw_usage=raw_usage,
        metadata=metadata,
    )


def build_usage_event_from_response(
    *,
    model: str,
    response: Any,
    call_type: str = "completion",
    provider: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> UsageEvent:
    """Create a usage event directly from an LLM response object."""

    input_tokens, output_tokens, total_tokens = _resolve_usage_counts(response)
    (
        reasoning_tokens,
        cached_input_tokens,
        cache_creation_input_tokens,
        raw_usage,
    ) = _resolve_usage_details(response)
    return build_usage_event(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        call_type=call_type,
        provider=provider,
        metadata=metadata,
        response=response,
        reasoning_tokens=reasoning_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        response_id=_usage_response_id(response),
        raw_usage=raw_usage,
    )


def _usage_response_id(response: Any) -> Optional[str]:
    response_id = getattr(response, "id", None)
    if response_id is None and isinstance(response, dict):
        response_id = response.get("id")
    return response_id if isinstance(response_id, str) else None


def coerce_usage_event(event: UsageEvent | dict[str, Any]) -> UsageEvent:
    if isinstance(event, UsageEvent):
        return event
    return UsageEvent(**event)


def serialize_usage_events(
    events: Iterable[UsageEvent | dict[str, Any]],
) -> list[dict[str, Any]]:
    return [coerce_usage_event(event).to_dict() for event in events]


def summarize_usage_events(
    events: Iterable[UsageEvent | dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate usage events into a compact summary."""

    coerced_events = [coerce_usage_event(event) for event in events]
    call_count = len(coerced_events)
    input_events = [event for event in coerced_events if event.input_tokens is not None]
    output_events = [
        event for event in coerced_events if event.output_tokens is not None
    ]
    total_events = [event for event in coerced_events if event.total_tokens is not None]
    reasoning_events = [
        event for event in coerced_events if event.reasoning_tokens is not None
    ]
    cached_events = [
        event for event in coerced_events if event.cached_input_tokens is not None
    ]
    cache_creation_events = [
        event
        for event in coerced_events
        if event.cache_creation_input_tokens is not None
    ]
    input_tokens = sum(int(event.input_tokens) for event in input_events)
    output_tokens = sum(int(event.output_tokens) for event in output_events)
    total_tokens = sum(int(event.total_tokens) for event in total_events)
    reasoning_tokens = sum(int(event.reasoning_tokens) for event in reasoning_events)
    cached_input_tokens = sum(int(event.cached_input_tokens) for event in cached_events)
    cache_creation_input_tokens = sum(
        int(event.cache_creation_input_tokens) for event in cache_creation_events
    )
    priced_events = [event for event in coerced_events if event.cost_usd is not None]
    cost_usd = sum(float(event.cost_usd) for event in priced_events)
    unpriced_call_count = call_count - len(priced_events)
    missing_input_token_count = call_count - len(input_events)
    missing_output_token_count = call_count - len(output_events)
    missing_total_token_count = call_count - len(total_events)
    missing_reasoning_token_count = call_count - len(reasoning_events)
    missing_cached_input_token_count = call_count - len(cached_events)
    missing_cache_creation_input_token_count = call_count - len(cache_creation_events)

    by_model: dict[str, dict[str, Any]] = {}
    by_call_type: dict[str, dict[str, Any]] = {}

    def _accumulate(bucket: dict[str, Any], event: UsageEvent) -> None:
        bucket["call_count"] += 1
        if event.input_tokens is None:
            bucket["missing_input_token_count"] += 1
        else:
            bucket["input_tokens"] += int(event.input_tokens)
        if event.output_tokens is None:
            bucket["missing_output_token_count"] += 1
        else:
            bucket["output_tokens"] += int(event.output_tokens)
        if event.total_tokens is None:
            bucket["missing_total_token_count"] += 1
        else:
            bucket["total_tokens"] += int(event.total_tokens)
        if event.reasoning_tokens is None:
            bucket["missing_reasoning_token_count"] += 1
        else:
            bucket["reasoning_tokens"] += int(event.reasoning_tokens)
        if event.cached_input_tokens is None:
            bucket["missing_cached_input_token_count"] += 1
        else:
            bucket["cached_input_tokens"] += int(event.cached_input_tokens)
        if event.cache_creation_input_tokens is None:
            bucket["missing_cache_creation_input_token_count"] += 1
        else:
            bucket["cache_creation_input_tokens"] += int(
                event.cache_creation_input_tokens
            )
        if event.cost_usd is None:
            bucket["unpriced_call_count"] += 1
        else:
            bucket["priced_call_count"] += 1
            bucket["cost_usd"] += float(event.cost_usd)

    for event in coerced_events:
        model_key = f"{event.provider}/{event.model}" if event.provider else event.model
        model_bucket = by_model.setdefault(
            model_key,
            {
                "call_count": 0,
                "priced_call_count": 0,
                "unpriced_call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "cached_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "missing_input_token_count": 0,
                "missing_output_token_count": 0,
                "missing_total_token_count": 0,
                "missing_reasoning_token_count": 0,
                "missing_cached_input_token_count": 0,
                "missing_cache_creation_input_token_count": 0,
                "cost_usd": 0.0,
            },
        )
        _accumulate(model_bucket, event)

        call_type_bucket = by_call_type.setdefault(
            event.call_type,
            {
                "call_count": 0,
                "priced_call_count": 0,
                "unpriced_call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "cached_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "missing_input_token_count": 0,
                "missing_output_token_count": 0,
                "missing_total_token_count": 0,
                "missing_reasoning_token_count": 0,
                "missing_cached_input_token_count": 0,
                "missing_cache_creation_input_token_count": 0,
                "cost_usd": 0.0,
            },
        )
        _accumulate(call_type_bucket, event)

    for bucket in list(by_model.values()) + list(by_call_type.values()):
        bucket["cost_usd"] = round(bucket["cost_usd"], 10)
        bucket["pricing_complete"] = bucket["unpriced_call_count"] == 0
        bucket["input_tokens_complete"] = bucket["missing_input_token_count"] == 0
        bucket["output_tokens_complete"] = bucket["missing_output_token_count"] == 0
        bucket["total_tokens_complete"] = bucket["missing_total_token_count"] == 0
        bucket["reasoning_tokens_complete"] = (
            bucket["missing_reasoning_token_count"] == 0
        )
        bucket["cached_input_tokens_complete"] = (
            bucket["missing_cached_input_token_count"] == 0
        )
        bucket["cache_creation_input_tokens_complete"] = (
            bucket["missing_cache_creation_input_token_count"] == 0
        )

    summary = {
        "call_count": call_count,
        "priced_call_count": len(priced_events),
        "unpriced_call_count": unpriced_call_count,
        "pricing_complete": unpriced_call_count == 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "missing_input_token_count": missing_input_token_count,
        "missing_output_token_count": missing_output_token_count,
        "missing_total_token_count": missing_total_token_count,
        "missing_reasoning_token_count": missing_reasoning_token_count,
        "missing_cached_input_token_count": missing_cached_input_token_count,
        "missing_cache_creation_input_token_count": missing_cache_creation_input_token_count,
        "input_tokens_complete": missing_input_token_count == 0,
        "output_tokens_complete": missing_output_token_count == 0,
        "total_tokens_complete": missing_total_token_count == 0,
        "reasoning_tokens_complete": missing_reasoning_token_count == 0,
        "cached_input_tokens_complete": missing_cached_input_token_count == 0,
        "cache_creation_input_tokens_complete": (
            missing_cache_creation_input_token_count == 0
        ),
        "cost_usd": round(cost_usd, 10),
        "by_model": by_model,
        "by_call_type": by_call_type,
    }

    if call_count == 0:
        summary["pricing_complete"] = True

    return summary
