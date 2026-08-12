"""Provider-native turn adapters for ICL-style systems.

The old ICL path stores only visible chat messages. Reasoning APIs often return
provider-specific state that must be round-tripped separately, so this module
keeps that state out of benchmark-visible artifacts while preserving it for the
next model call.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import litellm
from pydantic import BaseModel, ValidationError

from ...errors import _is_content_policy_refusal, as_provider_refusal
from ...usage import UsageEvent, build_usage_event
from .structured_output import (
    RETRYABLE_HTTP_STATUSES,
    completion_with_structured_output,
    extract_content,
    extract_http_status,
    schema_to_prompt_instruction,
    validate_with_coercion,
)

logger = logging.getLogger(__name__)

ProviderName = Literal["openai", "anthropic", "gemini", "litellm"]
ProviderMode = Literal["auto", "native", "litellm_chat"]
ContinuityMode = Literal[
    "provider_native_stateful",
    "encrypted_reasoning_stateless",
    "visible_context_only",
    "chat_completion_fallback",
]


@dataclass
class ProviderState:
    """Mutable provider state owned by one ICL-style system instance."""

    provider: ProviderName
    provider_mode: ProviderMode
    continuity_mode: ContinuityMode
    sent_message_count: int = 0
    previous_response_id: str | None = None
    native_messages: list[dict[str, Any]] = field(default_factory=list)
    encrypted_reasoning_items: list[dict[str, Any]] = field(default_factory=list)
    response_ids: list[str] = field(default_factory=list)
    hidden_state_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.sent_message_count = 0
        self.previous_response_id = None
        self.native_messages = []
        self.encrypted_reasoning_items = []
        self.response_ids = []
        self.hidden_state_used = False
        self.metadata = {}

    def to_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["native_message_count"] = len(self.native_messages)
        payload["encrypted_reasoning_item_count"] = len(self.encrypted_reasoning_items)
        payload.pop("native_messages", None)
        payload.pop("encrypted_reasoning_items", None)
        return payload


@dataclass
class ProviderTurnResult:
    action: BaseModel
    assistant_record: str
    usage_events: list[UsageEvent]
    metadata: dict[str, Any] = field(default_factory=dict)


def resolve_vertex_config(
    project_id: str | None = None,
    region: str | None = None,
) -> tuple[str | None, str]:
    """Resolve project_id and region for Vertex AI Anthropic calls.

    Region defaults to 'global' if not specified in arguments or environment.
    Project ID is resolved from explicit arguments, environment variables, or
    Google Application Default Credentials (ADC).
    """
    resolved_region = (
        region
        or os.environ.get("ANTHROPIC_VERTEX_LOCATION")
        or os.environ.get("CLOUD_ML_REGION")
        or os.environ.get("VERTEXAI_LOCATION")
        or os.environ.get("GOOGLE_CLOUD_REGION")
        or os.environ.get("LOCATION")
        or "global"
    )

    resolved_project = (
        project_id
        or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
        or os.environ.get("CLOUD_ML_PROJECT_ID")
        or os.environ.get("VERTEXAI_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT")
    )
    if not resolved_project:
        try:
            import google.auth

            _, default_project = google.auth.default()
            if default_project:
                resolved_project = default_project
        except Exception:
            pass

    return resolved_project, resolved_region


def is_vertex_available() -> bool:
    """Check if Vertex AI Google Cloud authentication is available."""
    if any(
        os.environ.get(k)
        for k in (
            "ANTHROPIC_VERTEX_PROJECT_ID",
            "CLOUD_ML_PROJECT_ID",
            "VERTEXAI_PROJECT",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_APPLICATION_CREDENTIALS",
        )
    ):
        return True
    try:
        import google.auth

        creds, _ = google.auth.default()
        return creds is not None
    except Exception:
        return False


def resolve_anthropic_auth_provider(
    model: str,
    configured_provider: str = "auto",
) -> Literal["direct", "vertex"]:
    """Resolve whether an Anthropic model call should use Direct Anthropic API or Vertex AI."""
    if configured_provider == "vertex":
        return "vertex"
    if configured_provider == "direct":
        return "direct"

    # In "auto" mode:
    model_lower = model.lower()
    if (
        model_lower.startswith("vertex_ai/")
        or model_lower.startswith("vertex/")
        or model_lower.startswith("google/")
        or os.environ.get("CLAUDE_CODE_USE_VERTEX") == "1"
        or os.environ.get("ANTHROPIC_AUTH_PROVIDER", "").lower() == "vertex"
    ):
        return "vertex"

    if os.environ.get("ANTHROPIC_API_KEY"):
        return "direct"

    # Fallback to Vertex AI if available
    if is_vertex_available():
        return "vertex"

    return "direct"


def detect_provider(model: str) -> ProviderName:
    model_lower = model.lower()
    if (
        "claude" in model_lower
        or model_lower.startswith("anthropic/")
        or model_lower.startswith("vertex_ai/claude")
        or model_lower.startswith("vertex/claude")
    ):
        return "anthropic"
    if "gemini" in model_lower or model_lower.startswith("google/"):
        return "gemini"
    if (
        model_lower.startswith("openai/")
        or model_lower.startswith("gpt-")
        or model_lower.startswith("o")
    ):
        return "openai"
    return "litellm"


def make_provider_state(
    model: str,
    provider_mode: ProviderMode = "auto",
    *,
    openai_store: bool = True,
) -> ProviderState:
    provider = detect_provider(model)
    if provider_mode == "litellm_chat":
        provider = "litellm"
        continuity: ContinuityMode = "chat_completion_fallback"
    elif provider == "openai":
        continuity = (
            "provider_native_stateful"
            if openai_store
            else "encrypted_reasoning_stateless"
        )
    elif provider == "anthropic":
        continuity = "provider_native_stateful"
    elif provider == "gemini":
        continuity = "visible_context_only"
    else:
        continuity = "chat_completion_fallback"
    return ProviderState(
        provider=provider,
        provider_mode=provider_mode,
        continuity_mode=continuity,
    )


class ProviderTurnClient:
    """Routes one benchmark turn through the best provider-specific API."""

    def __init__(
        self,
        *,
        model: str,
        system_prompt: str = "",
        provider_mode: ProviderMode = "auto",
        openai_store: bool = True,
        openai_include_encrypted_reasoning: bool = False,
        anthropic_max_tokens: int | None = None,
        anthropic_project_id: str | None = None,
        anthropic_region: str | None = None,
        anthropic_auth_provider: Literal["auto", "direct", "vertex"] = "auto",
        anthropic_stream: bool = False,
        content_policy_max_retries: int = 5,
        content_policy_retry_delay: float = 2.0,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.provider_mode = provider_mode
        self.openai_store = openai_store
        self.openai_include_encrypted_reasoning = openai_include_encrypted_reasoning
        self.anthropic_max_tokens = _resolve_model_max_output_tokens(
            model,
            anthropic_max_tokens,
        )
        self.anthropic_project_id = anthropic_project_id
        self.anthropic_region = anthropic_region
        self.anthropic_auth_provider = anthropic_auth_provider
        self.anthropic_stream = anthropic_stream
        self.content_policy_max_retries = max(0, int(content_policy_max_retries))
        self.content_policy_retry_delay = max(0.0, float(content_policy_retry_delay))
        self.state = make_provider_state(
            model,
            provider_mode,
            openai_store=openai_store,
        )

    def reset(self) -> None:
        self.state.reset()

    def state_metadata(self) -> dict[str, Any]:
        return self.state.to_metadata()

    def respond_structured(
        self,
        *,
        messages: list[dict[str, Any]],
        response_schema: type[BaseModel],
    ) -> ProviderTurnResult:
        if self.state.provider == "openai":
            return self._openai_structured(messages, response_schema)
        if self.state.provider == "anthropic":
            return self._anthropic_structured(messages, response_schema)
        # Gemini is intentionally explicit: no hidden reasoning/session primitive
        # is available through this adapter, so preserve visible chat behavior.
        return self._chat_structured(messages, response_schema)

    def _new_visible_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        sent = max(0, min(self.state.sent_message_count, len(messages)))
        if sent >= len(messages) and messages:
            # Visible FIFO truncation can make the old sent count larger than
            # the current list. The newest message is the active turn.
            return messages[-1:]
        return messages[sent:]

    def _mark_visible_messages_sent(self, messages: list[dict[str, Any]]) -> None:
        self.state.sent_message_count = len(messages)

    def _chat_structured(
        self,
        messages: list[dict[str, Any]],
        response_schema: type[BaseModel],
    ) -> ProviderTurnResult:
        parsed, usage = completion_with_structured_output(
            model=self.model,
            messages=messages,
            response_schema=response_schema,
        )
        self._mark_visible_messages_sent(messages)
        return ProviderTurnResult(
            action=parsed,
            assistant_record=parsed.model_dump_json(),
            usage_events=[usage],
            metadata=self.state_metadata(),
        )

    def _openai_input_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> str | list[dict[str, Any]]:
        new_messages = self._new_visible_messages(messages)
        source = new_messages if self.state.previous_response_id else messages
        converted = [
            {"role": m["role"], "content": str(m.get("content", ""))}
            for m in source
            if m.get("role") != "system"
        ]
        if not converted:
            return ""
        return converted

    def _openai_common_kwargs(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._openai_model(),
            "input": self._openai_input_messages(messages),
            "store": self.openai_store,
        }
        if not self.openai_store and self.state.encrypted_reasoning_items:
            existing_input = kwargs["input"]
            if not isinstance(existing_input, list):
                existing_input = [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": str(existing_input)}
                        ],
                    }
                ]
            kwargs["input"] = [*self.state.encrypted_reasoning_items, *existing_input]
            self.state.hidden_state_used = True
        if self.system_prompt:
            kwargs["instructions"] = self.system_prompt
        if self.openai_store and self.state.previous_response_id:
            kwargs["previous_response_id"] = self.state.previous_response_id
            self.state.hidden_state_used = True
        if not self.openai_store or self.openai_include_encrypted_reasoning:
            kwargs["include"] = ["reasoning.encrypted_content"]
        return kwargs

    def _openai_model(self) -> str:
        return self.model.removeprefix("openai/")

    def _openai_structured(
        self,
        messages: list[dict[str, Any]],
        response_schema: type[BaseModel],
    ) -> ProviderTurnResult:
        schema = _openai_strict_json_schema(response_schema.model_json_schema())

        def request_fn() -> Any:
            kwargs = self._openai_common_kwargs(messages)
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": response_schema.__name__,
                    "schema": schema,
                    "strict": True,
                }
            }
            try:
                return self._litellm_responses_with_retry(**kwargs)
            except Exception as exc:
                refusal = as_provider_refusal(
                    exc,
                    provider="openai",
                    model=self.model,
                    messages=messages,
                )
                if refusal is not None:
                    raise refusal from exc
                raise

        return self._call_with_parse_retry(
            messages=messages,
            response_schema=response_schema,
            provider_label="openai",
            request_fn=request_fn,
            text_fn=_response_output_text,
            capture_fn=self._capture_openai_response,
        )

    def _litellm_responses_with_retry(self, **kwargs: Any) -> Any:
        transient_max = 5
        policy_max = self.content_policy_max_retries
        transient_attempt = 0
        policy_attempt = 0
        while True:
            try:
                return litellm.responses(**kwargs)
            except Exception as exc:
                if _is_content_policy_refusal(exc):
                    if policy_attempt >= policy_max:
                        raise
                    wait = self.content_policy_retry_delay
                    logger.warning(
                        "OpenAI Responses content-policy refusal; retrying "
                        "in %.1fs (attempt %d/%d): %s",
                        wait,
                        policy_attempt + 1,
                        policy_max + 1,
                        str(exc)[:300],
                    )
                    policy_attempt += 1
                    if wait:
                        time.sleep(wait)
                    continue
                if (
                    not _is_retryable_responses_exception(exc)
                    or transient_attempt >= transient_max
                ):
                    raise
                wait = _responses_retry_delay(exc, transient_attempt)
                logger.warning(
                    "Transient OpenAI Responses error; retrying in %.1fs "
                    "(attempt %d/%d): %s",
                    wait,
                    transient_attempt + 1,
                    transient_max + 1,
                    str(exc)[:300],
                )
                transient_attempt += 1
                time.sleep(wait)

    def _capture_openai_response(self, response: Any) -> None:
        response_id = _object_value(response, "id")
        if isinstance(response_id, str):
            self.state.previous_response_id = response_id
            self.state.response_ids.append(response_id)
        encrypted_items = [
            _as_plain_dict(item)
            for item in (_object_value(response, "output") or [])
            if _object_value(item, "type") == "reasoning"
            and _object_value(item, "encrypted_content")
        ]
        if encrypted_items:
            self.state.encrypted_reasoning_items = encrypted_items

    def _anthropic_structured(
        self,
        messages: list[dict[str, Any]],
        response_schema: type[BaseModel],
    ) -> ProviderTurnResult:
        self._append_anthropic_user_messages(messages)
        prompt_messages = list(self.state.native_messages)
        if prompt_messages and prompt_messages[-1].get("role") == "user":
            prompt_messages[-1] = {
                **prompt_messages[-1],
                "content": _append_text_to_anthropic_content(
                    prompt_messages[-1].get("content", []),
                    schema_to_prompt_instruction(response_schema),
                ),
            }

        def request_fn() -> Any:
            payload = self._anthropic_payload(prompt_messages)
            return self._anthropic_request(payload)

        return self._call_with_parse_retry(
            messages=messages,
            response_schema=response_schema,
            provider_label="anthropic",
            request_fn=request_fn,
            text_fn=_anthropic_text,
            capture_fn=self._capture_anthropic_assistant,
        )

    def _call_with_parse_retry(
        self,
        *,
        messages: list[dict[str, Any]],
        response_schema: type[BaseModel],
        provider_label: str,
        request_fn: Callable[[], Any],
        text_fn: Callable[[Any], str],
        capture_fn: Callable[[Any], None],
    ) -> ProviderTurnResult:
        """Make a structured request with bounded re-sample on parse failure.

        Native provider paths (Anthropic Messages, OpenAI Responses) own their
        own HTTP-level retry inside ``request_fn``; this loop sits one layer
        above and re-samples when the model returns a 200 OK whose body fails
        to satisfy ``response_schema``. The chat-completion path already does
        the equivalent via ``completion_with_structured_output``.

        State capture (``capture_fn``) is deferred until parse succeeds so a
        failed attempt does not poison ``state.previous_response_id`` or
        append a malformed assistant message that the next attempt would
        re-send.
        """
        last_parse_error: Exception | None = None
        for attempt in range(_PARSE_RETRY_MAX_ATTEMPTS):
            response = request_fn()
            text = text_fn(response)
            try:
                parsed = validate_with_coercion(text, response_schema)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_parse_error = exc
                if attempt + 1 >= _PARSE_RETRY_MAX_ATTEMPTS:
                    raise
                wait = _PARSE_RETRY_BASE_SECONDS * (2**attempt)
                logger.warning(
                    "%s structured-output parse failure; re-sampling in "
                    "%.1fs (attempt %d/%d): %s",
                    provider_label,
                    wait,
                    attempt + 1,
                    _PARSE_RETRY_MAX_ATTEMPTS,
                    str(exc)[:300],
                )
                if wait:
                    time.sleep(wait)
                continue
            capture_fn(response)
            self._mark_visible_messages_sent(messages)
            return ProviderTurnResult(
                action=parsed,
                assistant_record=parsed.model_dump_json(),
                usage_events=[
                    self._usage_from_provider_response(response, provider_label)
                ],
                metadata=self.state_metadata(),
            )
        # Defensive: the loop above always returns or raises within the final
        # iteration. This line is unreachable but satisfies type checkers.
        raise RuntimeError("parse-retry loop exhausted unexpectedly") from (
            last_parse_error
        )

    def _append_anthropic_user_messages(self, messages: list[dict[str, Any]]) -> None:
        new_messages = self._new_visible_messages(messages)
        for message in new_messages:
            role = message.get("role")
            if role == "system":
                continue
            if role not in {"user", "assistant"}:
                role = "user"
            self.state.native_messages.append(
                {
                    "role": role,
                    "content": [
                        {"type": "text", "text": str(message.get("content", ""))}
                    ],
                }
            )

    def _anthropic_model(self) -> str:
        return (
            self.model.removeprefix("anthropic/")
            .removeprefix("vertex_ai/")
            .removeprefix("vertex/")
        )

    def _anthropic_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._anthropic_model(),
            "max_tokens": self.anthropic_max_tokens,
            "messages": messages,
        }
        if self.system_prompt:
            payload["system"] = [
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        # Cache the conversation prefix up to the previous turn. The current
        # user message (messages[-1]) changes every turn; messages[-2] is the
        # last completed assistant response and is stable — marking it caches
        # everything before it on subsequent turns.
        if len(messages) >= 2:
            penultimate = deepcopy(messages[-2])
            content = penultimate.get("content")
            if isinstance(content, list) and content:
                content[-1]["cache_control"] = {"type": "ephemeral"}
                messages[-2] = penultimate  # safe: messages is already a list copy
        return payload

    def _anthropic_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        auth_provider = resolve_anthropic_auth_provider(
            self.model, self.anthropic_auth_provider
        )
        if auth_provider == "vertex":
            return self._anthropic_vertex_request(payload)
        return self._anthropic_direct_request(payload)

    def _anthropic_vertex_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        from anthropic import AnthropicVertex

        project_id, region = resolve_vertex_config(
            project_id=self.anthropic_project_id,
            region=self.anthropic_region,
        )

        client = AnthropicVertex(region=region, project_id=project_id)
        model_name = payload.get("model", self._anthropic_model())

        kwargs: dict[str, Any] = {
            "model": model_name,
            "max_tokens": payload.get("max_tokens", self.anthropic_max_tokens),
            "messages": payload["messages"],
        }
        if "system" in payload:
            kwargs["system"] = payload["system"]

        for attempt in range(_ANTHROPIC_MAX_RETRIES + 1):
            try:
                if self.anthropic_stream:
                    with client.messages.stream(**kwargs) as stream:
                        response_msg = stream.get_final_message()
                        return response_msg.model_dump()
                else:
                    response_msg = client.messages.create(**kwargs)
                    return response_msg.model_dump()
            except Exception as exc:
                status = extract_http_status(exc)
                if (
                    status is not None
                    and status in _ANTHROPIC_RETRYABLE_HTTP_STATUSES
                    and attempt < _ANTHROPIC_MAX_RETRIES
                ):
                    wait = _anthropic_retry_delay(exc, attempt)
                    logger.warning(
                        "Transient Anthropic Vertex HTTP %s; retrying in %.1fs "
                        "(attempt %d/%d): %s",
                        status,
                        wait,
                        attempt + 1,
                        _ANTHROPIC_MAX_RETRIES + 1,
                        str(exc)[:200],
                    )
                    time.sleep(wait)
                    continue

                if (
                    isinstance(exc, (ConnectionError, OSError))
                    and attempt < _ANTHROPIC_MAX_RETRIES
                ):
                    wait = _anthropic_retry_delay(exc, attempt)
                    logger.warning(
                        "Transient Anthropic Vertex network error; retrying in %.1fs "
                        "(attempt %d/%d): %s",
                        wait,
                        attempt + 1,
                        _ANTHROPIC_MAX_RETRIES + 1,
                        str(exc)[:300],
                    )
                    time.sleep(wait)
                    continue

                refusal = as_provider_refusal(
                    exc,
                    provider="anthropic",
                    model=self.model,
                )
                if refusal is not None:
                    raise refusal from exc

                if attempt >= _ANTHROPIC_MAX_RETRIES:
                    raise RuntimeError(
                        f"Anthropic Vertex request failed: {exc}"
                    ) from exc
                raise
        raise RuntimeError("Anthropic Vertex retry loop exhausted unexpectedly")

    def _anthropic_direct_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY must be set for direct Anthropic API, or configure Vertex AI "
                "auth (e.g. gcloud auth application-default login, or set ANTHROPIC_VERTEX_PROJECT_ID / CLOUD_ML_REGION)."
            )
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        for attempt in range(_ANTHROPIC_MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if (
                    exc.code in _ANTHROPIC_RETRYABLE_HTTP_STATUSES
                    and attempt < _ANTHROPIC_MAX_RETRIES
                ):
                    wait = _anthropic_retry_delay(exc, attempt)
                    logger.warning(
                        "Transient Anthropic HTTP %d; retrying in %.1fs "
                        "(attempt %d/%d): %s",
                        exc.code,
                        wait,
                        attempt + 1,
                        _ANTHROPIC_MAX_RETRIES + 1,
                        detail[:200],
                    )
                    time.sleep(wait)
                    continue
                refusal = as_provider_refusal(
                    RuntimeError(detail),
                    provider="anthropic",
                    model=self.model,
                )
                if refusal is not None:
                    raise refusal from exc
                raise RuntimeError(f"Anthropic request failed: {detail}") from exc
            except OSError as exc:
                # Catches ssl.SSLError (TLS handshake / record-mac flaps),
                # urllib.error.URLError without an HTTP status (DNS, connection
                # reset, broken pipe), socket.timeout, ConnectionError, and
                # other transient socket-level failures.  All of these are
                # subclasses of OSError in Python 3.
                if attempt == _ANTHROPIC_MAX_RETRIES:
                    raise
                wait = _anthropic_retry_delay(exc, attempt)
                logger.warning(
                    "Transient Anthropic network error; retrying in %.1fs "
                    "(attempt %d/%d): %s",
                    wait,
                    attempt + 1,
                    _ANTHROPIC_MAX_RETRIES + 1,
                    str(exc)[:300],
                )
                time.sleep(wait)
        # The loop always returns or raises within the final iteration; this
        # line is unreachable but keeps mypy / pyright happy about return type.
        raise RuntimeError("Anthropic retry loop exhausted unexpectedly")

    def _capture_anthropic_assistant(self, response: dict[str, Any]) -> None:
        response_id = response.get("id")
        if isinstance(response_id, str):
            self.state.previous_response_id = response_id
            self.state.response_ids.append(response_id)
        content = response.get("content") or []
        if any(
            block.get("type") == "thinking"
            for block in content
            if isinstance(block, dict)
        ):
            self.state.hidden_state_used = True
        self.state.native_messages.append({"role": "assistant", "content": content})

    def _usage_from_provider_response(
        self,
        response: Any,
        provider: str,
    ) -> UsageEvent:
        usage = _object_value(response, "usage") or {}
        input_tokens = _usage_int(usage, "input_tokens", "prompt_tokens")
        output_tokens = _usage_int(usage, "output_tokens", "completion_tokens")
        total_tokens = _usage_int(usage, "total_tokens")
        if total_tokens is None and (
            input_tokens is not None or output_tokens is not None
        ):
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
        cached_input_tokens = _nested_usage_int(
            usage,
            ("input_tokens_details", "cached_tokens"),
            ("prompt_tokens_details", "cached_tokens"),
            ("cache_read_input_tokens",),
        )
        cache_creation_input_tokens = _nested_usage_int(
            usage,
            ("cache_creation_input_tokens",),
        )
        # Anthropic's native API returns input_tokens as only the newly-processed
        # tokens (cache reads are excluded). litellm's cost functions expect
        # prompt_tokens to be the full context total, so add the cache tokens back.
        # We also skip the response-based litellm.completion_cost path because it
        # can't correctly re-parse Anthropic's native usage fields, and use
        # litellm.cost_per_token with the normalised total instead.
        if provider == "anthropic":
            input_tokens = (
                (input_tokens or 0)
                + (cached_input_tokens or 0)
                + (cache_creation_input_tokens or 0)
            )
        metadata = {
            "provider_state": self.state_metadata(),
            "raw_usage": _as_plain_dict(usage),
            "reasoning_tokens": _nested_usage_int(
                usage,
                ("output_tokens_details", "reasoning_tokens"),
                ("completion_tokens_details", "reasoning_tokens"),
            ),
            "cached_input_tokens": cached_input_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "response_id": _object_value(response, "id"),
        }
        return build_usage_event(
            model=self.model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            call_type="completion",
            metadata=metadata,
            response=None if provider == "anthropic" else response,
            reasoning_tokens=metadata["reasoning_tokens"],
            cached_input_tokens=metadata["cached_input_tokens"],
            cache_creation_input_tokens=metadata["cache_creation_input_tokens"],
            response_id=metadata["response_id"],
            raw_usage=metadata["raw_usage"],
        )


def _object_value(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _as_plain_dict(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_as_plain_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _as_plain_dict(value) for key, value in obj.items()}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {
            key: _as_plain_dict(value)
            for key, value in vars(obj).items()
            if not key.startswith("_")
        }
    return str(obj)


def _response_output_text(response: Any) -> str:
    output_text = _object_value(response, "output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    parts: list[str] = []
    for item in _object_value(response, "output") or []:
        if _object_value(item, "type") != "message":
            continue
        for content in _object_value(item, "content") or []:
            text = _object_value(content, "text")
            if isinstance(text, str):
                parts.append(text)
    if parts:
        return "\n".join(parts)
    content = extract_content(response)
    if content is None:
        raise RuntimeError("Provider response did not include text output")
    return content


def _anthropic_text(response: dict[str, Any]) -> str:
    parts = [
        block.get("text", "")
        for block in response.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    if not parts:
        raise RuntimeError("Anthropic response did not include text content")
    return "\n".join(parts)


def _append_text_to_anthropic_content(
    content: list[dict[str, Any]],
    extra_text: str,
) -> list[dict[str, Any]]:
    copied = [dict(block) for block in content]
    for block in reversed(copied):
        if block.get("type") == "text":
            block["text"] = str(block.get("text", "")) + extra_text
            return copied
    copied.append({"type": "text", "text": extra_text})
    return copied


def _openai_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON schema accepted by OpenAI Responses strict mode.

    OpenAI's strict structured-output path requires every object schema to set
    ``additionalProperties: false`` and every property to be listed in
    ``required``. Pydantic leaves optional-default fields out of ``required``;
    the nullable type in the field schema still communicates optionality to the
    model while satisfying OpenAI's validator.
    """

    normalized = deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return

        node.pop("default", None)
        properties = node.get("properties")
        if isinstance(properties, dict):
            node["additionalProperties"] = False
            node["required"] = list(properties.keys())
            for child in properties.values():
                visit(child)

        for key in ("$defs", "definitions"):
            defs = node.get(key)
            if isinstance(defs, dict):
                for child in defs.values():
                    visit(child)

        for key in ("items", "anyOf", "oneOf", "allOf", "not"):
            if key in node:
                visit(node[key])

    visit(normalized)
    return normalized


def _usage_int(usage: Any, *names: str) -> int | None:
    for name in names:
        value = _object_value(usage, name)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _is_retryable_responses_exception(exc: Exception) -> bool:
    retryable_types = (
        "InternalServerError",
        "APIConnectionError",
        "Timeout",
        "ServiceUnavailableError",
        "RateLimitError",
        "BadGatewayError",
    )
    if exc.__class__.__name__ in retryable_types:
        return True
    litellm_types = tuple(
        cls
        for name in retryable_types
        for cls in [getattr(litellm, name, None)]
        if isinstance(cls, type)
    )
    if litellm_types and isinstance(exc, litellm_types):
        return True
    # Catches transient HTTP errors that aren't typed (e.g. Cloudflare 520-526
    # arriving from Anthropic via litellm.responses) by inspecting the
    # exception's status code or its message.
    status = extract_http_status(exc)
    if status is not None and status in RETRYABLE_HTTP_STATUSES:
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "badgateway",
            "bad gateway",
            "<!doctype html",
            "<html",
            "overloaded",
            "temporarily unavailable",
        )
    )


def _responses_retry_delay(exc: Exception, attempt: int) -> float:
    message = str(exc).lower()
    base = 15.0 if "rate" in message or "overloaded" in message else 2.0
    return base * (2**attempt)


# The Anthropic-native HTTP retry loop and the litellm/OpenAI Responses
# loop both consult ``RETRYABLE_HTTP_STATUSES`` from structured_output so
# the three retry surfaces stay in sync from a single source of truth.
_ANTHROPIC_RETRYABLE_HTTP_STATUSES: frozenset[int] = RETRYABLE_HTTP_STATUSES
_ANTHROPIC_MAX_RETRIES = 5

# Bounded re-sample for native paths when the model returns a 200 OK whose
# body fails to satisfy the response schema (stochastic JSON-quality slip
# from the provider). Three retries on top of the initial attempt; short
# backoff since these are not rate-limit conditions.
_PARSE_RETRY_MAX_ATTEMPTS = 4
_PARSE_RETRY_BASE_SECONDS = 0.5
_ANTHROPIC_BACKOFF_BASE = 2.0
_ANTHROPIC_RATE_LIMIT_BACKOFF_BASE = 15.0


def _anthropic_retry_delay(exc: BaseException, attempt: int) -> float:
    """Exponential backoff for the Anthropic native HTTP retry loop.

    Uses a longer base for rate-limit and overload signals so we don't
    immediately re-hit a saturated endpoint.
    """
    if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
        base = _ANTHROPIC_RATE_LIMIT_BACKOFF_BASE
    else:
        message = str(exc).lower()
        if "overloaded" in message or "rate" in message:
            base = _ANTHROPIC_RATE_LIMIT_BACKOFF_BASE
        else:
            base = _ANTHROPIC_BACKOFF_BASE
    return base * (2**attempt)


def _resolve_model_max_output_tokens(
    model: str,
    configured_max_output_tokens: int | None,
) -> int:
    if configured_max_output_tokens is not None:
        return configured_max_output_tokens
    for candidate in (
        model,
        model.removeprefix("vertex_ai/")
        .removeprefix("vertex/")
        .removeprefix("anthropic/"),
    ):
        try:
            info = litellm.get_model_info(candidate)
            max_output_tokens = info.get("max_output_tokens")
            if max_output_tokens:
                return int(max_output_tokens)
        except Exception:
            pass
    # Anthropic requires max_tokens. Keep a conservative fallback for custom
    # model ids that LiteLLM cannot describe.
    return 4096


def _nested_usage_int(usage: Any, *paths: tuple[str, ...]) -> int | None:
    for path in paths:
        current = usage
        for part in path:
            current = _object_value(current, part)
            if current is None:
                break
        if current is None:
            continue
        try:
            return int(current)
        except (TypeError, ValueError):
            continue
    return None
