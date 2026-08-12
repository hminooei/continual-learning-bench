"""
ICL system with notepad for tracking observations across interactions.

The model can maintain a notepad to jot down important observations, patterns,
and thoughts that persist across interactions within a task.
"""

import json
from typing import Any, ClassVar, Optional

import litellm
from pydantic import BaseModel, Field, create_model, model_validator

from ...errors import ProviderRefusalError
from ...interface import (
    ContinualLearningSystem,
    Observation,
    Query,
    Response,
    observation_marks_instance_complete,
)
from ...registry import register_system
from ..utils import (
    ProviderTurnClient,
    TokenBudgetTracker,
    completion_with_structured_output,
    count_tokens,
    resolve_context_token_limit,
)


class _FlatNotepadSchemaBase(BaseModel):
    """Base for `_build_flat_schema` outputs that unwraps wrapper responses."""

    _task_field_names: ClassVar[frozenset[str]] = frozenset()

    @model_validator(mode="before")
    @classmethod
    def _unwrap_envelope(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        task_fields = cls._task_field_names
        if not task_fields:
            return data

        if any(k in data for k in task_fields):
            return data

        nested = _find_nested_dict_with_keys(data, task_fields)
        if nested is None:
            return data

        result = {k: v for k, v in nested.items() if k in cls.model_fields}
        if "notepad_update" not in result and "notepad_update" in data:
            result["notepad_update"] = data["notepad_update"]
        return result


def _find_nested_dict_with_keys(
    obj: Any,
    expected: frozenset[str] | set[str],
) -> dict[str, Any] | None:
    """Depth-first search for the inner dict carrying any expected key."""
    if not isinstance(obj, dict):
        return None
    if any(k in obj for k in expected):
        return obj
    for v in obj.values():
        found = _find_nested_dict_with_keys(v, expected)
        if found is not None:
            return found
    return None


def _build_flat_schema(task_schema: type[BaseModel]) -> type[BaseModel]:
    """
    Merge task schema fields with a `notepad_update` field into one flat model.

    All fields live at the top level for normal structured responses. The
    returned model also accepts provider wrapper responses where the task fields
    are nested and only ``notepad_update`` sits at the top.
    """
    if "notepad_update" in task_schema.model_fields:
        raise ValueError(
            f"Task schema {task_schema.__name__} already defines 'notepad_update'; "
            "this conflicts with the icl_notepad system's reserved field."
        )
    field_defs: dict = {}
    for name, field_info in task_schema.model_fields.items():
        field_defs[name] = (field_info.annotation, field_info)
    field_defs["notepad_update"] = (
        Optional[str],
        Field(
            default=None,
            description=(
                "Optional: Update your notepad with new observations, patterns, "
                "or thoughts. If provided, this completely replaces the current "
                "notepad content. Leave as null to keep notepad unchanged."
            ),
        ),
    )
    wrapped = create_model(
        f"{task_schema.__name__}WithNotepad",
        __base__=_FlatNotepadSchemaBase,
        **field_defs,
    )
    wrapped._task_field_names = frozenset(task_schema.model_fields.keys())
    return wrapped


@register_system("icl_notepad")
class ICLNotepadSystem(ContinualLearningSystem):
    """
    ICL system with a persistent notepad for tracking observations.

    The system maintains both a conversation history and a notepad that the model
    can freely update to track important information across interactions.
    """

    def __init__(
        self,
        model: str = "gpt-5",
        max_tokens: int | None = None,
        system_prompt: str = "",
        name: str = "icl_notepad",
        reserve_tokens: int = 500,
        clear_context_between_instances: bool = True,
        clear_notepad_between_instances: bool = False,
        include_notepad: bool = True,
        stateless: bool = False,
        provider_mode: str = "auto",
        openai_store: bool = True,
        openai_include_encrypted_reasoning: bool = False,
        anthropic_max_tokens: int | None = None,
        anthropic_project_id: str | None = None,
        anthropic_region: str | None = None,
        anthropic_auth_provider: str = "auto",
        anthropic_stream: bool = False,
    ):
        """
        Initialize ICL notepad system.

        Args:
            model: LiteLLM model identifier
            max_tokens: Maximum context window size before truncation. If omitted,
                uses the model's known input/context limit with a safety buffer.
            system_prompt: System prompt for the LLM
            name: System identifier
            reserve_tokens: Tokens to reserve for system prompt + response
            clear_context_between_instances: If True, clear message history at
                instance boundaries (notepad is preserved).
            clear_notepad_between_instances: If True, clear notepad at instance
                boundaries. Combined with clear_context_between_instances, this
                makes the system fully independent across instances.
            include_notepad: If True, add a notepad_update field to the response
                schema so the model can update its notepad. Tasks can override
                per-query via query.metadata["include_notepad"]. Defaults to True.
            stateless: If True, enables fully stateless mode: clears context,
                clears notepad, and suppresses cross-instance feedback.
                Equivalent to setting clear_context_between_instances=True and
                clear_notepad_between_instances=True.
            provider_mode: "auto" routes known reasoning providers through native
                stateful APIs; "litellm_chat" preserves the old chat-completion path.
            openai_store: If False, OpenAI Responses requests use stateless encrypted
                reasoning items instead of server-side previous_response_id state.
            openai_include_encrypted_reasoning: Request encrypted reasoning items
                even when OpenAI server-side state is enabled.
            anthropic_max_tokens: Native Anthropic max_tokens response budget.
                If omitted, uses the model's known max output token limit.
            anthropic_project_id: Google Cloud project ID for Vertex AI Anthropic calls.
            anthropic_region: Google Cloud region for Vertex AI Anthropic calls (default: "global").
            anthropic_auth_provider: "auto" (default), "direct" (Anthropic API key), or "vertex" (Vertex AI).
            anthropic_stream: If True, use streaming Anthropic SDK API.
        """
        self.stateless = stateless
        if stateless:
            clear_context_between_instances = True
            clear_notepad_between_instances = True
        self._name = name
        self.model = model
        self.max_tokens = resolve_context_token_limit(model, max_tokens)
        self.reserve_tokens = reserve_tokens
        self.clear_context_between_instances = clear_context_between_instances
        self.clear_notepad_between_instances = clear_notepad_between_instances
        self.include_notepad = include_notepad
        self.system_prompt = system_prompt
        self.provider_mode = provider_mode
        self._provider_client = ProviderTurnClient(
            model=model,
            system_prompt=system_prompt,
            provider_mode=provider_mode,  # type: ignore[arg-type]
            openai_store=openai_store,
            openai_include_encrypted_reasoning=openai_include_encrypted_reasoning,
            anthropic_max_tokens=anthropic_max_tokens,
            anthropic_project_id=anthropic_project_id,
            anthropic_region=anthropic_region,
            anthropic_auth_provider=anthropic_auth_provider,  # type: ignore[arg-type]
            anthropic_stream=anthropic_stream,
        )

        # Context storage
        self.messages: list[dict[str, str]] = []
        self._token_budget = TokenBudgetTracker()

        # Notepad storage
        self.notepad: str = ""

        # Truncation tracking
        self.truncation_count: int = 0
        self.has_truncated_flag: bool = False

        # Interaction tracking
        self.interaction_count: int = 0
        self._at_instance_boundary: bool = True

        # Deferred feedback: stashed here when context is cleared at an
        # instance boundary so it can be injected into the next prompt.
        self._pending_feedback: str | None = None

    def respond(self, query: Query) -> Response:
        """
        Generate response using LLM with full context and notepad.

        Process:
        1. Prepend turn feedback and notepad to query
        2. Format query as user message
        3. Add to context and handle truncation
        4. Call LLM with structured output
        5. Update notepad if requested
        6. Store response in context
        7. Return Response object

        Args:
            query: Current query from the task

        Returns:
            Response with action and metadata
        """
        instance_boundary = self._at_instance_boundary

        # Build query content with notepad and feedback
        query_parts = []

        # Inject deferred feedback from the previous instance.
        if self._pending_feedback and instance_boundary:
            query_parts.append(
                f"FEEDBACK FROM PREVIOUS INSTANCE:\n{self._pending_feedback}"
            )
            self._pending_feedback = None

        # Include the notepad only at the start of a new instance;
        # within an instance it is already visible in the first message.
        if self.notepad and instance_boundary:
            query_parts.append(
                f"=== YOUR NOTEPAD ===\n{self.notepad}\n==================="
            )

        # Add main query
        if query.prompt:
            query_parts.append(query.prompt)

        query_content = "\n\n".join(query_parts) if query_parts else "(no content)"

        self.interaction_count += 1
        self._at_instance_boundary = False

        # Add user message to context
        self._add_message("user", query_content)

        include_notepad = (query.metadata or {}).get(
            "include_notepad", self.include_notepad
        )

        try:
            task_schema = query.response_schema
            if include_notepad:
                llm_schema = _build_flat_schema(task_schema)
            else:
                llm_schema = task_schema

            self._truncate_context(
                prefix_messages=self._system_messages(),
                extra_tokens=self._response_schema_tokens(llm_schema),
            )
            llm_messages = [*self._system_messages(), *self.messages]
            if self._provider_client.state.provider == "litellm":
                parsed, usage_event = completion_with_structured_output(
                    model=self.model,
                    messages=llm_messages,
                    response_schema=llm_schema,
                )
                usage_events = [usage_event]
            else:
                provider_result = self._provider_client.respond_structured(
                    messages=llm_messages,
                    response_schema=llm_schema,
                )
                parsed = provider_result.action
                usage_events = provider_result.usage_events
            for usage_event in usage_events:
                self._note_prompt_token_usage(usage_event.input_tokens)
                self.record_usage_event(usage_event)

            if include_notepad and getattr(parsed, "notepad_update", None) is not None:
                self.notepad = parsed.notepad_update

            if include_notepad:
                task_data = {k: getattr(parsed, k) for k in task_schema.model_fields}
                action = task_schema(**task_data)
            else:
                action = parsed
            assistant_record = parsed.model_dump_json()
        except ProviderRefusalError:
            raise
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}") from e

        self._add_message("assistant", assistant_record)
        self._sync_provider_visible_count()

        return Response(
            action=action,
            metadata={
                "interaction_count": self.interaction_count,
                "system_type": "icl_notepad",
                "model": self.model,
                "context_tokens": self._count_message_tokens(self.messages),
                "has_truncated": self.has_truncated_flag,
                "truncation_count": self.truncation_count,
                "notepad_length": len(self.notepad),
                "notepad_content": self.notepad,
                "provider_state": self._provider_client.state_metadata(),
            },
        )

    def reset(self) -> None:
        """
        Reset for new task instance.

        Clears all messages, notepad, pending observations, and resets counters.
        """
        self.messages = []
        self.notepad = ""
        self._token_budget.reset()
        self.truncation_count = 0
        self.has_truncated_flag = False
        self.interaction_count = 0
        self._at_instance_boundary = True
        self._pending_feedback = None
        self._provider_client.reset()

    def observe(
        self, observation: Observation, next_query: Query | None = None
    ) -> None:
        """Persist feedback and handle instance-boundary clearing semantics."""
        instance_complete = observation_marks_instance_complete(observation)
        content = observation.content.strip()

        if content and not (self.stateless and instance_complete):
            if instance_complete and self.clear_context_between_instances:
                # Stash feedback so it survives the context clear and can
                # be injected into the next turn's prompt.
                self._pending_feedback = content
            else:
                self._add_message("user", f"FEEDBACK: {content}")

        if instance_complete:
            if self.clear_context_between_instances:
                self.messages = []
                self._provider_client.reset()
            if self.clear_notepad_between_instances:
                self.notepad = ""
            self._at_instance_boundary = True
        else:
            self._at_instance_boundary = False

    @property
    def name(self) -> str:
        """Return system identifier."""
        return self._name

    def get_run_artifacts(self) -> dict[str, Any]:
        """Export final notepad and context state for analysis."""
        return {
            "artifact_type": "icl_notepad",
            "notepad": self.notepad,
            "notepad_length": len(self.notepad),
            "messages": list(self.messages),
            "message_count": len(self.messages),
            "interaction_count": self.interaction_count,
            "model": self.model,
            "context_tokens": self._count_message_tokens(self.messages),
            "has_truncated": self.has_truncated_flag,
            "truncation_count": self.truncation_count,
            "clear_context_between_instances": self.clear_context_between_instances,
            "clear_notepad_between_instances": self.clear_notepad_between_instances,
            "include_notepad": self.include_notepad,
            "stateless": self.stateless,
            "provider_state": self._provider_client.state_metadata(),
        }

    def _estimate_message_tokens(self, messages: list[dict[str, str]]) -> int:
        if not messages:
            return 0
        try:
            return litellm.token_counter(model=self.model, messages=messages)
        except Exception:
            total = 0
            for message in messages:
                total += 4
                total += count_tokens(self.model, message["role"])
                total += count_tokens(self.model, message["content"])
            total += 2
            return total

    def _system_messages(self) -> list[dict[str, str]]:
        if not self.system_prompt:
            return []
        return [{"role": "system", "content": self.system_prompt}]

    def _response_schema_tokens(self, schema: type[BaseModel] | None) -> int:
        if schema is None:
            return 0
        return count_tokens(self.model, json.dumps(schema.model_json_schema()))

    def _count_message_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count tokens using the latest API-reported prompt usage when available."""
        return self._token_budget.count(messages, self._estimate_message_tokens)

    def _note_prompt_token_usage(self, input_tokens: int | None) -> None:
        """Anchor future truncation decisions on the last real prompt usage."""
        self._token_budget.note_usage(
            messages=self.messages,
            input_tokens=input_tokens,
            estimate_fn=self._estimate_message_tokens,
        )

    def _sync_provider_visible_count(self) -> None:
        self._provider_client.state.sent_message_count = len(
            [*self._system_messages(), *self.messages]
        )

    def _add_message(self, role: str, content: str) -> None:
        """
        Add message to context and truncate if needed.

        Args:
            role: Message role ('user', 'assistant', or 'system')
            content: Message content
        """
        self.messages.append({"role": role, "content": content})
        self._truncate_context()

    def _truncate_context(
        self,
        *,
        prefix_messages: list[dict[str, str]] | None = None,
        extra_tokens: int = 0,
    ) -> None:
        """
        Remove oldest messages until under token limit.

        Uses FIFO strategy: removes messages from the start of the list
        until the total tokens (including reserve) fit within max_tokens.
        Always preserves at least the most recent message so the LLM
        call never receives an empty conversation.
        """
        while len(self.messages) > 1:
            current_tokens = self._count_message_tokens(
                [*(prefix_messages or []), *self.messages]
            ) + max(0, extra_tokens)
            available = self.max_tokens - self.reserve_tokens

            if current_tokens <= available:
                break

            self.messages.pop(0)
            self.truncation_count += 1
            self.has_truncated_flag = True
