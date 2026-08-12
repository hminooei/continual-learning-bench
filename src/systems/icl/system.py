"""
In-Context Learning (ICL) baseline system for continual learning benchmark.

This system accumulates all interactions in a linear context window with
token-based truncation from the start when limits are reached. It provides
a simple baseline for comparing continual learning approaches.
"""

import json
from typing import Any

import litellm
from pydantic import BaseModel

from ...errors import ProviderRefusalError
from ...interface import (
    ContinualLearningSystem,
    Observation,
    Query,
    Response,
)
from ...registry import register_system
from ..utils import (
    ProviderTurnClient,
    TokenBudgetTracker,
    completion_with_structured_output,
    count_tokens,
    resolve_context_token_limit,
)


@register_system("icl")
class ICLSystem(ContinualLearningSystem):
    """
    ICL baseline system that stores interactions in a linear context window.

    The system maintains a conversation history as a list of messages and
    truncates from the start (FIFO) when the token limit is exceeded.
    Observations are stored and prepended to the next query to ensure the
    LLM sees feedback in context.
    """

    def __init__(
        self,
        model: str = "gpt-5",
        max_tokens: int | None = None,
        system_prompt: str = "",
        name: str = "icl_baseline",
        reserve_tokens: int = 500,
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
        Initialize ICL baseline system.

        Args:
            model: LiteLLM model identifier
            max_tokens: Maximum context window size before truncation. If omitted,
                uses the model's known input/context limit with a safety buffer.
            system_prompt: System prompt for the LLM (default: empty string)
            name: System identifier
            reserve_tokens: Tokens to reserve for system prompt + response
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
        self._name = name
        self.model = model
        self.max_tokens = resolve_context_token_limit(model, max_tokens)
        self.reserve_tokens = reserve_tokens
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

        # Truncation tracking
        self.truncation_count: int = 0
        self.has_truncated_flag: bool = False

        # Interaction tracking
        self.interaction_count: int = 0

    def respond(self, query: Query) -> Response:
        """
        Generate response using LLM with full context.

        Process:
        1. Check if this turn includes feedback, prepend it to query
        2. Format query as user message (with feedback if present)
        3. Add to context and handle truncation
        4. Call LLM via LiteLLM
        5. Store response in context
        6. Return Response object

        Args:
            query: Current query from the task

        Returns:
            Response with action and metadata
        """
        query_content = query.prompt if query.prompt else "(no content)"

        self.interaction_count += 1

        # Add user message to context
        self._add_message("user", query_content)

        # Prepare messages for LLM call
        schema_tokens = self._response_schema_tokens(query.response_schema)
        self._truncate_context(
            prefix_messages=self._system_messages(),
            extra_tokens=schema_tokens,
        )
        llm_messages = [*self._system_messages(), *self.messages]

        usage_events = []

        # Call LLM with the active task interface.
        try:
            if self._provider_client.state.provider == "litellm":
                action, usage_event = completion_with_structured_output(
                    model=self.model,
                    messages=llm_messages,
                    response_schema=query.response_schema,
                )
                usage_events = [usage_event]
                assistant_record = action.model_dump_json()
            else:
                provider_result = self._provider_client.respond_structured(
                    messages=llm_messages,
                    response_schema=query.response_schema,
                )
                action = provider_result.action
                usage_events = provider_result.usage_events
                assistant_record = provider_result.assistant_record
        except ProviderRefusalError:
            raise
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}") from e

        for usage_event in usage_events:
            self._note_prompt_token_usage(usage_event.input_tokens)
            self.record_usage_event(usage_event)

        # Add assistant response to context
        self._add_message("assistant", assistant_record)
        self._sync_provider_visible_count()

        return Response(
            action=action,
            metadata={
                "interaction_count": self.interaction_count,
                "system_type": "icl",
                "model": self.model,
                "context_tokens": self._count_message_tokens(self.messages),
                "has_truncated": self.has_truncated_flag,
                "truncation_count": self.truncation_count,
                "provider_state": self._provider_client.state_metadata(),
            },
        )

    def reset(self) -> None:
        """
        Reset for new task instance.

        Clears all messages, pending observations, and resets counters.
        """
        self.messages = []
        self._token_budget.reset()
        self.truncation_count = 0
        self.has_truncated_flag = False
        self.interaction_count = 0
        self._provider_client.reset()

    def observe(
        self, observation: Observation, next_query: Query | None = None
    ) -> None:
        """Persist task feedback as context for future turns."""
        content = observation.content.strip()
        if not content:
            return
        self._add_message("user", f"FEEDBACK: {content}")

    @property
    def name(self) -> str:
        """Return system identifier."""
        return self._name

    def get_run_artifacts(self) -> dict[str, Any]:
        """Export the final ICL context state for analysis."""
        return {
            "artifact_type": "icl",
            "messages": list(self.messages),
            "message_count": len(self.messages),
            "interaction_count": self.interaction_count,
            "model": self.model,
            "context_tokens": self._count_message_tokens(self.messages),
            "has_truncated": self.has_truncated_flag,
            "truncation_count": self.truncation_count,
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

    def _is_anthropic_model(self) -> bool:
        """Check if the current model uses Anthropic's API."""
        model_lower = self.model.lower()
        return "anthropic/" in model_lower or "claude" in model_lower

    def _system_messages(self) -> list[dict]:
        if not self.system_prompt:
            return []
        if self._is_anthropic_model():
            return [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ]
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
        """
        while self.messages:
            current_tokens = self._count_message_tokens(
                [*(prefix_messages or []), *self.messages]
            ) + max(0, extra_tokens)
            available = self.max_tokens - self.reserve_tokens

            if current_tokens <= available:
                break

            # Remove oldest message
            self.messages.pop(0)
            self.truncation_count += 1
            self.has_truncated_flag = True

    def get_truncation_stats(self) -> dict:
        """
        Return truncation statistics.

        Returns:
            Dict with truncation metadata:
                - has_truncated: Whether truncation has occurred
                - truncation_count: Number of messages truncated
                - current_message_count: Current number of messages
                - current_tokens: Current token count
        """
        return {
            "has_truncated": self.has_truncated_flag,
            "truncation_count": self.truncation_count,
            "current_message_count": len(self.messages),
            "current_tokens": self._count_message_tokens(self.messages),
        }
