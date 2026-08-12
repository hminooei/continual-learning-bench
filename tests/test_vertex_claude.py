"""Tests for Vertex AI Anthropic Claude model integration."""

from unittest.mock import MagicMock, patch
import pytest
from pydantic import BaseModel

from src.systems.icl.system import ICLSystem
from src.systems.icl_notepad.system import ICLNotepadSystem
from src.systems.utils.provider_adapters import (
    ProviderTurnClient,
    detect_provider,
    resolve_anthropic_auth_provider,
    resolve_vertex_config,
)
from src.usage import _lookup_model_rates


class SampleSchema(BaseModel):
    decision: str
    confidence: float


def test_detect_provider_vertex_claude():
    assert detect_provider("vertex_ai/claude-3-5-sonnet") == "anthropic"
    assert detect_provider("vertex_ai/claude-3-5-sonnet-v2@20241022") == "anthropic"
    assert detect_provider("vertex/claude-opus-5") == "anthropic"
    assert detect_provider("claude-3-7-sonnet@20250219") == "anthropic"
    assert detect_provider("claude-opus-5") == "anthropic"
    assert detect_provider("anthropic/claude-3-5-haiku") == "anthropic"
    assert detect_provider("gpt-5") == "openai"
    assert detect_provider("gemini-2.5-flash") == "gemini"


def test_resolve_vertex_config_explicit():
    project, region = resolve_vertex_config(
        project_id="my-custom-project",
        region="us-east5",
    )
    assert project == "my-custom-project"
    assert region == "us-east5"


def test_resolve_vertex_config_env_vars(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "env-project-123")
    monkeypatch.setenv("ANTHROPIC_VERTEX_LOCATION", "europe-west1")
    project, region = resolve_vertex_config()
    assert project == "env-project-123"
    assert region == "europe-west1"


def test_resolve_vertex_config_cloud_ml_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_VERTEX_PROJECT_ID", raising=False)
    monkeypatch.delenv("ANTHROPIC_VERTEX_LOCATION", raising=False)
    monkeypatch.setenv("CLOUD_ML_PROJECT_ID", "cloud-ml-project")
    monkeypatch.setenv("CLOUD_ML_REGION", "us-central1")
    project, region = resolve_vertex_config()
    assert project == "cloud-ml-project"
    assert region == "us-central1"


def test_resolve_vertex_config_default_region(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_VERTEX_LOCATION", raising=False)
    monkeypatch.delenv("CLOUD_ML_REGION", raising=False)
    monkeypatch.delenv("VERTEXAI_LOCATION", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_REGION", raising=False)
    monkeypatch.delenv("LOCATION", raising=False)
    _, region = resolve_vertex_config(project_id="test-proj")
    assert region == "global"


def test_resolve_anthropic_auth_provider_explicit():
    assert (
        resolve_anthropic_auth_provider("claude-opus-5", configured_provider="vertex")
        == "vertex"
    )
    assert (
        resolve_anthropic_auth_provider("claude-opus-5", configured_provider="direct")
        == "direct"
    )


def test_resolve_anthropic_auth_provider_auto_vertex_model(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert resolve_anthropic_auth_provider("vertex_ai/claude-3-5-sonnet") == "vertex"


def test_resolve_anthropic_auth_provider_auto_with_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    assert resolve_anthropic_auth_provider("claude-opus-5") == "direct"


def test_resolve_anthropic_auth_provider_auto_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "test-project")
    assert resolve_anthropic_auth_provider("claude-opus-5") == "vertex"


def test_provider_turn_client_vertex_request(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "test-project")
    monkeypatch.setenv("ANTHROPIC_VERTEX_LOCATION", "global")

    from anthropic.types import Message, TextBlock, Usage

    fake_message = Message(
        id="msg_vertex_test",
        content=[
            TextBlock(text='{"decision": "call", "confidence": 0.95}', type="text")
        ],
        model="claude-3-5-sonnet",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage=Usage(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=10,
            cache_creation_input_tokens=5,
        ),
    )

    mock_client = MagicMock()
    mock_client.messages.create.return_value = fake_message

    with patch("anthropic.AnthropicVertex", return_value=mock_client):
        client = ProviderTurnClient(
            model="vertex_ai/claude-3-5-sonnet",
            anthropic_auth_provider="vertex",
        )
        result = client.respond_structured(
            messages=[{"role": "user", "content": "Choose an action"}],
            response_schema=SampleSchema,
        )

        assert isinstance(result.action, SampleSchema)
        assert result.action.decision == "call"
        assert result.action.confidence == 0.95
        assert len(result.usage_events) == 1
        event = result.usage_events[0]
        assert event.model == "vertex_ai/claude-3-5-sonnet"
        assert event.output_tokens == 20
        assert event.cached_input_tokens == 10
        # Check call parameters passed to create
        mock_client.messages.create.assert_called_once()
        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-3-5-sonnet"
        assert kwargs["max_tokens"] == client.anthropic_max_tokens


def test_provider_turn_client_vertex_stream(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "test-project")

    from anthropic.types import Message, TextBlock, Usage

    fake_message = Message(
        id="msg_vertex_stream",
        content=[
            TextBlock(text='{"decision": "fold", "confidence": 0.8}', type="text")
        ],
        model="claude-opus-5",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage=Usage(input_tokens=50, output_tokens=15),
    )

    mock_stream = MagicMock()
    mock_stream.get_final_message.return_value = fake_message
    mock_stream.__enter__.return_value = mock_stream

    mock_client = MagicMock()
    mock_client.messages.stream.return_value = mock_stream

    with patch("anthropic.AnthropicVertex", return_value=mock_client):
        client = ProviderTurnClient(
            model="claude-opus-5",
            anthropic_auth_provider="vertex",
            anthropic_stream=True,
        )
        result = client.respond_structured(
            messages=[{"role": "user", "content": "Action?"}],
            response_schema=SampleSchema,
        )

        assert result.action.decision == "fold"
        mock_client.messages.stream.assert_called_once()


def test_provider_turn_client_vertex_retry_on_rate_limit(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "test-project")

    import httpx
    import anthropic
    from anthropic.types import Message, TextBlock, Usage

    req = httpx.Request("POST", "https://example.com")
    resp_429 = httpx.Response(status_code=429, request=req)
    rate_limit_exc = anthropic.RateLimitError(
        message="Too many requests",
        response=resp_429,
        body={"error": "rate_limited"},
    )

    success_message = Message(
        id="msg_after_retry",
        content=[
            TextBlock(text='{"decision": "check", "confidence": 0.7}', type="text")
        ],
        model="claude-3-5-sonnet",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage=Usage(input_tokens=40, output_tokens=10),
    )

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [rate_limit_exc, success_message]

    with (
        patch("anthropic.AnthropicVertex", return_value=mock_client),
        patch("time.sleep"),
    ):
        client = ProviderTurnClient(
            model="vertex_ai/claude-3-5-sonnet",
            anthropic_auth_provider="vertex",
        )
        result = client.respond_structured(
            messages=[{"role": "user", "content": "Action?"}],
            response_schema=SampleSchema,
        )

        assert result.action.decision == "check"
        assert mock_client.messages.create.call_count == 2


def test_icl_system_with_vertex_params():
    sys = ICLSystem(
        model="vertex_ai/claude-3-5-sonnet",
        anthropic_project_id="test-proj",
        anthropic_region="us-east5",
        anthropic_auth_provider="vertex",
    )
    assert sys._provider_client.anthropic_project_id == "test-proj"
    assert sys._provider_client.anthropic_region == "us-east5"
    assert sys._provider_client.anthropic_auth_provider == "vertex"


def test_icl_notepad_system_with_vertex_params():
    sys = ICLNotepadSystem(
        model="claude-opus-5",
        anthropic_project_id="test-proj",
        anthropic_region="global",
        anthropic_auth_provider="vertex",
        anthropic_stream=True,
    )
    assert sys._provider_client.anthropic_project_id == "test-proj"
    assert sys._provider_client.anthropic_region == "global"
    assert sys._provider_client.anthropic_auth_provider == "vertex"
    assert sys._provider_client.anthropic_stream is True


def test_lookup_model_rates_vertex_claude():
    rates = _lookup_model_rates("vertex_ai/claude-3-5-sonnet", "anthropic")
    assert rates is not None
    assert "input_cost_per_token" in rates

    rates_v2 = _lookup_model_rates(
        "vertex_ai/claude-3-5-sonnet-v2@20241022", "anthropic"
    )
    assert rates_v2 is not None


def test_doctor_with_vertex_auth(monkeypatch, capsys):
    from src.commands.doctor import cmd_doctor

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "doc-test-project")
    monkeypatch.setenv("ANTHROPIC_VERTEX_LOCATION", "global")

    with pytest.raises(SystemExit) as exc_info:
        cmd_doctor([])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert (
        "Anthropic auth configured via Vertex AI (project: doc-test-project, region: global)"
        in out
    )
