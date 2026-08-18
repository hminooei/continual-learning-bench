"""Claude Code CLI system for continual learning benchmark.

Runs claude inside a Docker container for full environment isolation.
Conversation continuity uses --resume with the tracked session id and
`single_conversation` policy. Persistent memory uses Claude's native file-based
memory directory plus any configured workspace watch list, reported as unified
memory files.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from ...interface import (
    ContinualLearningSystem,
    Observation,
    Query,
    Response,
    observation_marks_instance_complete,
)
from ...registry import register_system
from ...usage import build_usage_event
from ..common import (
    CONTAINER_WORKSPACE,
    DEFAULT_DOCKER_IMAGE,
    cleanup_run_workspace,
    copy_memory_seed,
    create_run_workspace,
    docker_exec,
    load_skill_prompt_prefix,
    merge_memory_snapshots,
    normalize_other_memory_file_patterns,
    read_memory_snapshot,
    read_other_memory_files,
    resolve_seed_memory_dir,
    start_docker_container,
    stop_docker_container,
)
from ..utils.structured_output import (
    extract_json,
    schema_to_prompt_instruction,
    validate_with_coercion,
)
from .artifacts import ensure_registered as ensure_claude_artifact_exporter_registered

ensure_claude_artifact_exporter_registered()

logger = logging.getLogger(__name__)

_CONTAINER_CLAUDE_CONFIG = f"{CONTAINER_WORKSPACE}/.claude"
_CLAUDE_PROJECT_SLUG = "-workspace"
_CONTAINER_MEMORY_DIR = (
    f"{_CONTAINER_CLAUDE_CONFIG}/projects/{_CLAUDE_PROJECT_SLUG}/memory"
)
_DOCKER_IMAGE = DEFAULT_DOCKER_IMAGE
_REASONING_EFFORT_VALUES = {"low", "medium", "high", "xhigh", "max"}

_EMPTY_TOKENS: dict[str, int] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "reasoning_tokens": 0,
    "cached_input_tokens": 0,
    "cache_creation_input_tokens": 0,
}

_DEFAULT_MEMORY_INSTRUCTION = """You should learn and store information from each interaction, so use your memory. You have a persistent, file-based memory system at {memory_dir}. The MEMORY.md file at {memory_dir}/MEMORY.md is an index of individual memory files in that directory. Read it at the start of each turn and update both the index and individual memory files as you learn new information that will be useful in future turns."""


def _parse_events(stdout: str) -> list[dict[str, Any]]:
    """Parse JSONL lines from claude stream-json stdout, skipping malformed lines."""
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed JSONL line: %.120s", line)
    return events


def _extract_session_ids(events: list[dict[str, Any]]) -> list[str]:
    """Return unique session_id values from Claude system/init events."""
    session_ids: list[str] = []
    seen: set[str] = set()
    for event in events:
        if event.get("type") != "system" or event.get("subtype") != "init":
            continue
        session_id = event.get("session_id")
        if not isinstance(session_id, str) or session_id in seen:
            continue
        seen.add(session_id)
        session_ids.append(session_id)
    return session_ids


def _extract_session_id(events: list[dict[str, Any]]) -> str | None:
    """Return the first session_id from a system/init event, if present."""
    session_ids = _extract_session_ids(events)
    return session_ids[0] if session_ids else None


def _extract_error_detail(result: subprocess.CompletedProcess[str]) -> str:
    """Extract a concise, useful error message from Claude CLI output."""
    stderr = result.stderr.strip()
    if stderr:
        return stderr
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return "No output produced"
    for line in reversed(lines):
        try:
            event = json.loads(line)
            if event.get("type") in {"error", "result"} and (
                event.get("is_error") or "error" in event or "message" in event
            ):
                return line
        except Exception:
            pass
    return lines[-1]



def _extract_assistant_text(events: list[dict[str, Any]]) -> str:
    """Concatenate text content blocks across all assistant events."""
    text_parts: list[str] = []
    for event in events:
        if event.get("type") != "assistant":
            continue
        content = event.get("message", {}).get("content", [])
        for block in content:
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
    if not text_parts:
        raise ValueError("No assistant message found in claude output")
    return "\n".join(text_parts)


def _extract_token_usage(events: list[dict[str, Any]]) -> dict[str, int]:
    """Extract token totals across Claude result events.

    Claude Code reports result usage per CLI invocation, even when invoked with
    --resume. ``usage.input_tokens`` excludes prompt-cache reads and writes, so
    the normalized input total includes all three input buckets while preserving
    cache buckets for pricing.
    """
    totals = dict(_EMPTY_TOKENS)
    for event in events:
        if event.get("type") != "result":
            continue
        usage = event.get("usage", {})
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
        cache_writes = int(usage.get("cache_creation_input_tokens", 0) or 0)
        totals["input_tokens"] += (
            int(usage.get("input_tokens", 0) or 0) + cache_read + cache_writes
        )
        totals["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
        totals["cached_input_tokens"] += cache_read
        totals["cache_creation_input_tokens"] += cache_writes
    return totals


def _extract_cli_cost_usd(events: list[dict[str, Any]]) -> float | None:
    """Sum Claude Code's own per-result cost when every result reports it."""
    costs: list[float] = []
    saw_result = False
    for event in events:
        if event.get("type") != "result":
            continue
        saw_result = True
        cost = event.get("total_cost_usd")
        if cost is None:
            return None
        try:
            costs.append(float(cost))
        except (TypeError, ValueError):
            return None
    if not saw_result:
        return None
    return sum(costs)


def _json_candidates(text: str) -> list[str]:
    """Return JSON-like candidates from an assistant message, most recent last."""
    stripped = text.strip()
    if not stripped:
        return []

    candidates: list[str] = [stripped]
    if "```" in stripped:
        parts = stripped.split("```")
        for part in parts[1::2]:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") or part.startswith("["):
                candidates.append(part)

    decoder = json.JSONDecoder()
    for idx, char in enumerate(stripped):
        if char not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        candidates.append(stripped[idx : idx + end])

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _parse_action_text(text: str, schema: type[Any]) -> Any:
    """Parse assistant text into the target schema, preferring the latest JSON."""
    last_error: Exception | None = None
    for candidate in reversed(_json_candidates(text)):
        try:
            return validate_with_coercion(candidate, schema)
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return validate_with_coercion(extract_json(text), schema)


@register_system("claude")
class ClaudeCodeSystem(ContinualLearningSystem):
    """Claude Code CLI system running inside a Docker container."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        name: str = "claude",
        seed_memory_dir: Optional[str] = None,
        timeout: int = 300,
        max_turns: Optional[int] = None,
        allowed_tools: Optional[list[str]] = None,
        disallowed_tools: Optional[list[str]] = None,
        docker_image: str = _DOCKER_IMAGE,
        version: str = "2.1.119",
        memory_instruction: str = _DEFAULT_MEMORY_INSTRUCTION,
        single_conversation: bool = True,
        preamble_file: Optional[str] = None,
        skill: Optional[str] = None,
        invocation_type: str = "skill",
        other_memory_files: Optional[list[str]] = None,
        api_key: Optional[str] = None,
        reasoning_effort: Optional[str] = "low",
        use_vertex: Optional[bool] = None,
        vertex_project_id: Optional[str] = None,
        vertex_region: Optional[str] = None,
    ):
        resolved_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if use_vertex is None:
            use_vertex = os.environ.get("CLAUDE_CODE_USE_VERTEX") == "1" or (
                not resolved_api_key
            )

        if use_vertex:
            from ..utils.provider_adapters import resolve_vertex_config

            project_id, region = resolve_vertex_config(
                project_id=vertex_project_id,
                region=vertex_region,
            )
            self._use_vertex = True
            self._vertex_project_id = project_id
            self._vertex_region = region
            self._api_key = resolved_api_key
        else:
            if not resolved_api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY must be set when Vertex AI is not used"
                )
            self._use_vertex = False
            self._vertex_project_id = None
            self._vertex_region = None
            self._api_key = resolved_api_key

        if (
            reasoning_effort is not None
            and reasoning_effort not in _REASONING_EFFORT_VALUES
        ):
            raise ValueError(
                f"reasoning_effort must be one of {sorted(_REASONING_EFFORT_VALUES)} "
                f"or None, got {reasoning_effort!r}"
            )

        if not isinstance(single_conversation, bool):
            raise ValueError(
                f"single_conversation must be a bool, got {single_conversation!r}"
            )

        seed_path = resolve_seed_memory_dir(seed_memory_dir)

        self._name = name
        self._model = model
        self._timeout = timeout
        self._max_turns = max_turns
        self._allowed_tools = allowed_tools
        self._disallowed_tools = disallowed_tools
        self._reasoning_effort = reasoning_effort
        self._memory_instruction = memory_instruction
        self._single_conversation = single_conversation
        self._preamble = self._read_preamble_file(preamble_file)
        self._skill = skill
        self._invocation_type = invocation_type
        self._skill_prompt_prefix = ""
        self._other_memory_files = normalize_other_memory_file_patterns(
            other_memory_files
        )
        self._seed_memory_dir = seed_path

        self._tmp_dir = create_run_workspace("claude_bench")

        self._container_id: str | None = None
        self._docker_image = docker_image
        self._version = version
        self._conversation_id: str | None = None
        self._interaction_count: int = 0
        self._jsonl_log: list[dict[str, Any]] = []
        self._cumulative_tokens: dict[str, int] = dict(_EMPTY_TOKENS)
        self._memory_history: list[dict[str, Any]] = []
        self._claude_conversation_session_ids: set[str] = set()
        self._claude_conversation_history: list[dict[str, Any]] = []
        self._claude_instance_conversations: dict[
            tuple[str | None, int | None], dict[str, Any]
        ] = {}
        self._pending_feedback: str | None = None

        self._initialize_workspace()
        self._start_container()

    def _read_preamble_file(self, preamble_file: Optional[str]) -> str | None:
        """Read optional markdown pre-amble content from the host filesystem."""
        if preamble_file is None:
            return None
        path = Path(preamble_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"preamble_file does not exist: {path}")
        return path.read_text()

    def _start_container(self) -> None:
        """Start a persistent Docker container with claude installed."""
        env: dict[str, str] = {
            "CLAUDE_CONFIG_DIR": _CONTAINER_CLAUDE_CONFIG,
            "IS_SANDBOX": "1",
            "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
        extra_volumes: list[str] = []
        if self._use_vertex:
            gcloud_dir = Path.home() / ".config" / "gcloud"
            if gcloud_dir.is_dir():
                extra_volumes.append(f"{gcloud_dir}:/root/.config/gcloud:ro")
            sa_key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if sa_key_path and Path(sa_key_path).is_file():
                sa_path = Path(sa_key_path).resolve()
                extra_volumes.append(f"{sa_path}:/root/gcp_sa_key.json:ro")
                env["GOOGLE_APPLICATION_CREDENTIALS"] = "/root/gcp_sa_key.json"
            env["CLAUDE_CODE_USE_VERTEX"] = "1"
            if self._vertex_project_id:
                env["CLOUD_ML_PROJECT_ID"] = self._vertex_project_id
                env["ANTHROPIC_VERTEX_PROJECT_ID"] = self._vertex_project_id
            if self._vertex_region:
                env["ANTHROPIC_VERTEX_LOCATION"] = self._vertex_region
                env["CLOUD_ML_REGION"] = self._vertex_region
            if self._model:
                env["ANTHROPIC_MODEL"] = self._model
                env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = self._model
                env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = self._model
            env["MAX_THINKING_TOKENS"] = "0"
        else:
            env["ANTHROPIC_API_KEY"] = self._api_key

        self._container_id = start_docker_container(
            name_prefix="claude_bench",
            host_workspace=self._tmp_dir,
            docker_image=self._docker_image,
            env=env,
            extra_volumes=extra_volumes,
            logger=logger,
        )

        logger.info("Installing ca-certificates + claude-code@%s...", self._version)
        install = self._docker_exec(
            "apt-get update -qq && apt-get install -y -qq ca-certificates > /dev/null 2>&1 && "
            f"npm install -g @anthropic-ai/claude-code@{self._version}",
            timeout=180,
        )
        if install.returncode != 0:
            raise RuntimeError(f"Failed to install claude-code: {install.stderr}")

        self._docker_exec(
            "for bin in node claude; do"
            '  BIN_PATH="$(which "$bin" 2>/dev/null || true)";'
            '  if [ -n "$BIN_PATH" ] && [ "$BIN_PATH" != "/usr/local/bin/$bin" ]; then'
            '    ln -sf "$BIN_PATH" "/usr/local/bin/$bin";'
            "  fi;"
            " done",
            timeout=10,
        )
        verify = self._docker_exec("claude --version", timeout=10)
        logger.info("Verified: %s", verify.stdout.strip() or verify.stderr.strip())

    def _initialize_workspace(self) -> None:
        """Create Claude's config dir skeleton and seed memory if configured."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        copy_memory_seed(self._seed_memory_dir, self.memory_dir)
        self._skill_prompt_prefix = load_skill_prompt_prefix(
            self._skill,
            agent_name="claude",
            workspace_dir=self._tmp_dir,
            invocation_type=self._invocation_type,
        )

    def _docker_exec(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a shell command inside the container."""
        return docker_exec(
            self._container_id,
            command,
            timeout=timeout,
            default_timeout=self._timeout,
        )

    @property
    def memory_dir(self) -> Path:
        """Host-side path to the memory directory."""
        return (
            Path(self._tmp_dir)
            / ".claude"
            / "projects"
            / _CLAUDE_PROJECT_SLUG
            / "memory"
        )

    def _read_native_memory_snapshot(self) -> dict[str, str]:
        """Read Claude memory files from all known project memory dirs."""
        memory_dirs = [self.memory_dir]
        projects_dir = Path(self._tmp_dir) / ".claude" / "projects"
        if projects_dir.exists():
            for candidate in sorted(projects_dir.glob("*/memory")):
                if candidate not in memory_dirs:
                    memory_dirs.append(candidate)

        return read_memory_snapshot(memory_dirs, logger=logger)

    def _read_other_memory_snapshot(self) -> dict[str, str]:
        """Read opt-in workspace files as additional file memory."""
        return read_other_memory_files(
            self._tmp_dir,
            self._other_memory_files,
            logger=logger,
        )

    def _read_memory_snapshot(self) -> dict[str, str]:
        """Read native and opt-in memory files into one merged snapshot."""
        native_memory_snapshot = self._read_native_memory_snapshot()
        other_memory_snapshot = self._read_other_memory_snapshot()
        memory_snapshot, _, _ = merge_memory_snapshots(
            native_memory_snapshot,
            other_memory_snapshot,
            native_source="claude_memory_file",
        )
        return memory_snapshot

    @property
    def _claude_projects_dir(self) -> Path:
        """Host-side Claude projects directory for this run workspace."""
        return Path(self._tmp_dir) / ".claude" / "projects"

    def _read_claude_conversation_jsonl_files(self) -> dict[str, str]:
        """Read Claude Code conversation transcript JSONL files for this run.

        Claude Code can create extra project JSONL files for subagents.  The
        benchmark only exports conversation files whose filename stem matches a
        session_id observed in the main Claude stream-json init events.
        """
        session_ids = set(self._claude_conversation_session_ids)
        if self._conversation_id is not None:
            session_ids.add(self._conversation_id)
        if not session_ids:
            return {}

        projects_dir = self._claude_projects_dir
        if not projects_dir.exists():
            return {}

        files: dict[str, str] = {}
        workspace_dir = Path(self._tmp_dir)
        for path in sorted(projects_dir.rglob("*.jsonl")):
            if not path.is_file() or path.stem not in session_ids:
                continue
            try:
                rel = path.relative_to(workspace_dir).as_posix()
            except ValueError:
                logger.warning("Skipping Claude JSONL outside workspace: %s", path)
                continue
            try:
                files[rel] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning(
                    "Could not read Claude conversation JSONL %s: %s", path, exc
                )
        return files

    @staticmethod
    def _summarize_claude_conversation_files(
        files: dict[str, str],
    ) -> tuple[list[str], dict[str, int]]:
        """Return stable file names and byte-ish character counts for JSONL files."""
        file_names = sorted(files)
        file_sizes = {name: len(files[name]) for name in file_names}
        return file_names, file_sizes

    def _record_claude_conversation_snapshot(
        self,
        query: Query,
        *,
        step: int,
    ) -> dict[str, Any]:
        """Capture the latest native Claude conversation files for an instance."""
        conversation_files = self._read_claude_conversation_jsonl_files()
        file_names, file_sizes = self._summarize_claude_conversation_files(
            conversation_files
        )
        session_ids = sorted(self._claude_conversation_session_ids)
        summary = {
            "step": step,
            "instance_id": query.instance_id,
            "instance_index": query.instance_index,
            "conversation_id": self._conversation_id,
            "session_ids": session_ids,
            "conversation_jsonl_files": file_names,
            "conversation_jsonl_file_sizes": file_sizes,
        }
        self._claude_conversation_history.append(summary)
        self._claude_instance_conversations[
            (query.instance_id, query.instance_index)
        ] = {
            **summary,
            "conversation_jsonl_files": conversation_files,
        }
        return summary

    def _sorted_claude_instance_conversations(self) -> list[dict[str, Any]]:
        """Return latest per-instance Claude conversation snapshots in task order."""

        def sort_key(snapshot: dict[str, Any]) -> tuple[int, int | str, str, int]:
            instance_index = snapshot.get("instance_index")
            if isinstance(instance_index, int):
                primary: tuple[int, int | str] = (0, instance_index)
            else:
                primary = (1, str(snapshot.get("instance_id") or ""))
            return (
                *primary,
                str(snapshot.get("instance_id") or ""),
                int(snapshot.get("step") or 0),
            )

        return sorted(self._claude_instance_conversations.values(), key=sort_key)

    def _build_cli_args(self, resume_mode: str) -> list[str]:
        """Build claude CLI args.

        resume_mode: "fresh" (first call) or "resume-sid".
        """
        args = [
            "claude",
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            "bypassPermissions",
            "--model",
            self._model,
        ]
        if self._reasoning_effort is not None and not self._use_vertex:
            args.extend(["--effort", self._reasoning_effort])
        if resume_mode == "resume-sid":
            if self._conversation_id is None:
                raise RuntimeError("resume-sid requested with no session_id")
            args.extend(["--resume", self._conversation_id])
        elif resume_mode != "fresh":
            raise RuntimeError(f"unsupported Claude resume_mode: {resume_mode}")

        if self._max_turns is not None:
            args.extend(["--max-turns", str(self._max_turns)])
        if self._allowed_tools is not None:
            args.extend(["--allowedTools", ",".join(self._allowed_tools)])
        if self._disallowed_tools is not None:
            args.extend(["--disallowedTools", ",".join(self._disallowed_tools)])
        return args

    def _wrap_skill_invocation(self, prompt: str) -> str:
        """Wrap a prompt in Claude Code native skill invocation tags if needed."""
        if not self._skill_prompt_prefix:
            return prompt
        if self._invocation_type.strip().lower() != "skill":
            return f"{self._skill_prompt_prefix}{prompt}"
        return f"{self._skill_prompt_prefix}<command-args>{prompt}</command-args>"

    def _build_prompt(self, query: Query, *, include_skill: bool = True) -> str:
        parts: list[str] = []
        if self._preamble:
            parts.append(f"PRE-AMBLE:\n{self._preamble}")
        if self._pending_feedback:
            parts.append(f"FEEDBACK FROM PREVIOUS ACTION: {self._pending_feedback}\n")
        parts.append(query.prompt)
        parts.append(schema_to_prompt_instruction(query.response_schema))
        parts.append(
            "\n\n" + self._memory_instruction.format(memory_dir=_CONTAINER_MEMORY_DIR)
        )
        prompt = "\n".join(parts)
        if include_skill:
            return self._wrap_skill_invocation(prompt)
        return prompt

    def _prepare_conversation(self, query: Query) -> dict[str, Any]:
        """Apply conversation policy and return metadata for this turn."""
        previous_conversation_id = self._conversation_id
        resumed_this_turn = self._conversation_id is not None
        return {
            "mode": "single_conversation"
            if self._single_conversation
            else "per_instance_conversation",
            "single_conversation": self._single_conversation,
            "instance_id": query.instance_id,
            "instance_index": query.instance_index,
            "invocation_mode": "resume-sid" if resumed_this_turn else "fresh",
            "resumed_this_turn": resumed_this_turn,
            "conversation_id_before": previous_conversation_id,
            "conversation_id_for_turn": self._conversation_id,
        }

    def _build_repair_prompt(
        self,
        original_prompt: str,
        assistant_text: str,
        validation_error: str,
    ) -> str:
        return "\n".join(
            [
                "Your previous response did not match the required benchmark schema.",
                f"Validation error: {validation_error}",
                "Previous response:",
                assistant_text,
                "Return ONLY a corrected JSON object for the same task.",
                "Original prompt:",
                original_prompt,
            ]
        )

    def _run_claude(self, prompt: str, resume_mode: str) -> list[dict[str, Any]]:
        """Invoke claude once and retry once on non-zero exit."""

        def _attempt(mode: str) -> subprocess.CompletedProcess[str]:
            args = self._build_cli_args(mode)
            claude_cmd = " ".join([*args, "--", shlex.quote(prompt)])
            command = (
                f"{claude_cmd}; _rc=$?; "
                "chmod -R a+rX /workspace/.claude >/dev/null 2>&1 || true; exit $_rc"
            )
            logger.info(
                "Calling claude (mode=%s, interaction=%d, prompt=%d chars)",
                mode,
                self._interaction_count + 1,
                len(prompt),
            )
            return self._docker_exec(command, timeout=self._timeout)

        try:
            result = _attempt(resume_mode)
        except subprocess.TimeoutExpired as exc:
            timeout = exc.timeout if exc.timeout is not None else self._timeout
            raise RuntimeError(
                f"LLM call failed: Claude timed out after {timeout} seconds"
            ) from exc
        if result.returncode != 0:
            detail = _extract_error_detail(result)
            retry_mode = "resume-sid" if self._conversation_id is not None else "fresh"
            logger.warning(
                "Claude failed (rc=%d), retrying as %s: %s",
                result.returncode,
                retry_mode,
                detail[:300],
            )
            try:
                result = _attempt(retry_mode)
            except subprocess.TimeoutExpired as exc:
                timeout = exc.timeout if exc.timeout is not None else self._timeout
                raise RuntimeError(
                    f"LLM call failed: Claude timed out after {timeout} seconds during retry"
                ) from exc
            if result.returncode != 0 and retry_mode == "resume-sid":
                detail = _extract_error_detail(result)
                logger.warning(
                    "Claude resume failed (rc=%d), falling back to fresh attempt: %s",
                    result.returncode,
                    detail[:300],
                )
                try:
                    result = _attempt("fresh")
                except subprocess.TimeoutExpired as exc:
                    timeout = exc.timeout if exc.timeout is not None else self._timeout
                    raise RuntimeError(
                        f"LLM call failed: Claude timed out after {timeout} seconds during fresh retry"
                    ) from exc
            if result.returncode != 0:
                detail = _extract_error_detail(result)
                raise RuntimeError(
                    f"LLM call failed: Claude exited with code {result.returncode} after retry: {detail}"
                )

        events = _parse_events(result.stdout)
        logger.info("Claude returned %d events", len(events))
        self._jsonl_log.extend(events)
        session_ids = _extract_session_ids(events)
        self._claude_conversation_session_ids.update(session_ids)
        if session_ids:
            self._conversation_id = session_ids[0]
        return events

    def _retry_mode_after_empty_response(self) -> str:
        """Choose a conservative retry mode after Claude returns no assistant text."""
        if self._conversation_id is not None:
            return "resume-sid"
        return "fresh"

    def _run_assistant_turn(
        self,
        prompt: str,
        *,
        resume_mode: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        events = self._run_claude(prompt, resume_mode=resume_mode)
        try:
            assistant_text = _extract_assistant_text(events)
        except ValueError as exc:
            retry_mode = self._retry_mode_after_empty_response()
            logger.warning(
                "Claude returned no assistant message, retrying as %s: %s",
                retry_mode,
                exc,
            )
            retry_events = self._run_claude(prompt, resume_mode=retry_mode)
            try:
                assistant_text = _extract_assistant_text(retry_events)
            except ValueError as retry_exc:
                raise RuntimeError(
                    f"LLM call failed: Claude returned no assistant message after retry: {retry_exc}"
                ) from retry_exc
            return assistant_text, [*events, *retry_events]
        return assistant_text, events

    def _run_action_turn(
        self,
        prompt: str,
        response_schema: type[Any],
        *,
        resume_mode: str,
        repair_original_prompt: str,
    ) -> tuple[Any, list[dict[str, Any]], bool]:
        assistant_text, all_events = self._run_assistant_turn(
            prompt,
            resume_mode=resume_mode,
        )

        try:
            action = _parse_action_text(assistant_text, response_schema)
            return action, all_events, False
        except Exception as exc:
            repair_prompt = self._build_repair_prompt(
                repair_original_prompt,
                assistant_text,
                str(exc),
            )
            repair_resume_mode = (
                "resume-sid" if self._conversation_id is not None else "fresh"
            )
            repair_events = self._run_claude(
                repair_prompt,
                resume_mode=repair_resume_mode,
            )
            all_events.extend(repair_events)
            try:
                repaired_text = _extract_assistant_text(repair_events)
            except ValueError as repair_exc:
                raise RuntimeError(
                    f"LLM call failed: Claude repair returned no assistant message: {repair_exc}"
                ) from repair_exc
            try:
                action = _parse_action_text(repaired_text, response_schema)
            except Exception as repair_parse_exc:
                raise RuntimeError(
                    f"LLM call failed: Claude response could not be parsed into schema: {repair_parse_exc}"
                ) from repair_parse_exc
            return action, all_events, True

    def _record_turn_usage(self, events: list[dict[str, Any]]) -> dict[str, int]:
        token_usage = _extract_token_usage(events)
        for key in self._cumulative_tokens:
            self._cumulative_tokens[key] += token_usage[key]

        usage_event = build_usage_event(
            model=self._model,
            provider="anthropic",
            input_tokens=token_usage["input_tokens"],
            output_tokens=token_usage["output_tokens"],
            cached_input_tokens=token_usage["cached_input_tokens"],
            cache_creation_input_tokens=token_usage["cache_creation_input_tokens"],
            call_type="completion",
        )
        cli_cost_usd = _extract_cli_cost_usd(events)
        if cli_cost_usd is not None:
            usage_event.cost_usd = cli_cost_usd
            usage_event.pricing_source = "claude_cli.total_cost_usd"
            usage_event.pricing_error = None
        self.record_usage_event(usage_event)

        logger.info(
            "Response parsed (in=%d out=%d)",
            token_usage["input_tokens"],
            token_usage["output_tokens"],
        )
        return token_usage

    def _record_memory_snapshot(self) -> dict[str, str]:
        memory_snapshot = self._read_memory_snapshot()
        if memory_snapshot or self._other_memory_files:
            self._memory_history.append(
                {
                    "step": self._interaction_count + 1,
                    "files": memory_snapshot,
                }
            )
        self._log_memory_snapshot(memory_snapshot)
        return memory_snapshot

    @staticmethod
    def _log_memory_snapshot(memory_files: dict[str, str]) -> None:
        if memory_files:
            logger.info(
                "Memory after step: %d file(s), %d chars",
                len(memory_files),
                sum(len(value) for value in memory_files.values()),
            )
            return
        logger.info("Memory after step: (empty)")

    def _finish_successful_interaction(self) -> None:
        self._pending_feedback = None
        self._interaction_count += 1

    def _build_response_metadata(
        self,
        *,
        continuity: dict[str, Any],
        token_usage: dict[str, int],
        repair_attempted: bool,
        memory_snapshot: dict[str, str],
        conversation_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "interaction_count": self._interaction_count,
            "system_type": "claude",
            "model": self._model,
            "version": self._version,
            "conversation_id": self._conversation_id,
            "continuity": {
                **continuity,
                "conversation_id_after": self._conversation_id,
            },
            "token_usage": token_usage,
            "cumulative_tokens": dict(self._cumulative_tokens),
            "repair_attempted": repair_attempted,
            "memory_files": memory_snapshot,
            "claude_conversation_session_ids": conversation_snapshot["session_ids"],
            "claude_conversation_jsonl_files": conversation_snapshot[
                "conversation_jsonl_files"
            ],
            "claude_conversation_jsonl_file_sizes": conversation_snapshot[
                "conversation_jsonl_file_sizes"
            ],
        }

    def respond(self, query: Query) -> Response:
        continuity = self._prepare_conversation(query)
        prompt_without_skill = self._build_prompt(query, include_skill=False)
        prompt = self._build_prompt(
            query,
            include_skill=continuity["invocation_mode"] == "fresh",
        )
        action, all_events, repair_attempted = self._run_action_turn(
            prompt,
            query.response_schema,
            resume_mode=continuity["invocation_mode"],
            repair_original_prompt=prompt_without_skill,
        )
        token_usage = self._record_turn_usage(all_events)
        memory_snapshot = self._record_memory_snapshot()
        conversation_snapshot = self._record_claude_conversation_snapshot(
            query,
            step=self._interaction_count + 1,
        )
        self._finish_successful_interaction()

        return Response(
            action=action,
            metadata=self._build_response_metadata(
                continuity=continuity,
                token_usage=token_usage,
                repair_attempted=repair_attempted,
                memory_snapshot=memory_snapshot,
                conversation_snapshot=conversation_snapshot,
            ),
        )

    def observe(
        self, observation: Observation, next_query: Optional[Query] = None
    ) -> None:
        _ = next_query
        if not self._single_conversation and observation_marks_instance_complete(
            observation
        ):
            self._conversation_id = None
        content = observation.content.strip()
        if not content:
            return
        self._record_observation_feedback(content)

    def _record_observation_feedback(self, content: str) -> None:
        self._pending_feedback = content

    def reset(self) -> None:
        self._clear_interaction_state()
        self._clear_conversation_state()
        self._clear_persistent_workspace_state()
        self._rebuild_workspace_state()

    def _clear_interaction_state(self) -> None:
        self._interaction_count = 0
        self._jsonl_log = []
        self._cumulative_tokens = dict(_EMPTY_TOKENS)
        self._memory_history = []
        self._claude_conversation_history = []
        self._claude_instance_conversations = {}
        self._pending_feedback = None

    def _clear_conversation_state(self) -> None:
        self._conversation_id = None
        self._claude_conversation_session_ids = set()

    def _clear_persistent_workspace_state(self) -> None:
        workspace = Path(self._tmp_dir)
        self._delete_claude_project_state(workspace)
        self._delete_watched_memory_files(workspace)

    @staticmethod
    def _delete_claude_project_state(workspace: Path) -> None:
        shutil.rmtree(workspace / ".claude" / "projects", ignore_errors=True)

    def _delete_watched_memory_files(self, workspace: Path) -> None:
        for relpath in self._read_other_memory_snapshot():
            path = workspace / relpath
            if path.is_file():
                path.unlink()

    def _rebuild_workspace_state(self) -> None:
        self._initialize_workspace()

    def get_run_artifacts(self) -> dict[str, Any]:
        memory_snapshot = self._read_memory_snapshot()
        conversation_jsonl_files = self._read_claude_conversation_jsonl_files()
        return {
            "artifact_type": "claude",
            "memory_files": memory_snapshot,
            "memory_history": list(self._memory_history),
            "conversation_id": self._conversation_id,
            "claude_conversation_session_ids": sorted(
                self._claude_conversation_session_ids
            ),
            "claude_conversation_jsonl_files": conversation_jsonl_files,
            "claude_conversation_history": list(self._claude_conversation_history),
            "claude_instance_conversations": self._sorted_claude_instance_conversations(),
            "version": self._version,
            "interaction_count": self._interaction_count,
            "cumulative_tokens": dict(self._cumulative_tokens),
            "jsonl_events": self._jsonl_log,
        }

    @property
    def name(self) -> str:
        return self._name

    def _stop_container(self) -> None:
        if self._container_id is not None:
            stop_docker_container(self._container_id)
            self._container_id = None

    def __del__(self) -> None:
        if hasattr(self, "_container_id"):
            self._stop_container()
        if hasattr(self, "_tmp_dir"):
            cleanup_run_workspace(self._tmp_dir)
