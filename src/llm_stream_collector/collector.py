"""Core Collector implementation.

Provider-aware streamed-chunk reassembler. The caller is responsible for
SSE framing / network IO; they feed already-parsed JSON dicts (one per
streamed event) and then call `finalize()` to get a `CollectedMessage`
with the joined text, tool calls, usage, stop reason, and timing info.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Provider(str, Enum):
    """Which provider's chunk shape to expect."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    BEDROCK_ANTHROPIC = "bedrock_anthropic"
    BEDROCK_LLAMA = "bedrock_llama"
    GEMINI = "gemini"


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation reassembled from streamed chunks."""

    name: str
    arguments: dict[str, Any]
    id: str | None = None


@dataclass(frozen=True)
class CollectedMessage:
    """Final reassembled message returned by `Collector.finalize()`.

    Attributes:
        text: concatenated text content
        tool_calls: tool invocations the model emitted
        usage: provider-reported token usage if any was seen
        stop_reason: provider-reported stop reason if any was seen
        chunks_count: how many chunks were fed
        first_chunk_ms: ms between Collector construction and first chunk
        total_ms: ms between Collector construction and finalize
        raw_chunks: original chunks in the order they were fed
    """

    text: str
    tool_calls: list[ToolCall]
    usage: dict[str, Any] | None
    stop_reason: str | None
    chunks_count: int
    first_chunk_ms: float | None
    total_ms: float
    raw_chunks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _PartialToolCall:
    """Mutable tool-call accumulator used while assembling streamed args."""

    name: str | None = None
    arg_buffer: str = ""
    id: str | None = None
    index: int | None = None
    # An initial input dict shipped on the block-start event (Anthropic). Kept
    # separate from `arg_buffer` so that streamed JSON fragments augment it
    # instead of being string-concatenated onto a serialized dict (which would
    # produce invalid JSON and silently drop both).
    seed_args: dict[str, Any] | None = None


class Collector:
    """Streamed-chunk reassembler for a given provider.

    Construction starts the timing clock. Call `feed(chunk_dict)` for each
    streamed event in arrival order, then `finalize()` to materialize a
    `CollectedMessage`. Call `partial()` mid-stream for the text so far
    without finalizing (useful for live UIs).

    The collector does not parse SSE framing. Pass a `dict` per event.
    Empty or unrecognized chunks are ignored, not raised.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        keep_raw: bool = True,
    ) -> None:
        if not isinstance(provider, Provider):
            raise TypeError("provider must be a Provider enum value")
        self._provider = provider
        self._keep_raw = keep_raw

        # arrival timing
        self._start = time.monotonic()
        self._first_chunk_at: float | None = None
        self._finalized = False

        # accumulators
        self._text_parts: list[str] = []
        self._raw: list[dict[str, Any]] = []
        self._chunks_seen: int = 0
        self._usage: dict[str, Any] | None = None
        self._stop_reason: str | None = None

        # tool-call assembly, keyed differently per provider
        # Anthropic: by content_block index
        # OpenAI: by tool_calls[].index
        self._tool_calls: dict[int, _PartialToolCall] = {}
        self._tool_order: list[int] = []

    @property
    def provider(self) -> Provider:
        return self._provider

    @property
    def chunks_count(self) -> int:
        return self._chunks_seen

    # ---- public API ----

    def feed(self, chunk: dict[str, Any]) -> None:
        """Accept one parsed chunk and update state.

        Empty or `None` chunks, and chunks that don't carry any payload the
        current provider recognizes, are ignored without raising.
        """
        if self._finalized:
            raise RuntimeError("collector already finalized; cannot feed more chunks")
        if not isinstance(chunk, dict) or not chunk:
            return

        if self._first_chunk_at is None:
            self._first_chunk_at = time.monotonic()
        if self._keep_raw:
            self._raw.append(chunk)
        self._chunks_seen += 1

        if self._provider in (Provider.ANTHROPIC, Provider.BEDROCK_ANTHROPIC):
            self._feed_anthropic(chunk)
        elif self._provider is Provider.OPENAI:
            self._feed_openai(chunk)
        elif self._provider is Provider.BEDROCK_LLAMA:
            self._feed_bedrock_llama(chunk)
        elif self._provider is Provider.GEMINI:
            self._feed_gemini(chunk)

    def partial(self) -> str:
        """Return the text reassembled so far without finalizing."""
        return "".join(self._text_parts)

    def finalize(self) -> CollectedMessage:
        """Return a `CollectedMessage` snapshot. Safe to call once.

        Subsequent `feed()` calls raise after finalize; call `partial()` if
        you only need the text-so-far.
        """
        if self._finalized:
            raise RuntimeError("collector already finalized")
        self._finalized = True

        end = time.monotonic()
        total_ms = (end - self._start) * 1000.0
        first_ms = (
            (self._first_chunk_at - self._start) * 1000.0
            if self._first_chunk_at is not None
            else None
        )

        # finalize tool calls in arrival order
        finalized_tools: list[ToolCall] = []
        for idx in self._tool_order:
            partial_tc = self._tool_calls[idx]
            delta_args = _safe_json_loads(partial_tc.arg_buffer) if partial_tc.arg_buffer else {}
            if partial_tc.seed_args is not None:
                # Start from the seeded input and let streamed fragments win.
                args = {**partial_tc.seed_args, **delta_args}
            else:
                args = delta_args
            finalized_tools.append(
                ToolCall(
                    name=partial_tc.name or "",
                    arguments=args,
                    id=partial_tc.id,
                )
            )

        return CollectedMessage(
            text="".join(self._text_parts),
            tool_calls=finalized_tools,
            usage=self._usage,
            stop_reason=self._stop_reason,
            chunks_count=self._chunks_seen,
            first_chunk_ms=first_ms,
            total_ms=total_ms,
            raw_chunks=list(self._raw) if self._keep_raw else [],
        )

    # ---- per-provider feed handlers ----

    def _feed_anthropic(self, chunk: dict[str, Any]) -> None:
        # event-typed SSE shape, e.g.
        #   {"type": "message_start", "message": {...}}
        #   {"type": "content_block_start", "index": 0,
        #     "content_block": {"type": "text"|"tool_use", ...}}
        #   {"type": "content_block_delta", "index": 0,
        #     "delta": {"type": "text_delta"|"input_json_delta", ...}}
        #   {"type": "message_delta",
        #     "delta": {"stop_reason": ...}, "usage": {...}}
        event_type = chunk.get("type")
        if event_type == "message_start":
            msg = chunk.get("message") or {}
            usage = msg.get("usage")
            if isinstance(usage, dict):
                self._usage = dict(usage)
            stop = msg.get("stop_reason")
            if isinstance(stop, str):
                self._stop_reason = stop
            return

        if event_type == "content_block_start":
            idx = chunk.get("index")
            block = chunk.get("content_block") or {}
            if isinstance(idx, int) and block.get("type") == "tool_use":
                tc = self._tool_calls.setdefault(idx, _PartialToolCall(index=idx))
                if idx not in self._tool_order:
                    self._tool_order.append(idx)
                name = block.get("name")
                if isinstance(name, str):
                    tc.name = name
                tid = block.get("id")
                if isinstance(tid, str):
                    tc.id = tid
                # some servers include an initial input dict here
                initial_input = block.get("input")
                if isinstance(initial_input, dict) and initial_input:
                    tc.seed_args = dict(initial_input)
            return

        if event_type == "content_block_delta":
            delta = chunk.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                text = delta.get("text")
                if isinstance(text, str):
                    self._text_parts.append(text)
            elif dtype == "input_json_delta":
                idx = chunk.get("index")
                partial_json = delta.get("partial_json")
                if isinstance(idx, int) and isinstance(partial_json, str):
                    tc = self._tool_calls.setdefault(idx, _PartialToolCall(index=idx))
                    if idx not in self._tool_order:
                        self._tool_order.append(idx)
                    tc.arg_buffer += partial_json
            return

        if event_type == "message_delta":
            delta = chunk.get("delta") or {}
            stop = delta.get("stop_reason")
            if isinstance(stop, str):
                self._stop_reason = stop
            usage = chunk.get("usage")
            if isinstance(usage, dict):
                if self._usage is None:
                    self._usage = dict(usage)
                else:
                    # merge so output_tokens from message_delta wins
                    self._usage = {**self._usage, **usage}
            return

        # message_stop, ping, content_block_stop, etc: nothing to do.

    def _feed_openai(self, chunk: dict[str, Any]) -> None:
        # OpenAI chat.completions stream chunk:
        #   {"choices": [{"index": 0,
        #                 "delta": {"content": "...", "tool_calls": [...]},
        #                 "finish_reason": ...}],
        #    "usage": {...}}  # only on final chunk when include_usage=True
        choices = chunk.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str):
                    self._text_parts.append(content)

                tool_calls = delta.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        idx = tc.get("index")
                        if not isinstance(idx, int):
                            # OpenAI requires index on each tool_calls delta
                            continue
                        slot = self._tool_calls.setdefault(idx, _PartialToolCall(index=idx))
                        if idx not in self._tool_order:
                            self._tool_order.append(idx)
                        tid = tc.get("id")
                        if isinstance(tid, str) and slot.id is None:
                            slot.id = tid
                        function = tc.get("function") or {}
                        if isinstance(function, dict):
                            name = function.get("name")
                            if isinstance(name, str) and slot.name is None:
                                slot.name = name
                            args = function.get("arguments")
                            if isinstance(args, str):
                                slot.arg_buffer += args

                finish_reason = choice.get("finish_reason")
                if isinstance(finish_reason, str):
                    self._stop_reason = finish_reason

        # usage may appear in the final chunk when include_usage=True
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self._usage = dict(usage)

    def _feed_bedrock_llama(self, chunk: dict[str, Any]) -> None:
        # Bedrock Llama invokeWithResponseStream chunk shape (already JSON-decoded):
        #   {"generation": "...", "stop_reason": "stop"|"length"|None,
        #    "prompt_token_count": int, "generation_token_count": int}
        gen = chunk.get("generation")
        if isinstance(gen, str):
            self._text_parts.append(gen)

        stop = chunk.get("stop_reason")
        if isinstance(stop, str):
            self._stop_reason = stop

        # collect usage incrementally; Llama returns these on final chunks
        prompt_tokens = chunk.get("prompt_token_count")
        gen_tokens = chunk.get("generation_token_count")
        if isinstance(prompt_tokens, int) or isinstance(gen_tokens, int):
            if self._usage is None:
                self._usage = {}
            if isinstance(prompt_tokens, int):
                self._usage["prompt_token_count"] = prompt_tokens
            if isinstance(gen_tokens, int):
                self._usage["generation_token_count"] = gen_tokens

    def _feed_gemini(self, chunk: dict[str, Any]) -> None:
        # Gemini generateContentStream chunk shape:
        #   {"candidates": [{"content": {"parts": [{"text": "..."}, ...], "role": "model"},
        #                    "finishReason": "STOP"|...}],
        #    "usageMetadata": {"promptTokenCount": int, "candidatesTokenCount": int, ...}}
        candidates = chunk.get("candidates")
        if isinstance(candidates, list) and candidates:
            cand = candidates[0]
            if isinstance(cand, dict):
                content = cand.get("content") or {}
                parts = content.get("parts") if isinstance(content, dict) else None
                if isinstance(parts, list):
                    for part in parts:
                        if not isinstance(part, dict):
                            continue
                        text = part.get("text")
                        if isinstance(text, str):
                            self._text_parts.append(text)
                finish = cand.get("finishReason")
                if isinstance(finish, str):
                    self._stop_reason = finish

        usage = chunk.get("usageMetadata")
        if isinstance(usage, dict):
            self._usage = dict(usage)


def _safe_json_loads(blob: str) -> dict[str, Any]:
    """Parse a JSON object from a streamed-arguments string; return {} on
    failure so callers always get a dict back. We never raise during the
    text-assembly path; argument parsing failures are surfaced as an empty
    dict so a downstream consumer can decide how to react."""
    try:
        parsed = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
