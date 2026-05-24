"""llm-stream-collector - reassemble streamed LLM chunks across providers.

Caller is responsible for SSE framing and network IO; feed parsed JSON
dicts one at a time to a `Collector` and call `finalize()` to get a
`CollectedMessage` with the joined text, tool calls, usage, stop reason,
and timing info.

    from llm_stream_collector import Collector, Provider

    c = Collector(provider=Provider.ANTHROPIC)
    for chunk in iter_sse_events():  # already-parsed dicts
        c.feed(chunk)
    result = c.finalize()
    # result.text, result.tool_calls, result.usage, result.stop_reason
    # result.chunks_count, result.first_chunk_ms, result.total_ms

For live UIs, call `partial()` mid-stream for the text reassembled so far
without finalizing the collector.

Sibling projects:
  * `claude-stream-rs` - Rust Anthropic-only SSE parser.
  * `agenttap` - Python httpx-transport wire-level capture.
"""

from llm_stream_collector.collector import (
    CollectedMessage,
    Collector,
    Provider,
    ToolCall,
)

__version__ = "0.1.0"

__all__ = [
    "CollectedMessage",
    "Collector",
    "Provider",
    "ToolCall",
    "__version__",
]
