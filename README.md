# llm-stream-collector

[![PyPI](https://img.shields.io/pypi/v/llm-stream-collector.svg)](https://pypi.org/project/llm-stream-collector/)
[![Python](https://img.shields.io/pypi/pyversions/llm-stream-collector.svg)](https://pypi.org/project/llm-stream-collector/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Reassemble streamed LLM chunks into a final message: text, tool calls,
usage, stop reason, timing.**

Anthropic, OpenAI, Bedrock (Anthropic + Llama), and Gemini all stream
responses with different chunk shapes. This library is a small,
zero-dependency state machine that takes the parsed chunk dicts you
already have and gives you back one final normalized message.

Three things it does:

1. Joins text deltas in order.
2. Reassembles tool-call argument JSON fragments into a real dict.
3. Tracks arrival timing (time to first chunk, total elapsed) and a
   chunks count.

It does NOT do HTTP. It does NOT parse raw SSE byte frames. You feed it
one already-parsed JSON dict per event.

Sibling projects:

- [`claude-stream-rs`](https://crates.io/crates/claude-stream-rs): the
  Rust Anthropic-only SSE parser.
- [`agenttap`](https://github.com/MukundaKatta/agenttap): an httpx
  transport that captures the raw JSON sent over the wire.

## Install

```bash
pip install llm-stream-collector
```

## Anthropic

```python
from llm_stream_collector import Collector, Provider

c = Collector(provider=Provider.ANTHROPIC)
for chunk in iter_anthropic_sse():   # already-parsed dicts
    c.feed(chunk)
result = c.finalize()

result.text           # "the assembled message"
result.tool_calls     # [ToolCall(name=..., arguments={...}, id="toolu_...")]
result.usage          # {"input_tokens": 12, "output_tokens": 48}
result.stop_reason    # "end_turn" / "tool_use" / "max_tokens" / ...
result.chunks_count   # 17
result.first_chunk_ms
result.total_ms
```

## OpenAI

```python
from llm_stream_collector import Collector, Provider

c = Collector(provider=Provider.OPENAI)
for delta in openai_client.chat.completions.create(
    model="gpt-5.4",
    messages=[...],
    stream=True,
    stream_options={"include_usage": True},
):
    c.feed(delta.model_dump())
result = c.finalize()

result.text
result.tool_calls     # function-call arguments are joined into a dict
result.usage          # only present when include_usage=True
result.stop_reason    # "stop" / "tool_calls" / "length"
```

## Live UI: partial text without finalizing

```python
c = Collector(provider=Provider.ANTHROPIC)
for chunk in stream:
    c.feed(chunk)
    render(c.partial())   # text so far, no allocations beyond join
# you can still call finalize() at the end
```

## Other providers

```python
Collector(provider=Provider.BEDROCK_ANTHROPIC)  # same shape as ANTHROPIC
Collector(provider=Provider.BEDROCK_LLAMA)      # 'generation' + 'stop_reason'
Collector(provider=Provider.GEMINI)             # candidates[].content.parts[].text
```

## What it does NOT do

- No HTTP client. Bring your own.
- No SSE byte parser. Bring your own line splitter; feed parsed dicts.
- No retry, no backoff, no rate limiting. See `llm-retry` and
  `llm-circuit-breaker` on crates.io.
- No cost calculation. See `claude-cost`, `openai-cost`, `gemini-cost`,
  `bedrock-cost` on crates.io.
- No persistence. State lives in process.

## License

MIT
