"""Tests for the streamed-chunk Collector across all supported providers."""

from __future__ import annotations

import json
import time

import pytest

from llm_stream_collector import CollectedMessage, Collector, Provider, ToolCall

# ---------------------------------------------------------------------------
# Fixture chunk sequences. These mirror real provider chunk shapes but are
# hand-written so the tests don't hit the network.
# ---------------------------------------------------------------------------


def anthropic_text_only_chunks() -> list[dict]:
    return [
        {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": ", "},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "world!"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 7},
        },
        {"type": "message_stop"},
    ]


def anthropic_tool_use_chunks() -> list[dict]:
    return [
        {"type": "message_start", "message": {"usage": {"input_tokens": 9}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Looking that up. "},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "get_weather",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"city": "Aus'},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": 'tin", "unit"'},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": ': "celsius"}'},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 14},
        },
        {"type": "message_stop"},
    ]


def anthropic_multi_tool_chunks() -> list[dict]:
    """Two tool_use blocks in the same response."""
    return [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "first_tool"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"a": 1}'},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_2", "name": "second_tool"},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"b": 2}'},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 11},
        },
    ]


def openai_text_only_chunks() -> list[dict]:
    base = {"id": "chatcmpl-x", "object": "chat.completion.chunk", "model": "gpt-5.4"}
    return [
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        },
        {
            **base,
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        },
        {
            **base,
            "choices": [{"index": 0, "delta": {"content": " from "}, "finish_reason": None}],
        },
        {
            **base,
            "choices": [{"index": 0, "delta": {"content": "OpenAI."}, "finish_reason": None}],
        },
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {
            **base,
            "choices": [],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
        },
    ]


def openai_tool_call_chunks() -> list[dict]:
    base = {"id": "chatcmpl-y", "object": "chat.completion.chunk", "model": "gpt-5.4"}
    return [
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"city":"'}}
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": 'Austin","unit":"'}}
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": 'celsius"}'}}
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]


def openai_multi_tool_call_chunks() -> list[dict]:
    base = {"id": "chatcmpl-z", "object": "chat.completion.chunk", "model": "gpt-5.4"}
    return [
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "tool_a", "arguments": '{"x":1}'},
                            },
                            {
                                "index": 1,
                                "id": "call_2",
                                "type": "function",
                                "function": {"name": "tool_b", "arguments": '{"y":2}'},
                            },
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]


def bedrock_llama_chunks() -> list[dict]:
    return [
        {"generation": "Once ", "prompt_token_count": 8},
        {"generation": "upon ", "prompt_token_count": 8},
        {"generation": "a "},
        {"generation": "time.", "stop_reason": "stop", "generation_token_count": 4},
    ]


def gemini_chunks() -> list[dict]:
    return [
        {"candidates": [{"content": {"parts": [{"text": "Hello"}], "role": "model"}}]},
        {"candidates": [{"content": {"parts": [{"text": " from "}], "role": "model"}}]},
        {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Gemini."}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 6, "candidatesTokenCount": 7},
        },
    ]


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def test_anthropic_assembles_text_in_order():
    c = Collector(provider=Provider.ANTHROPIC)
    for chunk in anthropic_text_only_chunks():
        c.feed(chunk)
    result = c.finalize()
    assert isinstance(result, CollectedMessage)
    assert result.text == "Hello, world!"


def test_anthropic_captures_stop_reason_and_usage():
    c = Collector(provider=Provider.ANTHROPIC)
    for chunk in anthropic_text_only_chunks():
        c.feed(chunk)
    result = c.finalize()
    assert result.stop_reason == "end_turn"
    # input_tokens from message_start, output_tokens from message_delta
    assert result.usage is not None
    assert result.usage["input_tokens"] == 5
    assert result.usage["output_tokens"] == 7


def test_anthropic_assembles_single_tool_call():
    c = Collector(provider=Provider.ANTHROPIC)
    for chunk in anthropic_tool_use_chunks():
        c.feed(chunk)
    result = c.finalize()
    assert result.text == "Looking that up. "
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert isinstance(tc, ToolCall)
    assert tc.name == "get_weather"
    assert tc.id == "toolu_abc"
    assert tc.arguments == {"city": "Austin", "unit": "celsius"}
    assert result.stop_reason == "tool_use"


def test_anthropic_assembles_multiple_tool_calls_in_order():
    c = Collector(provider=Provider.ANTHROPIC)
    for chunk in anthropic_multi_tool_chunks():
        c.feed(chunk)
    result = c.finalize()
    assert len(result.tool_calls) == 2
    assert [tc.name for tc in result.tool_calls] == ["first_tool", "second_tool"]
    assert result.tool_calls[0].arguments == {"a": 1}
    assert result.tool_calls[1].arguments == {"b": 2}
    assert result.tool_calls[0].id == "toolu_1"
    assert result.tool_calls[1].id == "toolu_2"


def test_anthropic_partial_returns_text_so_far():
    c = Collector(provider=Provider.ANTHROPIC)
    chunks = anthropic_text_only_chunks()
    # feed first three real text deltas (index 2,3,4)
    c.feed(chunks[0])
    c.feed(chunks[1])
    c.feed(chunks[2])  # "Hello"
    assert c.partial() == "Hello"
    c.feed(chunks[3])  # ", "
    assert c.partial() == "Hello, "
    c.feed(chunks[4])  # "world!"
    assert c.partial() == "Hello, world!"
    # finalize still works after partial usage
    result = c.finalize()
    assert result.text == "Hello, world!"


def test_anthropic_initial_input_dict_on_tool_use_block():
    """If a tool_use block ships with a non-empty initial 'input' dict,
    that dict should be the starting point and later input_json_delta
    fragments should not stomp it (real Anthropic streams almost always
    send an empty initial input, but the path needs to be safe)."""
    c = Collector(provider=Provider.ANTHROPIC)
    c.feed({"type": "message_start", "message": {}})
    c.feed(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_seed",
                "name": "ping",
                "input": {"seeded": True},
            },
        }
    )
    c.feed({"type": "content_block_stop", "index": 0})
    result = c.finalize()
    assert result.tool_calls == [
        ToolCall(name="ping", arguments={"seeded": True}, id="toolu_seed")
    ]


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def test_openai_assembles_text_in_order():
    c = Collector(provider=Provider.OPENAI)
    for chunk in openai_text_only_chunks():
        c.feed(chunk)
    result = c.finalize()
    assert result.text == "Hello from OpenAI."


def test_openai_captures_finish_reason_and_usage():
    c = Collector(provider=Provider.OPENAI)
    for chunk in openai_text_only_chunks():
        c.feed(chunk)
    result = c.finalize()
    assert result.stop_reason == "stop"
    assert result.usage == {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}


def test_openai_assembles_single_tool_call_with_fragmented_args():
    c = Collector(provider=Provider.OPENAI)
    for chunk in openai_tool_call_chunks():
        c.feed(chunk)
    result = c.finalize()
    assert result.text == ""
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.name == "get_weather"
    assert tc.id == "call_abc"
    assert tc.arguments == {"city": "Austin", "unit": "celsius"}
    assert result.stop_reason == "tool_calls"


def test_openai_assembles_multiple_tool_calls_by_index():
    c = Collector(provider=Provider.OPENAI)
    for chunk in openai_multi_tool_call_chunks():
        c.feed(chunk)
    result = c.finalize()
    assert len(result.tool_calls) == 2
    assert [tc.name for tc in result.tool_calls] == ["tool_a", "tool_b"]
    assert result.tool_calls[0].arguments == {"x": 1}
    assert result.tool_calls[1].arguments == {"y": 2}
    assert [tc.id for tc in result.tool_calls] == ["call_1", "call_2"]


def test_openai_missing_index_on_tool_call_is_ignored():
    """OpenAI requires `index` on each tool_calls delta. A delta without
    one should be silently dropped rather than blowing up."""
    base = {"id": "x", "object": "chat.completion.chunk", "model": "gpt-5.4"}
    c = Collector(provider=Provider.OPENAI)
    c.feed(
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {  # no 'index' key
                                "id": "ghost",
                                "type": "function",
                                "function": {"name": "wat", "arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
    )
    c.feed({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    result = c.finalize()
    assert result.tool_calls == []
    assert result.stop_reason == "tool_calls"


# ---------------------------------------------------------------------------
# Bedrock Anthropic + Bedrock Llama
# ---------------------------------------------------------------------------


def test_bedrock_anthropic_reuses_anthropic_path():
    c = Collector(provider=Provider.BEDROCK_ANTHROPIC)
    for chunk in anthropic_text_only_chunks():
        c.feed(chunk)
    result = c.finalize()
    assert result.text == "Hello, world!"
    assert result.stop_reason == "end_turn"


def test_bedrock_llama_assembles_generation():
    c = Collector(provider=Provider.BEDROCK_LLAMA)
    for chunk in bedrock_llama_chunks():
        c.feed(chunk)
    result = c.finalize()
    assert result.text == "Once upon a time."
    assert result.stop_reason == "stop"
    assert result.usage == {"prompt_token_count": 8, "generation_token_count": 4}


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


def test_gemini_assembles_parts_and_usage_metadata():
    c = Collector(provider=Provider.GEMINI)
    for chunk in gemini_chunks():
        c.feed(chunk)
    result = c.finalize()
    assert result.text == "Hello from Gemini."
    assert result.stop_reason == "STOP"
    assert result.usage == {"promptTokenCount": 6, "candidatesTokenCount": 7}


def test_gemini_multi_part_in_single_chunk():
    c = Collector(provider=Provider.GEMINI)
    c.feed(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "A"}, {"text": "B"}, {"text": "C"}],
                        "role": "model",
                    }
                }
            ]
        }
    )
    result = c.finalize()
    assert result.text == "ABC"


# ---------------------------------------------------------------------------
# Cross-cutting behavior
# ---------------------------------------------------------------------------


def test_empty_and_garbage_chunks_are_ignored():
    c = Collector(provider=Provider.ANTHROPIC)
    c.feed({})  # empty dict
    c.feed({"type": "ping"})  # unknown event
    c.feed(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hi"},
        }
    )
    c.feed({"unrelated": "noise"})
    result = c.finalize()
    assert result.text == "Hi"


def test_finalize_records_chunks_count_and_timing():
    c = Collector(provider=Provider.OPENAI)
    # arrival before any chunks
    time.sleep(0.005)
    for chunk in openai_text_only_chunks():
        c.feed(chunk)
    result = c.finalize()
    assert result.chunks_count == len(openai_text_only_chunks())
    assert result.first_chunk_ms is not None
    # first chunk arrived after at least the 5ms pre-sleep
    assert result.first_chunk_ms >= 4.0
    assert result.total_ms >= result.first_chunk_ms


def test_finalize_with_no_chunks_has_none_first_chunk_ms():
    c = Collector(provider=Provider.ANTHROPIC)
    result = c.finalize()
    assert result.text == ""
    assert result.tool_calls == []
    assert result.usage is None
    assert result.stop_reason is None
    assert result.chunks_count == 0
    assert result.first_chunk_ms is None
    assert result.total_ms >= 0.0


def test_feed_after_finalize_raises():
    c = Collector(provider=Provider.ANTHROPIC)
    c.finalize()
    with pytest.raises(RuntimeError):
        c.feed({"type": "ping"})


def test_finalize_twice_raises():
    c = Collector(provider=Provider.ANTHROPIC)
    c.finalize()
    with pytest.raises(RuntimeError):
        c.finalize()


def test_raw_chunks_preserved_in_order():
    c = Collector(provider=Provider.GEMINI)
    chunks = gemini_chunks()
    for chunk in chunks:
        c.feed(chunk)
    result = c.finalize()
    assert result.raw_chunks == chunks
    # mutation of returned list does not affect future runs (defensive copy)
    result.raw_chunks.clear()
    # collector itself is finalized; just check the snapshot was a copy
    assert len(chunks) == 3


def test_keep_raw_false_drops_raw_but_keeps_count():
    c = Collector(provider=Provider.OPENAI, keep_raw=False)
    for chunk in openai_text_only_chunks():
        c.feed(chunk)
    result = c.finalize()
    assert result.raw_chunks == []
    assert result.chunks_count == len(openai_text_only_chunks())
    assert result.text == "Hello from OpenAI."


def test_partial_works_for_openai_too():
    c = Collector(provider=Provider.OPENAI)
    chunks = openai_text_only_chunks()
    c.feed(chunks[0])
    c.feed(chunks[1])  # "Hello"
    assert c.partial() == "Hello"
    c.feed(chunks[2])  # " from "
    assert c.partial() == "Hello from "


def test_unrecognized_argument_json_yields_empty_dict():
    """When the streamed-tool-call arguments don't parse as JSON, the
    library should return an empty dict rather than raising."""
    c = Collector(provider=Provider.OPENAI)
    base = {"id": "x", "object": "chat.completion.chunk", "model": "gpt-5.4"}
    c.feed(
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_bad",
                                "type": "function",
                                "function": {"name": "broken", "arguments": "not-json"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
    )
    c.feed({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    result = c.finalize()
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].arguments == {}


def test_provider_must_be_enum():
    with pytest.raises(TypeError):
        Collector(provider="anthropic")  # type: ignore[arg-type]


def test_chunks_count_property_tracks_feeds():
    c = Collector(provider=Provider.GEMINI)
    assert c.chunks_count == 0
    c.feed({"candidates": [{"content": {"parts": [{"text": "x"}]}}]})
    assert c.chunks_count == 1
    c.feed({})  # ignored
    assert c.chunks_count == 1


def test_anthropic_tool_use_with_no_input_delta_yields_empty_args():
    """A tool_use content_block that has no input_json_delta events should
    still produce a ToolCall with arguments={}, not raise."""
    c = Collector(provider=Provider.ANTHROPIC)
    c.feed({"type": "message_start", "message": {}})
    c.feed(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_quiet", "name": "noop"},
        }
    )
    c.feed({"type": "content_block_stop", "index": 0})
    result = c.finalize()
    assert result.tool_calls == [
        ToolCall(name="noop", arguments={}, id="toolu_quiet")
    ]


def test_collected_message_is_frozen():
    from dataclasses import FrozenInstanceError

    c = Collector(provider=Provider.ANTHROPIC)
    result = c.finalize()
    with pytest.raises(FrozenInstanceError):
        result.text = "mutated"  # type: ignore[misc]


def test_tool_call_is_frozen():
    from dataclasses import FrozenInstanceError

    tc = ToolCall(name="x", arguments={"a": 1}, id="id")
    with pytest.raises(FrozenInstanceError):
        tc.name = "y"  # type: ignore[misc]


def test_anthropic_tool_call_with_only_initial_input_dict():
    """If the tool_use block ships with the full input dict already and
    there are no input_json_delta events, that initial dict round-trips
    correctly even though it goes through the JSON buffer path."""
    c = Collector(provider=Provider.ANTHROPIC)
    c.feed({"type": "message_start", "message": {}})
    seed_input = {"city": "Austin", "unit": "celsius"}
    c.feed(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_full",
                "name": "get_weather",
                "input": seed_input,
            },
        }
    )
    c.feed({"type": "content_block_stop", "index": 0})
    result = c.finalize()
    assert result.tool_calls[0].arguments == seed_input
    # confirm the buffer round-tripped via JSON
    assert json.loads(json.dumps(seed_input)) == seed_input
