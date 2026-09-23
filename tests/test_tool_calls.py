"""Qwen tool-call XML parser used by the OpenAI server."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from serve_openai import normalize_messages, parse_qwen_tool_calls, request_tools


def test_parses_xml_into_openai_tool_calls():
    text = (
        "我來查。\n"
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=city>\n台北\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    calls, leftover = parse_qwen_tool_calls(text)
    assert leftover == "我來查。"
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "台北"}


def test_leaves_incomplete_xml_in_the_text():
    text = "查一下 <tool_call><function=get_weather>"
    calls, leftover = parse_qwen_tool_calls(text)
    assert calls == []
    assert leftover == text


def test_openai_argument_string_becomes_a_mapping():
    messages = normalize_messages([{
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_x",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\": \"台北\"}"},
        }],
    }])
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {"city": "台北"}


def test_tool_choice_none_drops_tools():
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    assert request_tools({"tools": tools, "tool_choice": "none"}) is None
    assert request_tools({"tools": tools}) == tools


if __name__ == "__main__":
    test_parses_xml_into_openai_tool_calls()
    test_leaves_incomplete_xml_in_the_text()
    test_openai_argument_string_becomes_a_mapping()
    test_tool_choice_none_drops_tools()
    print("tool-call tests passed")
