# ruff: noqa: E501, E741, B023, SIM102, E402, SIM115
"""Reproduce each leak scenario against whatever `vllm` is first on PYTHONPATH (no plugin loaded)."""

import os

from transformers import AutoTokenizer

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest as C
from vllm.parser import ParserManager

MODEL = os.environ["MODEL_PATH"]
tok = AutoTokenizer.from_pretrained(MODEL)
P = ParserManager.get_parser(
    tool_parser_name="minimax_m3",
    reasoning_parser_name="minimax_m3",
    enable_auto_tools=True,
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "w",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def prompt_ids(kwargs=None, msgs=None):
    msgs = msgs or [{"role": "user", "content": "hi"}]
    s = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, **(kwargs or {})
    )
    return s, tok.encode(s, add_special_tokens=False)


def stream(full, tools, ckw=None):
    req = C(model="m", messages=[{"role": "user", "content": "hi"}], tools=tools)
    p = P(tok, req.tools, chat_template_kwargs=ckw or {}, model_config=None)
    _, pids = prompt_ids(ckw)
    ids = tok.encode(full, add_special_tokens=False)
    R = Cn = ""
    for n, i in enumerate(ids):
        m = p.parse_delta(
            tok.decode([i]),
            [i],
            req,
            pids if n == 0 else None,
            finished=(n == len(ids) - 1),
        )
        if m:
            R += m.reasoning or ""
            Cn += m.content or ""
    return R, Cn


print(
    "== 1. normal reasoning stream (system prompt contains the literal <mm:think></mm:think>)"
)
for tools in (None, TOOLS):
    print(
        "   tools" if tools else "   no tools",
        stream("<mm:think>step.</mm:think>Answer.", tools),
    )

print("== 2. stray </mm:think> inside the answer")
for tools in (None, TOOLS):
    print(
        "   tools" if tools else "   no tools",
        stream("<mm:think>step.</mm:think>Hello.</mm:think>Again.", tools),
    )

print("== 3. structured-output gate: resolve_reasoning_ended(prompt) per thinking mode")
for label, kw in (
    ("adaptive (default)", {}),
    ("thinking_mode=enabled", {"thinking_mode": "enabled"}),
    ("thinking_mode=disabled", {"thinking_mode": "disabled"}),
):
    s, ids = prompt_ids(kw)
    req = C(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )
    p = P(tok, None, chat_template_kwargs=kw, model_config=None)
    print(
        f"   {label:24s} tail={s[-28:]!r} -> gate={req.resolve_reasoning_ended(p, ids)}"
    )
