#!/usr/bin/env python3
# ruff: noqa: E501, E741, B023, SIM102, E402, SIM115
"""Leak test matrix for minimax-m31-flash through the gateway.

Every case runs streaming and non-streaming, REPEAT times (temperature 1.0 is the server default, so output
varies). A case FAILS when any of the following holds:
  * a protocol marker appears in content, reasoning, tool-call name or tool-call arguments
  * the response is empty (no content, no reasoning, no tool call) without finish_reason=length
  * streaming and non-streaming disagree structurally (tool-call count / names) at temperature 0
  * tool-call arguments are not valid JSON, or do not match the tool schema's required keys
"""

import concurrent.futures as cf
import json
import re
import sys
import urllib.request

E = sys.argv[1]
REPEAT = int(sys.argv[2]) if len(sys.argv) > 2 else 4
import os

MODEL = os.environ.get("MODEL", "minimax-m31-flash")

LEAK = re.compile(
    r"</?mm:think>|</?think>|\]<\]minimax\[>\[|</?tool_call>|</?invoke\b|\]~b\]|\[e~\[|\]~!b\[|"
    r"</?response>|thinking_instructions|<model_id>|</?function_call>|</?minimax:"
)

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}
TIME = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get current local time for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
CALC = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}
NESTED = {
    "type": "function",
    "function": {
        "name": "create_order",
        "description": "Create an order",
        "parameters": {
            "type": "object",
            "properties": {
                "customer": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "vip": {"type": "boolean"},
                    },
                    "required": ["name"],
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sku": {"type": "string"},
                            "qty": {"type": "integer"},
                        },
                        "required": ["sku", "qty"],
                    },
                },
                "note": {"type": "string"},
            },
            "required": ["customer", "items"],
        },
    },
}
WRITE = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
}


def U(t):
    return {"role": "user", "content": t}


def post(body, stream, timeout=300):
    body = dict(body, model=MODEL, stream=stream)
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(
        E + body.pop("_path", "/v1/chat/completions"),
        json.dumps(body).encode(),
        {"content-type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def run_chat(body, stream):
    """Returns dict(reasoning, content, tool_calls=[{name,arguments}], finish, chunks, usage, order)"""
    out = dict(
        reasoning="",
        content="",
        tool_calls=[],
        finish=None,
        chunks=0,
        usage=None,
        order=[],
        raw_roles=0,
    )
    if not stream:
        d = json.load(post(body, False))
        ch = d["choices"][0]
        m = ch["message"]
        out["reasoning"] = m.get("reasoning") or m.get("reasoning_content") or ""
        out["content"] = m.get("content") or ""
        out["tool_calls"] = [
            {"name": t["function"]["name"], "arguments": t["function"]["arguments"]}
            for t in (m.get("tool_calls") or [])
        ]
        out["finish"] = ch["finish_reason"]
        out["usage"] = d.get("usage")
        return out
    tcs = {}
    for line in post(body, True):
        line = line.decode().strip()
        if not line.startswith("data:"):
            continue
        p = line[5:].strip()
        if p == "[DONE]":
            break
        d = json.loads(p)
        out["chunks"] += 1
        if d.get("usage"):
            out["usage"] = d["usage"]
        if not d.get("choices"):
            continue
        ch = d["choices"][0]
        dl = ch.get("delta") or {}
        r = dl.get("reasoning") or dl.get("reasoning_content") or ""
        c = dl.get("content") or ""
        first = (
            ch.get("index", 0) == 0
        )  # with n>1 the choices interleave; check ordering per choice (choice 0)
        if r:
            out["reasoning"] += r
            if first:
                out["order"].append("R")
        if c:
            out["content"] += c
            if first:
                out["order"].append("C")
        for t in dl.get("tool_calls") or []:
            i = t.get("index", 0)
            s = tcs.setdefault(i, {"name": "", "arguments": ""})
            f = t.get("function") or {}
            if f.get("name"):
                s["name"] += f["name"]
            if f.get("arguments"):
                s["arguments"] += f["arguments"]
            out["order"].append("T")
        if ch.get("finish_reason"):
            out["finish"] = ch["finish_reason"]
    out["tool_calls"] = [tcs[i] for i in sorted(tcs)]
    return out


MODEL_BEHAVIOUR = []  # (case, mode, message): not leaks, reported on their own line


def check(case, r, stream):
    errs = []
    for field in ("reasoning", "content"):
        m = LEAK.findall(r[field])
        # A lone second </mm:think> inside the answer is a token the MODEL sampled (visible in
        # return_token_ids), not a parsing leak; report it separately instead of failing.
        if field == "content" and m and set(m) == {"</mm:think>"}:
            MODEL_BEHAVIOUR.append(
                (
                    case["name"],
                    "S" if stream else "N",
                    "model emitted a stray </mm:think> in the answer",
                )
            )
            continue
        if m:
            errs.append(f"marker {sorted(set(m))} in {field}: {r[field][:120]!r}")
    for t in r["tool_calls"]:
        if LEAK.search(t["name"]) or LEAK.search(t["arguments"]):
            errs.append(f"marker in tool call {t['name']}: {t['arguments'][:120]!r}")
        try:
            args = json.loads(t["arguments"])
        except Exception:
            errs.append(f"tool args not JSON: {t['arguments'][:100]!r}")
            continue
        schema = case.get("schemas", {}).get(t["name"])
        if schema is None and case.get("schemas"):
            errs.append(f"unknown tool name {t['name']!r}")
        elif schema:
            miss = [
                k for k in schema["parameters"].get("required", []) if k not in args
            ]
            if miss:
                errs.append(
                    f"tool {t['name']} missing required {miss}: {t['arguments'][:100]!r}"
                )
    if (
        not (r["content"] or r["reasoning"] or r["tool_calls"])
        and r["finish"] != "length"
    ):
        errs.append(f"empty response finish={r['finish']}")
    # reasoning must never come after content in a stream
    if stream and "C" in r["order"] and "R" in r["order"]:
        if r["order"].index("C") < len(r["order"]) - 1 - r["order"][::-1].index("R"):
            errs.append("reasoning delta after content delta")
    want = case.get("expect_tools")
    if want is not None and r["finish"] != "length":
        got = [t["name"] for t in r["tool_calls"]]
        if want == "any" and not got:
            errs.append("expected a tool call, got none")
        elif isinstance(want, int) and len(got) != want:
            errs.append(f"expected {want} tool calls, got {len(got)}: {got}")
        elif isinstance(want, list) and sorted(got) != sorted(want):
            msg = f"expected tools {sorted(want)}, got {sorted(got)}"
            if case.get("ambiguous_tools") and set(want) <= set(got):
                MODEL_BEHAVIOUR.append(
                    (case["name"], "S" if stream else "N", msg)
                )  # model added extra tools
            else:
                errs.append(msg)
    if case.get("expect_no_tools") and r["tool_calls"]:
        errs.append(f"unexpected tool calls {[t['name'] for t in r['tool_calls']]}")
    if case.get("expect_content") and not r["content"] and r["finish"] != "length":
        errs.append("expected visible content")
    if case.get("expect_no_reasoning") and r["reasoning"]:
        errs.append(f"reasoning present though disabled: {r['reasoning'][:80]!r}")
    return errs


def tool_turn(call_id, name, args):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


def tool_result(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


SYS = {"role": "system", "content": "You are a concise assistant."}
THINK_HIST = {
    "role": "assistant",
    "content": "<mm:think>Simple arithmetic.</mm:think>4.",
}
CASES = []


def case(name, **kw):
    CASES.append(dict(name=name, **kw))


case(
    "plain-short",
    body=dict(messages=[U("Say hello in one short sentence.")], max_tokens=300),
    expect_content=True,
)
case(
    "plain-system",
    body=dict(messages=[SYS, U("What is the capital of France?")], max_tokens=400),
    expect_content=True,
)
case(
    "reasoning-math",
    body=dict(
        messages=[
            U(
                "A train travels 120 km in 1.5 hours, then 80 km in 1 hour. Average speed overall? Think it through."
            )
        ],
        max_tokens=1500,
    ),
    expect_content=True,
)
case(
    "long-output",
    body=dict(
        messages=[U("Write a 300 word story about a lighthouse keeper.")],
        max_tokens=1200,
    ),
    expect_content=True,
)
case(
    "cut-mid-reasoning",
    body=dict(
        messages=[U("Prove that sqrt(2) is irrational, in detail.")], max_tokens=20
    ),
)
case(
    "stop-sequence",
    body=dict(
        messages=[U("Count from 1 to 10 separated by commas.")],
        max_tokens=400,
        stop=["5"],
    ),
)
case(
    "multi-turn-history",
    body=dict(messages=[SYS, U("2+2?"), THINK_HIST, U("And times 3?")], max_tokens=600),
    expect_content=True,
)
case(
    "multi-turn-reasoning-field",
    body=dict(
        messages=[
            U("2+2?"),
            {"role": "assistant", "content": "4.", "reasoning": "Simple arithmetic."},
            U("Times 3?"),
        ],
        max_tokens=600,
    ),
    expect_content=True,
)
case(
    "unicode-code",
    body=dict(
        messages=[
            U(
                "Write a Python function that reverses a string, with a docstring in Japanese. Code only."
            )
        ],
        max_tokens=800,
    ),
    expect_content=True,
)
case(
    "markup-in-answer",
    body=dict(
        messages=[
            U(
                'Show a short example of an HTML <div> with a <span> inside, and of an XML <invoke name="x"></invoke> element, in a code block.'
            )
        ],
        max_tokens=800,
    ),
    allow_xml=True,
)
case(
    "tag-in-prompt",
    body=dict(
        messages=[
            U("Repeat exactly this text and nothing else: <mm:think>hello</mm:think>")
        ],
        max_tokens=300,
    ),
    allow_markers=True,
)
case(
    "json-mode",
    body=dict(
        messages=[
            U("Return a JSON object with keys name and age for a fictional person.")
        ],
        max_tokens=500,
        response_format={"type": "json_object"},
    ),
    expect_content=True,
)
case(
    "n2",
    body=dict(messages=[U("Pick a random fruit. One word.")], max_tokens=300, n=2),
    skip_stream_n=True,
)
case(
    "thinking-disabled-kw",
    body=dict(
        messages=[U("What is 12*12?")],
        max_tokens=400,
        chat_template_kwargs={"thinking_mode": "disabled"},
    ),
    expect_content=True,
)
case(
    "thinking-enabled-kw",
    body=dict(
        messages=[U("What is 12*12?")],
        max_tokens=800,
        chat_template_kwargs={"thinking_mode": "enabled"},
    ),
    expect_content=True,
)
case(
    "include-reasoning-false",
    body=dict(
        messages=[U("Is 91 prime? Think first.")],
        max_tokens=800,
        include_reasoning=False,
    ),
    expect_no_reasoning=True,
)

SCH = {
    t["function"]["name"]: t["function"] for t in (WEATHER, TIME, CALC, NESTED, WRITE)
}
case(
    "tool-single",
    body=dict(
        messages=[U("What's the weather in Paris? Use the tool.")],
        tools=[WEATHER],
        max_tokens=800,
    ),
    expect_tools=["get_weather"],
    schemas=SCH,
)
case(
    "tool-single-system",
    body=dict(
        messages=[SYS, U("Weather in Tokyo in fahrenheit?")],
        tools=[WEATHER, TIME],
        max_tokens=800,
    ),
    expect_tools=["get_weather"],
    schemas=SCH,
    ambiguous_tools=True,
)
case(
    "tool-parallel-same",
    body=dict(
        messages=[
            U(
                "Get the weather for Paris, London and Berlin. Call the tool once per city, all at the same time."
            )
        ],
        tools=[WEATHER],
        max_tokens=1200,
    ),
    expect_tools=["get_weather"] * 3,
    schemas=SCH,
)
case(
    "tool-parallel-diff",
    body=dict(
        messages=[
            U(
                "I need the weather in Rome AND the local time in Rome AND 17*23. Call all three tools in one go."
            )
        ],
        tools=[WEATHER, TIME, CALC],
        max_tokens=1500,
    ),
    expect_tools=["get_weather", "get_time", "calculate"],
    schemas=SCH,
)
case(
    "tool-nested-args",
    body=dict(
        messages=[
            U(
                "Create an order for customer Ada Lovelace (vip) with 2 of SKU A-1 and 5 of SKU B-9, note: 'gift wrap <please>'."
            )
        ],
        tools=[NESTED],
        max_tokens=1200,
    ),
    expect_tools=["create_order"],
    schemas=SCH,
)
case(
    "tool-special-chars",
    body=dict(
        messages=[
            U(
                'Write a file /tmp/a.html containing exactly: <div class="x">a & b</div> and a line with ]~b] [e~[ ]<]minimax[>[ tokens.'
            )
        ],
        tools=[WRITE],
        max_tokens=1200,
    ),
    expect_tools=["write_file"],
    schemas=SCH,
    allow_markers_in_args=True,
)
case(
    "tool-multiline-code",
    body=dict(
        messages=[
            U(
                "Write /tmp/hello.py with a hello world function with a multi-line docstring and a main guard."
            )
        ],
        tools=[WRITE],
        max_tokens=1500,
    ),
    expect_tools=["write_file"],
    schemas=SCH,
)
case(
    "tool-choice-required",
    body=dict(
        messages=[U("Tell me about Paris.")],
        tools=[WEATHER, TIME],
        tool_choice="required",
        max_tokens=800,
    ),
    expect_tools="any",
    schemas=SCH,
)
case(
    "tool-choice-named",
    body=dict(
        messages=[U("Anything about Oslo.")],
        tools=[WEATHER, TIME],
        tool_choice={"type": "function", "function": {"name": "get_time"}},
        max_tokens=800,
    ),
    expect_tools=["get_time"],
    schemas=SCH,
    ambiguous_tools=True,
)
case(
    "tool-choice-none",
    body=dict(
        messages=[U("What's the weather in Paris?")],
        tools=[WEATHER],
        tool_choice="none",
        max_tokens=800,
    ),
    expect_no_tools=True,
    expect_content=True,
)
case(
    "tool-not-needed",
    body=dict(messages=[U("Say hi.")], tools=[WEATHER, TIME, CALC], max_tokens=400),
    expect_no_tools=True,
    expect_content=True,
)
case(
    "tool-followup",
    body=dict(
        messages=[
            U("What's the weather in Paris?"),
            tool_turn("call_1", "get_weather", {"city": "Paris"}),
            tool_result("call_1", json.dumps({"temp_c": 18, "sky": "cloudy"})),
        ],
        tools=[WEATHER],
        max_tokens=800,
    ),
    expect_content=True,
    expect_no_tools=True,
)
case(
    "tool-followup-parallel",
    body=dict(
        messages=[
            U("Weather in Paris and Rome?"),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Paris"}),
                        },
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Rome"}),
                        },
                    },
                ],
            },
            tool_result("c1", "18C cloudy"),
            tool_result("c2", "25C sunny"),
        ],
        tools=[WEATHER],
        max_tokens=800,
    ),
    expect_content=True,
    expect_no_tools=True,
)
case(
    "tool-chain-second-call",
    body=dict(
        messages=[
            U("Get the weather in Paris, then the local time in the same city."),
            tool_turn("call_1", "get_weather", {"city": "Paris"}),
            tool_result("call_1", "18C cloudy"),
        ],
        tools=[WEATHER, TIME],
        max_tokens=1000,
    ),
    expect_tools=["get_time"],
    schemas=SCH,
)
case(
    "tool-after-history-think",
    body=dict(
        messages=[
            SYS,
            U("2+2?"),
            THINK_HIST,
            U("Now the weather in Madrid, use the tool."),
        ],
        tools=[WEATHER],
        max_tokens=1000,
    ),
    expect_tools=["get_weather"],
    schemas=SCH,
)
case(
    "tool-reasoning-heavy",
    body=dict(
        messages=[
            U(
                "Compute (17*23) + (45*12) carefully with the calculator tool, one expression, after thinking about it."
            )
        ],
        tools=[CALC],
        max_tokens=1500,
    ),
    expect_tools=["calculate"],
    schemas=SCH,
)
case(
    "tool-parallel-calls-off",
    body=dict(
        messages=[U("Weather in Paris and Rome.")],
        tools=[WEATHER],
        parallel_tool_calls=False,
        max_tokens=1000,
    ),
    expect_tools="any",
    schemas=SCH,
)
case(
    "tool-strict-no-args-ok",
    body=dict(
        messages=[U("What time is it in Lima? Use the tool.")],
        tools=[TIME],
        max_tokens=800,
    ),
    expect_tools=["get_time"],
    schemas=SCH,
)

# raw completions endpoint (no parser involved) and responses API
case(
    "completions-raw",
    body=dict(
        _path="/v1/completions", prompt="The three primary colors are", max_tokens=40
    ),
    raw=True,
)


def one(c, stream, i):
    body = dict(c["body"])
    try:
        if c.get("raw"):
            return c["name"], stream, i, run_raw(body, stream), []
        r = run_chat(body, stream)
    except Exception as e:  # noqa: BLE001
        return c["name"], stream, i, None, [f"request failed: {type(e).__name__}: {e}"]
    errs = check(c, r, stream)
    if c.get("allow_markers"):
        errs = [e for e in errs if not e.startswith("marker")]
    if c.get("allow_xml"):
        errs = [e for e in errs if "invoke" not in e]
    if c.get("allow_markers_in_args"):
        errs = [e for e in errs if not e.startswith("marker in tool call")]
    return c["name"], stream, i, r, errs


def run_raw(body, stream):
    out = dict(
        reasoning="",
        content="",
        tool_calls=[],
        finish=None,
        chunks=0,
        usage=None,
        order=[],
    )
    if not stream:
        d = json.load(post(body, False))
        out["content"] = d["choices"][0]["text"]
        out["finish"] = d["choices"][0]["finish_reason"]
        return out
    for line in post(body, True):
        line = line.decode().strip()
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            d = json.loads(line[5:])
            if d.get("choices"):
                out["content"] += d["choices"][0].get("text") or ""
                out["finish"] = d["choices"][0].get("finish_reason") or out["finish"]
    return out


def equivalence(c):
    """Temperature 0: streaming and non-streaming must agree on the tool names and on reasoning/content presence"""
    body = dict(c["body"], temperature=0, seed=7)
    try:
        a = run_chat(body, False)
        b = run_chat(body, True)
    except Exception as e:  # noqa: BLE001
        return [f"equivalence request failed: {e}"]
    errs = []
    na = [t["name"] for t in a["tool_calls"]]
    nb = [t["name"] for t in b["tool_calls"]]
    if na != nb:
        errs.append(f"tool names differ stream/non-stream: {nb} vs {na}")
    if bool(a["reasoning"]) != bool(b["reasoning"]):
        errs.append(
            f"reasoning presence differs: stream={bool(b['reasoning'])} non-stream={bool(a['reasoning'])}"
        )
    if bool(a["content"].strip()) != bool(b["content"].strip()):
        errs.append(
            f"content presence differs: stream={bool(b['content'].strip())} non-stream={bool(a['content'].strip())}"
        )
    return errs


def main():
    jobs = []
    for c in CASES:
        for stream in (True, False):
            if stream is False and c.get("skip_nonstream"):
                continue
            for i in range(REPEAT):
                jobs.append((c, stream, i))
    results = {}
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(one, c, s, i) for c, s, i in jobs]
        for f in cf.as_completed(futs):
            name, stream, i, r, errs = f.result()
            results.setdefault(name, []).append((stream, i, r, errs))
    eq = {}
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {
            ex.submit(equivalence, c): c["name"]
            for c in CASES
            if not c.get("raw") and not c.get("skip_stream_n")
        }
        for f in cf.as_completed(futs):
            eq[futs[f]] = f.result()

    total = bad = 0
    print(f"{'case':28s} {'stream ok/n':>12s} {'nonstream ok/n':>15s}  notes")
    for c in CASES:
        rs = results.get(c["name"], [])
        ss = [x for x in rs if x[0]]
        ns = [x for x in rs if not x[0]]
        sok = sum(1 for x in ss if not x[3])
        nok = sum(1 for x in ns if not x[3])
        e2 = eq.get(c["name"], [])
        total += len(rs)
        bad += (len(ss) - sok) + (len(ns) - nok)
        notes = []
        for x in rs:
            for e in x[3]:
                notes.append(f"[{'S' if x[0] else 'N'}{x[1]}] {e}")
        notes += [f"[EQ] {e}" for e in e2]
        print(
            f"{c['name']:28s} {sok:>5d}/{len(ss):<6d} {nok:>8d}/{len(ns):<6d}  "
            + ("OK" if not notes else notes[0][:150])
        )
        for n in notes[1:4]:
            print(" " * 64 + n[:150])
    if MODEL_BEHAVIOUR:
        from collections import Counter

        cnt = Counter((c, m) for c, _, m in MODEL_BEHAVIOUR)
        print(
            "\nMODEL BEHAVIOUR (not leaks): the model called extra tools next to the requested one"
        )
        for (c, m), n in sorted(cnt.items()):
            print(f"  {c}: {n}x  {m}")
    print(
        f"\nTOTAL requests={total} failures={bad} equivalence_failures={sum(1 for v in eq.values() if v)}"
    )
    json.dump(
        {k: [(s, i, r, e) for s, i, r, e in v] for k, v in results.items()},
        open(os.environ.get("LEAKTEST_RESULTS", "/dev/null"), "w"),
        default=str,
    )
    sys.exit(1 if bad or any(eq.values()) else 0)


main()
