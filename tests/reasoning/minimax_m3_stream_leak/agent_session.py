#!/usr/bin/env python3
# ruff: noqa: E501, E741, B023, SIM102, E402, SIM115
"""Long multi-turn streaming agent sessions: stream, execute fake tools, feed results back, repeat.
History is replayed the way clients do it, in two variants: content only, and content + reasoning field.
Checks every turn for protocol-marker leaks, JSON validity of tool arguments, empty turns, and that the
session converges to a final text answer."""

import concurrent.futures as cf
import json
import os
import re
import sys
import urllib.request

E = sys.argv[1]
SESSIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
LEAK = re.compile(
    r"</?mm:think>|</?think>|\]<\]minimax\[>\[|</?tool_call>|</?invoke\b|\]~b\]|\[e~\[|\]~!b\[|</?response>|thinking_instructions|<model_id>"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the test suite",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": [],
            },
        },
    },
]
FS = {
    "/app/main.py": "def add(a, b):\n    return a - b  # BUG\n",
    "/app/test_main.py": "from main import add\ndef test_add():\n    assert add(2, 3) == 5\n",
}


def fake_tool(FS, name, args):
    if name == "list_files":
        return json.dumps(sorted(FS))
    if name == "read_file":
        return FS.get(args.get("path"), "ERROR: not found")
    if name == "write_file":
        FS[args["path"]] = args["content"]
        return "ok"
    if name == "run_tests":
        return (
            "1 passed"
            if "a + b" in FS["/app/main.py"]
            else "FAILED test_add: assert -1 == 5"
        )
    return "ERROR: unknown tool"


def stream(messages):
    body = {
        "model": os.environ.get("MODEL", "minimax-m31-flash"),
        "messages": messages,
        "tools": TOOLS,
        "stream": True,
        "max_tokens": 3000,
    }
    req = urllib.request.Request(
        E + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"content-type": "application/json"},
    )
    reasoning = content = ""
    tcs = {}
    finish = None
    for line in urllib.request.urlopen(req, timeout=300):
        line = line.decode().strip()
        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
            continue
        d = json.loads(line[5:])
        if not d.get("choices"):
            continue
        ch = d["choices"][0]
        dl = ch.get("delta") or {}
        reasoning += dl.get("reasoning") or dl.get("reasoning_content") or ""
        content += dl.get("content") or ""
        for t in dl.get("tool_calls") or []:
            s = tcs.setdefault(
                t.get("index", 0), {"id": None, "name": "", "arguments": ""}
            )
            s["id"] = t.get("id") or s["id"]
            f = t.get("function") or {}
            s["name"] += f.get("name") or ""
            s["arguments"] += f.get("arguments") or ""
        finish = ch.get("finish_reason") or finish
    return reasoning, content, [tcs[i] for i in sorted(tcs)], finish


def session(n, with_reasoning):
    FS_local = dict(FS)
    msgs = [
        {
            "role": "system",
            "content": "You are a careful coding agent. Use the tools. Fix the bug, run the tests, then summarize.",
        },
        {
            "role": "user",
            "content": "The add function in /app/main.py is wrong and test_add fails. Find it, fix it, run the tests, and tell me what you changed.",
        },
    ]
    errs, turns, tool_calls = [], 0, 0
    for turn in range(14):
        try:
            r, c, tcs, fin = stream(msgs)
        except Exception as e:  # noqa: BLE001
            return (
                n,
                with_reasoning,
                turns,
                tool_calls,
                [f"request failed turn {turn}: {e}"],
                False,
            )
        turns += 1
        for field, v in (("reasoning", r), ("content", c)):
            m = LEAK.findall(v)
            if m:
                errs.append(
                    f"turn {turn}: marker {sorted(set(m))} in {field}: {v[:100]!r}"
                )
        if not (r or c or tcs) and fin != "length":
            errs.append(f"turn {turn}: empty response")
        asst = {"role": "assistant", "content": c or None}
        if with_reasoning and r:
            asst["reasoning"] = r
        if tcs:
            calls = []
            for i, t in enumerate(tcs):
                if LEAK.search(t["name"]) or LEAK.search(t["arguments"]):
                    errs.append(
                        f"turn {turn}: marker in tool call {t['name']}: {t['arguments'][:80]!r}"
                    )
                try:
                    args = json.loads(t["arguments"] or "{}")
                except Exception:
                    errs.append(f"turn {turn}: bad JSON args {t['arguments'][:80]!r}")
                    args = {}
                calls.append((t["id"] or f"call_{turn}_{i}", t["name"], args))
            asst["tool_calls"] = [
                {
                    "id": cid,
                    "type": "function",
                    "function": {"name": nm, "arguments": json.dumps(a)},
                }
                for cid, nm, a in calls
            ]
            msgs.append(asst)
            for cid, nm, a in calls:
                tool_calls += 1
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": cid,
                        "content": fake_tool(FS_local, nm, a),
                    }
                )
            continue
        msgs.append(asst)
        return n, with_reasoning, turns, tool_calls, errs, bool(c.strip())
    errs.append("did not converge in 14 turns")
    return n, with_reasoning, turns, tool_calls, errs, False


jobs = [(i, i % 2 == 1) for i in range(SESSIONS)]
bad = 0
with cf.ThreadPoolExecutor(8) as ex:
    for n, wr, turns, tc, errs, done in ex.map(lambda a: session(*a), jobs):
        flag = "OK " if not errs and done else "BAD"
        bad += flag == "BAD"
        print(
            f"session {n} history={'content+reasoning' if wr else 'content only':17s} turns={turns:2d} tool_calls={tc:2d} final_answer={done} {flag}"
        )
        for e in errs[:3]:
            print("    ", e)
print(f"\nAGENT SESSIONS {len(jobs)}  bad={bad}")
sys.exit(1 if bad else 0)
