# ruff: noqa: E501, E741, B023, SIM102, E402, SIM115
import concurrent.futures as cf
import json
import sys
import urllib.request
from collections import Counter

E = sys.argv[1]


def call(kw, stream):
    b = {
        "model": __import__("os").environ.get("MODEL", "minimax-m31-flash"),
        "messages": [
            {
                "role": "user",
                "content": "Return a JSON object with keys name and age for a fictional person.",
            }
        ],
        "max_tokens": 600,
        "response_format": {"type": "json_object"},
        "stream": stream,
    }
    if kw:
        b["chat_template_kwargs"] = kw
    r = urllib.request.urlopen(
        urllib.request.Request(
            E + "/v1/chat/completions",
            json.dumps(b).encode(),
            {"content-type": "application/json"},
        ),
        timeout=300,
    )
    if not stream:
        m = json.load(r)["choices"][0]["message"]
        return ("R" if m.get("reasoning") else "-") + ("C" if m.get("content") else "-")
    rs = c = ""
    for l in r:
        l = l.decode().strip()
        if l.startswith("data:") and l[5:].strip() != "[DONE]":
            d = json.loads(l[5:])
            if d.get("choices"):
                dl = d["choices"][0]["delta"]
                rs += dl.get("reasoning") or ""
                c += dl.get("content") or ""
    return ("R" if rs else "-") + ("C" if c else "-")


for label, kw in (
    ("adaptive (default)", None),
    ("thinking_mode=enabled", {"thinking_mode": "enabled"}),
    ("thinking_mode=disabled", {"thinking_mode": "disabled"}),
):
    with cf.ThreadPoolExecutor(8) as ex:
        s = Counter(ex.map(lambda _: call(kw, True), range(12)))
        n = Counter(ex.map(lambda _: call(kw, False), range(12)))
    print(
        f"{label:24s} stream {dict(s)}  non-stream {dict(n)}   (RC = reasoning+content, -C = content only, R- = JSON stuck in reasoning)"
    )
