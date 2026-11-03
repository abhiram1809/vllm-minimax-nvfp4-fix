# MiniMax-M3 streaming leak checks

Server-level checks (not collected by pytest) for reasoning/tool-call markers leaking into `delta.content`.
Run them against a running `vllm serve` with `--reasoning-parser minimax_m3 --tool-call-parser minimax_m3
--enable-auto-tool-choice`:

```bash
MODEL=<served-model-name> python tests/reasoning/minimax_m3_stream_leak/leaktest.py http://HOST:PORT 5
MODEL=<served-model-name> python tests/reasoning/minimax_m3_stream_leak/agent_session.py http://HOST:PORT 12
MODEL=<served-model-name> python tests/reasoning/minimax_m3_stream_leak/gate_modes.py http://HOST:PORT
MODEL_PATH=<checkpoint dir> python tests/reasoning/minimax_m3_stream_leak/repro_parser_unit.py   # no server needed
```

- `leaktest.py`: 36 cases x stream/non-stream x N repeats (plain, reasoning, multi-turn history with think
  blocks, thinking on/off, `n=2`, JSON mode, stop sequences, every `tool_choice` mode, parallel and nested tool
  calls, special characters in arguments, tool follow-ups, raw completions). Fails on any protocol marker in
  content, reasoning, or tool-call name/arguments. A lone extra `</mm:think>` is reported as model behaviour.
- `agent_session.py`: streaming agent loops that execute fake tools and replay history (content only, and
  content + reasoning).
- `gate_modes.py`: `response_format=json_object` under adaptive / enabled / disabled thinking; prints where
  the JSON lands (`-C` content, `R-` stuck in reasoning).
- `repro_parser_unit.py`: drives `DelegatingParser.parse_delta` directly with a real tokenizer.
