# Isolated direct KV control prototype

Build `llama-kv-control` with the normal CMake examples. Run:

```
./build/bin/llama-kv-control /path/to/model.gguf [q8_0]
```

CPU only, fixed 256-cell cache, ordinary `llama`/`qwen3` attention. The process
prints its initial inspection (including session identity and revision), then
accepts one JSON request per line. Engine logs go to stderr.

```
{"version":1,"session":"<initial session>","revision":0,"op":"kv.append","tokens":[4,5,6]}
{"version":1,"session":"<initial session>","op":"kv.inspect"}
{"version":1,"session":"<initial session>","revision":1,"op":"kv.drop","span":"<returned span id>"}
```

Each append creates a stable span and returns logits. Drop removes its KV from
future attention without changing the transcript or rotating surviving keys.
Downstream KV stays approximate/stale. Tail rollback/replay is tested for logit
agreement. `kv.reposition` explicitly fails. Failed validation does not mutate;
engine execution failure poisons the process session until restart.

Freed cells can be reused, including at positions beyond the number of cache
slots. Positions are guarded against INT32 overflow, not model training limits;
extrapolated RoPE quality is not guaranteed. Allocated KV buffers do not shrink.
`resident_kv_bytes` is null; `returned_to_os_bytes` is zero. Tombstones remain for
stable failure behavior, bounded to 4096 appended spans, then `span_limit`.

This is separate from llama-server and changes none of its SCR paths. One process
owns one context; no shared stale KV, slots, persistence or automatic full-prompt
replay. Session IDs guard accidental addressing; they are not authentication.

Protocol, live Python transport/tests and memory probe live in `rmnigm/local_clm`
(branch `clm/kv-control-m1`): `specs/kv-control.md`, `src/local_clm/kv`,
`tests/kv/test_direct_kv.py`, `scripts/kv_probe.py`.
