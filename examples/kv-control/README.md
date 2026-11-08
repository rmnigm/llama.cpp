# Direct KV control (fork experiment)

`common/kv-control.{h,cpp}` implements exclusive cache sessions with inspect,
append, drop and collision-checked positional relocation. Keys are re-rotated
synchronously before reposition succeeds. Stable handles, revision checks,
tombstones, bounded metadata and poisoned-state recovery apply to both transports.

This JSONL example is a CPU compatibility transport:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=ON -DGGML_NATIVE=OFF
cmake --build build --target llama-kv-control llama-server test-kv-control test-llama-archs llama-quantize -j 3
ctest --test-dir build -R '^test-kv-' --output-on-failure
build/bin/llama-kv-control /existing/model.gguf q8_0
```

The first JSONL response is inspect. Send one JSON object per line, with
`version:1`, its `session`, `op`, and the current `revision` for mutations.
`kv.append` accepts token IDs; `kv.drop` accepts a stable `span`; `kv.reposition`
accepts `span` and a destination start `position`. Append returns logits. Errors
are explicit `{"error":"code"}` values. The executable owns 256 CPU cache cells,
with no prompt restore or sharing. Its historical INT32_MAX position limit does
not establish quality outside the model's training range.

## llama-server

Opt in with `--kv-control`. `POST /v1/kv` creates/inspects/mutates/closes separately
owned contexts, serialized by the server task queue. Normal prefix/greedy/SCR
slots are unchanged. Direct continuation uses `/v1/completions` with nonempty
**token deltas** and `kv_control:{version:1,session:...,revision:...,mode:"append"}`.
Full direct-KV chat replay rejects; no original prompt/checkpoint is restored.
Direct generation is synchronous and bounded to 4096 output tokens, so it can
delay ordinary tasks on the shared task queue. Streaming and full chat-template
reconciliation are not implemented.

Ordinary full-attention llama/Qwen3, unit-magnitude normal/NeoX RoPE (none/linear
scaling), F16 and q8_0 K/V are supported. Other architectures, YaRN/nonunit RoPE,
SWA/recurrent/hybrid, multimodal, LoRA/control vectors, expert-cache and draft
configurations explicitly reject. Drop preserves surviving positions and marks
downstream KV stale. Reposition preserves old causal history; it is approximate
relative to exact text re-prefill. Drop is not secure erasure.

`--kv-control-max-sessions` (4) and `--kv-control-max-mib` (256) bound owned sessions
and their allocated KV, with preflight and post-allocation checks. These limits
exclude model weights, standard server KV and compute/allocator overhead. Drop
reclaims reusable cells, not resident RAM; fixed tensor allocation stays constant.
Close releases context ownership but RSS can remain high. Per-KV resident bytes
are unknown, not estimated from occupancy.

Canonical protocol, errors, conformance and exact CPU/CUDA/Metal commands:
[local_clm specification](https://github.com/rmnigm/local_clm/blob/clm/kv-control-hardening/specs/kv-control.md)
and [runbook](https://github.com/rmnigm/local_clm/blob/clm/kv-control-hardening/wiki/runbooks/direct-kv-smoke.md).
Generated CPU fixtures pass; CUDA/Metal and pretrained Qwen3-4B remain unverified.
