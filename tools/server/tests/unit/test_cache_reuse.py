import json
import os
from pathlib import Path

import pytest
from utils import ServerProcess


@pytest.fixture(scope="session", autouse=True)
def load_server_presets():
    # Override the download fixture: these tests use test-generate-models only.
    pass


@pytest.fixture
def server(tmp_path, monkeypatch, reuse_mode, cache_type):
    monkeypatch.setenv("LLAMA_ARG_N_GPU_LAYERS", "0")
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "none")
    monkeypatch.setenv("LLAMA_ARG_CACHE_REUSE_MODE", reuse_mode)
    root = Path(__file__).resolve().parents[4]
    binary = Path(os.environ.get("LLAMA_SERVER_BIN_PATH", root / "build/bin/llama-server"))
    model = binary.parent.parent / "tests/test-models/llama-dense.gguf"
    assert model.is_file(), "Run ctest --test-dir build -R '^test-generate-models$' first"
    instance = ServerProcess()
    instance.server_path = str(binary)
    instance.model_file = str(model)
    instance.model_hf_repo = None
    instance.model_hf_file = None
    instance.offline = True
    instance.n_slots = 1
    instance.n_ctx = 512
    instance.n_batch = 32
    instance.n_ubatch = 32
    instance.cache_ram = 0
    instance.ctk = cache_type
    instance.log_path = str(tmp_path / "server.log")
    instance.start()
    return instance


@pytest.fixture(params=["greedy", "scr"])
def reuse_mode(request):
    return request.param


@pytest.fixture(params=["f16", "q8_0"])
def cache_type(request):
    return request.param


@pytest.mark.parametrize("edit", ["delete", "replace", "equal-replace", "insert", "identical", "reorder", "gaps"])
def test_reuse_metrics(server, edit, reuse_mode):
    prefix = list(range(10, 30))
    middle = list(range(30, 40))
    suffix = list(range(40, 80))
    base = prefix + middle + suffix
    edited = {
        "delete": prefix + suffix,
        "replace": [*prefix, 90, 91, 92, *suffix],
        "equal-replace": prefix + [90] * len(middle) + suffix,
        "insert": [*prefix, 90, 91, 92, *middle, *suffix],
        "identical": base,
        "reorder": prefix + suffix + middle,
        "gaps": prefix + [90] * 40 + middle + [91] * 40 + suffix,
    }[edit]
    responses = []
    for prompt in (base, edited):
        response = server.make_request(
            "POST",
            "/completion",
            {
                "prompt": prompt,
                "n_predict": 1,
                "temperature": 0,
                "cache_prompt": True,
                "n_cache_reuse": 4,
            },
        )
        assert response.status_code == 200
        timings = response.body["timings"]
        assert timings["n_reused_prefix"] + timings["n_reused_shift"] + timings["n_prefilled"] == len(prompt)
        assert timings["n_prefilled"] == timings["prompt_n"]
        assert timings["n_reused_prefix"] + timings["n_reused_shift"] == timings["cache_n"]
        responses.append(timings)
    assert responses[0]["n_prefilled"] == len(base)
    timings = responses[1]
    if edit == "delete":
        assert timings["n_reused_prefix"] == len(prefix)
        assert timings["n_reused_shift"] == len(suffix) - 1
        assert timings["n_spans_relocated"] == 1
        assert timings["n_prefilled"] == 1
    elif edit == "identical":
        assert timings["n_reused_prefix"] == len(base) - 1
        assert timings["n_reused_shift"] == 0
        assert timings["n_prefilled"] == 1
    elif reuse_mode == "scr":
        expected = {
            "replace": (39, 1, 4),
            "equal-replace": (39, 1, 11),
            "insert": (49, 1, 4),
            "reorder": (49, 2, 1),
            "gaps": (49, 2, 81),
        }
        shifted, spans, prefilled = expected[edit]
        assert timings["n_reused_prefix"] == len(prefix)
        assert timings["n_reused_shift"] == shifted
        assert timings["n_spans_relocated"] == spans
        assert timings["n_prefilled"] == prefilled
    elif edit == "reorder":
        assert timings["n_reused_prefix"] == len(prefix)
        assert timings["n_reused_shift"] == len(suffix)
        assert timings["n_prefilled"] == len(middle)
    else:
        assert timings["n_reused_prefix"] == len(prefix)
        assert timings["n_reused_shift"] == 0
        assert timings["n_spans_relocated"] == 0
        assert timings["n_prefilled"] == len(edited) - len(prefix)
    server.stop()
    records = [
        json.loads(line.split("clm_reuse ", 1)[1])
        for line in Path(server.log_path).read_text().splitlines()
        if "clm_reuse " in line
    ]
    assert len(records) == 2
    for record, timings in zip(records, responses, strict=True):
        assert record.pop("id_slot") == 0
        assert record == {
            key: timings[key]
            for key in (
                "n_reused_prefix",
                "n_reused_shift",
                "n_prefilled",
                "n_spans_relocated",
                "prompt_ms",
            )
        }


def test_streamed_completion_metrics(server):
    base = list(range(10, 80))
    body = {"prompt": base, "max_tokens": 1, "temperature": 0, "cache_prompt": True, "n_cache_reuse": 4}
    assert server.make_request("POST", "/v1/completions", body).status_code == 200
    chunks = list(
        server.make_stream_request(
            "POST",
            "/v1/completions",
            {
                **body,
                "prompt": base[:20] + base[30:],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
    )
    timings = next(chunk["timings"] for chunk in reversed(chunks) if "timings" in chunk)
    usage = next(chunk["usage"] for chunk in reversed(chunks) if chunk.get("usage"))
    assert timings["n_reused_prefix"] == 20
    assert timings["n_reused_shift"] == 39
    assert timings["n_prefilled"] == 1
    assert timings["n_spans_relocated"] == 1
    assert usage["prompt_tokens"] == 60


@pytest.mark.parametrize("reuse_mode", ["scr"], indirect=True)
def test_shared_cache_falls_back_to_prefix(server):
    server.stop()
    server.n_slots = 2
    server.kv_unified = True
    server.start()
    base = list(range(10, 80))
    for prompt in (base, [*base[:20], 90, 91, 92, *base[30:]]):
        response = server.make_request(
            "POST",
            "/completion",
            {
                "prompt": prompt,
                "n_predict": 1,
                "temperature": 0,
                "cache_prompt": True,
                "n_cache_reuse": 4,
                "id_slot": 0,
            },
        )
        assert response.status_code == 200
    timings = response.body["timings"]
    assert timings["n_reused_prefix"] == 20
    assert timings["n_reused_shift"] == 0
    assert timings["n_spans_relocated"] == 0
    assert timings["n_prefilled"] == 43
