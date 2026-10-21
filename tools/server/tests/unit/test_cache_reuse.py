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
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_N_GPU_LAYERS", "0")
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
    instance.log_path = str(tmp_path / "server.log")
    instance.start()
    return instance


@pytest.mark.parametrize("edit", ["delete", "replace", "insert", "identical"])
def test_reuse_metrics(server, edit):
    prefix = list(range(10, 30))
    middle = list(range(30, 40))
    suffix = list(range(40, 80))
    base = prefix + middle + suffix
    edited = {
        "delete": prefix + suffix,
        "replace": [*prefix, 90, 91, 92, *suffix],
        "insert": [*prefix, 90, 91, 92, *middle, *suffix],
        "identical": base,
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
