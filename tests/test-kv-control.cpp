#include "kv-control.h"
#include "../src/llama-ext.h"
#include "../src/llama-kv-cache.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static common_json call(common_kv_session & s, const std::string & op, common_json fields = common_json::object()) {
    auto request = s.inspect();
    common_json req{{"version", 1}, {"session", request.at("session")}, {"revision", request.at("revision")}, {"op", op}};
    for (const auto & item : fields.items()) req[item.key()] = item.value();
    return s.apply(req);
}

static common_json append(common_kv_session & s, std::vector<llama_token> tokens) {
    return call(s, "kv.append", {{"tokens", tokens}});
}

static float drift(const common_json & a, const common_json & b) {
    require(a.size() == b.size(), "logit size mismatch");
    float result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const auto x = a[i].get<float>();
        const auto y = b[i].get<float>();
        require(std::isfinite(x) && std::isfinite(y), "non-finite logits");
        result = std::max(result, std::abs(x - y));
    }
    return result;
}

static common_json raw_append(llama_context * ctx, const std::vector<llama_token> & tokens, llama_pos start) {
    llama_batch_ext_ptr batch(llama_batch_ext_init(ctx));
    require(bool(batch), "raw batch allocation failed");
    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto idx = llama_batch_ext_add_token(batch.get(), 0, tokens[i]);
        const llama_pos pos = start + i;
        llama_batch_ext_set_pos(batch.get(), idx, &pos);
    }
    llama_batch_ext_set_output_logits(batch.get(), tokens.size() - 1, true);
    require(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "raw decode failed");
    llama_synchronize(ctx);
    const auto * logits = llama_get_logits_ith(ctx, -1);
    return std::vector<float>(logits, logits + llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx))));
}

static void reject(common_kv_session & s, const std::string & op, const common_json & fields, const char * code) {
    const auto before = s.inspect().dump();
    bool rejected = false;
    try {
        call(s, op, fields);
    } catch (const common_kv_error & e) {
        rejected = std::string(e.what()) == code;
    }
    require(rejected, "wrong rejection");
    require(before == s.inspect().dump(), "rejection changed state");
}

static common_json generate(common_kv_session & s, const std::vector<llama_token> & tokens, int32_t max_tokens) {
    const auto state = s.inspect();
    common_params_sampling sampling;
    sampling.temp = 0;
    return s.generate({{"version", 1}, {"session", state.at("session")}, {"revision", state.at("revision")},
                       {"op", "kv.generate"}, {"tokens", tokens}}, sampling, max_tokens);
}

static void test_active_capacity(llama_model * model, llama_context_params cp, size_t budget) {
    cp.n_ctx = 17;
    size_t estimate = 0;
    require(llama_kv_control_preflight(model, cp, &estimate), "small-capacity allocation estimate failed");
    require(estimate > 0 && estimate <= budget, "invalid small-capacity allocation estimate");
    common_kv_session active(model, cp, 17, estimate);
    common_kv_session isolated(model, cp, 17, budget);
    const auto initial = active.inspect();
    require(initial.at("allocated_capacity_tokens").get<uint32_t>() > initial.at("capacity_tokens").get<uint32_t>(), "test needs spare allocated cells");
    const auto allocated = initial.at("allocated_kv_bytes");
    require(allocated.get<size_t>() > 0, "missing allocated KV measurement");
    require(allocated.get<size_t>() <= estimate, "allocation exceeded preflight estimate");
    const auto anchor = append(active, {4, 5, 6, 7, 8}).at("spans").back().at("id");
    append(isolated, {4, 5, 6, 7, 8});
    const auto isolated_before = isolated.inspect();
    const std::vector<llama_token> tail{9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    auto state = append(active, tail);
    const auto baseline = state.at("logits");
    for (int i = 0; i < 32; ++i) {
        require(state.at("occupied_tokens") == 17 && state.at("reusable_tokens") == 0, "active budget not filled");
        reject(active, "kv.append", {{"tokens", std::vector<int>{21}}}, "invalid_tokens");
        const auto dropped = state.at("spans").back().at("id");
        state = call(active, "kv.drop", {{"span", dropped}});
        require(state.at("occupied_tokens") == 5 && state.at("reusable_tokens") == 12, "drop did not restore active budget");
        reject(active, "kv.drop", {{"span", dropped}}, "span_dropped");
        state = call(active, "kv.reposition", {{"span", anchor}, {"position", (i + 1) * 257}});
        require(state.at("occupied_tokens") == 5 && state.at("reusable_tokens") == 12, "reposition changed active capacity");
        state = append(active, tail);
        require(state.at("next_position") == (i + 1) * 257 + 17, "position limited by physical cell count");
        require(state.at("engine_position_max") == state.at("next_position").get<int>() - 1, "engine position disagrees with spans");
        require(state.at("allocated_kv_bytes") == allocated, "edits resized fixed KV allocation");
        require(state.at("allocated_capacity_tokens") == initial.at("allocated_capacity_tokens"), "edits changed physical capacity");
        require(state.at("resident_kv_bytes").is_null() && state.at("returned_to_os_bytes") == 0, "logical reuse claimed physical reclamation");
        require(drift(baseline, state.at("logits")) < 1e-3f, "repeated translation and refill changed logits");
        require(isolated.inspect() == isolated_before, "edits changed another session");
    }
    require(drift(baseline, append(isolated, tail).at("logits")) < 1e-5f, "edits changed another session's logits");
}

static void test_generation_capacity(llama_model * model, llama_context_params cp, size_t budget) {
    common_kv_session session(model, cp, 5, budget);
    append(session, {4, 5, 6});
    const auto before = session.inspect();
    bool rejected = false;
    try { generate(session, {7}, 2); }
    catch (const common_kv_error & e) { rejected = std::string(e.what()) == "invalid_tokens"; }
    require(rejected, "generation failed to reserve maximum output");
    require(session.inspect() == before, "rejected generation appended prompt delta");
    const auto generated = generate(session, {7}, 1);
    require(generated.at("revision") == before.at("revision").get<int>() + 1, "generation advanced revision more than once");
    require(generated.at("occupied_tokens") == 5 && generated.at("reusable_tokens") == 0, "final generated token not retained");
    const auto & output = generated.at("spans").back();
    require(output.at("begin") == 4 && output.at("end") == 5, "wrong generated span boundaries");
    require(generated.at("engine_position_max") == 4, "final generated token not decoded");
    call(session, "kv.drop", {{"span", output.at("id")}});
    const auto replay = append(session, generated.at("generated_tokens").get<std::vector<llama_token>>());
    require(replay.at("occupied_tokens") == 5, "generated cell could not be reused");
}

static void test_batched_append(llama_model * model, llama_context_params cp, size_t budget) {
    // Keep microbatches equal so quantized GEMV/GEMM differences do not mask wrapper errors.
    cp.n_ubatch = 4;
    common_kv_session reference(model, cp, 65, budget);
    cp.n_batch = 8;
    common_kv_session batched(model, cp, 65, budget);
    std::vector<llama_token> tokens;
    for (int i = 0; i < 65; ++i) tokens.push_back(4 + i % 17);
    const auto expected = append(reference, tokens);
    const auto actual = append(batched, tokens);
    require(actual.at("revision") == 1 && actual.at("spans").size() == 1, "batching split one append mutation");
    require(actual.at("occupied_tokens") == 65 && actual.at("engine_position_max") == 64, "batched append lost cells");
    require(drift(expected.at("logits"), actual.at("logits")) < 1e-4f, "batched append changed logits");
    call(batched, "kv.drop", {{"span", actual.at("spans")[0].at("id")}});
    require(batched.inspect().at("engine_position_max") == -1, "drop left batched cells in attention");
    require(drift(actual.at("logits"), append(batched, tokens).at("logits")) < 1e-5f, "batched refill changed logits");
}

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params();
    const char * gpu_layers = std::getenv("CLM_KV_N_GPU_LAYERS");
    mp.n_gpu_layers = gpu_layers ? std::stoi(gpu_layers) : 0;
    if (mp.n_gpu_layers > 0 && (!llama_supports_gpu_offload() ||
        (!ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) && !ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU)))) {
        std::fprintf(stderr, "GPU test requested but no GPU offload backend/device is available\n");
        return 2;
    }
    llama_model_ptr model(llama_model_load_from_file(argv[1], mp));
    if (!model) return 2;
    auto cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = cp.n_ubatch = 256;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.offload_kqv = mp.n_gpu_layers != 0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (std::string(argv[2]) == "q8_0") cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
    const size_t budget = 16 * 1024 * 1024;
    if (std::string(argv[2]) == "unsupported") {
        try {
            common_kv_session unsupported(model.get(), cp, 256, budget);
        } catch (const common_kv_error & e) {
            return std::string(e.what()) == "unsupported_architecture" ? 0 : 1;
        }
        return 1;
    }
    try {
        test_active_capacity(model.get(), cp, budget);
        test_generation_capacity(model.get(), cp, budget);
        test_batched_append(model.get(), cp, budget);
        auto scaled_cp = cp;
        scaled_cp.yarn_attn_factor = 1.1f;
        bool scaling_rejected = false;
        try { common_kv_session scaled(model.get(), scaled_cp, 256, budget); }
        catch (const common_kv_error & e) { scaling_rejected = std::string(e.what()) == "unsupported_configuration"; }
        require(scaling_rejected, "nonunit RoPE magnitude accepted");
        scaled_cp = cp;
        scaled_cp.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
        scaling_rejected = false;
        try { common_kv_session scaled(model.get(), scaled_cp, 256, budget); }
        catch (const common_kv_error & e) { scaling_rejected = std::string(e.what()) == "unsupported_configuration"; }
        require(scaling_rejected, "unverified YaRN accepted");
        scaled_cp = cp;
        scaled_cp.moe_cache_size = 1024 * 1024;
        scaling_rejected = false;
        try { common_kv_session scaled(model.get(), scaled_cp, 256, budget); }
        catch (const common_kv_error & e) { scaling_rejected = std::string(e.what()) == "unsupported_configuration"; }
        require(scaling_rejected, "unbudgeted MoE cache accepted");
        common_kv_session a(model.get(), cp, 256, budget);
        common_kv_session b(model.get(), cp, 256, budget);
        require(a.inspect().at("session") != b.inspect().at("session"), "session ID reused");
        append(a, {4, 5, 6, 7});
        append(b, {4, 5, 6, 7});
        const auto baseline = append(a, {8, 9, 10}).at("logits");
        for (int i = 0; i < 100; ++i) {
            const auto handle = a.inspect().at("spans").back().at("id");
            call(a, "kv.drop", {{"span", handle}});
            const auto replay = append(a, {8, 9, 10});
            require(drift(baseline, replay.at("logits")) < 1e-5f, "tail replay changed logits");
        }
        require(b.inspect().at("occupied_tokens") == 4, "another session changed");
        require(drift(baseline, append(b, {8, 9, 10}).at("logits")) < 1e-5f, "another session's logits changed");
        reject(b, "kv.drop", {{"span", a.inspect().at("spans")[0].at("id")}}, "unknown_span");
        reject(a, "kv.append", {{"tokens", std::vector<int>{-1}}}, "invalid_tokens");
        reject(a, "kv.drop", {{"span", "missing"}, {"revision", -1}}, "revision_conflict");
        const auto handle = a.inspect().at("spans")[0].at("id");
        reject(a, "kv.reposition", {{"span", handle}, {"position", -1}}, "invalid_position");
        reject(a, "kv.reposition", {{"span", handle}, {"position", 5}}, "position_collision");
        reject(a, "kv.reposition", {{"span", handle}, {"position", INT64_MAX}}, "invalid_request");
        reject(a, "kv.drop", {{"span", handle}, {"prompt", "resurrect"}}, "invalid_request");

        // Uniform translation preserves relative RoPE attention. Missing re-rotation fails this probe.
        common_kv_session shifted(model.get(), cp, 256, budget);
        append(shifted, {4, 5, 6, 7});
        const auto moving = shifted.inspect().at("spans")[0].at("id");
        call(shifted, "kv.reposition", {{"span", moving}, {"position", 19}});
        require(shifted.inspect().at("next_position") == 23, "wrong shifted position");
        auto shifted_logits = append(shifted, {8, 9, 10}).at("logits");
        const float shift_drift = drift(baseline, shifted_logits);
        require(shift_drift < 1e-4f, "uniform RoPE translation drift");
        // Negative control: relocate metadata, then deliberately restore the unrotated keys.
        llama_context_ptr broken(llama_init_from_model(model.get(), cp));
        require(bool(broken), "negative control context failed");
        raw_append(broken.get(), {4, 5, 6, 7}, 0);
        auto * cache = static_cast<llama_kv_cache *>(llama_get_memory(broken.get()));
        std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> saved_keys;
        for (const auto il : cache->get_layer_ids()) {
            auto * key = cache->get_k_storage(il);
            std::vector<uint8_t> bytes(ggml_nbytes(key));
            ggml_backend_tensor_get(key, bytes.data(), 0, bytes.size());
            saved_keys.emplace_back(key, std::move(bytes));
        }
        llama_memory_seq_add(llama_get_memory(broken.get()), 0, 0, 4, 19);
        require(llama_memory_apply_updates(broken.get()), "negative control shift failed");
        for (const auto & saved : saved_keys) ggml_backend_tensor_set(saved.first, saved.second.data(), 0, saved.second.size());
        const float unrotated_drift = drift(baseline, raw_append(broken.get(), {8, 9, 10}, 23));
        require(unrotated_drift > 2 * shift_drift + 1e-7f, "RoPE probe cannot detect missing key rotation");

        call(shifted, "kv.drop", {{"span", shifted.inspect().at("spans").back().at("id")}});
        for (int i = 0; i < 31; ++i) {
            call(shifted, "kv.reposition", {{"span", moving}, {"position", 0}});
            call(shifted, "kv.reposition", {{"span", moving}, {"position", 19}});
        }
        call(shifted, "kv.reposition", {{"span", moving}, {"position", 0}});
        const auto repeated = append(shifted, {8, 9, 10});
        const float repeated_drift = drift(baseline, repeated.at("logits"));
        require(repeated_drift < 1e-3f, "repeated quantized shift drift");
        reject(shifted, "kv.reposition", {{"span", moving}, {"position", 20}}, "shift_limit");

        common_kv_session capacity(model.get(), cp, 256, budget);
        append(capacity, std::vector<llama_token>(128, 4));
        const auto first = capacity.inspect().at("spans")[0].at("id");
        append(capacity, std::vector<llama_token>(128, 5));
        const auto allocated = capacity.allocated_bytes();
        call(capacity, "kv.drop", {{"span", first}});
        require(capacity.inspect().at("reusable_tokens") == 128, "cells not reclaimed");
        require(capacity.allocated_bytes() == allocated, "drop resized fixed allocation");
        append(capacity, std::vector<llama_token>(128, 6));
        require(capacity.inspect().at("next_position") == 384, "occupancy confused with positions");
        reject(capacity, "kv.append", {{"tokens", std::vector<int>{7}}}, "invalid_tokens");

        common_kv_session stale(model.get(), cp, 256, budget);
        append(stale, {4, 5});
        const auto middle = append(stale, {12, 13}).at("spans").back().at("id");
        append(stale, {20, 21});
        const auto before = append(stale, {30});
        call(stale, "kv.drop", {{"span", before.at("spans").back().at("id")}});
        call(stale, "kv.drop", {{"span", middle}});
        require(stale.inspect().at("spans")[2].at("stale") == true, "downstream stale flag lost");
        const auto after = append(stale, {30});
        require(drift(before.at("logits"), after.at("logits")) > 1e-6f, "drop did not change future attention");
        require(after.at("spans").back().at("stale") == true, "staleness not inherited");

        bool budget_rejected = false;
        try { common_kv_session too_big(model.get(), cp, 256, 1); }
        catch (const common_kv_error & e) { budget_rejected = std::string(e.what()) == "kv_budget_exceeded"; }
        require(budget_rejected, "byte budget ignored");
        common_kv_session limited(model.get(), cp, 3, budget, 4);
        append(limited, {4, 5, 6});
        reject(limited, "kv.append", {{"tokens", std::vector<int>{7}}}, "invalid_tokens");
        reject(limited, "kv.reposition", {{"span", limited.inspect().at("spans")[0].at("id")}, {"position", 2}}, "invalid_position");

        cp.abort_callback = [](void *) { return true; };
        common_kv_session aborted(model.get(), cp, 256, budget);
        bool failed = false;
        try { append(aborted, {4, 5, 6}); }
        catch (const common_kv_error & e) { failed = std::string(e.what()) == "engine_failure"; }
        require(failed && aborted.inspect().at("poisoned") == true, "failed decode not poisoned");
        reject(aborted, "kv.append", {{"tokens", std::vector<int>{4}}}, "session_poisoned");
        require(aborted.inspect().at("occupancy_valid") == false, "partial decode occupancy reported valid");
        bool abort_shift = false;
        cp.abort_callback = [](void * flag) { return *static_cast<bool *>(flag); };
        cp.abort_callback_data = &abort_shift;
        common_kv_session failed_shift(model.get(), cp, 256, budget);
        const auto shift_handle = append(failed_shift, {4, 5, 6}).at("spans")[0].at("id");
        abort_shift = true;
        failed = false;
        try { call(failed_shift, "kv.reposition", {{"span", shift_handle}, {"position", 12}}); }
        catch (const common_kv_error & e) { failed = std::string(e.what()) == "engine_failure"; }
        require(failed && failed_shift.inspect().at("poisoned") == true, "failed RoPE shift not poisoned");

        std::printf("{\"kv_type\":\"%s\",\"shift_max_logit_drift\":%.9g,\"64_shifts_max_logit_drift\":%.9g,\"unrotated_max_logit_drift\":%.9g,\"passed\":true}\n", argv[2], shift_drift, repeated_drift, unrotated_drift);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAILED: %s\n", e.what());
        return 1;
    }
    return 0;
}
