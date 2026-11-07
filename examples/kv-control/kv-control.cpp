#include "llama.h"
#include "llama-memory.h"
#include "json.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct span {
    std::string id;
    llama_pos begin, end;
    bool live = true;
    bool stale = false;
};

static int64_t integer(const json & j) {
    if (!j.is_number_integer() || (j.is_number_unsigned() && j.get<uint64_t>() > INT64_MAX)) {
        throw std::runtime_error("invalid_request");
    }
    return j.get<int64_t>();
}

// This adapter owns its context exclusively. Never attach it to a server slot.
class session {
    llama_context * ctx;
    const llama_vocab * vocab;
    std::vector<span> spans;
    std::string id;
    int64_t revision = 0;
    llama_pos next = 0;
    bool poisoned = false;

public:
    session(llama_context * ctx, const llama_vocab * vocab) : ctx(ctx), vocab(vocab) {
        std::random_device rng;
        for (int i = 0; i < 4; ++i) {
            id += std::to_string(rng()) + "-";
        }
    }

    json inspect() const {
        json entries = json::array();
        int occupied = 0;
        for (const auto & s : spans) {
            entries.push_back({{"id", s.id}, {"begin", s.begin}, {"end", s.end},
                               {"resident", s.live}, {"stale", s.stale}});
            if (s.live) {
                occupied += s.end - s.begin;
            }
        }
        size_t allocated = 0;
        for (const auto & item : llama_get_memory(ctx)->memory_breakdown()) {
            allocated += item.second;
        }
        return {{"version", 1}, {"session", id}, {"revision", revision}, {"spans", entries},
                {"next_position", next}, {"position_limit", INT32_MAX}, {"capacity_tokens", llama_n_ctx(ctx)},
                {"occupied_tokens", occupied}, {"allocated_kv_bytes", allocated},
                {"resident_kv_bytes", nullptr}, {"returned_to_os_bytes", 0},
                {"span_limit", 4096}, {"poisoned", poisoned}, {"reposition_supported", false}};
    }

    json apply(const json & req) {
        if (!req.is_object() || integer(req.at("version")) != 1) {
            throw std::runtime_error("invalid_request");
        }
        if (req.at("session").get<std::string>() != id) {
            throw std::runtime_error("session_mismatch");
        }
        const auto op = req.at("op").get<std::string>();
        if (op == "kv.inspect") {
            return inspect();
        }
        if (poisoned) {
            throw std::runtime_error("session_poisoned");
        }
        if (integer(req.at("revision")) != revision) {
            throw std::runtime_error("revision_conflict");
        }
        if (op == "kv.reposition") {
            throw std::runtime_error("unsupported_operation");
        }
        if (op == "kv.drop") {
            const auto handle = req.at("span").get<std::string>();
            auto it = std::find_if(spans.begin(), spans.end(), [&](const span & s) { return s.id == handle; });
            if (it == spans.end()) {
                throw std::runtime_error("unknown_span");
            }
            if (!it->live) {
                throw std::runtime_error("span_dropped");
            }
            llama_synchronize(ctx);
            if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, it->begin, it->end)) {
                poisoned = true;
                throw std::runtime_error("engine_failure");
            }
            it->live = false;
            for (auto & s : spans) {
                if (s.live && s.begin >= it->end) {
                    s.stale = true;
                }
            }
            // Tail removal makes the next engine position the end of the last live span.
            next = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
            ++revision;
            return inspect();
        }
        if (op != "kv.append") {
            throw std::runtime_error("unsupported_operation");
        }
        if (spans.size() >= 4096) {
            throw std::runtime_error("span_limit");
        }
        int occupied = 0;
        for (const auto & s : spans) {
            if (s.live) occupied += s.end - s.begin;
        }
        const auto & values = req.at("tokens");
        if (!values.is_array() || values.empty() || values.size() > llama_n_batch(ctx) ||
            values.size() > llama_n_ctx(ctx) - size_t(occupied) ||
            values.size() > size_t(INT32_MAX - next)) {
            throw std::runtime_error("invalid_tokens");
        }
        std::vector<llama_token> tokens;
        for (const auto & value : values) {
            const auto t = integer(value);
            if (t < 0 || t >= llama_vocab_n_tokens(vocab)) {
                throw std::runtime_error("invalid_tokens");
            }
            tokens.push_back(llama_token(t));
        }
        auto * batch = llama_batch_ext_init(ctx);
        if (!batch) {
            throw std::runtime_error("engine_failure");
        }
        for (size_t i = 0; i < tokens.size(); ++i) {
            const auto idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
            const llama_pos pos = next + i;
            llama_batch_ext_set_pos(batch, idx, &pos);
        }
        llama_batch_ext_set_output_logits(batch, tokens.size() - 1, true);
        const int result = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
        llama_batch_ext_free(batch);
        if (result != 0) {
            poisoned = true;
            throw std::runtime_error("engine_failure");
        }
        llama_synchronize(ctx);
        const bool stale = std::any_of(spans.begin(), spans.end(), [](const span & s) { return s.live && s.stale; });
        // A hole also makes newly computed states approximate relative to the original transcript.
        const auto end = next + llama_pos(tokens.size());
        spans.push_back({id + std::to_string(spans.size()), next, end, true, stale || occupied != next});
        next = end;
        ++revision;
        auto response = inspect();
        const auto * logits = llama_get_logits_ith(ctx, -1);
        response["logits"] = std::vector<float>(logits, logits + llama_vocab_n_tokens(vocab));
        return response;
    }
};

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "q8_0")) {
        std::cerr << "usage: llama-kv-control MODEL.gguf [q8_0]\n";
        return 1;
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model) return 1;
    char arch[128] = {};
    llama_model_meta_val_str(model.get(), "general.architecture", arch, sizeof(arch));
    if (std::string(arch) != "llama" && std::string(arch) != "qwen3") {
        std::cerr << "unsupported architecture (initial allowlist: llama, qwen3)\n";
        return 1;
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 256;
    cp.n_ubatch = 256;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.offload_kqv = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (argc == 3) cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) return 1;
    session state(ctx.get(), llama_model_get_vocab(model.get()));
    std::cout << state.inspect().dump() << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            std::cout << state.apply(json::parse(line)).dump() << std::endl;
        } catch (const json::exception &) {
            std::cout << json({{"error", "invalid_request"}}).dump() << std::endl;
        } catch (const std::runtime_error & e) {
            std::cout << json({{"error", e.what()}}).dump() << std::endl;
        }
    }
}
