#include "server-kv.h"

#include <algorithm>

server_kv_control::server_kv_control(llama_model * model, llama_context_params params, uint32_t max_tokens,
                                   size_t max_sessions, size_t max_bytes)
    : model(model), context_params(params), max_tokens(max_tokens), max_sessions(max_sessions), max_bytes(max_bytes) {}

common_json server_kv_control::execute(const common_json & req, const common_params_sampling & sampling, int32_t max_generate) {
    try {
        if (!req.is_object() || common_kv_integer(req.at("version")) != 1) throw common_kv_error("invalid_request");
        const auto op = req.at("op").get<std::string>();
        if (op == "kv.create") {
            for (const auto & item : req.items()) {
                if (item.key() != "version" && item.key() != "op" && item.key() != "capacity_tokens" &&
                    item.key() != "kv_byte_budget" && item.key() != "kv_type" && item.key() != "position_limit") {
                    throw common_kv_error("invalid_request");
                }
            }
            if (sessions.size() >= max_sessions) throw common_kv_error("session_limit");
            const auto capacity = common_kv_integer(req.at("capacity_tokens"));
            if (capacity < 1 || capacity > max_tokens) throw common_kv_error("invalid_request");
            const auto position_limit = req.contains("position_limit") ? common_kv_integer(req.at("position_limit")) : llama_model_n_ctx_train(model);
            if (position_limit < 1 || position_limit > llama_model_n_ctx_train(model)) throw common_kv_error("invalid_position");
            size_t allocated = 0;
            for (const auto & s : sessions) allocated += s.second->allocated_bytes();
            size_t budget = max_bytes - allocated;
            if (req.contains("kv_byte_budget")) {
                const auto requested = common_kv_integer(req.at("kv_byte_budget"));
                if (requested < 1) throw common_kv_error("invalid_request");
                budget = std::min(budget, size_t(requested));
            }
            if (!budget) throw common_kv_error("kv_budget_exceeded");
            auto cp = context_params;
            cp.n_ctx = capacity;
            cp.n_batch = std::min(cp.n_batch, uint32_t(capacity));
            cp.n_ubatch = std::min(cp.n_ubatch, cp.n_batch);
            cp.ctx_other = nullptr;
            cp.samplers = nullptr;
            cp.n_samplers = 0;
            const auto type = req.value("kv_type", "f16");
            if (type != "f16" && type != "q8_0") throw common_kv_error("unsupported_kv_type");
            cp.type_k = cp.type_v = type == "f16" ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            auto session = std::make_unique<common_kv_session>(model, cp, capacity, budget, llama_pos(position_limit));
            auto out = session->inspect();
            const auto inserted = sessions.emplace(out.at("session").get<std::string>(), std::move(session));
            if (!inserted.second) throw common_kv_error("engine_failure");
            return out;
        }
        const auto id = req.at("session").get<std::string>();
        auto it = sessions.find(id);
        if (it == sessions.end()) throw common_kv_error("session_mismatch");
        if (op == "kv.close") {
            for (const auto & item : req.items()) {
                if (item.key() != "op" && item.key() != "version" && item.key() != "session" && item.key() != "revision") {
                    throw common_kv_error("invalid_request");
                }
            }
            it->second->validate(req, true, true);
            sessions.erase(it);
            return {{"version", 1}, {"session", id}, {"closed", true}};
        }
        if (op == "kv.generate") return it->second->generate(req, sampling, max_generate);
        return it->second->apply(req);
    } catch (const common_json_error &) {
        return {{"error", "invalid_request"}};
    } catch (const common_kv_error & e) {
        return {{"error", e.what()}};
    } catch (const std::exception &) {
        return {{"error", "engine_failure"}};
    }
}
