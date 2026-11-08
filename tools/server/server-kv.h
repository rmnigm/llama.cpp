#pragma once

#include "kv-control.h"

#include <map>

// Owned by server_context; all access runs on the main task queue.
struct server_kv_control {
    llama_model * model;
    llama_context_params context_params;
    uint32_t max_tokens;
    size_t max_sessions;
    size_t max_bytes;
    std::map<std::string, std::unique_ptr<common_kv_session>> sessions;

    server_kv_control(llama_model * model, llama_context_params params, uint32_t max_tokens,
                      size_t max_sessions, size_t max_bytes);
    common_json execute(const common_json & request, const common_params_sampling & sampling, int32_t max_generate);
};
