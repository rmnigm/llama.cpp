#pragma once

#include "common.h"
#include "json.h"
#include "llama-cpp.h"

#include <stdexcept>

struct common_kv_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// An exclusive context. Call only from its owner's serialized task queue.
class common_kv_session {
public:
    common_kv_session(llama_model * model, llama_context_params params, uint32_t token_budget,
                      size_t byte_budget, llama_pos position_limit = INT32_MAX);
    common_json inspect() const;
    common_json apply(const common_json & request);
    common_json generate(const common_json & request, const common_params_sampling & sampling, int32_t max_tokens);
    void validate(const common_json & request, bool mutation, bool allow_poisoned = false) const;
    size_t allocated_bytes() const;

private:
    struct span {
        std::string id;
        llama_pos begin, end;
        bool live = true;
        bool stale = false;
        uint32_t shifts = 0;
    };
    llama_context_ptr ctx;
    const llama_vocab * vocab;
    std::vector<span> spans;
    std::string id;
    int64_t revision = 0;
    llama_pos next = 0;
    llama_pos position_limit;
    uint32_t token_budget;
    size_t byte_budget;
    bool poisoned = false;
    static constexpr size_t span_limit = 4096;
    static constexpr uint32_t shift_limit = 64;

    uint32_t occupied() const;
    std::vector<llama_token> parse_tokens(const common_json & request) const;
    void admit(size_t count, size_t new_spans) const;
    void append(const std::vector<llama_token> & tokens);
    void decode(const std::vector<llama_token> & tokens);
    void mark_stale(llama_pos from);
};

int64_t common_kv_integer(const common_json & value);
