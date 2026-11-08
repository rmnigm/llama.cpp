#include "kv-control.h"
#include "sampling.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <random>

int64_t common_kv_integer(const common_json & value) {
    // Bound before conversion, including unsigned JSON values. Revisions fit JS clients too.
    if (!value.is_number_integer() || value.get<double>() < -9007199254740991.0 ||
        value.get<double>() > 9007199254740991.0) {
        throw common_kv_error("invalid_request");
    }
    return value.get<int64_t>();
}

common_kv_session::common_kv_session(llama_model * model, llama_context_params params, uint32_t token_budget,
                                   size_t byte_budget, llama_pos position_limit)
    : vocab(llama_model_get_vocab(model)), position_limit(position_limit), token_budget(token_budget), byte_budget(byte_budget) {
    char arch[128] = {};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    const auto rope = llama_model_rope_type(model);
    if ((std::string(arch) != "llama" && std::string(arch) != "qwen3") || llama_model_n_swa(model) ||
        llama_model_is_recurrent(model) || llama_model_is_hybrid(model) ||
        (rope != LLAMA_ROPE_TYPE_NORM && rope != LLAMA_ROPE_TYPE_NEOX)) {
        throw common_kv_error("unsupported_architecture");
    }
    if ((params.type_k != GGML_TYPE_F16 && params.type_k != GGML_TYPE_Q8_0) || params.type_k != params.type_v) {
        throw common_kv_error("unsupported_kv_type");
    }
    if (!token_budget || token_budget > params.n_ctx || position_limit <= 0 || !byte_budget) {
        throw common_kv_error("invalid_request");
    }
    size_t required = 0;
    if (!llama_kv_control_preflight(model, params, &required)) throw common_kv_error("unsupported_configuration");
    if (required > byte_budget) throw common_kv_error("kv_budget_exceeded");
    params.n_seq_max = 1;
    params.kv_unified = true;
    params.embeddings = false;
    params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
    params.n_outputs_max = 0;
    params.n_outputs_max_per_seq = 0;
    ctx.reset(llama_init_from_model(model, params));
    if (!ctx || !llama_get_memory(ctx.get()) || !llama_memory_can_shift(llama_get_memory(ctx.get()))) {
        throw common_kv_error("engine_failure");
    }
    if (allocated_bytes() > byte_budget) {
        throw common_kv_error("kv_budget_exceeded");
    }
    std::random_device rng;
    for (int i = 0; i < 4; ++i) {
        id += std::to_string(rng()) + "-";
    }
    spans.reserve(span_limit);
}

size_t common_kv_session::allocated_bytes() const {
    size_t bytes = 0;
    for (const auto & item : llama_get_memory_breakdown(ctx.get())) {
        bytes += item.second.context;
    }
    return bytes;
}

uint32_t common_kv_session::occupied() const {
    uint32_t count = 0;
    for (const auto & s : spans) {
        if (s.live) count += s.end - s.begin;
    }
    return count;
}

common_json common_kv_session::inspect() const {
    common_json entries = common_json::array();
    for (const auto & s : spans) {
        entries.push_back(common_json{{"id", s.id}, {"begin", s.begin}, {"end", s.end},
                           {"resident", s.live}, {"stale", s.stale}, {"shifts", s.shifts}});
    }
    return {{"version", 1}, {"session", id}, {"revision", revision}, {"spans", entries},
            {"next_position", next}, {"position_limit", position_limit}, {"capacity_tokens", token_budget},
            {"allocated_capacity_tokens", llama_n_ctx(ctx.get())}, {"occupied_tokens", occupied()},
            {"reusable_tokens", token_budget - occupied()}, {"allocated_kv_bytes", allocated_bytes()},
            {"kv_byte_budget", byte_budget}, {"resident_kv_bytes", nullptr}, {"returned_to_os_bytes", 0},
            {"span_limit", span_limit}, {"shift_limit", shift_limit}, {"poisoned", poisoned},
            {"reposition_supported", true}, {"occupancy_valid", !poisoned},
            {"engine_position_max", llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0)}, {"prompt_sync", "append_only"}};
}

void common_kv_session::validate(const common_json & req, bool mutation, bool allow_poisoned) const {
    if (!req.is_object() || common_kv_integer(req.at("version")) != 1) {
        throw common_kv_error("invalid_request");
    }
    if (req.at("session").get<std::string>() != id) {
        throw common_kv_error("session_mismatch");
    }
    if (mutation) {
        const auto expected = common_kv_integer(req.at("revision"));
        if (expected != revision) throw common_kv_error("revision_conflict");
        if (poisoned && !allow_poisoned) throw common_kv_error("session_poisoned");
        if (revision >= 9007199254740991LL) throw common_kv_error("revision_limit");
    }
}

void common_kv_session::admit(size_t count, size_t new_spans) const {
    if (spans.size() + new_spans > span_limit) throw common_kv_error("span_limit");
    if (!count || count > token_budget - occupied() || count > size_t(position_limit - next)) {
        throw common_kv_error("invalid_tokens");
    }
}

std::vector<llama_token> common_kv_session::parse_tokens(const common_json & req) const {
    const auto & values = req.at("tokens");
    if (!values.is_array() || values.empty() || values.size() > token_budget) throw common_kv_error("invalid_tokens");
    std::vector<llama_token> tokens;
    tokens.reserve(values.size());
    for (const auto & value : values) {
        const auto token = common_kv_integer(value);
        if (token < 0 || token >= llama_vocab_n_tokens(vocab)) throw common_kv_error("invalid_tokens");
        tokens.push_back(llama_token(token));
    }
    return tokens;
}

void common_kv_session::decode(const std::vector<llama_token> & tokens) {
    size_t offset = 0;
    while (offset < tokens.size()) {
        const size_t count = std::min(tokens.size() - offset, size_t(llama_n_batch(ctx.get())));
        llama_batch_ext_ptr batch(llama_batch_ext_init(ctx.get()));
        if (!batch) throw common_kv_error("engine_failure");
        for (size_t i = 0; i < count; ++i) {
            const auto idx = llama_batch_ext_add_token(batch.get(), 0, tokens[offset + i]);
            const llama_pos pos = next + offset + i;
            llama_batch_ext_set_pos(batch.get(), idx, &pos);
        }
        llama_batch_ext_set_output_logits(batch.get(), count - 1, true);
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) throw common_kv_error("engine_failure");
        llama_synchronize(ctx.get());
        offset += count;
    }
}

void common_kv_session::append(const std::vector<llama_token> & tokens) {
    const bool stale = occupied() != uint32_t(next) || std::any_of(spans.begin(), spans.end(), [](const span & s) {
        return s.live && s.stale;
    });
    decode(tokens);
    const auto end = next + llama_pos(tokens.size());
    spans.push_back({id + std::to_string(spans.size()), next, end, true, stale, 0});
    next = end;
}

void common_kv_session::mark_stale(llama_pos from) {
    for (auto & s : spans) {
        if (s.live && s.end > from) s.stale = true;
    }
}

common_json common_kv_session::apply(const common_json & req) {
    const auto op = req.at("op").get<std::string>();
    for (const auto & item : req.items()) {
        const auto key = item.key();
        if (key != "version" && key != "session" && key != "revision" && key != "op" &&
            !(key == "span" && (op == "kv.drop" || op == "kv.reposition")) &&
            !(key == "position" && op == "kv.reposition") && !(key == "tokens" && op == "kv.append")) {
            throw common_kv_error("invalid_request");
        }
    }
    validate(req, op != "kv.inspect");
    if (op == "kv.inspect") return inspect();
    if (op == "kv.drop" || op == "kv.reposition") {
        const auto handle = req.at("span").get<std::string>();
        auto it = std::find_if(spans.begin(), spans.end(), [&](const span & s) { return s.id == handle; });
        if (it == spans.end()) throw common_kv_error("unknown_span");
        if (!it->live) throw common_kv_error("span_dropped");
        llama_pos target = it->begin;
        if (op == "kv.reposition") {
            const auto pos = common_kv_integer(req.at("position"));
            if (pos < 0 || pos > int64_t(position_limit) - (it->end - it->begin)) throw common_kv_error("invalid_position");
            target = llama_pos(pos);
            const auto end = pos + (it->end - it->begin);
            for (const auto & s : spans) {
                if (&s != &*it && s.live && pos < s.end && s.begin < end) throw common_kv_error("position_collision");
            }
            if (target != it->begin && it->shifts >= shift_limit) throw common_kv_error("shift_limit");
        }
        try {
            llama_synchronize(ctx.get());
            if (op == "kv.drop") {
                if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, it->begin, it->end)) throw common_kv_error("engine_failure");
                it->live = false;
                mark_stale(it->end);
            } else if (target != it->begin) {
                const auto delta = target - it->begin;
                llama_memory_seq_add(llama_get_memory(ctx.get()), 0, it->begin, it->end, delta);
                if (!llama_memory_apply_updates(ctx.get())) throw common_kv_error("engine_failure");
                mark_stale(std::min(target, it->begin));
                it->begin += delta;
                it->end += delta;
                it->stale = true;
                ++it->shifts;
            }
            next = llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) + 1;
            ++revision;
            return inspect();
        } catch (...) {
            poisoned = true;
            throw common_kv_error("engine_failure");
        }
    }
    if (op != "kv.append") throw common_kv_error("unsupported_operation");
    const auto tokens = parse_tokens(req);
    admit(tokens.size(), 1);
    try {
        append(tokens);
        ++revision;
        auto out = inspect();
        const auto * logits = llama_get_logits_ith(ctx.get(), -1);
        if (!logits) throw common_kv_error("engine_failure");
        out["logits"] = std::vector<float>(logits, logits + llama_vocab_n_tokens(vocab));
        return out;
    } catch (...) {
        poisoned = true;
        throw common_kv_error("engine_failure");
    }
}

common_json common_kv_session::generate(const common_json & req, const common_params_sampling & sampling, int32_t max_tokens) {
    validate(req, true);
    if (max_tokens < 1 || max_tokens > 4096) throw common_kv_error("invalid_request");
    const auto tokens = parse_tokens(req);
    admit(tokens.size() + size_t(max_tokens), 2);
    auto sampling_params = sampling;
    common_sampler_ptr sampler(common_sampler_init(llama_get_model(ctx.get()), sampling_params));
    if (!sampler) throw common_kv_error("invalid_request");
    try {
        append(tokens);
        for (const auto token : tokens) common_sampler_accept(sampler.get(), token, false);
        const auto begin = next;
        std::vector<llama_token> generated;
        std::string text;
        bool eos = false;
        for (int32_t i = 0; i < max_tokens; ++i) {
            const auto token = common_sampler_sample(sampler.get(), ctx.get(), -1);
            common_sampler_accept(sampler.get(), token, true);
            decode({token});
            ++next;
            generated.push_back(token);
            eos = llama_vocab_is_eog(vocab, token);
            if (eos) break;
            text += common_token_to_piece(vocab, token, false);
        }
        spans.push_back({id + std::to_string(spans.size()), begin, next, true, spans.back().stale, 0});
        ++revision;
        auto out = inspect();
        out["text"] = text;
        out["generated_tokens"] = generated;
        out["finish_reason"] = eos ? "stop" : "length";
        out["prompt_tokens"] = tokens.size();
        return out;
    } catch (...) {
        poisoned = true;
        throw common_kv_error("engine_failure");
    }
}
