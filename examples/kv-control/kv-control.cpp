#include "kv-control.h"

#include <iostream>
#include <limits>

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "q8_0")) {
        std::cerr << "usage: llama-kv-control MODEL.gguf [q8_0]\n";
        return 1;
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model_ptr model(llama_model_load_from_file(argv[1], mp));
    if (!model) return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = cp.n_ubatch = 256;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.offload_kqv = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (argc == 3) cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
    try {
        common_kv_session state(model.get(), cp, 256, std::numeric_limits<size_t>::max());
        std::cout << state.inspect().dump() << std::endl;
        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                std::cout << state.apply(common_json::parse(line)).dump() << std::endl;
            } catch (const common_json_error &) {
                std::cout << common_json({{"error", "invalid_request"}}).dump() << std::endl;
            } catch (const common_kv_error & e) {
                std::cout << common_json({{"error", e.what()}}).dump() << std::endl;
            }
        }
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
