#include "ggml-cpu-traits.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"

namespace ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

bool wsp_ggml_cpu_extra_compute_forward(struct wsp_ggml_compute_params * params, struct wsp_ggml_tensor * op) {
    for (auto extra : wsp_ggml_backend_cpu_get_extra_buffers_type()) {
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            if (tensor_traits && tensor_traits->compute_forward(params, op)) {
                return true;
            }
        }
    }
    return false;
}

bool wsp_ggml_cpu_extra_work_size(int n_threads, const struct wsp_ggml_tensor * op, size_t * size) {
    for (auto extra : wsp_ggml_backend_cpu_get_extra_buffers_type()) {
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            if (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                return true;
            }
        }
    }
    return false;
}
