// vad-cpu: a ggml backend tuned for small, latency-bound graphs (VAD models).
//
// It wraps the stock ggml-cpu backend and shares its device and host buffer
// type, so tensors never need copying between the two. On the first compute
// of a graph it builds a plan:
//   - ops it implements run on its own small-shape kernels,
//   - MUL_MAT -> ADD(bias) [-> RELU] chains are fused into one GEMM,
//   - weight matrices are packed once into a GEMM-friendly layout,
//   - anything else is delegated to ggml-cpu in contiguous node ranges.
// Later computes of the same graph replay the plan.
#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

ggml_backend_t ggml_backend_vad_cpu_init(int n_threads);
bool           ggml_backend_is_vad_cpu(ggml_backend_t backend);
const char *   ggml_backend_vad_cpu_kernels(void);   // selected ISA variant, e.g. "avx512"

// stats of the last built plan (for benchmarks / debugging)
struct ggml_vad_cpu_plan_stats {
    int n_nodes;        // nodes in the graph
    int n_steps;        // executed steps after fusion (excluding no-op views)
    int n_fused;        // nodes absorbed by fusion
    int n_delegated;    // nodes delegated to ggml-cpu
};
struct ggml_vad_cpu_plan_stats ggml_backend_vad_cpu_last_plan_stats(ggml_backend_t backend);

#ifdef __cplusplus
}
#endif
