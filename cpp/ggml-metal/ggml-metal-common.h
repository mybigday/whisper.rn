// helper functions for ggml-metal that are too difficult to implement in Objective-C

#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct wsp_ggml_tensor;
struct wsp_ggml_cgraph;

enum wsp_ggml_mem_range_type {
    MEM_RANGE_TYPE_SRC = 0,
    MEM_RANGE_TYPE_DST = 1,
};

// a helper object that can be used for reordering operations to improve concurrency
//
// the fundamental idea is that a set of tasks (either ggml ops, or something else) can run concurrently if they
//   don't write to a memory that is being read by another task or written to by another task in the set
//
// with this structure, we can add tasks to the set, setting memory constraints. we can also check if a new task
//   can be added to the set without violating the constraints (i.e. if it can be executed concurrently with the
//   tasks already in the set)
//
typedef struct wsp_ggml_mem_ranges * wsp_ggml_mem_ranges_t;

wsp_ggml_mem_ranges_t wsp_ggml_mem_ranges_init(int debug);
void wsp_ggml_mem_ranges_free(wsp_ggml_mem_ranges_t mrs);

// remove all ranges from the set
void wsp_ggml_mem_ranges_reset(wsp_ggml_mem_ranges_t mrs);

// add src or dst ranges to track
bool wsp_ggml_mem_ranges_add(wsp_ggml_mem_ranges_t mrs, const struct wsp_ggml_tensor * tensor);

// return false if:
// - new src range overlaps with any existing dst range
// - new dst range overlaps with any existing range (src or dst)
bool wsp_ggml_mem_ranges_check(wsp_ggml_mem_ranges_t mrs, const struct wsp_ggml_tensor * tensor);

// reorder the nodes in the graph to improve concurrency, while respecting fusion
//
// note: this implementation is generic and not specific to metal
//       if it proves to work well, we can start using it for other backends in the future
void wsp_ggml_graph_optimize(struct wsp_ggml_cgraph * gf);

#ifdef __cplusplus
}
#endif
