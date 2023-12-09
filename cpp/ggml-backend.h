#pragma once

#include "ggml.h"
#include "ggml-alloc.h"

#ifdef  __cplusplus
extern "C" {
#endif

    typedef struct wsp_ggml_backend_buffer_type * wsp_ggml_backend_buffer_type_t;
    typedef struct wsp_ggml_backend_buffer * wsp_ggml_backend_buffer_t;
    typedef struct wsp_ggml_backend * wsp_ggml_backend_t;
    typedef void * wsp_ggml_backend_graph_plan_t;

    //
    // Backend buffer
    //

    // buffer type
    WSP_GGML_API wsp_ggml_backend_buffer_t wsp_ggml_backend_buft_alloc_buffer(wsp_ggml_backend_buffer_type_t buft, size_t size);
    WSP_GGML_API size_t wsp_ggml_backend_buft_get_alignment (wsp_ggml_backend_buffer_type_t buft);
    WSP_GGML_API size_t wsp_ggml_backend_buft_get_alloc_size(wsp_ggml_backend_buffer_type_t buft, struct wsp_ggml_tensor * tensor);
    WSP_GGML_API bool wsp_ggml_backend_buft_supports_backend(wsp_ggml_backend_buffer_type_t buft, wsp_ggml_backend_t backend);

    // buffer
    WSP_GGML_API void   wsp_ggml_backend_buffer_free          (wsp_ggml_backend_buffer_t buffer);
    WSP_GGML_API void * wsp_ggml_backend_buffer_get_base      (wsp_ggml_backend_buffer_t buffer);
    WSP_GGML_API size_t wsp_ggml_backend_buffer_get_size      (wsp_ggml_backend_buffer_t buffer);
    WSP_GGML_API void   wsp_ggml_backend_buffer_init_tensor   (wsp_ggml_backend_buffer_t buffer, struct wsp_ggml_tensor * tensor);
    WSP_GGML_API size_t wsp_ggml_backend_buffer_get_alignment (wsp_ggml_backend_buffer_t buffer);
    WSP_GGML_API size_t wsp_ggml_backend_buffer_get_alloc_size(wsp_ggml_backend_buffer_t buffer, struct wsp_ggml_tensor * tensor);
    WSP_GGML_API wsp_ggml_backend_buffer_type_t wsp_ggml_backend_buffer_type(wsp_ggml_backend_buffer_t buffer);

    //
    // Backend
    //


    WSP_GGML_API const char * wsp_ggml_backend_name(wsp_ggml_backend_t backend);
    WSP_GGML_API void         wsp_ggml_backend_free(wsp_ggml_backend_t backend);

    WSP_GGML_API wsp_ggml_backend_buffer_type_t wsp_ggml_backend_get_default_buffer_type(wsp_ggml_backend_t backend);
    WSP_GGML_API wsp_ggml_backend_buffer_t      wsp_ggml_backend_alloc_buffer(wsp_ggml_backend_t backend, size_t size);
    WSP_GGML_API size_t                     wsp_ggml_backend_get_alignment(wsp_ggml_backend_t backend);

    WSP_GGML_API void wsp_ggml_backend_tensor_set_async(wsp_ggml_backend_t backend,       struct wsp_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    WSP_GGML_API void wsp_ggml_backend_tensor_get_async(wsp_ggml_backend_t backend, const struct wsp_ggml_tensor * tensor,       void * data, size_t offset, size_t size);

    WSP_GGML_API void wsp_ggml_backend_tensor_set(      struct wsp_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    WSP_GGML_API void wsp_ggml_backend_tensor_get(const struct wsp_ggml_tensor * tensor,       void * data, size_t offset, size_t size);

    WSP_GGML_API void wsp_ggml_backend_synchronize(wsp_ggml_backend_t backend);

    WSP_GGML_API wsp_ggml_backend_graph_plan_t wsp_ggml_backend_graph_plan_create (wsp_ggml_backend_t backend, struct wsp_ggml_cgraph * cgraph);

    WSP_GGML_API void wsp_ggml_backend_graph_plan_free   (wsp_ggml_backend_t backend, wsp_ggml_backend_graph_plan_t plan);
    WSP_GGML_API void wsp_ggml_backend_graph_plan_compute(wsp_ggml_backend_t backend, wsp_ggml_backend_graph_plan_t plan);
    WSP_GGML_API void wsp_ggml_backend_graph_compute     (wsp_ggml_backend_t backend, struct wsp_ggml_cgraph * cgraph);
    WSP_GGML_API bool wsp_ggml_backend_supports_op       (wsp_ggml_backend_t backend, const struct wsp_ggml_tensor * op);

    // tensor copy between different backends
    WSP_GGML_API void wsp_ggml_backend_tensor_copy(struct wsp_ggml_tensor * src, struct wsp_ggml_tensor * dst);
    WSP_GGML_API void wsp_ggml_backend_tensor_copy_async(wsp_ggml_backend_t backend, struct wsp_ggml_tensor * src, struct wsp_ggml_tensor * dst); // automatic fallback to sync copy

    //
    // CPU backend
    //

    WSP_GGML_API wsp_ggml_backend_t wsp_ggml_backend_cpu_init(void);

    WSP_GGML_API bool wsp_ggml_backend_is_cpu(wsp_ggml_backend_t backend);
    WSP_GGML_API void wsp_ggml_backend_cpu_set_n_threads(wsp_ggml_backend_t backend_cpu, int n_threads);

    // Create a backend buffer from an existing pointer
    WSP_GGML_API wsp_ggml_backend_buffer_t wsp_ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size);

    WSP_GGML_API wsp_ggml_backend_buffer_type_t wsp_ggml_backend_cpu_buffer_type(void);

    //
    // Backend registry
    //

    // The backend registry is a registry of all the available backends, and allows initializing backends in a generic way

    WSP_GGML_API size_t                     wsp_ggml_backend_reg_get_count(void);
    WSP_GGML_API size_t                     wsp_ggml_backend_reg_find_by_name(const char * name);
    WSP_GGML_API wsp_ggml_backend_t             wsp_ggml_backend_reg_init_backend_from_str(const char * backend_str); // str is name[:params]
    WSP_GGML_API const char *               wsp_ggml_backend_reg_get_name(size_t i);
    WSP_GGML_API wsp_ggml_backend_t             wsp_ggml_backend_reg_init_backend(size_t i, const char * params); // params is backend-specific
    WSP_GGML_API wsp_ggml_backend_buffer_type_t wsp_ggml_backend_reg_get_default_buffer_type(size_t i);
    WSP_GGML_API wsp_ggml_backend_buffer_t      wsp_ggml_backend_reg_alloc_buffer(size_t i, size_t size);

    //
    // Backend scheduler
    //

    // The backend scheduler allows for multiple backends to be used together
    // Handles compute buffer allocation, assignment of tensors to backends, and copying of tensors between backends
    // The backends are selected based on:
    // - the backend that supports the operation
    // - the location of the pre-allocated tensors (e.g. the weights)
    /*
      Example usage:

        sched = wsp_ggml_backend_sched_new({backend_gpu, backend_gpu2, backend_cpu}, num_backends);
        // sched is initialized with measure allocators and cannot be used until allocated with a measure graph

        // initialize buffers from a measure graph
        measure_graph = build_graph(sched); // use the allocr to allocate inputs as needed

        // in build_graph:
        build_graph(...) {
            // allocating tensors in a specific backend (optional, recommended: pre-allocate inputs in a different buffer)
            alloc_cpu = wsp_ggml_backend_sched_get_allocr(sched, backend_cpu);
            wsp_ggml_allocr_alloc(alloc_cpu, tensor);

            // manually assigning nodes to a backend (optional, shouldn't be needed in most cases)
            struct wsp_ggml_tensor * node = wsp_ggml_mul_mat(ctx, ...);
            wsp_ggml_backend_sched_set_node_backend(sched, node, backend_gpu);
        }

        // allocate backend buffers from measure graph
        wsp_ggml_backend_sched_init_measure(sched, measure_graph);

        // the scheduler is now ready to compute graphs

        // compute
        graph = build_graph(sched);
        wsp_ggml_backend_sched_graph_compute(sched, graph);
    */

    struct wsp_ggml_backend_sched;
    typedef struct wsp_ggml_backend_sched * wsp_ggml_backend_sched_t;

    // Initialize a backend scheduler
    WSP_GGML_API wsp_ggml_backend_sched_t wsp_ggml_backend_sched_new(wsp_ggml_backend_t * backends, int n_backends);

    WSP_GGML_API void wsp_ggml_backend_sched_free(wsp_ggml_backend_sched_t sched);

    // Initialize backend buffers from a measure graph
    WSP_GGML_API void wsp_ggml_backend_sched_init_measure(wsp_ggml_backend_sched_t sched, struct wsp_ggml_cgraph * measure_graph);

    WSP_GGML_API wsp_ggml_tallocr_t        wsp_ggml_backend_sched_get_tallocr(wsp_ggml_backend_sched_t sched, wsp_ggml_backend_t backend);
    WSP_GGML_API wsp_ggml_backend_buffer_t wsp_ggml_backend_sched_get_buffer (wsp_ggml_backend_sched_t sched, wsp_ggml_backend_t backend);

    WSP_GGML_API void wsp_ggml_backend_sched_set_node_backend(wsp_ggml_backend_sched_t sched, struct wsp_ggml_tensor * node, wsp_ggml_backend_t backend);

    // Allocate a graph on the backend scheduler
    WSP_GGML_API void wsp_ggml_backend_sched_graph_compute(
            wsp_ggml_backend_sched_t sched,
            struct wsp_ggml_cgraph * graph);


    //
    // Utils
    //

    struct wsp_ggml_backend_graph_copy {
        wsp_ggml_backend_buffer_t buffer;
        struct wsp_ggml_context * ctx_allocated;
        struct wsp_ggml_context * ctx_unallocated;
        struct wsp_ggml_cgraph * graph;
    };

    // Copy a graph to a different backend
    WSP_GGML_API struct wsp_ggml_backend_graph_copy wsp_ggml_backend_graph_copy(wsp_ggml_backend_t backend, struct wsp_ggml_cgraph * graph);
    WSP_GGML_API void                           wsp_ggml_backend_graph_copy_free(struct wsp_ggml_backend_graph_copy copy);

    typedef bool (*wsp_ggml_backend_eval_callback)(int node_index, struct wsp_ggml_tensor * t1, struct wsp_ggml_tensor * t2, void * user_data);

    // Compare the output of two backends
    WSP_GGML_API void wsp_ggml_backend_compare_graph_backend(wsp_ggml_backend_t backend1, wsp_ggml_backend_t backend2, struct wsp_ggml_cgraph * graph, wsp_ggml_backend_eval_callback callback, void * user_data);

    // Tensor initialization
    WSP_GGML_API void wsp_ggml_backend_tensor_alloc(wsp_ggml_backend_buffer_t buffer, struct wsp_ggml_tensor * tensor, void * addr);
    WSP_GGML_API void wsp_ggml_backend_view_init(wsp_ggml_backend_buffer_t buffer, struct wsp_ggml_tensor * tensor);


#ifdef  __cplusplus
}
#endif
