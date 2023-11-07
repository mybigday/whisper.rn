#pragma once

// ggml-backend internal header

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    //
    // Backend buffer
    //

    typedef void * wsp_ggml_backend_buffer_context_t;

    struct wsp_ggml_backend_buffer_i {
        void   (*free_buffer)   (wsp_ggml_backend_buffer_t buffer);
        void * (*get_base)      (wsp_ggml_backend_buffer_t buffer); // get base pointer
        size_t (*get_alloc_size)(wsp_ggml_backend_buffer_t buffer, struct wsp_ggml_tensor * tensor); // pre-allocation callback
        void   (*init_tensor)   (wsp_ggml_backend_buffer_t buffer, struct wsp_ggml_tensor * tensor); // post-allocation callback
        void   (*free_tensor)   (wsp_ggml_backend_buffer_t buffer, struct wsp_ggml_tensor * tensor); // pre-free callback
    };

    struct wsp_ggml_backend_buffer {
        struct wsp_ggml_backend_buffer_i iface;

        wsp_ggml_backend_t                backend;
        wsp_ggml_backend_buffer_context_t context;

        size_t size;
    };

    WSP_GGML_API wsp_ggml_backend_buffer_t wsp_ggml_backend_buffer_init(
            struct wsp_ggml_backend                  * backend,
            struct wsp_ggml_backend_buffer_i           iface,
                   wsp_ggml_backend_buffer_context_t   context,
                   size_t                          size);

    //
    // Backend
    //

    typedef void * wsp_ggml_backend_context_t;

    struct wsp_ggml_backend_i {
        const char * (*get_name)(wsp_ggml_backend_t backend);

        void (*free)(wsp_ggml_backend_t backend);

        // buffer allocation
        wsp_ggml_backend_buffer_t (*alloc_buffer)(wsp_ggml_backend_t backend, size_t size);

        // get buffer alignment
        size_t (*get_alignment)(wsp_ggml_backend_t backend);

        // tensor data access
        // these functions can be asynchronous, helper functions are provided for synchronous access that automatically call synchronize
        void (*set_tensor_async)(wsp_ggml_backend_t backend,       struct wsp_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*get_tensor_async)(wsp_ggml_backend_t backend, const struct wsp_ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        void (*synchronize)     (wsp_ggml_backend_t backend);

        // (optional) copy tensor between different backends, allow for single-copy tranfers
        void (*cpy_tensor_from)(wsp_ggml_backend_t backend, struct wsp_ggml_tensor * src, struct wsp_ggml_tensor * dst);
        void (*cpy_tensor_to)  (wsp_ggml_backend_t backend, struct wsp_ggml_tensor * src, struct wsp_ggml_tensor * dst);

        // compute graph with a plan
        wsp_ggml_backend_graph_plan_t (*graph_plan_create) (wsp_ggml_backend_t backend, struct wsp_ggml_cgraph * cgraph);
        void                      (*graph_plan_free)   (wsp_ggml_backend_t backend, wsp_ggml_backend_graph_plan_t plan);
        void                      (*graph_plan_compute)(wsp_ggml_backend_t backend, wsp_ggml_backend_graph_plan_t plan);

        // compute graph without a plan
        void (*graph_compute)(wsp_ggml_backend_t backend, struct wsp_ggml_cgraph * cgraph);

        // check if the backend supports an operation
        bool (*supports_op)(wsp_ggml_backend_t backend, const struct wsp_ggml_tensor * op);
    };

    struct wsp_ggml_backend {
        struct wsp_ggml_backend_i iface;

        wsp_ggml_backend_context_t context;
    };

#ifdef  __cplusplus
}
#endif
