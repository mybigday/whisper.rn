#pragma once

// ggml-backend internal header

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    //
    // Backend buffer
    //

    // buffer type
    typedef void * wsp_ggml_backend_buffer_type_context_t;

    struct wsp_ggml_backend_buffer_type_i {
        wsp_ggml_backend_buffer_t (*alloc_buffer)    (wsp_ggml_backend_buffer_type_t buft, size_t size);
        size_t                (*get_alignment)   (wsp_ggml_backend_buffer_type_t buft); // tensor alignment
        size_t                (*get_alloc_size)  (wsp_ggml_backend_buffer_type_t buft, struct wsp_ggml_tensor * tensor); // data size needed to allocate the tensor, including padding
        bool                  (*supports_backend)(wsp_ggml_backend_buffer_type_t buft, wsp_ggml_backend_t backend); // check if the buffer type is usable by the backend
    };

    struct wsp_ggml_backend_buffer_type {
        struct wsp_ggml_backend_buffer_type_i  iface;
        wsp_ggml_backend_buffer_type_context_t context;
    };

    // buffer
    typedef void * wsp_ggml_backend_buffer_context_t;

    struct wsp_ggml_backend_buffer_i {
        void     (*free_buffer)(wsp_ggml_backend_buffer_t buffer);
        //void     (*reset)      (wsp_ggml_backend_buffer_t buffer); // reset any internal state due to tensor initialization, such as tensor extras
        void *   (*get_base)   (wsp_ggml_backend_buffer_t buffer);
        void     (*init_tensor)(wsp_ggml_backend_buffer_t buffer, struct wsp_ggml_tensor * tensor);
        void     (*set_tensor) (wsp_ggml_backend_buffer_t buffer,       struct wsp_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void     (*get_tensor) (wsp_ggml_backend_buffer_t buffer, const struct wsp_ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        // (optional) copy tensor between different buffer-type, allow for single-copy tranfers
        void (*cpy_tensor_from)(wsp_ggml_backend_buffer_t buffer, struct wsp_ggml_tensor * src, struct wsp_ggml_tensor * dst);
        void (*cpy_tensor_to)  (wsp_ggml_backend_buffer_t buffer, struct wsp_ggml_tensor * src, struct wsp_ggml_tensor * dst);
    };

    struct wsp_ggml_backend_buffer {
        struct wsp_ggml_backend_buffer_i  iface;
        wsp_ggml_backend_buffer_type_t    buft;
        wsp_ggml_backend_buffer_context_t context;
        size_t size;
    };

    wsp_ggml_backend_buffer_t wsp_ggml_backend_buffer_init(
                   wsp_ggml_backend_buffer_type_t      buft,
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
        wsp_ggml_backend_buffer_type_t (*get_default_buffer_type)(wsp_ggml_backend_t backend);

        // (optional) asynchroneous tensor data access
        void (*set_tensor_async)(wsp_ggml_backend_t backend,       struct wsp_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*get_tensor_async)(wsp_ggml_backend_t backend, const struct wsp_ggml_tensor * tensor,       void * data, size_t offset, size_t size);

        // (optional) asynchroneous tensor copy
        void (*cpy_tensor_from_async)(wsp_ggml_backend_t backend, struct wsp_ggml_tensor * src, struct wsp_ggml_tensor * dst);
        void (*cpy_tensor_to_async)  (wsp_ggml_backend_t backend, struct wsp_ggml_tensor * src, struct wsp_ggml_tensor * dst);

        void (*synchronize)     (wsp_ggml_backend_t backend);

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


    //
    // Backend registry
    //

    typedef wsp_ggml_backend_t (*wsp_ggml_backend_init_fn)(const char * params, void * user_data);

    void wsp_ggml_backend_register(const char * name, wsp_ggml_backend_init_fn init_fn, wsp_ggml_backend_buffer_type_t default_buffer_type, void * user_data);

#ifdef  __cplusplus
}
#endif
