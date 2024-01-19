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
        const char *          (*WSP_GGML_CALL get_name)        (wsp_ggml_backend_buffer_type_t buft);
        wsp_ggml_backend_buffer_t (*WSP_GGML_CALL alloc_buffer)    (wsp_ggml_backend_buffer_type_t buft, size_t size);
        size_t                (*WSP_GGML_CALL get_alignment)   (wsp_ggml_backend_buffer_type_t buft); // tensor alignment
        size_t                (*WSP_GGML_CALL get_alloc_size)  (wsp_ggml_backend_buffer_type_t buft, const struct wsp_ggml_tensor * tensor); // data size needed to allocate the tensor, including padding
        bool                  (*WSP_GGML_CALL supports_backend)(wsp_ggml_backend_buffer_type_t buft, wsp_ggml_backend_t backend); // check if the buffer type is usable by the backend
        // check if tensor data is in host memory
        // should be equivalent to supports_backend(buft, wsp_ggml_backend_cpu_init())
        bool                  (*WSP_GGML_CALL is_host)         (wsp_ggml_backend_buffer_type_t buft);
    };

    struct wsp_ggml_backend_buffer_type {
        struct wsp_ggml_backend_buffer_type_i  iface;
        wsp_ggml_backend_buffer_type_context_t context;
    };

    // buffer
    typedef void * wsp_ggml_backend_buffer_context_t;

    struct wsp_ggml_backend_buffer_i {
        const char * (*WSP_GGML_CALL get_name)   (wsp_ggml_backend_buffer_t buffer);
        void         (*WSP_GGML_CALL free_buffer)(wsp_ggml_backend_buffer_t buffer);
        void *       (*WSP_GGML_CALL get_base)   (wsp_ggml_backend_buffer_t buffer);
        void         (*WSP_GGML_CALL init_tensor)(wsp_ggml_backend_buffer_t buffer, struct wsp_ggml_tensor * tensor);
        void         (*WSP_GGML_CALL set_tensor) (wsp_ggml_backend_buffer_t buffer,       struct wsp_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void         (*WSP_GGML_CALL get_tensor) (wsp_ggml_backend_buffer_t buffer, const struct wsp_ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        bool         (*WSP_GGML_CALL cpy_tensor) (wsp_ggml_backend_buffer_t buffer, const struct wsp_ggml_tensor * src, struct wsp_ggml_tensor * dst); // dst is in the buffer, src may be in any buffer
        void         (*WSP_GGML_CALL clear)      (wsp_ggml_backend_buffer_t buffer, uint8_t value);
        void         (*WSP_GGML_CALL reset)      (wsp_ggml_backend_buffer_t buffer); // reset any internal state due to tensor initialization, such as tensor extras
    };

    struct wsp_ggml_backend_buffer {
        struct wsp_ggml_backend_buffer_i  iface;
        wsp_ggml_backend_buffer_type_t    buft;
        wsp_ggml_backend_buffer_context_t context;
        size_t size;
        enum wsp_ggml_backend_buffer_usage usage;
    };

    WSP_GGML_CALL wsp_ggml_backend_buffer_t wsp_ggml_backend_buffer_init(
                   wsp_ggml_backend_buffer_type_t      buft,
            struct wsp_ggml_backend_buffer_i           iface,
                   wsp_ggml_backend_buffer_context_t   context,
                   size_t                          size);

    // do not use directly, use wsp_ggml_backend_tensor_copy instead
    bool wsp_ggml_backend_buffer_copy_tensor(const struct wsp_ggml_tensor * src, struct wsp_ggml_tensor * dst);

    //
    // Backend
    //

    typedef void * wsp_ggml_backend_context_t;

    struct wsp_ggml_backend_i {
        const char * (*WSP_GGML_CALL get_name)(wsp_ggml_backend_t backend);

        void (*WSP_GGML_CALL free)(wsp_ggml_backend_t backend);

        // buffer allocation
        wsp_ggml_backend_buffer_type_t (*WSP_GGML_CALL get_default_buffer_type)(wsp_ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        void (*WSP_GGML_CALL set_tensor_async)(wsp_ggml_backend_t backend,       struct wsp_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*WSP_GGML_CALL get_tensor_async)(wsp_ggml_backend_t backend, const struct wsp_ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        bool (*WSP_GGML_CALL cpy_tensor_async)(wsp_ggml_backend_t backend, const struct wsp_ggml_tensor * src, struct wsp_ggml_tensor * dst);

        // (optional) complete all pending operations
        void (*WSP_GGML_CALL synchronize)(wsp_ggml_backend_t backend);

        // compute graph with a plan
        wsp_ggml_backend_graph_plan_t (*WSP_GGML_CALL graph_plan_create) (wsp_ggml_backend_t backend, const struct wsp_ggml_cgraph * cgraph);
        void                      (*WSP_GGML_CALL graph_plan_free)   (wsp_ggml_backend_t backend, wsp_ggml_backend_graph_plan_t plan);
        void                      (*WSP_GGML_CALL graph_plan_compute)(wsp_ggml_backend_t backend, wsp_ggml_backend_graph_plan_t plan);

        // compute graph without a plan (async)
        bool (*WSP_GGML_CALL graph_compute)(wsp_ggml_backend_t backend, struct wsp_ggml_cgraph * cgraph);

        // check if the backend supports an operation
        bool (*WSP_GGML_CALL supports_op)(wsp_ggml_backend_t backend, const struct wsp_ggml_tensor * op);
    };

    struct wsp_ggml_backend {
        struct wsp_ggml_backend_i iface;

        wsp_ggml_backend_context_t context;
    };

    //
    // Backend registry
    //

    typedef wsp_ggml_backend_t (*WSP_GGML_CALL wsp_ggml_backend_init_fn)(const char * params, void * user_data);

    WSP_GGML_CALL void wsp_ggml_backend_register(const char * name, wsp_ggml_backend_init_fn init_fn, wsp_ggml_backend_buffer_type_t default_buffer_type, void * user_data);

#ifdef  __cplusplus
}
#endif
