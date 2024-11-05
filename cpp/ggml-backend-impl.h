#pragma once

// ggml-backend internal header

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    //
    // Backend buffer type
    //

    struct wsp_ggml_backend_buffer_type_i {
        const char *          (*get_name)      (wsp_ggml_backend_buffer_type_t buft);
        // allocate a buffer of this type
        wsp_ggml_backend_buffer_t (*alloc_buffer)  (wsp_ggml_backend_buffer_type_t buft, size_t size);
        // tensor alignment
        size_t                (*get_alignment) (wsp_ggml_backend_buffer_type_t buft);
        // (optional) max buffer size that can be allocated (defaults to SIZE_MAX)
        size_t                (*get_max_size)  (wsp_ggml_backend_buffer_type_t buft);
        // (optional) data size needed to allocate the tensor, including padding (defaults to wsp_ggml_nbytes)
        size_t                (*get_alloc_size)(wsp_ggml_backend_buffer_type_t buft, const struct wsp_ggml_tensor * tensor);
        // (optional) check if tensor data is in host memory (defaults to false)
        bool                  (*is_host)       (wsp_ggml_backend_buffer_type_t buft);
    };

    struct wsp_ggml_backend_buffer_type {
        struct wsp_ggml_backend_buffer_type_i  iface;
        wsp_ggml_backend_dev_t device;
        void * context;
    };

    //
    // Backend buffer
    //

    struct wsp_ggml_backend_buffer_i {
        const char * (*get_name)     (wsp_ggml_backend_buffer_t buffer);
        // (optional) free the buffer
        void         (*free_buffer)  (wsp_ggml_backend_buffer_t buffer);
        // base address of the buffer
        void *       (*get_base)     (wsp_ggml_backend_buffer_t buffer);
        // (optional) initialize a tensor in the buffer (eg. add tensor extras)
        void         (*init_tensor)  (wsp_ggml_backend_buffer_t buffer, struct wsp_ggml_tensor * tensor);
        // tensor data access
        void         (*memset_tensor)(wsp_ggml_backend_buffer_t buffer,       struct wsp_ggml_tensor * tensor,     uint8_t value, size_t offset, size_t size);
        void         (*set_tensor)   (wsp_ggml_backend_buffer_t buffer,       struct wsp_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void         (*get_tensor)   (wsp_ggml_backend_buffer_t buffer, const struct wsp_ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        // (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
        bool         (*cpy_tensor)   (wsp_ggml_backend_buffer_t buffer, const struct wsp_ggml_tensor * src, struct wsp_ggml_tensor * dst);
        // clear the entire buffer
        void         (*clear)        (wsp_ggml_backend_buffer_t buffer, uint8_t value);
        // (optional) reset any internal state due to tensor initialization, such as tensor extras
        void         (*reset)        (wsp_ggml_backend_buffer_t buffer);
    };

    struct wsp_ggml_backend_buffer {
        struct wsp_ggml_backend_buffer_i  iface;
        wsp_ggml_backend_buffer_type_t    buft;
        void * context;
        size_t size;
        enum wsp_ggml_backend_buffer_usage usage;
    };

    wsp_ggml_backend_buffer_t wsp_ggml_backend_buffer_init(
                   wsp_ggml_backend_buffer_type_t buft,
            struct wsp_ggml_backend_buffer_i      iface,
                   void *                     context,
                   size_t                     size);

    // do not use directly, use wsp_ggml_backend_tensor_copy instead
    bool wsp_ggml_backend_buffer_copy_tensor(const struct wsp_ggml_tensor * src, struct wsp_ggml_tensor * dst);

    // multi-buffer
    // buffer that contains a collection of buffers
    wsp_ggml_backend_buffer_t wsp_ggml_backend_multi_buffer_alloc_buffer(wsp_ggml_backend_buffer_t * buffers, size_t n_buffers);
    bool                  wsp_ggml_backend_buffer_is_multi_buffer(wsp_ggml_backend_buffer_t buffer);
    void                  wsp_ggml_backend_multi_buffer_set_usage(wsp_ggml_backend_buffer_t buffer, enum wsp_ggml_backend_buffer_usage usage);

    //
    // Backend (stream)
    //

    struct wsp_ggml_backend_i {
        const char * (*get_name)(wsp_ggml_backend_t backend);

        void (*free)(wsp_ggml_backend_t backend);

        // Will be moved to the device interface
        // buffer allocation
        wsp_ggml_backend_buffer_type_t (*get_default_buffer_type)(wsp_ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        void (*set_tensor_async)(wsp_ggml_backend_t backend,       struct wsp_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*get_tensor_async)(wsp_ggml_backend_t backend, const struct wsp_ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        bool (*cpy_tensor_async)(wsp_ggml_backend_t backend_src, wsp_ggml_backend_t backend_dst, const struct wsp_ggml_tensor * src, struct wsp_ggml_tensor * dst);

        // (optional) complete all pending operations
        void (*synchronize)(wsp_ggml_backend_t backend);

        // (optional) compute graph with a plan (not used currently)
        wsp_ggml_backend_graph_plan_t (*graph_plan_create) (wsp_ggml_backend_t backend, const struct wsp_ggml_cgraph * cgraph);
        void                      (*graph_plan_free)   (wsp_ggml_backend_t backend, wsp_ggml_backend_graph_plan_t plan);
        // update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
        void                      (*graph_plan_update) (wsp_ggml_backend_t backend, wsp_ggml_backend_graph_plan_t plan, const struct wsp_ggml_cgraph * cgraph);
        // compute the graph with the plan
        enum wsp_ggml_status          (*graph_plan_compute)(wsp_ggml_backend_t backend, wsp_ggml_backend_graph_plan_t plan);

        // compute graph (always async if supported by the backend)
        enum wsp_ggml_status          (*graph_compute)     (wsp_ggml_backend_t backend, struct wsp_ggml_cgraph * cgraph);

        // IMPORTANT: these functions have been moved to the device interface and will be removed from the backend interface
        //            new backends should implement the device interface instead
        // These functions are being moved to the device interface
        bool (*supports_op)  (wsp_ggml_backend_t backend, const struct wsp_ggml_tensor * op);
        bool (*supports_buft)(wsp_ggml_backend_t backend, wsp_ggml_backend_buffer_type_t buft);
        bool (*offload_op)   (wsp_ggml_backend_t backend, const struct wsp_ggml_tensor * op);

        // (optional) event synchronization
        // record an event on this stream
        void (*event_record)(wsp_ggml_backend_t backend, wsp_ggml_backend_event_t event);
        // wait for an event on on a different stream
        void (*event_wait)  (wsp_ggml_backend_t backend, wsp_ggml_backend_event_t event);
    };

    struct wsp_ggml_backend {
        wsp_ggml_guid_t guid;
        struct wsp_ggml_backend_i iface;
        wsp_ggml_backend_dev_t device;
        void * context;
    };

    struct wsp_ggml_backend_event {
        struct wsp_ggml_backend_device * device;
        void * context;
    };

    //
    // Backend device
    //

    // Note: if additional properties are needed, we should add a struct with all of them
    //       the current functions to obtain the properties can remain, since they are more convenient for often used properties
    struct wsp_ggml_backend_device_i {
        // device name: short identifier for this device, such as "CPU" or "CUDA0"
        const char * (*get_name)(wsp_ggml_backend_dev_t dev);

        // device description: short informative description of the device, could be the model name
        const char * (*get_description)(wsp_ggml_backend_dev_t dev);

        // device memory in bytes
        void         (*get_memory)(wsp_ggml_backend_dev_t dev, size_t * free, size_t * total);

        // device type
        enum wsp_ggml_backend_dev_type (*get_type)(wsp_ggml_backend_dev_t dev);

        // device properties
        void (*get_props)(wsp_ggml_backend_dev_t dev, struct wsp_ggml_backend_dev_props * props);

        // backend (stream) initialization
        wsp_ggml_backend_t (*init_backend)(wsp_ggml_backend_dev_t dev, const char * params);

        // preferred buffer type
        wsp_ggml_backend_buffer_type_t (*get_buffer_type)(wsp_ggml_backend_dev_t dev);

        // (optional) host buffer type (in system memory, typically this is a pinned memory buffer for faster transfers between host and device)
        wsp_ggml_backend_buffer_type_t (*get_host_buffer_type)(wsp_ggml_backend_dev_t dev);

        // (optional) buffer from pointer: create a buffer from a host pointer (useful for memory mapped models and importing data from other libraries)
        wsp_ggml_backend_buffer_t (*buffer_from_host_ptr)(wsp_ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size);

        // check if the backend can compute an operation
        bool (*supports_op)(wsp_ggml_backend_dev_t dev, const struct wsp_ggml_tensor * op);

        // check if the backend can use tensors allocated in a buffer type
        bool (*supports_buft)(wsp_ggml_backend_dev_t dev, wsp_ggml_backend_buffer_type_t buft);

        // (optional) check if the backend wants to run an operation, even if the weights are allocated in an incompatible buffer
        // these should be expensive operations that may benefit from running on this backend instead of the CPU backend
        bool (*offload_op)(wsp_ggml_backend_dev_t dev, const struct wsp_ggml_tensor * op);

        // (optional) event synchronization
        wsp_ggml_backend_event_t (*event_new)         (wsp_ggml_backend_dev_t dev);
        void                 (*event_free)        (wsp_ggml_backend_dev_t dev, wsp_ggml_backend_event_t event);
        void                 (*event_synchronize) (wsp_ggml_backend_dev_t dev, wsp_ggml_backend_event_t event);
    };

    struct wsp_ggml_backend_device {
        struct wsp_ggml_backend_device_i iface;
        wsp_ggml_backend_reg_t reg;
        void * context;
    };

    //
    // Backend (reg)
    //

    struct wsp_ggml_backend_reg_i {
        const char * (*get_name)(wsp_ggml_backend_reg_t reg);

        // enumerate available devices
        size_t             (*get_device_count)(wsp_ggml_backend_reg_t reg);
        wsp_ggml_backend_dev_t (*get_device)(wsp_ggml_backend_reg_t reg, size_t index);

        // (optional) get a pointer to a function in the backend
        // backends can add custom functions that are not part of the standard ggml-backend interface
        void * (*get_proc_address)(wsp_ggml_backend_reg_t reg, const char * name);
    };

    struct wsp_ggml_backend_reg {
        // int api_version; // TODO: for dynamic loading
        struct wsp_ggml_backend_reg_i iface;
        void * context;
    };


    // Internal backend registry API
    void wsp_ggml_backend_register(wsp_ggml_backend_reg_t reg);
    void wsp_ggml_backend_device_register(wsp_ggml_backend_dev_t device);
    // TODO: backends can be loaded as a dynamic library, in which case it needs to export this function
    // typedef wsp_ggml_backend_register_t * (*wsp_ggml_backend_init)(void);

#ifdef  __cplusplus
}
#endif
