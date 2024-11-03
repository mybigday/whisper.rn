#pragma once

#include "ggml.h"
#include "ggml-alloc.h"

#ifdef  __cplusplus
extern "C" {
#endif

    typedef struct wsp_ggml_backend_buffer_type * wsp_ggml_backend_buffer_type_t;
    typedef struct wsp_ggml_backend_buffer * wsp_ggml_backend_buffer_t;
    typedef struct wsp_ggml_backend_event * wsp_ggml_backend_event_t;
    typedef struct wsp_ggml_backend * wsp_ggml_backend_t;
    typedef void * wsp_ggml_backend_graph_plan_t;
    typedef struct wsp_ggml_backend_reg * wsp_ggml_backend_reg_t;
    typedef struct wsp_ggml_backend_device * wsp_ggml_backend_dev_t;


    //
    // Backend buffer type
    //

    WSP_GGML_API const char *          wsp_ggml_backend_buft_name          (wsp_ggml_backend_buffer_type_t buft);
    WSP_GGML_API wsp_ggml_backend_buffer_t wsp_ggml_backend_buft_alloc_buffer  (wsp_ggml_backend_buffer_type_t buft, size_t size);
    WSP_GGML_API size_t                wsp_ggml_backend_buft_get_alignment (wsp_ggml_backend_buffer_type_t buft);
    WSP_GGML_API size_t                wsp_ggml_backend_buft_get_max_size  (wsp_ggml_backend_buffer_type_t buft);
    WSP_GGML_API size_t                wsp_ggml_backend_buft_get_alloc_size(wsp_ggml_backend_buffer_type_t buft, struct wsp_ggml_tensor * tensor);
    WSP_GGML_API bool                  wsp_ggml_backend_buft_is_host       (wsp_ggml_backend_buffer_type_t buft);
    WSP_GGML_API wsp_ggml_backend_dev_t    wsp_ggml_backend_buft_get_device    (wsp_ggml_backend_buffer_type_t buft);

    //
    // Backend buffer
    //

    enum wsp_ggml_backend_buffer_usage {
        WSP_GGML_BACKEND_BUFFER_USAGE_ANY = 0,
        WSP_GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1,
        WSP_GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2,
    };

    WSP_GGML_API const char *                   wsp_ggml_backend_buffer_name          (wsp_ggml_backend_buffer_t buffer);
    WSP_GGML_API void                           wsp_ggml_backend_buffer_free          (wsp_ggml_backend_buffer_t buffer);
    WSP_GGML_API void *                         wsp_ggml_backend_buffer_get_base      (wsp_ggml_backend_buffer_t buffer);
    WSP_GGML_API size_t                         wsp_ggml_backend_buffer_get_size      (wsp_ggml_backend_buffer_t buffer);
    WSP_GGML_API void                           wsp_ggml_backend_buffer_init_tensor   (wsp_ggml_backend_buffer_t buffer, struct wsp_ggml_tensor * tensor);
    WSP_GGML_API size_t                         wsp_ggml_backend_buffer_get_alignment (wsp_ggml_backend_buffer_t buffer);
    WSP_GGML_API size_t                         wsp_ggml_backend_buffer_get_max_size  (wsp_ggml_backend_buffer_t buffer);
    WSP_GGML_API size_t                         wsp_ggml_backend_buffer_get_alloc_size(wsp_ggml_backend_buffer_t buffer, struct wsp_ggml_tensor * tensor);
    WSP_GGML_API void                           wsp_ggml_backend_buffer_clear         (wsp_ggml_backend_buffer_t buffer, uint8_t value);
    WSP_GGML_API bool                           wsp_ggml_backend_buffer_is_host       (wsp_ggml_backend_buffer_t buffer);
    WSP_GGML_API void                           wsp_ggml_backend_buffer_set_usage     (wsp_ggml_backend_buffer_t buffer, enum wsp_ggml_backend_buffer_usage usage);
    WSP_GGML_API enum wsp_ggml_backend_buffer_usage wsp_ggml_backend_buffer_get_usage     (wsp_ggml_backend_buffer_t buffer);
    WSP_GGML_API wsp_ggml_backend_buffer_type_t     wsp_ggml_backend_buffer_get_type      (wsp_ggml_backend_buffer_t buffer);
    WSP_GGML_API void                           wsp_ggml_backend_buffer_reset         (wsp_ggml_backend_buffer_t buffer);

    // tensor copy between different backends
    WSP_GGML_API void wsp_ggml_backend_tensor_copy(struct wsp_ggml_tensor * src, struct wsp_ggml_tensor * dst);

    //
    // Backend (stream)
    //

    WSP_GGML_API wsp_ggml_guid_t  wsp_ggml_backend_guid(wsp_ggml_backend_t backend);
    WSP_GGML_API const char * wsp_ggml_backend_name(wsp_ggml_backend_t backend);
    WSP_GGML_API void         wsp_ggml_backend_free(wsp_ggml_backend_t backend);

    WSP_GGML_API wsp_ggml_backend_buffer_type_t wsp_ggml_backend_get_default_buffer_type(wsp_ggml_backend_t backend);
    WSP_GGML_API wsp_ggml_backend_buffer_t      wsp_ggml_backend_alloc_buffer(wsp_ggml_backend_t backend, size_t size);
    WSP_GGML_API size_t                     wsp_ggml_backend_get_alignment(wsp_ggml_backend_t backend);
    WSP_GGML_API size_t                     wsp_ggml_backend_get_max_size(wsp_ggml_backend_t backend);

    WSP_GGML_API void wsp_ggml_backend_tensor_set_async(wsp_ggml_backend_t backend,       struct wsp_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    WSP_GGML_API void wsp_ggml_backend_tensor_get_async(wsp_ggml_backend_t backend, const struct wsp_ggml_tensor * tensor,       void * data, size_t offset, size_t size);

    // "offset" refers to the offset of the tensor data for setting/getting data
    WSP_GGML_API void wsp_ggml_backend_tensor_set(      struct wsp_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    WSP_GGML_API void wsp_ggml_backend_tensor_get(const struct wsp_ggml_tensor * tensor,       void * data, size_t offset, size_t size);
    WSP_GGML_API void wsp_ggml_backend_tensor_memset(   struct wsp_ggml_tensor * tensor,     uint8_t value, size_t offset, size_t size);

    WSP_GGML_API void wsp_ggml_backend_synchronize(wsp_ggml_backend_t backend);

    WSP_GGML_API wsp_ggml_backend_graph_plan_t wsp_ggml_backend_graph_plan_create(wsp_ggml_backend_t backend, struct wsp_ggml_cgraph * cgraph);
    WSP_GGML_API void                      wsp_ggml_backend_graph_plan_free  (wsp_ggml_backend_t backend, wsp_ggml_backend_graph_plan_t plan);

    WSP_GGML_API enum wsp_ggml_status wsp_ggml_backend_graph_plan_compute (wsp_ggml_backend_t backend, wsp_ggml_backend_graph_plan_t plan);
    WSP_GGML_API enum wsp_ggml_status wsp_ggml_backend_graph_compute      (wsp_ggml_backend_t backend, struct wsp_ggml_cgraph * cgraph);
    WSP_GGML_API enum wsp_ggml_status wsp_ggml_backend_graph_compute_async(wsp_ggml_backend_t backend, struct wsp_ggml_cgraph * cgraph);

    // NOTE: will be removed, use device version instead
    WSP_GGML_API bool wsp_ggml_backend_supports_op(wsp_ggml_backend_t backend, const struct wsp_ggml_tensor * op);
    WSP_GGML_API bool wsp_ggml_backend_supports_buft(wsp_ggml_backend_t backend, wsp_ggml_backend_buffer_type_t buft);
    WSP_GGML_API bool wsp_ggml_backend_offload_op(wsp_ggml_backend_t backend, const struct wsp_ggml_tensor * op);

    // asynchronous copy
    // the copy is performed after all the currently queued operations in backend_src
    // backend_dst will wait for the copy to complete before performing other operations
    // automatic fallback to sync copy if async is not supported
    WSP_GGML_API void wsp_ggml_backend_tensor_copy_async(wsp_ggml_backend_t backend_src, wsp_ggml_backend_t backend_dst, struct wsp_ggml_tensor * src, struct wsp_ggml_tensor * dst);

    WSP_GGML_API wsp_ggml_backend_dev_t wsp_ggml_backend_get_device(wsp_ggml_backend_t backend);

    //
    // Events
    //

    WSP_GGML_API wsp_ggml_backend_event_t wsp_ggml_backend_event_new(wsp_ggml_backend_dev_t device);
    WSP_GGML_API void                 wsp_ggml_backend_event_free(wsp_ggml_backend_event_t event);
    WSP_GGML_API void                 wsp_ggml_backend_event_record(wsp_ggml_backend_event_t event, wsp_ggml_backend_t backend);
    WSP_GGML_API void                 wsp_ggml_backend_event_synchronize(wsp_ggml_backend_event_t event);
    WSP_GGML_API void                 wsp_ggml_backend_event_wait(wsp_ggml_backend_t backend, wsp_ggml_backend_event_t event);

    //
    // Backend device
    //

    enum wsp_ggml_backend_dev_type {
        WSP_GGML_BACKEND_DEVICE_TYPE_CPU,
        WSP_GGML_BACKEND_DEVICE_TYPE_GPU,
        // devices with full capabilities (excludes backends such as BLAS that only support matrix multiplication)
        WSP_GGML_BACKEND_DEVICE_TYPE_CPU_FULL,
        WSP_GGML_BACKEND_DEVICE_TYPE_GPU_FULL
    };

    // functionality supported by the device
    struct wsp_ggml_backend_dev_caps {
        // asynchronous operations
        bool async;
        // pinned host buffer
        bool host_buffer;
        // creating buffers from host ptr
        bool buffer_from_host_ptr;
        // event synchronization
        bool events;
    };

    // all the device properties
    struct wsp_ggml_backend_dev_props {
        const char * name;
        const char * description;
        size_t memory_free;
        size_t memory_total;
        enum wsp_ggml_backend_dev_type type;
        struct wsp_ggml_backend_dev_caps caps;
    };

    WSP_GGML_API const char *                  wsp_ggml_backend_dev_name(wsp_ggml_backend_dev_t device);
    WSP_GGML_API const char *                  wsp_ggml_backend_dev_description(wsp_ggml_backend_dev_t device);
    WSP_GGML_API void                          wsp_ggml_backend_dev_memory(wsp_ggml_backend_dev_t device, size_t * free, size_t * total);
    WSP_GGML_API enum wsp_ggml_backend_dev_type    wsp_ggml_backend_dev_type(wsp_ggml_backend_dev_t device);
    WSP_GGML_API void                          wsp_ggml_backend_dev_get_props(wsp_ggml_backend_dev_t device, struct wsp_ggml_backend_dev_props * props);
    WSP_GGML_API wsp_ggml_backend_reg_t            wsp_ggml_backend_dev_backend_reg(wsp_ggml_backend_dev_t device);
    WSP_GGML_API wsp_ggml_backend_t                wsp_ggml_backend_dev_init(wsp_ggml_backend_dev_t device, const char * params);
    WSP_GGML_API wsp_ggml_backend_buffer_type_t    wsp_ggml_backend_dev_buffer_type(wsp_ggml_backend_dev_t device);
    WSP_GGML_API wsp_ggml_backend_buffer_type_t    wsp_ggml_backend_dev_host_buffer_type(wsp_ggml_backend_dev_t device);
    WSP_GGML_API wsp_ggml_backend_buffer_t         wsp_ggml_backend_dev_buffer_from_host_ptr(wsp_ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size);

    WSP_GGML_API bool                          wsp_ggml_backend_dev_supports_op(wsp_ggml_backend_dev_t device, const struct wsp_ggml_tensor * op);
    WSP_GGML_API bool                          wsp_ggml_backend_dev_supports_buft(wsp_ggml_backend_dev_t device, wsp_ggml_backend_buffer_type_t buft);
    WSP_GGML_API bool                          wsp_ggml_backend_dev_offload_op(wsp_ggml_backend_dev_t device, const struct wsp_ggml_tensor * op);

    //
    // Backend (reg)
    //

    WSP_GGML_API const char *       wsp_ggml_backend_reg_name(wsp_ggml_backend_reg_t reg);
    WSP_GGML_API size_t             wsp_ggml_backend_reg_dev_count(wsp_ggml_backend_reg_t reg);
    WSP_GGML_API wsp_ggml_backend_dev_t wsp_ggml_backend_reg_dev_get(wsp_ggml_backend_reg_t reg, size_t index);
    WSP_GGML_API void *             wsp_ggml_backend_reg_get_proc_address(wsp_ggml_backend_reg_t reg, const char * name);


    // Functions that may be obtained using wsp_ggml_backend_reg_get_proc_address
    typedef wsp_ggml_backend_buffer_type_t (*wsp_ggml_backend_split_buffer_type_t)(const float *);
    typedef void (*wsp_ggml_backend_set_n_threads_t)(wsp_ggml_backend_t, int);

    //
    // Backend registry
    //

    // Backend (reg) enumeration
    WSP_GGML_API size_t             wsp_ggml_backend_reg_count(void);
    WSP_GGML_API wsp_ggml_backend_reg_t wsp_ggml_backend_reg_get(size_t index);
    WSP_GGML_API wsp_ggml_backend_reg_t wsp_ggml_backend_reg_by_name(const char * name);

    // Device enumeration
    WSP_GGML_API size_t             wsp_ggml_backend_dev_count(void);
    WSP_GGML_API wsp_ggml_backend_dev_t wsp_ggml_backend_dev_get(size_t index);
    WSP_GGML_API wsp_ggml_backend_dev_t wsp_ggml_backend_dev_by_name(const char * name);
    WSP_GGML_API wsp_ggml_backend_dev_t wsp_ggml_backend_dev_by_type(enum wsp_ggml_backend_dev_type type);

    // Direct backend (stream) initialization
    // = wsp_ggml_backend_dev_init(wsp_ggml_backend_dev_by_name(name), params)
    WSP_GGML_API wsp_ggml_backend_t wsp_ggml_backend_init_by_name(const char * name, const char * params);
    // = wsp_ggml_backend_dev_init(wsp_ggml_backend_dev_by_type(type), params)
    WSP_GGML_API wsp_ggml_backend_t wsp_ggml_backend_init_by_type(enum wsp_ggml_backend_dev_type type, const char * params);
    // = wsp_ggml_backend_dev_init(wsp_ggml_backend_dev_by_type(GPU_FULL) OR wsp_ggml_backend_dev_by_type(CPU_FULL), NULL)
    WSP_GGML_API wsp_ggml_backend_t wsp_ggml_backend_init_best(void);

    //
    // Backend scheduler
    //

    // The backend scheduler allows for multiple backend devices to be used together
    // Handles compute buffer allocation, assignment of tensors to backends, and copying of tensors between backends
    // The backends are selected based on:
    // - the backend that supports the operation
    // - the location of the pre-allocated tensors (e.g. the weights)
    /*
      Example usage:

        // operations that use tensors allocated in a buffer with USAGE_WEIGHTS will be assigned
        // preferrably to run on the same backend as the buffer
        wsp_ggml_backend_buffer_set_usage(buf_weights, WSP_GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        sched = wsp_ggml_backend_sched_new({backend_gpu, backend_gpu2, backend_cpu}, NULL, num_backends, WSP_GGML_DEFAULT_GRAPH_SIZE, false);

        // initialize buffers from a max size graph (optional)
        reserve_graph = build_graph(sched, max_batch_size);

        // manually assign nodes to a backend (optional, should not be needed in most cases)
        struct wsp_ggml_tensor * node = wsp_ggml_mul_mat(ctx, ...);
        wsp_ggml_backend_sched_set_tensor_backend(sched, node, backend_gpu);

        wsp_ggml_backend_sched_reserve(sched, reserve_graph);

        // compute
        graph = build_graph(sched);
        wsp_ggml_backend_sched_graph_compute(sched, graph);

        // if there are graph inputs:
        wsp_ggml_backend_sched_reset(sched);
        wsp_ggml_backend_sched_alloc_graph(sched, graph);
        wsp_ggml_backend_tensor_set(input_tensor, ...);
        wsp_ggml_backend_sched_graph_compute(sched, graph);
    }
    */

    typedef struct wsp_ggml_backend_sched * wsp_ggml_backend_sched_t;

    // Evaluation callback for each node in the graph (set with wsp_ggml_backend_sched_set_eval_callback)
    // when ask == true, the scheduler wants to know if the user wants to observe this node
    // this allows the scheduler to batch nodes together in order to evaluate them in a single call
    //
    // when ask == false, the scheduler is passing the node tensor to the user for observation
    // if the user returns false, the scheduler will cancel the graph compute
    //
    typedef bool (*wsp_ggml_backend_sched_eval_callback)(struct wsp_ggml_tensor * t, bool ask, void * user_data);

    // Initialize a backend scheduler
    WSP_GGML_API wsp_ggml_backend_sched_t wsp_ggml_backend_sched_new(wsp_ggml_backend_t * backends, wsp_ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel);
    WSP_GGML_API void                 wsp_ggml_backend_sched_free(wsp_ggml_backend_sched_t sched);

    // Initialize backend buffers from a measure graph
    WSP_GGML_API bool                 wsp_ggml_backend_sched_reserve(wsp_ggml_backend_sched_t sched, struct wsp_ggml_cgraph * measure_graph); // returns success

    WSP_GGML_API int                  wsp_ggml_backend_sched_get_n_backends(wsp_ggml_backend_sched_t sched);
    WSP_GGML_API wsp_ggml_backend_t       wsp_ggml_backend_sched_get_backend(wsp_ggml_backend_sched_t sched, int i);

    // Get the number of splits of the last graph
    WSP_GGML_API int                  wsp_ggml_backend_sched_get_n_splits(wsp_ggml_backend_sched_t sched);
    WSP_GGML_API int                  wsp_ggml_backend_sched_get_n_copies(wsp_ggml_backend_sched_t sched);

    WSP_GGML_API size_t               wsp_ggml_backend_sched_get_buffer_size(wsp_ggml_backend_sched_t sched, wsp_ggml_backend_t backend);

    WSP_GGML_API void                 wsp_ggml_backend_sched_set_tensor_backend(wsp_ggml_backend_sched_t sched, struct wsp_ggml_tensor * node, wsp_ggml_backend_t backend);
    WSP_GGML_API wsp_ggml_backend_t       wsp_ggml_backend_sched_get_tensor_backend(wsp_ggml_backend_sched_t sched, struct wsp_ggml_tensor * node);

    // Allocate and compute graph on the backend scheduler
    WSP_GGML_API bool                 wsp_ggml_backend_sched_alloc_graph(wsp_ggml_backend_sched_t sched, struct wsp_ggml_cgraph * graph); // returns success
    WSP_GGML_API enum wsp_ggml_status     wsp_ggml_backend_sched_graph_compute(wsp_ggml_backend_sched_t sched, struct wsp_ggml_cgraph * graph);
    WSP_GGML_API enum wsp_ggml_status     wsp_ggml_backend_sched_graph_compute_async(wsp_ggml_backend_sched_t sched, struct wsp_ggml_cgraph * graph);
    WSP_GGML_API void                 wsp_ggml_backend_sched_synchronize(wsp_ggml_backend_sched_t sched);

    // Reset all assignments and allocators - must be called before changing the node backends
    WSP_GGML_API void                 wsp_ggml_backend_sched_reset(wsp_ggml_backend_sched_t sched);

    // Set a callback to be called for each resulting node during graph compute
    WSP_GGML_API void                 wsp_ggml_backend_sched_set_eval_callback(wsp_ggml_backend_sched_t sched, wsp_ggml_backend_sched_eval_callback callback, void * user_data);

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
    WSP_GGML_API bool wsp_ggml_backend_compare_graph_backend(wsp_ggml_backend_t backend1, wsp_ggml_backend_t backend2, struct wsp_ggml_cgraph * graph, wsp_ggml_backend_eval_callback callback, void * user_data);

    // Tensor initialization
    WSP_GGML_API void wsp_ggml_backend_tensor_alloc(wsp_ggml_backend_buffer_t buffer, struct wsp_ggml_tensor * tensor, void * addr);
    WSP_GGML_API void wsp_ggml_backend_view_init(struct wsp_ggml_tensor * tensor);

    //
    // CPU backend
    //

    WSP_GGML_API wsp_ggml_backend_t wsp_ggml_backend_cpu_init(void);

    WSP_GGML_API bool wsp_ggml_backend_is_cpu                (wsp_ggml_backend_t backend);
    WSP_GGML_API void wsp_ggml_backend_cpu_set_n_threads     (wsp_ggml_backend_t backend_cpu, int n_threads);
    WSP_GGML_API void wsp_ggml_backend_cpu_set_threadpool    (wsp_ggml_backend_t backend_cpu, wsp_ggml_threadpool_t threadpool);
    WSP_GGML_API void wsp_ggml_backend_cpu_set_abort_callback(wsp_ggml_backend_t backend_cpu, wsp_ggml_abort_callback abort_callback, void * abort_callback_data);

    // Create a backend buffer from an existing pointer
    WSP_GGML_API wsp_ggml_backend_buffer_t      wsp_ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size);
    WSP_GGML_API wsp_ggml_backend_buffer_type_t wsp_ggml_backend_cpu_buffer_type(void);

    WSP_GGML_API wsp_ggml_backend_reg_t wsp_ggml_backend_cpu_reg(void);

#ifdef WSP_GGML_USE_CPU_HBM
    WSP_GGML_API wsp_ggml_backend_buffer_type_t wsp_ggml_backend_cpu_hbm_buffer_type(void);
#endif

#ifdef  __cplusplus
}
#endif
