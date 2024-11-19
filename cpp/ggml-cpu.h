#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    // Scheduling priorities
    enum wsp_ggml_sched_priority {
        WSP_GGML_SCHED_PRIO_NORMAL,
        WSP_GGML_SCHED_PRIO_MEDIUM,
        WSP_GGML_SCHED_PRIO_HIGH,
        WSP_GGML_SCHED_PRIO_REALTIME
    };

    // Threadpool params
    // Use wsp_ggml_threadpool_params_default() or wsp_ggml_threadpool_params_init() to populate the defaults
    struct wsp_ggml_threadpool_params {
        bool                cpumask[WSP_GGML_MAX_N_THREADS]; // mask of cpu cores (all-zeros means use default affinity settings)
        int                 n_threads;                   // number of threads
        enum wsp_ggml_sched_priority prio;                   // thread priority
        uint32_t            poll;                        // polling level (0 - no polling, 100 - aggressive polling)
        bool                strict_cpu;                  // strict cpu placement
        bool                paused;                      // start in paused state
    };

    struct wsp_ggml_threadpool;     // forward declaration, see ggml.c

    typedef struct wsp_ggml_threadpool * wsp_ggml_threadpool_t;

    // the compute plan that needs to be prepared for wsp_ggml_graph_compute()
    // since https://github.com/ggerganov/ggml/issues/287
    struct wsp_ggml_cplan {
        size_t    work_size; // size of work buffer, calculated by `wsp_ggml_graph_plan()`
        uint8_t * work_data; // work buffer, to be allocated by caller before calling to `wsp_ggml_graph_compute()`

        int n_threads;
        struct wsp_ggml_threadpool * threadpool;

        // abort wsp_ggml_graph_compute when true
        wsp_ggml_abort_callback abort_callback;
        void *              abort_callback_data;
    };

    // numa strategies
    enum wsp_ggml_numa_strategy {
        WSP_GGML_NUMA_STRATEGY_DISABLED   = 0,
        WSP_GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
        WSP_GGML_NUMA_STRATEGY_ISOLATE    = 2,
        WSP_GGML_NUMA_STRATEGY_NUMACTL    = 3,
        WSP_GGML_NUMA_STRATEGY_MIRROR     = 4,
        WSP_GGML_NUMA_STRATEGY_COUNT
    };

    WSP_GGML_API void    wsp_ggml_numa_init(enum wsp_ggml_numa_strategy numa); // call once for better performance on NUMA systems
    WSP_GGML_API bool    wsp_ggml_is_numa(void); // true if init detected that system has >1 NUMA node

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_new_i32(struct wsp_ggml_context * ctx, int32_t value);
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_new_f32(struct wsp_ggml_context * ctx, float value);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_set_i32 (struct wsp_ggml_tensor * tensor, int32_t value);
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_set_f32 (struct wsp_ggml_tensor * tensor, float value);

    WSP_GGML_API int32_t wsp_ggml_get_i32_1d(const struct wsp_ggml_tensor * tensor, int i);
    WSP_GGML_API void    wsp_ggml_set_i32_1d(const struct wsp_ggml_tensor * tensor, int i, int32_t value);

    WSP_GGML_API int32_t wsp_ggml_get_i32_nd(const struct wsp_ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    WSP_GGML_API void    wsp_ggml_set_i32_nd(const struct wsp_ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

    WSP_GGML_API float   wsp_ggml_get_f32_1d(const struct wsp_ggml_tensor * tensor, int i);
    WSP_GGML_API void    wsp_ggml_set_f32_1d(const struct wsp_ggml_tensor * tensor, int i, float value);

    WSP_GGML_API float   wsp_ggml_get_f32_nd(const struct wsp_ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    WSP_GGML_API void    wsp_ggml_set_f32_nd(const struct wsp_ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

    WSP_GGML_API struct wsp_ggml_threadpool_params wsp_ggml_threadpool_params_default(int n_threads);
    WSP_GGML_API void                          wsp_ggml_threadpool_params_init   (struct wsp_ggml_threadpool_params * p, int n_threads);
    WSP_GGML_API bool                          wsp_ggml_threadpool_params_match  (const struct wsp_ggml_threadpool_params * p0, const struct wsp_ggml_threadpool_params * p1);
    WSP_GGML_API struct wsp_ggml_threadpool *      wsp_ggml_threadpool_new          (struct wsp_ggml_threadpool_params  * params);
    WSP_GGML_API void                          wsp_ggml_threadpool_free         (struct wsp_ggml_threadpool * threadpool);
    WSP_GGML_API int                           wsp_ggml_threadpool_get_n_threads(struct wsp_ggml_threadpool * threadpool);
    WSP_GGML_API void                          wsp_ggml_threadpool_pause        (struct wsp_ggml_threadpool * threadpool);
    WSP_GGML_API void                          wsp_ggml_threadpool_resume       (struct wsp_ggml_threadpool * threadpool);

    // wsp_ggml_graph_plan() has to be called before wsp_ggml_graph_compute()
    // when plan.work_size > 0, caller must allocate memory for plan.work_data
    WSP_GGML_API struct wsp_ggml_cplan wsp_ggml_graph_plan(
                  const struct wsp_ggml_cgraph * cgraph,
                                       int   n_threads, /* = WSP_GGML_DEFAULT_N_THREADS */
                    struct wsp_ggml_threadpool * threadpool /* = NULL */ );
    WSP_GGML_API enum wsp_ggml_status  wsp_ggml_graph_compute(struct wsp_ggml_cgraph * cgraph, struct wsp_ggml_cplan * cplan);

    // same as wsp_ggml_graph_compute() but the work data is allocated as a part of the context
    // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
    WSP_GGML_API enum wsp_ggml_status  wsp_ggml_graph_compute_with_ctx(struct wsp_ggml_context * ctx, struct wsp_ggml_cgraph * cgraph, int n_threads);

    // TODO: move to backend interface
    WSP_GGML_API int wsp_ggml_cpu_has_neon       (void);
    WSP_GGML_API int wsp_ggml_cpu_has_sve        (void);
    WSP_GGML_API int wsp_ggml_cpu_has_matmul_int8(void);
    // get the sve vector length in bytes
    WSP_GGML_API int wsp_ggml_cpu_get_sve_cnt(void);

    // Internal types and functions exposed for tests and benchmarks

    typedef void (*wsp_ggml_from_float_to_mat_t)
                                     (const float * WSP_GGML_RESTRICT x, void * WSP_GGML_RESTRICT y, int64_t nr, int64_t k, int64_t bs);
    typedef void (*wsp_ggml_vec_dot_t)  (int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT x, size_t bx,
                                       const void * WSP_GGML_RESTRICT y, size_t by, int nrc);
    typedef void (*wsp_ggml_gemv_t)     (int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT x,
                                       const void * WSP_GGML_RESTRICT y, int nr, int nc);
    typedef void (*wsp_ggml_gemm_t)     (int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT x,
                                       const void * WSP_GGML_RESTRICT y, int nr, int nc);

    struct wsp_ggml_type_traits_cpu {
        wsp_ggml_from_float_to_mat_t from_float_to_mat;
        wsp_ggml_vec_dot_t           vec_dot;
        enum wsp_ggml_type           vec_dot_type;
        int64_t                  nrows; // number of rows to process simultaneously
        int64_t                  ncols; // number of columns to process simultaneously
        wsp_ggml_gemv_t              gemv;
        wsp_ggml_gemm_t              gemm;
    };

    WSP_GGML_API const struct wsp_ggml_type_traits_cpu * wsp_ggml_get_type_traits_cpu(enum wsp_ggml_type type);

    WSP_GGML_API void wsp_ggml_cpu_init(void);

    //
    // CPU backend
    //

    WSP_GGML_API wsp_ggml_backend_t wsp_ggml_backend_cpu_init(void);

    WSP_GGML_API bool wsp_ggml_backend_is_cpu                (wsp_ggml_backend_t backend);
    WSP_GGML_API void wsp_ggml_backend_cpu_set_n_threads     (wsp_ggml_backend_t backend_cpu, int n_threads);
    WSP_GGML_API void wsp_ggml_backend_cpu_set_threadpool    (wsp_ggml_backend_t backend_cpu, wsp_ggml_threadpool_t threadpool);
    WSP_GGML_API void wsp_ggml_backend_cpu_set_abort_callback(wsp_ggml_backend_t backend_cpu, wsp_ggml_abort_callback abort_callback, void * abort_callback_data);

    WSP_GGML_API wsp_ggml_backend_reg_t wsp_ggml_backend_cpu_reg(void);

#ifdef WSP_GGML_USE_CPU_HBM
    WSP_GGML_API wsp_ggml_backend_buffer_type_t wsp_ggml_backend_cpu_hbm_buffer_type(void);
#endif

#ifdef __cplusplus
}
#endif
