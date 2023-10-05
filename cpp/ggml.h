#pragma once

//
// GGML Tensor Library
//
// This documentation is still a work in progress.
// If you wish some specific topics to be covered, feel free to drop a comment:
//
//   https://github.com/ggerganov/whisper.cpp/issues/40
//
// ## Overview
//
// This library implements:
//
//  - a set of tensor operations
//  - automatic differentiation
//  - basic optimization algorithms
//
// The aim of this library is to provide a minimalistic approach for various machine learning tasks. This includes,
// but is not limited to, the following:
//
//  - linear regression
//  - support vector machines
//  - neural networks
//
// The library allows the user to define a certain function using the available tensor operations. This function
// definition is represented internally via a computation graph. Each tensor operation in the function definition
// corresponds to a node in the graph. Having the computation graph defined, the user can choose to compute the
// function's value and/or its gradient with respect to the input variables. Optionally, the function can be optimized
// using one of the available optimization algorithms.
//
// For example, here we define the function: f(x) = a*x^2 + b
//
//   {
//       struct wsp_ggml_init_params params = {
//           .mem_size   = 16*1024*1024,
//           .mem_buffer = NULL,
//       };
//
//       // memory allocation happens here
//       struct wsp_ggml_context * ctx = wsp_ggml_init(params);
//
//       struct wsp_ggml_tensor * x = wsp_ggml_new_tensor_1d(ctx, WSP_GGML_TYPE_F32, 1);
//
//       wsp_ggml_set_param(ctx, x); // x is an input variable
//
//       struct wsp_ggml_tensor * a  = wsp_ggml_new_tensor_1d(ctx, WSP_GGML_TYPE_F32, 1);
//       struct wsp_ggml_tensor * b  = wsp_ggml_new_tensor_1d(ctx, WSP_GGML_TYPE_F32, 1);
//       struct wsp_ggml_tensor * x2 = wsp_ggml_mul(ctx, x, x);
//       struct wsp_ggml_tensor * f  = wsp_ggml_add(ctx, wsp_ggml_mul(ctx, a, x2), b);
//
//       ...
//   }
//
// Notice that the function definition above does not involve any actual computation. The computation is performed only
// when the user explicitly requests it. For example, to compute the function's value at x = 2.0:
//
//   {
//       ...
//
//       struct wsp_ggml_cgraph gf = wsp_ggml_build_forward(f);
//
//       // set the input variable and parameter values
//       wsp_ggml_set_f32(x, 2.0f);
//       wsp_ggml_set_f32(a, 3.0f);
//       wsp_ggml_set_f32(b, 4.0f);
//
//       wsp_ggml_graph_compute_with_ctx(ctx, &gf, n_threads);
//
//       printf("f = %f\n", wsp_ggml_get_f32_1d(f, 0));
//
//       ...
//   }
//
// The actual computation is performed in the wsp_ggml_graph_compute() function.
//
// The wsp_ggml_new_tensor_...() functions create new tensors. They are allocated in the memory buffer provided to the
// wsp_ggml_init() function. You have to be careful not to exceed the memory buffer size. Therefore, you have to know
// in advance how much memory you need for your computation. Alternatively, you can allocate a large enough memory
// and after defining the computation graph, call the wsp_ggml_used_mem() function to find out how much memory was
// actually needed.
//
// The wsp_ggml_set_param() function marks a tensor as an input variable. This is used by the automatic
// differentiation and optimization algorithms.
//
// The described approach allows to define the function graph once and then compute its forward or backward graphs
// multiple times. All computations will use the same memory buffer allocated in the wsp_ggml_init() function. This way
// the user can avoid the memory allocation overhead at runtime.
//
// The library supports multi-dimensional tensors - up to 4 dimensions. The FP16 and FP32 data types are first class
// citizens, but in theory the library can be extended to support FP8 and integer data types.
//
// Each tensor operation produces a new tensor. Initially the library was envisioned to support only the use of unary
// and binary operations. Most of the available operations fall into one of these two categories. With time, it became
// clear that the library needs to support more complex operations. The way to support these operations is not clear
// yet, but a few examples are demonstrated in the following operations:
//
//   - wsp_ggml_permute()
//   - wsp_ggml_conv_1d_1s()
//   - wsp_ggml_conv_1d_2s()
//
// For each tensor operator, the library implements a forward and backward computation function. The forward function
// computes the output tensor value given the input tensor values. The backward function computes the adjoint of the
// input tensors given the adjoint of the output tensor. For a detailed explanation of what this means, take a
// calculus class, or watch the following video:
//
//   What is Automatic Differentiation?
//   https://www.youtube.com/watch?v=wG_nF1awSSY
//
//
// ## Tensor data (struct wsp_ggml_tensor)
//
// The tensors are stored in memory via the wsp_ggml_tensor struct. The structure provides information about the size of
// the tensor, the data type, and the memory buffer where the tensor data is stored. Additionally, it contains
// pointers to the "source" tensors - i.e. the tensors that were used to compute the current tensor. For example:
//
//   {
//       struct wsp_ggml_tensor * c = wsp_ggml_add(ctx, a, b);
//
//       assert(c->src[0] == a);
//       assert(c->src[1] == b);
//   }
//
// The multi-dimensional tensors are stored in row-major order. The wsp_ggml_tensor struct contains fields for the
// number of elements in each dimension ("ne") as well as the number of bytes ("nb", a.k.a. stride). This allows
// to store tensors that are not contiguous in memory, which is useful for operations such as transposition and
// permutation. All tensor operations have to take the stride into account and not assume that the tensor is
// contiguous in memory.
//
// The data of the tensor is accessed via the "data" pointer. For example:
//
//   {
//       const int nx = 2;
//       const int ny = 3;
//
//       struct wsp_ggml_tensor * a = wsp_ggml_new_tensor_2d(ctx, WSP_GGML_TYPE_F32, nx, ny);
//
//       for (int y = 0; y < ny; y++) {
//           for (int x = 0; x < nx; x++) {
//               *(float *) ((char *) a->data + y*a->nb[1] + x*a->nb[0]) = x + y;
//           }
//       }
//
//       ...
//   }
//
// Alternatively, there are helper functions, such as wsp_ggml_get_f32_1d() and wsp_ggml_set_f32_1d() that can be used.
//
// ## The matrix multiplication operator (wsp_ggml_mul_mat)
//
// TODO
//
//
// ## Multi-threading
//
// TODO
//
//
// ## Overview of ggml.c
//
// TODO
//
//
// ## SIMD optimizations
//
// TODO
//
//
// ## Debugging ggml
//
// TODO
//
//

#ifdef WSP_GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef WSP_GGML_BUILD
#            define WSP_GGML_API __declspec(dllexport)
#        else
#            define WSP_GGML_API __declspec(dllimport)
#        endif
#    else
#        define WSP_GGML_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define WSP_GGML_API
#endif

// TODO: support for clang
#ifdef __GNUC__
#    define WSP_GGML_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define WSP_GGML_DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define WSP_GGML_DEPRECATED(func, hint) func
#endif

#ifndef __GNUC__
#    define WSP_GGML_ATTRIBUTE_FORMAT(...)
#elif defined(__MINGW32__)
#    define WSP_GGML_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#    define WSP_GGML_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define WSP_GGML_FILE_MAGIC   0x67676d6c // "ggml"
#define WSP_GGML_FILE_VERSION 1

#define WSP_GGML_QNT_VERSION        2    // bump this on quantization format changes
#define WSP_GGML_QNT_VERSION_FACTOR 1000 // do not change this

#define WSP_GGML_MAX_DIMS          4
#define WSP_GGML_MAX_NODES         4096
#define WSP_GGML_MAX_PARAMS        256
#define WSP_GGML_MAX_CONTEXTS      64
#define WSP_GGML_MAX_SRC           6
#define WSP_GGML_MAX_NAME          64
#define WSP_GGML_MAX_OP_PARAMS     32
#define WSP_GGML_DEFAULT_N_THREADS 4

#if UINTPTR_MAX == 0xFFFFFFFF
    #define WSP_GGML_MEM_ALIGN 4
#else
    #define WSP_GGML_MEM_ALIGN 16
#endif

#define WSP_GGML_EXIT_SUCCESS 0
#define WSP_GGML_EXIT_ABORTED 1

#define GGUF_MAGIC   0x46554747 // "GGUF"
#define GGUF_VERSION 2

#define GGUF_DEFAULT_ALIGNMENT 32

#define WSP_GGML_UNUSED(x) (void)(x)

#define WSP_GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

#define WSP_GGML_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "WSP_GGML_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

// used to copy the number of elements and stride in bytes of tensors into local variables.
// main purpose is to reduce code duplication and improve readability.
//
// example:
//
//    WSP_GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
//    WSP_GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb);
//
#define WSP_GGML_TENSOR_LOCALS_1(type, prefix, pointer, array) \
    const type prefix##0 = (pointer)->array[0]; \
    WSP_GGML_UNUSED(prefix##0);
#define WSP_GGML_TENSOR_LOCALS_2(type, prefix, pointer, array) \
    WSP_GGML_TENSOR_LOCALS_1    (type, prefix, pointer, array) \
    const type prefix##1 = (pointer)->array[1]; \
    WSP_GGML_UNUSED(prefix##1);
#define WSP_GGML_TENSOR_LOCALS_3(type, prefix, pointer, array) \
    WSP_GGML_TENSOR_LOCALS_2    (type, prefix, pointer, array) \
    const type prefix##2 = (pointer)->array[2]; \
    WSP_GGML_UNUSED(prefix##2);
#define WSP_GGML_TENSOR_LOCALS(type, prefix, pointer, array) \
    WSP_GGML_TENSOR_LOCALS_3  (type, prefix, pointer, array) \
    const type prefix##3 = (pointer)->array[3]; \
    WSP_GGML_UNUSED(prefix##3);

#ifdef  __cplusplus
extern "C" {
#endif

#if defined(__ARM_NEON) && defined(__CUDACC__)
    typedef half wsp_ggml_fp16_t;
#elif defined(__ARM_NEON)
    typedef __fp16 wsp_ggml_fp16_t;
#else
    typedef uint16_t wsp_ggml_fp16_t;
#endif

    // convert FP16 <-> FP32
    WSP_GGML_API float       wsp_ggml_fp16_to_fp32(wsp_ggml_fp16_t x);
    WSP_GGML_API wsp_ggml_fp16_t wsp_ggml_fp32_to_fp16(float x);

    WSP_GGML_API void wsp_ggml_fp16_to_fp32_row(const wsp_ggml_fp16_t * x, float * y, int n);
    WSP_GGML_API void wsp_ggml_fp32_to_fp16_row(const float * x, wsp_ggml_fp16_t * y, int n);

    struct wsp_ggml_object;
    struct wsp_ggml_context;

    enum wsp_ggml_type {
        WSP_GGML_TYPE_F32  = 0,
        WSP_GGML_TYPE_F16  = 1,
        WSP_GGML_TYPE_Q4_0 = 2,
        WSP_GGML_TYPE_Q4_1 = 3,
        // WSP_GGML_TYPE_Q4_2 = 4, support has been removed
        // WSP_GGML_TYPE_Q4_3 (5) support has been removed
        WSP_GGML_TYPE_Q5_0 = 6,
        WSP_GGML_TYPE_Q5_1 = 7,
        WSP_GGML_TYPE_Q8_0 = 8,
        WSP_GGML_TYPE_Q8_1 = 9,
        // k-quantizations
        WSP_GGML_TYPE_Q2_K = 10,
        WSP_GGML_TYPE_Q3_K = 11,
        WSP_GGML_TYPE_Q4_K = 12,
        WSP_GGML_TYPE_Q5_K = 13,
        WSP_GGML_TYPE_Q6_K = 14,
        WSP_GGML_TYPE_Q8_K = 15,
        WSP_GGML_TYPE_I8,
        WSP_GGML_TYPE_I16,
        WSP_GGML_TYPE_I32,
        WSP_GGML_TYPE_COUNT,
    };

    enum wsp_ggml_backend {
        WSP_GGML_BACKEND_CPU = 0,
        WSP_GGML_BACKEND_GPU = 10,
        WSP_GGML_BACKEND_GPU_SPLIT = 20,
    };

    // model file types
    enum wsp_ggml_ftype {
        WSP_GGML_FTYPE_UNKNOWN     = -1,
        WSP_GGML_FTYPE_ALL_F32     = 0,
        WSP_GGML_FTYPE_MOSTLY_F16  = 1,  // except 1d tensors
        WSP_GGML_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
        WSP_GGML_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
        WSP_GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
        WSP_GGML_FTYPE_MOSTLY_Q8_0 = 7,  // except 1d tensors
        WSP_GGML_FTYPE_MOSTLY_Q5_0 = 8,  // except 1d tensors
        WSP_GGML_FTYPE_MOSTLY_Q5_1 = 9,  // except 1d tensors
        WSP_GGML_FTYPE_MOSTLY_Q2_K = 10, // except 1d tensors
        WSP_GGML_FTYPE_MOSTLY_Q3_K = 11, // except 1d tensors
        WSP_GGML_FTYPE_MOSTLY_Q4_K = 12, // except 1d tensors
        WSP_GGML_FTYPE_MOSTLY_Q5_K = 13, // except 1d tensors
        WSP_GGML_FTYPE_MOSTLY_Q6_K = 14, // except 1d tensors
    };

    // available tensor operations:
    enum wsp_ggml_op {
        WSP_GGML_OP_NONE = 0,

        WSP_GGML_OP_DUP,
        WSP_GGML_OP_ADD,
        WSP_GGML_OP_ADD1,
        WSP_GGML_OP_ACC,
        WSP_GGML_OP_SUB,
        WSP_GGML_OP_MUL,
        WSP_GGML_OP_DIV,
        WSP_GGML_OP_SQR,
        WSP_GGML_OP_SQRT,
        WSP_GGML_OP_LOG,
        WSP_GGML_OP_SUM,
        WSP_GGML_OP_SUM_ROWS,
        WSP_GGML_OP_MEAN,
        WSP_GGML_OP_ARGMAX,
        WSP_GGML_OP_REPEAT,
        WSP_GGML_OP_REPEAT_BACK,
        WSP_GGML_OP_CONCAT,
        WSP_GGML_OP_SILU_BACK,
        WSP_GGML_OP_NORM, // normalize
        WSP_GGML_OP_RMS_NORM,
        WSP_GGML_OP_RMS_NORM_BACK,
        WSP_GGML_OP_GROUP_NORM,

        WSP_GGML_OP_MUL_MAT,
        WSP_GGML_OP_OUT_PROD,

        WSP_GGML_OP_SCALE,
        WSP_GGML_OP_SET,
        WSP_GGML_OP_CPY,
        WSP_GGML_OP_CONT,
        WSP_GGML_OP_RESHAPE,
        WSP_GGML_OP_VIEW,
        WSP_GGML_OP_PERMUTE,
        WSP_GGML_OP_TRANSPOSE,
        WSP_GGML_OP_GET_ROWS,
        WSP_GGML_OP_GET_ROWS_BACK,
        WSP_GGML_OP_DIAG,
        WSP_GGML_OP_DIAG_MASK_INF,
        WSP_GGML_OP_DIAG_MASK_ZERO,
        WSP_GGML_OP_SOFT_MAX,
        WSP_GGML_OP_SOFT_MAX_BACK,
        WSP_GGML_OP_ROPE,
        WSP_GGML_OP_ROPE_BACK,
        WSP_GGML_OP_ALIBI,
        WSP_GGML_OP_CLAMP,
        WSP_GGML_OP_CONV_1D,
        WSP_GGML_OP_CONV_2D,
        WSP_GGML_OP_CONV_TRANSPOSE_2D,
        WSP_GGML_OP_POOL_1D,
        WSP_GGML_OP_POOL_2D,

        WSP_GGML_OP_UPSCALE, // nearest interpolate

        WSP_GGML_OP_FLASH_ATTN,
        WSP_GGML_OP_FLASH_FF,
        WSP_GGML_OP_FLASH_ATTN_BACK,
        WSP_GGML_OP_WIN_PART,
        WSP_GGML_OP_WIN_UNPART,
        WSP_GGML_OP_GET_REL_POS,
        WSP_GGML_OP_ADD_REL_POS,

        WSP_GGML_OP_UNARY,

        WSP_GGML_OP_MAP_UNARY,
        WSP_GGML_OP_MAP_BINARY,

        WSP_GGML_OP_MAP_CUSTOM1_F32,
        WSP_GGML_OP_MAP_CUSTOM2_F32,
        WSP_GGML_OP_MAP_CUSTOM3_F32,

        WSP_GGML_OP_MAP_CUSTOM1,
        WSP_GGML_OP_MAP_CUSTOM2,
        WSP_GGML_OP_MAP_CUSTOM3,

        WSP_GGML_OP_CROSS_ENTROPY_LOSS,
        WSP_GGML_OP_CROSS_ENTROPY_LOSS_BACK,

        WSP_GGML_OP_COUNT,
    };

    enum wsp_ggml_unary_op {
        WSP_GGML_UNARY_OP_ABS,
        WSP_GGML_UNARY_OP_SGN,
        WSP_GGML_UNARY_OP_NEG,
        WSP_GGML_UNARY_OP_STEP,
        WSP_GGML_UNARY_OP_TANH,
        WSP_GGML_UNARY_OP_ELU,
        WSP_GGML_UNARY_OP_RELU,
        WSP_GGML_UNARY_OP_GELU,
        WSP_GGML_UNARY_OP_GELU_QUICK,
        WSP_GGML_UNARY_OP_SILU,
    };

    enum wsp_ggml_object_type {
        WSP_GGML_OBJECT_TENSOR,
        WSP_GGML_OBJECT_GRAPH,
        WSP_GGML_OBJECT_WORK_BUFFER
    };

    // ggml object
    struct wsp_ggml_object {
        size_t offs;
        size_t size;

        struct wsp_ggml_object * next;

        enum wsp_ggml_object_type type;

        char padding[4];
    };

    static const size_t WSP_GGML_OBJECT_SIZE = sizeof(struct wsp_ggml_object);

    // n-dimensional tensor
    struct wsp_ggml_tensor {
        enum wsp_ggml_type    type;
        enum wsp_ggml_backend backend;

        int     n_dims;
        int64_t ne[WSP_GGML_MAX_DIMS]; // number of elements
        size_t  nb[WSP_GGML_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = sizeof(type)
                                   // nb[1] = nb[0]   * ne[0] + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // compute data
        enum wsp_ggml_op op;

        // op params - allocated as int32_t for alignment
        int32_t op_params[WSP_GGML_MAX_OP_PARAMS / sizeof(int32_t)];

        bool is_param;

        struct wsp_ggml_tensor * grad;
        struct wsp_ggml_tensor * src[WSP_GGML_MAX_SRC];

        // performance
        int     perf_runs;
        int64_t perf_cycles;
        int64_t perf_time_us;

        struct wsp_ggml_tensor * view_src;
        size_t               view_offs;

        void * data;

        char name[WSP_GGML_MAX_NAME];

        void * extra; // extra things e.g. for ggml-cuda.cu

        char padding[4];
    };

    static const size_t WSP_GGML_TENSOR_SIZE = sizeof(struct wsp_ggml_tensor);

    // the compute plan that needs to be prepared for wsp_ggml_graph_compute()
    // since https://github.com/ggerganov/ggml/issues/287
    struct wsp_ggml_cplan {
        size_t    work_size; // size of work buffer, calculated by `wsp_ggml_graph_plan()`
        uint8_t * work_data; // work buffer, to be allocated by caller before calling to `wsp_ggml_graph_compute()`

        int n_threads;

        // the `n_tasks` of nodes, 1:1 mapping to cgraph nodes
        int n_tasks[WSP_GGML_MAX_NODES];

        // abort wsp_ggml_graph_compute when true
        bool (*abort_callback)(void * data);
        void * abort_callback_data;
    };

    // next prime after WSP_GGML_MAX_NODES
    // #define WSP_GGML_GRAPH_HASHTABLE_SIZE 4099
    // next prime after WSP_GGML_MAX_NODES * 2 (nodes + leafs)
    #define WSP_GGML_GRAPH_HASHTABLE_SIZE 8273

    // computation graph
    struct wsp_ggml_cgraph {
        int n_nodes;
        int n_leafs;

        struct wsp_ggml_tensor * nodes[WSP_GGML_MAX_NODES];
        struct wsp_ggml_tensor * grads[WSP_GGML_MAX_NODES];
        struct wsp_ggml_tensor * leafs[WSP_GGML_MAX_NODES];

        void * visited_hash_table[WSP_GGML_GRAPH_HASHTABLE_SIZE];

        // performance
        int     perf_runs;
        int64_t perf_cycles;
        int64_t perf_time_us;
    };

    static const size_t WSP_GGML_GRAPH_SIZE = sizeof(struct wsp_ggml_cgraph);

    // scratch buffer
    struct wsp_ggml_scratch {
        size_t offs;
        size_t size;
        void * data;
    };

    struct wsp_ggml_init_params {
        // memory pool
        size_t mem_size;   // bytes
        void * mem_buffer; // if NULL, memory will be allocated internally
        bool   no_alloc;   // don't allocate memory for the tensor data
    };


    // compute types

    // NOTE: the INIT or FINALIZE pass is not scheduled unless explicitly enabled.
    // This behavior was changed since https://github.com/ggerganov/llama.cpp/pull/1995.
    enum wsp_ggml_task_type {
        WSP_GGML_TASK_INIT = 0,
        WSP_GGML_TASK_COMPUTE,
        WSP_GGML_TASK_FINALIZE,
    };

    struct wsp_ggml_compute_params {
        enum wsp_ggml_task_type type;

        // ith = thread index, nth = number of threads
        int ith, nth;

        // work buffer for all threads
        size_t wsize;
        void * wdata;
    };

    // misc

    WSP_GGML_API void    wsp_ggml_time_init(void); // call this once at the beginning of the program
    WSP_GGML_API int64_t wsp_ggml_time_ms(void);
    WSP_GGML_API int64_t wsp_ggml_time_us(void);
    WSP_GGML_API int64_t wsp_ggml_cycles(void);
    WSP_GGML_API int64_t wsp_ggml_cycles_per_ms(void);

    WSP_GGML_API void    wsp_ggml_numa_init(void); // call once for better performance on NUMA systems
    WSP_GGML_API bool    wsp_ggml_is_numa(void); // true if init detected that system has >1 NUMA node

    WSP_GGML_API void    wsp_ggml_print_object (const struct wsp_ggml_object * obj);
    WSP_GGML_API void    wsp_ggml_print_objects(const struct wsp_ggml_context * ctx);

    WSP_GGML_API int64_t wsp_ggml_nelements   (const struct wsp_ggml_tensor * tensor);
    WSP_GGML_API int64_t wsp_ggml_nrows       (const struct wsp_ggml_tensor * tensor);
    WSP_GGML_API size_t  wsp_ggml_nbytes      (const struct wsp_ggml_tensor * tensor);
    WSP_GGML_API size_t  wsp_ggml_nbytes_pad  (const struct wsp_ggml_tensor * tensor); // same as wsp_ggml_nbytes() but padded to WSP_GGML_MEM_ALIGN
    WSP_GGML_API size_t  wsp_ggml_nbytes_split(const struct wsp_ggml_tensor * tensor, int nrows_split);

    WSP_GGML_API int     wsp_ggml_blck_size (enum wsp_ggml_type type);
    WSP_GGML_API size_t  wsp_ggml_type_size (enum wsp_ggml_type type); // size in bytes for all elements in a block
    WSP_GGML_API float   wsp_ggml_type_sizef(enum wsp_ggml_type type); // wsp_ggml_type_size()/wsp_ggml_blck_size() as float

    WSP_GGML_API const char * wsp_ggml_type_name(enum wsp_ggml_type type);
    WSP_GGML_API const char * wsp_ggml_op_name  (enum wsp_ggml_op   op);
    WSP_GGML_API const char * wsp_ggml_op_symbol(enum wsp_ggml_op   op);

    WSP_GGML_API size_t  wsp_ggml_element_size(const struct wsp_ggml_tensor * tensor);

    WSP_GGML_API bool    wsp_ggml_is_quantized(enum wsp_ggml_type type);

    // TODO: temporary until model loading of ggml examples is refactored
    WSP_GGML_API enum wsp_ggml_type wsp_ggml_ftype_to_wsp_ggml_type(enum wsp_ggml_ftype ftype);

    WSP_GGML_API bool wsp_ggml_is_transposed(const struct wsp_ggml_tensor * tensor);
    WSP_GGML_API bool wsp_ggml_is_contiguous(const struct wsp_ggml_tensor * tensor);
    WSP_GGML_API bool wsp_ggml_is_permuted  (const struct wsp_ggml_tensor * tensor);

    WSP_GGML_API bool wsp_ggml_are_same_shape(const struct wsp_ggml_tensor * t0, const struct wsp_ggml_tensor * t1);

    // use this to compute the memory overhead of a tensor
    WSP_GGML_API size_t wsp_ggml_tensor_overhead(void);

    // main

    WSP_GGML_API struct wsp_ggml_context * wsp_ggml_init(struct wsp_ggml_init_params params);
    WSP_GGML_API void                  wsp_ggml_free(struct wsp_ggml_context * ctx);

    WSP_GGML_API size_t  wsp_ggml_used_mem(const struct wsp_ggml_context * ctx);

    WSP_GGML_API size_t  wsp_ggml_set_scratch (struct wsp_ggml_context * ctx, struct wsp_ggml_scratch scratch);
    WSP_GGML_API bool    wsp_ggml_get_no_alloc(struct wsp_ggml_context * ctx);
    WSP_GGML_API void    wsp_ggml_set_no_alloc(struct wsp_ggml_context * ctx, bool no_alloc);

    WSP_GGML_API void *  wsp_ggml_get_mem_buffer     (const struct wsp_ggml_context * ctx);
    WSP_GGML_API size_t  wsp_ggml_get_mem_size       (const struct wsp_ggml_context * ctx);
    WSP_GGML_API size_t  wsp_ggml_get_max_tensor_size(const struct wsp_ggml_context * ctx);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_new_tensor(
            struct wsp_ggml_context * ctx,
            enum   wsp_ggml_type type,
            int    n_dims,
            const int64_t *ne);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_new_tensor_1d(
            struct wsp_ggml_context * ctx,
            enum   wsp_ggml_type type,
            int64_t ne0);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_new_tensor_2d(
            struct wsp_ggml_context * ctx,
            enum   wsp_ggml_type type,
            int64_t ne0,
            int64_t ne1);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_new_tensor_3d(
            struct wsp_ggml_context * ctx,
            enum   wsp_ggml_type type,
            int64_t ne0,
            int64_t ne1,
            int64_t ne2);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_new_tensor_4d(
            struct wsp_ggml_context * ctx,
            enum   wsp_ggml_type type,
            int64_t ne0,
            int64_t ne1,
            int64_t ne2,
            int64_t ne3);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_new_i32(struct wsp_ggml_context * ctx, int32_t value);
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_new_f32(struct wsp_ggml_context * ctx, float value);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_dup_tensor (struct wsp_ggml_context * ctx, const struct wsp_ggml_tensor * src);
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_view_tensor(struct wsp_ggml_context * ctx, struct wsp_ggml_tensor * src);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_get_tensor(struct wsp_ggml_context * ctx, const char * name);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_set_zero(struct wsp_ggml_tensor * tensor);
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_set_i32 (struct wsp_ggml_tensor * tensor, int32_t value);
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_set_f32 (struct wsp_ggml_tensor * tensor, float value);

    WSP_GGML_API int32_t wsp_ggml_get_i32_1d(const struct wsp_ggml_tensor * tensor, int i);
    WSP_GGML_API void    wsp_ggml_set_i32_1d(const struct wsp_ggml_tensor * tensor, int i, int32_t value);

    WSP_GGML_API float   wsp_ggml_get_f32_1d(const struct wsp_ggml_tensor * tensor, int i);
    WSP_GGML_API void    wsp_ggml_set_f32_1d(const struct wsp_ggml_tensor * tensor, int i, float value);

    WSP_GGML_API void *  wsp_ggml_get_data    (const struct wsp_ggml_tensor * tensor);
    WSP_GGML_API float * wsp_ggml_get_data_f32(const struct wsp_ggml_tensor * tensor);

    WSP_GGML_API enum wsp_ggml_unary_op wsp_ggml_get_unary_op(const struct wsp_ggml_tensor * tensor);

    WSP_GGML_API const char *         wsp_ggml_get_name   (const struct wsp_ggml_tensor * tensor);
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_set_name   (      struct wsp_ggml_tensor * tensor, const char * name);
    WSP_GGML_ATTRIBUTE_FORMAT(2, 3)
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_format_name(      struct wsp_ggml_tensor * tensor, const char * fmt, ...);

    //
    // operations on tensors with backpropagation
    //

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_dup(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    // in-place, returns view(a)
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_dup_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_add(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_add_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_add1(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_add1_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_acc(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_acc_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_sub(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_sub_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_mul(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_mul_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_div(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_div_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_sqr(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_sqr_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_sqrt(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_sqrt_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_log(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_log_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    // return scalar
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_sum(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    // sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_sum_rows(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    // mean along rows
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_mean(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    // argmax along rows
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_argmax(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    // if a is the same shape as b, and a is not parameter, return a
    // otherwise, return a new tensor: repeat(a) to fit in b
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_repeat(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_repeat_back(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    // concat a and b on dim 2
    // used in stable-diffusion
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_concat(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_abs(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_abs_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_sgn(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_sgn_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_neg(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_neg_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_step(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_step_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_tanh(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_tanh_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_elu(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_elu_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_relu(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_relu_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    // TODO: double-check this computation is correct
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_gelu(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_gelu_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_gelu_quick(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_gelu_quick_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_silu(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_silu_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    // a - x
    // b - dy
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_silu_back(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    // normalize along rows
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_norm(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            float                 eps);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_norm_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            float                 eps);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_rms_norm(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            float                 eps);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_rms_norm_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            float                 eps);

    // group normalize along ne0*ne1*n_groups
    // used in stable-diffusion
    // TODO: eps is hardcoded to 1e-6 for now
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_group_norm(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int                   n_groups);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_group_norm_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int                   n_groups);

    // a - x
    // b - dy
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_rms_norm_back(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b,
            float                 eps);

    // A: n columns, m rows
    // B: n columns, p rows  (i.e. we transpose it internally)
    // result is m columns, p rows
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_mul_mat(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    // A: m columns, n rows,
    // B: p columns, n rows,
    // result is m columns, p rows
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_out_prod(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    //
    // operations on tensors without backpropagation
    //

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_scale(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    // in-place, returns view(a)
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_scale_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    // b -> view(a,offset,nb1,nb2,3), return modified a
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_set(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    // b -> view(a,offset,nb1,nb2,3), return view(a)
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_set_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_set_1d(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b,
            size_t                offset);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_set_1d_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b,
            size_t                offset);

    // b -> view(a,offset,nb1,nb2,3), return modified a
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_set_2d(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b,
            size_t                nb1,
            size_t                offset);

    // b -> view(a,offset,nb1,nb2,3), return view(a)
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_set_2d_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b,
            size_t                nb1,
            size_t                offset);


    // a -> b, return view(b)
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_cpy(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    // a -> b, in-place, return view(b)
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_cpy_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    // make contiguous
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_cont(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    // make contiguous, in-place
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_cont_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    // return view(a), b specifies the new shape
    // TODO: when we start computing gradient, make a copy instead of view
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_reshape(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    // return view(a)
    // TODO: when we start computing gradient, make a copy instead of view
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_reshape_1d(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int64_t               ne0);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_reshape_2d(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1);

    // return view(a)
    // TODO: when we start computing gradient, make a copy instead of view
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_reshape_3d(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_reshape_4d(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3);

    // offset in bytes
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_view_1d(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int64_t               ne0,
            size_t                offset);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_view_2d(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            size_t                nb1, // row stride in bytes
            size_t                offset);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_view_3d(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            size_t                nb1, // row   stride in bytes
            size_t                nb2, // slice stride in bytes
            size_t                offset);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_view_4d(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3,
            size_t                nb1, // row   stride in bytes
            size_t                nb2, // slice stride in bytes
            size_t                nb3,
            size_t                offset);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_permute(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int                   axis0,
            int                   axis1,
            int                   axis2,
            int                   axis3);

    // alias for wsp_ggml_permute(ctx, a, 1, 0, 2, 3)
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_transpose(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_get_rows(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_get_rows_back(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b,
            struct wsp_ggml_tensor  * c);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_diag(
        struct wsp_ggml_context     * ctx,
        struct wsp_ggml_tensor      * a);

    // set elements above the diagonal to -INF
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_diag_mask_inf(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int                   n_past);

    // in-place, returns view(a)
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_diag_mask_inf_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int                   n_past);

    // set elements above the diagonal to 0
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_diag_mask_zero(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int                   n_past);

    // in-place, returns view(a)
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_diag_mask_zero_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int                   n_past);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_soft_max(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    // in-place, returns view(a)
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_soft_max_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_soft_max_back(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    // in-place, returns view(a)
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_soft_max_back_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    // rotary position embedding
    // if mode & 1 == 1, skip n_past elements
    // if mode & 2 == 1, GPT-NeoX style
    // if mode & 4 == 1, ChatGLM style
    // TODO: avoid creating a new tensor every time
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_rope(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int                   n_past,
            int                   n_dims,
            int                   mode,
            int                   n_ctx);

    // in-place, returns view(a)
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_rope_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int                   n_past,
            int                   n_dims,
            int                   mode,
            int                   n_ctx);

    // custom RoPE
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_rope_custom(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int                   n_past,
            int                   n_dims,
            int                   mode,
            int                   n_ctx,
            float                 freq_base,
            float                 freq_scale);

    // in-place, returns view(a)
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_rope_custom_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int                   n_past,
            int                   n_dims,
            int                   mode,
            int                   n_ctx,
            float                 freq_base,
            float                 freq_scale);

    // xPos RoPE, in-place, returns view(a)
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_rope_xpos_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int                   n_past,
            int                   n_dims,
            float                 base,
            bool                  down);

    // rotary position embedding backward, i.e compute dx from dy
    // a - dy
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_rope_back(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int                   n_past,
            int                   n_dims,
            int                   mode,
            int                   n_ctx,
            float                 freq_base,
            float                 freq_scale,
            float                 xpos_base,
            bool                  xpos_down);

    // alibi position embedding
    // in-place, returns view(a)
    struct wsp_ggml_tensor * wsp_ggml_alibi(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int                   n_past,
            int                   n_head,
            float                 bias_max);

    // clamp
    // in-place, returns view(a)
    struct wsp_ggml_tensor * wsp_ggml_clamp(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            float                 min,
            float                 max);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_conv_1d(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b,
            int                   s0,  // stride
            int                   p0,  // padding
            int                   d0); // dilation

    // conv_1d with padding = half
    // alias for wsp_ggml_conv_1d(a, b, s, a->ne[0]/2, d)
    WSP_GGML_API struct wsp_ggml_tensor* wsp_ggml_conv_1d_ph(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b,
            int                   s,
            int                   d);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_conv_2d(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b,
            int                   s0,
            int                   s1,
            int                   p0,
            int                   p1,
            int                   d0,
            int                   d1);


    // kernel size is a->ne[0] x a->ne[1]
    // stride is equal to kernel size
    // padding is zero
    // example:
    // a:     16   16    3  768
    // b:   1024 1024    3    1
    // res:   64   64  768    1
    // used in sam
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_conv_2d_sk_p0(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    // kernel size is a->ne[0] x a->ne[1]
    // stride is 1
    // padding is half
    // example:
    // a:      3    3    256  256
    // b:     64   64    256    1
    // res:   64   64    256    1
    // used in sam
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_conv_2d_s1_ph(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_conv_transpose_2d_p0(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b,
            int                   stride);

    enum wsp_ggml_op_pool {
        WSP_GGML_OP_POOL_MAX,
        WSP_GGML_OP_POOL_AVG,
        WSP_GGML_OP_POOL_COUNT,
    };

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_pool_1d(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            enum wsp_ggml_op_pool     op,
            int                   k0, // kernel size
            int                   s0, // stride
            int                   p0); // padding

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_pool_2d(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            enum wsp_ggml_op_pool     op,
            int                   k0,
            int                   k1,
            int                   s0,
            int                   s1,
            int                   p0,
            int                   p1);

    // nearest interpolate
    // used in stable-diffusion
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_upscale(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int                   scale_factor);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_flash_attn(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * q,
            struct wsp_ggml_tensor  * k,
            struct wsp_ggml_tensor  * v,
            bool                  masked);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_flash_attn_back(
           struct wsp_ggml_context * ctx,
           struct wsp_ggml_tensor  * q,
           struct wsp_ggml_tensor  * k,
           struct wsp_ggml_tensor  * v,
           struct wsp_ggml_tensor  * d,
           bool                  masked);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_flash_ff(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b0,
            struct wsp_ggml_tensor  * b1,
            struct wsp_ggml_tensor  * c0,
            struct wsp_ggml_tensor  * c1);

    // partition into non-overlapping windows with padding if needed
    // example:
    // a:   768   64   64    1
    // w:    14
    // res: 768   14   14    25
    // used in sam
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_win_part(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int                   w);

    // reverse of wsp_ggml_win_part
    // used in sam
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_win_unpart(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int                   w0,
            int                   h0,
            int                   w);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_unary(
            struct wsp_ggml_context * ctx,
             struct wsp_ggml_tensor * a,
             enum wsp_ggml_unary_op op);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_unary_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        enum wsp_ggml_unary_op op);

    // used in sam
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_get_rel_pos(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int                   qh,
            int                   kh);

    // used in sam

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_add_rel_pos(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * pw,
            struct wsp_ggml_tensor  * ph);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_add_rel_pos_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * pw,
            struct wsp_ggml_tensor  * ph);

    // custom operators

    typedef void (*wsp_ggml_unary_op_f32_t) (const int, float *, const float *);
    typedef void (*wsp_ggml_binary_op_f32_t)(const int, float *, const float *, const float *);

    typedef void (*wsp_ggml_custom1_op_f32_t)(struct wsp_ggml_tensor *, const struct wsp_ggml_tensor *);
    typedef void (*wsp_ggml_custom2_op_f32_t)(struct wsp_ggml_tensor *, const struct wsp_ggml_tensor *, const struct wsp_ggml_tensor *);
    typedef void (*wsp_ggml_custom3_op_f32_t)(struct wsp_ggml_tensor *, const struct wsp_ggml_tensor *, const struct wsp_ggml_tensor *, const struct wsp_ggml_tensor *);

    WSP_GGML_DEPRECATED(WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_unary_f32(
            struct wsp_ggml_context        * ctx,
            struct wsp_ggml_tensor         * a,
                   wsp_ggml_unary_op_f32_t   fun),
        "use wsp_ggml_map_custom1 instead");

    WSP_GGML_DEPRECATED(WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_unary_inplace_f32(
            struct wsp_ggml_context        * ctx,
            struct wsp_ggml_tensor         * a,
                   wsp_ggml_unary_op_f32_t   fun),
        "use wsp_ggml_map_custom1_inplace instead");

    WSP_GGML_DEPRECATED(WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_binary_f32(
            struct wsp_ggml_context         * ctx,
            struct wsp_ggml_tensor          * a,
            struct wsp_ggml_tensor          * b,
                   wsp_ggml_binary_op_f32_t   fun),
        "use wsp_ggml_map_custom2 instead");

    WSP_GGML_DEPRECATED(WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_binary_inplace_f32(
            struct wsp_ggml_context         * ctx,
            struct wsp_ggml_tensor          * a,
            struct wsp_ggml_tensor          * b,
                   wsp_ggml_binary_op_f32_t   fun),
        "use wsp_ggml_map_custom2_inplace instead");

    WSP_GGML_DEPRECATED(WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_custom1_f32(
            struct wsp_ggml_context          * ctx,
            struct wsp_ggml_tensor           * a,
                   wsp_ggml_custom1_op_f32_t   fun),
        "use wsp_ggml_map_custom1 instead");

    WSP_GGML_DEPRECATED(WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_custom1_inplace_f32(
            struct wsp_ggml_context          * ctx,
            struct wsp_ggml_tensor           * a,
                   wsp_ggml_custom1_op_f32_t   fun),
        "use wsp_ggml_map_custom1_inplace instead");

    WSP_GGML_DEPRECATED(WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_custom2_f32(
            struct wsp_ggml_context          * ctx,
            struct wsp_ggml_tensor           * a,
            struct wsp_ggml_tensor           * b,
                   wsp_ggml_custom2_op_f32_t   fun),
        "use wsp_ggml_map_custom2 instead");

    WSP_GGML_DEPRECATED(WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_custom2_inplace_f32(
            struct wsp_ggml_context          * ctx,
            struct wsp_ggml_tensor           * a,
            struct wsp_ggml_tensor           * b,
                   wsp_ggml_custom2_op_f32_t   fun),
        "use wsp_ggml_map_custom2_inplace instead");

    WSP_GGML_DEPRECATED(WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_custom3_f32(
            struct wsp_ggml_context          * ctx,
            struct wsp_ggml_tensor           * a,
            struct wsp_ggml_tensor           * b,
            struct wsp_ggml_tensor           * c,
                   wsp_ggml_custom3_op_f32_t   fun),
        "use wsp_ggml_map_custom3 instead");

    WSP_GGML_DEPRECATED(WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_custom3_inplace_f32(
            struct wsp_ggml_context          * ctx,
            struct wsp_ggml_tensor           * a,
            struct wsp_ggml_tensor           * b,
            struct wsp_ggml_tensor           * c,
                   wsp_ggml_custom3_op_f32_t   fun),
        "use wsp_ggml_map_custom3_inplace instead");

    // custom operators v2

    typedef void (*wsp_ggml_custom1_op_t)(struct wsp_ggml_tensor * dst , const struct wsp_ggml_tensor * a, int ith, int nth, void * userdata);
    typedef void (*wsp_ggml_custom2_op_t)(struct wsp_ggml_tensor * dst , const struct wsp_ggml_tensor * a, const struct wsp_ggml_tensor * b, int ith, int nth, void * userdata);
    typedef void (*wsp_ggml_custom3_op_t)(struct wsp_ggml_tensor * dst , const struct wsp_ggml_tensor * a, const struct wsp_ggml_tensor * b, const struct wsp_ggml_tensor * c, int ith, int nth, void * userdata);

    #define WSP_GGML_N_TASKS_MAX -1

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_custom1(
            struct wsp_ggml_context   * ctx,
            struct wsp_ggml_tensor    * a,
            wsp_ggml_custom1_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_custom1_inplace(
            struct wsp_ggml_context   * ctx,
            struct wsp_ggml_tensor    * a,
            wsp_ggml_custom1_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_custom2(
            struct wsp_ggml_context   * ctx,
            struct wsp_ggml_tensor    * a,
            struct wsp_ggml_tensor    * b,
            wsp_ggml_custom2_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_custom2_inplace(
            struct wsp_ggml_context   * ctx,
            struct wsp_ggml_tensor    * a,
            struct wsp_ggml_tensor    * b,
            wsp_ggml_custom2_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_custom3(
            struct wsp_ggml_context   * ctx,
            struct wsp_ggml_tensor    * a,
            struct wsp_ggml_tensor    * b,
            struct wsp_ggml_tensor    * c,
            wsp_ggml_custom3_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_custom3_inplace(
            struct wsp_ggml_context   * ctx,
            struct wsp_ggml_tensor    * a,
            struct wsp_ggml_tensor    * b,
            struct wsp_ggml_tensor    * c,
            wsp_ggml_custom3_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    // loss function

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_cross_entropy_loss(
            struct wsp_ggml_context         * ctx,
            struct wsp_ggml_tensor          * a,
            struct wsp_ggml_tensor          * b);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_cross_entropy_loss_back(
            struct wsp_ggml_context         * ctx,
            struct wsp_ggml_tensor          * a,
            struct wsp_ggml_tensor          * b,
            struct wsp_ggml_tensor          * c);

    //
    // automatic differentiation
    //

    WSP_GGML_API void wsp_ggml_set_param(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * tensor);


    WSP_GGML_API void wsp_ggml_build_forward_expand (struct wsp_ggml_cgraph * cgraph, struct wsp_ggml_tensor * tensor);
    WSP_GGML_API void wsp_ggml_build_backward_expand(struct wsp_ggml_context * ctx, struct wsp_ggml_cgraph * gf, struct wsp_ggml_cgraph * gb, bool keep);

    WSP_GGML_API struct wsp_ggml_cgraph wsp_ggml_build_forward (struct wsp_ggml_tensor * tensor);
    WSP_GGML_API struct wsp_ggml_cgraph wsp_ggml_build_backward(struct wsp_ggml_context * ctx, struct wsp_ggml_cgraph * gf, bool keep);

    // graph allocation in a context
    WSP_GGML_API struct wsp_ggml_cgraph * wsp_ggml_new_graph        (struct wsp_ggml_context * ctx);
    WSP_GGML_API struct wsp_ggml_cgraph * wsp_ggml_build_forward_ctx(struct wsp_ggml_context * ctx, struct wsp_ggml_tensor * tensor);
    WSP_GGML_API size_t wsp_ggml_graph_overhead(void);

    // wsp_ggml_graph_plan() has to be called before wsp_ggml_graph_compute()
    // when plan.work_size > 0, caller must allocate memory for plan.work_data
    WSP_GGML_API struct wsp_ggml_cplan wsp_ggml_graph_plan   (struct wsp_ggml_cgraph * cgraph, int n_threads /*= WSP_GGML_DEFAULT_N_THREADS*/);
    WSP_GGML_API               int wsp_ggml_graph_compute(struct wsp_ggml_cgraph * cgraph, struct wsp_ggml_cplan * cplan);
    WSP_GGML_API              void wsp_ggml_graph_reset  (struct wsp_ggml_cgraph * cgraph);

    // same as wsp_ggml_graph_compute() but the work data is allocated as a part of the context
    // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
    WSP_GGML_API void wsp_ggml_graph_compute_with_ctx(struct wsp_ggml_context * ctx, struct wsp_ggml_cgraph * cgraph, int n_threads);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_graph_get_tensor(struct wsp_ggml_cgraph * cgraph, const char * name);

    WSP_GGML_API void               wsp_ggml_graph_export(const struct wsp_ggml_cgraph * cgraph, const char * fname);
    WSP_GGML_API struct wsp_ggml_cgraph wsp_ggml_graph_import(const char * fname, struct wsp_ggml_context ** ctx_data, struct wsp_ggml_context ** ctx_eval);

    // print info and performance information for the graph
    WSP_GGML_API void wsp_ggml_graph_print(const struct wsp_ggml_cgraph * cgraph);

    // dump the graph into a file using the dot format
    WSP_GGML_API void wsp_ggml_graph_dump_dot(const struct wsp_ggml_cgraph * gb, const struct wsp_ggml_cgraph * gf, const char * filename);

    //
    // optimization
    //

    // optimization methods
    enum wsp_ggml_opt_type {
        WSP_GGML_OPT_ADAM,
        WSP_GGML_OPT_LBFGS,
    };

    // linesearch methods
    enum wsp_ggml_linesearch {
        WSP_GGML_LINESEARCH_DEFAULT = 1,

        WSP_GGML_LINESEARCH_BACKTRACKING_ARMIJO       = 0,
        WSP_GGML_LINESEARCH_BACKTRACKING_WOLFE        = 1,
        WSP_GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2,
    };

    // optimization return values
    enum wsp_ggml_opt_result {
        WSP_GGML_OPT_OK = 0,
        WSP_GGML_OPT_DID_NOT_CONVERGE,
        WSP_GGML_OPT_NO_CONTEXT,
        WSP_GGML_OPT_INVALID_WOLFE,
        WSP_GGML_OPT_FAIL,

        WSP_GGML_LINESEARCH_FAIL = -128,
        WSP_GGML_LINESEARCH_MINIMUM_STEP,
        WSP_GGML_LINESEARCH_MAXIMUM_STEP,
        WSP_GGML_LINESEARCH_MAXIMUM_ITERATIONS,
        WSP_GGML_LINESEARCH_INVALID_PARAMETERS,
    };

    typedef void (*wsp_ggml_opt_callback)(void * data, float * sched);

    // optimization parameters
    //
    //   see ggml.c (wsp_ggml_opt_default_params) for default values
    //
    struct wsp_ggml_opt_params {
        enum wsp_ggml_opt_type type;

        int n_threads;

        // delta-based convergence test
        //
        //   if past == 0 - disabled
        //   if past > 0:
        //     stop if |f(x) - f(x_past)| < delta * max(1, |f(x)|)
        //
        int past;
        float delta;

        // maximum number of iterations without improvement
        //
        //   if 0 - disabled
        //   if > 0:
        //     assume convergence if no cost improvement in this number of iterations
        //
        int max_no_improvement;

        bool print_forward_graph;
        bool print_backward_graph;

        // ADAM parameters
        struct {
            int n_iter;

            float sched; // schedule multiplier (fixed, decay or warmup)
            float decay; // weight decay for AdamW, use 0.0f to disable
            int   decay_min_ndim; // minimum number of tensor dimension to apply weight decay
            float alpha; // learning rate
            float beta1;
            float beta2;
            float eps;   // epsilon for numerical stability
            float eps_f; // epsilon for convergence test
            float eps_g; // epsilon for convergence test
            float gclip; // gradient clipping
        } adam;

        // LBFGS parameters
        struct {
            int m; // number of corrections to approximate the inv. Hessian
            int n_iter;
            int max_linesearch;

            float eps;      // convergence tolerance
            float ftol;     // line search tolerance
            float wolfe;
            float min_step;
            float max_step;

            enum wsp_ggml_linesearch linesearch;
        } lbfgs;
    };

    struct wsp_ggml_opt_context {
        struct wsp_ggml_context * ctx;
        struct wsp_ggml_opt_params params;

        int iter;
        int64_t nx; // number of parameter elements

        bool just_initialized;

        float loss_before;
        float loss_after;

        struct {
            struct wsp_ggml_tensor * m;  // first moment
            struct wsp_ggml_tensor * v;  // second moment
            struct wsp_ggml_tensor * pf; // past function values
            float fx_best;
            float fx_prev;
            int n_no_improvement;
        } adam;

        struct {
            struct wsp_ggml_tensor * x;    // current parameters
            struct wsp_ggml_tensor * xp;   // previous parameters
            struct wsp_ggml_tensor * g;    // current gradient
            struct wsp_ggml_tensor * gp;   // previous gradient
            struct wsp_ggml_tensor * d;    // search direction
            struct wsp_ggml_tensor * pf;   // past function values
            struct wsp_ggml_tensor * lmal; // the L-BFGS memory alpha
            struct wsp_ggml_tensor * lmys; // the L-BFGS memory ys
            struct wsp_ggml_tensor * lms;  // the L-BFGS memory s
            struct wsp_ggml_tensor * lmy;  // the L-BFGS memory y
            float fx_best;
            float step;
            int j;
            int k;
            int end;
            int n_no_improvement;
        } lbfgs;
    };

    WSP_GGML_API struct wsp_ggml_opt_params wsp_ggml_opt_default_params(enum wsp_ggml_opt_type type);

    // optimize the function defined by the tensor f
    WSP_GGML_API enum wsp_ggml_opt_result wsp_ggml_opt(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_opt_params params,
            struct wsp_ggml_tensor * f);

    // initialize optimizer context
    WSP_GGML_API void wsp_ggml_opt_init(
            struct wsp_ggml_context     * ctx,
            struct wsp_ggml_opt_context * opt,
            struct wsp_ggml_opt_params    params,
            int64_t                   nx);

    // continue optimizing the function defined by the tensor f
    WSP_GGML_API enum wsp_ggml_opt_result wsp_ggml_opt_resume(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_opt_context * opt,
            struct wsp_ggml_tensor * f);

    // continue optimizing the function defined by the tensor f
    WSP_GGML_API enum wsp_ggml_opt_result wsp_ggml_opt_resume_g(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_opt_context * opt,
            struct wsp_ggml_tensor * f,
            struct wsp_ggml_cgraph * gf,
            struct wsp_ggml_cgraph * gb,
            wsp_ggml_opt_callback callback,
            void * callback_data);

    //
    // quantization
    //

    WSP_GGML_API size_t wsp_ggml_quantize_q4_0(const float * src, void * dst, int n, int k, int64_t * hist);
    WSP_GGML_API size_t wsp_ggml_quantize_q4_1(const float * src, void * dst, int n, int k, int64_t * hist);
    WSP_GGML_API size_t wsp_ggml_quantize_q5_0(const float * src, void * dst, int n, int k, int64_t * hist);
    WSP_GGML_API size_t wsp_ggml_quantize_q5_1(const float * src, void * dst, int n, int k, int64_t * hist);
    WSP_GGML_API size_t wsp_ggml_quantize_q8_0(const float * src, void * dst, int n, int k, int64_t * hist);

    WSP_GGML_API size_t wsp_ggml_quantize_chunk(enum wsp_ggml_type type, const float * src, void * dst, int start, int n, int64_t * hist);

    //
    // gguf
    //

    enum gguf_type {
        GGUF_TYPE_UINT8   = 0,
        GGUF_TYPE_INT8    = 1,
        GGUF_TYPE_UINT16  = 2,
        GGUF_TYPE_INT16   = 3,
        GGUF_TYPE_UINT32  = 4,
        GGUF_TYPE_INT32   = 5,
        GGUF_TYPE_FLOAT32 = 6,
        GGUF_TYPE_BOOL    = 7,
        GGUF_TYPE_STRING  = 8,
        GGUF_TYPE_ARRAY   = 9,
        GGUF_TYPE_UINT64  = 10,
        GGUF_TYPE_INT64   = 11,
        GGUF_TYPE_FLOAT64 = 12,
        GGUF_TYPE_COUNT,       // marks the end of the enum
    };

    struct gguf_context;

    struct gguf_init_params {
        bool no_alloc;

        // if not NULL, create a wsp_ggml_context and allocate the tensor data in it
        struct wsp_ggml_context ** ctx;
    };

    WSP_GGML_API struct gguf_context * gguf_init_empty(void);
    WSP_GGML_API struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params);
    //WSP_GGML_API struct gguf_context * gguf_init_from_buffer(..);

    WSP_GGML_API void gguf_free(struct gguf_context * ctx);

    WSP_GGML_API const char * gguf_type_name(enum gguf_type type);

    WSP_GGML_API int    gguf_get_version    (const struct gguf_context * ctx);
    WSP_GGML_API size_t gguf_get_alignment  (const struct gguf_context * ctx);
    WSP_GGML_API size_t gguf_get_data_offset(const struct gguf_context * ctx);
    WSP_GGML_API void * gguf_get_data       (const struct gguf_context * ctx);

    WSP_GGML_API int          gguf_get_n_kv(const struct gguf_context * ctx);
    WSP_GGML_API int          gguf_find_key(const struct gguf_context * ctx, const char * key);
    WSP_GGML_API const char * gguf_get_key (const struct gguf_context * ctx, int i);

    WSP_GGML_API enum gguf_type gguf_get_kv_type (const struct gguf_context * ctx, int i);
    WSP_GGML_API enum gguf_type gguf_get_arr_type(const struct gguf_context * ctx, int i);

    // results are undefined if the wrong type is used for the key
    WSP_GGML_API uint8_t      gguf_get_val_u8  (const struct gguf_context * ctx, int i);
    WSP_GGML_API int8_t       gguf_get_val_i8  (const struct gguf_context * ctx, int i);
    WSP_GGML_API uint16_t     gguf_get_val_u16 (const struct gguf_context * ctx, int i);
    WSP_GGML_API int16_t      gguf_get_val_i16 (const struct gguf_context * ctx, int i);
    WSP_GGML_API uint32_t     gguf_get_val_u32 (const struct gguf_context * ctx, int i);
    WSP_GGML_API int32_t      gguf_get_val_i32 (const struct gguf_context * ctx, int i);
    WSP_GGML_API float        gguf_get_val_f32 (const struct gguf_context * ctx, int i);
    WSP_GGML_API uint64_t     gguf_get_val_u64 (const struct gguf_context * ctx, int i);
    WSP_GGML_API int64_t      gguf_get_val_i64 (const struct gguf_context * ctx, int i);
    WSP_GGML_API double       gguf_get_val_f64 (const struct gguf_context * ctx, int i);
    WSP_GGML_API bool         gguf_get_val_bool(const struct gguf_context * ctx, int i);
    WSP_GGML_API const char * gguf_get_val_str (const struct gguf_context * ctx, int i);
    WSP_GGML_API int          gguf_get_arr_n   (const struct gguf_context * ctx, int i);
    WSP_GGML_API const void * gguf_get_arr_data(const struct gguf_context * ctx, int i);
    WSP_GGML_API const char * gguf_get_arr_str (const struct gguf_context * ctx, int key_id, int i);

    WSP_GGML_API int    gguf_get_n_tensors    (const struct gguf_context * ctx);
    WSP_GGML_API int    gguf_find_tensor      (const struct gguf_context * ctx, const char * name);
    WSP_GGML_API size_t gguf_get_tensor_offset(const struct gguf_context * ctx, int i);
    WSP_GGML_API char * gguf_get_tensor_name  (const struct gguf_context * ctx, int i);

    // overrides existing values or adds a new one
    WSP_GGML_API void gguf_set_val_u8  (struct gguf_context * ctx, const char * key, uint8_t  val);
    WSP_GGML_API void gguf_set_val_i8  (struct gguf_context * ctx, const char * key, int8_t   val);
    WSP_GGML_API void gguf_set_val_u16 (struct gguf_context * ctx, const char * key, uint16_t val);
    WSP_GGML_API void gguf_set_val_i16 (struct gguf_context * ctx, const char * key, int16_t  val);
    WSP_GGML_API void gguf_set_val_u32 (struct gguf_context * ctx, const char * key, uint32_t val);
    WSP_GGML_API void gguf_set_val_i32 (struct gguf_context * ctx, const char * key, int32_t  val);
    WSP_GGML_API void gguf_set_val_f32 (struct gguf_context * ctx, const char * key, float    val);
    WSP_GGML_API void gguf_set_val_u64 (struct gguf_context * ctx, const char * key, uint64_t val);
    WSP_GGML_API void gguf_set_val_i64 (struct gguf_context * ctx, const char * key, int64_t  val);
    WSP_GGML_API void gguf_set_val_f64 (struct gguf_context * ctx, const char * key, double   val);
    WSP_GGML_API void gguf_set_val_bool(struct gguf_context * ctx, const char * key, bool     val);
    WSP_GGML_API void gguf_set_val_str (struct gguf_context * ctx, const char * key, const char * val);
    WSP_GGML_API void gguf_set_arr_data(struct gguf_context * ctx, const char * key, enum gguf_type type, const void * data, int n);
    WSP_GGML_API void gguf_set_arr_str (struct gguf_context * ctx, const char * key, const char ** data, int n);

    // set or add KV pairs from another context
    WSP_GGML_API void gguf_set_kv(struct gguf_context * ctx, struct gguf_context * src);

    // manage tensor info
    WSP_GGML_API void gguf_add_tensor(struct gguf_context * ctx, const struct wsp_ggml_tensor * tensor);
    WSP_GGML_API void gguf_set_tensor_type(struct gguf_context * ctx, const char * name, enum wsp_ggml_type type);
    WSP_GGML_API void gguf_set_tensor_data(struct gguf_context * ctx, const char * name, const void * data, size_t size);

    // writing gguf files can be done in 2 ways:
    //
    // - write the entire gguf_context to a binary file in a single pass:
    //
    //   gguf_write_to_file(ctx, fname);
    //
    // - first prepare a file with a placeholder for the meta data, write the tensor data, then write the meta data:
    //
    //   FILE * f = fopen(fname, "wb");
    //   fseek(f, gguf_get_meta_size(ctx), SEEK_SET);
    //   fwrite(f, ...);
    //   void * data = gguf_meta_get_meta_data(ctx);
    //   fseek(f, 0, SEEK_SET);
    //   fwrite(f, data, gguf_get_meta_size(ctx));
    //   free(data);
    //   fclose(f);
    //

    // write the entire context to a binary file
    WSP_GGML_API void gguf_write_to_file(const struct gguf_context * ctx, const char * fname, bool only_meta);

    // get the size in bytes of the meta data (header, kv pairs, tensor info) including padding
    WSP_GGML_API size_t gguf_get_meta_size(const struct gguf_context * ctx);
    WSP_GGML_API void   gguf_get_meta_data(const struct gguf_context * ctx, void * data);

    //
    // system info
    //

    WSP_GGML_API int wsp_ggml_cpu_has_avx        (void);
    WSP_GGML_API int wsp_ggml_cpu_has_avx2       (void);
    WSP_GGML_API int wsp_ggml_cpu_has_avx512     (void);
    WSP_GGML_API int wsp_ggml_cpu_has_avx512_vbmi(void);
    WSP_GGML_API int wsp_ggml_cpu_has_avx512_vnni(void);
    WSP_GGML_API int wsp_ggml_cpu_has_fma        (void);
    WSP_GGML_API int wsp_ggml_cpu_has_neon       (void);
    WSP_GGML_API int wsp_ggml_cpu_has_arm_fma    (void);
    WSP_GGML_API int wsp_ggml_cpu_has_metal      (void);
    WSP_GGML_API int wsp_ggml_cpu_has_f16c       (void);
    WSP_GGML_API int wsp_ggml_cpu_has_fp16_va    (void);
    WSP_GGML_API int wsp_ggml_cpu_has_wasm_simd  (void);
    WSP_GGML_API int wsp_ggml_cpu_has_blas       (void);
    WSP_GGML_API int wsp_ggml_cpu_has_cublas     (void);
    WSP_GGML_API int wsp_ggml_cpu_has_clblast    (void);
    WSP_GGML_API int wsp_ggml_cpu_has_gpublas    (void);
    WSP_GGML_API int wsp_ggml_cpu_has_sse3       (void);
    WSP_GGML_API int wsp_ggml_cpu_has_ssse3      (void);
    WSP_GGML_API int wsp_ggml_cpu_has_vsx        (void);

    //
    // Internal types and functions exposed for tests and benchmarks
    //

#ifdef  __cplusplus
// restrict not standard in C++
#define WSP_GGML_RESTRICT
#else
#define WSP_GGML_RESTRICT restrict
#endif
    typedef void (*wsp_ggml_to_float_t)  (const void  * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int k);
    typedef void (*wsp_ggml_from_float_t)(const float * WSP_GGML_RESTRICT x, void  * WSP_GGML_RESTRICT y, int k);
    typedef void (*wsp_ggml_vec_dot_t)   (const int n, float * WSP_GGML_RESTRICT s, const void * WSP_GGML_RESTRICT x, const void * WSP_GGML_RESTRICT y);

    typedef struct {
        const char      * type_name;
        int               blck_size;
        size_t            type_size;
        bool              is_quantized;
        wsp_ggml_to_float_t   to_float;
        wsp_ggml_from_float_t from_float;
        wsp_ggml_from_float_t from_float_reference;
        wsp_ggml_vec_dot_t    vec_dot;
        enum wsp_ggml_type    vec_dot_type;
    } wsp_ggml_type_traits_t;

    wsp_ggml_type_traits_t wsp_ggml_internal_get_type_traits(enum wsp_ggml_type type);

#ifdef  __cplusplus
}
#endif
