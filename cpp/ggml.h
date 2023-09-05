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
//       wsp_ggml_graph_compute(ctx0, &gf);
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
//       struct wsp_ggml_tensor * a = wsp_ggml_new_tensor_2d(ctx, WSP_GGML_TYPE_F32, 2, 3);
//
//       // a[1, 2] = 1.0f;
//       *(float *) ((char *) a->data + 2*a->nb[1] + 1*a->nb[0]) = 1.0f;
//
//       // a[2, 0] = 2.0f;
//       *(float *) ((char *) a->data + 0*a->nb[1] + 2*a->nb[0]) = 2.0f;
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
#define WSP_GGML_MAX_OPT           4
#define WSP_GGML_MAX_NAME          48
#define WSP_GGML_DEFAULT_N_THREADS 4

#define WSP_GGML_UNUSED(x) (void)(x)

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

#ifdef __ARM_NEON
    // we use the built-in 16-bit float type
    typedef __fp16 wsp_ggml_fp16_t;
#else
    typedef uint16_t wsp_ggml_fp16_t;
#endif

    // convert FP16 <-> FP32
    WSP_GGML_API float       wsp_ggml_fp16_to_fp32(wsp_ggml_fp16_t x);
    WSP_GGML_API wsp_ggml_fp16_t wsp_ggml_fp32_to_fp16(float x);

    WSP_GGML_API void wsp_ggml_fp16_to_fp32_row(const wsp_ggml_fp16_t * x, float * y, size_t n);
    WSP_GGML_API void wsp_ggml_fp32_to_fp16_row(const float * x, wsp_ggml_fp16_t * y, size_t n);

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
        WSP_GGML_OP_ABS,
        WSP_GGML_OP_SGN,
        WSP_GGML_OP_NEG,
        WSP_GGML_OP_STEP,
        WSP_GGML_OP_TANH,
        WSP_GGML_OP_ELU,
        WSP_GGML_OP_RELU,
        WSP_GGML_OP_GELU,
        WSP_GGML_OP_GELU_QUICK,
        WSP_GGML_OP_SILU,
        WSP_GGML_OP_SILU_BACK,
        WSP_GGML_OP_NORM, // normalize
        WSP_GGML_OP_RMS_NORM,
        WSP_GGML_OP_RMS_NORM_BACK,

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

        WSP_GGML_OP_FLASH_ATTN,
        WSP_GGML_OP_FLASH_FF,
        WSP_GGML_OP_FLASH_ATTN_BACK,
        WSP_GGML_OP_WIN_PART,
        WSP_GGML_OP_WIN_UNPART,

        WSP_GGML_OP_MAP_UNARY,
        WSP_GGML_OP_MAP_BINARY,

        WSP_GGML_OP_MAP_CUSTOM1,
        WSP_GGML_OP_MAP_CUSTOM2,
        WSP_GGML_OP_MAP_CUSTOM3,

        WSP_GGML_OP_CROSS_ENTROPY_LOSS,
        WSP_GGML_OP_CROSS_ENTROPY_LOSS_BACK,

        WSP_GGML_OP_COUNT,
    };


    // ggml object
    struct wsp_ggml_object {
        size_t offs;
        size_t size;

        struct wsp_ggml_object * next;

        char padding[8];
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

        bool is_param;

        struct wsp_ggml_tensor * grad;
        struct wsp_ggml_tensor * src0;
        struct wsp_ggml_tensor * src1;
        struct wsp_ggml_tensor * opt[WSP_GGML_MAX_OPT];

        // thread scheduling
        int n_tasks;

        // performance
        int     perf_runs;
        int64_t perf_cycles;
        int64_t perf_time_us;

        void * data;

        char name[WSP_GGML_MAX_NAME];

        void * extra; // extra things e.g. for ggml-cuda.cu

        char padding[4];
    };

    static const size_t WSP_GGML_TENSOR_SIZE = sizeof(struct wsp_ggml_tensor);

    // computation graph
    struct wsp_ggml_cgraph {
        int n_nodes;
        int n_leafs;
        int n_threads;

        size_t work_size;
        struct wsp_ggml_tensor * work;

        struct wsp_ggml_tensor * nodes[WSP_GGML_MAX_NODES];
        struct wsp_ggml_tensor * grads[WSP_GGML_MAX_NODES];
        struct wsp_ggml_tensor * leafs[WSP_GGML_MAX_NODES];

        // performance
        int     perf_runs;
        int64_t perf_cycles;
        int64_t perf_time_us;
    };

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
    WSP_GGML_API size_t  wsp_ggml_nbytes_split(const struct wsp_ggml_tensor * tensor, int nrows_split);

    WSP_GGML_API int     wsp_ggml_blck_size (enum wsp_ggml_type type);
    WSP_GGML_API size_t  wsp_ggml_type_size (enum wsp_ggml_type type); // size in bytes for all elements in a block
    WSP_GGML_API float   wsp_ggml_type_sizef(enum wsp_ggml_type type); // wsp_ggml_type_size()/wsp_ggml_blck_size() as float

    WSP_GGML_API const char * wsp_ggml_type_name(enum wsp_ggml_type type);
    WSP_GGML_API const char * wsp_ggml_op_name  (enum wsp_ggml_op   op);

    WSP_GGML_API size_t  wsp_ggml_element_size(const struct wsp_ggml_tensor * tensor);

    WSP_GGML_API bool    wsp_ggml_is_quantized(enum wsp_ggml_type type);

    // TODO: temporary until model loading of ggml examples is refactored
    WSP_GGML_API enum wsp_ggml_type wsp_ggml_ftype_to_wsp_ggml_type(enum wsp_ggml_ftype ftype);

    WSP_GGML_API bool wsp_ggml_is_transposed(const struct wsp_ggml_tensor * tensor);
    WSP_GGML_API bool wsp_ggml_is_contiguous(const struct wsp_ggml_tensor * tensor);
    WSP_GGML_API bool wsp_ggml_is_permuted  (const struct wsp_ggml_tensor * tensor);

    // use this to compute the memory overhead of a tensor
    WSP_GGML_API size_t wsp_ggml_tensor_overhead(void);

    // main

    WSP_GGML_API struct wsp_ggml_context * wsp_ggml_init(struct wsp_ggml_init_params params);
    WSP_GGML_API void                  wsp_ggml_free(struct wsp_ggml_context * ctx);

    WSP_GGML_API size_t  wsp_ggml_used_mem(const struct wsp_ggml_context * ctx);

    WSP_GGML_API size_t  wsp_ggml_set_scratch (struct wsp_ggml_context * ctx, struct wsp_ggml_scratch scratch);
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
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_view_tensor(struct wsp_ggml_context * ctx, const struct wsp_ggml_tensor * src);

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

    WSP_GGML_API const char *         wsp_ggml_get_name(const struct wsp_ggml_tensor * tensor);
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_set_name(struct wsp_ggml_tensor * tensor, const char * name);
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_format_name(struct wsp_ggml_tensor * tensor, const char * fmt, ...);

    //
    // operations on tensors with backpropagation
    //

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_dup(
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
    // TODO: eps is hardcoded to 1e-5 for now
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_norm(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_norm_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_rms_norm(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_rms_norm_inplace(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a);

    // a - x
    // b - dy
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_rms_norm_back(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b);

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

    // make contiguous
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_cont(
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

    // rotary position embedding backward, i.e compute dx from dy
    // a - dy
    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_rope_back(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            int                   n_past,
            int                   n_dims,
            int                   mode);

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

    // conv_1d with padding = half
    // alias for wsp_ggml_conv_1d(a, b, s, a->ne[0]/2, d)
    WSP_GGML_API struct wsp_ggml_tensor* wsp_ggml_conv_1d_ph(
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_tensor  * a,
            struct wsp_ggml_tensor  * b,
            int                   s,
            int                   d);

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

    // custom operators

    typedef void (*wsp_ggml_unary_op_f32_t) (const int, float *, const float *);
    typedef void (*wsp_ggml_binary_op_f32_t)(const int, float *, const float *, const float *);

    typedef void (*wsp_ggml_custom1_op_f32_t)(struct wsp_ggml_tensor *, const struct wsp_ggml_tensor *);
    typedef void (*wsp_ggml_custom2_op_f32_t)(struct wsp_ggml_tensor *, const struct wsp_ggml_tensor *, const struct wsp_ggml_tensor *);
    typedef void (*wsp_ggml_custom3_op_f32_t)(struct wsp_ggml_tensor *, const struct wsp_ggml_tensor *, const struct wsp_ggml_tensor *, const struct wsp_ggml_tensor *);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_unary_f32(
            struct wsp_ggml_context        * ctx,
            struct wsp_ggml_tensor         * a,
                   wsp_ggml_unary_op_f32_t   fun);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_unary_inplace_f32(
            struct wsp_ggml_context        * ctx,
            struct wsp_ggml_tensor         * a,
                   wsp_ggml_unary_op_f32_t   fun);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_binary_f32(
            struct wsp_ggml_context         * ctx,
            struct wsp_ggml_tensor          * a,
            struct wsp_ggml_tensor          * b,
                   wsp_ggml_binary_op_f32_t   fun);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_binary_inplace_f32(
            struct wsp_ggml_context         * ctx,
            struct wsp_ggml_tensor          * a,
            struct wsp_ggml_tensor          * b,
                   wsp_ggml_binary_op_f32_t   fun);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_custom1_f32(
            struct wsp_ggml_context          * ctx,
            struct wsp_ggml_tensor           * a,
                   wsp_ggml_custom1_op_f32_t   fun);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_custom1_inplace_f32(
            struct wsp_ggml_context          * ctx,
            struct wsp_ggml_tensor           * a,
                   wsp_ggml_custom1_op_f32_t   fun);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_custom2_f32(
            struct wsp_ggml_context          * ctx,
            struct wsp_ggml_tensor           * a,
            struct wsp_ggml_tensor           * b,
                   wsp_ggml_custom2_op_f32_t   fun);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_custom2_inplace_f32(
            struct wsp_ggml_context          * ctx,
            struct wsp_ggml_tensor           * a,
            struct wsp_ggml_tensor           * b,
                   wsp_ggml_custom2_op_f32_t   fun);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_custom3_f32(
            struct wsp_ggml_context          * ctx,
            struct wsp_ggml_tensor           * a,
            struct wsp_ggml_tensor           * b,
            struct wsp_ggml_tensor           * c,
                   wsp_ggml_custom3_op_f32_t   fun);

    WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_map_custom3_inplace_f32(
            struct wsp_ggml_context          * ctx,
            struct wsp_ggml_tensor           * a,
            struct wsp_ggml_tensor           * b,
            struct wsp_ggml_tensor           * c,
                   wsp_ggml_custom3_op_f32_t   fun);

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
            struct wsp_ggml_tensor * tensor);

    WSP_GGML_API void wsp_ggml_build_forward_expand(struct wsp_ggml_cgraph * cgraph, struct wsp_ggml_tensor * tensor);

    WSP_GGML_API struct wsp_ggml_cgraph wsp_ggml_build_forward (struct wsp_ggml_tensor * tensor);
    WSP_GGML_API struct wsp_ggml_cgraph wsp_ggml_build_backward(struct wsp_ggml_context * ctx, struct wsp_ggml_cgraph * gf, bool keep);

    WSP_GGML_API void wsp_ggml_graph_compute(struct wsp_ggml_context * ctx, struct wsp_ggml_cgraph * cgraph);
    WSP_GGML_API void wsp_ggml_graph_reset  (struct wsp_ggml_cgraph * cgraph);

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
            float alpha; // learning rate
            float beta1;
            float beta2;
            float eps;   // epsilon for numerical stability
            float eps_f; // epsilon for convergence test
            float eps_g; // epsilon for convergence test
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

        struct {
            struct wsp_ggml_tensor * x;  // view of the parameters
            struct wsp_ggml_tensor * g1; // gradient
            struct wsp_ggml_tensor * g2; // gradient squared
            struct wsp_ggml_tensor * m;  // first moment
            struct wsp_ggml_tensor * v;  // second moment
            struct wsp_ggml_tensor * mh; // first moment hat
            struct wsp_ggml_tensor * vh; // second moment hat
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
            struct wsp_ggml_context * ctx,
            struct wsp_ggml_opt_context * opt,
            struct wsp_ggml_opt_params params,
            int64_t nx);

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
            struct wsp_ggml_cgraph * gb);

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
    typedef void (*dequantize_row_q_t)(const void * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int k);
    typedef void (*quantize_row_q_t)  (const float * WSP_GGML_RESTRICT x, void * WSP_GGML_RESTRICT y, int k);
    typedef void (*vec_dot_q_t)       (const int n, float * WSP_GGML_RESTRICT s, const void * WSP_GGML_RESTRICT x, const void * WSP_GGML_RESTRICT y);

    typedef struct {
        dequantize_row_q_t dequantize_row_q;
        quantize_row_q_t   quantize_row_q;
        quantize_row_q_t   quantize_row_q_reference;
        quantize_row_q_t   quantize_row_q_dot;
        vec_dot_q_t        vec_dot_q;
        enum wsp_ggml_type     vec_dot_type;
    } quantize_fns_t;

    quantize_fns_t wsp_ggml_internal_get_quantize_fn(size_t i);

#ifdef  __cplusplus
}
#endif
