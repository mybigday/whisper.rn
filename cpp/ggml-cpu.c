#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#include "ggml-aarch64.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml.h"

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <signal.h>
#if defined(__gnu_linux__)
#include <syscall.h>
#endif

#ifdef WSP_GGML_USE_OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef WSP_GGML_USE_LLAMAFILE
#endif

#ifdef WSP_GGML_USE_LLAMAFILE
#include <llamafile/sgemm.h>
#endif

#if defined(_MSC_VER)
// disable "possible loss of data" to avoid hundreds of casts
// we should just be careful :)
#pragma warning(disable: 4244 4267)

// disable POSIX deprecation warnings
// these functions are never going away, anyway
#pragma warning(disable: 4996)

// unreachable code because of multiple instances of code after WSP_GGML_ABORT
#pragma warning(disable: 4702)
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define WSP_GGML_CACHE_LINE  64

#if defined(__clang__) || defined(__GNUC__)
#define WSP_GGML_CACHE_ALIGN __attribute__((aligned(WSP_GGML_CACHE_LINE)))
#endif

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define WSP_GGML_TSAN_ENABLED 1
#endif
#else  // __has_feature
#if defined(__SANITIZE_THREAD__)
#define WSP_GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED WSP_GGML_UNUSED
#define SWAP(x, y, T) do { T SWAP = x; (x) = y; (y) = SWAP; } while (0)

#if defined(WSP_GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

// floating point type used to accumulate sums
typedef double wsp_ggml_float;

#define WSP_GGML_GELU_FP16
#define WSP_GGML_GELU_QUICK_FP16

#define WSP_GGML_SOFT_MAX_UNROLL 4
#define WSP_GGML_VEC_DOT_UNROLL  2
#define WSP_GGML_VEC_MAD_UNROLL  32

//
// global data
//

// precomputed gelu table for f16 (128 KB)
static wsp_ggml_fp16_t wsp_ggml_table_gelu_f16[1 << 16];

// precomputed quick gelu table for f16 (128 KB)
static wsp_ggml_fp16_t wsp_ggml_table_gelu_quick_f16[1 << 16];

// precomputed f32 table for f16 (256 KB) (ggml-impl.h)
float wsp_ggml_table_f32_f16[1 << 16];

#if defined(__ARM_ARCH)
struct wsp_ggml_arm_arch_features_type {
    int has_neon;
    int has_i8mm;
    int has_sve;
    int sve_cnt;
} wsp_ggml_arm_arch_features = {-1, -1, -1, 0};
#endif


#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>


#if !defined(__clang__)
#define WSP_GGML_CACHE_ALIGN __declspec(align(WSP_GGML_CACHE_LINE))

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;
typedef atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

typedef enum {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

static void atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
static void atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
static LONG atomic_load(atomic_int * ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
static LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedExchangeAdd(ptr, inc);
}
static atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    return InterlockedExchange(ptr, 1);
}
static void atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
static void atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#else // clang
#include <stdatomic.h>
#endif

typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;
static int pthread_create(pthread_t * out, void * unused, thread_ret_t(*func)(void *), void * arg) {
    (void) unused;
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, NULL);
    if (handle == NULL)
    {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static int pthread_join(pthread_t thread, void * unused) {
    (void) unused;
    int ret = (int) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return ret;
}

static int sched_yield (void) {
    Sleep (0);
    return 0;
}
#else

#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#if defined(__FreeBSD__)
#include <pthread_np.h>
#endif

typedef void * thread_ret_t;

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

typedef pthread_t wsp_ggml_thread_t;

#ifdef WSP_GGML_USE_CPU_HBM
#include <hbwmalloc.h>
#endif

#if defined(__APPLE__)
#include <unistd.h>
#include <mach/mach.h>
#include <TargetConditionals.h>
#endif

//
// cache line
//

#if defined(__cpp_lib_hardware_interference_size)
#define CACHE_LINE_SIZE hardware_destructive_interference_size
#else
#if defined(__POWER9_VECTOR__)
#define CACHE_LINE_SIZE 128
#else
#define CACHE_LINE_SIZE 64
#endif
#endif

static const size_t CACHE_LINE_SIZE_F32 = CACHE_LINE_SIZE/sizeof(float);


static void wsp_ggml_vec_dot_f32(int n, float * restrict s, size_t bs, const float * restrict x, size_t bx, const float * restrict y, size_t by, int nrc);
static void wsp_ggml_vec_dot_f16(int n, float * restrict s, size_t bs, wsp_ggml_fp16_t * restrict x, size_t bx, wsp_ggml_fp16_t * restrict y, size_t by, int nrc);
static void wsp_ggml_vec_dot_bf16(int n, float * restrict s, size_t bs, wsp_ggml_bf16_t * restrict x, size_t bx, wsp_ggml_bf16_t * restrict y, size_t by, int nrc);

static const struct wsp_ggml_type_traits_cpu type_traits_cpu[WSP_GGML_TYPE_COUNT] = {
    [WSP_GGML_TYPE_F32] = {
        .vec_dot                  = (wsp_ggml_vec_dot_t) wsp_ggml_vec_dot_f32,
        .vec_dot_type             = WSP_GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_F16] = {
        .vec_dot                  = (wsp_ggml_vec_dot_t) wsp_ggml_vec_dot_f16,
        .vec_dot_type             = WSP_GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_Q4_0] = {
        .vec_dot                  = wsp_ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [WSP_GGML_TYPE_Q4_1] = {
        .vec_dot                  = wsp_ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_1,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [4] = { // WSP_GGML_TYPE_Q4_2
        .vec_dot                  = NULL,
        .vec_dot_type             = WSP_GGML_TYPE_COUNT,
        .nrows                    = 1,
    },
    [5] = { // WSP_GGML_TYPE_Q4_3
        .vec_dot                  = NULL,
        .vec_dot_type             = WSP_GGML_TYPE_COUNT,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_Q5_0] = {
        .vec_dot                  = wsp_ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_Q5_1] = {
        .vec_dot                  = wsp_ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_Q8_0] = {
        .from_float_to_mat        = wsp_quantize_mat_q8_0,
        .vec_dot                  = wsp_ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [WSP_GGML_TYPE_Q8_1] = {
        .vec_dot_type             = WSP_GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_Q2_K] = {
        .vec_dot                  = wsp_ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_Q3_K] = {
        .vec_dot                  = wsp_ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_Q4_K] = {
        .vec_dot                  = wsp_ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_Q5_K] = {
        .vec_dot                  = wsp_ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_Q6_K] = {
        .vec_dot                  = wsp_ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_IQ2_XXS] = {
        .vec_dot                  = wsp_ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_IQ2_XS] = {
        .vec_dot                  = wsp_ggml_vec_dot_iq2_xs_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_IQ3_XXS] = {
        .vec_dot                  = wsp_ggml_vec_dot_iq3_xxs_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_IQ3_S] = {
        .vec_dot                  = wsp_ggml_vec_dot_iq3_s_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_IQ2_S] = {
        .vec_dot                  = wsp_ggml_vec_dot_iq2_s_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_IQ1_S] = {
        .vec_dot                  = wsp_ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_IQ1_M] = {
        .vec_dot                  = wsp_ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_IQ4_NL] = {
        .vec_dot                  = wsp_ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_IQ4_XS] = {
        .vec_dot                  = wsp_ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_BF16] = {
        .vec_dot                  = (wsp_ggml_vec_dot_t) wsp_ggml_vec_dot_bf16,
        .vec_dot_type             = WSP_GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_Q4_0_4_4] = {
        .vec_dot                  = NULL,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_0,
        .nrows                    = 1,
        .ncols                    = 4,
        .gemv                     = wsp_ggml_gemv_q4_0_4x4_q8_0,
        .gemm                     = wsp_ggml_gemm_q4_0_4x4_q8_0,
    },
    [WSP_GGML_TYPE_Q4_0_4_8] = {
        .vec_dot                  = NULL,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_0,
        .nrows                    = 1,
        .ncols                    = 4,
        .gemv                     = wsp_ggml_gemv_q4_0_4x8_q8_0,
        .gemm                     = wsp_ggml_gemm_q4_0_4x8_q8_0,
    },
    [WSP_GGML_TYPE_Q4_0_8_8] = {
        .vec_dot                  = NULL,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_0,
        .nrows                    = 1,
        .ncols                    = 8,
        .gemv                     = wsp_ggml_gemv_q4_0_8x8_q8_0,
        .gemm                     = wsp_ggml_gemm_q4_0_8x8_q8_0,
    },
    [WSP_GGML_TYPE_TQ1_0] = {
        .vec_dot                  = wsp_ggml_vec_dot_tq1_0_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_TQ2_0] = {
        .vec_dot                  = wsp_ggml_vec_dot_tq2_0_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
};

const struct wsp_ggml_type_traits_cpu * wsp_ggml_get_type_traits_cpu(enum wsp_ggml_type type) {
    return &type_traits_cpu[type];
}

//
// simd mappings
//

// we define a common set of C macros which map to specific intrinsics based on the current architecture
// we then implement the fundamental computation operations below using only these macros
// adding support for new architectures requires to define the corresponding SIMD macros
//
// WSP_GGML_F32_STEP / WSP_GGML_F16_STEP
//   number of elements to process in a single step
//
// WSP_GGML_F32_EPR / WSP_GGML_F16_EPR
//   number of elements to fit in a single register
//

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)

#define WSP_GGML_SIMD

// F32 NEON

#define WSP_GGML_F32_STEP 16
#define WSP_GGML_F32_EPR  4

#define WSP_GGML_F32x4              float32x4_t
#define WSP_GGML_F32x4_ZERO         vdupq_n_f32(0.0f)
#define WSP_GGML_F32x4_SET1(x)      vdupq_n_f32(x)
#define WSP_GGML_F32x4_LOAD         vld1q_f32
#define WSP_GGML_F32x4_STORE        vst1q_f32
#define WSP_GGML_F32x4_FMA(a, b, c) vfmaq_f32(a, b, c)
#define WSP_GGML_F32x4_ADD          vaddq_f32
#define WSP_GGML_F32x4_MUL          vmulq_f32
#define WSP_GGML_F32x4_REDUCE_ONE(x) vaddvq_f32(x)
#define WSP_GGML_F32x4_REDUCE(res, x)                  \
{                                                  \
    int offset = WSP_GGML_F32_ARR >> 1;                \
    for (int i = 0; i < offset; ++i) {             \
        (x)[i] = vaddq_f32((x)[i], (x)[offset+i]); \
    }                                              \
    offset >>= 1;                                  \
    for (int i = 0; i < offset; ++i) {             \
        (x)[i] = vaddq_f32((x)[i], (x)[offset+i]); \
    }                                              \
    offset >>= 1;                                  \
    for (int i = 0; i < offset; ++i) {             \
        (x)[i] = vaddq_f32((x)[i], (x)[offset+i]); \
    }                                              \
    (res) = WSP_GGML_F32x4_REDUCE_ONE((x)[0]);         \
}

#define WSP_GGML_F32_VEC        WSP_GGML_F32x4
#define WSP_GGML_F32_VEC_ZERO   WSP_GGML_F32x4_ZERO
#define WSP_GGML_F32_VEC_SET1   WSP_GGML_F32x4_SET1
#define WSP_GGML_F32_VEC_LOAD   WSP_GGML_F32x4_LOAD
#define WSP_GGML_F32_VEC_STORE  WSP_GGML_F32x4_STORE
#define WSP_GGML_F32_VEC_FMA    WSP_GGML_F32x4_FMA
#define WSP_GGML_F32_VEC_ADD    WSP_GGML_F32x4_ADD
#define WSP_GGML_F32_VEC_MUL    WSP_GGML_F32x4_MUL
#define WSP_GGML_F32_VEC_REDUCE WSP_GGML_F32x4_REDUCE

// F16 NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #define WSP_GGML_F16_STEP 32
    #define WSP_GGML_F16_EPR  8

    #define WSP_GGML_F16x8              float16x8_t
    #define WSP_GGML_F16x8_ZERO         vdupq_n_f16(0.0f)
    #define WSP_GGML_F16x8_SET1(x)      vdupq_n_f16(x)
    #define WSP_GGML_F16x8_LOAD(x)      vld1q_f16((const wsp_ggml_fp16_internal_t *)(x))
    #define WSP_GGML_F16x8_STORE        vst1q_f16
    #define WSP_GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
    #define WSP_GGML_F16x8_ADD          vaddq_f16
    #define WSP_GGML_F16x8_MUL          vmulq_f16
    #define WSP_GGML_F16x8_REDUCE(res, x)                               \
    do {                                                            \
        int offset = WSP_GGML_F16_ARR >> 1;                             \
        for (int i = 0; i < offset; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[offset+i]);              \
        }                                                           \
        offset >>= 1;                                               \
        for (int i = 0; i < offset; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[offset+i]);              \
        }                                                           \
        offset >>= 1;                                               \
        for (int i = 0; i < offset; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[offset+i]);              \
        }                                                           \
        const float32x4_t t0 = vcvt_f32_f16(vget_low_f16 ((x)[0])); \
        const float32x4_t t1 = vcvt_f32_f16(vget_high_f16((x)[0])); \
        (res) = (wsp_ggml_float) vaddvq_f32(vaddq_f32(t0, t1));         \
    } while (0)

    #define WSP_GGML_F16_VEC                WSP_GGML_F16x8
    #define WSP_GGML_F16_VEC_ZERO           WSP_GGML_F16x8_ZERO
    #define WSP_GGML_F16_VEC_SET1           WSP_GGML_F16x8_SET1
    #define WSP_GGML_F16_VEC_LOAD(p, i)     WSP_GGML_F16x8_LOAD(p)
    #define WSP_GGML_F16_VEC_STORE(p, r, i) WSP_GGML_F16x8_STORE((wsp_ggml_fp16_internal_t *)(p), (r)[i])
    #define WSP_GGML_F16_VEC_FMA            WSP_GGML_F16x8_FMA
    #define WSP_GGML_F16_VEC_ADD            WSP_GGML_F16x8_ADD
    #define WSP_GGML_F16_VEC_MUL            WSP_GGML_F16x8_MUL
    #define WSP_GGML_F16_VEC_REDUCE         WSP_GGML_F16x8_REDUCE
#else
    // if FP16 vector arithmetic is not supported, we use FP32 instead
    // and take advantage of the vcvt_ functions to convert to/from FP16

    #define WSP_GGML_F16_STEP 16
    #define WSP_GGML_F16_EPR  4

    #define WSP_GGML_F32Cx4              float32x4_t
    #define WSP_GGML_F32Cx4_ZERO         vdupq_n_f32(0.0f)
    #define WSP_GGML_F32Cx4_SET1(x)      vdupq_n_f32(x)
    #define WSP_GGML_F32Cx4_LOAD(x)      vcvt_f32_f16(vld1_f16((const wsp_ggml_fp16_internal_t *)(x)))
    #define WSP_GGML_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
    #define WSP_GGML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #define WSP_GGML_F32Cx4_ADD          vaddq_f32
    #define WSP_GGML_F32Cx4_MUL          vmulq_f32
    #define WSP_GGML_F32Cx4_REDUCE       WSP_GGML_F32x4_REDUCE

    #define WSP_GGML_F16_VEC                WSP_GGML_F32Cx4
    #define WSP_GGML_F16_VEC_ZERO           WSP_GGML_F32Cx4_ZERO
    #define WSP_GGML_F16_VEC_SET1           WSP_GGML_F32Cx4_SET1
    #define WSP_GGML_F16_VEC_LOAD(p, i)     WSP_GGML_F32Cx4_LOAD(p)
    #define WSP_GGML_F16_VEC_STORE(p, r, i) WSP_GGML_F32Cx4_STORE((wsp_ggml_fp16_internal_t *)(p), r[i])
    #define WSP_GGML_F16_VEC_FMA            WSP_GGML_F32Cx4_FMA
    #define WSP_GGML_F16_VEC_ADD            WSP_GGML_F32Cx4_ADD
    #define WSP_GGML_F16_VEC_MUL            WSP_GGML_F32Cx4_MUL
    #define WSP_GGML_F16_VEC_REDUCE         WSP_GGML_F32Cx4_REDUCE
#endif

#elif defined(__AVX512F__)

#define WSP_GGML_SIMD

// F32 AVX512

#define WSP_GGML_F32_STEP 64
#define WSP_GGML_F32_EPR  16

#define WSP_GGML_F32x16         __m512
#define WSP_GGML_F32x16_ZERO    _mm512_setzero_ps()
#define WSP_GGML_F32x16_SET1(x) _mm512_set1_ps(x)
#define WSP_GGML_F32x16_LOAD    _mm512_loadu_ps
#define WSP_GGML_F32x16_STORE   _mm512_storeu_ps
// _mm512_fmadd_ps is defined in AVX512F so no guard is required
#define WSP_GGML_F32x16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#define WSP_GGML_F32x16_ADD     _mm512_add_ps
#define WSP_GGML_F32x16_MUL     _mm512_mul_ps
#define WSP_GGML_F32x16_REDUCE(res, x)                                    \
do {                                                                  \
    int offset = WSP_GGML_F32_ARR >> 1;                                   \
    for (int i = 0; i < offset; ++i) {                                \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                      \
    }                                                                 \
    offset >>= 1;                                                     \
    for (int i = 0; i < offset; ++i) {                                \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                      \
    }                                                                 \
    offset >>= 1;                                                     \
    for (int i = 0; i < offset; ++i) {                                \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                      \
    }                                                                 \
    res = _mm512_reduce_add_ps(x[0]);                                 \
} while (0)

// TODO: is this optimal ?

#define WSP_GGML_F32_VEC        WSP_GGML_F32x16
#define WSP_GGML_F32_VEC_ZERO   WSP_GGML_F32x16_ZERO
#define WSP_GGML_F32_VEC_SET1   WSP_GGML_F32x16_SET1
#define WSP_GGML_F32_VEC_LOAD   WSP_GGML_F32x16_LOAD
#define WSP_GGML_F32_VEC_STORE  WSP_GGML_F32x16_STORE
#define WSP_GGML_F32_VEC_FMA    WSP_GGML_F32x16_FMA
#define WSP_GGML_F32_VEC_ADD    WSP_GGML_F32x16_ADD
#define WSP_GGML_F32_VEC_MUL    WSP_GGML_F32x16_MUL
#define WSP_GGML_F32_VEC_REDUCE WSP_GGML_F32x16_REDUCE

// F16 AVX512

// F16 AVX

#define WSP_GGML_F16_STEP 64
#define WSP_GGML_F16_EPR  16

// AVX512 has FP16 extension (AVX512_FP16) but I don't have it on my machine so I use FP32 instead

#define WSP_GGML_F32Cx16             __m512
#define WSP_GGML_F32Cx16_ZERO        _mm512_setzero_ps()
#define WSP_GGML_F32Cx16_SET1(x)     _mm512_set1_ps(x)

// unlike  _mm256_cvt intrinsics that require F16C, _mm512_cvt is defined in AVX512F
// so F16C guard isn't required
#define WSP_GGML_F32Cx16_LOAD(x)     _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(x)))
#define WSP_GGML_F32Cx16_STORE(x, y) _mm256_storeu_si256((__m256i *)(x), _mm512_cvtps_ph(y, 0))

#define WSP_GGML_F32Cx16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#define WSP_GGML_F32Cx16_ADD         _mm512_add_ps
#define WSP_GGML_F32Cx16_MUL         _mm512_mul_ps
#define WSP_GGML_F32Cx16_REDUCE(res, x)                               \
do {                                                              \
    int offset = WSP_GGML_F32_ARR >> 1;                               \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    res = _mm512_reduce_add_ps(x[0]);                             \
} while (0)

#define WSP_GGML_F16_VEC                WSP_GGML_F32Cx16
#define WSP_GGML_F16_VEC_ZERO           WSP_GGML_F32Cx16_ZERO
#define WSP_GGML_F16_VEC_SET1           WSP_GGML_F32Cx16_SET1
#define WSP_GGML_F16_VEC_LOAD(p, i)     WSP_GGML_F32Cx16_LOAD(p)
#define WSP_GGML_F16_VEC_STORE(p, r, i) WSP_GGML_F32Cx16_STORE(p, r[i])
#define WSP_GGML_F16_VEC_FMA            WSP_GGML_F32Cx16_FMA
#define WSP_GGML_F16_VEC_ADD            WSP_GGML_F32Cx16_ADD
#define WSP_GGML_F16_VEC_MUL            WSP_GGML_F32Cx16_MUL
#define WSP_GGML_F16_VEC_REDUCE         WSP_GGML_F32Cx16_REDUCE

#elif defined(__AVX__)

#define WSP_GGML_SIMD

// F32 AVX

#define WSP_GGML_F32_STEP 32
#define WSP_GGML_F32_EPR  8

#define WSP_GGML_F32x8         __m256
#define WSP_GGML_F32x8_ZERO    _mm256_setzero_ps()
#define WSP_GGML_F32x8_SET1(x) _mm256_set1_ps(x)
#define WSP_GGML_F32x8_LOAD    _mm256_loadu_ps
#define WSP_GGML_F32x8_STORE   _mm256_storeu_ps
#if defined(__FMA__)
    #define WSP_GGML_F32x8_FMA(a, b, c) _mm256_fmadd_ps(b, c, a)
#else
    #define WSP_GGML_F32x8_FMA(a, b, c) _mm256_add_ps(_mm256_mul_ps(b, c), a)
#endif
#define WSP_GGML_F32x8_ADD     _mm256_add_ps
#define WSP_GGML_F32x8_MUL     _mm256_mul_ps
#define WSP_GGML_F32x8_REDUCE(res, x)                                 \
do {                                                              \
    int offset = WSP_GGML_F32_ARR >> 1;                               \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]),    \
                                 _mm256_extractf128_ps(x[0], 1)); \
    const __m128 t1 = _mm_hadd_ps(t0, t0);                        \
    res = (wsp_ggml_float) _mm_cvtss_f32(_mm_hadd_ps(t1, t1));        \
} while (0)
// TODO: is this optimal ?

#define WSP_GGML_F32_VEC        WSP_GGML_F32x8
#define WSP_GGML_F32_VEC_ZERO   WSP_GGML_F32x8_ZERO
#define WSP_GGML_F32_VEC_SET1   WSP_GGML_F32x8_SET1
#define WSP_GGML_F32_VEC_LOAD   WSP_GGML_F32x8_LOAD
#define WSP_GGML_F32_VEC_STORE  WSP_GGML_F32x8_STORE
#define WSP_GGML_F32_VEC_FMA    WSP_GGML_F32x8_FMA
#define WSP_GGML_F32_VEC_ADD    WSP_GGML_F32x8_ADD
#define WSP_GGML_F32_VEC_MUL    WSP_GGML_F32x8_MUL
#define WSP_GGML_F32_VEC_REDUCE WSP_GGML_F32x8_REDUCE

// F16 AVX

#define WSP_GGML_F16_STEP 32
#define WSP_GGML_F16_EPR  8

// F16 arithmetic is not supported by AVX, so we use F32 instead

#define WSP_GGML_F32Cx8             __m256
#define WSP_GGML_F32Cx8_ZERO        _mm256_setzero_ps()
#define WSP_GGML_F32Cx8_SET1(x)     _mm256_set1_ps(x)

#if defined(__F16C__)
// the  _mm256_cvt intrinsics require F16C
#define WSP_GGML_F32Cx8_LOAD(x)     _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x)))
#define WSP_GGML_F32Cx8_STORE(x, y) _mm_storeu_si128((__m128i *)(x), _mm256_cvtps_ph(y, 0))
#else
static inline __m256 __avx_f32cx8_load(wsp_ggml_fp16_t *x) {
    float tmp[8];

    for (int i = 0; i < 8; i++) {
        tmp[i] = WSP_GGML_FP16_TO_FP32(x[i]);
    }

    return _mm256_loadu_ps(tmp);
}
static inline void __avx_f32cx8_store(wsp_ggml_fp16_t *x, __m256 y) {
    float arr[8];

    _mm256_storeu_ps(arr, y);

    for (int i = 0; i < 8; i++)
        x[i] = WSP_GGML_FP32_TO_FP16(arr[i]);
}
#define WSP_GGML_F32Cx8_LOAD(x)     __avx_f32cx8_load(x)
#define WSP_GGML_F32Cx8_STORE(x, y) __avx_f32cx8_store(x, y)
#endif

#define WSP_GGML_F32Cx8_FMA         WSP_GGML_F32x8_FMA
#define WSP_GGML_F32Cx8_ADD         _mm256_add_ps
#define WSP_GGML_F32Cx8_MUL         _mm256_mul_ps
#define WSP_GGML_F32Cx8_REDUCE      WSP_GGML_F32x8_REDUCE

#define WSP_GGML_F16_VEC                WSP_GGML_F32Cx8
#define WSP_GGML_F16_VEC_ZERO           WSP_GGML_F32Cx8_ZERO
#define WSP_GGML_F16_VEC_SET1           WSP_GGML_F32Cx8_SET1
#define WSP_GGML_F16_VEC_LOAD(p, i)     WSP_GGML_F32Cx8_LOAD(p)
#define WSP_GGML_F16_VEC_STORE(p, r, i) WSP_GGML_F32Cx8_STORE(p, r[i])
#define WSP_GGML_F16_VEC_FMA            WSP_GGML_F32Cx8_FMA
#define WSP_GGML_F16_VEC_ADD            WSP_GGML_F32Cx8_ADD
#define WSP_GGML_F16_VEC_MUL            WSP_GGML_F32Cx8_MUL
#define WSP_GGML_F16_VEC_REDUCE         WSP_GGML_F32Cx8_REDUCE

#elif defined(__POWER9_VECTOR__)

#define WSP_GGML_SIMD

// F32 POWER9

#define WSP_GGML_F32_STEP 32
#define WSP_GGML_F32_EPR  4

#define WSP_GGML_F32x4              vector float
#define WSP_GGML_F32x4_ZERO         0.0f
#define WSP_GGML_F32x4_SET1         vec_splats
#define WSP_GGML_F32x4_LOAD(p)      vec_xl(0, p)
#define WSP_GGML_F32x4_STORE(p, r)  vec_xst(r, 0, p)
#define WSP_GGML_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#define WSP_GGML_F32x4_ADD          vec_add
#define WSP_GGML_F32x4_MUL          vec_mul
#define WSP_GGML_F32x4_REDUCE(res, x)              \
{                                              \
    int offset = WSP_GGML_F32_ARR >> 1;            \
    for (int i = 0; i < offset; ++i) {         \
        x[i] = vec_add(x[i], x[offset+i]);     \
    }                                          \
    offset >>= 1;                              \
    for (int i = 0; i < offset; ++i) {         \
        x[i] = vec_add(x[i], x[offset+i]);     \
    }                                          \
    offset >>= 1;                              \
    for (int i = 0; i < offset; ++i) {         \
        x[i] = vec_add(x[i], x[offset+i]);     \
    }                                          \
    res = vec_extract(x[0], 0) +               \
          vec_extract(x[0], 1) +               \
          vec_extract(x[0], 2) +               \
          vec_extract(x[0], 3);                \
}

#define WSP_GGML_F32_VEC        WSP_GGML_F32x4
#define WSP_GGML_F32_VEC_ZERO   WSP_GGML_F32x4_ZERO
#define WSP_GGML_F32_VEC_SET1   WSP_GGML_F32x4_SET1
#define WSP_GGML_F32_VEC_LOAD   WSP_GGML_F32x4_LOAD
#define WSP_GGML_F32_VEC_STORE  WSP_GGML_F32x4_STORE
#define WSP_GGML_F32_VEC_FMA    WSP_GGML_F32x4_FMA
#define WSP_GGML_F32_VEC_ADD    WSP_GGML_F32x4_ADD
#define WSP_GGML_F32_VEC_MUL    WSP_GGML_F32x4_MUL
#define WSP_GGML_F32_VEC_REDUCE WSP_GGML_F32x4_REDUCE

// F16 POWER9
#define WSP_GGML_F16_STEP       WSP_GGML_F32_STEP
#define WSP_GGML_F16_EPR        WSP_GGML_F32_EPR
#define WSP_GGML_F16_VEC        WSP_GGML_F32x4
#define WSP_GGML_F16_VEC_ZERO   WSP_GGML_F32x4_ZERO
#define WSP_GGML_F16_VEC_SET1   WSP_GGML_F32x4_SET1
#define WSP_GGML_F16_VEC_FMA    WSP_GGML_F32x4_FMA
#define WSP_GGML_F16_VEC_ADD    WSP_GGML_F32x4_ADD
#define WSP_GGML_F16_VEC_MUL    WSP_GGML_F32x4_MUL
#define WSP_GGML_F16_VEC_REDUCE WSP_GGML_F32x4_REDUCE
// Use vec_xl, not vec_ld, in case the load address is not aligned.
#define WSP_GGML_F16_VEC_LOAD(p, i) (i & 0x1) ?                   \
  vec_extract_fp32_from_shorth(vec_xl(0, p - WSP_GGML_F16_EPR)) : \
  vec_extract_fp32_from_shortl(vec_xl(0, p))
#define WSP_GGML_ENDIAN_BYTE(i) ((unsigned char *)&(uint16_t){1})[i]
#define WSP_GGML_F16_VEC_STORE(p, r, i)                             \
  if (i & 0x1)                                                  \
    vec_xst(vec_pack_to_short_fp32(r[i - WSP_GGML_ENDIAN_BYTE(1)],  \
                                   r[i - WSP_GGML_ENDIAN_BYTE(0)]), \
            0, p - WSP_GGML_F16_EPR)

#elif defined(__wasm_simd128__)

#define WSP_GGML_SIMD

// F32 WASM

#define WSP_GGML_F32_STEP 16
#define WSP_GGML_F32_EPR  4

#define WSP_GGML_F32x4              v128_t
#define WSP_GGML_F32x4_ZERO         wasm_f32x4_splat(0.0f)
#define WSP_GGML_F32x4_SET1(x)      wasm_f32x4_splat(x)
#define WSP_GGML_F32x4_LOAD         wasm_v128_load
#define WSP_GGML_F32x4_STORE        wasm_v128_store
#define WSP_GGML_F32x4_FMA(a, b, c) wasm_f32x4_add(wasm_f32x4_mul(b, c), a)
#define WSP_GGML_F32x4_ADD          wasm_f32x4_add
#define WSP_GGML_F32x4_MUL          wasm_f32x4_mul
#define WSP_GGML_F32x4_REDUCE(res, x)                  \
{                                                  \
    int offset = WSP_GGML_F32_ARR >> 1;                \
    for (int i = 0; i < offset; ++i) {             \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);  \
    }                                              \
    offset >>= 1;                                  \
    for (int i = 0; i < offset; ++i) {             \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);  \
    }                                              \
    offset >>= 1;                                  \
    for (int i = 0; i < offset; ++i) {             \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);  \
    }                                              \
    res = wasm_f32x4_extract_lane(x[0], 0) +       \
          wasm_f32x4_extract_lane(x[0], 1) +       \
          wasm_f32x4_extract_lane(x[0], 2) +       \
          wasm_f32x4_extract_lane(x[0], 3);        \
}

#define WSP_GGML_F32_VEC        WSP_GGML_F32x4
#define WSP_GGML_F32_VEC_ZERO   WSP_GGML_F32x4_ZERO
#define WSP_GGML_F32_VEC_SET1   WSP_GGML_F32x4_SET1
#define WSP_GGML_F32_VEC_LOAD   WSP_GGML_F32x4_LOAD
#define WSP_GGML_F32_VEC_STORE  WSP_GGML_F32x4_STORE
#define WSP_GGML_F32_VEC_FMA    WSP_GGML_F32x4_FMA
#define WSP_GGML_F32_VEC_ADD    WSP_GGML_F32x4_ADD
#define WSP_GGML_F32_VEC_MUL    WSP_GGML_F32x4_MUL
#define WSP_GGML_F32_VEC_REDUCE WSP_GGML_F32x4_REDUCE

// F16 WASM

#define WSP_GGML_F16_STEP 16
#define WSP_GGML_F16_EPR  4

inline static v128_t __wasm_f16x4_load(const wsp_ggml_fp16_t * p) {
    float tmp[4];

    tmp[0] = WSP_GGML_FP16_TO_FP32(p[0]);
    tmp[1] = WSP_GGML_FP16_TO_FP32(p[1]);
    tmp[2] = WSP_GGML_FP16_TO_FP32(p[2]);
    tmp[3] = WSP_GGML_FP16_TO_FP32(p[3]);

    return wasm_v128_load(tmp);
}

inline static void __wasm_f16x4_store(wsp_ggml_fp16_t * p, v128_t x) {
    float tmp[4];

    wasm_v128_store(tmp, x);

    p[0] = WSP_GGML_FP32_TO_FP16(tmp[0]);
    p[1] = WSP_GGML_FP32_TO_FP16(tmp[1]);
    p[2] = WSP_GGML_FP32_TO_FP16(tmp[2]);
    p[3] = WSP_GGML_FP32_TO_FP16(tmp[3]);
}

#define WSP_GGML_F16x4             v128_t
#define WSP_GGML_F16x4_ZERO        wasm_f32x4_splat(0.0f)
#define WSP_GGML_F16x4_SET1(x)     wasm_f32x4_splat(x)
#define WSP_GGML_F16x4_LOAD(x)     __wasm_f16x4_load(x)
#define WSP_GGML_F16x4_STORE(x, y) __wasm_f16x4_store(x, y)
#define WSP_GGML_F16x4_FMA         WSP_GGML_F32x4_FMA
#define WSP_GGML_F16x4_ADD         wasm_f32x4_add
#define WSP_GGML_F16x4_MUL         wasm_f32x4_mul
#define WSP_GGML_F16x4_REDUCE(res, x)                  \
{                                                  \
    int offset = WSP_GGML_F16_ARR >> 1;                \
    for (int i = 0; i < offset; ++i) {             \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);  \
    }                                              \
    offset >>= 1;                                  \
    for (int i = 0; i < offset; ++i) {             \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);  \
    }                                              \
    offset >>= 1;                                  \
    for (int i = 0; i < offset; ++i) {             \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);  \
    }                                              \
    res = wasm_f32x4_extract_lane(x[0], 0) +       \
          wasm_f32x4_extract_lane(x[0], 1) +       \
          wasm_f32x4_extract_lane(x[0], 2) +       \
          wasm_f32x4_extract_lane(x[0], 3);        \
}

#define WSP_GGML_F16_VEC                WSP_GGML_F16x4
#define WSP_GGML_F16_VEC_ZERO           WSP_GGML_F16x4_ZERO
#define WSP_GGML_F16_VEC_SET1           WSP_GGML_F16x4_SET1
#define WSP_GGML_F16_VEC_LOAD(p, i)     WSP_GGML_F16x4_LOAD(p)
#define WSP_GGML_F16_VEC_STORE(p, r, i) WSP_GGML_F16x4_STORE(p, r[i])
#define WSP_GGML_F16_VEC_FMA            WSP_GGML_F16x4_FMA
#define WSP_GGML_F16_VEC_ADD            WSP_GGML_F16x4_ADD
#define WSP_GGML_F16_VEC_MUL            WSP_GGML_F16x4_MUL
#define WSP_GGML_F16_VEC_REDUCE         WSP_GGML_F16x4_REDUCE

#elif defined(__SSE3__)

#define WSP_GGML_SIMD

// F32 SSE

#define WSP_GGML_F32_STEP 32
#define WSP_GGML_F32_EPR  4

#define WSP_GGML_F32x4         __m128
#define WSP_GGML_F32x4_ZERO    _mm_setzero_ps()
#define WSP_GGML_F32x4_SET1(x) _mm_set1_ps(x)
#define WSP_GGML_F32x4_LOAD    _mm_loadu_ps
#define WSP_GGML_F32x4_STORE   _mm_storeu_ps
#if defined(__FMA__)
    // TODO: Does this work?
    #define WSP_GGML_F32x4_FMA(a, b, c) _mm_fmadd_ps(b, c, a)
#else
    #define WSP_GGML_F32x4_FMA(a, b, c) _mm_add_ps(_mm_mul_ps(b, c), a)
#endif
#define WSP_GGML_F32x4_ADD     _mm_add_ps
#define WSP_GGML_F32x4_MUL     _mm_mul_ps
#define WSP_GGML_F32x4_REDUCE(res, x)                                 \
{                                                                 \
    int offset = WSP_GGML_F32_ARR >> 1;                               \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm_add_ps(x[i], x[offset+i]);                     \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm_add_ps(x[i], x[offset+i]);                     \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm_add_ps(x[i], x[offset+i]);                     \
    }                                                             \
    const __m128 t0 = _mm_hadd_ps(x[0], x[0]);                    \
    res = (wsp_ggml_float) _mm_cvtss_f32(_mm_hadd_ps(t0, t0));        \
}
// TODO: is this optimal ?

#define WSP_GGML_F32_VEC        WSP_GGML_F32x4
#define WSP_GGML_F32_VEC_ZERO   WSP_GGML_F32x4_ZERO
#define WSP_GGML_F32_VEC_SET1   WSP_GGML_F32x4_SET1
#define WSP_GGML_F32_VEC_LOAD   WSP_GGML_F32x4_LOAD
#define WSP_GGML_F32_VEC_STORE  WSP_GGML_F32x4_STORE
#define WSP_GGML_F32_VEC_FMA    WSP_GGML_F32x4_FMA
#define WSP_GGML_F32_VEC_ADD    WSP_GGML_F32x4_ADD
#define WSP_GGML_F32_VEC_MUL    WSP_GGML_F32x4_MUL
#define WSP_GGML_F32_VEC_REDUCE WSP_GGML_F32x4_REDUCE

// F16 SSE

#define WSP_GGML_F16_STEP 32
#define WSP_GGML_F16_EPR  4

static inline __m128 __sse_f16x4_load(wsp_ggml_fp16_t *x) {
    float tmp[4];

    tmp[0] = WSP_GGML_FP16_TO_FP32(x[0]);
    tmp[1] = WSP_GGML_FP16_TO_FP32(x[1]);
    tmp[2] = WSP_GGML_FP16_TO_FP32(x[2]);
    tmp[3] = WSP_GGML_FP16_TO_FP32(x[3]);

    return _mm_loadu_ps(tmp);
}

static inline void __sse_f16x4_store(wsp_ggml_fp16_t *x, __m128 y) {
    float arr[4];

    _mm_storeu_ps(arr, y);

    x[0] = WSP_GGML_FP32_TO_FP16(arr[0]);
    x[1] = WSP_GGML_FP32_TO_FP16(arr[1]);
    x[2] = WSP_GGML_FP32_TO_FP16(arr[2]);
    x[3] = WSP_GGML_FP32_TO_FP16(arr[3]);
}

#define WSP_GGML_F32Cx4             __m128
#define WSP_GGML_F32Cx4_ZERO        _mm_setzero_ps()
#define WSP_GGML_F32Cx4_SET1(x)     _mm_set1_ps(x)
#define WSP_GGML_F32Cx4_LOAD(x)     __sse_f16x4_load(x)
#define WSP_GGML_F32Cx4_STORE(x, y) __sse_f16x4_store(x, y)
#define WSP_GGML_F32Cx4_FMA         WSP_GGML_F32x4_FMA
#define WSP_GGML_F32Cx4_ADD         _mm_add_ps
#define WSP_GGML_F32Cx4_MUL         _mm_mul_ps
#define WSP_GGML_F32Cx4_REDUCE      WSP_GGML_F32x4_REDUCE

#define WSP_GGML_F16_VEC                 WSP_GGML_F32Cx4
#define WSP_GGML_F16_VEC_ZERO            WSP_GGML_F32Cx4_ZERO
#define WSP_GGML_F16_VEC_SET1            WSP_GGML_F32Cx4_SET1
#define WSP_GGML_F16_VEC_LOAD(p, i)      WSP_GGML_F32Cx4_LOAD(p)
#define WSP_GGML_F16_VEC_STORE(p, r, i)  WSP_GGML_F32Cx4_STORE(p, r[i])
#define WSP_GGML_F16_VEC_FMA             WSP_GGML_F32Cx4_FMA
#define WSP_GGML_F16_VEC_ADD             WSP_GGML_F32Cx4_ADD
#define WSP_GGML_F16_VEC_MUL             WSP_GGML_F32Cx4_MUL
#define WSP_GGML_F16_VEC_REDUCE          WSP_GGML_F32Cx4_REDUCE

#elif defined(__loongarch_asx)

#define WSP_GGML_SIMD

// F32 LASX
#define WSP_GGML_F32_STEP 32
#define WSP_GGML_F32_EPR  8

#define WSP_GGML_F32x8         __m256
#define WSP_GGML_F32x8_ZERO    (__m256)__lasx_xvldi(0)
#define WSP_GGML_F32x8_SET1(x) (__m256)__lasx_xvreplfr2vr_s((x))
#define WSP_GGML_F32x8_LOAD(x) (__m256)__lasx_xvld((x), 0)
#define WSP_GGML_F32x8_STORE(x,y)   __lasx_xvst((y), (x), 0)
#define WSP_GGML_F32x8_FMA(a, b, c) __lasx_xvfmadd_s(b, c, a)
#define WSP_GGML_F32x8_ADD     __lasx_xvfadd_s
#define WSP_GGML_F32x8_MUL     __lasx_xvfmul_s
#define WSP_GGML_F32x8_REDUCE(res, x)                                 \
do {                                                              \
    int offset = WSP_GGML_F32_ARR >> 1;                               \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[offset+i]);                  \
    }                                                             \
    float *tmp_p = (float *)&x[0]; \
    res = tmp_p[0] + tmp_p[1] + tmp_p[2] + tmp_p[3] + tmp_p[4] + tmp_p[5] + tmp_p[6] + tmp_p[7];  \
} while (0)
// TODO: is this optimal ?

#define WSP_GGML_F32_VEC        WSP_GGML_F32x8
#define WSP_GGML_F32_VEC_ZERO   WSP_GGML_F32x8_ZERO
#define WSP_GGML_F32_VEC_SET1   WSP_GGML_F32x8_SET1
#define WSP_GGML_F32_VEC_LOAD   WSP_GGML_F32x8_LOAD
#define WSP_GGML_F32_VEC_STORE  WSP_GGML_F32x8_STORE
#define WSP_GGML_F32_VEC_FMA    WSP_GGML_F32x8_FMA
#define WSP_GGML_F32_VEC_ADD    WSP_GGML_F32x8_ADD
#define WSP_GGML_F32_VEC_MUL    WSP_GGML_F32x8_MUL
#define WSP_GGML_F32_VEC_REDUCE WSP_GGML_F32x8_REDUCE

// F16 LASX

#define WSP_GGML_F16_STEP 32
#define WSP_GGML_F16_EPR  8

// F16 arithmetic is not supported by AVX, so we use F32 instead

#define WSP_GGML_F32Cx8          __m256
#define WSP_GGML_F32Cx8_ZERO    (__m256)__lasx_xvldi(0)
#define WSP_GGML_F32Cx8_SET1(x) (__m256)__lasx_xvreplgr2vr_w((x))

static inline __m256 __lasx_f32cx8_load(const wsp_ggml_fp16_t * x) {
    float tmp[8];

    for (int i = 0; i < 8; i++) {
        tmp[i] = WSP_GGML_FP16_TO_FP32(x[i]);
    }

    return (__m256)__lasx_xvld(tmp, 0);
}
static inline void __lasx_f32cx8_store(wsp_ggml_fp16_t * x, __m256 y) {
    float arr[8];

    __lasx_xvst(y, arr, 0);

    for (int i = 0; i < 8; i++) {
        x[i] = WSP_GGML_FP32_TO_FP16(arr[i]);
    }
}
#define WSP_GGML_F32Cx8_LOAD(x)     __lasx_f32cx8_load(x)
#define WSP_GGML_F32Cx8_STORE(x, y) __lasx_f32cx8_store(x, y)

#define WSP_GGML_F32Cx8_FMA         WSP_GGML_F32x8_FMA
#define WSP_GGML_F32Cx8_ADD         __lasx_xvfadd_s
#define WSP_GGML_F32Cx8_MUL         __lasx_xvfmul_s
#define WSP_GGML_F32Cx8_REDUCE      WSP_GGML_F32x8_REDUCE

#define WSP_GGML_F16_VEC                WSP_GGML_F32Cx8
#define WSP_GGML_F16_VEC_ZERO           WSP_GGML_F32Cx8_ZERO
#define WSP_GGML_F16_VEC_SET1           WSP_GGML_F32Cx8_SET1
#define WSP_GGML_F16_VEC_LOAD(p, i)     WSP_GGML_F32Cx8_LOAD(p)
#define WSP_GGML_F16_VEC_STORE(p, r, i) WSP_GGML_F32Cx8_STORE(p, r[i])
#define WSP_GGML_F16_VEC_FMA            WSP_GGML_F32Cx8_FMA
#define WSP_GGML_F16_VEC_ADD            WSP_GGML_F32Cx8_ADD
#define WSP_GGML_F16_VEC_MUL            WSP_GGML_F32Cx8_MUL
#define WSP_GGML_F16_VEC_REDUCE         WSP_GGML_F32Cx8_REDUCE

#elif defined(__loongarch_sx)

#define WSP_GGML_SIMD

// F32 LSX

#define WSP_GGML_F32_STEP 32
#define WSP_GGML_F32_EPR  4

#define WSP_GGML_F32x4         __m128
#define WSP_GGML_F32x4_ZERO    __lsx_vldi(0)
#define WSP_GGML_F32x4_SET1(x) __lsx_vinsgr2vr_w(__lsx_vldi(0),(x), 0)
#define WSP_GGML_F32x4_LOAD(x) __lsx_vld((x), 0)
#define WSP_GGML_F32x4_STORE((x),(y))   __lsx_vst((y), (x), 0)
#define WSP_GGML_F32x4_FMA(a, b, c) __lsx_vfmadd_s(b, c, a)
#define WSP_GGML_F32x4_ADD     __lsx_vfadd_s
#define WSP_GGML_F32x4_MUL     __lsx_vfmul_s
#define WSP_GGML_F32x4_REDUCE(res, x)                                 \
{                                                                 \
    int offset = WSP_GGML_F32_ARR >> 1;                               \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = __lsx_vfadd_s(x[i], x[offset+i]);                     \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = __lsx_vfadd_s(x[i], x[offset+i]);                     \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = __lsx_vfadd_s(x[i], x[offset+i]);                     \
    }                                                             \
    __m128i tmp = __lsx_vsrli_d((__m128i)x[0], 32); \
    tmp = (__m128i)__lsx_vfadd_s((__m128)tmp, x[0]); \
    tmp = __lsx_vpickev_w(__lsx_vldi(0), tmp); \
    const __m128 t0 = __lsx_vshuf4i_w(tmp, 0x88); \
    tmp = __lsx_vsrli_d((__m128i)t0, 32); \
    tmp = (__m128i)__lsx_vfadd_s((__m128)tmp, t0); \
    tmp = __lsx_vpickev_w(__lsx_vldi(0), tmp); \
    res = (wsp_ggml_float) __lsx_vpickve2gr_w(__lsx_vshuf4i_w(tmp, 0x88), 0);        \
}

#define WSP_GGML_F32_VEC        WSP_GGML_F32x4
#define WSP_GGML_F32_VEC_ZERO   WSP_GGML_F32x4_ZERO
#define WSP_GGML_F32_VEC_SET1   WSP_GGML_F32x4_SET1
#define WSP_GGML_F32_VEC_LOAD   WSP_GGML_F32x4_LOAD
#define WSP_GGML_F32_VEC_STORE  WSP_GGML_F32x4_STORE
#define WSP_GGML_F32_VEC_FMA    WSP_GGML_F32x4_FMA
#define WSP_GGML_F32_VEC_ADD    WSP_GGML_F32x4_ADD
#define WSP_GGML_F32_VEC_MUL    WSP_GGML_F32x4_MUL
#define WSP_GGML_F32_VEC_REDUCE WSP_GGML_F32x4_REDUCE

// F16 LSX

#define WSP_GGML_F16_STEP 32
#define WSP_GGML_F16_EPR  4

static inline __m128 __lsx_f16x4_load(const wsp_ggml_fp16_t * x) {
    float tmp[4];

    tmp[0] = WSP_GGML_FP16_TO_FP32(x[0]);
    tmp[1] = WSP_GGML_FP16_TO_FP32(x[1]);
    tmp[2] = WSP_GGML_FP16_TO_FP32(x[2]);
    tmp[3] = WSP_GGML_FP16_TO_FP32(x[3]);

    return __lsx_vld(tmp, 0);
}

static inline void __lsx_f16x4_store(wsp_ggml_fp16_t * x, __m128 y) {
    float arr[4];

    __lsx_vst(y, arr, 0);

    x[0] = WSP_GGML_FP32_TO_FP16(arr[0]);
    x[1] = WSP_GGML_FP32_TO_FP16(arr[1]);
    x[2] = WSP_GGML_FP32_TO_FP16(arr[2]);
    x[3] = WSP_GGML_FP32_TO_FP16(arr[3]);
}

#define WSP_GGML_F32Cx4             __m128
#define WSP_GGML_F32Cx4_ZERO        __lsx_vldi(0)
#define WSP_GGML_F32Cx4_SET1(x)     __lsx_vinsgr2vr_w(__lsx_vldi(0),(x), 0)
#define WSP_GGML_F32Cx4_LOAD(x)     __lsx_f16x4_load(x)
#define WSP_GGML_F32Cx4_STORE(x, y) __lsx_f16x4_store(x, y)
#define WSP_GGML_F32Cx4_FMA         WSP_GGML_F32x4_FMA
#define WSP_GGML_F32Cx4_ADD         __lsx_vfadd_s
#define WSP_GGML_F32Cx4_MUL         __lsx_vfmul_s
#define WSP_GGML_F32Cx4_REDUCE      WSP_GGML_F32x4_REDUCE

#define WSP_GGML_F16_VEC                 WSP_GGML_F32Cx4
#define WSP_GGML_F16_VEC_ZERO            WSP_GGML_F32Cx4_ZERO
#define WSP_GGML_F16_VEC_SET1            WSP_GGML_F32Cx4_SET1
#define WSP_GGML_F16_VEC_LOAD(p, i)      WSP_GGML_F32Cx4_LOAD(p)
#define WSP_GGML_F16_VEC_STORE(p, r, i)  WSP_GGML_F32Cx4_STORE(p, r[i])
#define WSP_GGML_F16_VEC_FMA             WSP_GGML_F32Cx4_FMA
#define WSP_GGML_F16_VEC_ADD             WSP_GGML_F32Cx4_ADD
#define WSP_GGML_F16_VEC_MUL             WSP_GGML_F32Cx4_MUL
#define WSP_GGML_F16_VEC_REDUCE          WSP_GGML_F32Cx4_REDUCE

#endif

// WSP_GGML_F32_ARR / WSP_GGML_F16_ARR
//   number of registers to use per step
#ifdef WSP_GGML_SIMD
#define WSP_GGML_F32_ARR (WSP_GGML_F32_STEP/WSP_GGML_F32_EPR)
#define WSP_GGML_F16_ARR (WSP_GGML_F16_STEP/WSP_GGML_F16_EPR)
#endif

//
// Threading defs
//

typedef pthread_t          wsp_ggml_thread_t;

#if defined(_WIN32)

typedef CONDITION_VARIABLE wsp_ggml_cond_t;
typedef SRWLOCK            wsp_ggml_mutex_t;

#define wsp_ggml_mutex_init(m)   InitializeSRWLock(m)
#define wsp_ggml_mutex_destroy(m)
#define wsp_ggml_mutex_lock(m)   AcquireSRWLockExclusive(m)
#define wsp_ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#define wsp_ggml_mutex_lock_shared(m)   AcquireSRWLockShared(m)
#define wsp_ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#define wsp_ggml_cond_init(c)    InitializeConditionVariable(c)
#define wsp_ggml_cond_destroy(c)
#define wsp_ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#define wsp_ggml_cond_broadcast(c) WakeAllConditionVariable(c)

#define wsp_ggml_thread_create pthread_create
#define wsp_ggml_thread_join   pthread_join

#else

typedef pthread_cond_t     wsp_ggml_cond_t;
typedef pthread_mutex_t    wsp_ggml_mutex_t;

#define wsp_ggml_mutex_init(m)          pthread_mutex_init(m, NULL)
#define wsp_ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define wsp_ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define wsp_ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define wsp_ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define wsp_ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define wsp_ggml_lock_init(x)    UNUSED(x)
#define wsp_ggml_lock_destroy(x) UNUSED(x)
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
#define wsp_ggml_lock_lock(x)    _mm_pause()
#else
#define wsp_ggml_lock_lock(x)    UNUSED(x)
#endif
#define wsp_ggml_lock_unlock(x)  UNUSED(x)

#define WSP_GGML_LOCK_INITIALIZER 0
#define wsp_ggml_cond_init(c)      pthread_cond_init(c, NULL)
#define wsp_ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define wsp_ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define wsp_ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define wsp_ggml_thread_create pthread_create
#define wsp_ggml_thread_join   pthread_join

#endif

// Threadpool def
struct wsp_ggml_threadpool {
    wsp_ggml_mutex_t mutex;       // mutex for cond.var
    wsp_ggml_cond_t  cond;        // cond.var for waiting for new work

    struct wsp_ggml_cgraph * cgraph;
    struct wsp_ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // incremented when there is work to be done (i.e each graph)
    atomic_int WSP_GGML_CACHE_ALIGN n_barrier;
    atomic_int WSP_GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_bool abort;        // Used for aborting processing of a graph

    struct wsp_ggml_compute_state * workers;   // per thread state
    int          n_threads_max; // number of threads in the pool
    atomic_int   n_threads_cur; // number of threads used in the current graph

    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    enum wsp_ggml_status ec;
};

// Per-thread state
struct wsp_ggml_compute_state {
#ifndef WSP_GGML_USE_OPENMP
    wsp_ggml_thread_t thrd;
    bool cpumask[WSP_GGML_MAX_N_THREADS];
    int  last_graph;
    bool pending;
#endif
    struct wsp_ggml_threadpool * threadpool;
    int ith;
};

struct wsp_ggml_compute_params {
    // ith = thread index, nth = number of threads
    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void * wdata;

    struct wsp_ggml_threadpool * threadpool;
};

//
// fundamental operations
//

inline static void wsp_ggml_vec_set_i8(const int n, int8_t * x, const int8_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void wsp_ggml_vec_set_i16(const int n, int16_t * x, const int16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void wsp_ggml_vec_set_i32(const int n, int32_t * x, const int32_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void wsp_ggml_vec_set_f16(const int n, wsp_ggml_fp16_t * x, const int32_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void wsp_ggml_vec_set_bf16(const int n, wsp_ggml_bf16_t * x, const wsp_ggml_bf16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void wsp_ggml_vec_add_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i] + y[i]; }
inline static void wsp_ggml_vec_add1_f32(const int n, float * z, const float * x, const float   v) { for (int i = 0; i < n; ++i) z[i]  = x[i] + v;    }
inline static void wsp_ggml_vec_acc_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i] += x[i];        }
inline static void wsp_ggml_vec_acc1_f32(const int n, float * y, const float   v)                  { for (int i = 0; i < n; ++i) y[i] += v;           }
inline static void wsp_ggml_vec_sub_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i] - y[i]; }
inline static void wsp_ggml_vec_set_f32 (const int n, float * x, const float   v)                  { for (int i = 0; i < n; ++i) x[i]  = v;           }
inline static void wsp_ggml_vec_cpy_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i]  = x[i];        }
inline static void wsp_ggml_vec_neg_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i]  = -x[i];       }
inline static void wsp_ggml_vec_mul_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i]*y[i];   }
inline static void wsp_ggml_vec_div_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i]/y[i];   }

static void wsp_ggml_vec_dot_f32(int n, float * restrict s, size_t bs, const float * restrict x, size_t bx, const float * restrict y, size_t by, int nrc) {
   assert(nrc == 1);
   UNUSED(nrc);
   UNUSED(bx);
   UNUSED(by);
   UNUSED(bs);

#if defined(WSP_GGML_SIMD)
    float sumf = 0.0f;
    const int np = (n & ~(WSP_GGML_F32_STEP - 1));

    WSP_GGML_F32_VEC sum[WSP_GGML_F32_ARR] = { WSP_GGML_F32_VEC_ZERO };

    WSP_GGML_F32_VEC ax[WSP_GGML_F32_ARR];
    WSP_GGML_F32_VEC ay[WSP_GGML_F32_ARR];

    for (int i = 0; i < np; i += WSP_GGML_F32_STEP) {
        for (int j = 0; j < WSP_GGML_F32_ARR; j++) {
            ax[j] = WSP_GGML_F32_VEC_LOAD(x + i + j*WSP_GGML_F32_EPR);
            ay[j] = WSP_GGML_F32_VEC_LOAD(y + i + j*WSP_GGML_F32_EPR);

            sum[j] = WSP_GGML_F32_VEC_FMA(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    WSP_GGML_F32_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += x[i]*y[i];
    }
#else
    // scalar
    wsp_ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (wsp_ggml_float)(x[i]*y[i]);
    }
#endif

    *s = sumf;
}

static void wsp_ggml_vec_dot_bf16(int n, float * restrict s, size_t bs, wsp_ggml_bf16_t * restrict x, size_t bx, wsp_ggml_bf16_t * restrict y, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    int i = 0;
    wsp_ggml_float sumf = 0;

#if defined(__AVX512BF16__)
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    for (; i + 64 <= n; i += 64) {
        c1 = _mm512_dpbf16_ps(c1, m512bh(_mm512_loadu_si512((x + i))),
                             m512bh(_mm512_loadu_si512((y + i))));
        c2 = _mm512_dpbf16_ps(c2, m512bh(_mm512_loadu_si512((x + i + 32))),
                             m512bh(_mm512_loadu_si512((y + i + 32))));
    }
    sumf += (wsp_ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (wsp_ggml_float)_mm512_reduce_add_ps(c2);

#elif defined(__AVX512F__)
#define LOAD(p) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(p))), 16))
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        c1 = _mm512_add_ps(_mm512_mul_ps(LOAD(x + i), LOAD(y + i)), c1);
        c2 = _mm512_add_ps(_mm512_mul_ps(LOAD(x + i + 16), LOAD(y + i + 16)), c2);
    }
    sumf += (wsp_ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (wsp_ggml_float)_mm512_reduce_add_ps(c2);

#undef LOAD
#elif defined(__AVX2__)
#define LOAD(p) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16))
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        c1 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i), LOAD(y + i)), c1);
        c2 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 8), LOAD(y + i + 8)), c2);
        c3 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 16), LOAD(y + i + 16)), c3);
        c4 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 24), LOAD(y + i + 24)), c4);
    }
    __m128 g;
    c1 = _mm256_add_ps(_mm256_add_ps(c1, c3),
                       _mm256_add_ps(c2, c4));
    g = _mm_add_ps(_mm256_extractf128_ps(c1, 1),
                   _mm256_castps256_ps128(c1));
    g = _mm_add_ps(g, _mm_movehl_ps(g, g));
    g = _mm_add_ss(g, _mm_movehdup_ps(g));
    sumf += (wsp_ggml_float)_mm_cvtss_f32(g);

#undef LOAD
#endif

    for (; i < n; ++i) {
        sumf += (wsp_ggml_float)(WSP_GGML_BF16_TO_FP32(x[i]) *
                             WSP_GGML_BF16_TO_FP32(y[i]));
    }
    *s = sumf;
}

static void wsp_ggml_vec_dot_f16(int n, float * restrict s, size_t bs, wsp_ggml_fp16_t * restrict x, size_t bx, wsp_ggml_fp16_t * restrict y, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    wsp_ggml_float sumf = 0.0;

#if defined(WSP_GGML_SIMD)
    const int np = (n & ~(WSP_GGML_F16_STEP - 1));

    WSP_GGML_F16_VEC sum[WSP_GGML_F16_ARR] = { WSP_GGML_F16_VEC_ZERO };

    WSP_GGML_F16_VEC ax[WSP_GGML_F16_ARR];
    WSP_GGML_F16_VEC ay[WSP_GGML_F16_ARR];

    for (int i = 0; i < np; i += WSP_GGML_F16_STEP) {
        for (int j = 0; j < WSP_GGML_F16_ARR; j++) {
            ax[j] = WSP_GGML_F16_VEC_LOAD(x + i + j*WSP_GGML_F16_EPR, j);
            ay[j] = WSP_GGML_F16_VEC_LOAD(y + i + j*WSP_GGML_F16_EPR, j);

            sum[j] = WSP_GGML_F16_VEC_FMA(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    WSP_GGML_F16_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += (wsp_ggml_float)(WSP_GGML_FP16_TO_FP32(x[i])*WSP_GGML_FP16_TO_FP32(y[i]));
    }
#else
    for (int i = 0; i < n; ++i) {
        sumf += (wsp_ggml_float)(WSP_GGML_FP16_TO_FP32(x[i])*WSP_GGML_FP16_TO_FP32(y[i]));
    }
#endif

    *s = sumf;
}

// compute WSP_GGML_VEC_DOT_UNROLL dot products at once
// xs - x row stride in bytes
inline static void wsp_ggml_vec_dot_f16_unroll(const int n, const int xs, float * restrict s, void * restrict xv, wsp_ggml_fp16_t * restrict y) {
    wsp_ggml_float sumf[WSP_GGML_VEC_DOT_UNROLL] = { 0.0 };

    wsp_ggml_fp16_t * restrict x[WSP_GGML_VEC_DOT_UNROLL];

    for (int i = 0; i < WSP_GGML_VEC_DOT_UNROLL; ++i) {
        x[i] = (wsp_ggml_fp16_t *) ((char *) xv + i*xs);
    }

#if defined(WSP_GGML_SIMD)
    const int np = (n & ~(WSP_GGML_F16_STEP - 1));

    WSP_GGML_F16_VEC sum[WSP_GGML_VEC_DOT_UNROLL][WSP_GGML_F16_ARR] = { { WSP_GGML_F16_VEC_ZERO } };

    WSP_GGML_F16_VEC ax[WSP_GGML_F16_ARR];
    WSP_GGML_F16_VEC ay[WSP_GGML_F16_ARR];

    for (int i = 0; i < np; i += WSP_GGML_F16_STEP) {
        for (int j = 0; j < WSP_GGML_F16_ARR; j++) {
            ay[j] = WSP_GGML_F16_VEC_LOAD(y + i + j*WSP_GGML_F16_EPR, j);

            for (int k = 0; k < WSP_GGML_VEC_DOT_UNROLL; ++k) {
                ax[j] = WSP_GGML_F16_VEC_LOAD(x[k] + i + j*WSP_GGML_F16_EPR, j);

                sum[k][j] = WSP_GGML_F16_VEC_FMA(sum[k][j], ax[j], ay[j]);
            }
        }
    }

    // reduce sum0..sum3 to sum0
    for (int k = 0; k < WSP_GGML_VEC_DOT_UNROLL; ++k) {
        WSP_GGML_F16_VEC_REDUCE(sumf[k], sum[k]);
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        for (int j = 0; j < WSP_GGML_VEC_DOT_UNROLL; ++j) {
            sumf[j] += (wsp_ggml_float)(WSP_GGML_FP16_TO_FP32(x[j][i])*WSP_GGML_FP16_TO_FP32(y[i]));
        }
    }
#else
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < WSP_GGML_VEC_DOT_UNROLL; ++j) {
            sumf[j] += (wsp_ggml_float)(WSP_GGML_FP16_TO_FP32(x[j][i])*WSP_GGML_FP16_TO_FP32(y[i]));
        }
    }
#endif

    for (int i = 0; i < WSP_GGML_VEC_DOT_UNROLL; ++i) {
        s[i] = sumf[i];
    }
}

inline static void wsp_ggml_vec_mad_f32(const int n, float * restrict y, const float * restrict x, const float v) {
#if defined(WSP_GGML_SIMD)
    const int np = (n & ~(WSP_GGML_F32_STEP - 1));

    WSP_GGML_F32_VEC vx = WSP_GGML_F32_VEC_SET1(v);

    WSP_GGML_F32_VEC ax[WSP_GGML_F32_ARR];
    WSP_GGML_F32_VEC ay[WSP_GGML_F32_ARR];

    for (int i = 0; i < np; i += WSP_GGML_F32_STEP) {
        for (int j = 0; j < WSP_GGML_F32_ARR; j++) {
            ax[j] = WSP_GGML_F32_VEC_LOAD(x + i + j*WSP_GGML_F32_EPR);
            ay[j] = WSP_GGML_F32_VEC_LOAD(y + i + j*WSP_GGML_F32_EPR);
            ay[j] = WSP_GGML_F32_VEC_FMA(ay[j], ax[j], vx);

            WSP_GGML_F32_VEC_STORE(y + i + j*WSP_GGML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] += x[i]*v;
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] += x[i]*v;
    }
#endif
}

inline static void wsp_ggml_vec_mad_f16(const int n, wsp_ggml_fp16_t * restrict y, const wsp_ggml_fp16_t * restrict x, const float v) {
#if defined(WSP_GGML_SIMD)
    const int np = (n & ~(WSP_GGML_F16_STEP - 1));

    WSP_GGML_F16_VEC vx = WSP_GGML_F16_VEC_SET1(v);

    WSP_GGML_F16_VEC ax[WSP_GGML_F16_ARR];
    WSP_GGML_F16_VEC ay[WSP_GGML_F16_ARR];

    for (int i = 0; i < np; i += WSP_GGML_F16_STEP) {
        for (int j = 0; j < WSP_GGML_F16_ARR; j++) {
            ax[j] = WSP_GGML_F16_VEC_LOAD(x + i + j*WSP_GGML_F16_EPR, j);
            ay[j] = WSP_GGML_F16_VEC_LOAD(y + i + j*WSP_GGML_F16_EPR, j);
            ay[j] = WSP_GGML_F16_VEC_FMA(ay[j], ax[j], vx);

            WSP_GGML_F16_VEC_STORE(y + i + j*WSP_GGML_F16_EPR, ay, j);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] = WSP_GGML_FP32_TO_FP16(WSP_GGML_FP16_TO_FP32(y[i]) + WSP_GGML_FP16_TO_FP32(x[i])*v);
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] = WSP_GGML_FP32_TO_FP16(WSP_GGML_FP16_TO_FP32(y[i]) + WSP_GGML_FP16_TO_FP32(x[i])*v);
    }
#endif
}

// xs and vs are byte strides of x and v
inline static void wsp_ggml_vec_mad_f32_unroll(const int n, const int xs, const int vs, float * restrict y, const float * restrict xv, const float * restrict vv) {

    const float * restrict x[WSP_GGML_VEC_MAD_UNROLL];
    const float * restrict v[WSP_GGML_VEC_MAD_UNROLL];

    for (int i = 0; i < WSP_GGML_VEC_MAD_UNROLL; ++i) {
        x[i] = (const float *) ((const char *) xv + i*xs);
        v[i] = (const float *) ((const char *) vv + i*vs);
    }

#if defined(WSP_GGML_SIMD)
    const int np = (n & ~(WSP_GGML_F32_STEP - 1));

    WSP_GGML_F32_VEC vx[WSP_GGML_VEC_MAD_UNROLL];

    for (int k = 0; k < WSP_GGML_VEC_MAD_UNROLL; ++k) {
        vx[k] = WSP_GGML_F32_VEC_SET1(v[k][0]);
    }

    WSP_GGML_F32_VEC ax[WSP_GGML_VEC_MAD_UNROLL][WSP_GGML_F32_ARR];
    WSP_GGML_F32_VEC ay[WSP_GGML_F32_ARR];

    for (int i = 0; i < np; i += WSP_GGML_F32_STEP) {
        for (int j = 0; j < WSP_GGML_F32_ARR; j++) {
            ay[j] = WSP_GGML_F32_VEC_LOAD(y + i + j*WSP_GGML_F32_EPR);

            for (int k = 0; k < WSP_GGML_VEC_MAD_UNROLL; ++k) {
                ax[k][j] = WSP_GGML_F32_VEC_LOAD(x[k] + i + j*WSP_GGML_F32_EPR);
                ay[j] = WSP_GGML_F32_VEC_FMA(ay[j], ax[k][j], vx[k]);
            }

            WSP_GGML_F32_VEC_STORE(y + i + j*WSP_GGML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int k = 0; k < WSP_GGML_VEC_MAD_UNROLL; ++k) {
        for (int i = np; i < n; ++i) {
            y[i] += x[k][i]*v[k][0];
        }
    }
#else
    // scalar
    for (int k = 0; k < WSP_GGML_VEC_MAD_UNROLL; ++k) {
        for (int i = 0; i < n; ++i) {
            y[i] += x[k][i]*v[k][0];
        }
    }
#endif
}

//inline static void wsp_ggml_vec_scale_f32(const int n, float * y, const float   v) { for (int i = 0; i < n; ++i) y[i] *= v;          }
inline static void wsp_ggml_vec_scale_f32(const int n, float * y, const float   v) {
#if defined(WSP_GGML_USE_ACCELERATE)
    vDSP_vsmul(y, 1, &v, y, 1, n);
#elif defined(WSP_GGML_SIMD)
    const int np = (n & ~(WSP_GGML_F32_STEP - 1));

    WSP_GGML_F32_VEC vx = WSP_GGML_F32_VEC_SET1(v);

    WSP_GGML_F32_VEC ay[WSP_GGML_F32_ARR];

    for (int i = 0; i < np; i += WSP_GGML_F32_STEP) {
        for (int j = 0; j < WSP_GGML_F32_ARR; j++) {
            ay[j] = WSP_GGML_F32_VEC_LOAD(y + i + j*WSP_GGML_F32_EPR);
            ay[j] = WSP_GGML_F32_VEC_MUL(ay[j], vx);

            WSP_GGML_F32_VEC_STORE(y + i + j*WSP_GGML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] *= v;
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] *= v;
    }
#endif
}

inline static void wsp_ggml_vec_scale_f16(const int n, wsp_ggml_fp16_t * y, const float v) {
#if defined(WSP_GGML_SIMD)
    const int np = (n & ~(WSP_GGML_F16_STEP - 1));

    WSP_GGML_F16_VEC vx = WSP_GGML_F16_VEC_SET1(v);

    WSP_GGML_F16_VEC ay[WSP_GGML_F16_ARR];

    for (int i = 0; i < np; i += WSP_GGML_F16_STEP) {
        for (int j = 0; j < WSP_GGML_F16_ARR; j++) {
            ay[j] = WSP_GGML_F16_VEC_LOAD(y + i + j*WSP_GGML_F16_EPR, j);
            ay[j] = WSP_GGML_F16_VEC_MUL(ay[j], vx);

            WSP_GGML_F16_VEC_STORE(y + i + j*WSP_GGML_F16_EPR, ay, j);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] = WSP_GGML_FP32_TO_FP16(WSP_GGML_FP16_TO_FP32(y[i])*v);
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] = WSP_GGML_FP32_TO_FP16(WSP_GGML_FP16_TO_FP32(y[i])*v);
    }
#endif
}

inline static void wsp_ggml_vec_norm_f32 (const int n, float * s, const float * x) { wsp_ggml_vec_dot_f32(n, s, 0, x, 0, x, 0, 1); *s = sqrtf(*s);   }
inline static void wsp_ggml_vec_sqr_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = x[i]*x[i];   }
inline static void wsp_ggml_vec_sqrt_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = sqrtf(x[i]); }
inline static void wsp_ggml_vec_log_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = logf(x[i]);  }
inline static void wsp_ggml_vec_sin_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = sinf(x[i]);  }
inline static void wsp_ggml_vec_cos_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = cosf(x[i]);  }
inline static void wsp_ggml_vec_abs_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = fabsf(x[i]); }
inline static void wsp_ggml_vec_sgn_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : ((x[i] < 0.f) ? -1.f : 0.f); }
inline static void wsp_ggml_vec_step_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : 0.f; }
inline static void wsp_ggml_vec_tanh_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = tanhf(x[i]);  }
inline static void wsp_ggml_vec_elu_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? x[i] : expm1f(x[i]); }
inline static void wsp_ggml_vec_relu_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? x[i] : 0.f; }
inline static void wsp_ggml_vec_leaky_relu_f32 (const int n, float * y, const float * x, const float ns) { for (int i = 0; i < n; ++i) y[i] = ((x[i] > 0.f) ? x[i] : 0.f) + ns * ((x[i] < 0.0f) ? x[i] : 0.f); }
inline static void wsp_ggml_vec_sigmoid_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = 1.f / (1.f + expf(-x[i])); }
// TODO: optimize performance
inline static void wsp_ggml_vec_hardswish_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = x[i] * fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f)); }
inline static void wsp_ggml_vec_hardsigmoid_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f)); }
inline static void wsp_ggml_vec_exp_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = expf(x[i]); }

static const float GELU_COEF_A     = 0.044715f;
static const float GELU_QUICK_COEF = -1.702f;
static const float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

inline static float wsp_ggml_gelu_f32(float x) {
    return 0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

inline static void wsp_ggml_vec_gelu_f16(const int n, wsp_ggml_fp16_t * y, const wsp_ggml_fp16_t * x) {
    const uint16_t * i16 = (const uint16_t *) x;
    for (int i = 0; i < n; ++i) {
        y[i] = wsp_ggml_table_gelu_f16[i16[i]];
    }
}

#ifdef WSP_GGML_GELU_FP16
inline static void wsp_ggml_vec_gelu_f32(const int n, float * y, const float * x) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        if (x[i] <= -10.0f) {
            y[i] = 0.0f;
        } else if (x[i] >= 10.0f) {
            y[i] = x[i];
        } else {
            wsp_ggml_fp16_t fp16 = WSP_GGML_FP32_TO_FP16(x[i]);
            memcpy(&t, &fp16, sizeof(uint16_t));
            y[i] = WSP_GGML_FP16_TO_FP32(wsp_ggml_table_gelu_f16[t]);
        }
    }
}
#else
inline static void wsp_ggml_vec_gelu_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = wsp_ggml_gelu_f32(x[i]);
    }
}
#endif

inline static float wsp_ggml_gelu_quick_f32(float x) {
    return x*(1.0f/(1.0f+expf(GELU_QUICK_COEF*x)));
}

//inline static void wsp_ggml_vec_gelu_quick_f16(const int n, wsp_ggml_fp16_t * y, const wsp_ggml_fp16_t * x) {
//    const uint16_t * i16 = (const uint16_t *) x;
//    for (int i = 0; i < n; ++i) {
//        y[i] = wsp_ggml_table_gelu_quick_f16[i16[i]];
//    }
//}

#ifdef WSP_GGML_GELU_QUICK_FP16
inline static void wsp_ggml_vec_gelu_quick_f32(const int n, float * y, const float * x) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        wsp_ggml_fp16_t fp16 = WSP_GGML_FP32_TO_FP16(x[i]);
        memcpy(&t, &fp16, sizeof(uint16_t));
        y[i] = WSP_GGML_FP16_TO_FP32(wsp_ggml_table_gelu_quick_f16[t]);
    }
}
#else
inline static void wsp_ggml_vec_gelu_quick_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = wsp_ggml_gelu_quick_f32(x[i]);
    }
}
#endif

// Sigmoid Linear Unit (SiLU) function
inline static float wsp_ggml_silu_f32(float x) {
    return x/(1.0f + expf(-x));
}

#if __FINITE_MATH_ONLY__
#error "some routines in ggml.c require non-finite math arithmetics -- pass -fno-finite-math-only to the compiler to fix"
#error "ref: https://github.com/ggerganov/llama.cpp/pull/7154#issuecomment-2143844461"
#endif

#if defined(__ARM_NEON) && defined(__aarch64__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static float32x4_t wsp_ggml_v_expf(float32x4_t x) {
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n,
                                    vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                  vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u), u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c)))
        return vfmaq_f32(k, j, k);
    const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static float32x4_t wsp_ggml_v_silu(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t neg_x = vsubq_f32(zero, x);
    const float32x4_t exp_neg_x = wsp_ggml_v_expf(neg_x);
    const float32x4_t one_plus_exp_neg_x = vaddq_f32(one, exp_neg_x);
    return vdivq_f32(x, one_plus_exp_neg_x);
}

#elif defined(__AVX512F__) && defined(__AVX512DQ__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static __m512 wsp_ggml_v_expf(__m512 x) {
  const __m512 r = _mm512_set1_ps(0x1.8p23f);
  const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
  const __m512 n = _mm512_sub_ps(z, r);
  const __m512 b =
      _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f),
                       _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
  const __mmask16 d =
      _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192), _CMP_GT_OQ);
  const __m512 u = _mm512_mul_ps(b, b);
  const __m512 j = _mm512_fmadd_ps(
      _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b,
                                      _mm512_set1_ps(0x1.573e2ep-5f)),
                      u,
                      _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b,
                                      _mm512_set1_ps(0x1.fffdb6p-2f))),
      u,
      _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0F)));
  const __m512 res = _mm512_scalef_ps(j, n);
  if (_mm512_kortestz(d, d))
    return res;
  const __m512 zero = _mm512_setzero_ps();
  const __m512 alt = _mm512_mask_blend_ps(
      _mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ), _mm512_set1_ps(INFINITY), zero);
  return _mm512_mask_blend_ps(d, res, alt);
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static __m512 wsp_ggml_v_silu(__m512 x) {
    const __m512 one = _mm512_set1_ps(1);
    const __m512 zero = _mm512_setzero_ps();
    const __m512 neg_x = _mm512_sub_ps(zero, x);
    const __m512 exp_neg_x = wsp_ggml_v_expf(neg_x);
    const __m512 one_plus_exp_neg_x = _mm512_add_ps(one, exp_neg_x);
    return _mm512_div_ps(x, one_plus_exp_neg_x);
}

#elif defined(__AVX2__) && defined(__FMA__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static __m256 wsp_ggml_v_expf(__m256 x) {
  const __m256 r = _mm256_set1_ps(0x1.8p23f);
  const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
  const __m256 n = _mm256_sub_ps(z, r);
  const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                    _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
  const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
  const __m256 k = _mm256_castsi256_ps(
      _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
  const __m256i c = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(126), _CMP_GT_OQ));
  const __m256 u = _mm256_mul_ps(b, b);
  const __m256 j = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                                                                   _mm256_set1_ps(0x1.573e2ep-5f)), u,
                                                   _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                                                                   _mm256_set1_ps(0x1.fffdb6p-2f))),
                                   u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
  if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
    return _mm256_fmadd_ps(j, k, k);
  const __m256i g = _mm256_and_si256(
      _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
      _mm256_set1_epi32(0x82000000u));
  const __m256 s1 =
      _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
  const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
  const __m256i d = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(192), _CMP_GT_OQ));
  return _mm256_or_ps(
      _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
      _mm256_andnot_ps(
          _mm256_castsi256_ps(d),
          _mm256_or_ps(
              _mm256_and_ps(_mm256_castsi256_ps(c),
                            _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
              _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static __m256 wsp_ggml_v_silu(__m256 x) {
    const __m256 one = _mm256_set1_ps(1);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 neg_x = _mm256_sub_ps(zero, x);
    const __m256 exp_neg_x = wsp_ggml_v_expf(neg_x);
    const __m256 one_plus_exp_neg_x = _mm256_add_ps(one, exp_neg_x);
    return _mm256_div_ps(x, one_plus_exp_neg_x);
}

#elif defined(__SSE2__) // __AVX2__ / __ARM_NEON

#if defined(__FMA__)
#define MADD128(x, y, z) _mm_fmadd_ps(x, y, z)
#define NMADD128(x, y, z) _mm_fnmadd_ps(x, y, z)
#else
#define MADD128(x, y, z) _mm_add_ps(_mm_mul_ps(x, y), z)
#define NMADD128(x, y, z) _mm_sub_ps(z, _mm_mul_ps(x, y))
#endif

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static __m128 wsp_ggml_v_expf(__m128 x) {
    const __m128 r = _mm_set1_ps(0x1.8p23f);
    const __m128 z = MADD128(x, _mm_set1_ps(0x1.715476p+0f), r);
    const __m128 n = _mm_sub_ps(z, r);
    const __m128 b =
        NMADD128(n, _mm_set1_ps(0x1.7f7d1cp-20f), NMADD128(n, _mm_set1_ps(0x1.62e4p-1f), x));
    const __m128i e = _mm_slli_epi32(_mm_castps_si128(z), 23);
    const __m128 k = _mm_castsi128_ps(_mm_add_epi32(e, _mm_castps_si128(_mm_set1_ps(1))));
    const __m128i c =
        _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(126)));
    const __m128 u = _mm_mul_ps(b, b);
    const __m128 j =
        MADD128(MADD128(MADD128(_mm_set1_ps(0x1.0e4020p-7f), b, _mm_set1_ps(0x1.573e2ep-5f)), u,
                        MADD128(_mm_set1_ps(0x1.555e66p-3f), b, _mm_set1_ps(0x1.fffdb6p-2f))),
                u, _mm_mul_ps(_mm_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm_movemask_epi8(c))
        return MADD128(j, k, k);
    const __m128i g = _mm_and_si128(_mm_castps_si128(_mm_cmple_ps(n, _mm_setzero_ps())),
                                    _mm_set1_epi32(0x82000000u));
    const __m128 s1 = _mm_castsi128_ps(_mm_add_epi32(g, _mm_set1_epi32(0x7f000000u)));
    const __m128 s2 = _mm_castsi128_ps(_mm_sub_epi32(e, g));
    const __m128i d =
        _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(192)));
    return _mm_or_ps(
        _mm_and_ps(_mm_castsi128_ps(d), _mm_mul_ps(s1, s1)),
        _mm_andnot_ps(_mm_castsi128_ps(d),
                      _mm_or_ps(_mm_and_ps(_mm_castsi128_ps(c), _mm_mul_ps(MADD128(s2, j, s2), s1)),
                                _mm_andnot_ps(_mm_castsi128_ps(c), MADD128(k, j, k)))));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static __m128 wsp_ggml_v_silu(__m128 x) {
    const __m128 one = _mm_set1_ps(1);
    const __m128 zero = _mm_setzero_ps();
    const __m128 neg_x = _mm_sub_ps(zero, x);
    const __m128 exp_neg_x = wsp_ggml_v_expf(neg_x);
    const __m128 one_plus_exp_neg_x = _mm_add_ps(one, exp_neg_x);
    return _mm_div_ps(x, one_plus_exp_neg_x);
}

#endif // __ARM_NEON / __AVX2__ / __SSE2__

static void wsp_ggml_vec_silu_f32(const int n, float * y, const float * x) {
    int i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i, wsp_ggml_v_silu(_mm512_loadu_ps(x + i)));
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, wsp_ggml_v_silu(_mm256_loadu_ps(x + i)));
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        _mm_storeu_ps(y + i, wsp_ggml_v_silu(_mm_loadu_ps(x + i)));
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        vst1q_f32(y + i, wsp_ggml_v_silu(vld1q_f32(x + i)));
    }
#endif
    for (; i < n; ++i) {
        y[i] = wsp_ggml_silu_f32(x[i]);
    }
}

static wsp_ggml_float wsp_ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max) {
    int i = 0;
    wsp_ggml_float sum = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        __m512 val = wsp_ggml_v_expf(_mm512_sub_ps(_mm512_loadu_ps(x + i),
                                               _mm512_set1_ps(max)));
        _mm512_storeu_ps(y + i, val);
        sum += (wsp_ggml_float)_mm512_reduce_add_ps(val);
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        __m256 val = wsp_ggml_v_expf(_mm256_sub_ps(_mm256_loadu_ps(x + i),
                                               _mm256_set1_ps(max)));
        _mm256_storeu_ps(y + i, val);
        __m128 val2 = _mm_add_ps(_mm256_extractf128_ps(val, 1),
                                 _mm256_castps256_ps128(val));
        val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
        val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
        sum += (wsp_ggml_float)_mm_cvtss_f32(val2);
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        __m128 val = wsp_ggml_v_expf(_mm_sub_ps(_mm_loadu_ps(x + i),
                                            _mm_set1_ps(max)));
        _mm_storeu_ps(y + i, val);
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
        val = _mm_add_ps(val, _mm_movehl_ps(val, val));
        val = _mm_add_ss(val, _mm_movehdup_ps(val));
#else
        __m128 tmp = _mm_shuffle_ps(val, val, _MM_SHUFFLE(2, 3, 0, 1));
        val = _mm_add_ps(val, tmp);
        tmp = _mm_movehl_ps(tmp, val);
        val = _mm_add_ss(val, tmp);
#endif
        sum += (wsp_ggml_float)_mm_cvtss_f32(val);
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        float32x4_t val = wsp_ggml_v_expf(vsubq_f32(vld1q_f32(x + i),
                                                vdupq_n_f32(max)));
        vst1q_f32(y + i, val);
        sum += (wsp_ggml_float)vaddvq_f32(val);
    }
#endif
    for (; i < n; ++i) {
        float val = expf(x[i] - max);
        sum += (wsp_ggml_float)val;
        y[i] = val;
    }
    return sum;
}

static wsp_ggml_float wsp_ggml_vec_log_soft_max_f32(const int n, float * y, const float * x, float max) {
    // log(soft_max) = log(soft_max_i / soft_max_sum) = log(soft_max_i) - log(soft_max_sum) = (logit_i - max) - log(soft_max_i)

    int i = 0;
    wsp_ggml_float sum = 0;
    for (; i < n; ++i) {
        float val = x[i] - max;
        y[i] = val;
        sum += (wsp_ggml_float)expf(val);
    }
    return sum = (wsp_ggml_float)logf(sum);
}

inline static float wsp_ggml_silu_backward_f32(float x, float dy) {
    const float s = 1.0f/(1.0f + expf(-x));
    return dy*s*(1.0f + x*(1.0f - s));
}

inline static void wsp_ggml_vec_silu_backward_f32(const int n, float * dx, const float * x, const float * dy) {
    for (int i = 0; i < n; ++i) {
        dx[i] = wsp_ggml_silu_backward_f32(x[i], dy[i]);
    }
}

inline static void wsp_ggml_vec_sum_f32(const int n, float * s, const float * x) {
#ifndef WSP_GGML_USE_ACCELERATE
    wsp_ggml_float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (wsp_ggml_float)x[i];
    }
    *s = sum;
#else
    vDSP_sve(x, 1, s, n);
#endif
}

inline static void wsp_ggml_vec_sum_f32_ggf(const int n, wsp_ggml_float * s, const float * x) {
    wsp_ggml_float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (wsp_ggml_float)x[i];
    }
    *s = sum;
}

inline static void wsp_ggml_vec_sum_f16_ggf(const int n, float * s, const wsp_ggml_fp16_t * x) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += WSP_GGML_FP16_TO_FP32(x[i]);
    }
    *s = sum;
}

inline static void wsp_ggml_vec_sum_bf16_ggf(const int n, float * s, const wsp_ggml_bf16_t * x) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += WSP_GGML_BF16_TO_FP32(x[i]);
    }
    *s = sum;
}

inline static void wsp_ggml_vec_max_f32(const int n, float * s, const float * x) {
#ifndef WSP_GGML_USE_ACCELERATE
    float max = -INFINITY;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
    }
    *s = max;
#else
    vDSP_maxv(x, 1, s, n);
#endif
}

inline static void wsp_ggml_vec_norm_inv_f32(const int n, float * s, const float * x) {
    wsp_ggml_vec_norm_f32(n, s, x);
    *s = 1.f/(*s);
}

inline static void wsp_ggml_vec_argmax_f32(const int n, int * s, const float * x) {
    float max = -INFINITY;
    int idx = 0;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
        if (max == x[i]) { idx = i; }
    }
    *s = idx;
}

// Helpers for polling loops
#if defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
static inline void wsp_ggml_thread_cpu_relax(void) {
    __asm__ volatile("yield" ::: "memory");
}
#elif defined(__x86_64__)
static inline void wsp_ggml_thread_cpu_relax(void) {
    _mm_pause();
}
#else
static inline void wsp_ggml_thread_cpu_relax(void) {;}
#endif

//
// NUMA support
//

#define WSP_GGML_NUMA_MAX_NODES 8
#define WSP_GGML_NUMA_MAX_CPUS 512

struct wsp_ggml_numa_node {
    uint32_t cpus[WSP_GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

struct wsp_ggml_numa_nodes {
    enum wsp_ggml_numa_strategy numa_strategy;
    struct wsp_ggml_numa_node nodes[WSP_GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_node; // node on which main process is execting
#if defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#else
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

struct wsp_ggml_state {
    struct wsp_ggml_numa_nodes numa;
};

// global state
static struct wsp_ggml_state g_state = {0};
static atomic_flag g_state_critical = ATOMIC_FLAG_INIT;

// TODO: move to threading file
// critical section via spin lock
void wsp_ggml_critical_section_start(void) {
    while (atomic_flag_test_and_set(&g_state_critical)) {
        // spin
        sched_yield();
    }
}

void wsp_ggml_critical_section_end(void) {
    atomic_flag_clear(&g_state_critical);
}

static void wsp_ggml_barrier(struct wsp_ggml_threadpool * tp) {
    int n_threads = atomic_load_explicit(&tp->n_threads_cur, memory_order_relaxed);
    if (n_threads == 1) {
        return;
    }

#ifdef WSP_GGML_USE_OPENMP
    #pragma omp barrier
#else
    int n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    int n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    if (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (fill seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        return;
    }

    // wait for other threads
    while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        wsp_ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef WSP_GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}

#if defined(__gnu_linux__)
static cpu_set_t wsp_ggml_get_numa_affinity(void) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    return cpuset;
}
#else
static uint32_t wsp_ggml_get_numa_affinity(void) {
    return 0; // no NUMA support
}
#endif

void wsp_ggml_numa_init(enum wsp_ggml_numa_strategy numa_flag) {
    if (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "wsp_ggml_numa_init: NUMA already initialized\n");

        return;
    }

#if defined(__gnu_linux__)
    struct stat st;
    char path[256];
    int rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    WSP_GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = wsp_ggml_get_numa_affinity();

    // enumerate nodes
    while (g_state.numa.n_nodes < WSP_GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        WSP_GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    while (g_state.numa.total_cpus < WSP_GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        WSP_GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.total_cpus;
    }

    WSP_GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    int getcpu_ret = 0;
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 28) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_node);
#else
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   if !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_node);
#endif

    if (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        return;
    }

    WSP_GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_node, current_cpu);

    for (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        struct wsp_ggml_numa_node * node = &g_state.numa.nodes[n];
        WSP_GGML_PRINT_DEBUG("CPUs on node %u:", n);
        node->n_cpus = 0;
        for (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            WSP_GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
            if (stat(path, &st) == 0) {
                node->cpus[node->n_cpus++] = c;
                WSP_GGML_PRINT_DEBUG(" %u", c);
            }
        }
        WSP_GGML_PRINT_DEBUG("\n");
    }

    if (wsp_ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        if (fptr != NULL) {
            char buf[42];
            if (fgets(buf, sizeof(buf), fptr) && strncmp(buf, "0\n", sizeof(buf)) != 0) {
                WSP_GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#else
    UNUSED(numa_flag);
    // TODO
#endif
}

bool wsp_ggml_is_numa(void) {
    return g_state.numa.n_nodes > 1;
}

#if defined(__ARM_ARCH)

#if defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if !defined(HWCAP2_I8MM)
#define HWCAP2_I8MM 0
#endif

static void wsp_ggml_init_arm_arch_features(void) {
#if defined(__linux__) && defined(__aarch64__)
    uint32_t hwcap = getauxval(AT_HWCAP);
    uint32_t hwcap2 = getauxval(AT_HWCAP2);

    wsp_ggml_arm_arch_features.has_neon = !!(hwcap & HWCAP_ASIMD);
    wsp_ggml_arm_arch_features.has_i8mm = !!(hwcap2 & HWCAP2_I8MM);
    wsp_ggml_arm_arch_features.has_sve  = !!(hwcap & HWCAP_SVE);

#if defined(__ARM_FEATURE_SVE)
    wsp_ggml_arm_arch_features.sve_cnt = PR_SVE_VL_LEN_MASK & prctl(PR_SVE_GET_VL);
#endif
#elif defined(__APPLE__)
    int oldp = 0;
    size_t size = sizeof(oldp);
    if (sysctlbyname("hw.optional.AdvSIMD", &oldp, &size, NULL, 0) != 0) {
        oldp = 0;
    }
    wsp_ggml_arm_arch_features.has_neon = oldp;

    if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &oldp, &size, NULL, 0) != 0) {
        oldp = 0;
    }
    wsp_ggml_arm_arch_features.has_i8mm = oldp;

    wsp_ggml_arm_arch_features.has_sve = 0;
    wsp_ggml_arm_arch_features.sve_cnt = 0;
#else
// Run-time CPU feature detection not implemented for this platform, fallback to compile time
#if defined(__ARM_NEON)
    wsp_ggml_arm_arch_features.has_neon = 1;
#else
    wsp_ggml_arm_arch_features.has_neon = 0;
#endif

#if defined(__ARM_FEATURE_MATMUL_INT8)
    wsp_ggml_arm_arch_features.has_i8mm = 1;
#else
    wsp_ggml_arm_arch_features.has_i8mm = 0;
#endif

#if defined(__ARM_FEATURE_SVE)
    wsp_ggml_arm_arch_features.has_sve = 1;
    wsp_ggml_arm_arch_features.sve_cnt = 16;
#else
    wsp_ggml_arm_arch_features.has_sve = 0;
    wsp_ggml_arm_arch_features.sve_cnt = 0;
#endif
#endif
}
#endif

struct wsp_ggml_tensor * wsp_ggml_new_i32(struct wsp_ggml_context * ctx, int32_t value) {
    WSP_GGML_ASSERT(!wsp_ggml_get_no_alloc(ctx));

    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_1d(ctx, WSP_GGML_TYPE_I32, 1);

    wsp_ggml_set_i32(result, value);

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_new_f32(struct wsp_ggml_context * ctx, float value) {
    WSP_GGML_ASSERT(!wsp_ggml_get_no_alloc(ctx));

    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_1d(ctx, WSP_GGML_TYPE_F32, 1);

    wsp_ggml_set_f32(result, value);

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_set_i32 (struct wsp_ggml_tensor * tensor, int32_t value) {
    const int n     = wsp_ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case WSP_GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    wsp_ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case WSP_GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    wsp_ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case WSP_GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    wsp_ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case WSP_GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(wsp_ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    wsp_ggml_vec_set_f16(nc, (wsp_ggml_fp16_t *)(data + i*n1), WSP_GGML_FP32_TO_FP16(value));
                }
            } break;
        case WSP_GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(wsp_ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    wsp_ggml_vec_set_bf16(nc, (wsp_ggml_bf16_t *)(data + i*n1), WSP_GGML_FP32_TO_BF16(value));
                }
            } break;
        case WSP_GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    wsp_ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

struct wsp_ggml_tensor * wsp_ggml_set_f32(struct wsp_ggml_tensor * tensor, float value) {
    const int n     = wsp_ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case WSP_GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    wsp_ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case WSP_GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    wsp_ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case WSP_GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    wsp_ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case WSP_GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(wsp_ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    wsp_ggml_vec_set_f16(nc, (wsp_ggml_fp16_t *)(data + i*n1), WSP_GGML_FP32_TO_FP16(value));
                }
            } break;
        case WSP_GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(wsp_ggml_bf16_t));
                for (int i = 0; i < n; i++) {
                    wsp_ggml_vec_set_bf16(nc, (wsp_ggml_bf16_t *)(data + i*n1), WSP_GGML_FP32_TO_BF16(value));
                }
            } break;
        case WSP_GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    wsp_ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

int32_t wsp_ggml_get_i32_1d(const struct wsp_ggml_tensor * tensor, int i) {
    if (!wsp_ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        wsp_ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return wsp_ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case WSP_GGML_TYPE_I8:
            {
                WSP_GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                return ((int8_t *)(tensor->data))[i];
            }
        case WSP_GGML_TYPE_I16:
            {
                WSP_GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                return ((int16_t *)(tensor->data))[i];
            }
        case WSP_GGML_TYPE_I32:
            {
                WSP_GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                return ((int32_t *)(tensor->data))[i];
            }
        case WSP_GGML_TYPE_F16:
            {
                WSP_GGML_ASSERT(tensor->nb[0] == sizeof(wsp_ggml_fp16_t));
                return WSP_GGML_FP16_TO_FP32(((wsp_ggml_fp16_t *)(tensor->data))[i]);
            }
        case WSP_GGML_TYPE_BF16:
            {
                WSP_GGML_ASSERT(tensor->nb[0] == sizeof(wsp_ggml_bf16_t));
                return WSP_GGML_BF16_TO_FP32(((wsp_ggml_bf16_t *)(tensor->data))[i]);
            }
        case WSP_GGML_TYPE_F32:
            {
                WSP_GGML_ASSERT(tensor->nb[0] == sizeof(float));
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

void wsp_ggml_set_i32_1d(const struct wsp_ggml_tensor * tensor, int i, int32_t value) {
    if (!wsp_ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        wsp_ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        wsp_ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case WSP_GGML_TYPE_I8:
            {
                WSP_GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case WSP_GGML_TYPE_I16:
            {
                WSP_GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case WSP_GGML_TYPE_I32:
            {
                WSP_GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case WSP_GGML_TYPE_F16:
            {
                WSP_GGML_ASSERT(tensor->nb[0] == sizeof(wsp_ggml_fp16_t));
                ((wsp_ggml_fp16_t *)(tensor->data))[i] = WSP_GGML_FP32_TO_FP16(value);
            } break;
        case WSP_GGML_TYPE_BF16:
            {
                WSP_GGML_ASSERT(tensor->nb[0] == sizeof(wsp_ggml_bf16_t));
                ((wsp_ggml_bf16_t *)(tensor->data))[i] = WSP_GGML_FP32_TO_BF16(value);
            } break;
        case WSP_GGML_TYPE_F32:
            {
                WSP_GGML_ASSERT(tensor->nb[0] == sizeof(float));
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

int32_t wsp_ggml_get_i32_nd(const struct wsp_ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case WSP_GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case WSP_GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case WSP_GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case WSP_GGML_TYPE_F16:
            return WSP_GGML_FP16_TO_FP32(((wsp_ggml_fp16_t *) data)[0]);
        case WSP_GGML_TYPE_BF16:
            return WSP_GGML_BF16_TO_FP32(((wsp_ggml_bf16_t *) data)[0]);
        case WSP_GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            WSP_GGML_ABORT("fatal error");
    }
}

void wsp_ggml_set_i32_nd(const struct wsp_ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case WSP_GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case WSP_GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case WSP_GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case WSP_GGML_TYPE_F16:
            {
                ((wsp_ggml_fp16_t *)(data))[0] = WSP_GGML_FP32_TO_FP16(value);
            } break;
        case WSP_GGML_TYPE_BF16:
            {
                ((wsp_ggml_bf16_t *)(data))[0] = WSP_GGML_FP32_TO_BF16(value);
            } break;
        case WSP_GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

float wsp_ggml_get_f32_1d(const struct wsp_ggml_tensor * tensor, int i) {
    if (!wsp_ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        wsp_ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return wsp_ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case WSP_GGML_TYPE_I8:
            {
                return ((int8_t *)(tensor->data))[i];
            }
        case WSP_GGML_TYPE_I16:
            {
                return ((int16_t *)(tensor->data))[i];
            }
        case WSP_GGML_TYPE_I32:
            {
                return ((int32_t *)(tensor->data))[i];
            }
        case WSP_GGML_TYPE_F16:
            {
                return WSP_GGML_FP16_TO_FP32(((wsp_ggml_fp16_t *)(tensor->data))[i]);
            }
        case WSP_GGML_TYPE_BF16:
            {
                return WSP_GGML_BF16_TO_FP32(((wsp_ggml_bf16_t *)(tensor->data))[i]);
            }
        case WSP_GGML_TYPE_F32:
            {
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

void wsp_ggml_set_f32_1d(const struct wsp_ggml_tensor * tensor, int i, float value) {
    if (!wsp_ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        wsp_ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        wsp_ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case WSP_GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case WSP_GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case WSP_GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case WSP_GGML_TYPE_F16:
            {
                ((wsp_ggml_fp16_t *)(tensor->data))[i] = WSP_GGML_FP32_TO_FP16(value);
            } break;
        case WSP_GGML_TYPE_BF16:
            {
                ((wsp_ggml_bf16_t *)(tensor->data))[i] = WSP_GGML_FP32_TO_BF16(value);
            } break;
        case WSP_GGML_TYPE_F32:
            {
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

float wsp_ggml_get_f32_nd(const struct wsp_ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case WSP_GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case WSP_GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case WSP_GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case WSP_GGML_TYPE_F16:
            return WSP_GGML_FP16_TO_FP32(((wsp_ggml_fp16_t *) data)[0]);
        case WSP_GGML_TYPE_BF16:
            return WSP_GGML_BF16_TO_FP32(((wsp_ggml_bf16_t *) data)[0]);
        case WSP_GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            WSP_GGML_ABORT("fatal error");
    }
}

void wsp_ggml_set_f32_nd(const struct wsp_ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case WSP_GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case WSP_GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case WSP_GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case WSP_GGML_TYPE_F16:
            {
                ((wsp_ggml_fp16_t *)(data))[0] = WSP_GGML_FP32_TO_FP16(value);
            } break;
        case WSP_GGML_TYPE_BF16:
            {
                ((wsp_ggml_bf16_t *)(data))[0] = WSP_GGML_FP32_TO_BF16(value);
            } break;
        case WSP_GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// wsp_ggml_compute_forward_dup

static void wsp_ggml_compute_forward_dup_same_cont(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    WSP_GGML_ASSERT(wsp_ggml_nelements(dst) == wsp_ggml_nelements(src0));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(dst) && wsp_ggml_is_contiguous(src0));
    WSP_GGML_ASSERT(src0->type == dst->type);

    const size_t nb0 = wsp_ggml_type_size(src0->type);

    const int ith = params->ith; // thread index
    const int nth = params->nth; // number of threads

    // parallelize by elements
    const int ne = wsp_ggml_nelements(dst);
    const int dr = (ne + nth - 1) / nth;
    const int ie0 = dr * ith;
    const int ie1 = MIN(ie0 + dr, ne);

    if (ie0 < ie1) {
        memcpy(
            ((char *)  dst->data + ie0*nb0),
            ((char *) src0->data + ie0*nb0),
            (ie1 - ie0) * nb0);
    }
}

static void wsp_ggml_compute_forward_dup_f16(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    WSP_GGML_ASSERT(wsp_ggml_nelements(dst) == wsp_ggml_nelements(src0));

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    const int ith = params->ith; // thread index
    const int nth = params->nth; // number of threads

    // parallelize by rows
    const int nr = ne01;
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == wsp_ggml_type_size(src0->type) && nb0 == wsp_ggml_type_size(dst->type)) {
        // copy by rows
        const size_t rs = ne00*nb00;
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        return;
    }

    // TODO: add more special-case implementations for tensor shapes/strides that can benefit from memcpy

    if (wsp_ggml_is_contiguous(dst)) {
        if (nb00 == sizeof(wsp_ggml_fp16_t)) {
            if (dst->type == WSP_GGML_TYPE_F16) {
                size_t id = 0;
                const size_t rs = ne00 * nb00;
                char * dst_ptr = (char *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const char * src0_ptr = (char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } else if (dst->type == WSP_GGML_TYPE_F32) {
                size_t id = 0;
                float * dst_ptr = (float *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const wsp_ggml_fp16_t * src0_ptr = (wsp_ggml_fp16_t *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                            for (int i00 = 0; i00 < ne00; i00++) {
                                dst_ptr[id] = WSP_GGML_FP16_TO_FP32(src0_ptr[i00]);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else if (wsp_ggml_get_type_traits(dst->type)->from_float) {
                wsp_ggml_from_float_t const wsp_quantize_row_q = wsp_ggml_get_type_traits(dst->type)->from_float;
                float * src0_f32 = (float *) params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

                size_t id = 0;
                size_t rs = nb0 * (ne00 / wsp_ggml_blck_size(dst->type));
                char * dst_ptr = (char *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const wsp_ggml_fp16_t * src0_ptr = (wsp_ggml_fp16_t *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                            for (int i00 = 0; i00 < ne00; i00++) {
                                src0_f32[i00] = WSP_GGML_FP16_TO_FP32(src0_ptr[i00]);
                            }

                            wsp_quantize_row_q(src0_f32, dst_ptr + id, ne00);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } else {
                WSP_GGML_ABORT("fatal error"); // TODO: implement
            }
        } else {
            //printf("%s: this is not optimal - fix me\n", __func__);

            if (dst->type == WSP_GGML_TYPE_F32) {
                size_t id = 0;
                float * dst_ptr = (float *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const wsp_ggml_fp16_t * src0_ptr = (wsp_ggml_fp16_t *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                                dst_ptr[id] = WSP_GGML_FP16_TO_FP32(*src0_ptr);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else if (dst->type == WSP_GGML_TYPE_F16) {
                size_t id = 0;
                wsp_ggml_fp16_t * dst_ptr = (wsp_ggml_fp16_t *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const wsp_ggml_fp16_t * src0_ptr = (wsp_ggml_fp16_t *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                                dst_ptr[id] = *src0_ptr;
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else {
                WSP_GGML_ABORT("fatal error"); // TODO: implement
            }
        }
        return;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    if (dst->type == WSP_GGML_TYPE_F16) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        memcpy(dst_ptr, src0_ptr, sizeof(wsp_ggml_fp16_t));

                        if (++i10 == ne00) {
                            i10 = 0;
                            if (++i11 == ne01) {
                                i11 = 0;
                                if (++i12 == ne02) {
                                    i12 = 0;
                                    if (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    } else if (dst->type == WSP_GGML_TYPE_F32) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        *(float *) dst_ptr = WSP_GGML_FP16_TO_FP32(*(const wsp_ggml_fp16_t *) src0_ptr);

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    } else {
        WSP_GGML_ABORT("fatal error"); // TODO: implement
    }
}

static void wsp_ggml_compute_forward_dup_bf16(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    WSP_GGML_ASSERT(wsp_ggml_nelements(dst) == wsp_ggml_nelements(src0));

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    const int ith = params->ith; // thread index
    const int nth = params->nth; // number of threads

    // parallelize by rows
    const int nr = ne01;
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == wsp_ggml_type_size(src0->type) && nb0 == wsp_ggml_type_size(dst->type)) {
        // copy by rows
        const size_t rs = ne00*nb00;
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        return;
    }

    // TODO: add more special-case implementations for tensor shapes/strides that can benefit from memcpy

    if (wsp_ggml_is_contiguous(dst)) {
        if (nb00 == sizeof(wsp_ggml_bf16_t)) {
            if (dst->type == WSP_GGML_TYPE_BF16) {
                size_t id = 0;
                const size_t rs = ne00 * nb00;
                char * dst_ptr = (char *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const char * src0_ptr = (char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } else if (dst->type == WSP_GGML_TYPE_F16) {
                size_t id = 0;
                wsp_ggml_fp16_t * dst_ptr = (wsp_ggml_fp16_t *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const wsp_ggml_bf16_t * src0_ptr = (wsp_ggml_bf16_t *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                            for (int i00 = 0; i00 < ne00; i00++) {
                                dst_ptr[id] = WSP_GGML_FP32_TO_FP16(WSP_GGML_BF16_TO_FP32(src0_ptr[i00]));
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else if (dst->type == WSP_GGML_TYPE_F32) {
                size_t id = 0;
                float * dst_ptr = (float *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const wsp_ggml_bf16_t * src0_ptr = (wsp_ggml_bf16_t *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                            for (int i00 = 0; i00 < ne00; i00++) {
                                dst_ptr[id] = WSP_GGML_BF16_TO_FP32(src0_ptr[i00]);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else if (wsp_ggml_get_type_traits(dst->type)->from_float) {
                wsp_ggml_from_float_t const wsp_quantize_row_q = wsp_ggml_get_type_traits(dst->type)->from_float;
                float * src0_f32 = (float *) params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

                size_t id = 0;
                size_t rs = nb0 * (ne00 / wsp_ggml_blck_size(dst->type));
                char * dst_ptr = (char *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const wsp_ggml_bf16_t * src0_ptr = (wsp_ggml_bf16_t *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                            for (int i00 = 0; i00 < ne00; i00++) {
                                src0_f32[i00] = WSP_GGML_BF16_TO_FP32(src0_ptr[i00]);
                            }

                            wsp_quantize_row_q(src0_f32, dst_ptr + id, ne00);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } else {
                WSP_GGML_ABORT("fatal error"); // TODO: implement
            }
        } else {
            //printf("%s: this is not optimal - fix me\n", __func__);

            if (dst->type == WSP_GGML_TYPE_F32) {
                size_t id = 0;
                float * dst_ptr = (float *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const wsp_ggml_bf16_t * src0_ptr = (wsp_ggml_bf16_t *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                                dst_ptr[id] = WSP_GGML_BF16_TO_FP32(*src0_ptr);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else if (dst->type == WSP_GGML_TYPE_BF16) {
                size_t id = 0;
                wsp_ggml_bf16_t * dst_ptr = (wsp_ggml_bf16_t *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const wsp_ggml_bf16_t * src0_ptr = (wsp_ggml_bf16_t *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                                dst_ptr[id] = *src0_ptr;
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else if (dst->type == WSP_GGML_TYPE_F16) {
                size_t id = 0;
                wsp_ggml_fp16_t * dst_ptr = (wsp_ggml_fp16_t *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const wsp_ggml_bf16_t * src0_ptr = (wsp_ggml_bf16_t *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                                dst_ptr[id] = WSP_GGML_FP32_TO_FP16(WSP_GGML_BF16_TO_FP32(*src0_ptr));
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else {
                WSP_GGML_ABORT("fatal error"); // TODO: implement
            }
        }
        return;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    if (dst->type == WSP_GGML_TYPE_BF16) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        memcpy(dst_ptr, src0_ptr, sizeof(wsp_ggml_bf16_t));

                        if (++i10 == ne00) {
                            i10 = 0;
                            if (++i11 == ne01) {
                                i11 = 0;
                                if (++i12 == ne02) {
                                    i12 = 0;
                                    if (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    } else if (dst->type == WSP_GGML_TYPE_F16) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        *(wsp_ggml_fp16_t *) dst_ptr = WSP_GGML_FP32_TO_FP16(WSP_GGML_BF16_TO_FP32(*(const wsp_ggml_bf16_t *) src0_ptr));

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    } else if (dst->type == WSP_GGML_TYPE_F32) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        *(float *) dst_ptr = WSP_GGML_BF16_TO_FP32(*(const wsp_ggml_bf16_t *) src0_ptr);

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    } else {
        WSP_GGML_ABORT("fatal error"); // TODO: implement
    }
}

static void wsp_ggml_compute_forward_dup_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    WSP_GGML_ASSERT(wsp_ggml_nelements(dst) == wsp_ggml_nelements(src0));

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    const int ith = params->ith; // thread index
    const int nth = params->nth; // number of threads

    // parallelize by rows
    const int nr = ne01;
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == wsp_ggml_type_size(src0->type) && nb0 == wsp_ggml_type_size(dst->type)) {
        // copy by rows
        const size_t rs = ne00*nb00;
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        return;
    }

    if (wsp_ggml_is_contiguous(dst)) {
        // TODO: simplify
        if (nb00 == sizeof(float)) {
            if (dst->type == WSP_GGML_TYPE_F32) {
                size_t id = 0;
                const size_t rs = ne00 * nb00;
                char * dst_ptr = (char *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const char * src0_ptr = (char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } else if (wsp_ggml_get_type_traits(dst->type)->from_float) {
                wsp_ggml_from_float_t const wsp_quantize_row_q = wsp_ggml_get_type_traits(dst->type)->from_float;

                size_t id = 0;
                size_t rs = nb0 * (ne00 / wsp_ggml_blck_size(dst->type));
                char * dst_ptr = (char *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const float * src0_ptr = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                            wsp_quantize_row_q(src0_ptr, dst_ptr + id, ne00);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } else {
                WSP_GGML_ABORT("fatal error"); // TODO: implement
            }
        } else {
            //printf("%s: this is not optimal - fix me\n", __func__);

            if (dst->type == WSP_GGML_TYPE_F32) {
                size_t id = 0;
                float * dst_ptr = (float *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const float * src0_ptr = (float *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                                dst_ptr[id] = *src0_ptr;
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else if (dst->type == WSP_GGML_TYPE_F16) {
                size_t id = 0;
                wsp_ggml_fp16_t * dst_ptr = (wsp_ggml_fp16_t *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const float * src0_ptr = (float *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                                dst_ptr[id] = WSP_GGML_FP32_TO_FP16(*src0_ptr);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else if (dst->type == WSP_GGML_TYPE_BF16) {
                size_t id = 0;
                wsp_ggml_bf16_t * dst_ptr = (wsp_ggml_bf16_t *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const float * src0_ptr = (float *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                                dst_ptr[id] = WSP_GGML_FP32_TO_BF16(*src0_ptr);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else {
                WSP_GGML_ABORT("fatal error"); // TODO: implement
            }
        }

        return;
    }

    // dst counters

    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    if (dst->type == WSP_GGML_TYPE_F32) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        memcpy(dst_ptr, src0_ptr, sizeof(float));

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    } else if (dst->type == WSP_GGML_TYPE_F16) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        *(wsp_ggml_fp16_t *) dst_ptr = WSP_GGML_FP32_TO_FP16(*(const float *) src0_ptr);

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    } else if (dst->type == WSP_GGML_TYPE_BF16) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        *(wsp_ggml_bf16_t *) dst_ptr = WSP_GGML_FP32_TO_BF16(*(const float *) src0_ptr);

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    } else {
        WSP_GGML_ABORT("fatal error"); // TODO: implement
    }
}

// A simplified version of wsp_ggml_compute_forward_dup that doesn't do float upcasting, and just plain old memcpy.
static void wsp_ggml_compute_forward_dup_bytes(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    WSP_GGML_ASSERT(wsp_ggml_nelements(dst) == wsp_ggml_nelements(src0));
    WSP_GGML_ASSERT(src0->type == dst->type);

    WSP_GGML_TENSOR_UNARY_OP_LOCALS;

    if (wsp_ggml_is_contiguous(src0) && wsp_ggml_is_contiguous(dst)) {
        wsp_ggml_compute_forward_dup_same_cont(params, dst);
        return;
    }

    const size_t type_size = wsp_ggml_type_size(src0->type);
    const int ith = params->ith; // thread index
    const int nth = params->nth; // number of threads


    // parallelize by rows
    const int nr = ne01;
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == type_size && nb0 == type_size) {
        // copy by rows
        const size_t rs = ne00 * type_size;
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        return;
    }

    if (wsp_ggml_is_contiguous(dst)) {
        size_t id = 0;
        char * dst_ptr = (char *) dst->data;
        const size_t rs = ne00 * type_size;

        if (nb00 == type_size) {
            // src0 is contigous on first dimension, copy by rows
            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    for (int64_t i01 = ir0; i01 < ir1; i01++) {
                        const char * src0_ptr = (char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                        memcpy(dst_ptr + id, src0_ptr, rs);
                        id += rs;
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        } else {
            //printf("%s: this is not optimal - fix me\n", __func__);

            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    for (int64_t i01 = ir0; i01 < ir1; i01++) {
                        for (int64_t i00 = 0; i00 < ne00; i00++) {
                            const char * src0_ptr = (char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, type_size);

                            id += type_size;
                        }
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        }

        return;
    }

    // dst counters

    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            i10 += ne00 * ir0;
            while (i10 >= ne0) {
                i10 -= ne0;
                if (++i11 == ne1) {
                    i11 = 0;
                    if (++i12 == ne2) {
                        i12 = 0;
                        if (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
            for (int64_t i01 = ir0; i01 < ir1; i01++) {
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                    memcpy(dst_ptr, src0_ptr, type_size);

                    if (++i10 == ne0) {
                        i10 = 0;
                        if (++i11 == ne1) {
                            i11 = 0;
                            if (++i12 == ne2) {
                                i12 = 0;
                                if (++i13 == ne3) {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
            i10 += ne00 * (ne01 - ir1);
            while (i10 >= ne0) {
                i10 -= ne0;
                if (++i11 == ne1) {
                    i11 = 0;
                    if (++i12 == ne2) {
                        i12 = 0;
                        if (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
        }
    }
}

static void wsp_ggml_compute_forward_dup(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (src0->type == dst->type) {
        wsp_ggml_compute_forward_dup_bytes(params, dst);
        return;
    }

    switch (src0->type) {
        case WSP_GGML_TYPE_F16:
            {
                wsp_ggml_compute_forward_dup_f16(params, dst);
            } break;
        case WSP_GGML_TYPE_BF16:
            {
                wsp_ggml_compute_forward_dup_bf16(params, dst);
            } break;
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_dup_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_add

static void wsp_ggml_compute_forward_add_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(wsp_ggml_can_repeat(src1, src0) && wsp_ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = wsp_ggml_nrows(src0);

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    WSP_GGML_ASSERT( nb0 == sizeof(float));
    WSP_GGML_ASSERT(nb00 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (nb10 == sizeof(float)) {
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;
            const int64_t nr0 = ne00 / ne10;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
            float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

            for (int64_t r = 0; r < nr0; ++r) {
#ifdef WSP_GGML_USE_ACCELERATE
                vDSP_vadd(src0_ptr + r*ne10, 1, src1_ptr, 1, dst_ptr + r*ne10, 1, ne10);
#else
                wsp_ggml_vec_add_f32(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
#endif
            }
        }
    } else {
        // src1 is not contiguous
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                const int64_t i10 = i0 % ne10;
                float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10);

                dst_ptr[i0] = src0_ptr[i0] + *src1_ptr;
            }
        }
    }
}

static void wsp_ggml_compute_forward_add_f16_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, src1) && wsp_ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = wsp_ggml_nrows(src0);

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    WSP_GGML_ASSERT(src0->type == WSP_GGML_TYPE_F16);
    WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_F32);

    if (dst->type == WSP_GGML_TYPE_F32) {
        WSP_GGML_ASSERT( nb0 == sizeof(float));
    }
    else {
        WSP_GGML_ASSERT(dst->type  == WSP_GGML_TYPE_F16);
        WSP_GGML_ASSERT( nb0 == sizeof(wsp_ggml_fp16_t));
    }

    WSP_GGML_ASSERT(nb00 == sizeof(wsp_ggml_fp16_t));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (nb10 == sizeof(float)) {
        if (dst->type == WSP_GGML_TYPE_F16) {
            for (int ir = ir0; ir < ir1; ++ir) {
                // src0, src1 and dst are same shape => same indices
                const int i3 = ir/(ne2*ne1);
                const int i2 = (ir - i3*ne2*ne1)/ne1;
                const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

                wsp_ggml_fp16_t * dst_ptr  = (wsp_ggml_fp16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1);
                wsp_ggml_fp16_t * src0_ptr = (wsp_ggml_fp16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
                float *       src1_ptr = (float *)       ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11);

                for (int i = 0; i < ne0; i++) {
                    dst_ptr[i] = WSP_GGML_FP32_TO_FP16(WSP_GGML_FP16_TO_FP32(src0_ptr[i]) + src1_ptr[i]);
                }
            }
        } else {
            for (int ir = ir0; ir < ir1; ++ir) {
                // src0, src1 and dst are same shape => same indices
                const int i3 = ir/(ne2*ne1);
                const int i2 = (ir - i3*ne2*ne1)/ne1;
                const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

                float *       dst_ptr  = (float *)       ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1);
                wsp_ggml_fp16_t * src0_ptr = (wsp_ggml_fp16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
                float *       src1_ptr = (float *)       ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11);

                for (int i = 0; i < ne0; i++) {
                    dst_ptr[i] = WSP_GGML_FP16_TO_FP32(src0_ptr[i]) + src1_ptr[i];
                }
            }
        }
    }
    else {
        // src1 is not contiguous
        WSP_GGML_ABORT("fatal error");
    }
}

static void wsp_ggml_compute_forward_add_bf16_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, src1) && wsp_ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = wsp_ggml_nrows(src0);

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    WSP_GGML_ASSERT(src0->type == WSP_GGML_TYPE_BF16);
    WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_F32);

    if (dst->type == WSP_GGML_TYPE_F32) {
        WSP_GGML_ASSERT( nb0 == sizeof(float));
    }
    else {
        WSP_GGML_ASSERT(dst->type  == WSP_GGML_TYPE_BF16);
        WSP_GGML_ASSERT( nb0 == sizeof(wsp_ggml_bf16_t));
    }

    WSP_GGML_ASSERT(nb00 == sizeof(wsp_ggml_bf16_t));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (nb10 == sizeof(float)) {
        if (dst->type == WSP_GGML_TYPE_BF16) {
            for (int ir = ir0; ir < ir1; ++ir) {
                // src0, src1 and dst are same shape => same indices
                const int i3 = ir/(ne2*ne1);
                const int i2 = (ir - i3*ne2*ne1)/ne1;
                const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

                wsp_ggml_bf16_t * dst_ptr  = (wsp_ggml_bf16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1);
                wsp_ggml_bf16_t * src0_ptr = (wsp_ggml_bf16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
                float *       src1_ptr = (float *)       ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11);

                for (int i = 0; i < ne0; i++) {
                    dst_ptr[i] = WSP_GGML_FP32_TO_BF16(WSP_GGML_BF16_TO_FP32(src0_ptr[i]) + src1_ptr[i]);
                }
            }
        } else {
            for (int ir = ir0; ir < ir1; ++ir) {
                // src0, src1 and dst are same shape => same indices
                const int i3 = ir/(ne2*ne1);
                const int i2 = (ir - i3*ne2*ne1)/ne1;
                const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

                float *       dst_ptr  = (float *)       ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1);
                wsp_ggml_bf16_t * src0_ptr = (wsp_ggml_bf16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
                float *       src1_ptr = (float *)       ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11);

                for (int i = 0; i < ne0; i++) {
                    dst_ptr[i] = WSP_GGML_BF16_TO_FP32(src0_ptr[i]) + src1_ptr[i];
                }
            }
        }
    }
    else {
        // src1 is not contiguous
        WSP_GGML_ABORT("fatal error");
    }
}

static void wsp_ggml_compute_forward_add_f16_f16(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, src1) && wsp_ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = wsp_ggml_nrows(src0);

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    WSP_GGML_ASSERT(src0->type == WSP_GGML_TYPE_F16);
    WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_F16);
    WSP_GGML_ASSERT(dst->type  == WSP_GGML_TYPE_F16);

    WSP_GGML_ASSERT( nb0 == sizeof(wsp_ggml_fp16_t));
    WSP_GGML_ASSERT(nb00 == sizeof(wsp_ggml_fp16_t));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (nb10 == sizeof(wsp_ggml_fp16_t)) {
        for (int ir = ir0; ir < ir1; ++ir) {
            // src0, src1 and dst are same shape => same indices
            const int i3 = ir/(ne2*ne1);
            const int i2 = (ir - i3*ne2*ne1)/ne1;
            const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

            wsp_ggml_fp16_t * dst_ptr  = (wsp_ggml_fp16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1);
            wsp_ggml_fp16_t * src0_ptr = (wsp_ggml_fp16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
            wsp_ggml_fp16_t * src1_ptr = (wsp_ggml_fp16_t *) ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11);

            for (int i = 0; i < ne0; i++) {
                dst_ptr[i] = WSP_GGML_FP32_TO_FP16(WSP_GGML_FP16_TO_FP32(src0_ptr[i]) + WSP_GGML_FP16_TO_FP32(src1_ptr[i]));
            }
        }
    }
    else {
        // src1 is not contiguous
        WSP_GGML_ABORT("fatal error");
    }
}

static void wsp_ggml_compute_forward_add_bf16_bf16(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, src1) && wsp_ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = wsp_ggml_nrows(src0);

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    WSP_GGML_ASSERT(src0->type == WSP_GGML_TYPE_BF16);
    WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_BF16);
    WSP_GGML_ASSERT(dst->type  == WSP_GGML_TYPE_BF16);

    WSP_GGML_ASSERT( nb0 == sizeof(wsp_ggml_bf16_t));
    WSP_GGML_ASSERT(nb00 == sizeof(wsp_ggml_bf16_t));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (nb10 == sizeof(wsp_ggml_bf16_t)) {
        for (int ir = ir0; ir < ir1; ++ir) {
            // src0, src1 and dst are same shape => same indices
            const int i3 = ir/(ne2*ne1);
            const int i2 = (ir - i3*ne2*ne1)/ne1;
            const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

            wsp_ggml_bf16_t * dst_ptr  = (wsp_ggml_bf16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1);
            wsp_ggml_bf16_t * src0_ptr = (wsp_ggml_bf16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
            wsp_ggml_bf16_t * src1_ptr = (wsp_ggml_bf16_t *) ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11);

            for (int i = 0; i < ne0; i++) {
                dst_ptr[i] = WSP_GGML_FP32_TO_BF16(WSP_GGML_BF16_TO_FP32(src0_ptr[i]) + WSP_GGML_BF16_TO_FP32(src1_ptr[i]));
            }
        }
    }
    else {
        // src1 is not contiguous
        WSP_GGML_ABORT("fatal error");
    }
}

static void wsp_ggml_compute_forward_add_q_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, src1) && wsp_ggml_are_same_shape(src0, dst));

    const int nr  = wsp_ggml_nrows(src0);

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum wsp_ggml_type type = src0->type;
    const enum wsp_ggml_type dtype = dst->type;
    wsp_ggml_to_float_t const wsp_dewsp_quantize_row_q = wsp_ggml_get_type_traits(type)->to_float;
    wsp_ggml_from_float_t const wsp_quantize_row_q = wsp_ggml_get_type_traits(dtype)->from_float;

    // we don't support permuted src0 or src1
    WSP_GGML_ASSERT(nb00 == wsp_ggml_type_size(type));
    WSP_GGML_ASSERT(nb10 == sizeof(float));

    // dst cannot be transposed or permuted
    WSP_GGML_ASSERT(nb0 <= nb1);
    WSP_GGML_ASSERT(nb1 <= nb2);
    WSP_GGML_ASSERT(nb2 <= nb3);

    WSP_GGML_ASSERT(wsp_ggml_is_quantized(src0->type));
    WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_F32);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    float * wdata = (float *) params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        const int i03 = ir/(ne02*ne01);
        const int i02 = (ir - i03*ne02*ne01)/ne01;
        const int i01 = (ir - i03*ne02*ne01 - i02*ne01);

        // src1 and dst are same shape as src0 => same indices
        const int i13 = i03;
        const int i12 = i02;
        const int i11 = i01;

        const int i3 = i03;
        const int i2 = i02;
        const int i1 = i01;

        void  * src0_row = (void *) ((char *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
        float * src1_row = (float *)((char *) src1->data + (i11*nb11 + i12*nb12 + i13*nb13));
        void  * dst_row  = (void *) ((char *)  dst->data + ( i1*nb1  +  i2*nb2  +  i3*nb3));

        assert(ne00 % 32 == 0);

        // unquantize row from src0 to temp buffer
        wsp_dewsp_quantize_row_q(src0_row, wdata, ne00);
        // add src1
        wsp_ggml_vec_acc_f32(ne00, wdata, src1_row);
        // quantize row to dst
        if (wsp_quantize_row_q != NULL) {
            wsp_quantize_row_q(wdata, dst_row, ne00);
        } else {
            memcpy(dst_row, wdata, ne0*nb0);
        }
    }
}

static void wsp_ggml_compute_forward_add(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                if (src1->type == WSP_GGML_TYPE_F32) {
                    wsp_ggml_compute_forward_add_f32(params, dst);
                }
                else {
                    WSP_GGML_ABORT("fatal error");
                }
            } break;
        case WSP_GGML_TYPE_F16:
            {
                if (src1->type == WSP_GGML_TYPE_F16) {
                    wsp_ggml_compute_forward_add_f16_f16(params, dst);
                }
                else if (src1->type == WSP_GGML_TYPE_F32) {
                    wsp_ggml_compute_forward_add_f16_f32(params, dst);
                }
                else {
                    WSP_GGML_ABORT("fatal error");
                }
            } break;
        case WSP_GGML_TYPE_BF16:
            {
                if (src1->type == WSP_GGML_TYPE_BF16) {
                    wsp_ggml_compute_forward_add_bf16_bf16(params, dst);
                }
                else if (src1->type == WSP_GGML_TYPE_F32) {
                    wsp_ggml_compute_forward_add_bf16_f32(params, dst);
                }
                else {
                    WSP_GGML_ABORT("fatal error");
                }
            } break;
        case WSP_GGML_TYPE_Q4_0:
        case WSP_GGML_TYPE_Q4_1:
        case WSP_GGML_TYPE_Q5_0:
        case WSP_GGML_TYPE_Q5_1:
        case WSP_GGML_TYPE_Q8_0:
        case WSP_GGML_TYPE_Q2_K:
        case WSP_GGML_TYPE_Q3_K:
        case WSP_GGML_TYPE_Q4_K:
        case WSP_GGML_TYPE_Q5_K:
        case WSP_GGML_TYPE_Q6_K:
        case WSP_GGML_TYPE_TQ1_0:
        case WSP_GGML_TYPE_TQ2_0:
        case WSP_GGML_TYPE_IQ2_XXS:
        case WSP_GGML_TYPE_IQ2_XS:
        case WSP_GGML_TYPE_IQ3_XXS:
        case WSP_GGML_TYPE_IQ1_S:
        case WSP_GGML_TYPE_IQ1_M:
        case WSP_GGML_TYPE_IQ4_NL:
        case WSP_GGML_TYPE_IQ4_XS:
        case WSP_GGML_TYPE_IQ3_S:
        case WSP_GGML_TYPE_IQ2_S:
        case WSP_GGML_TYPE_Q4_0_4_4:
        case WSP_GGML_TYPE_Q4_0_4_8:
        case WSP_GGML_TYPE_Q4_0_8_8:
            {
                wsp_ggml_compute_forward_add_q_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_add1

static void wsp_ggml_compute_forward_add1_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, dst));
    WSP_GGML_ASSERT(wsp_ggml_is_scalar(src1));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = wsp_ggml_nrows(src0);

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    WSP_GGML_ASSERT( nb0 == sizeof(float));
    WSP_GGML_ASSERT(nb00 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        const int i3 = ir/(ne2*ne1);
        const int i2 = (ir - i3*ne2*ne1)/ne1;
        const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

#ifdef WSP_GGML_USE_ACCELERATE
        UNUSED(wsp_ggml_vec_add1_f32);

        vDSP_vadd(
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01), 1,
                (float *) ((char *) src1->data), 0,
                (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ), 1,
                ne0);
#else
        wsp_ggml_vec_add1_f32(ne0,
                (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01),
               *(float *) src1->data);
#endif
    }
}

static void wsp_ggml_compute_forward_add1_f16_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, dst));
    WSP_GGML_ASSERT(wsp_ggml_is_scalar(src1));

    // scalar to add
    const float v = *(float *) src1->data;

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = wsp_ggml_nrows(src0);

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    WSP_GGML_ASSERT(src0->type == WSP_GGML_TYPE_F16);
    WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT(dst->type  == WSP_GGML_TYPE_F16);

    WSP_GGML_ASSERT( nb0 == sizeof(wsp_ggml_fp16_t));
    WSP_GGML_ASSERT(nb00 == sizeof(wsp_ggml_fp16_t));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        const int i3 = ir/(ne2*ne1);
        const int i2 = (ir - i3*ne2*ne1)/ne1;
        const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

        wsp_ggml_fp16_t * dst_ptr  = (wsp_ggml_fp16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        wsp_ggml_fp16_t * src0_ptr = (wsp_ggml_fp16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        for (int i = 0; i < ne0; i++) {
            dst_ptr[i] = WSP_GGML_FP32_TO_FP16(WSP_GGML_FP16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

static void wsp_ggml_compute_forward_add1_f16_f16(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, dst));
    WSP_GGML_ASSERT(wsp_ggml_is_scalar(src1));

    // scalar to add
    const float v = WSP_GGML_FP16_TO_FP32(*(wsp_ggml_fp16_t *) src1->data);

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = wsp_ggml_nrows(src0);

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    WSP_GGML_ASSERT(src0->type == WSP_GGML_TYPE_F16);
    WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_F16);
    WSP_GGML_ASSERT(dst->type  == WSP_GGML_TYPE_F16);

    WSP_GGML_ASSERT( nb0 == sizeof(wsp_ggml_fp16_t));
    WSP_GGML_ASSERT(nb00 == sizeof(wsp_ggml_fp16_t));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        const int i3 = ir/(ne2*ne1);
        const int i2 = (ir - i3*ne2*ne1)/ne1;
        const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

        wsp_ggml_fp16_t * dst_ptr  = (wsp_ggml_fp16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        wsp_ggml_fp16_t * src0_ptr = (wsp_ggml_fp16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        for (int i = 0; i < ne0; i++) {
            dst_ptr[i] = WSP_GGML_FP32_TO_FP16(WSP_GGML_FP16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

static void wsp_ggml_compute_forward_add1_q_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, dst));
    WSP_GGML_ASSERT(wsp_ggml_is_scalar(src1));

    // scalar to add
    const float v = *(float *) src1->data;

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = wsp_ggml_nrows(src0);

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    const enum wsp_ggml_type type = src0->type;
    wsp_ggml_to_float_t const wsp_dewsp_quantize_row_q = wsp_ggml_get_type_traits(type)->to_float;
    wsp_ggml_from_float_t const wsp_quantize_row_q = wsp_ggml_get_type_traits(type)->from_float;

    // we don't support permuted src0
    WSP_GGML_ASSERT(nb00 == wsp_ggml_type_size(type));

    // dst cannot be transposed or permuted
    WSP_GGML_ASSERT(nb0 <= nb1);
    WSP_GGML_ASSERT(nb1 <= nb2);
    WSP_GGML_ASSERT(nb2 <= nb3);

    WSP_GGML_ASSERT(wsp_ggml_is_quantized(src0->type));
    WSP_GGML_ASSERT(dst->type == src0->type);
    WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_F32);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    float * wdata = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32) * ith;

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        const int i3 = ir/(ne2*ne1);
        const int i2 = (ir - i3*ne2*ne1)/ne1;
        const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

        void  * src0_row = (void *) ((char *) src0->data + (i1*nb01 + i2*nb02 + i3*nb03));
        void  * dst_row  = (void *) ((char *)  dst->data + (i1*nb1  + i2*nb2  + i3*nb0 ));

        assert(ne0 % 32 == 0);

        // unquantize row from src0 to temp buffer
        wsp_dewsp_quantize_row_q(src0_row, wdata, ne0);
        // add src1
        wsp_ggml_vec_acc1_f32(ne0, wdata, v);
        // quantize row to dst
        wsp_quantize_row_q(wdata, dst_row, ne0);
    }
}

static void wsp_ggml_compute_forward_add1_bf16_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, dst));
    WSP_GGML_ASSERT(wsp_ggml_is_scalar(src1));

    // scalar to add
    const float v = *(float *) src1->data;

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = wsp_ggml_nrows(src0);

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    WSP_GGML_ASSERT(src0->type == WSP_GGML_TYPE_BF16);
    WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT(dst->type  == WSP_GGML_TYPE_BF16);

    WSP_GGML_ASSERT( nb0 == sizeof(wsp_ggml_bf16_t));
    WSP_GGML_ASSERT(nb00 == sizeof(wsp_ggml_bf16_t));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        const int i3 = ir/(ne2*ne1);
        const int i2 = (ir - i3*ne2*ne1)/ne1;
        const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

        wsp_ggml_bf16_t * dst_ptr  = (wsp_ggml_bf16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        wsp_ggml_bf16_t * src0_ptr = (wsp_ggml_bf16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        for (int i = 0; i < ne0; i++) {
            dst_ptr[i] = WSP_GGML_FP32_TO_BF16(WSP_GGML_BF16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

static void wsp_ggml_compute_forward_add1_bf16_bf16(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, dst));
    WSP_GGML_ASSERT(wsp_ggml_is_scalar(src1));

    // scalar to add
    const float v = WSP_GGML_BF16_TO_FP32(*(wsp_ggml_bf16_t *) src1->data);

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = wsp_ggml_nrows(src0);

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    WSP_GGML_ASSERT(src0->type == WSP_GGML_TYPE_BF16);
    WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_BF16);
    WSP_GGML_ASSERT(dst->type  == WSP_GGML_TYPE_BF16);

    WSP_GGML_ASSERT( nb0 == sizeof(wsp_ggml_bf16_t));
    WSP_GGML_ASSERT(nb00 == sizeof(wsp_ggml_bf16_t));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        const int i3 = ir/(ne2*ne1);
        const int i2 = (ir - i3*ne2*ne1)/ne1;
        const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

        wsp_ggml_bf16_t * dst_ptr  = (wsp_ggml_bf16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        wsp_ggml_bf16_t * src0_ptr = (wsp_ggml_bf16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        for (int i = 0; i < ne0; i++) {
            dst_ptr[i] = WSP_GGML_FP32_TO_BF16(WSP_GGML_BF16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

static void wsp_ggml_compute_forward_add1(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_add1_f32(params, dst);
            } break;
        case WSP_GGML_TYPE_F16:
            {
                if (src1->type == WSP_GGML_TYPE_F16) {
                    wsp_ggml_compute_forward_add1_f16_f16(params, dst);
                }
                else if (src1->type == WSP_GGML_TYPE_F32) {
                    wsp_ggml_compute_forward_add1_f16_f32(params, dst);
                }
                else {
                    WSP_GGML_ABORT("fatal error");
                }
            } break;
        case WSP_GGML_TYPE_BF16:
            {
                if (src1->type == WSP_GGML_TYPE_BF16) {
                    wsp_ggml_compute_forward_add1_bf16_bf16(params, dst);
                }
                else if (src1->type == WSP_GGML_TYPE_F32) {
                    wsp_ggml_compute_forward_add1_bf16_f32(params, dst);
                }
                else {
                    WSP_GGML_ABORT("fatal error");
                }
            } break;
        case WSP_GGML_TYPE_Q4_0:
        case WSP_GGML_TYPE_Q4_1:
        case WSP_GGML_TYPE_Q5_0:
        case WSP_GGML_TYPE_Q5_1:
        case WSP_GGML_TYPE_Q8_0:
        case WSP_GGML_TYPE_Q8_1:
        case WSP_GGML_TYPE_Q2_K:
        case WSP_GGML_TYPE_Q3_K:
        case WSP_GGML_TYPE_Q4_K:
        case WSP_GGML_TYPE_Q5_K:
        case WSP_GGML_TYPE_Q6_K:
        case WSP_GGML_TYPE_TQ1_0:
        case WSP_GGML_TYPE_TQ2_0:
        case WSP_GGML_TYPE_IQ2_XXS:
        case WSP_GGML_TYPE_IQ2_XS:
        case WSP_GGML_TYPE_IQ3_XXS:
        case WSP_GGML_TYPE_IQ1_S:
        case WSP_GGML_TYPE_IQ1_M:
        case WSP_GGML_TYPE_IQ4_NL:
        case WSP_GGML_TYPE_IQ4_XS:
        case WSP_GGML_TYPE_IQ3_S:
        case WSP_GGML_TYPE_IQ2_S:
        case WSP_GGML_TYPE_Q4_0_4_4:
        case WSP_GGML_TYPE_Q4_0_4_8:
        case WSP_GGML_TYPE_Q4_0_8_8:
            {
                wsp_ggml_compute_forward_add1_q_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_acc

static void wsp_ggml_compute_forward_acc_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, dst));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(dst) && wsp_ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during acc
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    bool   inplace = (bool) ((int32_t *) dst->op_params)[4];

    if (!inplace) {
        if (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((char *)  dst->data),
                ((char *) src0->data),
                wsp_ggml_nbytes(dst));
        }
        wsp_ggml_barrier(params->threadpool);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr = wsp_ggml_nrows(src1);
    const int nc = src1->ne[0];

    WSP_GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    WSP_GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during acc
    const size_t nb0 = wsp_ggml_element_size(src0);

    const size_t nb00 = nb0;
    const size_t nb01 = nb1;
    const size_t nb02 = nb2;
    const size_t nb03 = nb3;

    WSP_GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb0  + (ne11 == 0 ? 0 : ne11-1)*nb1  + (ne12 == 0 ? 0 : ne12-1)*nb2  + (ne13 == 0 ? 0 : ne13-1)*nb3  < wsp_ggml_nbytes(dst));
    WSP_GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb00 + (ne11 == 0 ? 0 : ne11-1)*nb01 + (ne12 == 0 ? 0 : ne12-1)*nb02 + (ne13 == 0 ? 0 : ne13-1)*nb03 < wsp_ggml_nbytes(src0));

    WSP_GGML_ASSERT(nb10 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        const int i3 = ir/(ne12*ne11);
        const int i2 = (ir - i3*ne12*ne11)/ne11;
        const int i1 = (ir - i3*ne12*ne11 - i2*ne11);

#ifdef WSP_GGML_USE_ACCELERATE
        vDSP_vadd(
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + offset), 1,
                (float *) ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11), 1,
                (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1  + offset), 1, nc);
#else
        wsp_ggml_vec_add_f32(nc,
                (float *) ((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + offset),
                (float *) ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
#endif
    }
}

static void wsp_ggml_compute_forward_acc(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_acc_f32(params, dst);
            } break;
        case WSP_GGML_TYPE_F16:
        case WSP_GGML_TYPE_BF16:
        case WSP_GGML_TYPE_Q4_0:
        case WSP_GGML_TYPE_Q4_1:
        case WSP_GGML_TYPE_Q5_0:
        case WSP_GGML_TYPE_Q5_1:
        case WSP_GGML_TYPE_Q8_0:
        case WSP_GGML_TYPE_Q8_1:
        case WSP_GGML_TYPE_Q2_K:
        case WSP_GGML_TYPE_Q3_K:
        case WSP_GGML_TYPE_Q4_K:
        case WSP_GGML_TYPE_Q5_K:
        case WSP_GGML_TYPE_Q6_K:
        case WSP_GGML_TYPE_TQ1_0:
        case WSP_GGML_TYPE_TQ2_0:
        case WSP_GGML_TYPE_IQ2_XXS:
        case WSP_GGML_TYPE_IQ2_XS:
        case WSP_GGML_TYPE_IQ3_XXS:
        case WSP_GGML_TYPE_IQ1_S:
        case WSP_GGML_TYPE_IQ1_M:
        case WSP_GGML_TYPE_IQ4_NL:
        case WSP_GGML_TYPE_IQ4_XS:
        case WSP_GGML_TYPE_IQ3_S:
        case WSP_GGML_TYPE_IQ2_S:
        case WSP_GGML_TYPE_Q4_0_4_4:
        case WSP_GGML_TYPE_Q4_0_4_8:
        case WSP_GGML_TYPE_Q4_0_8_8:
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_sub

static void wsp_ggml_compute_forward_sub_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    assert(wsp_ggml_can_repeat(src1, src0) && wsp_ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = wsp_ggml_nrows(src0);

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    WSP_GGML_ASSERT( nb0 == sizeof(float));
    WSP_GGML_ASSERT(nb00 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (nb10 == sizeof(float)) {
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;
            const int64_t nr0 = ne00 / ne10;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
            float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

            for (int64_t r = 0; r < nr0; ++r) {
#ifdef WSP_GGML_USE_ACCELERATE
                vDSP_vsub(src1_ptr, 1, src0_ptr + r*ne10, 1, dst_ptr + r*ne10, 1, ne10);
#else
                wsp_ggml_vec_sub_f32(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
#endif
            }
        }
    } else {
        // src1 is not contiguous
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                const int64_t i10 = i0 % ne10;
                float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10);

                dst_ptr[i0] = src0_ptr[i0] - *src1_ptr;
            }
        }
    }
}

static void wsp_ggml_compute_forward_sub(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_sub_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_mul

static void wsp_ggml_compute_forward_mul_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(wsp_ggml_can_repeat(src1, src0) && wsp_ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nr = wsp_ggml_nrows(src0);

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    WSP_GGML_ASSERT( nb0 == sizeof(float));
    WSP_GGML_ASSERT(nb00 == sizeof(float));

    if (nb10 == sizeof(float)) {
        for (int64_t ir = ith; ir < nr; ir += nth) {
            // src0 and dst are same shape => same indices
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;
            const int64_t nr0 = ne00 / ne10;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
            float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

            for (int64_t r = 0 ; r < nr0; ++r) {
#ifdef WSP_GGML_USE_ACCELERATE
                UNUSED(wsp_ggml_vec_mul_f32);

                vDSP_vmul(src0_ptr + r*ne10, 1, src1_ptr, 1, dst_ptr + r*ne10, 1, ne10);
#else
                wsp_ggml_vec_mul_f32(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
#endif
            }
        }
    } else {
        // src1 is not contiguous
        for (int64_t ir = ith; ir < nr; ir += nth) {
            // src0 and dst are same shape => same indices
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

            for (int64_t i0 = 0; i0 < ne00; ++i0) {
                const int64_t i10 = i0 % ne10;
                float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10);

                dst_ptr[i0] = src0_ptr[i0] * (*src1_ptr);
            }
        }
    }
}

static void wsp_ggml_compute_forward_mul(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_F32 && "only f32 src1 supported for now");

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_mul_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_div

static void wsp_ggml_compute_forward_div_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(wsp_ggml_can_repeat(src1, src0) && wsp_ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nr = wsp_ggml_nrows(src0);

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    WSP_GGML_ASSERT( nb0 == sizeof(float));
    WSP_GGML_ASSERT(nb00 == sizeof(float));

    if (nb10 == sizeof(float)) {
        for (int64_t ir = ith; ir < nr; ir += nth) {
            // src0 and dst are same shape => same indices
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;
            const int64_t nr0 = ne00 / ne10;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
            float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

            for (int64_t r = 0; r < nr0; ++r) {
#ifdef WSP_GGML_USE_ACCELERATE
                UNUSED(wsp_ggml_vec_div_f32);

                vDSP_vdiv(src1_ptr, 1, src0_ptr + r*ne10, 1, dst_ptr + r*ne10, 1, ne10);
#else
                wsp_ggml_vec_div_f32(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
#endif
            }
        }
    } else {
        // src1 is not contiguous
        for (int64_t ir = ith; ir < nr; ir += nth) {
            // src0 and dst are same shape => same indices
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

            for (int64_t i0 = 0; i0 < ne00; ++i0) {
                const int64_t i10 = i0 % ne10;
                float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10);

                dst_ptr[i0] = src0_ptr[i0] / (*src1_ptr);
            }
        }
    }
}

static void wsp_ggml_compute_forward_div(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_div_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_sqr

static void wsp_ggml_compute_forward_sqr_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(wsp_ggml_are_same_shape(src0, dst));

    const int n     = wsp_ggml_nrows(src0);
    const int nc    = src0->ne[0];

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        wsp_ggml_vec_sqr_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void wsp_ggml_compute_forward_sqr(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_sqr_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_sqrt

static void wsp_ggml_compute_forward_sqrt_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(wsp_ggml_are_same_shape(src0, dst));

    const int n  = wsp_ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        wsp_ggml_vec_sqrt_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void wsp_ggml_compute_forward_sqrt(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_sqrt_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_log

static void wsp_ggml_compute_forward_log_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, dst));

    const int n  = wsp_ggml_nrows(src0);
    const int nc = src0->ne[0];

    WSP_GGML_ASSERT( dst->nb[0] == sizeof(float));
    WSP_GGML_ASSERT(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        wsp_ggml_vec_log_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void wsp_ggml_compute_forward_log(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_log_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_sin

static void wsp_ggml_compute_forward_sin_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, dst));

    const int n  = wsp_ggml_nrows(src0);
    const int nc = src0->ne[0];

    WSP_GGML_ASSERT( dst->nb[0] == sizeof(float));
    WSP_GGML_ASSERT(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        wsp_ggml_vec_sin_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void wsp_ggml_compute_forward_sin(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_sin_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_cos

static void wsp_ggml_compute_forward_cos_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, dst));

    const int n  = wsp_ggml_nrows(src0);
    const int nc = src0->ne[0];

    WSP_GGML_ASSERT( dst->nb[0] == sizeof(float));
    WSP_GGML_ASSERT(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        wsp_ggml_vec_cos_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void wsp_ggml_compute_forward_cos(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_cos_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_sum

static void wsp_ggml_compute_forward_sum_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(wsp_ggml_is_scalar(dst));
    assert(src0->nb[0] == sizeof(float));

    WSP_GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    WSP_GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    wsp_ggml_float sum     = 0;
    wsp_ggml_float row_sum = 0;

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                wsp_ggml_vec_sum_f32_ggf(ne00,
                        &row_sum,
                        (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));
                sum += row_sum;
            }
        }
    }
    ((float *) dst->data)[0] = sum;
}

static void wsp_ggml_compute_forward_sum_f16(
    const struct wsp_ggml_compute_params * params,
          struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(wsp_ggml_is_scalar(dst));

    assert(src0->nb[0] == sizeof(wsp_ggml_fp16_t));

    WSP_GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    WSP_GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    float sum = 0;
    float row_sum = 0;

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                wsp_ggml_vec_sum_f16_ggf(ne00,
                    &row_sum,
                    (wsp_ggml_fp16_t *) ((char *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
                sum += row_sum;
            }
        }
    }
    ((wsp_ggml_fp16_t *) dst->data)[0] = WSP_GGML_FP32_TO_FP16(sum);
}

static void wsp_ggml_compute_forward_sum_bf16(
    const struct wsp_ggml_compute_params * params,
          struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(wsp_ggml_is_scalar(dst));

    assert(src0->nb[0] == sizeof(wsp_ggml_bf16_t));

    WSP_GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    WSP_GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    float sum = 0;
    float row_sum = 0;

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                wsp_ggml_vec_sum_bf16_ggf(ne00,
                    &row_sum,
                    (wsp_ggml_bf16_t *) ((char *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
                sum += row_sum;
            }
        }
    }
    ((wsp_ggml_bf16_t *) dst->data)[0] = WSP_GGML_FP32_TO_BF16(sum);
}

static void wsp_ggml_compute_forward_sum(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_sum_f32(params, dst);
            } break;
        case WSP_GGML_TYPE_F16:
            {
                wsp_ggml_compute_forward_sum_f16(params, dst);
            } break;
        case WSP_GGML_TYPE_BF16:
            {
                wsp_ggml_compute_forward_sum_bf16(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_sum_rows

static void wsp_ggml_compute_forward_sum_rows_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    WSP_GGML_ASSERT(src0->nb[0] == sizeof(float));
    WSP_GGML_ASSERT(dst->nb[0] == sizeof(float));

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    WSP_GGML_ASSERT(ne0 == 1);
    WSP_GGML_ASSERT(ne1 == ne01);
    WSP_GGML_ASSERT(ne2 == ne02);
    WSP_GGML_ASSERT(ne3 == ne03);

    for (int64_t i3 = 0; i3 < ne03; i3++) {
        for (int64_t i2 = 0; i2 < ne02; i2++) {
            for (int64_t i1 = 0; i1 < ne01; i1++) {
                float * src_row = (float *) ((char *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);
                float * dst_row = (float *) ((char *) dst->data  + i1*nb1  + i2*nb2  + i3*nb3);
                float row_sum = 0;
                wsp_ggml_vec_sum_f32(ne00, &row_sum, src_row);
                dst_row[0] = row_sum;
            }
        }
    }
}

static void wsp_ggml_compute_forward_sum_rows(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_sum_rows_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_mean

static void wsp_ggml_compute_forward_mean_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(src0->nb[0] == sizeof(float));

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    assert(ne0 == 1);
    assert(ne1 == ne01);
    assert(ne2 == ne02);
    assert(ne3 == ne03);

    UNUSED(ne0);
    UNUSED(ne1);
    UNUSED(ne2);
    UNUSED(ne3);

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                wsp_ggml_vec_sum_f32(ne00,
                        (float *) ((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));

                *(float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3) /= (float) ne00;
            }
        }
    }
}

static void wsp_ggml_compute_forward_mean(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_mean_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_argmax

static void wsp_ggml_compute_forward_argmax_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(src0->nb[0] == sizeof(float));
    assert(dst->nb[0] == sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];

    const size_t nb01 = src0->nb[1];
    const size_t nb0 = dst->nb[0];

    for (int64_t i1 = 0; i1 < ne01; i1++) {
        float * src = (float *) ((char *) src0->data + i1*nb01);
        int32_t * dst_ = (int32_t *) ((char *)  dst->data + i1*nb0);
        int v = 0;
        wsp_ggml_vec_argmax_f32(ne00, &v, src);
        dst_[0] = v;
    }
}

static void wsp_ggml_compute_forward_argmax(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_argmax_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_count_equal

static void wsp_ggml_compute_forward_count_equal_i32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_TENSOR_BINARY_OP_LOCALS;

    WSP_GGML_ASSERT(src0->type == WSP_GGML_TYPE_I32);
    WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_I32);
    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, src1));
    WSP_GGML_ASSERT(wsp_ggml_is_scalar(dst));
    WSP_GGML_ASSERT(dst->type == WSP_GGML_TYPE_I64);

    const int64_t nr = wsp_ggml_nrows(src0);

    const int ith = params->ith;
    const int nth = params->nth;

    int64_t * sums = (int64_t *) params->wdata;
    int64_t sum_thread = 0;

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 =  ir                        / (ne02*ne01);
        const int64_t i02 = (ir - i03*ne03)            /       ne01;
        const int64_t i01 =  ir - i03*ne03 - i02*ne02;

        const char * data0 = (const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01;
        const char * data1 = (const char *) src1->data + i03*nb13 + i02*nb12 + i01*nb11;

        for (int64_t i00 = 0; i00 < ne00; ++i00) {
            const int32_t val0 = *((const int32_t *) (data0 + i00*nb00));
            const int32_t val1 = *((const int32_t *) (data1 + i00*nb10));

            sum_thread += val0 == val1;
        }
    }
    if (ith != 0) {
        sums[ith] = sum_thread;
    }
    wsp_ggml_barrier(params->threadpool);

    if (ith != 0) {
        return;
    }

    for (int ith_other = 1; ith_other < nth; ++ith_other) {
        sum_thread += sums[ith_other];
    }
    *((int64_t *) dst->data) = sum_thread;
}

static void wsp_ggml_compute_forward_count_equal(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_I32:
            {
                wsp_ggml_compute_forward_count_equal_i32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_repeat

static void wsp_ggml_compute_forward_repeat_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    WSP_GGML_ASSERT(wsp_ggml_can_repeat(src0, dst));

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in wsp_ggml_can_repeat
    const int nr0 = (int)(ne0/ne00);
    const int nr1 = (int)(ne1/ne01);
    const int nr2 = (int)(ne2/ne02);
    const int nr3 = (int)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    WSP_GGML_ASSERT(nb0  == sizeof(float));
    WSP_GGML_ASSERT(nb00 == sizeof(float));

    // TODO: maybe this is not optimal?
    for                         (int i3 = 0; i3 < nr3;  i3++) {
        for                     (int k3 = 0; k3 < ne03; k3++) {
            for                 (int i2 = 0; i2 < nr2;  i2++) {
                for             (int k2 = 0; k2 < ne02; k2++) {
                    for         (int i1 = 0; i1 < nr1;  i1++) {
                        for     (int k1 = 0; k1 < ne01; k1++) {
                            for (int i0 = 0; i0 < nr0;  i0++) {
                                wsp_ggml_vec_cpy_f32(ne00,
                                        (float *) ((char *)  dst->data + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0),
                                        (float *) ((char *) src0->data + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01));
                            }
                        }
                    }
                }
            }
        }
    }
}

static void wsp_ggml_compute_forward_repeat_f16(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    WSP_GGML_ASSERT(wsp_ggml_can_repeat(src0, dst));

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in wsp_ggml_can_repeat
    const int nr0 = (int)(ne0/ne00);
    const int nr1 = (int)(ne1/ne01);
    const int nr2 = (int)(ne2/ne02);
    const int nr3 = (int)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    WSP_GGML_ASSERT(nb0  == sizeof(wsp_ggml_fp16_t));
    WSP_GGML_ASSERT(nb00 == sizeof(wsp_ggml_fp16_t));

    // TODO: maybe this is not optimal?
    for                         (int i3 = 0; i3 < nr3;  i3++) {
        for                     (int k3 = 0; k3 < ne03; k3++) {
            for                 (int i2 = 0; i2 < nr2;  i2++) {
                for             (int k2 = 0; k2 < ne02; k2++) {
                    for         (int i1 = 0; i1 < nr1;  i1++) {
                        for     (int k1 = 0; k1 < ne01; k1++) {
                            for (int i0 = 0; i0 < nr0;  i0++) {
                                wsp_ggml_fp16_t * y = (wsp_ggml_fp16_t *) ((char *)  dst->data + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0);
                                wsp_ggml_fp16_t * x = (wsp_ggml_fp16_t *) ((char *) src0->data + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01);
                                // wsp_ggml_vec_cpy_f16(ne00, y, x)
                                for (int i = 0; i < ne00; ++i) {
                                    y[i]  = x[i];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

static void wsp_ggml_compute_forward_repeat(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F16:
        case WSP_GGML_TYPE_BF16:
        case WSP_GGML_TYPE_I16:
            {
                wsp_ggml_compute_forward_repeat_f16(params, dst);
            } break;
        case WSP_GGML_TYPE_F32:
        case WSP_GGML_TYPE_I32:
            {
                wsp_ggml_compute_forward_repeat_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_repeat_back

static void wsp_ggml_compute_forward_repeat_back_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    WSP_GGML_ASSERT(wsp_ggml_can_repeat(dst, src0));

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in wsp_ggml_can_repeat
    const int nr0 = (int)(ne00/ne0);
    const int nr1 = (int)(ne01/ne1);
    const int nr2 = (int)(ne02/ne2);
    const int nr3 = (int)(ne03/ne3);

    // TODO: support for transposed / permuted tensors
    WSP_GGML_ASSERT(nb0  == sizeof(float));
    WSP_GGML_ASSERT(nb00 == sizeof(float));

    if (wsp_ggml_is_contiguous(dst)) {
        wsp_ggml_vec_set_f32(ne0*ne1*ne2*ne3, dst->data, 0);
    } else {
        for         (int k3 = 0; k3 < ne3; k3++) {
            for     (int k2 = 0; k2 < ne2; k2++) {
                for (int k1 = 0; k1 < ne1; k1++) {
                    wsp_ggml_vec_set_f32(ne0,
                        (float *) ((char *) dst->data + k1*nb1 + k2*nb2 + k3*nb3),
                        0);
                }
            }
        }
    }

    // TODO: maybe this is not optimal?
    for                         (int i3 = 0; i3 < nr3; i3++) {
        for                     (int k3 = 0; k3 < ne3; k3++) {
            for                 (int i2 = 0; i2 < nr2; i2++) {
                for             (int k2 = 0; k2 < ne2; k2++) {
                    for         (int i1 = 0; i1 < nr1; i1++) {
                        for     (int k1 = 0; k1 < ne1; k1++) {
                            for (int i0 = 0; i0 < nr0; i0++) {
                                wsp_ggml_vec_acc_f32(ne0,
                                        (float *) ((char *)  dst->data + (         k3)*nb3  + (         k2)*nb2  + (         k1)*nb1),
                                        (float *) ((char *) src0->data + (i3*ne3 + k3)*nb03 + (i2*ne2 + k2)*nb02 + (i1*ne1 + k1)*nb01 + (i0*ne0)*nb00));
                            }
                        }
                    }
                }
            }
        }
    }
}

static void wsp_ggml_compute_forward_repeat_back(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_repeat_back_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_concat

static void wsp_ggml_compute_forward_concat_f32(
    const struct wsp_ggml_compute_params * params,
    struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t dim = wsp_ggml_get_op_params_i32(dst, 0);

    WSP_GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    const float * x;

    // TODO: smarter multi-theading
    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = ith; i2 < ne2; i2 += nth) {
            for (int i1 = 0; i1 < ne1; i1++) {
                for (int i0 = 0; i0 < ne0; i0++) {
                    if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (const float *) ((const char *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } else {
                        x = (const float *) ((const char *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    float * y = (float *)((char *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

static void wsp_ggml_compute_forward_concat(
    const struct wsp_ggml_compute_params * params,
    struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
        case WSP_GGML_TYPE_I32:
            {
                wsp_ggml_compute_forward_concat_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_abs

static void wsp_ggml_compute_forward_abs_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(wsp_ggml_is_contiguous_1(src0));
    assert(wsp_ggml_is_contiguous_1(dst));
    assert(wsp_ggml_are_same_shape(src0, dst));

    const int n  = wsp_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        wsp_ggml_vec_abs_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void wsp_ggml_compute_forward_abs(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_abs_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_sgn

static void wsp_ggml_compute_forward_sgn_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(wsp_ggml_is_contiguous_1(src0));
    assert(wsp_ggml_is_contiguous_1(dst));
    assert(wsp_ggml_are_same_shape(src0, dst));

    const int n  = wsp_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        wsp_ggml_vec_sgn_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void wsp_ggml_compute_forward_sgn(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_sgn_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_neg

static void wsp_ggml_compute_forward_neg_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(wsp_ggml_is_contiguous_1(src0));
    assert(wsp_ggml_is_contiguous_1(dst));
    assert(wsp_ggml_are_same_shape(src0, dst));

    const int n  = wsp_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        wsp_ggml_vec_neg_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void wsp_ggml_compute_forward_neg(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_neg_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_step

static void wsp_ggml_compute_forward_step_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(wsp_ggml_is_contiguous_1(src0));
    assert(wsp_ggml_is_contiguous_1(dst));
    assert(wsp_ggml_are_same_shape(src0, dst));

    const int n  = wsp_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        wsp_ggml_vec_step_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void wsp_ggml_compute_forward_step(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_step_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_tanh

static void wsp_ggml_compute_forward_tanh_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(wsp_ggml_is_contiguous_1(src0));
    assert(wsp_ggml_is_contiguous_1(dst));
    assert(wsp_ggml_are_same_shape(src0, dst));

    const int n  = wsp_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        wsp_ggml_vec_tanh_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void wsp_ggml_compute_forward_tanh(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_tanh_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_elu

static void wsp_ggml_compute_forward_elu_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(wsp_ggml_is_contiguous_1(src0));
    assert(wsp_ggml_is_contiguous_1(dst));
    assert(wsp_ggml_are_same_shape(src0, dst));

    const int n  = wsp_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        wsp_ggml_vec_elu_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void wsp_ggml_compute_forward_elu(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_elu_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_relu

static void wsp_ggml_compute_forward_relu_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(wsp_ggml_is_contiguous_1(src0));
    assert(wsp_ggml_is_contiguous_1(dst));
    assert(wsp_ggml_are_same_shape(src0, dst));

    const int n  = wsp_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        wsp_ggml_vec_relu_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void wsp_ggml_compute_forward_relu(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_relu_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_sigmoid

static void wsp_ggml_compute_forward_sigmoid_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(wsp_ggml_is_contiguous_1(src0));
    assert(wsp_ggml_is_contiguous_1(dst));
    assert(wsp_ggml_are_same_shape(src0, dst));

    const int n  = wsp_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        wsp_ggml_vec_sigmoid_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void wsp_ggml_compute_forward_sigmoid(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_sigmoid_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_gelu

static void wsp_ggml_compute_forward_gelu_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    assert(wsp_ggml_is_contiguous_1(src0));
    assert(wsp_ggml_is_contiguous_1(dst));
    assert(wsp_ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = wsp_ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        wsp_ggml_vec_gelu_f32(nc,
                (float *) ((char *) dst->data  + i1*( dst->nb[1])),
                (float *) ((char *) src0->data + i1*(src0->nb[1])));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif
    }
}

static void wsp_ggml_compute_forward_gelu(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_gelu_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_gelu_quick

static void wsp_ggml_compute_forward_gelu_quick_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    assert(wsp_ggml_is_contiguous_1(src0));
    assert(wsp_ggml_is_contiguous_1(dst));
    assert(wsp_ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = wsp_ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        wsp_ggml_vec_gelu_quick_f32(nc,
                (float *) ((char *) dst->data  + i1*( dst->nb[1])),
                (float *) ((char *) src0->data + i1*(src0->nb[1])));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif
    }
}

static void wsp_ggml_compute_forward_gelu_quick(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_gelu_quick_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_silu

static void wsp_ggml_compute_forward_silu_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    assert(wsp_ggml_is_contiguous_1(src0));
    assert(wsp_ggml_is_contiguous_1(dst));
    assert(wsp_ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = wsp_ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        wsp_ggml_vec_silu_f32(nc,
                (float *) ((char *) dst->data  + i1*( dst->nb[1])),
                (float *) ((char *) src0->data + i1*(src0->nb[1])));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*(dst->nb[1])))[k];
            UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif
    }
}

static void wsp_ggml_compute_forward_silu(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_silu_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}
// wsp_ggml_compute_forward_leaky_relu

static void wsp_ggml_compute_forward_leaky_relu_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(wsp_ggml_is_contiguous_1(src0));
    assert(wsp_ggml_is_contiguous_1(dst));
    assert(wsp_ggml_are_same_shape(src0, dst));

    const int n  = wsp_ggml_nrows(src0);
    const int nc = src0->ne[0];

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    assert(dst->nb[0]  == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        wsp_ggml_vec_leaky_relu_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])), negative_slope);
    }
}

static void wsp_ggml_compute_forward_leaky_relu(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_leaky_relu_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_silu_back

static void wsp_ggml_compute_forward_silu_back_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * grad = dst->src[1];

    assert(wsp_ggml_is_contiguous_1(grad));
    assert(wsp_ggml_is_contiguous_1(src0));
    assert(wsp_ggml_is_contiguous_1(dst));
    assert(wsp_ggml_are_same_shape(src0, dst));
    assert(wsp_ggml_are_same_shape(src0, grad));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = wsp_ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        wsp_ggml_vec_silu_backward_f32(nc,
                (float *) ((char *) dst->data  + i1*( dst->nb[1])),
                (float *) ((char *) src0->data + i1*(src0->nb[1])),
                (float *) ((char *) grad->data + i1*(grad->nb[1])));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif
    }
}

static void wsp_ggml_compute_forward_silu_back(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_silu_back_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}


static void wsp_ggml_compute_forward_hardswish_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(wsp_ggml_is_contiguous_1(src0));
    assert(wsp_ggml_is_contiguous_1(dst));
    assert(wsp_ggml_are_same_shape(src0, dst));

    const int n  = wsp_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        wsp_ggml_vec_hardswish_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}
static void wsp_ggml_compute_forward_hardswish(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_hardswish_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

static void wsp_ggml_compute_forward_hardsigmoid_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(wsp_ggml_is_contiguous_1(src0));
    assert(wsp_ggml_is_contiguous_1(dst));
    assert(wsp_ggml_are_same_shape(src0, dst));

    const int n  = wsp_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        wsp_ggml_vec_hardsigmoid_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void wsp_ggml_compute_forward_hardsigmoid(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_hardsigmoid_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

static void wsp_ggml_compute_forward_exp_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(wsp_ggml_is_contiguous_1(src0));
    assert(wsp_ggml_is_contiguous_1(dst));
    assert(wsp_ggml_are_same_shape(src0, dst));

    const int n  = wsp_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        wsp_ggml_vec_exp_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void wsp_ggml_compute_forward_exp(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_exp_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}


// wsp_ggml_compute_forward_norm

static void wsp_ggml_compute_forward_norm_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, dst));

    WSP_GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    WSP_GGML_ASSERT(eps > 0.0f);

    // TODO: optimize
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                wsp_ggml_float sum = 0.0;
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (wsp_ggml_float)x[i00];
                }

                float mean = sum/ne00;

                float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                wsp_ggml_float sum2 = 0.0;
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    float v = x[i00] - mean;
                    y[i00] = v;
                    sum2 += (wsp_ggml_float)(v*v);
                }

                float variance = sum2/ne00;
                const float scale = 1.0f/sqrtf(variance + eps);

                wsp_ggml_vec_scale_f32(ne00, y, scale);
            }
        }
    }
}

static void wsp_ggml_compute_forward_norm(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_norm_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_group_rms_norm

static void wsp_ggml_compute_forward_rms_norm_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, dst));

    WSP_GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    WSP_GGML_ASSERT(eps > 0.0f);

    // TODO: optimize
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                wsp_ggml_float sum = 0.0;
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (wsp_ggml_float)(x[i00] * x[i00]);
                }

                const float mean = sum/ne00;

                float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                memcpy(y, x, ne00 * sizeof(float));
                // for (int i00 = 0; i00 < ne00; i00++) {
                //     y[i00] = x[i00];
                // }

                const float scale = 1.0f/sqrtf(mean + eps);

                wsp_ggml_vec_scale_f32(ne00, y, scale);
            }
        }
    }
}

static void wsp_ggml_compute_forward_rms_norm(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_rms_norm_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

static void wsp_ggml_compute_forward_rms_norm_back_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, dst) && wsp_ggml_are_same_shape(src0, src1));

    WSP_GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    // TODO: optimize
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                // src1 is same shape as src0 => same indices
                const int64_t i11 = i01;
                const int64_t i12 = i02;
                const int64_t i13 = i03;

                const float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                const float * dz = (float *) ((char *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                wsp_ggml_float sum_xx  = 0.0;
                wsp_ggml_float sum_xdz = 0.0;

                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum_xx  += (wsp_ggml_float)(x[i00] * x[i00]);
                    sum_xdz += (wsp_ggml_float)(x[i00] * dz[i00]);
                }

                //const float mean     = (float)(sum_xx)/ne00;
                const float mean_eps = (float)(sum_xx)/ne00 + eps;
                const float sum_eps  = (float)(sum_xx) + eps*ne00;
                //const float mean_xdz = (float)(sum_xdz)/ne00;
                // we could cache rms from forward pass to improve performance.
                // to do this implement wsp_ggml_rms and compose wsp_ggml_rms_norm using wsp_ggml_rms.
                //const float rms      = sqrtf(mean_eps);
                const float rrms     = 1.0f / sqrtf(mean_eps);
                //const float scale    = -rrms/(ne00 * mean_eps); // -1/(n*rms**3)

                {
                    // z = rms_norm(x)
                    //
                    // rms_norm(src0) =
                    //     scale(
                    //         src0,
                    //         div(
                    //             1,
                    //             sqrt(
                    //                 add(
                    //                     scale(
                    //                         sum(
                    //                             sqr(
                    //                                 src0)),
                    //                         (1.0/N)),
                    //                     eps))));

                    // postorder:
                    // ## op    args         grad
                    // 00 param src0         grad[#00]
                    // 01 const 1
                    // 02 sqr   (#00)        grad[#02]
                    // 03 sum   (#02)        grad[#03]
                    // 04 const 1/N
                    // 05 scale (#03, #04)   grad[#05]
                    // 06 const eps
                    // 07 add   (#05, #06)   grad[#07]
                    // 08 sqrt  (#07)        grad[#08]
                    // 09 div   (#01,#08)    grad[#09]
                    // 10 scale (#00,#09)    grad[#10]
                    //
                    // backward pass, given grad[#10]
                    // #10: scale
                    // grad[#00] += scale(grad[#10],#09)
                    // grad[#09] += sum(mul(grad[#10],#00))
                    // #09: div
                    // grad[#08] += neg(mul(grad[#09], div(#09,#08)))
                    // #08: sqrt
                    // grad[#07] += mul(grad[#08], div(0.5, #08))
                    // #07: add
                    // grad[#05] += grad[#07]
                    // #05: scale
                    // grad[#03] += scale(grad[#05],#04)
                    // #03: sum
                    // grad[#02] += repeat(grad[#03], #02)
                    // #02:
                    // grad[#00] += scale(mul(#00, grad[#02]), 2.0)
                    //
                    // substitute and simplify:
                    // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, grad[#02]), 2.0)
                    // grad[#02] = repeat(grad[#03], #02)
                    // grad[#02] = repeat(scale(grad[#05],#04), #02)
                    // grad[#02] = repeat(scale(grad[#07],#04), #02)
                    // grad[#02] = repeat(scale(mul(grad[#08], div(0.5, #08)),#04), #02)
                    // grad[#02] = repeat(scale(mul(neg(mul(grad[#09], div(#09,#08))), div(0.5, #08)),#04), #02)
                    // grad[#02] = repeat(scale(mul(neg(mul(sum(mul(grad[#10],#00)), div(#09,#08))), div(0.5, #08)),#04), #02)
                    // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(#09,#08) * div(0.5, #08) * (1/N)), #02)
                    // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(div(#01,#08),#08) * div(0.5, #08) * (1/N)), #02)
                    // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(1,#08*#08) * div(0.5, #08) * (1/N)), #02)
                    // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5, #08) * (1/N)), #02)
                    // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, grad[#02]), 2.0)
                    // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, repeat(-(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5, #08) * (1/N)), #02)), 2.0)
                    // grad[#00] = scale(grad(#10), #09) + scale(scale(#00, -(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5, #08) * (1/N))), 2.0)
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, -(sum(mul(grad[#10],#00)) * div(1,#07) * div(1,#08) * (1/N)))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(1,#07*#08) * (-1/N))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(1,#07*#08) * (-1/N))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(1,mean_eps*rms) * (-1/N))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(-1,rms*N*mean_eps))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(-1,rms*N*(sum_xx/N+eps)))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(-1,rms*N*sum_xx+rms*N*eps))
                    // grad[#00] = scale(dz, rrms) + scale(x, sum(mul(dz,x)) * div(-1,rms*N*mean_eps))
                    // grad[#00] = scale(dz, rrms) + scale(x, sum_xdz * div(-1,rms*N*mean_eps))
                    // a = b*c + d*e
                    // a = b*c*f/f + d*e*f/f
                    // a = (b*c*f + d*e*f)*(1/f)
                    // a = (b*c*(1/c) + d*e*(1/c))*(1/(1/c))
                    // a = (b + d*e/c)*c
                    // b = dz, c = rrms, d = x, e = sum_xdz * div(-1,rms*N*mean_eps)
                    // a = (dz + x*sum_xdz * div(-1,rms*N*mean_eps)/rrms)*rrms
                    // a = (dz + x*sum_xdz * div(-1,rms*N*mean_eps)*rms)*rrms
                    // a = (dz + x*sum_xdz * div(-rms,rms*N*mean_eps))*rrms
                    // a = (dz + x*sum_xdz * div(-1,N*mean_eps))*rrms
                    // a = (dz + x*div(-sum_xdz,N*mean_eps))*rrms
                    // a = (dz + x*div(-mean_xdz,mean_eps))*rrms
                    // grad[#00] = scale(dz + scale(x, div(-mean_xdz,mean_eps)),rrms)
                    // grad[#00] = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
                    // dx = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
                }
                // dx = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
                // post-order:
                // dx := x
                // dx := scale(dx,-mean_xdz/mean_eps)
                // dx := add(dx, dz)
                // dx := scale(dx, rrms)
                float * dx = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                wsp_ggml_vec_cpy_f32  (ne00, dx, x);
                // wsp_ggml_vec_scale_f32(ne00, dx, -mean_xdz/mean_eps);
                wsp_ggml_vec_scale_f32(ne00, dx, (float)(-sum_xdz)/sum_eps);
                wsp_ggml_vec_acc_f32  (ne00, dx, dz);
                wsp_ggml_vec_scale_f32(ne00, dx, rrms);
            }
        }
    }
}

static void wsp_ggml_compute_forward_rms_norm_back(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_rms_norm_back_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_group_norm

static void wsp_ggml_compute_forward_group_norm_f32(
    const struct wsp_ggml_compute_params * params,
    struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, dst));

    WSP_GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    // TODO: optimize

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));

    int n_channels = src0->ne[2];
    int n_groups = dst->op_params[0];
    int n_channels_per_group = (n_channels + n_groups - 1) / n_groups;
    for (int i = ith; i < n_groups; i += nth) {
        int start = i * n_channels_per_group;
        int end = start + n_channels_per_group;
        if (end > n_channels) {
            end = n_channels;
        }
        int step = end - start;

        for (int64_t i03 = 0; i03 < ne03; i03++) {
            wsp_ggml_float sum = 0.0;
            for (int64_t i02 = start; i02 < end; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    const float * x = (float *)((char *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    wsp_ggml_float sumr = 0.0;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        sumr += (wsp_ggml_float)x[i00];
                    }
                    sum += sumr;
                }
            }
            const float mean = sum / (ne00 * ne01 * step);

            wsp_ggml_float sum2 = 0.0;
            for (int64_t i02 = start; i02 < end; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    const float * x = (float *)((char *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    float * y = (float *)((char *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

                    wsp_ggml_float sumr = 0.0;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        float v = x[i00] - mean;
                        y[i00] = v;
                        sumr += (wsp_ggml_float)(v * v);
                    }
                    sum2 += sumr;
                }
            }
            const float variance = sum2 / (ne00 * ne01 * step);
            const float scale = 1.0f / sqrtf(variance + eps);

            for (int64_t i02 = start; i02 < end; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    float * y = (float *)((char *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);
                    wsp_ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

static void wsp_ggml_compute_forward_group_norm(
    const struct wsp_ggml_compute_params * params,
    struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_group_norm_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_mul_mat

static void wsp_ggml_compute_forward_mul_mat_one_chunk(
    const struct wsp_ggml_compute_params * params,
    struct wsp_ggml_tensor * dst,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    const enum wsp_ggml_type type = src0->type;

    const bool src1_cont = wsp_ggml_is_contiguous(src1);

    wsp_ggml_vec_dot_t const vec_dot      = type_traits_cpu[type].vec_dot;
    enum wsp_ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    const size_t row_size = wsp_ggml_row_size(vec_dot_type, ne10);

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    float tmp[32];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }
}

static void wsp_ggml_compute_forward_mul_mat(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum wsp_ggml_type type = src0->type;

    enum wsp_ggml_type           const vec_dot_type         = type_traits_cpu[type].vec_dot_type;
    wsp_ggml_from_float_t        const from_float           = wsp_ggml_get_type_traits(vec_dot_type)->from_float;
    wsp_ggml_from_float_to_mat_t const from_float_to_mat    = type_traits_cpu[vec_dot_type].from_float_to_mat;
    int64_t                  const vec_dot_num_rows     = type_traits_cpu[type].nrows;
    int64_t                  const matmul_num_cols      = type_traits_cpu[type].ncols;
    int64_t                  const blck_size_interleave = wsp_ggml_get_type_traits(type)->blck_size_interleave;
    wsp_ggml_gemv_t              const gemv                 = type_traits_cpu[type].gemv;
    wsp_ggml_gemm_t              const gemm                 = type_traits_cpu[type].gemm;

    WSP_GGML_ASSERT(ne0 == ne01);
    WSP_GGML_ASSERT(ne1 == ne11);
    WSP_GGML_ASSERT(ne2 == ne12);
    WSP_GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    WSP_GGML_ASSERT(nb00 == wsp_ggml_type_size(type));
    WSP_GGML_ASSERT(nb10 == wsp_ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    WSP_GGML_ASSERT(nb0 == sizeof(float));
    WSP_GGML_ASSERT(nb0 <= nb1);
    WSP_GGML_ASSERT(nb1 <= nb2);
    WSP_GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

#if WSP_GGML_USE_LLAMAFILE
    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const bool src1_cont = wsp_ggml_is_contiguous(src1);

    if (src1_cont) {
        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(ne01, ne11, ne00/wsp_ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/wsp_ggml_type_size(src0->type),
                                     (const char *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/wsp_ggml_type_size(src1->type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/wsp_ggml_type_size(dst->type),
                                     ith, nth,
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    goto UseGgmlGemm1;
        return;
    }
UseGgmlGemm1:;
#endif

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw1 = wsp_ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_F32);

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                int64_t i11_processed = 0;
                if ((wsp_ggml_n_dims(src1) == 2) && from_float_to_mat && gemm) {
                    for (int64_t i11 = ith * 4; i11 < ne11 - ne11 % 4; i11 += nth * 4) {
                        from_float_to_mat((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                                          (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                          4, ne10, blck_size_interleave);
                    }
                    i11_processed = ne11 - ne11 % 4;
                }
                for (int64_t i11 = i11_processed + ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                           (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                           ne10);
                }
            }
        }
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    wsp_ggml_barrier(params->threadpool);

#if WSP_GGML_USE_LLAMAFILE
    if (src1->type != vec_dot_type) {
        const void* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = wsp_ggml_row_size(vec_dot_type, ne10);

        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(ne01, ne11, ne00/wsp_ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/wsp_ggml_type_size(src0->type),
                                     (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/wsp_ggml_type_size(vec_dot_type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/wsp_ggml_type_size(dst->type),
                                     ith, nth,
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    goto UseGgmlGemm2;
        return;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const int64_t nr1 = ne1 * ne2 * ne3;

    // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
    int64_t num_rows_per_vec_dot = vec_dot_num_rows;
    // TODO: currently the mmla kernels support only even numbered rows/cols.
    // this check can be removed once they are extended to support odd numbered rows/cols too
    if ((nr0 % 2 != 0) || (ne11 % 2 != 0)) {
        num_rows_per_vec_dot = 1;
    }

    // Now select a reasonable chunk size.
    int chunk_size = 16;

    // We need to step up the size if it's small
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggerganov/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    if (nchunk0 * nchunk1 < nth * 4 || wsp_ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    if ((wsp_ggml_n_dims(src0) == 2) && gemv) {
        const void * src1_wdata      = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t src1_col_stride = wsp_ggml_is_contiguous(src1) || src1->type != vec_dot_type ? wsp_ggml_row_size(vec_dot_type, ne10) : nb11;
        int64_t src0_start = (ith * ne01) / nth;
        int64_t src0_end   = ((ith + 1) * ne01) / nth;
        src0_start = (src0_start % matmul_num_cols) ? src0_start + matmul_num_cols - (src0_start % matmul_num_cols): src0_start;
        src0_end   = (src0_end   % matmul_num_cols) ? src0_end   + matmul_num_cols - (src0_end   % matmul_num_cols): src0_end;
        if (src0_start >= src0_end) return;

        // If there are more than three rows in src1, use gemm; otherwise, use gemv.
        if (gemm && (ne11 > 3)) {
            gemm(ne00, (float *)((char *) dst->data) + src0_start, ne01, (const char *) src0->data + src0_start * nb01,
                 (const char *) src1_wdata, ne11 - ne11 % 4, src0_end - src0_start);
        }
        for (int iter = gemm ? ne11 - ne11 % 4 : 0; iter < ne11; iter++) {
            gemv(ne00, (float *)((char *) dst->data + (iter * nb1)) + src0_start, ne01,
                 (const char *) src0->data + src0_start * nb01, (const char *) src1_wdata + (src1_col_stride * iter), 1,
                 src0_end - src0_start);
        }
        return;
    }

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = ith;

    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        wsp_ggml_compute_forward_mul_mat_one_chunk(params, dst, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// wsp_ggml_compute_forward_mul_mat_id

static void wsp_ggml_compute_forward_mul_mat_id(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];
    const struct wsp_ggml_tensor * ids = dst->src[2];

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum wsp_ggml_type type = src0->type;

    const bool src1_cont = wsp_ggml_is_contiguous(src1);

    wsp_ggml_vec_dot_t    const vec_dot         = type_traits_cpu[type].vec_dot;
    enum wsp_ggml_type    const vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    wsp_ggml_from_float_t const from_float      = wsp_ggml_get_type_traits(vec_dot_type)->from_float;
    int64_t           const matmul_num_cols = type_traits_cpu[type].ncols;
    wsp_ggml_gemv_t       const gemv            = type_traits_cpu[type].gemv;

    // we don't support permuted src0 or src1
    WSP_GGML_ASSERT(nb00 == wsp_ggml_type_size(type));
    WSP_GGML_ASSERT(nb10 == wsp_ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    WSP_GGML_ASSERT(nb0 == sizeof(float));
    WSP_GGML_ASSERT(nb0 <= nb1);
    WSP_GGML_ASSERT(nb1 <= nb2);
    WSP_GGML_ASSERT(nb2 <= nb3);

    // row groups
    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_expert

    char * wdata_src1_end = (src1->type == vec_dot_type) ?
            (char *) params->wdata :
            (char *) params->wdata + WSP_GGML_PAD(wsp_ggml_row_size(vec_dot_type, wsp_ggml_nelements(src1)), sizeof(int64_t));

    struct mmid_row_mapping {
        int32_t i1;
        int32_t i2;
    };

    int64_t * matrix_row_counts = (int64_t *) (wdata_src1_end); // [n_as]
    struct mmid_row_mapping * matrix_rows = (struct mmid_row_mapping *)(matrix_row_counts + n_as); // [n_as][ne11]

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw1 = wsp_ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_F32);

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
    }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ne12 + (i1)]

    if (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));

        // group rows by src0 matrix
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                assert(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    wsp_ggml_barrier(params->threadpool);

    // compute each matrix multiplication in sequence
    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        const char * src0_cur = (const char *) src0->data + cur_a*nb02;

        const void * wdata    = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = wsp_ggml_row_size(vec_dot_type, ne10);

        const int64_t nr0 = ne01; // src0 rows
        const int64_t nr1 = cne1; // src1 rows

        if (((wsp_ggml_n_dims(src0) - 1) == 2) && gemv) {
            int64_t src0_cur_start = (ith * ne01) / nth;
            int64_t src0_cur_end   = ((ith + 1) * ne01) / nth;
            src0_cur_start = (src0_cur_start % matmul_num_cols) ? src0_cur_start + matmul_num_cols - (src0_cur_start % matmul_num_cols): src0_cur_start;
            src0_cur_end   = (src0_cur_end % matmul_num_cols) ? src0_cur_end + matmul_num_cols - (src0_cur_end % matmul_num_cols): src0_cur_end;
            if (src0_cur_start >= src0_cur_end) return;

            for (int ir1 = 0; ir1 < nr1; ir1++) {
                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);
                const int id       = row_mapping.i1; // selected expert index

                const int64_t  i11 = id % ne11;
                const int64_t  i12 = row_mapping.i2; // row index in src1

                const int64_t  i1 = id;  // selected expert index
                const int64_t  i2 = i12; // row

                const char * src1_col = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11        + i12 * ne11) * row_size
                    : (i11 * nb11 + i12 * nb12));

                gemv(ne00, (float *)((char *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start, ne01,
                     (const char *) src0_cur + src0_cur_start * nb01, src1_col, 1, src0_cur_end - src0_cur_start);
            }
            continue;
        }

        // distribute the thread work across the inner or outer loop based on which one is larger

        const int64_t nth0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        const int64_t nth1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows

        const int64_t ith0 = ith % nth0;
        const int64_t ith1 = ith / nth0;

        const int64_t dr0 = (nr0 + nth0 - 1)/nth0;
        const int64_t dr1 = (nr1 + nth1 - 1)/nth1;

        const int64_t ir010 = dr0*ith0;
        const int64_t ir011 = MIN(ir010 + dr0, nr0);

        const int64_t ir110 = dr1*ith1;
        const int64_t ir111 = MIN(ir110 + dr1, nr1);

        // threads with no work simply yield (not sure if it helps)
        //if (ir010 >= ir011 || ir110 >= ir111) {
        //    sched_yield();
        //    continue;
        //}

        // block-tiling attempt
        const int64_t blck_0 = 16;
        const int64_t blck_1 = 16;

        // attempt to reduce false-sharing (does not seem to make a difference)
        float tmp[16];

        for (int64_t iir1 = ir110; iir1 < ir111; iir1 += blck_1) {
            for (int64_t iir0 = ir010; iir0 < ir011; iir0 += blck_0) {
                for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir111; ++ir1) {
                    const int64_t _i12 = ir1; // logical row index for this expert

                    struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                    const int id       = row_mapping.i1; // selected expert index

                    const int64_t  i11 = id % ne11;
                    const int64_t  i12 = row_mapping.i2; // row index in src1

                    const int64_t  i1 = id;  // selected expert index
                    const int64_t  i2 = i12; // row

                    // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                    //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                    //       the original src1 data pointer, so we should index using the indices directly
                    // TODO: this is a bit of a hack, we should probably have a better way to handle this
                    const char * src1_col = (const char *) wdata +
                        (src1_cont || src1->type != vec_dot_type
                        ? (i11      + i12*ne11)*row_size
                        : (i11*nb11 + i12*nb12));

                    float * dst_col = (float *) ((char *) dst->data + (i1*nb1 + i2*nb2));

                    //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir011; ++ir0) {
                    //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                    //}

                    for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir011; ++ir0) {
                        vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                    }

                    memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir011) - iir0)*sizeof(float));
                }
            }
        }
    }

#undef MMID_MATRIX_ROW
}

// wsp_ggml_compute_forward_out_prod

static void wsp_ggml_compute_forward_out_prod_f32(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    WSP_GGML_ASSERT(dst->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT(src0->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_F32);

    const int ith = params->ith;
    const int nth = params->nth;

    WSP_GGML_ASSERT(ne0  == ne00);
    WSP_GGML_ASSERT(ne1  == ne10);
    WSP_GGML_ASSERT(ne2  == ne02);
    WSP_GGML_ASSERT(ne02 == ne12);
    WSP_GGML_ASSERT(ne3  == ne13);
    WSP_GGML_ASSERT(ne03 == ne13);

    // we don't support permuted src0 or src1
    WSP_GGML_ASSERT(nb00 == sizeof(float));

    // dst cannot be transposed or permuted
    WSP_GGML_ASSERT(nb0 == sizeof(float));
    // WSP_GGML_ASSERT(nb0 <= nb1);
    // WSP_GGML_ASSERT(nb1 <= nb2);
    // WSP_GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    if (ith == 0) {
        wsp_ggml_vec_set_f32(ne0*ne1*ne2*ne3, dst->data, 0);
    }
    wsp_ggml_barrier(params->threadpool);

    // dst[:,:,:,:] = 0
    // for i2,i3:
    //   for i1:
    //     for i01:
    //       for i0:
    //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

    // parallelize by last three dimensions

    // total rows in dst
    const int64_t nr = ne1*ne2*ne3;

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    // block-tiling attempt
    const int64_t blck_0 = MAX(WSP_GGML_VEC_MAD_UNROLL, 32);
    const int64_t blck_1 = 16;

    for (int64_t bir = ir0; bir < ir1; bir += blck_1) {
        const int64_t bir1 = MIN(bir + blck_1, ir1);
        for (int64_t bi01 = 0; bi01 < ne01; bi01 += blck_0) {
            const int64_t bne01 = MIN(bi01 + blck_0, ne01);
            for (int64_t ir = bir; ir < bir1; ++ir) {
                // dst indices
                const int64_t i3 = ir/(ne2*ne1);
                const int64_t i2 = (ir - i3*ne2*ne1)/ne1;
                const int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

                const int64_t i02 = i2;
                const int64_t i03 = i3;

                //const int64_t i10 = i1;
                const int64_t i12 = i2;
                const int64_t i13 = i3;

#if WSP_GGML_VEC_MAD_UNROLL > 2
                const int64_t bne01_unroll = bne01 - (bne01 % WSP_GGML_VEC_MAD_UNROLL);
                for (int64_t i01 = bi01; i01 < bne01_unroll; i01 += WSP_GGML_VEC_MAD_UNROLL) {
                    const int64_t i11 = i01;

                    float * s0 = (float *) ((char *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    float * s1 = (float *) ((char *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    float * d  = (float *) ((char *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

                    wsp_ggml_vec_mad_f32_unroll(ne0, nb01, nb11, d, s0, s1);
                }
                for (int64_t i01 = bne01_unroll; i01 < bne01; ++i01) {
                    const int64_t i11 = i01;

                    float * s0 = (float *) ((char *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    float * s1 = (float *) ((char *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    float * d  = (float *) ((char *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

                    wsp_ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#else
                for (int64_t i01 = bi01; i01 < bne01; ++i01) {
                    const int64_t i11 = i01;

                    float * s0 = (float *) ((char *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    float * s1 = (float *) ((char *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    float * d  = (float *) ((char *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

                    wsp_ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#endif
            }
        }
    }
}

static void wsp_ggml_compute_forward_out_prod_q_f32(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_TENSOR_BINARY_OP_LOCALS;

    const int ith = params->ith;
    const int nth = params->nth;

    const enum wsp_ggml_type type = src0->type;
    wsp_ggml_to_float_t const wsp_dewsp_quantize_row_q = wsp_ggml_get_type_traits(type)->to_float;

    WSP_GGML_ASSERT(ne02 == ne12);
    WSP_GGML_ASSERT(ne03 == ne13);
    WSP_GGML_ASSERT(ne2  == ne12);
    WSP_GGML_ASSERT(ne3  == ne13);

    // we don't support permuted src0 dim0
    WSP_GGML_ASSERT(nb00 == wsp_ggml_type_size(type));

    // dst dim0 cannot be transposed or permuted
    WSP_GGML_ASSERT(nb0 == sizeof(float));
    // WSP_GGML_ASSERT(nb0 <= nb1);
    // WSP_GGML_ASSERT(nb1 <= nb2);
    // WSP_GGML_ASSERT(nb2 <= nb3);

    WSP_GGML_ASSERT(ne0 == ne00);
    WSP_GGML_ASSERT(ne1 == ne10);
    WSP_GGML_ASSERT(ne2 == ne02);
    WSP_GGML_ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    if (ith == 0) {
        wsp_ggml_vec_set_f32(ne0*ne1*ne2*ne3, dst->data, 0);
    }
    wsp_ggml_barrier(params->threadpool);

    // parallelize by last three dimensions

    // total rows in dst
    const int64_t nr = ne1*ne2*ne3;

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    // dst[:,:,:,:] = 0
    // for i2,i3:
    //   for i1:
    //     for i01:
    //       for i0:
    //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

    float * wdata = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32) * ith;

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        // dst indices
        const int64_t i3 = ir/(ne2*ne1);
        const int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        const int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        const int64_t i02 = i2;
        const int64_t i03 = i3;

        //const int64_t i10 = i1;
        const int64_t i12 = i2;
        const int64_t i13 = i3;

        for (int64_t i01 = 0; i01 < ne01; ++i01) {
            const int64_t i11 = i01;

            float * s0 = (float *) ((char *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
            float * s1 = (float *) ((char *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            float * d  = (float *) ((char *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

            wsp_dewsp_quantize_row_q(s0, wdata, ne0);
            wsp_ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

static void wsp_ggml_compute_forward_out_prod(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_Q4_0:
        case WSP_GGML_TYPE_Q4_1:
        case WSP_GGML_TYPE_Q5_0:
        case WSP_GGML_TYPE_Q5_1:
        case WSP_GGML_TYPE_Q8_0:
        case WSP_GGML_TYPE_Q2_K:
        case WSP_GGML_TYPE_Q3_K:
        case WSP_GGML_TYPE_Q4_K:
        case WSP_GGML_TYPE_Q5_K:
        case WSP_GGML_TYPE_Q6_K:
        case WSP_GGML_TYPE_TQ1_0:
        case WSP_GGML_TYPE_TQ2_0:
        case WSP_GGML_TYPE_IQ2_XXS:
        case WSP_GGML_TYPE_IQ2_XS:
        case WSP_GGML_TYPE_IQ3_XXS:
        case WSP_GGML_TYPE_IQ1_S:
        case WSP_GGML_TYPE_IQ1_M:
        case WSP_GGML_TYPE_IQ4_NL:
        case WSP_GGML_TYPE_IQ4_XS:
        case WSP_GGML_TYPE_IQ3_S:
        case WSP_GGML_TYPE_IQ2_S:
        case WSP_GGML_TYPE_Q4_0_4_4:
        case WSP_GGML_TYPE_Q4_0_4_8:
        case WSP_GGML_TYPE_Q4_0_8_8:
            {
                wsp_ggml_compute_forward_out_prod_q_f32(params, dst);
            } break;
        case WSP_GGML_TYPE_F16:
            {
                WSP_GGML_ABORT("fatal error"); // todo
                // wsp_ggml_compute_forward_out_prod_f16_f32(params, dst);
            }
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_out_prod_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_scale

static void wsp_ggml_compute_forward_scale_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(src0));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(dst));
    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, dst));

    // scale factor
    float v;
    memcpy(&v, dst->op_params, sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = wsp_ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    const size_t nb01 = src0->nb[1];

    const size_t nb1 = dst->nb[1];

    for (int i1 = ir0; i1 < ir1; i1++) {
        if (dst->data != src0->data) {
            // src0 is same shape as dst => same indices
            memcpy((char *)dst->data + i1*nb1, (char *)src0->data + i1*nb01, nc * sizeof(float));
        }
        wsp_ggml_vec_scale_f32(nc, (float *) ((char *) dst->data + i1*nb1), v);
    }
}

static void wsp_ggml_compute_forward_scale(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_scale_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_set

static void wsp_ggml_compute_forward_set_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, dst));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(dst) && wsp_ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    bool   inplace = (bool) ((int32_t *) dst->op_params)[4];

    if (!inplace) {
        if (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((char *)  dst->data),
                ((char *) src0->data),
                wsp_ggml_nbytes(dst));
        }
        wsp_ggml_barrier(params->threadpool);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr = wsp_ggml_nrows(src1);
    const int nc = src1->ne[0];

    WSP_GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    WSP_GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    const size_t nb0 = wsp_ggml_element_size(src0);

    const int im0 = (ne10 == 0 ? 0 : ne10-1);
    const int im1 = (ne11 == 0 ? 0 : ne11-1);
    const int im2 = (ne12 == 0 ? 0 : ne12-1);
    const int im3 = (ne13 == 0 ? 0 : ne13-1);

    WSP_GGML_ASSERT(offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= wsp_ggml_nbytes(dst));

    WSP_GGML_ASSERT(nb10 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        const int i3 = ir/(ne12*ne11);
        const int i2 = (ir - i3*ne12*ne11)/ne11;
        const int i1 = (ir - i3*ne12*ne11 - i2*ne11);

        wsp_ggml_vec_cpy_f32(nc,
                (float *) ((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (float *) ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

static void wsp_ggml_compute_forward_set(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_set_f32(params, dst);
            } break;
        case WSP_GGML_TYPE_F16:
        case WSP_GGML_TYPE_BF16:
        case WSP_GGML_TYPE_Q4_0:
        case WSP_GGML_TYPE_Q4_1:
        case WSP_GGML_TYPE_Q5_0:
        case WSP_GGML_TYPE_Q5_1:
        case WSP_GGML_TYPE_Q8_0:
        case WSP_GGML_TYPE_Q8_1:
        case WSP_GGML_TYPE_Q2_K:
        case WSP_GGML_TYPE_Q3_K:
        case WSP_GGML_TYPE_Q4_K:
        case WSP_GGML_TYPE_Q5_K:
        case WSP_GGML_TYPE_Q6_K:
        case WSP_GGML_TYPE_TQ1_0:
        case WSP_GGML_TYPE_TQ2_0:
        case WSP_GGML_TYPE_IQ2_XXS:
        case WSP_GGML_TYPE_IQ2_XS:
        case WSP_GGML_TYPE_IQ3_XXS:
        case WSP_GGML_TYPE_IQ1_S:
        case WSP_GGML_TYPE_IQ1_M:
        case WSP_GGML_TYPE_IQ4_NL:
        case WSP_GGML_TYPE_IQ4_XS:
        case WSP_GGML_TYPE_IQ3_S:
        case WSP_GGML_TYPE_IQ2_S:
        case WSP_GGML_TYPE_Q4_0_4_4:
        case WSP_GGML_TYPE_Q4_0_4_8:
        case WSP_GGML_TYPE_Q4_0_8_8:
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_cpy

static void wsp_ggml_compute_forward_cpy(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {
    wsp_ggml_compute_forward_dup(params, dst);
}

// wsp_ggml_compute_forward_cont

static void wsp_ggml_compute_forward_cont(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {
    wsp_ggml_compute_forward_dup(params, dst);
}

// wsp_ggml_compute_forward_reshape

static void wsp_ggml_compute_forward_reshape(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {
    // NOP
    UNUSED(params);
    UNUSED(dst);
}

// wsp_ggml_compute_forward_view

static void wsp_ggml_compute_forward_view(
        const struct wsp_ggml_compute_params * params,
        const struct wsp_ggml_tensor * dst) {
    // NOP
    UNUSED(params);
    UNUSED(dst);
}

// wsp_ggml_compute_forward_permute

static void wsp_ggml_compute_forward_permute(
        const struct wsp_ggml_compute_params * params,
        const struct wsp_ggml_tensor * dst) {
    // NOP
    UNUSED(params);
    UNUSED(dst);
}

// wsp_ggml_compute_forward_transpose

static void wsp_ggml_compute_forward_transpose(
        const struct wsp_ggml_compute_params * params,
        const struct wsp_ggml_tensor * dst) {
    // NOP
    UNUSED(params);
    UNUSED(dst);
}

// wsp_ggml_compute_forward_get_rows

static void wsp_ggml_compute_forward_get_rows_q(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = wsp_ggml_nelements(src1);

    const enum wsp_ggml_type type = src0->type;
    wsp_ggml_to_float_t const wsp_dewsp_quantize_row_q = wsp_ggml_get_type_traits(type)->to_float;

    assert(ne0  == nc);
    assert(ne02 == ne11);
    assert(nb00 == wsp_ggml_type_size(type));
    assert(wsp_ggml_nrows(dst) == nr);

    const int ith = params->ith;
    const int nth = params->nth;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int64_t i = ir0; i < ir1; ++i) {
        const int64_t i12 = i/(ne11*ne10);
        const int64_t i11 = (i - i12*ne11*ne10)/ne10;
        const int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        const int64_t i01 = *(int32_t *) ((char *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        WSP_GGML_ASSERT(i01 >= 0 && i01 < ne01);

        wsp_dewsp_quantize_row_q(
                (const void *) ((char *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                     (float *) ((char *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

static void wsp_ggml_compute_forward_get_rows_f16(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = wsp_ggml_nelements(src1);

    assert(ne0  == nc);
    assert(ne02 == ne11);
    assert(nb00 == sizeof(wsp_ggml_fp16_t));
    assert(wsp_ggml_nrows(dst) == nr);

    const int ith = params->ith;
    const int nth = params->nth;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int64_t i = ir0; i < ir1; ++i) {
        const int64_t i12 = i/(ne11*ne10);
        const int64_t i11 = (i - i12*ne11*ne10)/ne10;
        const int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        const int64_t i01 = *(int32_t *) ((char *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        WSP_GGML_ASSERT(i01 >= 0 && i01 < ne01);

        wsp_ggml_fp16_to_fp32_row(
                (const void *) ((char *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                     (float *) ((char *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

static void wsp_ggml_compute_forward_get_rows_bf16(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = wsp_ggml_nelements(src1);

    assert(ne0  == nc);
    assert(ne02 == ne11);
    assert(nb00 == sizeof(wsp_ggml_bf16_t));
    assert(wsp_ggml_nrows(dst) == nr);

    const int ith = params->ith;
    const int nth = params->nth;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int64_t i = ir0; i < ir1; ++i) {
        const int64_t i12 = i/(ne11*ne10);
        const int64_t i11 = (i - i12*ne11*ne10)/ne10;
        const int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        const int64_t i01 = *(int32_t *) ((char *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        WSP_GGML_ASSERT(i01 >= 0 && i01 < ne01);

        wsp_ggml_bf16_to_fp32_row(
                (const void *) ((char *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                     (float *) ((char *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

static void wsp_ggml_compute_forward_get_rows_f32(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = wsp_ggml_nelements(src1);

    assert(ne0  == nc);
    assert(ne02 == ne11);
    assert(nb00 == sizeof(float));
    assert(wsp_ggml_nrows(dst) == nr);

    const int ith = params->ith;
    const int nth = params->nth;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int64_t i = ir0; i < ir1; ++i) {
        const int64_t i12 = i/(ne11*ne10);
        const int64_t i11 = (i - i12*ne11*ne10)/ne10;
        const int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        const int64_t i01 = *(int32_t *) ((char *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        WSP_GGML_ASSERT(i01 >= 0 && i01 < ne01);

        wsp_ggml_vec_cpy_f32(nc,
                (float *) ((char *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3),
                (float *) ((char *) src0->data + i01*nb01 + i11*nb02 + i12*nb03));
    }
}

static void wsp_ggml_compute_forward_get_rows(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_Q4_0:
        case WSP_GGML_TYPE_Q4_1:
        case WSP_GGML_TYPE_Q5_0:
        case WSP_GGML_TYPE_Q5_1:
        case WSP_GGML_TYPE_Q8_0:
        case WSP_GGML_TYPE_Q8_1:
        case WSP_GGML_TYPE_Q2_K:
        case WSP_GGML_TYPE_Q3_K:
        case WSP_GGML_TYPE_Q4_K:
        case WSP_GGML_TYPE_Q5_K:
        case WSP_GGML_TYPE_Q6_K:
        case WSP_GGML_TYPE_TQ1_0:
        case WSP_GGML_TYPE_TQ2_0:
        case WSP_GGML_TYPE_IQ2_XXS:
        case WSP_GGML_TYPE_IQ2_XS:
        case WSP_GGML_TYPE_IQ3_XXS:
        case WSP_GGML_TYPE_IQ1_S:
        case WSP_GGML_TYPE_IQ1_M:
        case WSP_GGML_TYPE_IQ4_NL:
        case WSP_GGML_TYPE_IQ4_XS:
        case WSP_GGML_TYPE_IQ3_S:
        case WSP_GGML_TYPE_IQ2_S:
        case WSP_GGML_TYPE_Q4_0_4_4:
        case WSP_GGML_TYPE_Q4_0_4_8:
        case WSP_GGML_TYPE_Q4_0_8_8:
            {
                wsp_ggml_compute_forward_get_rows_q(params, dst);
            } break;
        case WSP_GGML_TYPE_F16:
            {
                wsp_ggml_compute_forward_get_rows_f16(params, dst);
            } break;
        case WSP_GGML_TYPE_BF16:
            {
                wsp_ggml_compute_forward_get_rows_bf16(params, dst);
            } break;
        case WSP_GGML_TYPE_F32:
        case WSP_GGML_TYPE_I32:
            {
                wsp_ggml_compute_forward_get_rows_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }

    //static bool first = true;
    //printf("ne0 = %d, ne1 = %d, ne2 = %d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
    //if (first) {
    //    first = false;
    //} else {
    //    for (int k = 0; k < dst->ne[1]; ++k) {
    //        for (int j = 0; j < dst->ne[0]/16; ++j) {
    //            for (int i = 0; i < 16; ++i) {
    //                printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
    //            }
    //            printf("\n");
    //        }
    //        printf("\n");
    //    }
    //    printf("\n");
    //    exit(0);
    //}
}

// wsp_ggml_compute_forward_get_rows_back

static void wsp_ggml_compute_forward_get_rows_back_f32_f16(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    if (params->ith != 0) {
        return;
    }

    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(dst));

    // wsp_ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->data, 0, wsp_ggml_nbytes(dst));

    const int nc = src0->ne[0];
    const int nr = wsp_ggml_nelements(src1);

    WSP_GGML_ASSERT( dst->ne[0] == nc);
    WSP_GGML_ASSERT(src0->nb[0] == sizeof(wsp_ggml_fp16_t));

    for (int i = 0; i < nr; ++i) {
        const int r = ((int32_t *) src1->data)[i];

        for (int j = 0; j < nc; ++j) {
            wsp_ggml_fp16_t v = ((wsp_ggml_fp16_t *) ((char *) src0->data + i*src0->nb[1]))[j];
            ((float *) ((char *) dst->data + r*dst->nb[1]))[j] += WSP_GGML_FP16_TO_FP32(v);
        }
    }
}

static void wsp_ggml_compute_forward_get_rows_back_f32(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    if (params->ith != 0) {
        return;
    }

    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(dst));

    // wsp_ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->data, 0, wsp_ggml_nbytes(dst));

    const int nc = src0->ne[0];
    const int nr = wsp_ggml_nelements(src1);

    WSP_GGML_ASSERT( dst->ne[0] == nc);
    WSP_GGML_ASSERT(src0->nb[0] == sizeof(float));

    for (int i = 0; i < nr; ++i) {
        const int r = ((int32_t *) src1->data)[i];

        wsp_ggml_vec_add_f32(nc,
                (float *) ((char *)  dst->data + r*dst->nb[1]),
                (float *) ((char *)  dst->data + r*dst->nb[1]),
                (float *) ((char *) src0->data + i*src0->nb[1]));
    }
}

static void wsp_ggml_compute_forward_get_rows_back(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F16:
            {
                wsp_ggml_compute_forward_get_rows_back_f32_f16(params, dst);
            } break;
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_get_rows_back_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }

    //static bool first = true;
    //printf("ne0 = %d, ne1 = %d, ne2 = %d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
    //if (first) {
    //    first = false;
    //} else {
    //    for (int k = 0; k < dst->ne[1]; ++k) {
    //        for (int j = 0; j < dst->ne[0]/16; ++j) {
    //            for (int i = 0; i < 16; ++i) {
    //                printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
    //            }
    //            printf("\n");
    //        }
    //        printf("\n");
    //    }
    //    printf("\n");
    //    exit(0);
    //}
}

// wsp_ggml_compute_forward_diag

static void wsp_ggml_compute_forward_diag_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    // TODO: handle transposed/permuted matrices

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    WSP_GGML_ASSERT(ne00 == ne0);
    WSP_GGML_ASSERT(ne00 == ne1);
    WSP_GGML_ASSERT(ne01 == 1);
    WSP_GGML_ASSERT(ne02 == ne2);
    WSP_GGML_ASSERT(ne03 == ne3);

    WSP_GGML_ASSERT(nb00 == sizeof(float));
    WSP_GGML_ASSERT(nb0  == sizeof(float));

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            for (int i1 = 0; i1 < ne1; i1++) {
                float * d = (float *)((char *)  dst->data + i3*nb3  + i2*nb2 + i1*nb1);
                float * s = (float *)((char *) src0->data + i3*nb03 + i2*nb02);
                for (int i0 = 0; i0 < i1; i0++) {
                    d[i0] = 0;
                }
                d[i1] = s[i1];
                for (int i0 = i1+1; i0 < ne0; i0++) {
                    d[i0] = 0;
                }
            }
        }
    }
}

static void wsp_ggml_compute_forward_diag(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_diag_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_diag_mask_inf

static void wsp_ggml_compute_forward_diag_mask_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst,
        const float value) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    const int ith = params->ith;
    const int nth = params->nth;

    const int  n_past  = ((int32_t *) dst->op_params)[0];
    const bool inplace = src0->data == dst->data;

    WSP_GGML_ASSERT(n_past >= 0);

    if (!inplace) {
        if (ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            WSP_GGML_ASSERT(wsp_ggml_nelements(dst) == wsp_ggml_nelements(src0));
            WSP_GGML_ASSERT(wsp_ggml_is_contiguous(dst) && wsp_ggml_is_contiguous(src0));
            memcpy(
                ((char *)  dst->data),
                ((char *) src0->data),
                wsp_ggml_nbytes(dst));
        }
        wsp_ggml_barrier(params->threadpool);
    }

    // TODO: handle transposed/permuted matrices

    const int n  = wsp_ggml_nrows(src0);
    const int nc = src0->ne[0];
    const int nr = src0->ne[1];
    const int nz = n/nr;

    WSP_GGML_ASSERT( dst->nb[0] == sizeof(float));
    WSP_GGML_ASSERT(src0->nb[0] == sizeof(float));

    for (int k = 0; k < nz; k++) {
        for (int j = ith; j < nr; j += nth) {
            for (int i = n_past; i < nc; i++) {
                if (i > n_past + j) {
                    *(float *)((char *) dst->data + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = value;
                }
            }
        }
    }
}

static void wsp_ggml_compute_forward_diag_mask_inf(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_diag_mask_f32(params, dst, -INFINITY);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

static void wsp_ggml_compute_forward_diag_mask_zero(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_diag_mask_f32(params, dst, 0);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_soft_max

static void wsp_ggml_compute_forward_soft_max_f32(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    assert(wsp_ggml_is_contiguous(dst));
    assert(wsp_ggml_are_same_shape(src0, dst));

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));

    // TODO: handle transposed/permuted matrices

    const int ith = params->ith;
    const int nth = params->nth;

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    //const int64_t ne11 = src1 ? src1->ne[1] : 1;

    // TODO: is this supposed to be ceil instead of floor?
    //       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
    const uint32_t n_head      = ne02;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const int nc = src0->ne[0];
    const int nr = wsp_ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    float * wp = (float *) params->wdata + (nc + CACHE_LINE_SIZE_F32) * ith;

    const bool use_f16 = (src1 && src1->type == WSP_GGML_TYPE_F16);

    for (int i1 = ir0; i1 < ir1; i1++) {
        // ALiBi
        const uint32_t h = (i1/ne01)%ne02; // head
        const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        float * sp = (float *)((char *) src0->data + i1*src0->nb[1]);
        float * dp = (float *)((char *)  dst->data +  i1*dst->nb[1]);

        // broadcast the mask across rows
        wsp_ggml_fp16_t * mp_f16 = src1 ? (wsp_ggml_fp16_t *)((char *) src1->data) + (i1%ne01)*ne00 : NULL;
        float       * mp_f32 = src1 ? (float       *)((char *) src1->data) + (i1%ne01)*ne00 : NULL;

        wsp_ggml_vec_cpy_f32  (nc, wp, sp);
        wsp_ggml_vec_scale_f32(nc, wp, scale);
        if (mp_f32) {
            if (use_f16) {
                for (int i = 0; i < nc; ++i) {
                    wp[i] += slope*WSP_GGML_FP16_TO_FP32(mp_f16[i]);
                }
            } else {
                for (int i = 0; i < nc; ++i) {
                    wp[i] += slope*mp_f32[i];
                }
            }
        }

#ifndef NDEBUG
        for (int i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            assert(!isnan(wp[i]));
        }
#endif

        float max = -INFINITY;
        wsp_ggml_vec_max_f32(nc, &max, wp);

        wsp_ggml_float sum = wsp_ggml_vec_soft_max_f32(nc, dp, wp, max);
        assert(sum > 0.0);

        sum = 1.0/sum;
        wsp_ggml_vec_scale_f32(nc, dp, sum);

#ifndef NDEBUG
        for (int i = 0; i < nc; ++i) {
            assert(!isnan(dp[i]));
            assert(!isinf(dp[i]));
        }
#endif
    }
}

static void wsp_ggml_compute_forward_soft_max(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_soft_max_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}


// wsp_ggml_compute_forward_soft_max_back

static void wsp_ggml_compute_forward_soft_max_back_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(src0));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(src1));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(dst));
    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, dst));
    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src1, dst));

    // TODO: handle transposed/permuted matrices

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = wsp_ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float *dy = (float *)((char *) src0->data + i1*src0->nb[1]);
        float *y  = (float *)((char *) src1->data + i1*src1->nb[1]);
        float *dx = (float *)((char *) dst->data  + i1*dst->nb[1]);

#ifndef NDEBUG
        for (int i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            assert(!isnan(dy[i]));
            assert(!isnan(y[i]));
        }
#endif
        // Jii = yi - yi*yi
        // Jij = -yi*yj
        // J = diag(y)-y.T*y
        // dx = J * dy
        // dxk = sum_i(Jki * dyi)
        // dxk = sum_i(-yk*yi * dyi) - (-yk*yk)*dyk + (yk - yk*yk)*dyk
        // dxk = sum_i(-yk*yi * dyi) + yk*yk*dyk + yk*dyk - yk*yk*dyk
        // dxk = sum_i(-yk*yi * dyi) + yk*dyk
        // dxk = -yk * sum_i(yi * dyi) + yk*dyk
        // dxk = -yk * dot(y, dy) + yk*dyk
        // dxk = yk * (- dot(y, dy) + dyk)
        // dxk = yk * (dyk - dot(y, dy))
        //
        // post-order:
        // dot_y_dy := dot(y, dy)
        // dx := dy
        // dx := dx - dot_y_dy
        // dx := dx * y

        // linear runtime, no additional memory
        float dot_y_dy = 0;
        wsp_ggml_vec_dot_f32 (nc, &dot_y_dy, 0, y, 0, dy, 0, 1);
        wsp_ggml_vec_cpy_f32 (nc, dx, dy);
        wsp_ggml_vec_acc1_f32(nc, dx, -dot_y_dy);
        wsp_ggml_vec_mul_f32 (nc, dx, dx, y);

#ifndef NDEBUG
        for (int i = 0; i < nc; ++i) {
            assert(!isnan(dx[i]));
            assert(!isinf(dx[i]));
        }
#endif
    }
}

static void wsp_ggml_compute_forward_soft_max_back(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_soft_max_back_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_clamp

static void wsp_ggml_compute_forward_clamp_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    float min;
    float max;
    memcpy(&min, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max, (float *) dst->op_params + 1, sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int n  = wsp_ggml_nrows(src0);
    const int nc = src0->ne[0];

    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];

    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];

    WSP_GGML_ASSERT( nb0 == sizeof(float));
    WSP_GGML_ASSERT(nb00 == sizeof(float));

    for (int j = ith; j < n; j += nth) {
        float * dst_ptr  = (float *) ((char *)  dst->data + j*nb1);
        float * src0_ptr = (float *) ((char *) src0->data + j*nb01);

        for (int i = 0; i < nc; i++) {
            dst_ptr[i] = MAX(MIN(src0_ptr[i], max), min);
        }
    }
}

static void wsp_ggml_compute_forward_clamp(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_clamp_f32(params, dst);
            } break;
        case WSP_GGML_TYPE_F16:
        case WSP_GGML_TYPE_BF16:
        case WSP_GGML_TYPE_Q4_0:
        case WSP_GGML_TYPE_Q4_1:
        case WSP_GGML_TYPE_Q5_0:
        case WSP_GGML_TYPE_Q5_1:
        case WSP_GGML_TYPE_Q8_0:
        case WSP_GGML_TYPE_Q8_1:
        case WSP_GGML_TYPE_Q2_K:
        case WSP_GGML_TYPE_Q3_K:
        case WSP_GGML_TYPE_Q4_K:
        case WSP_GGML_TYPE_Q5_K:
        case WSP_GGML_TYPE_Q6_K:
        case WSP_GGML_TYPE_TQ1_0:
        case WSP_GGML_TYPE_TQ2_0:
        case WSP_GGML_TYPE_IQ2_XXS:
        case WSP_GGML_TYPE_IQ2_XS:
        case WSP_GGML_TYPE_IQ3_XXS:
        case WSP_GGML_TYPE_IQ1_S:
        case WSP_GGML_TYPE_IQ1_M:
        case WSP_GGML_TYPE_IQ4_NL:
        case WSP_GGML_TYPE_IQ4_XS:
        case WSP_GGML_TYPE_IQ3_S:
        case WSP_GGML_TYPE_IQ2_S:
        case WSP_GGML_TYPE_Q8_K:
        case WSP_GGML_TYPE_Q4_0_4_4:
        case WSP_GGML_TYPE_Q4_0_4_8:
        case WSP_GGML_TYPE_Q4_0_8_8:
        case WSP_GGML_TYPE_I8:
        case WSP_GGML_TYPE_I16:
        case WSP_GGML_TYPE_I32:
        case WSP_GGML_TYPE_I64:
        case WSP_GGML_TYPE_F64:
        case WSP_GGML_TYPE_COUNT:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_rope

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / MAX(0.001f, high - low);
    return 1 - MIN(1, MAX(0, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0, float ext_factor, float mscale,
    float * cos_theta, float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
static float wsp_ggml_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float)M_PI)) / (2 * logf(base));
}

static void wsp_ggml_rope_cache_init(
     float theta_base, float freq_scale, const float * freq_factors, float corr_dims[2], int64_t ne0, float ext_factor, float mscale,
     float * cache, float sin_sign, float theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    float theta = theta_base;
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

void wsp_ggml_rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    // start and end correction dims
    float start = floorf(wsp_ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end   =  ceilf(wsp_ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}

static void wsp_ggml_compute_forward_rope_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst,
        const bool forward) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];
    const struct wsp_ggml_tensor * src2 = dst->src[2];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    WSP_GGML_ASSERT(nb00 == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr = wsp_ggml_nrows(dst);

    WSP_GGML_ASSERT(n_dims <= ne0);
    WSP_GGML_ASSERT(n_dims % 2 == 0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    float corr_dims[2];
    wsp_ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const bool is_neox = mode & WSP_GGML_ROPE_TYPE_NEOX;

    const float * freq_factors = NULL;
    if (src2 != NULL) {
        WSP_GGML_ASSERT(src2->type == WSP_GGML_TYPE_F32);
        WSP_GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (const float *) src2->data;
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    const float sin_sign = forward ? 1.0f : -1.0f;

    const int32_t * pos = (const int32_t *) src1->data;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const int64_t p = pos[i2];

            float * cache = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32)*ith;
            wsp_ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);

            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;

                if (!is_neox) {
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];

                        const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                              float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        const float x0 = src[0];
                        const float x1 = src[1];

                        dst_data[0] = x0*cos_theta - x1*sin_theta;
                        dst_data[1] = x0*sin_theta + x1*cos_theta;
                    }
                } else {
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        const int64_t ic = i0/2;

                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];

                        const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                        float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

                        const float x0 = src[0];
                        const float x1 = src[n_dims/2];

                        dst_data[0]        = x0*cos_theta - x1*sin_theta;
                        dst_data[n_dims/2] = x0*sin_theta + x1*cos_theta;
                    }
                }

                for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                    const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                    float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                    dst_data[0] = src[0];
                    dst_data[1] = src[1];
                }
            }
        }
    }
}

// TODO: deduplicate f16/f32 code
static void wsp_ggml_compute_forward_rope_f16(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst,
        const bool forward) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];
    const struct wsp_ggml_tensor * src2 = dst->src[2];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    WSP_GGML_ASSERT(nb0 == sizeof(wsp_ggml_fp16_t));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr = wsp_ggml_nrows(dst);

    WSP_GGML_ASSERT(n_dims <= ne0);
    WSP_GGML_ASSERT(n_dims % 2 == 0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    float corr_dims[2];
    wsp_ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const bool is_neox = mode & WSP_GGML_ROPE_TYPE_NEOX;

    const float * freq_factors = NULL;
    if (src2 != NULL) {
        WSP_GGML_ASSERT(src2->type == WSP_GGML_TYPE_F32);
        WSP_GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (const float *) src2->data;
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    const float sin_sign = forward ? 1.0f : -1.0f;

    const int32_t * pos = (const int32_t *) src1->data;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const int64_t p = pos[i2];

            float * cache = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32)*ith;
            wsp_ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);

            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;

                if (!is_neox) {
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];

                        const wsp_ggml_fp16_t * const src = (wsp_ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                              wsp_ggml_fp16_t * dst_data  = (wsp_ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        const float x0 = WSP_GGML_FP16_TO_FP32(src[0]);
                        const float x1 = WSP_GGML_FP16_TO_FP32(src[1]);

                        dst_data[0] = WSP_GGML_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
                        dst_data[1] = WSP_GGML_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
                    }
                } else {
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        const int64_t ic = i0/2;

                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];

                        const wsp_ggml_fp16_t * const src = (wsp_ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                        wsp_ggml_fp16_t * dst_data  = (wsp_ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

                        const float x0 = WSP_GGML_FP16_TO_FP32(src[0]);
                        const float x1 = WSP_GGML_FP16_TO_FP32(src[n_dims/2]);

                        dst_data[0]        = WSP_GGML_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
                        dst_data[n_dims/2] = WSP_GGML_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
                    }
                }

                for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                    const wsp_ggml_fp16_t * const src = (wsp_ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                    wsp_ggml_fp16_t * dst_data  = (wsp_ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                    dst_data[0] = src[0];
                    dst_data[1] = src[1];
                }
            }
        }
    }
}

static void wsp_ggml_compute_forward_rope(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F16:
            {
                wsp_ggml_compute_forward_rope_f16(params, dst, true);
            } break;
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_rope_f32(params, dst, true);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_rope_back

static void wsp_ggml_compute_forward_rope_back(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F16:
            {
                wsp_ggml_compute_forward_rope_f16(params, dst, false);
            } break;
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_rope_f32(params, dst, false);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_conv_transpose_1d

static void wsp_ggml_compute_forward_conv_transpose_1d_f16_f32(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(src0->type == WSP_GGML_TYPE_F16);
    WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT( dst->type == WSP_GGML_TYPE_F32);

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const int nk = ne00*ne01*ne02;

    WSP_GGML_ASSERT(nb00 == sizeof(wsp_ggml_fp16_t));
    WSP_GGML_ASSERT(nb10 == sizeof(float));

    if (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            wsp_ggml_fp16_t * const wdata = (wsp_ggml_fp16_t *) params->wdata + 0;

            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    const wsp_ggml_fp16_t * const src = (wsp_ggml_fp16_t *)((char *) src0->data + i02*nb02 + i01*nb01);
                    wsp_ggml_fp16_t * dst_data = wdata + i01*ne00*ne02;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // permute source data (src1) from (L x Cin) to (Cin x L)
        {
            wsp_ggml_fp16_t * const wdata = (wsp_ggml_fp16_t *) params->wdata + nk;
            wsp_ggml_fp16_t * dst_data = wdata;

            for (int64_t i11 = 0; i11 < ne11; i11++) {
                const float * const src = (float *)((char *) src1->data + i11*nb11);
                for (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[i10*ne11 + i11] = WSP_GGML_FP32_TO_FP16(src[i10]);
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->data, 0, wsp_ggml_nbytes(dst));
    }
    wsp_ggml_barrier(params->threadpool);

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];

    // total rows in dst
    const int nr = ne1;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    wsp_ggml_fp16_t * const wdata     = (wsp_ggml_fp16_t *) params->wdata + 0;
    wsp_ggml_fp16_t * const wdata_src = wdata + nk;

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * dst_data = (float *)((char *) dst->data + i1*nb1);
        wsp_ggml_fp16_t * wdata_kernel = wdata + i1*ne02*ne00;
        for (int i10 = 0; i10 < ne10; i10++) {
            const int i1n = i10*ne11;
            for (int i00 = 0; i00 < ne00; i00++) {
                float v = 0;
                wsp_ggml_vec_dot_f16(ne02, &v, 0,
                        (wsp_ggml_fp16_t *)    wdata_src + i1n, 0,
                        (wsp_ggml_fp16_t *) wdata_kernel + i00*ne02, 0, 1);
                dst_data[i10*s0 + i00] += v;
            }
        }
    }
}

static void wsp_ggml_compute_forward_conv_transpose_1d_f32(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(src0->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT( dst->type == WSP_GGML_TYPE_F32);

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const int nk = ne00*ne01*ne02;

    WSP_GGML_ASSERT(nb00 == sizeof(float));
    WSP_GGML_ASSERT(nb10 == sizeof(float));

    if (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            float * const wdata = (float *) params->wdata + 0;

            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    const float * const src = (float *)((char *) src0->data + i02*nb02 + i01*nb01);
                    float * dst_data = wdata + i01*ne00*ne02;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            float * const wdata = (float *) params->wdata + nk;
            float * dst_data = wdata;

            for (int64_t i11 = 0; i11 < ne11; i11++) {
                const float * const src = (float *)((char *) src1->data + i11*nb11);
                for (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[i10*ne11 + i11] = src[i10];
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->data, 0, wsp_ggml_nbytes(dst));
    }
    wsp_ggml_barrier(params->threadpool);

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];

    // total rows in dst
    const int nr = ne1;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    float * const wdata     = (float *) params->wdata + 0;
    float * const wdata_src = wdata + nk;

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * dst_data = (float *)((char *) dst->data + i1*nb1);
        float * wdata_kernel = wdata + i1*ne02*ne00;
        for (int i10 = 0; i10 < ne10; i10++) {
            const int i1n = i10*ne11;
            for (int i00 = 0; i00 < ne00; i00++) {
                float v = 0;
                wsp_ggml_vec_dot_f32(ne02, &v, 0,
                        wdata_src + i1n, 0,
                        wdata_kernel + i00*ne02, 0, 1);
                dst_data[i10*s0 + i00] += v;
            }
        }
    }
}

static void wsp_ggml_compute_forward_conv_transpose_1d(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F16:
            {
                wsp_ggml_compute_forward_conv_transpose_1d_f16_f32(params, dst);
            } break;
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_conv_transpose_1d_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_im2col_f32
// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
static void wsp_ggml_compute_forward_im2col_f32(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT( dst->type == WSP_GGML_TYPE_F32);

    WSP_GGML_TENSOR_BINARY_OP_LOCALS;

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t *)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t *)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t *)(dst->op_params))[5];
    const bool is_2D = ((const int32_t *)(dst->op_params))[6] == 1;

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t N  = is_2D ? ne13 : ne12;
    const int64_t IC = is_2D ? ne12 : ne11;
    const int64_t IH = is_2D ? ne11 : 1;
    const int64_t IW = ne10;

    const int64_t KH = is_2D ? ne01 : 1;
    const int64_t KW = ne00;

    const int64_t OH = is_2D ? ne2 : 1;
    const int64_t OW = ne1;

    int ofs0 = is_2D ? nb13 : nb12;
    int ofs1 = is_2D ? nb12 : nb11;

    WSP_GGML_ASSERT(nb10 == sizeof(float));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        float * const wdata = (float *) dst->data;

        for (int64_t in = 0; in < N; in++) {
            for (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                for (int64_t iow = 0; iow < OW; iow++) {
                    for (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        float * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        const float * const src_data = (float *)((char *) src1->data + in*ofs0 + iic*ofs1); // [IH, IW]

                        for (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            for (int64_t ikw = 0; ikw < KW; ikw++) {
                                const int64_t iiw = iow*s0 + ikw*d0 - p0;
                                const int64_t iih = ioh*s1 + ikh*d1 - p1;

                                if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } else {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = (src_data[iih*IW + iiw]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


// wsp_ggml_compute_forward_im2col_f16
// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
static void wsp_ggml_compute_forward_im2col_f16(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(src0->type == WSP_GGML_TYPE_F16);
    WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT( dst->type == WSP_GGML_TYPE_F16);

    WSP_GGML_TENSOR_BINARY_OP_LOCALS;

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t *)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t *)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t *)(dst->op_params))[5];
    const bool is_2D = ((const int32_t *)(dst->op_params))[6] == 1;

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t N  = is_2D ? ne13 : ne12;
    const int64_t IC = is_2D ? ne12 : ne11;
    const int64_t IH = is_2D ? ne11 : 1;
    const int64_t IW = ne10;

    const int64_t KH = is_2D ? ne01 : 1;
    const int64_t KW = ne00;

    const int64_t OH = is_2D ? ne2 : 1;
    const int64_t OW = ne1;

    int ofs0 = is_2D ? nb13 : nb12;
    int ofs1 = is_2D ? nb12 : nb11;

    WSP_GGML_ASSERT(nb00 == sizeof(wsp_ggml_fp16_t));
    WSP_GGML_ASSERT(nb10 == sizeof(float));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        wsp_ggml_fp16_t * const wdata = (wsp_ggml_fp16_t *) dst->data;

        for (int64_t in = 0; in < N; in++) {
            for (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                for (int64_t iow = 0; iow < OW; iow++) {
                    for (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        wsp_ggml_fp16_t * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        const float * const src_data = (float *)((char *) src1->data + in*ofs0 + iic*ofs1); // [IH, IW]

                        for (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            for (int64_t ikw = 0; ikw < KW; ikw++) {
                                const int64_t iiw = iow*s0 + ikw*d0 - p0;
                                const int64_t iih = ioh*s1 + ikh*d1 - p1;

                                if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } else {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = WSP_GGML_FP32_TO_FP16(src_data[iih*IW + iiw]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

static void wsp_ggml_compute_forward_im2col(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {
    switch (dst->type) {
        case WSP_GGML_TYPE_F16:
            {
                wsp_ggml_compute_forward_im2col_f16(params, dst);
            } break;
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_im2col_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_im2col_back_f32

static void wsp_ggml_compute_forward_im2col_back_f32(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT( dst->type == WSP_GGML_TYPE_F32);

    WSP_GGML_TENSOR_BINARY_OP_LOCALS;

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t *)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t *)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t *)(dst->op_params))[5];
    const bool is_2D = ((const int32_t *)(dst->op_params))[6] == 1;

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t N  = is_2D ? ne3 : ne2;
    const int64_t IC = is_2D ? ne2 : ne1;
    const int64_t IH = is_2D ? ne1 : 1;
    const int64_t IW = ne0;

    const int64_t KH = is_2D ? ne01 : 1;
    const int64_t KW = ne00;

    const int64_t OH = is_2D ? ne12 : 1;
    const int64_t OW = ne11;

    int ofs0 = is_2D ? nb3 : nb2;
    int ofs1 = is_2D ? nb2 : nb1;

    WSP_GGML_ASSERT(nb0  == sizeof(float));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        float * const wdata = (float *) dst->data;

        for (int64_t in = 0; in < N; in++) {
            for (int64_t iic = ith; iic < IC; iic += nth) {
                for (int64_t iih = 0; iih < IH; iih++) {
                    for (int64_t iiw = 0; iiw < IW; iiw++) {

                        // micro kernel
                        float grad = 0.0f;
                        for (int64_t ikh = 0; ikh < KH; ikh++) {
                            for (int64_t ikw = 0; ikw < KW; ikw++) {
                                // For s0 > 1 some values were skipped over in the forward pass.
                                // These values have tmpw % s0 != 0 and need to be skipped in the backwards pass as well.
                                const int64_t tmpw = (iiw + p0 - ikw*d0);
                                if (tmpw % s0 != 0) {
                                    continue;
                                }
                                const int64_t iow = tmpw / s0;

                                // Equivalent logic as above except for s1.
                                int64_t ioh;
                                if (is_2D) {
                                    const int64_t tmph = iih + p1 - ikh*d1;

                                    if (tmph % s1 != 0) {
                                        continue;
                                    }

                                    ioh = tmph / s1;
                                } else {
                                    ioh = 0;
                                }

                                if (iow < 0 || iow >= OW || ioh < 0 || ioh >= OH) {
                                    continue;
                                }

                                const float * const src_data = (const float *) src1->data
                                    + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                                grad += src_data[iic*(KH*KW) + ikh*KW + ikw];
                            }
                        }
                        float * dst_data = (float *)((char *) wdata + (in*ofs0 + iic*ofs1)); // [IH, IW]
                        dst_data[iih*IW + iiw] = grad;
                    }
                }
            }
        }
    }
}

// wsp_ggml_compute_forward_conv_transpose_2d

static void wsp_ggml_compute_forward_conv_transpose_2d(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(src0->type == WSP_GGML_TYPE_F16);
    WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT( dst->type == WSP_GGML_TYPE_F32);

    WSP_GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const int nk = ne00*ne01*ne02*ne03;

    WSP_GGML_ASSERT(nb00 == sizeof(wsp_ggml_fp16_t));
    WSP_GGML_ASSERT(nb10 == sizeof(float));

    if (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
        {
            wsp_ggml_fp16_t * const wdata = (wsp_ggml_fp16_t *) params->wdata + 0;

            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    const wsp_ggml_fp16_t * const src = (wsp_ggml_fp16_t *)((char *) src0->data + i03*nb03 + i02*nb02);
                    wsp_ggml_fp16_t * dst_data = wdata + i02*ne01*ne00*ne03;
                    for (int64_t i01 = 0; i01 < ne01; i01++) {
                        for (int64_t i00 = 0; i00 < ne00; i00++) {
                            dst_data[i01*ne00*ne03 + i00*ne03 + i03] = src[i01 * ne00 + i00];
                        }
                    }
                }
            }
        }

        // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh)
        {
            wsp_ggml_fp16_t * const wdata = (wsp_ggml_fp16_t *) params->wdata + nk;
            for (int i12 = 0; i12 < ne12; i12++) {
                for (int i11 = 0; i11 < ne11; i11++) {
                    const float * const src = (float *)((char *) src1->data + i12*nb12 + i11*nb11);
                    wsp_ggml_fp16_t * dst_data = wdata + i11*ne10*ne12;
                    for (int i10 = 0; i10 < ne10; i10++) {
                        dst_data[i10*ne12 + i12] = WSP_GGML_FP32_TO_FP16(src[i10]);
                    }
                }
            }
        }

        memset(dst->data, 0, wsp_ggml_nbytes(dst));
    }
    wsp_ggml_barrier(params->threadpool);

    const int32_t stride = wsp_ggml_get_op_params_i32(dst, 0);

    // total patches in dst
    const int np = ne2;

    // patches per thread
    const int dp = (np + nth - 1)/nth;

    // patch range for this thread
    const int ip0 = dp*ith;
    const int ip1 = MIN(ip0 + dp, np);

    wsp_ggml_fp16_t * const wdata = (wsp_ggml_fp16_t *) params->wdata + 0;
    wsp_ggml_fp16_t * const wdata_src = wdata + nk;

    for (int i2 = ip0; i2 < ip1; i2++) { // Cout
        float * dst_data = (float *)((char *) dst->data + i2*nb2);
        wsp_ggml_fp16_t * wdata_kernel = wdata + i2*ne01*ne00*ne03;
        for (int i11 = 0; i11 < ne11; i11++) {
            for (int i10 = 0; i10 < ne10; i10++) {
                const int i1n = i11*ne10*ne12 + i10*ne12;
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i00 = 0; i00 < ne00; i00++) {
                        float v = 0;
                        wsp_ggml_vec_dot_f16(ne03, &v, 0,
                                wdata_src + i1n, 0,
                                wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                        dst_data[(i11*stride + i01)*ne0 + i10*stride + i00] += v;
                    }
                }
            }
        }
    }
}

// wsp_ggml_compute_forward_pool_1d_sk_p0

static void wsp_ggml_compute_forward_pool_1d_sk_p0(
        const struct wsp_ggml_compute_params * params,
        const enum wsp_ggml_op_pool op,
        const int k,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src = dst->src[0];

    assert(src->type == WSP_GGML_TYPE_F32 || src->type == WSP_GGML_TYPE_F16);

    if (params->ith != 0) {
        return;
    }

    const char * cdata = (const char *)src->data;
    const char * const data_end = cdata + wsp_ggml_nbytes(src);
    float * drow = (float *)dst->data;

    const int64_t rs = dst->ne[0];

    while (cdata < data_end) {
        const void * srow = (const void *)cdata;
        int j = 0;
        for (int64_t i = 0; i < rs; ++i) {
            switch (op) {
                case WSP_GGML_OP_POOL_AVG:   drow[i] = 0;        break;
                case WSP_GGML_OP_POOL_MAX:   drow[i] = -FLT_MAX; break;
                case WSP_GGML_OP_POOL_COUNT: WSP_GGML_ABORT("fatal error");
            }
            for (int ki = 0; ki < k; ++ki) {
                const float srow_j = (src->type == WSP_GGML_TYPE_F32) ? ((const float*)srow)[j] : WSP_GGML_FP16_TO_FP32(((const wsp_ggml_fp16_t*)srow)[j]);
                switch (op) {
                    case WSP_GGML_OP_POOL_AVG:                         drow[i] += srow_j; break;
                    case WSP_GGML_OP_POOL_MAX:   if (srow_j > drow[i]) drow[i]  = srow_j; break;
                    case WSP_GGML_OP_POOL_COUNT:                       WSP_GGML_ABORT("fatal error");
                }
                ++j;
            }
            switch (op) {
                case WSP_GGML_OP_POOL_AVG:         drow[i] /= k; break;
                case WSP_GGML_OP_POOL_MAX:                       break;
                case WSP_GGML_OP_POOL_COUNT: WSP_GGML_ABORT("fatal error");
            }
        }

        cdata += src->nb[1];
        drow  += rs;
    }
}

// wsp_ggml_compute_forward_pool_1d

static void wsp_ggml_compute_forward_pool_1d(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum wsp_ggml_op_pool op = opts[0];
    const int k0 = opts[1];
    const int s0 = opts[2];
    const int p0 = opts[3];
    WSP_GGML_ASSERT(p0 == 0); // padding not supported
    WSP_GGML_ASSERT(k0 == s0); // only s = k supported

    wsp_ggml_compute_forward_pool_1d_sk_p0(params, op, k0, dst);
}

// wsp_ggml_compute_forward_pool_2d

static void wsp_ggml_compute_forward_pool_2d(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src = dst->src[0];

    assert(src->type == WSP_GGML_TYPE_F32 || src->type == WSP_GGML_TYPE_F16);

    if (params->ith != 0) {
        return;
    }

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum wsp_ggml_op_pool op = opts[0];
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];
    const char * cdata = (const char*)src->data;
    const char * const data_end = cdata + wsp_ggml_nbytes(src);

    const int64_t px = dst->ne[0];
    const int64_t py = dst->ne[1];
    const int64_t pa = px * py;

    float * dplane = (float *)dst->data;

    const int ka = k0 * k1;
    const int offset0 = -p0;
    const int offset1 = -p1;

    while (cdata < data_end) {
        for (int oy = 0; oy < py; ++oy) {
            float * const drow = dplane + oy * px;
            for (int ox = 0; ox < px; ++ox) {
                float * const out =  drow + ox;
                switch (op) {
                    case WSP_GGML_OP_POOL_AVG:     *out = 0;        break;
                    case WSP_GGML_OP_POOL_MAX:     *out = -FLT_MAX; break;
                    case WSP_GGML_OP_POOL_COUNT: WSP_GGML_ABORT("fatal error");
                }

                const int ix = offset0 + ox * s0;
                const int iy = offset1 + oy * s1;

                for (int ky = 0; ky < k1; ++ky) {
                    if (iy + ky < 0 || iy + ky >= src->ne[1]) continue;
                    const void * srow = (const void *)(cdata + src->nb[1] * (iy + ky));
                    for (int kx = 0; kx < k0; ++kx) {
                        int j = ix + kx;
                        if (j < 0 || j >= src->ne[0]) continue;
                        const float srow_j = (src->type == WSP_GGML_TYPE_F32) ? ((const float*)srow)[j] : WSP_GGML_FP16_TO_FP32(((const wsp_ggml_fp16_t*)srow)[j]);
                        switch (op) {
                            case WSP_GGML_OP_POOL_AVG:                     *out += srow_j; break;
                            case WSP_GGML_OP_POOL_MAX: if (srow_j > *out)  *out  = srow_j; break;
                            case WSP_GGML_OP_POOL_COUNT:               WSP_GGML_ABORT("fatal error");
                        }
                    }
                }
                switch (op) {
                    case WSP_GGML_OP_POOL_AVG:           *out /= ka; break;
                    case WSP_GGML_OP_POOL_MAX:                       break;
                    case WSP_GGML_OP_POOL_COUNT: WSP_GGML_ABORT("fatal error");
                }
            }
        }

        cdata  += src->nb[2];
        dplane += pa;
    }
}

// wsp_ggml_compute_forward_pool_2d_back

static void wsp_ggml_compute_forward_pool_2d_back(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src  = dst->src[0];
    const struct wsp_ggml_tensor * dstf = dst->src[1]; // forward tensor of dst

    assert(dst->type == WSP_GGML_TYPE_F32 || dst->type == WSP_GGML_TYPE_F16);

    if (params->ith != 0) {
        return;
    }

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum wsp_ggml_op_pool op = opts[0];
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];

    char       * cdata  = (char       *) dst->data;
    const char * cdataf = (const char *) dstf->data;
    const char * const data_end = cdata + wsp_ggml_nbytes(dst);

    WSP_GGML_ASSERT(params->ith == 0);
    memset(cdata, 0, wsp_ggml_nbytes(dst));

    const int64_t px = src->ne[0];
    const int64_t py = src->ne[1];
    const int64_t pa = px * py;

    const float * splane = (const float *) src->data;

    const int ka = k0 * k1;
    const int offset0 = -p0;
    const int offset1 = -p1;

    while (cdata < data_end) {
        for (int oy = 0; oy < py; ++oy) {
            const float * const srow = splane + oy * px;
            for (int ox = 0; ox < px; ++ox) {
                const float grad0 = srow[ox];

                const int ix = offset0 + ox * s0;
                const int iy = offset1 + oy * s1;

                if (op == WSP_GGML_OP_POOL_MAX) {
                    float maxval = -FLT_MAX;
                    int kxmax = -1;
                    int kymax = -1;

                    for (int ky = 0; ky < k1; ++ky) {
                        if (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            continue;
                        }
                        const void * drowf = (const void *)(cdataf + dst->nb[1] * (iy + ky));
                        for (int kx = 0; kx < k0; ++kx) {
                            int j = ix + kx;
                            if (j < 0 || j >= dst->ne[0]) {
                                continue;
                            }

                            const float val = dst->type == WSP_GGML_TYPE_F32 ?
                                ((const float *) drowf)[j] : WSP_GGML_FP16_TO_FP32(((const wsp_ggml_fp16_t *) drowf)[j]);
                            if (val <= maxval) {
                                continue;
                            }

                            maxval = val;
                            kxmax = kx;
                            kymax = ky;
                        }
                    }

                    if (kxmax == -1 || kymax == -1) {
                        continue;
                    }

                    void * drow = (void *)(cdata + dst->nb[1] * (iy + kymax));
                    const int j = ix + kxmax;
                    if (dst->type == WSP_GGML_TYPE_F32) {
                        ((float *) drow)[j] += grad0;
                    } else {
                        ((wsp_ggml_fp16_t *) drow)[j] = WSP_GGML_FP32_TO_FP16(grad0 + WSP_GGML_FP16_TO_FP32(((const wsp_ggml_fp16_t *) drow)[j]));
                    }
                } else if (op == WSP_GGML_OP_POOL_AVG) {
                    const float grad = grad0 / ka;

                    for (int ky = 0; ky < k1; ++ky) {
                        if (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            continue;
                        }
                        void * drow = (void *)(cdata + dst->nb[1] * (iy + ky));
                        for (int kx = 0; kx < k0; ++kx) {
                            int j = ix + kx;
                            if (j < 0 || j >= dst->ne[0]) {
                                continue;
                            }

                            if (dst->type == WSP_GGML_TYPE_F32) {
                                ((float *) drow)[j] += grad;
                            } else {
                                ((wsp_ggml_fp16_t *) drow)[j] += WSP_GGML_FP32_TO_FP16(grad);
                            }
                        }
                    }
                } else {
                    WSP_GGML_ASSERT(false);
                }
            }
        }

        cdata  += dst->nb[2];
        cdataf += dst->nb[2];
        splane += pa;
    }
}

// wsp_ggml_compute_forward_upscale

static void wsp_ggml_compute_forward_upscale_f32(
    const struct wsp_ggml_compute_params * params,
    struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    WSP_GGML_ASSERT(src0->type == WSP_GGML_TYPE_F32);

    const int ith = params->ith;
    const int nth = params->nth;

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    const float sf0 = (float)ne0/src0->ne[0];
    const float sf1 = (float)ne1/src0->ne[1];
    const float sf2 = (float)ne2/src0->ne[2];
    const float sf3 = (float)ne3/src0->ne[3];

    // TODO: optimize

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        const int64_t i03 = i3 / sf3;
        for (int64_t i2 = ith; i2 < ne2; i2 += nth) {
            const int64_t i02 = i2 / sf2;
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                const int64_t i01 = i1 / sf1;
                for (int64_t i0 = 0; i0 < ne0; i0++) {
                    const int64_t i00 = i0 / sf0;

                    const float * x = (float *)((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          float * y = (float *)((char *)  dst->data +  i0*nb0  +  i1*nb1  +  i2*nb2  +  i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

static void wsp_ggml_compute_forward_upscale(
    const struct wsp_ggml_compute_params * params,
    struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_upscale_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}


// wsp_ggml_compute_forward_pad

static void wsp_ggml_compute_forward_pad_f32(
    const struct wsp_ggml_compute_params * params,
          struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    WSP_GGML_ASSERT(src0->nb[0] == sizeof(float));
    WSP_GGML_ASSERT( dst->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    float * dst_ptr = (float *) dst->data;

    // TODO: optimize

    for (int64_t i2 = 0; i2 < ne2; ++i2) {
        for (int64_t i1 = ith; i1 < ne1; i1 += nth) {
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                for (int64_t i3 = 0; i3 < ne3; ++i3) {
                    const int64_t dst_idx = i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0;

                    const float * src_ptr = (const float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);

                    if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        dst_ptr[dst_idx] = *src_ptr;
                    } else {
                        dst_ptr[dst_idx] = 0;
                    }
                }
            }
        }
    }
}

static void wsp_ggml_compute_forward_pad(
    const struct wsp_ggml_compute_params * params,
    struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_pad_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}


// wsp_ggml_compute_forward_arange

static void wsp_ggml_compute_forward_arange_f32(
    const struct wsp_ggml_compute_params * params,
    struct wsp_ggml_tensor * dst) {

    WSP_GGML_ASSERT(dst->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const float start = wsp_ggml_get_op_params_f32(dst, 0);
    const float stop  = wsp_ggml_get_op_params_f32(dst, 1);
    const float step  = wsp_ggml_get_op_params_f32(dst, 2);

    const int64_t steps = (int64_t) ceilf((stop - start) / step);

    WSP_GGML_ASSERT(wsp_ggml_nelements(dst) == steps);

    for (int64_t i = ith; i < steps; i+= nth) {
        float value = start + step * i;
        ((float *)dst->data)[i] = value;
    }
}

static void wsp_ggml_compute_forward_arange(
    const struct wsp_ggml_compute_params * params,
    struct wsp_ggml_tensor * dst) {
    switch (dst->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_arange_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

static void wsp_ggml_compute_forward_timestep_embedding_f32(
    const struct wsp_ggml_compute_params * params,
    struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    WSP_GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    const int dim = wsp_ggml_get_op_params_i32(dst, 0);
    const int max_period = wsp_ggml_get_op_params_i32(dst, 1);

    int half = dim / 2;

    for (int64_t i = 0; i < ne00; i++) {
        float * embed_data = (float *)((char *)  dst->data +  i*nb1);
        for (int64_t j = ith; j < half; j += nth) {
            float timestep = ((float *)src0->data)[i];
            float freq = (float)expf(-logf(max_period) * j / half);
            float arg = timestep * freq;
            embed_data[j] = cosf(arg);
            embed_data[j + half] = sinf(arg);
        }
        if (dim % 2 != 0 && ith == 0) {
            embed_data[dim] = 0.f;
        }
    }
}

static void wsp_ggml_compute_forward_timestep_embedding(
    const struct wsp_ggml_compute_params * params,
    struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_timestep_embedding_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_argsort

static void wsp_ggml_compute_forward_argsort_f32(
    const struct wsp_ggml_compute_params * params,
    struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    WSP_GGML_ASSERT(nb0 == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nr = wsp_ggml_nrows(src0);

    enum wsp_ggml_sort_order order = (enum wsp_ggml_sort_order) wsp_ggml_get_op_params_i32(dst, 0);

    for (int64_t i = ith; i < nr; i += nth) {
        int32_t * dst_data = (int32_t *)((char *) dst->data + i*nb1);
        const float * src_data = (float *)((char *) src0->data + i*nb01);

        for (int64_t j = 0; j < ne0; j++) {
            dst_data[j] = j;
        }

        // C doesn't have a functional sort, so we do a bubble sort instead
        for (int64_t j = 0; j < ne0; j++) {
            for (int64_t k = j + 1; k < ne0; k++) {
                if ((order == WSP_GGML_SORT_ORDER_ASC  && src_data[dst_data[j]] > src_data[dst_data[k]]) ||
                    (order == WSP_GGML_SORT_ORDER_DESC && src_data[dst_data[j]] < src_data[dst_data[k]])) {
                    int32_t tmp = dst_data[j];
                    dst_data[j] = dst_data[k];
                    dst_data[k] = tmp;
                }
            }
        }
    }
}

static void wsp_ggml_compute_forward_argsort(
    const struct wsp_ggml_compute_params * params,
    struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_argsort_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_flash_attn_ext

static void wsp_ggml_compute_forward_flash_attn_ext_f16(
        const struct wsp_ggml_compute_params * params,
        const struct wsp_ggml_tensor * q,
        const struct wsp_ggml_tensor * k,
        const struct wsp_ggml_tensor * v,
        const struct wsp_ggml_tensor * mask,
        struct wsp_ggml_tensor * dst) {

    WSP_GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    WSP_GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    WSP_GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    WSP_GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    WSP_GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    WSP_GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    WSP_GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    WSP_GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t D = neq0;
    const int64_t N = neq1;

    WSP_GGML_ASSERT(ne0 == D);
    WSP_GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    WSP_GGML_ASSERT(nbq0 == wsp_ggml_type_size(q->type));
    WSP_GGML_ASSERT(nbk0 == wsp_ggml_type_size(k->type));
    WSP_GGML_ASSERT(nbv0 == wsp_ggml_type_size(v->type));

    WSP_GGML_ASSERT(neq0 == D);
    WSP_GGML_ASSERT(nek0 == D);
    WSP_GGML_ASSERT(nev0 == D);

    WSP_GGML_ASSERT(neq1 == N);
    WSP_GGML_ASSERT(nev0 == D);

    // dst cannot be transposed or permuted
    WSP_GGML_ASSERT(nb0 == sizeof(float));
    WSP_GGML_ASSERT(nb0 <= nb1);
    WSP_GGML_ASSERT(nb1 <= nb2);
    WSP_GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    const int64_t rk2 = neq2/nek2;
    const int64_t rk3 = neq3/nek3;

    const int64_t rv2 = neq2/nev2;
    const int64_t rv3 = neq3/nev3;

    // parallelize by q rows using wsp_ggml_vec_dot_f32

    // total rows in q
    const int nr = neq1*neq2*neq3;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (float *) dst->op_params + 2, sizeof(float));

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    enum wsp_ggml_type    const k_vec_dot_type = type_traits_cpu[k->type].vec_dot_type;
    wsp_ggml_from_float_t const q_to_vec_dot   = wsp_ggml_get_type_traits(k_vec_dot_type)->from_float;
    wsp_ggml_vec_dot_t    const kq_vec_dot     = type_traits_cpu[k->type].vec_dot;
    wsp_ggml_to_float_t   const v_to_float     = wsp_ggml_get_type_traits(v->type)->to_float;

    WSP_GGML_ASSERT(q_to_vec_dot && "fattn: unsupported K-type");
    WSP_GGML_ASSERT(v_to_float   && "fattn: unsupported V-type");

    // loop over n_batch and n_head
    for (int ir = ir0; ir < ir1; ++ir) {
        // q indices
        const int iq3 = ir/(neq2*neq1);
        const int iq2 = (ir - iq3*neq2*neq1)/neq1;
        const int iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        const uint32_t h = iq2; // head index
        const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        float S = 0.0f;      // sum
        float M = -INFINITY; // maximum KQ value

        float       * VKQ32 = (float       *) params->wdata + ith*(3*D + CACHE_LINE_SIZE_F32); // FP32 VKQ accumulator
        float       * V32   =                 (VKQ32 + 1*D); // (temporary) FP32 V buffer
        wsp_ggml_fp16_t * VKQ16 = (wsp_ggml_fp16_t *) (VKQ32 + 1*D); // (temporary) FP16 VKQ accumulator
        wsp_ggml_fp16_t * Q_q   = (wsp_ggml_fp16_t *) (VKQ32 + 2*D); // (temporary) buffer for Q converted to quantized/FP16

        if (v->type == WSP_GGML_TYPE_F16) {
            memset(VKQ16, 0, D*sizeof(wsp_ggml_fp16_t));
        } else {
            memset(VKQ32, 0, D*sizeof(float));
        }

        const wsp_ggml_fp16_t * mp = mask ? (wsp_ggml_fp16_t *)((char *) mask->data + iq1*mask->nb[1]) : NULL;

        // k indices
        const int ik3 = iq3 / rk3;
        const int ik2 = iq2 / rk2;

        // v indices
        const int iv3 = iq3 / rv3;
        const int iv2 = iq2 / rv2;

        const float * pq = (const float *) ((char *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        q_to_vec_dot(pq, Q_q, D);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf
        for (int64_t ic = 0; ic < nek1; ++ic) {
            const float mv = mp ? slope*WSP_GGML_FP16_TO_FP32(mp[ic]) : 0.0f;
            if (mv == -INFINITY) {
                continue;
            }

            float s; // KQ value

            const char * k_data = (const char *) k->data + ( ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            kq_vec_dot(D, &s, 0, k_data, 0, Q_q, 0, 1);

            s = s*scale; // scale KQ value

            if (logit_softcap != 0.0f) {
                s = logit_softcap*tanhf(s);
            }

            s += mv; // apply mask

            const float Mold = M;

            float ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            float vs = 1.0f; // post-softmax KQ value, expf(s - M)

            const char * v_data = ((const char *) v->data + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

            if (v->type == WSP_GGML_TYPE_F16) {
                if (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    wsp_ggml_vec_scale_f16(D, VKQ16, ms);
                } else {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                wsp_ggml_vec_mad_f16(D, VKQ16, (const wsp_ggml_fp16_t *) v_data, vs);
            } else {
                if (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    wsp_ggml_vec_scale_f32(D, VKQ32, ms);
                } else {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                v_to_float(v_data, V32, D);

                // V += v*expf(s - M)
                wsp_ggml_vec_mad_f32(D, VKQ32, V32, vs);
            }

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        if (v->type == WSP_GGML_TYPE_F16) {
            for (int64_t d = 0; d < D; ++d) {
                VKQ32[d] = WSP_GGML_FP16_TO_FP32(VKQ16[d]);
            }
        }

        // V /= S
        const float S_inv = 1.0f/S;
        wsp_ggml_vec_scale_f32(D, VKQ32, S_inv);

        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        // original
        //memcpy((char *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3), V, nev0*sizeof(float));

        // permute(0, 2, 1, 3)
        memcpy((char *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
    }
}

static void wsp_ggml_compute_forward_flash_attn_ext(
        const struct wsp_ggml_compute_params * params,
        const struct wsp_ggml_tensor * q,
        const struct wsp_ggml_tensor * k,
        const struct wsp_ggml_tensor * v,
        const struct wsp_ggml_tensor * mask,
        struct wsp_ggml_tensor * dst) {
    switch (dst->op_params[3]) {
        case WSP_GGML_PREC_DEFAULT:
        case WSP_GGML_PREC_F32:
            {
                // uses F32 accumulators
                wsp_ggml_compute_forward_flash_attn_ext_f16(params, q, k, v, mask, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_flash_attn_back

static void wsp_ggml_compute_forward_flash_attn_back_f32(
        const struct wsp_ggml_compute_params * params,
        const bool masked,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * q = dst->src[0];
    const struct wsp_ggml_tensor * k = dst->src[1];
    const struct wsp_ggml_tensor * v = dst->src[2];
    const struct wsp_ggml_tensor * d = dst->src[3];

    WSP_GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    WSP_GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    WSP_GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    WSP_GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    WSP_GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    WSP_GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    WSP_GGML_TENSOR_LOCALS(int64_t, ned, d,   ne)
    WSP_GGML_TENSOR_LOCALS(size_t,  nbd, d,   nb)
    WSP_GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    WSP_GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t D = neq0;
    const int64_t N = neq1;
    const int64_t P = nek1 - N;
    const int64_t M = P + N;

    const int Mup  = wsp_ggml_up(M, WSP_GGML_SOFT_MAX_UNROLL);
    const int mxDM = MAX(D, Mup);

    // WSP_GGML_ASSERT(ne0 == D);
    // WSP_GGML_ASSERT(ne1 == N);
    WSP_GGML_ASSERT(P >= 0);

    WSP_GGML_ASSERT(nbq0 == sizeof(float));
    WSP_GGML_ASSERT(nbk0 == sizeof(float));
    WSP_GGML_ASSERT(nbv0 == sizeof(float));

    WSP_GGML_ASSERT(neq0 == D);
    WSP_GGML_ASSERT(nek0 == D);
    WSP_GGML_ASSERT(nev1 == D);
    WSP_GGML_ASSERT(ned0 == D);

    WSP_GGML_ASSERT(neq1 == N);
    WSP_GGML_ASSERT(nek1 == N + P);
    WSP_GGML_ASSERT(nev1 == D);
    WSP_GGML_ASSERT(ned1 == N);

    // dst cannot be transposed or permuted
    WSP_GGML_ASSERT(nb0 == sizeof(float));
    WSP_GGML_ASSERT(nb0 <= nb1);
    WSP_GGML_ASSERT(nb1 <= nb2);
    WSP_GGML_ASSERT(nb2 <= nb3);

    if (ith == 0) {
        memset(dst->data, 0, nb0*ne0*ne1*ne2*ne3);
    }
    wsp_ggml_barrier(params->threadpool);

    const int64_t elem_q = wsp_ggml_nelements(q);
    const int64_t elem_k = wsp_ggml_nelements(k);

    enum wsp_ggml_type result_type = dst->type;
    WSP_GGML_ASSERT(wsp_ggml_blck_size(result_type) == 1);
    const size_t tsize = wsp_ggml_type_size(result_type);

    const size_t offs_q = 0;
    const size_t offs_k = offs_q + WSP_GGML_PAD(elem_q * tsize, WSP_GGML_MEM_ALIGN);
    const size_t offs_v = offs_k + WSP_GGML_PAD(elem_k * tsize, WSP_GGML_MEM_ALIGN);

    void * grad_q = (char *) dst->data;
    void * grad_k = (char *) dst->data + offs_k;
    void * grad_v = (char *) dst->data + offs_v;

    const size_t nbgq1 = nb0*neq0;
    const size_t nbgq2 = nb0*neq0*neq1;
    const size_t nbgq3 = nb0*neq0*neq1*neq2;

    const size_t nbgk1 = nb0*nek0;
    const size_t nbgk2 = nb0*nek0*nek1;
    const size_t nbgk3 = nb0*nek0*nek1*neq2;

    const size_t nbgv1 = nb0*nev0;
    const size_t nbgv2 = nb0*nev0*nev1;
    const size_t nbgv3 = nb0*nev0*nev1*neq2;

    // parallelize by k rows using wsp_ggml_vec_dot_f32

    // total rows in k
    const int nr = nek2*nek3;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    const float scale = 1.0f/sqrtf(D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    // how often k2 (and v2) is repeated in q2
    int nrep = neq2/nek2;

    for (int ir = ir0; ir < ir1; ++ir) {
        // q indices
        const int ik3 = ir/(nek2);
        const int ik2 = ir - ik3*nek2;

        const int iq3 = ik3;
        const int id3 = ik3;
        const int iv3 = ik3;
        const int iv2 = ik2;

        for (int irep = 0; irep < nrep; ++irep) {
            const int iq2 = ik2 + irep*nek2;
            const int id2 = iq2;

            // (ik2 + irep*nek2) % nek2 == ik2
            for (int iq1 = 0; iq1 < neq1; ++iq1) {
                const int id1 = iq1;

                // not sure about CACHE_LINE_SIZE_F32..
                // - maybe it must not be multiplied by 2 and excluded from .. in SM 1*(..) offset?
                float * S  = (float *) params->wdata + ith*2*(mxDM + CACHE_LINE_SIZE_F32) + 0*(mxDM+CACHE_LINE_SIZE_F32);
                float * SM = (float *) params->wdata + ith*2*(mxDM + CACHE_LINE_SIZE_F32) + 1*(mxDM+CACHE_LINE_SIZE_F32);

                for (int i = M; i < Mup; ++i) {
                    S[i] = -INFINITY;
                }

                const int64_t masked_begin = masked ? (P + iq1 + 1) : M;
                for (int64_t ic = 0; ic < masked_begin; ++ic) {
                    // k indices
                    const int ik1 = ic;

                    // S indices
                    const int i1 = ik1;

                    wsp_ggml_vec_dot_f32(neq0,
                            S + i1, 0,
                            (float *) ((char *) k->data + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)), 0,
                            (float *) ((char *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)), 0, 1);
                }

                // scale
                wsp_ggml_vec_scale_f32(masked_begin, S, scale);

                for (int64_t i = masked_begin; i < M; i++) {
                    S[i] = -INFINITY;
                }

                // softmax
                // exclude known -INF S[..] values from max and loop
                // dont forget to set their SM values to zero
                {
                    float max = -INFINITY;
                    wsp_ggml_vec_max_f32(masked_begin, &max, S);

                    wsp_ggml_float sum = 0.0;
                    {
#ifdef WSP_GGML_SOFT_MAX_ACCELERATE
                        max = -max;
                        vDSP_vsadd(SM, 1, &max, SM, 1, Mup);
                        vvexpf(SM, SM, &Mup);
                        wsp_ggml_vec_sum_f32(Mup, &sum, SM);
#else
                        sum = wsp_ggml_vec_soft_max_f32(Mup, SM, S, max);
#endif
                    }

                    assert(sum > 0.0);

                    sum = 1.0/sum;
                    wsp_ggml_vec_scale_f32(masked_begin, SM, sum);

                }

                // step-by-step explanation
                {
                    // forward-process                    shape      grads from backward process
                    // parallel_for ik2,ik3:
                    //  for irep:
                    //   iq2 = ik2 + irep*nek2
                    //   k[:D,:M,:,:]                     [D,M,:,:]  grad[k][:D,:M,ik2,ik3]  += grad[kcur]
                    //   q[:D,:N,:,:]                     [D,N,:,:]  grad[q][:D,iq1,iq2,iq3] += grad[qcur]
                    //   v[:M,:D,:,:]                     [M,D,:,:]  grad[v][:M,:D,iv2,iv3]  += grad[vcur]
                    //   for iq1:
                    //    kcur   = k[:D,:M,ik2,ik3]       [D,M,1,1]  grad[kcur] = grad[S1].T @ qcur
                    //    qcur   = q[:D,iq1,iq2,iq3]      [D,1,1,1]  grad[qcur] = grad[S1]   @ kcur
                    //    vcur   = v[:M,:D,iv2,iv3]       [M,D,1,1]  grad[vcur] = grad[S5].T @ S4
                    //    S0     = -Inf                   [D,1,1,1]
                    //   ~S1[i]  = dot(kcur[:D,i], qcur)
                    //    S1     = qcur @ kcur.T          [M,1,1,1]  grad[S1]   = grad[S2] * scale
                    //    S2     = S1 * scale             [M,1,1,1]  grad[S2]   = diag_mask_zero(grad[S3], P)
                    //    S3     = diag_mask_inf(S2, P)   [M,1,1,1]  grad[S3]   = S4 * (grad[S4] - dot(S4, grad[S4]))
                    //    S4     = softmax(S3)            [M,1,1,1]  grad[S4]   = grad[S5] @ vcur
                    //   ~S5[i]  = dot(vcur[:,i], S4)
                    //    S5     = S4 @ vcur.T            [D,1,1,1]  grad[S5]   = d[:D,id1,id2,id3]
                    //   ~dst[i,iq1,iq2,iq3]  = S5[i]              ^
                    //    dst[:D,iq1,iq2,iq3] = S5                 | grad[dst[:D,iq1,iq2,iq3]] = d[:D,id1,id2,id3]
                    // dst                               backward-/ grad[dst]                 = d
                    //
                    // output gradients with their dependencies:
                    //
                    // grad[kcur] = grad[S1].T @ qcur
                    // grad[S1]   = diag_mask_zero(grad[S3], P) * scale
                    // grad[S3]   = S4 * (grad[S4] - dot(S4, grad[S4]))
                    // grad[S4]   = grad[S5] @ vcur
                    // grad[S4]   = d[:D,id1,id2,id3] @ vcur
                    // grad[qcur] = grad[S1]   @ kcur
                    // grad[vcur] = grad[S5].T @ S4
                    // grad[vcur] = d[:D,id1,id2,id3].T @ S4
                    //
                    // in post-order:
                    //
                    // S1         = qcur @ kcur.T
                    // S2         = S1 * scale
                    // S3         = diag_mask_inf(S2, P)
                    // S4         = softmax(S3)
                    // grad[S4]   = d[:D,id1,id2,id3] @ vcur
                    // grad[S3]   = S4 * (grad[S4] - dot(S4, grad[S4]))
                    // grad[S1]   = diag_mask_zero(grad[S3], P) * scale
                    // grad[qcur] = grad[S1]   @ kcur
                    // grad[kcur] = grad[S1].T @ qcur
                    // grad[vcur] = d[:D,id1,id2,id3].T @ S4
                    //
                    // using less variables (SM=S4):
                    //
                    // S             = diag_mask_inf(qcur @ kcur.T * scale, P)
                    // SM            = softmax(S)
                    // S             = d[:D,iq1,iq2,iq3] @ vcur
                    // dot_SM_gradSM = dot(SM, S)
                    // S             = SM * (S - dot(SM, S))
                    // S             = diag_mask_zero(S, P) * scale
                    //
                    // grad[q][:D,iq1,iq2,iq3] += S   @ kcur
                    // grad[k][:D,:M,ik2,ik3]  += S.T @ qcur
                    // grad[v][:M,:D,iv2,iv3]  += d[:D,id1,id2,id3].T @ SM
                }

                // S = gradSM = d[:D,id1,id2,id3] @ vcur[:,:,iv2,iv3]
                // S = d[:D,id1,id2,id3] @ vcur[:,:,iv2,iv3]
                // for ic:
                //   S[:M] += vcur[:M,ic,iv2,iv3] * d[ic,id1,id2,id3]
                // exclude known future zero S[..] values from operation
                wsp_ggml_vec_set_f32(masked_begin, S, 0);
                for (int64_t ic = 0; ic < D; ++ic) {
                    wsp_ggml_vec_mad_f32(masked_begin,
                            S,
                             (float *) ((char *) v->data + (          ic*nbv1  + iv2*nbv2 + iv3*nbv3)),
                            *(float *) ((char *) d->data + (ic*nbd0 + id1*nbd1 + id2*nbd2 + id3*nbd3)));
                }

                // S = SM * (S - dot(SM, S))
                float dot_SM_gradSM = 0;
                wsp_ggml_vec_dot_f32 (masked_begin, &dot_SM_gradSM, 0, SM, 0, S, 0, 1);
                wsp_ggml_vec_acc1_f32(M, S, -dot_SM_gradSM);
                wsp_ggml_vec_mul_f32 (masked_begin, S, S, SM);

                // S = diag_mask_zero(S, P) * scale
                // already done by above wsp_ggml_vec_set_f32

                // exclude known zero S[..] values from operation
                wsp_ggml_vec_scale_f32(masked_begin, S, scale);

                // S    shape [M,1]
                // SM   shape [M,1]
                // kcur shape [D,M]
                // qcur shape [D,1]
                // vcur shape [M,D]

                // grad[q][:D,iq1,iq2,iq3] += S @ kcur
                // grad[q][:D,iq1,iq2,iq3] += shape[M,1] @ shape[D,M]
                // for ic:
                //  grad[q][:D,iq1,iq2,iq3] += S[ic] * kcur[:D,ic,ik2,ik3]
                // exclude known zero S[..] values from loop
                for (int64_t ic = 0; ic < masked_begin; ++ic) {
                    wsp_ggml_vec_mad_f32(D,
                            (float *) ((char *) grad_q  + (iq1*nbgq1 + iq2*nbgq2  + iq3*nbgq3)),
                            (float *) ((char *) k->data + (ic*nbk1   + ik2*nbk2   + ik3*nbk3)),
                            S[ic]);
                }

                // grad[k][:D,:M,iq2,iq3] += S.T @ qcur
                // for ic:
                //  grad[k][:D,ic,iq2,iq3] += S.T[0,ic] * qcur[:D,0]
                //  grad[k][:D,ic,iq2,iq3] += S[ic]     * qcur[:D,0]
                // exclude known zero S[..] values from loop
                for (int64_t ic = 0; ic < masked_begin; ++ic) {
                    wsp_ggml_vec_mad_f32(D,
                            (float *) ((char *) grad_k  + (ic*nbgk1  + ik2*nbgk2  + ik3*nbgk3)),
                            (float *) ((char *) q->data + (iq1*nbq1  + iq2*nbq2   + iq3*nbq3)),
                            S[ic]);
                }

                // grad[v][:M,:D,iv2,iv3] += d[:D,id1,id2,id3].T       @ SM
                // for ic:
                //  grad[v][:M,ic,iv2,iv3] += d[:D,id1,id2,id3].T[0,ic] * SM[:M]
                //  grad[v][:M,ic,iv2,iv3] += d[ic,id1,id2,id3]         * SM[:M]
                // exclude known zero SM[..] values from mad
                for (int64_t ic = 0; ic < D; ++ic) {
                    wsp_ggml_vec_mad_f32(masked_begin,
                            (float *) ((char *) grad_v   + (          ic*nbgv1 + iv2*nbgv2 + iv3*nbgv3)),
                            SM,
                            *(float *) ((char *) d->data + (ic*nbd0 + id1*nbd1 + id2*nbd2  + id3*nbd3)));
                }
            }
        }
    }
}

static void wsp_ggml_compute_forward_flash_attn_back(
        const struct wsp_ggml_compute_params * params,
        const bool masked,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * q = dst->src[0];

    switch (q->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_flash_attn_back_f32(params, masked, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_ssm_conv

static void wsp_ggml_compute_forward_ssm_conv_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {
    const struct wsp_ggml_tensor * src0 = dst->src[0]; // conv_x
    const struct wsp_ggml_tensor * src1 = dst->src[1]; // conv1d.weight

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc  = src1->ne[0]; // d_conv
    const int ncs = src0->ne[0]; // d_conv - 1 + n_t
    const int nr  = src0->ne[1]; // d_inner
    const int n_t =  dst->ne[1]; // tokens per sequence
    const int n_s =  dst->ne[2]; // number of sequences in the batch

    WSP_GGML_ASSERT( dst->ne[0] == nr);
    WSP_GGML_ASSERT(src0->nb[0] == sizeof(float));
    WSP_GGML_ASSERT(src1->nb[0] == sizeof(float));
    WSP_GGML_ASSERT(src0->nb[1] == src0->ne[0]*sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);
    const int ir  = ir1 - ir0;

    for (int i3 = 0; i3 < n_s; ++i3) {
        for (int i2 = 0; i2 < n_t; ++i2) {
            // {d_conv - 1 + n_t, d_inner, n_seqs}
            // sliding window
            const float * s = (const float *) ((const char *) src0->data + ir0*(src0->nb[1]) + i2*(src0->nb[0]) + i3*(src0->nb[2])); // {d_conv, d_inner, n_s}
            const float * c = (const float *) ((const char *) src1->data + ir0*(src1->nb[1])); // {d_conv, d_inner}
            float * x = (float *) ((char *) dst->data + ir0*(dst->nb[0]) + i2*(dst->nb[1]) + i3*(dst->nb[2])); // {d_inner, n_t, n_s}

            // TODO: transpose the output for smaller strides for big batches?
            // d_inner
            for (int i1 = 0; i1 < ir; ++i1) {
                // rowwise dot product
                // NOTE: not using wsp_ggml_vec_dot_f32, because its sum is in double precision
                float sumf = 0.0f;

                // d_conv
                for (int i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];
                }
                x[i1] = sumf;
            }
        }
    }
}

static void wsp_ggml_compute_forward_ssm_conv(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_ssm_conv_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_ssm_scan

static void wsp_ggml_compute_forward_ssm_scan_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {
    const struct wsp_ggml_tensor * src0 = dst->src[0]; // s
    const struct wsp_ggml_tensor * src1 = dst->src[1]; // x
    const struct wsp_ggml_tensor * src2 = dst->src[2]; // dt
    const struct wsp_ggml_tensor * src3 = dst->src[3]; // A
    const struct wsp_ggml_tensor * src4 = dst->src[4]; // B
    const struct wsp_ggml_tensor * src5 = dst->src[5]; // C

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nc  = src0->ne[0]; // d_state
    const int64_t nr  = src0->ne[1]; // d_inner
    const int64_t n_t = src1->ne[1]; // number of tokens per sequence
    const int64_t n_s = src0->ne[2]; // number of sequences in the batch

    WSP_GGML_ASSERT(wsp_ggml_nelements(src1) + wsp_ggml_nelements(src0) == wsp_ggml_nelements(dst));
    WSP_GGML_ASSERT(src0->nb[0] == sizeof(float));
    WSP_GGML_ASSERT(src1->nb[0] == sizeof(float));
    WSP_GGML_ASSERT(src2->nb[0] == sizeof(float));
    WSP_GGML_ASSERT(src3->nb[0] == sizeof(float));
    WSP_GGML_ASSERT(src4->nb[0] == sizeof(float));
    WSP_GGML_ASSERT(src5->nb[0] == sizeof(float));
    // required for the dot product between s and C
    WSP_GGML_ASSERT(src0->nb[1] == src0->ne[0]*sizeof(float));
    // required for per-sequence offsets for states
    WSP_GGML_ASSERT(src0->nb[2] == src0->ne[0]*src0->ne[1]*sizeof(float));
    // required to get correct offset for state destination (i.e. src1->nb[3])
    WSP_GGML_ASSERT(src1->nb[3] == src1->ne[0]*src1->ne[1]*src1->ne[2]*sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);
    const int ir  = ir1 - ir0;

    for (int i3 = 0; i3 < n_s; ++i3) {
        for (int i2 = 0; i2 < n_t; ++i2) {
            const float * s0 = (const float *) ((const char *) src0->data + ir0*(src0->nb[1]) + i3*(src0->nb[2])); // {d_state, d_inner, n_s}
            const float * x  = (const float *) ((const char *) src1->data + ir0*(src1->nb[0]) + i2*(src1->nb[1]) + i3*(src1->nb[2])); // {d_inner, n_t, n_s}
            const float * dt = (const float *) ((const char *) src2->data + ir0*(src2->nb[0]) + i2*(src2->nb[1]) + i3*(src2->nb[2])); // {d_inner, n_t, n_s}
            const float * A  = (const float *) ((const char *) src3->data + ir0*(src3->nb[1])); // {d_state, d_inner}
            const float * B  = (const float *) ((const char *) src4->data +  i2*(src4->nb[1]) + i3*(src4->nb[2])); // {d_state, n_t, n_s}
            const float * C  = (const float *) ((const char *) src5->data +  i2*(src5->nb[1]) + i3*(src5->nb[2])); // {d_state, n_t, n_s}
                  float * y  = (      float *) ((      char *) dst->data  + ir0*(src1->nb[0]) + i2*(src1->nb[1]) + i3*(src1->nb[2])); // {d_inner, n_t, n_s}
                  float * s  = (      float *) ((      char *) dst->data  + ir0*(src0->nb[1]) + i3*(src0->nb[2]) +     src1->nb[3]);  // {d_state, d_inner, n_s}

            // use the output as the source for the next token-wise iterations
            if (i2 > 0) { s0 = s; }

            // d_inner
            for (int i1 = 0; i1 < ir; ++i1) {
                // ref: https://github.com/state-spaces/mamba/blob/34076d664838588a3c97727b263478ab9f621a07/mamba_ssm/ops/triton/selective_state_update.py#L78
                float dt_soft_plus = dt[i1] <= 20.0f ? log1pf(expf(dt[i1])) : dt[i1];
                float x_dt = x[i1] * dt_soft_plus;
                float sumf = 0.0f;
                // d_state
                for (int i0 = 0; i0 < nc; ++i0) {
                    int i = i0 + i1*nc;
                    // state = prev_state * dA + dB * x
                    float state = (s0[i] * expf(dt_soft_plus * A[i])) + (B[i0] * x_dt);
                    // y = rowwise_dotprod(state, C)
                    sumf += state * C[i0];
                    s[i] = state;
                }
                y[i1] = sumf;
            }
        }
    }
}

static void wsp_ggml_compute_forward_ssm_scan(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_ssm_scan_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_win_part

static void wsp_ggml_compute_forward_win_part_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {
    UNUSED(params);

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    WSP_GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    WSP_GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    const int32_t nep0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t nep1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t w    = ((const int32_t *)(dst->op_params))[2];

    assert(ne00 == ne0);
    assert(ne3  == nep0*nep1);

    // TODO: optimize / multi-thread
    for (int py = 0; py < nep1; ++py) {
        for (int px = 0; px < nep0; ++px) {
            const int64_t i3 = py*nep0 + px;
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        const int64_t i02 = py*w + i2;
                        const int64_t i01 = px*w + i1;
                        const int64_t i00 = i0;

                        const int64_t i = i3*ne2*ne1*ne0 + i2*ne1*ne0    + i1*ne0   + i0;
                        const int64_t j =                  i02*ne01*ne00 + i01*ne00 + i00;

                        if (py*w + i2 >= ne02 || px*w + i1 >= ne01) {
                            ((float *) dst->data)[i] = 0.0f;
                        } else {
                            ((float *) dst->data)[i] = ((float *) src0->data)[j];
                        }
                    }
                }
            }
        }
    }
}

static void wsp_ggml_compute_forward_win_part(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_win_part_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_win_unpart

static void wsp_ggml_compute_forward_win_unpart_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {
    UNUSED(params);

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    WSP_GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    WSP_GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    const int32_t w = ((const int32_t *)(dst->op_params))[0];

    // padding
    const int px = (w - ne1%w)%w;
    //const int py = (w - ne2%w)%w;

    const int npx = (px + ne1)/w;
    //const int npy = (py + ne2)/w;

    assert(ne0 == ne00);

    // TODO: optimize / multi-thread
    for (int64_t i2 = 0; i2 < ne2; ++i2) {
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                const int ip2 = i2/w;
                const int ip1 = i1/w;

                const int64_t i02 = i2%w;
                const int64_t i01 = i1%w;
                const int64_t i00 = i0;

                const int64_t i = (ip2*npx + ip1)*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00 + i00;
                const int64_t j =                                  i2*ne1*ne0    + i1*ne0   + i0;

                ((float *) dst->data)[j] = ((float *) src0->data)[i];
            }
        }
    }
}

static void wsp_ggml_compute_forward_win_unpart(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_win_unpart_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

//gmml_compute_forward_unary

static void wsp_ggml_compute_forward_unary(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const enum wsp_ggml_unary_op op = wsp_ggml_get_unary_op(dst);

    switch (op) {
        case WSP_GGML_UNARY_OP_ABS:
            {
                wsp_ggml_compute_forward_abs(params, dst);
            } break;
        case WSP_GGML_UNARY_OP_SGN:
            {
                wsp_ggml_compute_forward_sgn(params, dst);
            } break;
        case WSP_GGML_UNARY_OP_NEG:
            {
                wsp_ggml_compute_forward_neg(params, dst);
            } break;
        case WSP_GGML_UNARY_OP_STEP:
            {
                wsp_ggml_compute_forward_step(params, dst);
            } break;
        case WSP_GGML_UNARY_OP_TANH:
            {
                wsp_ggml_compute_forward_tanh(params, dst);
            } break;
        case WSP_GGML_UNARY_OP_ELU:
            {
                wsp_ggml_compute_forward_elu(params, dst);
            } break;
        case WSP_GGML_UNARY_OP_RELU:
            {
                wsp_ggml_compute_forward_relu(params, dst);
            } break;
        case WSP_GGML_UNARY_OP_SIGMOID:
            {
                wsp_ggml_compute_forward_sigmoid(params, dst);
            } break;
        case WSP_GGML_UNARY_OP_GELU:
            {
                wsp_ggml_compute_forward_gelu(params, dst);
            } break;
        case WSP_GGML_UNARY_OP_GELU_QUICK:
            {
                wsp_ggml_compute_forward_gelu_quick(params, dst);
            } break;
        case WSP_GGML_UNARY_OP_SILU:
            {
                wsp_ggml_compute_forward_silu(params, dst);
            } break;
        case WSP_GGML_UNARY_OP_HARDSWISH:
            {
                wsp_ggml_compute_forward_hardswish(params, dst);
            } break;
        case WSP_GGML_UNARY_OP_HARDSIGMOID:
            {
                wsp_ggml_compute_forward_hardsigmoid(params, dst);
            } break;
        case WSP_GGML_UNARY_OP_EXP:
            {
                wsp_ggml_compute_forward_exp(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_get_rel_pos

static void wsp_ggml_compute_forward_get_rel_pos_f16(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {
    UNUSED(params);

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

    WSP_GGML_TENSOR_UNARY_OP_LOCALS

    const int64_t w = ne1;

    wsp_ggml_fp16_t * src0_data = (wsp_ggml_fp16_t *) src0->data;
    wsp_ggml_fp16_t * dst_data  = (wsp_ggml_fp16_t *) dst->data;

    for (int64_t i2 = 0; i2 < ne2; ++i2) {
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            const int64_t pos = (w - i1 - 1) + i2;
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                dst_data[i2*ne1*ne0 + i1*ne0 + i0] = src0_data[pos*ne00 + i0];
            }
        }
    }
}

static void wsp_ggml_compute_forward_get_rel_pos(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F16:
        case WSP_GGML_TYPE_BF16:
            {
                wsp_ggml_compute_forward_get_rel_pos_f16(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_add_rel_pos

static void wsp_ggml_compute_forward_add_rel_pos_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];
    const struct wsp_ggml_tensor * src2 = dst->src[2];

    const bool inplace = (bool) ((int32_t *) dst->op_params)[0];
    if (!inplace) {
        if (params->ith == 0) {
            memcpy((char *) dst->data, (char *) src0->data, wsp_ggml_nbytes(dst));
        }
        wsp_ggml_barrier(params->threadpool);
    }
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

    float * src1_data = (float *) src1->data;
    float * src2_data = (float *) src2->data;
    float * dst_data  = (float *) dst->data;

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const int ith = params->ith;
    const int nth = params->nth;

    // total patches in dst
    const int np = ne13;

    // patches per thread
    const int dp = (np + nth - 1)/nth;

    // patch range for this thread
    const int ip0 = dp*ith;
    const int ip1 = MIN(ip0 + dp, np);

    for (int64_t i13 = ip0; i13 < ip1; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t i11 = 0; i11 < ne11; ++i11) {
                const int64_t jp1 = i13*ne12*ne11*ne10 + i12*ne11*ne10 + i11*ne10;
                for (int64_t i10 = 0; i10 < ne10; ++i10) {
                    const int64_t jp0  = jp1 + i10;
                    const float src1_e = src1_data[jp0];
                    const float src2_e = src2_data[jp0];

                    const int64_t jdh = jp0 * ne10;
                    const int64_t jdw = jdh - (ne10 - 1) * i10;

                    for (int64_t j = 0; j < ne10; ++j) {
                        dst_data[jdh + j     ] += src2_e;
                        dst_data[jdw + j*ne10] += src1_e;
                    }
                }
            }
        }
    }
}

static void wsp_ggml_compute_forward_add_rel_pos(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_add_rel_pos_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_rwkv_wkv6

static void wsp_ggml_compute_forward_rwkv_wkv6_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {
    const int64_t T = dst->src[1]->ne[3];
    const int64_t C = dst->ne[0];
    const int64_t HEADS = dst->src[1]->ne[2];
    const int64_t n_seqs = dst->src[5]->ne[1];
    const int64_t head_size = C / HEADS;

    float * dst_data = (float *) dst->data;
    float * state = ((float *) dst->data) + C * T;

    const int ith = params->ith;
    const int nth = params->nth;

    if (ith >= HEADS) {
        return;
    }

    const int h_start = (HEADS * ith) / nth;
    const int h_end = ((HEADS * (ith + 1)) / nth < HEADS) ?
                (HEADS * (ith + 1)) / nth : HEADS;

    float * k =          (float *) dst->src[0]->data;
    float * v =          (float *) dst->src[1]->data;
    float * r =          (float *) dst->src[2]->data;
    float * time_faaaa = (float *) dst->src[3]->data;
    float * time_decay = (float *) dst->src[4]->data;

    size_t t_stride = HEADS * head_size; // Same to C

    size_t h_stride = C / HEADS;
    WSP_GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = head_size * head_size;

    if (ith == 0) {
        memset(dst_data, 0, T * C * sizeof(float));
    }
    wsp_ggml_barrier(params->threadpool);


    #if defined(__AVX__) && !defined(__AVX512F__)
        #define WSP_GGML_F32X WSP_GGML_F32x8
        #define WSP_GGML_F32X_SET1 WSP_GGML_F32x8_SET1
        #define WSP_GGML_F32X_LOAD WSP_GGML_F32x8_LOAD
        #define WSP_GGML_F32X_STORE WSP_GGML_F32x8_STORE
        #define WSP_GGML_F32X_MUL WSP_GGML_F32x8_MUL
        #define WSP_GGML_F32X_FMA WSP_GGML_F32x8_FMA
        #define WKV_VECTOR_SIZE 8
    #elif defined(__AVX512F__)
        #define WSP_GGML_F32X WSP_GGML_F32x16
        #define WSP_GGML_F32X_SET1 WSP_GGML_F32x16_SET1
        #define WSP_GGML_F32X_LOAD WSP_GGML_F32x16_LOAD
        #define WSP_GGML_F32X_STORE WSP_GGML_F32x16_STORE
        #define WSP_GGML_F32X_MUL WSP_GGML_F32x16_MUL
        #define WSP_GGML_F32X_FMA WSP_GGML_F32x16_FMA
        #define WKV_VECTOR_SIZE 16
    #elif defined(__ARM_NEON) && defined(__aarch64__)
        #define WSP_GGML_F32X WSP_GGML_F32x4
        #define WSP_GGML_F32X_SET1 WSP_GGML_F32x4_SET1
        #define WSP_GGML_F32X_LOAD WSP_GGML_F32x4_LOAD
        #define WSP_GGML_F32X_STORE WSP_GGML_F32x4_STORE
        #define WSP_GGML_F32X_MUL WSP_GGML_F32x4_MUL
        #define WSP_GGML_F32X_FMA WSP_GGML_F32x4_FMA
        #define WKV_VECTOR_SIZE 4
    #endif

    #ifdef WKV_VECTOR_SIZE
        const int64_t vec_count = head_size / WKV_VECTOR_SIZE;

        for (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            float * state_cur = state + state_offset;
            float * state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[5]->data + state_offset;

            for (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                for (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_i_offset = h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    float k_val = k[t_h_i_offset];
                    float r_val = r[t_h_i_offset];
                    float time_faaaa_val = time_faaaa[h_i_offset];
                    float time_decay_val = time_decay[t_h_i_offset];

                    // Broadcast scalar values to vectors
                    WSP_GGML_F32X k_vec = WSP_GGML_F32X_SET1(k_val);
                    WSP_GGML_F32X r_vec = WSP_GGML_F32X_SET1(r_val);
                    WSP_GGML_F32X time_faaaa_vec = WSP_GGML_F32X_SET1(time_faaaa_val);
                    WSP_GGML_F32X time_decay_vec = WSP_GGML_F32X_SET1(time_decay_val);

                    for (int64_t j = 0; j < vec_count; j++) {
                        size_t base_j = j * WKV_VECTOR_SIZE;
                        size_t t_h_j_offset = t_h_offset + base_j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + base_j;

                        // Load x elements at once
                        WSP_GGML_F32X v_vec = WSP_GGML_F32X_LOAD(&v[t_h_j_offset]);
                        WSP_GGML_F32X prev_state_vec = WSP_GGML_F32X_LOAD(&state_prev[h_2d_i_j_offset]);
                        WSP_GGML_F32X dst_vec = WSP_GGML_F32X_LOAD(&dst_data[t_h_j_offset]);

                        // Compute kv = v * k
                        WSP_GGML_F32X kv_vec = WSP_GGML_F32X_MUL(v_vec, k_vec);

                        // Compute temp = kv * time_faaaa + prev_state
                        WSP_GGML_F32X temp_vec = WSP_GGML_F32X_FMA(prev_state_vec, kv_vec, time_faaaa_vec);

                        // Update dst: dst += temp * r
                        dst_vec = WSP_GGML_F32X_FMA(dst_vec, temp_vec, r_vec);
                        WSP_GGML_F32X_STORE(&dst_data[t_h_j_offset], dst_vec);

                        // Update state: state = prev_state * time_decay + kv
                        WSP_GGML_F32X new_state_vec = WSP_GGML_F32X_FMA(kv_vec, prev_state_vec, time_decay_vec);
                        WSP_GGML_F32X_STORE(&state_cur[h_2d_i_j_offset], new_state_vec);
                    }

                    // Handle remaining elements, this will not be used.
                    for (int64_t j = vec_count * WKV_VECTOR_SIZE; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;
                        float v_val = v[t_h_j_offset];
                        float kv_val = v_val * k_val;
                        float prev_state_val = state_prev[h_2d_i_j_offset];
                        float temp_val = kv_val * time_faaaa_val + prev_state_val;
                        dst_data[t_h_j_offset] += temp_val * r_val;
                        state_cur[h_2d_i_j_offset] = prev_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }

    #else
        // basically fused operations:
        // dst = r @ (time_faaaa * (k @ v) + state),
        // state = time_decay * state + (k @ v),
        // recursive through each token
        for (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            float * state_cur = state + state_offset;
            float * state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[5]->data + state_offset;

            for (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                for (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_i_offset = h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    float k_val = k[t_h_i_offset];
                    float r_val = r[t_h_i_offset];
                    float time_faaaa_val = time_faaaa[h_i_offset];
                    // RWKV v6: different time_decay for each token.
                    float time_decay_val = time_decay[t_h_i_offset];

                    for (int64_t j = 0; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;

                        float v_val = v[t_h_j_offset];
                        float kv_val = v_val * k_val;
                        float prev_state_val = state_prev[h_2d_i_j_offset];
                        float temp_val = kv_val * time_faaaa_val + prev_state_val;
                        dst_data[t_h_j_offset] += temp_val * r_val;
                        state_cur[h_2d_i_j_offset] = prev_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }
    #endif
}


static void wsp_ggml_compute_forward_rwkv_wkv6(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_rwkv_wkv6_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_map_unary

static void wsp_ggml_compute_forward_map_unary_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst,
        const wsp_ggml_unary_op_f32_t fun) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(wsp_ggml_is_contiguous_1(src0));
    assert(wsp_ggml_is_contiguous_1(dst));
    assert(wsp_ggml_are_same_shape(src0, dst));

    const int n  = wsp_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        fun(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void wsp_ggml_compute_forward_map_unary(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst,
        const wsp_ggml_unary_op_f32_t fun) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_map_unary_f32(params, dst, fun);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_map_binary

static void wsp_ggml_compute_forward_map_binary_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst,
        const wsp_ggml_binary_op_f32_t fun) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    if (params->ith != 0) {
        return;
    }

    assert(wsp_ggml_is_contiguous_1(src0));
    assert(wsp_ggml_is_contiguous_1(src1));
    assert(wsp_ggml_is_contiguous_1(dst));
    assert(wsp_ggml_are_same_shape(src0, src1) && wsp_ggml_are_same_shape(src0, dst));

    const int n  = wsp_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        fun(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])),
                (float *) ((char *) src1->data + i*(src1->nb[1])));
    }
}

static void wsp_ggml_compute_forward_map_binary(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst,
        const wsp_ggml_binary_op_f32_t fun) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_map_binary_f32(params, dst, fun);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_map_custom1

static void wsp_ggml_compute_forward_map_custom1_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst,
        const wsp_ggml_custom1_op_f32_t fun) {

    const struct wsp_ggml_tensor * a = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    fun(dst, a);
}

// wsp_ggml_compute_forward_map_custom2

static void wsp_ggml_compute_forward_map_custom2_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst,
        const wsp_ggml_custom2_op_f32_t fun) {

    const struct wsp_ggml_tensor * a = dst->src[0];
    const struct wsp_ggml_tensor * b = dst->src[1];

    if (params->ith != 0) {
        return;
    }

    fun(dst, a, b);
}

// wsp_ggml_compute_forward_map_custom3

static void wsp_ggml_compute_forward_map_custom3_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst,
        const wsp_ggml_custom3_op_f32_t fun) {

    const struct wsp_ggml_tensor * a = dst->src[0];
    const struct wsp_ggml_tensor * b = dst->src[1];
    const struct wsp_ggml_tensor * c = dst->src[1];

    if (params->ith != 0) {
        return;
    }

    fun(dst, a, b, c);
}

// wsp_ggml_compute_forward_map_custom1

static void wsp_ggml_compute_forward_map_custom1(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * a = dst->src[0];

    struct wsp_ggml_map_custom1_op_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    p.fun(dst, a, params->ith, params->nth, p.userdata);
}

// wsp_ggml_compute_forward_map_custom2

static void wsp_ggml_compute_forward_map_custom2(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * a = dst->src[0];
    const struct wsp_ggml_tensor * b = dst->src[1];

    struct wsp_ggml_map_custom2_op_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    p.fun(dst, a, b, params->ith, params->nth, p.userdata);
}

// wsp_ggml_compute_forward_map_custom3

static void wsp_ggml_compute_forward_map_custom3(
        const struct wsp_ggml_compute_params * params,
              struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * a = dst->src[0];
    const struct wsp_ggml_tensor * b = dst->src[1];
    const struct wsp_ggml_tensor * c = dst->src[2];

    struct wsp_ggml_map_custom3_op_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    p.fun(dst, a, b, c, params->ith, params->nth, p.userdata);
}

// wsp_ggml_compute_forward_cross_entropy_loss

static void wsp_ggml_compute_forward_cross_entropy_loss_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];

    WSP_GGML_ASSERT(src0->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT(src1->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT(src0->nb[0] == wsp_ggml_type_size(src0->type));
    WSP_GGML_ASSERT(src1->nb[0] == wsp_ggml_type_size(src1->type));
    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, src1));
    WSP_GGML_ASSERT(wsp_ggml_is_scalar(dst));
    WSP_GGML_ASSERT(dst->type == WSP_GGML_TYPE_F32);

    // TODO: handle transposed/permuted matrices
    const int64_t nc = src0->ne[0];
    const int64_t nr = wsp_ggml_nrows(src0);

    const int ith = params->ith;
    const int nth = params->nth;

    float * sums =  (float *) params->wdata;
    float * st   = ((float *) params->wdata) + nth + ith*nc;
    float sum_thread = 0.0f;

    WSP_GGML_ASSERT(params->wsize >= sizeof(float) * (nth + nth * nc));

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    for (int64_t i1 = ir0; i1 < ir1; ++i1) {
        const float * s0 = (const float *)((const char *) src0->data + i1*src0->nb[1]);
        const float * s1 = (const float *)((const char *) src1->data + i1*src1->nb[1]);

#ifndef NDEBUG
        for (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            assert(!isnan(s0[i]));
            assert(!isnan(s1[i]));
        }
#endif

        float max = -INFINITY;
        wsp_ggml_vec_max_f32(nc, &max, s0);
        const wsp_ggml_float sum_softmax = wsp_ggml_vec_log_soft_max_f32(nc, st, s0, max);
        assert(sum_softmax >= 0.0);

        wsp_ggml_vec_add1_f32(nc, st, st, -sum_softmax);
        wsp_ggml_vec_mul_f32(nc, st, st, s1);

        float sum_st = 0.0f;
        wsp_ggml_vec_sum_f32(nc, &sum_st, st);
        sum_thread += sum_st;

#ifndef NDEBUG
        for (int64_t i = 0; i < nc; ++i) {
            assert(!isnan(st[i]));
            assert(!isinf(st[i]));
        }
#endif
    }
    sums[ith] = sum_thread;
    wsp_ggml_barrier(params->threadpool);

    if (ith == 0) {
        float * dp = (float *) dst->data;
        wsp_ggml_vec_sum_f32(nth, dp, sums);
        dp[0] *= -1.0f / (float) nr;
    }
}

static void wsp_ggml_compute_forward_cross_entropy_loss(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_cross_entropy_loss_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// wsp_ggml_compute_forward_cross_entropy_loss_back

static void wsp_ggml_compute_forward_cross_entropy_loss_back_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];
    const struct wsp_ggml_tensor * src1 = dst->src[1];
    const struct wsp_ggml_tensor * opt0 = dst->src[2];

    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(dst));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(src0));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(src1));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(opt0));
    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, src1) && wsp_ggml_are_same_shape(src0, dst));

    const int64_t ith = params->ith;
    const int64_t nth = params->nth;

    // TODO: handle transposed/permuted matrices
    const int64_t nc = src0->ne[0];
    const int64_t nr = wsp_ggml_nrows(src0);

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    const float d_by_nr = ((const float *) opt0->data)[0] / (float) nr;

    for (int64_t i1 = ir0; i1 < ir1; i1++) {
        float * ds0 = (float *)((char *) dst->data  + i1*dst->nb[1]);
        float * s0  = (float *)((char *) src0->data + i1*src0->nb[1]);
        float * s1  = (float *)((char *) src1->data + i1*src1->nb[1]);

#ifndef NDEBUG
        for (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            assert(!isnan(s0[i]));
            assert(!isnan(s1[i]));
        }
#endif

        // soft_max
        float max = -INFINITY;
        wsp_ggml_vec_max_f32(nc, &max, s0);
        wsp_ggml_float sum = wsp_ggml_vec_soft_max_f32(nc, ds0, s0, max);
        assert(sum > 0.0);
        wsp_ggml_vec_scale_f32(nc, ds0, 1.0/sum);

        // grad(src0) = (softmax(src0) - src1) * grad(cross_entropy_loss(src0, src1)) / nr
        wsp_ggml_vec_sub_f32(nc, ds0, ds0, s1);
        wsp_ggml_vec_scale_f32(nc, ds0, d_by_nr);

#ifndef NDEBUG
        for (int64_t i = 0; i < nc; ++i) {
            assert(!isnan(ds0[i]));
            assert(!isinf(ds0[i]));
        }
#endif
    }
}

static void wsp_ggml_compute_forward_cross_entropy_loss_back(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_cross_entropy_loss_back_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

static void wsp_ggml_compute_forward_opt_step_adamw_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0        = dst->src[0];
    const struct wsp_ggml_tensor * src0_grad   = dst->src[1];
    const struct wsp_ggml_tensor * src0_grad_m = dst->src[2];
    const struct wsp_ggml_tensor * src0_grad_v = dst->src[3];
    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(src0, src0_grad));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = wsp_ggml_nrows(src0);

    WSP_GGML_TENSOR_UNARY_OP_LOCALS
    WSP_GGML_ASSERT(nb00 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    /* const float   gnorm = 1.0f; */
    int64_t       iter;   memcpy(&iter, &dst->op_params[0], sizeof(int64_t));
    const float   alpha = wsp_ggml_get_op_params_f32(dst, 2);
    const float   beta1 = wsp_ggml_get_op_params_f32(dst, 3);
    const float   beta2 = wsp_ggml_get_op_params_f32(dst, 4);
    const float   eps   = wsp_ggml_get_op_params_f32(dst, 5);
    const float   wd    = wsp_ggml_get_op_params_f32(dst, 6);

    const float beta1h  = alpha/(1.0f - powf(beta1, iter));
    const float beta2h  =  1.0f/(1.0f - powf(beta2, iter));

    for (int ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        const size_t offset = i03*nb03 + i02*nb02 + i01*nb01;

        float       * w = (float       *) ((char       *) src0->data        + offset); // weight
        const float * g = (const float *) ((const char *) src0_grad->data   + offset); // grad
        float       * m = (float       *) ((char       *) src0_grad_m->data + offset);
        float       * v = (float       *) ((char       *) src0_grad_v->data + offset);

        for (int i00 = 0; i00 < ne00; ++i00) {
            m[i00] = m[i00]*beta1 +        g[i00]*(1.0f - beta1);
            v[i00] = v[i00]*beta2 + g[i00]*g[i00]*(1.0f - beta2);

            const float mh =       m[i00]*beta1h;
            const float vh = sqrtf(v[i00]*beta2h) + eps;

            // The weight decay is applied independently of the Adam momenta m and v.
            // This is NOT equivalent to l2 regularization that adds w[i00]*w[i00] to the loss.
            // See: https://arxiv.org/pdf/1711.05101v3.pdf
            w[i00] = w[i00]*(1.0f - alpha*wd) - mh/vh;
        }
    }

    wsp_ggml_barrier(params->threadpool);
    if (ith != 0) {
        return;
    }

    iter++;
    memcpy(&dst->op_params[0], &iter, sizeof(int64_t));
}

static void wsp_ggml_compute_forward_opt_step_adamw(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_opt_step_adamw_f32(params, dst);
            } break;
        default:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}
/////////////////////////////////

static void wsp_ggml_compute_forward(struct wsp_ggml_compute_params * params, struct wsp_ggml_tensor * tensor) {
    WSP_GGML_ASSERT(params);

    if (tensor->op == WSP_GGML_OP_NONE || wsp_ggml_is_empty(tensor)) {
        return;
    }

    switch (tensor->op) {
        case WSP_GGML_OP_DUP:
            {
                wsp_ggml_compute_forward_dup(params, tensor);
            } break;
        case WSP_GGML_OP_ADD:
            {
                wsp_ggml_compute_forward_add(params, tensor);
            } break;
        case WSP_GGML_OP_ADD1:
            {
                wsp_ggml_compute_forward_add1(params, tensor);
            } break;
        case WSP_GGML_OP_ACC:
            {
                wsp_ggml_compute_forward_acc(params, tensor);
            } break;
        case WSP_GGML_OP_SUB:
            {
                wsp_ggml_compute_forward_sub(params, tensor);
            } break;
        case WSP_GGML_OP_MUL:
            {
                wsp_ggml_compute_forward_mul(params, tensor);
            } break;
        case WSP_GGML_OP_DIV:
            {
                wsp_ggml_compute_forward_div(params, tensor);
            } break;
        case WSP_GGML_OP_SQR:
            {
                wsp_ggml_compute_forward_sqr(params, tensor);
            } break;
        case WSP_GGML_OP_SQRT:
            {
                wsp_ggml_compute_forward_sqrt(params, tensor);
            } break;
        case WSP_GGML_OP_LOG:
            {
                wsp_ggml_compute_forward_log(params, tensor);
            } break;
        case WSP_GGML_OP_SIN:
            {
                wsp_ggml_compute_forward_sin(params, tensor);
            } break;
        case WSP_GGML_OP_COS:
            {
                wsp_ggml_compute_forward_cos(params, tensor);
            } break;
        case WSP_GGML_OP_SUM:
            {
                wsp_ggml_compute_forward_sum(params, tensor);
            } break;
        case WSP_GGML_OP_SUM_ROWS:
            {
                wsp_ggml_compute_forward_sum_rows(params, tensor);
            } break;
        case WSP_GGML_OP_MEAN:
            {
                wsp_ggml_compute_forward_mean(params, tensor);
            } break;
        case WSP_GGML_OP_ARGMAX:
            {
                wsp_ggml_compute_forward_argmax(params, tensor);
            } break;
        case WSP_GGML_OP_COUNT_EQUAL:
            {
                wsp_ggml_compute_forward_count_equal(params, tensor);
            } break;
        case WSP_GGML_OP_REPEAT:
            {
                wsp_ggml_compute_forward_repeat(params, tensor);
            } break;
        case WSP_GGML_OP_REPEAT_BACK:
            {
                wsp_ggml_compute_forward_repeat_back(params, tensor);
            } break;
        case WSP_GGML_OP_CONCAT:
            {
                wsp_ggml_compute_forward_concat(params, tensor);
            } break;
        case WSP_GGML_OP_SILU_BACK:
            {
                wsp_ggml_compute_forward_silu_back(params, tensor);
            } break;
        case WSP_GGML_OP_NORM:
            {
                wsp_ggml_compute_forward_norm(params, tensor);
            } break;
        case WSP_GGML_OP_RMS_NORM:
            {
                wsp_ggml_compute_forward_rms_norm(params, tensor);
            } break;
        case WSP_GGML_OP_RMS_NORM_BACK:
            {
                wsp_ggml_compute_forward_rms_norm_back(params, tensor);
            } break;
        case WSP_GGML_OP_GROUP_NORM:
            {
                wsp_ggml_compute_forward_group_norm(params, tensor);
            } break;
        case WSP_GGML_OP_MUL_MAT:
            {
                wsp_ggml_compute_forward_mul_mat(params, tensor);
            } break;
        case WSP_GGML_OP_MUL_MAT_ID:
            {
                wsp_ggml_compute_forward_mul_mat_id(params, tensor);
            } break;
        case WSP_GGML_OP_OUT_PROD:
            {
                wsp_ggml_compute_forward_out_prod(params, tensor);
            } break;
        case WSP_GGML_OP_SCALE:
            {
                wsp_ggml_compute_forward_scale(params, tensor);
            } break;
        case WSP_GGML_OP_SET:
            {
                wsp_ggml_compute_forward_set(params, tensor);
            } break;
        case WSP_GGML_OP_CPY:
            {
                wsp_ggml_compute_forward_cpy(params, tensor);
            } break;
        case WSP_GGML_OP_CONT:
            {
                wsp_ggml_compute_forward_cont(params, tensor);
            } break;
        case WSP_GGML_OP_RESHAPE:
            {
                wsp_ggml_compute_forward_reshape(params, tensor);
            } break;
        case WSP_GGML_OP_VIEW:
            {
                wsp_ggml_compute_forward_view(params, tensor);
            } break;
        case WSP_GGML_OP_PERMUTE:
            {
                wsp_ggml_compute_forward_permute(params, tensor);
            } break;
        case WSP_GGML_OP_TRANSPOSE:
            {
                wsp_ggml_compute_forward_transpose(params, tensor);
            } break;
        case WSP_GGML_OP_GET_ROWS:
            {
                wsp_ggml_compute_forward_get_rows(params, tensor);
            } break;
        case WSP_GGML_OP_GET_ROWS_BACK:
            {
                wsp_ggml_compute_forward_get_rows_back(params, tensor);
            } break;
        case WSP_GGML_OP_DIAG:
            {
                wsp_ggml_compute_forward_diag(params, tensor);
            } break;
        case WSP_GGML_OP_DIAG_MASK_INF:
            {
                wsp_ggml_compute_forward_diag_mask_inf(params, tensor);
            } break;
        case WSP_GGML_OP_DIAG_MASK_ZERO:
            {
                wsp_ggml_compute_forward_diag_mask_zero(params, tensor);
            } break;
        case WSP_GGML_OP_SOFT_MAX:
            {
                wsp_ggml_compute_forward_soft_max(params, tensor);
            } break;
        case WSP_GGML_OP_SOFT_MAX_BACK:
            {
                wsp_ggml_compute_forward_soft_max_back(params, tensor);
            } break;
        case WSP_GGML_OP_ROPE:
            {
                wsp_ggml_compute_forward_rope(params, tensor);
            } break;
        case WSP_GGML_OP_ROPE_BACK:
            {
                wsp_ggml_compute_forward_rope_back(params, tensor);
            } break;
        case WSP_GGML_OP_CLAMP:
            {
                wsp_ggml_compute_forward_clamp(params, tensor);
            } break;
        case WSP_GGML_OP_CONV_TRANSPOSE_1D:
            {
                wsp_ggml_compute_forward_conv_transpose_1d(params, tensor);
            } break;
        case WSP_GGML_OP_IM2COL:
            {
                wsp_ggml_compute_forward_im2col(params, tensor);
            } break;
        case WSP_GGML_OP_IM2COL_BACK:
            {
                wsp_ggml_compute_forward_im2col_back_f32(params, tensor);
            } break;
        case WSP_GGML_OP_CONV_TRANSPOSE_2D:
            {
                wsp_ggml_compute_forward_conv_transpose_2d(params, tensor);
            } break;
        case WSP_GGML_OP_POOL_1D:
            {
                wsp_ggml_compute_forward_pool_1d(params, tensor);
            } break;
        case WSP_GGML_OP_POOL_2D:
            {
                wsp_ggml_compute_forward_pool_2d(params, tensor);
            } break;
        case WSP_GGML_OP_POOL_2D_BACK:
            {
                wsp_ggml_compute_forward_pool_2d_back(params, tensor);
            } break;
        case WSP_GGML_OP_UPSCALE:
            {
                wsp_ggml_compute_forward_upscale(params, tensor);
            } break;
        case WSP_GGML_OP_PAD:
            {
                wsp_ggml_compute_forward_pad(params, tensor);
            } break;
        case WSP_GGML_OP_ARANGE:
            {
                wsp_ggml_compute_forward_arange(params, tensor);
            } break;
        case WSP_GGML_OP_TIMESTEP_EMBEDDING:
            {
                wsp_ggml_compute_forward_timestep_embedding(params, tensor);
            } break;
        case WSP_GGML_OP_ARGSORT:
            {
                wsp_ggml_compute_forward_argsort(params, tensor);
            } break;
        case WSP_GGML_OP_LEAKY_RELU:
            {
                wsp_ggml_compute_forward_leaky_relu(params, tensor);
            } break;
        case WSP_GGML_OP_FLASH_ATTN_EXT:
            {
                wsp_ggml_compute_forward_flash_attn_ext(params, tensor->src[0], tensor->src[1], tensor->src[2], tensor->src[3], tensor);
            } break;
        case WSP_GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = wsp_ggml_get_op_params_i32(tensor, 0);
                WSP_GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                wsp_ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } break;
        case WSP_GGML_OP_SSM_CONV:
            {
                wsp_ggml_compute_forward_ssm_conv(params, tensor);
            } break;
        case WSP_GGML_OP_SSM_SCAN:
            {
                wsp_ggml_compute_forward_ssm_scan(params, tensor);
            } break;
        case WSP_GGML_OP_WIN_PART:
            {
                wsp_ggml_compute_forward_win_part(params, tensor);
            } break;
        case WSP_GGML_OP_WIN_UNPART:
            {
                wsp_ggml_compute_forward_win_unpart(params, tensor);
            } break;
        case WSP_GGML_OP_UNARY:
            {
                wsp_ggml_compute_forward_unary(params, tensor);
            } break;
        case WSP_GGML_OP_GET_REL_POS:
            {
                wsp_ggml_compute_forward_get_rel_pos(params, tensor);
            } break;
        case WSP_GGML_OP_ADD_REL_POS:
            {
                wsp_ggml_compute_forward_add_rel_pos(params, tensor);
            } break;
        case WSP_GGML_OP_RWKV_WKV6:
            {
                wsp_ggml_compute_forward_rwkv_wkv6(params, tensor);
            } break;
        case WSP_GGML_OP_MAP_UNARY:
            {
                wsp_ggml_unary_op_f32_t fun;
                memcpy(&fun, tensor->op_params, sizeof(fun));
                wsp_ggml_compute_forward_map_unary(params, tensor, fun);
            }
            break;
        case WSP_GGML_OP_MAP_BINARY:
            {
                wsp_ggml_binary_op_f32_t fun;
                memcpy(&fun, tensor->op_params, sizeof(fun));
                wsp_ggml_compute_forward_map_binary(params, tensor, fun);
            }
            break;
        case WSP_GGML_OP_MAP_CUSTOM1_F32:
            {
                wsp_ggml_custom1_op_f32_t fun;
                memcpy(&fun, tensor->op_params, sizeof(fun));
                wsp_ggml_compute_forward_map_custom1_f32(params, tensor, fun);
            }
            break;
        case WSP_GGML_OP_MAP_CUSTOM2_F32:
            {
                wsp_ggml_custom2_op_f32_t fun;
                memcpy(&fun, tensor->op_params, sizeof(fun));
                wsp_ggml_compute_forward_map_custom2_f32(params, tensor, fun);
            }
            break;
        case WSP_GGML_OP_MAP_CUSTOM3_F32:
            {
                wsp_ggml_custom3_op_f32_t fun;
                memcpy(&fun, tensor->op_params, sizeof(fun));
                wsp_ggml_compute_forward_map_custom3_f32(params, tensor, fun);
            }
            break;
        case WSP_GGML_OP_MAP_CUSTOM1:
            {
                wsp_ggml_compute_forward_map_custom1(params, tensor);
            }
            break;
        case WSP_GGML_OP_MAP_CUSTOM2:
            {
                wsp_ggml_compute_forward_map_custom2(params, tensor);
            }
            break;
        case WSP_GGML_OP_MAP_CUSTOM3:
            {
                wsp_ggml_compute_forward_map_custom3(params, tensor);
            }
            break;
        case WSP_GGML_OP_CROSS_ENTROPY_LOSS:
            {
                wsp_ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            break;
        case WSP_GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                wsp_ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            break;
        case WSP_GGML_OP_OPT_STEP_ADAMW:
            {
                wsp_ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            break;
        case WSP_GGML_OP_NONE:
            {
                // nop
            } break;
        case WSP_GGML_OP_COUNT:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#if defined(__gnu_linux__)
static void set_numa_thread_affinity(int thread_n) {
    if (!wsp_ggml_is_numa()) {
        return;
    }

    int node_num;
    int rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    switch(g_state.numa.numa_strategy) {
        case WSP_GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            node_num = thread_n % g_state.numa.n_nodes;
            break;
        case WSP_GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            node_num = g_state.numa.current_node;
            break;
        case WSP_GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            if (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            return;
        default:
            return;
    }

    struct wsp_ggml_numa_node * node = &g_state.numa.nodes[node_num];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (size_t i = 0; i < node->n_cpus; ++i) {
        CPU_SET_S(node->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

static void clear_numa_thread_affinity(void) {
    if (!wsp_ggml_is_numa()) {
        return;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    int rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#else
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
static void set_numa_thread_affinity(int thread_n) { UNUSED(thread_n);  }
static void clear_numa_thread_affinity(void) {}
#endif

static int wsp_ggml_get_n_tasks(struct wsp_ggml_tensor * node, int n_threads) {
    int n_tasks = 0;

    if (wsp_ggml_is_empty(node)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        return n_tasks;
    }

    switch (node->op) {
        case WSP_GGML_OP_CPY:
        case WSP_GGML_OP_DUP:
        case WSP_GGML_OP_CONT:
        case WSP_GGML_OP_ADD:
        case WSP_GGML_OP_ADD1:
        case WSP_GGML_OP_ACC:
            {
                n_tasks = n_threads;
            } break;
        case WSP_GGML_OP_SUB:
        case WSP_GGML_OP_SQR:
        case WSP_GGML_OP_SQRT:
        case WSP_GGML_OP_LOG:
        case WSP_GGML_OP_SIN:
        case WSP_GGML_OP_COS:
        case WSP_GGML_OP_SUM:
        case WSP_GGML_OP_SUM_ROWS:
        case WSP_GGML_OP_MEAN:
        case WSP_GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } break;
        case WSP_GGML_OP_COUNT_EQUAL:
            {
                n_tasks = n_threads;
            } break;
        case WSP_GGML_OP_REPEAT:
        case WSP_GGML_OP_REPEAT_BACK:
        case WSP_GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } break;
        case WSP_GGML_OP_UNARY:
            switch (wsp_ggml_get_unary_op(node)) {
                case WSP_GGML_UNARY_OP_ABS:
                case WSP_GGML_UNARY_OP_SGN:
                case WSP_GGML_UNARY_OP_NEG:
                case WSP_GGML_UNARY_OP_STEP:
                case WSP_GGML_UNARY_OP_TANH:
                case WSP_GGML_UNARY_OP_ELU:
                case WSP_GGML_UNARY_OP_RELU:
                case WSP_GGML_UNARY_OP_SIGMOID:
                case WSP_GGML_UNARY_OP_HARDSWISH:
                case WSP_GGML_UNARY_OP_HARDSIGMOID:
                case WSP_GGML_UNARY_OP_EXP:
                    {
                        n_tasks = 1;
                    } break;

                case WSP_GGML_UNARY_OP_GELU:
                case WSP_GGML_UNARY_OP_GELU_QUICK:
                case WSP_GGML_UNARY_OP_SILU:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    WSP_GGML_ABORT("fatal error");
            }
            break;
        case WSP_GGML_OP_SILU_BACK:
        case WSP_GGML_OP_MUL:
        case WSP_GGML_OP_DIV:
        case WSP_GGML_OP_NORM:
        case WSP_GGML_OP_RMS_NORM:
        case WSP_GGML_OP_RMS_NORM_BACK:
        case WSP_GGML_OP_GROUP_NORM:
        case WSP_GGML_OP_CONCAT:
        case WSP_GGML_OP_MUL_MAT:
        case WSP_GGML_OP_MUL_MAT_ID:
        case WSP_GGML_OP_OUT_PROD:
            {
                n_tasks = n_threads;
            } break;
        case WSP_GGML_OP_GET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } break;
        case WSP_GGML_OP_SCALE:
        case WSP_GGML_OP_SET:
        case WSP_GGML_OP_RESHAPE:
        case WSP_GGML_OP_VIEW:
        case WSP_GGML_OP_PERMUTE:
        case WSP_GGML_OP_TRANSPOSE:
        case WSP_GGML_OP_GET_ROWS_BACK:
        case WSP_GGML_OP_DIAG:
            {
                n_tasks = 1;
            } break;
        case WSP_GGML_OP_DIAG_MASK_ZERO:
        case WSP_GGML_OP_DIAG_MASK_INF:
        case WSP_GGML_OP_SOFT_MAX_BACK:
        case WSP_GGML_OP_ROPE:
        case WSP_GGML_OP_ROPE_BACK:
        case WSP_GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } break;
        case WSP_GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } break;
        case WSP_GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, wsp_ggml_nrows(node->src[0]));
            } break;
        case WSP_GGML_OP_IM2COL:
        case WSP_GGML_OP_IM2COL_BACK:
        case WSP_GGML_OP_CONV_TRANSPOSE_1D:
        case WSP_GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } break;
        case WSP_GGML_OP_POOL_1D:
        case WSP_GGML_OP_POOL_2D:
        case WSP_GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } break;
        case WSP_GGML_OP_UPSCALE:
        case WSP_GGML_OP_PAD:
        case WSP_GGML_OP_ARANGE:
        case WSP_GGML_OP_TIMESTEP_EMBEDDING:
        case WSP_GGML_OP_ARGSORT:
        case WSP_GGML_OP_FLASH_ATTN_EXT:
        case WSP_GGML_OP_FLASH_ATTN_BACK:
        case WSP_GGML_OP_SSM_CONV:
        case WSP_GGML_OP_SSM_SCAN:
            {
                n_tasks = n_threads;
            } break;
        case WSP_GGML_OP_WIN_PART:
        case WSP_GGML_OP_WIN_UNPART:
        case WSP_GGML_OP_GET_REL_POS:
        case WSP_GGML_OP_RWKV_WKV6:
        case WSP_GGML_OP_MAP_UNARY:
        case WSP_GGML_OP_MAP_BINARY:
        case WSP_GGML_OP_MAP_CUSTOM1_F32:
        case WSP_GGML_OP_MAP_CUSTOM2_F32:
        case WSP_GGML_OP_MAP_CUSTOM3_F32:
            {
                n_tasks = 1;
            } break;
        case WSP_GGML_OP_MAP_CUSTOM1:
            {
                struct wsp_ggml_map_custom1_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == WSP_GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case WSP_GGML_OP_MAP_CUSTOM2:
            {
                struct wsp_ggml_map_custom2_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == WSP_GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case WSP_GGML_OP_MAP_CUSTOM3:
            {
                struct wsp_ggml_map_custom3_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == WSP_GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case WSP_GGML_OP_CROSS_ENTROPY_LOSS:
        case WSP_GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case WSP_GGML_OP_OPT_STEP_ADAMW:
            {
                n_tasks = n_threads;
            } break;
        case WSP_GGML_OP_NONE:
            {
                n_tasks = 1;
            } break;
        case WSP_GGML_OP_COUNT:
            {
                WSP_GGML_ABORT("fatal error");
            }
        default:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                if (node->op < WSP_GGML_OP_COUNT) {
                    fprintf(stderr, "%s\n", wsp_ggml_op_name(node->op));
                } else {
                    fprintf(stderr, "%d\n", node->op);
                }
                WSP_GGML_ABORT("fatal error");
            }
    }

    assert(n_tasks > 0);

    return n_tasks;
}

static thread_ret_t wsp_ggml_graph_compute_secondary_thread(void* data);

#if defined(_WIN32)
#include "windows.h"

// TODO: support > 64 CPUs
bool wsp_ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    assert(WSP_GGML_MAX_N_THREADS >= 64);

    for (int32_t i = 0; i < 8; i++) {
        int32_t idx = i * 8;
        uint8_t val = 0;
        val |= mask[idx + 0] << 0;
        val |= mask[idx + 1] << 1;
        val |= mask[idx + 2] << 2;
        val |= mask[idx + 3] << 3;
        val |= mask[idx + 4] << 4;
        val |= mask[idx + 5] << 5;
        val |= mask[idx + 6] << 6;
        val |= mask[idx + 7] << 7;
        bitmask |= (uint64_t)val << idx;
    }

    for (int32_t i = 64; i < WSP_GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            break;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    return m != 0;
}

static bool wsp_ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    switch (prio) {
        case WSP_GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        break;
        case WSP_GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  break;
        case WSP_GGML_SCHED_PRIO_HIGH:     p = THREAD_PRIORITY_HIGHEST;       break;
        case WSP_GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; break;
    }

    if (prio == WSP_GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    if (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/resource.h>

static bool wsp_ggml_thread_apply_affinity(const bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    return true;
}

static bool wsp_ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        case WSP_GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case WSP_GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case WSP_GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case WSP_GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == WSP_GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#elif defined(__gnu_linux__)
// TODO: this may not work on BSD, to be verified

static bool wsp_ggml_thread_apply_affinity(const bool * mask) {
    cpu_set_t cpuset;
    int err;

    CPU_ZERO(&cpuset);

    for (uint32_t i = 0; i < WSP_GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            WSP_GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, sizeof(cpuset), &cpuset);
    if (err < 0) {
        err = errno;
    }
#else
    err = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
    if (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        return false;
    }

    return true;
}

static bool wsp_ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        case WSP_GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case WSP_GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case WSP_GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case WSP_GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == WSP_GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#else // unsupported platforms

static bool wsp_ggml_thread_apply_affinity(const bool * mask) {
    UNUSED(mask);
    return true;
}

static bool wsp_ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    return true;
}

#endif

static bool wsp_ggml_thread_cpumask_is_valid(const bool * mask) {
    for (int i = 0; i < WSP_GGML_MAX_N_THREADS; i++) {
        if (mask[i]) { return true; }
    }
    return false;
}

static void wsp_ggml_thread_cpumask_next(const bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    if (!strict) {
        memcpy(local_mask, global_mask, WSP_GGML_MAX_N_THREADS);
        return;
    } else {
        memset(local_mask, 0, WSP_GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        for (int32_t i = 0; i < WSP_GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            if (idx >= WSP_GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= WSP_GGML_MAX_N_THREADS;
            }
            if (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                return;
            }
        }
    }
}

void wsp_ggml_threadpool_free(struct wsp_ggml_threadpool* threadpool) {
    if (!threadpool) return;

    const int n_threads = threadpool->n_threads_max;

#ifndef WSP_GGML_USE_OPENMP
    struct wsp_ggml_compute_state* workers = threadpool->workers;

    wsp_ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    wsp_ggml_cond_broadcast(&threadpool->cond);
    wsp_ggml_mutex_unlock(&threadpool->mutex);

    for (int j = 1; j < n_threads; j++) {
        int32_t rc = wsp_ggml_thread_join(workers[j].thrd, NULL);
        WSP_GGML_ASSERT(rc == WSP_GGML_EXIT_SUCCESS || rc == WSP_GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    wsp_ggml_mutex_destroy(&threadpool->mutex);
    wsp_ggml_cond_destroy(&threadpool->cond);
#endif // WSP_GGML_USE_OPENMP

    const size_t workers_size = sizeof(struct wsp_ggml_compute_state) * n_threads;
    wsp_ggml_aligned_free(threadpool->workers, workers_size);
    wsp_ggml_aligned_free(threadpool, sizeof(struct wsp_ggml_threadpool));
}

#ifndef WSP_GGML_USE_OPENMP
// pause/resume must be called under mutex
static void wsp_ggml_threadpool_pause_locked(struct wsp_ggml_threadpool * threadpool) {
    WSP_GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    wsp_ggml_cond_broadcast(&threadpool->cond);
}

static void wsp_ggml_threadpool_resume_locked(struct wsp_ggml_threadpool * threadpool) {
    WSP_GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    wsp_ggml_cond_broadcast(&threadpool->cond);
}
#endif

void wsp_ggml_threadpool_pause(struct wsp_ggml_threadpool * threadpool) {
#ifndef WSP_GGML_USE_OPENMP
    wsp_ggml_mutex_lock(&threadpool->mutex);
    if (!threadpool->pause) {
       wsp_ggml_threadpool_pause_locked(threadpool);
    }
    wsp_ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

void wsp_ggml_threadpool_resume(struct wsp_ggml_threadpool * threadpool) {
#ifndef WSP_GGML_USE_OPENMP
    wsp_ggml_mutex_lock(&threadpool->mutex);
    if (threadpool->pause) {
       wsp_ggml_threadpool_resume_locked(threadpool);
    }
    wsp_ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

struct wsp_ggml_cplan wsp_ggml_graph_plan(
          const struct wsp_ggml_cgraph * cgraph,
                               int   n_threads,
            struct wsp_ggml_threadpool * threadpool) {

    if (threadpool == NULL) {
        //WSP_GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    if (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads_max : WSP_GGML_DEFAULT_N_THREADS;
    }

    size_t work_size = 0;

    struct wsp_ggml_cplan cplan;
    memset(&cplan, 0, sizeof(struct wsp_ggml_cplan));

    int max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct wsp_ggml_tensor * node = cgraph->nodes[i];

        const int n_tasks = wsp_ggml_get_n_tasks(node, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        switch (node->op) {
            case WSP_GGML_OP_CPY:
            case WSP_GGML_OP_DUP:
                {
                    if (wsp_ggml_is_quantized(node->type) ||
                        // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                        (node->src[0]->type == WSP_GGML_TYPE_F16  && node->src[1] && node->src[1]->type == WSP_GGML_TYPE_BF16) ||
                        (node->src[0]->type == WSP_GGML_TYPE_BF16 && node->src[1] && node->src[1]->type == WSP_GGML_TYPE_F16)) {
                        cur = wsp_ggml_type_size(WSP_GGML_TYPE_F32) * node->ne[0] * n_tasks;
                    }
                } break;
            case WSP_GGML_OP_ADD:
            case WSP_GGML_OP_ADD1:
                {
                    if (wsp_ggml_is_quantized(node->src[0]->type)) {
                        cur = wsp_ggml_type_size(WSP_GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                    }
                } break;
            case WSP_GGML_OP_ACC:
                {
                    if (wsp_ggml_is_quantized(node->src[0]->type)) {
                        cur = wsp_ggml_type_size(WSP_GGML_TYPE_F32) * node->src[1]->ne[0] * n_tasks;
                    }
                } break;
            case WSP_GGML_OP_COUNT_EQUAL:
                {
                    cur = wsp_ggml_type_size(node->type)*n_tasks;
                } break;
            case WSP_GGML_OP_MUL_MAT:
                {
                    const enum wsp_ggml_type vec_dot_type = type_traits_cpu[node->src[0]->type].vec_dot_type;

                    if (node->src[1]->type != vec_dot_type) {
                        cur = wsp_ggml_row_size(vec_dot_type, wsp_ggml_nelements(node->src[1]));
                    }
                } break;
            case WSP_GGML_OP_MUL_MAT_ID:
                {
                    cur = 0;
                    const struct wsp_ggml_tensor * src0 = node->src[0];
                    const struct wsp_ggml_tensor * src1 = node->src[1];
                    const enum wsp_ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                    if (src1->type != vec_dot_type) {
                        cur += wsp_ggml_row_size(vec_dot_type, wsp_ggml_nelements(src1));
                    }
                    const int n_as = src0->ne[2];
                    cur += WSP_GGML_PAD(cur, sizeof(int64_t));       // align
                    cur += n_as * sizeof(int64_t);               // matrix_row_counts
                    cur += n_as * src1->ne[2] * sizeof(int64_t); // matrix_rows
                } break;
            case WSP_GGML_OP_OUT_PROD:
                {
                    if (wsp_ggml_is_quantized(node->src[0]->type)) {
                        cur = wsp_ggml_type_size(WSP_GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                    }
                } break;
            case WSP_GGML_OP_SOFT_MAX:
            case WSP_GGML_OP_ROPE:
                {
                    cur = wsp_ggml_type_size(WSP_GGML_TYPE_F32) * node->ne[0] * n_tasks;
                } break;
            case WSP_GGML_OP_CONV_TRANSPOSE_1D:
                {
                    WSP_GGML_ASSERT(node->src[0]->ne[3] == 1);
                    WSP_GGML_ASSERT(node->src[1]->ne[2] == 1);
                    WSP_GGML_ASSERT(node->src[1]->ne[3] == 1);

                    const int64_t ne00 = node->src[0]->ne[0];  // K
                    const int64_t ne01 = node->src[0]->ne[1];  // Cout
                    const int64_t ne02 = node->src[0]->ne[2];  // Cin

                    const int64_t ne10 = node->src[1]->ne[0];  // L
                    const int64_t ne11 = node->src[1]->ne[1];  // Cin

                    if ((node->src[0]->type == WSP_GGML_TYPE_F16 ||
                         node->src[0]->type == WSP_GGML_TYPE_BF16) &&
                        node->src[1]->type == WSP_GGML_TYPE_F32) {
                        cur += sizeof(wsp_ggml_fp16_t)*ne00*ne01*ne02;
                        cur += sizeof(wsp_ggml_fp16_t)*ne10*ne11;
                    } else if (node->src[0]->type == WSP_GGML_TYPE_F32 &&
                               node->src[1]->type == WSP_GGML_TYPE_F32) {
                        cur += sizeof(float)*ne00*ne01*ne02;
                        cur += sizeof(float)*ne10*ne11;
                    } else {
                        WSP_GGML_ABORT("fatal error");
                    }
                } break;
            case WSP_GGML_OP_CONV_TRANSPOSE_2D:
                {
                    const int64_t ne00 = node->src[0]->ne[0]; // W
                    const int64_t ne01 = node->src[0]->ne[1]; // H
                    const int64_t ne02 = node->src[0]->ne[2]; // Channels Out
                    const int64_t ne03 = node->src[0]->ne[3]; // Channels In

                    const int64_t ne10 = node->src[1]->ne[0]; // W
                    const int64_t ne11 = node->src[1]->ne[1]; // H
                    const int64_t ne12 = node->src[1]->ne[2]; // Channels In

                    cur += sizeof(wsp_ggml_fp16_t)*ne00*ne01*ne02*ne03;
                    cur += sizeof(wsp_ggml_fp16_t)*ne10*ne11*ne12;
                } break;
            case WSP_GGML_OP_FLASH_ATTN_EXT:
                {
                    const int64_t ne00 = node->src[0]->ne[0]; // D

                    cur = 3*sizeof(float)*ne00*n_tasks; // 3x head size/thread
                } break;
            case WSP_GGML_OP_FLASH_ATTN_BACK:
                {
                    const int64_t    D = node->src[0]->ne[0];
                    const int64_t ne11 = wsp_ggml_up(node->src[1]->ne[1], WSP_GGML_SOFT_MAX_UNROLL);
                    const int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in wsp_ggml_compute_forward_flash_attn_back
                    if (node->src[1]->type == WSP_GGML_TYPE_F32) {
                        cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                        cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                    } else if (node->src[1]->type == WSP_GGML_TYPE_F16) {
                        cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                        cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                    } else if (node->src[1]->type == WSP_GGML_TYPE_BF16) {
                        cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                        cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                    }
                } break;

            case WSP_GGML_OP_CROSS_ENTROPY_LOSS:
                {
                    cur = wsp_ggml_type_size(node->type)*(n_tasks + node->src[0]->ne[0]*n_tasks);
                } break;
            case WSP_GGML_OP_COUNT:
                {
                    WSP_GGML_ABORT("fatal error");
                }
            default:
                break;
        }

        work_size = MAX(work_size, cur);
    }

    if (work_size > 0) {
        work_size += CACHE_LINE_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = NULL;

    return cplan;
}

static thread_ret_t wsp_ggml_graph_compute_thread(void * data) {
    struct wsp_ggml_compute_state * state = (struct wsp_ggml_compute_state *) data;
    struct wsp_ggml_threadpool    * tp    = state->threadpool;

    const struct wsp_ggml_cgraph * cgraph = tp->cgraph;
    const struct wsp_ggml_cplan  * cplan  = tp->cplan;

    set_numa_thread_affinity(state->ith);

    struct wsp_ggml_compute_params params = {
        /*.ith       =*/ state->ith,
        /*.nth       =*/ atomic_load_explicit(&tp->n_threads_cur, memory_order_relaxed),
        /*.wsize     =*/ cplan->work_size,
        /*.wdata     =*/ cplan->work_data,
        /*.threadpool=*/ tp,
    };

    for (int node_n = 0; node_n < cgraph->n_nodes && !tp->abort; node_n++) {
        struct wsp_ggml_tensor * node = cgraph->nodes[node_n];

        wsp_ggml_compute_forward(&params, node);

        if (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            tp->abort = true;
            tp->ec    = WSP_GGML_STATUS_ABORTED;
        }

        wsp_ggml_barrier(state->threadpool);
    }

    return 0;
}

#ifndef WSP_GGML_USE_OPENMP

// check if thread is active
static inline bool wsp_ggml_graph_compute_thread_active(struct wsp_ggml_compute_state * state) {
    struct wsp_ggml_threadpool * threadpool = state->threadpool;
    int n_threads = atomic_load_explicit(&threadpool->n_threads_cur, memory_order_relaxed);
    return (state->ith < n_threads);
}

// check if thread is ready to proceed (exit from polling or sleeping)
static inline bool wsp_ggml_graph_compute_thread_ready(struct wsp_ggml_compute_state * state) {
    struct wsp_ggml_threadpool * threadpool = state->threadpool;

    if (state->pending || threadpool->stop || threadpool->pause) { return true; }

    // check for new graph/work
    int new_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    if (new_graph != state->last_graph) {
        state->pending    = wsp_ggml_graph_compute_thread_active(state);
        state->last_graph = new_graph;
    }

    return state->pending;
}

// sync thread state after polling
static inline void wsp_ggml_graph_compute_thread_sync(struct wsp_ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef WSP_GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

static inline bool wsp_ggml_graph_compute_poll_for_work(struct wsp_ggml_compute_state * state) {
    struct wsp_ggml_threadpool * threadpool = state->threadpool;

    // Skip polling for unused threads
    if (!wsp_ggml_graph_compute_thread_active(state)) {
        return state->pending;
    }

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    const uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    for (uint64_t i=0; !wsp_ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        wsp_ggml_thread_cpu_relax();
    }

    return state->pending;
}

static inline bool wsp_ggml_graph_compute_check_for_work(struct wsp_ggml_compute_state * state) {
    struct wsp_ggml_threadpool * threadpool = state->threadpool;

    if (wsp_ggml_graph_compute_poll_for_work(state)) {
        wsp_ggml_graph_compute_thread_sync(state);
        return state->pending;
    }

    wsp_ggml_mutex_lock_shared(&threadpool->mutex);
    while (!wsp_ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        WSP_GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        wsp_ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    wsp_ggml_mutex_unlock_shared(&threadpool->mutex);

    return state->pending;
}

static thread_ret_t wsp_ggml_graph_compute_secondary_thread(void* data) {
    struct wsp_ggml_compute_state * state = (struct wsp_ggml_compute_state *) data;
    struct wsp_ggml_threadpool * threadpool = state->threadpool;

    wsp_ggml_thread_apply_priority(threadpool->prio);
    if (wsp_ggml_thread_cpumask_is_valid(state->cpumask)) {
        wsp_ggml_thread_apply_affinity(state->cpumask);
    }

    while (true) {
        // Check if we need to sleep
        while (threadpool->pause) {
            WSP_GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            wsp_ggml_mutex_lock_shared(&threadpool->mutex);
            if (threadpool->pause) {
                wsp_ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            WSP_GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            wsp_ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        if (threadpool->stop) break;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        wsp_ggml_graph_compute_check_for_work(state);
        if (state->pending) {
            state->pending = false;

            wsp_ggml_graph_compute_thread(state);
        }
    }

    return (thread_ret_t) 0;
}

// Start processing new graph
static void wsp_ggml_graph_compute_kickoff(struct wsp_ggml_threadpool * threadpool, int n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    wsp_ggml_mutex_lock(&threadpool->mutex);

    WSP_GGML_PRINT_DEBUG("threadpool: n_threads_cur %d n_threads %d\n", threadpool->n_threads_cur, n_threads);

    // Update the number of active threads
    atomic_store_explicit(&threadpool->n_threads_cur, n_threads, memory_order_relaxed);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_fetch_add_explicit(&threadpool->n_graph, 1, memory_order_seq_cst);

    if (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       wsp_ggml_thread_apply_priority(threadpool->prio);
       if (wsp_ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           wsp_ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       wsp_ggml_threadpool_resume_locked(threadpool);
    } else {
       wsp_ggml_cond_broadcast(&threadpool->cond);
    }

    wsp_ggml_mutex_unlock(&threadpool->mutex);
}

#endif // WSP_GGML_USE_OPENMP

void wsp_ggml_threadpool_params_init(struct wsp_ggml_threadpool_params * p, int n_threads) {
    p->n_threads  = n_threads;
    p->prio       = 0;     // default priority (usually means normal or inherited)
    p->poll       = 50;    // hybrid-polling enabled
    p->strict_cpu = false; // no strict placement (all threads share same cpumask)
    p->paused     = false; // threads are ready to go
    memset(p->cpumask, 0, WSP_GGML_MAX_N_THREADS); // all-zero means use the default affinity (usually inherited)
}

struct wsp_ggml_threadpool_params wsp_ggml_threadpool_params_default(int n_threads) {
    struct wsp_ggml_threadpool_params p;
    wsp_ggml_threadpool_params_init(&p, n_threads);
    return p;
}

bool wsp_ggml_threadpool_params_match(const struct wsp_ggml_threadpool_params * p0, const struct wsp_ggml_threadpool_params * p1) {
    if (p0->n_threads      != p1->n_threads  )    return false;
    if (p0->prio           != p1->prio       )    return false;
    if (p0->poll           != p1->poll       )    return false;
    if (p0->strict_cpu     != p1->strict_cpu )    return false;
    return memcmp(p0->cpumask, p1->cpumask, WSP_GGML_MAX_N_THREADS) == 0;
}

static struct wsp_ggml_threadpool * wsp_ggml_threadpool_new_impl(
    struct wsp_ggml_threadpool_params * tpp,
               struct wsp_ggml_cgraph * cgraph,
                struct wsp_ggml_cplan * cplan) {

    struct wsp_ggml_threadpool * threadpool =
        wsp_ggml_aligned_malloc(sizeof(struct wsp_ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_graph          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = false;
        threadpool->workers          = NULL;
        threadpool->n_threads_max    = tpp->n_threads;
        threadpool->n_threads_cur    = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = WSP_GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    const size_t workers_size = sizeof(struct wsp_ggml_compute_state) * tpp->n_threads;
    struct wsp_ggml_compute_state * workers = wsp_ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    for (int j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ifndef WSP_GGML_USE_OPENMP
    wsp_ggml_mutex_init(&threadpool->mutex);
    wsp_ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    for (int j = 1; j < tpp->n_threads; j++) {
        wsp_ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = wsp_ggml_thread_create(&workers[j].thrd, NULL, wsp_ggml_graph_compute_secondary_thread, &workers[j]);
        WSP_GGML_ASSERT(rc == 0);
    }

    wsp_ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    if (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        wsp_ggml_thread_apply_priority(threadpool->prio);
        if (wsp_ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            wsp_ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // WSP_GGML_USE_OPENMP

    return threadpool;
}

struct wsp_ggml_threadpool * wsp_ggml_threadpool_new(struct wsp_ggml_threadpool_params * tpp) {
    return wsp_ggml_threadpool_new_impl(tpp, NULL, NULL);
}

enum wsp_ggml_status wsp_ggml_graph_compute(struct wsp_ggml_cgraph * cgraph, struct wsp_ggml_cplan * cplan) {
    wsp_ggml_cpu_init();

    WSP_GGML_ASSERT(cplan);
    WSP_GGML_ASSERT(cplan->n_threads > 0);
    WSP_GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != NULL);

    int n_threads                               = cplan->n_threads;
    struct wsp_ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    if (threadpool == NULL) {
        //WSP_GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        struct wsp_ggml_threadpool_params ttp = wsp_ggml_threadpool_params_default(n_threads);
        threadpool = wsp_ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } else {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = false;
        threadpool->ec               = WSP_GGML_STATUS_SUCCESS;
    }

#ifdef WSP_GGML_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_num_threads();
                atomic_store_explicit(&threadpool->n_threads_cur, n_threads, memory_order_relaxed);
            }

            wsp_ggml_graph_compute_thread(&threadpool->workers[omp_get_thread_num()]);
        }
    } else {
        atomic_store_explicit(&threadpool->n_threads_cur, 1, memory_order_relaxed);
        wsp_ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#else
    if (n_threads > threadpool->n_threads_max) {
        WSP_GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, threadpool->n_threads_max);
        n_threads = threadpool->n_threads_max;
    }

    // Kick all threads to start the new graph
    wsp_ggml_graph_compute_kickoff(threadpool, n_threads);

    // This is a work thread too
    wsp_ggml_graph_compute_thread(&threadpool->workers[0]);
#endif

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    enum wsp_ggml_status ret = threadpool->ec;

    if (disposable_threadpool) {
        wsp_ggml_threadpool_free(threadpool);
    }

    return ret;
}

enum wsp_ggml_status wsp_ggml_graph_compute_with_ctx(struct wsp_ggml_context * ctx, struct wsp_ggml_cgraph * cgraph, int n_threads) {
    struct wsp_ggml_cplan cplan = wsp_ggml_graph_plan(cgraph, n_threads, NULL);

    cplan.work_data = (uint8_t *)wsp_ggml_new_buffer(ctx, cplan.work_size);

    return wsp_ggml_graph_compute(cgraph, &cplan);
}

int wsp_ggml_cpu_has_neon(void) {
#if defined(__ARM_ARCH)
    return wsp_ggml_arm_arch_features.has_neon;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_sve(void) {
#if defined(__ARM_ARCH)
    return wsp_ggml_arm_arch_features.has_sve;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_matmul_int8(void) {
#if defined(__ARM_ARCH)
    return wsp_ggml_arm_arch_features.has_i8mm;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_get_sve_cnt(void) {
#if defined(__ARM_ARCH)
    return wsp_ggml_arm_arch_features.sve_cnt;
#else
    return 0;
#endif
}

void wsp_ggml_cpu_init(void) {
    // needed to initialize f16 tables
    {
        struct wsp_ggml_init_params params = { 0, NULL, false };
        struct wsp_ggml_context * ctx = wsp_ggml_init(params);
        wsp_ggml_free(ctx);
    }

    wsp_ggml_critical_section_start();

    static bool is_first_call = true;

    if (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            const uint64_t t_start = wsp_ggml_time_us(); UNUSED(t_start);

            for (int i = 0; i < (1 << 16); ++i) {
                union {
                    uint16_t u16;
                    wsp_ggml_fp16_t fp16;
                } u = {i};
                float f = WSP_GGML_FP16_TO_FP32(u.fp16);
                wsp_ggml_table_gelu_f16[i] = WSP_GGML_FP32_TO_FP16(wsp_ggml_gelu_f32(f));
                wsp_ggml_table_gelu_quick_f16[i] = WSP_GGML_FP32_TO_FP16(wsp_ggml_gelu_quick_f32(f));
            }

            const uint64_t t_end = wsp_ggml_time_us(); UNUSED(t_end);

            WSP_GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);
        }

#if defined(__ARM_ARCH)
        wsp_ggml_init_arm_arch_features();
#endif

        is_first_call = false;
    }

    wsp_ggml_critical_section_end();
}
