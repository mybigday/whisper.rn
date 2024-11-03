#define _CRT_SECURE_NO_DEPRECATE // Disables ridiculous "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml-quants.h"
#include "ggml.h"
#include "ggml-aarch64.h"

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

#if (defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)) && \
    (!defined(TARGET_OS_TV) && !defined(TARGET_OS_WATCH))

#include <sys/wait.h>

#if defined(__ANDROID__)
#include <unwind.h>
#include <dlfcn.h>
#include <stdio.h>

struct backtrace_state {
    void ** current;
    void ** end;
};

static _Unwind_Reason_Code unwind_callback(struct _Unwind_Context* context, void* arg) {
    struct backtrace_state * state = (struct backtrace_state *)arg;
    uintptr_t pc = _Unwind_GetIP(context);
    if (pc) {
        if (state->current == state->end) {
            return _URC_END_OF_STACK;
        } else {
            *state->current++ = (void*)pc;
        }
    }
    return _URC_NO_REASON;
}

static void wsp_ggml_print_backtrace_symbols(void) {
    const int max = 100;
    void* buffer[max];

    struct backtrace_state state = {buffer, buffer + max};
    _Unwind_Backtrace(unwind_callback, &state);

    int count = state.current - buffer;

    for (int idx = 0; idx < count; ++idx) {
        const void * addr = buffer[idx];
        const char * symbol = "";

        Dl_info info;
        if (dladdr(addr, &info) && info.dli_sname) {
            symbol = info.dli_sname;
        }

        fprintf(stderr, "%d: %p %s\n", idx, addr, symbol);
    }
}
#elif defined(__linux__) && defined(__GLIBC__)
#include <execinfo.h>
static void wsp_ggml_print_backtrace_symbols(void) {
    void * trace[100];
    int nptrs = backtrace(trace, sizeof(trace)/sizeof(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#else
static void wsp_ggml_print_backtrace_symbols(void) {
    // platform not supported
}
#endif

static void wsp_ggml_print_backtrace(void) {
    char attach[32];
    snprintf(attach, sizeof(attach), "attach %d", getpid());
    int pid = fork();
    if (pid == 0) {
        // try gdb
        execlp("gdb", "gdb", "--batch",
            "-ex", "set style enabled on",
            "-ex", attach,
            "-ex", "bt -frame-info source-and-location",
            "-ex", "detach",
            "-ex", "quit",
            (char *) NULL);
        // try lldb
        execlp("lldb", "lldb", "--batch",
            "-o", "bt",
            "-o", "quit",
            "-p", attach,
            (char *) NULL);
        exit(EXIT_FAILURE);
    } else {
        int wstatus;
        waitpid(pid, &wstatus, 0);
        if (WIFEXITED(wstatus)) {
            if (WEXITSTATUS(wstatus) == EXIT_FAILURE) {
                // gdb failed, fallback to backtrace_symbols
                wsp_ggml_print_backtrace_symbols();
            }
        }
    }
}
#else
static void wsp_ggml_print_backtrace(void) {
    // platform not supported
}
#endif

void wsp_ggml_abort(const char * file, int line, const char * fmt, ...) {
    fflush(stdout);

    fprintf(stderr, "%s:%d: ", file, line);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fprintf(stderr, "\n");

    wsp_ggml_print_backtrace();
    abort();
}

#define WSP_GGML_DEBUG 0

#define WSP_GGML_GELU_FP16
#define WSP_GGML_GELU_QUICK_FP16

#define WSP_GGML_SOFT_MAX_UNROLL 4
#define WSP_GGML_VEC_DOT_UNROLL  2
#define WSP_GGML_VEC_MAD_UNROLL  32

//
// logging
//

struct wsp_ggml_logger_state {
    wsp_ggml_log_callback log_callback;
    void * log_callback_user_data;
};
static struct wsp_ggml_logger_state g_logger_state = {wsp_ggml_log_callback_default, NULL};

static void wsp_ggml_log_internal_v(enum wsp_ggml_log_level level, const char * format, va_list args) {
    if (format == NULL) {
        return;
    }
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[128];
    int len = vsnprintf(buffer, 128, format, args);
    if (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } else {
        char * buffer2 = (char *) calloc(len + 1, sizeof(char));
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        free(buffer2);
    }
    va_end(args_copy);
}

void wsp_ggml_log_internal(enum wsp_ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    wsp_ggml_log_internal_v(level, format, args);
    va_end(args);
}

void wsp_ggml_log_callback_default(enum wsp_ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

#if (WSP_GGML_DEBUG >= 1)
#define WSP_GGML_PRINT_DEBUG(...) WSP_GGML_LOG_DEBUG(__VA_ARGS__)
#else
#define WSP_GGML_PRINT_DEBUG(...)
#endif

#if (WSP_GGML_DEBUG >= 5)
#define WSP_GGML_PRINT_DEBUG_5(...) WSP_GGML_LOG_DEBUG(__VA_ARGS__)
#else
#define WSP_GGML_PRINT_DEBUG_5(...)
#endif

#if (WSP_GGML_DEBUG >= 10)
#define WSP_GGML_PRINT_DEBUG_10(...) WSP_GGML_LOG_DEBUG(__VA_ARGS__)
#else
#define WSP_GGML_PRINT_DEBUG_10(...)
#endif

//
// end of logging block
//

#ifdef WSP_GGML_USE_ACCELERATE
// uncomment to use vDSP for soft max computation
// note: not sure if it is actually faster
//#define WSP_GGML_SOFT_MAX_ACCELERATE
#endif


void * wsp_ggml_aligned_malloc(size_t size) {
#if defined(_MSC_VER) || defined(__MINGW32__)
    return _aligned_malloc(size, TENSOR_ALIGNMENT);
#else
    if (size == 0) {
        WSP_GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for wsp_ggml_aligned_malloc!\n");
        return NULL;
    }
    void * aligned_memory = NULL;
#ifdef WSP_GGML_USE_CPU_HBM
    int result = hbw_posix_memalign(&aligned_memory, TENSOR_ALIGNMENT, size);
#elif TARGET_OS_OSX
    kern_return_t alloc_status = vm_allocate((vm_map_t) mach_task_self(), (vm_address_t *) &aligned_memory, size, VM_FLAGS_ANYWHERE);
    int result = EFAULT;
    switch (alloc_status) {
        case KERN_SUCCESS:
            result = 0;
            break;
        case KERN_INVALID_ADDRESS:
            result = EINVAL;
            break;
        case KERN_NO_SPACE:
            result = ENOMEM;
            break;
        default:
            result = EFAULT;
            break;
    }
#elif WSP_GGML_USE_METAL
    const long page_size = sysconf(_SC_PAGESIZE);
    int result = posix_memalign(&aligned_memory, MAX(TENSOR_ALIGNMENT, page_size), size);
#else
    int result = posix_memalign(&aligned_memory, TENSOR_ALIGNMENT, size);
#endif
    if (result != 0) {
        // Handle allocation failure
        const char *error_desc = "unknown allocation error";
        switch (result) {
            case EINVAL:
                error_desc = "invalid alignment value";
                break;
            case ENOMEM:
                error_desc = "insufficient memory";
                break;
        }
        WSP_GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size/(1024.0*1024.0));
        WSP_GGML_ABORT("fatal error");
        return NULL;
    }
    return aligned_memory;
#endif
}

void wsp_ggml_aligned_free(void * ptr, size_t size) {
    WSP_GGML_UNUSED(size);
#if defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#elif WSP_GGML_USE_CPU_HBM
    if (ptr != NULL) {
        hbw_free(ptr);
    }
#elif TARGET_OS_OSX
    if (ptr != NULL) {
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)ptr, size);
    }
#else
    free(ptr);
#endif
}


inline static void * wsp_ggml_malloc(size_t size) {
    if (size == 0) {
        WSP_GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for wsp_ggml_malloc!\n");
        return NULL;
    }
    void * result = malloc(size);
    if (result == NULL) {
        WSP_GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        WSP_GGML_ABORT("fatal error");
    }
    return result;
}

// calloc
inline static void * wsp_ggml_calloc(size_t num, size_t size) {
    if (num == 0 || size == 0) {
        WSP_GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for wsp_ggml_calloc!\n");
        return NULL;
    }
    void * result = calloc(num, size);
    if (result == NULL) {
        WSP_GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        WSP_GGML_ABORT("fatal error");
    }
    return result;
}

#define WSP_GGML_MALLOC(size)      wsp_ggml_malloc(size)
#define WSP_GGML_CALLOC(num, size) wsp_ggml_calloc(num, size)

#define WSP_GGML_FREE(ptr) free(ptr)

#define UNUSED WSP_GGML_UNUSED
#define SWAP(x, y, T) do { T SWAP = x; (x) = y; (y) = SWAP; } while (0)

#if defined(WSP_GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

// floating point type used to accumulate sums
typedef double wsp_ggml_float;

#undef MIN
#undef MAX

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

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

const char * wsp_ggml_status_to_string(enum wsp_ggml_status status) {
    switch (status) {
        case WSP_GGML_STATUS_ALLOC_FAILED: return "GGML status: error (failed to allocate memory)";
        case WSP_GGML_STATUS_FAILED:       return "GGML status: error (operation failed)";
        case WSP_GGML_STATUS_SUCCESS:      return "GGML status: success";
        case WSP_GGML_STATUS_ABORTED:      return "GGML status: warning (operation aborted)";
    }

    return "GGML status: unknown";
}

float wsp_ggml_fp16_to_fp32(wsp_ggml_fp16_t x) {
#define wsp_ggml_fp16_to_fp32 do_not_use__wsp_ggml_fp16_to_fp32__in_ggml
    return WSP_GGML_FP16_TO_FP32(x);
}

wsp_ggml_fp16_t wsp_ggml_fp32_to_fp16(float x) {
#define wsp_ggml_fp32_to_fp16 do_not_use__wsp_ggml_fp32_to_fp16__in_ggml
    return WSP_GGML_FP32_TO_FP16(x);
}

float wsp_ggml_bf16_to_fp32(wsp_ggml_bf16_t x) {
#define wsp_ggml_bf16_to_fp32 do_not_use__wsp_ggml_bf16_to_fp32__in_ggml
    return WSP_GGML_BF16_TO_FP32(x);  // it just left shifts
}

wsp_ggml_bf16_t wsp_ggml_fp32_to_bf16(float x) {
#define wsp_ggml_fp32_to_bf16 do_not_use__wsp_ggml_fp32_to_bf16__in_ggml
    return WSP_GGML_FP32_TO_BF16(x);
}

void wsp_ggml_fp16_to_fp32_row(const wsp_ggml_fp16_t * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = WSP_GGML_FP16_TO_FP32(x[i]);
    }
}

void wsp_ggml_fp32_to_fp16_row(const float * x, wsp_ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    for(; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#endif
    for (; i < n; i++) {
        y[i] = WSP_GGML_FP32_TO_FP16(x[i]);
    }
}

void wsp_ggml_bf16_to_fp32_row(const wsp_ggml_bf16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__AVX512F__)
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(y + i,
                         _mm512_castsi512_ps(
                             _mm512_slli_epi32(
                                 _mm512_cvtepu16_epi32(
                                     _mm256_loadu_si256(
                                         (const __m256i *)(x + i))),
                                 16)));
    }
#elif defined(__AVX2__)
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(y + i,
                         _mm256_castsi256_ps(
                             _mm256_slli_epi32(
                                 _mm256_cvtepu16_epi32(
                                     _mm_loadu_si128(
                                         (const __m128i *)(x + i))),
                                 16)));
    }
#endif
    for (; i < n; i++) {
        y[i] = WSP_GGML_BF16_TO_FP32(x[i]);
    }
}

void wsp_ggml_fp32_to_bf16_row_ref(const float * x, wsp_ggml_bf16_t * y, int64_t n) {
    for (int i = 0; i < n; i++) {
        y[i] = wsp_ggml_compute_fp32_to_bf16(x[i]);
    }
}

void wsp_ggml_fp32_to_bf16_row(const float * x, wsp_ggml_bf16_t * y, int64_t n) {
  int i = 0;
#if defined(__AVX512BF16__)
  // subnormals are flushed to zero on this platform
  for (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512(
            (__m512i *)(y + i),
            m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
  }
#endif
    for (; i < n; i++) {
        y[i] = WSP_GGML_FP32_TO_BF16(x[i]);
    }
}

bool wsp_ggml_guid_matches(wsp_ggml_guid_t guid_a, wsp_ggml_guid_t guid_b) {
    return memcmp(guid_a, guid_b, sizeof(wsp_ggml_guid)) == 0;
}

//
// timing
//

#if defined(_MSC_VER) || defined(__MINGW32__)
static int64_t timer_freq, timer_start;
void wsp_ggml_time_init(void) {
    LARGE_INTEGER t;
    QueryPerformanceFrequency(&t);
    timer_freq = t.QuadPart;

    // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
    // and the uptime is high enough.
    // We subtract the program start time to reduce the likelihood of that happening.
    QueryPerformanceCounter(&t);
    timer_start = t.QuadPart;
}
int64_t wsp_ggml_time_ms(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart-timer_start) * 1000) / timer_freq;
}
int64_t wsp_ggml_time_us(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart-timer_start) * 1000000) / timer_freq;
}
#else
void wsp_ggml_time_init(void) {}
int64_t wsp_ggml_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

int64_t wsp_ggml_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#endif

int64_t wsp_ggml_cycles(void) {
    return clock();
}

int64_t wsp_ggml_cycles_per_ms(void) {
    return CLOCKS_PER_SEC/1000;
}

//
// cross-platform UTF-8 file paths
//

#ifdef _WIN32
static wchar_t * wsp_ggml_mbstowcs(const char * mbs) {
    int wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, NULL, 0);
    if (!wlen) {
        errno = EINVAL;
        return NULL;
    }

    wchar_t * wbuf = WSP_GGML_MALLOC(wlen * sizeof(wchar_t));
    wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, wbuf, wlen);
    if (!wlen) {
        WSP_GGML_FREE(wbuf);
        errno = EINVAL;
        return NULL;
    }

    return wbuf;
}
#endif

FILE * wsp_ggml_fopen(const char * fname, const char * mode) {
#ifdef _WIN32
    FILE * file = NULL;

    // convert fname (UTF-8)
    wchar_t * wfname = wsp_ggml_mbstowcs(fname);
    if (wfname) {
        // convert mode (ANSI)
        wchar_t * wmode = WSP_GGML_MALLOC((strlen(mode) + 1) * sizeof(wchar_t));
        wchar_t * wmode_p = wmode;
        do {
            *wmode_p++ = (wchar_t)*mode;
        } while (*mode++);

        // open file
        file = _wfopen(wfname, wmode);

        WSP_GGML_FREE(wfname);
        WSP_GGML_FREE(wmode);
    }

    return file;
#else
    return fopen(fname, mode);
#endif
}

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

static const struct wsp_ggml_type_traits type_traits[WSP_GGML_TYPE_COUNT] = {
    [WSP_GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_size                = 1,
        .type_size                = sizeof(int8_t),
        .is_quantized             = false,
    },
    [WSP_GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_size                = 1,
        .type_size                = sizeof(int16_t),
        .is_quantized             = false,
    },
    [WSP_GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_size                = 1,
        .type_size                = sizeof(int32_t),
        .is_quantized             = false,
    },
    [WSP_GGML_TYPE_I64] = {
        .type_name                = "i64",
        .blck_size                = 1,
        .type_size                = sizeof(int64_t),
        .is_quantized             = false,
    },
    [WSP_GGML_TYPE_F64] = {
        .type_name                = "f64",
        .blck_size                = 1,
        .type_size                = sizeof(double),
        .is_quantized             = false,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_size                = 1,
        .type_size                = sizeof(float),
        .is_quantized             = false,
        .vec_dot                  = (wsp_ggml_vec_dot_t) wsp_ggml_vec_dot_f32,
        .vec_dot_type             = WSP_GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_size                = 1,
        .type_size                = sizeof(wsp_ggml_fp16_t),
        .is_quantized             = false,
        .to_float                 = (wsp_ggml_to_float_t) wsp_ggml_fp16_to_fp32_row,
        .from_float               = (wsp_ggml_from_float_t) wsp_ggml_fp32_to_fp16_row,
        .from_float_ref           = (wsp_ggml_from_float_t) wsp_ggml_fp32_to_fp16_row,
        .vec_dot                  = (wsp_ggml_vec_dot_t) wsp_ggml_vec_dot_f16,
        .vec_dot_type             = WSP_GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_size                = QK4_0,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = (wsp_ggml_to_float_t) wsp_dewsp_quantize_row_q4_0,
        .from_float               = wsp_quantize_row_q4_0,
        .from_float_ref           = (wsp_ggml_from_float_t) wsp_quantize_row_q4_0_ref,
        .vec_dot                  = wsp_ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [WSP_GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_size                = QK4_1,
        .type_size                = sizeof(block_q4_1),
        .is_quantized             = true,
        .to_float                 = (wsp_ggml_to_float_t) wsp_dewsp_quantize_row_q4_1,
        .from_float               = wsp_quantize_row_q4_1,
        .from_float_ref           = (wsp_ggml_from_float_t) wsp_quantize_row_q4_1_ref,
        .vec_dot                  = wsp_ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_1,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [4] = { // WSP_GGML_TYPE_Q4_2
        .type_name                = "DEPRECATED",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
        .to_float                 = NULL,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot                  = NULL,
        .vec_dot_type             = WSP_GGML_TYPE_COUNT,
        .nrows                    = 1,
    },
    [5] = { // WSP_GGML_TYPE_Q4_3
        .type_name                = "DEPRECATED",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
        .to_float                 = NULL,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot                  = NULL,
        .vec_dot_type             = WSP_GGML_TYPE_COUNT,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_Q5_0] = {
        .type_name                = "q5_0",
        .blck_size                = QK5_0,
        .type_size                = sizeof(block_q5_0),
        .is_quantized             = true,
        .to_float                 = (wsp_ggml_to_float_t) wsp_dewsp_quantize_row_q5_0,
        .from_float               = wsp_quantize_row_q5_0,
        .from_float_ref           = (wsp_ggml_from_float_t) wsp_quantize_row_q5_0_ref,
        .vec_dot                  = wsp_ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_size                = QK5_1,
        .type_size                = sizeof(block_q5_1),
        .is_quantized             = true,
        .to_float                 = (wsp_ggml_to_float_t) wsp_dewsp_quantize_row_q5_1,
        .from_float               = wsp_quantize_row_q5_1,
        .from_float_ref           = (wsp_ggml_from_float_t) wsp_quantize_row_q5_1_ref,
        .vec_dot                  = wsp_ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_size                = QK8_0,
        .type_size                = sizeof(block_q8_0),
        .is_quantized             = true,
        .to_float                 = (wsp_ggml_to_float_t) wsp_dewsp_quantize_row_q8_0,
        .from_float               = wsp_quantize_row_q8_0,
        .from_float_ref           = (wsp_ggml_from_float_t) wsp_quantize_row_q8_0_ref,
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
        .type_name                = "q8_1",
        .blck_size                = QK8_1,
        .type_size                = sizeof(block_q8_1),
        .is_quantized             = true,
        .from_float               = wsp_quantize_row_q8_1,
        .from_float_ref           = (wsp_ggml_from_float_t) wsp_quantize_row_q8_1_ref,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q2_K),
        .is_quantized             = true,
        .to_float                 = (wsp_ggml_to_float_t) wsp_dewsp_quantize_row_q2_K,
        .from_float               = wsp_quantize_row_q2_K,
        .from_float_ref           = (wsp_ggml_from_float_t) wsp_quantize_row_q2_K_ref,
        .vec_dot                  = wsp_ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q3_K),
        .is_quantized             = true,
        .to_float                 = (wsp_ggml_to_float_t) wsp_dewsp_quantize_row_q3_K,
        .from_float               = wsp_quantize_row_q3_K,
        .from_float_ref           = (wsp_ggml_from_float_t) wsp_quantize_row_q3_K_ref,
        .vec_dot                  = wsp_ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q4_K),
        .is_quantized             = true,
        .to_float                 = (wsp_ggml_to_float_t) wsp_dewsp_quantize_row_q4_K,
        .from_float               = wsp_quantize_row_q4_K,
        .from_float_ref           = (wsp_ggml_from_float_t) wsp_quantize_row_q4_K_ref,
        .vec_dot                  = wsp_ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q5_K),
        .is_quantized             = true,
        .to_float                 = (wsp_ggml_to_float_t) wsp_dewsp_quantize_row_q5_K,
        .from_float               = wsp_quantize_row_q5_K,
        .from_float_ref           = (wsp_ggml_from_float_t) wsp_quantize_row_q5_K_ref,
        .vec_dot                  = wsp_ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q6_K),
        .is_quantized             = true,
        .to_float                 = (wsp_ggml_to_float_t) wsp_dewsp_quantize_row_q6_K,
        .from_float               = wsp_quantize_row_q6_K,
        .from_float_ref           = (wsp_ggml_from_float_t) wsp_quantize_row_q6_K_ref,
        .vec_dot                  = wsp_ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq2_xxs),
        .is_quantized             = true,
        .to_float                 = (wsp_ggml_to_float_t) wsp_dewsp_quantize_row_iq2_xxs,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot                  = wsp_ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq2_xs),
        .is_quantized             = true,
        .to_float                 = (wsp_ggml_to_float_t) wsp_dewsp_quantize_row_iq2_xs,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot                  = wsp_ggml_vec_dot_iq2_xs_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq3_xxs),
        .is_quantized             = true,
        .to_float                 = (wsp_ggml_to_float_t) wsp_dewsp_quantize_row_iq3_xxs,
        .from_float               = wsp_quantize_row_iq3_xxs,
        .from_float_ref           = (wsp_ggml_from_float_t)wsp_quantize_row_iq3_xxs_ref,
        .vec_dot                  = wsp_ggml_vec_dot_iq3_xxs_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq3_s),
        .is_quantized             = true,
        .to_float                 = (wsp_ggml_to_float_t) wsp_dewsp_quantize_row_iq3_s,
        .from_float               = wsp_quantize_row_iq3_s,
        .from_float_ref           = (wsp_ggml_from_float_t)wsp_quantize_row_iq3_s_ref,
        .vec_dot                  = wsp_ggml_vec_dot_iq3_s_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq2_s),
        .is_quantized             = true,
        .to_float                 = (wsp_ggml_to_float_t) wsp_dewsp_quantize_row_iq2_s,
        .from_float               = wsp_quantize_row_iq2_s,
        .from_float_ref           = (wsp_ggml_from_float_t)wsp_quantize_row_iq2_s_ref,
        .vec_dot                  = wsp_ggml_vec_dot_iq2_s_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq1_s),
        .is_quantized             = true,
        .to_float                 = (wsp_ggml_to_float_t) wsp_dewsp_quantize_row_iq1_s,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot                  = wsp_ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq1_m),
        .is_quantized             = true,
        .to_float                 = (wsp_ggml_to_float_t) wsp_dewsp_quantize_row_iq1_m,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot                  = wsp_ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_size                = QK4_NL,
        .type_size                = sizeof(block_iq4_nl),
        .is_quantized             = true,
        .to_float                 = (wsp_ggml_to_float_t) wsp_dewsp_quantize_row_iq4_nl,
        .from_float               = wsp_quantize_row_iq4_nl,
        .from_float_ref           = (wsp_ggml_from_float_t)wsp_quantize_row_iq4_nl_ref,
        .vec_dot                  = wsp_ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq4_xs),
        .is_quantized             = true,
        .to_float                 = (wsp_ggml_to_float_t) wsp_dewsp_quantize_row_iq4_xs,
        .from_float               = wsp_quantize_row_iq4_xs,
        .from_float_ref           = (wsp_ggml_from_float_t)wsp_quantize_row_iq4_xs_ref,
        .vec_dot                  = wsp_ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q8_K),
        .is_quantized             = true,
        .from_float               = wsp_quantize_row_q8_K,
    },
    [WSP_GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_size                = 1,
        .type_size                = sizeof(wsp_ggml_bf16_t),
        .is_quantized             = false,
        .to_float                 = (wsp_ggml_to_float_t) wsp_ggml_bf16_to_fp32_row,
        .from_float               = (wsp_ggml_from_float_t) wsp_ggml_fp32_to_bf16_row,
        .from_float_ref           = (wsp_ggml_from_float_t) wsp_ggml_fp32_to_bf16_row_ref,
        .vec_dot                  = (wsp_ggml_vec_dot_t) wsp_ggml_vec_dot_bf16,
        .vec_dot_type             = WSP_GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_Q4_0_4_4] = {
        .type_name                = "q4_0_4x4",
        .blck_size                = QK4_0,
        .blck_size_interleave     = 4,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = NULL,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot                  = NULL,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_0,
        .nrows                    = 1,
        .ncols                    = 4,
        .gemv                     = wsp_ggml_gemv_q4_0_4x4_q8_0,
        .gemm                     = wsp_ggml_gemm_q4_0_4x4_q8_0,
    },
    [WSP_GGML_TYPE_Q4_0_4_8] = {
        .type_name                = "q4_0_4x8",
        .blck_size                = QK4_0,
        .blck_size_interleave     = 8,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = NULL,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot                  = NULL,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_0,
        .nrows                    = 1,
        .ncols                    = 4,
        .gemv                     = wsp_ggml_gemv_q4_0_4x8_q8_0,
        .gemm                     = wsp_ggml_gemm_q4_0_4x8_q8_0,
    },
    [WSP_GGML_TYPE_Q4_0_8_8] = {
        .type_name                = "q4_0_8x8",
        .blck_size                = QK4_0,
        .blck_size_interleave     = 8,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = NULL,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot                  = NULL,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_0,
        .nrows                    = 1,
        .ncols                    = 8,
        .gemv                     = wsp_ggml_gemv_q4_0_8x8_q8_0,
        .gemm                     = wsp_ggml_gemm_q4_0_8x8_q8_0,
    },
    [WSP_GGML_TYPE_TQ1_0] = {
        .type_name                = "tq1_0",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_tq1_0),
        .is_quantized             = true,
        .to_float                 = (wsp_ggml_to_float_t) wsp_dewsp_quantize_row_tq1_0,
        .from_float               = wsp_quantize_row_tq1_0,
        .from_float_ref           = (wsp_ggml_from_float_t) wsp_quantize_row_tq1_0_ref,
        .vec_dot                  = wsp_ggml_vec_dot_tq1_0_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [WSP_GGML_TYPE_TQ2_0] = {
        .type_name                = "tq2_0",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_tq2_0),
        .is_quantized             = true,
        .to_float                 = (wsp_ggml_to_float_t) wsp_dewsp_quantize_row_tq2_0,
        .from_float               = wsp_quantize_row_tq2_0,
        .from_float_ref           = (wsp_ggml_from_float_t) wsp_quantize_row_tq2_0_ref,
        .vec_dot                  = wsp_ggml_vec_dot_tq2_0_q8_K,
        .vec_dot_type             = WSP_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
};

// For internal test use
const struct wsp_ggml_type_traits * wsp_ggml_get_type_traits(enum wsp_ggml_type type) {
    WSP_GGML_ASSERT(type < WSP_GGML_TYPE_COUNT);
    return &type_traits[type];
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
// ggml object
//

struct wsp_ggml_object {
    size_t offs;
    size_t size;

    struct wsp_ggml_object * next;

    enum wsp_ggml_object_type type;

    char padding[4];
};

static const size_t WSP_GGML_OBJECT_SIZE = sizeof(struct wsp_ggml_object);

//
// ggml context
//

struct wsp_ggml_context {
    size_t mem_size;
    void * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;
    bool   no_alloc_save; // this is used to save the no_alloc state when using scratch buffers

    int    n_objects;

    struct wsp_ggml_object * objects_begin;
    struct wsp_ggml_object * objects_end;

    struct wsp_ggml_scratch scratch;
    struct wsp_ggml_scratch scratch_save;
};

struct wsp_ggml_context_container {
    bool used;

    struct wsp_ggml_context context;
};

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

//
// data types
//

static const char * WSP_GGML_OP_NAME[WSP_GGML_OP_COUNT] = {
    "NONE",

    "DUP",
    "ADD",
    "ADD1",
    "ACC",
    "SUB",
    "MUL",
    "DIV",
    "SQR",
    "SQRT",
    "LOG",
    "SIN",
    "COS",
    "SUM",
    "SUM_ROWS",
    "MEAN",
    "ARGMAX",
    "COUNT_EQUAL",
    "REPEAT",
    "REPEAT_BACK",
    "CONCAT",
    "SILU_BACK",
    "NORM",
    "RMS_NORM",
    "RMS_NORM_BACK",
    "GROUP_NORM",

    "MUL_MAT",
    "MUL_MAT_ID",
    "OUT_PROD",

    "SCALE",
    "SET",
    "CPY",
    "CONT",
    "RESHAPE",
    "VIEW",
    "PERMUTE",
    "TRANSPOSE",
    "GET_ROWS",
    "GET_ROWS_BACK",
    "DIAG",
    "DIAG_MASK_INF",
    "DIAG_MASK_ZERO",
    "SOFT_MAX",
    "SOFT_MAX_BACK",
    "ROPE",
    "ROPE_BACK",
    "CLAMP",
    "CONV_TRANSPOSE_1D",
    "IM2COL",
    "IM2COL_BACK",
    "CONV_TRANSPOSE_2D",
    "POOL_1D",
    "POOL_2D",
    "POOL_2D_BACK",
    "UPSCALE",
    "PAD",
    "ARANGE",
    "TIMESTEP_EMBEDDING",
    "ARGSORT",
    "LEAKY_RELU",

    "FLASH_ATTN_EXT",
    "FLASH_ATTN_BACK",
    "SSM_CONV",
    "SSM_SCAN",
    "WIN_PART",
    "WIN_UNPART",
    "GET_REL_POS",
    "ADD_REL_POS",
    "RWKV_WKV",

    "UNARY",

    "MAP_UNARY",
    "MAP_BINARY",

    "MAP_CUSTOM1_F32",
    "MAP_CUSTOM2_F32",
    "MAP_CUSTOM3_F32",

    "MAP_CUSTOM1",
    "MAP_CUSTOM2",
    "MAP_CUSTOM3",

    "CROSS_ENTROPY_LOSS",
    "CROSS_ENTROPY_LOSS_BACK",
    "OPT_STEP_ADAMW",
};

static_assert(WSP_GGML_OP_COUNT == 81, "WSP_GGML_OP_COUNT != 81");

static const char * WSP_GGML_OP_SYMBOL[WSP_GGML_OP_COUNT] = {
    "none",

    "x",
    "x+y",
    "x+y",
    "view(x,nb,offset)+=y->x",
    "x-y",
    "x*y",
    "x/y",
    "x^2",
    "√x",
    "log(x)",
    "sin(x)",
    "cos(x)",
    "Σx",
    "Σx_k",
    "Σx/n",
    "argmax(x)",
    "count_equal(x)",
    "repeat(x)",
    "repeat_back(x)",
    "concat(x, y)",
    "silu_back(x)",
    "norm(x)",
    "rms_norm(x)",
    "rms_norm_back(x)",
    "group_norm(x)",

    "X*Y",
    "X[i]*Y",
    "X*Y",

    "x*v",
    "y-\\>view(x)",
    "x-\\>y",
    "cont(x)",
    "reshape(x)",
    "view(x)",
    "permute(x)",
    "transpose(x)",
    "get_rows(x)",
    "get_rows_back(x)",
    "diag(x)",
    "diag_mask_inf(x)",
    "diag_mask_zero(x)",
    "soft_max(x)",
    "soft_max_back(x)",
    "rope(x)",
    "rope_back(x)",
    "clamp(x)",
    "conv_transpose_1d(x)",
    "im2col(x)",
    "im2col_back(x)",
    "conv_transpose_2d(x)",
    "pool_1d(x)",
    "pool_2d(x)",
    "pool_2d_back(x)",
    "upscale(x)",
    "pad(x)",
    "arange(start, stop, step)",
    "timestep_embedding(timesteps, dim, max_period)",
    "argsort(x)",
    "leaky_relu(x)",

    "flash_attn_ext(x)",
    "flash_attn_back(x)",
    "ssm_conv(x)",
    "ssm_scan(x)",
    "win_part(x)",
    "win_unpart(x)",
    "get_rel_pos(x)",
    "add_rel_pos(x)",
    "rwkv_wkv(k, v, r, tf, td, s)",

    "unary(x)",

    "f(x)",
    "f(x,y)",

    "custom_f32(x)",
    "custom_f32(x,y)",
    "custom_f32(x,y,z)",

    "custom(x)",
    "custom(x,y)",
    "custom(x,y,z)",

    "cross_entropy_loss(x,y)",
    "cross_entropy_loss_back(x,y)",
    "adamw(x)",
};

static_assert(WSP_GGML_OP_COUNT == 81, "WSP_GGML_OP_COUNT != 81");

static_assert(WSP_GGML_OP_POOL_COUNT == 2, "WSP_GGML_OP_POOL_COUNT != 2");


static const char * WSP_GGML_UNARY_OP_NAME[WSP_GGML_UNARY_OP_COUNT] = {
    "ABS",
    "SGN",
    "NEG",
    "STEP",
    "TANH",
    "ELU",
    "RELU",
    "SIGMOID",
    "GELU",
    "GELU_QUICK",
    "SILU",
    "HARDSWISH",
    "HARDSIGMOID",
    "EXP",
};

static_assert(WSP_GGML_UNARY_OP_COUNT == 14, "WSP_GGML_UNARY_OP_COUNT != 14");


static_assert(sizeof(struct wsp_ggml_object)%WSP_GGML_MEM_ALIGN == 0, "wsp_ggml_object size must be a multiple of WSP_GGML_MEM_ALIGN");
static_assert(sizeof(struct wsp_ggml_tensor)%WSP_GGML_MEM_ALIGN == 0, "wsp_ggml_tensor size must be a multiple of WSP_GGML_MEM_ALIGN");

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
static struct wsp_ggml_state g_state;
static atomic_flag g_state_critical = ATOMIC_FLAG_INIT;

// critical section via spin lock
inline static void wsp_ggml_critical_section_start(void) {
    while (atomic_flag_test_and_set(&g_state_critical)) {
        // spin
        sched_yield();
    }
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

// TODO: make this somehow automatically executed
//       some sort of "sentry" mechanism
inline static void wsp_ggml_critical_section_end(void) {
    atomic_flag_clear(&g_state_critical);
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

////////////////////////////////////////////////////////////////////////////////

void wsp_ggml_print_object(const struct wsp_ggml_object * obj) {
    WSP_GGML_LOG_INFO(" - wsp_ggml_object: type = %d, offset = %zu, size = %zu, next = %p\n",
            obj->type, obj->offs, obj->size, (const void *) obj->next);
}

void wsp_ggml_print_objects(const struct wsp_ggml_context * ctx) {
    struct wsp_ggml_object * obj = ctx->objects_begin;

    WSP_GGML_LOG_INFO("%s: objects in context %p:\n", __func__, (const void *) ctx);

    while (obj != NULL) {
        wsp_ggml_print_object(obj);
        obj = obj->next;
    }

    WSP_GGML_LOG_INFO("%s: --- end ---\n", __func__);
}

int64_t wsp_ggml_nelements(const struct wsp_ggml_tensor * tensor) {
    static_assert(WSP_GGML_MAX_DIMS == 4, "WSP_GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

int64_t wsp_ggml_nrows(const struct wsp_ggml_tensor * tensor) {
    static_assert(WSP_GGML_MAX_DIMS == 4, "WSP_GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

size_t wsp_ggml_nbytes(const struct wsp_ggml_tensor * tensor) {
    size_t nbytes;
    const size_t blck_size = wsp_ggml_blck_size(tensor->type);
    if (blck_size == 1) {
        nbytes = wsp_ggml_type_size(tensor->type);
        for (int i = 0; i < WSP_GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    else {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_size;
        for (int i = 1; i < WSP_GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    return nbytes;
}

size_t wsp_ggml_nbytes_pad(const struct wsp_ggml_tensor * tensor) {
    return WSP_GGML_PAD(wsp_ggml_nbytes(tensor), WSP_GGML_MEM_ALIGN);
}

int64_t wsp_ggml_blck_size(enum wsp_ggml_type type) {
    return type_traits[type].blck_size;
}

size_t wsp_ggml_type_size(enum wsp_ggml_type type) {
    return type_traits[type].type_size;
}

size_t wsp_ggml_row_size(enum wsp_ggml_type type, int64_t ne) {
    assert(ne % wsp_ggml_blck_size(type) == 0);
    return wsp_ggml_type_size(type)*ne/wsp_ggml_blck_size(type);
}

double wsp_ggml_type_sizef(enum wsp_ggml_type type) {
    return ((double)(type_traits[type].type_size))/type_traits[type].blck_size;
}

const char * wsp_ggml_type_name(enum wsp_ggml_type type) {
    return type < WSP_GGML_TYPE_COUNT ? type_traits[type].type_name : "NONE";
}

bool wsp_ggml_is_quantized(enum wsp_ggml_type type) {
    return type_traits[type].is_quantized;
}

const char * wsp_ggml_op_name(enum wsp_ggml_op op) {
    return WSP_GGML_OP_NAME[op];
}

const char * wsp_ggml_op_symbol(enum wsp_ggml_op op) {
    return WSP_GGML_OP_SYMBOL[op];
}

const char * wsp_ggml_unary_op_name(enum wsp_ggml_unary_op op) {
    return WSP_GGML_UNARY_OP_NAME[op];
}

const char * wsp_ggml_op_desc(const struct wsp_ggml_tensor * t) {
    if (t->op == WSP_GGML_OP_UNARY) {
        enum wsp_ggml_unary_op uop = wsp_ggml_get_unary_op(t);
        return wsp_ggml_unary_op_name(uop);
    }
    return wsp_ggml_op_name(t->op);
}

size_t wsp_ggml_element_size(const struct wsp_ggml_tensor * tensor) {
    return wsp_ggml_type_size(tensor->type);
}

bool wsp_ggml_is_scalar(const struct wsp_ggml_tensor * tensor) {
    static_assert(WSP_GGML_MAX_DIMS == 4, "WSP_GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool wsp_ggml_is_vector(const struct wsp_ggml_tensor * tensor) {
    static_assert(WSP_GGML_MAX_DIMS == 4, "WSP_GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool wsp_ggml_is_matrix(const struct wsp_ggml_tensor * tensor) {
    static_assert(WSP_GGML_MAX_DIMS == 4, "WSP_GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool wsp_ggml_is_3d(const struct wsp_ggml_tensor * tensor) {
    return tensor->ne[3] == 1;
}

int wsp_ggml_n_dims(const struct wsp_ggml_tensor * tensor) {
    for (int i = WSP_GGML_MAX_DIMS - 1; i >= 1; --i) {
        if (tensor->ne[i] > 1) {
            return i + 1;
        }
    }
    return 1;
}

static inline bool wsp_ggml_can_mul_mat(const struct wsp_ggml_tensor * t0, const struct wsp_ggml_tensor * t1) {
    static_assert(WSP_GGML_MAX_DIMS == 4, "WSP_GGML_MAX_DIMS is not 4 - update this function");

    return (t0->ne[0]           == t1->ne[0])  &&
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

static inline bool wsp_ggml_can_out_prod(const struct wsp_ggml_tensor * t0, const struct wsp_ggml_tensor * t1) {
    static_assert(WSP_GGML_MAX_DIMS == 4, "WSP_GGML_MAX_DIMS is not 4 - update this function");

    return (t0->ne[1] == t1->ne[1])   &&
           (t1->ne[2]%t0->ne[2] == 0) && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

enum wsp_ggml_type wsp_ggml_ftype_to_wsp_ggml_type(enum wsp_ggml_ftype ftype) {
    enum wsp_ggml_type wtype = WSP_GGML_TYPE_COUNT;

    switch (ftype) {
        case WSP_GGML_FTYPE_ALL_F32:              wtype = WSP_GGML_TYPE_F32;   break;
        case WSP_GGML_FTYPE_MOSTLY_F16:           wtype = WSP_GGML_TYPE_F16;   break;
        case WSP_GGML_FTYPE_MOSTLY_BF16:          wtype = WSP_GGML_TYPE_BF16;  break;
        case WSP_GGML_FTYPE_MOSTLY_Q4_0:          wtype = WSP_GGML_TYPE_Q4_0;  break;
        case WSP_GGML_FTYPE_MOSTLY_Q4_1:          wtype = WSP_GGML_TYPE_Q4_1;  break;
        case WSP_GGML_FTYPE_MOSTLY_Q5_0:          wtype = WSP_GGML_TYPE_Q5_0;  break;
        case WSP_GGML_FTYPE_MOSTLY_Q5_1:          wtype = WSP_GGML_TYPE_Q5_1;  break;
        case WSP_GGML_FTYPE_MOSTLY_Q8_0:          wtype = WSP_GGML_TYPE_Q8_0;  break;
        case WSP_GGML_FTYPE_MOSTLY_Q2_K:          wtype = WSP_GGML_TYPE_Q2_K;  break;
        case WSP_GGML_FTYPE_MOSTLY_Q3_K:          wtype = WSP_GGML_TYPE_Q3_K;  break;
        case WSP_GGML_FTYPE_MOSTLY_Q4_K:          wtype = WSP_GGML_TYPE_Q4_K;  break;
        case WSP_GGML_FTYPE_MOSTLY_Q5_K:          wtype = WSP_GGML_TYPE_Q5_K;  break;
        case WSP_GGML_FTYPE_MOSTLY_Q6_K:          wtype = WSP_GGML_TYPE_Q6_K;  break;
        case WSP_GGML_FTYPE_MOSTLY_IQ2_XXS:       wtype = WSP_GGML_TYPE_IQ2_XXS;  break;
        case WSP_GGML_FTYPE_MOSTLY_IQ2_XS:        wtype = WSP_GGML_TYPE_IQ2_XS;   break;
        case WSP_GGML_FTYPE_MOSTLY_IQ3_XXS:       wtype = WSP_GGML_TYPE_IQ3_XXS;  break;
        case WSP_GGML_FTYPE_MOSTLY_IQ1_S:         wtype = WSP_GGML_TYPE_IQ1_S;    break;
        case WSP_GGML_FTYPE_MOSTLY_IQ1_M:         wtype = WSP_GGML_TYPE_IQ1_M;    break;
        case WSP_GGML_FTYPE_MOSTLY_IQ4_NL:        wtype = WSP_GGML_TYPE_IQ4_NL;   break;
        case WSP_GGML_FTYPE_MOSTLY_IQ4_XS:        wtype = WSP_GGML_TYPE_IQ4_XS;   break;
        case WSP_GGML_FTYPE_MOSTLY_IQ3_S:         wtype = WSP_GGML_TYPE_IQ3_S;    break;
        case WSP_GGML_FTYPE_MOSTLY_IQ2_S:         wtype = WSP_GGML_TYPE_IQ2_S;    break;
        case WSP_GGML_FTYPE_MOSTLY_Q4_0_4_4:      wtype = WSP_GGML_TYPE_Q4_0_4_4; break;
        case WSP_GGML_FTYPE_MOSTLY_Q4_0_4_8:      wtype = WSP_GGML_TYPE_Q4_0_4_8; break;
        case WSP_GGML_FTYPE_MOSTLY_Q4_0_8_8:      wtype = WSP_GGML_TYPE_Q4_0_8_8; break;
        case WSP_GGML_FTYPE_UNKNOWN:              wtype = WSP_GGML_TYPE_COUNT; break;
        case WSP_GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: wtype = WSP_GGML_TYPE_COUNT; break;
    }

    WSP_GGML_ASSERT(wtype != WSP_GGML_TYPE_COUNT);

    return wtype;
}

size_t wsp_ggml_tensor_overhead(void) {
    return WSP_GGML_OBJECT_SIZE + WSP_GGML_TENSOR_SIZE;
}

bool wsp_ggml_is_transposed(const struct wsp_ggml_tensor * tensor) {
    return tensor->nb[0] > tensor->nb[1];
}

static bool wsp_ggml_is_contiguous_n(const struct wsp_ggml_tensor * tensor, int n) {
    size_t next_nb = wsp_ggml_type_size(tensor->type);
    if (tensor->ne[0] != wsp_ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) {
        return false;
    }
    next_nb *= tensor->ne[0]/wsp_ggml_blck_size(tensor->type);
    for (int i = 1; i < WSP_GGML_MAX_DIMS; i++) {
        if (tensor->ne[i] != 1) {
            if (i > n) {
                if (tensor->nb[i] != next_nb) {
                    return false;
                }
                next_nb *= tensor->ne[i];
            } else {
                // this dimension does not need to be contiguous
                next_nb = tensor->ne[i]*tensor->nb[i];
            }
        }
    }
    return true;
}

bool wsp_ggml_is_contiguous(const struct wsp_ggml_tensor * tensor) {
    return wsp_ggml_is_contiguous_0(tensor);
}

bool wsp_ggml_is_contiguous_0(const struct wsp_ggml_tensor * tensor) {
    return wsp_ggml_is_contiguous_n(tensor, 0);
}

bool wsp_ggml_is_contiguous_1(const struct wsp_ggml_tensor * tensor) {
    return wsp_ggml_is_contiguous_n(tensor, 1);
}

bool wsp_ggml_is_contiguous_2(const struct wsp_ggml_tensor * tensor) {
    return wsp_ggml_is_contiguous_n(tensor, 2);
}

bool wsp_ggml_is_permuted(const struct wsp_ggml_tensor * tensor) {
    static_assert(WSP_GGML_MAX_DIMS == 4, "WSP_GGML_MAX_DIMS is not 4 - update this function");

    return tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

static inline bool wsp_ggml_is_padded_1d(const struct wsp_ggml_tensor * tensor) {
    static_assert(WSP_GGML_MAX_DIMS == 4, "WSP_GGML_MAX_DIMS is not 4 - update this function");

    return
        tensor->nb[0] == wsp_ggml_type_size(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

bool wsp_ggml_is_empty(const struct wsp_ggml_tensor * tensor) {
    for (int i = 0; i < WSP_GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] == 0) {
            // empty if any dimension has no elements
            return true;
        }
    }
    return false;
}

bool wsp_ggml_are_same_shape(const struct wsp_ggml_tensor * t0, const struct wsp_ggml_tensor * t1) {
    static_assert(WSP_GGML_MAX_DIMS == 4, "WSP_GGML_MAX_DIMS is not 4 - update this function");

    return
        (t0->ne[0] == t1->ne[0]) &&
        (t0->ne[1] == t1->ne[1]) &&
        (t0->ne[2] == t1->ne[2]) &&
        (t0->ne[3] == t1->ne[3]);
}

bool wsp_ggml_are_same_stride(const struct wsp_ggml_tensor * t0, const struct wsp_ggml_tensor * t1) {
    static_assert(WSP_GGML_MAX_DIMS == 4, "WSP_GGML_MAX_DIMS is not 4 - update this function");

    return
        (t0->nb[0] == t1->nb[0]) &&
        (t0->nb[1] == t1->nb[1]) &&
        (t0->nb[2] == t1->nb[2]) &&
        (t0->nb[3] == t1->nb[3]);
}

// check if t1 can be represented as a repeatition of t0
bool wsp_ggml_can_repeat(const struct wsp_ggml_tensor * t0, const struct wsp_ggml_tensor * t1) {
    static_assert(WSP_GGML_MAX_DIMS == 4, "WSP_GGML_MAX_DIMS is not 4 - update this function");

    return wsp_ggml_is_empty(t0) ? wsp_ggml_is_empty(t1) :
        (t1->ne[0]%t0->ne[0] == 0) &&
        (t1->ne[1]%t0->ne[1] == 0) &&
        (t1->ne[2]%t0->ne[2] == 0) &&
        (t1->ne[3]%t0->ne[3] == 0);
}

static inline bool wsp_ggml_can_repeat_rows(const struct wsp_ggml_tensor * t0, const struct wsp_ggml_tensor * t1) {
    static_assert(WSP_GGML_MAX_DIMS == 4, "WSP_GGML_MAX_DIMS is not 4 - update this function");

    return (t0->ne[0] == t1->ne[0]) && wsp_ggml_can_repeat(t0, t1);
}

static inline int wsp_ggml_up32(int n) {
    return (n + 31) & ~31;
}

//static inline int wsp_ggml_up64(int n) {
//    return (n + 63) & ~63;
//}

static inline int wsp_ggml_up(int n, int m) {
    // assert m is a power of 2
    WSP_GGML_ASSERT((m & (m - 1)) == 0);
    return (n + m - 1) & ~(m - 1);
}

// assert that pointer is aligned to WSP_GGML_MEM_ALIGN
#define WSP_GGML_ASSERT_ALIGNED(ptr) \
    WSP_GGML_ASSERT(((uintptr_t) (ptr))%WSP_GGML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

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

struct wsp_ggml_context * wsp_ggml_init(struct wsp_ggml_init_params params) {
    // make this function thread safe
    wsp_ggml_critical_section_start();

    static bool is_first_call = true;

    if (is_first_call) {
        // initialize time system (required on Windows)
        wsp_ggml_time_init();

        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            const uint64_t t_start = wsp_ggml_time_us(); UNUSED(t_start);

            for (int i = 0; i < (1 << 16); ++i) {
                union {
                    uint16_t u16;
                    wsp_ggml_fp16_t fp16;
                } u = {i};
                float f = wsp_ggml_table_f32_f16[i] = WSP_GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                wsp_ggml_table_gelu_f16[i] = WSP_GGML_FP32_TO_FP16(wsp_ggml_gelu_f32(f));
                wsp_ggml_table_gelu_quick_f16[i] = WSP_GGML_FP32_TO_FP16(wsp_ggml_gelu_quick_f32(f));
            }

            const uint64_t t_end = wsp_ggml_time_us(); UNUSED(t_end);

            WSP_GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0f);
        }

        // initialize g_state
        {
            const uint64_t t_start = wsp_ggml_time_us(); UNUSED(t_start);

            g_state = (struct wsp_ggml_state) {
                /*.numa =*/ {
                    .n_nodes = 0,
                    .total_cpus = 0,
                },
            };

            const uint64_t t_end = wsp_ggml_time_us(); UNUSED(t_end);

            WSP_GGML_PRINT_DEBUG("%s: g_state initialized in %f ms\n", __func__, (t_end - t_start)/1000.0f);
        }

#if defined(__ARM_ARCH)
        wsp_ggml_init_arm_arch_features();
#endif

        is_first_call = false;
    }

    wsp_ggml_critical_section_end();

    struct wsp_ggml_context * ctx = WSP_GGML_MALLOC(sizeof(struct wsp_ggml_context));

    // allow to call wsp_ggml_init with 0 size
    if (params.mem_size == 0) {
        params.mem_size = WSP_GGML_MEM_ALIGN;
    }

    const size_t mem_size = params.mem_buffer ? params.mem_size : WSP_GGML_PAD(params.mem_size, WSP_GGML_MEM_ALIGN);

    *ctx = (struct wsp_ggml_context) {
        /*.mem_size           =*/ mem_size,
        /*.mem_buffer         =*/ params.mem_buffer ? params.mem_buffer : wsp_ggml_aligned_malloc(mem_size),
        /*.mem_buffer_owned   =*/ params.mem_buffer ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.no_alloc_save      =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ NULL,
        /*.objects_end        =*/ NULL,
        /*.scratch            =*/ { 0, 0, NULL, },
        /*.scratch_save       =*/ { 0, 0, NULL, },
    };

    WSP_GGML_ASSERT(ctx->mem_buffer != NULL);

    WSP_GGML_ASSERT_ALIGNED(ctx->mem_buffer);

    WSP_GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    return ctx;
}

void wsp_ggml_reset(struct wsp_ggml_context * ctx) {
    if (ctx == NULL) {
        return;
    }

    ctx->n_objects     = 0;
    ctx->objects_begin = NULL;
    ctx->objects_end   = NULL;
    ctx->scratch       = (struct wsp_ggml_scratch) { 0, 0, NULL, };
    ctx->scratch_save  = (struct wsp_ggml_scratch) { 0, 0, NULL, };
}

void wsp_ggml_free(struct wsp_ggml_context * ctx) {
    if (ctx == NULL) {
        return;
    }

    if (ctx->mem_buffer_owned) {
        wsp_ggml_aligned_free(ctx->mem_buffer, ctx->mem_size);
    }

    WSP_GGML_FREE(ctx);
}

size_t wsp_ggml_used_mem(const struct wsp_ggml_context * ctx) {
    return ctx->objects_end == NULL ? 0 : ctx->objects_end->offs + ctx->objects_end->size;
}

size_t wsp_ggml_set_scratch(struct wsp_ggml_context * ctx, struct wsp_ggml_scratch scratch) {
    const size_t result = ctx->scratch.data ? ctx->scratch.offs : 0;

    ctx->scratch = scratch;

    return result;
}

bool wsp_ggml_get_no_alloc(struct wsp_ggml_context * ctx) {
    return ctx->no_alloc;
}

void wsp_ggml_set_no_alloc(struct wsp_ggml_context * ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

void * wsp_ggml_get_mem_buffer(const struct wsp_ggml_context * ctx) {
    return ctx->mem_buffer;
}

size_t wsp_ggml_get_mem_size(const struct wsp_ggml_context * ctx) {
    return ctx->mem_size;
}

size_t wsp_ggml_get_max_tensor_size(const struct wsp_ggml_context * ctx) {
    size_t max_size = 0;

    for (struct wsp_ggml_tensor * tensor = wsp_ggml_get_first_tensor(ctx); tensor != NULL; tensor = wsp_ggml_get_next_tensor(ctx, tensor)) {
        size_t bytes = wsp_ggml_nbytes(tensor);
        max_size = MAX(max_size, bytes);
    }

    return max_size;
}

// IMPORTANT:
// when creating "opt" tensors, always save and load the scratch buffer
// this is an error prone process, but it is necessary to support inplace
// operators when using scratch buffers
// TODO: implement a better way
static void wsp_ggml_scratch_save(struct wsp_ggml_context * ctx) {
    // this is needed to allow opt tensors to store their data
    // TODO: again, need to find a better way
    ctx->no_alloc_save = ctx->no_alloc;
    ctx->no_alloc      = false;

    ctx->scratch_save = ctx->scratch;
    ctx->scratch.data = NULL;
}

static void wsp_ggml_scratch_load(struct wsp_ggml_context * ctx) {
    ctx->no_alloc = ctx->no_alloc_save;

    ctx->scratch = ctx->scratch_save;
}

////////////////////////////////////////////////////////////////////////////////

static struct wsp_ggml_object * wsp_ggml_new_object(struct wsp_ggml_context * ctx, enum wsp_ggml_object_type type, size_t size) {
    // always insert objects at the end of the context's memory pool
    struct wsp_ggml_object * obj_cur = ctx->objects_end;

    const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
    const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
    const size_t cur_end  = cur_offs + cur_size;

    // align to WSP_GGML_MEM_ALIGN
    size_t size_needed = WSP_GGML_PAD(size, WSP_GGML_MEM_ALIGN);

    char * const mem_buffer = ctx->mem_buffer;
    struct wsp_ggml_object * const obj_new = (struct wsp_ggml_object *)(mem_buffer + cur_end);

    if (cur_end + size_needed + WSP_GGML_OBJECT_SIZE > ctx->mem_size) {
        WSP_GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_end + size_needed + WSP_GGML_OBJECT_SIZE, ctx->mem_size);
        assert(false);
        return NULL;
    }

    *obj_new = (struct wsp_ggml_object) {
        .offs = cur_end + WSP_GGML_OBJECT_SIZE,
        .size = size_needed,
        .next = NULL,
        .type = type,
    };

    WSP_GGML_ASSERT_ALIGNED(mem_buffer + obj_new->offs);

    if (obj_cur != NULL) {
        obj_cur->next = obj_new;
    } else {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    return obj_new;
}

static struct wsp_ggml_tensor * wsp_ggml_new_tensor_impl(
        struct wsp_ggml_context * ctx,
        enum   wsp_ggml_type      type,
        int                   n_dims,
        const int64_t       * ne,
        struct wsp_ggml_tensor  * view_src,
        size_t                view_offs) {

    WSP_GGML_ASSERT(type >= 0 && type < WSP_GGML_TYPE_COUNT);
    WSP_GGML_ASSERT(n_dims >= 1 && n_dims <= WSP_GGML_MAX_DIMS);

    // find the base tensor and absolute offset
    if (view_src != NULL && view_src->view_src != NULL) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    size_t data_size = wsp_ggml_row_size(type, ne[0]);
    for (int i = 1; i < n_dims; i++) {
        data_size *= ne[i];
    }

    WSP_GGML_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <= wsp_ggml_nbytes(view_src));

    void * data = view_src != NULL ? view_src->data : NULL;
    if (data != NULL) {
        data = (char *) data + view_offs;
    }

    size_t obj_alloc_size = 0;

    if (view_src == NULL && !ctx->no_alloc) {
        if (ctx->scratch.data != NULL) {
            // allocate tensor data in the scratch buffer
            if (ctx->scratch.offs + data_size > ctx->scratch.size) {
                WSP_GGML_LOG_WARN("%s: not enough space in the scratch memory pool (needed %zu, available %zu)\n",
                        __func__, ctx->scratch.offs + data_size, ctx->scratch.size);
                assert(false);
                return NULL;
            }

            data = (char * const) ctx->scratch.data + ctx->scratch.offs;

            ctx->scratch.offs += data_size;
        } else {
            // allocate tensor data in the context's memory pool
            obj_alloc_size = data_size;
        }
    }

    struct wsp_ggml_object * const obj_new = wsp_ggml_new_object(ctx, WSP_GGML_OBJECT_TYPE_TENSOR, WSP_GGML_TENSOR_SIZE + obj_alloc_size);
    WSP_GGML_ASSERT(obj_new);

    // TODO: for recoverable errors, we would need to free the data allocated from the scratch buffer here

    struct wsp_ggml_tensor * const result = (struct wsp_ggml_tensor *)((char *)ctx->mem_buffer + obj_new->offs);

#ifdef __clang__
    // temporary until wsp_ggml_tensor::backend is removed
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

    *result = (struct wsp_ggml_tensor) {
        /*.type         =*/ type,
        /*.backend      =*/ WSP_GGML_BACKEND_TYPE_CPU,
        /*.buffer       =*/ NULL,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ WSP_GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
        /*.grad         =*/ NULL,
        /*.src          =*/ { NULL },
        /*.view_src     =*/ view_src,
        /*.view_offs    =*/ view_offs,
        /*.data         =*/ obj_alloc_size > 0 ? (void *)(result + 1) : data,
        /*.name         =*/ { 0 },
        /*.extra        =*/ NULL,
        ///*.padding      =*/ { 0 },
    };

#ifdef __clang__
    #pragma clang diagnostic pop
#endif

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //WSP_GGML_ASSERT_ALIGNED(result->data);

    for (int i = 0; i < n_dims; i++) {
        result->ne[i] = ne[i];
    }

    result->nb[0] = wsp_ggml_type_size(type);
    result->nb[1] = result->nb[0]*(result->ne[0]/wsp_ggml_blck_size(type));
    for (int i = 2; i < WSP_GGML_MAX_DIMS; i++) {
        result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
    }

    ctx->n_objects++;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_new_tensor(
        struct wsp_ggml_context * ctx,
        enum   wsp_ggml_type      type,
        int                   n_dims,
        const int64_t       * ne) {
    return wsp_ggml_new_tensor_impl(ctx, type, n_dims, ne, NULL, 0);
}

struct wsp_ggml_tensor * wsp_ggml_new_tensor_1d(
        struct wsp_ggml_context * ctx,
        enum   wsp_ggml_type      type,
        int64_t ne0) {
    return wsp_ggml_new_tensor(ctx, type, 1, &ne0);
}

struct wsp_ggml_tensor * wsp_ggml_new_tensor_2d(
        struct wsp_ggml_context * ctx,
        enum   wsp_ggml_type      type,
        int64_t ne0,
        int64_t ne1) {
    const int64_t ne[2] = { ne0, ne1 };
    return wsp_ggml_new_tensor(ctx, type, 2, ne);
}

struct wsp_ggml_tensor * wsp_ggml_new_tensor_3d(
        struct wsp_ggml_context * ctx,
        enum   wsp_ggml_type      type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) {
    const int64_t ne[3] = { ne0, ne1, ne2 };
    return wsp_ggml_new_tensor(ctx, type, 3, ne);
}

struct wsp_ggml_tensor * wsp_ggml_new_tensor_4d(
        struct wsp_ggml_context * ctx,
        enum   wsp_ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    return wsp_ggml_new_tensor(ctx, type, 4, ne);
}

struct wsp_ggml_tensor * wsp_ggml_new_i32(struct wsp_ggml_context * ctx, int32_t value) {
    wsp_ggml_scratch_save(ctx);

    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_1d(ctx, WSP_GGML_TYPE_I32, 1);

    wsp_ggml_scratch_load(ctx);

    wsp_ggml_set_i32(result, value);

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_new_f32(struct wsp_ggml_context * ctx, float value) {
    wsp_ggml_scratch_save(ctx);

    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_1d(ctx, WSP_GGML_TYPE_F32, 1);

    wsp_ggml_scratch_load(ctx);

    wsp_ggml_set_f32(result, value);

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_dup_tensor(struct wsp_ggml_context * ctx, const struct wsp_ggml_tensor * src) {
    return wsp_ggml_new_tensor(ctx, src->type, WSP_GGML_MAX_DIMS, src->ne);
}

static void wsp_ggml_set_op_params(struct wsp_ggml_tensor * tensor, const void * params, size_t params_size) {
    WSP_GGML_ASSERT(tensor != NULL); // silence -Warray-bounds warnings
    assert(params_size <= WSP_GGML_MAX_OP_PARAMS);
    memcpy(tensor->op_params, params, params_size);
}

static int32_t wsp_ggml_get_op_params_i32(const struct wsp_ggml_tensor * tensor, uint32_t i) {
    assert(i < WSP_GGML_MAX_OP_PARAMS / sizeof(int32_t));
    return ((const int32_t *)(tensor->op_params))[i];
}

static float wsp_ggml_get_op_params_f32(const struct wsp_ggml_tensor * tensor, uint32_t i) {
    assert(i < WSP_GGML_MAX_OP_PARAMS / sizeof(float));
    return ((const float *)(tensor->op_params))[i];
}

static void wsp_ggml_set_op_params_i32(struct wsp_ggml_tensor * tensor, uint32_t i, int32_t value) {
    assert(i < WSP_GGML_MAX_OP_PARAMS / sizeof(int32_t));
    ((int32_t *)(tensor->op_params))[i] = value;
}

static void wsp_ggml_set_op_params_f32(struct wsp_ggml_tensor * tensor, uint32_t i, float value) {
    assert(i < WSP_GGML_MAX_OP_PARAMS / sizeof(float));
    ((float *)(tensor->op_params))[i] = value;
}

struct wsp_ggml_tensor * wsp_ggml_set_zero(struct wsp_ggml_tensor * tensor) {
    if (wsp_ggml_is_empty(tensor)) {
        return tensor;
    }
    if (tensor->buffer) {
        wsp_ggml_backend_tensor_memset(tensor, 0, 0, wsp_ggml_nbytes(tensor));
    } else {
        WSP_GGML_ASSERT(tensor->data);
        memset(tensor->data, 0, wsp_ggml_nbytes(tensor));
    }
    return tensor;
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

void wsp_ggml_unravel_index(const struct wsp_ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3) {
    const int64_t ne2 = tensor->ne[2];
    const int64_t ne1 = tensor->ne[1];
    const int64_t ne0 = tensor->ne[0];

    const int64_t i3_ = (i/(ne2*ne1*ne0));
    const int64_t i2_ = (i - i3_*ne2*ne1*ne0)/(ne1*ne0);
    const int64_t i1_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0)/ne0;
    const int64_t i0_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0 - i1_*ne0);

    if (i0) {
        * i0 = i0_;
    }
    if (i1) {
        * i1 = i1_;
    }
    if (i2) {
        * i2 = i2_;
    }
    if (i3) {
        * i3 = i3_;
    }
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

void * wsp_ggml_get_data(const struct wsp_ggml_tensor * tensor) {
    return tensor->data;
}

float * wsp_ggml_get_data_f32(const struct wsp_ggml_tensor * tensor) {
    assert(tensor->type == WSP_GGML_TYPE_F32);
    return (float *)(tensor->data);
}

enum wsp_ggml_unary_op wsp_ggml_get_unary_op(const struct wsp_ggml_tensor * tensor) {
    WSP_GGML_ASSERT(tensor->op == WSP_GGML_OP_UNARY);
    return (enum wsp_ggml_unary_op) wsp_ggml_get_op_params_i32(tensor, 0);
}

const char * wsp_ggml_get_name(const struct wsp_ggml_tensor * tensor) {
    return tensor->name;
}

struct wsp_ggml_tensor * wsp_ggml_set_name(struct wsp_ggml_tensor * tensor, const char * name) {
    size_t i;
    for (i = 0; i < sizeof(tensor->name) - 1 && name[i] != '\0'; i++) {
        tensor->name[i] = name[i];
    }
    tensor->name[i] = '\0';
    return tensor;
}

struct wsp_ggml_tensor * wsp_ggml_format_name(struct wsp_ggml_tensor * tensor, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(tensor->name, sizeof(tensor->name), fmt, args);
    va_end(args);
    return tensor;
}

struct wsp_ggml_tensor * wsp_ggml_view_tensor(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * src) {
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_impl(ctx, src->type, WSP_GGML_MAX_DIMS, src->ne, src, 0);
    wsp_ggml_format_name(result, "%s (view)", src->name);

    for (int i = 0; i < WSP_GGML_MAX_DIMS; i++) {
        result->nb[i] = src->nb[i];
    }

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_get_first_tensor(const struct wsp_ggml_context * ctx) {
    struct wsp_ggml_object * obj = ctx->objects_begin;

    char * const mem_buffer = ctx->mem_buffer;

    while (obj != NULL) {
        if (obj->type == WSP_GGML_OBJECT_TYPE_TENSOR) {
            return (struct wsp_ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    return NULL;
}

struct wsp_ggml_tensor * wsp_ggml_get_next_tensor(const struct wsp_ggml_context * ctx, struct wsp_ggml_tensor * tensor) {
    struct wsp_ggml_object * obj = (struct wsp_ggml_object *) ((char *)tensor - WSP_GGML_OBJECT_SIZE);
    obj = obj->next;

    char * const mem_buffer = ctx->mem_buffer;

    while (obj != NULL) {
        if (obj->type == WSP_GGML_OBJECT_TYPE_TENSOR) {
            return (struct wsp_ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    return NULL;
}

struct wsp_ggml_tensor * wsp_ggml_get_tensor(struct wsp_ggml_context * ctx, const char * name) {
    struct wsp_ggml_object * obj = ctx->objects_begin;

    char * const mem_buffer = ctx->mem_buffer;

    while (obj != NULL) {
        if (obj->type == WSP_GGML_OBJECT_TYPE_TENSOR) {
            struct wsp_ggml_tensor * cur = (struct wsp_ggml_tensor *)(mem_buffer + obj->offs);
            if (strcmp(cur->name, name) == 0) {
                return cur;
            }
        }

        obj = obj->next;
    }

    return NULL;
}

////////////////////////////////////////////////////////////////////////////////

// wsp_ggml_dup

static struct wsp_ggml_tensor * wsp_ggml_dup_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        bool                  inplace) {
    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    result->op     = WSP_GGML_OP_DUP;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_dup(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_dup_impl(ctx, a, false);
}

struct wsp_ggml_tensor * wsp_ggml_dup_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_dup_impl(ctx, a, true);
}

// wsp_ggml_add

static struct wsp_ggml_tensor * wsp_ggml_add_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        bool                  inplace) {
    WSP_GGML_ASSERT(wsp_ggml_can_repeat(b, a));

    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    result->op     = WSP_GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_add(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    return wsp_ggml_add_impl(ctx, a, b, false);
}

struct wsp_ggml_tensor * wsp_ggml_add_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    return wsp_ggml_add_impl(ctx, a, b, true);
}

// wsp_ggml_add_cast

static struct wsp_ggml_tensor * wsp_ggml_add_cast_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        enum   wsp_ggml_type      type) {
    // TODO: support less-strict constraint
    //       WSP_GGML_ASSERT(wsp_ggml_can_repeat(b, a));
    WSP_GGML_ASSERT(wsp_ggml_can_repeat_rows(b, a));

    // currently only supported for quantized input and f16
    WSP_GGML_ASSERT(wsp_ggml_is_quantized(a->type) ||
                a->type == WSP_GGML_TYPE_F16 ||
                a->type == WSP_GGML_TYPE_BF16);

    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor(ctx, type, WSP_GGML_MAX_DIMS, a->ne);

    result->op     = WSP_GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_add_cast(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        enum   wsp_ggml_type      type) {
    return wsp_ggml_add_cast_impl(ctx, a, b, type);
}

// wsp_ggml_add1

static struct wsp_ggml_tensor * wsp_ggml_add1_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        bool                  inplace) {
    WSP_GGML_ASSERT(wsp_ggml_is_scalar(b));
    WSP_GGML_ASSERT(wsp_ggml_is_padded_1d(a));

    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    result->op     = WSP_GGML_OP_ADD1;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_add1(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    return wsp_ggml_add1_impl(ctx, a, b, false);
}

struct wsp_ggml_tensor * wsp_ggml_add1_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    return wsp_ggml_add1_impl(ctx, a, b, true);
}

// wsp_ggml_acc

static struct wsp_ggml_tensor * wsp_ggml_acc_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    WSP_GGML_ASSERT(wsp_ggml_nelements(b) <= wsp_ggml_nelements(a));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(a));
    WSP_GGML_ASSERT(a->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT(b->type == WSP_GGML_TYPE_F32);

    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    wsp_ggml_set_op_params(result, params, sizeof(params));

    result->op     = WSP_GGML_OP_ACC;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_acc(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    return wsp_ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

struct wsp_ggml_tensor * wsp_ggml_acc_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    return wsp_ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

// wsp_ggml_sub

static struct wsp_ggml_tensor * wsp_ggml_sub_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        bool                  inplace) {
    WSP_GGML_ASSERT(wsp_ggml_can_repeat(b, a));

    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    result->op     = WSP_GGML_OP_SUB;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_sub(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    return wsp_ggml_sub_impl(ctx, a, b, false);
}

struct wsp_ggml_tensor * wsp_ggml_sub_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    return wsp_ggml_sub_impl(ctx, a, b, true);
}

// wsp_ggml_mul

static struct wsp_ggml_tensor * wsp_ggml_mul_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        bool                  inplace) {
    WSP_GGML_ASSERT(wsp_ggml_can_repeat(b, a));

    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    result->op     = WSP_GGML_OP_MUL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_mul(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    return wsp_ggml_mul_impl(ctx, a, b, false);
}

struct wsp_ggml_tensor * wsp_ggml_mul_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    return wsp_ggml_mul_impl(ctx, a, b, true);
}

// wsp_ggml_div

static struct wsp_ggml_tensor * wsp_ggml_div_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        bool                  inplace) {
    WSP_GGML_ASSERT(wsp_ggml_can_repeat(b, a));

    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    result->op     = WSP_GGML_OP_DIV;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_div(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    return wsp_ggml_div_impl(ctx, a, b, false);
}

struct wsp_ggml_tensor * wsp_ggml_div_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    return wsp_ggml_div_impl(ctx, a, b, true);
}

// wsp_ggml_sqr

static struct wsp_ggml_tensor * wsp_ggml_sqr_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        bool                  inplace) {
    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    result->op     = WSP_GGML_OP_SQR;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_sqr(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_sqr_impl(ctx, a, false);
}

struct wsp_ggml_tensor * wsp_ggml_sqr_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_sqr_impl(ctx, a, true);
}

// wsp_ggml_sqrt

static struct wsp_ggml_tensor * wsp_ggml_sqrt_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        bool                  inplace) {
    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    result->op     = WSP_GGML_OP_SQRT;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_sqrt(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_sqrt_impl(ctx, a, false);
}

struct wsp_ggml_tensor * wsp_ggml_sqrt_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_sqrt_impl(ctx, a, true);
}

// wsp_ggml_log

static struct wsp_ggml_tensor * wsp_ggml_log_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        bool                  inplace) {
    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    result->op     = WSP_GGML_OP_LOG;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_log(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_log_impl(ctx, a, false);
}

struct wsp_ggml_tensor * wsp_ggml_log_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_log_impl(ctx, a, true);
}

// wsp_ggml_sin

static struct wsp_ggml_tensor * wsp_ggml_sin_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        bool                  inplace) {
    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    result->op     = WSP_GGML_OP_SIN;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_sin(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_sin_impl(ctx, a, false);
}

struct wsp_ggml_tensor * wsp_ggml_sin_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_sin_impl(ctx, a, true);
}

// wsp_ggml_cos

static struct wsp_ggml_tensor * wsp_ggml_cos_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        bool                  inplace) {
    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    result->op     = WSP_GGML_OP_COS;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_cos(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_cos_impl(ctx, a, false);
}

struct wsp_ggml_tensor * wsp_ggml_cos_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_cos_impl(ctx, a, true);
}

// wsp_ggml_sum

struct wsp_ggml_tensor * wsp_ggml_sum(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = WSP_GGML_OP_SUM;
    result->src[0] = a;

    return result;
}

// wsp_ggml_sum_rows

struct wsp_ggml_tensor * wsp_ggml_sum_rows(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    int64_t ne[WSP_GGML_MAX_DIMS] = { 1 };
    for (int i = 1; i < WSP_GGML_MAX_DIMS; ++i) {
        ne[i] = a->ne[i];
    }

    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor(ctx, a->type, WSP_GGML_MAX_DIMS, ne);

    result->op     = WSP_GGML_OP_SUM_ROWS;
    result->src[0] = a;

    return result;
}

// wsp_ggml_mean

struct wsp_ggml_tensor * wsp_ggml_mean(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor(ctx, WSP_GGML_TYPE_F32, 4, ne);

    result->op     = WSP_GGML_OP_MEAN;
    result->src[0] = a;

    return result;
}

// wsp_ggml_argmax

struct wsp_ggml_tensor * wsp_ggml_argmax(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    WSP_GGML_ASSERT(wsp_ggml_is_matrix(a));

    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_1d(ctx, WSP_GGML_TYPE_I32, a->ne[1]);

    result->op     = WSP_GGML_OP_ARGMAX;
    result->src[0] = a;

    return result;
}

// wsp_ggml_count_equal

struct wsp_ggml_tensor * wsp_ggml_count_equal(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(a, b));

    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_1d(ctx, WSP_GGML_TYPE_I64, 1);

    result->op     = WSP_GGML_OP_COUNT_EQUAL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// wsp_ggml_repeat

struct wsp_ggml_tensor * wsp_ggml_repeat(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    WSP_GGML_ASSERT(wsp_ggml_can_repeat(a, b));

    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor(ctx, a->type, WSP_GGML_MAX_DIMS, b->ne);

    result->op     = WSP_GGML_OP_REPEAT;
    result->src[0] = a;

    return result;
}

// wsp_ggml_repeat_back

struct wsp_ggml_tensor * wsp_ggml_repeat_back(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    WSP_GGML_ASSERT(wsp_ggml_can_repeat(b, a));

    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor(ctx, a->type, WSP_GGML_MAX_DIMS, b->ne);

    result->op     = WSP_GGML_OP_REPEAT_BACK;
    result->src[0] = a;

    return result;
}

// wsp_ggml_concat

struct wsp_ggml_tensor * wsp_ggml_concat(
    struct wsp_ggml_context * ctx,
    struct wsp_ggml_tensor  * a,
    struct wsp_ggml_tensor  * b,
    int                   dim) {
    WSP_GGML_ASSERT(dim >= 0 && dim < WSP_GGML_MAX_DIMS);

    int64_t ne[WSP_GGML_MAX_DIMS];
    for (int d = 0; d < WSP_GGML_MAX_DIMS; ++d) {
        if (d == dim) {
            ne[d] = a->ne[d] + b->ne[d];
            continue;
        }
        WSP_GGML_ASSERT(a->ne[d] == b->ne[d]);
        ne[d] = a->ne[d];
    }

    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor(ctx, a->type, WSP_GGML_MAX_DIMS, ne);

    wsp_ggml_set_op_params_i32(result, 0, dim);

    result->op     = WSP_GGML_OP_CONCAT;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// wsp_ggml_abs

struct wsp_ggml_tensor * wsp_ggml_abs(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary(ctx, a, WSP_GGML_UNARY_OP_ABS);
}

struct wsp_ggml_tensor * wsp_ggml_abs_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary_inplace(ctx, a, WSP_GGML_UNARY_OP_ABS);
}

// wsp_ggml_sgn

struct wsp_ggml_tensor * wsp_ggml_sgn(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary(ctx, a, WSP_GGML_UNARY_OP_SGN);
}

struct wsp_ggml_tensor * wsp_ggml_sgn_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary_inplace(ctx, a, WSP_GGML_UNARY_OP_SGN);
}

// wsp_ggml_neg

struct wsp_ggml_tensor * wsp_ggml_neg(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary(ctx, a, WSP_GGML_UNARY_OP_NEG);
}

struct wsp_ggml_tensor * wsp_ggml_neg_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary_inplace(ctx, a, WSP_GGML_UNARY_OP_NEG);
}

// wsp_ggml_step

struct wsp_ggml_tensor * wsp_ggml_step(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary(ctx, a, WSP_GGML_UNARY_OP_STEP);
}

struct wsp_ggml_tensor * wsp_ggml_step_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary_inplace(ctx, a, WSP_GGML_UNARY_OP_STEP);
}

// wsp_ggml_tanh

struct wsp_ggml_tensor * wsp_ggml_tanh(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary(ctx, a, WSP_GGML_UNARY_OP_TANH);
}

struct wsp_ggml_tensor * wsp_ggml_tanh_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary_inplace(ctx, a, WSP_GGML_UNARY_OP_TANH);
}

// wsp_ggml_elu

struct wsp_ggml_tensor * wsp_ggml_elu(
    struct wsp_ggml_context * ctx,
    struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary(ctx, a, WSP_GGML_UNARY_OP_ELU);
}

struct wsp_ggml_tensor * wsp_ggml_elu_inplace(
    struct wsp_ggml_context * ctx,
    struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary_inplace(ctx, a, WSP_GGML_UNARY_OP_ELU);
}

// wsp_ggml_relu

struct wsp_ggml_tensor * wsp_ggml_relu(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary(ctx, a, WSP_GGML_UNARY_OP_RELU);
}

struct wsp_ggml_tensor * wsp_ggml_relu_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary_inplace(ctx, a, WSP_GGML_UNARY_OP_RELU);
}

// wsp_ggml_leaky_relu

struct wsp_ggml_tensor * wsp_ggml_leaky_relu(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        float                 negative_slope,
        bool                  inplace) {
    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    wsp_ggml_set_op_params(result, &negative_slope, sizeof(negative_slope));

    result->op     = WSP_GGML_OP_LEAKY_RELU;
    result->src[0] = a;

    return result;
}

// wsp_ggml_sigmoid

struct wsp_ggml_tensor * wsp_ggml_sigmoid(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary(ctx, a, WSP_GGML_UNARY_OP_SIGMOID);
}

struct wsp_ggml_tensor * wsp_ggml_sigmoid_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary_inplace(ctx, a, WSP_GGML_UNARY_OP_SIGMOID);
}

// wsp_ggml_gelu

struct wsp_ggml_tensor * wsp_ggml_gelu(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary(ctx, a, WSP_GGML_UNARY_OP_GELU);
}

struct wsp_ggml_tensor * wsp_ggml_gelu_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary_inplace(ctx, a, WSP_GGML_UNARY_OP_GELU);
}

// wsp_ggml_gelu_quick

struct wsp_ggml_tensor * wsp_ggml_gelu_quick(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary(ctx, a, WSP_GGML_UNARY_OP_GELU_QUICK);
}

struct wsp_ggml_tensor * wsp_ggml_gelu_quick_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary_inplace(ctx, a, WSP_GGML_UNARY_OP_GELU_QUICK);
}

// wsp_ggml_silu

struct wsp_ggml_tensor * wsp_ggml_silu(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary(ctx, a, WSP_GGML_UNARY_OP_SILU);
}

struct wsp_ggml_tensor * wsp_ggml_silu_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary_inplace(ctx, a, WSP_GGML_UNARY_OP_SILU);
}

// wsp_ggml_silu_back

struct wsp_ggml_tensor * wsp_ggml_silu_back(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    struct wsp_ggml_tensor * result = wsp_ggml_dup_tensor(ctx, a);

    result->op     = WSP_GGML_OP_SILU_BACK;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// ggml hardswish

struct wsp_ggml_tensor * wsp_ggml_hardswish(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary(ctx, a, WSP_GGML_UNARY_OP_HARDSWISH);
}

// ggml hardsigmoid

struct wsp_ggml_tensor * wsp_ggml_hardsigmoid(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary(ctx, a, WSP_GGML_UNARY_OP_HARDSIGMOID);
}

// ggml exp

struct wsp_ggml_tensor * wsp_ggml_exp(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary(ctx, a, WSP_GGML_UNARY_OP_EXP);
}

struct wsp_ggml_tensor * wsp_ggml_exp_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_unary_inplace(ctx, a, WSP_GGML_UNARY_OP_EXP);
}

// wsp_ggml_norm

static struct wsp_ggml_tensor * wsp_ggml_norm_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        float                 eps,
        bool                  inplace) {
    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    wsp_ggml_set_op_params(result, &eps, sizeof(eps));

    result->op     = WSP_GGML_OP_NORM;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_norm(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        float                 eps) {
    return wsp_ggml_norm_impl(ctx, a, eps, false);
}

struct wsp_ggml_tensor * wsp_ggml_norm_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        float                 eps) {
    return wsp_ggml_norm_impl(ctx, a, eps, true);
}

// wsp_ggml_rms_norm

static struct wsp_ggml_tensor * wsp_ggml_rms_norm_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        float                 eps,
        bool                  inplace) {
    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    wsp_ggml_set_op_params(result, &eps, sizeof(eps));

    result->op     = WSP_GGML_OP_RMS_NORM;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_rms_norm(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        float                 eps) {
    return wsp_ggml_rms_norm_impl(ctx, a, eps, false);
}

struct wsp_ggml_tensor * wsp_ggml_rms_norm_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        float                 eps) {
    return wsp_ggml_rms_norm_impl(ctx, a, eps, true);
}

// wsp_ggml_rms_norm_back

struct wsp_ggml_tensor * wsp_ggml_rms_norm_back(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        float                 eps) {
    struct wsp_ggml_tensor * result = wsp_ggml_dup_tensor(ctx, a);

    wsp_ggml_set_op_params(result, &eps, sizeof(eps));

    result->op     = WSP_GGML_OP_RMS_NORM_BACK;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// wsp_ggml_group_norm

static struct wsp_ggml_tensor * wsp_ggml_group_norm_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int                   n_groups,
        float                 eps,
        bool                  inplace) {
    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    wsp_ggml_set_op_params_i32(result, 0, n_groups);
    wsp_ggml_set_op_params_f32(result, 1, eps);

    result->op     = WSP_GGML_OP_GROUP_NORM;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_group_norm(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int                   n_groups,
        float                 eps) {
    return wsp_ggml_group_norm_impl(ctx, a, n_groups, eps, false);
}

struct wsp_ggml_tensor * wsp_ggml_group_norm_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int                   n_groups,
        float                 eps) {
    return wsp_ggml_group_norm_impl(ctx, a, n_groups, eps, true);
}

// wsp_ggml_mul_mat

struct wsp_ggml_tensor * wsp_ggml_mul_mat(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    WSP_GGML_ASSERT(wsp_ggml_can_mul_mat(a, b));
    WSP_GGML_ASSERT(!wsp_ggml_is_transposed(a));

    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor(ctx, WSP_GGML_TYPE_F32, 4, ne);

    result->op     = WSP_GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

void wsp_ggml_mul_mat_set_prec(
        struct wsp_ggml_tensor * a,
        enum wsp_ggml_prec       prec) {
    WSP_GGML_ASSERT(a->op == WSP_GGML_OP_MUL_MAT);

    const int32_t prec_i32 = (int32_t) prec;

    wsp_ggml_set_op_params_i32(a, 0, prec_i32);
}

// wsp_ggml_mul_mat_id

/*
    c = wsp_ggml_mul_mat_id(ctx, as, b, ids);

    as  -> [cols, rows, n_expert]
    ids -> [n_experts_used, n_tokens] (i32)
    b   -> [cols, n_expert_used, n_tokens]
    c   -> [rows, n_expert_used, n_tokens]

    in b, n_experts_used can be broadcasted to match the n_expert_used of ids

    c ~= as[:,:,i] @ b[:,i%r,t], i = ids[e,t] for all e,t in ids
*/
struct wsp_ggml_tensor * wsp_ggml_mul_mat_id(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * as,
        struct wsp_ggml_tensor  * b,
        struct wsp_ggml_tensor  * ids) {
    WSP_GGML_ASSERT(!wsp_ggml_is_transposed(as));
    WSP_GGML_ASSERT(ids->type == WSP_GGML_TYPE_I32);

    WSP_GGML_ASSERT(as->ne[3] == 1); // as is 3d (one matrix per expert)
    WSP_GGML_ASSERT(b->ne[3] == 1); // b is 3d
    WSP_GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1); // ids is 2d
    WSP_GGML_ASSERT(ids->ne[1] == b->ne[2]); // must have an expert list per b row
    WSP_GGML_ASSERT(as->ne[0] == b->ne[0]); // can_mul_mat
    WSP_GGML_ASSERT(ids->ne[0] % b->ne[1] == 0); // can broadcast

    const int64_t ne[4] = { as->ne[1], ids->ne[0], b->ne[2], 1 };
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor(ctx, WSP_GGML_TYPE_F32, 4, ne);

    result->op     = WSP_GGML_OP_MUL_MAT_ID;
    result->src[0] = as;
    result->src[1] = b;
    result->src[2] = ids;

    return result;
}

// wsp_ggml_out_prod

struct wsp_ggml_tensor * wsp_ggml_out_prod(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    WSP_GGML_ASSERT(wsp_ggml_can_out_prod(a, b));
    WSP_GGML_ASSERT(!wsp_ggml_is_transposed(a));

    // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
    const int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor(ctx, WSP_GGML_TYPE_F32, 4, ne);

    result->op     = WSP_GGML_OP_OUT_PROD;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// wsp_ggml_scale

static struct wsp_ggml_tensor * wsp_ggml_scale_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        float                 s,
        bool                  inplace) {
    WSP_GGML_ASSERT(wsp_ggml_is_padded_1d(a));

    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    wsp_ggml_set_op_params(result, &s, sizeof(s));

    result->op     = WSP_GGML_OP_SCALE;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_scale(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        float                 s) {
    return wsp_ggml_scale_impl(ctx, a, s, false);
}

struct wsp_ggml_tensor * wsp_ggml_scale_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        float                 s) {
    return wsp_ggml_scale_impl(ctx, a, s, true);
}

// wsp_ggml_set

static struct wsp_ggml_tensor * wsp_ggml_set_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    WSP_GGML_ASSERT(wsp_ggml_nelements(a) >= wsp_ggml_nelements(b));

    // make a view of the destination
    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    WSP_GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    wsp_ggml_set_op_params(result, params, sizeof(params));

    result->op     = WSP_GGML_OP_SET;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_set(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    return wsp_ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

struct wsp_ggml_tensor * wsp_ggml_set_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    return wsp_ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

struct wsp_ggml_tensor * wsp_ggml_set_1d(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        size_t                offset) {
    return wsp_ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, false);
}

struct wsp_ggml_tensor * wsp_ggml_set_1d_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        size_t                offset) {
    return wsp_ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, true);
}

struct wsp_ggml_tensor * wsp_ggml_set_2d(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    return wsp_ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, false);
}

struct wsp_ggml_tensor * wsp_ggml_set_2d_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    return wsp_ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, true);
}

// wsp_ggml_cpy

static struct wsp_ggml_tensor * wsp_ggml_cpy_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    WSP_GGML_ASSERT(wsp_ggml_nelements(a) == wsp_ggml_nelements(b));

    // make a view of the destination
    struct wsp_ggml_tensor * result = wsp_ggml_view_tensor(ctx, b);
    if (strlen(b->name) > 0) {
        wsp_ggml_format_name(result, "%s (copy of %s)", b->name, a->name);
    } else {
        wsp_ggml_format_name(result, "%s (copy)", a->name);
    }

    result->op     = WSP_GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_cpy(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor * a,
        struct wsp_ggml_tensor * b) {
    return wsp_ggml_cpy_impl(ctx, a, b);
}

struct wsp_ggml_tensor * wsp_ggml_cast(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        enum   wsp_ggml_type      type) {
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor(ctx, type, WSP_GGML_MAX_DIMS, a->ne);
    wsp_ggml_format_name(result, "%s (copy)", a->name);

    result->op     = WSP_GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = result;

    return result;
}

// wsp_ggml_cont

static struct wsp_ggml_tensor * wsp_ggml_cont_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    struct wsp_ggml_tensor * result = wsp_ggml_dup_tensor(ctx, a);
    wsp_ggml_format_name(result, "%s (cont)", a->name);

    result->op     = WSP_GGML_OP_CONT;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_cont(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor * a) {
    return wsp_ggml_cont_impl(ctx, a);
}

// make contiguous, with new shape
WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_cont_1d(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int64_t               ne0) {
    return wsp_ggml_cont_4d(ctx, a, ne0, 1, 1, 1);
}

WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_cont_2d(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    return wsp_ggml_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_cont_3d(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    return wsp_ggml_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

struct wsp_ggml_tensor * wsp_ggml_cont_4d(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    WSP_GGML_ASSERT(wsp_ggml_nelements(a) == (ne0*ne1*ne2*ne3));

    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
    wsp_ggml_format_name(result, "%s (cont)", a->name);

    result->op     = WSP_GGML_OP_CONT;
    result->src[0] = a;

    return result;
}

// wsp_ggml_reshape

struct wsp_ggml_tensor * wsp_ggml_reshape(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor * a,
        struct wsp_ggml_tensor * b) {
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(a));
    // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
    WSP_GGML_ASSERT(wsp_ggml_nelements(a) == wsp_ggml_nelements(b));

    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_impl(ctx, a->type, WSP_GGML_MAX_DIMS, b->ne, a, 0);
    wsp_ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = WSP_GGML_OP_RESHAPE;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_reshape_1d(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int64_t               ne0) {
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(a));
    WSP_GGML_ASSERT(wsp_ggml_nelements(a) == ne0);

    const int64_t ne[1] = { ne0 };
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
    wsp_ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = WSP_GGML_OP_RESHAPE;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_reshape_2d(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(a));
    WSP_GGML_ASSERT(wsp_ggml_nelements(a) == ne0*ne1);

    const int64_t ne[2] = { ne0, ne1 };
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    wsp_ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = WSP_GGML_OP_RESHAPE;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_reshape_3d(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(a));
    WSP_GGML_ASSERT(wsp_ggml_nelements(a) == ne0*ne1*ne2);

    const int64_t ne[3] = { ne0, ne1, ne2 };
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
    wsp_ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = WSP_GGML_OP_RESHAPE;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_reshape_4d(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(a));
    WSP_GGML_ASSERT(wsp_ggml_nelements(a) == ne0*ne1*ne2*ne3);

    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
    wsp_ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = WSP_GGML_OP_RESHAPE;
    result->src[0] = a;

    return result;
}

static struct wsp_ggml_tensor * wsp_ggml_view_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int                   n_dims,
        const int64_t       * ne,
        size_t                offset) {
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, offset);
    wsp_ggml_format_name(result, "%s (view)", a->name);

    wsp_ggml_set_op_params(result, &offset, sizeof(offset));

    result->op     = WSP_GGML_OP_VIEW;
    result->src[0] = a;

    return result;
}

// wsp_ggml_view_1d

struct wsp_ggml_tensor * wsp_ggml_view_1d(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int64_t               ne0,
        size_t                offset) {
    struct wsp_ggml_tensor * result = wsp_ggml_view_impl(ctx, a, 1, &ne0, offset);

    return result;
}

// wsp_ggml_view_2d

struct wsp_ggml_tensor * wsp_ggml_view_2d(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1,
        size_t                offset) {
    const int64_t ne[2] = { ne0, ne1 };

    struct wsp_ggml_tensor * result = wsp_ggml_view_impl(ctx, a, 2, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = result->nb[1]*ne1;
    result->nb[3] = result->nb[2];

    return result;
}

// wsp_ggml_view_3d

struct wsp_ggml_tensor * wsp_ggml_view_3d(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        size_t                nb1,
        size_t                nb2,
        size_t                offset) {
    const int64_t ne[3] = { ne0, ne1, ne2 };

    struct wsp_ggml_tensor * result = wsp_ggml_view_impl(ctx, a, 3, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = result->nb[2]*ne2;

    return result;
}

// wsp_ggml_view_4d

struct wsp_ggml_tensor * wsp_ggml_view_4d(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };

    struct wsp_ggml_tensor * result = wsp_ggml_view_impl(ctx, a, 4, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = nb3;

    return result;
}

// wsp_ggml_permute

struct wsp_ggml_tensor * wsp_ggml_permute(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int                   axis0,
        int                   axis1,
        int                   axis2,
        int                   axis3) {
    WSP_GGML_ASSERT(axis0 >= 0 && axis0 < WSP_GGML_MAX_DIMS);
    WSP_GGML_ASSERT(axis1 >= 0 && axis1 < WSP_GGML_MAX_DIMS);
    WSP_GGML_ASSERT(axis2 >= 0 && axis2 < WSP_GGML_MAX_DIMS);
    WSP_GGML_ASSERT(axis3 >= 0 && axis3 < WSP_GGML_MAX_DIMS);

    WSP_GGML_ASSERT(axis0 != axis1);
    WSP_GGML_ASSERT(axis0 != axis2);
    WSP_GGML_ASSERT(axis0 != axis3);
    WSP_GGML_ASSERT(axis1 != axis2);
    WSP_GGML_ASSERT(axis1 != axis3);
    WSP_GGML_ASSERT(axis2 != axis3);

    struct wsp_ggml_tensor * result = wsp_ggml_view_tensor(ctx, a);
    wsp_ggml_format_name(result, "%s (permuted)", a->name);

    int ne[WSP_GGML_MAX_DIMS];
    int nb[WSP_GGML_MAX_DIMS];

    ne[axis0] = a->ne[0];
    ne[axis1] = a->ne[1];
    ne[axis2] = a->ne[2];
    ne[axis3] = a->ne[3];

    nb[axis0] = a->nb[0];
    nb[axis1] = a->nb[1];
    nb[axis2] = a->nb[2];
    nb[axis3] = a->nb[3];

    result->ne[0] = ne[0];
    result->ne[1] = ne[1];
    result->ne[2] = ne[2];
    result->ne[3] = ne[3];

    result->nb[0] = nb[0];
    result->nb[1] = nb[1];
    result->nb[2] = nb[2];
    result->nb[3] = nb[3];

    result->op     = WSP_GGML_OP_PERMUTE;
    result->src[0] = a;

    int32_t params[] = { axis0, axis1, axis2, axis3 };
    wsp_ggml_set_op_params(result, params, sizeof(params));

    return result;
}

// wsp_ggml_transpose

struct wsp_ggml_tensor * wsp_ggml_transpose(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    struct wsp_ggml_tensor * result = wsp_ggml_view_tensor(ctx, a);
    wsp_ggml_format_name(result, "%s (transposed)", a->name);

    result->ne[0] = a->ne[1];
    result->ne[1] = a->ne[0];

    result->nb[0] = a->nb[1];
    result->nb[1] = a->nb[0];

    result->op     = WSP_GGML_OP_TRANSPOSE;
    result->src[0] = a;

    return result;
}

// wsp_ggml_get_rows

struct wsp_ggml_tensor * wsp_ggml_get_rows(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    WSP_GGML_ASSERT(a->ne[2] == b->ne[1]);
    WSP_GGML_ASSERT(b->ne[3] == 1);
    WSP_GGML_ASSERT(b->type == WSP_GGML_TYPE_I32);

    // TODO: implement non F32 return
    enum wsp_ggml_type type = WSP_GGML_TYPE_F32;
    if (a->type == WSP_GGML_TYPE_I32) {
        type = a->type;
    }
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

    result->op     = WSP_GGML_OP_GET_ROWS;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// wsp_ggml_get_rows_back

struct wsp_ggml_tensor * wsp_ggml_get_rows_back(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        struct wsp_ggml_tensor  * c) {
    WSP_GGML_ASSERT(wsp_ggml_is_matrix(a) && wsp_ggml_is_vector(b) && b->type == WSP_GGML_TYPE_I32);
    WSP_GGML_ASSERT(wsp_ggml_is_matrix(c) && (a->ne[0] == c->ne[0]));

    // TODO: implement non F32 return
    //struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_2d(ctx, WSP_GGML_TYPE_F32, c->ne[0], c->ne[1]);

    result->op     = WSP_GGML_OP_GET_ROWS_BACK;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// wsp_ggml_diag

struct wsp_ggml_tensor * wsp_ggml_diag(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    WSP_GGML_ASSERT(a->ne[1] == 1);

    const int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor(ctx, a->type, 4, ne);

    result->op     = WSP_GGML_OP_DIAG;
    result->src[0] = a;

    return result;
}

// wsp_ggml_diag_mask_inf

static struct wsp_ggml_tensor * wsp_ggml_diag_mask_inf_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int                   n_past,
        bool                  inplace) {
    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    wsp_ggml_set_op_params(result, params, sizeof(params));

    result->op     = WSP_GGML_OP_DIAG_MASK_INF;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_diag_mask_inf(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int                   n_past) {
    return wsp_ggml_diag_mask_inf_impl(ctx, a, n_past, false);
}

struct wsp_ggml_tensor * wsp_ggml_diag_mask_inf_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int                   n_past) {
    return wsp_ggml_diag_mask_inf_impl(ctx, a, n_past, true);
}

// wsp_ggml_diag_mask_zero

static struct wsp_ggml_tensor * wsp_ggml_diag_mask_zero_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int                   n_past,
        bool                  inplace) {
    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    wsp_ggml_set_op_params(result, params, sizeof(params));

    result->op     = WSP_GGML_OP_DIAG_MASK_ZERO;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_diag_mask_zero(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int                   n_past) {
    return wsp_ggml_diag_mask_zero_impl(ctx, a, n_past, false);
}

struct wsp_ggml_tensor * wsp_ggml_diag_mask_zero_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int                   n_past) {
    return wsp_ggml_diag_mask_zero_impl(ctx, a, n_past, true);
}

// wsp_ggml_soft_max

static struct wsp_ggml_tensor * wsp_ggml_soft_max_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * mask,
        float                 scale,
        float                 max_bias,
        bool                  inplace) {
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(a));

    if (mask) {
        WSP_GGML_ASSERT(mask->type == WSP_GGML_TYPE_F16 || mask->type == WSP_GGML_TYPE_F32);
        WSP_GGML_ASSERT(wsp_ggml_is_contiguous(mask));
        WSP_GGML_ASSERT(wsp_ggml_is_matrix(mask));
        WSP_GGML_ASSERT(mask->ne[0] == a->ne[0]);
        WSP_GGML_ASSERT(mask->ne[1] >= a->ne[1]);
    }

    if (max_bias > 0.0f) {
        WSP_GGML_ASSERT(mask);
    }

    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    float params[] = { scale, max_bias };
    wsp_ggml_set_op_params(result, params, sizeof(params));

    result->op     = WSP_GGML_OP_SOFT_MAX;
    result->src[0] = a;
    result->src[1] = mask;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_soft_max(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_soft_max_impl(ctx, a, NULL, 1.0f, 0.0f, false);
}

struct wsp_ggml_tensor * wsp_ggml_soft_max_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a) {
    return wsp_ggml_soft_max_impl(ctx, a, NULL, 1.0f, 0.0f, true);
}

struct wsp_ggml_tensor * wsp_ggml_soft_max_ext(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * mask,
        float                 scale,
        float                 max_bias) {
    return wsp_ggml_soft_max_impl(ctx, a, mask, scale, max_bias, false);
}

// wsp_ggml_soft_max_back

static struct wsp_ggml_tensor * wsp_ggml_soft_max_back_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        bool                  inplace) {
    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    result->op     = WSP_GGML_OP_SOFT_MAX_BACK;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_soft_max_back(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    return wsp_ggml_soft_max_back_impl(ctx, a, b, false);
}

struct wsp_ggml_tensor * wsp_ggml_soft_max_back_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    return wsp_ggml_soft_max_back_impl(ctx, a, b, true);
}

// wsp_ggml_rope

static struct wsp_ggml_tensor * wsp_ggml_rope_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        struct wsp_ggml_tensor  * c,
        int                   n_dims,
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  inplace) {
    WSP_GGML_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported");

    WSP_GGML_ASSERT(wsp_ggml_is_vector(b));
    WSP_GGML_ASSERT(b->type == WSP_GGML_TYPE_I32);
    WSP_GGML_ASSERT(a->ne[2] == b->ne[0]);

    if (c) {
        WSP_GGML_ASSERT(c->type == WSP_GGML_TYPE_F32);
        WSP_GGML_ASSERT(c->ne[0] >= n_dims / 2);
    }

    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    int32_t params[11] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
    memcpy(params +  5, &freq_base,    sizeof(float));
    memcpy(params +  6, &freq_scale,   sizeof(float));
    memcpy(params +  7, &ext_factor,   sizeof(float));
    memcpy(params +  8, &attn_factor,  sizeof(float));
    memcpy(params +  9, &beta_fast,    sizeof(float));
    memcpy(params + 10, &beta_slow,    sizeof(float));
    wsp_ggml_set_op_params(result, params, sizeof(params));

    result->op     = WSP_GGML_OP_ROPE;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_rope(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        int                   n_dims,
        int                   mode) {
    return wsp_ggml_rope_impl(
        ctx, a, b, NULL, n_dims, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    );
}

struct wsp_ggml_tensor * wsp_ggml_rope_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        int                   n_dims,
        int                   mode) {
    return wsp_ggml_rope_impl(
        ctx, a, b, NULL, n_dims, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
    );
}

struct wsp_ggml_tensor * wsp_ggml_rope_ext(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        struct wsp_ggml_tensor  * c,
        int                   n_dims,
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return wsp_ggml_rope_impl(
        ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

struct wsp_ggml_tensor * wsp_ggml_rope_ext_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        struct wsp_ggml_tensor  * c,
        int                   n_dims,
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return wsp_ggml_rope_impl(
        ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

struct wsp_ggml_tensor * wsp_ggml_rope_custom(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        int                   n_dims,
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return wsp_ggml_rope_impl(
        ctx, a, b, NULL, n_dims, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

struct wsp_ggml_tensor * wsp_ggml_rope_custom_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        int                   n_dims,
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return wsp_ggml_rope_impl(
        ctx, a, b, NULL, n_dims, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

// wsp_ggml_rope_back

struct wsp_ggml_tensor * wsp_ggml_rope_back(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        struct wsp_ggml_tensor  * c,
        int                   n_dims,
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    WSP_GGML_ASSERT(wsp_ggml_is_vector(b));
    WSP_GGML_ASSERT(b->type == WSP_GGML_TYPE_I32);
    WSP_GGML_ASSERT(a->ne[2] == b->ne[0]);

    struct wsp_ggml_tensor * result = wsp_ggml_dup_tensor(ctx, a);

    int32_t params[11] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
    memcpy(params +  5, &freq_base,    sizeof(float));
    memcpy(params +  6, &freq_scale,   sizeof(float));
    memcpy(params +  7, &ext_factor,   sizeof(float));
    memcpy(params +  8, &attn_factor,  sizeof(float));
    memcpy(params +  9, &beta_fast,    sizeof(float));
    memcpy(params + 10, &beta_slow,    sizeof(float));
    wsp_ggml_set_op_params(result, params, sizeof(params));

    result->op     = WSP_GGML_OP_ROPE_BACK;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    return result;
}

// wsp_ggml_clamp

struct wsp_ggml_tensor * wsp_ggml_clamp(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        float                 min,
        float                 max) {
    // TODO: when implement backward, fix this:
    struct wsp_ggml_tensor * result = wsp_ggml_view_tensor(ctx, a);

    float params[] = { min, max };
    wsp_ggml_set_op_params(result, params, sizeof(params));

    result->op     = WSP_GGML_OP_CLAMP;
    result->src[0] = a;

    return result;
}

// wsp_ggml_conv_1d

static int64_t wsp_ggml_calc_conv_output_size(int64_t ins, int64_t ks, int s, int p, int d) {
    return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_conv_1d(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        int                   s0,
        int                   p0,
        int                   d0) {
    struct wsp_ggml_tensor * im2col = wsp_ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, WSP_GGML_TYPE_F16); // [N, OL, IC * K]

    struct wsp_ggml_tensor * result =
        wsp_ggml_mul_mat(ctx,
                wsp_ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])), // [N, OL, IC * K] => [N*OL, IC * K]
                wsp_ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));                    // [OC，IC, K] => [OC, IC * K]

    result = wsp_ggml_reshape_3d(ctx, result, im2col->ne[1], a->ne[2], im2col->ne[2]); // [N, OC, OL]

    return result;
}

// wsp_ggml_conv_1d_ph

struct wsp_ggml_tensor* wsp_ggml_conv_1d_ph(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        int                   s,
        int                   d) {
    return wsp_ggml_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

// wsp_ggml_conv_transpose_1d

static int64_t wsp_ggml_calc_conv_transpose_1d_output_size(int64_t ins, int64_t ks, int s, int p, int d) {
    return (ins - 1) * s - 2 * p + d * (ks - 1) + 1;
}

WSP_GGML_API struct wsp_ggml_tensor * wsp_ggml_conv_transpose_1d(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        int                   s0,
        int                   p0,
        int                   d0) {
    WSP_GGML_ASSERT(wsp_ggml_is_matrix(b));
    WSP_GGML_ASSERT(a->ne[2] == b->ne[1]);
    WSP_GGML_ASSERT(a->ne[3] == 1);

    WSP_GGML_ASSERT(p0 == 0);
    WSP_GGML_ASSERT(d0 == 1);

    const int64_t ne[4] = {
        wsp_ggml_calc_conv_transpose_1d_output_size(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
        a->ne[1], b->ne[2], 1,
    };
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor(ctx, WSP_GGML_TYPE_F32, 4, ne);

    int32_t params[] = { s0, p0, d0 };
    wsp_ggml_set_op_params(result, params, sizeof(params));

    result->op     = WSP_GGML_OP_CONV_TRANSPOSE_1D;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// wsp_ggml_conv_depthwise

struct wsp_ggml_tensor * wsp_ggml_conv_depthwise_2d(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        int                   s0,
        int                   s1,
        int                   p0,
        int                   p1,
        int                   d0,
        int                   d1) {
    struct wsp_ggml_tensor * new_a = wsp_ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    struct wsp_ggml_tensor * im2col = wsp_ggml_im2col(ctx, new_a,
                                        wsp_ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
                                        s0, s1, p0, p1, d0, d1, true, WSP_GGML_TYPE_F16); // [N * IC, OH, OW, KH * KW]
    struct wsp_ggml_tensor * new_b = wsp_ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

    new_a = wsp_ggml_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2],  new_a->ne[3], 1);                       // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
    struct wsp_ggml_tensor * result = wsp_ggml_mul_mat(ctx, new_a, new_b);
    result = wsp_ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]); // [N, OC, OH, OW]

    return result;
}
// wsp_ggml_conv_2d

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
struct wsp_ggml_tensor * wsp_ggml_im2col(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        int                   s0,
        int                   s1,
        int                   p0,
        int                   p1,
        int                   d0,
        int                   d1,
        bool                  is_2D,
        enum wsp_ggml_type        dst_type) {
    if(is_2D) {
        WSP_GGML_ASSERT(a->ne[2] == b->ne[2]);
    } else {
        WSP_GGML_ASSERT(a->ne[1] == b->ne[1]);
        WSP_GGML_ASSERT(b->ne[3] == 1);
    }

    const int64_t OH = is_2D ? wsp_ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    const int64_t OW =         wsp_ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);

    WSP_GGML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
    WSP_GGML_ASSERT((OW > 0)           && "b too small compared to a");

    const int64_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ?      b->ne[3] : 1,
    };

    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    wsp_ggml_set_op_params(result, params, sizeof(params));

    result->op     = WSP_GGML_OP_IM2COL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_im2col_back(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        int64_t             * ne,
        int                   s0,
        int                   s1,
        int                   p0,
        int                   p1,
        int                   d0,
        int                   d1,
        bool                  is_2D) {
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor(ctx, WSP_GGML_TYPE_F32, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    wsp_ggml_set_op_params(result, params, sizeof(params));

    result->op     = WSP_GGML_OP_IM2COL_BACK;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OC, OH, OW]
struct wsp_ggml_tensor * wsp_ggml_conv_2d(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        int                   s0,
        int                   s1,
        int                   p0,
        int                   p1,
        int                   d0,
        int                   d1) {
    struct wsp_ggml_tensor * im2col = wsp_ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a->type); // [N, OH, OW, IC * KH * KW]

    struct wsp_ggml_tensor * result =
        wsp_ggml_mul_mat(ctx,
                wsp_ggml_reshape_2d(ctx, im2col, im2col->ne[0],  im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
                wsp_ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]),  a->ne[3]));                       // [OC，IC, KH, KW] => [OC, IC * KH * KW]

    result = wsp_ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
    result = wsp_ggml_cont(ctx, wsp_ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OH, OW]


    return result;
}

// wsp_ggml_conv_2d_sk_p0

struct wsp_ggml_tensor * wsp_ggml_conv_2d_sk_p0(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    return wsp_ggml_conv_2d(ctx, a, b, a->ne[0], a->ne[1], 0, 0, 1, 1);
}

// wsp_ggml_conv_2d_s1_ph

struct wsp_ggml_tensor * wsp_ggml_conv_2d_s1_ph(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    return wsp_ggml_conv_2d(ctx, a, b, 1, 1, a->ne[0] / 2, a->ne[1] / 2, 1, 1);
}

// wsp_ggml_conv_transpose_2d_p0

static int64_t wsp_ggml_calc_conv_transpose_output_size(int64_t ins, int64_t ks, int s, int p) {
    return (ins - 1) * s - 2 * p + ks;
}

struct wsp_ggml_tensor * wsp_ggml_conv_transpose_2d_p0(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        int                   stride) {
    WSP_GGML_ASSERT(a->ne[3] == b->ne[2]);

    const int64_t ne[4] = {
        wsp_ggml_calc_conv_transpose_output_size(b->ne[0], a->ne[0], stride, 0 /*p0*/),
        wsp_ggml_calc_conv_transpose_output_size(b->ne[1], a->ne[1], stride, 0 /*p1*/),
        a->ne[2], b->ne[3],
    };

    struct wsp_ggml_tensor* result = wsp_ggml_new_tensor(ctx, WSP_GGML_TYPE_F32, 4, ne);

    wsp_ggml_set_op_params_i32(result, 0, stride);

    result->op     = WSP_GGML_OP_CONV_TRANSPOSE_2D;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// wsp_ggml_pool_*

static int64_t wsp_ggml_calc_pool_output_size(int64_t ins, int ks, int s, float p) {
    return (ins + 2 * p - ks) / s + 1;
}

// wsp_ggml_pool_1d

struct wsp_ggml_tensor * wsp_ggml_pool_1d(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        enum wsp_ggml_op_pool     op,
        int                   k0,
        int                   s0,
        int                   p0) {
    const int64_t ne[4] = {
        wsp_ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        a->ne[1],
        a->ne[2],
        a->ne[3],
    };
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor(ctx, WSP_GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, s0, p0 };
    wsp_ggml_set_op_params(result, params, sizeof(params));

    result->op     = WSP_GGML_OP_POOL_1D;
    result->src[0] = a;

    return result;
}

// wsp_ggml_pool_2d

struct wsp_ggml_tensor * wsp_ggml_pool_2d(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        enum wsp_ggml_op_pool     op,
        int                   k0,
        int                   k1,
        int                   s0,
        int                   s1,
        float                 p0,
        float                 p1) {
    struct wsp_ggml_tensor * result;
    const int64_t ne[4] = {
        wsp_ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        wsp_ggml_calc_pool_output_size(a->ne[1], k1, s1, p1),
        a->ne[2],
        a->ne[3],
    };
    result = wsp_ggml_new_tensor(ctx, WSP_GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    wsp_ggml_set_op_params(result, params, sizeof(params));

    result->op     = WSP_GGML_OP_POOL_2D;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_pool_2d_back(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * af,
        enum wsp_ggml_op_pool     op,
        int                   k0,
        int                   k1,
        int                   s0,
        int                   s1,
        float                 p0,
        float                 p1) {
    struct wsp_ggml_tensor * result;
    result = wsp_ggml_new_tensor(ctx, WSP_GGML_TYPE_F32, 4, af->ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    wsp_ggml_set_op_params(result, params, sizeof(params));

    result->op     = WSP_GGML_OP_POOL_2D_BACK;
    result->src[0] = a;
    result->src[1] = af;

    return result;
}

// wsp_ggml_upscale

static struct wsp_ggml_tensor * wsp_ggml_upscale_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int                   ne0,
        int                   ne1,
        int                   ne2,
        int                   ne3) {
    WSP_GGML_ASSERT(a->ne[0] <= ne0);
    WSP_GGML_ASSERT(a->ne[1] <= ne1);
    WSP_GGML_ASSERT(a->ne[2] <= ne2);
    WSP_GGML_ASSERT(a->ne[3] <= ne3);

    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    result->op     = WSP_GGML_OP_UPSCALE;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_upscale(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int                   scale_factor) {
    return wsp_ggml_upscale_impl(ctx, a, a->ne[0] * scale_factor, a->ne[1] * scale_factor, a->ne[2], a->ne[3]);
}

struct wsp_ggml_tensor * wsp_ggml_upscale_ext(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int                   ne0,
        int                   ne1,
        int                   ne2,
        int                   ne3) {
    return wsp_ggml_upscale_impl(ctx, a, ne0, ne1, ne2, ne3);
}

// wsp_ggml_pad

struct wsp_ggml_tensor * wsp_ggml_pad(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int                   p0,
        int                   p1,
        int                   p2,
        int                   p3) {
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + p0,
            a->ne[1] + p1,
            a->ne[2] + p2,
            a->ne[3] + p3);

    result->op     = WSP_GGML_OP_PAD;
    result->src[0] = a;

    return result;
}

// wsp_ggml_arange

struct wsp_ggml_tensor * wsp_ggml_arange(
        struct wsp_ggml_context * ctx,
        float                 start,
        float                 stop,
        float                 step) {
    WSP_GGML_ASSERT(stop > start);

    const int64_t steps = (int64_t) ceilf((stop - start) / step);

    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_1d(ctx, WSP_GGML_TYPE_F32, steps);

    wsp_ggml_set_op_params_f32(result, 0, start);
    wsp_ggml_set_op_params_f32(result, 1, stop);
    wsp_ggml_set_op_params_f32(result, 2, step);

    result->op = WSP_GGML_OP_ARANGE;

    return result;
}

// wsp_ggml_timestep_embedding

struct wsp_ggml_tensor * wsp_ggml_timestep_embedding(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * timesteps,
        int                   dim,
        int                   max_period) {
    int actual_dim = dim;
    if (dim % 2 != 0) {
        actual_dim = dim + 1;
    }

    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_2d(ctx, WSP_GGML_TYPE_F32, actual_dim, timesteps->ne[0]);

    wsp_ggml_set_op_params_i32(result, 0, dim);
    wsp_ggml_set_op_params_i32(result, 1, max_period);

    result->op     = WSP_GGML_OP_TIMESTEP_EMBEDDING;
    result->src[0] = timesteps;

    return result;
}

// wsp_ggml_argsort

struct wsp_ggml_tensor * wsp_ggml_argsort(
        struct wsp_ggml_context  * ctx,
        struct wsp_ggml_tensor   * a,
        enum wsp_ggml_sort_order   order) {
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor(ctx, WSP_GGML_TYPE_I32, WSP_GGML_MAX_DIMS, a->ne);

    wsp_ggml_set_op_params_i32(result, 0, (int32_t) order);

    result->op     = WSP_GGML_OP_ARGSORT;
    result->src[0] = a;

    return result;
}

// wsp_ggml_top_k

struct wsp_ggml_tensor * wsp_ggml_top_k(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int                   k) {
    WSP_GGML_ASSERT(a->ne[0] >= k);

    struct wsp_ggml_tensor * result = wsp_ggml_argsort(ctx, a, WSP_GGML_SORT_ORDER_DESC);

    result = wsp_ggml_view_4d(ctx, result,
                k, result->ne[1], result->ne[2], result->ne[3],
                   result->nb[1], result->nb[2], result->nb[3],
                0);

    return result;
}

// wsp_ggml_flash_attn_ext

struct wsp_ggml_tensor * wsp_ggml_flash_attn_ext(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * q,
        struct wsp_ggml_tensor  * k,
        struct wsp_ggml_tensor  * v,
        struct wsp_ggml_tensor  * mask,
        float                 scale,
        float                 max_bias,
        float                 logit_softcap) {
    WSP_GGML_ASSERT(wsp_ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    if (mask) {
        WSP_GGML_ASSERT(wsp_ggml_is_contiguous(mask));
        WSP_GGML_ASSERT(mask->ne[2] == 1);
        WSP_GGML_ASSERT(mask->ne[3] == 1);
        WSP_GGML_ASSERT(mask->ne[1] >= WSP_GGML_PAD(q->ne[1], WSP_GGML_KQ_MASK_PAD) &&
                "the Flash-Attention kernel requires the mask to be padded to WSP_GGML_KQ_MASK_PAD and at least n_queries big");
        //WSP_GGML_ASSERT(wsp_ggml_can_repeat_rows(mask, qk));
    }

    if (max_bias > 0.0f) {
        WSP_GGML_ASSERT(mask);
    }

    bool is_node = false;

    // permute(0, 2, 1, 3)
    int64_t ne[4] = { q->ne[0], q->ne[2], q->ne[1], q->ne[3] };
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor(ctx, WSP_GGML_TYPE_F32, 4, ne);

    float params[] = { scale, max_bias, logit_softcap };
    wsp_ggml_set_op_params(result, params, sizeof(params));

    result->op   = WSP_GGML_OP_FLASH_ATTN_EXT;
    result->grad = is_node ? wsp_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = mask;

    return result;
}

void wsp_ggml_flash_attn_ext_set_prec(
        struct wsp_ggml_tensor * a,
        enum wsp_ggml_prec       prec) {
    WSP_GGML_ASSERT(a->op == WSP_GGML_OP_FLASH_ATTN_EXT);

    const int32_t prec_i32 = (int32_t) prec;

    wsp_ggml_set_op_params_i32(a, 3, prec_i32); // scale is on first pos, max_bias on second
}

// wsp_ggml_flash_attn_back

struct wsp_ggml_tensor * wsp_ggml_flash_attn_back(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * q,
        struct wsp_ggml_tensor  * k,
        struct wsp_ggml_tensor  * v,
        struct wsp_ggml_tensor  * d,
        bool                  masked) {
    WSP_GGML_ABORT("TODO: adapt to wsp_ggml_flash_attn_ext() changes");

    WSP_GGML_ASSERT(wsp_ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    // d shape [D,N,ne2,ne3]
    // q shape [D,N,ne2,ne3]
    // k shape [D,M,kvne2,ne3]
    // v shape [M,D,kvne2,ne3]

    const int64_t     D = q->ne[0];
    const int64_t     N = q->ne[1];
    const int64_t     M = k->ne[1];
    const int64_t   ne2 = q->ne[2];
    const int64_t   ne3 = q->ne[3];
    const int64_t kvne2 = k->ne[2];

    WSP_GGML_ASSERT(k->ne[0] == D);
    WSP_GGML_ASSERT(v->ne[0] == M);
    WSP_GGML_ASSERT(v->ne[1] == D);
    WSP_GGML_ASSERT(d->ne[0] == D);
    WSP_GGML_ASSERT(d->ne[1] == N);
    WSP_GGML_ASSERT(k->ne[2] == kvne2);
    WSP_GGML_ASSERT(k->ne[3] == ne3);
    WSP_GGML_ASSERT(v->ne[2] == kvne2);
    WSP_GGML_ASSERT(v->ne[3] == ne3);
    WSP_GGML_ASSERT(d->ne[2] == ne2);
    WSP_GGML_ASSERT(d->ne[3] == ne3);

    WSP_GGML_ASSERT(ne2 % kvne2 == 0);

    bool is_node = false;

    if (q->grad || k->grad || v->grad) {
        // when using this operation (in backwards pass) these grads are set.
        // we don't want to create (big) grad of our result, so is_node is false.
        is_node = false;
    }

    // store gradients of q, k and v as continuous tensors concatenated in result.
    // note: v and gradv are actually transposed, i.e. v->ne[0] != D.
    const int64_t elem_q = wsp_ggml_nelements(q);
    const int64_t elem_k = wsp_ggml_nelements(k);
    const int64_t elem_v = wsp_ggml_nelements(v);

    enum wsp_ggml_type result_type = WSP_GGML_TYPE_F32;
    WSP_GGML_ASSERT(wsp_ggml_blck_size(result_type) == 1);
    const size_t tsize = wsp_ggml_type_size(result_type);

    const size_t offs_q = 0;
    const size_t offs_k = offs_q + WSP_GGML_PAD(elem_q * tsize, WSP_GGML_MEM_ALIGN);
    const size_t offs_v = offs_k + WSP_GGML_PAD(elem_k * tsize, WSP_GGML_MEM_ALIGN);
    const size_t end    = offs_v + WSP_GGML_PAD(elem_v * tsize, WSP_GGML_MEM_ALIGN);

    const size_t nelements = (end + tsize - 1)/tsize;

    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_1d(ctx, WSP_GGML_TYPE_F32, nelements);

    int32_t masked_i = masked ? 1 : 0;
    wsp_ggml_set_op_params(result, &masked_i, sizeof(masked_i));

    result->op   = WSP_GGML_OP_FLASH_ATTN_BACK;
    result->grad = is_node ? wsp_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = d;

    return result;
}

// wsp_ggml_ssm_conv

struct wsp_ggml_tensor * wsp_ggml_ssm_conv(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * sx,
        struct wsp_ggml_tensor  * c) {
    WSP_GGML_ASSERT(wsp_ggml_is_3d(sx));
    WSP_GGML_ASSERT(wsp_ggml_is_matrix(c));

    const int64_t d_conv  = c->ne[0];
    const int64_t d_inner = c->ne[1];
    const int64_t n_t     = sx->ne[0] - d_conv + 1; // tokens per sequence
    const int64_t n_s     = sx->ne[2];

    // TODO: maybe support other strides than 1?
    WSP_GGML_ASSERT(sx->ne[0] == d_conv - 1 + n_t);
    WSP_GGML_ASSERT(sx->ne[1] == d_inner);
    WSP_GGML_ASSERT(n_t >= 0);

    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_3d(ctx, WSP_GGML_TYPE_F32, d_inner, n_t, n_s);

    result->op     = WSP_GGML_OP_SSM_CONV;
    result->src[0] = sx;
    result->src[1] = c;

    return result;
}

// wsp_ggml_ssm_scan

struct wsp_ggml_tensor * wsp_ggml_ssm_scan(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * s,
        struct wsp_ggml_tensor  * x,
        struct wsp_ggml_tensor  * dt,
        struct wsp_ggml_tensor  * A,
        struct wsp_ggml_tensor  * B,
        struct wsp_ggml_tensor  * C) {
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(s));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(x));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(dt));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(A));
    WSP_GGML_ASSERT(wsp_ggml_is_matrix(A));
    WSP_GGML_ASSERT(wsp_ggml_is_3d(B));
    WSP_GGML_ASSERT(wsp_ggml_is_3d(s));
    WSP_GGML_ASSERT(B->nb[0] == wsp_ggml_type_size(B->type));
    WSP_GGML_ASSERT(C->nb[0] == wsp_ggml_type_size(C->type));
    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(x, dt));
    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(B, C));

    {
        const int64_t d_state      = s->ne[0];
        const int64_t d_inner      = s->ne[1];
        const int64_t n_seq_tokens = x->ne[1];
        const int64_t n_seqs       = x->ne[2];

        WSP_GGML_ASSERT(s->ne[2] == n_seqs);
        WSP_GGML_ASSERT(x->ne[0] == d_inner);
        WSP_GGML_ASSERT(A->ne[0] == d_state);
        WSP_GGML_ASSERT(A->ne[1] == d_inner);
        WSP_GGML_ASSERT(B->ne[0] == d_state);
        WSP_GGML_ASSERT(B->ne[1] == n_seq_tokens);
        WSP_GGML_ASSERT(B->ne[2] == n_seqs);
    }

    // concatenated y + ssm_states
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_1d(ctx, WSP_GGML_TYPE_F32, wsp_ggml_nelements(x) + wsp_ggml_nelements(s));

    result->op   = WSP_GGML_OP_SSM_SCAN;
    result->src[0] = s;
    result->src[1] = x;
    result->src[2] = dt;
    result->src[3] = A;
    result->src[4] = B;
    result->src[5] = C;

    return result;
}

// wsp_ggml_win_part

struct wsp_ggml_tensor * wsp_ggml_win_part(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int                   w) {
    WSP_GGML_ASSERT(a->ne[3] == 1);
    WSP_GGML_ASSERT(a->type  == WSP_GGML_TYPE_F32);

    // padding
    const int px = (w - a->ne[1]%w)%w;
    const int py = (w - a->ne[2]%w)%w;

    const int npx = (px + a->ne[1])/w;
    const int npy = (py + a->ne[2])/w;
    const int np  = npx*npy;

    const int64_t ne[4] = { a->ne[0], w, w, np, };
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor(ctx, WSP_GGML_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, w };
    wsp_ggml_set_op_params(result, params, sizeof(params));

    result->op     = WSP_GGML_OP_WIN_PART;
    result->src[0] = a;

    return result;
}

// wsp_ggml_win_unpart

struct wsp_ggml_tensor * wsp_ggml_win_unpart(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int                   w0,
        int                   h0,
        int                   w) {
    WSP_GGML_ASSERT(a->type == WSP_GGML_TYPE_F32);

    const int64_t ne[4] = { a->ne[0], w0, h0, 1, };
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor(ctx, WSP_GGML_TYPE_F32, 3, ne);

    int32_t params[] = { w };
    wsp_ggml_set_op_params(result, params, sizeof(params));

    result->op     = WSP_GGML_OP_WIN_UNPART;
    result->src[0] = a;

    return result;
}

// wsp_ggml_get_rel_pos

struct wsp_ggml_tensor * wsp_ggml_get_rel_pos(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        int                   qh,
        int                   kh) {
    WSP_GGML_ASSERT(qh == kh);
    WSP_GGML_ASSERT(2*MAX(qh, kh) - 1 == a->ne[1]);

    const int64_t ne[4] = { a->ne[0], kh, qh, 1, };
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor(ctx, WSP_GGML_TYPE_F16, 3, ne);

    result->op     = WSP_GGML_OP_GET_REL_POS;
    result->src[0] = a;

    return result;
}

// wsp_ggml_add_rel_pos

static struct wsp_ggml_tensor * wsp_ggml_add_rel_pos_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * pw,
        struct wsp_ggml_tensor  * ph,
        bool                  inplace) {
    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(pw, ph));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(a));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(pw));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(ph));
    WSP_GGML_ASSERT(ph->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT(pw->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT(pw->ne[3] == a->ne[2]);
    WSP_GGML_ASSERT(pw->ne[0]*pw->ne[0] == a->ne[0]);
    WSP_GGML_ASSERT(pw->ne[1]*pw->ne[2] == a->ne[1]);

    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);
    wsp_ggml_set_op_params_i32(result, 0, inplace ? 1 : 0);

    result->op     = WSP_GGML_OP_ADD_REL_POS;
    result->src[0] = a;
    result->src[1] = pw;
    result->src[2] = ph;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_add_rel_pos(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * pw,
        struct wsp_ggml_tensor  * ph) {
    return wsp_ggml_add_rel_pos_impl(ctx, a, pw, ph, false);
}

struct wsp_ggml_tensor * wsp_ggml_add_rel_pos_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * pw,
        struct wsp_ggml_tensor  * ph) {
    return wsp_ggml_add_rel_pos_impl(ctx, a, pw, ph, true);
}

// wsp_ggml_rwkv_wkv

struct wsp_ggml_tensor * wsp_ggml_rwkv_wkv(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * k,
        struct wsp_ggml_tensor  * v,
        struct wsp_ggml_tensor  * r,
        struct wsp_ggml_tensor  * tf,
        struct wsp_ggml_tensor  * td,
        struct wsp_ggml_tensor  * state) {
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(k));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(v));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(r));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(tf));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(td));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(state));

    const int64_t S = k->ne[0];
    const int64_t H = k->ne[2];
    const int64_t n_tokens = k->ne[3];
    const int64_t n_seqs = state->ne[1];
    {
        WSP_GGML_ASSERT(k->ne[1] == 1);
        WSP_GGML_ASSERT(v->ne[0] == 1 && v->ne[1] == S && v->ne[2] == H && v->ne[3] == n_tokens);
        WSP_GGML_ASSERT(r->ne[0] == 1 && r->ne[1] == S && r->ne[2] == H && r->ne[3] == n_tokens);
        // TODO: RWKV v4 and v5
        WSP_GGML_ASSERT(td->ne[0] == 1 && td->ne[1] == S && td->ne[2] == H && td->ne[3] == n_tokens);
        WSP_GGML_ASSERT(wsp_ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    const int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor(ctx, WSP_GGML_TYPE_F32, 4, ne);

    result->op     = WSP_GGML_OP_RWKV_WKV;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = r;
    result->src[3] = tf;
    result->src[4] = td;
    result->src[5] = state;

    return result;
}

// wsp_ggml_unary

static struct wsp_ggml_tensor * wsp_ggml_unary_impl(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        enum wsp_ggml_unary_op    op,
        bool                  inplace) {
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous_1(a));

    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    wsp_ggml_set_op_params_i32(result, 0, (int32_t) op);

    result->op     = WSP_GGML_OP_UNARY;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_unary(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        enum wsp_ggml_unary_op    op) {
    return wsp_ggml_unary_impl(ctx, a, op, false);
}

struct wsp_ggml_tensor * wsp_ggml_unary_inplace(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        enum wsp_ggml_unary_op    op) {
    return wsp_ggml_unary_impl(ctx, a, op, true);
}

// wsp_ggml_map_unary

static struct wsp_ggml_tensor * wsp_ggml_map_unary_impl_f32(
        struct wsp_ggml_context        * ctx,
        struct wsp_ggml_tensor         * a,
        const  wsp_ggml_unary_op_f32_t   fun,
        bool                         inplace) {
    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    wsp_ggml_set_op_params(result, (const void *) &fun, sizeof(fun));

    result->op     = WSP_GGML_OP_MAP_UNARY;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_map_unary_f32(
        struct wsp_ggml_context        * ctx,
        struct wsp_ggml_tensor         * a,
        const  wsp_ggml_unary_op_f32_t   fun) {
    return wsp_ggml_map_unary_impl_f32(ctx, a, fun, false);
}

struct wsp_ggml_tensor * wsp_ggml_map_unary_inplace_f32(
        struct wsp_ggml_context        * ctx,
        struct wsp_ggml_tensor         * a,
        const  wsp_ggml_unary_op_f32_t   fun) {
    return wsp_ggml_map_unary_impl_f32(ctx, a, fun, true);
}

// wsp_ggml_map_binary

static struct wsp_ggml_tensor * wsp_ggml_map_binary_impl_f32(
        struct wsp_ggml_context         * ctx,
        struct wsp_ggml_tensor          * a,
        struct wsp_ggml_tensor          * b,
        const  wsp_ggml_binary_op_f32_t   fun,
        bool                          inplace) {
    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(a, b));

    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    wsp_ggml_set_op_params(result, (const void *) &fun, sizeof(fun));

    result->op     = WSP_GGML_OP_MAP_BINARY;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_map_binary_f32(
        struct wsp_ggml_context         * ctx,
        struct wsp_ggml_tensor          * a,
        struct wsp_ggml_tensor          * b,
        const  wsp_ggml_binary_op_f32_t   fun) {
    return wsp_ggml_map_binary_impl_f32(ctx, a, b, fun, false);
}

struct wsp_ggml_tensor * wsp_ggml_map_binary_inplace_f32(
        struct wsp_ggml_context         * ctx,
        struct wsp_ggml_tensor          * a,
        struct wsp_ggml_tensor          * b,
        const  wsp_ggml_binary_op_f32_t   fun) {
    return wsp_ggml_map_binary_impl_f32(ctx, a, b, fun, true);
}

// wsp_ggml_map_custom1_f32

static struct wsp_ggml_tensor * wsp_ggml_map_custom1_impl_f32(
        struct wsp_ggml_context          * ctx,
        struct wsp_ggml_tensor           * a,
        const  wsp_ggml_custom1_op_f32_t   fun,
        bool                           inplace) {
    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    wsp_ggml_set_op_params(result, (const void *) &fun, sizeof(fun));

    result->op     = WSP_GGML_OP_MAP_CUSTOM1_F32;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_map_custom1_f32(
        struct wsp_ggml_context          * ctx,
        struct wsp_ggml_tensor           * a,
        const  wsp_ggml_custom1_op_f32_t   fun) {
    return wsp_ggml_map_custom1_impl_f32(ctx, a, fun, false);
}

struct wsp_ggml_tensor * wsp_ggml_map_custom1_inplace_f32(
        struct wsp_ggml_context          * ctx,
        struct wsp_ggml_tensor           * a,
        const  wsp_ggml_custom1_op_f32_t   fun) {
    return wsp_ggml_map_custom1_impl_f32(ctx, a, fun, true);
}

// wsp_ggml_map_custom2_f32

static struct wsp_ggml_tensor * wsp_ggml_map_custom2_impl_f32(
        struct wsp_ggml_context          * ctx,
        struct wsp_ggml_tensor           * a,
        struct wsp_ggml_tensor           * b,
        const  wsp_ggml_custom2_op_f32_t   fun,
        bool                           inplace) {
    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    wsp_ggml_set_op_params(result, (const void *) &fun, sizeof(fun));

    result->op     = WSP_GGML_OP_MAP_CUSTOM2_F32;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_map_custom2_f32(
        struct wsp_ggml_context          * ctx,
        struct wsp_ggml_tensor           * a,
        struct wsp_ggml_tensor           * b,
        const  wsp_ggml_custom2_op_f32_t   fun) {
    return wsp_ggml_map_custom2_impl_f32(ctx, a, b, fun, false);
}

struct wsp_ggml_tensor * wsp_ggml_map_custom2_inplace_f32(
        struct wsp_ggml_context          * ctx,
        struct wsp_ggml_tensor           * a,
        struct wsp_ggml_tensor           * b,
        const  wsp_ggml_custom2_op_f32_t   fun) {
    return wsp_ggml_map_custom2_impl_f32(ctx, a, b, fun, true);
}

// wsp_ggml_map_custom3_f32

static struct wsp_ggml_tensor * wsp_ggml_map_custom3_impl_f32(
        struct wsp_ggml_context          * ctx,
        struct wsp_ggml_tensor           * a,
        struct wsp_ggml_tensor           * b,
        struct wsp_ggml_tensor           * c,
        const  wsp_ggml_custom3_op_f32_t   fun,
        bool                           inplace) {
    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    wsp_ggml_set_op_params(result, (const void *) &fun, sizeof(fun));

    result->op     = WSP_GGML_OP_MAP_CUSTOM3_F32;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_map_custom3_f32(
        struct wsp_ggml_context          * ctx,
        struct wsp_ggml_tensor           * a,
        struct wsp_ggml_tensor           * b,
        struct wsp_ggml_tensor           * c,
        const  wsp_ggml_custom3_op_f32_t   fun) {
    return wsp_ggml_map_custom3_impl_f32(ctx, a, b, c, fun, false);
}

struct wsp_ggml_tensor * wsp_ggml_map_custom3_inplace_f32(
        struct wsp_ggml_context          * ctx,
        struct wsp_ggml_tensor           * a,
        struct wsp_ggml_tensor           * b,
        struct wsp_ggml_tensor           * c,
        const  wsp_ggml_custom3_op_f32_t   fun) {
    return wsp_ggml_map_custom3_impl_f32(ctx, a, b, c, fun, true);
}

// wsp_ggml_map_custom1
struct wsp_ggml_map_custom1_op_params {
    wsp_ggml_custom1_op_t  fun;
    int                n_tasks;
    void             * userdata;
};

static struct wsp_ggml_tensor * wsp_ggml_map_custom1_impl(
        struct wsp_ggml_context      * ctx,
        struct wsp_ggml_tensor       * a,
        const  wsp_ggml_custom1_op_t   fun,
        int                        n_tasks,
        void                     * userdata,
        bool                       inplace) {
    WSP_GGML_ASSERT(n_tasks == WSP_GGML_N_TASKS_MAX || n_tasks > 0);

    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    struct wsp_ggml_map_custom1_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    wsp_ggml_set_op_params(result, (const void *) &params, sizeof(params));

    result->op     = WSP_GGML_OP_MAP_CUSTOM1;
    result->src[0] = a;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_map_custom1(
        struct wsp_ggml_context      * ctx,
        struct wsp_ggml_tensor       * a,
        const  wsp_ggml_custom1_op_t   fun,
        int                        n_tasks,
        void                     * userdata) {
    return wsp_ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, false);
}

struct wsp_ggml_tensor * wsp_ggml_map_custom1_inplace(
        struct wsp_ggml_context      * ctx,
        struct wsp_ggml_tensor       * a,
        const  wsp_ggml_custom1_op_t   fun,
        int                        n_tasks,
        void                     * userdata) {
    return wsp_ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, true);
}

// wsp_ggml_map_custom2

struct wsp_ggml_map_custom2_op_params {
    wsp_ggml_custom2_op_t   fun;
    int                 n_tasks;
    void              * userdata;
};

static struct wsp_ggml_tensor * wsp_ggml_map_custom2_impl(
        struct wsp_ggml_context      * ctx,
        struct wsp_ggml_tensor       * a,
        struct wsp_ggml_tensor       * b,
        const  wsp_ggml_custom2_op_t   fun,
        int                        n_tasks,
        void                     * userdata,
        bool                       inplace) {
    WSP_GGML_ASSERT(n_tasks == WSP_GGML_N_TASKS_MAX || n_tasks > 0);

    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    struct wsp_ggml_map_custom2_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    wsp_ggml_set_op_params(result, (const void *) &params, sizeof(params));

    result->op     = WSP_GGML_OP_MAP_CUSTOM2;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_map_custom2(
        struct wsp_ggml_context      * ctx,
        struct wsp_ggml_tensor       * a,
        struct wsp_ggml_tensor       * b,
        const  wsp_ggml_custom2_op_t   fun,
        int                        n_tasks,
        void                     * userdata) {
    return wsp_ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, false);
}

struct wsp_ggml_tensor * wsp_ggml_map_custom2_inplace(
        struct wsp_ggml_context      * ctx,
        struct wsp_ggml_tensor       * a,
        struct wsp_ggml_tensor       * b,
        const  wsp_ggml_custom2_op_t   fun,
        int                        n_tasks,
        void                     * userdata) {
    return wsp_ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, true);
}

// wsp_ggml_map_custom3

struct wsp_ggml_map_custom3_op_params {
    wsp_ggml_custom3_op_t fun;
    int n_tasks;
    void * userdata;
};

static struct wsp_ggml_tensor * wsp_ggml_map_custom3_impl(
        struct wsp_ggml_context      * ctx,
        struct wsp_ggml_tensor       * a,
        struct wsp_ggml_tensor       * b,
        struct wsp_ggml_tensor       * c,
        const  wsp_ggml_custom3_op_t   fun,
        int                        n_tasks,
        void                     * userdata,
        bool                       inplace) {
    WSP_GGML_ASSERT(n_tasks == WSP_GGML_N_TASKS_MAX || n_tasks > 0);

    struct wsp_ggml_tensor * result = inplace ? wsp_ggml_view_tensor(ctx, a) : wsp_ggml_dup_tensor(ctx, a);

    struct wsp_ggml_map_custom3_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    wsp_ggml_set_op_params(result, (const void *) &params, sizeof(params));

    result->op     = WSP_GGML_OP_MAP_CUSTOM3;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    return result;
}

struct wsp_ggml_tensor * wsp_ggml_map_custom3(
        struct wsp_ggml_context      * ctx,
        struct wsp_ggml_tensor       * a,
        struct wsp_ggml_tensor       * b,
        struct wsp_ggml_tensor       * c,
        const  wsp_ggml_custom3_op_t   fun,
        int                        n_tasks,
        void                     * userdata) {
    return wsp_ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, false);
}

struct wsp_ggml_tensor * wsp_ggml_map_custom3_inplace(
        struct wsp_ggml_context      * ctx,
        struct wsp_ggml_tensor       * a,
        struct wsp_ggml_tensor       * b,
        struct wsp_ggml_tensor       * c,
        const  wsp_ggml_custom3_op_t   fun,
        int                        n_tasks,
        void                     * userdata) {
    return wsp_ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, true);
}

// wsp_ggml_cross_entropy_loss

struct wsp_ggml_tensor * wsp_ggml_cross_entropy_loss(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b) {
    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(a, b));

    struct wsp_ggml_tensor * result = wsp_ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = WSP_GGML_OP_CROSS_ENTROPY_LOSS;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// wsp_ggml_cross_entropy_loss_back

struct wsp_ggml_tensor * wsp_ggml_cross_entropy_loss_back(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * b,
        struct wsp_ggml_tensor  * c) {
    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(a, b));
    WSP_GGML_ASSERT(wsp_ggml_is_scalar(c));

    struct wsp_ggml_tensor * result = wsp_ggml_dup_tensor(ctx, a);

    result->op     = WSP_GGML_OP_CROSS_ENTROPY_LOSS_BACK;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    return result;
}

// opt_step_adamw

struct wsp_ggml_tensor * wsp_ggml_opt_step_adamw(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_tensor  * a,
        struct wsp_ggml_tensor  * grad,
        float                 alpha,
        float                 beta1,
        float                 beta2,
        float                 eps,
        float                 wd) {
    WSP_GGML_ASSERT(a->flags & WSP_GGML_TENSOR_FLAG_PARAM);
    WSP_GGML_ASSERT(wsp_ggml_are_same_shape(a, grad));
    WSP_GGML_ASSERT(alpha >  0.0f);
    WSP_GGML_ASSERT(beta1 >= 0.0f && beta1 <= 1.0f);
    WSP_GGML_ASSERT(beta2 >= 0.0f && beta2 <= 1.0f);
    WSP_GGML_ASSERT(eps   >= 0.0f);
    WSP_GGML_ASSERT(wd    >= 0.0f && wd    <= 1.0f);

    struct wsp_ggml_tensor * result = wsp_ggml_view_tensor(ctx, a);

    const int64_t iter = 1;
    memcpy(&result->op_params[0], &iter, sizeof(int64_t));
    wsp_ggml_set_op_params_f32(result, 2, alpha);
    wsp_ggml_set_op_params_f32(result, 3, beta1);
    wsp_ggml_set_op_params_f32(result, 4, beta2);
    wsp_ggml_set_op_params_f32(result, 5, eps);
    wsp_ggml_set_op_params_f32(result, 6, wd);

    result->op     = WSP_GGML_OP_OPT_STEP_ADAMW;
    result->src[0] = a;
    result->src[1] = grad;
    result->src[2] = wsp_ggml_dup_tensor(ctx, grad);
    result->src[3] = wsp_ggml_dup_tensor(ctx, grad);

    return result;
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
            } else if (type_traits[dst->type].from_float) {
                wsp_ggml_from_float_t const wsp_quantize_row_q = type_traits[dst->type].from_float;
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
            } else if (type_traits[dst->type].from_float) {
                wsp_ggml_from_float_t const wsp_quantize_row_q = type_traits[dst->type].from_float;
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
            } else if (type_traits[dst->type].from_float) {
                wsp_ggml_from_float_t const wsp_quantize_row_q = type_traits[dst->type].from_float;

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
    wsp_ggml_to_float_t const wsp_dewsp_quantize_row_q = type_traits[type].to_float;
    wsp_ggml_from_float_t const wsp_quantize_row_q = type_traits[dtype].from_float;

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
    wsp_ggml_to_float_t const wsp_dewsp_quantize_row_q = type_traits[type].to_float;
    wsp_ggml_from_float_t const wsp_quantize_row_q = type_traits[type].from_float;

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

    wsp_ggml_vec_dot_t const vec_dot      = type_traits[type].vec_dot;
    enum wsp_ggml_type const vec_dot_type = type_traits[type].vec_dot_type;

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

    enum wsp_ggml_type           const vec_dot_type         = type_traits[type].vec_dot_type;
    wsp_ggml_from_float_t        const from_float           = type_traits[vec_dot_type].from_float;
    wsp_ggml_from_float_to_mat_t const from_float_to_mat    = type_traits[vec_dot_type].from_float_to_mat;
    int64_t                  const vec_dot_num_rows     = type_traits[type].nrows;
    int64_t                  const matmul_num_cols      = type_traits[type].ncols;
    int64_t                  const blck_size_interleave = type_traits[type].blck_size_interleave;
    wsp_ggml_gemv_t              const gemv                 = type_traits[type].gemv;
    wsp_ggml_gemm_t              const gemm                 = type_traits[type].gemm;

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

    wsp_ggml_vec_dot_t    const vec_dot         = type_traits[type].vec_dot;
    enum wsp_ggml_type    const vec_dot_type    = type_traits[type].vec_dot_type;
    wsp_ggml_from_float_t const from_float      = type_traits[vec_dot_type].from_float;
    int64_t           const matmul_num_cols = type_traits[type].ncols;
    wsp_ggml_gemv_t       const gemv            = type_traits[type].gemv;

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
    wsp_ggml_to_float_t const wsp_dewsp_quantize_row_q = type_traits[type].to_float;

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
    wsp_ggml_to_float_t const wsp_dewsp_quantize_row_q = type_traits[type].to_float;

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

    enum wsp_ggml_type    const k_vec_dot_type = type_traits[k->type].vec_dot_type;
    wsp_ggml_from_float_t const q_to_vec_dot   = type_traits[k_vec_dot_type].from_float;
    wsp_ggml_vec_dot_t    const kq_vec_dot     = type_traits[k->type].vec_dot;
    wsp_ggml_to_float_t   const v_to_float     = type_traits[v->type].to_float;

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

// wsp_ggml_compute_forward_rwkv_wkv

static void wsp_ggml_compute_forward_rwkv_wkv_f32(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {
    const size_t T = dst->src[1]->ne[3];
    const size_t C = dst->ne[0];
    const size_t H = dst->src[1]->ne[2];
    const size_t n_seqs = dst->src[5]->ne[1];

    float * dst_data = (float *) dst->data;
    float * state = ((float *) dst->data) + C * T;

    if (params->ith != 0) {
        return;
    }

    memset(dst_data, 0, T * C * sizeof(float));

    float * k =          (float *) dst->src[0]->data;
    float * v =          (float *) dst->src[1]->data;
    float * r =          (float *) dst->src[2]->data;
    float * time_faaaa = (float *) dst->src[3]->data;
    float * time_decay = (float *) dst->src[4]->data;

    size_t t_stride = H * (C / H);

    size_t h_stride = C / H;
    size_t h_stride_2d = (C / H) * (C / H);

    // basically fused operations:
    // dst = r @ (time_faaaa * (k @ v) + state),
    // state = time_decay * state + (k @ v),
    // recursive through each token
    for (size_t t = 0; t < T; t++) {
        size_t t_offset = t * t_stride;
        size_t state_offset = (C / H) * C * (t / (T / n_seqs));
        float * state_cur = state + state_offset;
        float * state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[5]->data + state_offset;

        for (size_t h = 0; h < H; h++) {
            size_t h_offset = h * h_stride;
            size_t t_h_offset = t_offset + h_offset;
            size_t h_2d_offset = h * h_stride_2d;

            for (size_t i = 0; i < C / H; i++) {
                size_t t_h_i_offset = t_h_offset + i;
                size_t h_i_offset = h_offset + i;
                size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                float k_val = k[t_h_i_offset];
                float r_val = r[t_h_i_offset];
                float time_faaaa_val = time_faaaa[h_i_offset];
                // RWKV v6: different time_decay for each token.
                float time_decay_val = time_decay[t_h_i_offset];

                for (size_t j = 0; j < C / H; j ++) {
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
}

static void wsp_ggml_compute_forward_rwkv_wkv(
        const struct wsp_ggml_compute_params * params,
        struct wsp_ggml_tensor * dst) {

    const struct wsp_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case WSP_GGML_TYPE_F32:
            {
                wsp_ggml_compute_forward_rwkv_wkv_f32(params, dst);
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
        case WSP_GGML_OP_RWKV_WKV:
            {
                wsp_ggml_compute_forward_rwkv_wkv(params, tensor);
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

////////////////////////////////////////////////////////////////////////////////

struct wsp_ggml_hash_set wsp_ggml_hash_set_new(size_t size) {
    size = wsp_ggml_hash_size(size);
    struct wsp_ggml_hash_set result;
    result.size = size;
    result.keys = WSP_GGML_MALLOC(sizeof(struct wsp_ggml_tensor *) * size);
    result.used = WSP_GGML_CALLOC(wsp_ggml_bitset_size(size), sizeof(wsp_ggml_bitset_t));
    return result;
}

void wsp_ggml_hash_set_reset(struct wsp_ggml_hash_set * hash_set) {
    memset(hash_set->used, 0, sizeof(wsp_ggml_bitset_t) * wsp_ggml_bitset_size(hash_set->size));
}

void wsp_ggml_hash_set_free(struct wsp_ggml_hash_set * hash_set) {
    WSP_GGML_FREE(hash_set->used);
    WSP_GGML_FREE(hash_set->keys);
}

size_t wsp_ggml_hash_size(size_t min_sz) {
    // next primes after powers of two
    static const size_t primes[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    static const size_t n_primes = sizeof(primes)/sizeof(primes[0]);

    // find the smallest prime that is larger or equal than min_sz
    size_t l = 0;
    size_t r = n_primes;
    while (l < r) {
        size_t m = (l + r)/2;
        if (primes[m] < min_sz) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    size_t sz = l < n_primes ? primes[l] : min_sz | 1;
    return sz;
}

struct hash_map {
    struct wsp_ggml_hash_set set;
    struct wsp_ggml_tensor ** vals;
};

static struct hash_map * wsp_ggml_new_hash_map(size_t size) {
    struct hash_map * result = WSP_GGML_MALLOC(sizeof(struct hash_map));
    result->set = wsp_ggml_hash_set_new(size);
    result->vals = WSP_GGML_CALLOC(result->set.size, sizeof(struct wsp_ggml_tensor *));
    return result;
}

static void wsp_ggml_hash_map_free(struct hash_map * map) {
    wsp_ggml_hash_set_free(&map->set);
    WSP_GGML_FREE(map->vals);
    WSP_GGML_FREE(map);
}

// gradient checkpointing

static struct wsp_ggml_tensor * wsp_ggml_recompute_graph_node(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_cgraph  * graph,
        struct hash_map     * replacements,
        struct wsp_ggml_tensor  * node) {

    if (node == NULL) {
        return NULL;
    }

    if (node->flags & WSP_GGML_TENSOR_FLAG_PARAM) {
        return node;
    }

    if (!wsp_ggml_hash_contains(&graph->visited_hash_set, node)) {
        return node;
    }

    int count_children = 0;
    for (int k = 0; k < WSP_GGML_MAX_SRC; ++k) {
        if (node->src[k]) {
            ++count_children;
        }
    }

    if (count_children == 0) {
        return node;
    }

    size_t i = wsp_ggml_hash_find(&replacements->set, node);
    WSP_GGML_ASSERT(i != WSP_GGML_HASHSET_FULL); // assert that not full
    if (replacements->set.keys[i] == node) {
        return replacements->vals[i];
    }

    struct wsp_ggml_tensor * clone = wsp_ggml_new_tensor(ctx, node->type, WSP_GGML_MAX_DIMS, node->ne);

    // insert clone into replacements
    WSP_GGML_ASSERT(replacements->set.keys[i] == NULL); // assert that we don't overwrite
    replacements->set.keys[i] = node;
    replacements->vals[i] = clone;

    clone->op       = node->op;
    clone->grad     = node->grad;
    clone->flags    = node->flags;
    clone->extra    = node->extra;
    for (int k = 0; k < WSP_GGML_MAX_DIMS; ++k) {
        clone->nb[k] = node->nb[k];
    }
    for (int k = 0; k < WSP_GGML_MAX_SRC; ++k) {
        clone->src[k] = wsp_ggml_recompute_graph_node(ctx, graph, replacements, node->src[k]);
    }
    if (node->view_src != NULL) {
        clone->data = (node->view_src->data == NULL)
                        ? NULL // view_src not yet allocated
                        : (char *) node->view_src->data // view_src already allocated
                                 + node->view_offs;
        clone->view_src  = node->view_src;
        clone->view_offs = node->view_offs;
    }

    WSP_GGML_ASSERT(sizeof(node->op_params) == sizeof(int32_t) * (WSP_GGML_MAX_OP_PARAMS / sizeof(int32_t)));
    WSP_GGML_ASSERT(sizeof(node->name)      == WSP_GGML_MAX_NAME);
    memcpy(clone->op_params, node->op_params, sizeof(node->op_params));
    wsp_ggml_format_name(clone, "%s (clone)", wsp_ggml_get_name(node));

    return clone;
}

void wsp_ggml_build_backward_gradient_checkpointing(
        struct wsp_ggml_context   * ctx,
        struct wsp_ggml_cgraph    * gf,
        struct wsp_ggml_cgraph    * gb,
        struct wsp_ggml_cgraph    * gb_tmp,
        struct wsp_ggml_tensor  * * checkpoints,
        int                     n_checkpoints) {
    wsp_ggml_graph_cpy(gf, gb_tmp);
    wsp_ggml_build_backward_expand(ctx, gf, gb_tmp, false);

    if (n_checkpoints <= 0) {
        wsp_ggml_graph_cpy(gb_tmp, gb);
        return;
    }

    struct hash_map * replacements = wsp_ggml_new_hash_map(gf->n_nodes + gf->n_leafs + n_checkpoints);

    // insert checkpoints in replacements
    for (int i = 0; i < n_checkpoints; ++i) {
        size_t k = wsp_ggml_hash_find(&replacements->set, checkpoints[i]);
        WSP_GGML_ASSERT(k != WSP_GGML_HASHSET_FULL); // assert that not full
        WSP_GGML_ASSERT(replacements->set.keys[k] == NULL); // assert that we don't overwrite
        replacements->set.keys[k] = checkpoints[i];
        replacements->vals[k]     = checkpoints[i];
    }

    wsp_ggml_graph_cpy(gf, gb);
    // rewrite gb_tmp->nodes[gf->n_nodes:gb_tmp->n_nodes],
    // replacing references to gb_tmp->nodes[0:gf->n_nodes] ( == gf->nodes[0:gf->n_nodes]),
    // by recomputing them from checkpoints
    for (int i = gf->n_nodes; i<gb_tmp->n_nodes; ++i) {
        struct wsp_ggml_tensor * node = gb_tmp->nodes[i];
        for (int k = 0; k < WSP_GGML_MAX_SRC; ++k) {
            // insert new tensors recomputing src, reusing already made replacements,
            // remember replacements: remember new tensors with mapping from corresponding gf nodes
            // recurse for input tensors,
            // unless (i.e. terminating when) input tensors are replacements (like checkpoints)
            node->src[k] = wsp_ggml_recompute_graph_node(ctx, gf, replacements, node->src[k]);
        }
        // insert rewritten backward node with replacements made into resulting backward graph gb
        wsp_ggml_build_forward_expand(gb, node);
    }

    wsp_ggml_hash_map_free(replacements);
}

// utility functions to change gradients
// if a is in acc_table, modify gradients in-place and mark result as gradient accumulator
// else if a is in zero_table, replace a
// else, just add/subtract/etc. the gradients

static struct wsp_ggml_tensor * wsp_ggml_add_or_set(
        struct wsp_ggml_context  * ctx,
        struct wsp_ggml_tensor   * a,
        struct wsp_ggml_tensor   * b,
        struct wsp_ggml_hash_set * zero_table,
        struct wsp_ggml_hash_set * acc_table) {
    if (wsp_ggml_hash_contains(acc_table, a)) {
        struct wsp_ggml_tensor * ret = wsp_ggml_add_impl(ctx, a, b, true);
        const size_t insert_result = wsp_ggml_hash_insert(acc_table, ret);
        WSP_GGML_ASSERT(insert_result != WSP_GGML_HASHSET_FULL);
        WSP_GGML_ASSERT(insert_result != WSP_GGML_HASHSET_ALREADY_EXISTS);
        return ret;
    }
    if (wsp_ggml_hash_contains(zero_table, a)) {
        return b;
    }
    return wsp_ggml_add_impl(ctx, a, b, false);
}

static struct wsp_ggml_tensor * wsp_ggml_acc_or_set(
        struct wsp_ggml_context  * ctx,
        struct wsp_ggml_tensor   * a,
        struct wsp_ggml_tensor   * b,
        const  size_t          nb1,
        const  size_t          nb2,
        const  size_t          nb3,
        const  size_t          offset,
        struct wsp_ggml_hash_set * zero_table,
        struct wsp_ggml_hash_set * acc_table) {
    if (wsp_ggml_hash_contains(acc_table, a)) {
        struct wsp_ggml_tensor * ret = wsp_ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
        const size_t insert_result = wsp_ggml_hash_insert(acc_table, ret);
        WSP_GGML_ASSERT(insert_result != WSP_GGML_HASHSET_FULL);
        WSP_GGML_ASSERT(insert_result != WSP_GGML_HASHSET_ALREADY_EXISTS);
        return ret;
    }
    if (wsp_ggml_hash_contains(zero_table, a)) {
        struct wsp_ggml_tensor * a_zero = wsp_ggml_scale(ctx, a, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
        return wsp_ggml_acc_impl(ctx, a_zero, b, nb1, nb2, nb3, offset, false);
    }
    return wsp_ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

static struct wsp_ggml_tensor * wsp_ggml_add1_or_set(
        struct wsp_ggml_context  * ctx,
        struct wsp_ggml_tensor   * a,
        struct wsp_ggml_tensor   * b,
        struct wsp_ggml_hash_set * zero_table,
        struct wsp_ggml_hash_set * acc_table) {
    if (wsp_ggml_hash_contains(acc_table, a)) {
        struct wsp_ggml_tensor * ret = wsp_ggml_add1_impl(ctx, a, b, true);
        const size_t insert_result = wsp_ggml_hash_insert(acc_table, ret);
        WSP_GGML_ASSERT(insert_result != WSP_GGML_HASHSET_FULL);
        WSP_GGML_ASSERT(insert_result != WSP_GGML_HASHSET_ALREADY_EXISTS);
        return ret;
    }
    if (wsp_ggml_hash_contains(zero_table, a)) {
        return wsp_ggml_repeat(ctx, b, a);
    }
    return wsp_ggml_add1_impl(ctx, a, b, false);
}

static struct wsp_ggml_tensor * wsp_ggml_sub_or_set(
        struct wsp_ggml_context  * ctx,
        struct wsp_ggml_tensor   * a,
        struct wsp_ggml_tensor   * b,
        struct wsp_ggml_hash_set * zero_table,
        struct wsp_ggml_hash_set * acc_table) {
    if (wsp_ggml_hash_contains(acc_table, a)) {
        struct wsp_ggml_tensor * ret = wsp_ggml_sub_impl(ctx, a, b, true);
        const size_t insert_result = wsp_ggml_hash_insert(acc_table, ret);
        WSP_GGML_ASSERT(insert_result != WSP_GGML_HASHSET_FULL);
        WSP_GGML_ASSERT(insert_result != WSP_GGML_HASHSET_ALREADY_EXISTS);
        return ret;
    }
    if (wsp_ggml_hash_contains(zero_table, a)) {
        return wsp_ggml_neg(ctx, b);
    }
    return wsp_ggml_sub_impl(ctx, a, b, false);
}

static void wsp_ggml_compute_backward(struct wsp_ggml_context * ctx, struct wsp_ggml_tensor * tensor, struct wsp_ggml_hash_set * zero_table, struct wsp_ggml_hash_set * acc_table) {
    struct wsp_ggml_tensor * src0 = tensor->src[0];
    struct wsp_ggml_tensor * src1 = tensor->src[1];
    struct wsp_ggml_tensor * src2 = tensor->src[2];

    switch (tensor->op) {
        case WSP_GGML_OP_DUP:
            {
                if (src0->grad) {
                    src0->grad = wsp_ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_ADD:
            {
                if (src0->grad) {
                    src0->grad = wsp_ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table, acc_table);
                }
                if (src1->grad) {
                    if (wsp_ggml_are_same_shape(src0, src1)) {
                        src1->grad = wsp_ggml_add_or_set(ctx, src1->grad,                       tensor->grad,        zero_table, acc_table);
                    } else {
                        src1->grad = wsp_ggml_add_or_set(ctx, src1->grad, wsp_ggml_repeat_back(ctx, tensor->grad, src1), zero_table, acc_table);
                    }
                }
            } break;
        case WSP_GGML_OP_ADD1:
            {
                if (src0->grad) {
                    src0->grad = wsp_ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table, acc_table);
                }
                if (src1->grad) {
                    src1->grad = wsp_ggml_add_or_set(ctx,
                        src1->grad,
                        wsp_ggml_mean(ctx, tensor->grad), // TODO: should probably be sum instead of mean
                        zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_ACC:
            {
                if (src0->grad) {
                    src0->grad = wsp_ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table, acc_table);
                }
                if (src1->grad) {
                    const size_t nb1     = ((int32_t *) tensor->op_params)[0];
                    const size_t nb2     = ((int32_t *) tensor->op_params)[1];
                    const size_t nb3     = ((int32_t *) tensor->op_params)[2];
                    const size_t offset  = ((int32_t *) tensor->op_params)[3];

                    struct wsp_ggml_tensor * tensor_grad_view = wsp_ggml_view_4d(ctx,
                        tensor->grad,
                        src1->grad->ne[0],
                        src1->grad->ne[1],
                        src1->grad->ne[2],
                        src1->grad->ne[3],
                        nb1, nb2, nb3, offset);

                    src1->grad =
                        wsp_ggml_add_or_set(ctx,
                            src1->grad,
                            wsp_ggml_reshape(ctx,
                                wsp_ggml_cont(ctx, tensor_grad_view),
                                src1->grad),
                            zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_SUB:
            {
                if (src0->grad) {
                    src0->grad = wsp_ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table, acc_table);
                }
                if (src1->grad) {
                    src1->grad = wsp_ggml_sub_or_set(ctx, src1->grad, tensor->grad, zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_MUL:
            {
                if (src0->grad) {
                    src0->grad =
                        wsp_ggml_add_or_set(ctx,
                                src0->grad,
                                wsp_ggml_mul(ctx, src1, tensor->grad),
                                zero_table, acc_table);
                }
                if (src1->grad) {
                    src1->grad =
                        wsp_ggml_add_or_set(ctx,
                                src1->grad,
                                wsp_ggml_mul(ctx, src0, tensor->grad),
                                zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_DIV:
            {
                if (src0->grad) {
                    src0->grad =
                        wsp_ggml_add_or_set(ctx,
                                src0->grad,
                                wsp_ggml_div(ctx, tensor->grad, src1),
                                zero_table, acc_table);
                }
                if (src1->grad) {
                    src1->grad =
                        wsp_ggml_sub_or_set(ctx,
                                src1->grad,
                                wsp_ggml_mul(ctx,
                                    tensor->grad,
                                    wsp_ggml_div(ctx, tensor, src1)),
                                zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_SQR:
            {
                if (src0->grad) {
                    src0->grad =
                        wsp_ggml_add_or_set(ctx,
                                src0->grad,
                                wsp_ggml_scale(ctx,
                                    wsp_ggml_mul(ctx, src0, tensor->grad),
                                    2.0f),
                                zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_SQRT:
            {
                if (src0->grad) {
                    src0->grad =
                        wsp_ggml_add_or_set(ctx,
                                src0->grad,
                                wsp_ggml_scale(ctx,
                                    wsp_ggml_div(ctx,
                                        tensor->grad,
                                        tensor),
                                    0.5f),
                                zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_LOG:
            {
                if (src0->grad) {
                    src0->grad =
                        wsp_ggml_add_or_set(ctx,
                                src0->grad,
                                wsp_ggml_div(ctx,
                                    tensor->grad,
                                    src0),
                                zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_SIN:
            {
                if (src0->grad) {
                    src0->grad =
                        wsp_ggml_add_or_set(ctx,
                                src0->grad,
                                wsp_ggml_mul(ctx,
                                    tensor->grad,
                                    wsp_ggml_cos(ctx, src0)),
                                zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_COS:
            {
                if (src0->grad) {
                    src0->grad =
                        wsp_ggml_sub_or_set(ctx,
                                src0->grad,
                                wsp_ggml_mul(ctx,
                                    tensor->grad,
                                    wsp_ggml_sin(ctx, src0)),
                                zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_SUM:
            {
                if (src0->grad) {
                    src0->grad =
                        wsp_ggml_add1_or_set(ctx,
                                src0->grad,
                                tensor->grad,
                                zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_SUM_ROWS:
            {
                if (src0->grad) {
                    src0->grad =
                        wsp_ggml_add_or_set(ctx,
                                src0->grad,
                                wsp_ggml_repeat(ctx,
                                    tensor->grad,
                                    src0->grad),
                                zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_MEAN:
        case WSP_GGML_OP_ARGMAX:
        case WSP_GGML_OP_COUNT_EQUAL:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: implement
            }
        case WSP_GGML_OP_REPEAT:
            {
                // necessary for llama
                if (src0->grad) {
                    src0->grad = wsp_ggml_add_or_set(ctx,
                            src0->grad,
                            wsp_ggml_repeat_back(ctx, tensor->grad, src0->grad),
                            zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_REPEAT_BACK:
            {
                if (src0->grad) {
                    // TODO: test this
                    src0->grad = wsp_ggml_add_or_set(ctx,
                            src0->grad,
                            wsp_ggml_repeat(ctx, tensor->grad, src0->grad),
                            zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_CONCAT:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: implement
            }
        case WSP_GGML_OP_SILU_BACK:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_NORM:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_RMS_NORM:
            {
                // necessary for llama
                if (src0->grad) {
                    float eps;
                    memcpy(&eps, tensor->op_params, sizeof(float));

                    src0->grad = wsp_ggml_add_or_set(ctx,
                            src0->grad,
                            wsp_ggml_rms_norm_back(ctx, src0, tensor->grad, eps),
                            zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_RMS_NORM_BACK:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_GROUP_NORM:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_MUL_MAT:
            {
                // https://cs231n.github.io/optimization-2/#staged
                // # forward pass
                // s0 = np.random.randn(5, 10)
                // s1 = np.random.randn(10, 3)
                // t = s0.dot(s1)

                // # now suppose we had the gradient on t from above in the circuit
                // dt = np.random.randn(*t.shape) # same shape as t
                // ds0 = dt.dot(s1.T) #.T gives the transpose of the matrix
                // ds1 = t.T.dot(dt)

                // tensor.shape [m,p,qq,rr]
                // src0.shape   [n,m,q1,r1]
                // src1.shape   [n,p,qq,rr]

                // necessary for llama
                if (src0->grad) {
                    struct wsp_ggml_tensor * s1_tg =
                        wsp_ggml_out_prod(ctx, // [n,m,qq,rr]
                            src1,          // [n,p,qq,rr]
                            tensor->grad); // [m,p,qq,rr]
                    const int64_t qq = s1_tg->ne[2];
                    const int64_t rr = s1_tg->ne[3];
                    const int64_t q1 = src0->ne[2];
                    const int64_t r1 = src0->ne[3];
                    const bool ne2_broadcasted = qq > q1;
                    const bool ne3_broadcasted = rr > r1;
                    if (ne2_broadcasted || ne3_broadcasted) {
                        // sum broadcast repetitions of s1_tg into shape of src0
                        s1_tg = wsp_ggml_repeat_back(ctx, s1_tg, src0);
                    }
                    src0->grad =
                        wsp_ggml_add_or_set(ctx,
                                src0->grad, // [n,m,q1,r1]
                                s1_tg,      // [n,m,q1,r1]
                                zero_table, acc_table);
                }
                if (src1->grad) {
                    src1->grad =
                        wsp_ggml_add_or_set(ctx,
                                src1->grad,                            // [n,p,qq,rr]
                                // wsp_ggml_mul_mat(ctx,                   // [n,p,qq,rr]
                                //     wsp_ggml_cont(ctx,                  // [m,n,q1,r1]
                                //         wsp_ggml_transpose(ctx, src0)), // [m,n,q1,r1]
                                //     tensor->grad),                  // [m,p,qq,rr]

                                // // when src0 is bigger than tensor->grad (this is mostly the case in llama),
                                // // avoid transpose of src0, rather transpose smaller tensor->grad
                                // // and then use wsp_ggml_out_prod
                                wsp_ggml_out_prod(ctx,                  // [n,p,qq,rr]
                                    src0,                           // [n,m,q1,r1]
                                    wsp_ggml_transpose(ctx,             // [p,m,qq,rr]
                                        tensor->grad)),             // [m,p,qq,rr]
                                zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_MUL_MAT_ID:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_OUT_PROD:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_SCALE:
            {
                // necessary for llama
                if (src0->grad) {
                    float s;
                    memcpy(&s, tensor->op_params, sizeof(float));

                    src0->grad =
                        wsp_ggml_add_or_set(ctx,
                            src0->grad,
                            wsp_ggml_scale_impl(ctx, tensor->grad, s, false),
                            zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_SET:
            {
                const size_t nb1     = ((int32_t *) tensor->op_params)[0];
                const size_t nb2     = ((int32_t *) tensor->op_params)[1];
                const size_t nb3     = ((int32_t *) tensor->op_params)[2];
                const size_t offset  = ((int32_t *) tensor->op_params)[3];

                struct wsp_ggml_tensor * tensor_grad_view = NULL;

                if (src0->grad || src1->grad) {
                    WSP_GGML_ASSERT(src0->type == tensor->type);
                    WSP_GGML_ASSERT(tensor->grad->type == tensor->type);
                    WSP_GGML_ASSERT(!src1->grad || src1->grad->type == tensor->grad->type);

                    tensor_grad_view = wsp_ggml_view_4d(ctx,
                        tensor->grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                        nb1, nb2, nb3, offset);
                }

                if (src0->grad) {
                    src0->grad = wsp_ggml_add_or_set(ctx,
                        src0->grad,
                        wsp_ggml_acc_impl(ctx,
                            tensor->grad,
                            wsp_ggml_neg(ctx, tensor_grad_view),
                            nb1, nb2, nb3, offset, false),
                        zero_table, acc_table);
                }

                if (src1->grad) {
                    src1->grad =
                        wsp_ggml_add_or_set(ctx,
                            src1->grad,
                            wsp_ggml_reshape(ctx,
                                wsp_ggml_cont(ctx, tensor_grad_view),
                                src1->grad),
                            zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_CPY:
            {
                // necessary for llama
                // cpy overwrites value of src1 by src0 and returns view(src1)
                // the overwriting is mathematically equivalent to:
                // tensor = src0 * 1 + src1 * 0
                if (src0->grad) {
                    // dsrc0 = dtensor * 1
                    src0->grad = wsp_ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table, acc_table);
                }
                if (src1->grad) {
                    // dsrc1 = dtensor * 0 -> noop
                }
            } break;
        case WSP_GGML_OP_CONT:
            {
                // same as cpy
                if (src0->grad) {
                    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(src0->grad));
                    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(tensor->grad));
                    src0->grad = wsp_ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_RESHAPE:
            {
                // necessary for llama
                if (src0->grad) {
                    src0->grad =
                        wsp_ggml_add_or_set(ctx, src0->grad,
                            wsp_ggml_reshape(ctx,
                                wsp_ggml_is_contiguous(tensor->grad)
                                    ? tensor->grad
                                    : wsp_ggml_cont(ctx, tensor->grad),
                                src0->grad),
                        zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_VIEW:
            {
                // necessary for llama
                if (src0->grad) {
                    size_t offset;

                    memcpy(&offset, tensor->op_params, sizeof(offset));

                    size_t nb1 = tensor->nb[1];
                    size_t nb2 = tensor->nb[2];
                    size_t nb3 = tensor->nb[3];

                    if (src0->type != src0->grad->type) {
                        // gradient is typically F32, but src0 could be other type
                        size_t ng = wsp_ggml_element_size(src0->grad);
                        size_t n0 = wsp_ggml_element_size(src0);
                        WSP_GGML_ASSERT(offset % n0 == 0);
                        WSP_GGML_ASSERT(nb1 % n0 == 0);
                        WSP_GGML_ASSERT(nb2 % n0 == 0);
                        WSP_GGML_ASSERT(nb3 % n0 == 0);
                        offset = (offset / n0) * ng;
                        nb1 = (nb1 / n0) * ng;
                        nb2 = (nb2 / n0) * ng;
                        nb3 = (nb3 / n0) * ng;
                    }

                    src0->grad = wsp_ggml_acc_or_set(ctx, src0->grad, tensor->grad, nb1, nb2, nb3, offset, zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_PERMUTE:
            {
                // necessary for llama
                if (src0->grad) {
                    int32_t * axes = (int32_t *) tensor->op_params;
                    int axis0 = axes[0] & 0x3;
                    int axis1 = axes[1] & 0x3;
                    int axis2 = axes[2] & 0x3;
                    int axis3 = axes[3] & 0x3;
                    int axes_backward[4] = {0,0,0,0};
                    axes_backward[axis0] = 0;
                    axes_backward[axis1] = 1;
                    axes_backward[axis2] = 2;
                    axes_backward[axis3] = 3;
                    src0->grad =
                        wsp_ggml_add_or_set(ctx, src0->grad,
                            wsp_ggml_permute(ctx,
                                tensor->grad,
                                axes_backward[0],
                                axes_backward[1],
                                axes_backward[2],
                                axes_backward[3]),
                            zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_TRANSPOSE:
            {
                // necessary for llama
                if (src0->grad) {
                    src0->grad =
                        wsp_ggml_add_or_set(ctx, src0->grad,
                            wsp_ggml_transpose(ctx, tensor->grad),
                        zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_GET_ROWS:
            {
                // necessary for llama (only for tokenizer)
                if (src0->grad) {
                    src0->grad =
                        wsp_ggml_add_or_set(ctx, src0->grad,
                            // last wsp_ggml_get_rows_back argument src0->grad is only
                            // necessary to setup correct output shape
                            wsp_ggml_get_rows_back(ctx, tensor->grad, src1, src0->grad),
                        zero_table, acc_table);
                }
                if (src1->grad) {
                    // noop
                }
            } break;
        case WSP_GGML_OP_GET_ROWS_BACK:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_DIAG:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_DIAG_MASK_INF:
            {
                // necessary for llama
                if (src0->grad) {
                    const int n_past = ((int32_t *) tensor->op_params)[0];
                    src0->grad =
                        wsp_ggml_add_or_set(ctx, src0->grad,
                            /* wsp_ggml_diag_mask_inf_impl() shouldn't be here */
                            /* ref:  https://github.com/ggerganov/llama.cpp/pull/4203#discussion_r1412377992 */
                            wsp_ggml_diag_mask_zero_impl(ctx, tensor->grad, n_past, false),
                        zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_DIAG_MASK_ZERO:
            {
                // necessary for llama
                if (src0->grad) {
                    const int n_past = ((int32_t *) tensor->op_params)[0];
                    src0->grad =
                        wsp_ggml_add_or_set(ctx, src0->grad,
                            wsp_ggml_diag_mask_zero_impl(ctx, tensor->grad, n_past, false),
                        zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_SOFT_MAX:
            {
                // necessary for llama
                if (src0->grad) {
                    src0->grad =
                        wsp_ggml_add_or_set(ctx, src0->grad,
                            wsp_ggml_soft_max_back(ctx, tensor->grad, tensor),
                        zero_table, acc_table);
                }
                WSP_GGML_ASSERT((!src1 || !src1->grad) && "backward pass for softmax mask not implemented");
            } break;
        case WSP_GGML_OP_SOFT_MAX_BACK:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_ROPE:
            {
                // necessary for llama
                if (src0->grad) {
                    //const int n_past = ((int32_t *) tensor->op_params)[0];
                    const int n_dims     = ((int32_t *) tensor->op_params)[1];
                    const int mode       = ((int32_t *) tensor->op_params)[2];
                    //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                    const int n_ctx_orig = ((int32_t *) tensor->op_params)[4];
                    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;

                    memcpy(&freq_base,   (int32_t *) tensor->op_params +  5, sizeof(float));
                    memcpy(&freq_scale,  (int32_t *) tensor->op_params +  6, sizeof(float));
                    memcpy(&ext_factor,  (int32_t *) tensor->op_params +  7, sizeof(float));
                    memcpy(&attn_factor, (int32_t *) tensor->op_params +  8, sizeof(float));
                    memcpy(&beta_fast,   (int32_t *) tensor->op_params +  9, sizeof(float));
                    memcpy(&beta_slow,   (int32_t *) tensor->op_params + 10, sizeof(float));

                    src0->grad = wsp_ggml_add_or_set(ctx,
                            src0->grad,
                            wsp_ggml_rope_back(ctx,
                                tensor->grad,
                                src1,
                                src2,
                                n_dims,
                                mode,
                                n_ctx_orig,
                                freq_base,
                                freq_scale,
                                ext_factor,
                                attn_factor,
                                beta_fast,
                                beta_slow),
                            zero_table, acc_table);
                }
                WSP_GGML_ASSERT((!src2 || !src2->grad) && "gradients for freq factors not implemented");
            } break;
        case WSP_GGML_OP_ROPE_BACK:
            {
                if (src0->grad) {
                    //const int n_past = ((int32_t *) tensor->op_params)[0];
                    const int n_dims     = ((int32_t *) tensor->op_params)[1];
                    const int mode       = ((int32_t *) tensor->op_params)[2];
                    //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                    const int n_ctx_orig = ((int32_t *) tensor->op_params)[4];
                    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;

                    memcpy(&freq_base,   (int32_t *) tensor->op_params +  5, sizeof(float));
                    memcpy(&freq_scale,  (int32_t *) tensor->op_params +  6, sizeof(float));
                    memcpy(&ext_factor,  (int32_t *) tensor->op_params +  7, sizeof(float));
                    memcpy(&attn_factor, (int32_t *) tensor->op_params +  8, sizeof(float));
                    memcpy(&beta_fast,   (int32_t *) tensor->op_params +  9, sizeof(float));
                    memcpy(&beta_slow,   (int32_t *) tensor->op_params + 10, sizeof(float));

                    src0->grad = wsp_ggml_add_or_set(ctx,
                            src0->grad,
                            wsp_ggml_rope_impl(ctx,
                                tensor->grad,
                                src1,
                                src2,
                                n_dims,
                                mode,
                                n_ctx_orig,
                                freq_base,
                                freq_scale,
                                ext_factor,
                                attn_factor,
                                beta_fast,
                                beta_slow,
                                false),
                            zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_CLAMP:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_CONV_TRANSPOSE_1D:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_IM2COL:
            {
                if (src1->grad) {
                    const int32_t s0    = wsp_ggml_get_op_params_i32(tensor, 0);
                    const int32_t s1    = wsp_ggml_get_op_params_i32(tensor, 1);
                    const int32_t p0    = wsp_ggml_get_op_params_i32(tensor, 2);
                    const int32_t p1    = wsp_ggml_get_op_params_i32(tensor, 3);
                    const int32_t d0    = wsp_ggml_get_op_params_i32(tensor, 4);
                    const int32_t d1    = wsp_ggml_get_op_params_i32(tensor, 5);
                    const bool    is_2D = wsp_ggml_get_op_params_i32(tensor, 6) == 1;

                    src1->grad = wsp_ggml_add_or_set(ctx,
                            src1->grad,
                            wsp_ggml_im2col_back(ctx, src0, tensor->grad, src1->ne, s0, s1, p0, p1, d0, d1, is_2D),
                            zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_IM2COL_BACK:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_CONV_TRANSPOSE_2D:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_POOL_1D:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_POOL_2D:
            {
                if (src0->grad) {
                    const enum wsp_ggml_op_pool op = wsp_ggml_get_op_params_i32(tensor, 0);
                    const      int32_t      k0 = wsp_ggml_get_op_params_i32(tensor, 1);
                    const      int32_t      k1 = wsp_ggml_get_op_params_i32(tensor, 2);
                    const      int32_t      s0 = wsp_ggml_get_op_params_i32(tensor, 3);
                    const      int32_t      s1 = wsp_ggml_get_op_params_i32(tensor, 4);
                    const      int32_t      p0 = wsp_ggml_get_op_params_i32(tensor, 5);
                    const      int32_t      p1 = wsp_ggml_get_op_params_i32(tensor, 6);

                    src0->grad = wsp_ggml_add_or_set(ctx,
                            src0->grad,
                            wsp_ggml_pool_2d_back(ctx, tensor->grad, src0, op, k0, k1, s0, s1, p0, p1),
                            zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_POOL_2D_BACK:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_UPSCALE:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_PAD:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_ARANGE:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_TIMESTEP_EMBEDDING:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_ARGSORT:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_LEAKY_RELU:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_FLASH_ATTN_EXT:
            {
                WSP_GGML_ABORT("FA backward pass not adapted after rework");
                struct wsp_ggml_tensor * flash_grad = NULL;
                if (src0->grad || src1->grad || tensor->src[2]->grad) {
                    int32_t t = wsp_ggml_get_op_params_i32(tensor, 0);
                    WSP_GGML_ASSERT(t == 0 || t == 1);
                    bool masked = t != 0;
                    flash_grad =
                        wsp_ggml_flash_attn_back(ctx,
                            src0,
                            src1,
                            tensor->src[2],
                            tensor->grad,
                            masked);
                }

                const int64_t elem_q = wsp_ggml_nelements(src0);
                const int64_t elem_k = wsp_ggml_nelements(src1);
                const int64_t elem_v = wsp_ggml_nelements(src2);

                enum wsp_ggml_type result_type = flash_grad->type;
                WSP_GGML_ASSERT(wsp_ggml_blck_size(result_type) == 1);
                const size_t tsize = wsp_ggml_type_size(result_type);

                const size_t offs_q = 0;
                const size_t offs_k = offs_q + WSP_GGML_PAD(elem_q * tsize, WSP_GGML_MEM_ALIGN);
                const size_t offs_v = offs_k + WSP_GGML_PAD(elem_k * tsize, WSP_GGML_MEM_ALIGN);

                if (src0->grad) {
                    struct wsp_ggml_tensor * view_q = wsp_ggml_view_1d(ctx, flash_grad, elem_q, offs_q);
                    struct wsp_ggml_tensor * grad_q = wsp_ggml_reshape(ctx, view_q, src0);
                    src0->grad = wsp_ggml_add_or_set(ctx,
                            src0->grad,
                            grad_q,
                            zero_table, acc_table);
                }
                if (src1->grad) {
                    struct wsp_ggml_tensor * view_k = wsp_ggml_view_1d(ctx, flash_grad, elem_k, offs_k);
                    struct wsp_ggml_tensor * grad_k = wsp_ggml_reshape(ctx, view_k, src1);
                    src1->grad = wsp_ggml_add_or_set(ctx,
                            src1->grad,
                            grad_k,
                            zero_table, acc_table);
                }
                if (src2->grad) {
                    struct wsp_ggml_tensor * view_v = wsp_ggml_view_1d(ctx, flash_grad, elem_v, offs_v);
                    struct wsp_ggml_tensor * grad_v = wsp_ggml_reshape(ctx, view_v, src2);
                    src2->grad = wsp_ggml_add_or_set(ctx,
                            src2->grad,
                            grad_v,
                            zero_table, acc_table);
                }
            } break;
        case WSP_GGML_OP_FLASH_ATTN_BACK:
            {
                WSP_GGML_ABORT("fatal error"); // not supported
            }
        case WSP_GGML_OP_SSM_CONV:
        case WSP_GGML_OP_SSM_SCAN:
            {
                WSP_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case WSP_GGML_OP_WIN_PART:
        case WSP_GGML_OP_WIN_UNPART:
        case WSP_GGML_OP_UNARY:
            {
                switch (wsp_ggml_get_unary_op(tensor)) {
                    case WSP_GGML_UNARY_OP_ABS:
                        {
                            if (src0->grad) {
                                src0->grad =
                                    wsp_ggml_add_or_set(ctx,
                                            src0->grad,
                                            wsp_ggml_mul(ctx,
                                                wsp_ggml_sgn(ctx, src0),
                                                tensor->grad),
                                            zero_table, acc_table);
                            }
                        } break;
                    case WSP_GGML_UNARY_OP_SGN:
                        {
                            if (src0->grad) {
                                // noop
                            }
                        } break;
                    case WSP_GGML_UNARY_OP_NEG:
                        {
                            if (src0->grad) {
                                src0->grad = wsp_ggml_sub_or_set(ctx, src0->grad, tensor->grad, zero_table, acc_table);
                            }
                        } break;
                    case WSP_GGML_UNARY_OP_STEP:
                        {
                            if (src0->grad) {
                                // noop
                            }
                        } break;
                    case WSP_GGML_UNARY_OP_TANH:
                        {
                            WSP_GGML_ABORT("fatal error"); // TODO: not implemented
                        }
                    case WSP_GGML_UNARY_OP_ELU:
                        {
                            WSP_GGML_ABORT("fatal error"); // TODO: not implemented
                        }
                    case WSP_GGML_UNARY_OP_RELU:
                        {
                            if (src0->grad) {
                                src0->grad = wsp_ggml_add_or_set(ctx,
                                        src0->grad,
                                        wsp_ggml_mul(ctx,
                                            wsp_ggml_step(ctx, src0),
                                            tensor->grad),
                                        zero_table, acc_table);
                            }
                        } break;
                    case WSP_GGML_UNARY_OP_SIGMOID:
                        {
                            WSP_GGML_ABORT("fatal error"); // TODO: not implemented
                        }
                    case WSP_GGML_UNARY_OP_GELU:
                        {
                            WSP_GGML_ABORT("fatal error"); // TODO: not implemented
                        }
                    case WSP_GGML_UNARY_OP_GELU_QUICK:
                        {
                            WSP_GGML_ABORT("fatal error"); // TODO: not implemented
                        }
                    case WSP_GGML_UNARY_OP_SILU:
                        {
                            // necessary for llama
                            if (src0->grad) {
                                src0->grad = wsp_ggml_add_or_set(ctx,
                                        src0->grad,
                                        wsp_ggml_silu_back(ctx, src0, tensor->grad),
                                        zero_table, acc_table);
                            }
                        } break;
                    case WSP_GGML_UNARY_OP_EXP:
                        {
                            if (src0->grad) {
                                src0->grad = wsp_ggml_add_or_set(ctx,
                                        src0->grad,
                                        wsp_ggml_mul(ctx, tensor, tensor->grad),
                                        zero_table, acc_table);
                            }
                        } break;
                    default:
                        WSP_GGML_ABORT("fatal error");
                }
            } break;
        case WSP_GGML_OP_GET_REL_POS:
        case WSP_GGML_OP_ADD_REL_POS:
        case WSP_GGML_OP_RWKV_WKV:
        case WSP_GGML_OP_MAP_UNARY:
        case WSP_GGML_OP_MAP_BINARY:
        case WSP_GGML_OP_MAP_CUSTOM1_F32:
        case WSP_GGML_OP_MAP_CUSTOM2_F32:
        case WSP_GGML_OP_MAP_CUSTOM3_F32:
        case WSP_GGML_OP_MAP_CUSTOM1:
        case WSP_GGML_OP_MAP_CUSTOM2:
        case WSP_GGML_OP_MAP_CUSTOM3:
            {
                WSP_GGML_ABORT("fatal error"); // not supported
            }
        case WSP_GGML_OP_CROSS_ENTROPY_LOSS:
            {
                if (src0->grad) {
                    src0->grad = wsp_ggml_add_or_set(ctx,
                                src0->grad,
                                wsp_ggml_cross_entropy_loss_back(ctx,
                                    src0,
                                    src1,
                                    tensor->grad),
                                zero_table, acc_table);
                }
                WSP_GGML_ASSERT(!src1->grad && "backward pass for labels not implemented");
            } break;
        case WSP_GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                WSP_GGML_ABORT("fatal error"); // not supported
            }
        case WSP_GGML_OP_OPT_STEP_ADAMW:
            {
                WSP_GGML_ABORT("fatal error"); // not supported
            }
        case WSP_GGML_OP_NONE:
            {
                // nop
            } break;
        case WSP_GGML_OP_COUNT:
            {
                WSP_GGML_ABORT("fatal error");
            }
    }

    for (int i = 0; i < WSP_GGML_MAX_SRC; ++i) {
        if (tensor->src[i] && tensor->src[i]->grad) {
            WSP_GGML_ASSERT(wsp_ggml_are_same_shape(tensor->src[i], tensor->src[i]->grad));
        }
    }
}

static void wsp_ggml_visit_parents(struct wsp_ggml_cgraph * cgraph, struct wsp_ggml_tensor * node) {
    if (node->grad == NULL) {
        // this usually happens when we generate intermediate nodes from constants in the backward pass
        // it can also happen during forward pass, if the user performs computations with constants
        if (node->op != WSP_GGML_OP_NONE) {
            //WSP_GGML_PRINT_DEBUG("%s: warning: node %p has no grad, but op %d\n", __func__, (void *) node, node->op);
        }
    }

    // check if already visited
    if (wsp_ggml_hash_insert(&cgraph->visited_hash_set, node) == WSP_GGML_HASHSET_ALREADY_EXISTS) {
        return;
    }

    for (int i = 0; i < WSP_GGML_MAX_SRC; ++i) {
        const int k =
            (cgraph->order == WSP_GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) ? i :
            (cgraph->order == WSP_GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT) ? (WSP_GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i*/ i;
        if (node->src[k]) {
            wsp_ggml_visit_parents(cgraph, node->src[k]);
        }
    }

    if (node->op == WSP_GGML_OP_NONE && !(node->flags & WSP_GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        WSP_GGML_ASSERT(cgraph->n_leafs < cgraph->size);

        if (strlen(node->name) == 0) {
            wsp_ggml_format_name(node, "leaf_%d", cgraph->n_leafs);
        }

        cgraph->leafs[cgraph->n_leafs] = node;
        cgraph->n_leafs++;
    } else {
        WSP_GGML_ASSERT(cgraph->n_nodes < cgraph->size);

        if (strlen(node->name) == 0) {
            wsp_ggml_format_name(node, "node_%d", cgraph->n_nodes);
        }

        cgraph->nodes[cgraph->n_nodes] = node;
        cgraph->n_nodes++;
    }
}

static void wsp_ggml_build_forward_impl(struct wsp_ggml_cgraph * cgraph, struct wsp_ggml_tensor * tensor, bool expand) {
    if (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to wsp_ggml_build_forward_expand
        wsp_ggml_graph_clear(cgraph);
    }

    const int n0 = cgraph->n_nodes;

    wsp_ggml_visit_parents(cgraph, tensor);

    const int n_new = cgraph->n_nodes - n0;
    WSP_GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    if (n_new > 0) {
        // the last added node should always be starting point
        WSP_GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

void wsp_ggml_build_forward_expand(struct wsp_ggml_cgraph * cgraph, struct wsp_ggml_tensor * tensor) {
    wsp_ggml_build_forward_impl(cgraph, tensor, true);
}

void wsp_ggml_build_backward_expand(struct wsp_ggml_context * ctx, struct wsp_ggml_cgraph * gf, struct wsp_ggml_cgraph * gb, bool accumulate) {
    WSP_GGML_ASSERT(gf->n_nodes > 0);
    WSP_GGML_ASSERT(gf->grads);

    for (int i = 0; i < gf->n_nodes; ++i) {
        struct wsp_ggml_tensor * node = gf->nodes[i];

        if (node->type == WSP_GGML_TYPE_I32) {
            continue;
        }

        bool needs_grad = node->flags & WSP_GGML_TENSOR_FLAG_PARAM;
        bool ignore_src[WSP_GGML_MAX_SRC] = {false};
        switch (node->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            case WSP_GGML_OP_IM2COL:      // only used for its shape
            case WSP_GGML_OP_IM2COL_BACK: // same as IM2COL
                ignore_src[0] = true;
                break;
            case WSP_GGML_OP_UNARY: {
                const enum wsp_ggml_unary_op uop = wsp_ggml_get_unary_op(node);
                // SGN and STEP unary ops are piecewise constant
                if (uop == WSP_GGML_UNARY_OP_SGN || uop == WSP_GGML_UNARY_OP_STEP) {
                    ignore_src[0] = true;
                }
            } break;

            // gradients in node->src[1] for one reason or another have no effect on output gradients
            case WSP_GGML_OP_CPY:           // gradients in CPY target  are irrelevant
            case WSP_GGML_OP_GET_ROWS:      // row indices not differentiable
            case WSP_GGML_OP_GET_ROWS_BACK: // same as for GET_ROWS
            case WSP_GGML_OP_ROPE:          // positions not differentiable
                ignore_src[1] = true;
                break;

            default:
                break;
        }
        for (int j = 0; j < WSP_GGML_MAX_SRC; ++j) {
            if (!node->src[j] || !node->src[j]->grad || ignore_src[j]) {
                continue;
            }
            WSP_GGML_ASSERT(node->src[j]->type == WSP_GGML_TYPE_F32 || node->src[j]->type == WSP_GGML_TYPE_F16);
            needs_grad = true;
            break;
        }
        if (!needs_grad) {
            continue;
        }

        // inplace operations are currently not supported
        WSP_GGML_ASSERT(!node->view_src || node->op == WSP_GGML_OP_CPY || node->op == WSP_GGML_OP_VIEW ||
            node->op == WSP_GGML_OP_RESHAPE || node->op == WSP_GGML_OP_PERMUTE || node->op == WSP_GGML_OP_TRANSPOSE);

        // create a new tensor with the same type and shape as the node and set it as grad
        node->grad = wsp_ggml_dup_tensor(ctx, node);
    }

    // keep tables of original gradients for replacement/accumulation logic
    struct wsp_ggml_hash_set zero_table = wsp_ggml_hash_set_new(gf->size);
    struct wsp_ggml_hash_set acc_table  = wsp_ggml_hash_set_new(gf->size);
    for (int i = 0; i < gf->n_nodes; i++) {
        struct wsp_ggml_tensor * node = gf->nodes[i];

        if (node->grad) {
            {
                const size_t insert_result = wsp_ggml_hash_insert(&zero_table, node->grad);
                WSP_GGML_ASSERT(insert_result != WSP_GGML_HASHSET_FULL);
                WSP_GGML_ASSERT(insert_result != WSP_GGML_HASHSET_ALREADY_EXISTS);
            }

            // only gradients of trainable parameters should be accumulated
            if (accumulate && (node->flags & WSP_GGML_TENSOR_FLAG_PARAM)) {
                const size_t insert_result = wsp_ggml_hash_insert(&acc_table, node->grad);
                WSP_GGML_ASSERT(insert_result != WSP_GGML_HASHSET_FULL);
                WSP_GGML_ASSERT(insert_result != WSP_GGML_HASHSET_ALREADY_EXISTS);
            }
        }
    }

    for (int i = gf->n_nodes - 1; i >= 0; i--) {
        struct wsp_ggml_tensor * node = gf->nodes[i];

        // inplace operations to add gradients are not created by wsp_ggml_compute_backward except for gradient accumulation
        // use allocator to automatically make inplace operations
        if (node->grad) {
            wsp_ggml_compute_backward(ctx, node, &zero_table, &acc_table);
        }
    }

    for (int i = 0; i < gf->n_nodes; i++) {
        struct wsp_ggml_tensor * node = gf->nodes[i];

        if (node->flags & WSP_GGML_TENSOR_FLAG_PARAM) {
            WSP_GGML_PRINT_DEBUG("%s: found root node %p\n", __func__, (void *) node);
            wsp_ggml_build_forward_expand(gb, node->grad);
        }
    }

    wsp_ggml_hash_set_free(&zero_table);
    wsp_ggml_hash_set_free(&acc_table);
}

void wsp_ggml_build_opt_adamw(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_cgraph  * gf,
        struct wsp_ggml_cgraph  * gb,
        float                 alpha,
        float                 beta1,
        float                 beta2,
        float                 eps,
        float                 wd) {
    for (int i = 0; i < gf->n_nodes; i++) {
        struct wsp_ggml_tensor * node = gf->nodes[i];

        if (node->flags & WSP_GGML_TENSOR_FLAG_PARAM) {
            WSP_GGML_PRINT_DEBUG("%s: found root node %p\n", __func__, (void *) node);
            struct wsp_ggml_tensor * opt_step = wsp_ggml_opt_step_adamw(ctx, node, node->grad, alpha, beta1, beta2, eps, wd);
            wsp_ggml_build_forward_expand(gb, opt_step);
        }
    }
}


static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {
    void * ptr = *p;
    ptr = (void *) WSP_GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}

static size_t wsp_ggml_graph_nbytes(size_t size, bool grads) {
    size_t hash_size = wsp_ggml_hash_size(size * 2);
    void * p = 0;
    incr_ptr_aligned(&p, sizeof(struct wsp_ggml_cgraph), 1);
    incr_ptr_aligned(&p, size * sizeof(struct wsp_ggml_tensor *), sizeof(struct wsp_ggml_tensor *)); // nodes
    incr_ptr_aligned(&p, size * sizeof(struct wsp_ggml_tensor *), sizeof(struct wsp_ggml_tensor *)); // leafs
    incr_ptr_aligned(&p, hash_size * sizeof(struct wsp_ggml_tensor *), sizeof(struct wsp_ggml_tensor *)); // hash keys
    if (grads) {
        incr_ptr_aligned(&p, size * sizeof(struct wsp_ggml_tensor *), sizeof(struct wsp_ggml_tensor *)); // grads
    }
    incr_ptr_aligned(&p, wsp_ggml_bitset_size(hash_size) * sizeof(wsp_ggml_bitset_t), sizeof(wsp_ggml_bitset_t));

    size_t nbytes = (size_t) p;
    return nbytes;
}

size_t wsp_ggml_graph_overhead_custom(size_t size, bool grads) {
    return WSP_GGML_OBJECT_SIZE + WSP_GGML_PAD(wsp_ggml_graph_nbytes(size, grads), WSP_GGML_MEM_ALIGN);
}

size_t wsp_ggml_graph_overhead(void) {
    return wsp_ggml_graph_overhead_custom(WSP_GGML_DEFAULT_GRAPH_SIZE, false);
}

struct wsp_ggml_cgraph * wsp_ggml_new_graph_custom(struct wsp_ggml_context * ctx, size_t size, bool grads) {
    const size_t obj_size = wsp_ggml_graph_nbytes(size, grads);
    struct wsp_ggml_object * obj = wsp_ggml_new_object(ctx, WSP_GGML_OBJECT_TYPE_GRAPH, obj_size);
    struct wsp_ggml_cgraph * cgraph = (struct wsp_ggml_cgraph *) ((char *) ctx->mem_buffer + obj->offs);

    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_size = wsp_ggml_hash_size(size * 2);

    void * p = cgraph + 1;

    struct wsp_ggml_tensor ** nodes_ptr = incr_ptr_aligned(&p, size * sizeof(struct wsp_ggml_tensor *), sizeof(struct wsp_ggml_tensor *));
    struct wsp_ggml_tensor ** leafs_ptr = incr_ptr_aligned(&p, size * sizeof(struct wsp_ggml_tensor *), sizeof(struct wsp_ggml_tensor *));
    struct wsp_ggml_tensor ** hash_keys_ptr = incr_ptr_aligned(&p, hash_size * sizeof(struct wsp_ggml_tensor *), sizeof(struct wsp_ggml_tensor *));
    struct wsp_ggml_tensor ** grads_ptr = grads ? incr_ptr_aligned(&p, size * sizeof(struct wsp_ggml_tensor *), sizeof(struct wsp_ggml_tensor *)) : NULL;
    wsp_ggml_bitset_t * hash_used = incr_ptr_aligned(&p, wsp_ggml_bitset_size(hash_size) * sizeof(wsp_ggml_bitset_t), sizeof(wsp_ggml_bitset_t));

    // check that we allocated the correct amount of memory
    assert(obj_size == (size_t)((char *)p - (char *)cgraph));

    *cgraph = (struct wsp_ggml_cgraph) {
        /*.size         =*/ size,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.hash_table   =*/ { hash_size, hash_used, hash_keys_ptr },
        /*.order        =*/ WSP_GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT,
    };

    wsp_ggml_hash_set_reset(&cgraph->visited_hash_set);

    return cgraph;
}

struct wsp_ggml_cgraph * wsp_ggml_new_graph(struct wsp_ggml_context * ctx) {
    return wsp_ggml_new_graph_custom(ctx, WSP_GGML_DEFAULT_GRAPH_SIZE, false);
}

struct wsp_ggml_cgraph wsp_ggml_graph_view(struct wsp_ggml_cgraph * cgraph0, int i0, int i1) {
    struct wsp_ggml_cgraph cgraph = {
        /*.size         =*/ 0,
        /*.n_nodes      =*/ i1 - i0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ cgraph0->nodes + i0,
        /*.grads        =*/ cgraph0->grads ? cgraph0->grads + i0 : NULL,
        /*.leafs        =*/ NULL,
        /*.hash_table   =*/ { 0, NULL, NULL },
        /*.order        =*/ cgraph0->order,
    };

    return cgraph;
}

void wsp_ggml_graph_cpy(struct wsp_ggml_cgraph * src, struct wsp_ggml_cgraph * dst) {
    WSP_GGML_ASSERT(dst->size >= src->n_leafs);
    WSP_GGML_ASSERT(dst->size >= src->n_nodes);
    WSP_GGML_ASSERT(dst->visited_hash_set.size >= src->visited_hash_set.size);

    dst->n_leafs = src->n_leafs;
    dst->n_nodes = src->n_nodes;
    dst->order   = src->order;

    for (int i = 0; i < src->n_leafs; ++i) {
        dst->leafs[i] = src->leafs[i];
    }

    for (int i = 0; i < src->n_nodes; ++i) {
        dst->nodes[i] = src->nodes[i];
    }

    if (src->grads) {
        WSP_GGML_ASSERT(dst->grads != NULL);
        for (int i = 0; i < src->n_nodes; ++i) {
            dst->grads[i] = src->grads[i];
        }
    }

    for (size_t i = 0; i < src->visited_hash_set.size; ++i) {
        // copy all hashset keys (tensors) that are in use
        if (wsp_ggml_bitset_get(src->visited_hash_set.used, i)) {
            wsp_ggml_hash_insert(&dst->visited_hash_set, src->visited_hash_set.keys[i]);
        }
    }
}

struct wsp_ggml_cgraph * wsp_ggml_graph_dup(struct wsp_ggml_context * ctx, struct wsp_ggml_cgraph * cgraph) {
    struct wsp_ggml_cgraph * result = wsp_ggml_new_graph_custom(ctx, cgraph->size, cgraph->grads != NULL);
    wsp_ggml_graph_cpy(cgraph, result);
    return result;
}

void wsp_ggml_graph_reset(struct wsp_ggml_cgraph * cgraph) {
    WSP_GGML_ASSERT(cgraph->grads != NULL);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct wsp_ggml_tensor * node = cgraph->nodes[i];

        // initial gradients of loss should be 1, 0 otherwise
        if (node->grad) {
            if (node->flags & WSP_GGML_TENSOR_FLAG_LOSS) {
                WSP_GGML_ASSERT(node->grad->buffer);
                WSP_GGML_ASSERT(node->type == WSP_GGML_TYPE_F32);
                WSP_GGML_ASSERT(wsp_ggml_is_scalar(node));

                const float onef = 1.0f;
                wsp_ggml_backend_tensor_set(node->grad, &onef, 0, wsp_ggml_nbytes(node->grad));
            } else {
                wsp_ggml_set_zero(node->grad);
            }
        }

        WSP_GGML_ASSERT(node);
        if (node->op == WSP_GGML_OP_OPT_STEP_ADAMW) {
            // set iteration to 1 and clear momenta
            wsp_ggml_set_op_params_i32(node, 0, 1);
            wsp_ggml_set_zero(node->src[2]);
            wsp_ggml_set_zero(node->src[3]);
        }
    }
}

void wsp_ggml_graph_clear(struct wsp_ggml_cgraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    wsp_ggml_hash_set_reset(&cgraph->visited_hash_set);
}

int wsp_ggml_graph_size(struct wsp_ggml_cgraph * cgraph) {
    return cgraph->size;
}

struct wsp_ggml_tensor * wsp_ggml_graph_node(struct wsp_ggml_cgraph * cgraph, int i) {
    if (i < 0) {
        WSP_GGML_ASSERT(cgraph->n_nodes + i >= 0);
        return cgraph->nodes[cgraph->n_nodes + i];
    }

    WSP_GGML_ASSERT(i < cgraph->n_nodes);
    return cgraph->nodes[i];
}

struct wsp_ggml_tensor ** wsp_ggml_graph_nodes(struct wsp_ggml_cgraph * cgraph) {
    return cgraph->nodes;
}

int wsp_ggml_graph_n_nodes(struct wsp_ggml_cgraph * cgraph) {
    return cgraph->n_nodes;
}

void wsp_ggml_graph_add_node(struct wsp_ggml_cgraph * cgraph, struct wsp_ggml_tensor * tensor) {
    WSP_GGML_ASSERT(cgraph->size > cgraph->n_nodes);
    cgraph->nodes[cgraph->n_nodes] = tensor;
    cgraph->n_nodes++;
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
        case WSP_GGML_OP_RWKV_WKV:
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
        WSP_GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
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
                    const enum wsp_ggml_type vec_dot_type = type_traits[node->src[0]->type].vec_dot_type;

                    if (node->src[1]->type != vec_dot_type) {
                        cur = wsp_ggml_row_size(vec_dot_type, wsp_ggml_nelements(node->src[1]));
                    }
                } break;
            case WSP_GGML_OP_MUL_MAT_ID:
                {
                    cur = 0;
                    const struct wsp_ggml_tensor * src0 = node->src[0];
                    const struct wsp_ggml_tensor * src1 = node->src[1];
                    const enum wsp_ggml_type vec_dot_type = type_traits[src0->type].vec_dot_type;
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
    WSP_GGML_ASSERT(cplan);
    WSP_GGML_ASSERT(cplan->n_threads > 0);
    WSP_GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != NULL);

    int n_threads                               = cplan->n_threads;
    struct wsp_ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    if (threadpool == NULL) {
        WSP_GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
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

    struct wsp_ggml_object * obj = wsp_ggml_new_object(ctx, WSP_GGML_OBJECT_TYPE_WORK_BUFFER, cplan.work_size);

    cplan.work_data = (uint8_t *)ctx->mem_buffer + obj->offs;

    return wsp_ggml_graph_compute(cgraph, &cplan);
}

struct wsp_ggml_tensor * wsp_ggml_graph_get_tensor(struct wsp_ggml_cgraph * cgraph, const char * name) {
    for (int i = 0; i < cgraph->n_leafs; i++) {
        struct wsp_ggml_tensor * leaf = cgraph->leafs[i];

        if (strcmp(leaf->name, name) == 0) {
            return leaf;
        }
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct wsp_ggml_tensor * node = cgraph->nodes[i];

        if (strcmp(node->name, name) == 0) {
            return node;
        }
    }

    return NULL;
}

static void wsp_ggml_graph_export_leaf(const struct wsp_ggml_tensor * tensor, FILE * fout) {
    const int64_t * ne = tensor->ne;
    const size_t  * nb = tensor->nb;

    fprintf(fout, "%-6s %-12s %8d %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 " %16zu %16zu %16zu %16zu %16p %32s\n",
            wsp_ggml_type_name(tensor->type),
            wsp_ggml_op_name  (tensor->op),
            wsp_ggml_n_dims(tensor),
            ne[0], ne[1], ne[2], ne[3],
            nb[0], nb[1], nb[2], nb[3],
            tensor->data,
            tensor->name);
}

static void wsp_ggml_graph_export_node(const struct wsp_ggml_tensor * tensor, const char * arg, FILE * fout) {
    const int64_t * ne = tensor->ne;
    const size_t  * nb = tensor->nb;

    fprintf(fout, "%-6s %-6s %-12s %8d %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 " %16zu %16zu %16zu %16zu %16p %32s\n",
            arg,
            wsp_ggml_type_name(tensor->type),
            wsp_ggml_op_name  (tensor->op),
            wsp_ggml_n_dims(tensor),
            ne[0], ne[1], ne[2], ne[3],
            nb[0], nb[1], nb[2], nb[3],
            tensor->data,
            tensor->name);
}

void wsp_ggml_graph_export(const struct wsp_ggml_cgraph * cgraph, const char * fname) {
    uint64_t size_eval = 0;

    // compute size of intermediate results
    // TODO: does not take into account scratch buffers !!!!
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        size_eval += wsp_ggml_nbytes_pad(cgraph->nodes[i]);
    }

    // print
    {
        FILE * fout = stdout;

        fprintf(fout, "\n");
        fprintf(fout, "%-16s %8x\n", "magic",        WSP_GGML_FILE_MAGIC);
        fprintf(fout, "%-16s %8d\n", "version",      WSP_GGML_FILE_VERSION);
        fprintf(fout, "%-16s %8d\n", "leafs",        cgraph->n_leafs);
        fprintf(fout, "%-16s %8d\n", "nodes",        cgraph->n_nodes);
        fprintf(fout, "%-16s %" PRIu64 "\n", "eval", size_eval);

        // header
        fprintf(fout, "\n");
        fprintf(fout, "%-6s %-12s %8s %8s %8s %8s %8s %16s %16s %16s %16s %16s %16s\n",
                "TYPE", "OP", "NDIMS", "NE0", "NE1", "NE2", "NE3", "NB0", "NB1", "NB2", "NB3", "DATA", "NAME");

        for (int i = 0; i < cgraph->n_leafs; ++i) {
            wsp_ggml_graph_export_leaf(cgraph->leafs[i], fout);

            WSP_GGML_ASSERT(cgraph->leafs[i]->op   == WSP_GGML_OP_NONE);
            WSP_GGML_ASSERT(cgraph->leafs[i]->src[0] == NULL);
            WSP_GGML_ASSERT(cgraph->leafs[i]->src[1] == NULL);
        }

        // header
        fprintf(fout, "\n");
        fprintf(fout, "%-6s %-6s %-12s %8s %8s %8s %8s %8s %16s %16s %16s %16s %8s %16s %16s\n",
                "ARG", "TYPE", "OP", "NDIMS", "NE0", "NE1", "NE2", "NE3", "NB0", "NB1", "NB2", "NB3", "NTASKS", "DATA", "NAME");

        for (int i = 0; i < cgraph->n_nodes; ++i) {
            wsp_ggml_graph_export_node(cgraph->nodes[i], "DST", fout);

            for (int j = 0; j < WSP_GGML_MAX_SRC; ++j) {
                if (cgraph->nodes[i]->src[j]) {
                    wsp_ggml_graph_export_node(cgraph->nodes[i]->src[j], "SRC", fout);
                }
            }

            fprintf(fout, "\n");
        }

        fprintf(fout, "\n");
    }

    // write binary data
    {
        FILE * fout = wsp_ggml_fopen(fname, "wb");

        if (!fout) {
            fprintf(stderr, "%s: failed to open %s: %s\n", __func__, fname, strerror(errno));
            return;
        }

        // header
        {
            const uint32_t magic   = WSP_GGML_FILE_MAGIC;
            const uint32_t version = WSP_GGML_FILE_VERSION;
            const uint32_t n_leafs = cgraph->n_leafs;
            const uint32_t n_nodes = cgraph->n_nodes;

            fwrite(&magic,     sizeof(uint32_t), 1, fout);
            fwrite(&version,   sizeof(uint32_t), 1, fout);
            fwrite(&n_leafs,   sizeof(uint32_t), 1, fout);
            fwrite(&n_nodes,   sizeof(uint32_t), 1, fout);
            fwrite(&size_eval, sizeof(uint64_t), 1, fout);
        }

        // leafs
        {
            for (int i = 0; i < cgraph->n_leafs; ++i) {
                const struct wsp_ggml_tensor * tensor = cgraph->leafs[i];

                const uint32_t type   = tensor->type;
                const uint32_t op     = tensor->op;
                const int32_t  flags  = tensor->flags;

                fwrite(&type,   sizeof(uint32_t), 1, fout);
                fwrite(&op,     sizeof(uint32_t), 1, fout);
                fwrite(&flags,  sizeof(int32_t),  1, fout);

                for (int j = 0; j < WSP_GGML_MAX_DIMS; ++j) {
                    const uint64_t ne = tensor->ne[j];
                    const uint64_t nb = tensor->nb[j];

                    fwrite(&ne, sizeof(uint64_t), 1, fout);
                    fwrite(&nb, sizeof(uint64_t), 1, fout);
                }

                fwrite(tensor->name,      sizeof(char), WSP_GGML_MAX_NAME,      fout);
                fwrite(tensor->op_params, sizeof(char), WSP_GGML_MAX_OP_PARAMS, fout);

                // dump the data
                // TODO: pad this to 32 byte boundary
                {
                    const size_t size = wsp_ggml_nbytes(tensor);

                    fwrite(tensor->data, sizeof(char), size, fout);
                }
            }
        }

        // nodes
        {
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                const struct wsp_ggml_tensor * tensor = cgraph->nodes[i];

                const uint32_t type   = tensor->type;
                const uint32_t op     = tensor->op;
                const int32_t  flags  = tensor->flags;

                fwrite(&type,   sizeof(uint32_t), 1, fout);
                fwrite(&op,     sizeof(uint32_t), 1, fout);
                fwrite(&flags,  sizeof(int32_t),  1, fout);

                for (int j = 0; j < WSP_GGML_MAX_DIMS; ++j) {
                    const uint64_t ne = tensor->ne[j];
                    const uint64_t nb = tensor->nb[j];

                    fwrite(&ne, sizeof(uint64_t), 1, fout);
                    fwrite(&nb, sizeof(uint64_t), 1, fout);
                }

                fwrite(tensor->name,      sizeof(char), WSP_GGML_MAX_NAME,      fout);
                fwrite(tensor->op_params, sizeof(char), WSP_GGML_MAX_OP_PARAMS, fout);

                // output the op arguments
                {
                    struct wsp_ggml_tensor * args[WSP_GGML_MAX_SRC] = { NULL };

                    for (int j = 0; j < WSP_GGML_MAX_SRC; ++j) {
                        args[j] = tensor->src[j];
                    }

                    for (int j = 0; j < WSP_GGML_MAX_SRC; ++j) {
                        if (args[j]) {
                            int32_t idx = -1;

                            // check if leaf
                            {
                                for (int k = 0; k < cgraph->n_leafs; ++k) {
                                    if (args[j] == cgraph->leafs[k]) {
                                        idx = k;
                                        break;
                                    }
                                }
                            }

                            // check if node
                            if (idx == -1) {
                                for (int k = 0; k < cgraph->n_nodes; ++k) {
                                    if (args[j] == cgraph->nodes[k]) {
                                        idx = cgraph->n_leafs + k;
                                        break;
                                    }
                                }
                            }

                            if (idx == -1) {
                                fprintf(stderr, "%s: failed to find tensor, arg = %d, node = %d\n", __func__, j, i);
                                fclose(fout);
                                return;
                            }

                            fwrite(&idx, sizeof(int32_t), 1, fout);
                        } else {
                            const int32_t nul = -1;

                            fwrite(&nul, sizeof(int32_t), 1, fout);
                        }
                    }
                }

                // dump the data
                // TODO: pad this to 32 byte boundary
                if ((flags & WSP_GGML_TENSOR_FLAG_PARAM)) {
                    const size_t size = wsp_ggml_nbytes(tensor);

                    fwrite(tensor->data, sizeof(char), size, fout);
                }
            }
        }

        fclose(fout);
    }
}

struct wsp_ggml_cgraph * wsp_ggml_graph_import(const char * fname, struct wsp_ggml_context ** ctx_data, struct wsp_ggml_context ** ctx_eval) {
    assert(*ctx_data == NULL);
    assert(*ctx_eval == NULL);

    struct wsp_ggml_cgraph * result = NULL;

    struct wsp_ggml_tensor * data = NULL;

    // read file into data
    {
        FILE * fin = wsp_ggml_fopen(fname, "rb");
        if (!fin) {
            fprintf(stderr, "%s: failed to open %s: %s\n", __func__, fname, strerror(errno));
            return result;
        }

        size_t fsize = 0;

        fseek(fin, 0, SEEK_END);
        fsize = ftell(fin);
        fseek(fin, 0, SEEK_SET);

        // create the data context
        {
            const size_t overhead = 1*wsp_ggml_tensor_overhead();

            struct wsp_ggml_init_params params = {
                .mem_size   = fsize + overhead,
                .mem_buffer = NULL,
                .no_alloc   = false,
            };

            *ctx_data = wsp_ggml_init(params);

            if (!*ctx_data) {
                fprintf(stderr, "%s: failed to create ggml context\n", __func__);
                fclose(fin);
                return result;
            }
        }

        data = wsp_ggml_new_tensor_1d(*ctx_data, WSP_GGML_TYPE_I8, fsize);

        {
            const size_t ret = fread(data->data, sizeof(char), fsize, fin);
            if (ret != fsize) {
                fprintf(stderr, "%s: failed to read %s\n", __func__, fname);
                fclose(fin);
                return result;
            }
        }

        fclose(fin);
    }

    // populate result
    {
        char * ptr = (char *) data->data;

        const uint32_t magic = *(const uint32_t *) ptr; ptr += sizeof(magic);

        if (magic != WSP_GGML_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid magic number, got %08x\n", __func__, magic);
            return result;
        }

        const uint32_t version = *(const uint32_t *) ptr; ptr += sizeof(version);

        if (version != WSP_GGML_FILE_VERSION) {
            fprintf(stderr, "%s: invalid version number\n", __func__);
            return result;
        }

        const uint32_t n_leafs   = *(const uint32_t *) ptr; ptr += sizeof(n_leafs);
        const uint32_t n_nodes   = *(const uint32_t *) ptr; ptr += sizeof(n_nodes);
        const uint64_t size_eval = *(const uint64_t *) ptr; ptr += sizeof(size_eval);
        const int     graph_size = MAX(n_leafs, n_nodes);

        // create the data context
        {
            const size_t overhead = (n_leafs + n_nodes)*wsp_ggml_tensor_overhead() + wsp_ggml_graph_overhead_custom(graph_size, false);

            struct wsp_ggml_init_params params = {
                .mem_size   = size_eval + overhead,
                .mem_buffer = NULL,
                .no_alloc   = true,
            };

            *ctx_eval = wsp_ggml_init(params);

            if (!*ctx_eval) {
                fprintf(stderr, "%s: failed to create ggml context\n", __func__);
                return result;
            }
        }

        result = wsp_ggml_new_graph_custom(*ctx_eval, graph_size, false);

        result->n_leafs = n_leafs;
        result->n_nodes = n_nodes;


        // leafs
        {
            uint32_t type;
            uint32_t op;
            int32_t  flags;

            for (uint32_t i = 0; i < n_leafs; ++i) {
                type   = *(const uint32_t *) ptr; ptr += sizeof(type);
                op     = *(const uint32_t *) ptr; ptr += sizeof(op);
                flags  = *(const int32_t  *) ptr; ptr += sizeof(flags);

                int64_t ne[WSP_GGML_MAX_DIMS];
                size_t  nb[WSP_GGML_MAX_DIMS];

                for (int j = 0; j < WSP_GGML_MAX_DIMS; ++j) {
                    uint64_t ne_cur;
                    uint64_t nb_cur;

                    ne_cur = *(const uint64_t *) ptr; ptr += sizeof(ne_cur);
                    nb_cur = *(const uint64_t *) ptr; ptr += sizeof(nb_cur);

                    ne[j] = ne_cur;
                    nb[j] = nb_cur;
                }

                struct wsp_ggml_tensor * tensor = wsp_ggml_new_tensor(*ctx_eval, (enum wsp_ggml_type) type, WSP_GGML_MAX_DIMS, ne);

                tensor->op    = (enum wsp_ggml_op) op;
                tensor->flags = flags;

                memcpy(tensor->name,      ptr, WSP_GGML_MAX_NAME);      ptr += WSP_GGML_MAX_NAME;
                memcpy(tensor->op_params, ptr, WSP_GGML_MAX_OP_PARAMS); ptr += WSP_GGML_MAX_OP_PARAMS;

                for (int j = 0; j < WSP_GGML_MAX_DIMS; ++j) {
                    tensor->nb[j] = nb[j];
                }

                tensor->data = (void *) ptr; ptr += wsp_ggml_nbytes(tensor);

                result->leafs[i] = tensor;

                fprintf(stderr, "%s: loaded leaf %u: '%16s', %9zu bytes\n", __func__, i, tensor->name, wsp_ggml_nbytes(tensor));
            }
        }

        wsp_ggml_set_no_alloc(*ctx_eval, false);

        // nodes
        {
            uint32_t type;
            uint32_t op;
            int32_t  flags;

            for (uint32_t i = 0; i < n_nodes; ++i) {
                type   = *(const uint32_t *) ptr; ptr += sizeof(type);
                op     = *(const uint32_t *) ptr; ptr += sizeof(op);
                flags  = *(const int32_t  *) ptr; ptr += sizeof(flags);

                enum wsp_ggml_op eop = (enum wsp_ggml_op) op;

                int64_t ne[WSP_GGML_MAX_DIMS];
                size_t  nb[WSP_GGML_MAX_DIMS];

                for (int j = 0; j < WSP_GGML_MAX_DIMS; ++j) {
                    uint64_t ne_cur;
                    uint64_t nb_cur;

                    ne_cur = *(const uint64_t *) ptr; ptr += sizeof(ne_cur);
                    nb_cur = *(const uint64_t *) ptr; ptr += sizeof(nb_cur);

                    ne[j] = ne_cur;
                    nb[j] = nb_cur;
                }

                const char * ptr_name      = ptr; ptr += WSP_GGML_MAX_NAME;
                const char * ptr_op_params = ptr; ptr += WSP_GGML_MAX_OP_PARAMS;

                const int32_t * ptr_arg_idx = (const int32_t *) ptr; ptr += WSP_GGML_MAX_SRC*sizeof(int32_t);

                struct wsp_ggml_tensor * args[WSP_GGML_MAX_SRC] = { NULL };

                // parse args
                for (int j = 0; j < WSP_GGML_MAX_SRC; ++j) {
                    const int32_t arg_idx = ptr_arg_idx[j];

                    if (arg_idx == -1) {
                        continue;
                    }

                    if (arg_idx < result->n_leafs) {
                        args[j] = result->leafs[arg_idx];
                    } else {
                        args[j] = result->nodes[arg_idx - result->n_leafs];
                    }
                }

                // create the tensor
                // "view" operations are handled differently
                // TODO: handle inplace ops - currently a copy is always made

                struct wsp_ggml_tensor * tensor = NULL;

                switch (eop) {
                    // TODO: implement other view ops
                    case WSP_GGML_OP_RESHAPE:
                        {
                            tensor = wsp_ggml_reshape_4d(*ctx_eval, args[0], ne[0], ne[1], ne[2], ne[3]);
                        } break;
                    case WSP_GGML_OP_VIEW:
                        {
                            tensor = wsp_ggml_view_4d(*ctx_eval, args[0], ne[0], ne[1], ne[2], ne[3], 0, 0, 0, 0);

                            size_t offs;
                            memcpy(&offs, ptr_op_params, sizeof(offs));

                            tensor->data = ((char *) tensor->data) + offs;
                        } break;
                    case WSP_GGML_OP_TRANSPOSE:
                        {
                            tensor = wsp_ggml_transpose(*ctx_eval, args[0]);
                        } break;
                    case WSP_GGML_OP_PERMUTE:
                        {
                            tensor = wsp_ggml_view_4d(*ctx_eval, args[0], ne[0], ne[1], ne[2], ne[3], 0, 0, 0, 0);
                        } break;
                    default:
                        {
                            tensor = wsp_ggml_new_tensor(*ctx_eval, (enum wsp_ggml_type) type, WSP_GGML_MAX_DIMS, ne);

                            tensor->op = eop;
                        } break;
                }

                memcpy(tensor->name,      ptr_name,      WSP_GGML_MAX_NAME);
                memcpy(tensor->op_params, ptr_op_params, WSP_GGML_MAX_OP_PARAMS);

                for (int j = 0; j < WSP_GGML_MAX_DIMS; ++j) {
                    tensor->nb[j] = nb[j];
                }

                for (int j = 0; j < WSP_GGML_MAX_SRC; ++j) {
                    tensor->src[j] = args[j];
                }

                result->nodes[i] = tensor;

                // TODO tensor data is be duplicated due to wsp_ggml_new_tensor call above
                if (flags & WSP_GGML_TENSOR_FLAG_PARAM) {
                    tensor->data = (void *) ptr; ptr += wsp_ggml_nbytes(tensor);
                }

                fprintf(stderr, "%s: loaded node %u: '%16s', %9zu bytes\n", __func__, i, tensor->name, wsp_ggml_nbytes(tensor));
            }
        }
    }

    return result;
}

void wsp_ggml_graph_print(const struct wsp_ggml_cgraph * cgraph) {
    WSP_GGML_LOG_INFO("=== GRAPH ===\n");

    WSP_GGML_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct wsp_ggml_tensor * node = cgraph->nodes[i];

        WSP_GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                node->ne[0], node->ne[1], node->ne[2],
                wsp_ggml_op_name(node->op), (node->flags & WSP_GGML_TENSOR_FLAG_PARAM) ? "x" : node->grad ? "g" : " ");
    }

    WSP_GGML_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
    for (int i = 0; i < cgraph->n_leafs; i++) {
        struct wsp_ggml_tensor * node = cgraph->leafs[i];

        WSP_GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                node->ne[0], node->ne[1],
                wsp_ggml_op_name(node->op),
                wsp_ggml_get_name(node));
    }

    WSP_GGML_LOG_INFO("========================================\n");
}

// check if node is part of the graph
static bool wsp_ggml_graph_find(const struct wsp_ggml_cgraph * cgraph, const struct wsp_ggml_tensor * node) {
    if (cgraph == NULL) {
        return true;
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i] == node) {
            return true;
        }
    }

    return false;
}

static struct wsp_ggml_tensor * wsp_ggml_graph_get_parent(const struct wsp_ggml_cgraph * cgraph, const struct wsp_ggml_tensor * node) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct wsp_ggml_tensor * parent = cgraph->nodes[i];

        if (parent->grad == node) {
            return parent;
        }
    }

    return NULL;
}

static void wsp_ggml_graph_dump_dot_node_edge(FILE * fp, const struct wsp_ggml_cgraph * gb, struct wsp_ggml_tensor * node, struct wsp_ggml_tensor * parent, const char * label)  {
    struct wsp_ggml_tensor * gparent = wsp_ggml_graph_get_parent(gb, node);
    struct wsp_ggml_tensor * gparent0 = wsp_ggml_graph_get_parent(gb, parent);
    fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
            gparent0 ? (void *) gparent0 : (void *) parent,
            gparent0 ? "g" : "x",
            gparent ? (void *) gparent : (void *) node,
            gparent ? "g" : "x",
            gparent ? "empty" : "vee",
            gparent ? "dashed" : "solid",
            label);
}

static void wsp_ggml_graph_dump_dot_leaf_edge(FILE * fp, struct wsp_ggml_tensor * node, struct wsp_ggml_tensor * parent, const char * label)  {
    fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ label = \"%s\"; ]\n",
            (void *) parent, "x",
            (void *) node, "x",
            label);
}

void wsp_ggml_graph_dump_dot(const struct wsp_ggml_cgraph * gb, const struct wsp_ggml_cgraph * gf, const char * filename) {
    char color[16];

    FILE * fp = wsp_ggml_fopen(filename, "w");
    WSP_GGML_ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    fprintf(fp, "  rankdir = TB;\n");

    for (int i = 0; i < gb->n_nodes; i++) {
        struct wsp_ggml_tensor * node = gb->nodes[i];

        if (wsp_ggml_graph_get_parent(gb, node) != NULL) {
            continue;
        }

        if (node->flags & WSP_GGML_TENSOR_FLAG_PARAM) {
            snprintf(color, sizeof(color), "yellow");
        } else if (node->grad) {
            if (wsp_ggml_graph_find(gf, node)) {
                snprintf(color, sizeof(color), "green");
            } else {
                snprintf(color, sizeof(color), "lightblue");
            }
        } else {
            snprintf(color, sizeof(color), "white");
        }

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"",
                (void *) node, color);

        if (strlen(node->name) > 0) {
            fprintf(fp, "%s (%s)|", node->name, wsp_ggml_type_name(node->type));
        } else {
            fprintf(fp, "(%s)|", wsp_ggml_type_name(node->type));
        }

        if (wsp_ggml_is_matrix(node)) {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, node->ne[0], node->ne[1], wsp_ggml_op_symbol(node->op));
        } else {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, node->ne[0], node->ne[1], node->ne[2], wsp_ggml_op_symbol(node->op));
        }

        if (node->grad) {
            fprintf(fp, " | <g>%s\"; ]\n", wsp_ggml_op_symbol(node->grad->op));
        } else {
            fprintf(fp, "\"; ]\n");
        }
    }

    for (int i = 0; i < gb->n_leafs; i++) {
        struct wsp_ggml_tensor * node = gb->leafs[i];

        snprintf(color, sizeof(color), "pink");

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"<x>",
                (void *) node, color);

        if (strlen(node->name) > 0) {
            fprintf(fp, "%s (%s)|", node->name, wsp_ggml_type_name(node->type));
        } else {
            fprintf(fp, "(%s)|", wsp_ggml_type_name(node->type));
        }

        fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, node->ne[0], node->ne[1]);
        if (wsp_ggml_nelements(node) < 5 && node->data != NULL) {
            fprintf(fp, " | (");
            for (int j = 0; j < wsp_ggml_nelements(node); j++) {
                if (node->type == WSP_GGML_TYPE_I8 || node->type == WSP_GGML_TYPE_I16 || node->type == WSP_GGML_TYPE_I32) {
                    fprintf(fp, "%d", wsp_ggml_get_i32_1d(node, j));
                }
                else if (node->type == WSP_GGML_TYPE_F32 ||
                         node->type == WSP_GGML_TYPE_F16 ||
                         node->type == WSP_GGML_TYPE_BF16) {
                    fprintf(fp, "%.1e", (double)wsp_ggml_get_f32_1d(node, j));
                }
                else {
                    fprintf(fp, "#");
                }
                if (j < wsp_ggml_nelements(node) - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, ")");
        }
        fprintf(fp, "\"; ]\n");
    }

    for (int i = 0; i < gb->n_nodes; i++) {
        struct wsp_ggml_tensor * node = gb->nodes[i];

        for (int j = 0; j < WSP_GGML_MAX_SRC; j++) {
            if (node->src[j]) {
                char label[16];
                snprintf(label, sizeof(label), "src %d", j);
                wsp_ggml_graph_dump_dot_node_edge(fp, gb, node, node->src[j], label);
            }
        }
    }

    for (int i = 0; i < gb->n_leafs; i++) {
        struct wsp_ggml_tensor * node = gb->leafs[i];

        for (int j = 0; j < WSP_GGML_MAX_SRC; j++) {
            if (node->src[j]) {
                char label[16];
                snprintf(label, sizeof(label), "src %d", j);
                wsp_ggml_graph_dump_dot_leaf_edge(fp, node, node->src[j], label);
            }
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);

    WSP_GGML_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

static void wsp_ggml_opt_set_params(int np, struct wsp_ggml_tensor * const ps[], const float * x) {
    int i = 0;
    for (int p = 0; p < np; ++p) {
        const int64_t ne = wsp_ggml_nelements(ps[p]) ;
        // TODO: add function to set tensor from array
        for (int64_t j = 0; j < ne; ++j) {
            wsp_ggml_set_f32_1d(ps[p], j, x[i++]);
        }
    }
}

static void wsp_ggml_opt_get_params(int np, struct wsp_ggml_tensor * const ps[], float * x) {
    int i = 0;
    for (int p = 0; p < np; ++p) {
        const int64_t ne = wsp_ggml_nelements(ps[p]) ;
        // TODO: add function to get all elements at once
        for (int64_t j = 0; j < ne; ++j) {
            x[i++] = wsp_ggml_get_f32_1d(ps[p], j);
        }
    }
}

static void wsp_ggml_opt_get_grad(int np, struct wsp_ggml_tensor * const ps[], float * g) {
    int64_t i = 0;
    for (int p = 0; p < np; ++p) {
        const int64_t ne = wsp_ggml_nelements(ps[p]) ;
        // TODO: add function to get all elements at once
        for (int64_t j = 0; j < ne; ++j) {
            g[i++] = wsp_ggml_get_f32_1d(ps[p]->grad, j);
        }
    }
}

static void wsp_ggml_opt_acc_grad(int np, struct wsp_ggml_tensor * const ps[], float * g, float scale) {
    int64_t i = 0;
    for (int p = 0; p < np; ++p) {
        const int64_t ne = wsp_ggml_nelements(ps[p]) ;
        // TODO: add function to get all elements at once
        for (int64_t j = 0; j < ne; ++j) {
            g[i++] += wsp_ggml_get_f32_1d(ps[p]->grad, j) * scale;
        }
    }
}

//
// Using AdamW - ref: https://arxiv.org/pdf/1711.05101v3.pdf
//
// (Original Adam - ref: https://arxiv.org/pdf/1412.6980.pdf)
//

static enum wsp_ggml_opt_result wsp_ggml_opt_adam(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_opt_context * opt,
        struct wsp_ggml_opt_params params,
        struct wsp_ggml_tensor * f,
        struct wsp_ggml_cgraph * gf,
        struct wsp_ggml_cgraph * gb,
        wsp_ggml_opt_callback callback,
        void * callback_data) {
    WSP_GGML_ASSERT(wsp_ggml_is_scalar(f));
    WSP_GGML_ASSERT(f->type == WSP_GGML_TYPE_F32);

    // these will store the parameters we want to optimize
    struct wsp_ggml_tensor * ps[WSP_GGML_MAX_PARAMS];

    int np = 0;
    int64_t nx = 0;
    for (int i = 0; i < gf->n_nodes; ++i) {
        if (gf->nodes[i]->flags & WSP_GGML_TENSOR_FLAG_PARAM) {
            WSP_GGML_PRINT_DEBUG("found param %d: grad->op = %d\n", np, gf->nodes[i]->grad->op);

            WSP_GGML_ASSERT(np < WSP_GGML_MAX_PARAMS);

            ps[np++] = gf->nodes[i];
            nx += wsp_ggml_nelements(gf->nodes[i]);
        }
    }

    if ((opt->params.type != params.type) || (opt->nx != nx) || (opt->params.past != params.past)) {
        int iter = opt->iter;
        wsp_ggml_opt_init(opt->ctx, opt, params, nx);
        opt->iter = iter;
    }

    // constants
    float sched = params.adam.sched;
    const float alpha = params.adam.alpha;
    const float decay = params.adam.decay * alpha;
    const float beta1 = params.adam.beta1;
    const float beta2 = params.adam.beta2;
    const float eps   = params.adam.eps;
    const float gclip = params.adam.gclip;
    const int decay_min_ndim = params.adam.decay_min_ndim;
    const int n_accum = MAX(1, params.n_gradient_accumulation);
    const float accum_norm = 1.0f / (float) n_accum;

    float * g  = opt->adam.g->data;  // gradients
    float * m  = opt->adam.m->data;  // first moment
    float * v  = opt->adam.v->data;  // second moment

    float * pf = params.past > 0 ? opt->adam.pf->data : NULL; // past function values

    struct wsp_ggml_cplan cplan = wsp_ggml_graph_plan(gb, params.n_threads, NULL);
    struct wsp_ggml_object * obj = wsp_ggml_new_object(ctx, WSP_GGML_OBJECT_TYPE_WORK_BUFFER, cplan.work_size);
    cplan.work_data = (uint8_t *)ctx->mem_buffer + obj->offs;

    bool cancel = false;

    // compute the function value
    float fx = 0;
    wsp_ggml_set_zero(opt->adam.g);
    for (int accum_step = 0; accum_step < n_accum; ++accum_step) {
        if (callback) {
            callback(callback_data, accum_step, &sched, &cancel);
            if (cancel) {
                return WSP_GGML_OPT_RESULT_CANCEL;
            }
        }
        // wsp_ggml_graph_reset  (gf);
        wsp_ggml_set_f32      (f->grad, 1.0f);
        wsp_ggml_graph_compute(gb, &cplan);
        wsp_ggml_opt_acc_grad(np, ps, g, accum_norm);
        fx += wsp_ggml_get_f32_1d(f, 0);
    }
    fx *= accum_norm;

    opt->adam.fx_prev = fx;
    opt->adam.fx_best = opt->adam.fx_prev;
    if (pf) {
        pf[opt->iter % params.past] = opt->adam.fx_prev;
    }

    opt->loss_before = opt->adam.fx_prev;
    opt->loss_after  = opt->adam.fx_prev;

    // initialize
    if (opt->just_initialized) {
        opt->adam.n_no_improvement = 0;
        opt->just_initialized = false;
    }

    float * fx_best = &opt->adam.fx_best;
    float * fx_prev = &opt->adam.fx_prev;
    int * n_no_improvement = &opt->adam.n_no_improvement;

    int iter0 = opt->iter;

    // run the optimizer
    for (int t = 0; t < params.adam.n_iter; ++t) {
        opt->iter = iter0 + t + 1;
        WSP_GGML_PRINT_DEBUG  ("=== iter %d ===\n", t);

        WSP_GGML_PRINT_DEBUG  ("f      = %10.6f\n", wsp_ggml_get_f32_1d(f, 0));
        WSP_GGML_PRINT_DEBUG_5("df/dx0 = %10.6f\n", wsp_ggml_get_f32_1d(ps[0]->grad, 0));
        WSP_GGML_PRINT_DEBUG_5("df/dx1 = %10.6f\n", wsp_ggml_get_f32_1d(ps[1]->grad, 0));

        for (int i = 0; i < np; ++i) {
            WSP_GGML_PRINT_DEBUG("param %d: %10.6f, g = %10.6f\n", i,
                    wsp_ggml_get_f32_1d(ps[i], 0), wsp_ggml_get_f32_1d(ps[i]->grad, 0));
        }

        const int64_t t_start_wall = wsp_ggml_time_us();
        const int64_t t_start_cpu = wsp_ggml_cycles();
        UNUSED(t_start_wall);
        UNUSED(t_start_cpu);

        {
            float gnorm = 1.0f;
            if (gclip > 0.0f) {
                // gradient clipping
                wsp_ggml_float sum = 0.0;
                for (int64_t i = 0; i < nx; ++i) {
                    sum += (wsp_ggml_float)(g[i]*g[i]);
                }
                wsp_ggml_float norm = sqrt(sum);
                if (norm > (wsp_ggml_float) gclip) {
                    gnorm = (float) ((wsp_ggml_float) gclip / norm);
                }
            }
            const float beta1h = alpha*sched/(1.0f - powf(beta1, opt->iter));
            const float beta2h =        1.0f/(1.0f - powf(beta2, opt->iter));
            int64_t i = 0;
            for (int p = 0; p < np; ++p) {
                const int64_t ne = wsp_ggml_nelements(ps[p]);
                const float p_decay = ((wsp_ggml_n_dims(ps[p]) >= decay_min_ndim) ? decay : 0.0f) * sched;
                for (int64_t j = 0; j < ne; ++j) {
                    float x  = wsp_ggml_get_f32_1d(ps[p], j);
                    float g_ = g[i]*gnorm;
                    m[i] = m[i]*beta1 +    g_*(1.0f - beta1);
                    v[i] = v[i]*beta2 + g_*g_*(1.0f - beta2);
                    float mh = m[i]*beta1h;
                    float vh = v[i]*beta2h;
                    vh = sqrtf(vh) + eps;
                    x  = x*(1.0f - p_decay) - mh/vh;
                    wsp_ggml_set_f32_1d(ps[p], j, x);
                    ++i;
                }
            }
        }

        fx = 0;
        wsp_ggml_set_zero(opt->adam.g);
        for (int accum_step = 0; accum_step < n_accum; ++accum_step) {
            if (callback) {
                callback(callback_data, accum_step, &sched, &cancel);
                if (cancel) {
                    return WSP_GGML_OPT_RESULT_CANCEL;;
                }
            }
            // wsp_ggml_graph_reset  (gf);
            wsp_ggml_set_f32      (f->grad, 1.0f);
            wsp_ggml_graph_compute(gb, &cplan);
            wsp_ggml_opt_acc_grad(np, ps, g, accum_norm);
            fx += wsp_ggml_get_f32_1d(f, 0);
        }
        fx *= accum_norm;

        opt->loss_after = fx;

        // check convergence
        if (fabsf(fx - fx_prev[0])/fx < params.adam.eps_f) {
            WSP_GGML_PRINT_DEBUG("converged\n");

            return WSP_GGML_OPT_RESULT_OK;
        }

        // delta-based convergence test
        if (pf != NULL) {
            // need at least params.past iterations to start checking for convergence
            if (params.past <= iter0 + t) {
                const float rate = (pf[(iter0 + t)%params.past] - fx)/fx;

                if (fabsf(rate) < params.delta) {
                    return WSP_GGML_OPT_RESULT_OK;
                }
            }

            pf[(iter0 + t)%params.past] = fx;
        }

        // check for improvement
        if (params.max_no_improvement > 0) {
            if (fx_best[0] > fx) {
                fx_best[0] = fx;
                n_no_improvement[0] = 0;
            } else {
                ++n_no_improvement[0];

                if (n_no_improvement[0] >= params.max_no_improvement) {
                    return WSP_GGML_OPT_RESULT_OK;
                }
            }
        }

        fx_prev[0] = fx;

        {
            const int64_t t_end_cpu = wsp_ggml_cycles();
            WSP_GGML_PRINT_DEBUG("time iter:      %5.3f s\n", ((float)(t_end_cpu - t_start_cpu))/CLOCKS_PER_SEC);
            UNUSED(t_end_cpu);

            const int64_t t_end_wall = wsp_ggml_time_us();
            WSP_GGML_PRINT_DEBUG("wall time iter: %5.3f s\n", (t_end_wall - t_start_wall)/1e6);
            UNUSED(t_end_wall);
        }
    }

    return WSP_GGML_OPT_RESULT_DID_NOT_CONVERGE;
}

//
// L-BFGS
//
// the L-BFGS implementation below is based on the following implementation:
//
//   https://github.com/chokkan/liblbfgs
//

struct wsp_ggml_lbfgs_iteration_data {
    float alpha;
    float ys;
    float * s;
    float * y;
};

static enum wsp_ggml_opt_result linesearch_backtracking(
        const struct wsp_ggml_opt_params * params,
        int nx,
        float * x,
        float * fx,
        float * g,
        float * d,
        float * step,
        const float * xp,
        struct wsp_ggml_tensor * f,
        struct wsp_ggml_cgraph * gb,
        struct wsp_ggml_cplan  * cplan,
        const int np,
        struct wsp_ggml_tensor * ps[],
        bool * cancel,
        wsp_ggml_opt_callback callback,
        void * callback_data) {
    int count = 0;

    float width  = 0.0f;
    float dg     = 0.0f;
    float finit  = 0.0f;
    float dginit = 0.0f;
    float dgtest = 0.0f;

    const float dec = 0.5f;
    const float inc = 2.1f;

    const int n_accum = MAX(1, params->n_gradient_accumulation);
    const float accum_norm = 1.0f / (float) n_accum;

    if (*step <= 0.f) {
        return WSP_GGML_LINESEARCH_INVALID_PARAMETERS;
    }

    // compute the initial gradient in the search direction
    wsp_ggml_vec_dot_f32(nx, &dginit, 0, g, 0, d, 0, 1);

    // make sure that d points to a descent direction
    if (0 < dginit) {
        return WSP_GGML_LINESEARCH_FAIL;
    }

    // initialize local variables
    finit = *fx;
    dgtest = params->lbfgs.ftol*dginit;

    while (true) {
        wsp_ggml_vec_cpy_f32(nx, x, xp);
        wsp_ggml_vec_mad_f32(nx, x, d, *step);

        // evaluate the function and gradient values
        {
            wsp_ggml_opt_set_params(np, ps, x);

            *fx = 0;
            memset(g, 0, sizeof(float)*nx);
            for (int accum_step = 0; accum_step < n_accum; ++accum_step) {
                if (callback) {
                    // LBFG-S does not support learning rate -> ignore learning schedule
                    float sched = 0;
                    callback(callback_data, accum_step, &sched, cancel);
                    if (*cancel) {
                        return WSP_GGML_OPT_RESULT_CANCEL;
                    }
                }
                // wsp_ggml_graph_reset  (gf);
                wsp_ggml_set_f32      (f->grad, 1.0f);
                wsp_ggml_graph_compute(gb, cplan);
                wsp_ggml_opt_acc_grad(np, ps, g, accum_norm);
                *fx += wsp_ggml_get_f32_1d(f, 0);
            }
            *fx *= accum_norm;

        }

        ++count;

        if (*fx > finit + (*step)*dgtest) {
            width = dec;
        } else {
            // Armijo condition is satisfied
            if (params->lbfgs.linesearch == WSP_GGML_LINESEARCH_BACKTRACKING_ARMIJO) {
                return count;
            }

            wsp_ggml_vec_dot_f32(nx, &dg, 0, g, 0, d, 0, 1);

            // check the Wolfe condition
            if (dg < params->lbfgs.wolfe * dginit) {
                width = inc;
            } else {
                if(params->lbfgs.linesearch == WSP_GGML_LINESEARCH_BACKTRACKING_WOLFE) {
                    // regular Wolfe conditions
                    return count;
                }

                if(dg > -params->lbfgs.wolfe*dginit) {
                    width = dec;
                } else {
                    // strong Wolfe condition (WSP_GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE)
                    return count;
                }
            }
        }

        if (*step < params->lbfgs.min_step) {
            return WSP_GGML_LINESEARCH_MINIMUM_STEP;
        }
        if (*step > params->lbfgs.max_step) {
            return WSP_GGML_LINESEARCH_MAXIMUM_STEP;
        }
        if (params->lbfgs.max_linesearch <= count) {
            return WSP_GGML_LINESEARCH_MAXIMUM_ITERATIONS;
        }

        (*step) *= width;
    }

    WSP_GGML_ABORT("line search failed");

    //return WSP_GGML_LINESEARCH_FAIL;
}

static enum wsp_ggml_opt_result wsp_ggml_opt_lbfgs(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_opt_context * opt,
        struct wsp_ggml_opt_params params,
        struct wsp_ggml_tensor * f,
        struct wsp_ggml_cgraph * gf,
        struct wsp_ggml_cgraph * gb,
        wsp_ggml_opt_callback callback,
        void * callback_data) {
    if (params.lbfgs.linesearch == WSP_GGML_LINESEARCH_BACKTRACKING_WOLFE ||
        params.lbfgs.linesearch == WSP_GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE) {
        if (params.lbfgs.wolfe <= params.lbfgs.ftol || 1.f <= params.lbfgs.wolfe) {
            return WSP_GGML_OPT_RESULT_INVALID_WOLFE;
        }
    }

    const int m = params.lbfgs.m;

    // these will store the parameters we want to optimize
    struct wsp_ggml_tensor * ps[WSP_GGML_MAX_PARAMS];

    int np = 0;
    int nx = 0;
    for (int i = 0; i < gf->n_nodes; ++i) {
        if (gf->nodes[i]->flags & WSP_GGML_TENSOR_FLAG_PARAM) {
            WSP_GGML_PRINT_DEBUG("found param %d: grad->op = %d\n", np, gf->nodes[i]->grad->op);

            WSP_GGML_ASSERT(np < WSP_GGML_MAX_PARAMS);

            ps[np++] = gf->nodes[i];
            nx += wsp_ggml_nelements(gf->nodes[i]);
        }
    }

    if ((opt->params.type != params.type) || (opt->nx != nx) || (opt->params.past != params.past) || (opt->params.lbfgs.m != params.lbfgs.m)) {
        int iter = opt->iter;
        wsp_ggml_opt_init(ctx, opt, params, nx);
        opt->iter = iter;
    }

    struct wsp_ggml_cplan cplan = wsp_ggml_graph_plan(gb, params.n_threads, NULL);
    struct wsp_ggml_object * obj = wsp_ggml_new_object(ctx, WSP_GGML_OBJECT_TYPE_WORK_BUFFER, cplan.work_size);
    cplan.work_data = (uint8_t *)ctx->mem_buffer + obj->offs;

    float * x  = opt->lbfgs.x->data;  // current parameters
    float * xp = opt->lbfgs.xp->data; // previous parameters
    float * g  = opt->lbfgs.g->data;  // current gradient
    float * gp = opt->lbfgs.gp->data; // previous gradient
    float * d  = opt->lbfgs.d->data;  // search direction

    float * pf = params.past > 0 ? opt->lbfgs.pf->data : NULL; // past function values

    const int n_accum = MAX(1, params.n_gradient_accumulation);
    const float accum_norm = 1.0f / (float) n_accum;

    float fx    = 0.0f; // cost function value
    float xnorm = 0.0f; // ||x||
    float gnorm = 0.0f; // ||g||

    // initialize x from the graph nodes
    wsp_ggml_opt_get_params(np, ps, x);

    // the L-BFGS memory
    float * lm_alpha = opt->lbfgs.lmal->data;
    float * lm_ys    = opt->lbfgs.lmys->data;
    float * lm_s     = opt->lbfgs.lms->data;
    float * lm_y     = opt->lbfgs.lmy->data;

    bool cancel = false;

    // evaluate the function value and its gradient
    {
        wsp_ggml_opt_set_params(np, ps, x);

        fx = 0;
        memset(g, 0, sizeof(float)*nx);
        for (int accum_step = 0; accum_step < n_accum; ++accum_step) {
            if (callback) {
                // LBFG-S does not support learning rate -> ignore learning schedule
                float sched = 0;
                callback(callback_data, accum_step, &sched, &cancel);
                if (cancel) {
                    return WSP_GGML_OPT_RESULT_CANCEL;
                }
            }
            // wsp_ggml_graph_reset  (gf);
            wsp_ggml_set_f32      (f->grad, 1.0f);
            wsp_ggml_graph_compute(gb, &cplan);
            wsp_ggml_opt_acc_grad(np, ps, g, accum_norm);
            fx += wsp_ggml_get_f32_1d(f, 0);
        }
        fx *= accum_norm;

        opt->loss_before = fx;
        opt->loss_after  = fx;
    }

    // search direction = -gradient
    wsp_ggml_vec_neg_f32(nx, d, g);

    // ||x||, ||g||
    wsp_ggml_vec_norm_f32(nx, &xnorm, x);
    wsp_ggml_vec_norm_f32(nx, &gnorm, g);

    if (xnorm < 1.0f) {
        xnorm = 1.0f;
    }

    // already optimized
    if (gnorm/xnorm <= params.lbfgs.eps) {
        return WSP_GGML_OPT_RESULT_OK;
    }

    if (opt->just_initialized) {
        if (pf) {
            pf[0] = fx;
        }
        opt->lbfgs.fx_best = fx;

        // initial step
        wsp_ggml_vec_norm_inv_f32(nx, &opt->lbfgs.step, d);
        opt->lbfgs.j                = 0;
        opt->lbfgs.k                = 1;
        opt->lbfgs.end              = 0;
        opt->lbfgs.n_no_improvement = 0;
        opt->just_initialized       = false;
    }

    float * fx_best        = &opt->lbfgs.fx_best;
    float * step           = &opt->lbfgs.step;
    int * j                = &opt->lbfgs.j;
    int * k                = &opt->lbfgs.k;
    int * end              = &opt->lbfgs.end;
    int * n_no_improvement = &opt->lbfgs.n_no_improvement;

    int ls     = 0;
    int bound  = 0;

    float ys   = 0.0f;
    float yy   = 0.0f;
    float beta = 0.0f;

    int it = 0;

    while (true) {
        // store the current position and gradient vectors
        wsp_ggml_vec_cpy_f32(nx, xp, x);
        wsp_ggml_vec_cpy_f32(nx, gp, g);

        // TODO: instead of passing &cancel here, use the return code of the linesearch
        //       to determine if the optimization should be cancelled
        //       this is a simple change, but not doing this atm, since I don't have a nice
        //       way to test and don't want to break something with so many changes lined up
        ls = linesearch_backtracking(&params, nx, x, &fx, g, d, step, xp, f, gb, &cplan, np, ps, &cancel, callback, callback_data);
        if (cancel) {
            return WSP_GGML_OPT_RESULT_CANCEL;
        }

        if (ls < 0) {
            // linesearch failed - go back to the previous point and return
            wsp_ggml_vec_cpy_f32(nx, x, xp);
            wsp_ggml_vec_cpy_f32(nx, g, gp);

            return ls;
        }

        opt->loss_after = fx;

        wsp_ggml_vec_norm_f32(nx, &xnorm, x);
        wsp_ggml_vec_norm_f32(nx, &gnorm, g);

        WSP_GGML_PRINT_DEBUG("f = %10.6f\n", wsp_ggml_get_f32_1d(f, 0));

        if (xnorm < 1.0f) {
            xnorm = 1.0f;
        }
        if (gnorm/xnorm <= params.lbfgs.eps) {
            // converged
            return WSP_GGML_OPT_RESULT_OK;
        }

        // delta-based convergence test
        if (pf != NULL) {
            // need at least params.past iterations to start checking for convergence
            if (params.past <= k[0]) {
                const float rate = (pf[k[0]%params.past] - fx)/fx;

                if (fabsf(rate) < params.delta) {
                    return WSP_GGML_OPT_RESULT_OK;
                }
            }

            pf[k[0]%params.past] = fx;
        }

        // check for improvement
        if (params.max_no_improvement > 0) {
            if (fx < fx_best[0]) {
                fx_best[0] = fx;
                n_no_improvement[0] = 0;
            } else {
                n_no_improvement[0]++;

                if (n_no_improvement[0] >= params.max_no_improvement) {
                    return WSP_GGML_OPT_RESULT_OK;
                }
            }
        }

        if (params.lbfgs.n_iter != 0 && params.lbfgs.n_iter < it + 1) {
            // reached the maximum number of iterations
            return WSP_GGML_OPT_RESULT_DID_NOT_CONVERGE;
        }

        // update vectors s and y:
        //   s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
        //   y_{k+1} = g_{k+1} - g_{k}.
        //
        wsp_ggml_vec_sub_f32(nx, &lm_s[end[0]*nx], x, xp);
        wsp_ggml_vec_sub_f32(nx, &lm_y[end[0]*nx], g, gp);

        // compute scalars ys and yy:
        //     ys = y^t \cdot s    -> 1 / \rho.
        //     yy = y^t \cdot y.
        //
        wsp_ggml_vec_dot_f32(nx, &ys, 0, &lm_y[end[0]*nx], 0, &lm_s[end[0]*nx], 0, 1);
        wsp_ggml_vec_dot_f32(nx, &yy, 0, &lm_y[end[0]*nx], 0, &lm_y[end[0]*nx], 0, 1);

        lm_ys[end[0]] = ys;

        // find new search direction
        //   ref: https://en.wikipedia.org/wiki/Limited-memory_BFGS

        bound = (m <= k[0]) ? m : k[0];
        k[0]++;
        it++;
        end[0] = (end[0] + 1)%m;

        // initialize search direction with -g
        wsp_ggml_vec_neg_f32(nx, d, g);

        j[0] = end[0];
        for (int i = 0; i < bound; ++i) {
            j[0] = (j[0] + m - 1) % m;
            // \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}
            wsp_ggml_vec_dot_f32(nx, &lm_alpha[j[0]], 0, &lm_s[j[0]*nx], 0, d, 0, 1);
            lm_alpha[j[0]] /= lm_ys[j[0]];
            // q_{i} = q_{i+1} - \alpha_{i} y_{i}
            wsp_ggml_vec_mad_f32(nx, d, &lm_y[j[0]*nx], -lm_alpha[j[0]]);
        }

        wsp_ggml_vec_scale_f32(nx, d, ys/yy);

        for (int i = 0; i < bound; ++i) {
            // \beta_{j} = \rho_{j} y^t_{j} \cdot \gamma_{i}
            wsp_ggml_vec_dot_f32(nx, &beta, 0, &lm_y[j[0]*nx], 0, d, 0, 1);
            beta /= lm_ys[j[0]];
            // \gamma_{i+1} = \gamma_{i} + (\alpha_{j} - \beta_{j}) s_{j}
            wsp_ggml_vec_mad_f32(nx, d, &lm_s[j[0]*nx], lm_alpha[j[0]] - beta);
            j[0] = (j[0] + 1)%m;
        }

        step[0] = 1.0;
    }

    WSP_GGML_ABORT("lbfgs failed");

    //return WSP_GGML_OPT_RESULT_DID_NOT_CONVERGE;
}

struct wsp_ggml_opt_params wsp_ggml_opt_default_params(enum wsp_ggml_opt_type type) {
    struct wsp_ggml_opt_params result;

    switch (type) {
        case WSP_GGML_OPT_TYPE_ADAM:
            {
                result = (struct wsp_ggml_opt_params) {
                    .type       = WSP_GGML_OPT_TYPE_ADAM,
                    .graph_size = WSP_GGML_DEFAULT_GRAPH_SIZE,
                    .n_threads  = 1, // FIXME: WSP_GGML_DEFAULT_N_THREADS ?
                    .past       = 0,
                    .delta      = 1e-5f,

                    .max_no_improvement = 100,

                    .print_forward_graph  = true,
                    .print_backward_graph = true,

                    .n_gradient_accumulation = 1,

                    .adam = {
                        .n_iter = 10000,
                        .sched  = 1.000f,
                        .decay  = 0.0f,
                        .decay_min_ndim = 2,
                        .alpha  = 0.001f,
                        .beta1  = 0.9f,
                        .beta2  = 0.999f,
                        .eps    = 1e-8f,
                        .eps_f  = 1e-5f,
                        .eps_g  = 1e-3f,
                        .gclip  = 0.0f,
                    },
                };
            } break;
        case WSP_GGML_OPT_TYPE_LBFGS:
            {
                result = (struct wsp_ggml_opt_params) {
                    .type       = WSP_GGML_OPT_TYPE_LBFGS,
                    .graph_size = WSP_GGML_DEFAULT_GRAPH_SIZE,
                    .n_threads  = 1,
                    .past       = 0,
                    .delta      = 1e-5f,

                    .max_no_improvement = 0,

                    .print_forward_graph  = true,
                    .print_backward_graph = true,

                    .n_gradient_accumulation = 1,

                    .lbfgs = {
                        .m              = 6,
                        .n_iter         = 100,
                        .max_linesearch = 20,

                        .eps      = 1e-5f,
                        .ftol     = 1e-4f,
                        .wolfe    = 0.9f,
                        .min_step = 1e-20f,
                        .max_step = 1e+20f,

                        .linesearch = WSP_GGML_LINESEARCH_DEFAULT,
                    },
                };
            } break;
    }

    return result;
}

WSP_GGML_API void wsp_ggml_opt_init(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_opt_context * opt,
        struct wsp_ggml_opt_params params,
        int64_t nx) {
    opt->ctx = ctx;
    opt->params = params;
    opt->iter = 0;
    opt->nx = nx;
    opt->just_initialized = true;
    if (opt->ctx == NULL) {
        struct wsp_ggml_init_params ctx_opt_params;
        if (opt->params.type == WSP_GGML_OPT_TYPE_ADAM) {
            ctx_opt_params.mem_size = WSP_GGML_MEM_ALIGN*3 + wsp_ggml_tensor_overhead()*3 + wsp_ggml_type_size(WSP_GGML_TYPE_F32)*nx*3;
            if (opt->params.past > 0) {
                ctx_opt_params.mem_size += WSP_GGML_MEM_ALIGN + wsp_ggml_tensor_overhead() + wsp_ggml_type_size(WSP_GGML_TYPE_F32)*opt->params.past;
            }
        } else if (opt->params.type == WSP_GGML_OPT_TYPE_LBFGS) {
            ctx_opt_params.mem_size = WSP_GGML_MEM_ALIGN*9 + wsp_ggml_tensor_overhead()*9 + wsp_ggml_type_size(WSP_GGML_TYPE_F32)*(nx*5 + opt->params.lbfgs.m*2 + nx*opt->params.lbfgs.m*2);
            if (opt->params.past > 0) {
                ctx_opt_params.mem_size += WSP_GGML_MEM_ALIGN + wsp_ggml_tensor_overhead() + wsp_ggml_type_size(WSP_GGML_TYPE_F32)*opt->params.past;
            }
        }
        ctx_opt_params.mem_buffer = NULL;
        ctx_opt_params.no_alloc   = false;

        opt->ctx = wsp_ggml_init(ctx_opt_params);
    }
    switch (opt->params.type) {
        case WSP_GGML_OPT_TYPE_ADAM:
            {
                opt->adam.g  = wsp_ggml_new_tensor_1d(opt->ctx, WSP_GGML_TYPE_F32, nx);
                opt->adam.m  = wsp_ggml_new_tensor_1d(opt->ctx, WSP_GGML_TYPE_F32, nx);
                opt->adam.v  = wsp_ggml_new_tensor_1d(opt->ctx, WSP_GGML_TYPE_F32, nx);
                opt->adam.pf = params.past > 0
                    ? wsp_ggml_new_tensor_1d(opt->ctx, WSP_GGML_TYPE_F32, params.past)
                    : NULL;
                wsp_ggml_set_zero(opt->adam.m);
                wsp_ggml_set_zero(opt->adam.v);
                if (opt->adam.pf) {
                    wsp_ggml_set_zero(opt->adam.pf);
                }
            } break;
        case WSP_GGML_OPT_TYPE_LBFGS:
            {
                opt->lbfgs.x  = wsp_ggml_new_tensor_1d(opt->ctx, WSP_GGML_TYPE_F32, nx);
                opt->lbfgs.xp = wsp_ggml_new_tensor_1d(opt->ctx, WSP_GGML_TYPE_F32, nx);
                opt->lbfgs.g  = wsp_ggml_new_tensor_1d(opt->ctx, WSP_GGML_TYPE_F32, nx);
                opt->lbfgs.gp = wsp_ggml_new_tensor_1d(opt->ctx, WSP_GGML_TYPE_F32, nx);
                opt->lbfgs.d  = wsp_ggml_new_tensor_1d(opt->ctx, WSP_GGML_TYPE_F32, nx);
                opt->lbfgs.pf = params.past > 0
                    ? wsp_ggml_new_tensor_1d(opt->ctx, WSP_GGML_TYPE_F32, params.past)
                    : NULL;
                opt->lbfgs.lmal = wsp_ggml_new_tensor_1d(opt->ctx, WSP_GGML_TYPE_F32, params.lbfgs.m);
                opt->lbfgs.lmys = wsp_ggml_new_tensor_1d(opt->ctx, WSP_GGML_TYPE_F32, params.lbfgs.m);
                opt->lbfgs.lms  = wsp_ggml_new_tensor_2d(opt->ctx, WSP_GGML_TYPE_F32, nx, params.lbfgs.m);
                opt->lbfgs.lmy  = wsp_ggml_new_tensor_2d(opt->ctx, WSP_GGML_TYPE_F32, nx, params.lbfgs.m);
                wsp_ggml_set_zero(opt->lbfgs.x);
                wsp_ggml_set_zero(opt->lbfgs.xp);
                wsp_ggml_set_zero(opt->lbfgs.g);
                wsp_ggml_set_zero(opt->lbfgs.gp);
                wsp_ggml_set_zero(opt->lbfgs.d);
                if (opt->lbfgs.pf) {
                    wsp_ggml_set_zero(opt->lbfgs.pf);
                }
                wsp_ggml_set_zero(opt->lbfgs.lmal);
                wsp_ggml_set_zero(opt->lbfgs.lmys);
                wsp_ggml_set_zero(opt->lbfgs.lms);
                wsp_ggml_set_zero(opt->lbfgs.lmy);
            } break;
    }
}

enum wsp_ggml_opt_result wsp_ggml_opt(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_opt_params params,
        struct wsp_ggml_tensor * f) {
    bool free_ctx = false;
    if (ctx == NULL) {
        struct wsp_ggml_init_params params_ctx = {
            .mem_size   = 16*1024*1024,
            .mem_buffer = NULL,
            .no_alloc   = false,
        };

        ctx = wsp_ggml_init(params_ctx);
        if (ctx == NULL) {
            return WSP_GGML_OPT_RESULT_NO_CONTEXT;
        }

        free_ctx = true;
    }

    enum wsp_ggml_opt_result result = WSP_GGML_OPT_RESULT_OK;

    struct wsp_ggml_opt_context * opt = (struct wsp_ggml_opt_context *) alloca(sizeof(struct wsp_ggml_opt_context));

    wsp_ggml_opt_init(ctx, opt, params, 0);
    result = wsp_ggml_opt_resume(ctx, opt, f);

    if (free_ctx) {
        wsp_ggml_free(ctx);
    }

    return result;
}

enum wsp_ggml_opt_result wsp_ggml_opt_resume(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_opt_context * opt,
        struct wsp_ggml_tensor * f) {

    // build forward + backward compute graphs
    struct wsp_ggml_cgraph * gf = wsp_ggml_new_graph_custom(ctx, opt->params.graph_size, true);
    wsp_ggml_build_forward_expand(gf, f);

    struct wsp_ggml_cgraph * gb = wsp_ggml_graph_dup(ctx, gf);
    wsp_ggml_build_backward_expand(ctx, gf, gb, false);

    return wsp_ggml_opt_resume_g(ctx, opt, f, gf, gb, NULL, NULL);
}

enum wsp_ggml_opt_result wsp_ggml_opt_resume_g(
        struct wsp_ggml_context * ctx,
        struct wsp_ggml_opt_context * opt,
        struct wsp_ggml_tensor * f,
        struct wsp_ggml_cgraph * gf,
        struct wsp_ggml_cgraph * gb,
        wsp_ggml_opt_callback callback,
        void * callback_data) {

    WSP_GGML_ASSERT(f->grad && "wsp_ggml_set_param must be called for at least one ancestor");

    // build forward + backward compute graphs
    enum wsp_ggml_opt_result result = WSP_GGML_OPT_RESULT_OK;

    switch (opt->params.type) {
        case WSP_GGML_OPT_TYPE_ADAM:
            {
                result = wsp_ggml_opt_adam(ctx, opt, opt->params, f, gf, gb, callback, callback_data);
            } break;
        case WSP_GGML_OPT_TYPE_LBFGS:
            {
                result = wsp_ggml_opt_lbfgs(ctx, opt, opt->params, f, gf, gb, callback, callback_data);
            } break;
    }

    if (opt->params.print_forward_graph) {
        wsp_ggml_graph_print   (gf);
        wsp_ggml_graph_dump_dot(gf, NULL, "opt-forward.dot");
    }

    if (opt->params.print_backward_graph) {
        wsp_ggml_graph_print   (gb);
        wsp_ggml_graph_dump_dot(gb, gf, "opt-backward.dot");
    }

    return result;
}

////////////////////////////////////////////////////////////////////////////////

void wsp_ggml_set_input(struct wsp_ggml_tensor * tensor) {
    tensor->flags |= WSP_GGML_TENSOR_FLAG_INPUT;
}

void wsp_ggml_set_output(struct wsp_ggml_tensor * tensor) {
    tensor->flags |= WSP_GGML_TENSOR_FLAG_OUTPUT;
}

void wsp_ggml_set_param(struct wsp_ggml_context * ctx, struct wsp_ggml_tensor * tensor) {
    WSP_GGML_UNUSED(ctx); // TODO: remove this parameter
    tensor->flags |= WSP_GGML_TENSOR_FLAG_PARAM;
}

void wsp_ggml_set_loss(struct wsp_ggml_tensor * tensor) {
    WSP_GGML_ASSERT(wsp_ggml_is_scalar(tensor));
    WSP_GGML_ASSERT(tensor->type == WSP_GGML_TYPE_F32);
    tensor->flags |= WSP_GGML_TENSOR_FLAG_LOSS;
}

////////////////////////////////////////////////////////////////////////////////

void wsp_ggml_wsp_quantize_init(enum wsp_ggml_type type) {
    wsp_ggml_critical_section_start();

    switch (type) {
        case WSP_GGML_TYPE_IQ2_XXS:
        case WSP_GGML_TYPE_IQ2_XS:
        case WSP_GGML_TYPE_IQ2_S:
        case WSP_GGML_TYPE_IQ1_S:
        case WSP_GGML_TYPE_IQ1_M:   iq2xs_init_impl(type); break;
        case WSP_GGML_TYPE_IQ3_XXS: iq3xs_init_impl(256); break;
        case WSP_GGML_TYPE_IQ3_S:   iq3xs_init_impl(512); break;
        default: // nothing
            break;
    }

    wsp_ggml_critical_section_end();
}

void wsp_ggml_wsp_quantize_free(void) {
    wsp_ggml_critical_section_start();

    iq2xs_free_impl(WSP_GGML_TYPE_IQ2_XXS);
    iq2xs_free_impl(WSP_GGML_TYPE_IQ2_XS);
    iq2xs_free_impl(WSP_GGML_TYPE_IQ1_S);
    iq3xs_free_impl(256);

    wsp_ggml_critical_section_end();
}

bool wsp_ggml_wsp_quantize_requires_imatrix(enum wsp_ggml_type type) {
    return
        type == WSP_GGML_TYPE_IQ2_XXS ||
        type == WSP_GGML_TYPE_IQ2_XS  ||
        type == WSP_GGML_TYPE_IQ1_S;//   ||
        //type == WSP_GGML_TYPE_IQ1_M;
}

size_t wsp_ggml_wsp_quantize_chunk(
        enum wsp_ggml_type   type,
           const float * src,
                  void * dst,
               int64_t   start,
               int64_t   nrows,
               int64_t   n_per_row,
           const float * imatrix) {
    const int64_t n = (int64_t) nrows * n_per_row;

    if (wsp_ggml_wsp_quantize_requires_imatrix(type)) {
        WSP_GGML_ASSERT(imatrix != NULL);
    }

    WSP_GGML_ASSERT(start % type_traits[type].blck_size == 0);
    WSP_GGML_ASSERT(start % n_per_row == 0);

    wsp_ggml_wsp_quantize_init(type); // this is noop if already initialized

    const size_t start_row = start / n_per_row;
    const size_t row_size  = wsp_ggml_row_size(type, n_per_row);

    size_t result = 0;

    switch (type) {
        case WSP_GGML_TYPE_Q4_0:    result = wsp_quantize_q4_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_Q4_1:    result = wsp_quantize_q4_1(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_Q5_0:    result = wsp_quantize_q5_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_Q5_1:    result = wsp_quantize_q5_1(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_Q8_0:    result = wsp_quantize_q8_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_Q2_K:    result = wsp_quantize_q2_K(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_Q3_K:    result = wsp_quantize_q3_K(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_Q4_K:    result = wsp_quantize_q4_K(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_Q5_K:    result = wsp_quantize_q5_K(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_Q6_K:    result = wsp_quantize_q6_K(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_TQ1_0:   result = wsp_quantize_tq1_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_TQ2_0:   result = wsp_quantize_tq2_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_IQ2_XXS: result = wsp_quantize_iq2_xxs(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_IQ2_XS:  result = wsp_quantize_iq2_xs (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_IQ3_XXS: result = wsp_quantize_iq3_xxs(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_IQ3_S:   result = wsp_quantize_iq3_s  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_IQ2_S:   result = wsp_quantize_iq2_s  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_IQ1_S:   result = wsp_quantize_iq1_s  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_IQ1_M:   result = wsp_quantize_iq1_m  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_IQ4_NL:  result = wsp_quantize_iq4_nl (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_IQ4_XS:  result = wsp_quantize_iq4_xs (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_Q4_0_4_4: result = wsp_quantize_q4_0_4x4(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_Q4_0_4_8: result = wsp_quantize_q4_0_4x8(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_Q4_0_8_8: result = wsp_quantize_q4_0_8x8(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case WSP_GGML_TYPE_F16:
            {
                size_t elemsize = sizeof(wsp_ggml_fp16_t);
                wsp_ggml_fp32_to_fp16_row(src + start, (wsp_ggml_fp16_t *)dst + start, n);
                result = n * elemsize;
            } break;
        case WSP_GGML_TYPE_BF16:
            {
                size_t elemsize = sizeof(wsp_ggml_bf16_t);
                wsp_ggml_fp32_to_bf16_row_ref(src + start, (wsp_ggml_bf16_t *)dst + start, n);
                result = n * elemsize;
            } break;
        case WSP_GGML_TYPE_F32:
            {
                size_t elemsize = sizeof(float);
                result = n * elemsize;
                memcpy((uint8_t *)dst + start * elemsize, src + start, result);
            } break;
        default:
            assert(false);
    }

    WSP_GGML_ASSERT(result == nrows * row_size);

    return result;
}

////////////////////////////////////////////////////////////////////////////////

struct wsp_gguf_str {
    uint64_t n;  // GGUFv2
    char * data;
};

static const size_t WSP_GGUF_TYPE_SIZE[WSP_GGUF_TYPE_COUNT] = {
    [WSP_GGUF_TYPE_UINT8]   = sizeof(uint8_t),
    [WSP_GGUF_TYPE_INT8]    = sizeof(int8_t),
    [WSP_GGUF_TYPE_UINT16]  = sizeof(uint16_t),
    [WSP_GGUF_TYPE_INT16]   = sizeof(int16_t),
    [WSP_GGUF_TYPE_UINT32]  = sizeof(uint32_t),
    [WSP_GGUF_TYPE_INT32]   = sizeof(int32_t),
    [WSP_GGUF_TYPE_FLOAT32] = sizeof(float),
    [WSP_GGUF_TYPE_BOOL]    = sizeof(bool),
    [WSP_GGUF_TYPE_STRING]  = sizeof(struct wsp_gguf_str),
    [WSP_GGUF_TYPE_UINT64]  = sizeof(uint64_t),
    [WSP_GGUF_TYPE_INT64]   = sizeof(int64_t),
    [WSP_GGUF_TYPE_FLOAT64] = sizeof(double),
    [WSP_GGUF_TYPE_ARRAY]   = 0, // undefined
};
static_assert(WSP_GGUF_TYPE_COUNT == 13, "WSP_GGUF_TYPE_COUNT != 13");

static const char * WSP_GGUF_TYPE_NAME[WSP_GGUF_TYPE_COUNT] = {
    [WSP_GGUF_TYPE_UINT8]   = "u8",
    [WSP_GGUF_TYPE_INT8]    = "i8",
    [WSP_GGUF_TYPE_UINT16]  = "u16",
    [WSP_GGUF_TYPE_INT16]   = "i16",
    [WSP_GGUF_TYPE_UINT32]  = "u32",
    [WSP_GGUF_TYPE_INT32]   = "i32",
    [WSP_GGUF_TYPE_FLOAT32] = "f32",
    [WSP_GGUF_TYPE_BOOL]    = "bool",
    [WSP_GGUF_TYPE_STRING]  = "str",
    [WSP_GGUF_TYPE_ARRAY]   = "arr",
    [WSP_GGUF_TYPE_UINT64]  = "u64",
    [WSP_GGUF_TYPE_INT64]   = "i64",
    [WSP_GGUF_TYPE_FLOAT64] = "f64",
};
static_assert(WSP_GGUF_TYPE_COUNT == 13, "WSP_GGUF_TYPE_COUNT != 13");

union wsp_gguf_value {
    uint8_t  uint8;
    int8_t   int8;
    uint16_t uint16;
    int16_t  int16;
    uint32_t uint32;
    int32_t  int32;
    float    float32;
    uint64_t uint64;
    int64_t  int64;
    double   float64;
    bool     bool_;

    struct wsp_gguf_str str;

    struct {
        enum wsp_gguf_type type;

        uint64_t n;  // GGUFv2
        void * data;
    } arr;
};

struct wsp_gguf_kv {
    struct wsp_gguf_str key;

    enum  wsp_gguf_type  type;
    union wsp_gguf_value value;
};

struct wsp_gguf_header {
    char magic[4];

    uint32_t version;
    uint64_t n_tensors; // GGUFv2
    uint64_t n_kv;      // GGUFv2
};

struct wsp_gguf_tensor_info {
    struct wsp_gguf_str name;

    uint32_t n_dims;
    uint64_t ne[WSP_GGML_MAX_DIMS];

    enum wsp_ggml_type type;

    uint64_t offset; // offset from start of `data`, must be a multiple of `ALIGNMENT`

    // for writing API
    const void * data;
    size_t size;
};

struct wsp_gguf_context {
    struct wsp_gguf_header header;

    struct wsp_gguf_kv          * kv;
    struct wsp_gguf_tensor_info * infos;

    size_t alignment;
    size_t offset;    // offset of `data` from beginning of file
    size_t size;      // size of `data` in bytes

    //uint8_t * padding;
    void * data;
};

static size_t wsp_gguf_type_size(enum wsp_gguf_type type) {
    WSP_GGML_ASSERT(0 <= type && type < WSP_GGUF_TYPE_COUNT);
    return WSP_GGUF_TYPE_SIZE[type];
}

static void wsp_gguf_tensor_info_sanitize(struct wsp_gguf_tensor_info * info) {
    WSP_GGML_ASSERT(info->n_dims <= WSP_GGML_MAX_DIMS);
    WSP_GGML_ASSERT(0 <= info->type && info->type < WSP_GGML_TYPE_COUNT);

    for (uint32_t i = 0; i < info->n_dims; ++i) {
        WSP_GGML_ASSERT(info->ne[i] > 0);
    }

    // prevent overflow for total number of elements
    WSP_GGML_ASSERT(INT64_MAX/info->ne[1] > info->ne[0]);
    WSP_GGML_ASSERT(INT64_MAX/info->ne[2] > info->ne[0]*info->ne[1]);
    WSP_GGML_ASSERT(INT64_MAX/info->ne[3] > info->ne[0]*info->ne[1]*info->ne[2]);
}

static bool wsp_gguf_fread_el(FILE * file, void * dst, size_t size, size_t * offset) {
    const size_t n = fread(dst, 1, size, file);
    *offset += n;
    return n == size;
}

static bool wsp_gguf_fread_str(FILE * file, struct wsp_gguf_str * p, size_t * offset) {
    p->n    = 0;
    p->data = NULL;

    bool ok = true;

    ok = ok && wsp_gguf_fread_el(file, &p->n, sizeof(p->n), offset);

    // early exit if string length is invalid, prevents from integer overflow
    if (p->n == SIZE_MAX) {
        fprintf(stderr, "%s: invalid string length (%" PRIu64 ")\n", __func__, p->n);
        return false;
    }

    p->data = WSP_GGML_CALLOC(p->n + 1, 1);

    ok = ok && wsp_gguf_fread_el(file,  p->data, p->n, offset);

    return ok;
}

static void wsp_gguf_free_kv(struct wsp_gguf_kv * kv) {
    if (kv->key.data) {
        WSP_GGML_FREE(kv->key.data);
    }

    if (kv->type == WSP_GGUF_TYPE_STRING) {
        if (kv->value.str.data) {
            WSP_GGML_FREE(kv->value.str.data);
        }
    }

    if (kv->type == WSP_GGUF_TYPE_ARRAY) {
        if (kv->value.arr.data) {
            if (kv->value.arr.type == WSP_GGUF_TYPE_STRING) {
                for (uint64_t j = 0; j < kv->value.arr.n; ++j) {
                    struct wsp_gguf_str * str = &((struct wsp_gguf_str *) kv->value.arr.data)[j];
                    if (str->data) {
                        WSP_GGML_FREE(str->data);
                    }
                }
            }
            WSP_GGML_FREE(kv->value.arr.data);
        }
    }
}

struct wsp_gguf_context * wsp_gguf_init_empty(void) {
    struct wsp_gguf_context * ctx = WSP_GGML_CALLOC(1, sizeof(struct wsp_gguf_context));

    memcpy(ctx->header.magic, WSP_GGUF_MAGIC, sizeof(ctx->header.magic));
    ctx->header.version   = WSP_GGUF_VERSION;
    ctx->header.n_tensors = 0;
    ctx->header.n_kv      = 0;

    ctx->kv    = NULL;
    ctx->infos = NULL;

    ctx->alignment = WSP_GGUF_DEFAULT_ALIGNMENT;
    ctx->offset    = 0;
    ctx->size      = 0;

    ctx->data = NULL;

    return ctx;
}

struct wsp_gguf_context * wsp_gguf_init_from_file(const char * fname, struct wsp_gguf_init_params params) {
    FILE * file = wsp_ggml_fopen(fname, "rb");
    if (!file) {
        fprintf(stderr, "%s: failed to open '%s': '%s'\n", __func__, fname, strerror(errno));
        return NULL;
    }

    // offset from start of file
    size_t offset = 0;

    char magic[4];

    // check the magic before making allocations
    {
        wsp_gguf_fread_el(file, &magic, sizeof(magic), &offset);

        for (uint32_t i = 0; i < sizeof(magic); i++) {
            if (magic[i] != WSP_GGUF_MAGIC[i]) {
                fprintf(stderr, "%s: invalid magic characters '%c%c%c%c'\n", __func__, magic[0], magic[1], magic[2], magic[3]);
                fclose(file);
                return NULL;
            }
        }
    }

    bool ok = true;

    struct wsp_gguf_context * ctx = WSP_GGML_CALLOC(1, sizeof(struct wsp_gguf_context));

    // read the header
    {
        strncpy(ctx->header.magic, magic, 4);

        ctx->kv    = NULL;
        ctx->infos = NULL;
        ctx->data  = NULL;

        ok = ok && wsp_gguf_fread_el(file, &ctx->header.version,   sizeof(ctx->header.version),   &offset);
        ok = ok && wsp_gguf_fread_el(file, &ctx->header.n_tensors, sizeof(ctx->header.n_tensors), &offset);
        ok = ok && wsp_gguf_fread_el(file, &ctx->header.n_kv,      sizeof(ctx->header.n_kv),      &offset);

        if (ctx->header.version == 1) {
            fprintf(stderr, "%s: GGUFv1 is no longer supported. please use a more up-to-date version\n", __func__);
            fclose(file);
            wsp_gguf_free(ctx);
            return NULL;
        }

        // sanity-checks to prevent from integer/buffer overflows

        ok = ok && (ctx->header.n_tensors < (SIZE_MAX/2)/sizeof(struct wsp_gguf_tensor_info));
        ok = ok && (ctx->header.n_tensors < (SIZE_MAX/2)/wsp_ggml_tensor_overhead());
        ok = ok && (ctx->header.n_kv      < (SIZE_MAX/2)/sizeof(struct wsp_gguf_kv));

        if (!ok) {
            fprintf(stderr, "%s: failed to read header\n", __func__);
            fclose(file);
            wsp_gguf_free(ctx);
            return NULL;
        }
    }

    // read the kv pairs
    {
        const uint64_t n_kv = ctx->header.n_kv;

        // header.n_kv will hold the actual value of pairs that were successfully read in the loop below
        ctx->header.n_kv = 0;
        ctx->kv = WSP_GGML_CALLOC(n_kv, sizeof(struct wsp_gguf_kv));

        for (uint64_t i = 0; i < n_kv; ++i) {
            struct wsp_gguf_kv * kv = &ctx->kv[i];

            //fprintf(stderr, "%s: reading kv %d\n", __func__, i);

            ok = ok && wsp_gguf_fread_str(file, &kv->key,                    &offset);
            ok = ok && wsp_gguf_fread_el (file, &kv->type, sizeof(kv->type), &offset);

            //fprintf(stderr, "%s: reading kv with key %s\n", __func__, kv->key.data);

            switch (kv->type) {
                case WSP_GGUF_TYPE_UINT8:   ok = ok && wsp_gguf_fread_el (file, &kv->value.uint8,   sizeof(kv->value.uint8),   &offset); break;
                case WSP_GGUF_TYPE_INT8:    ok = ok && wsp_gguf_fread_el (file, &kv->value.int8,    sizeof(kv->value.int8),    &offset); break;
                case WSP_GGUF_TYPE_UINT16:  ok = ok && wsp_gguf_fread_el (file, &kv->value.uint16,  sizeof(kv->value.uint16),  &offset); break;
                case WSP_GGUF_TYPE_INT16:   ok = ok && wsp_gguf_fread_el (file, &kv->value.int16,   sizeof(kv->value.int16),   &offset); break;
                case WSP_GGUF_TYPE_UINT32:  ok = ok && wsp_gguf_fread_el (file, &kv->value.uint32,  sizeof(kv->value.uint32),  &offset); break;
                case WSP_GGUF_TYPE_INT32:   ok = ok && wsp_gguf_fread_el (file, &kv->value.int32,   sizeof(kv->value.int32),   &offset); break;
                case WSP_GGUF_TYPE_FLOAT32: ok = ok && wsp_gguf_fread_el (file, &kv->value.float32, sizeof(kv->value.float32), &offset); break;
                case WSP_GGUF_TYPE_UINT64:  ok = ok && wsp_gguf_fread_el (file, &kv->value.uint64,  sizeof(kv->value.uint64),  &offset); break;
                case WSP_GGUF_TYPE_INT64:   ok = ok && wsp_gguf_fread_el (file, &kv->value.int64,   sizeof(kv->value.int64),   &offset); break;
                case WSP_GGUF_TYPE_FLOAT64: ok = ok && wsp_gguf_fread_el (file, &kv->value.float64, sizeof(kv->value.float64), &offset); break;
                case WSP_GGUF_TYPE_BOOL:    ok = ok && wsp_gguf_fread_el (file, &kv->value.bool_,   sizeof(kv->value.bool_),   &offset); break;
                case WSP_GGUF_TYPE_STRING:  ok = ok && wsp_gguf_fread_str(file, &kv->value.str,                                &offset); break;
                case WSP_GGUF_TYPE_ARRAY:
                    {
                        ok = ok && wsp_gguf_fread_el(file, &kv->value.arr.type, sizeof(kv->value.arr.type), &offset);
                        ok = ok && wsp_gguf_fread_el(file, &kv->value.arr.n,    sizeof(kv->value.arr.n),    &offset);

                        switch (kv->value.arr.type) {
                            case WSP_GGUF_TYPE_UINT8:
                            case WSP_GGUF_TYPE_INT8:
                            case WSP_GGUF_TYPE_UINT16:
                            case WSP_GGUF_TYPE_INT16:
                            case WSP_GGUF_TYPE_UINT32:
                            case WSP_GGUF_TYPE_INT32:
                            case WSP_GGUF_TYPE_FLOAT32:
                            case WSP_GGUF_TYPE_UINT64:
                            case WSP_GGUF_TYPE_INT64:
                            case WSP_GGUF_TYPE_FLOAT64:
                            case WSP_GGUF_TYPE_BOOL:
                                {
                                    // prevent from integer overflow in the malloc below
                                    if (kv->value.arr.n >= SIZE_MAX/wsp_gguf_type_size(kv->value.arr.type)) {
                                        fprintf(stderr, "%s: array size is too large (%" PRIu64 ")\n", __func__, kv->value.arr.n);
                                        fclose(file);
                                        wsp_gguf_free(ctx);
                                        return NULL;
                                    }

                                    kv->value.arr.data = WSP_GGML_CALLOC(kv->value.arr.n, wsp_gguf_type_size(kv->value.arr.type));

                                    ok = ok && wsp_gguf_fread_el(file, kv->value.arr.data, kv->value.arr.n * wsp_gguf_type_size(kv->value.arr.type), &offset);
                                } break;
                            case WSP_GGUF_TYPE_STRING:
                                {
                                    // prevent from integer overflow in the malloc below
                                    if (kv->value.arr.n >= SIZE_MAX/sizeof(struct wsp_gguf_str)) {
                                        fprintf(stderr, "%s: array size is too large (%" PRIu64 ")\n", __func__, kv->value.arr.n);
                                        fclose(file);
                                        wsp_gguf_free(ctx);
                                        return NULL;
                                    }

                                    kv->value.arr.data = WSP_GGML_CALLOC(kv->value.arr.n, sizeof(struct wsp_gguf_str));

                                    for (uint64_t j = 0; j < kv->value.arr.n; ++j) {
                                        ok = ok && wsp_gguf_fread_str(file, &((struct wsp_gguf_str *) kv->value.arr.data)[j], &offset);
                                    }
                                } break;
                            case WSP_GGUF_TYPE_ARRAY:
                            default: WSP_GGML_ABORT("invalid type");
                        }
                    } break;
                default: WSP_GGML_ABORT("invalid type");
            }

            if (!ok) {
                break;
            }

            ctx->header.n_kv++;
        }

        if (!ok) {
            fprintf(stderr, "%s: failed to read key-value pairs\n", __func__);
            fclose(file);
            wsp_gguf_free(ctx);
            return NULL;
        }
    }

    // read the tensor infos
    if (ctx->header.n_tensors > 0) {
        ctx->infos = WSP_GGML_CALLOC(ctx->header.n_tensors, sizeof(struct wsp_gguf_tensor_info));

        for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
            struct wsp_gguf_tensor_info * info = &ctx->infos[i];

            for (int j = 0; j < WSP_GGML_MAX_DIMS; ++j) {
                info->ne[j] = 1;
            }

            ok = ok && wsp_gguf_fread_str(file, &info->name,                          &offset);
            ok = ok && wsp_gguf_fread_el (file, &info->n_dims, sizeof(info->n_dims),  &offset);

            ok = ok && (info->n_dims <= WSP_GGML_MAX_DIMS);

            for (uint32_t j = 0; j < info->n_dims; ++j) {
                ok = ok && wsp_gguf_fread_el(file, &info->ne[j], sizeof(info->ne[j]), &offset);
            }

            ok = ok && wsp_gguf_fread_el (file, &info->type,   sizeof(info->type),    &offset);
            ok = ok && wsp_gguf_fread_el (file, &info->offset, sizeof(info->offset),  &offset);

            // TODO: return an error instead of crashing with WSP_GGML_ASSERT
            wsp_gguf_tensor_info_sanitize(info);

            // make sure there is no duplicated tensor names
            for (uint64_t j = 0; j < i && ok; ++j) {
                if (strcmp(info->name.data, ctx->infos[j].name.data) == 0) {
                    fprintf(stderr, "%s: duplicated tensor name %s\n", __func__, info->name.data);
                    ok = false;
                }
            }

            if (!ok) {
                fprintf(stderr, "%s: failed to read tensor info\n", __func__);
                fclose(file);
                wsp_gguf_free(ctx);
                return NULL;
            }
        }
    }

    ctx->alignment = WSP_GGUF_DEFAULT_ALIGNMENT;

    int alignment_idx = wsp_gguf_find_key(ctx, "general.alignment");
    if (alignment_idx != -1) {
        ctx->alignment = wsp_gguf_get_val_u32(ctx, alignment_idx);
    }

    // we require the data section to be aligned, so take into account any padding
    {
        const size_t offset_pad = offset % ctx->alignment;

        if (offset_pad != 0) {
            offset += ctx->alignment - offset_pad;
            fseek(file, offset, SEEK_SET);
        }
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = offset;

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
            struct wsp_gguf_tensor_info * info = &ctx->infos[i];

            const int64_t ne =
                (int64_t) info->ne[0] *
                (int64_t) info->ne[1] *
                (int64_t) info->ne[2] *
                (int64_t) info->ne[3];

            if (wsp_ggml_blck_size(info->type) == 0 || ne % wsp_ggml_blck_size(info->type) != 0) {
                fprintf(stderr, "%s: tensor '%s' of type %d (%s) number of elements (%" PRId64 ") is not a multiple of block size (%" PRId64 ")\n",
                        __func__, info->name.data, (int) info->type, wsp_ggml_type_name(info->type), ne, wsp_ggml_blck_size(info->type));
                fclose(file);
                wsp_gguf_free(ctx);
                return NULL;
            }

            const size_t size_cur = wsp_ggml_row_size(info->type, ne);

            ctx->size += WSP_GGML_PAD(size_cur, ctx->alignment);
        }
    }

    // load the tensor data only if requested
    if (params.ctx != NULL) {
        // if the provided wsp_gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created wsp_ggml_context as well, and point the "data" members of
        // the wsp_ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new wsp_ggml_context
        const size_t mem_size =
            params.no_alloc ?
            (ctx->header.n_tensors    )*wsp_ggml_tensor_overhead() :
            (ctx->header.n_tensors + 1)*wsp_ggml_tensor_overhead() + ctx->size;

        struct wsp_ggml_init_params pdata = {
            .mem_size   = mem_size,
            .mem_buffer = NULL,
            .no_alloc   = params.no_alloc,
        };

        *params.ctx = wsp_ggml_init(pdata);
        if (*params.ctx == NULL) {
            fprintf(stderr, "%s: failed to initialize context\n", __func__);
            fclose(file);
            wsp_gguf_free(ctx);
            return NULL;
        }

        struct wsp_ggml_context * ctx_data = *params.ctx;

        struct wsp_ggml_tensor * data = NULL;

        if (!params.no_alloc) {
            data = wsp_ggml_new_tensor_1d(ctx_data, WSP_GGML_TYPE_I8, ctx->size);

            ok = ok && data != NULL;

            // read the binary blob with the tensor data
            ok = ok && wsp_gguf_fread_el(file, data->data, ctx->size, &offset);

            if (!ok) {
                fprintf(stderr, "%s: failed to read tensor data\n", __func__);
                fclose(file);
                wsp_ggml_free(ctx_data);
                wsp_gguf_free(ctx);
                return NULL;
            }

            ctx->data = data->data;
        }

        wsp_ggml_set_no_alloc(ctx_data, true);

        // create the tensors
        for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
            const int64_t ne[WSP_GGML_MAX_DIMS] = {
                ctx->infos[i].ne[0],
                ctx->infos[i].ne[1],
                ctx->infos[i].ne[2],
                ctx->infos[i].ne[3],
            };

            struct wsp_ggml_tensor * cur = wsp_ggml_new_tensor(ctx_data, ctx->infos[i].type, ctx->infos[i].n_dims, ne);

            ok = ok && cur != NULL;

            if (!ok) {
                break;
            }

            wsp_ggml_set_name(cur, ctx->infos[i].name.data);

            // point the data member to the appropriate location in the binary blob using the tensor infos
            if (!params.no_alloc) {
              //cur->data = (char *) data->data + ctx->infos[i].offset - ctx->offset; // offset from start of file
                cur->data = (char *) data->data + ctx->infos[i].offset;               // offset from data
            }
        }

        if (!ok) {
            fprintf(stderr, "%s: failed to read the tensor data\n", __func__);
            fclose(file);
            wsp_ggml_free(ctx_data);
            wsp_gguf_free(ctx);
            return NULL;
        }

        wsp_ggml_set_no_alloc(ctx_data, params.no_alloc);
    }

    fclose(file);

    return ctx;
}

void wsp_gguf_free(struct wsp_gguf_context * ctx) {
    if (ctx == NULL) {
        return;
    }

    if (ctx->kv) {
        // free string memory - not great..
        for (uint64_t i = 0; i < ctx->header.n_kv; ++i) {
            wsp_gguf_free_kv(&ctx->kv[i]);
        }

        WSP_GGML_FREE(ctx->kv);
    }

    if (ctx->infos) {
        for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
            struct wsp_gguf_tensor_info * info = &ctx->infos[i];

            if (info->name.data) {
                WSP_GGML_FREE(info->name.data);
            }
        }

        WSP_GGML_FREE(ctx->infos);
    }

    WSP_GGML_FREE(ctx);
}

const char * wsp_gguf_type_name(enum wsp_gguf_type type) {
    return WSP_GGUF_TYPE_NAME[type];
}

int wsp_gguf_get_version(const struct wsp_gguf_context * ctx) {
    return ctx->header.version;
}

size_t wsp_gguf_get_alignment(const struct wsp_gguf_context * ctx) {
    return ctx->alignment;
}

size_t wsp_gguf_get_data_offset(const struct wsp_gguf_context * ctx) {
    return ctx->offset;
}

void * wsp_gguf_get_data(const struct wsp_gguf_context * ctx) {
    return ctx->data;
}

int wsp_gguf_get_n_kv(const struct wsp_gguf_context * ctx) {
    return ctx->header.n_kv;
}

int wsp_gguf_find_key(const struct wsp_gguf_context * ctx, const char * key) {
    // return -1 if key not found
    int keyfound = -1;

    const int n_kv = wsp_gguf_get_n_kv(ctx);

    for (int i = 0; i < n_kv; ++i) {
        if (strcmp(key, wsp_gguf_get_key(ctx, i)) == 0) {
            keyfound = i;
            break;
        }
    }

    return keyfound;
}

const char * wsp_gguf_get_key(const struct wsp_gguf_context * ctx, int key_id) {
    WSP_GGML_ASSERT(key_id >= 0 && key_id < wsp_gguf_get_n_kv(ctx));
    return ctx->kv[key_id].key.data;
}

enum wsp_gguf_type wsp_gguf_get_kv_type(const struct wsp_gguf_context * ctx, int key_id) {
    WSP_GGML_ASSERT(key_id >= 0 && key_id < wsp_gguf_get_n_kv(ctx));
    return ctx->kv[key_id].type;
}

enum wsp_gguf_type wsp_gguf_get_arr_type(const struct wsp_gguf_context * ctx, int key_id) {
    WSP_GGML_ASSERT(key_id >= 0 && key_id < wsp_gguf_get_n_kv(ctx));
    WSP_GGML_ASSERT(ctx->kv[key_id].type == WSP_GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].value.arr.type;
}

const void * wsp_gguf_get_arr_data(const struct wsp_gguf_context * ctx, int key_id) {
    WSP_GGML_ASSERT(key_id >= 0 && key_id < wsp_gguf_get_n_kv(ctx));
    WSP_GGML_ASSERT(ctx->kv[key_id].type == WSP_GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].value.arr.data;
}

const char * wsp_gguf_get_arr_str(const struct wsp_gguf_context * ctx, int key_id, int i) {
    WSP_GGML_ASSERT(key_id >= 0 && key_id < wsp_gguf_get_n_kv(ctx));
    WSP_GGML_ASSERT(ctx->kv[key_id].type == WSP_GGUF_TYPE_ARRAY);
    struct wsp_gguf_kv * kv = &ctx->kv[key_id];
    struct wsp_gguf_str * str = &((struct wsp_gguf_str *) kv->value.arr.data)[i];
    return str->data;
}

int wsp_gguf_get_arr_n(const struct wsp_gguf_context * ctx, int key_id) {
    WSP_GGML_ASSERT(key_id >= 0 && key_id < wsp_gguf_get_n_kv(ctx));
    WSP_GGML_ASSERT(ctx->kv[key_id].type == WSP_GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].value.arr.n;
}

uint8_t wsp_gguf_get_val_u8(const struct wsp_gguf_context * ctx, int key_id) {
    WSP_GGML_ASSERT(key_id >= 0 && key_id < wsp_gguf_get_n_kv(ctx));
    WSP_GGML_ASSERT(ctx->kv[key_id].type == WSP_GGUF_TYPE_UINT8);
    return ctx->kv[key_id].value.uint8;
}

int8_t wsp_gguf_get_val_i8(const struct wsp_gguf_context * ctx, int key_id) {
    WSP_GGML_ASSERT(key_id >= 0 && key_id < wsp_gguf_get_n_kv(ctx));
    WSP_GGML_ASSERT(ctx->kv[key_id].type == WSP_GGUF_TYPE_INT8);
    return ctx->kv[key_id].value.int8;
}

uint16_t wsp_gguf_get_val_u16(const struct wsp_gguf_context * ctx, int key_id) {
    WSP_GGML_ASSERT(key_id >= 0 && key_id < wsp_gguf_get_n_kv(ctx));
    WSP_GGML_ASSERT(ctx->kv[key_id].type == WSP_GGUF_TYPE_UINT16);
    return ctx->kv[key_id].value.uint16;
}

int16_t wsp_gguf_get_val_i16(const struct wsp_gguf_context * ctx, int key_id) {
    WSP_GGML_ASSERT(key_id >= 0 && key_id < wsp_gguf_get_n_kv(ctx));
    WSP_GGML_ASSERT(ctx->kv[key_id].type == WSP_GGUF_TYPE_INT16);
    return ctx->kv[key_id].value.int16;
}

uint32_t wsp_gguf_get_val_u32(const struct wsp_gguf_context * ctx, int key_id) {
    WSP_GGML_ASSERT(key_id >= 0 && key_id < wsp_gguf_get_n_kv(ctx));
    WSP_GGML_ASSERT(ctx->kv[key_id].type == WSP_GGUF_TYPE_UINT32);
    return ctx->kv[key_id].value.uint32;
}

int32_t wsp_gguf_get_val_i32(const struct wsp_gguf_context * ctx, int key_id) {
    WSP_GGML_ASSERT(key_id >= 0 && key_id < wsp_gguf_get_n_kv(ctx));
    WSP_GGML_ASSERT(ctx->kv[key_id].type == WSP_GGUF_TYPE_INT32);
    return ctx->kv[key_id].value.int32;
}

float wsp_gguf_get_val_f32(const struct wsp_gguf_context * ctx, int key_id) {
    WSP_GGML_ASSERT(key_id >= 0 && key_id < wsp_gguf_get_n_kv(ctx));
    WSP_GGML_ASSERT(ctx->kv[key_id].type == WSP_GGUF_TYPE_FLOAT32);
    return ctx->kv[key_id].value.float32;
}

uint64_t wsp_gguf_get_val_u64(const struct wsp_gguf_context * ctx, int key_id) {
    WSP_GGML_ASSERT(key_id >= 0 && key_id < wsp_gguf_get_n_kv(ctx));
    WSP_GGML_ASSERT(ctx->kv[key_id].type == WSP_GGUF_TYPE_UINT64);
    return ctx->kv[key_id].value.uint64;
}

int64_t wsp_gguf_get_val_i64(const struct wsp_gguf_context * ctx, int key_id) {
    WSP_GGML_ASSERT(key_id >= 0 && key_id < wsp_gguf_get_n_kv(ctx));
    WSP_GGML_ASSERT(ctx->kv[key_id].type == WSP_GGUF_TYPE_INT64);
    return ctx->kv[key_id].value.int64;
}

double wsp_gguf_get_val_f64(const struct wsp_gguf_context * ctx, int key_id) {
    WSP_GGML_ASSERT(key_id >= 0 && key_id < wsp_gguf_get_n_kv(ctx));
    WSP_GGML_ASSERT(ctx->kv[key_id].type == WSP_GGUF_TYPE_FLOAT64);
    return ctx->kv[key_id].value.float64;
}

bool wsp_gguf_get_val_bool(const struct wsp_gguf_context * ctx, int key_id) {
    WSP_GGML_ASSERT(key_id >= 0 && key_id < wsp_gguf_get_n_kv(ctx));
    WSP_GGML_ASSERT(ctx->kv[key_id].type == WSP_GGUF_TYPE_BOOL);
    return ctx->kv[key_id].value.bool_;
}

const char * wsp_gguf_get_val_str(const struct wsp_gguf_context * ctx, int key_id) {
    WSP_GGML_ASSERT(key_id >= 0 && key_id < wsp_gguf_get_n_kv(ctx));
    WSP_GGML_ASSERT(ctx->kv[key_id].type == WSP_GGUF_TYPE_STRING);
    return ctx->kv[key_id].value.str.data;
}

const void * wsp_gguf_get_val_data(const struct wsp_gguf_context * ctx, int key_id) {
    WSP_GGML_ASSERT(key_id >= 0 && key_id < wsp_gguf_get_n_kv(ctx));
    WSP_GGML_ASSERT(ctx->kv[key_id].type != WSP_GGUF_TYPE_ARRAY);
    WSP_GGML_ASSERT(ctx->kv[key_id].type != WSP_GGUF_TYPE_STRING);
    return &ctx->kv[key_id].value;
}

int wsp_gguf_get_n_tensors(const struct wsp_gguf_context * ctx) {
    return ctx->header.n_tensors;
}

int wsp_gguf_find_tensor(const struct wsp_gguf_context * ctx, const char * name) {
    // return -1 if tensor not found
    int tensorfound = -1;

    const int n_tensors = wsp_gguf_get_n_tensors(ctx);

    for (int i = 0; i < n_tensors; ++i) {
        if (strcmp(name, wsp_gguf_get_tensor_name(ctx, i)) == 0) {
            tensorfound = i;
            break;
        }
    }

    return tensorfound;
}

size_t wsp_gguf_get_tensor_offset(const struct wsp_gguf_context * ctx, int i) {
    return ctx->infos[i].offset;
}

char * wsp_gguf_get_tensor_name(const struct wsp_gguf_context * ctx, int i) {
    return ctx->infos[i].name.data;
}

enum wsp_ggml_type wsp_gguf_get_tensor_type(const struct wsp_gguf_context * ctx, int i) {
    return ctx->infos[i].type;
}

// returns the index
static int wsp_gguf_get_or_add_key(struct wsp_gguf_context * ctx, const char * key) {
    const int idx = wsp_gguf_find_key(ctx, key);
    if (idx >= 0) {
        return idx;
    }

    const int n_kv = wsp_gguf_get_n_kv(ctx);

    ctx->kv = realloc(ctx->kv, (n_kv + 1) * sizeof(struct wsp_gguf_kv));
    ctx->kv[n_kv].key.n    = strlen(key);
    ctx->kv[n_kv].key.data = strdup(key);
    ctx->header.n_kv++;

    return n_kv;
}

void wsp_gguf_remove_key(struct wsp_gguf_context * ctx, const char * key) {
    const int idx = wsp_gguf_find_key(ctx, key);
    if (idx >= 0) {
        const int n_kv = wsp_gguf_get_n_kv(ctx);
        wsp_gguf_free_kv(&ctx->kv[idx]);
        for (int i = idx; i < n_kv-1; ++i) {
            ctx->kv[i] = ctx->kv[i+1];
        }
        ctx->kv = realloc(ctx->kv, (n_kv - 1) * sizeof(struct wsp_gguf_kv));
        ctx->header.n_kv--;
    }
}

void wsp_gguf_set_val_u8(struct wsp_gguf_context * ctx, const char * key, uint8_t val) {
    const int idx = wsp_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type        = WSP_GGUF_TYPE_UINT8;
    ctx->kv[idx].value.uint8 = val;
}

void wsp_gguf_set_val_i8(struct wsp_gguf_context * ctx, const char * key, int8_t val) {
    const int idx = wsp_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type       = WSP_GGUF_TYPE_INT8;
    ctx->kv[idx].value.int8 = val;
}

void wsp_gguf_set_val_u16(struct wsp_gguf_context * ctx, const char * key, uint16_t val) {
    const int idx = wsp_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type         = WSP_GGUF_TYPE_UINT16;
    ctx->kv[idx].value.uint16 = val;
}

void wsp_gguf_set_val_i16(struct wsp_gguf_context * ctx, const char * key, int16_t val) {
    const int idx = wsp_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type        = WSP_GGUF_TYPE_INT16;
    ctx->kv[idx].value.int16 = val;
}

void wsp_gguf_set_val_u32(struct wsp_gguf_context * ctx, const char * key, uint32_t val) {
    const int idx = wsp_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type         = WSP_GGUF_TYPE_UINT32;
    ctx->kv[idx].value.uint32 = val;
}

void wsp_gguf_set_val_i32(struct wsp_gguf_context * ctx, const char * key, int32_t val) {
    const int idx = wsp_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type        = WSP_GGUF_TYPE_INT32;
    ctx->kv[idx].value.int32 = val;
}

void wsp_gguf_set_val_f32(struct wsp_gguf_context * ctx, const char * key, float val) {
    const int idx = wsp_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type          = WSP_GGUF_TYPE_FLOAT32;
    ctx->kv[idx].value.float32 = val;
}

void wsp_gguf_set_val_u64(struct wsp_gguf_context * ctx, const char * key, uint64_t val) {
    const int idx = wsp_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type         = WSP_GGUF_TYPE_UINT64;
    ctx->kv[idx].value.uint64 = val;
}

void wsp_gguf_set_val_i64(struct wsp_gguf_context * ctx, const char * key, int64_t val) {
    const int idx = wsp_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type        = WSP_GGUF_TYPE_INT64;
    ctx->kv[idx].value.int64 = val;
}

void wsp_gguf_set_val_f64(struct wsp_gguf_context * ctx, const char * key, double val) {
    const int idx = wsp_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type          = WSP_GGUF_TYPE_FLOAT64;
    ctx->kv[idx].value.float64 = val;
}

void wsp_gguf_set_val_bool(struct wsp_gguf_context * ctx, const char * key, bool val) {
    const int idx = wsp_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type        = WSP_GGUF_TYPE_BOOL;
    ctx->kv[idx].value.bool_ = val;
}

void wsp_gguf_set_val_str(struct wsp_gguf_context * ctx, const char * key, const char * val) {
    const int idx = wsp_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type           = WSP_GGUF_TYPE_STRING;
    ctx->kv[idx].value.str.n    = strlen(val);
    ctx->kv[idx].value.str.data = strdup(val);
}

void wsp_gguf_set_arr_data(struct wsp_gguf_context * ctx, const char * key, enum wsp_gguf_type type, const void * data, int n) {
    const int idx = wsp_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type           = WSP_GGUF_TYPE_ARRAY;
    ctx->kv[idx].value.arr.type = type;
    ctx->kv[idx].value.arr.n    = n;
    ctx->kv[idx].value.arr.data = WSP_GGML_CALLOC(n, wsp_gguf_type_size(type));
    memcpy(ctx->kv[idx].value.arr.data, data, n*wsp_gguf_type_size(type));
}

void wsp_gguf_set_arr_str(struct wsp_gguf_context * ctx, const char * key, const char ** data, int n) {
    const int idx = wsp_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type           = WSP_GGUF_TYPE_ARRAY;
    ctx->kv[idx].value.arr.type = WSP_GGUF_TYPE_STRING;
    ctx->kv[idx].value.arr.n    = n;
    ctx->kv[idx].value.arr.data = WSP_GGML_CALLOC(n, sizeof(struct wsp_gguf_str));
    for (int i = 0; i < n; i++) {
        struct wsp_gguf_str * str = &((struct wsp_gguf_str *)ctx->kv[idx].value.arr.data)[i];
        str->n    = strlen(data[i]);
        str->data = strdup(data[i]);
    }
}

// set or add KV pairs from another context
void wsp_gguf_set_kv(struct wsp_gguf_context * ctx, struct wsp_gguf_context * src) {
    for (uint32_t i = 0; i < src->header.n_kv; i++) {
        switch (src->kv[i].type) {
            case WSP_GGUF_TYPE_UINT8:   wsp_gguf_set_val_u8  (ctx, src->kv[i].key.data, src->kv[i].value.uint8);    break;
            case WSP_GGUF_TYPE_INT8:    wsp_gguf_set_val_i8  (ctx, src->kv[i].key.data, src->kv[i].value.int8);     break;
            case WSP_GGUF_TYPE_UINT16:  wsp_gguf_set_val_u16 (ctx, src->kv[i].key.data, src->kv[i].value.uint16);   break;
            case WSP_GGUF_TYPE_INT16:   wsp_gguf_set_val_i16 (ctx, src->kv[i].key.data, src->kv[i].value.int16);    break;
            case WSP_GGUF_TYPE_UINT32:  wsp_gguf_set_val_u32 (ctx, src->kv[i].key.data, src->kv[i].value.uint32);   break;
            case WSP_GGUF_TYPE_INT32:   wsp_gguf_set_val_i32 (ctx, src->kv[i].key.data, src->kv[i].value.int32);    break;
            case WSP_GGUF_TYPE_FLOAT32: wsp_gguf_set_val_f32 (ctx, src->kv[i].key.data, src->kv[i].value.float32);  break;
            case WSP_GGUF_TYPE_UINT64:  wsp_gguf_set_val_u64 (ctx, src->kv[i].key.data, src->kv[i].value.uint64);   break;
            case WSP_GGUF_TYPE_INT64:   wsp_gguf_set_val_i64 (ctx, src->kv[i].key.data, src->kv[i].value.int64);    break;
            case WSP_GGUF_TYPE_FLOAT64: wsp_gguf_set_val_f64 (ctx, src->kv[i].key.data, src->kv[i].value.float64);  break;
            case WSP_GGUF_TYPE_BOOL:    wsp_gguf_set_val_bool(ctx, src->kv[i].key.data, src->kv[i].value.bool_);    break;
            case WSP_GGUF_TYPE_STRING:  wsp_gguf_set_val_str (ctx, src->kv[i].key.data, src->kv[i].value.str.data); break;
            case WSP_GGUF_TYPE_ARRAY:
                {
                    if (src->kv[i].value.arr.type == WSP_GGUF_TYPE_STRING) {
                        const char ** data = WSP_GGML_CALLOC(src->kv[i].value.arr.n, sizeof(char *));
                        for (uint32_t j = 0; j < src->kv[i].value.arr.n; j++) {
                            data[j] = ((struct wsp_gguf_str *)src->kv[i].value.arr.data)[j].data;
                        }
                        wsp_gguf_set_arr_str(ctx, src->kv[i].key.data, data, src->kv[i].value.arr.n);
                        WSP_GGML_FREE((void *)data);
                    } else if (src->kv[i].value.arr.type == WSP_GGUF_TYPE_ARRAY) {
                        WSP_GGML_ABORT("nested arrays not supported");
                    } else {
                        wsp_gguf_set_arr_data(ctx, src->kv[i].key.data, src->kv[i].value.arr.type, src->kv[i].value.arr.data, src->kv[i].value.arr.n);
                    }
                } break;
            default: WSP_GGML_ABORT("invalid type");
        }
    }
}

void wsp_gguf_add_tensor(
             struct wsp_gguf_context * ctx,
        const struct wsp_ggml_tensor * tensor) {
    WSP_GGML_ASSERT(tensor);
    if (wsp_gguf_find_tensor(ctx, tensor->name) != -1) {
        WSP_GGML_ABORT("duplicated tensor name");
    }

    const int idx = ctx->header.n_tensors;
    ctx->infos = realloc(ctx->infos, (idx + 1)*sizeof(struct wsp_gguf_tensor_info));

    ctx->infos[idx].name.n    = strlen(tensor->name);
    ctx->infos[idx].name.data = strdup(tensor->name);

    for (int i = 0; i < WSP_GGML_MAX_DIMS; ++i) {
        ctx->infos[idx].ne[i] = 1;
    }

    ctx->infos[idx].n_dims = wsp_ggml_n_dims(tensor);
    for (uint32_t i = 0; i < ctx->infos[idx].n_dims; i++) {
        ctx->infos[idx].ne[i] = tensor->ne[i];
    }

    ctx->infos[idx].type   = tensor->type;
    ctx->infos[idx].offset = 0;
    ctx->infos[idx].data   = tensor->data;
    ctx->infos[idx].size   = wsp_ggml_nbytes(tensor);

    if (ctx->header.n_tensors > 0) {
        ctx->infos[idx].offset = ctx->infos[idx - 1].offset + WSP_GGML_PAD(ctx->infos[idx - 1].size, ctx->alignment);
    }

    ctx->header.n_tensors++;
}

void wsp_gguf_set_tensor_type(struct wsp_gguf_context * ctx, const char * name, enum wsp_ggml_type type) {
    const int idx = wsp_gguf_find_tensor(ctx, name);
    if (idx < 0) {
        WSP_GGML_ABORT("tensor not found");
    }

    ctx->infos[idx].type = type;
}

void wsp_gguf_set_tensor_data(struct wsp_gguf_context * ctx, const char * name, const void * data, size_t size) {
    const int idx = wsp_gguf_find_tensor(ctx, name);
    if (idx < 0) {
        WSP_GGML_ABORT("tensor not found");
    }

    ctx->infos[idx].data = data;
    ctx->infos[idx].size = size;

    // update offsets
    for (uint32_t i = idx + 1; i < ctx->header.n_tensors; ++i) {
        ctx->infos[i].offset = ctx->infos[i - 1].offset + WSP_GGML_PAD(ctx->infos[i - 1].size, ctx->alignment);
    }
}

//static void wsp_gguf_fwrite_str(FILE * file, const struct wsp_gguf_str * val) {
//    fwrite(&val->n,   sizeof(val->n),    1, file);
//    fwrite(val->data, sizeof(char), val->n, file);
//}
//
//static void wsp_gguf_fwrite_el(FILE * file, const void * val, size_t size) {
//    fwrite(val, sizeof(char), size, file);
//}

struct wsp_gguf_buf {
    void * data;
    size_t size;
    size_t offset;
};

static struct wsp_gguf_buf wsp_gguf_buf_init(size_t size) {
    struct wsp_gguf_buf buf = {
        /*buf.data   =*/ size == 0 ? NULL : WSP_GGML_CALLOC(1, size),
        /*buf.size   =*/ size,
        /*buf.offset =*/ 0,
    };

    return buf;
}

static void wsp_gguf_buf_free(struct wsp_gguf_buf buf) {
    if (buf.data) {
        WSP_GGML_FREE(buf.data);
    }
}

static void wsp_gguf_buf_grow(struct wsp_gguf_buf * buf, size_t size) {
    if (buf->offset + size > buf->size) {
        buf->size = 1.5*(buf->offset + size);
        if (buf->data) {
            buf->data = realloc(buf->data, buf->size);
        }
    }
}

static void wsp_gguf_bwrite_str(struct wsp_gguf_buf * buf, const struct wsp_gguf_str * val) {
    wsp_gguf_buf_grow(buf, sizeof(val->n) + val->n);

    if (buf->data) {
        memcpy((char *) buf->data + buf->offset, &val->n, sizeof(val->n));
    }
    buf->offset += sizeof(val->n);

    if (buf->data) {
        memcpy((char *) buf->data + buf->offset, val->data, val->n);
    }
    buf->offset += val->n;
}

static void wsp_gguf_bwrite_el(struct wsp_gguf_buf * buf, const void * val, size_t el_size) {
    wsp_gguf_buf_grow(buf, el_size);

    if (buf->data) {
        memcpy((char *) buf->data + buf->offset, val, el_size);
    }
    buf->offset += el_size;
}

static void wsp_gguf_write_to_buf(const struct wsp_gguf_context * ctx, struct wsp_gguf_buf * buf, bool only_meta) {
    // write header
    wsp_gguf_bwrite_el(buf, &ctx->header.magic,     sizeof(ctx->header.magic));
    wsp_gguf_bwrite_el(buf, &ctx->header.version,   sizeof(ctx->header.version));
    wsp_gguf_bwrite_el(buf, &ctx->header.n_tensors, sizeof(ctx->header.n_tensors));
    wsp_gguf_bwrite_el(buf, &ctx->header.n_kv,      sizeof(ctx->header.n_kv));

    // write key-value pairs
    for (uint32_t i = 0; i < ctx->header.n_kv; ++i) {
        struct wsp_gguf_kv * kv = &ctx->kv[i];

        wsp_gguf_bwrite_str(buf, &kv->key);
        wsp_gguf_bwrite_el (buf, &kv->type, sizeof(kv->type));

        switch (kv->type) {
            case WSP_GGUF_TYPE_UINT8:   wsp_gguf_bwrite_el( buf, &kv->value.uint8,   sizeof(kv->value.uint8)  ); break;
            case WSP_GGUF_TYPE_INT8:    wsp_gguf_bwrite_el (buf, &kv->value.int8,    sizeof(kv->value.int8)   ); break;
            case WSP_GGUF_TYPE_UINT16:  wsp_gguf_bwrite_el (buf, &kv->value.uint16,  sizeof(kv->value.uint16) ); break;
            case WSP_GGUF_TYPE_INT16:   wsp_gguf_bwrite_el (buf, &kv->value.int16,   sizeof(kv->value.int16)  ); break;
            case WSP_GGUF_TYPE_UINT32:  wsp_gguf_bwrite_el (buf, &kv->value.uint32,  sizeof(kv->value.uint32) ); break;
            case WSP_GGUF_TYPE_INT32:   wsp_gguf_bwrite_el (buf, &kv->value.int32,   sizeof(kv->value.int32)  ); break;
            case WSP_GGUF_TYPE_FLOAT32: wsp_gguf_bwrite_el (buf, &kv->value.float32, sizeof(kv->value.float32)); break;
            case WSP_GGUF_TYPE_UINT64:  wsp_gguf_bwrite_el (buf, &kv->value.uint64,  sizeof(kv->value.uint64) ); break;
            case WSP_GGUF_TYPE_INT64:   wsp_gguf_bwrite_el (buf, &kv->value.int64,   sizeof(kv->value.int64)  ); break;
            case WSP_GGUF_TYPE_FLOAT64: wsp_gguf_bwrite_el (buf, &kv->value.float64, sizeof(kv->value.float64)); break;
            case WSP_GGUF_TYPE_BOOL:    wsp_gguf_bwrite_el (buf, &kv->value.bool_,   sizeof(kv->value.bool_)  ); break;
            case WSP_GGUF_TYPE_STRING:  wsp_gguf_bwrite_str(buf, &kv->value.str                               ); break;
            case WSP_GGUF_TYPE_ARRAY:
                {
                    wsp_gguf_bwrite_el(buf, &kv->value.arr.type, sizeof(kv->value.arr.type));
                    wsp_gguf_bwrite_el(buf, &kv->value.arr.n,    sizeof(kv->value.arr.n)   );

                    switch (kv->value.arr.type) {
                        case WSP_GGUF_TYPE_UINT8:
                        case WSP_GGUF_TYPE_INT8:
                        case WSP_GGUF_TYPE_UINT16:
                        case WSP_GGUF_TYPE_INT16:
                        case WSP_GGUF_TYPE_UINT32:
                        case WSP_GGUF_TYPE_INT32:
                        case WSP_GGUF_TYPE_FLOAT32:
                        case WSP_GGUF_TYPE_UINT64:
                        case WSP_GGUF_TYPE_INT64:
                        case WSP_GGUF_TYPE_FLOAT64:
                        case WSP_GGUF_TYPE_BOOL:
                            {
                                wsp_gguf_bwrite_el(buf, kv->value.arr.data, kv->value.arr.n * wsp_gguf_type_size(kv->value.arr.type));
                            } break;
                        case WSP_GGUF_TYPE_STRING:
                            {
                                for (uint32_t j = 0; j < kv->value.arr.n; ++j) {
                                    wsp_gguf_bwrite_str(buf, &((struct wsp_gguf_str *) kv->value.arr.data)[j]);
                                }
                            } break;
                        case WSP_GGUF_TYPE_ARRAY:
                        default: WSP_GGML_ABORT("invalid type");
                    }
                } break;
            default: WSP_GGML_ABORT("invalid type");
        }
    }

    // write tensor infos
    for (uint32_t i = 0; i < ctx->header.n_tensors; ++i) {
        struct wsp_gguf_tensor_info * info = &ctx->infos[i];

        wsp_gguf_bwrite_str(buf, &info->name);
        wsp_gguf_bwrite_el (buf, &info->n_dims, sizeof(info->n_dims));
        for (uint32_t j = 0; j < info->n_dims; ++j) {
            wsp_gguf_bwrite_el(buf, &info->ne[j], sizeof(info->ne[j]));
        }
        wsp_gguf_bwrite_el(buf, &info->type,   sizeof(info->type));
        wsp_gguf_bwrite_el(buf, &info->offset, sizeof(info->offset));
    }

    // we require the data section to be aligned, so take into account any padding
    {
        const size_t offset     = buf->offset;
        const size_t offset_pad = WSP_GGML_PAD(offset, ctx->alignment);

        if (offset_pad != offset) {
            uint8_t pad = 0;
            for (size_t i = 0; i < offset_pad - offset; ++i) {
                wsp_gguf_bwrite_el(buf, &pad, sizeof(pad));
            }
        }
    }

    if (only_meta) {
        return;
    }

    size_t offset = 0;

    // write tensor data
    for (uint32_t i = 0; i < ctx->header.n_tensors; ++i) {
        struct wsp_gguf_tensor_info * info = &ctx->infos[i];

        const size_t size     = info->size;
        const size_t size_pad = WSP_GGML_PAD(size, ctx->alignment);

        wsp_gguf_bwrite_el(buf, info->data, size);

        if (size_pad != size) {
            uint8_t pad = 0;
            for (size_t j = 0; j < size_pad - size; ++j) {
                wsp_gguf_bwrite_el(buf, &pad, sizeof(pad));
            }
        }

        WSP_GGML_ASSERT(offset == info->offset);

        offset += size_pad;
    }
}

void wsp_gguf_write_to_file(const struct wsp_gguf_context * ctx, const char * fname, bool only_meta) {
    FILE * file = wsp_ggml_fopen(fname, "wb");
    if (!file) {
        WSP_GGML_ABORT("failed to open file for writing");
    }

    struct wsp_gguf_buf buf = wsp_gguf_buf_init(16*1024);

    wsp_gguf_write_to_buf(ctx, &buf, only_meta);

    fwrite(buf.data, 1, buf.offset, file);

    wsp_gguf_buf_free(buf);

    fclose(file);
}

size_t wsp_gguf_get_meta_size(const struct wsp_gguf_context * ctx) {
    // no allocs - only compute size
    struct wsp_gguf_buf buf = wsp_gguf_buf_init(0);

    wsp_gguf_write_to_buf(ctx, &buf, true);

    return buf.offset;
}

void wsp_gguf_get_meta_data(const struct wsp_gguf_context * ctx, void * data) {
    struct wsp_gguf_buf buf = wsp_gguf_buf_init(16*1024);

    wsp_gguf_write_to_buf(ctx, &buf, true);

    memcpy(data, buf.data, buf.offset);

    wsp_gguf_buf_free(buf);
}

////////////////////////////////////////////////////////////////////////////////

int wsp_ggml_cpu_has_avx(void) {
#if defined(__AVX__)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_avx_vnni(void) {
#if defined(__AVXVNNI__)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_avx512(void) {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_avx512_vbmi(void) {
#if defined(__AVX512VBMI__)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_avx512_vnni(void) {
#if defined(__AVX512VNNI__)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_avx512_bf16(void) {
#if defined(__AVX512BF16__)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_amx_int8(void) {
#if defined(__AMX_INT8__)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_fma(void) {
#if defined(__FMA__)
    return 1;
#else
    return 0;
#endif
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

int wsp_ggml_cpu_has_arm_fma(void) {
#if defined(__ARM_FEATURE_FMA)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_riscv_v(void) {
#if defined(__riscv_v_intrinsic)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_metal(void) {
#if defined(WSP_GGML_USE_METAL)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_f16c(void) {
#if defined(__F16C__)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_fp16_va(void) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_wasm_simd(void) {
#if defined(__wasm_simd128__)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_blas(void) {
#if defined(WSP_GGML_USE_BLAS) || defined(WSP_GGML_USE_CUDA) || defined(WSP_GGML_USE_VULKAN) || defined(WSP_GGML_USE_SYCL)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_cuda(void) {
#if defined(WSP_GGML_USE_CUDA)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_vulkan(void) {
#if defined(WSP_GGML_USE_VULKAN)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_kompute(void) {
#if defined(WSP_GGML_USE_KOMPUTE)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_sycl(void) {
#if defined(WSP_GGML_USE_SYCL)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_rpc(void) {
#if defined(WSP_GGML_USE_RPC)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_cann(void) {
#if defined(WSP_GGML_USE_CANN)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_llamafile(void) {
#if defined(WSP_GGML_USE_LLAMAFILE)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_gpublas(void) {
    return wsp_ggml_cpu_has_cuda() || wsp_ggml_cpu_has_vulkan() || wsp_ggml_cpu_has_kompute() || wsp_ggml_cpu_has_sycl();
}

int wsp_ggml_cpu_has_sse3(void) {
#if defined(__SSE3__)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_ssse3(void) {
#if defined(__SSSE3__)
    return 1;
#else
    return 0;
#endif
}

int wsp_ggml_cpu_has_vsx(void) {
#if defined(__POWER9_VECTOR__)
    return 1;
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

void wsp_ggml_log_set(wsp_ggml_log_callback log_callback, void * user_data) {
    g_logger_state.log_callback = log_callback ? log_callback : wsp_ggml_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}
////////////////////////////////////////////////////////////////////////////////
