#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include "rn-whisper.h"
#include "ggml.h"
#include "ggml-cpu.h"

#if defined(__linux__)
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#define DEFAULT_MAX_AUDIO_SEC 30;

namespace rnwhisper {

const char * system_info(void) {
  static std::string s;
  s = "";
  if (ggml_cpu_has_avx() == 1) s += "AVX ";
  if (ggml_cpu_has_avx2() == 1) s += "AVX2 ";
  if (ggml_cpu_has_avx512() == 1) s += "AVX512 ";
  if (ggml_cpu_has_fma() == 1) s += "FMA ";
  if (ggml_cpu_has_neon() == 1) s += "NEON ";
  if (ggml_cpu_has_arm_fma() == 1) s += "ARM_FMA ";
  if (ggml_cpu_has_f16c() == 1) s += "F16C ";
  if (ggml_cpu_has_fp16_va() == 1) s += "FP16_VA ";
  if (ggml_cpu_has_sse3() == 1) s += "SSE3 ";
  if (ggml_cpu_has_ssse3() == 1) s += "SSSE3 ";
  if (ggml_cpu_has_vsx() == 1) s += "VSX ";
#ifdef WHISPER_USE_COREML
  s += "COREML ";
#endif
  s.erase(s.find_last_not_of(" ") + 1);
  return s.c_str();
}

std::string bench(struct whisper_context * ctx, int n_threads) {
    const int n_mels = whisper_model_n_mels(ctx);

    if (int ret = whisper_set_mel(ctx, nullptr, 0, n_mels)) {
        return "error: failed to set mel: " + std::to_string(ret);
    }
    // heat encoder
    if (int ret = whisper_encode(ctx, 0, n_threads) != 0) {
        return "error: failed to encode: " + std::to_string(ret);
    }

    whisper_token tokens[512];
    memset(tokens, 0, sizeof(tokens));

    // prompt heat
    if (int ret = whisper_decode(ctx, tokens, 256, 0, n_threads) != 0) {
        return "error: failed to decode: " + std::to_string(ret);
    }

    // text-generation heat
    if (int ret = whisper_decode(ctx, tokens, 1, 256, n_threads) != 0) {
        return "error: failed to decode: " + std::to_string(ret);
    }

    whisper_reset_timings(ctx);

    // actual run
    if (int ret = whisper_encode(ctx, 0, n_threads) != 0) {
        return "error: failed to encode: " + std::to_string(ret);
    }

    // text-generation
    for (int i = 0; i < 256; i++) {
        if (int ret = whisper_decode(ctx, tokens, 1, i, n_threads) != 0) {
            return "error: failed to decode: " + std::to_string(ret);
        }
    }

    // batched decoding
    for (int i = 0; i < 64; i++) {
        if (int ret = whisper_decode(ctx, tokens, 5, 0, n_threads) != 0) {
            return "error: failed to decode: " + std::to_string(ret);
        }
    }

    // prompt processing
    for (int i = 0; i < 16; i++) {
        if (int ret = whisper_decode(ctx, tokens, 256, 0, n_threads) != 0) {
            return "error: failed to decode: " + std::to_string(ret);
        }
    }

    const struct whisper_timings * timings = whisper_get_timings(ctx);

    return std::string("[") +
        "\"" + system_info() + "\"," +
        std::to_string(n_threads) + "," +
        std::to_string(timings->encode_ms) + "," +
        std::to_string(timings->decode_ms) + "," +
        std::to_string(timings->batchd_ms) + "," +
        std::to_string(timings->prompt_ms) + "]";
}

bool job::is_aborted() {
    return aborted;
}

void job::abort() {
    aborted = true;
}

job::~job() {
    RNWHISPER_LOG_INFO("rnwhisper::job::%s: job_id: %d\n", __func__, job_id);
    if (threadpool != nullptr) {
        ggml_threadpool_free(threadpool);
    }
    if (restore_affinity) {
        restore_affinity();
    }
}

// parse_cpu_range and parse_hex_cpu_mask are llama.cpp common's
// parse_cpu_range / parse_cpu_mask (common/common.cpp), with std::stoull's
// exceptions turned into a false return.
static bool parse_cpu_range(const std::string & range, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    size_t dash_loc = range.find('-');
    if (dash_loc == std::string::npos) {
        return false;
    }

    size_t start_i;
    size_t end_i;

    try {
        if (dash_loc == 0) {
            start_i = 0;
        } else {
            start_i = std::stoull(range.substr(0, dash_loc));
            if (start_i >= GGML_MAX_N_THREADS) {
                return false;
            }
        }

        if (dash_loc == range.length() - 1) {
            end_i = GGML_MAX_N_THREADS - 1;
        } else {
            end_i = std::stoull(range.substr(dash_loc + 1));
            if (end_i >= GGML_MAX_N_THREADS) {
                return false;
            }
        }
    } catch (const std::exception &) {
        return false;
    }

    for (size_t i = start_i; i <= end_i; i++) {
        boolmask[i] = true;
    }

    return true;
}

static bool parse_hex_cpu_mask(const std::string & mask, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    // Discard potential 0x prefix
    size_t start_i = 0;
    if (mask.length() >= 2 && mask.substr(0, 2) == "0x") {
        start_i = 2;
    }

    size_t num_digits = mask.length() - start_i;
    num_digits = std::min<size_t>(num_digits, 128);

    size_t end_i = num_digits + start_i;

    for (size_t i = start_i, n = (num_digits*4 - 1); i < end_i; i++, n-=4) {
        char c = mask.at(i);
        int8_t id = c;

        if ((c >= '0' && c <= '9')) {
            id -= '0';
        } else if (c >= 'a' && c <= 'f') {
            id -= 'a' - 10;
        } else if (c >= 'A' && c <= 'F') {
            id -= 'A' - 10;
        } else {
            return false;
        }

        boolmask[  n  ] = boolmask[  n  ] || ((id & 8) != 0);
        boolmask[n - 1] = boolmask[n - 1] || ((id & 4) != 0);
        boolmask[n - 2] = boolmask[n - 2] || ((id & 2) != 0);
        boolmask[n - 3] = boolmask[n - 3] || ((id & 1) != 0);
    }

    return true;
}

bool parse_cpu_mask(const std::string & mask, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    // A hex mask never contains '-', so the two forms cannot be confused.
    if (mask.find('-') != std::string::npos) {
        return parse_cpu_range(mask, boolmask);
    }
    return parse_hex_cpu_mask(mask, boolmask);
}

// ggml_threadpool_new pins the calling thread (it is worker 0) to the mask and
// nothing undoes that; ggml's clear_numa_thread_affinity is a no-op outside
// NUMA mode. Job threads are reused, and threads they spawn inherit the
// affinity, so a later job without a mask would stay pinned. Affinity is a
// no-op in ggml on Apple platforms, so only Linux/Android need restoring.
[[maybe_unused]] static std::function<void()> save_thread_affinity() {
#if defined(__linux__)
    cpu_set_t saved;
    CPU_ZERO(&saved);
    const pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
    if (sched_getaffinity(tid, sizeof(saved), &saved) != 0) {
        return nullptr;
    }
    return [tid, saved]() {
        sched_setaffinity(tid, sizeof(saved), &saved);
    };
#else
    return nullptr;
#endif
}

std::unordered_map<int, job*> job_map;
std::mutex job_map_mutex;

ggml_abort_callback_t set_ggml_abort_callback(ggml_abort_callback_t callback) {
    return ggml_set_abort_callback(callback);
}

void job_abort_all() {
    std::lock_guard<std::mutex> lock(job_map_mutex);
    for (auto it = job_map.begin(); it != job_map.end(); ++it) {
        it->second->abort();
    }
}

job* job_new(int job_id, struct whisper_full_params params, const cpu_params * cpu) {
    job* ctx = new job();
    ctx->job_id = job_id;
    ctx->params = params;
    // Without a mask, ggml keeps creating its own pool per graph, as before.
    // ggml cannot set thread affinity on Apple platforms, so a pool would
    // change nothing there but the thread lifetime; keep the default path.
#if !defined(__APPLE__)
    if (cpu != nullptr && cpu->has_mask() && params.n_threads > 0) {
        ggml_threadpool_params tpp = ggml_threadpool_params_default(params.n_threads);
        std::memcpy(tpp.cpumask, cpu->mask, sizeof(tpp.cpumask));
        tpp.strict_cpu = cpu->strict;

        ctx->restore_affinity = save_thread_affinity();
        ctx->threadpool = ggml_threadpool_new(&tpp);
        if (ctx->threadpool == nullptr) {
            RNWHISPER_LOG_WARN("rnwhisper::%s: failed to create threadpool, using the default threads\n", __func__);
        }
    }
#else
    (void) cpu;
#endif
    ctx->params.threadpool = ctx->threadpool;
    ctx->params.encoder_begin_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, void * user_data) {
        job *j = (job*)user_data;
        return !j->is_aborted();
    };
    ctx->params.abort_callback = [](void * user_data) {
        job *j = (job*)user_data;
        return j->is_aborted();
    };
    ctx->params.encoder_begin_callback_user_data = ctx;
    ctx->params.abort_callback_user_data = ctx;

    std::lock_guard<std::mutex> lock(job_map_mutex);
    job_map[job_id] = ctx;

    return ctx;
}

job* job_get(int job_id) {
    std::lock_guard<std::mutex> lock(job_map_mutex);
    if (job_map.find(job_id) != job_map.end()) {
        return job_map[job_id];
    }
    return nullptr;
}

void job_remove(int job_id) {
    std::lock_guard<std::mutex> lock(job_map_mutex);
    if (job_map.find(job_id) != job_map.end()) {
        delete job_map[job_id];
    }
    job_map.erase(job_id);
}

}
