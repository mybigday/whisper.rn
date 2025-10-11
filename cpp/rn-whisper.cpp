#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>
#include "rn-whisper.h"

#define DEFAULT_MAX_AUDIO_SEC 30;

namespace rnwhisper {

const char * system_info(void) {
  static std::string s;
  s = "";
  if (wsp_ggml_cpu_has_avx() == 1) s += "AVX ";
  if (wsp_ggml_cpu_has_avx2() == 1) s += "AVX2 ";
  if (wsp_ggml_cpu_has_avx512() == 1) s += "AVX512 ";
  if (wsp_ggml_cpu_has_fma() == 1) s += "FMA ";
  if (wsp_ggml_cpu_has_neon() == 1) s += "NEON ";
  if (wsp_ggml_cpu_has_arm_fma() == 1) s += "ARM_FMA ";
  if (wsp_ggml_cpu_has_f16c() == 1) s += "F16C ";
  if (wsp_ggml_cpu_has_fp16_va() == 1) s += "FP16_VA ";
  if (wsp_ggml_cpu_has_sse3() == 1) s += "SSE3 ";
  if (wsp_ggml_cpu_has_ssse3() == 1) s += "SSSE3 ";
  if (wsp_ggml_cpu_has_vsx() == 1) s += "VSX ";
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
}

std::unordered_map<int, job*> job_map;

void job_abort_all() {
    for (auto it = job_map.begin(); it != job_map.end(); ++it) {
        it->second->abort();
    }
}

job* job_new(int job_id, struct whisper_full_params params) {
    job* ctx = new job();
    ctx->job_id = job_id;
    ctx->params = params;

    job_map[job_id] = ctx;

    // Abort handler
    params.encoder_begin_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, void * user_data) {
        job *j = (job*)user_data;
        return !j->is_aborted();
    };
    params.encoder_begin_callback_user_data = job_map[job_id];
    params.abort_callback = [](void * user_data) {
        job *j = (job*)user_data;
        return j->is_aborted();
    };
    params.abort_callback_user_data = job_map[job_id];

    return job_map[job_id];
}

job* job_get(int job_id) {
    if (job_map.find(job_id) != job_map.end()) {
        return job_map[job_id];
    }
    return nullptr;
}

void job_remove(int job_id) {
    if (job_map.find(job_id) != job_map.end()) {
        delete job_map[job_id];
    }
    job_map.erase(job_id);
}

}
