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

void high_pass_filter(std::vector<float> & data, float cutoff, float sample_rate) {
    const float rc = 1.0f / (2.0f * M_PI * cutoff);
    const float dt = 1.0f / sample_rate;
    const float alpha = dt / (rc + dt);

    float y = data[0];

    for (size_t i = 1; i < data.size(); i++) {
        y = alpha * (y + data[i] - data[i - 1]);
        data[i] = y;
    }
}

bool vad_simple_impl(std::vector<float> & pcmf32, int sample_rate, int last_ms, float vad_thold, float freq_thold, bool verbose) {
    const int n_samples      = pcmf32.size();
    const int n_samples_last = (sample_rate * last_ms) / 1000;

    if (n_samples_last >= n_samples) {
        // not enough samples - assume no speech
        return false;
    }

    if (freq_thold > 0.0f) {
        high_pass_filter(pcmf32, freq_thold, sample_rate);
    }

    float energy_all  = 0.0f;
    float energy_last = 0.0f;

    for (int i = 0; i < n_samples; i++) {
        energy_all += fabsf(pcmf32[i]);

        if (i >= n_samples - n_samples_last) {
        energy_last += fabsf(pcmf32[i]);
        }
    }

    energy_all  /= n_samples;
    energy_last /= n_samples_last;

    if (verbose) {
        RNWHISPER_LOG_INFO("%s: energy_all: %f, energy_last: %f, vad_thold: %f, freq_thold: %f\n", __func__, energy_all, energy_last, vad_thold, freq_thold);
    }

    if (energy_last > vad_thold*energy_all) {
        return false;
    }

    return true;
}

void job::set_realtime_params(
    vad_params params,
    int sec,
    int slice_sec,
    float min_sec,
    const char* output_path
) {
    vad = params;
    if (vad.vad_ms < 2000) vad.vad_ms = 2000;
    audio_sec = sec > 0 ? sec : DEFAULT_MAX_AUDIO_SEC;
    audio_slice_sec = slice_sec > 0 && slice_sec < audio_sec ? slice_sec : audio_sec;
    audio_min_sec = min_sec >= 0.5 && min_sec <= audio_slice_sec ? min_sec : 1.0f;
    audio_output_path = output_path;
}

bool job::vad_simple(int slice_index, int n_samples, int n) {
    if (slice_index >= pcm_slices.size()) return !vad.use_vad;
    if (!vad.use_vad) return true;

    short* pcm = pcm_slices[slice_index];
    int sample_size = (int) (WHISPER_SAMPLE_RATE * vad.vad_ms / 1000);
    if (n_samples + n > sample_size) {
        int start = n_samples + n - sample_size;
        std::vector<float> pcmf32(sample_size);
        for (int i = 0; i < sample_size; i++) {
            pcmf32[i] = (float)pcm[i + start] / 32768.0f;
        }
        return vad_simple_impl(pcmf32, WHISPER_SAMPLE_RATE, vad.last_ms, vad.vad_thold, vad.freq_thold, vad.verbose);
    }
    return false;
}

void job::put_pcm_data(short* data, int slice_index, int n_samples, int n) {
    if (pcm_slices.size() == slice_index) {
        int n_slices = (int) (WHISPER_SAMPLE_RATE * audio_slice_sec);
        pcm_slices.push_back(new short[n_slices]);
    }
    short* pcm = pcm_slices[slice_index];
    for (int i = 0; i < n; i++) {
        pcm[i + n_samples] = data[i];
    }
}

float* job::pcm_slice_to_f32(int slice_index, int size) {
    if (pcm_slices.size() > slice_index) {
        float* pcmf32 = new float[size];
        for (int i = 0; i < size; i++) {
            pcmf32[i] = (float)pcm_slices[slice_index][i] / 32768.0f;
        }
        return pcmf32;
    }
    return nullptr;
}

bool job::is_aborted() {
    return aborted;
}

void job::abort() {
    aborted = true;
}

job::~job() {
    RNWHISPER_LOG_INFO("rnwhisper::job::%s: job_id: %d\n", __func__, job_id);

    for (size_t i = 0; i < pcm_slices.size(); i++) {
        delete[] pcm_slices[i];
    }
    pcm_slices.clear();
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
