#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>
#include "rn-whisper.h"

namespace rnwhisper {

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
        fprintf(stderr, "%s: energy_all: %f, energy_last: %f, vad_thold: %f, freq_thold: %f\n", __func__, energy_all, energy_last, vad_thold, freq_thold);
    }

    if (energy_last > vad_thold*energy_all) {
        return false;
    }

    return true;
}

job::~job() {
    fprintf(stderr, "%s: job_id: %d\n", __func__, job_id);
}

void job::set_vad_params(vad_params params) {
    vad = params;
    if (vad.vad_ms < 2000) vad.vad_ms = 2000;
}

bool job::vad_simple(short* pcm, int n_samples, int n) {
    if (!vad.use_vad) return true;

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

bool job::is_aborted() {
    return aborted;
}

void job::abort() {
    aborted = true;
}

std::unordered_map<int, job> job_map;

void job_abort_all() {
    for (auto it = job_map.begin(); it != job_map.end(); ++it) {
        it->second.abort();
    }
}

job* job_new(int job_id, struct whisper_full_params params) {
    job ctx;
    ctx.job_id = job_id;
    ctx.params = params;

    // Abort handler
    params.encoder_begin_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, void * user_data) {
        job *j = (job*)user_data;
        return !j->is_aborted();
    };
    params.encoder_begin_callback_user_data = &ctx;
    params.abort_callback = [](void * user_data) {
        job *j = (job*)user_data;
        return j->is_aborted();
    };
    params.abort_callback_user_data = &ctx;

    job_map[job_id] = ctx;
    return &job_map[job_id];
}

job* job_get(int job_id) {
    if (job_map.find(job_id) != job_map.end()) {
        return &job_map[job_id];
    }
    return nullptr;
}

void job_remove(int job_id) {
    job_map.erase(job_id);
}

}
