#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>
#include "whisper.h"
#include "rn-whisper.h"

namespace rnwhisper {

job::~job() {
    fprintf(stderr, "%s: job_id: %d\n", __func__, job_id);
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

job job_new(int job_id) {
    job ctx;
    ctx.job_id = job_id;
    job_map[job_id] = ctx;
    return ctx;
}

void job_remove(int job_id) {
    job_map.erase(job_id);
}

job* job_get(int job_id) {
    if (job_map.find(job_id) != job_map.end()) {
        return &job_map[job_id];
    }
    return nullptr;
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

bool vad_simple(std::vector<float> & pcmf32, int sample_rate, int last_ms, float vad_thold, float freq_thold, bool verbose) {
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

}
