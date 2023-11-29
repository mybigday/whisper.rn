#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>
#include "whisper.h"

extern "C" {

std::unordered_map<int, bool> abort_map;

bool* rn_whisper_assign_abort_map(int job_id) {
    abort_map[job_id] = false;
    return &abort_map[job_id];
}

void rn_whisper_remove_abort_map(int job_id) {
    if (abort_map.find(job_id) != abort_map.end()) {
        abort_map.erase(job_id);
    }
}

void rn_whisper_abort_transcribe(int job_id) {
    if (abort_map.find(job_id) != abort_map.end()) {
        abort_map[job_id] = true;
    }
}

bool rn_whisper_transcribe_is_aborted(int job_id) {
    if (abort_map.find(job_id) != abort_map.end()) {
        return abort_map[job_id];
    }
    return false;
}

void rn_whisper_abort_all_transcribe() {
    for (auto it = abort_map.begin(); it != abort_map.end(); ++it) {
        it->second = true;
    }
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

bool rn_whisper_vad_simple(std::vector<float> & pcmf32, int sample_rate, int last_ms, float vad_thold, float freq_thold, bool verbose) {
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
