#ifndef RNWHISPER_H
#define RNWHISPER_H

#include <string>
#include <vector>
#include "whisper.h"

namespace rnwhisper {

struct job {
    int job_id;
    whisper_full_params params;
    bool aborted = false;
    ~job();
    bool is_aborted();
    void abort();
};

void job_abort_all();
job* job_new(int job_id, struct whisper_full_params params);
void job_remove(int job_id);
job* job_get(int job_id);

void high_pass_filter(std::vector<float> & data, float cutoff, float sample_rate);
bool vad_simple(std::vector<float> & pcmf32, int sample_rate, int last_ms, float vad_thold, float freq_thold, bool verbose);

} // namespace rnwhisper

#endif // RNWHISPER_H