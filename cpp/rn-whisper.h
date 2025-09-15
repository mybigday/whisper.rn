#ifndef RNWHISPER_H
#define RNWHISPER_H

#include <string>
#include <vector>
#include "whisper.h"
#include "rn-whisper-log.h"

namespace rnwhisper {

std::string bench(whisper_context * ctx, int n_threads);

struct vad_params {
    bool use_vad = false;
    float vad_thold = 0.6f;
    float freq_thold = 100.0f;
    int vad_ms = 2000;
    int last_ms = 1000;
    bool verbose = false;
};

struct job {
    int job_id;
    bool aborted = false;
    whisper_full_params params;

    ~job();
    bool is_aborted();
    void abort();
};

void job_abort_all();
job* job_new(int job_id, struct whisper_full_params params);
void job_remove(int job_id);
job* job_get(int job_id);

} // namespace rnwhisper

#endif // RNWHISPER_H
