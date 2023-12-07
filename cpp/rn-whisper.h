#ifndef RNWHISPER_H
#define RNWHISPER_H

#include <string>
#include <vector>
#include "whisper.h"

namespace rnwhisper {

struct vad_params {
    bool use_vad = false;
    float vad_thold = 0.1;
    float freq_thold = 0.1;
    int vad_ms = 2000;
    int last_ms = 1000;
    bool verbose = false;
};

struct job {
    int job_id;
    bool aborted = false;
    whisper_full_params params;
    vad_params vad; // Realtime transcription only
    
    ~job();
    void set_vad_params(vad_params vad);
    bool vad_simple(short* pcm, int n_samples, int n);
    bool is_aborted();
    void abort();
};

void job_abort_all();
job* job_new(int job_id, struct whisper_full_params params);
void job_remove(int job_id);
job* job_get(int job_id);

} // namespace rnwhisper

#endif // RNWHISPER_H