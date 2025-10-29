#ifndef RNWHISPER_H
#define RNWHISPER_H

#include <string>
#include <vector>
#include "whisper.h"
#include "rn-whisper-log.h"
#include "rn-audioutils.h"

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
    int n_processors = 1;

    ~job();
    bool is_aborted();
    void abort();

    // Realtime transcription only:
    vad_params vad;
    int audio_sec = 0;
    int audio_slice_sec = 0;
    float audio_min_sec = 0;
    const char* audio_output_path = nullptr;
    std::vector<short *> pcm_slices;
    void set_realtime_params(vad_params vad, int sec, int slice_sec, float min_sec, const char* output_path);
    bool vad_simple(int slice_index, int n_samples, int n);
    void put_pcm_data(short* pcm, int slice_index, int n_samples, int n);
    float* pcm_slice_to_f32(int slice_index, int size);
};

void job_abort_all();
job* job_new(int job_id, struct whisper_full_params params);
void job_remove(int job_id);
job* job_get(int job_id);

} // namespace rnwhisper

#endif // RNWHISPER_H
