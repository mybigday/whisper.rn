
#ifdef __cplusplus
#include <string>
#include <whisper.h>
extern "C" {
#endif

bool* rn_whisper_assign_abort_map(int job_id);
void rn_whisper_remove_abort_map(int job_id);
void rn_whisper_abort_transcribe(int job_id);
bool rn_whisper_transcribe_is_aborted(int job_id);
void rn_whisper_abort_all_transcribe();
bool rn_whisper_vad_simple(std::vector<float> & pcmf32, int sample_rate, int last_ms, float vad_thold, float freq_thold, bool verbose);

#ifdef __cplusplus
}
#endif
