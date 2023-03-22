
#ifdef __cplusplus
#include <string>
#include <whisper.h>
extern "C" {
#endif

void rn_whisper_convert_prompt(
  struct whisper_context * ctx,
  struct whisper_full_params params,
  std::string * prompt
);

bool* rn_whisper_assign_abort_map(int job_id);
void rn_whisper_remove_abort_map(int job_id);
void rn_whisper_abort_transcribe(int job_id);
void rn_whisper_abort_all_transcribe();

#ifdef __cplusplus
}
#endif