
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

#ifdef __cplusplus
}
#endif