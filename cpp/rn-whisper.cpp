#include <cstdio>
#include <string>
#include <vector>
#include "whisper.h"

extern "C" {

void rn_whisper_convert_prompt(
  struct whisper_context * ctx,
  struct whisper_full_params params,
  std::string * prompt
) {
  std::vector<whisper_token> prompt_tokens;
  if (!prompt->empty()) {
    prompt_tokens.resize(1024);
    prompt_tokens.resize(whisper_tokenize(ctx, prompt->c_str(), prompt_tokens.data(), prompt_tokens.size()));

    // fprintf(stderr, "\n");
    // fprintf(stderr, "initial prompt: '%s'\n", prompt->c_str());
    // fprintf(stderr, "initial tokens: [ ");
    // for (int i = 0; i < (int) prompt_tokens.size(); ++i) {
    //     fprintf(stderr, "%d ", prompt_tokens[i]);
    // }
    // fprintf(stderr, "]\n");

    params.prompt_tokens = prompt_tokens.data();
    params.prompt_n_tokens = prompt_tokens.size();
  }
}

}