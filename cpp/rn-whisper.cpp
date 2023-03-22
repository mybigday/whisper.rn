#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>
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

void rn_whisper_abort_all_transcribe() {
  for (auto it = abort_map.begin(); it != abort_map.end(); ++it) {
    it->second = true;
  }
}

}