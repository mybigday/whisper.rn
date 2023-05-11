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

void rn_whisper_abort_all_transcribe() {
  for (auto it = abort_map.begin(); it != abort_map.end(); ++it) {
    it->second = true;
  }
}

}