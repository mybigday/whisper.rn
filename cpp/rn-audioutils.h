#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include "whisper.h"

namespace rnaudioutils {

std::vector<uint8_t> concat_short_buffers(const std::vector<short*>& buffers, const std::vector<int>& slice_n_samples);
void save_wav_file(const std::vector<uint8_t>& raw, const std::string& file);

} // namespace rnaudioutils
