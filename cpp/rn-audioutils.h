#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include "whisper.h"

namespace rnaudioutils {

void append_wav_data(const short* data, const int n_samples, const std::string& file);
void add_wav_header_to_file(const std::string& file, const int data_size);

} // namespace rnaudioutils
