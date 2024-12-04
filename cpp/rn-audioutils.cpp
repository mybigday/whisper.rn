#include "rn-audioutils.h"
#include "rn-whisper-log.h"

namespace rnaudioutils {

void append_wav_data(const short* data, const int n_samples, const std::string& file) {
    std::ofstream output(file, std::ios::binary | std::ios::app);
    output.write(reinterpret_cast<const char*>(data), n_samples * sizeof(short));
    output.close();
}

void add_wav_header_to_file(const std::string& file, const int data_size) {
    std::ofstream output(file, std::ios::binary | std::ios::app);

    if (!output.is_open()) {
        RNWHISPER_LOG_ERROR("Failed to open file for writing: %s\n", file.c_str());
        return;
    }

    output.seekp(0, std::ios::beg);

    output.write("RIFF", 4);
    int32_t chunk_size = 36 + static_cast<int32_t>(data_size);
    output.write(reinterpret_cast<char*>(&chunk_size), sizeof(chunk_size));
    output.write("WAVE", 4);
    output.write("fmt ", 4);
    int32_t sub_chunk_size = 16;
    output.write(reinterpret_cast<char*>(&sub_chunk_size), sizeof(sub_chunk_size));
    short audio_format = 1;
    output.write(reinterpret_cast<char*>(&audio_format), sizeof(audio_format));
    short num_channels = 1;
    output.write(reinterpret_cast<char*>(&num_channels), sizeof(num_channels));
    int32_t sample_rate = WHISPER_SAMPLE_RATE;
    output.write(reinterpret_cast<char*>(&sample_rate), sizeof(sample_rate));
    int32_t byte_rate = WHISPER_SAMPLE_RATE * 2;
    output.write(reinterpret_cast<char*>(&byte_rate), sizeof(byte_rate));
    short block_align = 2;
    output.write(reinterpret_cast<char*>(&block_align), sizeof(block_align));
    short bits_per_sample = 16;
    output.write(reinterpret_cast<char*>(&bits_per_sample), sizeof(bits_per_sample));
    output.write("data", 4);
    int32_t sub_chunk2_size = static_cast<int32_t>(data_size);
    output.write(reinterpret_cast<char*>(&sub_chunk2_size), sizeof(sub_chunk2_size));

    output.close();
}

} // namespace rnaudioutils
