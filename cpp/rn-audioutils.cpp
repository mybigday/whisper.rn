#include "rn-audioutils.h"
#include "rn-whisper-log.h"

#define WHISPER_SAMPLE_RATE 16000

namespace rnaudioutils {

// 1) Initialize: write a placeholder WAV header.
bool WavWriter::initialize(const std::string &filePath,
                           int sampleRate,
                           int numChannels,
                           int bitsPerSample) 
{
    // Store fields in header
    header.sampleRate    = sampleRate;
    header.numChannels   = numChannels;
    header.bitsPerSample = bitsPerSample;
    header.byteRate      = sampleRate * numChannels * (bitsPerSample / 8);
    header.blockAlign    = numChannels * (bitsPerSample / 8);

    // Try opening file
    outFile.open(filePath, std::ios::binary);
    if (!outFile.is_open()) {
        RNWHISPER_LOG_ERROR("WavWriter::initialize: Failed to open file: %s\n", filePath.c_str());
        return false;
    }

    // Write the placeholder header
    outFile.write(reinterpret_cast<const char*>(&header), sizeof(WavHeader));
    if (!outFile.good()) {
        RNWHISPER_LOG_ERROR("WavWriter::initialize: Failed to write placeholder header\n");
        outFile.close();
        return false;
    }

    isOpen = true;
    totalSamplesWritten = 0;
    RNWHISPER_LOG_INFO("WavWriter::initialize: Created WAV file: %s\n", filePath.c_str());
    return true;
}

// 2) Append PCM samples in real-time (16-bit).
bool WavWriter::appendSamples(const short *samples, size_t nSamples) {
    if (!isOpen) {
        RNWHISPER_LOG_ERROR("WavWriter::appendSamples: File not open!\n");
        return false;
    }

    // Write raw 16-bit samples
    outFile.write(reinterpret_cast<const char*>(samples), nSamples * sizeof(short));
    if (!outFile.good()) {
        RNWHISPER_LOG_ERROR("WavWriter::appendSamples: Failed to write PCM samples\n");
        return false;
    }

    // Keep track of how many samples total
    totalSamplesWritten += nSamples;
    return true;
}

// 3) Finalize: fix the chunk sizes in the WAV header & close the file.
void WavWriter::finalize() {
    if (!isOpen) {
        return; // Already closed or never opened
    }

    // Current file position = total file size
    std::streampos fileSize = outFile.tellp();

    // subChunk2Size = number of bytes of audio data
    // totalSamplesWritten is # of samples (each sample is 16 bits, 2 bytes)
    uint32_t dataSize = static_cast<uint32_t>(totalSamplesWritten * sizeof(short));
    // chunkSize = 4 (WAVE) + 8 (fmt chunk) + 16 (fmt data) + 8 (data chunk) + dataSize
    //            = 36 + dataSize
    uint32_t chunkSize = 36 + dataSize;

    // Seek back and update header fields:
    outFile.seekp(4, std::ios::beg);
    outFile.write(reinterpret_cast<const char*>(&chunkSize), sizeof(chunkSize));

    outFile.seekp(40, std::ios::beg);
    outFile.write(reinterpret_cast<const char*>(&dataSize), sizeof(dataSize));

    outFile.close();
    isOpen = false;

    RNWHISPER_LOG_INFO("WavWriter::finalize: Finalized WAV. Total samples = %zu\n", totalSamplesWritten);
}


std::vector<uint8_t> concat_short_buffers(const std::vector<short*>& buffers, const std::vector<int>& slice_n_samples) {
    
    RNWHISPER_LOG_INFO("Concating short buffer");
    std::vector<uint8_t> output_data;

    for (size_t i = 0; i < buffers.size(); i++) {
        int size = slice_n_samples[i]; // Number of shorts
        short* slice = buffers[i];

        // Copy each short as two bytes
        for (int j = 0; j < size; j++) {
            output_data.push_back(static_cast<uint8_t>(slice[j] & 0xFF));         // Lower byte
            output_data.push_back(static_cast<uint8_t>((slice[j] >> 8) & 0xFF));  // Higher byte
        }
    }

    return output_data;
}

std::vector<uint8_t> remove_trailing_zeros(const std::vector<uint8_t>& audio_data) {
    auto last = std::find_if(audio_data.rbegin(), audio_data.rend(), [](uint8_t byte) { return byte != 0; });
    return std::vector<uint8_t>(audio_data.begin(), last.base());
}

void save_wav_file(const std::vector<uint8_t>& raw, const std::string& file) {

    RNWHISPER_LOG_INFO("Saving audio file: %s\n", file.c_str());
    std::vector<uint8_t> data = remove_trailing_zeros(raw);
    
    std::ofstream output(file, std::ios::binary);

    if (!output.is_open()) {
        RNWHISPER_LOG_ERROR("Failed to open file for writing: %s\n", file.c_str());
        return;
    }

    // WAVE header
    output.write("RIFF", 4);
    int32_t chunk_size = 36 + static_cast<int32_t>(data.size());
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
    int32_t sub_chunk2_size = static_cast<int32_t>(data.size());
    output.write(reinterpret_cast<char*>(&sub_chunk2_size), sizeof(sub_chunk2_size));
    output.write(reinterpret_cast<const char*>(data.data()), data.size());

    output.close();

    RNWHISPER_LOG_INFO("Saved audio file: %s\n", file.c_str());
}

void raw_to_wav(const std::string& rawFilePath, const std::string& wavFilePath) {
    // 1) Read entire raw file
    std::ifstream in(rawFilePath, std::ios::binary);
    if (!in.is_open()) {
        RNWHISPER_LOG_ERROR("Cannot open raw file: %s\n", rawFilePath.c_str());
        return;
    }
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), {});
    in.close();

    // 2) Save as WAV
    save_wav_file(data, wavFilePath);
}

} // namespace rnaudioutils
