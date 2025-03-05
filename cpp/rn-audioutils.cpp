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

void WavWriter::finalize() {
    if (!isOpen) {
        return; // Already closed or never opened
    }

    // Get the total file size
    outFile.flush(); // Ensure all data is written
    std::streampos fileSize = outFile.tellp();
    if (fileSize < 44) {
        RNWHISPER_LOG_ERROR("WavWriter::finalize: File size too small for WAV\n");
        outFile.close();
        isOpen = false;
        return;
    }

    // Cast fileSize to a compatible type for arithmetic
    uint32_t totalFileSize = static_cast<uint32_t>(fileSize);
    // Calculate sizes
    uint32_t dataSize = static_cast<uint32_t>(totalFileSize - 44); // Size of PCM data
    uint32_t chunkSize = 36 + dataSize; // Total size minus RIFF header

    // Update header fields
    outFile.seekp(4, std::ios::beg);
    outFile.write(reinterpret_cast<const char*>(&chunkSize), sizeof(chunkSize));

    outFile.seekp(40, std::ios::beg);
    outFile.write(reinterpret_cast<const char*>(&dataSize), sizeof(dataSize));

    // Close the file
    outFile.close();
    isOpen = false;

    RNWHISPER_LOG_INFO("WavWriter::finalize: Finalized WAV. Total file size = %lld bytes\n", totalFileSize);
}
} // namespace rnaudioutils
