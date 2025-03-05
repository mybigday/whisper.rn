#ifndef RN_AUDIOUTILS_H
#define RN_AUDIOUTILS_H

#include <string>
#include <fstream>
#include <vector>
#include <cstdint>

namespace rnaudioutils {

// Simple WAV header struct (16-bit PCM, 1 channel @ 16000 Hz).
// Adjust fields if your format is different.
#pragma pack(push, 1)
struct WavHeader {
    char riff[4] = {'R','I','F','F'};
    uint32_t chunkSize = 36;      // Will fix later
    char wave[4] = {'W','A','V','E'};
    char fmt[4] = {'f','m','t',' '};
    uint32_t subChunkSize = 16;   // PCM
    uint16_t audioFormat = 1;     // PCM = 1
    uint16_t numChannels = 1;     // mono
    uint32_t sampleRate = 16000;  // 16 kHz
    uint32_t byteRate = 0;        // Will compute
    uint16_t blockAlign = 0;      // Will compute
    uint16_t bitsPerSample = 16;  // 16-bit
    char data[4] = {'d','a','t','a'};
    uint32_t subChunk2Size = 0;   // Will fix later
};
#pragma pack(pop)

class WavWriter {
public:
    // Call this first:
    bool initialize(const std::string &filePath,
                    int sampleRate = 16000,
                    int numChannels = 1,
                    int bitsPerSample = 16);

    // Call this repeatedly to append PCM data:
    bool appendSamples(const short *samples, size_t nSamples);

    // Call this once at the end to fix header sizes & close file:
    void finalize();

private:
    std::ofstream outFile;
    WavHeader header;
    bool isOpen = false;
    size_t totalSamplesWritten = 0;
};
} // namespace rnaudioutils

#endif // RN_AUDIOUTILS_H
