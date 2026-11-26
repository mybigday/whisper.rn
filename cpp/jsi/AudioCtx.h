#pragma once

#include <algorithm>
#include <cstdint>

namespace rnwhisper_jsi {

// Whisper encoder audio context fitted to the input length, for audioCtx: 'auto'.
//
// One encoder position covers 20 ms (two 10 ms mel frames) = 320 samples at
// 16 kHz. The fitted context adds ~2 s (100 positions) of headroom, is aligned
// to 64, never goes below 256 and never exceeds maxAudioCtx (the model's
// n_audio_ctx, 1500 for the stock 30 s window).
//
// Kept header-only and free of whisper.cpp/JSI dependencies so it is easy to
// test and is compiled into the JSI glue on every platform, independent of the
// prebuilt core library version.
inline int fitAudioCtx(int64_t nSamples, int maxAudioCtx) {
    const int64_t samples = std::max<int64_t>(0, nSamples);
    const int64_t frames = (samples + 319) / 320;
    const int64_t padded = ((frames + 100 + 63) / 64) * 64;
    const int64_t fitted = std::max<int64_t>(256, padded);
    return static_cast<int>(std::min<int64_t>(maxAudioCtx, fitted));
}

// Samples whisper_full actually encodes, after the offset/duration options.
inline int64_t effectiveAudioSamples(int64_t nSamples, int offsetMs, int durationMs) {
    constexpr int64_t kSampleRate = 16000;
    int64_t samples = nSamples - std::max<int64_t>(0, offsetMs) * kSampleRate / 1000;
    if (durationMs > 0) {
        samples = std::min<int64_t>(samples, int64_t(durationMs) * kSampleRate / 1000);
    }
    return std::max<int64_t>(0, samples);
}

} // namespace rnwhisper_jsi
