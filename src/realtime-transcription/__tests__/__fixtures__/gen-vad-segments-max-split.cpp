// Generates vad-segments-max-split.json: offline references (vad.cpp's
// vad_segments_from_probs, i.e. silero-vad's get_speech_timestamps_from_probs)
// for StreamingVadSegmenter's max-duration split.
//
//   g++ -O2 -std=c++17 -Ivendor/vad.cpp/include \
//     src/realtime-transcription/__tests__/__fixtures__/gen-vad-segments-max-split.cpp \
//     vendor/vad.cpp/src/segments.cpp -o /tmp/gen-vad-segments &&
//   /tmp/gen-vad-segments > src/realtime-transcription/__tests__/__fixtures__/vad-segments-max-split.json
#include "vad.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    const int hop = 512, sr = 16000;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    std::printf("[\n");
    const int n_cases = 16;
    for (int c = 0; c < n_cases; ++c) {
        vad_segment_params p = vad_segment_default_params();
        p.threshold = std::round((0.4f + 0.2f * u(rng)) * 100.f) / 100.f + 0.0005f;
        p.neg_threshold = -1.f;
        p.min_speech_ms = (int) (250 * u(rng));
        p.min_silence_ms = 100 + (int) (400 * u(rng));
        p.speech_pad_ms = (int) (120 * u(rng));
        p.max_speech_s = 1.f + 3.f * u(rng);
        // long speech runs with short pauses (shorter than min silence) and some long ones
        std::vector<float> probs;
        const int n = 400 + (int) (400 * u(rng));
        while ((int) probs.size() < n) {
            int speech = 3 + (int) (60 * u(rng));
            for (int i = 0; i < speech; ++i) probs.push_back(0.55f + 0.45f * u(rng));
            float r = u(rng);
            int pause = r < 0.7f ? 1 + (int) (5 * u(rng)) : r < 0.9f ? 1 + (int) (12 * u(rng)) : 10 + (int) (40 * u(rng));
            for (int i = 0; i < pause; ++i) probs.push_back(0.3f * u(rng));
            if (u(rng) < 0.2f) probs.push_back(0.3f + 0.4f * u(rng));  // ambiguous frame
        }
        probs.resize(n);
        for (auto &x : probs) x = std::round(x * 1000.f) / 1000.f;
        std::vector<vad_segment> segs(4096);
        int ns = vad_segments_from_probs(probs.data(), n, 1, hop, sr, 0, p, segs.data(), (int) segs.size());
        std::printf("  {\"params\": {\"threshold\": %.6g, \"negThreshold\": -1, \"minSpeechDurationMs\": %d, "
                    "\"minSilenceDurationMs\": %d, \"speechPadMs\": %d, \"maxSpeechDurationS\": %.6g},\n   \"probs\": [",
                    p.threshold, p.min_speech_ms, p.min_silence_ms, p.speech_pad_ms, p.max_speech_s);
        for (int i = 0; i < n; ++i) std::printf("%s%.3f", i ? "," : "", probs[i]);
        std::printf("],\n   \"segments\": [");
        for (int i = 0; i < ns; ++i) std::printf("%s[%lld,%lld]", i ? "," : "", (long long) segs[i].start, (long long) segs[i].end);
        std::printf("]}%s\n", c + 1 < n_cases ? "," : "");
    }
    std::printf("]\n");
}
