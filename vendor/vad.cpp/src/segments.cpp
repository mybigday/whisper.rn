// Port of silero-vad v6.2.3 get_speech_timestamps_from_probs (MIT, Silero Team),
// sample-domain output, use_max_poss_sil_at_max_speech = True, step = 1.
#include "vad.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

vad_segment_params vad_segment_default_params(void) {
    vad_segment_params p;
    p.threshold = 0.5f;
    p.neg_threshold = -1.0f;
    p.min_speech_ms = 250;
    p.min_silence_ms = 100;
    p.speech_pad_ms = 30;
    p.max_speech_s = 0.0f;
    return p;
}

namespace {

struct seg { double start = -1, end = -1; };

// one channel: probs[i * stride]
std::vector<seg> segment_channel(const float * probs, int stride, int n_probs, int chunk_size, int sr, int64_t n_samples,
                                 const vad_segment_params & p) {
    const double audio_len = n_samples > 0 ? (double) n_samples : (double) n_probs * chunk_size;
    const double min_speech  = sr * p.min_speech_ms / 1000.0;
    const double pad         = sr * p.speech_pad_ms / 1000.0;
    const double max_speech  = p.max_speech_s > 0 ? sr * (double) p.max_speech_s - chunk_size - 2 * pad
                                                  : std::numeric_limits<double>::infinity();
    const double min_silence = sr * p.min_silence_ms / 1000.0;
    const double min_silence_at_max = sr * 98 / 1000.0;
    const float neg = p.neg_threshold >= 0 ? p.neg_threshold : std::max(p.threshold - 0.15f, 0.01f);

    std::vector<seg> speeches;
    seg cur;   // start < 0: empty
    bool triggered = false;
    double temp_end = 0, prev_end = 0, next_start = 0;
    std::vector<std::pair<double, double>> possible_ends;

    for (int i = 0; i < n_probs; ++i) {
        const float prob = probs[(size_t) i * stride];
        const double s = (double) chunk_size * i;

        if (prob >= p.threshold && temp_end) {
            const double sil = s - temp_end;
            if (sil > min_silence_at_max) possible_ends.emplace_back(temp_end, sil);
            temp_end = 0;
            if (next_start < prev_end) next_start = s;
        }
        if (prob >= p.threshold && !triggered) {
            triggered = true;
            cur.start = s;
            continue;
        }
        if (triggered && s - cur.start > max_speech) {
            if (!possible_ends.empty()) {
                auto best = *std::max_element(possible_ends.begin(), possible_ends.end(),
                                              [](auto & a, auto & b) { return a.second < b.second; });
                prev_end = best.first;
                cur.end = prev_end;
                speeches.push_back(cur);
                cur = seg();
                next_start = prev_end + best.second;
                if (next_start < prev_end + s) cur.start = next_start;
                else triggered = false;
                prev_end = next_start = temp_end = 0;
                possible_ends.clear();
            } else {
                cur.end = s;
                speeches.push_back(cur);
                cur = seg();
                prev_end = next_start = temp_end = 0;
                triggered = false;
                possible_ends.clear();
                continue;
            }
        }
        if (prob < neg && triggered) {
            if (!temp_end) temp_end = s;
            if (s - temp_end < min_silence) continue;
            cur.end = temp_end;
            if (cur.end - cur.start > min_speech) speeches.push_back(cur);
            cur = seg();
            prev_end = next_start = temp_end = 0;
            triggered = false;
            possible_ends.clear();
            continue;
        }
    }
    if (cur.start >= 0 && audio_len - cur.start > min_speech) {
        cur.end = audio_len;
        speeches.push_back(cur);
    }

    const int n = (int) speeches.size();
    for (int i = 0; i < n; ++i) {
        seg & sp = speeches[i];
        if (i == 0) sp.start = std::floor(std::max(0.0, sp.start - pad));
        if (i != n - 1) {
            const double silence = speeches[i + 1].start - sp.end;
            if (silence < 2 * pad) {
                sp.end += std::floor(silence / 2);
                speeches[i + 1].start = std::floor(std::max(0.0, speeches[i + 1].start - std::floor(silence / 2)));
            } else {
                sp.end = std::floor(std::min(audio_len, sp.end + pad));
                speeches[i + 1].start = std::floor(std::max(0.0, speeches[i + 1].start - pad));
            }
        } else {
            sp.end = std::floor(std::min(audio_len, sp.end + pad));
        }
    }
    return speeches;
}

} // namespace

int vad_segments_from_probs(const float * probs, int n_frames, int n_outputs, int hop, int sr, int64_t n_samples,
                            vad_segment_params p, vad_segment * out, int max_out) {
    if (n_outputs < 1) return 0;
    std::vector<vad_segment> all;
    for (int ch = 0; ch < n_outputs; ++ch)
        for (const seg & s : segment_channel(probs + ch, n_outputs, n_frames, hop, sr, n_samples, p))
            all.push_back({ (int64_t) s.start, (int64_t) s.end, ch });
    std::stable_sort(all.begin(), all.end(), [](const vad_segment & a, const vad_segment & b) {
        return a.start != b.start ? a.start < b.start : a.channel < b.channel;
    });
    for (size_t i = 0; i < all.size() && (int) i < max_out; ++i) out[i] = all[i];
    return (int) all.size();
}
