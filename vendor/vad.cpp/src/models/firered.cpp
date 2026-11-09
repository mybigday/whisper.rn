// FireRedVAD (FireRedTeam/FireRedVAD, Apache-2.0): DFSMN on Kaldi fbank
// features, one probability per 10 ms frame. Two variants:
//  - Stream-VAD: causal (lookback only), streaming with per-layer caches;
//  - VAD: each FSMN layer also sees N2 future frames (lookahead), so it runs
//    offline over a whole utterance (zero padding beyond its ends, as the
//    reference), with lookback and lookahead merged into one N1+N2 tap conv.
//
// Features (kaldi-native-fbank defaults used by FireRedVAD): 16 kHz int16-scaled
// samples, 25 ms Povey window, 10 ms hop, snip_edges, DC removal, pre-emphasis
// 0.97, 512-point power spectrum, 80 mel bins (20 Hz .. 8 kHz), log with
// FLT_EPSILON floor, then CMVN. The spectrum is a CPU FFT op (frame_dft.h); CMVN is
// folded into the first Linear layer.
//
// Model: fc1 (80->H) relu, fc2 (H->P) relu, FSMN, (R-1) x [fc1 relu, fc2 (no bias),
// FSMN, + residual], M x [Linear relu], Linear(H->1), sigmoid. FSMN(p) = p + causal
// depthwise conv over the last N1 frames (ggml_ssm_conv).
//
// Per-stream state: [audio context 240 | R x FSMN cache (N1-1 frames x P, time-major)].
#include "common.h"
#include "frame_dft.h"

#include <cfloat>
#include <cmath>
#include <cstring>

namespace vad {

namespace {

constexpr int SR = 16000, WIN = 400, HOP = 160, CTX = WIN - HOP, NFFT = 512, NBIN = NFFT / 2 + 1, NMEL = 80;


// exact: activations must stay F32 on every backend. The power spectrum spans
// ~1e-10 .. 1e11, far outside F16, so the mel projection may not be converted
// (ggml-vulkan converts src1 to F16 and rejects F32-only mul_mat, so the
// scheduler runs that one op on the CPU).
ggml_tensor * mm(ggml_context * c, ggml_tensor * w, ggml_tensor * x, const graph_opts & o, bool exact = false) {
    if (!o.host && !ggml_is_contiguous(x)) x = ggml_cont(c, x);
    ggml_tensor * y = ggml_mul_mat(c, w, x);
    ggml_prec_set_acc(y, GGML_PREC_F32);
    ggml_prec_set_src(y, o.host || exact ? GGML_PREC_F32 : GGML_PREC_F16, 1);
    if (exact && !o.host && o.pin_cpu) o.pin_cpu->push_back(y);
    return y;
}

class firered : public arch {
public:

    // hyper-parameters from the metadata; tensors are checked by create_weights
    bool configure(const model_file & f, std::string & err) {
        auto kv = [&](const char * k, int64_t & v) {
            auto it = f.kv_int.find(k);
            if (it == f.kv_int.end()) { err = std::string("firered: missing metadata ") + k; return false; }
            v = it->second;
            return true;
        };
        int64_t sr, win, hop, nmel;
        if (!kv("preprocessor.sample_rate", sr) || !kv("preprocessor.frame_length", win) ||
            !kv("preprocessor.frame_shift", hop) || !kv("preprocessor.num_mel_bins", nmel) ||
            !kv("n_blocks", R_) || !kv("n_dnn", M_) || !kv("hidden", H_) || !kv("proj", P_) || !kv("lookback", N1_) ||
            !kv("lookahead", N2_)) return false;
        if (sr != SR || win != WIN || hop != HOP || nmel != NMEL || N1_ < 2 || N2_ < 0 || R_ < 1 || M_ < 1) {
            err = "firered: unsupported configuration"; return false;
        }
        return true;
    }

    const char * name() const override { return "firered"; }
    int sample_rate() const override { return SR; }
    int chunk_size() const override { return HOP; }
    int warmup_samples() const override { return CTX; }
    int state_size() const override { return N2_ > 0 ? 0 : CTX + (int) (R_ * (N1_ - 1) * P_); }
    bool streaming() const override { return N2_ == 0; }
    int lookback_frames() const override { return (int) (R_ * (N1_ - 1)); }
    int lookahead_frames() const override { return (int) (R_ * N2_); }
    int64_t audio_samples(int T) const override { return N2_ > 0 ? CTX + (int64_t) HOP * T : (int64_t) HOP * T; }
    size_t max_nodes(int, int) const override { return 256 + (size_t) R_ * 24; }

    bool create_weights(weight_builder & wb) override {
        // fbank front end: int16 scale, DC removal, pre-emphasis, Povey window, 512-point FFT -> [Re(0..256), Im(0..256)]
        {
            std::vector<float> win(WIN);
            for (int n = 0; n < WIN; ++n) win[n] = (float) std::pow(0.5 - 0.5 * std::cos(2 * M_PI * n / (WIN - 1)), 0.85);
            if (!dft_.init(WIN, NFFT, std::move(win), 32768.0f, true, 0.97f)) return false;
        }
        // mel filterbank [257 -> 80] (Kaldi: bins 0..255, Nyquist bin unused)
        {
            auto mel = [](double f) { return 1127.0 * std::log(1.0 + f / 700.0); };
            const double lo = mel(20.0), hi = mel(SR / 2.0), delta = (hi - lo) / (NMEL + 1);
            std::vector<float> w((size_t) NMEL * NBIN, 0.0f);
            for (int b = 0; b < NMEL; ++b) {
                const double l = lo + b * delta, cnt = l + delta, r = cnt + delta;
                for (int i = 0; i < NFFT / 2; ++i) {
                    const double m = mel((double) i * SR / NFFT);
                    double v = 0;
                    if (m > l && m < r) v = m <= cnt ? (m - l) / (cnt - l) : (r - m) / (r - cnt);
                    w[(size_t) b * NBIN + i] = (float) v;
                }
            }
            mel_ = wb.constant("fbank.mel", { NBIN, NMEL }, w);
        }
        // model (docs/gguf.md): CMVN is folded into fc1 by the converter; FSMN taps are
        // [N1 + N2, P] (lookback then lookahead)
        fc1_w_ = wb.mat("fc1.w", NMEL, H_);
        fc1_b_ = wb.f32("fc1.b", { H_ });
        fc2_w_ = wb.mat("fc2.w", H_, P_);
        fc2_b_ = wb.f32("fc2.b", { P_ });
        for (int l = 0; l < R_; ++l) fsmn_w_.push_back(wb.f32("fsmn" + std::to_string(l) + ".w", { N1_ + N2_, P_ }));
        for (int i = 0; i < R_ - 1; ++i) {
            const std::string q = "blk" + std::to_string(i) + ".";
            blk_fc1_w_.push_back(wb.mat(q + "fc1.w", P_, H_));
            blk_fc1_b_.push_back(wb.f32(q + "fc1.b", { H_ }));
            blk_fc2_w_.push_back(wb.mat(q + "fc2.w", H_, P_));
        }
        for (int i = 0; i < M_; ++i) {
            const std::string q = "dnn" + std::to_string(i) + ".";
            dnn_w_.push_back(wb.mat(q + "w", i == 0 ? P_ : H_, H_));
            dnn_b_.push_back(wb.f32(q + "b", { H_ }));
        }
        out_w_ = wb.mat("out.w", H_, 1);
        out_b_ = wb.f32("out.b", { 1 });
        return wb.ok();
    }

    graph_io build_graph(ggml_context * c, ggml_cgraph * gf, int B, int T, const graph_opts & o, ggml_tensor * state) override {
        graph_io io;
        io.audio = ggml_new_tensor_2d(c, GGML_TYPE_F32, audio_samples(T), B);
        ggml_set_name(io.audio, "audio"); ggml_set_input(io.audio);
        // fbank: streaming prepends the stored 240-sample context; offline gets it in the input
        ggml_tensor * full = io.audio;
        ggml_tensor * mask = nullptr;
        if (N2_ == 0) {
            ggml_tensor * ctx0 = ggml_view_2d(c, state, CTX, B, state->nb[1], 0);
            full = ggml_concat(c, o.host ? ctx0 : ggml_cont(c, ctx0), io.audio, 0);             // [240 + 160T, B]
        } else {
            io.frame_mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, T, B);
            ggml_set_name(io.frame_mask, "frame_mask"); ggml_set_input(io.frame_mask);
            mask = ggml_reshape_3d(c, io.frame_mask, 1, T, B);
        }
        ggml_tensor * frames = sliding_windows(c, full, WIN, T, HOP, B, full->nb[1], o);
        ggml_tensor * spec = dft_.build(c, frames, o);                                              // [514, T, B]
        ggml_tensor * re = ggml_view_3d(c, spec, NBIN, T, B, spec->nb[1], spec->nb[2], 0);
        ggml_tensor * im = ggml_view_3d(c, spec, NBIN, T, B, spec->nb[1], spec->nb[2], NBIN * sizeof(float));
        ggml_tensor * pw = ggml_add(c, ggml_mul(c, re, re), ggml_mul(c, im, im));               // [257, T, B]
        ggml_tensor * x = mm(c, mel_, pw, o, true);                                                // [80, T, B]
        x = ggml_log(c, ggml_clamp(c, x, FLT_EPSILON, FLT_MAX));

        // DFSMN
        x = ggml_relu(c, ggml_add(c, mm(c, fc1_w_, x, o), fc1_b_));
        ggml_tensor * p = ggml_relu(c, ggml_add(c, mm(c, fc2_w_, x, o), fc2_b_));             // [P, T, B]
        ggml_tensor * memory = N2_ ? fsmn_offline(c, p, 0, mask) : fsmn(c, p, 0, T, state, o, gf);
        for (int i = 0; i < R_ - 1; ++i) {
            ggml_tensor * h = ggml_relu(c, ggml_add(c, mm(c, blk_fc1_w_[i], memory, o), blk_fc1_b_[i]));
            ggml_tensor * pp = mm(c, blk_fc2_w_[i], h, o);
            memory = ggml_add(c, N2_ ? fsmn_offline(c, pp, i + 1, mask) : fsmn(c, pp, i + 1, T, state, o, gf), memory);
        }
        x = memory;
        for (int i = 0; i < M_; ++i) x = ggml_relu(c, ggml_add(c, mm(c, dnn_w_[i], x, o), dnn_b_[i]));
        ggml_tensor * y = ggml_sigmoid(c, ggml_add(c, mm(c, out_w_, x, o), out_b_));           // [1, T, B]
        io.probs = ggml_reshape_2d(c, y, T, B);
        ggml_set_name(io.probs, "probs"); ggml_set_output(io.probs);

        ggml_build_forward_expand(gf, io.probs);
        if (N2_ > 0) return io;
        // new audio context, in place (FSMN caches are written by fsmn())
        const size_t sb = state->nb[1];
        ggml_tensor * ctx1 = ggml_view_2d(c, full, CTX, B, full->nb[1], (size_t) HOP * T * sizeof(float));
        ggml_build_forward_expand(gf, ggml_cpy(c, ctx1, ggml_view_2d(c, state, CTX, B, sb, 0)));
        return io;
    }

private:
    // p [P, T, B] -> p + causal depthwise conv over the cached and current frames.
    // The layer's cache (last N1-1 frames of p, time-major) is updated in place.
    ggml_tensor * fsmn(ggml_context * c, ggml_tensor * p, int l, int T, ggml_tensor * state, const graph_opts & o,
                       ggml_cgraph * gf) {
        const int64_t C = N1_ - 1, B = state->ne[1];
        const size_t off = (CTX + (size_t) l * C * P_) * sizeof(float);
        ggml_tensor * cache = ggml_view_3d(c, state, C, P_, B, C * sizeof(float), state->nb[1], off);        // [C, P, B]
        ggml_tensor * pt = ggml_cont(c, ggml_permute(c, p, 1, 0, 2, 3));                                     // [T, P, B]
        ggml_tensor * sx = ggml_concat(c, o.host ? cache : ggml_cont(c, cache), pt, 0);                      // [C+T, P, B]
        ggml_tensor * y = ggml_add(c, p, ggml_ssm_conv(c, sx, fsmn_w_[l]));                                // [P, T, B]
        ggml_tensor * tail = ggml_view_3d(c, sx, C, P_, B, sx->nb[1], sx->nb[2], (size_t) T * sizeof(float));
        ggml_build_forward_expand(gf, ggml_cpy(c, tail, cache));
        return y;
    }

    // offline FSMN: p (zeroed beyond the utterance) padded with N1-1 zeros before and
    // N2 after, one conv over frames t-N1+1 .. t+N2
    ggml_tensor * fsmn_offline(ggml_context * c, ggml_tensor * p, int l, ggml_tensor * mask) {
        ggml_tensor * pm = ggml_mul(c, p, mask);                                                     // [P, T, B]
        ggml_tensor * pt = ggml_cont(c, ggml_permute(c, pm, 1, 0, 2, 3));                            // [T, P, B]
        ggml_tensor * sx = ggml_pad_ext(c, pt, (int) (N1_ - 1), (int) N2_, 0, 0, 0, 0, 0, 0);       // [N1-1+T+N2, P, B]
        return ggml_add(c, pm, ggml_ssm_conv(c, sx, fsmn_w_[l]));
    }

    int64_t R_ = 0, M_ = 0, H_ = 0, P_ = 0, N1_ = 0, N2_ = 0;
    frame_dft dft_;
    ggml_tensor * mel_ = nullptr, * fc1_w_ = nullptr, * fc1_b_ = nullptr, * fc2_w_ = nullptr, * fc2_b_ = nullptr;
    ggml_tensor * out_w_ = nullptr, * out_b_ = nullptr;
    std::vector<ggml_tensor *> fsmn_w_, blk_fc1_w_, blk_fc1_b_, blk_fc2_w_, dnn_w_, dnn_b_;
};

} // namespace

std::unique_ptr<arch> make_firered(const model_file & file, std::string & err) {
    auto m = std::make_unique<firered>();
    if (!m->configure(file, err)) return nullptr;
    return m;
}

} // namespace vad
