// smart-turn-v3 (pipecat-ai, BSD-2-Clause): turn-completion classifier on the
// last 8 s of audio. Not a per-frame VAD: one probability per 8 s window.
//
// Front end = transformers WhisperFeatureExtractor(chunk_length=8, do_normalize):
// zero-mean/unit-variance waveform, STFT (n_fft 400, hop 160, periodic Hann,
// center reflect padding), power, slaney mel (80), log10 floored at 1e-10,
// max(x, max(x) - 8), (x + 4) / 4 -> [80 mel, 800 frames].
// Model = Whisper encoder (conv1 k3, conv2 k3 s2, GELU, positions, pre-LN
// transformer layers, final LN) -> attention pooling over time -> classifier
// (Linear, LayerNorm, GELU, Linear, GELU, Linear) -> sigmoid.
#include "common.h"
#include "frame_dft.h"

#include <cfloat>
#include <cmath>

namespace vad {

namespace {

constexpr int SR = 16000, NFFT = 400, HOP = 160, NBIN = NFFT / 2 + 1, NMEL = 80;


// see firered.cpp: exact = activations must stay F32 on every backend
ggml_tensor * mm(ggml_context * c, ggml_tensor * w, ggml_tensor * x, const graph_opts & o, bool exact = false) {
    if (!o.host && !ggml_is_contiguous(x)) x = ggml_cont(c, x);
    ggml_tensor * y = ggml_mul_mat(c, w, x);
    ggml_prec_set_acc(y, GGML_PREC_F32);
    ggml_prec_set_src(y, o.host || exact ? GGML_PREC_F32 : GGML_PREC_F16, 1);
    if (exact && !o.host && o.pin_cpu) o.pin_cpu->push_back(y);
    return y;
}

class smart_turn : public arch {
public:

    bool configure(const model_file & f, std::string & err) {
        auto kv = [&](const char * k, int64_t & v) {
            auto it = f.kv_int.find(k);
            if (it == f.kv_int.end()) { err = std::string("smart_turn: missing metadata ") + k; return false; }
            v = it->second;
            return true;
        };
        int64_t sr, nm, nf, hop;
        if (!kv("preprocessor.sample_rate", sr) || !kv("window_samples", window_) || !kv("preprocessor.n_mels", nm) ||
            !kv("preprocessor.n_fft", nf) || !kv("preprocessor.hop", hop) || !kv("n_layers", L_) || !kv("d_model", D_) || !kv("n_heads", NH_) || !kv("n_ctx", NCTX_)) return false;
        if (sr != SR || nm != NMEL || nf != NFFT || hop != HOP || D_ % NH_ != 0 || window_ % HOP != 0 ||
            window_ / HOP / 2 != NCTX_) { err = "smart_turn: unsupported configuration"; return false; }
        return true;
    }

    const char * name() const override { return "smart_turn"; }
    bool is_turn() const override { return true; }
    int sample_rate() const override { return SR; }
    int chunk_size() const override { return (int) window_; }
    int state_size() const override { return 0; }
    size_t max_nodes(int, int) const override { return 256 + (size_t) L_ * 64; }

    bool create_weights(weight_builder & wb) override {
        // STFT: periodic Hann window, 400-point FFT -> [Re(0..200), Im(0..200)]
        {
            std::vector<float> win(NFFT);
            for (int n = 0; n < NFFT; ++n) win[n] = (float) (0.5 - 0.5 * std::cos(2 * M_PI * n / NFFT));
            if (!dft_.init(NFFT, NFFT, std::move(win))) return false;
        }
        // model (docs/gguf.md); conv weights are (Cout, 3, Cin) to match the [3 * Cin] window view
        const int64_t F = 4 * D_;
        // always F32: its input, the power spectrum, reaches ~1e5 and overflows F16, so
        // a backend that converts activations to the weight type (ggml-cpu, GPUs) must not see F16
        mel_ = wb.f32("fe.mel", { NBIN, NMEL });
        conv1_w_ = wb.mat("conv1.w", 3 * NMEL, D_);
        conv1_b_ = wb.f32("conv1.b", { D_ });
        conv2_w_ = wb.mat("conv2.w", 3 * D_, D_);
        conv2_b_ = wb.f32("conv2.b", { D_ });
        pos_ = wb.f32("pos", { D_, NCTX_ });
        for (int l = 0; l < L_; ++l) {
            const std::string q = "l" + std::to_string(l) + ".";
            layer L;
            L.ln1_w = wb.f32(q + "ln1.w", { D_ });
            L.ln1_b = wb.f32(q + "ln1.b", { D_ });
            L.wq = wb.mat(q + "wq", D_, D_);
            L.bq = wb.f32(q + "bq", { D_ });
            L.wk = wb.mat(q + "wk", D_, D_);
            L.wv = wb.mat(q + "wv", D_, D_);
            L.bv = wb.f32(q + "bv", { D_ });
            L.wo = wb.mat(q + "wo", D_, D_);
            L.bo = wb.f32(q + "bo", { D_ });
            L.ln2_w = wb.f32(q + "ln2.w", { D_ });
            L.ln2_b = wb.f32(q + "ln2.b", { D_ });
            L.w1 = wb.mat(q + "fc1.w", D_, F);
            L.b1 = wb.f32(q + "fc1.b", { F });
            L.w2 = wb.mat(q + "fc2.w", F, D_);
            L.b2 = wb.f32(q + "fc2.b", { D_ });
            layers_.push_back(L);
        }
        lnf_w_ = wb.f32("lnf.w", { D_ });
        lnf_b_ = wb.f32("lnf.b", { D_ });
        pa0_w_ = wb.mat("pa0.w", D_, 256);
        pa0_b_ = wb.f32("pa0.b", { 256 });
        pa2_w_ = wb.mat("pa2.w", 256, 1);
        pa2_b_ = wb.f32("pa2.b", { 1 });
        c0_w_ = wb.mat("c0.w", D_, 256);
        c0_b_ = wb.f32("c0.b", { 256 });
        c1_w_ = wb.f32("c1.w", { 256 });
        c1_b_ = wb.f32("c1.b", { 256 });
        c4_w_ = wb.mat("c4.w", 256, 64);
        c4_b_ = wb.f32("c4.b", { 64 });
        c6_w_ = wb.mat("c6.w", 64, 1);
        c6_b_ = wb.f32("c6.b", { 1 });
        return wb.ok();
    }

    graph_io build_graph(ggml_context * c, ggml_cgraph * gf, int B, int T, const graph_opts & o, ggml_tensor *) override {
        GGML_ASSERT(T == 1);
        graph_io io;
        const int64_t NF = window_ / HOP, D = D_, HD = D_ / NH_;
        io.audio = ggml_new_tensor_2d(c, GGML_TYPE_F32, window_, B);
        ggml_set_name(io.audio, "audio"); ggml_set_input(io.audio);

        // --- Whisper log-mel [80, 800, B]
        ggml_tensor * x = ggml_norm(c, io.audio, 1e-7f);                                         // zero mean, unit variance
        x = ggml_pad_reflect_1d(c, x, NFFT / 2, NFFT / 2);                                       // center=True
        ggml_tensor * frames = sliding_windows(c, x, NFFT, NF, HOP, B, x->nb[1], o);    // last frame dropped
        ggml_tensor * spec = dft_.build(c, frames, o);                                           // [402, 800, B]
        ggml_tensor * re = ggml_view_3d(c, spec, NBIN, NF, B, spec->nb[1], spec->nb[2], 0);
        ggml_tensor * im = ggml_view_3d(c, spec, NBIN, NF, B, spec->nb[1], spec->nb[2], NBIN * sizeof(float));
        ggml_tensor * pw = ggml_add(c, ggml_mul(c, re, re), ggml_mul(c, im, im));
        ggml_tensor * mel = mm(c, mel_, pw, o, true);                                            // [80, 800, B]
        ggml_tensor * ls = ggml_scale(c, ggml_log(c, ggml_clamp(c, mel, 1e-10f, FLT_MAX)), (float) (1.0 / std::log(10.0)));
        ggml_tensor * mx = ggml_pool_1d(c, ggml_reshape_2d(c, ls, NMEL * NF, B), GGML_OP_POOL_MAX, (int) (NMEL * NF), (int) (NMEL * NF), 0);
        ggml_tensor * floor = ggml_reshape_3d(c, ggml_scale_bias(c, mx, 1.0f, -8.0f), 1, 1, B);   // max - 8
        // (max(ls, floor) + 4) / 4 = (relu(ls - floor) + floor + 4) / 4
        x = ggml_add(c, ggml_relu(c, ggml_sub(c, ls, floor)), floor);
        x = ggml_scale_bias(c, x, 0.25f, 1.0f);                                                 // [80, 800, B]

        // --- encoder
        x = conv(c, x, conv1_w_, conv1_b_, 1, o);                                                // [384, 800, B]
        x = conv(c, x, conv2_w_, conv2_b_, 2, o);                                                // [384, 400, B]
        x = ggml_add(c, x, pos_);
        const int64_t N = x->ne[1];
        for (const layer & L : layers_) {
            ggml_tensor * h = ln(c, x, L.ln1_w, L.ln1_b);
            ggml_tensor * q = ggml_add(c, mm(c, L.wq, h, o), L.bq);
            ggml_tensor * k = mm(c, L.wk, h, o);
            ggml_tensor * v = ggml_add(c, mm(c, L.wv, h, o), L.bv);
            // heads: [HD, NH, N, B] -> [HD, N, NH, B]
            q = ggml_permute(c, ggml_reshape_4d(c, q, HD, NH_, N, B), 0, 2, 1, 3);
            k = ggml_permute(c, ggml_reshape_4d(c, k, HD, NH_, N, B), 0, 2, 1, 3);
            if (!o.host) { q = ggml_cont(c, q); k = ggml_cont(c, k); }
            ggml_tensor * kq = ggml_mul_mat(c, k, q);                                           // [N(k), N(q), NH, B]
            ggml_prec_set_acc(kq, GGML_PREC_F32);
            kq = ggml_soft_max_ext(c, kq, nullptr, 1.0f / std::sqrt((float) HD), 0.0f);
            ggml_tensor * vt = ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, v, HD, NH_, N, B), 1, 2, 0, 3));   // [N, HD, NH, B]
            ggml_tensor * a = ggml_mul_mat(c, vt, kq);                                          // [HD, N(q), NH, B]
            ggml_prec_set_acc(a, GGML_PREC_F32);
            a = ggml_reshape_3d(c, ggml_cont(c, ggml_permute(c, a, 0, 2, 1, 3)), D, N, B);      // [D, N, B]
            x = ggml_add(c, x, ggml_add(c, mm(c, L.wo, a, o), L.bo));
            h = ln(c, x, L.ln2_w, L.ln2_b);
            h = ggml_gelu_erf(c, ggml_add(c, mm(c, L.w1, h, o), L.b1));
            x = ggml_add(c, x, ggml_add(c, mm(c, L.w2, h, o), L.b2));
        }
        x = ln(c, x, lnf_w_, lnf_b_);                                                           // [D, N, B]

        // --- attention pooling over time
        ggml_tensor * s = ggml_tanh(c, ggml_add(c, mm(c, pa0_w_, x, o), pa0_b_));               // [256, N, B]
        s = ggml_add(c, mm(c, pa2_w_, s, o), pa2_b_);                                           // [1, N, B]
        s = ggml_soft_max(c, ggml_reshape_2d(c, s, N, B));                                      // [N, B]
        ggml_tensor * xt = ggml_cont(c, ggml_permute(c, x, 1, 0, 2, 3));                        // [N, D, B]
        ggml_tensor * pooled = ggml_mul_mat(c, xt, ggml_reshape_3d(c, s, N, 1, B));             // [D, 1, B]
        ggml_prec_set_acc(pooled, GGML_PREC_F32);
        pooled = ggml_reshape_2d(c, pooled, D, B);

        // --- classifier
        ggml_tensor * y = ggml_add(c, mm(c, c0_w_, pooled, o), c0_b_);
        y = ggml_gelu_erf(c, ln(c, y, c1_w_, c1_b_));
        y = ggml_gelu_erf(c, ggml_add(c, mm(c, c4_w_, y, o), c4_b_));
        y = ggml_sigmoid(c, ggml_add(c, mm(c, c6_w_, y, o), c6_b_));                            // [1, B]
        io.probs = ggml_reshape_2d(c, y, 1, B);
        ggml_set_name(io.probs, "probs"); ggml_set_output(io.probs);
        ggml_build_forward_expand(gf, io.probs);
        return io;
    }

private:
    struct layer {
        ggml_tensor * ln1_w, * ln1_b, * wq, * bq, * wk, * wv, * bv, * wo, * bo, * ln2_w, * ln2_b, * w1, * b1, * w2, * b2;
    };

    static ggml_tensor * ln(ggml_context * c, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
        return ggml_add(c, ggml_mul(c, ggml_norm(c, x, 1e-5f), w), b);
    }

    // x [Cin, T, B] -> conv1d k3 p1 (stride) + GELU -> [Cout, Tout, B]
    ggml_tensor * conv(ggml_context * c, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride, const graph_opts & o) {
        const int64_t C = x->ne[0], Tn = x->ne[1], B = x->ne[2];
        ggml_tensor * xp = ggml_pad_ext(c, x, 0, 0, 1, 1, 0, 0, 0, 0);
        const int64_t Tout = (Tn + 2 - 3) / stride + 1;
        ggml_tensor * col = sliding_windows(c, xp, 3 * C, Tout, stride * C, B, xp->nb[2], o);
        return ggml_gelu_erf(c, ggml_add(c, mm(c, w, col, o), b));
    }

    int64_t window_ = 0, L_ = 0, D_ = 0, NH_ = 0, NCTX_ = 0;
    frame_dft dft_;
    ggml_tensor * mel_ = nullptr, * conv1_w_ = nullptr, * conv1_b_ = nullptr, * conv2_w_ = nullptr, * conv2_b_ = nullptr;
    ggml_tensor * pos_ = nullptr, * lnf_w_ = nullptr, * lnf_b_ = nullptr;
    ggml_tensor * pa0_w_ = nullptr, * pa0_b_ = nullptr, * pa2_w_ = nullptr, * pa2_b_ = nullptr;
    ggml_tensor * c0_w_ = nullptr, * c0_b_ = nullptr, * c1_w_ = nullptr, * c1_b_ = nullptr, * c4_w_ = nullptr, * c4_b_ = nullptr;
    ggml_tensor * c6_w_ = nullptr, * c6_b_ = nullptr;
    std::vector<layer> layers_;
};

} // namespace

std::unique_ptr<arch> make_smart_turn(const model_file & file, std::string & err) {
    auto m = std::make_unique<smart_turn>();
    if (!m->configure(file, err)) return nullptr;
    return m;
}

} // namespace vad
