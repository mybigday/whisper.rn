// Silero VAD v5 / v6 (16 kHz branch).
//
// Official streaming semantics (silero_vad OnnxWrapper): every 512-sample chunk
// is prefixed with the last 64 samples of the previous chunk, reflect-padded by
// 64 on the right, then STFT (periodic Hann, n_fft 256, hop 128, 4 frames)
// -> |X| (129 bins) -> 4 x [conv1d k3 p1 + relu] (strides 1,2,2,1) -> LSTMCell(128)
// -> relu -> conv1d k1 -> sigmoid.
//
// Graph layout is channel-major ([C, T, B], channels in ne0) so rows are long
// and every conv is a single mul_mat over an overlapping strided view.
// Per-stream state (320 floats): [audio context 64 | h 128 | c 128].
#include "common.h"

#include <cstring>

namespace vad {

namespace {

constexpr int N_CHUNK = 512, N_CTX = 64, N_IN = N_CHUNK + N_CTX, N_FFT = 256, HOP = 128, N_FRAMES = 4;
constexpr int N_BINS = N_FFT / 2 + 1, H = 128, STATE = N_CTX + 2 * H;


// concatenate along dim with a balanced tree (O(n log n) copies instead of O(n^2))
ggml_tensor * concat_all(ggml_context * c, std::vector<ggml_tensor *> xs, int dim) {
    while (xs.size() > 1) {
        std::vector<ggml_tensor *> next;
        for (size_t i = 0; i + 1 < xs.size(); i += 2) next.push_back(ggml_concat(c, xs[i], xs[i + 1], dim));
        if (xs.size() % 2) next.push_back(xs.back());
        xs.swap(next);
    }
    return xs[0];
}

// mul_mat with F32 accumulation. On the CPU the activations stay F32 too. On
// accelerators, requiring F32 activations makes some backends (Vulkan) reject the
// op and push it to the CPU, so only the accumulator is pinned there and strided
// views are made contiguous (Vulkan converts non-contiguous src1 to F16).
ggml_tensor * mm(ggml_context * c, ggml_tensor * w, ggml_tensor * x, const graph_opts & o) {
    if (!o.host && !ggml_is_contiguous(x)) x = ggml_cont(c, x);
    ggml_tensor * y = ggml_mul_mat(c, w, x);
    ggml_prec_set_acc(y, GGML_PREC_F32);
    ggml_prec_set_src(y, o.host ? GGML_PREC_F32 : GGML_PREC_F16, 1);
    return y;
}

class silero : public arch {
public:

    const char * name() const override { return "silero"; }
    int sample_rate() const override { return 16000; }
    int chunk_size() const override { return N_CHUNK; }
    int state_size() const override { return STATE; }
    size_t max_nodes(int, int T) const override { return 128 + (size_t) T * 32; }

    // graph-ready tensors (docs/gguf.md), at the file's precision
    bool create_weights(weight_builder & wb) override {
        basis_ = wb.mat("stft.basis", N_FFT, 2 * N_BINS);
        const int cin[4] = {N_BINS, 128, 64, 64}, cout[4] = {128, 64, 64, 128};
        for (int l = 0; l < 4; ++l) {
            const std::string p = "enc" + std::to_string(l) + ".";
            enc_w_[l] = wb.mat(p + "w", 3 * cin[l], cout[l]);   // (Cout, 3, Cin)
            enc_b_[l] = wb.f32(p + "b", { cout[l] });
        }
        w_ih_ = wb.mat("lstm.w_ih", H, 4 * H);
        w_hh_ = wb.mat("lstm.w_hh", H, 4 * H);
        lstm_b_ = wb.f32("lstm.b", { 4 * H });                    // b_ih + b_hh
        dec_w_ = wb.mat("dec.w", H, 1);
        dec_b_ = wb.f32("dec.b", { 1 });
        return wb.ok();
    }

    graph_io build_graph(ggml_context * c, ggml_cgraph * gf, int B, int T, const graph_opts & o, ggml_tensor * state) override {
        graph_io io;
        const int64_t TB = (int64_t) T * B;
        io.audio = ggml_new_tensor_2d(c, GGML_TYPE_F32, (int64_t) N_CHUNK * T, B);
        ggml_set_name(io.audio, "audio"); ggml_set_input(io.audio);
        const size_t sb = state->nb[1];
        auto region = [&](int64_t off, int64_t n) { return ggml_view_2d(c, state, n, B, sb, off * sizeof(float)); };
        ggml_tensor * ctx0 = region(0, N_CTX);
        ggml_tensor * h    = region(N_CTX, H);
        ggml_tensor * cs   = region(N_CTX + H, H);

        // [ctx | audio] per stream, then T overlapping 576-sample windows, hop 512
        ggml_tensor * full = ggml_concat(c, o.host ? ctx0 : ggml_cont(c, ctx0), io.audio, 0);  // [64 + 512T, B]
        ggml_tensor * win = sliding_windows(c, full, N_IN, T, N_CHUNK, B, full->nb[1], o);
        ggml_tensor * x = ggml_reshape_2d(c, ggml_cont(c, win), N_IN, TB);                      // [576, T*B]
        x = ggml_pad_reflect_1d(c, x, 0, N_FFT / 4);                                            // [640, T*B]

        // STFT: 4 frames of 256, hop 128 -> |X| [129, 4, T*B]
        ggml_tensor * frames = sliding_windows(c, x, N_FFT, N_FRAMES, HOP, TB, x->nb[1], o);
        ggml_tensor * st = mm(c, basis_, frames, o);                                      // [258, 4, T*B]
        ggml_tensor * re = ggml_view_3d(c, st, N_BINS, N_FRAMES, TB, st->nb[1], st->nb[2], 0);
        ggml_tensor * im = ggml_view_3d(c, st, N_BINS, N_FRAMES, TB, st->nb[1], st->nb[2], N_BINS * sizeof(float));
        x = ggml_sqrt(c, ggml_add(c, ggml_mul(c, re, re), ggml_mul(c, im, im)));

        x = conv(c, x, 0, 1, o);
        x = conv(c, x, 1, 2, o);
        x = conv(c, x, 2, 2, o);
        x = conv(c, x, 3, 1, o);                                                                  // [128, 1, T*B]
        x = ggml_reshape_2d(c, x, H, TB);

        // LSTM: input projection for all steps at once, recurrence unrolled over T
        ggml_tensor * gi = ggml_add(c, mm(c, w_ih_, x, o), lstm_b_);                    // [512, T*B], column t + T*b
        std::vector<ggml_tensor *> hs;
        for (int t = 0; t < T; ++t) {
            ggml_tensor * g = ggml_view_2d(c, gi, 4 * H, B, gi->nb[1] * T, gi->nb[1] * t);
            g = ggml_add(c, g, mm(c, w_hh_, h, o));                                       // [512, B]
            auto gate = [&](int k) { return ggml_view_2d(c, g, H, B, g->nb[1], (size_t) k * H * sizeof(float)); };
            ggml_tensor * ig = ggml_sigmoid(c, gate(0));
            ggml_tensor * fg = ggml_sigmoid(c, gate(1));
            ggml_tensor * gg = ggml_tanh(c, gate(2));
            ggml_tensor * og = ggml_sigmoid(c, gate(3));
            cs = ggml_add(c, ggml_mul(c, fg, cs), ggml_mul(c, ig, gg));
            h  = ggml_mul(c, og, ggml_tanh(c, cs));
            hs.push_back(T > 1 ? ggml_reshape_3d(c, h, H, 1, B) : h);
        }
        ggml_tensor * hseq = T > 1 ? concat_all(c, hs, 1) : hs[0];                              // [128, T, B]

        ggml_tensor * y = mm(c, dec_w_, ggml_relu(c, hseq), o);                           // [1, T, B]
        y = ggml_sigmoid(c, ggml_add(c, y, dec_b_));
        io.probs = ggml_reshape_2d(c, y, T, B);
        ggml_set_name(io.probs, "probs"); ggml_set_output(io.probs);

        ggml_build_forward_expand(gf, io.probs);
        // new state, written in place (each write depends on the reads of its region)
        ggml_tensor * ctx1 = ggml_view_2d(c, full, N_CTX, B, full->nb[1], (size_t) N_CHUNK * T * sizeof(float));
        ggml_build_forward_expand(gf, ggml_cpy(c, ctx1, region(0, N_CTX)));
        ggml_build_forward_expand(gf, ggml_cpy(c, h, region(N_CTX, H)));
        ggml_build_forward_expand(gf, ggml_cpy(c, cs, region(N_CTX + H, H)));
        return io;
    }

private:
    // x [Cin, T, B] -> conv1d k3 p1 + relu -> [Cout, Tout, B]
    ggml_tensor * conv(ggml_context * c, ggml_tensor * x, int l, int stride, const graph_opts & o) {
        const int64_t C = x->ne[0], Tn = x->ne[1], B = x->ne[2];
        ggml_tensor * xp = ggml_pad_ext(c, x, 0, 0, 1, 1, 0, 0, 0, 0);                          // [C, T+2, B]
        const int64_t Tout = (Tn + 2 - 3) / stride + 1;
        ggml_tensor * col = sliding_windows(c, xp, 3 * C, Tout, stride * C, B, xp->nb[2], o);
        return ggml_relu(c, ggml_add(c, mm(c, enc_w_[l], col, o), enc_b_[l]));
    }

    ggml_tensor * basis_ = nullptr, * enc_w_[4] = {}, * enc_b_[4] = {};
    ggml_tensor * w_ih_ = nullptr, * w_hh_ = nullptr, * lstm_b_ = nullptr, * dec_w_ = nullptr, * dec_b_ = nullptr;
};

} // namespace

std::unique_ptr<arch> make_silero(const model_file &, std::string &) {
    return std::make_unique<silero>();
}

} // namespace vad
