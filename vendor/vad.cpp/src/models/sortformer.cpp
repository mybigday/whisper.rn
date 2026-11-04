// Streaming Sortformer speaker diarization: NVIDIA Nemotron-3-Diarization
// (OpenMDW-1.1), loaded from the official GGUF schema (general.architecture =
// "sortformer", sortformer.version = "v3") or from tools/convert_sortformer.py.
//
// Front end (transformers NemotronAsrStreamingFeatureExtractor): preemphasis 0.97,
// STFT n_fft 512, symmetric Hann 400, hop 160, centered with zero padding, power,
// mel (preprocessor.fb, 128), log(x + 2^-24); no normalization. Frames are stacked
// by 8 (80 ms encoder frames) and projected to 512 (the "embeddings").
// Encoder: input LayerNorm, 31 pre-LN transformer layers (fused QKV without bias,
// NeoX RoPE, GELU MLP), final LayerNorm. Head: Linear 512 -> 192, sub-pixel conv1d
// (k3) back to 10 ms frames, ReLU-Linear-ReLU-Linear -> 8 speaker logits, sigmoid.
//
// Streaming (host side, see sortformer_session): every step encodes the speaker
// cache + FIFO + the new chunk and its look-ahead; chunk frames then join the FIFO,
// whose overflow moves to the Arrival-Order Speaker Cache, compressed by per-speaker
// scores (top-k) when full. Ported from transformers' Nemotron3DiarizationSpeakerCache.
#include "common.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace vad {

namespace {

constexpr int SR = 16000, HOP = 160, NFFT = 512, WIN = 400, NBIN = NFFT / 2 + 1;

ggml_tensor * mm(ggml_context * c, ggml_tensor * w, ggml_tensor * x, const graph_opts & o, bool exact = false) {
    if (!o.host && !ggml_is_contiguous(x)) x = ggml_cont(c, x);
    ggml_tensor * y = ggml_mul_mat(c, w, x);
    ggml_prec_set_acc(y, GGML_PREC_F32);
    ggml_prec_set_src(y, o.host || exact ? GGML_PREC_F32 : GGML_PREC_F16, 1);
    if (exact && !o.host && o.pin_cpu) o.pin_cpu->push_back(y);
    return y;
}

ggml_tensor * ln(ggml_context * c, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    return ggml_add(c, ggml_mul(c, ggml_norm(c, x, 1e-5f), w), b);
}

// Model-card configurations (encoder frames): the offline mode of the reference
// forward, and the FIFO sizes of its streaming presets.
constexpr int OFFLINE_CHUNK = 340, OFFLINE_RC = 40, OFFLINE_FIFO = 40, OFFLINE_UPDATE = 300;
constexpr int STREAM_FIFO = 264, STREAM_UPDATE = 222, DEFAULT_CHUNK = 9;

struct scoring {
    int sil_frames = 1;
    double threshold = 0.25, boost_latest = 0.05, strong_rate = 0.75, weak_rate = 1.5, min_pos_rate = 0.5;
};

class sortformer : public arch {
public:
    bool configure(model_file & f, std::string & err) {
        auto gi = [&](const char * k, int64_t & v) {
            auto it = f.kv_int.find(k);
            if (it == f.kv_int.end()) { err = std::string("sortformer: missing metadata ") + k; return false; }
            v = it->second;
            return true;
        };
        auto gf = [&](const char * k, double & v) {
            auto it = f.kv_float.find(k);
            if (it == f.kv_float.end()) { err = std::string("sortformer: missing metadata ") + k; return false; }
            v = it->second;
            return true;
        };
        auto gs = [&](const char * k) -> std::string { auto it = f.kv_str.find(k); return it == f.kv_str.end() ? "" : it->second; };
        if (gs("version") != "v3" || gs("encoder.type") != "transformer_rope" || gs("encoder.subsampling_type") != "feature_stacking") {
            err = "sortformer: only v3 (transformer_rope encoder, feature stacking) is supported";
            return false;
        }
        int64_t sr, nfft, feat, sub, up, hr, qkvb, bias, nsil, cache;
        double ws, wstride, rot;
        if (!gi("encoder.d_model", D_) || !gi("encoder.n_layers", L_) || !gi("encoder.n_heads", NH_) || !gi("encoder.d_ff", FF_) ||
            !gi("encoder.subsampling_factor", sub) || !gi("encoder.feat_in", feat) || !gi("encoder.qkv_bias", qkvb) ||
            !gi("encoder.use_bias", bias) || !gf("encoder.rope_base", rope_base_) || !gf("encoder.rotary_fraction", rot) ||
            !gi("transformer.hidden_size", H_) || !gi("num_speakers", NS_) || !gi("upsample_factor", up) ||
            !gi("high_resolution", hr) || !gi("preprocessor.sample_rate", sr) || !gi("preprocessor.n_fft", nfft) ||
            !gi("preprocessor.features", feat_) || !gf("preprocessor.window_size", ws) || !gf("preprocessor.window_stride", wstride) ||
            !gf("preprocessor.preemph", preemph_) || !gf("preprocessor.log_zero_guard", guard_) ||
            !gi("scoring.spkcache_sil_frames_per_spk", nsil) || !gf("scoring.pred_score_threshold", sc_.threshold) ||
            !gf("scoring.scores_boost_latest", sc_.boost_latest) || !gf("scoring.strong_boost_rate", sc_.strong_rate) ||
            !gf("scoring.weak_boost_rate", sc_.weak_rate) || !gf("scoring.min_pos_scores_rate", sc_.min_pos_rate) ||
            !gi("streaming.spkcache_len", cache)) return false;
        sc_.sil_frames = (int) nsil;
        cache_len_ = (int) cache;
        SUB_ = (int) sub;
        if (sr != SR || nfft != NFFT || std::lround(ws * SR) != WIN || std::lround(wstride * SR) != HOP || feat != feat_ ||
            up != sub || !hr || qkvb || !bias || rot != 1.0 || D_ % NH_ != 0 || (D_ / NH_) % 2 != 0 ||
            gs("preprocessor.normalize") != "NA" || cache_len_ < (1 + sc_.sil_frames) * NS_) {
            err = "sortformer: unsupported configuration";
            return false;
        }
        // host copy of the learned silence embedding (fills reserved speaker-cache slots)
        auto it = f.tensors.find("learnable_sil_emb");
        if (it == f.tensors.end() || it->second.nelements() != D_ || !it->second.is_float()) {
            err = "sortformer: missing or invalid tensor learnable_sil_emb";
            return false;
        }
        sil_ = it->second.to_f32();
        // the sub-pixel conv weight (Cout, Cin, K) is used as a mul_mat over [K * Cin] windows
        auto cw = f.tensors.find("subpixel_upsample.weight");
        if (cw == f.tensors.end()) { err = "sortformer: missing tensor subpixel_upsample.weight"; return false; }
        host_tensor w;
        if (!conv_to_window_layout(cw->second, H_ * SUB_, H_, 3, w)) {
            err = "sortformer: subpixel_upsample.weight must be F32/F16/BF16 (Cout, Cin, 3)";
            return false;
        }
        cw->second = std::move(w);
        return true;
    }

    const char * name() const override { return "sortformer"; }
    int sample_rate() const override { return SR; }
    int chunk_size() const override { return HOP; }          // one output frame per 10 ms
    int state_size() const override { return 0; }
    int lookahead_frames() const override { return OFFLINE_RC * SUB_; }
    int n_outputs() const override { return (int) NS_; }
    bool is_speakers() const override { return true; }
    bool host_driven() const override { return true; }
    size_t max_nodes(int, int) const override { return 0; }
    graph_io build_graph(ggml_context *, ggml_cgraph *, int, int, const graph_opts &, ggml_tensor *) override {
        GGML_ABORT("sortformer: host-driven model");
    }

    bool create_weights(weight_builder & wb) override {
        // STFT basis: symmetric Hann(400) centered in 512 samples x DFT, rows [Re(0..256), Im(0..256)]
        std::vector<float> basis((size_t) NFFT * 2 * NBIN);
        const int off = (NFFT - WIN) / 2;
        for (int k = 0; k < NBIN; ++k)
            for (int n = 0; n < NFFT; ++n) {
                const int m = n - off;
                const double w = m >= 0 && m < WIN ? 0.5 - 0.5 * std::cos(2 * M_PI * m / (WIN - 1)) : 0.0;
                const double ang = 2 * M_PI * (double) ((int64_t) k * n % NFFT) / NFFT;
                basis[(size_t) k * NFFT + n] = (float) (w * std::cos(ang));
                basis[(size_t) (NBIN + k) * NFFT + n] = (float) (-w * std::sin(ang));
            }
        basis_ = wb.constant("fe.basis", { NFFT, 2 * NBIN }, basis);
        fb_ = wb.mat("preprocessor.fb", NBIN, feat_);
        proj_in_ = wb.mat("encoder.pre_encode.proj.weight", feat_ * SUB_, D_);
        in_w_ = wb.f32("encoder.embed_norm.weight", { D_ });
        in_b_ = wb.f32("encoder.embed_norm.bias", { D_ });
        for (int l = 0; l < L_; ++l) {
            const std::string p = "encoder.layers." + std::to_string(l) + ".";
            layer L;
            L.n1w = wb.f32(p + "norm1.weight", { D_ });
            L.n1b = wb.f32(p + "norm1.bias", { D_ });
            L.qkv = wb.mat(p + "attn.w_qkv.weight", D_, 3 * D_);
            L.ow = wb.mat(p + "attn.out_proj.weight", D_, D_);
            L.ob = wb.f32(p + "attn.out_proj.bias", { D_ });
            L.n2w = wb.f32(p + "norm2.weight", { D_ });
            L.n2b = wb.f32(p + "norm2.bias", { D_ });
            L.w1 = wb.mat(p + "ffn.net.0.weight", D_, FF_);
            L.b1 = wb.f32(p + "ffn.net.0.bias", { FF_ });
            L.w2 = wb.mat(p + "ffn.net.3.weight", FF_, D_);
            L.b2 = wb.f32(p + "ffn.net.3.bias", { D_ });
            layers_.push_back(L);
        }
        out_w_ = wb.f32("encoder.final_norm.weight", { D_ });
        out_b_ = wb.f32("encoder.final_norm.bias", { D_ });
        proj_w_ = wb.mat("encoder_proj.weight", D_, H_);
        proj_b_ = wb.f32("encoder_proj.bias", { H_ });
        up_w_ = wb.mat("subpixel_upsample.weight", 3 * H_, H_ * SUB_);
        up_b_ = wb.f32("subpixel_upsample.bias", { H_ * SUB_ });
        h1_w_ = wb.mat("head.first_hidden_to_hidden.weight", H_, H_);
        h1_b_ = wb.f32("head.first_hidden_to_hidden.bias", { H_ });
        h2_w_ = wb.mat("head.single_hidden_to_spks.weight", H_, NS_);
        h2_b_ = wb.f32("head.single_hidden_to_spks.bias", { NS_ });
        return wb.ok();
    }

    // Embeddings of Fe encoder frames: audio = preemphasized samples starting
    // NFFT/2 before the first mel frame, (SUB*Fe - 1)*HOP + NFFT long;
    // mask = 1 for mel frames inside the audio. Output [D, Fe].
    custom_graph embed_graph(ggml_context * c, ggml_cgraph * gf, const graph_opts & o, int Fe, int B = 1) const {
        custom_graph g;
        const int64_t NF = (int64_t) SUB_ * Fe;
        ggml_tensor * audio = ggml_new_tensor_2d(c, GGML_TYPE_F32, (NF - 1) * HOP + NFFT, B);
        ggml_tensor * mask = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, NF, B);
        ggml_set_input(audio); ggml_set_input(mask);
        ggml_tensor * frames = sliding_windows(c, audio, NFFT, NF, HOP, B, audio->nb[1], o);   // [512, NF, B]
        ggml_tensor * spec = mm(c, basis_, frames, o, true);
        ggml_tensor * re = ggml_view_3d(c, spec, NBIN, NF, B, spec->nb[1], spec->nb[2], 0);
        ggml_tensor * im = ggml_view_3d(c, spec, NBIN, NF, B, spec->nb[1], spec->nb[2], NBIN * sizeof(float));
        ggml_tensor * pw = ggml_add(c, ggml_mul(c, re, re), ggml_mul(c, im, im));
        ggml_tensor * mel = mm(c, fb_, pw, o, true);                                            // [128, NF, B]
        ggml_tensor * lm = ggml_mul(c, ggml_log(c, ggml_scale_bias(c, mel, 1.0f, (float) guard_)), mask);
        ggml_tensor * stacked = ggml_reshape_3d(c, lm, feat_ * SUB_, Fe, B);                   // 8 frames per row
        ggml_tensor * emb = mm(c, proj_in_, stacked, o, true);                                  // [D, Fe, B]
        ggml_set_output(emb);
        ggml_build_forward_expand(gf, emb);
        g.inputs = { audio, mask };
        g.outputs = { emb };
        return g;
    }

    // Encoder + head over B sequences of S (<= Sb) embeddings each: emb [D, Sb, B];
    // kq_mask [Sb, Sb, 1, B] F16 (0 / -inf per key); seq_mask [1, Sb, B] (1 inside the
    // sequence); pos [Sb]. Output: speaker probabilities [NS, SUB * Sb, B].
    custom_graph encoder_graph(ggml_context * c, ggml_cgraph * gf, const graph_opts & o, int Sb, int B = 1) const {
        custom_graph g;
        const int64_t D = D_, HD = D_ / NH_, NH = NH_;
        ggml_tensor * emb = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, Sb, B);
        ggml_tensor * kq_mask = ggml_new_tensor_4d(c, GGML_TYPE_F16, Sb, Sb, 1, B);
        ggml_tensor * seq_mask = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, Sb, B);
        ggml_tensor * pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, Sb);
        for (ggml_tensor * t : { emb, kq_mask, seq_mask, pos }) ggml_set_input(t);

        ggml_tensor * x = ln(c, emb, in_w_, in_b_);
        for (const layer & L : layers_) {
            ggml_tensor * h = ln(c, x, L.n1w, L.n1b);
            ggml_tensor * qkv = mm(c, L.qkv, h, o);                                             // [3D, Sb, B]
            auto head = [&](int i) {
                return ggml_view_4d(c, qkv, HD, NH, Sb, B, HD * sizeof(float), qkv->nb[1], qkv->nb[2], i * D * sizeof(float));
            };
            ggml_tensor * q = head(0), * k = head(1), * v = head(2);
            if (!o.host) { q = ggml_cont(c, q); k = ggml_cont(c, k); }
            q = ggml_rope_ext(c, q, pos, nullptr, (int) HD, GGML_ROPE_TYPE_NEOX, 0, (float) rope_base_, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k = ggml_rope_ext(c, k, pos, nullptr, (int) HD, GGML_ROPE_TYPE_NEOX, 0, (float) rope_base_, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            q = ggml_permute(c, q, 0, 2, 1, 3);                                                 // [HD, Sb, NH, B]
            k = ggml_permute(c, k, 0, 2, 1, 3);
            if (!o.host) { q = ggml_cont(c, q); k = ggml_cont(c, k); }
            ggml_tensor * kq = ggml_mul_mat(c, k, q);                                           // [Sb(k), Sb(q), NH, B]
            ggml_prec_set_acc(kq, GGML_PREC_F32);
            kq = ggml_soft_max_ext(c, kq, kq_mask, 1.0f / std::sqrt((float) HD), 0.0f);
            ggml_tensor * vt = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));                    // [Sb, HD, NH, B]
            ggml_tensor * a = ggml_mul_mat(c, vt, kq);                                          // [HD, Sb(q), NH, B]
            ggml_prec_set_acc(a, GGML_PREC_F32);
            a = ggml_reshape_3d(c, ggml_cont(c, ggml_permute(c, a, 0, 2, 1, 3)), D, Sb, B);    // [D, Sb, B]
            x = ggml_add(c, x, ggml_add(c, mm(c, L.ow, a, o), L.ob));
            h = ln(c, x, L.n2w, L.n2b);
            h = ggml_gelu_erf(c, ggml_add(c, mm(c, L.w1, h, o), L.b1));
            x = ggml_add(c, x, ggml_add(c, mm(c, L.w2, h, o), L.b2));
        }
        x = ln(c, x, out_w_, out_b_);
        // frames past the sequence (bucket padding) must read as the conv's zero padding
        ggml_tensor * y = ggml_mul(c, ggml_add(c, mm(c, proj_w_, x, o), proj_b_), seq_mask);   // [H, Sb, B]
        ggml_tensor * yp = ggml_pad_ext(c, y, 0, 0, 1, 1, 0, 0, 0, 0);                         // [H, Sb + 2, B]
        ggml_tensor * col = sliding_windows(c, yp, 3 * H_, Sb, H_, B, yp->nb[2], o);           // [3H, Sb, B]
        ggml_tensor * u = ggml_add(c, mm(c, up_w_, col, o), up_b_);                             // [H * SUB, Sb, B]
        u = ggml_reshape_3d(c, u, H_, (int64_t) SUB_ * Sb, B);                                 // sub-pixel shuffle
        u = ggml_add(c, mm(c, h1_w_, ggml_relu(c, u), o), h1_b_);
        u = ggml_add(c, mm(c, h2_w_, ggml_relu(c, u), o), h2_b_);
        ggml_tensor * p = ggml_sigmoid(c, u);                                                   // [NS, SUB * Sb, B]
        ggml_set_output(p);
        ggml_build_forward_expand(gf, p);
        g.inputs = { emb, kq_mask, seq_mask, pos };
        g.outputs = { p };
        return g;
    }

    size_t encoder_nodes() const { return 64 + (size_t) L_ * 48; }

    std::unique_ptr<session> new_session(int slice_samples, int lookahead_samples, bool offline, std::string & err) const override;
    bool streams_push(graph_runner & g, session * const * s, int n, const float * const * x, const int64_t * len,
                      float * const * out) const override;

    int SUB_ = 8;
    int64_t D_ = 0, L_ = 0, NH_ = 0, FF_ = 0, H_ = 0, NS_ = 0, feat_ = 0;
    double rope_base_ = 10000.0, preemph_ = 0.97, guard_ = 0.0;
    int cache_len_ = 0;
    scoring sc_;
    std::vector<float> sil_;

private:
    struct layer {
        ggml_tensor * n1w, * n1b, * qkv, * ow, * ob, * n2w, * n2b, * w1, * b1, * w2, * b2;
    };
    ggml_tensor * basis_ = nullptr, * fb_ = nullptr, * proj_in_ = nullptr, * in_w_ = nullptr, * in_b_ = nullptr;
    ggml_tensor * out_w_ = nullptr, * out_b_ = nullptr, * proj_w_ = nullptr, * proj_b_ = nullptr;
    ggml_tensor * up_w_ = nullptr, * up_b_ = nullptr, * h1_w_ = nullptr, * h1_b_ = nullptr, * h2_w_ = nullptr, * h2_b_ = nullptr;
    std::vector<layer> layers_;
};

// ------------------------------------------------------------------ session

class sortformer_session : public session {
public:
    sortformer_session(const sortformer & m, int chunk, int rc, int fifo, int update, bool offline)
        : m_(m), chunk_(chunk), rc_(rc), fifo_len_(fifo), update_(update), offline_(offline) {
        E_ = m_.SUB_ * HOP;
        D_ = (int) m_.D_;
        NS_ = (int) m_.NS_;
        reset();
    }

    void reset() override {
        pre_.clear(); pre0_ = 0; total_ = 0; last_ = 0.0f;
        emb_.clear(); emb0_ = 0;
        emitted_ = 0;
        cache_emb_.clear(); cache_prob_.clear(); n_cache_ = 0; compressed_ = false;
        fifo_emb_.clear(); n_fifo_ = 0;
    }

    // a non-final step over encoder frames [c, c + chunk + rc) needs the samples the
    // reference processor feeds for it: up to (c + chunk + rc) * E + WIN - NFFT / 2
    int64_t steps_ready(int64_t total) const {
        if (offline_) return 0;
        const int64_t need0 = (int64_t) (emitted_ + chunk_ + rc_) * E_ + WIN - NFFT / 2;
        return total < need0 ? 0 : 1 + (total - need0) / ((int64_t) chunk_ * E_);
    }

    int64_t n_pending(int64_t n) const override { return steps_ready(total_ + n) * chunk_ * m_.SUB_; }
    int64_t n_flush() const override { return std::max<int64_t>(0, total_ / HOP - (int64_t) emitted_ * m_.SUB_); }
    int64_t n_buffered() const override { return total_ - std::min<int64_t>(total_, (int64_t) emitted_ * E_); }

    bool push(graph_runner & g, const float * x, int64_t n, float * out) override {
        const int64_t steps = append(x, n);
        const int64_t valid = total_ / HOP;   // never reached by non-final steps
        for (int64_t s = 0; s < steps; ++s) {
            if (!step(g, chunk_, rc_, valid, out)) return false;
            out += (size_t) chunk_ * m_.SUB_ * NS_;
        }
        return true;
    }

    // adds samples; returns the streaming steps they complete
    int64_t append(const float * x, int64_t n) {
        const int64_t steps = steps_ready(total_ + n);
        pre_.reserve(pre_.size() + n);
        for (int64_t i = 0; i < n; ++i) {
            const float v = total_ + i == 0 ? x[i] : x[i] - (float) m_.preemph_ * (i ? x[i - 1] : last_);
            pre_.push_back(v);
        }
        if (n > 0) last_ = x[n - 1];
        total_ += n;
        return steps;
    }

    bool flush(graph_runner & g, float * out) override {
        const int64_t V = total_ / HOP;                        // mel frames inside the audio
        int64_t left = V - (int64_t) emitted_ * m_.SUB_;
        if (left <= 0) return true;
        std::vector<float> buf;
        if (!offline_) {
            // the reference's last streaming chunk: every remaining frame, no look-ahead
            const int R = (int) ((left + m_.SUB_ - 1) / m_.SUB_);
            buf.resize((size_t) R * m_.SUB_ * NS_);
            if (!step(g, R, 0, V, buf.data())) return false;
        } else {
            // the reference offline forward: one more (masked) mel frame, chunks of
            // OFFLINE_CHUNK with up to OFFLINE_RC look-ahead frames
            const int64_t E = (V + 1 + m_.SUB_ - 1) / m_.SUB_;
            buf.resize((size_t) (E - emitted_) * m_.SUB_ * NS_);
            float * o = buf.data();
            while (emitted_ < E) {
                const int ch = (int) std::min<int64_t>(chunk_, E - emitted_);
                const int rc = (int) std::min<int64_t>(rc_, E - emitted_ - ch);
                if (!step(g, ch, rc, V, o)) return false;
                o += (size_t) ch * m_.SUB_ * NS_;
            }
        }
        std::copy(buf.begin(), buf.begin() + (size_t) left * NS_, out);
        emitted_ = (int) ((V + m_.SUB_ - 1) / m_.SUB_);
        return true;
    }

private:
    // embeddings [emb0_, upto) are kept in emb_; computes the missing ones
    bool ensure_embeds(graph_runner & g, int64_t upto, int64_t valid) {
        embed_job j;
        while (next_embed(upto, valid, j)) {
            embed_job * p = &j;
            if (!run_embeds(g, m_, &p, 1)) return false;
        }
        return true;
    }

public:
    // the input of the next embedding run towards `upto` embeddings (false: none missing)
    struct embed_job {
        sortformer_session * s = nullptr;
        int want = 0, Fe = 0;
        std::vector<float> audio, mask;
    };

    bool next_embed(int64_t upto, int64_t valid, embed_job & j) {
        const int64_t e0 = emb0_ + (int64_t) emb_.size() / D_;
        if (e0 >= upto) return false;
        j.s = this;
        j.want = (int) std::min<int64_t>(512, upto - e0);
        j.Fe = 16;
        while (j.Fe < j.want) j.Fe *= 2;
        const int64_t NF = (int64_t) m_.SUB_ * j.Fe;
        j.audio.assign((size_t) ((NF - 1) * HOP + NFFT), 0.0f);
        j.mask.assign((size_t) NF, 0.0f);
        const int64_t a0 = e0 * E_ - NFFT / 2;
        for (size_t i = 0; i < j.audio.size(); ++i) {
            const int64_t s = a0 + (int64_t) i;
            if (s >= 0 && s < total_) j.audio[i] = pre_[(size_t) (s - pre0_)];
        }
        for (int64_t k = 0; k < NF; ++k) j.mask[(size_t) k] = e0 * m_.SUB_ + k < valid ? 1.0f : 0.0f;
        return true;
    }

    // embedding runs of several sessions, batched by Fe (B = power of two >= their count)
    static bool run_embeds(graph_runner & g, const sortformer & m, embed_job * const * jobs, int n) {
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return jobs[a]->Fe < jobs[b]->Fe; });
        for (int g0 = 0; g0 < n;) {
            int g1 = g0;
            const int Fe = jobs[order[g0]]->Fe;
            while (g1 < n && jobs[order[g1]]->Fe == Fe) ++g1;
            const int cnt = g1 - g0;
            int B = 1;
            while (B < cnt) B *= 2;
            const size_t na = jobs[order[g0]]->audio.size(), nm = jobs[order[g0]]->mask.size();
            std::vector<float> audio(na * B, 0.0f), mask(nm * B, 0.0f), e((size_t) m.D_ * Fe * B);
            for (int b = 0; b < cnt; ++b) {
                const embed_job & j = *jobs[order[g0 + b]];
                std::copy(j.audio.begin(), j.audio.end(), audio.begin() + (size_t) b * na);
                std::copy(j.mask.begin(), j.mask.end(), mask.begin() + (size_t) b * nm);
            }
            if (!g.run("sortformer.embed." + std::to_string(Fe) + "." + std::to_string(B), 256,
                       [&m, Fe, B](ggml_context * c, ggml_cgraph * gf, const graph_opts & o) { return m.embed_graph(c, gf, o, Fe, B); },
                       { audio.data(), mask.data() }, { e.data() })) return false;
            for (int b = 0; b < cnt; ++b) {
                embed_job & j = *jobs[order[g0 + b]];
                const float * eb = e.data() + (size_t) b * m.D_ * Fe;
                j.s->emb_.insert(j.s->emb_.end(), eb, eb + (size_t) m.D_ * j.want);
            }
            g0 = g1;
        }
        return true;
    }

    // embeddings the next streaming step needs
    int64_t step_upto() const { return emitted_ + chunk_ + rc_; }

private:
    // One step: chunk frames [emitted_, emitted_ + ch) with rc look-ahead frames.
    // Writes ch * SUB frames of NS probabilities; updates FIFO and speaker cache.
    bool step(graph_runner & g, int ch, int rc, int64_t valid, float * out) {
        step_job j;
        if (!prepare(g, ch, rc, valid, out, j)) return false;
        step_job * p = &j;
        return run_jobs(g, m_, &p, 1);
    }

public:
    // the encoder input of one step (cache + FIFO + chunk + look-ahead, padded to Sb)
    struct step_job {
        sortformer_session * s = nullptr;
        int ch = 0, nc = 0, nf = 0, S = 0, Sb = 0;
        std::vector<float> seq, seq_mask, key;
        std::vector<ggml_fp16_t> kq;
        float * out = nullptr;
    };

    // streaming steps of several sessions as batched computes: jobs of the same Sb share one
    // graph of B = power of two >= their count (padding sequences fully masked but one key)
    static bool run_jobs(graph_runner & g, const sortformer & m, step_job * const * jobs, int n) {
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return jobs[a]->Sb < jobs[b]->Sb; });
        const int64_t D = m.D_, NSO = m.NS_ * m.SUB_;
        for (int g0 = 0; g0 < n;) {
            int g1 = g0;
            const int Sb = jobs[order[g0]]->Sb;
            while (g1 < n && jobs[order[g1]]->Sb == Sb) ++g1;
            const int cnt = g1 - g0;
            int B = 1;
            while (B < cnt) B *= 2;
            std::vector<float> seq((size_t) D * Sb * B, 0.0f), seq_mask((size_t) Sb * B, 0.0f);
            std::vector<ggml_fp16_t> kq((size_t) Sb * Sb * B, ggml_fp32_to_fp16(-INFINITY));
            for (int b = 0; b < B; ++b) {
                if (b < cnt) {
                    const step_job & j = *jobs[order[g0 + b]];
                    std::copy(j.seq.begin(), j.seq.end(), seq.begin() + (size_t) b * D * Sb);
                    std::copy(j.seq_mask.begin(), j.seq_mask.end(), seq_mask.begin() + (size_t) b * Sb);
                    std::copy(j.kq.begin(), j.kq.end(), kq.begin() + (size_t) b * Sb * Sb);
                } else {
                    for (int q = 0; q < Sb; ++q) kq[(size_t) b * Sb * Sb + (size_t) q * Sb] = ggml_fp32_to_fp16(0.0f);
                }
            }
            std::vector<int32_t> pos((size_t) Sb);
            std::iota(pos.begin(), pos.end(), 0);
            std::vector<float> probs((size_t) NSO * Sb * B);
            if (!g.run("sortformer.encoder." + std::to_string(Sb) + "." + std::to_string(B), m.encoder_nodes(),
                       [&m, Sb, B](ggml_context * c, ggml_cgraph * gf, const graph_opts & o) { return m.encoder_graph(c, gf, o, Sb, B); },
                       { seq.data(), kq.data(), seq_mask.data(), pos.data() }, { probs.data() })) return false;
            for (int b = 0; b < cnt; ++b) {
                step_job & j = *jobs[order[g0 + b]];
                j.s->finish(j, probs.data() + (size_t) b * NSO * Sb);
            }
            g0 = g1;
        }
        return true;
    }

    // embeddings and encoder input of the next step (chunk ch, look-ahead rc); its output goes to out
    bool prepare(graph_runner & g, int ch, int rc, int64_t valid, float * out, step_job & j) {
        const int64_t c0 = emitted_;
        if (!ensure_embeds(g, c0 + ch + rc, valid)) return false;
        const int nc = n_cache_, nf = n_fifo_, S = nc + nf + ch + rc;
        int Sb = 64;
        while (Sb < S) Sb += 64;
        j.s = this; j.ch = ch; j.nc = nc; j.nf = nf; j.S = S; j.Sb = Sb; j.out = out;
        j.seq.assign((size_t) D_ * Sb, 0.0f);
        j.seq_mask.assign((size_t) Sb, 0.0f);
        j.key.assign((size_t) S, 1.0f);
        std::copy(cache_emb_.begin(), cache_emb_.begin() + (size_t) nc * D_, j.seq.begin());
        std::copy(fifo_emb_.begin(), fifo_emb_.begin() + (size_t) nf * D_, j.seq.begin() + (size_t) nc * D_);
        std::copy(emb_.begin() + (size_t) (c0 - emb0_) * D_, emb_.begin() + (size_t) (c0 - emb0_ + ch + rc) * D_,
                  j.seq.begin() + (size_t) (nc + nf) * D_);
        for (int i = 0; i < S; ++i) j.seq_mask[i] = 1.0f;
        // an encoder frame whose first mel frame lies past the audio is masked (offline tail)
        for (int i = 0; i < ch + rc; ++i) j.key[nc + nf + i] = (c0 + i) * m_.SUB_ < valid ? 1.0f : 0.0f;
        j.kq.resize((size_t) Sb * Sb);
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
        for (int k = 0; k < Sb; ++k) {
            const ggml_fp16_t v = k < S && j.key[k] > 0 ? zero : ninf;
            for (int q = 0; q < Sb; ++q) j.kq[(size_t) q * Sb + k] = v;
        }
        return true;
    }

    // probs [NS, SUB * Sb] of the step: writes the chunk's outputs, updates FIFO and speaker cache
    void finish(step_job & j, const float * probs) {
        const int nc = j.nc, nf = j.nf, S = j.S, ch = j.ch;
        const size_t o0 = (size_t) (nc + nf) * m_.SUB_ * NS_;
        std::copy(probs + o0, probs + o0 + (size_t) ch * m_.SUB_ * NS_, j.out);

        // probabilities per encoder frame: mean of its SUB frames, zero when masked
        std::vector<float> pooled((size_t) S * NS_);
        for (int f = 0; f < S; ++f)
            for (int s = 0; s < NS_; ++s) {
                double acc = 0;
                for (int k = 0; k < m_.SUB_; ++k) acc += probs[((size_t) f * m_.SUB_ + k) * NS_ + s];
                pooled[(size_t) f * NS_ + s] = (float) (acc / m_.SUB_) * j.key[f];
            }
        update(j.seq, pooled, nc, nf, ch);
        emitted_ += ch;
        // drop what later steps no longer need
        const int64_t drop = std::min<int64_t>(emitted_ - emb0_, (int64_t) emb_.size() / D_);
        emb_.erase(emb_.begin(), emb_.begin() + (size_t) drop * D_);
        emb0_ += drop;
        const int64_t keep = std::max<int64_t>(0, (emb0_ + (int64_t) emb_.size() / D_) * E_ - NFFT / 2);
        if (keep > pre0_) {
            const int64_t k = std::min<int64_t>(keep - pre0_, (int64_t) pre_.size());
            pre_.erase(pre_.begin(), pre_.begin() + k);
            pre0_ += k;
        }
    }

    int chunk() const { return chunk_; }
    int rc() const { return rc_; }
    int64_t valid() const { return total_ / HOP; }

private:
    // Nemotron3DiarizationSpeakerCache.update with no look-back frames
    void update(const std::vector<float> & seq, const std::vector<float> & probs, int nc, int nf, int ch) {
        const int chunk_start = nc + nf;
        std::vector<float> fifo((size_t) (nf + ch) * D_), fprob((size_t) (nf + ch) * NS_);
        std::copy(fifo_emb_.begin(), fifo_emb_.begin() + (size_t) nf * D_, fifo.begin());
        std::copy(seq.begin() + (size_t) chunk_start * D_, seq.begin() + (size_t) (chunk_start + ch) * D_, fifo.begin() + (size_t) nf * D_);
        std::copy(probs.begin() + (size_t) nc * NS_, probs.begin() + (size_t) (nc + nf) * NS_, fprob.begin());
        std::copy(probs.begin() + (size_t) chunk_start * NS_, probs.begin() + (size_t) (chunk_start + ch) * NS_, fprob.begin() + (size_t) nf * NS_);
        const int n = nf + ch;
        int popped = 0;
        if (n > fifo_len_) popped = std::min(std::max(update_, n - fifo_len_), n);
        if (popped) {
            std::vector<float> ce((size_t) (nc + popped) * D_), cp((size_t) (nc + popped) * NS_);
            std::copy(cache_emb_.begin(), cache_emb_.begin() + (size_t) nc * D_, ce.begin());
            std::copy(fifo.begin(), fifo.begin() + (size_t) popped * D_, ce.begin() + (size_t) nc * D_);
            // an uncompressed cache holds plain frames, re-estimated by this step
            const std::vector<float> & stored = compressed_ ? cache_prob_ : probs;
            std::copy(stored.begin(), stored.begin() + (size_t) nc * NS_, cp.begin());
            std::copy(fprob.begin(), fprob.begin() + (size_t) popped * NS_, cp.begin() + (size_t) nc * NS_);
            if (nc + popped > m_.cache_len_) {
                compress(ce, cp);
                compressed_ = true;
            }
            cache_emb_ = std::move(ce);
            cache_prob_ = std::move(cp);
            n_cache_ = (int) (cache_emb_.size() / D_);
        }
        fifo_emb_.assign(fifo.begin() + (size_t) popped * D_, fifo.end());
        n_fifo_ = n - popped;
    }

    // top-k indices of v (k largest, ties broken by lower index)
    static std::vector<int> topk(const std::vector<double> & v, int k) {
        std::vector<int> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);
        k = std::min<int>(k, (int) idx.size());
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return v[a] > v[b] || (v[a] == v[b] && a < b); });
        idx.resize(k);
        return idx;
    }

    // keeps cache_len frames: the most important per speaker by score, grouped by
    // speaker in their original order, with sil_frames silence slots per speaker
    void compress(std::vector<float> & emb, std::vector<float> & prob) const {
        const scoring & sc = m_.sc_;
        const int N = (int) (prob.size() / NS_), K = m_.cache_len_, NSIL = sc.sil_frames;
        const double NEG = -std::numeric_limits<double>::infinity(), POS = std::numeric_limits<double>::infinity();
        const int budget = K / NS_ - NSIL;
        const int min_pos = (int) std::floor(budget * sc.min_pos_rate);
        const int n_strong = (int) std::floor(budget * sc.strong_rate), n_weak = (int) std::floor(budget * sc.weak_rate);
        // scores [N][NS] (float math as the reference)
        std::vector<double> score((size_t) N * NS_);
        std::vector<int> n_pos(NS_, 0);
        for (int f = 0; f < N; ++f) {
            float sum_lc = 0.0f;
            for (int s = 0; s < NS_; ++s) sum_lc += std::log(std::max(1.0f - prob[(size_t) f * NS_ + s], (float) sc.threshold));
            for (int s = 0; s < NS_; ++s) {
                const float p = prob[(size_t) f * NS_ + s];
                const float lp = std::log(std::max(p, (float) sc.threshold)), lc = std::log(std::max(1.0f - p, (float) sc.threshold));
                const float v = lp - lc + sum_lc - std::log(0.5f);
                score[(size_t) f * NS_ + s] = p > 0.5f ? v : NEG;
                if (p > 0.5f && v > 0) ++n_pos[s];
            }
        }
        for (int f = 0; f < N; ++f)
            for (int s = 0; s < NS_; ++s) {
                double & v = score[(size_t) f * NS_ + s];
                const float p = prob[(size_t) f * NS_ + s];
                if (!(v > 0) && p > 0.5f && n_pos[s] >= min_pos) v = NEG;
                if (f >= K) v = (float) (v + sc.boost_latest);
            }
        auto boost = [&](int k, float b) {
            for (int s = 0; s < NS_; ++s) {
                std::vector<double> col(N);
                for (int f = 0; f < N; ++f) col[f] = score[(size_t) f * NS_ + s];
                for (int f : topk(col, k)) score[(size_t) f * NS_ + s] = (float) (score[(size_t) f * NS_ + s] + b);
            }
        };
        boost(n_strong, -2.0f * std::log(0.5f));
        boost(n_weak, -std::log(0.5f));
        // flattened [speaker][frame + silence slots]
        const int NSC = N + NSIL;
        std::vector<double> flat((size_t) NS_ * NSC);
        for (int s = 0; s < NS_; ++s)
            for (int f = 0; f < NSC; ++f) flat[(size_t) s * NSC + f] = f < N ? score[(size_t) f * NS_ + s] : POS;
        std::vector<int> pick = topk(flat, K);
        const int sentinel = NS_ * NSC;
        for (int & i : pick) if (flat[i] == NEG) i = sentinel;
        std::sort(pick.begin(), pick.end());
        std::vector<float> ne((size_t) K * D_), np((size_t) K * NS_, 0.0f);
        for (int r = 0; r < K; ++r) {
            const int f = pick[r] == sentinel ? N : std::min(pick[r] % NSC, N);
            if (f == N) std::copy(m_.sil_.begin(), m_.sil_.end(), ne.begin() + (size_t) r * D_);
            else {
                std::copy(emb.begin() + (size_t) f * D_, emb.begin() + (size_t) (f + 1) * D_, ne.begin() + (size_t) r * D_);
                std::copy(prob.begin() + (size_t) f * NS_, prob.begin() + (size_t) (f + 1) * NS_, np.begin() + (size_t) r * NS_);
            }
        }
        emb = std::move(ne);
        prob = std::move(np);
    }

    const sortformer & m_;
    const int chunk_, rc_, fifo_len_, update_;
    const bool offline_;
    int E_ = 0, D_ = 0, NS_ = 0;
    std::vector<float> pre_;             // preemphasized audio, pre_[0] is sample pre0_
    int64_t pre0_ = 0, total_ = 0;
    float last_ = 0.0f;
    std::vector<float> emb_;             // embeddings [emb0_, ...) not yet in the FIFO
    int64_t emb0_ = 0;
    int emitted_ = 0;                    // encoder frames emitted
    std::vector<float> cache_emb_, cache_prob_, fifo_emb_;
    int n_cache_ = 0, n_fifo_ = 0;
    bool compressed_ = false;
};

// Pushes to several streams: every round, the next streaming step of each stream that has one,
// batched across streams by sequence length (each stream keeps its own masks).
bool sortformer::streams_push(graph_runner & g, session * const * ss, int n, const float * const * x, const int64_t * len,
                              float * const * out) const {
    std::vector<sortformer_session *> s(n);
    std::vector<int64_t> steps(n);
    std::vector<float *> o(out, out + n);
    for (int i = 0; i < n; ++i) {
        s[i] = static_cast<sortformer_session *>(ss[i]);
        steps[i] = s[i]->append(x[i], len[i]);
    }
    std::vector<sortformer_session::step_job> jobs(n);
    std::vector<sortformer_session::embed_job> ejobs(n);
    for (;;) {
        // the embeddings of this round's steps, batched across streams
        for (;;) {
            std::vector<sortformer_session::embed_job *> er;
            for (int i = 0; i < n; ++i)
                if (steps[i] > 0 && s[i]->next_embed(s[i]->step_upto(), s[i]->valid(), ejobs[i])) er.push_back(&ejobs[i]);
            if (er.empty()) break;
            if (!sortformer_session::run_embeds(g, *this, er.data(), (int) er.size())) return false;
        }
        std::vector<sortformer_session::step_job *> round;
        for (int i = 0; i < n; ++i) {
            if (steps[i] == 0) continue;
            if (!s[i]->prepare(g, s[i]->chunk(), s[i]->rc(), s[i]->valid(), o[i], jobs[i])) return false;
            round.push_back(&jobs[i]);
            o[i] += (size_t) s[i]->chunk() * SUB_ * NS_;
            --steps[i];
        }
        if (round.empty()) return true;
        if (!sortformer_session::run_jobs(g, *this, round.data(), (int) round.size())) return false;
    }
}

std::unique_ptr<session> sortformer::new_session(int slice_samples, int lookahead_samples, bool offline, std::string & err) const {
    const int E = SUB_ * HOP;
    if (offline) return std::make_unique<sortformer_session>(*this, OFFLINE_CHUNK, OFFLINE_RC, OFFLINE_FIFO, OFFLINE_UPDATE, true);
    const int chunk = slice_samples > 0 ? std::max(1, (slice_samples + E - 1) / E) : DEFAULT_CHUNK;
    int rc;
    if (lookahead_samples >= 0) {
        rc = (lookahead_samples + E - 1) / E;
        if (rc > OFFLINE_RC) { err = "lookahead_samples exceeds the model maximum of " + std::to_string(OFFLINE_RC * E); return nullptr; }
    } else {
        // the model card's presets, else half a chunk
        switch (chunk) {
            case 9: rc = 4; break;
            case 6: rc = 2; break;
            case 3: rc = 1; break;
            default: rc = std::min(OFFLINE_RC, chunk >= OFFLINE_CHUNK ? OFFLINE_RC : chunk / 2); break;
        }
    }
    // large chunks use the offline FIFO sizes, small ones the streaming presets'
    const bool big = chunk >= OFFLINE_CHUNK;
    return std::make_unique<sortformer_session>(*this, chunk, rc, big ? OFFLINE_FIFO : STREAM_FIFO, big ? OFFLINE_UPDATE : STREAM_UPDATE, false);
}

} // namespace

std::unique_ptr<arch> make_sortformer(model_file & file, std::string & err) {
    auto m = std::make_unique<sortformer>();
    if (!m->configure(file, err)) return nullptr;
    return m;
}

} // namespace vad
