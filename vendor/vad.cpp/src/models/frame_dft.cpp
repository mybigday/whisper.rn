#include "frame_dft.h"

#include <algorithm>
#include <cmath>

namespace vad {

bool frame_dft::init(int win_, int nfft_, std::vector<float> window_, float scale_, bool remove_dc_, float preemph_) {
    win = win_; nfft = nfft_; nbin = nfft_ / 2 + 1;
    window = std::move(window_); scale = scale_; remove_dc = remove_dc_; preemph = preemph_;
    if (win <= 0 || win > nfft || nfft % 2 || (int) window.size() != win) return false;
    const int n = nfft / 2;
    stages_.clear();
    for (int len = n, s = 1; len > 1;) {   // radix 4 first, then 2, 3, 5
        const int p = len % 4 == 0 ? 4 : len % 2 == 0 ? 2 : len % 3 == 0 ? 3 : len % 5 == 0 ? 5 : 0;
        if (!p) return false;
        stage st; st.p = p; st.m = len / p; st.s = s;
        st.wr.resize((size_t) st.m * p); st.wi.resize((size_t) st.m * p);
        for (int j = 0; j < st.m; ++j)
            for (int r = 0; r < p; ++r) {
                const double a = -2.0 * M_PI * j * r / len;
                st.wr[(size_t) j * p + r] = (float) std::cos(a);
                st.wi[(size_t) j * p + r] = (float) std::sin(a);
            }
        stages_.push_back(std::move(st));
        len /= p; s *= p;
    }
    sr_.resize(n + 1); si_.resize(n + 1);
    for (int k = 0; k <= n; ++k) {
        const double a = -2.0 * M_PI * k / nfft;
        sr_[k] = (float) std::cos(a); si_[k] = (float) std::sin(a);
    }
    return true;
}

namespace {

// One Stockham stage (decimation in frequency, self-sorting), on L frames at once
// (element e of frame l at [e * L + l], so every inner loop runs over the L lanes):
//   y[k + s (p j + r)] = w^(j r) sum_q x[k + s (j + q m)] e^(-2 pi i q r / p)
template <int P, int L>
void stage_run(int m, int s, const float * wr, const float * wi, const float * xr, const float * xi, float * yr, float * yi) {
    float cw[P * P], sw[P * P];   // e^(-2 pi i q r / P), used by the direct radices
    if (P != 2 && P != 4)
        for (int i = 0; i < P * P; ++i) {
            const double a = -2.0 * M_PI * ((i / P) * (i % P) % P) / P;
            cw[i] = (float) std::cos(a); sw[i] = (float) std::sin(a);
        }
    for (int j = 0; j < m; ++j) {
        const float * twr = wr + (size_t) j * P, * twi = wi + (size_t) j * P;
        for (int k = 0; k < s; ++k) {
            float ar[P][L], ai[P][L], br[P][L], bi[P][L];
            for (int q = 0; q < P; ++q)
                for (int l = 0; l < L; ++l) {
                    ar[q][l] = xr[(size_t) (k + s * (j + q * m)) * L + l];
                    ai[q][l] = xi[(size_t) (k + s * (j + q * m)) * L + l];
                }
            for (int l = 0; l < L; ++l) {
                if (P == 2) {
                    br[0][l] = ar[0][l] + ar[1][l]; bi[0][l] = ai[0][l] + ai[1][l];
                    br[1][l] = ar[0][l] - ar[1][l]; bi[1][l] = ai[0][l] - ai[1][l];
                } else if (P == 4) {
                    const float s0r = ar[0][l] + ar[2][l], s0i = ai[0][l] + ai[2][l], d0r = ar[0][l] - ar[2][l], d0i = ai[0][l] - ai[2][l];
                    const float s1r = ar[1][l] + ar[3][l], s1i = ai[1][l] + ai[3][l], d1r = ar[1][l] - ar[3][l], d1i = ai[1][l] - ai[3][l];
                    br[0][l] = s0r + s1r; bi[0][l] = s0i + s1i;
                    br[2][l] = s0r - s1r; bi[2][l] = s0i - s1i;
                    br[1][l] = d0r + d1i; bi[1][l] = d0i - d1r;   // d0 - i d1
                    br[3][l] = d0r - d1i; bi[3][l] = d0i + d1r;   // d0 + i d1
                } else {   // 3, 5: direct
                    for (int r = 0; r < P; ++r) {
                        float sr = ar[0][l], si = ai[0][l];
                        for (int q = 1; q < P; ++q) {
                            const float c = cw[q * P + r], sn = sw[q * P + r];
                            sr += ar[q][l] * c - ai[q][l] * sn;
                            si += ar[q][l] * sn + ai[q][l] * c;
                        }
                        br[r][l] = sr; bi[r][l] = si;
                    }
                }
            }
            for (int r = 0; r < P; ++r) {
                const float c = twr[r], sn = twi[r];
                float * oyr = yr + (size_t) (k + s * (P * j + r)) * L, * oyi = yi + (size_t) (k + s * (P * j + r)) * L;
                for (int l = 0; l < L; ++l) {
                    oyr[l] = br[r][l] * c - bi[r][l] * sn;
                    oyi[l] = br[r][l] * sn + bi[r][l] * c;
                }
            }
        }
    }
}

} // namespace

template <int L>
void frame_dft::run_lanes(const float * const * x, float * const * re, float * const * im, float * work) const {
    const int n = nfft / 2;
    float * ar = work, * ai = work + (size_t) n * L, * br = work + (size_t) 2 * n * L, * bi = work + (size_t) 3 * n * L;
    // preprocessed frames, zero padded, packed as z[t] = y[2t] + i y[2t + 1]
    float mean[L], prev[L];
    for (int l = 0; l < L; ++l) {
        mean[l] = 0; prev[l] = 0;
        if (remove_dc) {
            for (int t = 0; t < win; ++t) mean[l] += x[l][t];
            mean[l] /= (float) win;
        }
    }
    for (int t = 0; t < nfft; ++t) {
        float * dst = (t & 1 ? ai : ar) + (size_t) (t >> 1) * L;
        for (int l = 0; l < L; ++l) {
            float y = 0;
            if (t < win) {
                const float v = (x[l][t] - mean[l]) * scale;
                y = (preemph != 0.0f ? v - preemph * (t == 0 ? v : prev[l]) : v) * window[t];
                prev[l] = v;
            }
            dst[l] = y;
        }
    }
    for (const stage & st : stages_) {
        switch (st.p) {
            case 4: stage_run<4, L>(st.m, st.s, st.wr.data(), st.wi.data(), ar, ai, br, bi); break;
            case 2: stage_run<2, L>(st.m, st.s, st.wr.data(), st.wi.data(), ar, ai, br, bi); break;
            case 3: stage_run<3, L>(st.m, st.s, st.wr.data(), st.wi.data(), ar, ai, br, bi); break;
            default: stage_run<5, L>(st.m, st.s, st.wr.data(), st.wi.data(), ar, ai, br, bi); break;
        }
        std::swap(ar, br); std::swap(ai, bi);
    }
    // real split: X[k] = E + w^k O, E = (Z[k] + conj Z[n - k]) / 2, O = -i (Z[k] - conj Z[n - k]) / 2
    for (int k = 0; k < nbin; ++k) {
        const float * za = ar + (size_t) (k % n) * L, * zai = ai + (size_t) (k % n) * L;
        const float * zb = ar + (size_t) ((n - k) % n) * L, * zbi = ai + (size_t) ((n - k) % n) * L;
        for (int l = 0; l < L; ++l) {
            const float zr = za[l], zi = zai[l], cr = zb[l], ci = -zbi[l];
            const float er = 0.5f * (zr + cr), ei = 0.5f * (zi + ci);
            const float orr = 0.5f * (zi - ci), oi = -0.5f * (zr - cr);
            re[l][k] = er + sr_[k] * orr - si_[k] * oi;
            im[l][k] = ei + sr_[k] * oi + si_[k] * orr;
        }
    }
}

void frame_dft::run(const float * x, float * re, float * im, float * work) const {
    run_lanes<1>(&x, &re, &im, work);
}

void frame_dft::run8(const float * const * x, float * const * re, float * const * im, float * work) const {
    run_lanes<8>(x, re, im, work);
}

namespace {

void frame_dft_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    const auto * d = (const frame_dft *) ud;
    const ggml_tensor * a = dst->src[0];
    const int64_t T = a->ne[1], B = a->ne[2], rows = T * B;
    thread_local std::vector<float> work, xs;
    if (work.size() < (size_t) 8 * d->work_size()) work.resize((size_t) 8 * d->work_size());
    const bool contig_rows = a->nb[0] == sizeof(float);
    if (!contig_rows && xs.size() < (size_t) 8 * d->win) xs.resize((size_t) 8 * d->win);
    // rows in groups of 8 (one FFT over 8 frames), the groups split over the threads
    const int64_t groups = (rows + 7) / 8, per = (groups + nth - 1) / nth;
    const int64_t g0 = std::min(groups, per * ith), g1 = std::min(groups, g0 + per);
    const float * xp[8]; float * rp[8], * ip[8];
    for (int64_t gi = g0; gi < g1; ++gi) {
        const int64_t r0 = gi * 8, nr = std::min<int64_t>(8, rows - r0);
        for (int64_t i = 0; i < nr; ++i) {
            const int64_t r = r0 + i, t = r % T, b = r / T;
            const char * src = (const char *) a->data + t * a->nb[1] + b * a->nb[2];
            if (contig_rows) xp[i] = (const float *) src;
            else {
                float * xr = xs.data() + i * d->win;
                for (int n = 0; n < d->win; ++n) xr[n] = *(const float *) (src + n * a->nb[0]);
                xp[i] = xr;
            }
            float * y = (float *) ((char *) dst->data + t * dst->nb[1] + b * dst->nb[2]);
            rp[i] = y; ip[i] = y + d->nbin;
        }
        if (nr == 8) d->run8(xp, rp, ip, work.data());
        else for (int64_t i = 0; i < nr; ++i) d->run(xp[i], rp[i], ip[i], work.data());
    }
}

} // namespace

ggml_tensor * frame_dft::build(ggml_context * c, ggml_tensor * frames, const graph_opts & o) const {
    GGML_ASSERT(frames->type == GGML_TYPE_F32 && frames->ne[0] == win && frames->ne[3] == 1);
    ggml_tensor * args[1] = { frames };
    ggml_tensor * y = ggml_custom_4d(c, GGML_TYPE_F32, 2 * nbin, frames->ne[1], frames->ne[2], 1, args, 1, frame_dft_op,
                                     GGML_N_TASKS_MAX, (void *) this);
    if (!o.host && o.pin_cpu) o.pin_cpu->push_back(y);   // CPU-only op: keep it (and its input) on the CPU
    return y;
}

} // namespace vad
