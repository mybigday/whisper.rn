// Small-shape CPU kernels used by the vad-cpu ggml backend.
// Plain C++ written so that compilers auto-vectorize the inner loops (no
// -ffast-math needed). This file is compiled once per ISA variant with
// VADK_NS set to the variant's namespace (see kernels_*.cpp and CMake).
#ifndef VADK_NS
#error "define VADK_NS before including kernels_impl.inc"
#endif

#include "kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>
#if defined(__AVX512F__) || defined(__F16C__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__wasm_simd128__)
#include <wasm_simd128.h>
#endif

namespace VADK_NS {

using vadk::UOp;
using vadk::BOp;

// ---------------------------------------------------------------- GEMM
// Y[m][n] = (bias ? bias[n] : 0) + sum_k X[m][k] * WT[k][n]   (+ optional relu)
// X rows have stride ldx (floats); WT is packed [K][ldw] as F32 or IEEE half;
// Y rows have stride ldy. The (MB x NT) accumulator tile stays in registers
// across the K loop.

// IEEE half -> float with integer ops and one multiply (vectorizes on every ISA).
// Normals and subnormals are exact; inf/NaN are not handled (weights only).
static inline float h2f(uint16_t h) {
    uint32_t u = (uint32_t) (h & 0x7fff) << 13;
    float f;
    std::memcpy(&f, &u, 4);
    f *= 5.192296858534828e+33f;   // 2^112: rebias the exponent (15 -> 127)
    std::memcpy(&u, &f, 4);
    u |= (uint32_t) (h & 0x8000) << 16;
    std::memcpy(&f, &u, 4);
    return f;
}
// convert n halves (n % 8 == 0) to floats
static inline void h2f_row(const uint16_t * __restrict h, float * __restrict f, int n) {
#if defined(__AVX512F__)
    int i = 0;
    for (; i + 16 <= n; i += 16) _mm512_storeu_ps(f + i, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (h + i))));
    for (; i < n; i += 8) _mm256_storeu_ps(f + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (h + i))));
#elif defined(__F16C__)
    for (int i = 0; i < n; i += 8) _mm256_storeu_ps(f + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (h + i))));
#elif defined(__aarch64__)
    for (int i = 0; i < n; i += 4) vst1q_f32(f + i, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(h + i))));
#else
    for (int i = 0; i < n; ++i) f[i] = h2f(h[i]);
#endif
}

// Accumulator tile width in floats (MB rows x TILE/MB columns). It must fit the vector
// register file with room for the weight row: 8 zmm / 16 ymm on x86, but 16 of NEON's 32
// q registers and 8 of the 16 that WASM SIMD maps to. Q8_0 tiles hold a partial sum as
// well, so they use half (except on AVX-512). Tile width does not change any column's summation order.
#if defined(__aarch64__)
constexpr int TILE = 64, TILE_Q8 = 32;
#elif defined(__wasm_simd128__)
constexpr int TILE = 32, TILE_Q8 = 16;
#else
constexpr int TILE = 128, TILE_Q8 = 128;   // 8 + 8 zmm
#endif

#if defined(__aarch64__)
// NEON accumulation for the narrow tiles: accumulators stay in registers (GCC's
// auto-vectorized version stores them to the stack every k), and F16 / int8 weights
// are widened in registers. Same arithmetic as the generic loop: fma in k order.
template <int NV> static inline void neon_wload(const float * w, float32x4_t * v) {
    for (int j = 0; j < NV; ++j) v[j] = vld1q_f32(w + 4 * j);
}
template <int NV> static inline void neon_wload(const uint16_t * w, float32x4_t * v) {
    for (int j = 0; j + 1 < NV; j += 2) {
        const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(w + 4 * j));
        v[j] = vcvt_f32_f16(vget_low_f16(h));
        v[j + 1] = vcvt_high_f32_f16(h);
    }
    if (NV & 1) v[NV - 1] = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(w + 4 * (NV - 1))));
}
template <int NV> static inline void neon_wload(const int8_t * w, float32x4_t * v) {
    static_assert(NV % 2 == 0, "int8 tiles are a multiple of 8 columns");
    int j = 0;
    for (; j + 4 <= NV; j += 4) {
        const int8x16_t q = vld1q_s8(w + 4 * j);
        const int16x8_t lo = vmovl_s8(vget_low_s8(q)), hi = vmovl_high_s8(q);
        v[j] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
        v[j + 1] = vcvtq_f32_s32(vmovl_high_s16(lo));
        v[j + 2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
        v[j + 3] = vcvtq_f32_s32(vmovl_high_s16(hi));
    }
    if (j < NV) {   // 8 columns left: load exactly 8 bytes
        const int16x8_t lo = vmovl_s8(vld1_s8(w + 4 * j));
        v[j] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
        v[j + 1] = vcvtq_f32_s32(vmovl_high_s16(lo));
    }
}
// acc[m][*] += sum_{k in [k0, k1)} X[m][k] * W[k][*]
template <int MB, int NV, typename TW>
static inline void neon_accum(const float * __restrict X, int64_t ldx, const TW * __restrict W, int64_t ldw, int k0, int k1,
                              float32x4_t (&acc)[MB][NV]) {
    for (int k = k0; k < k1; ++k) {
        float32x4_t w[NV];
        neon_wload<NV>(W + (int64_t) k * ldw, w);
        for (int m = 0; m < MB; ++m) {
            const float32x4_t xv = vld1q_dup_f32(X + m * ldx + k);
            for (int j = 0; j < NV; ++j) acc[m][j] = vfmaq_f32(acc[m][j], w[j], xv);
        }
    }
}
#endif

template <int MB, int NT, typename TW>
static inline void gemm_tile(const float * __restrict X, int64_t ldx, const TW * __restrict WT, int64_t ldw, int K,
                             const float * __restrict bias, bool relu, float * __restrict Y, int64_t ldy, int n_valid) {
    float acc[MB][NT];
#if defined(__aarch64__)
    {
        constexpr int NV = NT / 4;
        float32x4_t a[MB][NV];
        for (int m = 0; m < MB; ++m)
            for (int j = 0; j < NV; ++j) a[m][j] = vdupq_n_f32(0.0f);
        neon_accum<MB, NV>(X, ldx, WT, ldw, 0, K, a);
        for (int m = 0; m < MB; ++m)
            for (int j = 0; j < NV; ++j) vst1q_f32(&acc[m][4 * j], a[m][j]);
    }
#else
    for (int m = 0; m < MB; ++m)
        for (int j = 0; j < NT; ++j) acc[m][j] = 0.0f;
    for (int k = 0; k < K; ++k) {
        const TW * wr = WT + (int64_t) k * ldw;
        if constexpr (std::is_same<TW, float>::value) {
            for (int m = 0; m < MB; ++m) {
                const float xv = X[m * ldx + k];
                for (int j = 0; j < NT; ++j) acc[m][j] += xv * wr[j];
            }
        } else {
            alignas(64) float w[NT];   // convert the row once, reuse it for every row of X
            h2f_row(wr, w, NT);
            for (int m = 0; m < MB; ++m) {
                const float xv = X[m * ldx + k];
                for (int j = 0; j < NT; ++j) acc[m][j] += xv * w[j];
            }
        }
    }
#endif
    for (int m = 0; m < MB; ++m) {
        float * y = Y + m * ldy;
        for (int j = 0; j < n_valid; ++j) {
            float v = acc[m][j] + (bias ? bias[j] : 0.0f);
            y[j] = relu ? (v > 0 ? v : 0) : v;
        }
    }
}

template <int MB, typename TW>
static inline void gemm_rows(const float * X, int64_t ldx, const TW * WT, int64_t ldw, int K, int N,
                             const float * bias, bool relu, float * Y, int64_t ldy) {
    constexpr int NT_BIG = TILE / MB >= 4 ? TILE / MB : 4;
    constexpr int NT_TAIL = NT_BIG < 32 ? NT_BIG : 32;   // divides 32: WT is zero-padded to a multiple of 32 columns
    int n0 = 0;
    for (; n0 + NT_BIG <= N; n0 += NT_BIG)
        gemm_tile<MB, NT_BIG>(X, ldx, WT + n0, ldw, K, bias ? bias + n0 : nullptr, relu, Y + n0, ldy, NT_BIG);
    for (; n0 < N; n0 += NT_TAIL)
        gemm_tile<MB, NT_TAIL>(X, ldx, WT + n0, ldw, K, bias ? bias + n0 : nullptr, relu, Y + n0, ldy, N - n0 < NT_TAIL ? N - n0 : NT_TAIL);
}

// M rows of X, any M. For more than a few rows, loop over column panels first so
// one K x 128 panel of WT stays in cache while every row block streams over it.
#if defined(__aarch64__)
// One row of X with F32 weights of 64 KB or more (aarch64): stream the weights row by
// row (k outer) with the accumulators of a 512-column panel in L1. The register tiles
// read 256 bytes per weight row with a large stride, which a Cortex-A78 streams ~2-3x
// slower once the matrix is out of L1/L2. (F16 / Q8_0 rows are bound by the widening
// instead, and several rows by FMA: the tiles stay faster there.) Per column the
// arithmetic is the same (fma in k order from zero, bias last): identical results.
constexpr int KO_P = 512;
template <int MB, typename TW>
static void gemm_kouter(const float * __restrict X, int64_t ldx, const TW * __restrict WT, int64_t ldw, int K, int N,
                        const float * __restrict bias, bool relu, float * __restrict Y, int64_t ldy) {
    alignas(16) float acc[MB][KO_P];
    for (int n0 = 0; n0 < N; n0 += KO_P) {
        const int nw = N - n0 < KO_P ? N - n0 : KO_P;
        const int n8 = (nw + 7) / 8 * 8;   // WT is zero-padded to a multiple of 32 columns
        for (int m = 0; m < MB; ++m)
            for (int j = 0; j < n8; j += 4) vst1q_f32(acc[m] + j, vdupq_n_f32(0.0f));
        for (int k = 0; k < K; ++k) {
            const TW * w = WT + (int64_t) k * ldw + n0;
            float32x4_t xv[MB];
            for (int m = 0; m < MB; ++m) xv[m] = vld1q_dup_f32(X + m * ldx + k);
            for (int j = 0; j < n8; j += 8) {
                float32x4_t w2[2];
                neon_wload<2>(w + j, w2);
                for (int m = 0; m < MB; ++m) {
                    vst1q_f32(acc[m] + j,     vfmaq_f32(vld1q_f32(acc[m] + j),     w2[0], xv[m]));
                    vst1q_f32(acc[m] + j + 4, vfmaq_f32(vld1q_f32(acc[m] + j + 4), w2[1], xv[m]));
                }
            }
        }
        for (int m = 0; m < MB; ++m) {
            float * y = Y + m * ldy + n0;
            for (int j = 0; j < nw; ++j) {
                float v = acc[m][j] + (bias ? bias[n0 + j] : 0.0f);
                y[j] = relu ? (v > 0 ? v : 0) : v;
            }
        }
    }
}
#endif

template <typename TW>
static void gemm_t(const float * X, int64_t ldx, int M, const TW * WT, int64_t ldw, int K, int N,
                   const float * bias, bool relu, float * Y, int64_t ldy) {
#if defined(__aarch64__)
    if constexpr (std::is_same<TW, float>::value)
        if (M == 1 && (int64_t) K * N * 4 >= (64 << 10)) {
            gemm_kouter<1, float>(X, ldx, WT, ldw, K, N, bias, relu, Y, ldy);
            return;
        }
#endif
    constexpr int PANEL = 128;
    const int panel = M > 4 ? PANEL : N;
    for (int n0 = 0; n0 < N; n0 += panel) {
        const int nw = N - n0 < panel ? N - n0 : panel;
        const TW * W = WT + n0;
        const float * b = bias ? bias + n0 : nullptr;
        float * Yp = Y + n0;
        int m = 0;
        for (; m + 4 <= M; m += 4) gemm_rows<4>(X + m * ldx, ldx, W, ldw, K, nw, b, relu, Yp + m * ldy, ldy);
        for (; m + 2 <= M; m += 2) gemm_rows<2>(X + m * ldx, ldx, W, ldw, K, nw, b, relu, Yp + m * ldy, ldy);
        for (; m < M; ++m)         gemm_rows<1>(X + m * ldx, ldx, W, ldw, K, nw, b, relu, Yp + m * ldy, ldy);
    }
}

static void gemm(const float * X, int64_t ldx, int M, const float * WT, int64_t ldw, int K, int N,
                 const float * bias, bool relu, float * Y, int64_t ldy) {
    gemm_t(X, ldx, M, WT, ldw, K, N, bias, relu, Y, ldy);
}

// int8 weights: WQ [K][ldw] int8, WS [K/32][ldw] float scales (K % 32 == 0).
// Within a 32-row block the unscaled weights are accumulated into a partial sum,
// which is scaled once per block (no per-element multiply).
template <int MB, int NT>
static inline void gemm_tile_q8(const float * __restrict X, int64_t ldx, const int8_t * __restrict WQ, const float * __restrict WS,
                                int64_t ldw, int K, const float * __restrict bias, bool relu, float * __restrict Y, int64_t ldy, int n_valid) {
    float acc[MB][NT];
#if defined(__aarch64__)
    {
        constexpr int NV = NT / 4;
        float32x4_t a[MB][NV];
        for (int m = 0; m < MB; ++m)
            for (int j = 0; j < NV; ++j) a[m][j] = vdupq_n_f32(0.0f);
        for (int k0 = 0; k0 < K; k0 += 32) {
            float32x4_t part[MB][NV];
            for (int m = 0; m < MB; ++m)
                for (int j = 0; j < NV; ++j) part[m][j] = vdupq_n_f32(0.0f);
            neon_accum<MB, NV>(X, ldx, WQ, ldw, k0, k0 + 32, part);
            const float * sc = WS + (int64_t) (k0 >> 5) * ldw;
            for (int j = 0; j < NV; ++j) {
                const float32x4_t s4 = vld1q_f32(sc + 4 * j);
                for (int m = 0; m < MB; ++m) a[m][j] = vfmaq_f32(a[m][j], part[m][j], s4);
            }
        }
        for (int m = 0; m < MB; ++m)
            for (int j = 0; j < NV; ++j) vst1q_f32(&acc[m][4 * j], a[m][j]);
    }
#else
    for (int m = 0; m < MB; ++m)
        for (int j = 0; j < NT; ++j) acc[m][j] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 32) {
        float part[MB][NT];
        for (int m = 0; m < MB; ++m)
            for (int j = 0; j < NT; ++j) part[m][j] = 0.0f;
        for (int k = k0; k < k0 + 32; ++k) {
            const int8_t * q = WQ + (int64_t) k * ldw;
            for (int m = 0; m < MB; ++m) {
                const float xv = X[m * ldx + k];
                for (int j = 0; j < NT; ++j) part[m][j] += xv * (float) q[j];
            }
        }
        const float * sc = WS + (int64_t) (k0 >> 5) * ldw;
        for (int m = 0; m < MB; ++m)
            for (int j = 0; j < NT; ++j) acc[m][j] += part[m][j] * sc[j];
    }
#endif
    for (int m = 0; m < MB; ++m) {
        float * y = Y + m * ldy;
        for (int j = 0; j < n_valid; ++j) {
            float v = acc[m][j] + (bias ? bias[j] : 0.0f);
            y[j] = relu ? (v > 0 ? v : 0) : v;
        }
    }
}

template <int MB>
static inline void gemm_rows_q8(const float * X, int64_t ldx, const int8_t * WQ, const float * WS, int64_t ldw, int K, int N,
                                const float * bias, bool relu, float * Y, int64_t ldy) {
    constexpr int NT_BIG = TILE_Q8 / MB >= 4 ? TILE_Q8 / MB : 4;
    constexpr int NT_TAIL = NT_BIG < 32 ? NT_BIG : 32;
    int n0 = 0;
    for (; n0 + NT_BIG <= N; n0 += NT_BIG)
        gemm_tile_q8<MB, NT_BIG>(X, ldx, WQ + n0, WS + n0, ldw, K, bias ? bias + n0 : nullptr, relu, Y + n0, ldy, NT_BIG);
    for (; n0 < N; n0 += NT_TAIL)
        gemm_tile_q8<MB, NT_TAIL>(X, ldx, WQ + n0, WS + n0, ldw, K, bias ? bias + n0 : nullptr, relu, Y + n0, ldy, N - n0 < NT_TAIL ? N - n0 : NT_TAIL);
}

static void gemm_q8(const float * X, int64_t ldx, int M, const int8_t * WQ, const float * WS, int64_t ldw, int K, int N,
                    const float * bias, bool relu, float * Y, int64_t ldy) {
    constexpr int PANEL = 128;
    const int panel = M > 4 ? PANEL : N;
    for (int n0 = 0; n0 < N; n0 += panel) {
        const int nw = N - n0 < panel ? N - n0 : panel;
        const float * b = bias ? bias + n0 : nullptr;
        int m = 0;
        for (; m + 4 <= M; m += 4) gemm_rows_q8<4>(X + m * ldx, ldx, WQ + n0, WS + n0, ldw, K, nw, b, relu, Y + n0 + m * ldy, ldy);
        for (; m + 2 <= M; m += 2) gemm_rows_q8<2>(X + m * ldx, ldx, WQ + n0, WS + n0, ldw, K, nw, b, relu, Y + n0 + m * ldy, ldy);
        for (; m < M; ++m)         gemm_rows_q8<1>(X + m * ldx, ldx, WQ + n0, WS + n0, ldw, K, nw, b, relu, Y + n0 + m * ldy, ldy);
    }
}

static void gemm_f16(const float * X, int64_t ldx, int M, const uint16_t * WT, int64_t ldw, int K, int N,
                     const float * bias, bool relu, float * Y, int64_t ldy) {
    gemm_t(X, ldx, M, WT, ldw, K, N, bias, relu, Y, ldy);
}

// ---------------------------------------------------------------- many-row GEMM
// BLIS-style blocking for M >= gemm_big_min_rows (offline / batched graphs):
//   for each K block (KC): for each NR-column panel: the KC x NR weight panel is
//   converted to F32 into an L1-resident buffer (this is where F16 / Q8_0 weights
//   are expanded, once per panel instead of once per row block), then every MR-row
//   block of X runs a register-tiled MR x NR microkernel over it.
// The caller keeps M and N per call moderate (2D task tiling), so the X block
// (M x KC) stays in L2 across panels. Partial sums go through Y between K blocks.
// The arithmetic is the narrow kernels' exactly (sums from zero in k order, bias
// added last; Q8_0: unscaled 32-row partial sums, scaled once per block), so a
// stream (narrow kernel) and an offline run (this one) give identical results.
#if defined(__AVX512F__) || (defined(__AVX2__) && defined(__FMA__)) || defined(__aarch64__) || defined(__wasm_simd128__)
#define VADK_HAS_GEMM_BIG 1
#if defined(__AVX512F__)
constexpr int BIG_MR = 12, BIG_MR_Q8 = 6, BIG_NR = 32;    // 32 zmm: 24 (Q8: 12 + 12) accumulators
#elif defined(__AVX2__)
constexpr int BIG_MR = 6, BIG_MR_Q8 = 3, BIG_NR = 16;     // 16 ymm: 12 (Q8: 6 + 6) accumulators
#else
constexpr int BIG_MR = 6, BIG_MR_Q8 = 3, BIG_NR = 16;     // NEON 32 q-regs: 24 (Q8: 12 + 12); WASM: same shape
#endif
constexpr int BIG_KC = 256;   // multiple of 32 (Q8_0 blocks)

// B panel [kc][BIG_NR] (F32; Q8_0: unscaled) from packed weights at row k0, column n0;
// Q8_0 scales go to sp [kc / 32][BIG_NR]
static inline void big_pack(const vadk::weights & w, int k0, int kc, int n0, float * __restrict bp, float * __restrict sp) {
    for (int k = 0; k < kc; ++k) {
        const int64_t r = (int64_t) (k0 + k) * w.ldw + n0;
        float * d = bp + k * BIG_NR;
        if (w.f32) std::memcpy(d, w.f32 + r, sizeof(float) * BIG_NR);
        else if (w.f16) h2f_row(w.f16 + r, d, BIG_NR);
        else for (int j = 0; j < BIG_NR; ++j) d[j] = (float) w.q8[r + j];
    }
    if (w.q8)
        for (int k = 0; k < kc; k += 32)
            std::memcpy(sp + (k >> 5) * BIG_NR, w.qs + (int64_t) ((k0 + k) >> 5) * w.ldw + n0, sizeof(float) * BIG_NR);
}

#if defined(__AVX512F__)
using vreg = __m512;
constexpr int VL = 16;
static inline vreg vload(const float * p) { return _mm512_loadu_ps(p); }
static inline void vstore(float * p, vreg v) { _mm512_storeu_ps(p, v); }
static inline vreg vset1(float x) { return _mm512_set1_ps(x); }
static inline vreg vfma(vreg a, vreg b, vreg c) { return _mm512_fmadd_ps(a, b, c); }
static inline vreg vadd(vreg a, vreg b) { return _mm512_add_ps(a, b); }
static inline vreg vmax0(vreg a) { return _mm512_max_ps(a, _mm512_setzero_ps()); }
static inline vreg vzero() { return _mm512_setzero_ps(); }
#elif defined(__AVX2__)
using vreg = __m256;
constexpr int VL = 8;
static inline vreg vload(const float * p) { return _mm256_loadu_ps(p); }
static inline void vstore(float * p, vreg v) { _mm256_storeu_ps(p, v); }
static inline vreg vset1(float x) { return _mm256_set1_ps(x); }
static inline vreg vfma(vreg a, vreg b, vreg c) { return _mm256_fmadd_ps(a, b, c); }
static inline vreg vadd(vreg a, vreg b) { return _mm256_add_ps(a, b); }
static inline vreg vmax0(vreg a) { return _mm256_max_ps(a, _mm256_setzero_ps()); }
static inline vreg vzero() { return _mm256_setzero_ps(); }
#elif defined(__aarch64__)
using vreg = float32x4_t;
constexpr int VL = 4;
static inline vreg vload(const float * p) { return vld1q_f32(p); }
static inline void vstore(float * p, vreg v) { vst1q_f32(p, v); }
static inline vreg vset1(float x) { return vdupq_n_f32(x); }
static inline vreg vfma(vreg a, vreg b, vreg c) { return vfmaq_f32(c, a, b); }   // the narrow kernels' contracted a * b + c
static inline vreg vadd(vreg a, vreg b) { return vaddq_f32(a, b); }
static inline vreg vmax0(vreg a) { return vbslq_f32(vcgtq_f32(a, vdupq_n_f32(0.0f)), a, vdupq_n_f32(0.0f)); }   // v > 0 ? v : 0
static inline vreg vzero() { return vdupq_n_f32(0.0f); }
#else   // WASM SIMD128: no FMA, so mul + add, as the narrow kernels compile there
using vreg = v128_t;
constexpr int VL = 4;
static inline vreg vload(const float * p) { return wasm_v128_load(p); }
static inline void vstore(float * p, vreg v) { wasm_v128_store(p, v); }
static inline vreg vset1(float x) { return wasm_f32x4_splat(x); }
static inline vreg vfma(vreg a, vreg b, vreg c) { return wasm_f32x4_add(wasm_f32x4_mul(a, b), c); }
static inline vreg vadd(vreg a, vreg b) { return wasm_f32x4_add(a, b); }
static inline vreg vmax0(vreg a) { return wasm_f32x4_pmax(wasm_f32x4_splat(0.0f), a); }   // 0 < v ? v : 0
static inline vreg vzero() { return wasm_f32x4_splat(0.0f); }
#endif
constexpr int NV = BIG_NR / VL;   // vector registers per tile row

// C[MR][BIG_NR] (+)= X[MR][kc] . bp[kc][BIG_NR]; Y rows of full width BIG_NR.
// first: start from zero (else from Y); last: add the bias, then the ReLU.
template <int MR, bool Q8>
static inline void big_micro(int kc, const float * __restrict X, int64_t ldx, const float * __restrict bp, const float * __restrict sp,
                             float * __restrict Y, int64_t ldy, const float * __restrict bias, bool first, bool last, bool relu) {
    vreg c[MR][NV];
    for (int m = 0; m < MR; ++m)
        for (int v = 0; v < NV; ++v) c[m][v] = first ? vzero() : vload(Y + m * ldy + v * VL);
    const float * x[MR];
    for (int m = 0; m < MR; ++m) x[m] = X + m * ldx;
    if constexpr (!Q8) {
        for (int k = 0; k < kc; ++k) {
            vreg b[NV];
            for (int v = 0; v < NV; ++v) b[v] = vload(bp + k * BIG_NR + v * VL);
            for (int m = 0; m < MR; ++m) {
                const vreg a = vset1(x[m][k]);
                for (int v = 0; v < NV; ++v) c[m][v] = vfma(a, b[v], c[m][v]);
            }
        }
    } else {
        for (int k0 = 0; k0 < kc; k0 += 32) {
            vreg p[MR][NV];
            for (int m = 0; m < MR; ++m)
                for (int v = 0; v < NV; ++v) p[m][v] = vzero();
            for (int k = k0; k < k0 + 32; ++k) {
                vreg b[NV];
                for (int v = 0; v < NV; ++v) b[v] = vload(bp + k * BIG_NR + v * VL);
                for (int m = 0; m < MR; ++m) {
                    const vreg a = vset1(x[m][k]);
                    for (int v = 0; v < NV; ++v) p[m][v] = vfma(a, b[v], p[m][v]);
                }
            }
            for (int v = 0; v < NV; ++v) {
                const vreg sc = vload(sp + (k0 >> 5) * BIG_NR + v * VL);
                for (int m = 0; m < MR; ++m) c[m][v] = vfma(p[m][v], sc, c[m][v]);
            }
        }
    }
    for (int m = 0; m < MR; ++m)
        for (int v = 0; v < NV; ++v) {
            vreg r = c[m][v];
            if (last && bias) r = vadd(r, vload(bias + v * VL));
            vstore(Y + m * ldy + v * VL, last && relu ? vmax0(r) : r);
        }
}

template <int MR, bool Q8>
static void big_rows(int kc, const float * X, int64_t ldx, const float * bp, const float * sp, float * Y, int64_t ldy,
                     const float * bias, bool first, bool last, bool relu, int mr) {
    if (mr == MR) big_micro<MR, Q8>(kc, X, ldx, bp, sp, Y, ldy, bias, first, last, relu);
    else if constexpr (MR > 1) big_rows<MR - 1, Q8>(kc, X, ldx, bp, sp, Y, ldy, bias, first, last, relu, mr);
}

template <bool Q8>
static void gemm_big_t(const float * X, int64_t ldx, int M, const vadk::weights & w, int K, int N,
                       const float * bias, bool relu, float * Y, int64_t ldy) {
    constexpr int MR = Q8 ? BIG_MR_Q8 : BIG_MR;
    alignas(64) float bp[BIG_KC * BIG_NR];
    alignas(64) float sp[BIG_KC / 32 * BIG_NR];
    alignas(64) float tile[MR * BIG_NR];
    alignas(64) float btile[BIG_NR];
    std::memset(tile, 0, sizeof tile);
    for (int k0 = 0; k0 < K; k0 += BIG_KC) {
        const int kc = K - k0 < BIG_KC ? K - k0 : BIG_KC;
        const bool first = k0 == 0, last = k0 + kc == K;
        for (int n0 = 0; n0 < N; n0 += BIG_NR) {
            big_pack(w, k0, kc, n0, bp, sp);   // weights are zero padded to a multiple of 32 columns
            const int nv = N - n0 < BIG_NR ? N - n0 : BIG_NR;
            const float * b = bias ? bias + n0 : nullptr;
            if (b && nv < BIG_NR) { std::memset(btile, 0, sizeof btile); std::memcpy(btile, b, sizeof(float) * nv); b = btile; }
            for (int m0 = 0; m0 < M; m0 += MR) {
                const int mr = M - m0 < MR ? M - m0 : MR;
                const float * Xb = X + (int64_t) m0 * ldx + k0;
                float * Yb = Y + (int64_t) m0 * ldy + n0;
                if (nv == BIG_NR) {
                    big_rows<MR, Q8>(kc, Xb, ldx, bp, sp, Yb, ldy, b, first, last, relu, mr);
                } else {   // right edge: through a full-width tile
                    if (!first)
                        for (int m = 0; m < mr; ++m) std::memcpy(tile + m * BIG_NR, Yb + m * ldy, sizeof(float) * nv);
                    big_rows<MR, Q8>(kc, Xb, ldx, bp, sp, tile, BIG_NR, b, first, last, relu, mr);
                    for (int m = 0; m < mr; ++m) std::memcpy(Yb + m * ldy, tile + m * BIG_NR, sizeof(float) * nv);
                }
            }
        }
    }
}

static void gemm_big(const float * X, int64_t ldx, int M, const vadk::weights & w, int K, int N,
                     const float * bias, bool relu, float * Y, int64_t ldy) {
    if (w.q8) gemm_big_t<true>(X, ldx, M, w, K, N, bias, relu, Y, ldy);
    else      gemm_big_t<false>(X, ldx, M, w, K, N, bias, relu, Y, ldy);
}
#endif

// ---------------------------------------------------------------- math
// Written so every loop below vectorizes: clamps as ternaries (std::fmin/fmax keep
// NaN semantics that block vectorization), magic-number rounding, copysign.
// expf: Cody-Waite range reduction + degree-6 polynomial, ~1-2 ulp.
static inline float exp_f(float x) {
    x = x < -87.0f ? -87.0f : x;
    x = x > 88.0f ? 88.0f : x;
    const float t = x * 1.44269504088896341f;
    const float n = (t + 12582912.0f) - 12582912.0f;   // round to nearest (1.5 * 2^23); no -ffast-math here
    const float r = (x - n * 0.693359375f) - n * -2.12194440e-4f;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = p * r * r + r + 1.0f;
    return p * __builtin_bit_cast(float, ((int32_t) n + 127) << 23);   // 2^n
}
static inline float sigmoid_f(float x) { return 1.0f / (1.0f + exp_f(-x)); }
// erf: Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7)
static inline float erf_f(float x) {
    const float a = std::fabs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float p = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    return std::copysign(1.0f - p * exp_f(-a * a), x);
}
// logf: exponent split + Cephes polynomial on [sqrt(1/2), sqrt(2)), ~1 ulp; 0 -> -inf,
// negative / NaN -> NaN, +inf -> +inf, denormals rescaled first
static inline float log_f(float x) {
    const bool den = x < 1.17549435e-38f;
    const float xs = den ? x * 8388608.0f : x;   // 2^23
    const int32_t bits = __builtin_bit_cast(int32_t, xs);
    float e = (float) (((bits >> 23) & 0xff) - 127) - (den ? 23.0f : 0.0f);
    float m = __builtin_bit_cast(float, (bits & 0x007fffff) | 0x3f800000);   // [1, 2)
    const bool hi = m > 1.41421356f;
    m = hi ? m * 0.5f : m;
    e = hi ? e + 1.0f : e;
    const float f = m - 1.0f, z = f * f;
    float p = 7.0376836292e-2f;
    p = p * f - 1.1514610310e-1f;
    p = p * f + 1.1676998740e-1f;
    p = p * f - 1.2420140846e-1f;
    p = p * f + 1.4249322787e-1f;
    p = p * f - 1.6668057665e-1f;
    p = p * f + 2.0000714765e-1f;
    p = p * f - 2.4999993993e-1f;
    p = p * f + 3.3333331174e-1f;
    float y = p * f * z;
    y += e * -2.12194440e-4f;
    y += -0.5f * z;
    float r = f + y + e * 0.693359375f;
    r = x == __builtin_inff() ? x : r;
    r = x == 0.0f ? -__builtin_inff() : r;
    return x < 0.0f || x != x ? __builtin_nanf("") : r;
}
static inline float tanh_f(float x) {
    // tanh(x) = sign(x) * (1 - e) / (1 + e), e = exp(-2|x|): no cancellation for large |x|
    const float e = exp_f(-2.0f * std::fabs(x));
    return std::copysign((1.0f - e) / (1.0f + e), x);
}

static void unary(UOp op, const float * __restrict x, float * __restrict y, int n) {
    switch (op) {
        case UOp::RELU:    for (int i = 0; i < n; ++i) y[i] = x[i] > 0 ? x[i] : 0; break;
        case UOp::SIGMOID: for (int i = 0; i < n; ++i) y[i] = sigmoid_f(x[i]); break;
        case UOp::TANH:    for (int i = 0; i < n; ++i) y[i] = tanh_f(x[i]); break;
        case UOp::SQRT:    for (int i = 0; i < n; ++i) y[i] = std::sqrt(x[i]); break;
        case UOp::NEG:     for (int i = 0; i < n; ++i) y[i] = -x[i]; break;
        case UOp::ABS:     for (int i = 0; i < n; ++i) y[i] = std::fabs(x[i]); break;
        case UOp::LOG:     for (int i = 0; i < n; ++i) y[i] = log_f(x[i]); break;
        case UOp::EXP:     for (int i = 0; i < n; ++i) y[i] = exp_f(x[i]); break;
        case UOp::GELU_ERF: for (int i = 0; i < n; ++i) y[i] = 0.5f * x[i] * (1.0f + erf_f(x[i] * 0.70710678118654752f)); break;
    }
}

// y = a op b, b either full row (bn == n) or a broadcast scalar (bn == 1)
static void binary(BOp op, const float * __restrict a, const float * __restrict b, int bn, float * y, int n) {
    if (bn == n) {
        switch (op) {
            case BOp::ADD: for (int i = 0; i < n; ++i) y[i] = a[i] + b[i]; break;
            case BOp::SUB: for (int i = 0; i < n; ++i) y[i] = a[i] - b[i]; break;
            case BOp::MUL: for (int i = 0; i < n; ++i) y[i] = a[i] * b[i]; break;
            case BOp::DIV: for (int i = 0; i < n; ++i) y[i] = a[i] / b[i]; break;
        }
    } else {
        const float s = b[0];
        switch (op) {
            case BOp::ADD: for (int i = 0; i < n; ++i) y[i] = a[i] + s; break;
            case BOp::SUB: for (int i = 0; i < n; ++i) y[i] = a[i] - s; break;
            case BOp::MUL: for (int i = 0; i < n; ++i) y[i] = a[i] * s; break;
            case BOp::DIV: for (int i = 0; i < n; ++i) y[i] = a[i] / s; break;
        }
    }
}

// ---------------------------------------------------------------- reduced precision
// ARMv8.2 fp16 + dotprod only; vad-cpu uses these for F16 / Q8_0 weights when present
// (approximate activations, as ggml does, in exchange for speed):
//   gemm_f16_lp: weights in the file's [N][K] layout, activations rounded to fp16, fp16
//     FMA over blocks of 64 k (8 products per lane), each block widened into F32
//     accumulators (ggml-cpu accumulates the whole row in fp16). A block that
//     overflows fp16 is recomputed exactly.
//   gemm_q8_lp: activations quantized per 32-block to int8 (like ggml's Q8_0 x Q8_0),
//     int8 dot products (SDOT), one F32 scale per block. Weights packed k-interleaved:
//     WQ4[k / 4][ldw][4] (the 4 consecutive k of a column are adjacent).
// Results differ slightly from the exact kernels (and between platforms).
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__ARM_FEATURE_DOTPROD)
#define VADK_HAS_LP 1

// Weights in the file's row-major layout W[n][k] (row stride ldr elements): every
// output is a dot product over a contiguous weight row, 4 columns at a time, so the
// weights stream sequentially. fp16 FMA over blocks of 64 k (each lane sums 8
// products), then widened into F32 accumulators.
static inline float lp_dot_exact(const float * x, const uint16_t * w, int k0, int k1, float acc) {
    for (int k = k0; k < k1; ++k) acc = std::fma(x[k], h2f(w[k]), acc);
    return acc;
}

static void lp_cols_f16(const float * __restrict x, const float16_t * __restrict xh, const uint16_t * __restrict W, int64_t ldr,
                        int K, int n0, int n1, const float * __restrict bias, bool relu, float * __restrict y) {
    constexpr int KB = 64;
    const int K8 = K / 8 * 8;
    int n = n0;
    for (; n + 4 <= n1; n += 4) {
        const float16_t * w[4];
        for (int c = 0; c < 4; ++c) w[c] = (const float16_t *) (W + (int64_t) (n + c) * ldr);
        float32x4_t acc[4];
        for (int c = 0; c < 4; ++c) acc[c] = vdupq_n_f32(0.0f);
        for (int k0 = 0; k0 < K8; k0 += KB) {
            const int k1 = k0 + KB < K8 ? k0 + KB : K8;
            float16x8_t p0 = vdupq_n_f16(0), p1 = p0, p2 = p0, p3 = p0;
            for (int k = k0; k < k1; k += 8) {
                const float16x8_t xv = vld1q_f16(xh + k);
                p0 = vfmaq_f16(p0, vld1q_f16(w[0] + k), xv);
                p1 = vfmaq_f16(p1, vld1q_f16(w[1] + k), xv);
                p2 = vfmaq_f16(p2, vld1q_f16(w[2] + k), xv);
                p3 = vfmaq_f16(p3, vld1q_f16(w[3] + k), xv);
            }
            acc[0] = vaddq_f32(acc[0], vaddq_f32(vcvt_f32_f16(vget_low_f16(p0)), vcvt_high_f32_f16(p0)));
            acc[1] = vaddq_f32(acc[1], vaddq_f32(vcvt_f32_f16(vget_low_f16(p1)), vcvt_high_f32_f16(p1)));
            acc[2] = vaddq_f32(acc[2], vaddq_f32(vcvt_f32_f16(vget_low_f16(p2)), vcvt_high_f32_f16(p2)));
            acc[3] = vaddq_f32(acc[3], vaddq_f32(vcvt_f32_f16(vget_low_f16(p3)), vcvt_high_f32_f16(p3)));
        }
        for (int c = 0; c < 4; ++c) {
            float d = vaddvq_f32(acc[c]);
            // an fp16 overflow (or inf/NaN input) leaves inf/NaN: redo the column exactly
            if (!std::isfinite(d)) d = lp_dot_exact(x, (const uint16_t *) w[c], 0, K8, 0.0f);
            float v = lp_dot_exact(x, (const uint16_t *) w[c], K8, K, d) + (bias ? bias[n + c] : 0.0f);
            y[n + c] = relu ? (v > 0 ? v : 0) : v;
        }
    }
    for (; n < n1; ++n) {
        float v = lp_dot_exact(x, W + (int64_t) n * ldr, 0, K, 0.0f) + (bias ? bias[n] : 0.0f);
        y[n] = relu ? (v > 0 ? v : 0) : v;
    }
}

// M > 1: packed [K][ldw] weights, register tile of MB rows x 8*NV columns, fp16 FMA over
// blocks of 16 k widened into F32 accumulators (each weight row read once per MB rows)
template <int MB, int NV>
static inline void lp_tile_f16(const float * __restrict X, int64_t ldx, const float16_t * __restrict XH, int ldxh,
                               const uint16_t * __restrict W, int64_t ldw, int K, const float * __restrict bias, bool relu,
                               float * __restrict Y, int64_t ldy, int n_valid) {
    constexpr int KB = 16;
    float32x4_t acc[MB][2 * NV];
    for (int m = 0; m < MB; ++m)
        for (int j = 0; j < 2 * NV; ++j) acc[m][j] = vdupq_n_f32(0.0f);
    for (int k0 = 0; k0 < K; k0 += KB) {
        const int k1 = k0 + KB < K ? k0 + KB : K;
        float16x8_t p[MB][NV];
        for (int m = 0; m < MB; ++m)
            for (int j = 0; j < NV; ++j) p[m][j] = vdupq_n_f16(0);
        for (int k = k0; k < k1; ++k) {
            const float16_t * wr = (const float16_t *) W + (int64_t) k * ldw;
            float16x8_t w[NV];
            for (int j = 0; j < NV; ++j) w[j] = vld1q_f16(wr + 8 * j);
            for (int m = 0; m < MB; ++m) {
                const float16x8_t xv = vld1q_dup_f16(XH + m * ldxh + k);
                for (int j = 0; j < NV; ++j) p[m][j] = vfmaq_f16(p[m][j], w[j], xv);
            }
        }
        for (int m = 0; m < MB; ++m)
            for (int j = 0; j < NV; ++j) {
                acc[m][2 * j]     = vaddq_f32(acc[m][2 * j],     vcvt_f32_f16(vget_low_f16(p[m][j])));
                acc[m][2 * j + 1] = vaddq_f32(acc[m][2 * j + 1], vcvt_high_f32_f16(p[m][j]));
            }
    }
    for (int m = 0; m < MB; ++m) {
        alignas(16) float a[8 * NV];
        for (int j = 0; j < 2 * NV; ++j) vst1q_f32(a + 4 * j, acc[m][j]);
        float * y = Y + m * ldy;
        for (int j = 0; j < n_valid; ++j) {
            float v = a[j];
            if (!std::isfinite(v)) {   // fp16 overflow (or inf/NaN input): this output exactly
                v = 0.0f;
                for (int k = 0; k < K; ++k) v = std::fma(X[m * ldx + k], h2f(W[(int64_t) k * ldw + j]), v);
            }
            v += bias ? bias[j] : 0.0f;
            y[j] = relu ? (v > 0 ? v : 0) : v;
        }
    }
}

template <int MB>
static inline void lp_rows_f16(const float * X, int64_t ldx, const float16_t * XH, int ldxh, const uint16_t * W, int64_t ldw,
                               int K, int N, const float * bias, bool relu, float * Y, int64_t ldy) {
    constexpr int NV = MB == 2 ? 4 : 2;   // 32 / 16 columns
    int n0 = 0;
    for (; n0 + 8 * NV <= N; n0 += 8 * NV)
        lp_tile_f16<MB, NV>(X, ldx, XH, ldxh, W + n0, ldw, K, bias ? bias + n0 : nullptr, relu, Y + n0, ldy, 8 * NV);
    for (; n0 < N; n0 += 16)   // W is zero-padded to a multiple of 32 columns
        lp_tile_f16<MB, 2>(X, ldx, XH, ldxh, W + n0, ldw, K, bias ? bias + n0 : nullptr, relu, Y + n0, ldy, N - n0 < 16 ? N - n0 : 16);
}

// one row: W in the file layout [N][ldr]; more rows: WT packed [K][ldw]
static void gemm_f16_lp(const float * X, int64_t ldx, int M, const uint16_t * W, int64_t ldr, const uint16_t * WT, int64_t ldw,
                        int K, int N, const float * bias, bool relu, float * Y, int64_t ldy) {
    static thread_local std::vector<float16_t> xh;
    const int RB = M == 1 ? 1 : 4;
    if (xh.size() < (size_t) RB * K) xh.resize((size_t) RB * K);
    float16_t * XH = xh.data();
    for (int m = 0; m < M; m += RB) {
        const int mr = M - m < RB ? M - m : RB;
        for (int r = 0; r < mr; ++r) {   // activations to fp16
            const float * x = X + (int64_t) (m + r) * ldx;
            int k = 0;
            for (; k + 4 <= K; k += 4) vst1_f16(XH + r * K + k, vcvt_f16_f32(vld1q_f32(x + k)));
            for (; k < K; ++k) XH[r * K + k] = (float16_t) x[k];
        }
        const float * Xm = X + (int64_t) m * ldx;
        float * Ym = Y + (int64_t) m * ldy;
        if (M == 1)       lp_cols_f16(Xm, XH, W, ldr, K, 0, N, bias, relu, Ym);
        else if (mr == 4) lp_rows_f16<4>(Xm, ldx, XH, K, WT, ldw, K, N, bias, relu, Ym, ldy);
        else if (mr >= 2) {
            lp_rows_f16<2>(Xm, ldx, XH, K, WT, ldw, K, N, bias, relu, Ym, ldy);
            if (mr == 3) lp_cols_f16(Xm + 2 * ldx, XH + 2 * K, W, ldr, K, 0, N, bias, relu, Ym + 2 * ldy);
        } else lp_cols_f16(Xm, XH, W, ldr, K, 0, N, bias, relu, Ym);
    }
}

// Q8_0: X row quantized per 32-block: xq int8, dx = max|x| / 127
static inline void lp_quant_row(const float * x, int K, int8_t * q, float * d) {
    for (int b = 0; b < K / 32; ++b) {
        float32x4_t mx = vdupq_n_f32(0.0f);
        for (int i = 0; i < 32; i += 4) mx = vmaxq_f32(mx, vabsq_f32(vld1q_f32(x + 32 * b + i)));
        const float amax = vmaxvq_f32(mx), dd = amax / 127.0f, id = dd > 0 ? 1.0f / dd : 0.0f;
        d[b] = dd;
        for (int i = 0; i < 32; i += 8) {
            const int32x4_t a = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x + 32 * b + i), id));
            const int32x4_t c = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x + 32 * b + i + 4), id));
            const int16x8_t h = vcombine_s16(vqmovn_s32(a), vqmovn_s32(c));
            vst1_s8(q + 32 * b + i, vqmovn_s16(h));
        }
    }
}

template <int MB, int NV>   // NV int32x4 vectors per row (4 * NV columns)
static inline void lp_tile_q8(const int8_t * __restrict XQ, const float * __restrict XD, int KB, const int8_t * __restrict W4,
                              const float * __restrict WS, int64_t ldw, const float * __restrict bias, bool relu,
                              float * __restrict Y, int64_t ldy, int n_valid) {
    float32x4_t acc[MB][NV];
    for (int m = 0; m < MB; ++m)
        for (int j = 0; j < NV; ++j) acc[m][j] = vdupq_n_f32(0.0f);
    for (int b = 0; b < KB; ++b) {
        int32x4_t s[MB][NV];
        for (int m = 0; m < MB; ++m)
            for (int j = 0; j < NV; ++j) s[m][j] = vdupq_n_s32(0);
        for (int g = 0; g < 8; ++g) {   // 4 k per group
            const int8_t * wg = W4 + ((int64_t) (b * 8 + g) * ldw) * 4;
            int8x16_t w[NV];
            for (int j = 0; j < NV; ++j) w[j] = vld1q_s8(wg + 16 * j);
            for (int m = 0; m < MB; ++m) {
                int32_t xi;
                std::memcpy(&xi, XQ + m * KB * 32 + b * 32 + g * 4, 4);
                const int8x16_t xv = vreinterpretq_s8_s32(vdupq_n_s32(xi));
                for (int j = 0; j < NV; ++j) s[m][j] = vdotq_s32(s[m][j], w[j], xv);
            }
        }
        const float * sc = WS + (int64_t) b * ldw;
        for (int j = 0; j < NV; ++j) {
            const float32x4_t s4 = vld1q_f32(sc + 4 * j);
            for (int m = 0; m < MB; ++m)
                acc[m][j] = vfmaq_f32(acc[m][j], vcvtq_f32_s32(s[m][j]), vmulq_n_f32(s4, XD[m * KB + b]));
        }
    }
    for (int m = 0; m < MB; ++m) {
        alignas(16) float a[4 * NV];
        for (int j = 0; j < NV; ++j) vst1q_f32(a + 4 * j, acc[m][j]);
        float * y = Y + m * ldy;
        for (int j = 0; j < n_valid; ++j) {
            float v = a[j] + (bias ? bias[j] : 0.0f);
            y[j] = relu ? (v > 0 ? v : 0) : v;
        }
    }
}

template <int MB>
static inline void lp_rows_q8(const int8_t * XQ, const float * XD, int KB, const int8_t * W4, const float * WS, int64_t ldw, int N,
                              const float * bias, bool relu, float * Y, int64_t ldy) {
    constexpr int NV = MB == 1 ? 8 : MB == 2 ? 4 : 4;   // 32 / 16 / 16 columns
    int n0 = 0;
    for (; n0 + 4 * NV <= N; n0 += 4 * NV)
        lp_tile_q8<MB, NV>(XQ, XD, KB, W4 + 4 * n0, WS + n0, ldw, bias ? bias + n0 : nullptr, relu, Y + n0, ldy, 4 * NV);
    for (; n0 < N; n0 += 16)
        lp_tile_q8<MB, 4>(XQ, XD, KB, W4 + 4 * n0, WS + n0, ldw, bias ? bias + n0 : nullptr, relu, Y + n0, ldy, N - n0 < 16 ? N - n0 : 16);
}

// One row of X: dot products over the file's block_q8_0 rows ({fp16 d, int8 qs[32]},
// row stride ldr bytes), 4 columns at a time sharing the x loads, so the weights stream
// sequentially (the k-interleaved tile reads them with a large stride).
static void lp_cols_q8(const int8_t * __restrict xq, const float * __restrict xd, int KB, const uint8_t * __restrict W, int64_t ldr,
                       int N, const float * __restrict bias, bool relu, float * __restrict y) {
    auto d_of = [](const uint8_t * blk) { uint16_t h; std::memcpy(&h, blk, 2); return h2f(h); };
    int n = 0;
    for (; n + 4 <= N; n += 4) {
        const uint8_t * r[4] = { W + (int64_t) n * ldr, W + (int64_t) (n + 1) * ldr, W + (int64_t) (n + 2) * ldr, W + (int64_t) (n + 3) * ldr };
        float32x4_t acc[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
        for (int b = 0; b < KB; ++b) {
            const int8x16_t x0 = vld1q_s8(xq + 32 * b), x1 = vld1q_s8(xq + 32 * b + 16);
            for (int c = 0; c < 4; ++c) {
                const uint8_t * blk = r[c] + 34 * b;
                const int32x4_t s = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8((const int8_t *) blk + 2), x0),
                                              vld1q_s8((const int8_t *) blk + 18), x1);
                acc[c] = vfmaq_n_f32(acc[c], vcvtq_f32_s32(s), xd[b] * d_of(blk));
            }
        }
        for (int c = 0; c < 4; ++c) {
            const float v = vaddvq_f32(acc[c]) + (bias ? bias[n + c] : 0.0f);
            y[n + c] = relu ? (v > 0 ? v : 0) : v;
        }
    }
    for (; n < N; ++n) {
        const uint8_t * row = W + (int64_t) n * ldr;
        float s = 0.0f;
        for (int b = 0; b < KB; ++b) {
            int32_t is = 0;
            for (int i = 0; i < 32; ++i) is += (int32_t) xq[32 * b + i] * ((const int8_t *) row + 34 * b + 2)[i];
            s += (float) is * xd[b] * d_of(row + 34 * b);
        }
        const float v = s + (bias ? bias[n] : 0.0f);
        y[n] = relu ? (v > 0 ? v : 0) : v;
    }
}

static void gemm_q8_lp(const float * X, int64_t ldx, int M, const uint8_t * W, int64_t ldr, const int8_t * W4, const float * WS,
                       int64_t ldw, int K, int N, const float * bias, bool relu, float * Y, int64_t ldy) {
    constexpr int RB = 4;
    static thread_local std::vector<int8_t> xq;
    static thread_local std::vector<float> xd;
    if (xq.size() < (size_t) RB * K) { xq.resize((size_t) RB * K); xd.resize((size_t) RB * K / 32 + 1); }
    int8_t * XQ = xq.data();
    float * XD = xd.data();
    const int KB = K / 32;
    for (int m = 0; m < M; m += RB) {
        const int mr = M - m < RB ? M - m : RB;
        for (int r = 0; r < mr; ++r) lp_quant_row(X + (int64_t) (m + r) * ldx, K, XQ + r * K, XD + r * KB);
        float * Ym = Y + (int64_t) m * ldy;
        if (M == 1) { lp_cols_q8(XQ, XD, KB, W, ldr, N, bias, relu, Ym); return; }
        if (mr == 4)      lp_rows_q8<4>(XQ, XD, KB, W4, WS, ldw, N, bias, relu, Ym, ldy);
        else if (mr == 2) lp_rows_q8<2>(XQ, XD, KB, W4, WS, ldw, N, bias, relu, Ym, ldy);
        else for (int r = 0; r < mr; ++r)
            lp_rows_q8<1>(XQ + r * K, XD + r * KB, KB, W4, WS, ldw, N, bias, relu, Ym + r * ldy, ldy);
    }
}
#endif

// x86 AVX-512 VNNI: gemm_q8_lp with activations quantized per 32-block to int8 and
// VPDPBUSD (u8 x s8 -> s32). The weights are the unsigned operand: W4 holds w ^ 0x80
// (= w + 128), so each block sum is corrected by 128 * sum(x block).
#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512F__)
#define VADK_HAS_LP_Q8_VNNI 1

// X row quantized per 32-block: xq int8, xd = max|x| / 127, xs = 128 * sum(xq)
static inline void vnni_quant_row(const float * x, int K, int8_t * q, float * d, int32_t * xs) {
    for (int b = 0; b < K / 32; ++b) {
        const __m512 a0 = _mm512_loadu_ps(x + 32 * b), a1 = _mm512_loadu_ps(x + 32 * b + 16);
        const float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(a0), _mm512_abs_ps(a1)));
        const float dd = amax / 127.0f, id = dd > 0 ? 1.0f / dd : 0.0f;
        d[b] = dd;
        const __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(a0, _mm512_set1_ps(id)));   // round to nearest even
        const __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(a1, _mm512_set1_ps(id)));
        _mm_storeu_si128((__m128i *) (q + 32 * b), _mm512_cvtsepi32_epi8(i0));
        _mm_storeu_si128((__m128i *) (q + 32 * b + 16), _mm512_cvtsepi32_epi8(i1));
        int32_t sum = 0;
        for (int i = 0; i < 32; ++i) sum += q[32 * b + i];
        xs[b] = 128 * sum;
    }
}

template <int MB, int NV>   // NV vectors of 16 columns
static inline void vnni_tile_q8(const int8_t * __restrict XQ, const float * __restrict XD, const int32_t * __restrict XS, int KB,
                                const int8_t * __restrict W4, const float * __restrict WS, int64_t ldw,
                                const float * __restrict bias, bool relu, float * __restrict Y, int64_t ldy, int n_valid) {
    __m512 acc[MB][NV];
    for (int m = 0; m < MB; ++m)
        for (int j = 0; j < NV; ++j) acc[m][j] = _mm512_setzero_ps();
    for (int b = 0; b < KB; ++b) {
        __m512i s[MB][NV];
        for (int m = 0; m < MB; ++m)
            for (int j = 0; j < NV; ++j) s[m][j] = _mm512_set1_epi32(-XS[m * KB + b]);   // start at -128 sum(x)
        for (int g = 0; g < 8; ++g) {   // 4 k per group
            const int8_t * wg = W4 + ((int64_t) (b * 8 + g) * ldw) * 4;
            __m512i w[NV];
            for (int j = 0; j < NV; ++j) w[j] = _mm512_loadu_si512((const void *) (wg + 64 * j));
            for (int m = 0; m < MB; ++m) {
                int32_t xi;
                std::memcpy(&xi, XQ + m * KB * 32 + b * 32 + g * 4, 4);
                const __m512i xv = _mm512_set1_epi32(xi);
                for (int j = 0; j < NV; ++j) s[m][j] = _mm512_dpbusd_epi32(s[m][j], w[j], xv);
            }
        }
        const float * sc = WS + (int64_t) b * ldw;
        for (int j = 0; j < NV; ++j) {
            const __m512 s16 = _mm512_loadu_ps(sc + 16 * j);
            for (int m = 0; m < MB; ++m)
                acc[m][j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s[m][j]), _mm512_mul_ps(s16, _mm512_set1_ps(XD[m * KB + b])), acc[m][j]);
        }
    }
    for (int m = 0; m < MB; ++m) {
        alignas(64) float a[16 * NV];
        for (int j = 0; j < NV; ++j) _mm512_store_ps(a + 16 * j, acc[m][j]);
        float * y = Y + m * ldy;
        for (int j = 0; j < n_valid; ++j) {
            float v = a[j] + (bias ? bias[j] : 0.0f);
            y[j] = relu ? (v > 0 ? v : 0) : v;
        }
    }
}

template <int MB>
static inline void vnni_rows_q8(const int8_t * XQ, const float * XD, const int32_t * XS, int KB, const int8_t * W4, const float * WS,
                                int64_t ldw, int N, const float * bias, bool relu, float * Y, int64_t ldy) {
    for (int n0 = 0; n0 < N; n0 += 32)   // packed columns are zero padded to ldw (a multiple of 32)
        vnni_tile_q8<MB, 2>(XQ, XD, XS, KB, W4 + 4 * n0, WS + n0, ldw, bias ? bias + n0 : nullptr, relu, Y + n0, ldy,
                            N - n0 < 32 ? N - n0 : 32);
}

static void gemm_q8_lp(const float * X, int64_t ldx, int M, const uint8_t *, int64_t, const int8_t * W4, const float * WS,
                       int64_t ldw, int K, int N, const float * bias, bool relu, float * Y, int64_t ldy) {
    constexpr int RB = 4;
    static thread_local std::vector<int8_t> xq;
    static thread_local std::vector<float> xd;
    static thread_local std::vector<int32_t> xs;
    const int KB = K / 32;
    if (xq.size() < (size_t) RB * K) { xq.resize((size_t) RB * K); xd.resize((size_t) RB * KB + 1); xs.resize((size_t) RB * KB + 1); }
    for (int m = 0; m < M; m += RB) {
        const int mr = M - m < RB ? M - m : RB;
        for (int r = 0; r < mr; ++r) vnni_quant_row(X + (int64_t) (m + r) * ldx, K, xq.data() + r * K, xd.data() + r * KB, xs.data() + r * KB);
        float * Ym = Y + (int64_t) m * ldy;
        if (mr == 4)      vnni_rows_q8<4>(xq.data(), xd.data(), xs.data(), KB, W4, WS, ldw, N, bias, relu, Ym, ldy);
        else if (mr == 3) vnni_rows_q8<3>(xq.data(), xd.data(), xs.data(), KB, W4, WS, ldw, N, bias, relu, Ym, ldy);
        else if (mr == 2) vnni_rows_q8<2>(xq.data(), xd.data(), xs.data(), KB, W4, WS, ldw, N, bias, relu, Ym, ldy);
        else              vnni_rows_q8<1>(xq.data(), xd.data(), xs.data(), KB, W4, WS, ldw, N, bias, relu, Ym, ldy);
    }
}
#endif

#ifdef VADK_HAS_GEMM_BIG
#define VADK_BIG gemm_big, 16
#else
#define VADK_BIG nullptr, 0
#endif
#if defined(VADK_HAS_LP)
#define VADK_LP gemm_f16_lp, gemm_q8_lp, false
#elif defined(VADK_HAS_LP_Q8_VNNI)
#define VADK_LP nullptr, gemm_q8_lp, true
#else
#define VADK_LP nullptr, nullptr, false
#endif
extern const vadk::table table = { VADK_NAME, gemm, gemm_f16, gemm_q8, VADK_BIG, unary, binary, VADK_LP };

} // namespace VADK_NS
