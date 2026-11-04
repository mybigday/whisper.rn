// Kernel table of the vad-cpu backend; one implementation per ISA variant,
// selected at runtime (x86) or at build time (other targets).
#pragma once

#include <cstdint>

namespace vadk {

enum class UOp { RELU, SIGMOID, TANH, SQRT, NEG, ABS, LOG, GELU_ERF, EXP };
enum class BOp { ADD, SUB, MUL, DIV };

// packed weights [K][ldw] of one type (see gemm, gemm_f16, gemm_q8)
struct weights {
    const float * f32 = nullptr;
    const uint16_t * f16 = nullptr;
    const int8_t * q8 = nullptr;
    const float * qs = nullptr;
    int64_t ldw = 0;
};

struct table {
    const char * name;
    // Y[m][n] = bias[n] + sum_k X[m][k] * WT[k][n] (+relu); WT packed [K][ldw], ldw % 32 == 0, zero padded
    void (*gemm)(const float * X, int64_t ldx, int M, const float * WT, int64_t ldw, int K, int N,
                 const float * bias, bool relu, float * Y, int64_t ldy);
    // same with WT stored as IEEE half (bit patterns)
    void (*gemm_f16)(const float * X, int64_t ldx, int M, const uint16_t * WT, int64_t ldw, int K, int N,
                     const float * bias, bool relu, float * Y, int64_t ldy);
    // int8 weights: WQ [K][ldw], WS [K/32][ldw] per-(32-block, column) scales, K % 32 == 0
    void (*gemm_q8)(const float * X, int64_t ldx, int M, const int8_t * WQ, const float * WS, int64_t ldw, int K, int N,
                    const float * bias, bool relu, float * Y, int64_t ldy);
    // Many-row GEMM (M >= gemm_big_min_rows): cache-blocked, register-tiled. Same
    // contract as gemm / gemm_f16 / gemm_q8, with the weights given by w (exactly one
    // of f32, f16 or q8 + qs set, already offset to the first column).
    void (*gemm_big)(const float * X, int64_t ldx, int M, const struct weights & w, int K, int N,
                     const float * bias, bool relu, float * Y, int64_t ldy);
    int gemm_big_min_rows;   // 0: gemm_big not available (callers use gemm*)
    void (*unary)(UOp op, const float * x, float * y, int n);
    // y = a op b, b a full row (bn == n) or a broadcast scalar (bn == 1)
    void (*binary)(BOp op, const float * a, const float * b, int bn, float * y, int n);
    // Reduced precision (null where unsupported; ARMv8.2 fp16 + dotprod, x86 AVX-512 VNNI for Q8_0):
    // like gemm_f16 with activations rounded to fp16 and summed in fp16 over short blocks
    // (outputs that overflow are recomputed exactly). Needs the weights both in the file
    // layout W[n][k] (row stride ldr; used for one row of X) and packed WT [K][ldw].
    void (*gemm_f16_lp)(const float * X, int64_t ldx, int M, const uint16_t * W, int64_t ldr, const uint16_t * WT, int64_t ldw,
                        int K, int N, const float * bias, bool relu, float * Y, int64_t ldy);
    // like gemm_q8 with activations quantized per 32-block to int8. Needs the weights
    // both as the file's block_q8_0 rows W (row stride ldr bytes; used for one row of X)
    // and packed k-interleaved W4[k / 4][ldw][4] + scales WS (see pack_q8_interleaved)
    void (*gemm_q8_lp)(const float * X, int64_t ldx, int M, const uint8_t * W, int64_t ldr, const int8_t * W4, const float * WS,
                       int64_t ldw, int K, int N, const float * bias, bool relu, float * Y, int64_t ldy);
    // W4 for gemm_q8_lp holds w ^ 0x80 (x86 VNNI: the weights are the unsigned operand)
    bool q8_lp_unsigned;
};

// [K][ldw] int8 -> k-interleaved [K / 4][ldw][4] (K % 4 == 0) for gemm_q8_lp; flip: store w ^ 0x80
inline void pack_q8_interleaved(const int8_t * src, int64_t K, int64_t ldw, int8_t * dst, bool flip = false) {
    for (int64_t k = 0; k < K; ++k)
        for (int64_t n = 0; n < ldw; ++n)
            dst[((k / 4) * ldw + n) * 4 + k % 4] = flip ? (int8_t) ((uint8_t) src[k * ldw + n] ^ 0x80) : src[k * ldw + n];
}

const table & get();   // best variant for this CPU

// all variants this CPU can run (best last); returns the count
int available(const table ** out, int max);

} // namespace vadk
