#pragma once

#define WSP_GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

// Quantization
void wsp_quantize_row_q4_0_ref(const float * WSP_GGML_RESTRICT x, block_q4_0 * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_q4_1_ref(const float * WSP_GGML_RESTRICT x, block_q4_1 * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_q5_0_ref(const float * WSP_GGML_RESTRICT x, block_q5_0 * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_q5_1_ref(const float * WSP_GGML_RESTRICT x, block_q5_1 * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_q8_0_ref(const float * WSP_GGML_RESTRICT x, block_q8_0 * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_q8_1_ref(const float * WSP_GGML_RESTRICT x, block_q8_1 * WSP_GGML_RESTRICT y, int64_t k);

void wsp_quantize_row_q2_K_ref(const float * WSP_GGML_RESTRICT x, block_q2_K * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_q3_K_ref(const float * WSP_GGML_RESTRICT x, block_q3_K * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_q4_K_ref(const float * WSP_GGML_RESTRICT x, block_q4_K * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_q5_K_ref(const float * WSP_GGML_RESTRICT x, block_q5_K * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_q6_K_ref(const float * WSP_GGML_RESTRICT x, block_q6_K * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_q8_K_ref(const float * WSP_GGML_RESTRICT x, block_q8_K * WSP_GGML_RESTRICT y, int64_t k);

void wsp_quantize_row_tq1_0_ref(const float * WSP_GGML_RESTRICT x, block_tq1_0 * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_tq2_0_ref(const float * WSP_GGML_RESTRICT x, block_tq2_0 * WSP_GGML_RESTRICT y, int64_t k);

void wsp_quantize_row_iq3_xxs_ref(const float * WSP_GGML_RESTRICT x, block_iq3_xxs * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_iq4_nl_ref (const float * WSP_GGML_RESTRICT x, block_iq4_nl  * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_iq4_xs_ref (const float * WSP_GGML_RESTRICT x, block_iq4_xs  * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_iq3_s_ref  (const float * WSP_GGML_RESTRICT x, block_iq3_s   * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_iq2_s_ref  (const float * WSP_GGML_RESTRICT x, block_iq2_s   * WSP_GGML_RESTRICT y, int64_t k);

void wsp_quantize_row_q4_0(const float * WSP_GGML_RESTRICT x, void * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_q4_1(const float * WSP_GGML_RESTRICT x, void * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_q5_0(const float * WSP_GGML_RESTRICT x, void * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_q5_1(const float * WSP_GGML_RESTRICT x, void * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_q8_0(const float * WSP_GGML_RESTRICT x, void * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_q8_1(const float * WSP_GGML_RESTRICT x, void * WSP_GGML_RESTRICT y, int64_t k);

void wsp_quantize_row_q2_K(const float * WSP_GGML_RESTRICT x, void * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_q3_K(const float * WSP_GGML_RESTRICT x, void * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_q4_K(const float * WSP_GGML_RESTRICT x, void * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_q5_K(const float * WSP_GGML_RESTRICT x, void * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_q6_K(const float * WSP_GGML_RESTRICT x, void * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_q8_K(const float * WSP_GGML_RESTRICT x, void * WSP_GGML_RESTRICT y, int64_t k);

void wsp_quantize_row_tq1_0(const float * WSP_GGML_RESTRICT x, void * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_tq2_0(const float * WSP_GGML_RESTRICT x, void * WSP_GGML_RESTRICT y, int64_t k);

void wsp_quantize_row_iq3_xxs(const float * WSP_GGML_RESTRICT x, void * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_iq4_nl (const float * WSP_GGML_RESTRICT x, void * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_iq4_xs (const float * WSP_GGML_RESTRICT x, void * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_iq3_s  (const float * WSP_GGML_RESTRICT x, void * WSP_GGML_RESTRICT y, int64_t k);
void wsp_quantize_row_iq2_s  (const float * WSP_GGML_RESTRICT x, void * WSP_GGML_RESTRICT y, int64_t k);

// Dequantization
void wsp_dewsp_quantize_row_q4_0(const block_q4_0 * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);
void wsp_dewsp_quantize_row_q4_1(const block_q4_1 * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);
void wsp_dewsp_quantize_row_q5_0(const block_q5_0 * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);
void wsp_dewsp_quantize_row_q5_1(const block_q5_1 * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);
void wsp_dewsp_quantize_row_q8_0(const block_q8_0 * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);
//void wsp_dewsp_quantize_row_q8_1(const block_q8_1 * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);

void wsp_dewsp_quantize_row_q2_K(const block_q2_K * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);
void wsp_dewsp_quantize_row_q3_K(const block_q3_K * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);
void wsp_dewsp_quantize_row_q4_K(const block_q4_K * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);
void wsp_dewsp_quantize_row_q5_K(const block_q5_K * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);
void wsp_dewsp_quantize_row_q6_K(const block_q6_K * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);
void wsp_dewsp_quantize_row_q8_K(const block_q8_K * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);

void wsp_dewsp_quantize_row_tq1_0(const block_tq1_0 * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);
void wsp_dewsp_quantize_row_tq2_0(const block_tq2_0 * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);

void wsp_dewsp_quantize_row_iq2_xxs(const block_iq2_xxs * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);
void wsp_dewsp_quantize_row_iq2_xs (const block_iq2_xs  * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);
void wsp_dewsp_quantize_row_iq2_s  (const block_iq2_s   * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);
void wsp_dewsp_quantize_row_iq3_xxs(const block_iq3_xxs * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);
void wsp_dewsp_quantize_row_iq1_s  (const block_iq1_s   * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);
void wsp_dewsp_quantize_row_iq1_m  (const block_iq1_m   * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);
void wsp_dewsp_quantize_row_iq4_nl (const block_iq4_nl  * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);
void wsp_dewsp_quantize_row_iq4_xs (const block_iq4_xs  * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);
void wsp_dewsp_quantize_row_iq3_s  (const block_iq3_s   * WSP_GGML_RESTRICT x, float * WSP_GGML_RESTRICT y, int64_t k);

// Dot product
void wsp_ggml_vec_dot_q4_0_q8_0(int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT vx, size_t bx, const void * WSP_GGML_RESTRICT vy, size_t by, int nrc);
void wsp_ggml_vec_dot_q4_1_q8_1(int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT vx, size_t bx, const void * WSP_GGML_RESTRICT vy, size_t by, int nrc);
void wsp_ggml_vec_dot_q5_0_q8_0(int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT vx, size_t bx, const void * WSP_GGML_RESTRICT vy, size_t by, int nrc);
void wsp_ggml_vec_dot_q5_1_q8_1(int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT vx, size_t bx, const void * WSP_GGML_RESTRICT vy, size_t by, int nrc);
void wsp_ggml_vec_dot_q8_0_q8_0(int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT vx, size_t bx, const void * WSP_GGML_RESTRICT vy, size_t by, int nrc);

void wsp_ggml_vec_dot_q2_K_q8_K(int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT vx, size_t bx, const void * WSP_GGML_RESTRICT vy, size_t by, int nrc);
void wsp_ggml_vec_dot_q3_K_q8_K(int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT vx, size_t bx, const void * WSP_GGML_RESTRICT vy, size_t by, int nrc);
void wsp_ggml_vec_dot_q4_K_q8_K(int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT vx, size_t bx, const void * WSP_GGML_RESTRICT vy, size_t by, int nrc);
void wsp_ggml_vec_dot_q5_K_q8_K(int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT vx, size_t bx, const void * WSP_GGML_RESTRICT vy, size_t by, int nrc);
void wsp_ggml_vec_dot_q6_K_q8_K(int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT vx, size_t bx, const void * WSP_GGML_RESTRICT vy, size_t by, int nrc);

void wsp_ggml_vec_dot_tq1_0_q8_K(int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT vx, size_t bx, const void * WSP_GGML_RESTRICT vy, size_t by, int nrc);
void wsp_ggml_vec_dot_tq2_0_q8_K(int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT vx, size_t bx, const void * WSP_GGML_RESTRICT vy, size_t by, int nrc);

void wsp_ggml_vec_dot_iq2_xxs_q8_K(int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT vx, size_t bx, const void * WSP_GGML_RESTRICT vy, size_t by, int nrc);
void wsp_ggml_vec_dot_iq2_xs_q8_K (int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT vx, size_t bx, const void * WSP_GGML_RESTRICT vy, size_t by, int nrc);
void wsp_ggml_vec_dot_iq2_s_q8_K  (int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT vx, size_t bx, const void * WSP_GGML_RESTRICT vy, size_t by, int nrc);
void wsp_ggml_vec_dot_iq3_xxs_q8_K(int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT vx, size_t bx, const void * WSP_GGML_RESTRICT vy, size_t by, int nrc);
void wsp_ggml_vec_dot_iq1_s_q8_K  (int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT vx, size_t bx, const void * WSP_GGML_RESTRICT vy, size_t by, int nrc);
void wsp_ggml_vec_dot_iq1_m_q8_K  (int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT vx, size_t bx, const void * WSP_GGML_RESTRICT vy, size_t by, int nrc);
void wsp_ggml_vec_dot_iq4_nl_q8_0 (int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT vx, size_t bx, const void * WSP_GGML_RESTRICT vy, size_t by, int nrc);
void wsp_ggml_vec_dot_iq4_xs_q8_K (int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT vx, size_t bx, const void * WSP_GGML_RESTRICT vy, size_t by, int nrc);
void wsp_ggml_vec_dot_iq3_s_q8_K  (int n, float * WSP_GGML_RESTRICT s, size_t bs, const void * WSP_GGML_RESTRICT vx, size_t bx, const void * WSP_GGML_RESTRICT vy, size_t by, int nrc);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
size_t wsp_quantize_iq2_xxs(const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t wsp_quantize_iq2_xs (const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t wsp_quantize_iq2_s  (const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t wsp_quantize_iq3_xxs(const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t wsp_quantize_iq1_s  (const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t wsp_quantize_iq1_m  (const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t wsp_quantize_iq4_nl (const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t wsp_quantize_iq4_xs (const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t wsp_quantize_iq3_s  (const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

size_t wsp_quantize_tq1_0(const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t wsp_quantize_tq2_0(const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

size_t wsp_quantize_q2_K(const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t wsp_quantize_q3_K(const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t wsp_quantize_q4_K(const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t wsp_quantize_q5_K(const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t wsp_quantize_q6_K(const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t wsp_quantize_q4_0(const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t wsp_quantize_q4_1(const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t wsp_quantize_q5_0(const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t wsp_quantize_q5_1(const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t wsp_quantize_q8_0(const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

void iq2xs_init_impl(enum wsp_ggml_type type);
void iq2xs_free_impl(enum wsp_ggml_type type);
void iq3xs_init_impl(int grid_size);
void iq3xs_free_impl(int grid_size);

#ifdef __cplusplus
}
#endif
