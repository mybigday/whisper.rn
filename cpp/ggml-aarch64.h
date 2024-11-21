#pragma once

#include "ggml.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
size_t wsp_quantize_q4_0_4x4(const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t wsp_quantize_q4_0_4x8(const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t wsp_quantize_q4_0_8x8(const float * WSP_GGML_RESTRICT src, void * WSP_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

#ifdef __cplusplus
}
#endif

