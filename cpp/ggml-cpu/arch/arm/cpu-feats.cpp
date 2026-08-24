#include "ggml-backend-impl.h"
#include "ggml-feats.h"

#if defined(__aarch64__) || defined(_M_ARM64)

static int wsp_ggml_backend_cpu_aarch64_score() {
    int score = 1;
    const wsp_ggml_feats_arch64_runtime_t af = wsp_ggml_feats_get_arch64_runtime();
    WSP_GGML_UNUSED(af);

#ifdef WSP_GGML_USE_DOTPROD
    if (!af.has_dotprod) { return 0; }
    score += 1<<1;
#endif
#ifdef WSP_GGML_USE_FP16_VECTOR_ARITHMETIC
    if (!af.has_fp16) { return 0; }
    score += 1<<2;
#endif
#ifdef WSP_GGML_USE_SVE
    if (!af.has_sve) { return 0; }
    score += 1<<3;
#endif
#ifdef WSP_GGML_USE_MATMUL_INT8
    if (!af.has_i8mm) { return 0; }
    score += 1<<4;
#endif
#ifdef WSP_GGML_USE_SVE2
    if (!af.has_sve2) { return 0; }
    score += 1<<5;
#endif
#ifdef WSP_GGML_USE_SME
    if (!af.has_sme) { return 0; }
    score += 1<<6;
#endif

    return score;
}

WSP_GGML_BACKEND_DL_SCORE_IMPL(wsp_ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
