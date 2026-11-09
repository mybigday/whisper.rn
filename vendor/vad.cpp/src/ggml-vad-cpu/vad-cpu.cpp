#include "vad-cpu.h"
#include "kernels.h"
#include "thread_pool.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace {

const vadk::table & kern() { static const vadk::table & t = vadk::get(); return t; }

enum class StepKind { GEMM, UNARY, BINARY, COPY, COPY_STRIDED, CONCAT, PAD, PAD_REFLECT, CLAMP, SSM_CONV, FSMN, NORM, SCALE, SOFT_MAX, ROPE, CUSTOM, DELEGATE };

struct Step {
    StepKind kind;
    ggml_tensor * dst = nullptr;
    ggml_tensor * a = nullptr;      // GEMM: weight (src0); others: src0
    ggml_tensor * b = nullptr;      // GEMM: activations (src1); BINARY: src1
    ggml_tensor * bias = nullptr;   // GEMM
    bool relu = false;              // GEMM
    const float * packed = nullptr; // GEMM: packed weight [K][ldw] (null: pack per run)
    const uint16_t * packed16 = nullptr;   // GEMM: packed F16 weight [K][ldw] (F16 weights)
    const int8_t * packedq = nullptr;      // GEMM: packed Q8_0 weight [K][ldw] + scales [K/32][ldw]
    const float * packeds = nullptr;
    bool q8_interleaved = false;           // packedq is k-interleaved for gemm_q8_lp
    int64_t ldw = 0;
    vadk::UOp uop{};
    vadk::BOp bop{};
    int i0 = 0, i1 = 0;             // DELEGATE: node range
    ggml_tensor * w = nullptr;      // FSMN: conv weight [K, P]
    ggml_tensor * cache_out = nullptr;   // FSMN: CPY node writing the new cache
};

struct Plan {
    uint64_t sig = 0;
    std::vector<Step> steps;
    ggml_vad_cpu_plan_stats stats{};
};

struct Context {
    ggml_backend_t cpu = nullptr;
    std::unique_ptr<vadk::thread_pool> pool;   // null when single-threaded
    std::unordered_map<const ggml_cgraph *, Plan> plans;
    std::unordered_map<const ggml_tensor *, std::vector<float>> packed;   // weight -> [K][ldw]
    std::unordered_map<const ggml_tensor *, std::vector<uint16_t>> packed16;
    std::unordered_map<const ggml_tensor *, std::pair<std::vector<int8_t>, std::vector<float>>> packedq;
    std::unordered_map<const ggml_tensor *, std::vector<int8_t>> packedq4;   // k-interleaved (gemm_q8_lp)
    std::vector<float> scratch, scratch_w, scratch_x;
    std::vector<float> rope_cs;                       // ROPE: [n_pos][n_dims / 2][cos, sin], reused while the key matches
    struct { const void * pos = nullptr; int64_t n_pos = 0, n_dims = 0; float base = 0; uint64_t compute = 0; } rope_key;
    const ggml_tensor * mask_src = nullptr;           // SOFT_MAX: F16 mask converted once per compute
    const void * mask_data = nullptr;
    uint64_t mask_compute = 0, n_compute = 0;
    std::vector<float> mask_f32;
    ggml_vad_cpu_plan_stats last{};
};

bool is_f32_rows(const ggml_tensor * t) { return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float); }

// plain rope (NORMAL or NEOX, one position per i2, no YaRN or frequency factors)
bool rope_supported(const ggml_tensor * t) {
    const int32_t * p = (const int32_t *) t->op_params;
    const int n_dims = p[1], mode = p[2];
    float freq_scale, ext_factor, attn_factor;
    std::memcpy(&freq_scale, p + 6, 4); std::memcpy(&ext_factor, p + 7, 4); std::memcpy(&attn_factor, p + 8, 4);
    const ggml_tensor * a = t->src[0], * pos = t->src[1];
    return (mode == 0 || mode == GGML_ROPE_TYPE_NEOX) && !t->src[2] && freq_scale == 1.0f && ext_factor == 0.0f &&
           attn_factor == 1.0f && n_dims > 0 && n_dims % 2 == 0 && n_dims <= a->ne[0] && is_f32_rows(a) && is_f32_rows(t) &&
           ggml_are_same_shape(t, a) && pos->type == GGML_TYPE_I32 && ggml_is_contiguous(pos) && pos->ne[0] == a->ne[2];
}

bool is_noop(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_NONE: case GGML_OP_VIEW: case GGML_OP_RESHAPE: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

const ggml_tensor * base_of(const ggml_tensor * t) {
    while (t->view_src) t = t->view_src;
    return t;
}

bool is_weight(const ggml_tensor * t) {
    return t->op == GGML_OP_NONE && t->view_src == nullptr && t->buffer &&
           ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

uint64_t graph_signature(const ggml_cgraph * g) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    const int n = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(g));
    mix(n);
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * t = ggml_graph_node(const_cast<ggml_cgraph *>(g), i);
        mix((uint64_t) (uintptr_t) t); mix(t->op); mix((uint64_t) (uintptr_t) t->data);
        mix((uint64_t) (uintptr_t) t->src[0]); mix((uint64_t) (uintptr_t) t->src[1]);
    }
    return h;
}

// ------------------------------------------------------------- plan building

// src0 either a 2D matrix shared by every batch, or one matrix per batch (same
// batch dims as src1, e.g. attention scores); rows of src0 must be contiguous
bool is_weight(const ggml_tensor * t);

bool can_gemm(const ggml_tensor * mm) {
    const ggml_tensor * a = mm->src[0], * b = mm->src[1];
    if (a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_Q8_0)   // reduced-precision weights: packed once, 2D only
        return is_weight(a) && ggml_is_contiguous(a) && a->ne[2] == 1 && a->ne[3] == 1 && is_f32_rows(b) && is_f32_rows(mm);
    if (a->type != GGML_TYPE_F32 || a->nb[0] != sizeof(float) || !is_f32_rows(b) || !is_f32_rows(mm)) return false;
    const bool shared = a->ne[2] == 1 && a->ne[3] == 1;
    return shared || (a->ne[2] == b->ne[2] && a->ne[3] == b->ne[3]);
}

bool is_bias_for(const ggml_tensor * bias, int64_t n) {
    return bias->type == GGML_TYPE_F32 && ggml_is_contiguous(bias) && bias->ne[0] == n &&
           bias->ne[1] == 1 && bias->ne[2] == 1 && bias->ne[3] == 1;
}

bool unary_of(const ggml_tensor * t, vadk::UOp & op) {
    if (t->op == GGML_OP_SQRT) { op = vadk::UOp::SQRT; return true; }
    if (t->op == GGML_OP_LOG)  { op = vadk::UOp::LOG;  return true; }
    if (t->op != GGML_OP_UNARY) return false;
    switch (ggml_get_unary_op(t)) {
        case GGML_UNARY_OP_RELU:    op = vadk::UOp::RELU;    return true;
        case GGML_UNARY_OP_SIGMOID: op = vadk::UOp::SIGMOID; return true;
        case GGML_UNARY_OP_TANH:    op = vadk::UOp::TANH;    return true;
        case GGML_UNARY_OP_NEG:     op = vadk::UOp::NEG;     return true;
        case GGML_UNARY_OP_ABS:     op = vadk::UOp::ABS;     return true;
        case GGML_UNARY_OP_GELU_ERF: op = vadk::UOp::GELU_ERF; return true;
        default: return false;
    }
}

bool binary_of(const ggml_tensor * t, vadk::BOp & op) {
    switch (t->op) {
        case GGML_OP_ADD: op = vadk::BOp::ADD; break;
        case GGML_OP_SUB: op = vadk::BOp::SUB; break;
        case GGML_OP_MUL: op = vadk::BOp::MUL; break;
        case GGML_OP_DIV: op = vadk::BOp::DIV; break;
        default: return false;
    }
    const ggml_tensor * a = t->src[0], * b = t->src[1];
    return is_f32_rows(a) && is_f32_rows(b) && is_f32_rows(t) && ggml_are_same_shape(a, t) &&
           (b->ne[0] == t->ne[0] || b->ne[0] == 1) && ggml_can_repeat(b, t);
}

const float * pack_weight(Context & ctx, const ggml_tensor * a, int64_t & ldw) {
    const int64_t K = a->ne[0], N = a->ne[1];
    ldw = (N + 31) / 32 * 32;
    auto it = ctx.packed.find(a);
    if (it != ctx.packed.end()) return it->second.data();
    std::vector<float> & w = ctx.packed[a];
    w.assign((size_t) K * ldw, 0.0f);
    const float * src = (const float *) a->data;
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) w[(size_t) k * ldw + n] = src[n * K + k];
    return w.data();
}

// F16 weight [K, N] -> [K][ldw] half bit patterns, transposed, zero padded
const uint16_t * pack_weight16(Context & ctx, const ggml_tensor * a, int64_t & ldw) {
    const int64_t K = a->ne[0], N = a->ne[1];
    ldw = (N + 31) / 32 * 32;
    auto it = ctx.packed16.find(a);
    if (it != ctx.packed16.end()) return it->second.data();
    std::vector<uint16_t> & w = ctx.packed16[a];
    w.assign((size_t) K * ldw, 0);
    const uint16_t * src = (const uint16_t *) a->data;
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) w[(size_t) k * ldw + n] = src[n * K + k];
    return w.data();
}

// Q8_0 weight [K, N] (blocks of 32 along K) -> int8 [K][ldw] + scales [K/32][ldw]
void pack_weight_q8(Context & ctx, const ggml_tensor * a, int64_t & ldw, const int8_t *& q, const float *& sc) {
    const int64_t K = a->ne[0], N = a->ne[1], KB = K / 32;
    ldw = (N + 31) / 32 * 32;
    auto it = ctx.packedq.find(a);
    if (it == ctx.packedq.end()) {
        auto & e = ctx.packedq[a];
        e.first.assign((size_t) K * ldw, 0);
        e.second.assign((size_t) KB * ldw, 0.0f);
        struct blk { ggml_fp16_t d; int8_t qs[32]; };
        static_assert(sizeof(blk) == 34, "block_q8_0 layout");
        for (int64_t n = 0; n < N; ++n) {
            const blk * row = (const blk *) ((const char *) a->data + n * a->nb[1]);
            for (int64_t b = 0; b < KB; ++b) {
                e.second[(size_t) b * ldw + n] = ggml_fp16_to_fp32(row[b].d);
                for (int i = 0; i < 32; ++i) e.first[(size_t) (b * 32 + i) * ldw + n] = row[b].qs[i];
            }
        }
        it = ctx.packedq.find(a);
    }
    q = it->second.first.data();
    sc = it->second.second.data();
}

Plan build_plan(Context & ctx, ggml_cgraph * g) {
    Plan plan;
    const int n = ggml_graph_n_nodes(g);
    plan.stats.n_nodes = n;

    // number of consumers of every (base) tensor; graph outputs are pinned
    std::unordered_map<const ggml_tensor *, int> uses;
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * t = ggml_graph_node(g, i);
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) uses[base_of(t->src[s])]++;
        if (t->flags & GGML_TENSOR_FLAG_OUTPUT) uses[t] += 1000;
    }
    auto single_use = [&](const ggml_tensor * t) { auto it = uses.find(t); return it != uses.end() && it->second == 1; };

    // Streaming causal depthwise conv (FSMN / short conv) as built by the models:
    //   sx  = CONCAT(cache [C, P, B], CONT(PERMUTE(q [P, T, B])), dim 0)   [C+T, P, B]
    //   y   = SSM_CONV(sx, w [C+1, P])                                    [P, T, B]
    //   CPY(VIEW(sx, last C frames), cache region)
    // runs as one FSMN step: no tiny per-row copies, cache updated in place.
    struct fsmn_match { ggml_tensor * cache, * q, * cpy; };
    std::unordered_map<const ggml_tensor *, fsmn_match> fsmn;
    std::unordered_map<const ggml_tensor *, bool> absorbed;
    for (int i = 0; i < n; ++i) {
        ggml_tensor * t = ggml_graph_node(g, i);
        if (t->op != GGML_OP_SSM_CONV) continue;
        ggml_tensor * sx = t->src[0], * w = t->src[1];
        if (sx->op != GGML_OP_CONCAT || ggml_get_op_params_i32(sx, 0) != 0 || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w)) continue;
        ggml_tensor * cache = sx->src[0], * bp = sx->src[1];
        if (!(bp->op == GGML_OP_CONT || bp->op == GGML_OP_CPY || bp->op == GGML_OP_DUP) || bp->src[0]->op != GGML_OP_PERMUTE) continue;
        ggml_tensor * q = bp->src[0]->src[0];
        const int64_t C = cache->ne[0], P = q->ne[0], T = q->ne[1];
        if (!is_f32_rows(q) || !is_f32_rows(cache) || !is_f32_rows(t) || bp->ne[0] != T || bp->ne[1] != P ||
            cache->ne[1] != P || cache->ne[2] != q->ne[2] || cache->ne[3] != q->ne[3] || w->ne[0] != C + 1 || w->ne[1] != P) continue;
        ggml_tensor * cpy = nullptr;
        // the cache write may come before or after the conv in node order (it only depends on sx)
        for (int j = 0; j < n && !cpy; ++j) {
            ggml_tensor * u = ggml_graph_node(g, j);
            if (u->op == GGML_OP_CPY && u->src[0]->view_src == sx && u->src[0]->view_offs == (size_t) T * sizeof(float) &&
                ggml_are_same_shape(u, cache) && is_f32_rows(u)) cpy = u;
        }
        if (!cpy || !single_use(bp) || uses[sx] != 3) continue;   // users: ssm_conv, tail view, cpy
        fsmn[t] = { cache, q, cpy };
        absorbed[sx] = absorbed[bp] = absorbed[cpy] = true;
    }

    int deleg_start = -1;
    auto flush = [&](int end) {
        if (deleg_start >= 0) {
            Step s; s.kind = StepKind::DELEGATE; s.i0 = deleg_start; s.i1 = end;
            plan.steps.push_back(s);
            deleg_start = -1;
        }
    };

    for (int i = 0; i < n; ++i) {
        ggml_tensor * t = ggml_graph_node(g, i);
        if (is_noop(t)) continue;   // views: nothing to compute (also inside delegated ranges)
        if (absorbed.count(t)) { plan.stats.n_fused++; continue; }

        Step s;
        bool ok = false;
        vadk::UOp uop; vadk::BOp bop;

        if (fsmn.count(t)) {
            const fsmn_match & f = fsmn[t];
            s.kind = StepKind::FSMN; s.a = f.cache; s.b = f.q; s.w = t->src[1]; s.cache_out = f.cpy; s.dst = t; ok = true;
        } else if (t->op == GGML_OP_MUL_MAT && can_gemm(t)) {
            s.kind = StepKind::GEMM; s.a = t->src[0]; s.b = t->src[1]; s.dst = t;
            if (s.a->type == GGML_TYPE_F16) s.packed16 = pack_weight16(ctx, s.a, s.ldw);
            else if (s.a->type == GGML_TYPE_Q8_0) {
                pack_weight_q8(ctx, s.a, s.ldw, s.packedq, s.packeds);
                if (kern().gemm_q8_lp) {
                    auto & q4 = ctx.packedq4[s.a];
                    if (q4.empty()) {
                        q4.resize((size_t) s.a->ne[0] * s.ldw);
                        vadk::pack_q8_interleaved(s.packedq, s.a->ne[0], s.ldw, q4.data(), kern().q8_lp_unsigned);
                    }
                    s.packedq = q4.data();
                    s.q8_interleaved = true;
                }
            }
            else if (is_weight(s.a) && ggml_is_contiguous(s.a) && s.a->ne[2] == 1 && s.a->ne[3] == 1) s.packed = pack_weight(ctx, s.a, s.ldw);
            // fuse  mm -> add(bias) -> relu
            int j = i + 1;
            if (j < n && single_use(t) && t->src[0]->ne[2] == 1 && t->src[0]->ne[3] == 1) {
                ggml_tensor * add = ggml_graph_node(g, j);
                if (add->op == GGML_OP_ADD && is_f32_rows(add) && ggml_are_same_shape(add, t)) {
                    ggml_tensor * other = add->src[0] == t ? add->src[1] : (add->src[1] == t ? add->src[0] : nullptr);
                    if (other && is_bias_for(other, t->ne[0])) {
                        s.bias = other; s.dst = add; i = j; plan.stats.n_fused++;
                        ++j;
                        if (j < n && single_use(add)) {
                            ggml_tensor * r = ggml_graph_node(g, j);
                            if (r->op == GGML_OP_UNARY && ggml_get_unary_op(r) == GGML_UNARY_OP_RELU &&
                                r->src[0] == add && is_f32_rows(r)) {
                                s.relu = true; s.dst = r; i = j; plan.stats.n_fused++;
                            }
                        }
                    }
                }
            }
            ok = true;
        } else if (unary_of(t, uop) && is_f32_rows(t->src[0]) && is_f32_rows(t) && ggml_are_same_shape(t, t->src[0])) {
            s.kind = StepKind::UNARY; s.uop = uop; s.a = t->src[0]; s.dst = t; ok = true;
        } else if (binary_of(t, bop)) {
            s.kind = StepKind::BINARY; s.bop = bop; s.a = t->src[0]; s.b = t->src[1]; s.dst = t; ok = true;
        } else if ((t->op == GGML_OP_CPY || t->op == GGML_OP_DUP || t->op == GGML_OP_CONT) &&
                   is_f32_rows(t->src[0]) && is_f32_rows(t) && t->src[0]->ne[0] == t->ne[0] &&
                   ggml_nrows(t->src[0]) == ggml_nrows(t)) {
            s.kind = StepKind::COPY; s.a = t->src[0]; s.dst = t; ok = true;
        } else if ((t->op == GGML_OP_CPY || t->op == GGML_OP_DUP || t->op == GGML_OP_CONT) &&
                   t->src[0]->type == GGML_TYPE_F32 && t->type == GGML_TYPE_F32 && ggml_nelements(t->src[0]) == ggml_nelements(t)) {
            s.kind = StepKind::COPY_STRIDED; s.a = t->src[0]; s.dst = t; ok = true;
        } else if (t->op == GGML_OP_CLAMP && is_f32_rows(t->src[0]) && is_f32_rows(t) && ggml_are_same_shape(t, t->src[0])) {
            s.kind = StepKind::CLAMP; s.a = t->src[0]; s.dst = t; ok = true;
        } else if (t->op == GGML_OP_SSM_CONV && is_f32_rows(t->src[0]) && t->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(t->src[1]) && is_f32_rows(t)) {
            s.kind = StepKind::SSM_CONV; s.a = t->src[0]; s.b = t->src[1]; s.dst = t; ok = true;
        } else if (t->op == GGML_OP_CUSTOM) {   // ggml_custom_4d: the op's own function on vad-cpu's threads
            s.kind = StepKind::CUSTOM; s.dst = t; ok = true;
        } else if (t->op == GGML_OP_NORM && is_f32_rows(t->src[0]) && is_f32_rows(t) && ggml_are_same_shape(t, t->src[0])) {
            s.kind = StepKind::NORM; s.a = t->src[0]; s.dst = t; ok = true;
        } else if (t->op == GGML_OP_SCALE && is_f32_rows(t->src[0]) && is_f32_rows(t) && ggml_are_same_shape(t, t->src[0])) {
            s.kind = StepKind::SCALE; s.a = t->src[0]; s.dst = t; ok = true;
        } else if (t->op == GGML_OP_SOFT_MAX && !t->src[2] && is_f32_rows(t->src[0]) && is_f32_rows(t) &&
                   ggml_are_same_shape(t, t->src[0]) && ((const float *) t->op_params)[1] == 0.0f &&
                   (!t->src[1] || ((t->src[1]->type == GGML_TYPE_F32 || t->src[1]->type == GGML_TYPE_F16) &&
                                   ggml_is_contiguous(t->src[1]) && t->src[1]->ne[0] == t->ne[0] && t->src[1]->ne[1] >= t->ne[1]))) {
            s.kind = StepKind::SOFT_MAX; s.a = t->src[0]; s.b = t->src[1]; s.dst = t; ok = true;
        } else if (t->op == GGML_OP_ROPE && rope_supported(t)) {
            s.kind = StepKind::ROPE; s.a = t->src[0]; s.b = t->src[1]; s.dst = t; ok = true;
        } else if (t->op == GGML_OP_CONCAT && is_f32_rows(t->src[0]) && is_f32_rows(t->src[1]) && is_f32_rows(t)) {
            s.kind = StepKind::CONCAT; s.a = t->src[0]; s.b = t->src[1]; s.dst = t; ok = true;
        } else if (t->op == GGML_OP_PAD && is_f32_rows(t->src[0]) && is_f32_rows(t) && ggml_get_op_params_i32(t, 8) == 0) {
            s.kind = StepKind::PAD; s.a = t->src[0]; s.dst = t; ok = true;
        } else if (t->op == GGML_OP_PAD_REFLECT_1D && is_f32_rows(t->src[0]) && is_f32_rows(t)) {
            s.kind = StepKind::PAD_REFLECT; s.a = t->src[0]; s.dst = t; ok = true;
        }

        if (ok) {
            flush(i);
            plan.steps.push_back(s);
        } else {
            if (deleg_start < 0) deleg_start = i;
            plan.stats.n_delegated++;
        }
    }
    flush(n);
    plan.stats.n_steps = (int) plan.steps.size();
    return plan;
}

// ------------------------------------------------------------- execution

// work below this many elements runs on the calling thread (dispatch costs ~1 us)
constexpr int64_t PAR_MIN_ELEMS = 1 << 15;
constexpr double  PAR_MIN_FLOPS = 1 << 16;
#if defined(__aarch64__)
// Cortex-A cores pay more per dispatch (slower cross-core wake-up and cache-line
// transfers), and big.LITTLE parts mix fast and slow cores: split few-row GEMMs only
// when there is real work, and in 2 chunks per thread so fast cores can steal.
constexpr double  PAR_MIN_WBYTES = 1 << 19;
constexpr double  PAR_FEW_FLOPS = 1 << 20;
constexpr int     GEMV_TASKS_PER_THREAD = 2;
constexpr int     BIG_MIN_TASKS_PER_THREAD = 4;   // small many-row GEMMs too: slow cores must not hold a whole share
#else
// few-row GEMMs: a 128 KB matrix (~1.2 us on one Zen 5 core) loses more to the
// dispatch than it gains from a split; 256 KB+ (e.g. an LSTM step) pays off
constexpr double  PAR_MIN_WBYTES = 3 << 16;
constexpr double  PAR_FEW_FLOPS = 1 << 17;
constexpr int     GEMV_TASKS_PER_THREAD = 1;
constexpr int     BIG_MIN_TASKS_PER_THREAD = 1;
#endif

// f(i1, i2, i3) for every row of t, split across the pool when large enough
template <typename F>
void for_rows(Context & ctx, const ggml_tensor * t, F && f) {
    const int64_t n1 = t->ne[1], n2 = t->ne[2], R = n1 * n2 * t->ne[3];
    auto range = [&](int64_t r0, int64_t r1) {
        if (r0 >= r1) return;
        int64_t i1 = r0 % n1, i2 = (r0 / n1) % n2, i3 = r0 / (n1 * n2);   // divide once per range
        for (int64_t r = r0; r < r1; ++r) {
            f(i1, i2, i3);
            if (++i1 == n1) { i1 = 0; if (++i2 == n2) { i2 = 0; ++i3; } }
        }
    };
    const int nt = ctx.pool ? ctx.pool->size() : 1;
    if (nt == 1 || R < 2 || R * t->ne[0] < PAR_MIN_ELEMS) { range(0, R); return; }
    const int n_tasks = (int) std::min<int64_t>(R, nt * 4);
    ctx.pool->run(n_tasks, [&](int i) { range(R * i / n_tasks, R * (i + 1) / n_tasks); });
}

inline float * row(const ggml_tensor * t, int64_t i1, int64_t i2, int64_t i3) {
    return (float *) ((char *) t->data + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
}

bool overlaps(const ggml_tensor * x, const ggml_tensor * y) {
    const char * x0 = (const char *) x->data, * x1 = x0 + ggml_nbytes(x);
    const char * y0 = (const char *) y->data, * y1 = y0 + ggml_nbytes(y);
    return x0 < y1 && y0 < x1;
}

// src0 rows (any strides, nb0 == 4) of batch (i2, i3) -> WT [K][ldw], zero padded
void pack_rows(const ggml_tensor * a, int64_t i2, int64_t i3, int64_t ldw, float * WT) {
    const int64_t K = a->ne[0], N = a->ne[1];
    for (int64_t k = 0; k < K; ++k) std::fill(WT + k * ldw + N, WT + (k + 1) * ldw, 0.0f);
    for (int64_t nn = 0; nn < N; ++nn) {
        const float * src = row(a, nn, i2, i3);
        for (int64_t k = 0; k < K; ++k) WT[k * ldw + nn] = src[k];
    }
}

// one src0 matrix per batch (attention): pack each, then (batch x row block) tasks
// Many-row GEMM through kern().gemm_big, tiled over (row block x column block)
// so each task's X block stays in L2 and no task streams the whole weight matrix.
bool use_big(int64_t M) { return kern().gemm_big && M >= kern().gemm_big_min_rows; }

void run_big(Context & ctx, const float * X, int64_t ldx, int64_t M, const vadk::weights & w, int K, int N,
             const float * bias, bool relu, float * Y, int64_t ldy) {
    const int nt = ctx.pool ? ctx.pool->size() : 1;
    if (nt == 1 || 2.0 * M * N * K < PAR_MIN_FLOPS) {
        kern().gemm_big(X, ldx, (int) M, w, K, N, bias, relu, Y, ldy);
        return;
    }
    // ~4 tasks per thread: row blocks of 48..192 rows, column blocks multiples of 64;
    // small matrices (e.g. 64 x 128) go down to 24 rows x 32 columns so every thread has work
    int64_t mb = 192, nb = (N + 63) / 64 * 64;
    auto tasks = [&] { return ((M + mb - 1) / mb) * ((N + nb - 1) / nb); };
    while (tasks() < 4 * nt && nb > 128) nb = (nb / 2 + 63) / 64 * 64;
    while (tasks() < 4 * nt && mb > 48) mb /= 2;
    while (tasks() < BIG_MIN_TASKS_PER_THREAD * nt && nb > 32) nb = (nb / 2 + 31) / 32 * 32;
    while (tasks() < BIG_MIN_TASKS_PER_THREAD * nt && mb > 24) mb /= 2;
    const int64_t n_m = (M + mb - 1) / mb, n_n = (N + nb - 1) / nb;
    ctx.pool->run((int) (n_m * n_n), [&](int i) {
        const int64_t im = i % n_m, in = i / n_m;
        const int64_t m0 = im * mb, n0 = in * nb, m1 = std::min(M, m0 + mb), n1 = std::min<int64_t>(N, n0 + nb);
        vadk::weights wt = w;
        if (wt.f32) wt.f32 += n0;
        if (wt.f16) wt.f16 += n0;
        if (wt.q8) { wt.q8 += n0; wt.qs += n0; }
        kern().gemm_big(X + m0 * ldx, ldx, (int) (m1 - m0), wt, K, (int) (n1 - n0), bias ? bias + n0 : nullptr, relu,
                        Y + m0 * ldy + n0, ldy);
    });
}

void run_gemm_batched(Context & ctx, const Step & s) {
    const ggml_tensor * a = s.a, * b = s.b, * d = s.dst;
    const int64_t K = a->ne[0], N = a->ne[1], M = b->ne[1], NB = a->ne[2] * a->ne[3];
    const int64_t ldw = (N + 31) / 32 * 32;
    ctx.scratch_w.resize((size_t) NB * K * ldw);
    float * WT = ctx.scratch_w.data();
    const int nt = ctx.pool ? ctx.pool->size() : 1;
    auto pack = [&](int64_t bi) { pack_rows(a, bi % a->ne[2], bi / a->ne[2], ldw, WT + bi * K * ldw); };
    if (nt == 1) for (int64_t bi = 0; bi < NB; ++bi) pack(bi);
    else ctx.pool->run((int) NB, [&](int bi) { pack(bi); });
    const bool alias = overlaps(d, b);
    const int64_t ldx = b->nb[1] / sizeof(float), ldy = d->nb[1] / sizeof(float);
    if (!alias && use_big(M)) {
        // one (batch, row block) per task, each over all columns
        const int64_t mbb = std::max<int64_t>(48, std::min<int64_t>(192, (M * NB + nt * 4 - 1) / (nt * 4)));
        const int64_t per_b = (M + mbb - 1) / mbb;
        auto task_b = [&](int i) {
            const int64_t bi = i / per_b, m0 = (i % per_b) * mbb, m1 = std::min(M, m0 + mbb);
            const int64_t i2 = bi % b->ne[2], i3 = bi / b->ne[2];
            vadk::weights w;
            w.f32 = WT + bi * K * ldw;
            w.ldw = ldw;
            kern().gemm_big(row(b, m0, i2, i3), ldx, (int) (m1 - m0), w, (int) K, (int) N, nullptr, false, row(d, m0, i2, i3), ldy);
        };
        if (nt == 1) for (int64_t i = 0; i < per_b * NB; ++i) task_b((int) i);
        else ctx.pool->run((int) (per_b * NB), task_b);
        return;
    }
    const int64_t mb = nt == 1 ? M : std::max<int64_t>(4, ((M * NB) / (nt * 4) + 3) / 4 * 4);
    const int64_t per = (M + mb - 1) / mb;
    auto task = [&](int64_t i) {
        const int64_t bi = i / per, m0 = (i % per) * mb, m1 = std::min(M, m0 + mb);
        const int64_t i2 = bi % b->ne[2], i3 = bi / b->ne[2];
        const float * X = row(b, m0, i2, i3);
        float * Y = row(d, m0, i2, i3);
        if (alias) {   // rare: through a private buffer
            std::vector<float> tmp((size_t) (m1 - m0) * N);
            kern().gemm(X, ldx, (int) (m1 - m0), WT + bi * K * ldw, ldw, (int) K, (int) N, nullptr, false, tmp.data(), N);
            for (int64_t m = 0; m < m1 - m0; ++m) std::memcpy(Y + m * ldy, tmp.data() + m * N, sizeof(float) * N);
        } else {
            kern().gemm(X, ldx, (int) (m1 - m0), WT + bi * K * ldw, ldw, (int) K, (int) N, nullptr, false, Y, ldy);
        }
    };
    if (nt == 1 || alias) for (int64_t i = 0; i < per * NB; ++i) task(i);
    else ctx.pool->run((int) (per * NB), [&](int i) { task(i); });
}

void run_gemm(Context & ctx, const Step & s) {
    const ggml_tensor * a = s.a, * b = s.b, * d = s.dst;
    if (a->ne[2] * a->ne[3] > 1) { run_gemm_batched(ctx, s); return; }
    const int K = (int) a->ne[0], N = (int) a->ne[1], M = (int) b->ne[1];
    const float * W = s.packed;
    const uint16_t * W16 = s.packed16;
    const int8_t * WQ = s.packedq;
    const float * WS = s.packeds;
    int64_t ldw = s.ldw;
    if (!W && !W16 && !WQ) {
        ldw = (N + 31) / 32 * 32;
        ctx.scratch_w.resize((size_t) K * ldw);
        pack_rows(a, 0, 0, ldw, ctx.scratch_w.data());
        W = ctx.scratch_w.data();
    }
    const bool f16_lp = W16 && kern().gemm_f16_lp;
    const int64_t ldr16 = a->nb[1] / 2;   // F16 weight rows in the file layout (gemm_f16_lp)
    const bool lp = f16_lp || s.q8_interleaved;   // reduced-precision kernels handle every M
    // columns [n0, n0 + nw) of the packed weight
    auto G = [&](const float * Xp, int64_t ldxp, int Mp, int n0, int nw, const float * bp, float * Yp, int64_t ldyp) {
        if (WQ && s.q8_interleaved) kern().gemm_q8_lp(Xp, ldxp, Mp, (const uint8_t *) a->data + (int64_t) n0 * a->nb[1], a->nb[1],
                                                      WQ + 4 * n0, WS + n0, ldw, K, nw, bp, s.relu, Yp, ldyp);
        else if (WQ)  kern().gemm_q8(Xp, ldxp, Mp, WQ + n0, WS + n0, ldw, K, nw, bp, s.relu, Yp, ldyp);
        else if (W16 && f16_lp) kern().gemm_f16_lp(Xp, ldxp, Mp, (const uint16_t *) a->data + (int64_t) n0 * ldr16, ldr16, W16 + n0, ldw,
                                                    K, nw, bp, s.relu, Yp, ldyp);
        else if (W16) kern().gemm_f16(Xp, ldxp, Mp, W16 + n0, ldw, K, nw, bp, s.relu, Yp, ldyp);
        else     kern().gemm    (Xp, ldxp, Mp, W   + n0, ldw, K, nw, bp, s.relu, Yp, ldyp);
    };
    const float * bias = s.bias ? (const float *) s.bias->data : nullptr;
    // Flatten (ne1 x ne2 x ne3) into one row dimension so the weights are read
    // once per compute, not once per batch. X is packed into scratch when its rows
    // are not uniformly strided (e.g. a [C, T, B] window view) or when the
    // allocator placed dst on top of the (dead) input.
    const int64_t Mt = b->ne[1] * b->ne[2] * b->ne[3];
    const bool alias = overlaps(d, b);
    const bool y_uniform = d->nb[2] == d->ne[1] * d->nb[1] && d->nb[3] == d->ne[2] * d->nb[2];
    const bool x_uniform = b->nb[2] == b->ne[1] * b->nb[1] && b->nb[3] == b->ne[2] * b->nb[2];
    const float * X = (const float *) b->data;
    int64_t ldx = b->nb[1] / sizeof(float);
    if (alias || (!x_uniform && Mt > M)) {
        ctx.scratch_x.resize((size_t) Mt * K);   // never aliases d
        int64_t r = 0;
        for (int64_t i3 = 0; i3 < b->ne[3]; ++i3)
            for (int64_t i2 = 0; i2 < b->ne[2]; ++i2)
                for (int64_t i1 = 0; i1 < b->ne[1]; ++i1, ++r)
                    std::memcpy(ctx.scratch_x.data() + r * K, row(b, i1, i2, i3), sizeof(float) * K);
        X = ctx.scratch_x.data();
        ldx = K;
    }
    if (!y_uniform) {   // rare: per batch, X rows are now [batch][M] when packed
        const int64_t ldy = d->nb[1] / sizeof(float);
        const bool packed = X != (const float *) b->data;
        for (int64_t i3 = 0, bi = 0; i3 < b->ne[3]; ++i3)
            for (int64_t i2 = 0; i2 < b->ne[2]; ++i2, ++bi)
                G(packed ? X + bi * M * ldx : row(b, 0, i2, i3), ldx, M, 0, N, bias, row(d, 0, i2, i3), ldy);
        return;
    }
    float * Y = (float *) d->data;
    const int64_t ldy = d->nb[1] / sizeof(float);
    if (use_big(Mt) && !lp) {
        vadk::weights w;
        w.f32 = W; w.f16 = W16; w.q8 = WQ; w.qs = WS; w.ldw = ldw;
        if (WQ || W16) w.f32 = nullptr;
        run_big(ctx, X, ldx, Mt, w, K, N, bias, s.relu, Y, ldy);
        return;
    }
    const int nt = ctx.pool ? ctx.pool->size() : 1;
    const double flops = 2.0 * Mt * N * K;
    // few rows: bound by reading the weights. Split over all threads once the weight is
    // big enough; each thread then keeps its own slice in its L2 across calls, so a
    // model whose weights overflow one core's L2 streams them from several.
    const double wbytes = (double) K * N * (WQ ? 1.0 : W16 ? 2.0 : 4.0);
    const bool gemv_split = Mt < 4 * nt && wbytes >= PAR_MIN_WBYTES;
    if (nt == 1 || (flops < PAR_FEW_FLOPS && !gemv_split)) {
        G(X, ldx, (int) Mt, 0, N, bias, Y, ldy);
        return;
    }
    if (Mt >= 4 * nt) {
        // row blocks (multiples of 4); each task streams the weights once
        const int64_t mb = std::max<int64_t>(4, (Mt / (nt * 2) + 3) / 4 * 4);
        const int n_tasks = (int) ((Mt + mb - 1) / mb);
        ctx.pool->run(n_tasks, [&](int i) {
            const int64_t m0 = i * mb, m1 = std::min(Mt, m0 + mb);
            G(X + m0 * ldx, ldx, (int) (m1 - m0), 0, N, bias, Y + m0 * ldy, ldy);
        });
    } else {
        // few rows (streaming GEMV, bound by reading the weights): split the columns over
        // as many threads as the work pays for; the pool hands each thread the same
        // columns every call, so its weight slice stays in its own L2
        const int use = (gemv_split ? nt : (int) std::max(2.0, std::min<double>(nt, flops / PAR_FEW_FLOPS))) * GEMV_TASKS_PER_THREAD;
        const int nb = std::max(32, (N / use + 31) / 32 * 32);
        const int n_tasks = (N + nb - 1) / nb;
        ctx.pool->run(n_tasks, [&](int i) {
            const int n0 = i * nb, n1 = std::min(N, n0 + nb);
            G(X, ldx, (int) Mt, n0, n1 - n0, bias ? bias + n0 : nullptr, Y + n0, ldy);
        });
    }
}

void run_step(Context & ctx, ggml_cgraph * g, const Step & s) {
    switch (s.kind) {
        case StepKind::GEMM: run_gemm(ctx, s); break;
        case StepKind::UNARY: {
            const ggml_tensor * a = s.a, * d = s.dst;
            if (ggml_is_contiguous(a) && ggml_is_contiguous(d) && (!ctx.pool || ggml_nelements(d) < PAR_MIN_ELEMS)) {
                kern().unary(s.uop, (const float *) a->data, (float *) d->data, (int) ggml_nelements(d));
            } else {
                for_rows(ctx, d, [&](int64_t i1, int64_t i2, int64_t i3) {
                    kern().unary(s.uop, row(a, i1, i2, i3), row(d, i1, i2, i3), (int) d->ne[0]);
                });
            }
        } break;
        case StepKind::BINARY: {
            const ggml_tensor * a = s.a, * b = s.b, * d = s.dst;
            for_rows(ctx, d, [&](int64_t i1, int64_t i2, int64_t i3) {
                kern().binary(s.bop, row(a, i1, i2, i3), row(b, i1 % b->ne[1], i2 % b->ne[2], i3 % b->ne[3]),
                             (int) b->ne[0], row(d, i1, i2, i3), (int) d->ne[0]);
            });
        } break;
        case StepKind::COPY: {
            const ggml_tensor * a = s.a, * d = s.dst;
            if (ggml_is_contiguous(a) && ggml_is_contiguous(d)) {
                if (a->data != d->data) std::memmove(d->data, a->data, ggml_nbytes(d));
            } else {
                // same row length, same number of rows: map rows by logical row index
                const int64_t a1 = a->ne[1], a2 = a->ne[2];
                for_rows(ctx, d, [&](int64_t j1, int64_t j2, int64_t j3) {
                    const int64_t r = j1 + d->ne[1] * (j2 + d->ne[2] * j3);
                    std::memmove(row(d, j1, j2, j3), row(a, r % a1, (r / a1) % a2, r / (a1 * a2)), sizeof(float) * d->ne[0]);
                });
            }
        } break;
        case StepKind::COPY_STRIDED: {
            // element-wise copy in logical order (e.g. cont of a permuted view):
            // walk the source with carry counters instead of dividing per element
            const ggml_tensor * a = s.a, * d = s.dst;
            const int64_t R = ggml_nrows(d), n0 = d->ne[0];
            auto range = [&](int64_t r0, int64_t r1) {
                int64_t e = r0 * n0;
                int64_t i[4] = { e % a->ne[0], (e / a->ne[0]) % a->ne[1], (e / (a->ne[0] * a->ne[1])) % a->ne[2],
                                 e / (a->ne[0] * a->ne[1] * a->ne[2]) };
                for (int64_t r = r0; r < r1; ++r) {
                    float * y = row(d, r % d->ne[1], (r / d->ne[1]) % d->ne[2], r / (d->ne[1] * d->ne[2]));
                    for (int64_t k = 0; k < n0; ++k) {
                        y[k] = *(const float *) ((const char *) a->data + i[0] * a->nb[0] + i[1] * a->nb[1] + i[2] * a->nb[2] + i[3] * a->nb[3]);
                        if (++i[0] == a->ne[0]) { i[0] = 0; if (++i[1] == a->ne[1]) { i[1] = 0; if (++i[2] == a->ne[2]) { i[2] = 0; ++i[3]; } } }
                    }
                }
            };
            const int nt = ctx.pool ? ctx.pool->size() : 1;
            if (nt == 1 || R * n0 < PAR_MIN_ELEMS) range(0, R);
            else ctx.pool->run(nt * 4, [&](int t) { range(R * t / (nt * 4), R * (t + 1) / (nt * 4)); });
        } break;
        case StepKind::CLAMP: {
            const ggml_tensor * a = s.a, * d = s.dst;
            float lo, hi;
            std::memcpy(&lo, (const char *) d->op_params, sizeof(float));
            std::memcpy(&hi, (const char *) d->op_params + sizeof(float), sizeof(float));
            for_rows(ctx, d, [&](int64_t i1, int64_t i2, int64_t i3) {
                const float * x = row(a, i1, i2, i3);
                float * y = row(d, i1, i2, i3);
                for (int64_t k = 0; k < d->ne[0]; ++k) y[k] = x[k] < lo ? lo : (x[k] > hi ? hi : x[k]);
            });
        } break;
        case StepKind::SSM_CONV: {
            // dst[c, t, s] = sum_k sx[t + k, c, s] * w[k, c]; sx rows are time-contiguous,
            // so for each (channel, stream) accumulate all T outputs with K axpys, then scatter
            const ggml_tensor * sx = s.a, * w = s.b, * d = s.dst;
            const int64_t K = w->ne[0], C = w->ne[1], T = d->ne[1], NS = d->ne[2] * d->ne[3];
            const float * W = (const float *) w->data;
            if (T < 8) {
                // few frames: one output row (all channels of a frame) at a time
                for_rows(ctx, d, [&](int64_t t, int64_t s2, int64_t s3) {
                    float * y = row(d, t, s2, s3);
                    for (int64_t c = 0; c < C; ++c) {
                        const float * x = row(sx, c, s2, s3) + t;
                        const float * wc = W + c * K;
                        float a = 0;
                        for (int64_t k = 0; k < K; ++k) a += x[k] * wc[k];
                        y[c] = a;
                    }
                });
                break;
            }
            // tasks own (stream, 16-channel block, frame block) tiles, so each writes whole
            // 64-byte output lines: no two threads share a cache line of dst
            constexpr int64_t CB = 16, TB = 64;
            const int64_t n_cb = (C + CB - 1) / CB, n_tb = (T + TB - 1) / TB, J = NS * n_cb * n_tb;
            auto work = [&](int64_t j0, int64_t j1) {
                float acc[CB][TB];
                for (int64_t j = j0; j < j1; ++j) {
                    const int64_t tb = j % n_tb, cb = (j / n_tb) % n_cb, st = j / (n_tb * n_cb);
                    const int64_t s2 = st % d->ne[2], s3 = st / d->ne[2];
                    const int64_t c0 = cb * CB, c1 = std::min(C, c0 + CB), t0 = tb * TB, nt_ = std::min(T, t0 + TB) - t0;
                    for (int64_t c = c0; c < c1; ++c) {
                        const float * x = row(sx, c, s2, s3) + t0;
                        const float * wc = W + c * K;
                        float * a = acc[c - c0];
                        for (int64_t t = 0; t < nt_; ++t) a[t] = 0.0f;
                        for (int64_t k = 0; k < K; ++k) {
                            const float wk = wc[k];
                            const float * xk = x + k;
                            for (int64_t t = 0; t < nt_; ++t) a[t] += wk * xk[t];
                        }
                    }
                    for (int64_t t = 0; t < nt_; ++t) {
                        float * y = row(d, t0 + t, s2, s3);
                        for (int64_t c = c0; c < c1; ++c) y[c] = acc[c - c0][t];
                    }
                }
            };
            const int nt = ctx.pool ? ctx.pool->size() : 1;
            if (nt == 1 || C * NS * T * K < PAR_MIN_ELEMS || J < 2) work(0, J);
            else {
                const int n_tasks = (int) std::min<int64_t>(J, nt * 4);
                ctx.pool->run(n_tasks, [&](int i) { work(J * i / n_tasks, J * (i + 1) / n_tasks); });
            }
        } break;
        case StepKind::FSMN: {
            // per (channel, stream): x = [cache (C) | q[c, 0..T)], y[c, t] = sum_k w[k, c] x[t + k],
            // new cache = x[T, T + C). Each channel only touches its own cache row, so the
            // cache can be updated in place (cache_out may alias cache).
            const ggml_tensor * cache = s.a, * q = s.b, * d = s.dst, * co = s.cache_out;
            const int64_t C = cache->ne[0], P = q->ne[0], T = q->ne[1], K = C + 1, n2 = q->ne[2], NS = n2 * q->ne[3];
            const float * W = (const float *) s.w->data;
            const int nt = ctx.pool ? ctx.pool->size() : 1;
            if (T == 1) {
                // one frame (streaming): streams outer, channels inner
                auto work = [&](int64_t st0, int64_t st1) {
                    for (int64_t st = st0; st < st1; ++st) {
                        const int64_t s2 = st % n2, s3 = st / n2;
                        const char * cb = (const char *) row(cache, 0, s2, s3);
                        char * ob = (char *) row(co, 0, s2, s3);
                        const float * q0 = row(q, 0, s2, s3);
                        float * y0 = row(d, 0, s2, s3);
                        for (int64_t c = 0; c < P; ++c) {
                            const float * cr = (const float *) (cb + c * cache->nb[1]);
                            float * cw = (float *) (ob + c * co->nb[1]);
                            const float * wc = W + c * K;
                            const float qc = q0[c];
                            float acc = qc * wc[C];
                            for (int64_t k = 0; k < C; ++k) acc += cr[k] * wc[k];
                            y0[c] = acc;
                            for (int64_t k = 0; k + 1 < C; ++k) cw[k] = cr[k + 1];   // cw may equal cr: forward copy is safe
                            cw[C - 1] = qc;
                        }
                    }
                };
                if (nt == 1 || NS * P * K < PAR_MIN_ELEMS || NS < 2) work(0, NS);
                else {
                    const int n_tasks = (int) std::min<int64_t>(NS, nt * 4);
                    ctx.pool->run(n_tasks, [&](int i) { work(NS * i / n_tasks, NS * (i + 1) / n_tasks); });
                }
                break;
            }
            // several frames: tasks own (stream, 16-channel block) tiles, so output rows are
            // written in whole 64-byte lines and no two threads share one
            constexpr int64_t CB = 16;
            const int64_t n_cb = (P + CB - 1) / CB, J = NS * n_cb;
            auto work = [&](int64_t j0, int64_t j1) {
                std::vector<float> xs((size_t) (C + T)), acc((size_t) (CB * T));
                float * x = xs.data();
                for (int64_t j = j0; j < j1; ++j) {
                    const int64_t st = j / n_cb, c0 = (j % n_cb) * CB, c1 = std::min(P, c0 + CB);
                    const int64_t s2 = st % n2, s3 = st / n2;
                    const char * cb = (const char *) row(cache, 0, s2, s3);
                    char * ob = (char *) row(co, 0, s2, s3);
                    for (int64_t c = c0; c < c1; ++c) {
                        const float * cr = (const float *) (cb + c * cache->nb[1]);
                        for (int64_t k = 0; k < C; ++k) x[k] = cr[k];
                        for (int64_t t = 0; t < T; ++t) x[C + t] = row(q, t, s2, s3)[c];
                        const float * wc = W + c * K;
                        float * a = acc.data() + (c - c0) * T;
                        // same summation order as before (k ascending per output)
                        for (int64_t t = 0; t < T; ++t) a[t] = 0.0f;
                        for (int64_t k = 0; k < K; ++k) {
                            const float wk = wc[k];
                            const float * xk = x + k;
                            for (int64_t t = 0; t < T; ++t) a[t] += xk[t] * wk;
                        }
                        float * cw = (float *) (ob + c * co->nb[1]);
                        for (int64_t k = 0; k < C; ++k) cw[k] = x[T + k];
                    }
                    for (int64_t t = 0; t < T; ++t) {
                        float * y = row(d, t, s2, s3);
                        for (int64_t c = c0; c < c1; ++c) y[c] = acc[(c - c0) * T + t];
                    }
                }
            };
            if (nt == 1 || NS * P * T * K < PAR_MIN_ELEMS || J < 2) work(0, J);
            else {
                const int n_tasks = (int) std::min<int64_t>(J, nt * 4);
                ctx.pool->run(n_tasks, [&](int i) { work(J * i / n_tasks, J * (i + 1) / n_tasks); });
            }
        } break;
        case StepKind::CUSTOM: {
            ggml_custom_op_params p;
            std::memcpy(&p, s.dst->op_params, sizeof p);
            const int nt = ctx.pool ? ctx.pool->size() : 1;
            const int n_tasks = p.n_tasks == GGML_N_TASKS_MAX ? nt : std::max(1, std::min(p.n_tasks, nt));
            if (n_tasks == 1) p.fun(s.dst, 0, 1, p.userdata);
            else ctx.pool->run(n_tasks, [&](int i) { p.fun(s.dst, i, n_tasks, p.userdata); });
        } break;
        case StepKind::NORM: {
            const ggml_tensor * a = s.a, * d = s.dst;
            float eps; std::memcpy(&eps, d->op_params, sizeof(float));
            const int64_t n0 = d->ne[0];
            for_rows(ctx, d, [&](int64_t i1, int64_t i2, int64_t i3) {
                const float * x = row(a, i1, i2, i3);
                float * y = row(d, i1, i2, i3);
                double sum = 0;
                for (int64_t k = 0; k < n0; ++k) sum += x[k];
                const float mean = (float) (sum / n0);
                double var = 0;
                for (int64_t k = 0; k < n0; ++k) { const float v = x[k] - mean; var += v * v; }
                const float inv = 1.0f / std::sqrt((float) (var / n0) + eps);
                for (int64_t k = 0; k < n0; ++k) y[k] = (x[k] - mean) * inv;
            });
        } break;
        case StepKind::SCALE: {
            const ggml_tensor * a = s.a, * d = s.dst;
            float sc, bi; std::memcpy(&sc, d->op_params, sizeof(float)); std::memcpy(&bi, (const char *) d->op_params + sizeof(float), sizeof(float));
            for_rows(ctx, d, [&](int64_t i1, int64_t i2, int64_t i3) {
                const float * x = row(a, i1, i2, i3);
                float * y = row(d, i1, i2, i3);
                for (int64_t k = 0; k < d->ne[0]; ++k) y[k] = x[k] * sc + bi;
            });
        } break;
        case StepKind::SOFT_MAX: {
            // softmax(x * scale + mask); mask rows broadcast over ne2/ne3 as in ggml (max_bias 0)
            const ggml_tensor * a = s.a, * m = s.b, * d = s.dst;
            float sc; std::memcpy(&sc, d->op_params, sizeof(float));
            const int64_t n0 = d->ne[0];
            const float * M = nullptr;
            if (m && m->type == GGML_TYPE_F32) M = (const float *) m->data;
            else if (m) {
                // the same mask usually feeds every layer: convert it once per compute
                if (ctx.mask_src != m || ctx.mask_data != m->data || ctx.mask_compute != ctx.n_compute) {
                    ctx.mask_f32.resize((size_t) ggml_nelements(m));
                    ggml_fp16_to_fp32_row((const ggml_fp16_t *) m->data, ctx.mask_f32.data(), ggml_nelements(m));
                    ctx.mask_src = m; ctx.mask_data = m->data; ctx.mask_compute = ctx.n_compute;
                }
                M = ctx.mask_f32.data();
            }
            for_rows(ctx, d, [&](int64_t i1, int64_t i2, int64_t i3) {
                const float * x = row(a, i1, i2, i3);
                float * y = row(d, i1, i2, i3);
                const float * mr = M ? M + ((i3 % m->ne[3]) * m->ne[2] * m->ne[1] + (i2 % m->ne[2]) * m->ne[1] + i1) * n0 : nullptr;
                float mx = -INFINITY;
                if (mr) for (int64_t k = 0; k < n0; ++k) { y[k] = x[k] * sc + mr[k]; mx = y[k] > mx ? y[k] : mx; }
                else    for (int64_t k = 0; k < n0; ++k) { y[k] = x[k] * sc;         mx = y[k] > mx ? y[k] : mx; }
                if (mx == -INFINITY) { for (int64_t k = 0; k < n0; ++k) y[k] = 0.0f; return; }   // fully masked row
                for (int64_t k = 0; k < n0; ++k) y[k] -= mx;
                kern().unary(vadk::UOp::EXP, y, y, (int) n0);
                double sum = 0;
                for (int64_t k = 0; k < n0; ++k) sum += y[k];
                const float inv = (float) (1.0 / sum);
                for (int64_t k = 0; k < n0; ++k) y[k] *= inv;
            });
        } break;
        case StepKind::ROPE: {
            const ggml_tensor * a = s.a, * d = s.dst;
            const int32_t * p = (const int32_t *) d->op_params;
            const int64_t n_dims = p[1], half = n_dims / 2, n_pos = a->ne[2];
            const bool neox = p[2] == GGML_ROPE_TYPE_NEOX;
            float base; std::memcpy(&base, p + 5, 4);
            const int32_t * pos = (const int32_t *) s.b->data;
            // cos/sin per (position, pair), computed as ggml does (theta *= theta_scale in float)
            const float theta_scale = powf(base, -2.0f / (float) n_dims);
            auto & key = ctx.rope_key;
            if (key.pos != pos || key.n_pos != n_pos || key.n_dims != n_dims || key.base != base || key.compute != ctx.n_compute) {
                key.pos = pos; key.n_pos = n_pos; key.n_dims = n_dims; key.base = base; key.compute = ctx.n_compute;
                ctx.rope_cs.resize((size_t) n_pos * half * 2);
                for (int64_t i = 0; i < n_pos; ++i) {
                    float theta = (float) pos[i];
                    float * cs = ctx.rope_cs.data() + i * half * 2;
                    for (int64_t j = 0; j < half; ++j, theta *= theta_scale) { cs[2 * j] = cosf(theta); cs[2 * j + 1] = sinf(theta); }
                }
            }
            const int64_t n0 = a->ne[0];
            for_rows(ctx, d, [&](int64_t i1, int64_t i2, int64_t i3) {
                const float * x = row(a, i1, i2, i3);
                float * y = row(d, i1, i2, i3);
                const float * cs = ctx.rope_cs.data() + i2 * half * 2;
                if (neox) {
                    for (int64_t j = 0; j < half; ++j) {
                        const float x0 = x[j], x1 = x[j + half], c = cs[2 * j], sn = cs[2 * j + 1];
                        y[j] = x0 * c - x1 * sn;
                        y[j + half] = x0 * sn + x1 * c;
                    }
                } else {
                    for (int64_t j = 0; j < half; ++j) {
                        const float x0 = x[2 * j], x1 = x[2 * j + 1], c = cs[2 * j], sn = cs[2 * j + 1];
                        y[2 * j] = x0 * c - x1 * sn;
                        y[2 * j + 1] = x0 * sn + x1 * c;
                    }
                }
                for (int64_t k = n_dims; k < n0; ++k) y[k] = x[k];
            });
        } break;
        case StepKind::CONCAT: {
            const ggml_tensor * a = s.a, * b = s.b, * d = s.dst;
            const int dim = ggml_get_op_params_i32(d, 0);
            if (dim == 0) {
                for_rows(ctx, d, [&](int64_t i1, int64_t i2, int64_t i3) {
                    float * y = row(d, i1, i2, i3);
                    std::memcpy(y, row(a, i1, i2, i3), sizeof(float) * a->ne[0]);
                    std::memcpy(y + a->ne[0], row(b, i1, i2, i3), sizeof(float) * b->ne[0]);
                });
            } else {
                for_rows(ctx, d, [&](int64_t i1, int64_t i2, int64_t i3) {
                    const int64_t idx[4] = { 0, i1, i2, i3 };
                    const float * x;
                    if (idx[dim] < a->ne[dim]) {
                        x = row(a, i1, i2, i3);
                    } else {
                        int64_t j[4] = { 0, i1, i2, i3 };
                        j[dim] -= a->ne[dim];
                        x = row(b, j[1], j[2], j[3]);
                    }
                    std::memcpy(row(d, i1, i2, i3), x, sizeof(float) * d->ne[0]);
                });
            }
        } break;
        case StepKind::PAD: {
            const ggml_tensor * a = s.a, * d = s.dst;
            const int lp[4] = { ggml_get_op_params_i32(d, 0), ggml_get_op_params_i32(d, 2),
                                ggml_get_op_params_i32(d, 4), ggml_get_op_params_i32(d, 6) };
            for_rows(ctx, d, [&](int64_t i1, int64_t i2, int64_t i3) {
                float * y = row(d, i1, i2, i3);
                const int64_t j1 = i1 - lp[1], j2 = i2 - lp[2], j3 = i3 - lp[3];
                if (j1 < 0 || j1 >= a->ne[1] || j2 < 0 || j2 >= a->ne[2] || j3 < 0 || j3 >= a->ne[3]) {
                    std::memset(y, 0, sizeof(float) * d->ne[0]);
                    return;
                }
                const float * x = row(a, j1, j2, j3);
                for (int64_t k = 0; k < d->ne[0]; ++k) {
                    const int64_t kk = k - lp[0];
                    y[k] = (kk >= 0 && kk < a->ne[0]) ? x[kk] : 0.0f;
                }
            });
        } break;
        case StepKind::PAD_REFLECT: {
            const ggml_tensor * a = s.a, * d = s.dst;
            const int p0 = ggml_get_op_params_i32(d, 0), p1 = ggml_get_op_params_i32(d, 1);
            const int64_t n = a->ne[0];
            for_rows(ctx, d, [&](int64_t i1, int64_t i2, int64_t i3) {
                const float * x = row(a, i1, i2, i3);
                float * y = row(d, i1, i2, i3);
                for (int k = 0; k < p0; ++k) y[k] = x[p0 - k];
                std::memcpy(y + p0, x, sizeof(float) * n);
                for (int k = 0; k < p1; ++k) y[p0 + n + k] = x[n - 2 - k];
            });
        } break;
        case StepKind::DELEGATE: {
            // Same as ggml_graph_view, which ggml does not export from shared builds.
            ggml_cgraph v = {};
            v.n_nodes = s.i1 - s.i0;
            v.nodes = g->nodes + s.i0;
            v.use_counts = g->use_counts;
            v.visited_hash_set = g->visited_hash_set;
            v.order = g->order;
            ggml_backend_graph_compute(ctx.cpu, &v);
        } break;
    }
}

// ------------------------------------------------------------- backend glue

const char * vad_cpu_get_name(ggml_backend_t) { return "vad-cpu"; }

void vad_cpu_free(ggml_backend_t backend) {
    Context * ctx = (Context *) backend->context;
    ggml_backend_free(ctx->cpu);
    delete ctx;
    delete backend;
}

ggml_status vad_cpu_graph_compute(ggml_backend_t backend, ggml_cgraph * g) {
    Context & ctx = *(Context *) backend->context;
    ++ctx.n_compute;
    const uint64_t sig = graph_signature(g);
    Plan & plan = ctx.plans[g];
    if (plan.sig != sig || plan.steps.empty()) {
        plan = build_plan(ctx, g);
        plan.sig = sig;
        ctx.last = plan.stats;
    }
    static const bool profile = getenv("VAD_CPU_PROFILE") != nullptr;
    if (!profile) {
        for (const Step & s : plan.steps) run_step(ctx, g, s);
        return GGML_STATUS_SUCCESS;
    }
    // per step-kind wall time, printed at exit
    static struct prof {
        double acc[(int) StepKind::DELEGATE + 1] = {}; long calls = 0;
        ~prof() {
            // order of StepKind
            static const char * names[] = { "gemm", "unary", "binary", "copy", "copy_strided", "concat", "pad", "pad_reflect",
                                             "clamp", "ssm_conv", "fsmn", "norm", "scale", "soft_max", "rope", "custom", "delegate" };
            fprintf(stderr, "vad-cpu profile (us per compute over %ld):", calls);
            for (int k = 0; k <= (int) StepKind::DELEGATE; ++k) if (acc[k] > 0) fprintf(stderr, " %s %.2f", names[k], acc[k] / calls);
            fprintf(stderr, "\n");
        }
    } P;
    static const bool detail = getenv("VAD_CPU_PROFILE")[0] == '2';
    for (const Step & s : plan.steps) {
        const int64_t t0 = ggml_time_us();
        run_step(ctx, g, s);
        const int64_t dt = ggml_time_us() - t0;
        P.acc[(int) s.kind] += dt;
        if (detail && s.dst)
            fprintf(stderr, "step %d %s [%ld %ld %ld %ld] %ld us\n", (int) s.kind, ggml_op_desc(s.dst), (long) s.dst->ne[0],
                    (long) s.dst->ne[1], (long) s.dst->ne[2], (long) s.dst->ne[3], (long) dt);
    }
    ++P.calls;
    return GGML_STATUS_SUCCESS;
}

ggml_guid_t vad_cpu_guid() {
    static ggml_guid guid = { 0x76, 0x61, 0x64, 0x2d, 0x63, 0x70, 0x75, 0x2d, 0x62, 0x61, 0x63, 0x6b, 0x65, 0x6e, 0x64, 0x01 };
    return &guid;
}

} // namespace

extern "C" ggml_backend_t ggml_backend_vad_cpu_init(int n_threads) {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) return nullptr;
    ggml_backend_cpu_set_n_threads(cpu, n_threads);
    Context * ctx = new Context();
    ctx->cpu = cpu;
    if (n_threads > 1) ctx->pool = std::make_unique<vadk::thread_pool>(n_threads);
    ggml_backend_i iface = {};
    iface.get_name = vad_cpu_get_name;
    iface.free = vad_cpu_free;
    iface.graph_compute = vad_cpu_graph_compute;
    return new ggml_backend{ vad_cpu_guid(), iface, cpu->device, ctx };
}

extern "C" bool ggml_backend_is_vad_cpu(ggml_backend_t backend) {
    return backend && ggml_guid_matches(backend->guid, vad_cpu_guid());
}

extern "C" const char * ggml_backend_vad_cpu_kernels(void) { return kern().name; }

extern "C" ggml_vad_cpu_plan_stats ggml_backend_vad_cpu_last_plan_stats(ggml_backend_t backend) {
    return ((Context *) backend->context)->last;
}
