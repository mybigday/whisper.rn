#include "ggml-metal-ops.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-metal-impl.h"
#include "ggml-metal-common.h"
#include "ggml-metal-device.h"

#include <cassert>
#include <algorithm>
#include <limits>
#include <cmath>

static wsp_ggml_metal_buffer_id wsp_ggml_metal_get_buffer_id(const wsp_ggml_tensor * t) {
    if (!t) {
        return { nullptr, 0 };
    }

    wsp_ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    wsp_ggml_metal_buffer_t ctx = (wsp_ggml_metal_buffer_t) buffer->context;

    return wsp_ggml_metal_buffer_get_id(ctx, t);
}

struct wsp_ggml_metal_op {
    wsp_ggml_metal_op(
        wsp_ggml_metal_device_t dev,
        wsp_ggml_metal_cmd_buf_t cmd_buf,
        wsp_ggml_cgraph * gf,
        int  idx_start,
        int  idx_end,
        bool use_fusion,
        bool use_concurrency,
        bool use_capture,
        int  debug_graph,
        int  debug_fusion) {
        this->dev             = dev;
        this->lib             = wsp_ggml_metal_device_get_library(dev);
        this->enc             = wsp_ggml_metal_encoder_init(cmd_buf, use_concurrency);
        this->mem_ranges      = wsp_ggml_mem_ranges_init(debug_graph);
        this->idx_start       = idx_start;
        this->idx_end         = idx_end;
        this->use_fusion      = use_fusion;
        this->use_concurrency = use_concurrency;
        this->use_capture     = use_capture;
        this->debug_graph     = debug_graph;
        this->debug_fusion    = debug_fusion;
        this->gf              = gf;

        idxs.reserve(gf->n_nodes);

        // filter empty nodes
        // TODO: this can be removed when the allocator starts filtering them earlier
        //       https://github.com/ggml-org/llama.cpp/pull/16130#issuecomment-3327905830
        for (int i = idx_start; i < idx_end; i++) {
            if (!wsp_ggml_op_is_empty(gf->nodes[i]->op) && !wsp_ggml_is_empty(gf->nodes[i])) {
                idxs.push_back(i);
            }
        }
    }

    ~wsp_ggml_metal_op() {
        wsp_ggml_metal_encoder_end_encoding(this->enc);
        wsp_ggml_metal_encoder_free(this->enc);
        wsp_ggml_mem_ranges_free(this->mem_ranges);
    }

    int n_nodes() const {
        return idxs.size();
    }

    wsp_ggml_tensor * node(int i) const {
        assert(i >= 0 && i < (int) idxs.size());
        return wsp_ggml_graph_node(gf, idxs[i]);
    }

    bool can_fuse(int i0, const wsp_ggml_op * ops, int n_ops) const {
        assert(use_fusion);
        assert(i0 >= 0 && i0 < n_nodes());

        if (i0 + n_ops > n_nodes()) {
            return false;
        }

        return wsp_ggml_can_fuse_ext(gf, idxs.data() + i0, ops, n_ops);
    }

    wsp_ggml_metal_device_t  dev;
    wsp_ggml_metal_library_t lib;
    wsp_ggml_metal_encoder_t enc;
    wsp_ggml_mem_ranges_t    mem_ranges;

    bool use_fusion;
    bool use_concurrency;
    bool use_capture;

    int debug_graph;
    int debug_fusion;

private:
    wsp_ggml_cgraph * gf;

    int idx_start;
    int idx_end;

    // non-empty node indices
    std::vector<int> idxs;
};

wsp_ggml_metal_op_t wsp_ggml_metal_op_init(
        wsp_ggml_metal_device_t dev,
        wsp_ggml_metal_cmd_buf_t cmd_buf,
        wsp_ggml_cgraph * gf,
        int idx_start,
        int idx_end,
        bool use_fusion,
        bool use_concurrency,
        bool use_capture,
        int debug_graph,
        int debug_fusion) {
    wsp_ggml_metal_op_t res = new wsp_ggml_metal_op(
        dev,
        cmd_buf,
        gf,
        idx_start,
        idx_end,
        use_fusion,
        use_concurrency,
        use_capture,
        debug_graph,
        debug_fusion);

    return res;
}

void wsp_ggml_metal_op_free(wsp_ggml_metal_op_t ctx) {
    delete ctx;
}

int wsp_ggml_metal_op_n_nodes(wsp_ggml_metal_op_t ctx) {
    return ctx->n_nodes();
}

static bool wsp_ggml_metal_op_concurrency_reset(wsp_ggml_metal_op_t ctx) {
    if (!ctx->mem_ranges) {
        return true;
    }

    wsp_ggml_metal_encoder_memory_barrier(ctx->enc);

    wsp_ggml_mem_ranges_reset(ctx->mem_ranges);

    return true;
}

static bool wsp_ggml_metal_op_concurrency_check(wsp_ggml_metal_op_t ctx, const wsp_ggml_tensor * node) {
    if (!ctx->mem_ranges) {
        return false;
    }

    return wsp_ggml_mem_ranges_check(ctx->mem_ranges, node);
}

static bool wsp_ggml_metal_op_concurrency_add(wsp_ggml_metal_op_t ctx, const wsp_ggml_tensor * node) {
    if (!ctx->mem_ranges) {
        return true;
    }

    return wsp_ggml_mem_ranges_add(ctx->mem_ranges, node);
}

static int wsp_ggml_metal_op_encode_impl(wsp_ggml_metal_op_t ctx, int idx) {
    struct wsp_ggml_tensor * node = ctx->node(idx);

    //WSP_GGML_LOG_INFO("%s: encoding node %3d, op = %8s\n", __func__, idx, wsp_ggml_op_name(node->op));

    if (wsp_ggml_is_empty(node)) {
        return 1;
    }

    switch (node->op) {
        case WSP_GGML_OP_NONE:
        case WSP_GGML_OP_RESHAPE:
        case WSP_GGML_OP_VIEW:
        case WSP_GGML_OP_TRANSPOSE:
        case WSP_GGML_OP_PERMUTE:
            {
                // noop -> next node
                if (ctx->debug_graph > 0) {
                    WSP_GGML_LOG_DEBUG("%s: node[%5d] - %-12s %s\n", __func__, idx, wsp_ggml_op_name(node->op), "(noop)");
                }
            } return 1;
        default:
            {
            } break;
    }

    if (!wsp_ggml_metal_device_supports_op(ctx->dev, node)) {
        WSP_GGML_LOG_ERROR("%s: error: unsupported op '%s'\n", __func__, wsp_ggml_op_desc(node));
        WSP_GGML_ABORT("unsupported op");
    }

    int n_fuse = 1;

    // check if the current node can run concurrently with other nodes before it
    // the condition is that:
    //  - the current node cannot write to any previous src or dst ranges
    //  - the current node cannot read from any previous dst ranges
    //
    // if the condition is not satisfied, we put a memory barrier and clear all ranges
    // otherwise, we add the new ranges to the encoding context and process the node concurrently
    //
    {
        const bool is_concurrent = wsp_ggml_metal_op_concurrency_check(ctx, node);

        if (!is_concurrent) {
            wsp_ggml_metal_op_concurrency_reset(ctx);
        }

        if (ctx->debug_graph > 0) {
            WSP_GGML_LOG_DEBUG("%s: node[%5d] - %-12s %s\n", __func__, idx, wsp_ggml_op_name(node->op), is_concurrent ? "(concurrent)" : "");
        }
        if (ctx->debug_graph > 1) {
            WSP_GGML_TENSOR_LOCALS( int64_t, ne0, node->src[0], ne);
            WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, node->src[0], nb);
            WSP_GGML_TENSOR_LOCALS( int64_t, ne1, node->src[1], ne);
            WSP_GGML_TENSOR_LOCALS(uint64_t, nb1, node->src[1], nb);
            WSP_GGML_TENSOR_LOCALS( int64_t, ne2, node->src[2], ne);
            WSP_GGML_TENSOR_LOCALS(uint64_t, nb2, node->src[2], nb);
            WSP_GGML_TENSOR_LOCALS( int64_t, ne3, node->src[3], ne);
            WSP_GGML_TENSOR_LOCALS(uint64_t, nb3, node->src[3], nb);
            WSP_GGML_TENSOR_LOCALS( int64_t, ne,  node,         ne);
            WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  node,         nb);

            if (node->src[0]) {
                WSP_GGML_LOG_DEBUG("%s: src0 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, wsp_ggml_type_name(node->src[0]->type), ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03,
                        wsp_ggml_is_contiguous(node->src[0]), node->src[0]->name);
            }
            if (node->src[1]) {
                WSP_GGML_LOG_DEBUG("%s: src1 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, wsp_ggml_type_name(node->src[1]->type), ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13,
                        wsp_ggml_is_contiguous(node->src[1]), node->src[1]->name);
            }
            if (node->src[2]) {
                WSP_GGML_LOG_DEBUG("%s: src2 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, wsp_ggml_type_name(node->src[2]->type), ne20, ne21, ne22, ne23, nb20, nb21, nb22, nb23,
                        wsp_ggml_is_contiguous(node->src[2]), node->src[2]->name);
            }
            if (node->src[3]) {
                WSP_GGML_LOG_DEBUG("%s: src3 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, wsp_ggml_type_name(node->src[3]->type), ne30, ne31, ne32, ne33, nb30, nb31, nb32, nb33,
                        wsp_ggml_is_contiguous(node->src[3]), node->src[3]->name);
            }
            if (node) {
                WSP_GGML_LOG_DEBUG("%s: node  - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], 1, %s\n", __func__, wsp_ggml_type_name(node->type), ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3,
                        node->name);
            }
        }
    }

    switch (node->op) {
        case WSP_GGML_OP_CONCAT:
            {
                n_fuse = wsp_ggml_metal_op_concat(ctx, idx);
            } break;
        case WSP_GGML_OP_ADD:
        case WSP_GGML_OP_SUB:
        case WSP_GGML_OP_MUL:
        case WSP_GGML_OP_DIV:
            {
                n_fuse = wsp_ggml_metal_op_bin(ctx, idx);
            } break;
        case WSP_GGML_OP_ADD_ID:
            {
                n_fuse = wsp_ggml_metal_op_add_id(ctx, idx);
            } break;
        case WSP_GGML_OP_REPEAT:
            {
                n_fuse = wsp_ggml_metal_op_repeat(ctx, idx);
            } break;
        case WSP_GGML_OP_ACC:
            {
                n_fuse = wsp_ggml_metal_op_acc(ctx, idx);
            } break;
        case WSP_GGML_OP_SCALE:
            {
                n_fuse = wsp_ggml_metal_op_scale(ctx, idx);
            } break;
        case WSP_GGML_OP_CLAMP:
            {
                n_fuse = wsp_ggml_metal_op_clamp(ctx, idx);
            } break;
        case WSP_GGML_OP_SQR:
        case WSP_GGML_OP_SQRT:
        case WSP_GGML_OP_SIN:
        case WSP_GGML_OP_COS:
        case WSP_GGML_OP_LOG:
        case WSP_GGML_OP_UNARY:
            {
                n_fuse = wsp_ggml_metal_op_unary(ctx, idx);
            } break;
        case WSP_GGML_OP_GLU:
            {
                n_fuse = wsp_ggml_metal_op_glu(ctx, idx);
            } break;
        case WSP_GGML_OP_SUM:
            {
                n_fuse = wsp_ggml_metal_op_sum(ctx, idx);
            } break;
        case WSP_GGML_OP_SUM_ROWS:
        case WSP_GGML_OP_MEAN:
            {
                n_fuse = wsp_ggml_metal_op_sum_rows(ctx, idx);
            } break;
        case WSP_GGML_OP_CUMSUM:
            {
                n_fuse = wsp_ggml_metal_op_cumsum(ctx, idx);
            } break;
        case WSP_GGML_OP_SOFT_MAX:
            {
                n_fuse = wsp_ggml_metal_op_soft_max(ctx, idx);
            } break;
        case WSP_GGML_OP_SSM_CONV:
            {
                n_fuse = wsp_ggml_metal_op_ssm_conv(ctx, idx);
            } break;
        case WSP_GGML_OP_SSM_SCAN:
            {
                n_fuse = wsp_ggml_metal_op_ssm_scan(ctx, idx);
            } break;
        case WSP_GGML_OP_RWKV_WKV6:
        case WSP_GGML_OP_RWKV_WKV7:
            {
                n_fuse = wsp_ggml_metal_op_rwkv(ctx, idx);
            } break;
        case WSP_GGML_OP_MUL_MAT:
            {
                n_fuse = wsp_ggml_metal_op_mul_mat(ctx, idx);
            } break;
        case WSP_GGML_OP_MUL_MAT_ID:
            {
                n_fuse = wsp_ggml_metal_op_mul_mat_id(ctx, idx);
            } break;
        case WSP_GGML_OP_GET_ROWS:
            {
                n_fuse = wsp_ggml_metal_op_get_rows(ctx, idx);
            } break;
        case WSP_GGML_OP_SET_ROWS:
            {
                n_fuse = wsp_ggml_metal_op_set_rows(ctx, idx);
            } break;
        case WSP_GGML_OP_L2_NORM:
            {
                n_fuse = wsp_ggml_metal_op_l2_norm(ctx, idx);
            } break;
        case WSP_GGML_OP_GROUP_NORM:
            {
                n_fuse = wsp_ggml_metal_op_group_norm(ctx, idx);
            } break;
        case WSP_GGML_OP_NORM:
        case WSP_GGML_OP_RMS_NORM:
            {
                n_fuse = wsp_ggml_metal_op_norm(ctx, idx);
            } break;
        case WSP_GGML_OP_ROPE:
            {
                n_fuse = wsp_ggml_metal_op_rope(ctx, idx);
            } break;
        case WSP_GGML_OP_IM2COL:
            {
                n_fuse = wsp_ggml_metal_op_im2col(ctx, idx);
            } break;
        case WSP_GGML_OP_CONV_2D:
            {
                n_fuse = wsp_ggml_metal_op_conv_2d(ctx, idx);
            } break;
        case WSP_GGML_OP_CONV_TRANSPOSE_1D:
            {
                n_fuse = wsp_ggml_metal_op_conv_transpose_1d(ctx, idx);
            } break;
        case WSP_GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_fuse = wsp_ggml_metal_op_conv_transpose_2d(ctx, idx);
            } break;
        case WSP_GGML_OP_UPSCALE:
            {
                n_fuse = wsp_ggml_metal_op_upscale(ctx, idx);
            } break;
        case WSP_GGML_OP_PAD:
            {
                n_fuse = wsp_ggml_metal_op_pad(ctx, idx);
            } break;
        case WSP_GGML_OP_PAD_REFLECT_1D:
            {
                n_fuse = wsp_ggml_metal_op_pad_reflect_1d(ctx, idx);
            } break;
        case WSP_GGML_OP_ARANGE:
            {
                n_fuse = wsp_ggml_metal_op_arange(ctx, idx);
            } break;
        case WSP_GGML_OP_TIMESTEP_EMBEDDING:
            {
                n_fuse = wsp_ggml_metal_op_timestep_embedding(ctx, idx);
            } break;
        case WSP_GGML_OP_ARGSORT:
            {
                n_fuse = wsp_ggml_metal_op_argsort(ctx, idx);
            } break;
        case WSP_GGML_OP_LEAKY_RELU:
            {
                n_fuse = wsp_ggml_metal_op_leaky_relu(ctx, idx);
            } break;
        case WSP_GGML_OP_FLASH_ATTN_EXT:
            {
                n_fuse = wsp_ggml_metal_op_flash_attn_ext(ctx, idx);
            } break;
        case WSP_GGML_OP_DUP:
        case WSP_GGML_OP_CPY:
        case WSP_GGML_OP_CONT:
            {
                n_fuse = wsp_ggml_metal_op_cpy(ctx, idx);
            } break;
        case WSP_GGML_OP_POOL_2D:
            {
                n_fuse = wsp_ggml_metal_op_pool_2d(ctx, idx);
            } break;
        case WSP_GGML_OP_ARGMAX:
            {
                n_fuse = wsp_ggml_metal_op_argmax(ctx, idx);
            } break;
        case WSP_GGML_OP_OPT_STEP_ADAMW:
            {
                n_fuse = wsp_ggml_metal_op_opt_step_adamw(ctx, idx);
            } break;
        case WSP_GGML_OP_OPT_STEP_SGD:
            {
                n_fuse = wsp_ggml_metal_op_opt_step_sgd(ctx, idx);
            } break;
       default:
            {
                WSP_GGML_LOG_ERROR("%s: error: node %3d, op = %8s not implemented\n", __func__, idx, wsp_ggml_op_name(node->op));
                WSP_GGML_ABORT("fatal error");
            }
    }

    if (ctx->debug_graph > 0) {
        if (n_fuse > 1) {
            WSP_GGML_LOG_DEBUG("%s:               fuse %d ops\n", __func__, n_fuse);
        }
    }

    // update the mem ranges in the encoding context
    for (int i = 0; i < n_fuse; ++i) {
        if (!wsp_ggml_metal_op_concurrency_add(ctx, ctx->node(idx + i))) {
            wsp_ggml_metal_op_concurrency_reset(ctx);
        }
    }

    return n_fuse;
}

int wsp_ggml_metal_op_encode(wsp_ggml_metal_op_t ctx, int idx) {
    if (ctx->use_capture) {
        wsp_ggml_metal_encoder_debug_group_push(ctx->enc, wsp_ggml_op_desc(ctx->node(idx)));
    }

    int res = wsp_ggml_metal_op_encode_impl(ctx, idx);
    if (idx + res > ctx->n_nodes()) {
        WSP_GGML_ABORT("fusion error: nodes spanning multiple encoders have been fused. this indicates a bug in the fusion logic %s",
                "https://github.com/ggml-org/llama.cpp/pull/14849");
    }

    if (ctx->use_capture) {
        wsp_ggml_metal_encoder_debug_group_pop(ctx->enc);
    }

    return res;
}

int wsp_ggml_metal_op_concat(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t dim = ((const int32_t *) op->op_params)[0];

    wsp_ggml_metal_kargs_concat args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.dim  =*/ dim,
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_base(lib, WSP_GGML_OP_CONCAT);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[1]), 2);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         3);

    const int nth = std::min(1024, ne0);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_repeat(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_repeat(lib, op->type);

    wsp_ggml_metal_kargs_repeat args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         2);

    const int nth = std::min(wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_acc(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    WSP_GGML_ASSERT(op->src[0]->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT(op->src[1]->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT(op->type         == WSP_GGML_TYPE_F32);

    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(op->src[0]));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(op->src[1]));

    const size_t pnb1 = ((const int32_t *) op->op_params)[0];
    const size_t pnb2 = ((const int32_t *) op->op_params)[1];
    const size_t pnb3 = ((const int32_t *) op->op_params)[2];
    const size_t offs = ((const int32_t *) op->op_params)[3];

    const bool inplace = (bool) ((const int32_t *) op->op_params)[4];

    if (!inplace) {
        // run a separete kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        //const id<MTLComputePipelineState> pipeline = ctx->pipelines[WSP_GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].obj;
        wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

        wsp_ggml_metal_kargs_cpy args = {
            /*.nk0  =*/ ne00,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.ne2  =*/ ne2,
            /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,
            /*.nb1  =*/ nb1,
            /*.nb2  =*/ nb2,
            /*.nb3  =*/ nb3,
        };

        wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
        wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
        wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         2);

        const int nth = std::min(wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

        wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

        wsp_ggml_metal_op_concurrency_reset(ctx);
    }

    wsp_ggml_metal_kargs_bin args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ pnb1,
        /*.nb02 =*/ pnb2,
        /*.nb03 =*/ pnb3,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ pnb1,
        /*.nb2  =*/ pnb2,
        /*.nb3  =*/ pnb3,
        /*.offs =*/ offs,
        /*.o1   =*/ { 0 },
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_bin(lib, WSP_GGML_OP_ADD, 1, false);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[1]), 2);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         3);

    const int nth = std::min(wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ne11, ne12, ne13, nth, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_scale(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float scale;
    float bias;
    memcpy(&scale, ((const int32_t *) op->op_params) + 0, sizeof(float));
    memcpy(&bias,  ((const int32_t *) op->op_params) + 1, sizeof(float));

    wsp_ggml_metal_kargs_scale args = {
        /*.scale =*/ scale,
        /*.bias  =*/ bias,
    };

    int64_t n = wsp_ggml_nelements(op);

    if (n % 4 == 0) {
        n /= 4;
    }

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_unary(lib, op);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         2);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, 1, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_clamp(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float min;
    float max;
    memcpy(&min, ((const int32_t *) op->op_params) + 0, sizeof(float));
    memcpy(&max, ((const int32_t *) op->op_params) + 1, sizeof(float));

    wsp_ggml_metal_kargs_clamp args = {
        /*.min =*/ min,
        /*.max =*/ max,
    };

    int64_t n = wsp_ggml_nelements(op);

    if (n % 4 == 0) {
        n /= 4;
    }

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_unary(lib, op);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         2);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, 1, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_unary(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    int64_t n = wsp_ggml_nelements(op);

    if (n % 4 == 0) {
        n /= 4;
    }

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_unary(lib, op);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         1);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, 1, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_glu(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    if (op->src[1]) {
        WSP_GGML_ASSERT(wsp_ggml_are_same_shape(op->src[0], op->src[1]));
    }

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_glu(lib, op);

    const int32_t swp = wsp_ggml_get_op_params_i32(op, 1);
    const float alpha = wsp_ggml_get_op_params_f32(op, 2);
    const float limit = wsp_ggml_get_op_params_f32(op, 3);

    const int32_t i00 = swp ? ne0 : 0;
    const int32_t i10 = swp ? 0 : ne0;

    wsp_ggml_metal_kargs_glu args = {
        /*.ne00 =*/ ne00,
        /*.nb01 =*/ nb01,
        /*.ne10 =*/ op->src[1] ? ne10 : ne00,
        /*.nb11 =*/ op->src[1] ? nb11 : nb01,
        /*.ne0  =*/ ne0,
        /*.nb1  =*/ nb1,
        /*.i00  =*/ op->src[1] ? 0 : i00,
        /*.i10  =*/ op->src[1] ? 0 : i10,
        /*.alpha=*/ alpha,
        /*.limit=*/ limit
    };

    const int64_t nrows = wsp_ggml_nrows(op->src[0]);

    const int32_t nth = std::min(wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00/2);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    if (op->src[1]) {
        wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[1]), 2);
    } else {
        wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 2);
    }
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         3);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_sum(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op  = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    const uint64_t n = (uint64_t) wsp_ggml_nelements(op->src[0]);

    wsp_ggml_metal_kargs_sum args = {
        /*.np =*/ n,
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_sum(lib, op);

    int nth = 32; // SIMD width

    while (nth < (int) n && nth < wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (int) n);

    const int nsg = (nth + 31) / 32;

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         2);

    wsp_ggml_metal_encoder_set_threadgroup_memory_size(enc, nsg * sizeof(float), 0);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_sum_rows(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    wsp_ggml_metal_kargs_sum_rows args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_sum_rows(lib, op);

    int nth = 32; // SIMD width

    while (nth < ne00 && nth < wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, ne00);

    const size_t smem = wsp_ggml_metal_pipeline_get_smem(pipeline);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         2);

    wsp_ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_cumsum(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_ASSERT(wsp_ggml_is_contiguous_rows(op->src[0]));

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    wsp_ggml_metal_pipeline_t pipeline_blk = wsp_ggml_metal_library_get_pipeline_cumsum_blk(lib, op);

    int nth = 1;
    while (nth < ne00 && 2*nth <= wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_blk)) {
        nth *= 2;
    }

    WSP_GGML_ASSERT(ne00 <= nth*nth);

    const int64_t net0 = (ne00 + nth - 1) / nth;
    const int64_t net1 = ne01;
    const int64_t net2 = ne02;
    const int64_t net3 = ne03;

    const uint64_t nbt0 = sizeof(float);
    const uint64_t nbt1 = net0*nbt0;
    const uint64_t nbt2 = net1*nbt1;
    const uint64_t nbt3 = net2*nbt2;

    const size_t smem = WSP_GGML_PAD(32*sizeof(float), 16);

    wsp_ggml_metal_buffer_id bid_src0 = wsp_ggml_metal_get_buffer_id(op->src[0]);
    wsp_ggml_metal_buffer_id bid_dst  = wsp_ggml_metal_get_buffer_id(op);

    wsp_ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += wsp_ggml_nbytes(op);

    {
        wsp_ggml_metal_kargs_cumsum_blk args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.net0 =*/ net0,
            /*.net1 =*/ net1,
            /*.net2 =*/ net2,
            /*.net3 =*/ net3,
            /*.nbt0 =*/ nbt0,
            /*.nbt1 =*/ nbt1,
            /*.nbt2 =*/ nbt2,
            /*.nbt3 =*/ nbt3,
            /*.outb =*/ ne00 > nth,
        };

        wsp_ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
        wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        wsp_ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        wsp_ggml_metal_encoder_set_buffer  (enc, bid_tmp,  2);
        wsp_ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);

        wsp_ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        wsp_ggml_metal_encoder_dispatch_threadgroups(enc, net0*ne01, ne02, ne03, nth, 1, 1);
    }

    if (ne00 > nth) {
        wsp_ggml_metal_op_concurrency_reset(ctx);

        {
            wsp_ggml_metal_kargs_cumsum_blk args = {
                /*.ne00 =*/ net0,
                /*.ne01 =*/ net1,
                /*.ne02 =*/ net2,
                /*.ne03 =*/ net3,
                /*.nb00 =*/ nbt0,
                /*.nb01 =*/ nbt1,
                /*.nb02 =*/ nbt2,
                /*.nb03 =*/ nbt3,
                /*.net0 =*/ net0,
                /*.net1 =*/ net1,
                /*.net2 =*/ net2,
                /*.net3 =*/ net3,
                /*.nbt0 =*/ nbt0,
                /*.nbt1 =*/ nbt1,
                /*.nbt2 =*/ nbt2,
                /*.nbt3 =*/ nbt3,
                /*.outb =*/ false,
            };

            wsp_ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
            wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_tmp, 2);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_tmp, 3);

            wsp_ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            wsp_ggml_metal_encoder_dispatch_threadgroups(enc, net1, net2, net3, nth, 1, 1);
        }

        wsp_ggml_metal_op_concurrency_reset(ctx);

        {
            wsp_ggml_metal_pipeline_t pipeline_add = wsp_ggml_metal_library_get_pipeline_cumsum_add(lib, op);

            wsp_ggml_metal_kargs_cumsum_add args = {
                /*.ne00 =*/ ne00,
                /*.ne01 =*/ ne01,
                /*.ne02 =*/ ne02,
                /*.ne03 =*/ ne03,
                /*.nb00 =*/ nb00,
                /*.nb01 =*/ nb01,
                /*.nb02 =*/ nb02,
                /*.nb03 =*/ nb03,
                /*.net0 =*/ net0,
                /*.net1 =*/ net1,
                /*.net2 =*/ net2,
                /*.net3 =*/ net3,
                /*.nbt0 =*/ nbt0,
                /*.nbt1 =*/ nbt1,
                /*.nbt2 =*/ nbt2,
                /*.nbt3 =*/ nbt3,
            };

            wsp_ggml_metal_encoder_set_pipeline(enc, pipeline_add);
            wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

            wsp_ggml_metal_encoder_dispatch_threadgroups(enc, net0*ne01, ne02, ne03, nth, 1, 1);
        }
    }

    return 1;
}

int wsp_ggml_metal_op_get_rows(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_get_rows(lib, op->src[0]->type);

    wsp_ggml_metal_kargs_get_rows args = {
        /*.ne00t =*/ wsp_ggml_is_quantized(op->src[0]->type) ? ne00/16 : ne00,
        /*.ne00  =*/ ne00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne10  =*/ ne10,
        /*.nb10  =*/ nb10,
        /*.nb11  =*/ nb11,
        /*.nb12  =*/ nb12,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
    };

    const int nth = std::min(args.ne00t, wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const int nw0 = (args.ne00t + nth - 1)/nth;

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[1]), 2);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         3);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, nw0*ne10, ne11, ne12, nth, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_set_rows(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_set_rows(lib, op->src[1]->type, op->type);

    const int32_t nk0 = ne0/wsp_ggml_blck_size(op->type);

    int nth = 32; // SIMD width

    while (nth < nk0 && nth < wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    int nrptg = 1;
    if (nth > nk0) {
        nrptg = (nth + nk0 - 1)/nk0;
        nth   = nk0;

        if (nrptg*nth > wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
            nrptg--;
        }
    }

    nth = std::min(nth, nk0);

    wsp_ggml_metal_kargs_set_rows args = {
        /*.nk0  =*/ nk0,
        /*.ne01 =*/ ne01,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[1]), 2);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         3);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);

    return 1;
}

int wsp_ggml_metal_op_soft_max(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float scale;
    float max_bias;

    memcpy(&scale,    ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias, ((const int32_t *) op->op_params) + 1, sizeof(max_bias));

    const uint32_t n_head      = op->src[0]->ne[2];
    const  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // softmax

    wsp_ggml_metal_kargs_soft_max args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne11        =*/ ne11,
        /*.ne12        =*/ ne12,
        /*.ne13        =*/ ne13,
        /*.nb11        =*/ nb11,
        /*.nb12        =*/ nb12,
        /*.nb13        =*/ nb13,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.scale       =*/ scale,
        /*.max_bias    =*/ max_bias,
        /*.m0          =*/ m0,
        /*.m1          =*/ m1,
        /*.n_head_log2 =*/ n_head_log2,
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_soft_max(lib, op);

    int nth = 32; // SIMD width

    if (ne00%4 == 0) {
        while (nth < ne00/4 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    } else {
        while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    }

    const size_t smem = wsp_ggml_metal_pipeline_get_smem(pipeline);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer(enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    if (op->src[1]) {
        wsp_ggml_metal_encoder_set_buffer(enc, wsp_ggml_metal_get_buffer_id(op->src[1]), 2);
    } else {
        wsp_ggml_metal_encoder_set_buffer(enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 2);
    }
    if (op->src[2]) {
        wsp_ggml_metal_encoder_set_buffer(enc, wsp_ggml_metal_get_buffer_id(op->src[2]), 3);
    } else {
        wsp_ggml_metal_encoder_set_buffer(enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    wsp_ggml_metal_encoder_set_buffer(enc, wsp_ggml_metal_get_buffer_id(op), 4);

    wsp_ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_ssm_conv(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    wsp_ggml_metal_kargs_ssm_conv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_ssm_conv(lib, op);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer(enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer(enc, wsp_ggml_metal_get_buffer_id(op->src[1]), 2);
    wsp_ggml_metal_encoder_set_buffer(enc, wsp_ggml_metal_get_buffer_id(op),         3);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne1, ne02, 1, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_ssm_scan(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne4, op->src[4], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb4, op->src[4], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne5, op->src[5], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb5, op->src[5], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne6, op->src[6], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb6, op->src[6], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const wsp_ggml_tensor * src3 = op->src[3];
    const wsp_ggml_tensor * src4 = op->src[4];
    const wsp_ggml_tensor * src5 = op->src[5];
    const wsp_ggml_tensor * src6 = op->src[6];

    WSP_GGML_ASSERT(src3);
    WSP_GGML_ASSERT(src4);
    WSP_GGML_ASSERT(src5);
    WSP_GGML_ASSERT(src6);

    const int64_t d_state      = ne00;
    const int64_t d_inner      = ne01;
    const int64_t n_head       = ne02;
    const int64_t n_group      = ne41;
    const int64_t n_seq_tokens = ne12;
    const int64_t n_seqs       = ne13;

    wsp_ggml_metal_kargs_ssm_scan args = {
        /*.d_state      =*/ d_state,
        /*.d_inner      =*/ d_inner,
        /*.n_head       =*/ n_head,
        /*.n_group      =*/ n_group,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seqs       =*/ n_seqs,
        /*.s_off        =*/ wsp_ggml_nelements(op->src[1]) * sizeof(float),
        /*.nb00         =*/ nb00,
        /*.nb01         =*/ nb01,
        /*.nb02         =*/ nb02,
        /*.nb03         =*/ nb03,
        /*.nb10         =*/ nb10,
        /*.nb11         =*/ nb11,
        /*.nb12         =*/ nb12,
        /*.ns12         =*/ nb12/nb10,
        /*.nb13         =*/ nb13,
        /*.nb20         =*/ nb20,
        /*.nb21         =*/ nb21,
        /*.ns21         =*/ nb21/nb20,
        /*.nb22         =*/ nb22,
        /*.ne30         =*/ ne30,
        /*.nb31         =*/ nb31,
        /*.nb41         =*/ nb41,
        /*.nb42         =*/ nb42,
        /*.ns42         =*/ nb42/nb40,
        /*.nb43         =*/ nb43,
        /*.nb51         =*/ nb51,
        /*.nb52         =*/ nb52,
        /*.ns52         =*/ nb52/nb50,
        /*.nb53         =*/ nb53,
        /*.nb0          =*/ nb0,
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_ssm_scan(lib, op);

    WSP_GGML_ASSERT(d_state <= wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const size_t sms = wsp_ggml_metal_pipeline_get_smem(pipeline);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[1]), 2);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[2]), 3);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[3]), 4);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[4]), 5);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[5]), 6);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[6]), 7);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         8);

    wsp_ggml_metal_encoder_set_threadgroup_memory_size(enc, sms, 0);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, d_inner, n_head, n_seqs, d_state, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_rwkv(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int64_t B = op->op == WSP_GGML_OP_RWKV_WKV6 ? op->src[5]->ne[1] : op->src[6]->ne[1];
    const int64_t T = op->src[0]->ne[2];
    const int64_t C = op->ne[0];
    const int64_t H = op->src[0]->ne[1];

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_rwkv(lib, op);

    int ida = 0;

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), ida++);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[1]), ida++);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[2]), ida++);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[3]), ida++);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[4]), ida++);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[5]), ida++);
    if (op->op == WSP_GGML_OP_RWKV_WKV7) {
        wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[6]), ida++);
    }
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         ida++);
    wsp_ggml_metal_encoder_set_bytes   (enc, (void *) &B, sizeof(B), ida++);
    wsp_ggml_metal_encoder_set_bytes   (enc, (void *) &T, sizeof(T), ida++);
    wsp_ggml_metal_encoder_set_bytes   (enc, (void *) &C, sizeof(C), ida++);
    wsp_ggml_metal_encoder_set_bytes   (enc, (void *) &H, sizeof(H), ida++);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, B * H, 1, 1, C/H, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_cpy(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

    WSP_GGML_ASSERT(ne00 % wsp_ggml_blck_size(op->src[0]->type) == 0);

    int64_t nk0 = ne00;
    if (wsp_ggml_is_quantized(op->src[0]->type)) {
        nk0 = ne00/16;
    } else if (wsp_ggml_is_quantized(op->type)) {
        nk0 = ne00/wsp_ggml_blck_size(op->type);
    }

    int nth = std::min<int>(nk0, wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;

    // TODO: relax this constraint in the future
    if (wsp_ggml_blck_size(op->src[0]->type) == 1 && wsp_ggml_blck_size(op->type) == 1) {
        if (nth > nk0) {
            nrptg = (nth + nk0 - 1)/nk0;
            nth   = nk0;

            if (nrptg*nth > wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
                nrptg--;
            }
        }
    }

    nth = std::min<int>(nth, nk0);

    wsp_ggml_metal_kargs_cpy args = {
        /*.nk0  =*/ nk0,
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    const int nw0 = nrptg == 1 ? (nk0 + nth - 1)/nth : 1;

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         2);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, nw0*(ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);

    return 1;
}

int wsp_ggml_metal_op_pool_2d(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t * opts = op->op_params;
    wsp_ggml_op_pool op_pool = (wsp_ggml_op_pool) opts[0];

    const int32_t k0 = opts[1];
    const int32_t k1 = opts[2];
    const int32_t s0 = opts[3];
    const int32_t s1 = opts[4];
    const int32_t p0 = opts[5];
    const int32_t p1 = opts[6];

    const int64_t IH = op->src[0]->ne[1];
    const int64_t IW = op->src[0]->ne[0];

    const int64_t N  = op->ne[3];
    const int64_t OC = op->ne[2];
    const int64_t OH = op->ne[1];
    const int64_t OW = op->ne[0];

    const int64_t np = N * OC * OH * OW;

    wsp_ggml_metal_kargs_pool_2d args_pool_2d = {
        /* .k0 = */ k0,
        /* .k1 = */ k1,
        /* .s0 = */ s0,
        /* .s1 = */ s1,
        /* .p0 = */ p0,
        /* .p1 = */ p1,
        /* .IH = */ IH,
        /* .IW = */ IW,
        /* .OH = */ OH,
        /* .OW = */ OW,
        /* .np = */ np
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_pool_2d(lib, op, op_pool);

    const int nth = std::min(wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) np);
    const int ntg = (np + nth - 1) / nth;

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args_pool_2d, sizeof(args_pool_2d), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         2);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_mul_mat(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    const wsp_ggml_metal_device_props * props_dev = wsp_ggml_metal_device_get_props(ctx->dev);

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    WSP_GGML_ASSERT(ne00 == ne10);

    WSP_GGML_ASSERT(ne12 % ne02 == 0);
    WSP_GGML_ASSERT(ne13 % ne03 == 0);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    // find the break-even point where the matrix-matrix kernel becomes more efficient compared
    // to the matrix-vector kernel
    const int ne11_mm_min = 8;

    // first try to use small-batch mat-mv kernels
    // these should be efficient for BS [2, ~8]
    if (op->src[1]->type == WSP_GGML_TYPE_F32 && (ne00%128 == 0) &&
        (
         (
          (
           op->src[0]->type == WSP_GGML_TYPE_F32  || // TODO: helper function
           op->src[0]->type == WSP_GGML_TYPE_F16  ||
           op->src[0]->type == WSP_GGML_TYPE_Q4_0 ||
           op->src[0]->type == WSP_GGML_TYPE_Q4_1 ||
           op->src[0]->type == WSP_GGML_TYPE_Q5_0 ||
           op->src[0]->type == WSP_GGML_TYPE_Q5_1 ||
           op->src[0]->type == WSP_GGML_TYPE_Q8_0 ||
           op->src[0]->type == WSP_GGML_TYPE_MXFP4 ||
           op->src[0]->type == WSP_GGML_TYPE_IQ4_NL ||
           false) && (ne11 >= 2 && ne11 <= 8)
         ) ||
         (
          (
           op->src[0]->type == WSP_GGML_TYPE_Q4_K ||
           op->src[0]->type == WSP_GGML_TYPE_Q5_K ||
           op->src[0]->type == WSP_GGML_TYPE_Q6_K ||
           false) && (ne11 >= 4 && ne11 <= 8)
         )
        )
       ) {
        // TODO: determine the optimal parameters based on grid utilization
        //       I still don't know why we should not always use the maximum available threads:
        //
        //       nsg = pipeline.maxTotalThreadsPerThreadgroup / 32
        //
        //       my current hypothesis is that the work grid is not evenly divisible for different nsg
        //       values and there can be some tail effects when nsg is high. need to confirm this
        //
        const int nsg    = 2;                 // num simdgroups per threadgroup

        // num threads along row per simdgroup
        int16_t nxpsg = 0;
        if (ne00 % 256 == 0 && ne11 < 3) {
            nxpsg = 16;
        } else if (ne00 % 128 == 0) {
            nxpsg = 8;
        } else {
            nxpsg = 4;
        }

        const int16_t nypsg  = 32/nxpsg;          // num threads along col per simdgroup (i.e. a simdgroup processes that many src0 rows at a time)
        const int16_t r0ptg  = nypsg*nsg;         // num src0 rows per threadgroup
              int16_t r1ptg  = 4;                 // num src1 rows per threadgroup

        // note: not sure how optimal are those across all different hardware. there might be someting cleverer
        switch (ne11) {
            case 2:
                r1ptg = 2; break;
            case 3:
            case 6:
                r1ptg = 3; break;
            case 4:
            case 7:
            case 8:
                r1ptg = 4; break;
            case 5:
                r1ptg = 5; break;
            default:
                WSP_GGML_ABORT("unsupported ne11");
        };

        wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_mul_mv_ext(lib, op->src[0]->type, op->src[1]->type, nsg, nxpsg, r1ptg);

        wsp_ggml_metal_kargs_mul_mv_ext args = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne10  =*/ ne10,
            /*.ne11  =*/ ne11,
            /*.ne12  =*/ ne12,
            /*.nb10  =*/ nb10,
            /*.nb11  =*/ nb11,
            /*.nb12  =*/ nb12,
            /*.nb13  =*/ nb13,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.r2    =*/ r2,
            /*.r3    =*/ r3,
        };

        wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
        wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
        wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[1]), 2);
        wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         3);

        wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + r0ptg - 1)/r0ptg), ((ne11 + r1ptg - 1)/r1ptg), ne12*ne13, 32, nsg, 1);
    } else if (
        !wsp_ggml_is_transposed(op->src[0]) &&
        !wsp_ggml_is_transposed(op->src[1]) &&
        // for now the matrix-matrix multiplication kernel only works on A14+/M1+ SoCs
        // AMD GPU and older A-chips will reuse matrix-vector multiplication kernel
        props_dev->has_simdgroup_mm && ne00 >= 64 && ne11 > ne11_mm_min) {
        //WSP_GGML_LOG_INFO("matrix: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case WSP_GGML_TYPE_F32:  WSP_GGML_ASSERT(nb01 % 16 == 0); break;
        //    case WSP_GGML_TYPE_F16:  WSP_GGML_ASSERT(nb01 % 8  == 0); break;
        //    case WSP_GGML_TYPE_BF16: WSP_GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_mul_mm(lib, op);

        wsp_ggml_metal_kargs_mul_mm args = {
            /*.ne00 =*/ ne00,
            /*.ne02 =*/ ne02,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
        };

        wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
        wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
        wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[1]), 2);
        wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         3);

        const size_t smem = wsp_ggml_metal_pipeline_get_smem(pipeline);

        wsp_ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ((ne11 + 31)/32), ((ne01 + 63)/64), ne12*ne13, 128, 1, 1);
    } else {
        wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_mul_mv(lib, op);

        const int nr0 = wsp_ggml_metal_pipeline_get_nr0(pipeline);
        const int nr1 = wsp_ggml_metal_pipeline_get_nr1(pipeline);
        const int nsg = wsp_ggml_metal_pipeline_get_nsg(pipeline);

        const size_t smem = wsp_ggml_metal_pipeline_get_smem(pipeline);

        wsp_ggml_metal_kargs_mul_mv args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nr0  =*/ nr0,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
        };

        wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
        wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
        wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[1]), 2);
        wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         3);

        wsp_ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        if (op->src[0]->type == WSP_GGML_TYPE_F32 ||
            op->src[0]->type == WSP_GGML_TYPE_F16 ||
            op->src[0]->type == WSP_GGML_TYPE_BF16 ||
            op->src[0]->type == WSP_GGML_TYPE_Q8_0) {
            wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0 - 1)/(nr0)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
        } else {
            wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0*nsg - 1)/(nr0*nsg)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
        }
    }

    return 1;
}

size_t wsp_ggml_metal_op_mul_mat_id_extra_tpe(const wsp_ggml_tensor * op) {
    assert(op->op == WSP_GGML_OP_MUL_MAT_ID);

    const int64_t ne02 = op->src[0]->ne[2]; // n_expert

    return wsp_ggml_type_size(WSP_GGML_TYPE_I32)*ne02;
}

size_t wsp_ggml_metal_op_mul_mat_id_extra_ids(const wsp_ggml_tensor * op) {
    assert(op->op == WSP_GGML_OP_MUL_MAT_ID);

    const int64_t ne02 = op->src[0]->ne[2]; // n_expert
    const int64_t ne21 = op->src[2]->ne[1]; // n_token

    return wsp_ggml_type_size(WSP_GGML_TYPE_I32)*ne02*ne21;
}

int wsp_ggml_metal_op_mul_mat_id(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    const wsp_ggml_metal_device_props * props_dev = wsp_ggml_metal_device_get_props(ctx->dev);

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // src2 = ids
    WSP_GGML_ASSERT(op->src[2]->type == WSP_GGML_TYPE_I32);

    WSP_GGML_ASSERT(!wsp_ggml_is_transposed(op->src[0]));
    WSP_GGML_ASSERT(!wsp_ggml_is_transposed(op->src[1]));

    WSP_GGML_ASSERT(ne03 == 1);
    WSP_GGML_ASSERT(ne13 == 1);

    wsp_ggml_metal_buffer_id bid_src0 = wsp_ggml_metal_get_buffer_id(op->src[0]);
    wsp_ggml_metal_buffer_id bid_src1 = wsp_ggml_metal_get_buffer_id(op->src[1]);
    wsp_ggml_metal_buffer_id bid_src2 = wsp_ggml_metal_get_buffer_id(op->src[2]);
    wsp_ggml_metal_buffer_id bid_dst  = wsp_ggml_metal_get_buffer_id(op);

    const uint32_t r2 = 1;
    const uint32_t r3 = 1;

    // find the break-even point where the matrix-matrix kernel becomes more efficient compared
    // to the matrix-vector kernel
    // ne20 = n_used_experts
    // ne21 = n_rows (batch size)
    const int ne21_mm_id_min = 32;

    if (props_dev->has_simdgroup_mm && ne00 >= 64 && (ne21 >= ne21_mm_id_min)) {
        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case WSP_GGML_TYPE_F32:  WSP_GGML_ASSERT(nb01 % 16 == 0); break;
        //    case WSP_GGML_TYPE_F16:  WSP_GGML_ASSERT(nb01 % 8  == 0); break;
        //    case WSP_GGML_TYPE_BF16: WSP_GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        // extra buffers for intermediate id mapping
        wsp_ggml_metal_buffer_id bid_tpe = bid_dst;
        bid_tpe.offs += wsp_ggml_nbytes(op);

        wsp_ggml_metal_buffer_id bid_ids = bid_tpe;
        bid_ids.offs += wsp_ggml_metal_op_mul_mat_id_extra_tpe(op);

        {
            wsp_ggml_metal_kargs_mul_mm_id_map0 args = {
                ne02,
                ne10,
                ne11, // n_expert_used (bcast)
                nb11,
                nb12,
                ne21, // n_tokens
                ne20, // n_expert_used
                nb21,
            };

            wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_mul_mm_id_map0(lib, ne02, ne20);

            const size_t smem = wsp_ggml_metal_pipeline_get_smem(pipeline);

            WSP_GGML_ASSERT(ne02 <= wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

            WSP_GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

            wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
            wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_src2, 1);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_tpe,  2);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_ids,  3);

            wsp_ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            wsp_ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, ne02, 1, 1);
        }

        // this barrier is always needed because the next kernel has to wait for the id maps to be computed
        wsp_ggml_metal_op_concurrency_reset(ctx);

        {
            wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_mul_mm_id(lib, op);

            wsp_ggml_metal_kargs_mul_mm_id args = {
                /*.ne00  =*/ ne00,
                /*.ne02  =*/ ne02,
                /*.nb01  =*/ nb01,
                /*.nb02  =*/ nb02,
                /*.nb03  =*/ nb03,
                /*.ne11  =*/ ne11, // n_expert_used (bcast)
                /*.nb10  =*/ nb10,
                /*.nb11  =*/ nb11,
                /*.nb12  =*/ nb12,
                /*.nb13  =*/ nb13,
                /*.ne20  =*/ ne20, // n_expert_used
                /*.ne21  =*/ ne21, // n_tokens
                /*.ne0   =*/ ne0,
                /*.ne1   =*/ ne1,
                /*.r2    =*/ r2,
                /*.r3    =*/ r3,
            };

            wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
            wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_tpe,  3);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_ids,  4);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_dst,  5);

            const size_t smem = wsp_ggml_metal_pipeline_get_smem(pipeline);

            wsp_ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            wsp_ggml_metal_encoder_dispatch_threadgroups(enc, (ne21 + 31)/32, (ne01 + 63)/64, ne02, 128, 1, 1);
        }
    } else {
        wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_mul_mv_id(lib, op);

        const int nr0 = wsp_ggml_metal_pipeline_get_nr0(pipeline);
        const int nr1 = wsp_ggml_metal_pipeline_get_nr1(pipeline);
        const int nsg = wsp_ggml_metal_pipeline_get_nsg(pipeline);

        const size_t smem = wsp_ggml_metal_pipeline_get_smem(pipeline);

        wsp_ggml_metal_kargs_mul_mv_id args = {
            /*.nei0 =*/ ne20,
            /*.nei1 =*/ ne21,
            /*.nbi1 =*/ nb21,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.ne13 =*/ ne13,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nb1  =*/ nb1,
            /*.nr0  =*/ nr0,
        };

        if (wsp_ggml_is_quantized(op->src[0]->type)) {
            WSP_GGML_ASSERT(ne00 >= nsg*nr0);
        }

        wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
        wsp_ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        wsp_ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        wsp_ggml_metal_encoder_set_buffer(enc, bid_src1, 2);
        wsp_ggml_metal_encoder_set_buffer(enc, bid_dst,  3);
        wsp_ggml_metal_encoder_set_buffer(enc, bid_src2, 4);

        const int64_t _ne1 = 1;
        const int64_t ne123 = ne20*ne21;

        wsp_ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        if (op->src[0]->type == WSP_GGML_TYPE_F32 ||
            op->src[0]->type == WSP_GGML_TYPE_F16 ||
            op->src[0]->type == WSP_GGML_TYPE_BF16 ||
            op->src[0]->type == WSP_GGML_TYPE_Q8_0) {
            wsp_ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0 - 1)/(nr0), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        } else {
            wsp_ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0*nsg - 1)/(nr0*nsg), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        }
    }

    return 1;
}

int wsp_ggml_metal_op_add_id(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    WSP_GGML_ASSERT(op->src[0]->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT(op->src[1]->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT(op->src[2]->type == WSP_GGML_TYPE_I32);
    WSP_GGML_ASSERT(op->type         == WSP_GGML_TYPE_F32);

    WSP_GGML_ASSERT(wsp_ggml_is_contiguous_rows(op->src[0]));

    wsp_ggml_metal_kargs_add_id args = {
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb11 =*/ nb11,
        /*.nb21 =*/ nb21,
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_base(lib, WSP_GGML_OP_ADD_ID);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[1]), 2);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[2]), 3);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         4);

    const int nth = std::min(wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, 1, nth, 1, 1);

    return 1;
}

bool wsp_ggml_metal_op_flash_attn_ext_use_vec(const wsp_ggml_tensor * op) {
    assert(op->op == WSP_GGML_OP_FLASH_ATTN_EXT);

    const int64_t ne00 = op->src[0]->ne[0]; // head size
    const int64_t ne01 = op->src[0]->ne[1]; // batch size

    // use vec kernel if the batch size is small and if the head size is supported
    return (ne01 < 20) && (ne00 % 32 == 0);
}

size_t wsp_ggml_metal_op_flash_attn_ext_extra_pad(const wsp_ggml_tensor * op) {
    assert(op->op == WSP_GGML_OP_FLASH_ATTN_EXT);

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    const bool has_mask = op->src[3] != nullptr;

    if (wsp_ggml_metal_op_flash_attn_ext_use_vec(op)) {
        // note: always reserve the padding space to avoid graph reallocations
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_VEC_NCPSG != 0;
        const bool has_kvpad = true;

        if (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_VEC_NCPSG*(
                nb11*ne12*ne13 +
                nb21*ne22*ne23 +
                (has_mask ? wsp_ggml_type_size(WSP_GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    } else {
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_NCPSG != 0;
        const bool has_kvpad = true;

        if (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_NCPSG*(
                nb11*ne12*ne13 +
                nb21*ne22*ne23 +
                (has_mask ? wsp_ggml_type_size(WSP_GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    }

    return res;
}

size_t wsp_ggml_metal_op_flash_attn_ext_extra_blk(const wsp_ggml_tensor * op) {
    assert(op->op == WSP_GGML_OP_FLASH_ATTN_EXT);

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
  //WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
  //WSP_GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
  //WSP_GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
  //WSP_GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
  //WSP_GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    const bool has_mask = op->src[3] != nullptr;

    if (!has_mask) {
        return res;
    }

    const bool is_vec = wsp_ggml_metal_op_flash_attn_ext_use_vec(op);

    // this optimization is not useful for the vector kernels
    // note: always reserve the blk buffer to avoid graph reallocations
    //if (is_vec) {
    //    return res;
    //}

    const int nqptg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NQPTG : OP_FLASH_ATTN_EXT_NQPTG;
    const int ncpsg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NCPSG : OP_FLASH_ATTN_EXT_NCPSG;

    const int64_t ne1 = (ne01 + nqptg - 1)/nqptg;
    const int64_t ne0 = (ne30 + ncpsg - 1)/ncpsg;

    res += WSP_GGML_PAD(wsp_ggml_type_size(WSP_GGML_TYPE_I8)*ne0*ne1*ne32*ne33, 32);

    return res;
}

size_t wsp_ggml_metal_op_flash_attn_ext_extra_tmp(const wsp_ggml_tensor * op) {
    assert(op->op == WSP_GGML_OP_FLASH_ATTN_EXT);

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
  //WSP_GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
  //WSP_GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
  //WSP_GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
  //WSP_GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    // note: always reserve the temp buffer to avoid graph reallocations
    //if (wsp_ggml_metal_op_flash_attn_ext_use_vec(op)) {
    if (true) {
        const int64_t nwg = 32;
        const int64_t ne01_max = std::min(ne01, 32);

        // temp buffer for writing the results from each workgroup
        // - ne20: the size of the Value head
        // -  + 2: the S and M values for each intermediate result
        res += wsp_ggml_type_size(WSP_GGML_TYPE_F32)*(ne01_max*ne02*ne03*nwg*(ne20 + 2));
    }

    return res;
}

int wsp_ggml_metal_op_flash_attn_ext(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    const wsp_ggml_metal_device_props * props_dev = wsp_ggml_metal_device_get_props(ctx->dev);

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS( int32_t, nb,  op,         nb);

    WSP_GGML_ASSERT(ne00 % 4 == 0);

    WSP_GGML_ASSERT(op->src[0]->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT(op->src[1]->type == op->src[2]->type);

    //WSP_GGML_ASSERT(wsp_ggml_are_same_shape (src1, src2));
    WSP_GGML_ASSERT(ne11 == ne21);
    WSP_GGML_ASSERT(ne12 == ne22);

    WSP_GGML_ASSERT(!op->src[3] || op->src[3]->type == WSP_GGML_TYPE_F16);
    WSP_GGML_ASSERT(!op->src[3] || op->src[3]->ne[1] >= op->src[0]->ne[1] &&
            "the Flash-Attention Metal kernel requires the mask to be at least n_queries big");

    float scale;
    float max_bias;
    float logit_softcap;

    memcpy(&scale,         ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias,      ((const int32_t *) op->op_params) + 1, sizeof(max_bias));
    memcpy(&logit_softcap, ((const int32_t *) op->op_params) + 2, sizeof(logit_softcap));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const bool has_mask  = op->src[3] != NULL;
    const bool has_sinks = op->src[4] != NULL;
    const bool has_bias  = max_bias != 0.0f;
    const bool has_scap  = logit_softcap != 0.0f;

    const uint32_t n_head      = op->src[0]->ne[2];
    const  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    WSP_GGML_ASSERT(ne01 < 65536);

    wsp_ggml_metal_buffer_id bid_src0 = wsp_ggml_metal_get_buffer_id(op->src[0]);
    wsp_ggml_metal_buffer_id bid_src1 = wsp_ggml_metal_get_buffer_id(op->src[1]);
    wsp_ggml_metal_buffer_id bid_src2 = wsp_ggml_metal_get_buffer_id(op->src[2]);
    wsp_ggml_metal_buffer_id bid_src3 = has_mask  ? wsp_ggml_metal_get_buffer_id(op->src[3]) : bid_src0;
    wsp_ggml_metal_buffer_id bid_src4 = has_sinks ? wsp_ggml_metal_get_buffer_id(op->src[4]) : bid_src0;

    wsp_ggml_metal_buffer_id bid_dst = wsp_ggml_metal_get_buffer_id(op);

    wsp_ggml_metal_buffer_id bid_pad = bid_dst;
    bid_pad.offs += wsp_ggml_nbytes(op);

    wsp_ggml_metal_buffer_id bid_blk = bid_pad;
    bid_blk.offs += wsp_ggml_metal_op_flash_attn_ext_extra_pad(op);

    wsp_ggml_metal_buffer_id bid_tmp = bid_blk;
    bid_tmp.offs += wsp_ggml_metal_op_flash_attn_ext_extra_blk(op);

    if (!wsp_ggml_metal_op_flash_attn_ext_use_vec(op)) {
        // half8x8 kernel
        const int nqptg = OP_FLASH_ATTN_EXT_NQPTG; // queries per threadgroup
        const int ncpsg = OP_FLASH_ATTN_EXT_NCPSG; // cache values per simdgroup

        WSP_GGML_ASSERT(nqptg <= 32);
        WSP_GGML_ASSERT(nqptg  % 8  == 0);
        WSP_GGML_ASSERT(ncpsg  % 32 == 0);

        bool need_sync = false;

        const bool has_kvpad = ne11 % ncpsg != 0;

        if (has_kvpad) {
            assert(wsp_ggml_metal_op_flash_attn_ext_extra_pad(op) != 0);

            wsp_ggml_metal_kargs_flash_attn_ext_pad args0 = {
                /*.ne11    =*/ne11,
                /*.ne_12_2 =*/ne12,
                /*.ne_12_3 =*/ne13,
                /*.nb11    =*/nb11,
                /*.nb12    =*/nb12,
                /*.nb13    =*/nb13,
                /*.nb21    =*/nb21,
                /*.nb22    =*/nb22,
                /*.nb23    =*/nb23,
                /*.ne31    =*/ne31,
                /*.ne32    =*/ne32,
                /*.ne33    =*/ne33,
                /*.nb31    =*/nb31,
                /*.nb32    =*/nb32,
                /*.nb33    =*/nb33,
            };

            wsp_ggml_metal_pipeline_t pipeline0 = wsp_ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            wsp_ggml_metal_encoder_set_pipeline(enc, pipeline0);
            wsp_ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_src2, 2);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_pad,  4);

            assert(ne12 == ne22);
            assert(ne13 == ne23);

            wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, std::max(ne12, ne32), std::max(ne13, ne33), 32, 1, 1);

            need_sync = true;
        }

        if (has_mask) {
            assert(wsp_ggml_metal_op_flash_attn_ext_extra_blk(op) != 0);

            wsp_ggml_metal_kargs_flash_attn_ext_blk args0 = {
                /*.ne01 =*/ ne01,
                /*.ne30 =*/ ne30,
                /*.ne31 =*/ ne31,
                /*.ne32 =*/ ne32,
                /*.ne33 =*/ ne33,
                /*.nb31 =*/ nb31,
                /*.nb32 =*/ nb32,
                /*.nb33 =*/ nb33,
            };

            wsp_ggml_metal_pipeline_t pipeline0 = wsp_ggml_metal_library_get_pipeline_flash_attn_ext_blk(lib, op, nqptg, ncpsg);

            wsp_ggml_metal_encoder_set_pipeline(enc, pipeline0);
            wsp_ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_src3, 1);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_blk,  2);

            const int32_t nblk1 = ((ne01 + nqptg - 1)/nqptg);
            const int32_t nblk0 = ((ne30 + ncpsg - 1)/ncpsg);

            wsp_ggml_metal_encoder_dispatch_threadgroups(enc, nblk0, nblk1, ne32*ne33, 32, 1, 1);

            need_sync = true;
        }

        if (need_sync) {
            wsp_ggml_metal_op_concurrency_reset(ctx);
        }

        const int is_q = wsp_ggml_is_quantized(op->src[1]->type) ? 1 : 0;

        // 2*(2*ncpsg)
        // ncpsg soft_max values + ncpsg mask values
        //
        // 16*32*(nsg)
        // the shared memory needed for the simdgroups to load the KV cache
        // each thread loads (dequantizes) 16 head elements, there are 32 threads in th SG
        //
#define FATTN_SMEM(nsg) (WSP_GGML_PAD((nqptg*(ne00 + 2*WSP_GGML_PAD(ne20, 64) + 2*(2*ncpsg)) + is_q*(16*32*(nsg)))*(sizeof(float)/2), 16))

        //int64_t nsgmax = 4;
        //
        //if (is_q) {
        //    nsgmax = 2;
        //    while (true) {
        //        const size_t smem = FATTN_SMEM(nsgmax);
        //        if (smem > props_dev->max_theadgroup_memory_size) {
        //            break;
        //        }
        //        nsgmax *= 2;
        //    }
        //    nsgmax /= 2;
        //}

        // simdgroups per threadgroup (a.k.a. warps)
        //nsg = ne01 <= nqptg ? MAX(4, MIN(nsgmax, MIN(ne11/ncpsg, (int64_t) pipeline.maxTotalThreadsPerThreadgroup/32))) : 4;
        int32_t nsg = 4;

        const size_t smem = FATTN_SMEM(nsg);

        wsp_ggml_metal_kargs_flash_attn_ext args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ int32_t(nb11/nb10),
            /*.nb11          =*/ nb11,
            /*.nb12          =*/ nb12,
            /*.nb13          =*/ nb13,
            /*.ns20          =*/ int32_t(nb21/nb20),
            /*.nb21          =*/ nb21,
            /*.nb22          =*/ nb22,
            /*.nb23          =*/ nb23,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_flash_attn_ext(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, nsg);

        wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
        wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        wsp_ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        wsp_ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
        wsp_ggml_metal_encoder_set_buffer  (enc, bid_src2, 3);
        wsp_ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        wsp_ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
        wsp_ggml_metal_encoder_set_buffer  (enc, bid_pad,  6);
        wsp_ggml_metal_encoder_set_buffer  (enc, bid_blk,  7);
        wsp_ggml_metal_encoder_set_buffer  (enc, bid_dst,  8);

        wsp_ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        wsp_ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02, ne03, 32, nsg, 1);
#undef FATTN_SMEM
    } else {
        // half4x4 kernel
        const int nqptg = OP_FLASH_ATTN_EXT_VEC_NQPTG; // queries per threadgroup
        const int ncpsg = OP_FLASH_ATTN_EXT_VEC_NCPSG; // cache values per simdgroup !! sync with kernel template arguments !!
        const int nkpsg = 1*ncpsg;

        WSP_GGML_ASSERT(nqptg <= 32);
        WSP_GGML_ASSERT(nqptg  % 1  == 0);
        WSP_GGML_ASSERT(ncpsg  % 32 == 0);

        bool need_sync = false;

        const bool has_kvpad = ne11 % ncpsg != 0;

        if (has_kvpad) {
            assert(wsp_ggml_metal_op_flash_attn_ext_extra_pad(op) != 0);

            wsp_ggml_metal_kargs_flash_attn_ext_pad args0 = {
                /*.ne11    =*/ne11,
                /*.ne_12_2 =*/ne12,
                /*.ne_12_3 =*/ne13,
                /*.nb11    =*/nb11,
                /*.nb12    =*/nb12,
                /*.nb13    =*/nb13,
                /*.nb21    =*/nb21,
                /*.nb22    =*/nb22,
                /*.nb23    =*/nb23,
                /*.ne31    =*/ne31,
                /*.ne32    =*/ne32,
                /*.ne33    =*/ne33,
                /*.nb31    =*/nb31,
                /*.nb32    =*/nb32,
                /*.nb33    =*/nb33,
            };

            wsp_ggml_metal_pipeline_t pipeline0 = wsp_ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            wsp_ggml_metal_encoder_set_pipeline(enc, pipeline0);
            wsp_ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_src2, 2);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);
            wsp_ggml_metal_encoder_set_buffer  (enc, bid_pad,  4);

            assert(ne12 == ne22);
            assert(ne13 == ne23);

            wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, std::max(ne12, ne32), std::max(ne13, ne33), 32, 1, 1);

            need_sync = true;
        }

        if (need_sync) {
            wsp_ggml_metal_op_concurrency_reset(ctx);
        }

        // ne00 + 2*ncpsg*(nsg)
        // for each query, we load it as f16 in shared memory (ne00)
        // and store the soft_max values and the mask
        //
        // ne20*(nsg)
        // each simdgroup has a full f32 head vector in shared mem to accumulate results
        //
#define FATTN_SMEM(nsg) (WSP_GGML_PAD((nqptg*(WSP_GGML_PAD(ne00, 128) + 4*ncpsg*(nsg)) + 2*WSP_GGML_PAD(ne20, 128)*(nsg))*(sizeof(float)/2), 16))

        int64_t nsgmax = 2;
        while (true) {
            const size_t smem = FATTN_SMEM(nsgmax);
            // avoid using more than half of the threadgroup memory - can cause slow downs especially for large head sizes
            if (smem > props_dev->max_theadgroup_memory_size/2) {
                break;
            }
            nsgmax *= 2;
        }
        nsgmax /= 2;

        // simdgroups per threadgroup (a.k.a. warps)
        //const int64_t nsgt = MAX(2, MIN(nsgmax, MIN((ne11 + nkpsg - 1)/(nkpsg), (int64_t) pipeline.maxTotalThreadsPerThreadgroup/32)));
        const int64_t nsgt = MAX(2, MIN(nsgmax, MIN((ne11 + nkpsg - 1)/(nkpsg), (int64_t) 1024/32)));

        int64_t nsg = 1;
        while (nsg <= nsgt) {
            nsg *= 2;
        }
        nsg /= 2;

        // workgroups
        // each workgroup handles nsg*nkpsg cache values
        int32_t nwg = 1;
        if (false) {
            // for small KV caches, we could launch a single workgroup and write the results directly to dst/
            // however, this does not lead to significant improvement, so disabled
            nwg = 1;
            nsg = 4;
        } else {
            nwg = 32;
            nsg = 1;
            while (2*nwg*nsg*nkpsg < ne11 && nsg < 4) {
                nsg *= 2;
            }
        }

        wsp_ggml_metal_kargs_flash_attn_ext_vec args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ int32_t(nb11/nb10),
            /*.nb11          =*/ nb11,
            /*.nb12          =*/ nb12,
            /*.nb13          =*/ nb13,
            /*.ns20          =*/ int32_t(nb21/nb20),
            /*.nb21          =*/ nb21,
            /*.nb22          =*/ nb22,
            /*.nb23          =*/ nb23,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_flash_attn_ext_vec(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, nsg, nwg);

        WSP_GGML_ASSERT(nsg*32 <= wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
        wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        wsp_ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        wsp_ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
        wsp_ggml_metal_encoder_set_buffer  (enc, bid_src2, 3);
        wsp_ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        wsp_ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);

        const size_t smem = FATTN_SMEM(nsg);

        //printf("smem: %zu, max: %zu, nsg = %d, nsgmax = %d\n", smem, props_dev->max_theadgroup_memory_size, (int) nsg, (int) nsgmax);
        WSP_GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

        if (nwg == 1) {
            assert(wsp_ggml_metal_op_flash_attn_ext_extra_tmp(op) == 0);

            // using 1 workgroup -> write the result directly into dst
            wsp_ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            wsp_ggml_metal_encoder_set_buffer(enc, bid_dst, 7);

            wsp_ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            wsp_ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02, ne03*nwg, 32, nsg, 1);
        } else {
            // sanity checks
            assert(wsp_ggml_metal_op_flash_attn_ext_extra_tmp(op) != 0);

            WSP_GGML_ASSERT(ne01*ne02*ne03 == ne1*ne2*ne3);
            WSP_GGML_ASSERT((uint64_t)ne1*ne2*ne3 <= (1u << 31));

            // write the results from each workgroup into a temp buffer
            wsp_ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            wsp_ggml_metal_encoder_set_buffer(enc, bid_tmp, 7);

            wsp_ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
            wsp_ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02, ne03*nwg, 32, nsg, 1);

            // sync the 2 kernels
            wsp_ggml_metal_op_concurrency_reset(ctx);

            // reduce the results from the workgroups
            {
                const int32_t nrows = ne1*ne2*ne3;

                wsp_ggml_metal_kargs_flash_attn_ext_vec_reduce args0 = {
                    nrows,
                };

                wsp_ggml_metal_pipeline_t pipeline0 = wsp_ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(lib, op, ne20, nwg);

                wsp_ggml_metal_encoder_set_pipeline(enc, pipeline0);
                wsp_ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
                wsp_ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
                wsp_ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

                wsp_ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, 32*nwg, 1, 1);
            }
        }
#undef FATTN_SMEM
    }

    return 1;
}

int wsp_ggml_metal_op_bin(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    const bool use_fusion = ctx->use_fusion;

    const int debug_fusion = ctx->debug_fusion;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    WSP_GGML_ASSERT(op->src[0]->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT(op->src[1]->type == WSP_GGML_TYPE_F32);

    WSP_GGML_ASSERT(wsp_ggml_is_contiguous_rows(op->src[0]));
    WSP_GGML_ASSERT(wsp_ggml_is_contiguous_rows(op->src[1]));

    bool bcast_row = false;

    wsp_ggml_metal_buffer_id bid_src0 = wsp_ggml_metal_get_buffer_id(op->src[0]);
    wsp_ggml_metal_buffer_id bid_src1 = wsp_ggml_metal_get_buffer_id(op->src[1]);
    wsp_ggml_metal_buffer_id bid_dst  = wsp_ggml_metal_get_buffer_id(op);

    wsp_ggml_metal_kargs_bin args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.offs =*/ 0,
        /*.o1   =*/ { bid_src1.offs },
    };

    wsp_ggml_op fops[8];

    int n_fuse = 1;

    // c[0] = add(a,    b[0])
    // c[1] = add(c[0], b[1])
    // c[2] = add(c[1], b[2])
    // ...
    if (use_fusion) {
        fops[0] = WSP_GGML_OP_ADD;
        fops[1] = WSP_GGML_OP_ADD;
        fops[2] = WSP_GGML_OP_ADD;
        fops[3] = WSP_GGML_OP_ADD;
        fops[4] = WSP_GGML_OP_ADD;
        fops[5] = WSP_GGML_OP_ADD;
        fops[6] = WSP_GGML_OP_ADD;
        fops[7] = WSP_GGML_OP_ADD;

        // note: in metal, we sometimes encode the graph in parallel so we have to avoid fusing ops
        //       across splits. idx_end indicates the last node in the current split
        for (n_fuse = 0; n_fuse <= 6; ++n_fuse) {
            if (!ctx->can_fuse(idx + n_fuse, fops + n_fuse, 2)) {
                break;
            }

            wsp_ggml_tensor * f0 = ctx->node(idx + n_fuse);
            wsp_ggml_tensor * f1 = ctx->node(idx + n_fuse + 1);

            if (f0 != f1->src[0]) {
                break;
            }

            // b[0] === b[1] === ...
            if (!wsp_ggml_are_same_layout(f0->src[1], f1->src[1])) {
                break;
            }

            // only fuse ops if src1 is in the same Metal buffer
            wsp_ggml_metal_buffer_id bid_fuse = wsp_ggml_metal_get_buffer_id(f1->src[1]);
            if (bid_fuse.metal != bid_src1.metal) {
                break;
            }

            //ctx->fuse_cnt[ops[n_fuse + 1]->op]++;

            args.o1[n_fuse + 1] = bid_fuse.offs;
        }

        ++n_fuse;

        if (debug_fusion > 1 && n_fuse > 1) {
            WSP_GGML_LOG_DEBUG("%s: fuse: ADD x %d\n", __func__, n_fuse);
        }
    }

    // the offsets of src1 and all fused buffers are relative to the start of the src1 buffer
    bid_src1.offs = 0;

    wsp_ggml_metal_pipeline_t pipeline = nullptr;

    if (wsp_ggml_nelements(op->src[1]) == ne10 && wsp_ggml_is_contiguous(op->src[1]) && ne00 % 4 == 0 && ne10 % 4 == 0) {
        WSP_GGML_ASSERT(wsp_ggml_is_contiguous(op->src[0]));

        // src1 is a row
        WSP_GGML_ASSERT(ne11 == 1);

        pipeline = wsp_ggml_metal_library_get_pipeline_bin(lib, op->op, n_fuse, true);

        bcast_row = true;
    } else {
        pipeline = wsp_ggml_metal_library_get_pipeline_bin(lib, op->op, n_fuse, false);
    }

    if (n_fuse > 1) {
        bid_dst = wsp_ggml_metal_get_buffer_id(ctx->node(idx + n_fuse - 1));

        for (int i = 1; i < n_fuse; ++i) {
            if (!wsp_ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
                wsp_ggml_metal_op_concurrency_reset(ctx);

                break;
            }
        }
    }

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
    wsp_ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);

    if (bcast_row) {
        const int64_t n = wsp_ggml_nelements(op)/4;

        wsp_ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, 1, 1, 1);
    } else {
        int nth = 32;

        while (16*nth < ne0 && nth < wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
            nth *= 2;
        }

        wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
    }

    return n_fuse;
}

int wsp_ggml_metal_op_l2_norm(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    int nth = 32; // SIMD width

    wsp_ggml_metal_kargs_l2_norm args = {
        /*.ne00   =*/ ne00,
        /*.ne00_4 =*/ ne00/4,
        /*.nb01   =*/ nb01,
        /*.eps    =*/ eps,
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_l2_norm(lib, op);

    while (nth < ne00/4 && nth < wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, ne00/4);

    const size_t smem = wsp_ggml_metal_pipeline_get_smem(pipeline);

    const int64_t nrows = wsp_ggml_nrows(op->src[0]);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         2);

    wsp_ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_group_norm(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t ngrp = ((const int32_t *) op->op_params)[0];

    float eps;
    memcpy(&eps, op->op_params + 1, sizeof(float));

    wsp_ggml_metal_kargs_group_norm args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ngrp =*/ ngrp,
        /*.eps  =*/ eps,
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_group_norm(lib, op);

    int nth = 32; // SIMD width
    //while (nth < ne00/4 && nth < wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
    //    nth *= 2;
    //}

    //nth = std::min(nth, wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    //nth = std::min(nth, ne00/4);

    const size_t smem = wsp_ggml_metal_pipeline_get_smem(pipeline);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         2);

    wsp_ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ngrp, 1, 1, nth, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_norm(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    const bool use_fusion = ctx->use_fusion;

    const int debug_fusion = ctx->debug_fusion;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    wsp_ggml_metal_buffer_id bid_src0 = wsp_ggml_metal_get_buffer_id(op->src[0]);
    wsp_ggml_metal_buffer_id bid_dst  = wsp_ggml_metal_get_buffer_id(op);

    wsp_ggml_metal_kargs_norm args = {
        /*.ne00   =*/ ne00,
        /*.ne00_t =*/ ne00 % 4 == 0 ? ne00/4 : ne00,
        /*.nb1    =*/ nb1,
        /*.nb2    =*/ nb2,
        /*.nb3    =*/ nb3,
        /*.eps    =*/ eps,
        /*.nef1   =*/ { ne01 },
        /*.nef2   =*/ { ne02 },
        /*.nef3   =*/ { ne03 },
        /*.nbf1   =*/ { nb01 },
        /*.nbf2   =*/ { nb02 },
        /*.nbf3   =*/ { nb03 },
    };

    wsp_ggml_op fops[8];

    int n_fuse = 1;

    wsp_ggml_metal_buffer_id bid_fuse[2] = { bid_src0, bid_src0 };

    // d[0] = norm(a)
    // d[1] = mul(d[0], b)
    // d[2] = add(d[1], c)
    if (use_fusion) {
        fops[0] = op->op;
        fops[1] = WSP_GGML_OP_MUL;
        fops[2] = WSP_GGML_OP_ADD;

        for (n_fuse = 0; n_fuse <= 1; ++n_fuse) {
            if (!ctx->can_fuse(idx + n_fuse, fops + n_fuse, 2)) {
                break;
            }

            wsp_ggml_tensor * f0 = ctx->node(idx + n_fuse);
            wsp_ggml_tensor * f1 = ctx->node(idx + n_fuse + 1);

            if (f0 != f1->src[0]) {
                break;
            }

            if (f1->src[1]->ne[0] != op->ne[0]) {
                break;
            }

            if (!wsp_ggml_is_contiguous_rows(f1->src[1])) {
                break;
            }

            if (f1->type != WSP_GGML_TYPE_F32) {
                break;
            }

            //ctx->fuse_cnt[f1->op]++;

            bid_fuse[n_fuse] = wsp_ggml_metal_get_buffer_id(f1->src[1]);

            args.nef1[n_fuse + 1] = f1->src[1]->ne[1];
            args.nef2[n_fuse + 1] = f1->src[1]->ne[2];
            args.nef3[n_fuse + 1] = f1->src[1]->ne[3];

            args.nbf1[n_fuse + 1] = f1->src[1]->nb[1];
            args.nbf2[n_fuse + 1] = f1->src[1]->nb[2];
            args.nbf3[n_fuse + 1] = f1->src[1]->nb[3];
        }

        ++n_fuse;

        if (debug_fusion > 1 && n_fuse > 1) {
            if (n_fuse == 2) {
                WSP_GGML_LOG_DEBUG("%s: fuse: %s + MUL\n", __func__, wsp_ggml_op_name(op->op));
            }
            if (n_fuse == 3) {
                WSP_GGML_LOG_DEBUG("%s: fuse: %s + MUL + ADD\n", __func__, wsp_ggml_op_name(op->op));
            }
        }
    }

    if (n_fuse > 1) {
        bid_dst = wsp_ggml_metal_get_buffer_id(ctx->node(idx + n_fuse - 1));

        for (int i = 1; i < n_fuse; ++i) {
            if (!wsp_ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
                wsp_ggml_metal_op_concurrency_reset(ctx);

                break;
            }
        }
    }

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_norm(lib, op, n_fuse);

    int nth = 32; // SIMD width

    while (nth < args.ne00_t && nth < wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, args.ne00_t);

    const size_t smem = wsp_ggml_metal_pipeline_get_smem(pipeline);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, bid_src0,    1);
    wsp_ggml_metal_encoder_set_buffer  (enc, bid_fuse[0], 2);
    wsp_ggml_metal_encoder_set_buffer  (enc, bid_fuse[1], 3);
    wsp_ggml_metal_encoder_set_buffer  (enc, bid_dst,     4);

    wsp_ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return n_fuse;
}

int wsp_ggml_metal_op_rope(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // make sure we have one or more position id(ne10) per token(ne02)
    WSP_GGML_ASSERT(ne10 % ne02 == 0);
    WSP_GGML_ASSERT(ne10 >= ne02);

    const int nth = std::min(1024, ne00);

    const int n_past     = ((const int32_t *) op->op_params)[0];
    const int n_dims     = ((const int32_t *) op->op_params)[1];
  //const int mode       = ((const int32_t *) op->op_params)[2];
    // skip 3, n_ctx, used in GLM RoPE, unimplemented in metal
    const int n_ctx_orig = ((const int32_t *) op->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (const int32_t *) op->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) op->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) op->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) op->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) op->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) op->op_params + 10, sizeof(float));

    // mrope
    const int sect_0 = ((const int32_t *) op->op_params)[11];
    const int sect_1 = ((const int32_t *) op->op_params)[12];
    const int sect_2 = ((const int32_t *) op->op_params)[13];
    const int sect_3 = ((const int32_t *) op->op_params)[14];

    wsp_ggml_metal_kargs_rope args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.ne03        =*/ ne03,
        /*.nb00        =*/ nb00,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne0         =*/ ne0,
        /*.ne1         =*/ ne1,
        /*.ne2         =*/ ne2,
        /*.ne3         =*/ ne3,
        /*.nb0         =*/ nb0,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.n_past      =*/ n_past,
        /*.n_dims      =*/ n_dims,
        /*.n_ctx_orig  =*/ n_ctx_orig,
        /*.freq_base   =*/ freq_base,
        /*.freq_scale  =*/ freq_scale,
        /*.ext_factor  =*/ ext_factor,
        /*.attn_factor =*/ attn_factor,
        /*.beta_fast   =*/ beta_fast,
        /*.beta_slow   =*/ beta_slow,
        /* sect_0      =*/ sect_0,
        /* sect_1      =*/ sect_1,
        /* sect_2      =*/ sect_2,
        /* sect_3      =*/ sect_3,
        /* src2        =*/ op->src[2] != nullptr,
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_rope(lib, op);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[1]), 2);
    if (op->src[2]) {
        wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[2]), 3);
    } else {
        wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         4);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_im2col(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];
    const int32_t s1 = ((const int32_t *)(op->op_params))[1];
    const int32_t p0 = ((const int32_t *)(op->op_params))[2];
    const int32_t p1 = ((const int32_t *)(op->op_params))[3];
    const int32_t d0 = ((const int32_t *)(op->op_params))[4];
    const int32_t d1 = ((const int32_t *)(op->op_params))[5];

    const bool is_2D = ((const int32_t *)(op->op_params))[6] == 1;

    const int32_t N  = op->src[1]->ne[is_2D ? 3 : 2];
    const int32_t IC = op->src[1]->ne[is_2D ? 2 : 1];
    const int32_t IH = is_2D ? op->src[1]->ne[1] : 1;
    const int32_t IW =         op->src[1]->ne[0];

    const int32_t KH = is_2D ? op->src[0]->ne[1] : 1;
    const int32_t KW =         op->src[0]->ne[0];

    const int32_t OH = is_2D ? op->ne[2] : 1;
    const int32_t OW =         op->ne[1];

    const int32_t CHW = IC * KH * KW;

    const uint64_t ofs0 = op->src[1]->nb[is_2D ? 3 : 2] / 4;
    const uint64_t ofs1 = op->src[1]->nb[is_2D ? 2 : 1] / 4;

    wsp_ggml_metal_kargs_im2col args = {
        /*.ofs0 =*/ ofs0,
        /*.ofs1 =*/ ofs1,
        /*.IW   =*/ IW,
        /*.IH   =*/ IH,
        /*.CHW  =*/ CHW,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
        /*.N    =*/ N,
        /*.KH   =*/ KH,
        /*.KW   =*/ KW,
        /*.KHW  =*/ KH * KW,
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_im2col(lib, op);

    WSP_GGML_ASSERT(KH*KW <= wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const uint64_t ntptg0 = std::min(wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)/(KH*KW), N);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[1]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         2);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, IC, OH, OW, ntptg0, KH, KW);

    return 1;
}

int wsp_ggml_metal_op_conv_2d(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    WSP_GGML_ASSERT(wsp_ggml_is_contiguous(op->src[0]));
    WSP_GGML_ASSERT(op->src[1]->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT(op->type == WSP_GGML_TYPE_F32);
    WSP_GGML_ASSERT(op->src[0]->type == WSP_GGML_TYPE_F16 || op->src[0]->type == WSP_GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t *) op->op_params)[0];
    const int32_t s1 = ((const int32_t *) op->op_params)[1];
    const int32_t p0 = ((const int32_t *) op->op_params)[2];
    const int32_t p1 = ((const int32_t *) op->op_params)[3];
    const int32_t d0 = ((const int32_t *) op->op_params)[4];
    const int32_t d1 = ((const int32_t *) op->op_params)[5];

    wsp_ggml_metal_kargs_conv_2d args = {
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.IW   =*/ ne10,
        /*.IH   =*/ ne11,
        /*.KW   =*/ ne00,
        /*.KH   =*/ ne01,
        /*.IC   =*/ ne02,
        /*.OC   =*/ ne03,
        /*.OW   =*/ ne0,
        /*.OH   =*/ ne1,
        /*.N    =*/ ne3,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_conv_2d(lib, op);

    int nth = wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
    nth = std::min(nth, 256);
    nth = std::max(nth, 1);

    const uint64_t n_out = wsp_ggml_nelements(op);

    uint64_t tg = (n_out + nth - 1)/nth;
    tg = std::max<uint64_t>(tg, 1);
    tg = std::min<uint64_t>(tg, (uint64_t) std::numeric_limits<int>::max());

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[1]), 2);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         3);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, tg, 1, 1, nth, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_conv_transpose_1d(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];

    const int32_t IC = op->src[1]->ne[1];
    const int32_t IL = op->src[1]->ne[0];

    const int32_t K  = op->src[0]->ne[0];

    const int32_t OL = op->ne[0];
    const int32_t OC = op->ne[1];

    wsp_ggml_metal_kargs_conv_transpose_1d args = {
        /*.IC  =*/ IC,
        /*.IL  =*/ IL,
        /*.K   =*/ K,
        /*.s0  =*/ s0,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_conv_transpose_1d(lib, op);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[1]), 2);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         3);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, OL, OC, 1, 1, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_conv_transpose_2d(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];

    const int32_t IC = op->src[1]->ne[2];
    const int32_t IH = op->src[1]->ne[1];
    const int32_t IW = op->src[1]->ne[0];

    const int32_t KH = op->src[0]->ne[1];
    const int32_t KW = op->src[0]->ne[0];

    const int32_t OW = op->ne[0];
    const int32_t OH = op->ne[1];
    const int32_t OC = op->ne[2];

    wsp_ggml_metal_kargs_conv_transpose_2d args = {
        /*.IC  =*/ IC,
        /*.IH  =*/ IH,
        /*.IW  =*/ IW,
        /*.KH  =*/ KH,
        /*.KW  =*/ KW,
        /*.OC  =*/ OC,
        /*.s0  =*/ s0,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
        /*.nb2 =*/ nb2,
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_conv_transpose_2d(lib, op);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[1]), 2);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         3);

    // Metal requires buffer size to be multiple of 16 bytes
    const size_t smem = WSP_GGML_PAD(KW * KH * sizeof(float), 16);
    wsp_ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, OW, OH, OC, KW, KH, 1);

    return 1;
}

int wsp_ggml_metal_op_upscale(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const float sf0 = (float)ne0/op->src[0]->ne[0];
    const float sf1 = (float)ne1/op->src[0]->ne[1];
    const float sf2 = (float)ne2/op->src[0]->ne[2];
    const float sf3 = (float)ne3/op->src[0]->ne[3];

    wsp_ggml_metal_kargs_upscale args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0 =*/ ne0,
        /*.ne1 =*/ ne1,
        /*.ne2 =*/ ne2,
        /*.ne3 =*/ ne3,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
        /*.nb2 =*/ nb2,
        /*.nb3 =*/ nb3,
        /*.sf0 =*/ sf0,
        /*.sf1 =*/ sf1,
        /*.sf2 =*/ sf2,
        /*.sf3 =*/ sf3
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_upscale(lib, op);

    const int nth = std::min(wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         2);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_pad(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    wsp_ggml_metal_kargs_pad args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_pad(lib, op);

    const int nth = std::min(1024, ne0);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         2);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_pad_reflect_1d(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    wsp_ggml_metal_kargs_pad_reflect_1d args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.p0 =*/ ((const int32_t *)(op->op_params))[0],
        /*.p1 =*/ ((const int32_t *)(op->op_params))[1]
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_pad_reflect_1d(lib, op);

    const int nth = std::min(1024, ne0);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         2);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_arange(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float start;
    float step;

    memcpy(&start, ((const int32_t *) op->op_params) + 0, sizeof(float));
    memcpy(&step,  ((const int32_t *) op->op_params) + 2, sizeof(float));

    wsp_ggml_metal_kargs_arange args = {
        /*.ne0   =*/ ne0,
        /*.start =*/ start,
        /*.step  =*/ step
    };

    const int nth = std::min(1024, ne0);

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_arange(lib, op);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op), 1);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_timestep_embedding(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int dim        = op->op_params[0];
    const int max_period = op->op_params[1];

    wsp_ggml_metal_kargs_timestep_embedding args = {
        /*.nb1 =*/ nb1,
        /*.dim =*/ dim,
        /*.max_period =*/ max_period,
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_timestep_embedding(lib, op);

    const int nth = std::max(1, std::min(1024, dim/2));

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         2);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, ne00, 1, 1, nth, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_argmax(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    wsp_ggml_metal_kargs_argmax args = {
        /*.ne00 = */ ne00,
        /*.nb01 = */ nb01,
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_argmax(lib, op);

    const int64_t nrows = wsp_ggml_nrows(op->src[0]);

    int nth = 32; // SIMD width
    while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
        nth *= 2;
    }

    const size_t smem = wsp_ggml_metal_pipeline_get_smem(pipeline);

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         2);

    wsp_ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_argsort(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_ASSERT(wsp_ggml_is_contiguous_rows(op->src[0]));

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_argsort(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    int nth = 1;
    while (nth < ne00 && 2*nth <= wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    const int npr = (ne00 + nth - 1)/nth;

    // Metal kernels require the buffer size to be multiple of 16 bytes
    // https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443142-setthreadgroupmemorylength
    const size_t smem = WSP_GGML_PAD(nth*sizeof(int32_t), 16);

    wsp_ggml_metal_buffer_id bid_src0 = wsp_ggml_metal_get_buffer_id(op->src[0]);
    wsp_ggml_metal_buffer_id bid_dst  = wsp_ggml_metal_get_buffer_id(op);

    wsp_ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += wsp_ggml_nbytes(op);

    if ((int) ceil(std::log(npr) / std::log(2)) % 2 == 1) {
        std::swap(bid_dst, bid_tmp);
    }

    wsp_ggml_metal_kargs_argsort args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
    };

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    wsp_ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, npr*ne01, ne02, ne03, nth, 1, 1);

    wsp_ggml_metal_pipeline_t pipeline_merge = wsp_ggml_metal_library_get_pipeline_argsort_merge(lib, op);

    int len = nth;

    while (len < ne00) {
        wsp_ggml_metal_op_concurrency_reset(ctx);

        wsp_ggml_metal_kargs_argsort_merge args_merge = {
            .ne00 = ne00,
            .ne01 = ne01,
            .ne02 = ne02,
            .ne03 = ne03,
            .nb00 = nb00,
            .nb01 = nb01,
            .nb02 = nb02,
            .nb03 = nb03,
            .len  = len,
        };

        // merges per row
        const int nm = (ne00 + 2*len - 1) / (2*len);

        const int nth = std::min(512, wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_merge));

        wsp_ggml_metal_encoder_set_pipeline(enc, pipeline_merge);
        wsp_ggml_metal_encoder_set_bytes   (enc, &args_merge, sizeof(args_merge), 0);
        wsp_ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        wsp_ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);
        wsp_ggml_metal_encoder_set_buffer  (enc, bid_tmp,  3);

        wsp_ggml_metal_encoder_dispatch_threadgroups(enc, nm*ne01, ne02, ne03, nth, 1, 1);

        std::swap(bid_dst, bid_tmp);

        len <<= 1;
    }

    return 1;
}

int wsp_ggml_metal_op_leaky_relu(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float slope;
    memcpy(&slope, op->op_params, sizeof(float));

    wsp_ggml_metal_kargs_leaky_relu args = {
        /*.slope =*/ slope
    };

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_unary(lib, op);

    int64_t n = wsp_ggml_nelements(op);

    if (n % 4 == 0) {
        n /= 4;
    }

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), 1);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op),         2);

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, 1, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_opt_step_adamw(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_opt_step_adamw(lib, op);

    const int64_t np = wsp_ggml_nelements(op->src[0]);
    wsp_ggml_metal_kargs_opt_step_adamw args = {
        /*.np =*/ np,
    };

    int ida = 0;

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), ida++);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), ida++);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[1]), ida++);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[2]), ida++);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[3]), ida++);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[4]), ida++);

    const int nth = std::min(wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);
    const int64_t n = (np + nth - 1) / nth;

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}

int wsp_ggml_metal_op_opt_step_sgd(wsp_ggml_metal_op_t ctx, int idx) {
    wsp_ggml_tensor * op = ctx->node(idx);

    wsp_ggml_metal_library_t lib = ctx->lib;
    wsp_ggml_metal_encoder_t enc = ctx->enc;

    WSP_GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    WSP_GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    WSP_GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    wsp_ggml_metal_pipeline_t pipeline = wsp_ggml_metal_library_get_pipeline_opt_step_sgd(lib, op);

    const int64_t np = wsp_ggml_nelements(op->src[0]);
    wsp_ggml_metal_kargs_opt_step_sgd args = {
        /*.np =*/ np,
    };

    int ida = 0;

    wsp_ggml_metal_encoder_set_pipeline(enc, pipeline);
    wsp_ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), ida++);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[0]), ida++);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[1]), ida++);
    wsp_ggml_metal_encoder_set_buffer  (enc, wsp_ggml_metal_get_buffer_id(op->src[2]), ida++);

    const int nth = std::min(wsp_ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);
    const int64_t n = (np + nth - 1) / nth;

    wsp_ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}
