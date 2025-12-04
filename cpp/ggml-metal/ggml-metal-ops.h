#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wsp_ggml_metal_op * wsp_ggml_metal_op_t;

wsp_ggml_metal_op_t wsp_ggml_metal_op_init(
        wsp_ggml_metal_device_t dev,
        wsp_ggml_metal_cmd_buf_t cmd_buf,
        struct wsp_ggml_cgraph * gf,
        int  idx_start,
        int  idx_end,
        bool use_fusion,
        bool use_concurrency,
        bool use_capture,
        int  debug_graph,
        int  debug_fusion);

void wsp_ggml_metal_op_free(wsp_ggml_metal_op_t ctx);

int wsp_ggml_metal_op_n_nodes(wsp_ggml_metal_op_t ctx);

int wsp_ggml_metal_op_encode(wsp_ggml_metal_op_t ctx, int idx);

//
// available ops:
//

// tokens per expert
size_t wsp_ggml_metal_op_mul_mat_id_extra_tpe(const struct wsp_ggml_tensor * op);

// id map [n_tokens, n_expert]
size_t wsp_ggml_metal_op_mul_mat_id_extra_ids(const struct wsp_ggml_tensor * op);

// return true if we should use the FA vector kernel for this op
bool wsp_ggml_metal_op_flash_attn_ext_use_vec(const struct wsp_ggml_tensor * op);

size_t wsp_ggml_metal_op_flash_attn_ext_extra_pad(const struct wsp_ggml_tensor * op);
size_t wsp_ggml_metal_op_flash_attn_ext_extra_blk(const struct wsp_ggml_tensor * op);
size_t wsp_ggml_metal_op_flash_attn_ext_extra_tmp(const struct wsp_ggml_tensor * op);

int wsp_ggml_metal_op_concat            (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_repeat            (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_acc               (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_scale             (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_clamp             (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_unary             (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_glu               (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_sum               (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_sum_rows          (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_cumsum            (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_get_rows          (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_set_rows          (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_soft_max          (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_ssm_conv          (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_ssm_scan          (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_rwkv              (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_cpy               (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_pool_2d           (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_mul_mat           (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_mul_mat_id        (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_add_id            (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_flash_attn_ext    (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_bin               (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_l2_norm           (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_group_norm        (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_norm              (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_rope              (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_im2col            (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_conv_2d           (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_conv_transpose_1d (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_conv_transpose_2d (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_upscale           (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_pad               (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_pad_reflect_1d    (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_arange            (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_timestep_embedding(wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_argmax            (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_argsort           (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_leaky_relu        (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_opt_step_adamw    (wsp_ggml_metal_op_t ctx, int idx);
int wsp_ggml_metal_op_opt_step_sgd      (wsp_ggml_metal_op_t ctx, int idx);

#ifdef __cplusplus
}
#endif
