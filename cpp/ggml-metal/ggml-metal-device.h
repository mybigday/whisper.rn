#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

struct wsp_ggml_metal_buffer_id {
    void * metal; // id<MTLBuffer>
    size_t offs;
};

typedef struct wsp_ggml_metal_device * wsp_ggml_metal_device_t;

//
// MTLFunctionConstantValues wrapper
//

typedef struct wsp_ggml_metal_cv * wsp_ggml_metal_cv_t;

wsp_ggml_metal_cv_t wsp_ggml_metal_cv_init(void);
void wsp_ggml_metal_cv_free(wsp_ggml_metal_cv_t cv);

void wsp_ggml_metal_cv_set_int16(wsp_ggml_metal_cv_t cv, int16_t value, int32_t idx);
void wsp_ggml_metal_cv_set_int32(wsp_ggml_metal_cv_t cv, int32_t value, int32_t idx);
void wsp_ggml_metal_cv_set_bool (wsp_ggml_metal_cv_t cv, bool    value, int32_t idx);

//
// MTLComputePipelineState wrapper
//

typedef struct wsp_ggml_metal_pipeline * wsp_ggml_metal_pipeline_t;

wsp_ggml_metal_pipeline_t wsp_ggml_metal_pipeline_init(void);
void wsp_ggml_metal_pipeline_free(wsp_ggml_metal_pipeline_t pipeline);

void wsp_ggml_metal_pipeline_set_nsg(wsp_ggml_metal_pipeline_t pipeline, int nsg);
int  wsp_ggml_metal_pipeline_get_nsg(wsp_ggml_metal_pipeline_t pipeline);

void wsp_ggml_metal_pipeline_set_nr0(wsp_ggml_metal_pipeline_t pipeline, int nr0);
int  wsp_ggml_metal_pipeline_get_nr0(wsp_ggml_metal_pipeline_t pipeline);

void wsp_ggml_metal_pipeline_set_nr1(wsp_ggml_metal_pipeline_t pipeline, int nr1);
int  wsp_ggml_metal_pipeline_get_nr1(wsp_ggml_metal_pipeline_t pipeline);

void   wsp_ggml_metal_pipeline_set_smem(wsp_ggml_metal_pipeline_t pipeline, size_t smem);
size_t wsp_ggml_metal_pipeline_get_smem(wsp_ggml_metal_pipeline_t pipeline);

int wsp_ggml_metal_pipeline_max_theads_per_threadgroup(wsp_ggml_metal_pipeline_t pipeline);

// a collection of pipelines
typedef struct wsp_ggml_metal_pipelines * wsp_ggml_metal_pipelines_t;

wsp_ggml_metal_pipelines_t wsp_ggml_metal_pipelines_init(void);
void wsp_ggml_metal_pipelines_free(wsp_ggml_metal_pipelines_t ppls);

void                  wsp_ggml_metal_pipelines_add(wsp_ggml_metal_pipelines_t ppls, const char * name, wsp_ggml_metal_pipeline_t pipeline);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_pipelines_get(wsp_ggml_metal_pipelines_t ppls, const char * name);

//
// MTLCommandBuffer wrapper
//

typedef void * wsp_ggml_metal_cmd_buf_t;

//
// MTLComputeCommandEncoder wrapper
//

typedef struct wsp_ggml_metal_encoder * wsp_ggml_metal_encoder_t;

wsp_ggml_metal_encoder_t wsp_ggml_metal_encoder_init(wsp_ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent);
void wsp_ggml_metal_encoder_free(wsp_ggml_metal_encoder_t encoder);

void wsp_ggml_metal_encoder_debug_group_push(wsp_ggml_metal_encoder_t encoder, const char * name);
void wsp_ggml_metal_encoder_debug_group_pop (wsp_ggml_metal_encoder_t encoder);

void wsp_ggml_metal_encoder_set_pipeline(wsp_ggml_metal_encoder_t encoder, wsp_ggml_metal_pipeline_t pipeline);

void wsp_ggml_metal_encoder_set_bytes (wsp_ggml_metal_encoder_t encoder, void * data, size_t size, int idx);
void wsp_ggml_metal_encoder_set_buffer(wsp_ggml_metal_encoder_t encoder, struct wsp_ggml_metal_buffer_id buffer, int idx);

void wsp_ggml_metal_encoder_set_threadgroup_memory_size(wsp_ggml_metal_encoder_t encoder, size_t size, int idx);

void wsp_ggml_metal_encoder_dispatch_threadgroups(wsp_ggml_metal_encoder_t encoder, int tg0, int tg1, int tg2, int tptg0, int tptg1, int tptg2);

void wsp_ggml_metal_encoder_memory_barrier(wsp_ggml_metal_encoder_t encoder);

void wsp_ggml_metal_encoder_end_encoding(wsp_ggml_metal_encoder_t encoder);

//
// MTLLibrary wrapper
//

typedef struct wsp_ggml_metal_library * wsp_ggml_metal_library_t;

wsp_ggml_metal_library_t wsp_ggml_metal_library_init            (wsp_ggml_metal_device_t dev);
wsp_ggml_metal_library_t wsp_ggml_metal_library_init_from_source(wsp_ggml_metal_device_t dev, const char * source, bool verbose);

void wsp_ggml_metal_library_free(wsp_ggml_metal_library_t lib);

wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline    (wsp_ggml_metal_library_t lib, const char * name);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_compile_pipeline(wsp_ggml_metal_library_t lib, const char * base, const char * name, wsp_ggml_metal_cv_t cv);

wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_base              (wsp_ggml_metal_library_t lib, enum wsp_ggml_op op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_cpy               (wsp_ggml_metal_library_t lib, enum wsp_ggml_type tsrc, enum wsp_ggml_type tdst);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_pool_2d           (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op, enum wsp_ggml_op_pool op_pool);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_get_rows          (wsp_ggml_metal_library_t lib, enum wsp_ggml_type tsrc);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_set_rows          (wsp_ggml_metal_library_t lib, enum wsp_ggml_type tidx, enum wsp_ggml_type tdst);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_repeat            (wsp_ggml_metal_library_t lib, enum wsp_ggml_type tsrc);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_unary             (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_glu               (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_sum               (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_sum_rows          (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_cumsum_blk        (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_cumsum_add        (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_soft_max          (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_ssm_conv          (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_ssm_scan          (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_rwkv              (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_mul_mv_ext        (wsp_ggml_metal_library_t lib, enum wsp_ggml_type tsrc0, enum wsp_ggml_type tsrc1, int nsg, int nxpsg, int r1ptg);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_mul_mm            (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_mul_mv            (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_mul_mm_id_map0    (wsp_ggml_metal_library_t lib, int ne02, int ne20);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_mul_mm_id         (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_mul_mv_id         (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_argmax            (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_argsort           (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_argsort_merge     (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_bin               (wsp_ggml_metal_library_t lib, enum wsp_ggml_op op, int32_t n_fuse, bool row);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_l2_norm           (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_group_norm        (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_norm              (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op, int32_t n_fuse);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_rope              (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_im2col            (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_conv_transpose_1d (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_conv_transpose_2d (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_conv_2d           (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_upscale           (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_pad               (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_pad_reflect_1d    (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_arange            (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_timestep_embedding(wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_opt_step_adamw    (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);
wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_opt_step_sgd      (wsp_ggml_metal_library_t lib, const struct wsp_ggml_tensor * op);

wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        wsp_ggml_metal_library_t lib,
        const struct wsp_ggml_tensor * op,
        bool    has_mask,
        int32_t ncpsg);

wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        wsp_ggml_metal_library_t lib,
        const struct wsp_ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg);

wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_flash_attn_ext(
        wsp_ggml_metal_library_t lib,
        const struct wsp_ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg);

wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        wsp_ggml_metal_library_t lib,
        const struct wsp_ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        int32_t nwg);

wsp_ggml_metal_pipeline_t wsp_ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        wsp_ggml_metal_library_t lib,
        const struct wsp_ggml_tensor * op,
        int32_t dv,
        int32_t nwg);

//
// device
//

struct wsp_ggml_metal_device_props {
    char name[128];

    size_t max_buffer_size;
    size_t max_working_set_size;
    size_t max_theadgroup_memory_size;

    bool has_simdgroup_reduction;
    bool has_simdgroup_mm;
    bool has_unified_memory;
    bool has_bfloat;
    bool has_tensor;
    bool use_residency_sets;
    bool use_shared_buffers;

    bool supports_gpu_family_apple7;
};

wsp_ggml_metal_device_t wsp_ggml_metal_device_init(void);
void wsp_ggml_metal_device_free(wsp_ggml_metal_device_t dev);

// return a singleton that is automatically destroyed when the program exits
wsp_ggml_metal_device_t wsp_ggml_metal_device_get(void);

void * wsp_ggml_metal_device_get_obj  (wsp_ggml_metal_device_t dev); // id<MTLDevice>
void * wsp_ggml_metal_device_get_queue(wsp_ggml_metal_device_t dev); // id<MTLCommandQueue>

wsp_ggml_metal_library_t wsp_ggml_metal_device_get_library(wsp_ggml_metal_device_t dev);

void wsp_ggml_metal_device_get_memory(wsp_ggml_metal_device_t dev, size_t * free, size_t * total);
bool wsp_ggml_metal_device_supports_op(wsp_ggml_metal_device_t dev, const struct wsp_ggml_tensor * op);

const struct wsp_ggml_metal_device_props * wsp_ggml_metal_device_get_props(wsp_ggml_metal_device_t dev);

//
// device buffers
//

typedef struct wsp_ggml_metal_buffer * wsp_ggml_metal_buffer_t;

wsp_ggml_metal_buffer_t wsp_ggml_metal_buffer_init(wsp_ggml_metal_device_t dev, size_t size, bool shared);
wsp_ggml_metal_buffer_t wsp_ggml_metal_buffer_map (wsp_ggml_metal_device_t dev, void * ptr, size_t size, size_t max_tensor_size);

void   wsp_ggml_metal_buffer_free     (wsp_ggml_metal_buffer_t buf);
void * wsp_ggml_metal_buffer_get_base (wsp_ggml_metal_buffer_t buf);
bool   wsp_ggml_metal_buffer_is_shared(wsp_ggml_metal_buffer_t buf);

void   wsp_ggml_metal_buffer_memset_tensor(wsp_ggml_metal_buffer_t buf, struct wsp_ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
void   wsp_ggml_metal_buffer_set_tensor   (wsp_ggml_metal_buffer_t buf, struct wsp_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void   wsp_ggml_metal_buffer_get_tensor   (wsp_ggml_metal_buffer_t buf, const struct wsp_ggml_tensor * tensor, void * data, size_t offset, size_t size);
void   wsp_ggml_metal_buffer_clear        (wsp_ggml_metal_buffer_t buf, uint8_t value);

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
struct wsp_ggml_metal_buffer_id wsp_ggml_metal_buffer_get_id(wsp_ggml_metal_buffer_t buf, const struct wsp_ggml_tensor * t);

#ifdef __cplusplus
}
#endif
