#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// backend context
//

typedef struct wsp_ggml_metal * wsp_ggml_metal_t;

wsp_ggml_metal_t wsp_ggml_metal_init(wsp_ggml_metal_device_t dev);
void wsp_ggml_metal_free(wsp_ggml_metal_t ctx);

void wsp_ggml_metal_synchronize(wsp_ggml_metal_t ctx);

void wsp_ggml_metal_set_tensor_async(wsp_ggml_metal_t ctx, struct wsp_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void wsp_ggml_metal_get_tensor_async(wsp_ggml_metal_t ctx, const struct wsp_ggml_tensor * tensor, void * data, size_t offset, size_t size);

enum wsp_ggml_status wsp_ggml_metal_graph_compute (wsp_ggml_metal_t ctx, struct wsp_ggml_cgraph * gf);
void             wsp_ggml_metal_graph_optimize(wsp_ggml_metal_t ctx, struct wsp_ggml_cgraph * gf);

void wsp_ggml_metal_set_n_cb            (wsp_ggml_metal_t ctx, int n_cb);
void wsp_ggml_metal_set_abort_callback  (wsp_ggml_metal_t ctx, wsp_ggml_abort_callback abort_callback, void * user_data);
bool wsp_ggml_metal_supports_family     (wsp_ggml_metal_t ctx, int family);
void wsp_ggml_metal_capture_next_compute(wsp_ggml_metal_t ctx);

#ifdef __cplusplus
}
#endif
