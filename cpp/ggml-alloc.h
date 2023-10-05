#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif


WSP_GGML_API struct wsp_ggml_allocr * wsp_ggml_allocr_new(void * data, size_t size, size_t alignment);
WSP_GGML_API struct wsp_ggml_allocr * wsp_ggml_allocr_new_measure(size_t alignment);

// tell the allocator to parse nodes following the order described in the list
// you should call this if your graph are optimized to execute out-of-order
WSP_GGML_API void   wsp_ggml_allocr_set_parse_seq(struct wsp_ggml_allocr * alloc, const int * list, int n);

WSP_GGML_API void   wsp_ggml_allocr_free(struct wsp_ggml_allocr * alloc);
WSP_GGML_API bool   wsp_ggml_allocr_is_measure(struct wsp_ggml_allocr * alloc);
WSP_GGML_API void   wsp_ggml_allocr_reset(struct wsp_ggml_allocr * alloc);
WSP_GGML_API void   wsp_ggml_allocr_alloc(struct wsp_ggml_allocr * alloc, struct wsp_ggml_tensor * tensor);
WSP_GGML_API size_t wsp_ggml_allocr_alloc_graph(struct wsp_ggml_allocr * alloc, struct wsp_ggml_cgraph * graph);


#ifdef  __cplusplus
}
#endif
