#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct wsp_ggml_backend;
struct wsp_ggml_backend_buffer;

//
// Legacy API
//

typedef struct wsp_ggml_allocr * wsp_ggml_allocr_t;

// initialize allocator for use with CPU backend only
WSP_GGML_API wsp_ggml_allocr_t wsp_ggml_allocr_new(void * data, size_t size, size_t alignment);
WSP_GGML_API wsp_ggml_allocr_t wsp_ggml_allocr_new_measure(size_t alignment);

// initialize allocator for use with ggml-backend
WSP_GGML_API wsp_ggml_allocr_t wsp_ggml_allocr_new_from_buffer(struct wsp_ggml_backend_buffer * buffer);
WSP_GGML_API wsp_ggml_allocr_t wsp_ggml_allocr_new_from_backend(struct wsp_ggml_backend * backend, size_t size); // allocates an owned buffer
WSP_GGML_API wsp_ggml_allocr_t wsp_ggml_allocr_new_measure_from_backend(struct wsp_ggml_backend * backend);

WSP_GGML_API struct wsp_ggml_backend_buffer * wsp_ggml_allocr_get_buffer(wsp_ggml_allocr_t alloc);

// tell the allocator to parse nodes following the order described in the list
// you should call this if your graph are optimized to execute out-of-order
WSP_GGML_API void   wsp_ggml_allocr_set_parse_seq(wsp_ggml_allocr_t alloc, const int * list, int n);

WSP_GGML_API void   wsp_ggml_allocr_free       (wsp_ggml_allocr_t alloc);
WSP_GGML_API bool   wsp_ggml_allocr_is_measure (wsp_ggml_allocr_t alloc);
WSP_GGML_API void   wsp_ggml_allocr_reset      (wsp_ggml_allocr_t alloc);
WSP_GGML_API void   wsp_ggml_allocr_alloc      (wsp_ggml_allocr_t alloc, struct wsp_ggml_tensor * tensor);
WSP_GGML_API size_t wsp_ggml_allocr_max_size   (wsp_ggml_allocr_t alloc);

WSP_GGML_API size_t wsp_ggml_allocr_alloc_graph(wsp_ggml_allocr_t alloc, struct wsp_ggml_cgraph * graph);

//
// ggml-backend v2 API
//

// Seperate tensor and graph allocator objects
// This is necessary for multi-backend allocation because the graph allocator needs to use multiple tensor allocators
// The original API is kept as a wrapper around the new API

// Tensor allocator
typedef struct wsp_ggml_tallocr * wsp_ggml_tallocr_t;

WSP_GGML_API wsp_ggml_tallocr_t wsp_ggml_tallocr_new(void * data, size_t size, size_t alignment);
WSP_GGML_API wsp_ggml_tallocr_t wsp_ggml_tallocr_new_measure(size_t alignment);
WSP_GGML_API wsp_ggml_tallocr_t wsp_ggml_tallocr_new_from_buffer(struct wsp_ggml_backend_buffer * buffer);
WSP_GGML_API wsp_ggml_tallocr_t wsp_ggml_tallocr_new_from_backend(struct wsp_ggml_backend * backend, size_t size); // allocates an owned buffer
WSP_GGML_API wsp_ggml_tallocr_t wsp_ggml_tallocr_new_measure_from_backend(struct wsp_ggml_backend * backend);

WSP_GGML_API struct wsp_ggml_backend_buffer * wsp_ggml_tallocr_get_buffer(wsp_ggml_tallocr_t talloc);

WSP_GGML_API void   wsp_ggml_tallocr_free       (wsp_ggml_tallocr_t talloc);
WSP_GGML_API bool   wsp_ggml_tallocr_is_measure (wsp_ggml_tallocr_t talloc);
WSP_GGML_API void   wsp_ggml_tallocr_reset      (wsp_ggml_tallocr_t talloc);
WSP_GGML_API void   wsp_ggml_tallocr_alloc      (wsp_ggml_tallocr_t talloc, struct wsp_ggml_tensor * tensor);
WSP_GGML_API size_t wsp_ggml_tallocr_max_size   (wsp_ggml_tallocr_t talloc);


// Graph allocator
typedef struct wsp_ggml_gallocr * wsp_ggml_gallocr_t;

WSP_GGML_API wsp_ggml_gallocr_t wsp_ggml_gallocr_new(void);
WSP_GGML_API void   wsp_ggml_gallocr_free(wsp_ggml_gallocr_t galloc);

WSP_GGML_API void   wsp_ggml_gallocr_set_parse_seq(wsp_ggml_gallocr_t galloc, const int * list, int n);
WSP_GGML_API size_t wsp_ggml_gallocr_alloc_graph(wsp_ggml_gallocr_t galloc, wsp_ggml_tallocr_t talloc, struct wsp_ggml_cgraph * graph);

// Allocate tensors from the allocators given by the hash table
WSP_GGML_API void   wsp_ggml_gallocr_alloc_graph_n(
                    wsp_ggml_gallocr_t galloc,
                    struct wsp_ggml_cgraph * graph,
                    struct wsp_ggml_hash_set hash_set,
                    wsp_ggml_tallocr_t * hash_node_talloc);

#ifdef  __cplusplus
}
#endif
