#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define MAX(a, b) ((a) > (b) ? (a) : (b))


// backend buffer type

const char * wsp_ggml_backend_buft_name(wsp_ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name(buft);
}

WSP_GGML_CALL wsp_ggml_backend_buffer_t wsp_ggml_backend_buft_alloc_buffer(wsp_ggml_backend_buffer_type_t buft, size_t size) {
    return buft->iface.alloc_buffer(buft, size);
}

size_t wsp_ggml_backend_buft_get_alignment(wsp_ggml_backend_buffer_type_t buft) {
    return buft->iface.get_alignment(buft);
}

WSP_GGML_CALL size_t wsp_ggml_backend_buft_get_alloc_size(wsp_ggml_backend_buffer_type_t buft, struct wsp_ggml_tensor * tensor) {
    // get_alloc_size is optional, defaults to wsp_ggml_nbytes
    if (buft->iface.get_alloc_size) {
        return buft->iface.get_alloc_size(buft, tensor);
    }
    return wsp_ggml_nbytes(tensor);
}

bool wsp_ggml_backend_buft_supports_backend(wsp_ggml_backend_buffer_type_t buft, wsp_ggml_backend_t backend) {
    return buft->iface.supports_backend(buft, backend);
}

bool wsp_ggml_backend_buft_is_host(wsp_ggml_backend_buffer_type_t buft) {
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

// backend buffer

WSP_GGML_CALL wsp_ggml_backend_buffer_t wsp_ggml_backend_buffer_init(
               wsp_ggml_backend_buffer_type_t      buft,
        struct wsp_ggml_backend_buffer_i           iface,
               wsp_ggml_backend_buffer_context_t   context,
               size_t                          size) {
    wsp_ggml_backend_buffer_t buffer = malloc(sizeof(struct wsp_ggml_backend_buffer));

    WSP_GGML_ASSERT(iface.get_base != NULL);

    (*buffer) = (struct wsp_ggml_backend_buffer) {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ WSP_GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * wsp_ggml_backend_buffer_name(wsp_ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name(buffer);
}

void wsp_ggml_backend_buffer_free(wsp_ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    free(buffer);
}

size_t wsp_ggml_backend_buffer_get_size(wsp_ggml_backend_buffer_t buffer) {
    return buffer->size;
}

void * wsp_ggml_backend_buffer_get_base(wsp_ggml_backend_buffer_t buffer) {
    void * base = buffer->iface.get_base(buffer);

    WSP_GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

WSP_GGML_CALL void wsp_ggml_backend_buffer_init_tensor(wsp_ggml_backend_buffer_t buffer, struct wsp_ggml_tensor * tensor) {
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    }
}

size_t wsp_ggml_backend_buffer_get_alignment (wsp_ggml_backend_buffer_t buffer) {
    return wsp_ggml_backend_buft_get_alignment(wsp_ggml_backend_buffer_get_type(buffer));
}

size_t wsp_ggml_backend_buffer_get_alloc_size(wsp_ggml_backend_buffer_t buffer, struct wsp_ggml_tensor * tensor) {
    return wsp_ggml_backend_buft_get_alloc_size(wsp_ggml_backend_buffer_get_type(buffer), tensor);
}

void wsp_ggml_backend_buffer_clear(wsp_ggml_backend_buffer_t buffer, uint8_t value) {
    buffer->iface.clear(buffer, value);
}

bool wsp_ggml_backend_buffer_is_host(wsp_ggml_backend_buffer_t buffer) {
    return wsp_ggml_backend_buft_is_host(wsp_ggml_backend_buffer_get_type(buffer));
}

void wsp_ggml_backend_buffer_set_usage(wsp_ggml_backend_buffer_t buffer, enum wsp_ggml_backend_buffer_usage usage) {
    buffer->usage = usage;
}

wsp_ggml_backend_buffer_type_t wsp_ggml_backend_buffer_get_type(wsp_ggml_backend_buffer_t buffer) {
    return buffer->buft;
}

void wsp_ggml_backend_buffer_reset(wsp_ggml_backend_buffer_t buffer) {
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool wsp_ggml_backend_buffer_copy_tensor(const struct wsp_ggml_tensor * src, struct wsp_ggml_tensor * dst) {
    wsp_ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return src->buffer->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

const char * wsp_ggml_backend_name(wsp_ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void wsp_ggml_backend_free(wsp_ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

wsp_ggml_backend_buffer_type_t wsp_ggml_backend_get_default_buffer_type(wsp_ggml_backend_t backend) {
    return backend->iface.get_default_buffer_type(backend);
}

wsp_ggml_backend_buffer_t wsp_ggml_backend_alloc_buffer(wsp_ggml_backend_t backend, size_t size) {
    return wsp_ggml_backend_buft_alloc_buffer(wsp_ggml_backend_get_default_buffer_type(backend), size);
}

size_t wsp_ggml_backend_get_alignment(wsp_ggml_backend_t backend) {
    return wsp_ggml_backend_buft_get_alignment(wsp_ggml_backend_get_default_buffer_type(backend));
}

void wsp_ggml_backend_tensor_set_async(wsp_ggml_backend_t backend, struct wsp_ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    WSP_GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    WSP_GGML_ASSERT(offset + size <= wsp_ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        wsp_ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void wsp_ggml_backend_tensor_get_async(wsp_ggml_backend_t backend, const struct wsp_ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    WSP_GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    WSP_GGML_ASSERT(offset + size <= wsp_ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        wsp_ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

WSP_GGML_CALL void wsp_ggml_backend_tensor_set(struct wsp_ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    wsp_ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    WSP_GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    WSP_GGML_ASSERT(buf != NULL && "tensor buffer not set");
    WSP_GGML_ASSERT(offset + size <= wsp_ggml_nbytes(tensor) && "tensor write out of bounds");

    tensor->buffer->iface.set_tensor(buf, tensor, data, offset, size);
}

WSP_GGML_CALL void wsp_ggml_backend_tensor_get(const struct wsp_ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    wsp_ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    WSP_GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    WSP_GGML_ASSERT(tensor->buffer != NULL && "tensor buffer not set");
    WSP_GGML_ASSERT(offset + size <= wsp_ggml_nbytes(tensor) && "tensor read out of bounds");

    tensor->buffer->iface.get_tensor(buf, tensor, data, offset, size);
}

void wsp_ggml_backend_synchronize(wsp_ggml_backend_t backend) {
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

wsp_ggml_backend_graph_plan_t wsp_ggml_backend_graph_plan_create(wsp_ggml_backend_t backend, struct wsp_ggml_cgraph * cgraph) {
    return backend->iface.graph_plan_create(backend, cgraph);
}

void wsp_ggml_backend_graph_plan_free(wsp_ggml_backend_t backend, wsp_ggml_backend_graph_plan_t plan) {
    backend->iface.graph_plan_free(backend, plan);
}

void wsp_ggml_backend_graph_plan_compute(wsp_ggml_backend_t backend, wsp_ggml_backend_graph_plan_t plan) {
    backend->iface.graph_plan_compute(backend, plan);
}

bool wsp_ggml_backend_graph_compute(wsp_ggml_backend_t backend, struct wsp_ggml_cgraph * cgraph) {
    return backend->iface.graph_compute(backend, cgraph);
}

bool wsp_ggml_backend_supports_op(wsp_ggml_backend_t backend, const struct wsp_ggml_tensor * op) {
    return backend->iface.supports_op(backend, op);
}

// backend copy

static bool wsp_ggml_are_same_layout(const struct wsp_ggml_tensor * a, const struct wsp_ggml_tensor * b) {
    if (a->type != b->type) {
        return false;
    }
    for (int i = 0; i < WSP_GGML_MAX_DIMS; i++) {
        if (a->ne[i] != b->ne[i]) {
            return false;
        }
        if (a->nb[i] != b->nb[i]) {
            return false;
        }
    }
    return true;
}

void wsp_ggml_backend_tensor_copy(struct wsp_ggml_tensor * src, struct wsp_ggml_tensor * dst) {
    WSP_GGML_ASSERT(wsp_ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (wsp_ggml_backend_buffer_is_host(src->buffer)) {
        wsp_ggml_backend_tensor_set(dst, src->data, 0, wsp_ggml_nbytes(src));
    } else if (wsp_ggml_backend_buffer_is_host(dst->buffer)) {
        wsp_ggml_backend_tensor_get(src, dst->data, 0, wsp_ggml_nbytes(src));
    } else if (!wsp_ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        fprintf(stderr, "%s: warning: slow copy from %s to %s\n", __func__, wsp_ggml_backend_buffer_name(src->buffer), wsp_ggml_backend_buffer_name(dst->buffer));
#endif
        size_t nbytes = wsp_ggml_nbytes(src);
        void * data = malloc(nbytes);
        wsp_ggml_backend_tensor_get(src, data, 0, nbytes);
        wsp_ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void wsp_ggml_backend_tensor_copy_async(wsp_ggml_backend_t backend, struct wsp_ggml_tensor * src, struct wsp_ggml_tensor * dst) {
    WSP_GGML_ASSERT(wsp_ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (wsp_ggml_backend_buft_supports_backend(src->buffer->buft, backend) && wsp_ggml_backend_buft_supports_backend(dst->buffer->buft, backend)) {
        if (backend->iface.cpy_tensor_async != NULL) {
            if (backend->iface.cpy_tensor_async(backend, src, dst)) {
                return;
            }
        }
    }

    size_t nbytes = wsp_ggml_nbytes(src);
    if (wsp_ggml_backend_buffer_is_host(src->buffer)) {
        wsp_ggml_backend_tensor_set_async(backend, dst, src->data, 0, nbytes);
    }
    else {
        wsp_ggml_backend_tensor_copy(src, dst);
    }
}


// backend registry

#define WSP_GGML_MAX_BACKENDS_REG 16

struct wsp_ggml_backend_reg {
    char name[128];
    wsp_ggml_backend_init_fn init_fn;
    wsp_ggml_backend_buffer_type_t default_buffer_type;
    void * user_data;
};

static struct wsp_ggml_backend_reg wsp_ggml_backend_registry[WSP_GGML_MAX_BACKENDS_REG];
static size_t wsp_ggml_backend_registry_count = 0;

WSP_GGML_CALL static wsp_ggml_backend_t wsp_ggml_backend_reg_cpu_init(const char * params, void * user_data);

WSP_GGML_CALL static void wsp_ggml_backend_registry_init(void) {
    static bool initialized = false;

    if (initialized) {
        return;
    }

    initialized = true;

    wsp_ggml_backend_register("CPU", wsp_ggml_backend_reg_cpu_init, wsp_ggml_backend_cpu_buffer_type(), NULL);

    // add forward decls here to avoid including the backend headers
#ifdef WSP_GGML_USE_CUBLAS
    extern WSP_GGML_CALL void wsp_ggml_backend_cuda_reg_devices(void);
    wsp_ggml_backend_cuda_reg_devices();
#endif

#ifdef WSP_GGML_USE_METAL
    extern WSP_GGML_CALL wsp_ggml_backend_t wsp_ggml_backend_reg_metal_init(const char * params, void * user_data);
    extern WSP_GGML_CALL wsp_ggml_backend_buffer_type_t wsp_ggml_backend_metal_buffer_type(void);
    wsp_ggml_backend_register("Metal", wsp_ggml_backend_reg_metal_init, wsp_ggml_backend_metal_buffer_type(), NULL);
#endif
}

WSP_GGML_CALL void wsp_ggml_backend_register(const char * name, wsp_ggml_backend_init_fn init_fn, wsp_ggml_backend_buffer_type_t default_buffer_type, void * user_data) {
    WSP_GGML_ASSERT(wsp_ggml_backend_registry_count < WSP_GGML_MAX_BACKENDS_REG);

    size_t id = wsp_ggml_backend_registry_count;

    wsp_ggml_backend_registry[id] = (struct wsp_ggml_backend_reg) {
        /* .name                = */ {0},
        /* .fn                  = */ init_fn,
        /* .default_buffer_type = */ default_buffer_type,
        /* .user_data           = */ user_data,
    };

    snprintf(wsp_ggml_backend_registry[id].name, sizeof(wsp_ggml_backend_registry[id].name), "%s", name);

#ifndef NDEBUG
    fprintf(stderr, "%s: registered backend %s\n", __func__, name);
#endif

    wsp_ggml_backend_registry_count++;
}

size_t wsp_ggml_backend_reg_get_count(void) {
    wsp_ggml_backend_registry_init();

    return wsp_ggml_backend_registry_count;
}

size_t wsp_ggml_backend_reg_find_by_name(const char * name) {
    wsp_ggml_backend_registry_init();

    for (size_t i = 0; i < wsp_ggml_backend_registry_count; i++) {
        // TODO: case insensitive in a portable way
        if (strcmp(wsp_ggml_backend_registry[i].name, name) == 0) {
            return i;
        }
    }

    // not found
    return SIZE_MAX;
}

// init from backend:params string
wsp_ggml_backend_t wsp_ggml_backend_reg_init_backend_from_str(const char * backend_str) {
    wsp_ggml_backend_registry_init();

    const char * params = strchr(backend_str, ':');
    char backend_name[128];
    if (params == NULL) {
        snprintf(backend_name, sizeof(backend_name), "%s", backend_str);
        params = "";
    } else {
        snprintf(backend_name, sizeof(backend_name), "%.*s", (int)(params - backend_str), backend_str);
        params++;
    }

    size_t backend_i = wsp_ggml_backend_reg_find_by_name(backend_name);

    if (backend_i == SIZE_MAX) {
        fprintf(stderr, "%s: backend %s not found\n", __func__, backend_name);
        return NULL;
    }

    return wsp_ggml_backend_reg_init_backend(backend_i, params);
}

const char * wsp_ggml_backend_reg_get_name(size_t i) {
    wsp_ggml_backend_registry_init();

    WSP_GGML_ASSERT(i < wsp_ggml_backend_registry_count);
    return wsp_ggml_backend_registry[i].name;
}

wsp_ggml_backend_t wsp_ggml_backend_reg_init_backend(size_t i, const char * params) {
    wsp_ggml_backend_registry_init();

    WSP_GGML_ASSERT(i < wsp_ggml_backend_registry_count);
    return wsp_ggml_backend_registry[i].init_fn(params, wsp_ggml_backend_registry[i].user_data);
}

wsp_ggml_backend_buffer_type_t wsp_ggml_backend_reg_get_default_buffer_type(size_t i) {
    wsp_ggml_backend_registry_init();

    WSP_GGML_ASSERT(i < wsp_ggml_backend_registry_count);
    return wsp_ggml_backend_registry[i].default_buffer_type;
}

wsp_ggml_backend_buffer_t wsp_ggml_backend_reg_alloc_buffer(size_t i, size_t size) {
    wsp_ggml_backend_registry_init();

    WSP_GGML_ASSERT(i < wsp_ggml_backend_registry_count);
    return wsp_ggml_backend_buft_alloc_buffer(wsp_ggml_backend_registry[i].default_buffer_type, size);
}

// backend CPU

WSP_GGML_CALL static const char * wsp_ggml_backend_cpu_buffer_name(wsp_ggml_backend_buffer_t buffer) {
    return "CPU";

    WSP_GGML_UNUSED(buffer);
}

WSP_GGML_CALL static void * wsp_ggml_backend_cpu_buffer_get_base(wsp_ggml_backend_buffer_t buffer) {
    return (void *)buffer->context;
}

WSP_GGML_CALL static void wsp_ggml_backend_cpu_buffer_free_buffer(wsp_ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

WSP_GGML_CALL static void wsp_ggml_backend_cpu_buffer_set_tensor(wsp_ggml_backend_buffer_t buffer, struct wsp_ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *)tensor->data + offset, data, size);

    WSP_GGML_UNUSED(buffer);
}

WSP_GGML_CALL static void wsp_ggml_backend_cpu_buffer_get_tensor(wsp_ggml_backend_buffer_t buffer, const struct wsp_ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *)tensor->data + offset, size);

    WSP_GGML_UNUSED(buffer);
}

WSP_GGML_CALL static bool wsp_ggml_backend_cpu_buffer_cpy_tensor(wsp_ggml_backend_buffer_t buffer, const struct wsp_ggml_tensor * src, struct wsp_ggml_tensor * dst) {
    if (wsp_ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, wsp_ggml_nbytes(src));
        return true;
    }
    return false;

    WSP_GGML_UNUSED(buffer);
}

WSP_GGML_CALL static void wsp_ggml_backend_cpu_buffer_clear(wsp_ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static struct wsp_ggml_backend_buffer_i cpu_backend_buffer_i = {
    /* .get_name        = */ wsp_ggml_backend_cpu_buffer_name,
    /* .free_buffer     = */ wsp_ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ wsp_ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .set_tensor      = */ wsp_ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ wsp_ggml_backend_cpu_buffer_get_tensor,
    /* .cpy_tensor      = */ wsp_ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ wsp_ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// for buffers from ptr, free is not called
static struct wsp_ggml_backend_buffer_i cpu_backend_buffer_i_from_ptr = {
    /* .get_name        = */ wsp_ggml_backend_cpu_buffer_name,
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ wsp_ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .set_tensor      = */ wsp_ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ wsp_ggml_backend_cpu_buffer_get_tensor,
    /* .cpy_tensor      = */ wsp_ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ wsp_ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const size_t TENSOR_ALIGNMENT = 64; // should be enough for AVX 512

WSP_GGML_CALL static const char * wsp_ggml_backend_cpu_buffer_type_get_name(wsp_ggml_backend_buffer_type_t buft) {
    return "CPU";

    WSP_GGML_UNUSED(buft);
}

WSP_GGML_CALL static wsp_ggml_backend_buffer_t wsp_ggml_backend_cpu_buffer_type_alloc_buffer(wsp_ggml_backend_buffer_type_t buft, size_t size) {
    size += TENSOR_ALIGNMENT;   // malloc may return an address that is not aligned
    void * data = malloc(size); // TODO: maybe use WSP_GGML_ALIGNED_MALLOC?

    WSP_GGML_ASSERT(data != NULL && "failed to allocate buffer");

    return wsp_ggml_backend_buffer_init(buft, cpu_backend_buffer_i, data, size);
}

WSP_GGML_CALL static size_t wsp_ggml_backend_cpu_buffer_type_get_alignment(wsp_ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    WSP_GGML_UNUSED(buft);
}

WSP_GGML_CALL static bool wsp_ggml_backend_cpu_buffer_type_supports_backend(wsp_ggml_backend_buffer_type_t buft, wsp_ggml_backend_t backend) {
    return wsp_ggml_backend_is_cpu(backend);

    WSP_GGML_UNUSED(buft);
}

WSP_GGML_CALL static bool wsp_ggml_backend_cpu_buffer_type_is_host(wsp_ggml_backend_buffer_type_t buft) {
    return true;

    WSP_GGML_UNUSED(buft);
}

WSP_GGML_CALL wsp_ggml_backend_buffer_type_t wsp_ggml_backend_cpu_buffer_type(void) {
    static struct wsp_ggml_backend_buffer_type wsp_ggml_backend_cpu_buffer_type = {
        /* .iface = */ {
            /* .get_name         = */ wsp_ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ wsp_ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ wsp_ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_alloc_size   = */ NULL, // defaults to wsp_ggml_nbytes
            /* .supports_backend = */ wsp_ggml_backend_cpu_buffer_type_supports_backend,
            /* .is_host          = */ wsp_ggml_backend_cpu_buffer_type_is_host,
        },
        /* .context = */ NULL,
    };

    return &wsp_ggml_backend_cpu_buffer_type;
}

#ifdef WSP_GGML_USE_CPU_HBM

// buffer type HBM

#include <hbwmalloc.h>

WSP_GGML_CALL static const char * wsp_ggml_backend_cpu_hbm_buffer_type_get_name(wsp_ggml_backend_buffer_type_t buft) {
    return "CPU_HBM";

    WSP_GGML_UNUSED(buft);
}

WSP_GGML_CALL static const char * wsp_ggml_backend_cpu_hbm_buffer_get_name(wsp_ggml_backend_buffer_t buf) {
    return "CPU_HBM";

    WSP_GGML_UNUSED(buf);
}

WSP_GGML_CALL static void wsp_ggml_backend_cpu_hbm_buffer_free_buffer(wsp_ggml_backend_buffer_t buffer) {
    hbw_free(buffer->context);
}

WSP_GGML_CALL static wsp_ggml_backend_buffer_t wsp_ggml_backend_cpu_hbm_buffer_type_alloc_buffer(wsp_ggml_backend_buffer_type_t buft, size_t size) {
    //void * ptr = hbw_malloc(size);
    void * ptr;
    int result = hbw_posix_memalign(&ptr, wsp_ggml_backend_cpu_buffer_type_get_alignment(buft), size);
    if (result != 0) {
        fprintf(stderr, "failed to allocate HBM buffer of size %zu\n", size);
        return NULL;
    }

    wsp_ggml_backend_buffer_t buffer = wsp_ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.get_name = wsp_ggml_backend_cpu_hbm_buffer_get_name;
    buffer->iface.free_buffer = wsp_ggml_backend_cpu_hbm_buffer_free_buffer;

    return buffer;
}

wsp_ggml_backend_buffer_type_t wsp_ggml_backend_cpu_hbm_buffer_type(void) {
    static struct wsp_ggml_backend_buffer_type wsp_ggml_backend_cpu_buffer_type_hbm = {
        /* .iface    = */ {
            /* .get_name         = */ wsp_ggml_backend_cpu_hbm_buffer_type_get_name,
            /* .alloc_buffer     = */ wsp_ggml_backend_cpu_hbm_buffer_type_alloc_buffer,
            /* .get_alignment    = */ wsp_ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_alloc_size   = */ NULL, // defaults to wsp_ggml_nbytes
            /* .supports_backend = */ wsp_ggml_backend_cpu_buffer_type_supports_backend,
            /* .is_host          = */ wsp_ggml_backend_cpu_buffer_type_is_host,
        },
        /* .context  = */ NULL,
    };

    return &wsp_ggml_backend_cpu_buffer_type_hbm;
}
#endif

struct wsp_ggml_backend_cpu_context {
    int n_threads;
    void * work_data;
    size_t work_size;
};

WSP_GGML_CALL static const char * wsp_ggml_backend_cpu_name(wsp_ggml_backend_t backend) {
    return "CPU";

    WSP_GGML_UNUSED(backend);
}

WSP_GGML_CALL static void wsp_ggml_backend_cpu_free(wsp_ggml_backend_t backend) {
    struct wsp_ggml_backend_cpu_context * cpu_ctx = (struct wsp_ggml_backend_cpu_context *)backend->context;
    free(cpu_ctx->work_data);
    free(cpu_ctx);
    free(backend);
}

WSP_GGML_CALL static wsp_ggml_backend_buffer_type_t wsp_ggml_backend_cpu_get_default_buffer_type(wsp_ggml_backend_t backend) {
    return wsp_ggml_backend_cpu_buffer_type();

    WSP_GGML_UNUSED(backend);
}

struct wsp_ggml_backend_plan_cpu {
    struct wsp_ggml_cplan cplan;
    struct wsp_ggml_cgraph cgraph;
};

WSP_GGML_CALL static wsp_ggml_backend_graph_plan_t wsp_ggml_backend_cpu_graph_plan_create(wsp_ggml_backend_t backend, const struct wsp_ggml_cgraph * cgraph) {
    struct wsp_ggml_backend_cpu_context * cpu_ctx = (struct wsp_ggml_backend_cpu_context *)backend->context;

    struct wsp_ggml_backend_plan_cpu * cpu_plan = malloc(sizeof(struct wsp_ggml_backend_plan_cpu));

    cpu_plan->cplan = wsp_ggml_graph_plan(cgraph, cpu_ctx->n_threads);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    if (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = malloc(cpu_plan->cplan.work_size);
    }

    return cpu_plan;
}

WSP_GGML_CALL static void wsp_ggml_backend_cpu_graph_plan_free(wsp_ggml_backend_t backend, wsp_ggml_backend_graph_plan_t plan) {
    struct wsp_ggml_backend_plan_cpu * cpu_plan = (struct wsp_ggml_backend_plan_cpu *)plan;

    free(cpu_plan->cplan.work_data);
    free(cpu_plan);

    WSP_GGML_UNUSED(backend);
}

WSP_GGML_CALL static void wsp_ggml_backend_cpu_graph_plan_compute(wsp_ggml_backend_t backend, wsp_ggml_backend_graph_plan_t plan) {
    struct wsp_ggml_backend_plan_cpu * cpu_plan = (struct wsp_ggml_backend_plan_cpu *)plan;

    wsp_ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    WSP_GGML_UNUSED(backend);
}

WSP_GGML_CALL static bool wsp_ggml_backend_cpu_graph_compute(wsp_ggml_backend_t backend, struct wsp_ggml_cgraph * cgraph) {
    struct wsp_ggml_backend_cpu_context * cpu_ctx = (struct wsp_ggml_backend_cpu_context *)backend->context;

    struct wsp_ggml_cplan cplan = wsp_ggml_graph_plan(cgraph, cpu_ctx->n_threads);

    if (cpu_ctx->work_size < cplan.work_size) {
        // TODO: may be faster to free and use malloc to avoid the copy
        cpu_ctx->work_data = realloc(cpu_ctx->work_data, cplan.work_size);
        cpu_ctx->work_size = cplan.work_size;
    }

    cplan.work_data = cpu_ctx->work_data;

    wsp_ggml_graph_compute(cgraph, &cplan);
    return true;
}

WSP_GGML_CALL static bool wsp_ggml_backend_cpu_supports_op(wsp_ggml_backend_t backend, const struct wsp_ggml_tensor * op) {
    switch (op->op) {
        case WSP_GGML_OP_CPY:
            return op->type != WSP_GGML_TYPE_IQ2_XXS && op->type != WSP_GGML_TYPE_IQ2_XS; // missing type_traits.from_float
        case WSP_GGML_OP_MUL_MAT:
            return op->src[1]->type == WSP_GGML_TYPE_F32 || op->src[1]->type == wsp_ggml_internal_get_type_traits(op->src[0]->type).vec_dot_type;
        default:
            return true;
    }

    WSP_GGML_UNUSED(backend);
}

static struct wsp_ggml_backend_i cpu_backend_i = {
    /* .get_name                = */ wsp_ggml_backend_cpu_name,
    /* .free                    = */ wsp_ggml_backend_cpu_free,
    /* .get_default_buffer_type = */ wsp_ggml_backend_cpu_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ wsp_ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ wsp_ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_compute      = */ wsp_ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ wsp_ggml_backend_cpu_graph_compute,
    /* .supports_op             = */ wsp_ggml_backend_cpu_supports_op,
};

wsp_ggml_backend_t wsp_ggml_backend_cpu_init(void) {
    struct wsp_ggml_backend_cpu_context * ctx = malloc(sizeof(struct wsp_ggml_backend_cpu_context));

    ctx->n_threads = WSP_GGML_DEFAULT_N_THREADS;
    ctx->work_data = NULL;
    ctx->work_size = 0;

    wsp_ggml_backend_t cpu_backend = malloc(sizeof(struct wsp_ggml_backend));

    *cpu_backend = (struct wsp_ggml_backend) {
        /* .interface = */ cpu_backend_i,
        /* .context   = */ ctx
    };
    return cpu_backend;
}

WSP_GGML_CALL bool wsp_ggml_backend_is_cpu(wsp_ggml_backend_t backend) {
    return backend && backend->iface.get_name == wsp_ggml_backend_cpu_name;
}

void wsp_ggml_backend_cpu_set_n_threads(wsp_ggml_backend_t backend_cpu, int n_threads) {
    WSP_GGML_ASSERT(wsp_ggml_backend_is_cpu(backend_cpu));

    struct wsp_ggml_backend_cpu_context * ctx = (struct wsp_ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

WSP_GGML_CALL wsp_ggml_backend_buffer_t wsp_ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    return wsp_ggml_backend_buffer_init(wsp_ggml_backend_cpu_buffer_type(), cpu_backend_buffer_i_from_ptr, ptr, size);
}

WSP_GGML_CALL static wsp_ggml_backend_t wsp_ggml_backend_reg_cpu_init(const char * params, void * user_data) {
    return wsp_ggml_backend_cpu_init();

    WSP_GGML_UNUSED(params);
    WSP_GGML_UNUSED(user_data);
}


// scheduler

#define WSP_GGML_MAX_BACKENDS 16
#define WSP_GGML_MAX_SPLITS 256
#define WSP_GGML_MAX_SPLIT_INPUTS 16

struct wsp_ggml_backend_sched_split {
    wsp_ggml_tallocr_t tallocr;
    int i_start;
    int i_end;
    struct wsp_ggml_tensor * inputs[WSP_GGML_MAX_SPLIT_INPUTS];
    int n_inputs;
    // graph view of this split
    struct wsp_ggml_cgraph graph;
};

struct wsp_ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split

    int n_backends;
    wsp_ggml_backend_t backends[WSP_GGML_MAX_BACKENDS];
    wsp_ggml_backend_buffer_type_t bufts[WSP_GGML_MAX_BACKENDS];
    wsp_ggml_tallocr_t  tallocs[WSP_GGML_MAX_BACKENDS];

    wsp_ggml_gallocr_t galloc;

    // hash keys of the nodes in the graph
    struct wsp_ggml_hash_set    hash_set;
    // hash values (arrays of [hash_set.size])
    wsp_ggml_tallocr_t *        node_talloc;                     // tallocr assigned to each node (indirectly this is the backend)
    struct wsp_ggml_tensor * (* node_copies)[WSP_GGML_MAX_BACKENDS]; // copies of each node for each destination backend

    // copy of the graph with modified inputs
    struct wsp_ggml_cgraph * graph;

    struct wsp_ggml_backend_sched_split splits[WSP_GGML_MAX_SPLITS];
    int n_splits;

    struct wsp_ggml_context * ctx;

    // align context_buffer to WSP_GGML_MEM_ALIGN
    #ifdef _MSC_VER
    __declspec(align(WSP_GGML_MEM_ALIGN))
    #else
    __attribute__((aligned(WSP_GGML_MEM_ALIGN)))
    #endif
    char context_buffer[WSP_GGML_MAX_SPLITS*WSP_GGML_MAX_SPLIT_INPUTS*sizeof(struct wsp_ggml_tensor) + sizeof(struct wsp_ggml_cgraph)];

    wsp_ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;
};

#define hash_id(node) wsp_ggml_hash_find_or_insert(sched->hash_set, node)
#define node_allocr(node) sched->node_talloc[hash_id(node)]

static bool wsp_ggml_is_view_op(enum wsp_ggml_op op) {
    return op == WSP_GGML_OP_VIEW || op == WSP_GGML_OP_RESHAPE || op == WSP_GGML_OP_PERMUTE || op == WSP_GGML_OP_TRANSPOSE;
}

// returns the priority of the backend, lower is better
static int sched_backend_prio(wsp_ggml_backend_sched_t sched, wsp_ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return INT_MAX;
}

static int sched_allocr_prio(wsp_ggml_backend_sched_t sched, wsp_ggml_tallocr_t allocr) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->tallocs[i] == allocr) {
            return i;
        }
    }
    return INT_MAX;
}

static wsp_ggml_tallocr_t sched_allocr_from_buffer(wsp_ggml_backend_sched_t sched, wsp_ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return NULL;
    }

    // check if this is already allocate in a allocr buffer (from user manual allocations)
    for (int i = 0; i < sched->n_backends; i++) {
        if (wsp_ggml_tallocr_get_buffer(sched->tallocs[i]) == buffer) {
            return sched->tallocs[i];
        }
    }

    // find highest prio backend that supports the buffer type
    for (int i = 0; i < sched->n_backends; i++) {
        if (wsp_ggml_backend_buft_supports_backend(buffer->buft, sched->backends[i])) {
            return sched->tallocs[i];
        }
    }
    WSP_GGML_ASSERT(false && "tensor buffer type not supported by any backend");
}

static wsp_ggml_backend_t get_allocr_backend(wsp_ggml_backend_sched_t sched, wsp_ggml_tallocr_t allocr) {
    if (allocr == NULL) {
        return NULL;
    }
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->tallocs[i] == allocr) {
            return sched->backends[i];
        }
    }
    WSP_GGML_UNREACHABLE();
}

#if 0
static char causes[WSP_GGML_DEFAULT_GRAPH_SIZE*16 + WSP_GGML_MAX_SPLITS*WSP_GGML_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static wsp_ggml_tallocr_t sched_allocr_from_cur(wsp_ggml_backend_sched_t sched, struct wsp_ggml_tensor * node) {
    // assign pre-allocated nodes to their backend
    // dst
    wsp_ggml_tallocr_t cur_allocr = sched_allocr_from_buffer(sched, node->buffer);
    if (cur_allocr != NULL) {
        SET_CAUSE(node, "1.dst");
        return cur_allocr;
    }
    // view_src
    if (node->view_src != NULL) {
        cur_allocr = sched_allocr_from_buffer(sched, node->view_src->buffer);
        if (cur_allocr != NULL) {
            SET_CAUSE(node, "1.vsrc");
            return cur_allocr;
        }
    }
    // assign nodes that use weights to the backend of the weights
    for (int i = 0; i < WSP_GGML_MAX_SRC; i++) {
        const struct wsp_ggml_tensor * src = node->src[i];
        if (src == NULL) {
            break;
        }
        if (src->buffer != NULL && src->buffer->usage == WSP_GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            wsp_ggml_tallocr_t src_allocr = sched_allocr_from_buffer(sched, src->buffer);
            // operations with weights are always run on the same backend as the weights
            SET_CAUSE(node, "1.wgt%d", i);
            return src_allocr;
        }
    }

    return NULL;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        sprintf(buffer, "%zuM", size/1024/1024);
    } else {
        sprintf(buffer, "%zuK", size/1024);
    }
    return buffer;
}

static void sched_print_assignments(wsp_ggml_backend_sched_t sched, struct wsp_ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            wsp_ggml_backend_t split_backend = get_allocr_backend(sched, sched->splits[cur_split].tallocr);
            fprintf(stderr, "\n## SPLIT #%d: %s # %d inputs: ", cur_split, wsp_ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                fprintf(stderr, "[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(wsp_ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            fprintf(stderr, "\n");
            cur_split++;
        }
        struct wsp_ggml_tensor * node = graph->nodes[i];
        if (wsp_ggml_is_view_op(node->op)) {
            continue;
        }
        wsp_ggml_tallocr_t node_allocr = node_allocr(node);
        wsp_ggml_backend_t node_backend = node_allocr ? get_allocr_backend(sched, node_allocr) : NULL; // FIXME:
        fprintf(stderr, "node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s]:", i, wsp_ggml_op_name(node->op), node->name,
            fmt_size(wsp_ggml_nbytes(node)), node_allocr ? wsp_ggml_backend_name(node_backend) : "NULL", GET_CAUSE(node));
        for (int j = 0; j < WSP_GGML_MAX_SRC; j++) {
            struct wsp_ggml_tensor * src = node->src[j];
            if (src == NULL) {
                break;
            }
            wsp_ggml_tallocr_t src_allocr = node_allocr(src);
            wsp_ggml_backend_t src_backend = src_allocr ? get_allocr_backend(sched, src_allocr) : NULL;
            fprintf(stderr, " %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                fmt_size(wsp_ggml_nbytes(src)), src_backend ? wsp_ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
        }
        fprintf(stderr, "\n");
    }
}

// creates a copy of the tensor with the same memory layout
static struct wsp_ggml_tensor * wsp_ggml_dup_tensor_layout(struct wsp_ggml_context * ctx, const struct wsp_ggml_tensor * tensor) {
    struct wsp_ggml_tensor * dup = wsp_ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < WSP_GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}


//#define DEBUG_PASS1
//#define DEBUG_PASS2
//#define DEBUG_PASS3
//#define DEBUG_PASS4

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
static void sched_split_graph(wsp_ggml_backend_sched_t sched, struct wsp_ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->is_reset = false;

    struct wsp_ggml_init_params params = {
        /* .mem_size =   */ sizeof(sched->context_buffer),
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    wsp_ggml_free(sched->ctx);

    sched->ctx = wsp_ggml_init(params);
    if (sched->ctx == NULL) {
        fprintf(stderr, "%s: failed to initialize context\n", __func__);
        WSP_GGML_ASSERT(false);
    }

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct wsp_ggml_tensor * leaf = graph->leafs[i];
        if (node_allocr(leaf) != NULL) {
            // do not overwrite user assignments
            continue;
        }
        node_allocr(leaf) = sched_allocr_from_cur(sched, leaf);
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct wsp_ggml_tensor * node = graph->nodes[i];
        if (node_allocr(node) != NULL) {
            // do not overwrite user assignments
            continue;
        }
        node_allocr(node) = sched_allocr_from_cur(sched, node);
        // src
        for (int j = 0; j < WSP_GGML_MAX_SRC; j++) {
            struct wsp_ggml_tensor * src = node->src[j];
            if (src == NULL) {
                break;
            }
            if (node_allocr(src) == NULL) {
                node_allocr(src) = sched_allocr_from_cur(sched, src);
            }
        }
    }
#ifdef DEBUG_PASS1
    fprintf(stderr, "PASS 1 ASSIGNMENTS\n"); sched_print_assignments(sched, graph);
#endif

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops

    // pass 2.1 expand gpu up
    {
        wsp_ggml_tallocr_t cur_allocr = NULL;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct wsp_ggml_tensor * node = graph->nodes[i];
            if (wsp_ggml_is_view_op(node->op)) {
                continue;
            }
            wsp_ggml_tallocr_t node_allocr = node_allocr(node);
            if (node_allocr != NULL) {
                if (sched_allocr_prio(sched, node_allocr) == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_allocr = NULL;
                } else {
                    cur_allocr = node_allocr;
                }
            } else {
                node_allocr(node) = cur_allocr;
                SET_CAUSE(node, "2.1");
            }
        }
    }

    // pass 2.2 expand gpu down
    {
        wsp_ggml_tallocr_t cur_allocr = NULL;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct wsp_ggml_tensor * node = graph->nodes[i];
            if (wsp_ggml_is_view_op(node->op)) {
                continue;
            }
            wsp_ggml_tallocr_t node_allocr = node_allocr(node);
            if (node_allocr != NULL) {
                if (sched_allocr_prio(sched, node_allocr) == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_allocr = NULL;
                } else {
                    cur_allocr = node_allocr;
                }
            } else {
                node_allocr(node) = cur_allocr;
                SET_CAUSE(node, "2.2");
            }
        }
    }

    // pass 2.3 expand rest up
    {
        wsp_ggml_tallocr_t cur_allocr = NULL;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct wsp_ggml_tensor * node = graph->nodes[i];
            if (wsp_ggml_is_view_op(node->op)) {
                continue;
            }
            wsp_ggml_tallocr_t node_allocr = node_allocr(node);
            if (node_allocr != NULL) {
                cur_allocr = node_allocr;
            } else {
                node_allocr(node) = cur_allocr;
                SET_CAUSE(node, "2.3");
            }
        }
    }

    // pass 2.4 expand rest down
    {
        wsp_ggml_tallocr_t cur_allocr = NULL;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct wsp_ggml_tensor * node = graph->nodes[i];
            if (wsp_ggml_is_view_op(node->op)) {
                continue;
            }
            wsp_ggml_tallocr_t node_allocr = node_allocr(node);
            if (node_allocr != NULL) {
                cur_allocr = node_allocr;
            } else {
                node_allocr(node) = cur_allocr;
                SET_CAUSE(node, "2.4");
            }
        }
    }
#ifdef DEBUG_PASS2
    fprintf(stderr, "PASS 2 ASSIGNMENTS\n"); sched_print_assignments(sched, graph);
#endif

    // pass 3: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct wsp_ggml_tensor * node = graph->nodes[i];
        wsp_ggml_tallocr_t cur_allocr = node_allocr(node);
        if (node->view_src != NULL && cur_allocr == NULL) {
            cur_allocr = node_allocr(node) = node_allocr(node->view_src);
            SET_CAUSE(node, "3.vsrc");
        }
        for (int j = 0; j < WSP_GGML_MAX_SRC; j++) {
            struct wsp_ggml_tensor * src = node->src[j];
            if (src == NULL) {
                break;
            }
            wsp_ggml_tallocr_t src_allocr = node_allocr(src);
            if (src_allocr == NULL) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    node_allocr(src) = node_allocr(src->view_src);
                    SET_CAUSE(src, "3.vsrc");
                } else {
                    node_allocr(src) = cur_allocr;
                    SET_CAUSE(src, "3.cur");
                }
            }
        }
    }
#ifdef DEBUG_PASS3
    fprintf(stderr, "PASS 3 ASSIGNMENTS\n"); sched_print_assignments(sched, graph);
#endif

    // pass 4: split graph, find tensors that need to be copied
    {
        int cur_split = 0;
        // find the backend of the first split, skipping view ops
        for (int i = 0; i < graph->n_nodes; i++) {
            struct wsp_ggml_tensor * node = graph->nodes[i];
            if (!wsp_ggml_is_view_op(node->op)) {
                sched->splits[0].tallocr = node_allocr(node);
                break;
            }
        }
        sched->splits[0].i_start = 0;
        sched->splits[0].n_inputs = 0;
        memset(sched->splits[0].inputs, 0, sizeof(sched->splits[0].inputs)); //HACK
        wsp_ggml_tallocr_t cur_allocr = sched->splits[0].tallocr;
        size_t cur_backend_id = sched_allocr_prio(sched, cur_allocr);
        for (int i = 0; i < graph->n_nodes; i++) {
            struct wsp_ggml_tensor * node = graph->nodes[i];

            if (wsp_ggml_is_view_op(node->op)) {
                continue;
            }

            wsp_ggml_tallocr_t node_allocr = node_allocr(node);

            WSP_GGML_ASSERT(node_allocr != NULL); // all nodes should be assigned by now

            if (node_allocr != cur_allocr) {
                sched->splits[cur_split].i_end = i;
                cur_split++;
                WSP_GGML_ASSERT(cur_split < WSP_GGML_MAX_SPLITS);
                sched->splits[cur_split].tallocr = node_allocr;
                sched->splits[cur_split].i_start = i;
                sched->splits[cur_split].n_inputs = 0;
                cur_allocr = node_allocr;
                cur_backend_id = sched_allocr_prio(sched, cur_allocr);
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < WSP_GGML_MAX_SRC; j++) {
                struct wsp_ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    break;
                }
                wsp_ggml_tallocr_t src_allocr = node_allocr(src);
                WSP_GGML_ASSERT(src_allocr != NULL); // all inputs should be assigned by now
                if (src_allocr != node_allocr) {
                    // check if the input is already in the split
                    bool found = false;
                    for (int k = 0; k < sched->splits[cur_split].n_inputs; k++) {
                        if (sched->splits[cur_split].inputs[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        int n_inputs = sched->splits[cur_split].n_inputs++;
                        //printf("split %d input %d: %s (%s)\n", cur_split, n_inputs, src->name, wsp_ggml_backend_name(get_allocr_backend(sched, src_allocr)));
                        WSP_GGML_ASSERT(n_inputs < WSP_GGML_MAX_SPLIT_INPUTS);
                        sched->splits[cur_split].inputs[n_inputs] = src;
                    }

                    // create a copy of the input in the split's backend
                    size_t id = hash_id(src);
                    if (sched->node_copies[id][cur_backend_id] == NULL) {
                        wsp_ggml_backend_t backend = get_allocr_backend(sched, cur_allocr);
                        struct wsp_ggml_tensor * tensor_copy = wsp_ggml_dup_tensor_layout(sched->ctx, src);
                        wsp_ggml_format_name(tensor_copy, "%s#%s", wsp_ggml_backend_name(backend), src->name);

                        sched->node_copies[id][cur_backend_id] = tensor_copy;
                        node_allocr(tensor_copy) = cur_allocr;
                        SET_CAUSE(tensor_copy, "4.cpy");
                    }
                    node->src[j] = sched->node_copies[id][cur_backend_id];
                }
            }
        }
        sched->splits[cur_split].i_end = graph->n_nodes;
        sched->n_splits = cur_split + 1;
    }
#ifdef DEBUG_PASS4
    fprintf(stderr, "PASS 4 ASSIGNMENTS\n"); sched_print_assignments(sched, graph);
#endif

#ifndef NDEBUG
    // sanity check: all sources should have the same backend as the node
    for (int i = 0; i < graph->n_nodes; i++) {
        struct wsp_ggml_tensor * node = graph->nodes[i];
        wsp_ggml_tallocr_t node_allocr = node_allocr(node);
        if (node_allocr == NULL) {
            fprintf(stderr, "!!!!!!! %s has no backend\n", node->name);
        }
        if (node->view_src != NULL && node_allocr != node_allocr(node->view_src)) {
            fprintf(stderr, "!!!!!!! %s has backend %s, view_src %s has backend %s\n",
                node->name, node_allocr ? wsp_ggml_backend_name(get_allocr_backend(sched, node_allocr)) : "NULL",
                node->view_src->name, node_allocr(node->view_src) ? wsp_ggml_backend_name(get_allocr_backend(sched, node_allocr(node->view_src))) : "NULL");
        }
        for (int j = 0; j < WSP_GGML_MAX_SRC; j++) {
            struct wsp_ggml_tensor * src = node->src[j];
            if (src == NULL) {
                break;
            }
            wsp_ggml_tallocr_t src_allocr = node_allocr(src);
            if (src_allocr != node_allocr /* && src_backend != NULL */) { // ignore nulls for now
                fprintf(stderr, "!!!! %s has backend %s, src %d (%s) has backend %s\n",
                    node->name, node_allocr ? wsp_ggml_backend_name(get_allocr_backend(sched, node_allocr)) : "NULL",
                    j, src->name, src_allocr ? wsp_ggml_backend_name(get_allocr_backend(sched, src_allocr)) : "NULL");
            }
            if (src->view_src != NULL && src_allocr != node_allocr(src->view_src)) {
                fprintf(stderr, "!!!!!!! [src] %s has backend %s, view_src %s has backend %s\n",
                    src->name, src_allocr ? wsp_ggml_backend_name(get_allocr_backend(sched, src_allocr)) : "NULL",
                    src->view_src->name, node_allocr(src->view_src) ? wsp_ggml_backend_name(get_allocr_backend(sched, node_allocr(src->view_src))) : "NULL");
            }
        }
    }
    fflush(stderr);
#endif

    // create copies of the graph for each split
    // FIXME: avoid this copy, pass split inputs to wsp_ggml_gallocr_alloc_graph_n in some other way
    struct wsp_ggml_cgraph * graph_copy = wsp_ggml_new_graph_custom(sched->ctx, graph->n_nodes + sched->n_splits*WSP_GGML_MAX_SPLIT_INPUTS, false);
    for (int i = 0; i < sched->n_splits; i++) {
        struct wsp_ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = wsp_ggml_graph_view(graph, split->i_start, split->i_end);

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            struct wsp_ggml_tensor * input = split->inputs[j];
            struct wsp_ggml_tensor * input_cpy = sched->node_copies[hash_id(input)][sched_allocr_prio(sched, split->tallocr)];
            // add a dependency to the input source so that it is not freed before the copy is done
            WSP_GGML_ASSERT(input_cpy->src[0] == NULL || input_cpy->src[0] == input);
            input_cpy->src[0] = input;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }
    sched->graph = graph_copy;
}

static void sched_alloc_splits(wsp_ggml_backend_sched_t sched) {
    wsp_ggml_gallocr_alloc_graph_n(
        sched->galloc,
        sched->graph,
        sched->hash_set,
        sched->node_talloc);
}

static void sched_compute_splits(wsp_ggml_backend_sched_t sched) {
    uint64_t copy_us[WSP_GGML_MAX_BACKENDS] = {0};
    uint64_t compute_us[WSP_GGML_MAX_BACKENDS] = {0};

    struct wsp_ggml_backend_sched_split * splits = sched->splits;

    for (int i = 0; i < sched->n_splits; i++) {
        struct wsp_ggml_backend_sched_split * split = &splits[i];
        wsp_ggml_backend_t split_backend = get_allocr_backend(sched, split->tallocr);
        int split_backend_id = sched_backend_prio(sched, split_backend);

        // copy the input tensors to the split backend
        uint64_t copy_start_us = wsp_ggml_time_us();
        for (int j = 0; j < split->n_inputs; j++) {
            struct wsp_ggml_tensor * input = split->inputs[j];
            struct wsp_ggml_tensor * input_cpy = sched->node_copies[hash_id(input)][split_backend_id];

            WSP_GGML_ASSERT(input->buffer != NULL);
            WSP_GGML_ASSERT(input_cpy->buffer != NULL);

            // TODO: avoid this copy if it was already copied in a previous split, and the input didn't change
            // this is important to avoid copying constants such as KQ_mask and inp_pos multiple times
            wsp_ggml_backend_tensor_copy_async(split_backend, input, input_cpy);
        }
        //wsp_ggml_backend_synchronize(split_backend); // necessary to measure copy time
        int64_t copy_end_us = wsp_ggml_time_us();
        copy_us[split_backend_id] += copy_end_us - copy_start_us;

#if 0
        char split_filename[WSP_GGML_MAX_NAME];
        snprintf(split_filename, WSP_GGML_MAX_NAME, "split_%i_%s.dot", i, wsp_ggml_backend_name(split_backend));
        wsp_ggml_graph_dump_dot(split->graph, NULL, split_filename);
#endif


        uint64_t compute_start_us = wsp_ggml_time_us();
        if (!sched->callback_eval) {
            wsp_ggml_backend_graph_compute(split_backend, &split->graph);
          //wsp_ggml_backend_synchronize(split_backend); // necessary to measure compute time
        } else {
            // similar to wsp_ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct wsp_ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct wsp_ggml_cgraph gv = wsp_ggml_graph_view(&split->graph, j0, j1 + 1);

                wsp_ggml_backend_graph_compute(split_backend, &gv);

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }
        uint64_t compute_end_us = wsp_ggml_time_us();
        compute_us[split_backend_id] += compute_end_us - compute_start_us;
    }

#if 0
    // per-backend timings
    fprintf(stderr, "sched_compute_splits times (%d splits):\n", sched->n_splits);
    for (int i = 0; i < sched->n_backends; i++) {
        if (copy_us[i] > 0 || compute_us[i] > 0) {
            fprintf(stderr, "\t%5.5s: %lu us copy, %lu us compute\n", wsp_ggml_backend_name(sched->backends[i]), copy_us[i], compute_us[i]);
        }
    }
#endif
}

static void sched_reset(wsp_ggml_backend_sched_t sched) {
    for (int i = 0; i < sched->n_backends; i++) {
        wsp_ggml_tallocr_reset(sched->tallocs[i]);
    }
    // reset state for the next run
    size_t hash_size = sched->hash_set.size;
    memset(sched->hash_set.keys, 0, sizeof(sched->hash_set.keys[0]) * hash_size);
    memset(sched->node_talloc,   0, sizeof(sched->node_talloc[0])   * hash_size);
    memset(sched->node_copies,   0, sizeof(sched->node_copies[0])   * hash_size);

    sched->is_reset = true;
}

wsp_ggml_backend_sched_t wsp_ggml_backend_sched_new(wsp_ggml_backend_t * backends, wsp_ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size) {
    WSP_GGML_ASSERT(n_backends > 0);
    WSP_GGML_ASSERT(n_backends <= WSP_GGML_MAX_BACKENDS);

    struct wsp_ggml_backend_sched * sched = calloc(sizeof(struct wsp_ggml_backend_sched), 1);

    // initialize hash table
    sched->hash_set    = wsp_ggml_hash_set_new(graph_size + WSP_GGML_MAX_SPLITS*WSP_GGML_MAX_SPLIT_INPUTS);
    sched->node_talloc = calloc(sizeof(sched->node_talloc[0]) * sched->hash_set.size, 1);
    sched->node_copies = calloc(sizeof(sched->node_copies[0]) * sched->hash_set.size, 1);

    sched->n_backends = n_backends;
    for (int i = 0; i < n_backends; i++) {
        sched->backends[i] = backends[i];
        sched->bufts[i] = bufts ? bufts[i] : wsp_ggml_backend_get_default_buffer_type(backends[i]);
    }

    sched->galloc = wsp_ggml_gallocr_new();

    // init measure allocs for each backend
    for (int i = 0; i < n_backends; i++) {
        sched->tallocs[i] = wsp_ggml_tallocr_new_measure_from_buft(sched->bufts[i]);
    }

    sched_reset(sched);

    return sched;
}

void wsp_ggml_backend_sched_free(wsp_ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    for (int i = 0; i < sched->n_backends; i++) {
        wsp_ggml_tallocr_free(sched->tallocs[i]);
    }
    wsp_ggml_gallocr_free(sched->galloc);
    wsp_ggml_free(sched->ctx);
    free(sched->hash_set.keys);
    free(sched->node_talloc);
    free(sched->node_copies);
    free(sched);
}

void wsp_ggml_backend_sched_init_measure(wsp_ggml_backend_sched_t sched, struct wsp_ggml_cgraph * measure_graph) {
    WSP_GGML_ASSERT(wsp_ggml_tallocr_is_measure(sched->tallocs[0])); // can only be initialized once

    sched_split_graph(sched, measure_graph);
    sched_alloc_splits(sched);

    // allocate buffers and reset allocators
    for (int i = 0; i < sched->n_backends; i++) {
        size_t size = wsp_ggml_tallocr_max_size(sched->tallocs[i]);
        wsp_ggml_tallocr_free(sched->tallocs[i]);
        sched->tallocs[i] = wsp_ggml_tallocr_new_from_buft(sched->bufts[i], size);
    }

    sched_reset(sched);
}

void wsp_ggml_backend_sched_graph_compute(wsp_ggml_backend_sched_t sched, struct wsp_ggml_cgraph * graph) {
    WSP_GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + WSP_GGML_MAX_SPLITS*WSP_GGML_MAX_SPLIT_INPUTS);

    if (!sched->is_reset) {
        sched_reset(sched);
    }

    sched_split_graph(sched, graph);
    sched_alloc_splits(sched);
    sched_compute_splits(sched);
}

void wsp_ggml_backend_sched_reset(wsp_ggml_backend_sched_t sched) {
    sched_reset(sched);
}


void wsp_ggml_backend_sched_set_eval_callback(wsp_ggml_backend_sched_t sched, wsp_ggml_backend_sched_eval_callback callback, void * user_data) {
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

int wsp_ggml_backend_sched_get_n_splits(wsp_ggml_backend_sched_t sched) {
    return sched->n_splits;
}

wsp_ggml_tallocr_t wsp_ggml_backend_sched_get_tallocr(wsp_ggml_backend_sched_t sched, wsp_ggml_backend_t backend) {
    int backend_index = sched_backend_prio(sched, backend);
    WSP_GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    return sched->tallocs[backend_index];
}

wsp_ggml_backend_buffer_t wsp_ggml_backend_sched_get_buffer(wsp_ggml_backend_sched_t sched, wsp_ggml_backend_t backend) {
    int backend_index = sched_backend_prio(sched, backend);
    WSP_GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    return wsp_ggml_tallocr_get_buffer(sched->tallocs[backend_index]);
}

void wsp_ggml_backend_sched_set_node_backend(wsp_ggml_backend_sched_t sched, struct wsp_ggml_tensor * node, wsp_ggml_backend_t backend) {
    int backend_index = sched_backend_prio(sched, backend);
    WSP_GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    node_allocr(node) = sched->tallocs[backend_index];
}

wsp_ggml_backend_t wsp_ggml_backend_sched_get_node_backend(wsp_ggml_backend_sched_t sched, struct wsp_ggml_tensor * node) {
    wsp_ggml_tallocr_t allocr = node_allocr(node);
    if (allocr == NULL) {
        return NULL;
    }
    return get_allocr_backend(sched, allocr);
}

// utils

void wsp_ggml_backend_view_init(wsp_ggml_backend_buffer_t buffer, struct wsp_ggml_tensor * tensor) {
    WSP_GGML_ASSERT(tensor->buffer == NULL);
    //WSP_GGML_ASSERT(tensor->data == NULL); // views of pre-allocated tensors may have the data set in wsp_ggml_new_tensor, but still need to be initialized by the backend
    WSP_GGML_ASSERT(tensor->view_src != NULL);
    WSP_GGML_ASSERT(tensor->view_src->buffer != NULL);
    WSP_GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    tensor->backend = tensor->view_src->backend;
    wsp_ggml_backend_buffer_init_tensor(buffer, tensor);
}

void wsp_ggml_backend_tensor_alloc(wsp_ggml_backend_buffer_t buffer, struct wsp_ggml_tensor * tensor, void * addr) {
    WSP_GGML_ASSERT(tensor->buffer == NULL);
    WSP_GGML_ASSERT(tensor->data == NULL);
    WSP_GGML_ASSERT(tensor->view_src == NULL);
    WSP_GGML_ASSERT(addr >= wsp_ggml_backend_buffer_get_base(buffer));
    WSP_GGML_ASSERT((char *)addr + wsp_ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
                (char *)wsp_ggml_backend_buffer_get_base(buffer) + wsp_ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    wsp_ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct wsp_ggml_tensor * graph_dup_tensor(struct wsp_ggml_hash_set hash_set, struct wsp_ggml_tensor ** node_copies,
    struct wsp_ggml_context * ctx_allocated, struct wsp_ggml_context * ctx_unallocated, struct wsp_ggml_tensor * src) {

    WSP_GGML_ASSERT(src != NULL);
    WSP_GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = wsp_ggml_hash_insert(hash_set, src);
    if (id == WSP_GGML_HASHTABLE_ALREADY_EXISTS) {
        return node_copies[wsp_ggml_hash_find(hash_set, src)];
    }

    struct wsp_ggml_tensor * dst = wsp_ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    wsp_ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < WSP_GGML_MAX_SRC; i++) {
        struct wsp_ggml_tensor * s = src->src[i];
        if (s == NULL) {
            break;
        }
        dst->src[i] = graph_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_init_tensor(struct wsp_ggml_hash_set hash_set, struct wsp_ggml_tensor ** node_copies, bool * node_init, struct wsp_ggml_tensor * src) {
    size_t id = wsp_ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct wsp_ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_init_tensor(hash_set, node_copies, node_init, src->view_src);
        wsp_ggml_backend_view_init(dst->view_src->buffer, dst);
    }
    else {
        wsp_ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < WSP_GGML_MAX_SRC; i++) {
        struct wsp_ggml_tensor * s = src->src[i];
        if (s == NULL) {
            break;
        }
        graph_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct wsp_ggml_backend_graph_copy wsp_ggml_backend_graph_copy(wsp_ggml_backend_t backend, struct wsp_ggml_cgraph * graph) {
    struct wsp_ggml_hash_set hash_set = {
        /* .size = */ graph->visited_hash_table.size,
        /* .keys = */ calloc(sizeof(hash_set.keys[0]) * graph->visited_hash_table.size, 1)
    };
    struct wsp_ggml_tensor ** node_copies = calloc(sizeof(node_copies[0]) * hash_set.size, 1);
    bool * node_init = calloc(sizeof(node_init[0]) * hash_set.size, 1);

    struct wsp_ggml_init_params params = {
        /* .mem_size   = */ wsp_ggml_tensor_overhead()*hash_set.size + wsp_ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct wsp_ggml_context * ctx_allocated = wsp_ggml_init(params);
    struct wsp_ggml_context * ctx_unallocated = wsp_ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        fprintf(stderr, "failed to allocate context for graph copy\n");
        free(hash_set.keys);
        free(node_copies);
        free(node_init);
        wsp_ggml_free(ctx_allocated);
        wsp_ggml_free(ctx_unallocated);
        return (struct wsp_ggml_backend_graph_copy) {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct wsp_ggml_tensor * node = graph->nodes[i];
        graph_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    wsp_ggml_backend_buffer_t buffer = wsp_ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        fprintf(stderr, "failed to allocate buffer for graph copy\n");
        free(hash_set.keys);
        free(node_copies);
        free(node_init);
        wsp_ggml_free(ctx_allocated);
        wsp_ggml_free(ctx_unallocated);
        return (struct wsp_ggml_backend_graph_copy) {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", wsp_ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct wsp_ggml_tensor * node = graph->nodes[i];
        graph_init_tensor(hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct wsp_ggml_cgraph * graph_copy = wsp_ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct wsp_ggml_tensor * node = graph->nodes[i];
        struct wsp_ggml_tensor * node_copy = node_copies[wsp_ggml_hash_find(hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    free(hash_set.keys);
    free(node_copies);
    free(node_init);

    return (struct wsp_ggml_backend_graph_copy) {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void wsp_ggml_backend_graph_copy_free(struct wsp_ggml_backend_graph_copy copy) {
    wsp_ggml_backend_buffer_free(copy.buffer);
    wsp_ggml_free(copy.ctx_allocated);
    wsp_ggml_free(copy.ctx_unallocated);
}

bool wsp_ggml_backend_compare_graph_backend(wsp_ggml_backend_t backend1, wsp_ggml_backend_t backend2, struct wsp_ggml_cgraph * graph, wsp_ggml_backend_eval_callback callback, void * user_data) {
    struct wsp_ggml_backend_graph_copy copy = wsp_ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct wsp_ggml_cgraph * g1 = graph;
    struct wsp_ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    for (int i = 0; i < g1->n_nodes; i++) {
        //printf("eval %d/%d\n", i, g1->n_nodes);
        struct wsp_ggml_tensor * t1 = g1->nodes[i];
        struct wsp_ggml_tensor * t2 = g2->nodes[i];

        assert(t1->op == t2->op && wsp_ggml_are_same_layout(t1, t2));

        struct wsp_ggml_cgraph g1v = wsp_ggml_graph_view(g1, i, i + 1);
        struct wsp_ggml_cgraph g2v = wsp_ggml_graph_view(g2, i, i + 1);

        wsp_ggml_backend_graph_compute(backend1, &g1v);
        wsp_ggml_backend_graph_compute(backend2, &g2v);

        if (wsp_ggml_is_view_op(t1->op)) {
            continue;
        }

        // compare results, calculate rms etc
        if (!callback(i, t1, t2, user_data)) {
            break;
        }
    }

    wsp_ggml_backend_graph_copy_free(copy);

    return true;
}
