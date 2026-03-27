#include "ggml-metal.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-metal-device.h"
#include "ggml-metal-context.h"
#include "ggml-metal-ops.h"

#include <mutex>
#include <string>

#define WSP_GGML_METAL_NAME "MTL"
#define WSP_GGML_METAL_MAX_DEVICES 16

// number of Metal devices
// note: can be overridden with WSP_GGML_METAL_DEVICES env to simulate virtual devices
static int g_devices = 1;

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

static void wsp_ggml_backend_metal_buffer_shared_free_buffer(wsp_ggml_backend_buffer_t buffer) {
    wsp_ggml_metal_buffer_t ctx = (wsp_ggml_metal_buffer_t)buffer->context;

    WSP_GGML_ASSERT(wsp_ggml_metal_buffer_is_shared(ctx));

    wsp_ggml_metal_buffer_free(ctx);
}

static void * wsp_ggml_backend_metal_buffer_shared_get_base(wsp_ggml_backend_buffer_t buffer) {
    wsp_ggml_metal_buffer_t ctx = (wsp_ggml_metal_buffer_t)buffer->context;

    WSP_GGML_ASSERT(wsp_ggml_metal_buffer_is_shared(ctx));

    return wsp_ggml_metal_buffer_get_base(ctx);
}

static void wsp_ggml_backend_metal_buffer_shared_memset_tensor(wsp_ggml_backend_buffer_t buffer, wsp_ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    wsp_ggml_metal_buffer_t ctx = (wsp_ggml_metal_buffer_t)buffer->context;

    WSP_GGML_ASSERT(wsp_ggml_metal_buffer_is_shared(ctx));

    wsp_ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

static void wsp_ggml_backend_metal_buffer_shared_set_tensor(wsp_ggml_backend_buffer_t buffer, wsp_ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    wsp_ggml_metal_buffer_t ctx = (wsp_ggml_metal_buffer_t)buffer->context;

    WSP_GGML_ASSERT(wsp_ggml_metal_buffer_is_shared(ctx));

    wsp_ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

static void wsp_ggml_backend_metal_buffer_shared_get_tensor(wsp_ggml_backend_buffer_t buffer, const wsp_ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    wsp_ggml_metal_buffer_t ctx = (wsp_ggml_metal_buffer_t)buffer->context;

    WSP_GGML_ASSERT(wsp_ggml_metal_buffer_is_shared(ctx));

    wsp_ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

static bool wsp_ggml_backend_metal_buffer_shared_cpy_tensor(wsp_ggml_backend_buffer_t buffer, const wsp_ggml_tensor * src, wsp_ggml_tensor * dst) {
    wsp_ggml_metal_buffer_t ctx = (wsp_ggml_metal_buffer_t)buffer->context;

    WSP_GGML_ASSERT(wsp_ggml_metal_buffer_is_shared(ctx));

    WSP_GGML_UNUSED(buffer);
    WSP_GGML_UNUSED(src);
    WSP_GGML_UNUSED(dst);

    return false;
}

static void wsp_ggml_backend_metal_buffer_shared_clear(wsp_ggml_backend_buffer_t buffer, uint8_t value) {
    wsp_ggml_metal_buffer_t ctx = (wsp_ggml_metal_buffer_t)buffer->context;

    WSP_GGML_ASSERT(wsp_ggml_metal_buffer_is_shared(ctx));

    wsp_ggml_metal_buffer_clear(ctx, value);
}

static wsp_ggml_backend_buffer_i wsp_ggml_backend_metal_buffer_shared_i = {
    /* .free_buffer     = */ wsp_ggml_backend_metal_buffer_shared_free_buffer,
    /* .get_base        = */ wsp_ggml_backend_metal_buffer_shared_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ wsp_ggml_backend_metal_buffer_shared_memset_tensor,
    /* .set_tensor      = */ wsp_ggml_backend_metal_buffer_shared_set_tensor,
    /* .get_tensor      = */ wsp_ggml_backend_metal_buffer_shared_get_tensor,
    /* .cpy_tensor      = */ wsp_ggml_backend_metal_buffer_shared_cpy_tensor,
    /* .clear           = */ wsp_ggml_backend_metal_buffer_shared_clear,
    /* .reset           = */ NULL,
};

// private buffer

static void wsp_ggml_backend_metal_buffer_private_free_buffer(wsp_ggml_backend_buffer_t buffer) {
    wsp_ggml_metal_buffer_t ctx = (wsp_ggml_metal_buffer_t)buffer->context;

    WSP_GGML_ASSERT(!wsp_ggml_metal_buffer_is_shared(ctx));

    wsp_ggml_metal_buffer_free(ctx);
}

static void * wsp_ggml_backend_metal_buffer_private_get_base(wsp_ggml_backend_buffer_t buffer) {
    wsp_ggml_metal_buffer_t ctx = (wsp_ggml_metal_buffer_t)buffer->context;

    WSP_GGML_ASSERT(!wsp_ggml_metal_buffer_is_shared(ctx));

    return wsp_ggml_metal_buffer_get_base(ctx);
}

static void wsp_ggml_backend_metal_buffer_private_memset_tensor(wsp_ggml_backend_buffer_t buffer, wsp_ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    wsp_ggml_metal_buffer_t ctx = (wsp_ggml_metal_buffer_t)buffer->context;

    WSP_GGML_ASSERT(!wsp_ggml_metal_buffer_is_shared(ctx));

    wsp_ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

static void wsp_ggml_backend_metal_buffer_private_set_tensor(wsp_ggml_backend_buffer_t buffer, wsp_ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    wsp_ggml_metal_buffer_t ctx = (wsp_ggml_metal_buffer_t)buffer->context;

    WSP_GGML_ASSERT(!wsp_ggml_metal_buffer_is_shared(ctx));

    wsp_ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

static void wsp_ggml_backend_metal_buffer_private_get_tensor(wsp_ggml_backend_buffer_t buffer, const wsp_ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    wsp_ggml_metal_buffer_t ctx = (wsp_ggml_metal_buffer_t)buffer->context;

    WSP_GGML_ASSERT(!wsp_ggml_metal_buffer_is_shared(ctx));

    wsp_ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

static bool wsp_ggml_backend_metal_buffer_private_cpy_tensor(wsp_ggml_backend_buffer_t buffer, const wsp_ggml_tensor * src, wsp_ggml_tensor * dst) {
    wsp_ggml_metal_buffer_t ctx = (wsp_ggml_metal_buffer_t)buffer->context;

    WSP_GGML_ASSERT(!wsp_ggml_metal_buffer_is_shared(ctx));

    WSP_GGML_UNUSED(buffer);
    WSP_GGML_UNUSED(src);
    WSP_GGML_UNUSED(dst);

    return false;
}

static void wsp_ggml_backend_metal_buffer_private_clear(wsp_ggml_backend_buffer_t buffer, uint8_t value) {
    wsp_ggml_metal_buffer_t ctx = (wsp_ggml_metal_buffer_t)buffer->context;

    WSP_GGML_ASSERT(!wsp_ggml_metal_buffer_is_shared(ctx));

    wsp_ggml_metal_buffer_clear(ctx, value);
}

static wsp_ggml_backend_buffer_i wsp_ggml_backend_metal_buffer_private_i = {
    /* .free_buffer     = */ wsp_ggml_backend_metal_buffer_private_free_buffer,
    /* .get_base        = */ wsp_ggml_backend_metal_buffer_private_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ wsp_ggml_backend_metal_buffer_private_memset_tensor,
    /* .set_tensor      = */ wsp_ggml_backend_metal_buffer_private_set_tensor,
    /* .get_tensor      = */ wsp_ggml_backend_metal_buffer_private_get_tensor,
    /* .cpy_tensor      = */ wsp_ggml_backend_metal_buffer_private_cpy_tensor,
    /* .clear           = */ wsp_ggml_backend_metal_buffer_private_clear,
    /* .reset           = */ NULL,
};

static bool wsp_ggml_backend_buffer_is_metal(wsp_ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == wsp_ggml_backend_metal_buffer_shared_free_buffer ||
           buffer->iface.free_buffer == wsp_ggml_backend_metal_buffer_private_free_buffer;
}

//
// buffer types
//

struct wsp_ggml_backend_metal_buffer_type {
    int device;
    std::string name;
};

struct wsp_ggml_backend_metal_buffer_type_deleter {
    void operator()(wsp_ggml_backend_metal_buffer_type * ctx) const {
        delete ctx;
    }
};

typedef std::unique_ptr<wsp_ggml_backend_metal_buffer_type, wsp_ggml_backend_metal_buffer_type_deleter> wsp_ggml_backend_metal_buffer_type_ptr;

// common method for allocating shread or private Metal buffers
static wsp_ggml_backend_buffer_t wsp_ggml_backend_metal_buffer_type_alloc_buffer(wsp_ggml_backend_buffer_type_t buft, size_t size, bool shared) {
    wsp_ggml_metal_device_t ctx_dev = (wsp_ggml_metal_device_t)buft->device->context;
    wsp_ggml_metal_buffer_t res = wsp_ggml_metal_buffer_init(ctx_dev, size, shared);

    wsp_ggml_backend_buffer_i buf_i = wsp_ggml_metal_buffer_is_shared(res)
        ? wsp_ggml_backend_metal_buffer_shared_i
        : wsp_ggml_backend_metal_buffer_private_i;

    return wsp_ggml_backend_buffer_init(buft, buf_i, res, size);
}

static size_t wsp_ggml_backend_metal_buffer_type_get_alloc_size(wsp_ggml_backend_buffer_type_t buft, const wsp_ggml_tensor * tensor) {
    size_t res = wsp_ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    switch (tensor->op) {
        case WSP_GGML_OP_MUL_MAT_ID:
            {
                res += wsp_ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += wsp_ggml_metal_op_mul_mat_id_extra_ids(tensor);
            } break;
        case WSP_GGML_OP_FLASH_ATTN_EXT:
            {
                res += wsp_ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += wsp_ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += wsp_ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
            } break;
        case WSP_GGML_OP_CUMSUM:
        case WSP_GGML_OP_ARGSORT:
            {
                res *= 2;
            } break;
        case WSP_GGML_OP_TOP_K:
            {
                res = 2*sizeof(int32_t)*wsp_ggml_nelements(tensor->src[0]);
            } break;
        default:
            break;
    }

    return res;

    WSP_GGML_UNUSED(buft);
}

// default (shared) buffer type

static const char * wsp_ggml_backend_metal_buffer_type_shared_get_name(wsp_ggml_backend_buffer_type_t buft) {
    wsp_ggml_backend_metal_buffer_type * ctx = (wsp_ggml_backend_metal_buffer_type *)buft->context;

    return ctx->name.c_str();
}

static wsp_ggml_backend_buffer_t wsp_ggml_backend_metal_buffer_type_shared_alloc_buffer(wsp_ggml_backend_buffer_type_t buft, size_t size) {
    return wsp_ggml_backend_metal_buffer_type_alloc_buffer(buft, size, true);
}

static size_t wsp_ggml_backend_metal_buffer_type_shared_get_alignment(wsp_ggml_backend_buffer_type_t buft) {
    return 32;

    WSP_GGML_UNUSED(buft);
}

static size_t wsp_ggml_backend_metal_buffer_type_shared_get_max_size(wsp_ggml_backend_buffer_type_t buft) {
    wsp_ggml_metal_device_t ctx_dev = (wsp_ggml_metal_device_t)buft->device->context;

    return wsp_ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t wsp_ggml_backend_metal_buffer_type_shared_get_alloc_size(wsp_ggml_backend_buffer_type_t buft, const wsp_ggml_tensor * tensor) {
    return wsp_ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool wsp_ggml_backend_metal_buffer_type_shared_is_host(wsp_ggml_backend_buffer_type_t buft) {
    return false;

    WSP_GGML_UNUSED(buft);
}

static wsp_ggml_backend_buffer_type_t wsp_ggml_backend_metal_buffer_type_shared(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<wsp_ggml_backend_buffer_type> bufts;
    static std::vector<wsp_ggml_backend_metal_buffer_type_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            wsp_ggml_backend_metal_buffer_type * raw_ctx =
                new wsp_ggml_backend_metal_buffer_type {
                    /* .device = */ i,
                    /* .name   = */ WSP_GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            wsp_ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name         = */ wsp_ggml_backend_metal_buffer_type_shared_get_name,
                    /* .alloc_buffer     = */ wsp_ggml_backend_metal_buffer_type_shared_alloc_buffer,
                    /* .get_alignment    = */ wsp_ggml_backend_metal_buffer_type_shared_get_alignment,
                    /* .get_max_size     = */ wsp_ggml_backend_metal_buffer_type_shared_get_max_size,
                    /* .get_alloc_size   = */ wsp_ggml_backend_metal_buffer_type_shared_get_alloc_size,
                    /* .is_host          = */ wsp_ggml_backend_metal_buffer_type_shared_is_host,
                },
                /* .device  = */ wsp_ggml_backend_reg_dev_get(wsp_ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// default (private) buffer type

static const char * wsp_ggml_backend_metal_buffer_type_private_get_name(wsp_ggml_backend_buffer_type_t buft) {
    wsp_ggml_backend_metal_buffer_type * ctx = (wsp_ggml_backend_metal_buffer_type *)buft->context;

    return ctx->name.c_str();
}

static wsp_ggml_backend_buffer_t wsp_ggml_backend_metal_buffer_type_private_alloc_buffer(wsp_ggml_backend_buffer_type_t buft, size_t size) {
    return wsp_ggml_backend_metal_buffer_type_alloc_buffer(buft, size, false);
}

static size_t wsp_ggml_backend_metal_buffer_type_private_get_alignment(wsp_ggml_backend_buffer_type_t buft) {
    return 32;

    WSP_GGML_UNUSED(buft);
}

static size_t wsp_ggml_backend_metal_buffer_type_private_get_max_size(wsp_ggml_backend_buffer_type_t buft) {
    wsp_ggml_metal_device_t ctx_dev = (wsp_ggml_metal_device_t)buft->device->context;

    return wsp_ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t wsp_ggml_backend_metal_buffer_type_private_get_alloc_size(wsp_ggml_backend_buffer_type_t buft, const wsp_ggml_tensor * tensor) {
    return wsp_ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool wsp_ggml_backend_metal_buffer_type_private_is_host(wsp_ggml_backend_buffer_type_t buft) {
    return false;

    WSP_GGML_UNUSED(buft);
}

static wsp_ggml_backend_buffer_type_t wsp_ggml_backend_metal_buffer_type_private(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<wsp_ggml_backend_buffer_type> bufts;
    static std::vector<wsp_ggml_backend_metal_buffer_type_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            wsp_ggml_backend_metal_buffer_type * raw_ctx = new wsp_ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ WSP_GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            wsp_ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name         = */ wsp_ggml_backend_metal_buffer_type_private_get_name,
                    /* .alloc_buffer     = */ wsp_ggml_backend_metal_buffer_type_private_alloc_buffer,
                    /* .get_alignment    = */ wsp_ggml_backend_metal_buffer_type_private_get_alignment,
                    /* .get_max_size     = */ wsp_ggml_backend_metal_buffer_type_private_get_max_size,
                    /* .get_alloc_size   = */ wsp_ggml_backend_metal_buffer_type_private_get_alloc_size,
                    /* .is_host          = */ wsp_ggml_backend_metal_buffer_type_private_is_host,
                },
                /* .device  = */ wsp_ggml_backend_reg_dev_get(wsp_ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// mapped buffer type

static const char * wsp_ggml_backend_metal_buffer_type_mapped_get_name(wsp_ggml_backend_buffer_type_t buft) {
    wsp_ggml_backend_metal_buffer_type * ctx = (wsp_ggml_backend_metal_buffer_type *)buft->context;

    return ctx->name.c_str();
}

static wsp_ggml_backend_buffer_t wsp_ggml_backend_metal_buffer_type_mapped_alloc_buffer(wsp_ggml_backend_buffer_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    return wsp_ggml_backend_metal_buffer_type_alloc_buffer(buft, size, true);
}

static size_t wsp_ggml_backend_metal_buffer_type_mapped_get_alignment(wsp_ggml_backend_buffer_type_t buft) {
    return 32;

    WSP_GGML_UNUSED(buft);
}

static size_t wsp_ggml_backend_metal_buffer_type_mapped_get_max_size(wsp_ggml_backend_buffer_type_t buft) {
    wsp_ggml_metal_device_t ctx_dev = (wsp_ggml_metal_device_t)buft->device->context;

    return wsp_ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t wsp_ggml_backend_metal_buffer_type_mapped_get_alloc_size(wsp_ggml_backend_buffer_type_t buft, const wsp_ggml_tensor * tensor) {
    return wsp_ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool wsp_ggml_backend_metal_buffer_type_mapped_is_host(wsp_ggml_backend_buffer_type_t buft) {
    return false;

    WSP_GGML_UNUSED(buft);
}

static wsp_ggml_backend_buffer_type_t wsp_ggml_backend_metal_buffer_type_mapped(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<wsp_ggml_backend_buffer_type> bufts;
    static std::vector<wsp_ggml_backend_metal_buffer_type_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            wsp_ggml_backend_metal_buffer_type * raw_ctx = new wsp_ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ WSP_GGML_METAL_NAME + std::to_string(i) + "_Mapped"
            };
            ctxs.emplace_back(raw_ctx);

            // note: not obvious, but this buffer type still needs to implement .alloc_buffer:
            //       https://github.com/ggml-org/llama.cpp/pull/15832#discussion_r2333177099
            wsp_ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name         = */ wsp_ggml_backend_metal_buffer_type_mapped_get_name,
                    /* .alloc_buffer     = */ wsp_ggml_backend_metal_buffer_type_mapped_alloc_buffer,
                    /* .get_alignment    = */ wsp_ggml_backend_metal_buffer_type_mapped_get_alignment,
                    /* .get_max_size     = */ wsp_ggml_backend_metal_buffer_type_mapped_get_max_size,
                    /* .get_alloc_size   = */ wsp_ggml_backend_metal_buffer_type_mapped_get_alloc_size,
                    /* .is_host          = */ wsp_ggml_backend_metal_buffer_type_mapped_is_host,
                },
                /* .device  = */ wsp_ggml_backend_reg_dev_get(wsp_ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// backend

static const char * wsp_ggml_backend_metal_name(wsp_ggml_backend_t backend) {
    wsp_ggml_metal_t ctx = (wsp_ggml_metal_t)backend->context;

    return wsp_ggml_metal_get_name(ctx);
}

static void wsp_ggml_backend_metal_free(wsp_ggml_backend_t backend) {
    wsp_ggml_metal_t ctx = (wsp_ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    wsp_ggml_metal_synchronize(ctx);

    wsp_ggml_metal_free(ctx);

    free(backend);
}

static void wsp_ggml_backend_metal_synchronize(wsp_ggml_backend_t backend) {
    wsp_ggml_metal_t ctx = (wsp_ggml_metal_t)backend->context;

    wsp_ggml_metal_synchronize(ctx);
}

static void wsp_ggml_backend_metal_set_tensor_async(wsp_ggml_backend_t backend, wsp_ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    wsp_ggml_metal_t ctx = (wsp_ggml_metal_t)backend->context;

    wsp_ggml_metal_set_tensor_async(ctx, tensor, data, offset, size);
}

static void wsp_ggml_backend_metal_get_tensor_async(wsp_ggml_backend_t backend, const wsp_ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    wsp_ggml_metal_t ctx = (wsp_ggml_metal_t)backend->context;

    wsp_ggml_metal_get_tensor_async(ctx, tensor, data, offset, size);
}

static bool wsp_ggml_backend_metal_cpy_tensor_async(wsp_ggml_backend_t backend_src, wsp_ggml_backend_t backend_dst, const wsp_ggml_tensor * src, wsp_ggml_tensor * dst) {
    if (!wsp_ggml_backend_is_metal(backend_src) || !wsp_ggml_backend_is_metal(backend_dst)) {
        return false;
    }

    if (!wsp_ggml_backend_buffer_is_metal(src->buffer) || !wsp_ggml_backend_buffer_is_metal(dst->buffer)) {
        return false;
    }

    wsp_ggml_metal_t ctx_src = (wsp_ggml_metal_t)backend_src->context;
    wsp_ggml_metal_t ctx_dst = (wsp_ggml_metal_t)backend_dst->context;

    //wsp_ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //wsp_ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //wsp_ggml_metal_buffer_t buf_ctx_src = (wsp_ggml_metal_buffer_t)buf_src->context;
    //wsp_ggml_metal_buffer_t buf_ctx_dst = (wsp_ggml_metal_buffer_t)buf_dst->context;

    return wsp_ggml_metal_cpy_tensor_async(ctx_src, ctx_dst, src, dst);
}

static enum wsp_ggml_status wsp_ggml_backend_metal_graph_compute(wsp_ggml_backend_t backend, wsp_ggml_cgraph * cgraph) {
    wsp_ggml_metal_t ctx = (wsp_ggml_metal_t)backend->context;

    return wsp_ggml_metal_graph_compute(ctx, cgraph);
}

static void wsp_ggml_backend_metal_event_record(wsp_ggml_backend_t backend, wsp_ggml_backend_event_t event) {
    wsp_ggml_metal_t ctx = (wsp_ggml_metal_t)backend->context;
    wsp_ggml_metal_event_t ev = (wsp_ggml_metal_event_t)event->context;

    wsp_ggml_metal_event_record(ctx, ev);
}

static void wsp_ggml_backend_metal_event_wait(wsp_ggml_backend_t backend, wsp_ggml_backend_event_t event) {
    wsp_ggml_metal_t ctx = (wsp_ggml_metal_t)backend->context;
    wsp_ggml_metal_event_t ev = (wsp_ggml_metal_event_t)event->context;

    wsp_ggml_metal_event_wait(ctx, ev);
}

static void wsp_ggml_backend_metal_graph_optimize(wsp_ggml_backend_t backend, wsp_ggml_cgraph * cgraph) {
    wsp_ggml_metal_t ctx = (wsp_ggml_metal_t)backend->context;

    wsp_ggml_metal_graph_optimize(ctx, cgraph);
}

static void wsp_ggml_backend_metal_set_n_cb(wsp_ggml_backend_t backend, int n_cb) {
    WSP_GGML_ASSERT(wsp_ggml_backend_is_metal(backend));

    wsp_ggml_metal_t ctx = (wsp_ggml_metal_t)backend->context;

    wsp_ggml_metal_set_n_cb(ctx, n_cb);
}

static wsp_ggml_backend_i wsp_ggml_backend_metal_i = {
    /* .get_name                = */ wsp_ggml_backend_metal_name,
    /* .free                    = */ wsp_ggml_backend_metal_free,
    /* .set_tensor_async        = */ wsp_ggml_backend_metal_set_tensor_async,
    /* .get_tensor_async        = */ wsp_ggml_backend_metal_get_tensor_async,
    /* .cpy_tensor_async        = */ wsp_ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ wsp_ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ wsp_ggml_backend_metal_graph_compute,
    /* .event_record            = */ wsp_ggml_backend_metal_event_record,
    /* .event_wait              = */ wsp_ggml_backend_metal_event_wait,
    /* .graph_optimize          = */ wsp_ggml_backend_metal_graph_optimize,
};

static wsp_ggml_guid_t wsp_ggml_backend_metal_guid(void) {
    static wsp_ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    return &guid;
}

wsp_ggml_backend_t wsp_ggml_backend_metal_init(void) {
    wsp_ggml_backend_dev_t dev = wsp_ggml_backend_reg_dev_get(wsp_ggml_backend_metal_reg(), 0);
    wsp_ggml_metal_device_t ctx_dev = (wsp_ggml_metal_device_t)dev->context;

    wsp_ggml_metal_t ctx = wsp_ggml_metal_init(ctx_dev);
    if (ctx == NULL) {
        WSP_GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    wsp_ggml_backend_t backend = (wsp_ggml_backend_t) malloc(sizeof(wsp_ggml_backend));

    *backend = {
        /* .guid      = */ wsp_ggml_backend_metal_guid(),
        /* .interface = */ wsp_ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    wsp_ggml_backend_metal_set_n_cb(backend, 1);

    return backend;
}

bool wsp_ggml_backend_is_metal(wsp_ggml_backend_t backend) {
    return backend != NULL && wsp_ggml_guid_matches(backend->guid, wsp_ggml_backend_metal_guid());
}

void wsp_ggml_backend_metal_set_abort_callback(wsp_ggml_backend_t backend, wsp_ggml_abort_callback abort_callback, void * user_data) {
    WSP_GGML_ASSERT(wsp_ggml_backend_is_metal(backend));

    wsp_ggml_metal_t ctx = (wsp_ggml_metal_t)backend->context;

    wsp_ggml_metal_set_abort_callback(ctx, abort_callback, user_data);
}

bool wsp_ggml_backend_metal_supports_family(wsp_ggml_backend_t backend, int family) {
    WSP_GGML_ASSERT(wsp_ggml_backend_is_metal(backend));

    wsp_ggml_metal_t ctx = (wsp_ggml_metal_t)backend->context;

    return wsp_ggml_metal_supports_family(ctx, family);
}

void wsp_ggml_backend_metal_capture_next_compute(wsp_ggml_backend_t backend) {
    WSP_GGML_ASSERT(wsp_ggml_backend_is_metal(backend));

    wsp_ggml_metal_t ctx = (wsp_ggml_metal_t)backend->context;

    wsp_ggml_metal_capture_next_compute(ctx);
}

// backend device

static const char * wsp_ggml_backend_metal_device_get_name(wsp_ggml_backend_dev_t dev) {
    wsp_ggml_metal_device_t ctx_dev = (wsp_ggml_metal_device_t)dev->context;

    const wsp_ggml_metal_device_props * props_dev = wsp_ggml_metal_device_get_props(ctx_dev);

    return props_dev->name;
}

static const char * wsp_ggml_backend_metal_device_get_description(wsp_ggml_backend_dev_t dev) {
    wsp_ggml_metal_device_t ctx_dev = (wsp_ggml_metal_device_t)dev->context;

    return wsp_ggml_metal_device_get_props(ctx_dev)->desc;
}

static void wsp_ggml_backend_metal_device_get_memory(wsp_ggml_backend_dev_t dev, size_t * free, size_t * total) {
    wsp_ggml_metal_device_t ctx_dev = (wsp_ggml_metal_device_t)dev->context;

    wsp_ggml_metal_device_get_memory(ctx_dev, free, total);
}

static enum wsp_ggml_backend_dev_type wsp_ggml_backend_metal_device_get_type(wsp_ggml_backend_dev_t dev) {
    return WSP_GGML_BACKEND_DEVICE_TYPE_GPU;

    WSP_GGML_UNUSED(dev);
}

static void wsp_ggml_backend_metal_device_get_props(wsp_ggml_backend_dev_t dev, wsp_ggml_backend_dev_props * props) {
    props->name        = wsp_ggml_backend_metal_device_get_name(dev);
    props->description = wsp_ggml_backend_metal_device_get_description(dev);
    props->type        = wsp_ggml_backend_metal_device_get_type(dev);

    wsp_ggml_backend_metal_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                = */ true,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ true,
    };
}

static wsp_ggml_backend_t wsp_ggml_backend_metal_device_init_backend(wsp_ggml_backend_dev_t dev, const char * params) {
    wsp_ggml_metal_device_t ctx_dev = (wsp_ggml_metal_device_t)dev->context;

    wsp_ggml_metal_t ctx = wsp_ggml_metal_init(ctx_dev);
    if (ctx == NULL) {
        WSP_GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    wsp_ggml_backend_t backend = (wsp_ggml_backend_t) malloc(sizeof(wsp_ggml_backend));

    *backend = {
        /* .guid      = */ wsp_ggml_backend_metal_guid(),
        /* .interface = */ wsp_ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    wsp_ggml_backend_metal_set_n_cb(backend, 1);

    return backend;

    WSP_GGML_UNUSED(params);
}

static wsp_ggml_backend_buffer_type_t wsp_ggml_backend_metal_device_get_buffer_type(wsp_ggml_backend_dev_t dev) {
    wsp_ggml_metal_device_t ctx_dev = (wsp_ggml_metal_device_t)dev->context;

    const wsp_ggml_metal_device_props * props_dev = wsp_ggml_metal_device_get_props(ctx_dev);

    return props_dev->use_shared_buffers ? wsp_ggml_backend_metal_buffer_type_shared(props_dev->device) : wsp_ggml_backend_metal_buffer_type_private(props_dev->device);
}

static wsp_ggml_backend_buffer_t wsp_ggml_backend_metal_device_buffer_mapped(wsp_ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    wsp_ggml_metal_device_t ctx_dev = (wsp_ggml_metal_device_t)dev->context;

    wsp_ggml_metal_buffer_t res = wsp_ggml_metal_buffer_map(ctx_dev, ptr, size, max_tensor_size);

    const wsp_ggml_metal_device_props * props_dev = wsp_ggml_metal_device_get_props(ctx_dev);

    return wsp_ggml_backend_buffer_init(wsp_ggml_backend_metal_buffer_type_mapped(props_dev->device), wsp_ggml_backend_metal_buffer_shared_i, res, size);
}

static bool wsp_ggml_backend_metal_device_supports_op(wsp_ggml_backend_dev_t dev, const wsp_ggml_tensor * op) {
    wsp_ggml_metal_device_t ctx_dev = (wsp_ggml_metal_device_t)dev->context;

    return wsp_ggml_metal_device_supports_op(ctx_dev, op);
}

static bool wsp_ggml_backend_metal_device_supports_buft(wsp_ggml_backend_dev_t dev, wsp_ggml_backend_buffer_type_t buft) {
    return
        buft->device == dev && (
        buft->iface.get_name == wsp_ggml_backend_metal_buffer_type_shared_get_name ||
        buft->iface.get_name == wsp_ggml_backend_metal_buffer_type_private_get_name ||
        buft->iface.get_name == wsp_ggml_backend_metal_buffer_type_mapped_get_name);

    WSP_GGML_UNUSED(dev);
}

static int64_t get_op_batch_size(const wsp_ggml_tensor * op) {
    switch (op->op) {
        case WSP_GGML_OP_MUL_MAT:
            return op->ne[1];
        case WSP_GGML_OP_MUL_MAT_ID:
            return op->ne[2];
        default:
            return wsp_ggml_nrows(op);
    }
}

static bool wsp_ggml_backend_metal_device_offload_op(wsp_ggml_backend_dev_t dev, const wsp_ggml_tensor * op) {
    wsp_ggml_metal_device_t ctx_dev = (wsp_ggml_metal_device_t)dev->context;

    return (op->op == WSP_GGML_OP_MUL_MAT ||
            op->op == WSP_GGML_OP_MUL_MAT_ID) &&
            get_op_batch_size(op) >= wsp_ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_size;
}

static wsp_ggml_backend_event_t wsp_ggml_backend_metal_device_event_new(wsp_ggml_backend_dev_t dev) {
    wsp_ggml_metal_device_t ctx_dev = (wsp_ggml_metal_device_t)dev->context;

    wsp_ggml_metal_event_t event = wsp_ggml_metal_device_event_init(ctx_dev);
    WSP_GGML_ASSERT(event);

    wsp_ggml_backend_event_t ev = new wsp_ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    return ev;
}

static void wsp_ggml_backend_metal_device_event_free(wsp_ggml_backend_dev_t dev, wsp_ggml_backend_event_t event) {
    wsp_ggml_metal_device_t ctx_dev = (wsp_ggml_metal_device_t)dev->context;

    wsp_ggml_metal_event_t ev = (wsp_ggml_metal_event_t)event->context;

    wsp_ggml_metal_device_event_free(ctx_dev, ev);

    delete event;
}

static void wsp_ggml_backend_metal_device_event_synchronize(wsp_ggml_backend_dev_t dev, wsp_ggml_backend_event_t event) {
    wsp_ggml_metal_device_t ctx_dev = (wsp_ggml_metal_device_t)dev->context;

    wsp_ggml_metal_event_t evt = (wsp_ggml_metal_event_t)event->context;

    wsp_ggml_metal_device_event_synchronize(ctx_dev, evt);
}

static wsp_ggml_backend_device_i wsp_ggml_backend_metal_device_i = {
    /* .get_name             = */ wsp_ggml_backend_metal_device_get_name,
    /* .get_description      = */ wsp_ggml_backend_metal_device_get_description,
    /* .get_memory           = */ wsp_ggml_backend_metal_device_get_memory,
    /* .get_type             = */ wsp_ggml_backend_metal_device_get_type,
    /* .get_props            = */ wsp_ggml_backend_metal_device_get_props,
    /* .init_backend         = */ wsp_ggml_backend_metal_device_init_backend,
    /* .get_buffer_type      = */ wsp_ggml_backend_metal_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ wsp_ggml_backend_metal_device_buffer_mapped,
    /* .supports_op          = */ wsp_ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ wsp_ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ wsp_ggml_backend_metal_device_offload_op,
    /* .event_new            = */ wsp_ggml_backend_metal_device_event_new,
    /* .event_free           = */ wsp_ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ wsp_ggml_backend_metal_device_event_synchronize,
};

// backend registry

struct wsp_ggml_backend_metal_reg {
    std::vector<wsp_ggml_backend_dev_t> devices;
};

typedef struct wsp_ggml_backend_metal_reg * wsp_ggml_backend_metal_reg_t;

static wsp_ggml_backend_metal_reg_t wsp_ggml_backend_metal_reg_init(void) {
    wsp_ggml_backend_metal_reg_t ctx = new struct wsp_ggml_backend_metal_reg;

    return ctx;
}

static void wsp_ggml_backend_metal_reg_free(wsp_ggml_backend_metal_reg_t ctx) {
    delete ctx;
}

struct wsp_ggml_backend_metal_reg_deleter {
    void operator()(wsp_ggml_backend_metal_reg_t ctx) {
        wsp_ggml_backend_metal_reg_free(ctx);
    }
};

typedef std::unique_ptr<struct wsp_ggml_backend_metal_reg, wsp_ggml_backend_metal_reg_deleter> wsp_ggml_backend_metal_reg_ptr;

static const char * wsp_ggml_backend_metal_reg_get_name(wsp_ggml_backend_reg_t reg) {
    return WSP_GGML_METAL_NAME;

    WSP_GGML_UNUSED(reg);
}

static size_t wsp_ggml_backend_metal_reg_device_count(wsp_ggml_backend_reg_t reg) {
    wsp_ggml_backend_metal_reg_t ctx = (wsp_ggml_backend_metal_reg_t)reg->context;
    return ctx->devices.size();
}

static wsp_ggml_backend_dev_t wsp_ggml_backend_metal_reg_device_get(wsp_ggml_backend_reg_t reg, size_t index) {
    wsp_ggml_backend_metal_reg_t ctx = (wsp_ggml_backend_metal_reg_t)reg->context;
    WSP_GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static wsp_ggml_backend_feature g_wsp_ggml_backend_metal_features[] = {
#if defined(WSP_GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#endif
    { NULL, NULL },
};

static wsp_ggml_backend_feature * wsp_ggml_backend_metal_get_features(wsp_ggml_backend_reg_t reg) {
    return g_wsp_ggml_backend_metal_features;

    WSP_GGML_UNUSED(reg);
}

static void * wsp_ggml_backend_metal_get_proc_address(wsp_ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "wsp_ggml_backend_get_features") == 0) {
        return (void *)wsp_ggml_backend_metal_get_features;
    }

    return NULL;

    WSP_GGML_UNUSED(reg);
}

static wsp_ggml_backend_reg_i wsp_ggml_backend_metal_reg_i = {
    /* .get_name         = */ wsp_ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ wsp_ggml_backend_metal_reg_device_count,
    /* .get_device       = */ wsp_ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ wsp_ggml_backend_metal_get_proc_address,
};

static wsp_ggml_backend_dev_t wsp_ggml_backend_metal_device_init(wsp_ggml_backend_reg_t reg, int device) {
    return new wsp_ggml_backend_device {
        /* .iface   = */ wsp_ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ wsp_ggml_metal_device_get(device),
    };
}

static void wsp_ggml_backend_metal_device_free(wsp_ggml_backend_dev_t dev) {
    delete dev;
}

struct wsp_ggml_backend_device_deleter {
    void operator()(wsp_ggml_backend_dev_t ctx) {
        wsp_ggml_backend_metal_device_free(ctx);
    }
};

typedef std::unique_ptr<wsp_ggml_backend_device, wsp_ggml_backend_device_deleter> wsp_ggml_backend_device_ptr;

wsp_ggml_backend_reg_t wsp_ggml_backend_metal_reg(void) {
    static wsp_ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        const char * env = getenv("WSP_GGML_METAL_DEVICES");
        if (env) {
            g_devices = atoi(env);
        }

        static std::vector<wsp_ggml_backend_device_ptr> devs;

        if (!initialized) {
            static wsp_ggml_backend_metal_reg_ptr reg_ctx(wsp_ggml_backend_metal_reg_init());

            for (int i = 0; i < g_devices; ++i) {
                auto * dev = wsp_ggml_backend_metal_device_init(&reg, i);
                devs.emplace_back(dev);

                reg_ctx->devices.push_back(dev);
            }

            reg = {
                /* .api_version = */ WSP_GGML_BACKEND_API_VERSION,
                /* .iface       = */ wsp_ggml_backend_metal_reg_i,
                /* .context     = */ reg_ctx.get(),
            };
        }

        initialized = true;
    }

    return &reg;
}

WSP_GGML_BACKEND_DL_IMPL(wsp_ggml_backend_metal_reg)
