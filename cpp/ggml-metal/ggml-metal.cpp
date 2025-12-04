#include "ggml-metal.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-metal-device.h"
#include "ggml-metal-context.h"
#include "ggml-metal-ops.h"

// globals

// initialized in wsp_ggml_backend_metal_reg
static wsp_ggml_backend_reg    g_wsp_ggml_metal_reg;
static wsp_ggml_backend_device g_wsp_ggml_metal_device;

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

//
// buffer types
//

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
        default:
            break;
    }

    return res;

    WSP_GGML_UNUSED(buft);
}

// default (shared) buffer type

static const char * wsp_ggml_backend_metal_buffer_type_shared_get_name(wsp_ggml_backend_buffer_type_t buft) {
    return "Metal";

    WSP_GGML_UNUSED(buft);
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

static wsp_ggml_backend_buffer_type_t wsp_ggml_backend_metal_buffer_type_shared(void) {
    static wsp_ggml_backend_buffer_type wsp_ggml_backend_buffer_type_metal = {
        /* .iface = */ {
            /* .get_name         = */ wsp_ggml_backend_metal_buffer_type_shared_get_name,
            /* .alloc_buffer     = */ wsp_ggml_backend_metal_buffer_type_shared_alloc_buffer,
            /* .get_alignment    = */ wsp_ggml_backend_metal_buffer_type_shared_get_alignment,
            /* .get_max_size     = */ wsp_ggml_backend_metal_buffer_type_shared_get_max_size,
            /* .get_alloc_size   = */ wsp_ggml_backend_metal_buffer_type_shared_get_alloc_size,
            /* .is_host          = */ wsp_ggml_backend_metal_buffer_type_shared_is_host,
        },
        /* .device  = */ &g_wsp_ggml_metal_device,
        /* .context = */ NULL,
    };

    return &wsp_ggml_backend_buffer_type_metal;
}

// default (private) buffer type

static const char * wsp_ggml_backend_metal_buffer_type_private_get_name(wsp_ggml_backend_buffer_type_t buft) {
    return "Metal_Private";

    WSP_GGML_UNUSED(buft);
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

static wsp_ggml_backend_buffer_type_t wsp_ggml_backend_metal_buffer_type_private(void) {
    static wsp_ggml_backend_buffer_type wsp_ggml_backend_buffer_type_metal = {
        /* .iface = */ {
            /* .get_name         = */ wsp_ggml_backend_metal_buffer_type_private_get_name,
            /* .alloc_buffer     = */ wsp_ggml_backend_metal_buffer_type_private_alloc_buffer,
            /* .get_alignment    = */ wsp_ggml_backend_metal_buffer_type_private_get_alignment,
            /* .get_max_size     = */ wsp_ggml_backend_metal_buffer_type_private_get_max_size,
            /* .get_alloc_size   = */ wsp_ggml_backend_metal_buffer_type_private_get_alloc_size,
            /* .is_host          = */ wsp_ggml_backend_metal_buffer_type_private_is_host,
        },
        /* .device  = */ &g_wsp_ggml_metal_device,
        /* .context = */ NULL,
    };

    return &wsp_ggml_backend_buffer_type_metal;
}

// mapped buffer type

static const char * wsp_ggml_backend_metal_buffer_type_mapped_get_name(wsp_ggml_backend_buffer_type_t buft) {
    return "Metal_Mapped";

    WSP_GGML_UNUSED(buft);
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

static wsp_ggml_backend_buffer_type_t wsp_ggml_backend_metal_buffer_type_mapped(void) {
    // note: not obvious, but this buffer type still needs to implement .alloc_buffer:
    //       https://github.com/ggml-org/llama.cpp/pull/15832#discussion_r2333177099
    static wsp_ggml_backend_buffer_type wsp_ggml_backend_buffer_type_mapped_metal = {
        /* .iface = */ {
            /* .get_name         = */ wsp_ggml_backend_metal_buffer_type_mapped_get_name,
            /* .alloc_buffer     = */ wsp_ggml_backend_metal_buffer_type_mapped_alloc_buffer,
            /* .get_alignment    = */ wsp_ggml_backend_metal_buffer_type_mapped_get_alignment,
            /* .get_max_size     = */ wsp_ggml_backend_metal_buffer_type_mapped_get_max_size,
            /* .get_alloc_size   = */ wsp_ggml_backend_metal_buffer_type_mapped_get_alloc_size,
            /* .is_host          = */ wsp_ggml_backend_metal_buffer_type_mapped_is_host,
        },
        /* .device  = */ &g_wsp_ggml_metal_device,
        /* .context = */ NULL,
    };

    return &wsp_ggml_backend_buffer_type_mapped_metal;
}

// backend

static const char * wsp_ggml_backend_metal_name(wsp_ggml_backend_t backend) {
    return "Metal";

    WSP_GGML_UNUSED(backend);
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
    return false;

    WSP_GGML_UNUSED(backend_src);
    WSP_GGML_UNUSED(backend_dst);
    WSP_GGML_UNUSED(src);
    WSP_GGML_UNUSED(dst);
}

static enum wsp_ggml_status wsp_ggml_backend_metal_graph_compute(wsp_ggml_backend_t backend, wsp_ggml_cgraph * cgraph) {
    wsp_ggml_metal_t ctx = (wsp_ggml_metal_t)backend->context;

    return wsp_ggml_metal_graph_compute(ctx, cgraph);
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

    // the events API is needed only for multi-GPU setups, so likely no need to implement it for Metal
    // in any case, these docs seem relevant if we ever decide to implement it:
    // https://developer.apple.com/documentation/metal/mtlcommandbuffer#Synchronizing-Passes-with-Events
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
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
    return "Metal";

    WSP_GGML_UNUSED(dev);
}

static const char * wsp_ggml_backend_metal_device_get_description(wsp_ggml_backend_dev_t dev) {
    wsp_ggml_metal_device_t ctx_dev = (wsp_ggml_metal_device_t)dev->context;

    return wsp_ggml_metal_device_get_props(ctx_dev)->name;
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
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static wsp_ggml_backend_t wsp_ggml_backend_metal_device_init(wsp_ggml_backend_dev_t dev, const char * params) {
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

    return props_dev->use_shared_buffers ? wsp_ggml_backend_metal_buffer_type_shared() : wsp_ggml_backend_metal_buffer_type_private();
}

static wsp_ggml_backend_buffer_t wsp_ggml_backend_metal_device_buffer_mapped(wsp_ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    wsp_ggml_metal_device_t ctx_dev = (wsp_ggml_metal_device_t)dev->context;

    wsp_ggml_metal_buffer_t res = wsp_ggml_metal_buffer_map(ctx_dev, ptr, size, max_tensor_size);

    return wsp_ggml_backend_buffer_init(wsp_ggml_backend_metal_buffer_type_mapped(), wsp_ggml_backend_metal_buffer_shared_i, res, size);
}

static bool wsp_ggml_backend_metal_device_supports_op(wsp_ggml_backend_dev_t dev, const wsp_ggml_tensor * op) {
    wsp_ggml_metal_device_t ctx_dev = (wsp_ggml_metal_device_t)dev->context;

    return wsp_ggml_metal_device_supports_op(ctx_dev, op);
}

static bool wsp_ggml_backend_metal_device_supports_buft(wsp_ggml_backend_dev_t dev, wsp_ggml_backend_buffer_type_t buft) {
    return
        buft->iface.get_name == wsp_ggml_backend_metal_buffer_type_shared_get_name ||
        buft->iface.get_name == wsp_ggml_backend_metal_buffer_type_private_get_name ||
        buft->iface.get_name == wsp_ggml_backend_metal_buffer_type_mapped_get_name;

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
    const int min_batch_size = 32;

    return (op->op == WSP_GGML_OP_MUL_MAT ||
            op->op == WSP_GGML_OP_MUL_MAT_ID) &&
            get_op_batch_size(op) >= min_batch_size;

    WSP_GGML_UNUSED(dev);
    WSP_GGML_UNUSED(op);
}

static wsp_ggml_backend_device_i wsp_ggml_backend_metal_device_i = {
    /* .get_name             = */ wsp_ggml_backend_metal_device_get_name,
    /* .get_description      = */ wsp_ggml_backend_metal_device_get_description,
    /* .get_memory           = */ wsp_ggml_backend_metal_device_get_memory,
    /* .get_type             = */ wsp_ggml_backend_metal_device_get_type,
    /* .get_props            = */ wsp_ggml_backend_metal_device_get_props,
    /* .init_backend         = */ wsp_ggml_backend_metal_device_init,
    /* .get_buffer_type      = */ wsp_ggml_backend_metal_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ wsp_ggml_backend_metal_device_buffer_mapped,
    /* .supports_op          = */ wsp_ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ wsp_ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ wsp_ggml_backend_metal_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend registry

static const char * wsp_ggml_backend_metal_reg_get_name(wsp_ggml_backend_reg_t reg) {
    return "Metal";

    WSP_GGML_UNUSED(reg);
}

static size_t wsp_ggml_backend_metal_reg_device_count(wsp_ggml_backend_reg_t reg) {
    return 1;

    WSP_GGML_UNUSED(reg);
}

static wsp_ggml_backend_dev_t wsp_ggml_backend_metal_reg_device_get(wsp_ggml_backend_reg_t reg, size_t index) {
    WSP_GGML_ASSERT(index == 0);

    return &g_wsp_ggml_metal_device;

    WSP_GGML_UNUSED(reg);
    WSP_GGML_UNUSED(index);
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
    /* .device_count     = */ wsp_ggml_backend_metal_reg_device_count,
    /* .device_get       = */ wsp_ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ wsp_ggml_backend_metal_get_proc_address,
};

wsp_ggml_backend_reg_t wsp_ggml_backend_metal_reg(void) {
    {
        g_wsp_ggml_metal_reg = {
            /* .api_version = */ WSP_GGML_BACKEND_API_VERSION,
            /* .iface       = */ wsp_ggml_backend_metal_reg_i,
            /* .context     = */ NULL,
        };

        g_wsp_ggml_metal_device = {
            /* .iface   = */ wsp_ggml_backend_metal_device_i,
            /* .reg     = */ &g_wsp_ggml_metal_reg,
            /* .context = */ wsp_ggml_metal_device_get(),
        };
    }

    return &g_wsp_ggml_metal_reg;
}

WSP_GGML_BACKEND_DL_IMPL(wsp_ggml_backend_metal_reg)
