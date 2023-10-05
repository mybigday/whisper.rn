#import "ggml-metal.h"

#import "ggml.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// TODO: temporary - reuse llama.cpp logging
#ifdef WSP_GGML_METAL_NDEBUG
#define metal_printf(...)
#else
#define metal_printf(...) fprintf(stderr, __VA_ARGS__)
#endif

#define UNUSED(x) (void)(x)

#define WSP_GGML_MAX_CONCUR (2*WSP_GGML_MAX_NODES)

struct wsp_ggml_metal_buffer {
    const char * name;

    void   * data;
    size_t   size;

    id<MTLBuffer> metal;
};

struct wsp_ggml_metal_context {
    int n_cb;

    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;

    id<MTLCommandBuffer>         command_buffers [WSP_GGML_METAL_MAX_COMMAND_BUFFERS];
    id<MTLComputeCommandEncoder> command_encoders[WSP_GGML_METAL_MAX_COMMAND_BUFFERS];

    dispatch_queue_t d_queue;

    int n_buffers;
    struct wsp_ggml_metal_buffer buffers[WSP_GGML_METAL_MAX_BUFFERS];

    int concur_list[WSP_GGML_MAX_CONCUR];
    int concur_list_len;

    // custom kernels
#define WSP_GGML_METAL_DECL_KERNEL(name) \
    id<MTLFunction>             function_##name; \
    id<MTLComputePipelineState> pipeline_##name

    WSP_GGML_METAL_DECL_KERNEL(add);
    WSP_GGML_METAL_DECL_KERNEL(add_row); // TODO: avoid this extra kernel, instead extend the "add" kernel to support broadcast
    WSP_GGML_METAL_DECL_KERNEL(mul);
    WSP_GGML_METAL_DECL_KERNEL(mul_row); // TODO: avoid this extra kernel, instead extend the "mul" kernel to support broadcast
    WSP_GGML_METAL_DECL_KERNEL(scale);
    WSP_GGML_METAL_DECL_KERNEL(silu);
    WSP_GGML_METAL_DECL_KERNEL(relu);
    WSP_GGML_METAL_DECL_KERNEL(gelu);
    WSP_GGML_METAL_DECL_KERNEL(soft_max);
    WSP_GGML_METAL_DECL_KERNEL(soft_max_4);
    WSP_GGML_METAL_DECL_KERNEL(diag_mask_inf);
    WSP_GGML_METAL_DECL_KERNEL(diag_mask_inf_8);
    WSP_GGML_METAL_DECL_KERNEL(get_rows_f32);
    WSP_GGML_METAL_DECL_KERNEL(get_rows_f16);
    WSP_GGML_METAL_DECL_KERNEL(get_rows_q4_0);
    WSP_GGML_METAL_DECL_KERNEL(get_rows_q4_1);
    WSP_GGML_METAL_DECL_KERNEL(get_rows_q8_0);
    WSP_GGML_METAL_DECL_KERNEL(get_rows_q2_K);
    WSP_GGML_METAL_DECL_KERNEL(get_rows_q3_K);
    WSP_GGML_METAL_DECL_KERNEL(get_rows_q4_K);
    WSP_GGML_METAL_DECL_KERNEL(get_rows_q5_K);
    WSP_GGML_METAL_DECL_KERNEL(get_rows_q6_K);
    WSP_GGML_METAL_DECL_KERNEL(rms_norm);
    WSP_GGML_METAL_DECL_KERNEL(norm);
    WSP_GGML_METAL_DECL_KERNEL(mul_mat_f32_f32);
    WSP_GGML_METAL_DECL_KERNEL(mul_mat_f16_f32);
    WSP_GGML_METAL_DECL_KERNEL(mul_mat_f16_f32_1row);
    WSP_GGML_METAL_DECL_KERNEL(mul_mat_f16_f32_l4);
    WSP_GGML_METAL_DECL_KERNEL(mul_mat_q4_0_f32);
    WSP_GGML_METAL_DECL_KERNEL(mul_mat_q4_1_f32);
    WSP_GGML_METAL_DECL_KERNEL(mul_mat_q8_0_f32);
    WSP_GGML_METAL_DECL_KERNEL(mul_mat_q2_K_f32);
    WSP_GGML_METAL_DECL_KERNEL(mul_mat_q3_K_f32);
    WSP_GGML_METAL_DECL_KERNEL(mul_mat_q4_K_f32);
    WSP_GGML_METAL_DECL_KERNEL(mul_mat_q5_K_f32);
    WSP_GGML_METAL_DECL_KERNEL(mul_mat_q6_K_f32);
    WSP_GGML_METAL_DECL_KERNEL(mul_mm_f32_f32);
    WSP_GGML_METAL_DECL_KERNEL(mul_mm_f16_f32);
    WSP_GGML_METAL_DECL_KERNEL(mul_mm_q4_0_f32);
    WSP_GGML_METAL_DECL_KERNEL(mul_mm_q4_1_f32);
    WSP_GGML_METAL_DECL_KERNEL(mul_mm_q8_0_f32);
    WSP_GGML_METAL_DECL_KERNEL(mul_mm_q2_K_f32);
    WSP_GGML_METAL_DECL_KERNEL(mul_mm_q3_K_f32);
    WSP_GGML_METAL_DECL_KERNEL(mul_mm_q4_K_f32);
    WSP_GGML_METAL_DECL_KERNEL(mul_mm_q5_K_f32);
    WSP_GGML_METAL_DECL_KERNEL(mul_mm_q6_K_f32);
    WSP_GGML_METAL_DECL_KERNEL(rope);
    WSP_GGML_METAL_DECL_KERNEL(alibi_f32);
    WSP_GGML_METAL_DECL_KERNEL(cpy_f32_f16);
    WSP_GGML_METAL_DECL_KERNEL(cpy_f32_f32);
    WSP_GGML_METAL_DECL_KERNEL(cpy_f16_f16);

#undef WSP_GGML_METAL_DECL_KERNEL
};

// MSL code
// TODO: move the contents here when ready
//       for now it is easier to work in a separate file
static NSString * const msl_library_source = @"see metal.metal";

// Here to assist with NSBundle Path Hack
@interface GGMLMetalClass : NSObject
@end
@implementation GGMLMetalClass
@end

struct wsp_ggml_metal_context * wsp_ggml_metal_init(int n_cb) {
    metal_printf("%s: allocating\n", __func__);

    id <MTLDevice> device;
    NSString * s;

#if TARGET_OS_OSX
    // Show all the Metal device instances in the system
    NSArray * devices = MTLCopyAllDevices();
    for (device in devices) {
        s = [device name];
        metal_printf("%s: found device: %s\n", __func__, [s UTF8String]);
    }
#endif

    // Pick and show default Metal device
    device = MTLCreateSystemDefaultDevice();
    s = [device name];
    metal_printf("%s: picking default device: %s\n", __func__, [s UTF8String]);

    // Configure context
    struct wsp_ggml_metal_context * ctx = malloc(sizeof(struct wsp_ggml_metal_context));
    ctx->device = device;
    ctx->n_cb   = MIN(n_cb, WSP_GGML_METAL_MAX_BUFFERS);
    ctx->queue  = [ctx->device newCommandQueue];
    ctx->n_buffers = 0;
    ctx->concur_list_len = 0;

    ctx->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

#ifdef WSP_GGML_SWIFT
    // load the default.metallib file
    {
        NSError * error = nil;

        NSBundle * bundle = [NSBundle bundleForClass:[GGMLMetalClass class]];
        NSString * llamaBundlePath = [bundle pathForResource:@"llama_llama" ofType:@"bundle"];
        NSBundle * llamaBundle = [NSBundle bundleWithPath:llamaBundlePath];
        NSString * libPath = [llamaBundle pathForResource:@"default" ofType:@"metallib"];
        NSURL * libURL = [NSURL fileURLWithPath:libPath];

        // Load the metallib file into a Metal library
        ctx->library = [ctx->device newLibraryWithURL:libURL error:&error];

        if (error) {
            metal_printf("%s: error: %s\n", __func__, [[error description] UTF8String]);
            return NULL;
        }
    }
#else
    UNUSED(msl_library_source);

    // read the source from "ggml-metal.metal" into a string and use newLibraryWithSource
    {
        NSError * error = nil;

        //NSString * path = [[NSBundle mainBundle] pathForResource:@"../../examples/metal/metal" ofType:@"metal"];
        NSBundle * bundle = [NSBundle bundleForClass:[GGMLMetalClass class]];
        NSString * path   = [bundle pathForResource:@"ggml-metal" ofType:@"metal"];
        metal_printf("%s: loading '%s'\n", __func__, [path UTF8String]);

        NSString * src  = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
        if (error) {
            metal_printf("%s: error: %s\n", __func__, [[error description] UTF8String]);
            return NULL;
        }

#ifdef WSP_GGML_QKK_64
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.preprocessorMacros = @{ @"QK_K" : @(64) };
        ctx->library = [ctx->device newLibraryWithSource:src options:options error:&error];
#else
        ctx->library = [ctx->device newLibraryWithSource:src options:nil error:&error];
#endif
        if (error) {
            metal_printf("%s: error: %s\n", __func__, [[error description] UTF8String]);
            return NULL;
        }
    }
#endif

    // load kernels
    {
        NSError * error = nil;
#define WSP_GGML_METAL_ADD_KERNEL(name) \
        ctx->function_##name = [ctx->library newFunctionWithName:@"kernel_"#name]; \
        ctx->pipeline_##name = [ctx->device newComputePipelineStateWithFunction:ctx->function_##name error:&error]; \
        metal_printf("%s: loaded %-32s %16p | th_max = %4d | th_width = %4d\n", __func__, "kernel_"#name, (__bridge void *) ctx->pipeline_##name, \
                (int) ctx->pipeline_##name.maxTotalThreadsPerThreadgroup, \
                (int) ctx->pipeline_##name.threadExecutionWidth); \
        if (error) { \
            metal_printf("%s: load pipeline error: %s\n", __func__, [[error description] UTF8String]); \
            return NULL; \
        }

        WSP_GGML_METAL_ADD_KERNEL(add);
        WSP_GGML_METAL_ADD_KERNEL(add_row);
        WSP_GGML_METAL_ADD_KERNEL(mul);
        WSP_GGML_METAL_ADD_KERNEL(mul_row);
        WSP_GGML_METAL_ADD_KERNEL(scale);
        WSP_GGML_METAL_ADD_KERNEL(silu);
        WSP_GGML_METAL_ADD_KERNEL(relu);
        WSP_GGML_METAL_ADD_KERNEL(gelu);
        WSP_GGML_METAL_ADD_KERNEL(soft_max);
        WSP_GGML_METAL_ADD_KERNEL(soft_max_4);
        WSP_GGML_METAL_ADD_KERNEL(diag_mask_inf);
        WSP_GGML_METAL_ADD_KERNEL(diag_mask_inf_8);
        WSP_GGML_METAL_ADD_KERNEL(get_rows_f32);
        WSP_GGML_METAL_ADD_KERNEL(get_rows_f16);
        WSP_GGML_METAL_ADD_KERNEL(get_rows_q4_0);
        WSP_GGML_METAL_ADD_KERNEL(get_rows_q4_1);
        WSP_GGML_METAL_ADD_KERNEL(get_rows_q8_0);
        WSP_GGML_METAL_ADD_KERNEL(get_rows_q2_K);
        WSP_GGML_METAL_ADD_KERNEL(get_rows_q3_K);
        WSP_GGML_METAL_ADD_KERNEL(get_rows_q4_K);
        WSP_GGML_METAL_ADD_KERNEL(get_rows_q5_K);
        WSP_GGML_METAL_ADD_KERNEL(get_rows_q6_K);
        WSP_GGML_METAL_ADD_KERNEL(rms_norm);
        WSP_GGML_METAL_ADD_KERNEL(norm);
        WSP_GGML_METAL_ADD_KERNEL(mul_mat_f32_f32);
        WSP_GGML_METAL_ADD_KERNEL(mul_mat_f16_f32);
        WSP_GGML_METAL_ADD_KERNEL(mul_mat_f16_f32_1row);
        WSP_GGML_METAL_ADD_KERNEL(mul_mat_f16_f32_l4);
        WSP_GGML_METAL_ADD_KERNEL(mul_mat_q4_0_f32);
        WSP_GGML_METAL_ADD_KERNEL(mul_mat_q4_1_f32);
        WSP_GGML_METAL_ADD_KERNEL(mul_mat_q8_0_f32);
        WSP_GGML_METAL_ADD_KERNEL(mul_mat_q2_K_f32);
        WSP_GGML_METAL_ADD_KERNEL(mul_mat_q3_K_f32);
        WSP_GGML_METAL_ADD_KERNEL(mul_mat_q4_K_f32);
        WSP_GGML_METAL_ADD_KERNEL(mul_mat_q5_K_f32);
        WSP_GGML_METAL_ADD_KERNEL(mul_mat_q6_K_f32);
        WSP_GGML_METAL_ADD_KERNEL(mul_mm_f32_f32);
        WSP_GGML_METAL_ADD_KERNEL(mul_mm_f16_f32);
        WSP_GGML_METAL_ADD_KERNEL(mul_mm_q4_0_f32);
        WSP_GGML_METAL_ADD_KERNEL(mul_mm_q8_0_f32);
        WSP_GGML_METAL_ADD_KERNEL(mul_mm_q4_1_f32);
        WSP_GGML_METAL_ADD_KERNEL(mul_mm_q2_K_f32);
        WSP_GGML_METAL_ADD_KERNEL(mul_mm_q3_K_f32);
        WSP_GGML_METAL_ADD_KERNEL(mul_mm_q4_K_f32);
        WSP_GGML_METAL_ADD_KERNEL(mul_mm_q5_K_f32);
        WSP_GGML_METAL_ADD_KERNEL(mul_mm_q6_K_f32);
        WSP_GGML_METAL_ADD_KERNEL(rope);
        WSP_GGML_METAL_ADD_KERNEL(alibi_f32);
        WSP_GGML_METAL_ADD_KERNEL(cpy_f32_f16);
        WSP_GGML_METAL_ADD_KERNEL(cpy_f32_f32);
        WSP_GGML_METAL_ADD_KERNEL(cpy_f16_f16);

#undef WSP_GGML_METAL_ADD_KERNEL
    }

    metal_printf("%s: hasUnifiedMemory              = %s\n",       __func__, ctx->device.hasUnifiedMemory ? "true" : "false");
#if TARGET_OS_OSX
    metal_printf("%s: recommendedMaxWorkingSetSize  = %8.2f MB\n", __func__, ctx->device.recommendedMaxWorkingSetSize / 1024.0 / 1024.0);
    if (ctx->device.maxTransferRate != 0) {
        metal_printf("%s: maxTransferRate               = %8.2f MB/s\n", __func__, ctx->device.maxTransferRate / 1024.0 / 1024.0);
    } else {
        metal_printf("%s: maxTransferRate               = built-in GPU\n", __func__);
    }
#endif

    return ctx;
}

void wsp_ggml_metal_free(struct wsp_ggml_metal_context * ctx) {
    metal_printf("%s: deallocating\n", __func__);
#define WSP_GGML_METAL_DEL_KERNEL(name) \

    WSP_GGML_METAL_DEL_KERNEL(add);
    WSP_GGML_METAL_DEL_KERNEL(add_row);
    WSP_GGML_METAL_DEL_KERNEL(mul);
    WSP_GGML_METAL_DEL_KERNEL(mul_row);
    WSP_GGML_METAL_DEL_KERNEL(scale);
    WSP_GGML_METAL_DEL_KERNEL(silu);
    WSP_GGML_METAL_DEL_KERNEL(relu);
    WSP_GGML_METAL_DEL_KERNEL(gelu);
    WSP_GGML_METAL_DEL_KERNEL(soft_max);
    WSP_GGML_METAL_DEL_KERNEL(soft_max_4);
    WSP_GGML_METAL_DEL_KERNEL(diag_mask_inf);
    WSP_GGML_METAL_DEL_KERNEL(diag_mask_inf_8);
    WSP_GGML_METAL_DEL_KERNEL(get_rows_f32);
    WSP_GGML_METAL_DEL_KERNEL(get_rows_f16);
    WSP_GGML_METAL_DEL_KERNEL(get_rows_q4_0);
    WSP_GGML_METAL_DEL_KERNEL(get_rows_q4_1);
    WSP_GGML_METAL_DEL_KERNEL(get_rows_q8_0);
    WSP_GGML_METAL_DEL_KERNEL(get_rows_q2_K);
    WSP_GGML_METAL_DEL_KERNEL(get_rows_q3_K);
    WSP_GGML_METAL_DEL_KERNEL(get_rows_q4_K);
    WSP_GGML_METAL_DEL_KERNEL(get_rows_q5_K);
    WSP_GGML_METAL_DEL_KERNEL(get_rows_q6_K);
    WSP_GGML_METAL_DEL_KERNEL(rms_norm);
    WSP_GGML_METAL_DEL_KERNEL(norm);
    WSP_GGML_METAL_DEL_KERNEL(mul_mat_f32_f32);
    WSP_GGML_METAL_DEL_KERNEL(mul_mat_f16_f32);
    WSP_GGML_METAL_DEL_KERNEL(mul_mat_f16_f32_1row);
    WSP_GGML_METAL_DEL_KERNEL(mul_mat_f16_f32_l4);
    WSP_GGML_METAL_DEL_KERNEL(mul_mat_q4_0_f32);
    WSP_GGML_METAL_DEL_KERNEL(mul_mat_q4_1_f32);
    WSP_GGML_METAL_DEL_KERNEL(mul_mat_q8_0_f32);
    WSP_GGML_METAL_DEL_KERNEL(mul_mat_q2_K_f32);
    WSP_GGML_METAL_DEL_KERNEL(mul_mat_q3_K_f32);
    WSP_GGML_METAL_DEL_KERNEL(mul_mat_q4_K_f32);
    WSP_GGML_METAL_DEL_KERNEL(mul_mat_q5_K_f32);
    WSP_GGML_METAL_DEL_KERNEL(mul_mat_q6_K_f32);
    WSP_GGML_METAL_DEL_KERNEL(mul_mm_f32_f32);
    WSP_GGML_METAL_DEL_KERNEL(mul_mm_f16_f32);
    WSP_GGML_METAL_DEL_KERNEL(mul_mm_q4_0_f32);
    WSP_GGML_METAL_DEL_KERNEL(mul_mm_q8_0_f32);
    WSP_GGML_METAL_DEL_KERNEL(mul_mm_q4_1_f32);
    WSP_GGML_METAL_DEL_KERNEL(mul_mm_q2_K_f32);
    WSP_GGML_METAL_DEL_KERNEL(mul_mm_q3_K_f32);
    WSP_GGML_METAL_DEL_KERNEL(mul_mm_q4_K_f32);
    WSP_GGML_METAL_DEL_KERNEL(mul_mm_q5_K_f32);
    WSP_GGML_METAL_DEL_KERNEL(mul_mm_q6_K_f32);
    WSP_GGML_METAL_DEL_KERNEL(rope);
    WSP_GGML_METAL_DEL_KERNEL(alibi_f32);
    WSP_GGML_METAL_DEL_KERNEL(cpy_f32_f16);
    WSP_GGML_METAL_DEL_KERNEL(cpy_f32_f32);
    WSP_GGML_METAL_DEL_KERNEL(cpy_f16_f16);

#undef WSP_GGML_METAL_DEL_KERNEL

    free(ctx);
}

void * wsp_ggml_metal_host_malloc(size_t n) {
    void * data = NULL;
    const int result = posix_memalign((void **) &data, sysconf(_SC_PAGESIZE), n);
    if (result != 0) {
        metal_printf("%s: error: posix_memalign failed\n", __func__);
        return NULL;
    }

    return data;
}

void wsp_ggml_metal_host_free(void * data) {
    free(data);
}

void wsp_ggml_metal_set_n_cb(struct wsp_ggml_metal_context * ctx, int n_cb) {
    ctx->n_cb = MIN(n_cb, WSP_GGML_METAL_MAX_BUFFERS);
}

int wsp_ggml_metal_if_optimized(struct wsp_ggml_metal_context * ctx) {
    return ctx->concur_list_len;
}

int * wsp_ggml_metal_get_concur_list(struct wsp_ggml_metal_context * ctx) {
    return ctx->concur_list;
}

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
static id<MTLBuffer> wsp_ggml_metal_get_buffer(struct wsp_ggml_metal_context * ctx, struct wsp_ggml_tensor * t, size_t * offs) {
    //metal_printf("%s: data tensor '%16s', offs_data = %8ld, offs_eval = %8ld, offs_cach = %8ld\n", __func__, t->name, offs_data, offs_eval, offs_cach);

    const int64_t tsize = wsp_ggml_nbytes(t);

    // find the view that contains the tensor fully
    for (int i = 0; i < ctx->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) ctx->buffers[i].data;

        //metal_printf("ioffs = %10ld, tsize = %10ld, sum = %10ld, ctx->buffers[%d].size = %10ld, name = %s\n", ioffs, tsize, ioffs + tsize, i, ctx->buffers[i].size, ctx->buffers[i].name);
        if (ioffs >= 0 && ioffs + tsize <= (int64_t) ctx->buffers[i].size) {
            *offs = (size_t) ioffs;

            //metal_printf("%s: '%s' tensor '%16s', offs = %8ld\n", __func__, ctx->buffers[i].name, t->name, *offs);

            return ctx->buffers[i].metal;
        }
    }

    metal_printf("%s: error: buffer is nil\n", __func__);

    return nil;
}

bool wsp_ggml_metal_add_buffer(
        struct wsp_ggml_metal_context * ctx,
                     const char * name,
                           void * data,
                         size_t   size,
                         size_t   max_size) {
    if (ctx->n_buffers >= WSP_GGML_METAL_MAX_BUFFERS) {
        metal_printf("%s: too many buffers\n", __func__);
        return false;
    }

    if (data) {
        // verify that the buffer does not overlap with any of the existing buffers
        for (int i = 0; i < ctx->n_buffers; ++i) {
            const int64_t ioffs = (int64_t) data - (int64_t) ctx->buffers[i].data;

            if (ioffs >= 0 && ioffs < (int64_t) ctx->buffers[i].size) {
                metal_printf("%s: error: buffer '%s' overlaps with '%s'\n", __func__, name, ctx->buffers[i].name);
                return false;
            }
        }

        const size_t size_page = sysconf(_SC_PAGESIZE);

        size_t size_aligned = size;
        if ((size_aligned % size_page) != 0) {
            size_aligned += (size_page - (size_aligned % size_page));
        }

        // the buffer fits into the max buffer size allowed by the device
        if (size_aligned <= ctx->device.maxBufferLength) {
            ctx->buffers[ctx->n_buffers].name = name;
            ctx->buffers[ctx->n_buffers].data = data;
            ctx->buffers[ctx->n_buffers].size = size;

            ctx->buffers[ctx->n_buffers].metal = [ctx->device newBufferWithBytesNoCopy:data length:size_aligned options:MTLResourceStorageModeShared deallocator:nil];

            if (ctx->buffers[ctx->n_buffers].metal == nil) {
                metal_printf("%s: failed to allocate '%-16s' buffer, size = %8.2f MB\n", __func__, name, size_aligned / 1024.0 / 1024.0);
                return false;
            }

            metal_printf("%s: allocated '%-16s' buffer, size = %8.2f MB", __func__, name, size_aligned / 1024.0 / 1024.0);

            ++ctx->n_buffers;
        } else {
            // this overlap between the views will guarantee that the tensor with the maximum size will fully fit into
            // one of the views
            const size_t size_ovlp = ((max_size + size_page - 1) / size_page + 1) * size_page; // round-up 2 pages just in case
            const size_t size_step = ctx->device.maxBufferLength - size_ovlp;
            const size_t size_view = ctx->device.maxBufferLength;

            for (size_t i = 0; i < size; i += size_step) {
                const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

                ctx->buffers[ctx->n_buffers].name = name;
                ctx->buffers[ctx->n_buffers].data = (void *) ((uint8_t *) data + i);
                ctx->buffers[ctx->n_buffers].size = size_step_aligned;

                ctx->buffers[ctx->n_buffers].metal = [ctx->device newBufferWithBytesNoCopy:(void *) ((uint8_t *) data + i) length:size_step_aligned options:MTLResourceStorageModeShared deallocator:nil];

                if (ctx->buffers[ctx->n_buffers].metal == nil) {
                    metal_printf("%s: failed to allocate '%-16s' buffer, size = %8.2f MB\n", __func__, name, size_step_aligned / 1024.0 / 1024.0);
                    return false;
                }

                metal_printf("%s: allocated '%-16s' buffer, size = %8.2f MB, offs = %12ld", __func__, name, size_step_aligned / 1024.0 / 1024.0, i);
                if (i + size_step < size) {
                    metal_printf("\n");
                }

                ++ctx->n_buffers;
            }
        }

#if TARGET_OS_OSX
        metal_printf(", (%8.2f / %8.2f)",
                ctx->device.currentAllocatedSize / 1024.0 / 1024.0,
                ctx->device.recommendedMaxWorkingSetSize / 1024.0 / 1024.0);

        if (ctx->device.currentAllocatedSize > ctx->device.recommendedMaxWorkingSetSize) {
            metal_printf(", warning: current allocated size is greater than the recommended max working set size\n");
        } else {
            metal_printf("\n");
        }
#else
        metal_printf(", (%8.2f)\n", ctx->device.currentAllocatedSize / 1024.0 / 1024.0);
#endif
    }

    return true;
}

void wsp_ggml_metal_set_tensor(
        struct wsp_ggml_metal_context * ctx,
        struct wsp_ggml_tensor * t) {
    size_t offs;
    id<MTLBuffer> id_dst = wsp_ggml_metal_get_buffer(ctx, t, &offs);

    memcpy((void *) ((uint8_t *) id_dst.contents + offs), t->data, wsp_ggml_nbytes(t));
}

void wsp_ggml_metal_get_tensor(
        struct wsp_ggml_metal_context * ctx,
        struct wsp_ggml_tensor * t) {
    size_t offs;
    id<MTLBuffer> id_src = wsp_ggml_metal_get_buffer(ctx, t, &offs);

    memcpy(t->data, (void *) ((uint8_t *) id_src.contents + offs), wsp_ggml_nbytes(t));
}

void wsp_ggml_metal_graph_find_concurrency(
        struct wsp_ggml_metal_context * ctx,
        struct wsp_ggml_cgraph * gf, bool check_mem) {
    int search_depth = gf->n_nodes; //we only find concurrency in this range to avoid wasting too much time
    int nodes_unused[WSP_GGML_MAX_CONCUR];

    for (int i = 0; i < WSP_GGML_MAX_CONCUR; i++) { ctx->concur_list[i] = 0; }
    for (int i = 0; i < gf->n_nodes;     i++) { nodes_unused[i]     = 1; }
    ctx->concur_list_len = 0;

    int n_left    = gf->n_nodes;
    int n_start   = 0; // all nodes before n_start at nodes_unused array have been sorted and store back to ctx->concur_list
    int level_pos = 0; // at ctx->concur_list, the last layer (level) ends at level_pos

    while (n_left > 0) {
        // number of nodes at a layer (that can be issued concurrently)
        int concurrency = 0;
        for (int i = n_start; i < ((n_start + search_depth > gf->n_nodes) ? gf->n_nodes : n_start + search_depth); i++) {
            if (nodes_unused[i]) {
                // if the requirements for gf->nodes[i] are satisfied
                int exe_flag = 1;

                // scan all srcs
                for (int src_ind = 0; src_ind < WSP_GGML_MAX_SRC; src_ind++) {
                    struct wsp_ggml_tensor * src_cur = gf->nodes[i]->src[src_ind];
                    if (src_cur) {
                        // if is leaf nodes it's satisfied.
                        // TODO: wsp_ggml_is_leaf()
                        if (src_cur->op == WSP_GGML_OP_NONE && src_cur->grad == NULL) {
                            continue;
                        }

                        // otherwise this src should be the output from previous nodes.
                        int is_found = 0;

                        // scan 2*search_depth back because we inserted barrier.
                        //for (int j = ((level_pos - 2*search_depth) < 0 ? 0 : (level_pos - 2*search_depth)); j < level_pos; j++) {
                        for (int j = MAX(0, level_pos - 2*search_depth); j < level_pos; j++) {
                            if (ctx->concur_list[j] >= 0 && gf->nodes[ctx->concur_list[j]] == src_cur) {
                                is_found = 1;
                                break;
                            }
                        }
                        if (is_found == 0) {
                            exe_flag = 0;
                            break;
                        }
                    }
                }
                if (exe_flag && check_mem) {
                    // check if nodes[i]'s data will be overwritten by a node before nodes[i].
                    // if node[5] and node[3] write to the same memory region, then we can't issue node[5] before node[3]
                    int64_t data_start = (int64_t) gf->nodes[i]->data;
                    int64_t length     = (int64_t) wsp_ggml_nbytes(gf->nodes[i]);
                    for (int j = n_start; j < i; j++) {
                        if (nodes_unused[j] && gf->nodes[j]->op != WSP_GGML_OP_RESHAPE \
                                            && gf->nodes[j]->op != WSP_GGML_OP_VIEW \
                                            && gf->nodes[j]->op != WSP_GGML_OP_TRANSPOSE \
                                            && gf->nodes[j]->op != WSP_GGML_OP_PERMUTE) {
                            if (((int64_t)gf->nodes[j]->data) >= data_start + length || \
                                ((int64_t)gf->nodes[j]->data) + (int64_t) wsp_ggml_nbytes(gf->nodes[j]) <= data_start) {
                                continue;
                            }

                            exe_flag = 0;
                        }
                    }
                }
                if (exe_flag) {
                    ctx->concur_list[level_pos + concurrency] = i;
                    nodes_unused[i] = 0;
                    concurrency++;
                    ctx->concur_list_len++;
                }
            }
        }
        n_left -= concurrency;
        // adding a barrier different layer
        ctx->concur_list[level_pos + concurrency] = -1;
        ctx->concur_list_len++;
        // jump all sorted nodes at nodes_bak
        while (!nodes_unused[n_start]) {
            n_start++;
        }
        level_pos += concurrency + 1;
    }

    if (ctx->concur_list_len > WSP_GGML_MAX_CONCUR) {
        metal_printf("%s: too many elements for metal ctx->concur_list!\n", __func__);
    }
}

void wsp_ggml_metal_graph_compute(
        struct wsp_ggml_metal_context * ctx,
               struct wsp_ggml_cgraph * gf) {
    @autoreleasepool {

    // if there is ctx->concur_list, dispatch concurrently
    // else fallback to serial dispatch
    MTLComputePassDescriptor * edesc = MTLComputePassDescriptor.computePassDescriptor;

    const bool has_concur = ctx->concur_list_len && ctx->concur_list_len <= WSP_GGML_MAX_CONCUR;

    const int n_nodes  = has_concur ? ctx->concur_list_len      : gf->n_nodes;
    edesc.dispatchType = has_concur ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial;

    // create multiple command buffers and enqueue them
    // then, we encode the graph into the command buffers in parallel

    const int n_cb = ctx->n_cb;

    for (int i = 0; i < n_cb; ++i) {
        ctx->command_buffers[i] = [ctx->queue commandBuffer];

        // enqueue the command buffers in order to specify their execution order
        [ctx->command_buffers[i] enqueue];

        ctx->command_encoders[i] = [ctx->command_buffers[i] computeCommandEncoderWithDescriptor: edesc];
    }

    for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
        const int n_nodes_per_cb = (n_nodes + n_cb - 1) / n_cb;

        dispatch_async(ctx->d_queue, ^{
            size_t offs_src0 = 0;
            size_t offs_src1 = 0;
            size_t offs_dst  = 0;

            id<MTLCommandBuffer> command_buffer  = ctx->command_buffers[cb_idx];
            id<MTLComputeCommandEncoder> encoder = ctx->command_encoders[cb_idx];

            const int node_start =                                      (cb_idx + 0) * n_nodes_per_cb;
            const int node_end   = MIN((cb_idx == n_cb - 1) ? n_nodes : (cb_idx + 1) * n_nodes_per_cb, n_nodes);

            for (int ind = node_start; ind < node_end; ++ind) {
                const int i = has_concur ? ctx->concur_list[ind] : ind;

                if (i == -1) {
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    continue;
                }

                //metal_printf("%s: encoding node %3d, op = %8s\n", __func__, i, wsp_ggml_op_name(gf->nodes[i]->op));

                struct wsp_ggml_tensor * src0 = gf->nodes[i]->src[0];
                struct wsp_ggml_tensor * src1 = gf->nodes[i]->src[1];
                struct wsp_ggml_tensor * dst  = gf->nodes[i];

                const int64_t  ne00 = src0 ? src0->ne[0] : 0;
                const int64_t  ne01 = src0 ? src0->ne[1] : 0;
                const int64_t  ne02 = src0 ? src0->ne[2] : 0;
                const int64_t  ne03 = src0 ? src0->ne[3] : 0;

                const uint64_t nb00 = src0 ? src0->nb[0] : 0;
                const uint64_t nb01 = src0 ? src0->nb[1] : 0;
                const uint64_t nb02 = src0 ? src0->nb[2] : 0;
                const uint64_t nb03 = src0 ? src0->nb[3] : 0;

                const int64_t  ne10 = src1 ? src1->ne[0] : 0;
                const int64_t  ne11 = src1 ? src1->ne[1] : 0;
                const int64_t  ne12 = src1 ? src1->ne[2] : 0;
                const int64_t  ne13 = src1 ? src1->ne[3] : 0; UNUSED(ne13);

                const uint64_t nb10 = src1 ? src1->nb[0] : 0;
                const uint64_t nb11 = src1 ? src1->nb[1] : 0;
                const uint64_t nb12 = src1 ? src1->nb[2] : 0;
                const uint64_t nb13 = src1 ? src1->nb[3] : 0; UNUSED(nb13);

                const int64_t  ne0  = dst ? dst->ne[0] : 0;
                const int64_t  ne1  = dst ? dst->ne[1] : 0;
                const int64_t  ne2  = dst ? dst->ne[2] : 0;
                const int64_t  ne3  = dst ? dst->ne[3] : 0;

                const uint64_t nb0  = dst ? dst->nb[0] : 0;
                const uint64_t nb1  = dst ? dst->nb[1] : 0;
                const uint64_t nb2  = dst ? dst->nb[2] : 0;
                const uint64_t nb3  = dst ? dst->nb[3] : 0;

                const enum wsp_ggml_type src0t = src0 ? src0->type : WSP_GGML_TYPE_COUNT;
                const enum wsp_ggml_type src1t = src1 ? src1->type : WSP_GGML_TYPE_COUNT;
                const enum wsp_ggml_type dstt  = dst  ? dst->type  : WSP_GGML_TYPE_COUNT;

                id<MTLBuffer> id_src0 = src0 ? wsp_ggml_metal_get_buffer(ctx, src0, &offs_src0) : nil;
                id<MTLBuffer> id_src1 = src1 ? wsp_ggml_metal_get_buffer(ctx, src1, &offs_src1) : nil;
                id<MTLBuffer> id_dst  = dst  ? wsp_ggml_metal_get_buffer(ctx, dst,  &offs_dst)  : nil;

                //metal_printf("%s: op - %s\n", __func__, wsp_ggml_op_name(dst->op));
                //if (src0) {
                //    metal_printf("%s: src0 - %4s [%5lld, %5lld, %5lld], %d, %s\n", __func__, wsp_ggml_type_name(src0t), ne00, ne01, ne02,
                //            wsp_ggml_is_contiguous(src0), src0->name);
                //}
                //if (src1) {
                //    metal_printf("%s: src1 - %4s [%5lld, %5lld, %5lld], %d, %s\n", __func__, wsp_ggml_type_name(src1t), ne10, ne11, ne12,
                //            wsp_ggml_is_contiguous(src1), src1->name);
                //}
                //if (dst) {
                //    metal_printf("%s: dst  - %4s [%5lld, %5lld, %5lld], 1, %s\n",  __func__, wsp_ggml_type_name(dstt),  ne0,  ne1,  ne2,
                //            dst->name);
                //}

                switch (dst->op) {
                    case WSP_GGML_OP_NONE:
                    case WSP_GGML_OP_RESHAPE:
                    case WSP_GGML_OP_VIEW:
                    case WSP_GGML_OP_TRANSPOSE:
                    case WSP_GGML_OP_PERMUTE:
                        {
                            // noop
                        } break;
                    case WSP_GGML_OP_ADD:
                        {
                            WSP_GGML_ASSERT(wsp_ggml_is_contiguous(src0));
                            WSP_GGML_ASSERT(wsp_ggml_is_contiguous(src1));

                            // utilize float4
                            WSP_GGML_ASSERT(ne00 % 4 == 0);
                            const int64_t nb = ne00/4;

                            if (wsp_ggml_nelements(src1) == ne10) {
                                // src1 is a row
                                WSP_GGML_ASSERT(ne11 == 1);
                                [encoder setComputePipelineState:ctx->pipeline_add_row];
                            } else {
                                [encoder setComputePipelineState:ctx->pipeline_add];
                            }
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                            [encoder setBytes:&nb     length:sizeof(nb) atIndex:3];

                            const int64_t n = wsp_ggml_nelements(dst)/4;

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case WSP_GGML_OP_MUL:
                        {
                            WSP_GGML_ASSERT(wsp_ggml_is_contiguous(src0));
                            WSP_GGML_ASSERT(wsp_ggml_is_contiguous(src1));

                            // utilize float4
                            WSP_GGML_ASSERT(ne00 % 4 == 0);
                            const int64_t nb = ne00/4;

                            if (wsp_ggml_nelements(src1) == ne10) {
                                // src1 is a row
                                WSP_GGML_ASSERT(ne11 == 1);
                                [encoder setComputePipelineState:ctx->pipeline_mul_row];
                            } else {
                                [encoder setComputePipelineState:ctx->pipeline_mul];
                            }
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                            [encoder setBytes:&nb     length:sizeof(nb) atIndex:3];

                            const int64_t n = wsp_ggml_nelements(dst)/4;

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case WSP_GGML_OP_SCALE:
                        {
                            WSP_GGML_ASSERT(wsp_ggml_is_contiguous(src0));

                            const float scale = *(const float *) src1->data;

                            [encoder setComputePipelineState:ctx->pipeline_scale];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&scale length:sizeof(scale) atIndex:2];

                            const int64_t n = wsp_ggml_nelements(dst)/4;

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case WSP_GGML_OP_UNARY:
                        switch (wsp_ggml_get_unary_op(gf->nodes[i])) {
                            case WSP_GGML_UNARY_OP_SILU:
                                {
                                    [encoder setComputePipelineState:ctx->pipeline_silu];
                                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                                    const int64_t n = wsp_ggml_nelements(dst)/4;

                                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                                } break;
                            case WSP_GGML_UNARY_OP_RELU:
                                {
                                    [encoder setComputePipelineState:ctx->pipeline_relu];
                                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                                    const int64_t n = wsp_ggml_nelements(dst);

                                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                                } break;
                            case WSP_GGML_UNARY_OP_GELU:
                                {
                                    [encoder setComputePipelineState:ctx->pipeline_gelu];
                                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                                    const int64_t n = wsp_ggml_nelements(dst)/4;

                                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                                } break;
                            default:
                                {
                                    metal_printf("%s: node %3d, op = %8s not implemented\n", __func__, i, wsp_ggml_op_name(dst->op));
                                    WSP_GGML_ASSERT(false);
                                }
                        } break;
                    case WSP_GGML_OP_SOFT_MAX:
                        {
                            const int nth = 32;

                            if (ne00%4 == 0) {
                                [encoder setComputePipelineState:ctx->pipeline_soft_max_4];
                            } else {
                                [encoder setComputePipelineState:ctx->pipeline_soft_max];
                            }
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:2];
                            [encoder setBytes:&ne01 length:sizeof(ne01) atIndex:3];
                            [encoder setBytes:&ne02 length:sizeof(ne02) atIndex:4];

                            [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    case WSP_GGML_OP_DIAG_MASK_INF:
                        {
                            const int n_past = ((int32_t *)(dst->op_params))[0];

                            if (ne00%8 == 0) {
                                [encoder setComputePipelineState:ctx->pipeline_diag_mask_inf_8];
                            } else {
                                [encoder setComputePipelineState:ctx->pipeline_diag_mask_inf];
                            }
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00   length:sizeof(ne00) atIndex:2];
                            [encoder setBytes:&ne01   length:sizeof(ne01) atIndex:3];
                            [encoder setBytes:&n_past length:sizeof(int)  atIndex:4];

                            if (ne00%8 == 0) {
                                [encoder dispatchThreadgroups:MTLSizeMake(ne00*ne01*ne02/8, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                            }
                            else {
                                [encoder dispatchThreadgroups:MTLSizeMake(ne00, ne01, ne02) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                            }
                        } break;
                    case WSP_GGML_OP_MUL_MAT:
                        {
                            // TODO: needs to be updated after PR: https://github.com/ggerganov/ggml/pull/224

                            WSP_GGML_ASSERT(ne00 == ne10);
                            // WSP_GGML_ASSERT(ne02 == ne12); // Should be checked on individual data types until broadcast is implemented everywhere
                            uint gqa = ne12/ne02;
                            WSP_GGML_ASSERT(ne03 == ne13);

                            // for now the matrix-matrix multiplication kernel only works on A14+/M1+ SoCs
                            // AMD GPU and older A-chips will reuse matrix-vector multiplication kernel
                            if (!wsp_ggml_is_transposed(src0) &&
                                !wsp_ggml_is_transposed(src1) &&
                                src1t == WSP_GGML_TYPE_F32 &&
                                [ctx->device supportsFamily:MTLGPUFamilyApple7] &&
                                ne00%32 == 0 &&
                                ne11 > 1) {
                                switch (src0->type) {
                                    case WSP_GGML_TYPE_F32:  [encoder setComputePipelineState:ctx->pipeline_mul_mm_f32_f32];  break;
                                    case WSP_GGML_TYPE_F16:  [encoder setComputePipelineState:ctx->pipeline_mul_mm_f16_f32];  break;
                                    case WSP_GGML_TYPE_Q4_0: [encoder setComputePipelineState:ctx->pipeline_mul_mm_q4_0_f32]; break;
                                    case WSP_GGML_TYPE_Q4_1: [encoder setComputePipelineState:ctx->pipeline_mul_mm_q4_1_f32]; break;
                                    case WSP_GGML_TYPE_Q8_0: [encoder setComputePipelineState:ctx->pipeline_mul_mm_q8_0_f32]; break;
                                    case WSP_GGML_TYPE_Q2_K: [encoder setComputePipelineState:ctx->pipeline_mul_mm_q2_K_f32]; break;
                                    case WSP_GGML_TYPE_Q3_K: [encoder setComputePipelineState:ctx->pipeline_mul_mm_q3_K_f32]; break;
                                    case WSP_GGML_TYPE_Q4_K: [encoder setComputePipelineState:ctx->pipeline_mul_mm_q4_K_f32]; break;
                                    case WSP_GGML_TYPE_Q5_K: [encoder setComputePipelineState:ctx->pipeline_mul_mm_q5_K_f32]; break;
                                    case WSP_GGML_TYPE_Q6_K: [encoder setComputePipelineState:ctx->pipeline_mul_mm_q6_K_f32]; break;
                                    default: WSP_GGML_ASSERT(false && "MUL MAT-MAT not implemented");
                                }
                                [encoder setBuffer:id_src0 offset:offs_src0    atIndex:0];
                                [encoder setBuffer:id_src1 offset:offs_src1    atIndex:1];
                                [encoder setBuffer:id_dst  offset:offs_dst     atIndex:2];
                                [encoder setBytes:&ne00    length:sizeof(ne00) atIndex:3];
                                [encoder setBytes:&ne02    length:sizeof(ne02) atIndex:4];
                                [encoder setBytes:&nb01    length:sizeof(nb01) atIndex:5];
                                [encoder setBytes:&nb02    length:sizeof(nb02) atIndex:6];
                                [encoder setBytes:&ne12    length:sizeof(ne12) atIndex:7];
                                [encoder setBytes:&nb10    length:sizeof(nb10) atIndex:8];
                                [encoder setBytes:&nb11    length:sizeof(nb11) atIndex:9];
                                [encoder setBytes:&nb12    length:sizeof(nb12) atIndex:10];
                                [encoder setBytes:&ne0     length:sizeof(ne0)  atIndex:11];
                                [encoder setBytes:&ne1     length:sizeof(ne1)  atIndex:12];
                                [encoder setBytes:&gqa     length:sizeof(gqa)  atIndex:13];
                                [encoder setThreadgroupMemoryLength:8192 atIndex:0];
                                [encoder dispatchThreadgroups:MTLSizeMake( (ne11+31)/32, (ne01+63) / 64, ne12) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                            } else {
                                int nth0 = 32;
                                int nth1 = 1;
                                int nrows = 1;

                                // use custom matrix x vector kernel
                                switch (src0t) {
                                    case WSP_GGML_TYPE_F32:
                                        {
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_f32_f32];
                                            nrows = 4;
                                        } break;
                                    case WSP_GGML_TYPE_F16:
                                        {
                                            nth0 = 32;
                                            nth1 = 1;
                                            if (ne11 * ne12 < 4) {
                                                [encoder setComputePipelineState:ctx->pipeline_mul_mat_f16_f32_1row];
                                            } else if (ne00 >= 128 && ne01 >= 8 && ne00%4 == 0) {
                                                [encoder setComputePipelineState:ctx->pipeline_mul_mat_f16_f32_l4];
                                                nrows = ne11;
                                            } else {
                                                [encoder setComputePipelineState:ctx->pipeline_mul_mat_f16_f32];
                                                nrows = 4;
                                            }
                                        } break;
                                    case WSP_GGML_TYPE_Q4_0:
                                        {
                                            WSP_GGML_ASSERT(ne02 == 1);
                                            WSP_GGML_ASSERT(ne12 == 1);

                                            nth0 = 8;
                                            nth1 = 8;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q4_0_f32];
                                        } break;
                                    case WSP_GGML_TYPE_Q4_1:
                                        {
                                            WSP_GGML_ASSERT(ne02 == 1);
                                            WSP_GGML_ASSERT(ne12 == 1);

                                            nth0 = 8;
                                            nth1 = 8;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q4_1_f32];
                                        } break;
                                    case WSP_GGML_TYPE_Q8_0:
                                        {
                                            WSP_GGML_ASSERT(ne02 == 1);
                                            WSP_GGML_ASSERT(ne12 == 1);

                                            nth0 = 8;
                                            nth1 = 8;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q8_0_f32];
                                        } break;
                                    case WSP_GGML_TYPE_Q2_K:
                                        {
                                            WSP_GGML_ASSERT(ne02 == 1);
                                            WSP_GGML_ASSERT(ne12 == 1);

                                            nth0 = 2;
                                            nth1 = 32;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q2_K_f32];
                                        } break;
                                    case WSP_GGML_TYPE_Q3_K:
                                        {
                                            WSP_GGML_ASSERT(ne02 == 1);
                                            WSP_GGML_ASSERT(ne12 == 1);

                                            nth0 = 2;
                                            nth1 = 32;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q3_K_f32];
                                        } break;
                                    case WSP_GGML_TYPE_Q4_K:
                                        {
                                            WSP_GGML_ASSERT(ne02 == 1);
                                            WSP_GGML_ASSERT(ne12 == 1);

                                            nth0 = 4; //1;
                                            nth1 = 8; //32;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q4_K_f32];
                                        } break;
                                    case WSP_GGML_TYPE_Q5_K:
                                        {
                                            WSP_GGML_ASSERT(ne02 == 1);
                                            WSP_GGML_ASSERT(ne12 == 1);

                                            nth0 = 2;
                                            nth1 = 32;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q5_K_f32];
                                        } break;
                                    case WSP_GGML_TYPE_Q6_K:
                                        {
                                            WSP_GGML_ASSERT(ne02 == 1);
                                            WSP_GGML_ASSERT(ne12 == 1);

                                            nth0 = 2;
                                            nth1 = 32;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q6_K_f32];
                                        } break;
                                    default:
                                        {
                                            metal_printf("Asserting on type %d\n",(int)src0t);
                                            WSP_GGML_ASSERT(false && "not implemented");
                                        }
                                };

                                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                                [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:3];
                                [encoder setBytes:&ne01 length:sizeof(ne01) atIndex:4];
                                [encoder setBytes:&ne02 length:sizeof(ne02) atIndex:5];
                                [encoder setBytes:&nb00 length:sizeof(nb00) atIndex:6];
                                [encoder setBytes:&nb01 length:sizeof(nb01) atIndex:7];
                                [encoder setBytes:&nb02 length:sizeof(nb02) atIndex:8];
                                [encoder setBytes:&ne10 length:sizeof(ne10) atIndex:9];
                                [encoder setBytes:&ne11 length:sizeof(ne11) atIndex:10];
                                [encoder setBytes:&ne12 length:sizeof(ne12) atIndex:11];
                                [encoder setBytes:&nb10 length:sizeof(nb10) atIndex:12];
                                [encoder setBytes:&nb11 length:sizeof(nb11) atIndex:13];
                                [encoder setBytes:&nb12 length:sizeof(nb12) atIndex:14];
                                [encoder setBytes:&ne0  length:sizeof(ne0)  atIndex:15];
                                [encoder setBytes:&ne1  length:sizeof(ne1)  atIndex:16];
                                [encoder setBytes:&gqa  length:sizeof(gqa)  atIndex:17];

                                if (src0t == WSP_GGML_TYPE_Q4_0 || src0t == WSP_GGML_TYPE_Q4_1 || src0t == WSP_GGML_TYPE_Q8_0 ||
                                    src0t == WSP_GGML_TYPE_Q2_K) {// || src0t == WSP_GGML_TYPE_Q4_K) {
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 7)/8, ne11, ne12) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                                }
                                else if (src0t == WSP_GGML_TYPE_Q4_K) {
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, ne11, ne12) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                                }
                                else if (src0t == WSP_GGML_TYPE_Q3_K) {
#ifdef WSP_GGML_QKK_64
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 1)/2, ne11, ne12) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
#else
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, ne11, ne12) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
#endif
                                }
                                else if (src0t == WSP_GGML_TYPE_Q5_K) {
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, ne11, ne12) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                                }
                                else if (src0t == WSP_GGML_TYPE_Q6_K) {
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 1)/2, ne11, ne12) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                                } else {
                                    int64_t ny = (ne11 + nrows - 1)/nrows;
                                    [encoder dispatchThreadgroups:MTLSizeMake(ne01, ny, ne12) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                                }
                            }
                        } break;
                    case WSP_GGML_OP_GET_ROWS:
                        {
                            switch (src0->type) {
                                case WSP_GGML_TYPE_F32:  [encoder setComputePipelineState:ctx->pipeline_get_rows_f32];  break;
                                case WSP_GGML_TYPE_F16:  [encoder setComputePipelineState:ctx->pipeline_get_rows_f16];  break;
                                case WSP_GGML_TYPE_Q4_0: [encoder setComputePipelineState:ctx->pipeline_get_rows_q4_0]; break;
                                case WSP_GGML_TYPE_Q4_1: [encoder setComputePipelineState:ctx->pipeline_get_rows_q4_1]; break;
                                case WSP_GGML_TYPE_Q8_0: [encoder setComputePipelineState:ctx->pipeline_get_rows_q8_0]; break;
                                case WSP_GGML_TYPE_Q2_K: [encoder setComputePipelineState:ctx->pipeline_get_rows_q2_K]; break;
                                case WSP_GGML_TYPE_Q3_K: [encoder setComputePipelineState:ctx->pipeline_get_rows_q3_K]; break;
                                case WSP_GGML_TYPE_Q4_K: [encoder setComputePipelineState:ctx->pipeline_get_rows_q4_K]; break;
                                case WSP_GGML_TYPE_Q5_K: [encoder setComputePipelineState:ctx->pipeline_get_rows_q5_K]; break;
                                case WSP_GGML_TYPE_Q6_K: [encoder setComputePipelineState:ctx->pipeline_get_rows_q6_K]; break;
                                default: WSP_GGML_ASSERT(false && "not implemented");
                            }

                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                            [encoder setBytes:&ne00 length:sizeof( int64_t) atIndex:3];
                            [encoder setBytes:&nb01 length:sizeof(uint64_t) atIndex:4];
                            [encoder setBytes:&nb1  length:sizeof(uint64_t) atIndex:5];

                            const int64_t n = wsp_ggml_nelements(src1);

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case WSP_GGML_OP_RMS_NORM:
                        {
                            float eps;
                            memcpy(&eps, dst->op_params, sizeof(float));

                            const int nth = 512;

                            [encoder setComputePipelineState:ctx->pipeline_rms_norm];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00 length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&nb01 length:sizeof(uint64_t) atIndex:3];
                            [encoder setBytes:&eps  length:sizeof(   float) atIndex:4];
                            [encoder setThreadgroupMemoryLength:nth/32*sizeof(float) atIndex:0];

                            const int64_t nrows = wsp_ggml_nrows(src0);

                            [encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    case WSP_GGML_OP_NORM:
                        {
                            float eps;
                            memcpy(&eps, dst->op_params, sizeof(float));

                            const int nth = 256;

                            [encoder setComputePipelineState:ctx->pipeline_norm];
                            [encoder setBuffer:id_src0 offset:offs_src0        atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst         atIndex:1];
                            [encoder setBytes:&ne00    length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&nb01    length:sizeof(uint64_t) atIndex:3];
                            [encoder setBytes:&eps     length:sizeof(   float) atIndex:4];
                            [encoder setThreadgroupMemoryLength:nth*sizeof(float) atIndex:0];

                            const int64_t nrows = wsp_ggml_nrows(src0);

                            [encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    case WSP_GGML_OP_ALIBI:
                        {
                            WSP_GGML_ASSERT((src0t == WSP_GGML_TYPE_F32));

                            const int n_past = ((int32_t *) dst->op_params)[0]; UNUSED(n_past);
                            const int n_head = ((int32_t *) dst->op_params)[1];
                            float max_bias;
                            memcpy(&max_bias, (int32_t *) dst->op_params + 2, sizeof(float));

                            if (__builtin_popcount(n_head) != 1) {
                                WSP_GGML_ASSERT(false && "only power-of-two n_head implemented");
                            }

                            const int n_heads_log2_floor = 1 << (int) floor(log2(n_head));
                            const float m0 = powf(2.0f, -(max_bias) / n_heads_log2_floor);

                            [encoder setComputePipelineState:ctx->pipeline_alibi_f32];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00 length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&ne01 length:sizeof( int64_t) atIndex:3];
                            [encoder setBytes:&ne02 length:sizeof( int64_t) atIndex:4];
                            [encoder setBytes:&ne03 length:sizeof( int64_t) atIndex:5];
                            [encoder setBytes:&nb00 length:sizeof(uint64_t) atIndex:6];
                            [encoder setBytes:&nb01 length:sizeof(uint64_t) atIndex:7];
                            [encoder setBytes:&nb02 length:sizeof(uint64_t) atIndex:8];
                            [encoder setBytes:&nb03 length:sizeof(uint64_t) atIndex:9];
                            [encoder setBytes:&ne0  length:sizeof( int64_t) atIndex:10];
                            [encoder setBytes:&ne1  length:sizeof( int64_t) atIndex:11];
                            [encoder setBytes:&ne2  length:sizeof( int64_t) atIndex:12];
                            [encoder setBytes:&ne3  length:sizeof( int64_t) atIndex:13];
                            [encoder setBytes:&nb0  length:sizeof(uint64_t) atIndex:14];
                            [encoder setBytes:&nb1  length:sizeof(uint64_t) atIndex:15];
                            [encoder setBytes:&nb2  length:sizeof(uint64_t) atIndex:16];
                            [encoder setBytes:&nb3  length:sizeof(uint64_t) atIndex:17];
                            [encoder setBytes:&m0  length:sizeof(    float) atIndex:18];

                            const int nth = 32;

                            [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    case WSP_GGML_OP_ROPE:
                        {
                            const int n_past = ((int32_t *) dst->op_params)[0];
                            const int n_dims = ((int32_t *) dst->op_params)[1];
                            const int mode   = ((int32_t *) dst->op_params)[2];

                            float freq_base;
                            float freq_scale;
                            memcpy(&freq_base,  (int32_t *) dst->op_params + 4, sizeof(float));
                            memcpy(&freq_scale, (int32_t *) dst->op_params + 5, sizeof(float));

                            [encoder setComputePipelineState:ctx->pipeline_rope];
                            [encoder setBuffer:id_src0 offset:offs_src0        atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst         atIndex:1];
                            [encoder setBytes:&ne00    length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&ne01    length:sizeof( int64_t) atIndex:3];
                            [encoder setBytes:&ne02    length:sizeof( int64_t) atIndex:4];
                            [encoder setBytes:&ne03    length:sizeof( int64_t) atIndex:5];
                            [encoder setBytes:&nb00    length:sizeof(uint64_t) atIndex:6];
                            [encoder setBytes:&nb01    length:sizeof(uint64_t) atIndex:7];
                            [encoder setBytes:&nb02    length:sizeof(uint64_t) atIndex:8];
                            [encoder setBytes:&nb03    length:sizeof(uint64_t) atIndex:9];
                            [encoder setBytes:&ne0     length:sizeof( int64_t) atIndex:10];
                            [encoder setBytes:&ne1     length:sizeof( int64_t) atIndex:11];
                            [encoder setBytes:&ne2     length:sizeof( int64_t) atIndex:12];
                            [encoder setBytes:&ne3     length:sizeof( int64_t) atIndex:13];
                            [encoder setBytes:&nb0     length:sizeof(uint64_t) atIndex:14];
                            [encoder setBytes:&nb1     length:sizeof(uint64_t) atIndex:15];
                            [encoder setBytes:&nb2     length:sizeof(uint64_t) atIndex:16];
                            [encoder setBytes:&nb3     length:sizeof(uint64_t) atIndex:17];
                            [encoder setBytes:&n_past  length:sizeof(     int) atIndex:18];
                            [encoder setBytes:&n_dims  length:sizeof(     int) atIndex:19];
                            [encoder setBytes:&mode    length:sizeof(     int) atIndex:20];
                            [encoder setBytes:&freq_base  length:sizeof(float) atIndex:21];
                            [encoder setBytes:&freq_scale length:sizeof(float) atIndex:22];

                            [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                        } break;
                    case WSP_GGML_OP_DUP:
                    case WSP_GGML_OP_CPY:
                    case WSP_GGML_OP_CONT:
                        {
                            const int nth = 32;

                            switch (src0t) {
                                case WSP_GGML_TYPE_F32:
                                    {
                                        switch (dstt) {
                                            case WSP_GGML_TYPE_F16: [encoder setComputePipelineState:ctx->pipeline_cpy_f32_f16]; break;
                                            case WSP_GGML_TYPE_F32: [encoder setComputePipelineState:ctx->pipeline_cpy_f32_f32]; break;
                                            default: WSP_GGML_ASSERT(false && "not implemented");
                                        };
                                    } break;
                                case WSP_GGML_TYPE_F16:
                                    {
                                        switch (dstt) {
                                            case WSP_GGML_TYPE_F16: [encoder setComputePipelineState:ctx->pipeline_cpy_f16_f16]; break;
                                            case WSP_GGML_TYPE_F32: WSP_GGML_ASSERT(false && "cpy_f16_f32 not implemented"); break;
                                            default: WSP_GGML_ASSERT(false && "not implemented");
                                        };
                                    } break;
                                default: WSP_GGML_ASSERT(false && "not implemented");
                            }

                            [encoder setBuffer:id_src0 offset:offs_src0        atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst         atIndex:1];
                            [encoder setBytes:&ne00    length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&ne01    length:sizeof( int64_t) atIndex:3];
                            [encoder setBytes:&ne02    length:sizeof( int64_t) atIndex:4];
                            [encoder setBytes:&ne03    length:sizeof( int64_t) atIndex:5];
                            [encoder setBytes:&nb00    length:sizeof(uint64_t) atIndex:6];
                            [encoder setBytes:&nb01    length:sizeof(uint64_t) atIndex:7];
                            [encoder setBytes:&nb02    length:sizeof(uint64_t) atIndex:8];
                            [encoder setBytes:&nb03    length:sizeof(uint64_t) atIndex:9];
                            [encoder setBytes:&ne0     length:sizeof( int64_t) atIndex:10];
                            [encoder setBytes:&ne1     length:sizeof( int64_t) atIndex:11];
                            [encoder setBytes:&ne2     length:sizeof( int64_t) atIndex:12];
                            [encoder setBytes:&ne3     length:sizeof( int64_t) atIndex:13];
                            [encoder setBytes:&nb0     length:sizeof(uint64_t) atIndex:14];
                            [encoder setBytes:&nb1     length:sizeof(uint64_t) atIndex:15];
                            [encoder setBytes:&nb2     length:sizeof(uint64_t) atIndex:16];
                            [encoder setBytes:&nb3     length:sizeof(uint64_t) atIndex:17];

                            [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    default:
                        {
                            metal_printf("%s: node %3d, op = %8s not implemented\n", __func__, i, wsp_ggml_op_name(dst->op));
                            WSP_GGML_ASSERT(false);
                        }
                }
            }

            if (encoder != nil) {
                [encoder endEncoding];
                encoder = nil;
            }

            [command_buffer commit];
        });
    }

    // wait for all threads to finish
    dispatch_barrier_sync(ctx->d_queue, ^{});

    // check status of command buffers
    // needed to detect if the device ran out-of-memory for example (#1881)
    for (int i = 0; i < n_cb; i++) {
        [ctx->command_buffers[i] waitUntilCompleted];

        MTLCommandBufferStatus status = (MTLCommandBufferStatus) [ctx->command_buffers[i] status];
        if (status != MTLCommandBufferStatusCompleted) {
            metal_printf("%s: command buffer %d failed with status %lu\n", __func__, i, status);
            WSP_GGML_ASSERT(false);
        }
    }

    }
}
