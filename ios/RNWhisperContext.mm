#import "RNWhisperContext.h"
#import <Metal/Metal.h>
#include <vector>

#define NUM_BYTES_PER_BUFFER 16 * 1024

@implementation RNWhisperContext

static void whisper_log_callback_default(wsp_ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
#ifndef WHISPER_DEBUG
    if (level == WSP_GGML_LOG_LEVEL_DEBUG) {
        return;
    }
#endif
    fputs(text, stderr);
    fflush(stderr);
}

static void* retained_log_block = nullptr;

+ (void)toggleNativeLog:(BOOL)enabled onEmitLog:(void (^)(NSString *level, NSString *text))onEmitLog {
  if (enabled) {
      void (^copiedBlock)(NSString *, NSString *) = [onEmitLog copy];
      retained_log_block = (__bridge_retained void *)(copiedBlock);
      whisper_log_set([](enum wsp_ggml_log_level level, const char * text, void * data) {
          whisper_log_callback_default(level, text, data);
          NSString *levelStr = @"";
          if (level == WSP_GGML_LOG_LEVEL_ERROR) {
              levelStr = @"error";
          } else if (level == WSP_GGML_LOG_LEVEL_INFO) {
              levelStr = @"info";
          } else if (level == WSP_GGML_LOG_LEVEL_WARN) {
              levelStr = @"warn";
          }

          NSString *textStr = [NSString stringWithUTF8String:text];
          // NOTE: Convert to UTF-8 string may fail
          if (!textStr) {
              return;
          }
          void (^block)(NSString *, NSString *) = (__bridge void (^)(NSString *, NSString *))(data);
          block(levelStr, textStr);
      }, retained_log_block);
  } else {
      whisper_log_set(whisper_log_callback_default, nullptr);
      if (retained_log_block) {
          CFRelease(retained_log_block);
          retained_log_block = nullptr;
      }
  }
}

+ (instancetype)initWithModelPath:(NSString *)modelPath
    contextId:(int)contextId
    noCoreML:(BOOL)noCoreML
    noMetal:(BOOL)noMetal
    useFlashAttn:(BOOL)useFlashAttn
{
    RNWhisperContext *context = [[RNWhisperContext alloc] init];
    context->contextId = contextId;
    struct whisper_context_params cparams;
    NSString *reasonNoMetal = @"";
    cparams.use_gpu = !noMetal;
    cparams.flash_attn = useFlashAttn;

    // TODO: Expose dtw_token_timestamps and dtw_aheads_preset
    cparams.dtw_token_timestamps = false;
    // cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE;

    cparams.use_coreml = !noCoreML;
#ifndef WHISPER_USE_COREML
    if (cparams.use_coreml) {
        NSLog(@"[RNWhisper] CoreML is not enabled in this build, ignoring use_coreml option");
        cparams.use_coreml = false;
    }
#endif

#ifndef WSP_GGML_USE_METAL
    if (cparams.use_gpu) {
        NSLog(@"[RNWhisper] ggml-metal is not enabled in this build, ignoring use_gpu option");
        cparams.use_gpu = false;
    }
    reasonNoMetal = @"Metal is not enabled in this build";
#endif

#ifdef WSP_GGML_USE_METAL
    if (cparams.use_gpu) {
        cparams.gpu_device = 0;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        // Check ggml-metal availability
        BOOL supportsGgmlMetal = [device supportsFamily:MTLGPUFamilyApple7];
        if (@available(iOS 16.0, tvOS 16.0, *)) {
            supportsGgmlMetal = supportsGgmlMetal && [device supportsFamily:MTLGPUFamilyMetal3];
        }
        if (!supportsGgmlMetal) {
            cparams.use_gpu = false;
            reasonNoMetal = @"Metal is not supported in this device";
        }

#if TARGET_OS_SIMULATOR
        // Use the backend, but no layers because not supported fully on simulator
        cparams.use_gpu = false;
        reasonNoMetal = @"Metal is not supported in simulator";
#endif

        device = nil;
    }
#endif // WSP_GGML_USE_METAL

    if (cparams.use_gpu && cparams.use_coreml) {
        NSLog(@"[RNWhisper] Both use_gpu and use_coreml are enabled, ignoring use_coreml option");
        cparams.use_coreml = false; // Skip CoreML if Metal is enabled
    }

    context->ctx = whisper_init_from_file_with_params([modelPath UTF8String], cparams);
    context->dQueue = dispatch_queue_create(
        [[NSString stringWithFormat:@"RNWhisperContext-%d", contextId] UTF8String],
        DISPATCH_QUEUE_SERIAL
    );
    context->isMetalEnabled = cparams.use_gpu;
    context->reasonNoMetal = reasonNoMetal;
    return context;
}

- (bool)isMetalEnabled {
    return isMetalEnabled;
}

- (NSString *)reasonNoMetal {
    return reasonNoMetal;
}

- (struct whisper_context *)getContext {
    return self->ctx;
}

- (dispatch_queue_t)getDispatchQueue {
    return self->dQueue;
}

- (bool)isTranscribing {
    return self->isTranscribing;
}

struct rnwhisper_segments_callback_data {
    void (^onNewSegments)(NSDictionary *);
    int total_n_new;
    bool tdrzEnable;
};

- (void)transcribeData:(int)jobId
    audioData:(float *)audioData
    audioDataCount:(int)audioDataCount
    options:(NSDictionary *)options
    onProgress:(void (^)(int))onProgress
    onNewSegments:(void (^)(NSDictionary *))onNewSegments
    onEnd:(void (^)(int))onEnd
{
    dispatch_async(dQueue, ^{
        self->isTranscribing = true;

        whisper_full_params params = [self createParams:options jobId:jobId];

        if (options[@"onProgress"] && [options[@"onProgress"] boolValue]) {
            params.progress_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, int progress, void * user_data) {
                void (^onProgress)(int) = (__bridge void (^)(int))user_data;
                onProgress(progress);
            };
            params.progress_callback_user_data = (__bridge void *)(onProgress);
        }

        if (options[@"onNewSegments"] && [options[@"onNewSegments"] boolValue]) {
            params.new_segment_callback = [](struct whisper_context * ctx, struct whisper_state * /*state*/, int n_new, void * user_data) {
                struct rnwhisper_segments_callback_data *data = (struct rnwhisper_segments_callback_data *)user_data;
                data->total_n_new += n_new;

                NSString *text = @"";
                NSMutableArray *segments = [[NSMutableArray alloc] init];
                for (int i = data->total_n_new - n_new; i < data->total_n_new; i++) {
                    const char * text_cur = whisper_full_get_segment_text(ctx, i);
                    NSMutableString *mutable_ns_text = [NSMutableString stringWithUTF8String:text_cur];

                    if (data->tdrzEnable && whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                        [mutable_ns_text appendString:@" [SPEAKER_TURN]"];
                    }

                    text = [text stringByAppendingString:mutable_ns_text];

                    const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                    const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
                    NSDictionary *segment = @{
                        @"text": [NSString stringWithString:mutable_ns_text],
                        @"t0": [NSNumber numberWithLongLong:t0],
                        @"t1": [NSNumber numberWithLongLong:t1]
                    };
                    [segments addObject:segment];
                }

                NSDictionary *result = @{
                    @"nNew": [NSNumber numberWithInt:n_new],
                    @"totalNNew": [NSNumber numberWithInt:data->total_n_new],
                    @"result": text,
                    @"segments": segments
                };
                void (^onNewSegments)(NSDictionary *) = (void (^)(NSDictionary *))data->onNewSegments;
                onNewSegments(result);
            };
            struct rnwhisper_segments_callback_data user_data = {
                .onNewSegments = onNewSegments,
                .tdrzEnable = options[@"tdrzEnable"] && [options[@"tdrzEnable"] boolValue],
                .total_n_new = 0,
            };
            params.new_segment_callback_user_data = &user_data;
        }

        rnwhisper::job* job = rnwhisper::job_new(jobId, params);
        self->job = job;
        int code = [self fullTranscribe:job audioData:audioData audioDataCount:audioDataCount];
        rnwhisper::job_remove(jobId);
        self->job = nullptr;
        self->isTranscribing = false;
        onEnd(code);
    });
}

- (void)stopTranscribe:(int)jobId {
    if (self->job != nullptr) self->job->abort();
    self->isTranscribing = false;
    dispatch_barrier_sync(dQueue, ^{});
}

- (void)stopCurrentTranscribe {
    if (self->job == nullptr) return;
    [self stopTranscribe:self->job->job_id];
}

- (struct whisper_full_params)createParams:(NSDictionary *)options jobId:(int)jobId {
    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    const int n_threads = options[@"maxThreads"] != nil ?
      [options[@"maxThreads"] intValue] : 0;

    const int max_threads = (int) [[NSProcessInfo processInfo] processorCount];
    // Use 2 threads by default on 4-core devices, 4 threads on more cores
    const int default_n_threads = max_threads == 4 ? 2 : MIN(4, max_threads);

    if (options[@"beamSize"] != nil) {
        params.strategy = WHISPER_SAMPLING_BEAM_SEARCH;
        params.beam_search.beam_size = [options[@"beamSize"] intValue];
    }

    params.print_realtime   = false;
    params.print_progress   = false;
    params.print_timestamps = false;
    params.print_special    = false;
    params.translate        = options[@"translate"] != nil ? [options[@"translate"] boolValue] : false;
    params.language         = options[@"language"] != nil ? strdup([options[@"language"] UTF8String]) : "auto";
    params.n_threads        = n_threads > 0 ? n_threads : default_n_threads;
    params.offset_ms        = 0;
    params.no_context       = true;
    params.single_segment   = false;

    if (options[@"maxLen"] != nil) {
        params.max_len = [options[@"maxLen"] intValue];
    }
    params.token_timestamps = options[@"tokenTimestamps"] != nil ? [options[@"tokenTimestamps"] boolValue] : false;
    params.tdrz_enable = options[@"tdrzEnable"] != nil ? [options[@"tdrzEnable"] boolValue] : false;

    if (options[@"bestOf"] != nil) {
        params.greedy.best_of = [options[@"bestOf"] intValue];
    }
    if (options[@"maxContext"] != nil) {
        params.n_max_text_ctx = [options[@"maxContext"] intValue];
    }
    if (options[@"offset"] != nil) {
        params.offset_ms = [options[@"offset"] intValue];
    }
    if (options[@"duration"] != nil) {
        params.duration_ms = [options[@"duration"] intValue];
    }
    if (options[@"wordThold"] != nil) {
        params.thold_pt = [options[@"wordThold"] intValue];
    }
    if (options[@"temperature"] != nil) {
        params.temperature = [options[@"temperature"] floatValue];
    }
    if (options[@"temperatureInc"] != nil) {
        params.temperature_inc = [options[@"temperature_inc"] floatValue];
    }
    if (options[@"prompt"] != nil) {
        params.initial_prompt = strdup([options[@"prompt"] UTF8String]);
    }

    return params;
}

- (int)fullTranscribe:(rnwhisper::job *)job
  audioData:(float *)audioData
  audioDataCount:(int)audioDataCount
{
    whisper_reset_timings(self->ctx);
    int code = whisper_full(self->ctx, job->params, audioData, audioDataCount);
    if (job && job->is_aborted()) code = -999;
    // if (code == 0) {
    //     whisper_print_timings(self->ctx);
    // }
    return code;
}

- (NSMutableDictionary *)getTextSegments:(NSDictionary *)options {
    NSString *text = @"";
    int n_segments = whisper_full_n_segments(self->ctx);

    NSMutableArray *segments = [[NSMutableArray alloc] init];
    for (int i = 0; i < n_segments; i++) {
        const char * text_cur = whisper_full_get_segment_text(self->ctx, i);
        NSMutableString *mutable_ns_text = [NSMutableString stringWithUTF8String:text_cur];

        // Simplified condition
        if (options[@"tdrzEnable"] &&
            [options[@"tdrzEnable"] boolValue] &&
            whisper_full_get_segment_speaker_turn_next(self->ctx, i)) {
            [mutable_ns_text appendString:@" [SPEAKER_TURN]"];
        }

        text = [text stringByAppendingString:mutable_ns_text];

        const int64_t t0 = whisper_full_get_segment_t0(self->ctx, i);
        const int64_t t1 = whisper_full_get_segment_t1(self->ctx, i);
        NSDictionary *segment = @{
            @"text": [NSString stringWithString:mutable_ns_text],
            @"t0": [NSNumber numberWithLongLong:t0],
            @"t1": [NSNumber numberWithLongLong:t1]
        };
        [segments addObject:segment];
    }
    NSMutableDictionary *result = [[NSMutableDictionary alloc] init];
    result[@"result"] = text;
    result[@"segments"] = segments;
    return result;
}

- (NSString *)bench:(int)maxThreads {
    const int n_threads = maxThreads > 0 ? maxThreads : 0;

    const int max_threads = (int) [[NSProcessInfo processInfo] processorCount];
    // Use 2 threads by default on 4-core devices, 4 threads on more cores
    const int default_n_threads = max_threads == 4 ? 2 : MIN(4, max_threads);
    NSString *result = [NSString stringWithUTF8String:rnwhisper::bench(self->ctx, n_threads).c_str()];
    return result;
}

- (void)invalidate {
    [self stopCurrentTranscribe];
    whisper_free(self->ctx);
}

@end
