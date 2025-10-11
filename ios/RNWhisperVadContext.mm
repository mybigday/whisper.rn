#import "RNWhisperVadContext.h"
#import "RNWhisperAudioUtils.h"
#import <Metal/Metal.h>

@implementation RNWhisperVadContext

+ (instancetype)initWithModelPath:(NSString *)modelPath contextId:(int)contextId noMetal:(BOOL)noMetal nThreads:(NSNumber *)nThreads {
    RNWhisperVadContext *context = [[RNWhisperVadContext alloc] init];

    context->contextId = contextId;
    context->dQueue = dispatch_queue_create("rnwhisper.vad.serial_queue", DISPATCH_QUEUE_SERIAL);
    NSString *reasonNoMetal = @"";

    // Set up VAD context parameters
    struct whisper_vad_context_params ctx_params = whisper_vad_default_context_params();
    ctx_params.use_gpu = !noMetal;
    if (nThreads != nil) {
        ctx_params.n_threads = [nThreads intValue];
    }

#ifdef WSP_GGML_USE_METAL
    if (ctx_params.use_gpu) {
        // TODO: GPU VAD is forced disabled until the performance is improved (ref: whisper.cpp/whisper_vad_init_context)
        ctx_params.use_gpu = false;
        // ctx_params.gpu_device = 0;

        // id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        // // Check ggml-metal availability
        // BOOL supportsGgmlMetal = [device supportsFamily:MTLGPUFamilyApple7];
        // if (@available(iOS 16.0, tvOS 16.0, *)) {
        //     supportsGgmlMetal = supportsGgmlMetal && [device supportsFamily:MTLGPUFamilyMetal3];
        // }
        // if (!supportsGgmlMetal) {
        //   ctx_params.use_gpu = false;
        //     reasonNoMetal = @"Metal is not supported in this device";
        // }
        // device = nil;

#if TARGET_OS_SIMULATOR
        // Use the backend, but no layers because not supported fully on simulator
        ctx_params.use_gpu = false;
        reasonNoMetal = @"Metal is not supported in simulator";
#endif
    }
#endif // WSP_GGML_USE_METAL

    // Initialize VAD context
    context->vctx = whisper_vad_init_from_file_with_params([modelPath UTF8String], ctx_params);

    if (context->vctx == NULL) {
        NSLog(@"Failed to initialize VAD context from model: %@", modelPath);
        return nil;
    }

    // Check GPU status
    context->isMetalEnabled = ctx_params.use_gpu;
    context->reasonNoMetal = reasonNoMetal;

    return context;
}

- (bool)isMetalEnabled {
    return isMetalEnabled;
}

- (NSString *)reasonNoMetal {
    return reasonNoMetal;
}

- (struct whisper_vad_context *)getVadContext {
    return vctx;
}

- (dispatch_queue_t)getDispatchQueue {
    return dQueue;
}

- (NSArray *)detectSpeech:(float *)samples samplesCount:(int)samplesCount options:(NSDictionary *)options {
    if (vctx == NULL) {
        NSLog(@"VAD context is null");
        return @[];
    }

    // Run VAD detection
    bool speechDetected = whisper_vad_detect_speech(vctx, samples, samplesCount);
    if (!speechDetected) {
        return @[];
    }

    // Get VAD parameters
    struct whisper_vad_params vad_params = whisper_vad_default_params();

    if ([options objectForKey:@"threshold"]) {
        vad_params.threshold = [[options objectForKey:@"threshold"] floatValue];
    }
    if ([options objectForKey:@"minSpeechDurationMs"]) {
        vad_params.min_speech_duration_ms = [[options objectForKey:@"minSpeechDurationMs"] intValue];
    }
    if ([options objectForKey:@"minSilenceDurationMs"]) {
        vad_params.min_silence_duration_ms = [[options objectForKey:@"minSilenceDurationMs"] intValue];
    }
    if ([options objectForKey:@"maxSpeechDurationS"]) {
        vad_params.max_speech_duration_s = [[options objectForKey:@"maxSpeechDurationS"] floatValue];
    }
    if ([options objectForKey:@"speechPadMs"]) {
        vad_params.speech_pad_ms = [[options objectForKey:@"speechPadMs"] intValue];
    }
    if ([options objectForKey:@"samplesOverlap"]) {
        vad_params.samples_overlap = [[options objectForKey:@"samplesOverlap"] floatValue];
    }

    // Get segments from VAD probabilities
    struct whisper_vad_segments * segments = whisper_vad_segments_from_probs(vctx, vad_params);
    if (segments == NULL) {
        return @[];
    }

    // Convert segments to NSArray
    NSMutableArray *result = [[NSMutableArray alloc] init];
    int n_segments = whisper_vad_segments_n_segments(segments);

    for (int i = 0; i < n_segments; i++) {
        float t0 = whisper_vad_segments_get_segment_t0(segments, i);
        float t1 = whisper_vad_segments_get_segment_t1(segments, i);

        NSDictionary *segment = @{
            @"t0": @(t0),
            @"t1": @(t1)
        };
        [result addObject:segment];
    }

    // Clean up
    whisper_vad_free_segments(segments);

    return result;
}

- (void)invalidate {
    if (vctx != NULL) {
        whisper_vad_free(vctx);
        vctx = NULL;
    }
}

- (void)dealloc {
    [self invalidate];
}

@end
