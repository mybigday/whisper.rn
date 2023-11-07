#import "RNWhisperContext.h"
#import "RNWhisperAudioUtils.h"
#import <Metal/Metal.h>
#include <vector>

#define NUM_BYTES_PER_BUFFER 16 * 1024

@implementation RNWhisperContext

+ (instancetype)initWithModelPath:(NSString *)modelPath
    contextId:(int)contextId
    noCoreML:(BOOL)noCoreML
    noMetal:(BOOL)noMetal
{
    RNWhisperContext *context = [[RNWhisperContext alloc] init];
    context->contextId = contextId;
    struct whisper_context_params cparams;
    NSString *reasonNoMetal = @"";
    cparams.use_gpu = !noMetal;

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
#endif

#ifdef WSP_GGML_USE_METAL
    if (cparams.use_gpu) {
#if TARGET_OS_SIMULATOR
        NSLog(@"[RNWhisper] ggml-metal is not available in simulator, ignoring use_gpu option: %@", reasonNoMetal);
        cparams.use_gpu = false;
#else // TARGET_OS_SIMULATOR
        // Check ggml-metal availability
        NSError * error = nil;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLLibrary> library = [device
            newLibraryWithSource:@"#include <metal_stdlib>\n"
                                    "using namespace metal;"
                                    "kernel void test() { simd_sum(0); }"
            options:nil
            error:&error
        ];
        if (error) {
            reasonNoMetal = [error localizedDescription];
        } else {
            id<MTLFunction> kernel = [library newFunctionWithName:@"test"];
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernel error:&error];
            if (pipeline == nil) {
                reasonNoMetal = [error localizedDescription];
                NSLog(@"[RNWhisper] ggml-metal is not available, ignoring use_gpu option: %@", reasonNoMetal);
                cparams.use_gpu = false;
            }
        }
#endif // TARGET_OS_SIMULATOR
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

- (void)prepareRealtime:(NSDictionary *)options {
    self->recordState.options = options;

    self->recordState.dataFormat.mSampleRate = WHISPER_SAMPLE_RATE; // 16000
    self->recordState.dataFormat.mFormatID = kAudioFormatLinearPCM;
    self->recordState.dataFormat.mFramesPerPacket = 1;
    self->recordState.dataFormat.mChannelsPerFrame = 1; // mono
    self->recordState.dataFormat.mBytesPerFrame = 2;
    self->recordState.dataFormat.mBytesPerPacket = 2;
    self->recordState.dataFormat.mBitsPerChannel = 16;
    self->recordState.dataFormat.mReserved = 0;
    self->recordState.dataFormat.mFormatFlags = kLinearPCMFormatFlagIsSignedInteger;

    int maxAudioSecOpt = options[@"realtimeAudioSec"] != nil ? [options[@"realtimeAudioSec"] intValue] : 0;
    int maxAudioSec = maxAudioSecOpt > 0 ? maxAudioSecOpt : DEFAULT_MAX_AUDIO_SEC;
    self->recordState.maxAudioSec = maxAudioSec;

    int realtimeAudioSliceSec = options[@"realtimeAudioSliceSec"] != nil ? [options[@"realtimeAudioSliceSec"] intValue] : 0;
    int audioSliceSec = realtimeAudioSliceSec > 0 && realtimeAudioSliceSec < maxAudioSec ? realtimeAudioSliceSec : maxAudioSec;

    self->recordState.audioOutputPath = options[@"audioOutputPath"];

    self->recordState.useVad = options[@"useVad"] != nil ? [options[@"useVad"] boolValue] : false;
    self->recordState.vadMs = options[@"vadMs"] != nil ? [options[@"vadMs"] intValue] : 2000;
    if (self->recordState.vadMs < 2000) self->recordState.vadMs = 2000;

    self->recordState.vadThold = options[@"vadThold"] != nil ? [options[@"vadThold"] floatValue] : 0.6f;
    self->recordState.vadFreqThold = options[@"vadFreqThold"] != nil ? [options[@"vadFreqThold"] floatValue] : 100.0f;

    self->recordState.audioSliceSec = audioSliceSec;
    self->recordState.isUseSlices = audioSliceSec < maxAudioSec;

    self->recordState.sliceIndex = 0;
    self->recordState.transcribeSliceIndex = 0;
    self->recordState.nSamplesTranscribing = 0;

    [self freeBufferIfNeeded];
    self->recordState.shortBufferSlices = [NSMutableArray new];

    int16_t *audioBufferI16 = (int16_t *) malloc(audioSliceSec * WHISPER_SAMPLE_RATE * sizeof(int16_t));
    [self->recordState.shortBufferSlices addObject:[NSValue valueWithPointer:audioBufferI16]];

    self->recordState.sliceNSamples = [NSMutableArray new];
    [self->recordState.sliceNSamples addObject:[NSNumber numberWithInt:0]];

    self->recordState.isRealtime = true;
    self->recordState.isTranscribing = false;
    self->recordState.isCapturing = false;
    self->recordState.isStoppedByAction = false;

    self->recordState.mSelf = self;
}

- (void)freeBufferIfNeeded {
    if (self->recordState.shortBufferSlices != nil) {
        for (int i = 0; i < [self->recordState.shortBufferSlices count]; i++) {
            int16_t *audioBufferI16 = (int16_t *) [self->recordState.shortBufferSlices[i] pointerValue];
            free(audioBufferI16);
        }
        self->recordState.shortBufferSlices = nil;
    }
}

bool vad(RNWhisperContextRecordState *state, int16_t* audioBufferI16, int nSamples, int n)
{
    bool isSpeech = true;
    if (!state->isTranscribing && state->useVad) {
        int sampleSize = (int) (WHISPER_SAMPLE_RATE * state->vadMs / 1000);
        if (nSamples + n > sampleSize) {
            int start = nSamples + n - sampleSize;
            std::vector<float> audioBufferF32Vec(sampleSize);
            for (int i = 0; i < sampleSize; i++) {
                audioBufferF32Vec[i] = (float)audioBufferI16[i + start] / 32768.0f;
            }
            isSpeech = rn_whisper_vad_simple(audioBufferF32Vec, WHISPER_SAMPLE_RATE, 1000, state->vadThold, state->vadFreqThold, false);
            NSLog(@"[RNWhisper] VAD result: %d", isSpeech);
        } else {
            isSpeech = false;
        }
    }
    return isSpeech;
}

void AudioInputCallback(void * inUserData,
    AudioQueueRef inAQ,
    AudioQueueBufferRef inBuffer,
    const AudioTimeStamp * inStartTime,
    UInt32 inNumberPacketDescriptions,
    const AudioStreamPacketDescription * inPacketDescs)
{
    RNWhisperContextRecordState *state = (RNWhisperContextRecordState *)inUserData;

    if (!state->isCapturing) {
        NSLog(@"[RNWhisper] Not capturing, ignoring audio");
        if (!state->isTranscribing) {
            [state->mSelf finishRealtimeTranscribe:state result:@{}];
        }
        return;
    }

    int totalNSamples = 0;
    for (int i = 0; i < [state->sliceNSamples count]; i++) {
        totalNSamples += [[state->sliceNSamples objectAtIndex:i] intValue];
    }

    const int n = inBuffer->mAudioDataByteSize / 2;

    int nSamples = [state->sliceNSamples[state->sliceIndex] intValue];

    if (totalNSamples + n > state->maxAudioSec * WHISPER_SAMPLE_RATE) {
        NSLog(@"[RNWhisper] Audio buffer is full, stop capturing");
        state->isCapturing = false;
        [state->mSelf stopAudio];
        if (
            !state->isTranscribing &&
            nSamples == state->nSamplesTranscribing &&
            state->sliceIndex == state->transcribeSliceIndex
        ) {
            [state->mSelf finishRealtimeTranscribe:state result:@{}];
        } else if (
            !state->isTranscribing &&
            nSamples != state->nSamplesTranscribing
        ) {
            int16_t* audioBufferI16 = (int16_t*) [state->shortBufferSlices[state->sliceIndex] pointerValue];
            if (!vad(state, audioBufferI16, nSamples, 0)) {
                [state->mSelf finishRealtimeTranscribe:state result:@{}];
                return;
            }
            state->isTranscribing = true;
            dispatch_async([state->mSelf getDispatchQueue], ^{
                [state->mSelf fullTranscribeSamples:state];
            });
        }
        return;
    }

    int audioSliceSec = state->audioSliceSec;
    if (nSamples + n > audioSliceSec * WHISPER_SAMPLE_RATE) {
        // next slice
        state->sliceIndex++;
        nSamples = 0;
        int16_t* audioBufferI16 = (int16_t*) malloc(audioSliceSec * WHISPER_SAMPLE_RATE * sizeof(int16_t));
        [state->shortBufferSlices addObject:[NSValue valueWithPointer:audioBufferI16]];
        [state->sliceNSamples addObject:[NSNumber numberWithInt:0]];
    }

    // Append to buffer
    NSLog(@"[RNWhisper] Slice %d has %d samples", state->sliceIndex, nSamples);

    int16_t* audioBufferI16 = (int16_t*) [state->shortBufferSlices[state->sliceIndex] pointerValue];
    for (int i = 0; i < n; i++) {
        audioBufferI16[nSamples + i] = ((short*)inBuffer->mAudioData)[i];
    }

    bool isSpeech = vad(state, audioBufferI16, nSamples, n);
    nSamples += n;
    state->sliceNSamples[state->sliceIndex] = [NSNumber numberWithInt:nSamples];

    AudioQueueEnqueueBuffer(state->queue, inBuffer, 0, NULL);

    if (!isSpeech) return;

    if (!state->isTranscribing) {
        state->isTranscribing = true;
        dispatch_async([state->mSelf getDispatchQueue], ^{
            [state->mSelf fullTranscribeSamples:state];
        });
    }
}

- (void)finishRealtimeTranscribe:(RNWhisperContextRecordState*) state result:(NSDictionary*)result {
    // Save wav if needed
    if (state->audioOutputPath != nil) {
        // TODO: Append in real time so we don't need to keep all slices & also reduce memory usage
        [RNWhisperAudioUtils
            saveWavFile:[RNWhisperAudioUtils concatShortBuffers:state->shortBufferSlices
                            sliceNSamples:state->sliceNSamples]
            audioOutputFile:state->audioOutputPath
        ];
    }
    state->transcribeHandler(state->jobId, @"end", result);
}

- (void)fullTranscribeSamples:(RNWhisperContextRecordState*) state {
    int nSamplesOfIndex = [[state->sliceNSamples objectAtIndex:state->transcribeSliceIndex] intValue];
    state->nSamplesTranscribing = nSamplesOfIndex;
    NSLog(@"[RNWhisper] Transcribing %d samples", state->nSamplesTranscribing);

    int16_t* audioBufferI16 = (int16_t*) [state->shortBufferSlices[state->transcribeSliceIndex] pointerValue];
    float* audioBufferF32 = (float*) malloc(state->nSamplesTranscribing * sizeof(float));
    // convert I16 to F32
    for (int i = 0; i < state->nSamplesTranscribing; i++) {
        audioBufferF32[i] = (float)audioBufferI16[i] / 32768.0f;
    }
    CFTimeInterval timeStart = CACurrentMediaTime();
    struct whisper_full_params params = [state->mSelf getParams:state->options jobId:state->jobId];
    int code = [state->mSelf fullTranscribe:state->jobId params:params audioData:audioBufferF32 audioDataCount:state->nSamplesTranscribing];
    free(audioBufferF32);
    CFTimeInterval timeEnd = CACurrentMediaTime();
    const float timeRecording = (float) state->nSamplesTranscribing / (float) state->dataFormat.mSampleRate;

    NSDictionary* base = @{
        @"code": [NSNumber numberWithInt:code],
        @"processTime": [NSNumber numberWithInt:(timeEnd - timeStart) * 1E3],
        @"recordingTime": [NSNumber numberWithInt:timeRecording * 1E3],
        @"isUseSlices": @(state->isUseSlices),
        @"sliceIndex": @(state->transcribeSliceIndex),
    };

    NSMutableDictionary* result = [base mutableCopy];

    if (code == 0) {
        result[@"data"] = [state->mSelf getTextSegments];
    } else {
        result[@"error"] = [NSString stringWithFormat:@"Transcribe failed with code %d", code];
    }

    nSamplesOfIndex = [[state->sliceNSamples objectAtIndex:state->transcribeSliceIndex] intValue];

    bool isStopped = state->isStoppedByAction || (
        !state->isCapturing &&
        state->nSamplesTranscribing == nSamplesOfIndex &&
        state->sliceIndex == state->transcribeSliceIndex
    );

    if (
      // If no more samples on current slice, move to next slice
      state->nSamplesTranscribing == nSamplesOfIndex &&
      state->transcribeSliceIndex != state->sliceIndex
    ) {
        state->transcribeSliceIndex++;
        state->nSamplesTranscribing = 0;
    }

    bool continueNeeded = !state->isCapturing &&
        state->nSamplesTranscribing != nSamplesOfIndex;

    if (isStopped && !continueNeeded) {
        NSLog(@"[RNWhisper] Transcribe end");
        result[@"isStoppedByAction"] = @(state->isStoppedByAction);
        result[@"isCapturing"] = @(false);

        [state->mSelf finishRealtimeTranscribe:state result:result];
    } else if (code == 0) {
        result[@"isCapturing"] = @(true);
        state->transcribeHandler(state->jobId, @"transcribe", result);
    } else {
        result[@"isCapturing"] = @(true);
        state->transcribeHandler(state->jobId, @"transcribe", result);
    }

    if (continueNeeded) {
        state->isTranscribing = true;
        // Finish transcribing the rest of the samples
        [self fullTranscribeSamples:state];
    }
    state->isTranscribing = false;
}

- (bool)isCapturing {
    return self->recordState.isCapturing;
}

- (bool)isTranscribing {
    return self->recordState.isTranscribing;
}

- (bool)isStoppedByAction {
    return self->recordState.isStoppedByAction;
}

- (OSStatus)transcribeRealtime:(int)jobId
    options:(NSDictionary *)options
    onTranscribe:(void (^)(int, NSString *, NSDictionary *))onTranscribe
{
    self->recordState.transcribeHandler = onTranscribe;
    self->recordState.jobId = jobId;
    [self prepareRealtime:options];

    OSStatus status = AudioQueueNewInput(
        &self->recordState.dataFormat,
        AudioInputCallback,
        &self->recordState,
        NULL,
        kCFRunLoopCommonModes,
        0,
        &self->recordState.queue
    );

    if (status == 0) {
        for (int i = 0; i < NUM_BUFFERS; i++) {
            AudioQueueAllocateBuffer(self->recordState.queue, NUM_BYTES_PER_BUFFER, &self->recordState.buffers[i]);
            AudioQueueEnqueueBuffer(self->recordState.queue, self->recordState.buffers[i], 0, NULL);
        }
        status = AudioQueueStart(self->recordState.queue, NULL);
        if (status == 0) {
            self->recordState.isCapturing = true;
        }
    }
    return status;
}

struct rnwhisper_segments_callback_data {
    void (^onNewSegments)(NSDictionary *);
    int total_n_new;
};

- (void)transcribeFile:(int)jobId
    audioData:(float *)audioData
    audioDataCount:(int)audioDataCount
    options:(NSDictionary *)options
    onProgress:(void (^)(int))onProgress
    onNewSegments:(void (^)(NSDictionary *))onNewSegments
    onEnd:(void (^)(int))onEnd
{
    dispatch_async(dQueue, ^{
        self->recordState.isStoppedByAction = false;
        self->recordState.isTranscribing = true;
        self->recordState.jobId = jobId;

        whisper_full_params params = [self getParams:options jobId:jobId];
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
                    text = [text stringByAppendingString:[NSString stringWithUTF8String:text_cur]];

                    const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                    const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
                    NSDictionary *segment = @{
                        @"text": [NSString stringWithUTF8String:text_cur],
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
                .total_n_new = 0
            };
            params.new_segment_callback_user_data = &user_data;
        }
        int code = [self fullTranscribe:jobId params:params audioData:audioData audioDataCount:audioDataCount];
        self->recordState.jobId = -1;
        self->recordState.isTranscribing = false;
        onEnd(code);
    });
}

- (void)stopAudio {
    AudioQueueStop(self->recordState.queue, true);
    for (int i = 0; i < NUM_BUFFERS; i++) {
        AudioQueueFreeBuffer(self->recordState.queue, self->recordState.buffers[i]);
    }
    AudioQueueDispose(self->recordState.queue, true);
}

- (void)stopTranscribe:(int)jobId {
    rn_whisper_abort_transcribe(jobId);
    if (self->recordState.isRealtime && self->recordState.isCapturing) {
        [self stopAudio];
        if (!self->recordState.isTranscribing) {
            // Handle for VAD case
            self->recordState.transcribeHandler(jobId, @"end", @{});
        }
    }
    self->recordState.isCapturing = false;
    self->recordState.isStoppedByAction = true;
    dispatch_barrier_sync(dQueue, ^{});
}

- (void)stopCurrentTranscribe {
    if (!self->recordState.jobId) {
        return;
    }
    [self stopTranscribe:self->recordState.jobId];
}

- (struct whisper_full_params)getParams:(NSDictionary *)options jobId:(int)jobId {
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
    params.speed_up         = options[@"speedUp"] != nil ? [options[@"speedUp"] boolValue] : false;
    params.translate        = options[@"translate"] != nil ? [options[@"translate"] boolValue] : false;
    params.language         = options[@"language"] != nil ? [options[@"language"] UTF8String] : "auto";
    params.n_threads        = n_threads > 0 ? n_threads : default_n_threads;
    params.offset_ms        = 0;
    params.no_context       = true;
    params.single_segment   = false;

    if (options[@"maxLen"] != nil) {
        params.max_len = [options[@"maxLen"] intValue];
    }
    params.token_timestamps = options[@"tokenTimestamps"] != nil ? [options[@"tokenTimestamps"] boolValue] : false;

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
        params.initial_prompt = [options[@"prompt"] UTF8String];
    }

    // abort handler
    params.encoder_begin_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, void * user_data) {
        bool is_aborted = *(bool*)user_data;
        return !is_aborted;
    };
    params.encoder_begin_callback_user_data = rn_whisper_assign_abort_map(jobId);
    params.abort_callback = [](void * user_data) {
        bool is_aborted = *(bool*)user_data;
        return is_aborted;
    };
    params.abort_callback_user_data = rn_whisper_assign_abort_map(jobId);

    return params;
}

- (int)fullTranscribe:(int)jobId
  params:(struct whisper_full_params)params
  audioData:(float *)audioData
  audioDataCount:(int)audioDataCount
{
    whisper_reset_timings(self->ctx);

    int code = whisper_full(self->ctx, params, audioData, audioDataCount);
    rn_whisper_remove_abort_map(jobId);
    // if (code == 0) {
    //     whisper_print_timings(self->ctx);
    // }
    return code;
}

- (NSMutableDictionary *)getTextSegments {
    NSString *text = @"";
    int n_segments = whisper_full_n_segments(self->ctx);

    NSMutableArray *segments = [[NSMutableArray alloc] init];
    for (int i = 0; i < n_segments; i++) {
        const char * text_cur = whisper_full_get_segment_text(self->ctx, i);
        text = [text stringByAppendingString:[NSString stringWithUTF8String:text_cur]];

        const int64_t t0 = whisper_full_get_segment_t0(self->ctx, i);
        const int64_t t1 = whisper_full_get_segment_t1(self->ctx, i);
        NSDictionary *segment = @{
            @"text": [NSString stringWithUTF8String:text_cur],
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

- (void)invalidate {
    [self stopCurrentTranscribe];
    whisper_free(self->ctx);
    [self freeBufferIfNeeded];
}

@end
