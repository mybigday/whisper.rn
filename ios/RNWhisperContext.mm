#import "RNWhisperContext.h"
#import <Metal/Metal.h>
#import <React/RCTLog.h>
#include <vector>
#include <unicode/ustring.h>

#define NUM_BYTES_PER_BUFFER 16 * 1024

@implementation RNWhisperContext

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

    // TODO: Figure out why it leads to re-init crash
    cparams.dtw_token_timestamps = false;

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

- (void)prepareRealtime:(int)jobId options:(NSDictionary *)options {
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

    self->recordState.isRealtime = true;
    self->recordState.isTranscribing = false;
    self->recordState.isCapturing = false;
    self->recordState.isStoppedByAction = false;

    self->recordState.sliceIndex = 0;
    self->recordState.transcribeSliceIndex = 0;
    self->recordState.nSamplesTranscribing = 0;

    self->recordState.sliceNSamples.clear();
    self->recordState.sliceNSamples.push_back(0);

    self->recordState.job = rnwhisper::job_new(jobId, [self createParams:options jobId:jobId]);
    self->recordState.job->set_realtime_params(
        {
            .use_vad = options[@"useVad"] != nil ? [options[@"useVad"] boolValue] : false,
            .vad_ms = options[@"vadMs"] != nil ? [options[@"vadMs"] intValue] : 2000,
            .vad_thold = options[@"vadThold"] != nil ? [options[@"vadThold"] floatValue] : 0.6f,
            .freq_thold = options[@"vadFreqThold"] != nil ? [options[@"vadFreqThold"] floatValue] : 100.0f
        },
        options[@"realtimeAudioSec"] != nil ? [options[@"realtimeAudioSec"] intValue] : 0,
        options[@"realtimeAudioSliceSec"] != nil ? [options[@"realtimeAudioSliceSec"] intValue] : 0,
        options[@"realtimeAudioMinSec"] != nil ? [options[@"realtimeAudioMinSec"] floatValue] : 0,
        options[@"audioOutputPath"] != nil ? [options[@"audioOutputPath"] UTF8String] : nullptr
    );
    self->recordState.isUseSlices = self->recordState.job->audio_slice_sec < self->recordState.job->audio_sec;

    self->recordState.currentVolumeLevel = -1;
    self->recordState.mSelf = self;
}

bool vad(RNWhisperContextRecordState *state, int sliceIndex, int nSamples, int n)
{
    int currentVolumeLevel;
    if (state->isTranscribing) return true;
    return state->job->vad_simple(sliceIndex, nSamples, n);
}

float calculateRMS(AudioQueueBufferRef buffer) {
    short *data = (short *)buffer->mAudioData;
    int frameCount = buffer->mAudioDataByteSize / sizeof(short);
    double sum = 0.0;
    for (int i = 0; i < frameCount; i++) {
        float sample = data[i] / 32768.0f;
        sum += sample * sample;
    }
    return sqrt(sum / frameCount);
}


void AudioInputCallback(void * inUserData,
    AudioQueueRef inAQ,
    AudioQueueBufferRef inBuffer,
    const AudioTimeStamp * inStartTime,
    UInt32 inNumberPacketDescriptions,
    const AudioStreamPacketDescription * inPacketDescs)
{
    RNWhisperContextRecordState *state = (RNWhisperContextRecordState *)inUserData;
    NSLog(@"[custom-RNWhisper] AudioInputCallback");

    if (!state->isCapturing || state->isPaused) {
        if (!state->isTranscribing) {
            [state->mSelf finishRealtimeTranscribe:state result:@{}];
            return;
        }
        AudioQueueEnqueueBuffer(state->queue, inBuffer, 0, NULL);
        return;
    }
    // Calculate RMS
    float rms = calculateRMS(inBuffer);

    // Determine volume level based on RMS
    int volumeLevel = 0;
    if (rms < 0.01) {
        volumeLevel = 0; // Very quiet or silent
    } else if (rms < 0.05) {
        volumeLevel = 1; // Quiet
    } else if (rms < 0.1) {
        volumeLevel = 2; // Moderate
    } else if (rms < 0.15) {
        volumeLevel = 3; // Moderately loud
    } else if (rms < 0.2) {
        volumeLevel = 4; // Loud
    } else if (rms < 0.3) {
        volumeLevel = 5; // Very loud
    } else {
        volumeLevel = 6; // Extremely loud
    }

    if (volumeLevel != state->currentVolumeLevel) {
        state->currentVolumeLevel = volumeLevel;

        // Prepare result dictionary
        NSDictionary *result = @{
            @"volume": @(volumeLevel)
        };

        // Call transcribeHandler with eventType 'volumeChange'
        state->transcribeHandler(state->job->job_id, @"volumeChange", result);
    }

    int totalNSamples = 0;
    for (int i = 0; i < state->sliceNSamples.size(); i++) {
        totalNSamples += state->sliceNSamples[i];
    }

    const int n = inBuffer->mAudioDataByteSize / 2;

    int nSamples = state->sliceNSamples[state->sliceIndex];

    if (totalNSamples + n > state->job->audio_sec * WHISPER_SAMPLE_RATE) {
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
            bool isSamplesEnough = nSamples / WHISPER_SAMPLE_RATE >= state->job->audio_min_sec;
            if (!isSamplesEnough || !vad(state, state->sliceIndex, nSamples, 0)) {
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

    if (nSamples + n > state->job->audio_slice_sec * WHISPER_SAMPLE_RATE) {
        // next slice
        state->sliceIndex++;
        nSamples = 0;
        state->sliceNSamples.push_back(0);
    }

    NSLog(@"[custom-RNWhisper] Slice %d has %d samples, put %d samples", state->sliceIndex, nSamples, n);

    state->job->put_pcm_data((short*) inBuffer->mAudioData, state->sliceIndex, nSamples, n);

    bool isSpeech = vad(state, state->sliceIndex, nSamples, n);
    nSamples += n;
    state->sliceNSamples[state->sliceIndex] = nSamples;

    AudioQueueEnqueueBuffer(state->queue, inBuffer, 0, NULL);

    bool isSamplesEnough = nSamples / WHISPER_SAMPLE_RATE >= state->job->audio_min_sec;
    if (!isSamplesEnough || !isSpeech) return;

    if (!state->isTranscribing) {
        state->isTranscribing = true;
        dispatch_async([state->mSelf getDispatchQueue], ^{
            [state->mSelf fullTranscribeSamples:state];
        });
    }
}

- (void)finishRealtimeTranscribe:(RNWhisperContextRecordState*) state result:(NSDictionary*)result {
    // Save wav if needed
    NSLog(@"[custom-RNWhisper] audio output path: %s", state->job->audio_output_path);
    if (state->job->audio_output_path != nullptr) {
        NSLog(@"[custom-RNWhisper] audio output path not null");
        // TODO: Append in real time so we don't need to keep all slices & also reduce memory usage
        rnaudioutils::save_wav_file(
            rnaudioutils::concat_short_buffers(state->job->pcm_slices, state->sliceNSamples),
            state->job->audio_output_path
        );
        NSLog(@"[custom-RNWhisper] save_wav_file excuted");
    }
    state->transcribeHandler(state->job->job_id, @"end", result);
    rnwhisper::job_remove(state->job->job_id);
}


- (void)pauseAudio {
    if (self->recordState.queue != NULL && self->recordState.isCapturing && !self->recordState.isPaused) {
        NSLog(@"[RNWhisper] Pausing audio queue");
        AudioQueuePause(self->recordState.queue);
        self->recordState.isPaused = true;
    }
}

- (void)resumeAudio {
    if (self->recordState.queue != NULL && self->recordState.isCapturing && self->recordState.isPaused) {
        NSLog(@"[RNWhisper] Resuming audio queue");
        AudioQueueStart(self->recordState.queue, NULL);
        self->recordState.isPaused = false;
    }
}

- (void)fullTranscribeSamples:(RNWhisperContextRecordState*) state {
    int nSamplesOfIndex = state->sliceNSamples[state->transcribeSliceIndex];
    state->nSamplesTranscribing = nSamplesOfIndex;
    NSLog(@"[RNWhisper] Transcribing %d samples", state->nSamplesTranscribing);

    float* pcmf32 = state->job->pcm_slice_to_f32(state->transcribeSliceIndex, state->nSamplesTranscribing);

    CFTimeInterval timeStart = CACurrentMediaTime();
    int code = [state->mSelf fullTranscribe:state->job audioData:pcmf32 audioDataCount:state->nSamplesTranscribing];
    free(pcmf32);
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

    nSamplesOfIndex = state->sliceNSamples[state->transcribeSliceIndex];

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
        state->transcribeHandler(state->job->job_id, @"transcribe", result);
    } else {
        result[@"isCapturing"] = @(true);
        state->transcribeHandler(state->job->job_id, @"transcribe", result);
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

    NSLog(@"[custom-RNWhisper] Start RealTime transcribe");
    self->recordState.transcribeHandler = onTranscribe;

    [self prepareRealtime:jobId options:options];

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
    bool tdrzEnable;
    // Add fields for partial data accumulation:
    __strong NSMutableData *tempData;
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
        self->recordState.isStoppedByAction = false;
        self->recordState.isTranscribing = true;

        whisper_full_params params = [self createParams:options jobId:jobId];

        if (options[@"onProgress"] && [options[@"onProgress"] boolValue]) {
            params.progress_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, int progress, void * user_data) {
                void (^onProgress)(int) = (__bridge void (^)(int))user_data;
                onProgress(progress);
            };
            params.progress_callback_user_data = (__bridge void *)(onProgress);
        }
        if (options[@"onNewSegments"] && [options[@"onNewSegments"] boolValue]) {
            // Allocate and initialize callback data on the heap
            struct rnwhisper_segments_callback_data *user_data = (struct rnwhisper_segments_callback_data *)malloc(sizeof(struct rnwhisper_segments_callback_data));
            // Copy the block to ensure it is heap-allocated and remains valid
            user_data->onNewSegments = [onNewSegments copy];
            user_data->tdrzEnable = options[@"tdrzEnable"] && [options[@"tdrzEnable"] boolValue];
            user_data->total_n_new = 0;
            user_data->tempData = [NSMutableData data];

            params.new_segment_callback = [](struct whisper_context * ctx, struct whisper_state * /*state*/, int n_new, void * ud) {
                struct rnwhisper_segments_callback_data *data = (struct rnwhisper_segments_callback_data *)ud;
                data->total_n_new += n_new;

                NSString *combinedText = @"";
                NSMutableArray *segments = [[NSMutableArray alloc] init];

                NSMutableData *tempData = data->tempData;

                for (int i = data->total_n_new - n_new; i < data->total_n_new; i++) {
                    const char *text_cur = whisper_full_get_segment_text(ctx, i);
                    if (text_cur == NULL) {
                        NSLog(@"[custom-RNWhisper] text_cur is NULL for segment %d", i);
                        continue;
                    }

                    size_t text_cur_length = strlen(text_cur);
                    NSLog(@"[custom-RNWhisper] text_cur for segment %d (length %zu): \"%s\"", i, text_cur_length, text_cur);

                    [tempData appendBytes:text_cur length:text_cur_length];
                    [tempData appendBytes:"" length:1];
                    char *buffer = (char *)[tempData mutableBytes];

                    if (is_valid_utf8(buffer)) {
                        NSString *ns_text = [NSString stringWithUTF8String:buffer];
                        if (!ns_text) {
                            NSLog(@"[custom-RNWhisper] Unable to convert buffer to NSString for segment %d", i);
                            [tempData setLength:0];
                            continue;
                        }

                        [tempData setLength:0];

                        NSMutableString *mutable_ns_text = [NSMutableString stringWithString:ns_text];
                        if (data->tdrzEnable && whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                            [mutable_ns_text appendString:@" [SPEAKER_TURN]"];
                        }

                        combinedText = [combinedText stringByAppendingString:mutable_ns_text];

                        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

                        NSDictionary *segment = @{
                            @"text": [NSString stringWithString:mutable_ns_text],
                            @"t0": @(t0),
                            @"t1": @(t1)
                        };
                        [segments addObject:segment];

                        NSLog(@"[custom-RNWhisper] Final ns_text for segment %d: \"%@\"", i, ns_text);
                    } else {
                        [tempData setLength:[tempData length] - 1];
                        NSLog(@"[custom-RNWhisper] Current buffer not valid UTF-8 yet, waiting for next segment.");
                    }
                }

                NSDictionary *result = @{
                    @"nNew": @(n_new),
                    @"totalNNew": @(data->total_n_new),
                    @"result": combinedText,
                    @"segments": segments
                };

                void (^onNewSegmentsBlock)(NSDictionary *) = data->onNewSegments;
                onNewSegmentsBlock(result);
            };

            params.new_segment_callback_user_data = user_data;
        }


        rnwhisper::job* job = rnwhisper::job_new(jobId, params);
        self->recordState.job = job;
        int code = [self fullTranscribe:job audioData:audioData audioDataCount:audioDataCount];
        rnwhisper::job_remove(jobId);
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
    NSLog(@"[custom-RNWhisper] Stop transcribe");
    if (self->recordState.job) self->recordState.job->abort();
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
    if (self->recordState.job == nullptr) return;
    [self stopTranscribe:self->recordState.job->job_id];
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
    params.split_on_word = false;
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

// Helper function to check if a given C-string is valid UTF-8
static BOOL is_valid_utf8(const char *str) {
    if (!str) return NO;
    UErrorCode error = U_ZERO_ERROR;
    // Attempt conversion to UTF-16 using ICU to validate UTF-8
    u_strFromUTF8(NULL, 0, NULL, str, -1, &error);
    return (error != U_INVALID_CHAR_FOUND);
}

- (NSMutableDictionary *)getTextSegments {
    NSString *text = @"";
    int n_segments = whisper_full_n_segments(self->ctx);

    NSMutableArray *segments = [[NSMutableArray alloc] init];
    NSLog(@"[custom-RNWhisper] getTextSegments");

    // Instead of an NSString temp buffer, let's store raw bytes.
    // We'll convert to NSString only after validating UTF-8.
    NSMutableData *tempData = [NSMutableData data];

    for (int i = 0; i < n_segments; i++) {
        const char *text_cur = whisper_full_get_segment_text(self->ctx, i);

        if (text_cur == NULL) {
            NSLog(@"[custom-RNWhisper] text_cur is NULL for segment %d", i);
            continue;
        }

        size_t text_cur_length = strlen(text_cur);
        NSLog(@"[custom-RNWhisper] text_cur for segment %d (length %zu): \"%s\"", i, text_cur_length, text_cur);

        // Append this segment's raw bytes to the temporary buffer.
        [tempData appendBytes:text_cur length:text_cur_length];

        // After appending, check if the entire tempData forms a valid UTF-8 string.
        // We must first null-terminate for safe checking.
        [tempData appendBytes:"" length:1];
        char *buffer = (char *)[tempData mutableBytes];

        if (is_valid_utf8(buffer)) {
            // If valid, convert the entire buffer to NSString
            NSString *ns_text = [NSString stringWithUTF8String:buffer];
            if (!ns_text) {
                // If still can't form NSString, fallback to a safe encoding or skip
                NSLog(@"[custom-RNWhisper] Still unable to form a valid NSString from buffer for segment %d", i);
                [tempData setLength:0];
                continue;
            }

            // Clear the tempData now that we've consumed it
            [tempData setLength:0];

            // From here, handle speaker turn if needed
            NSMutableString *mutable_ns_text = [NSMutableString stringWithString:ns_text];

            if (self->recordState.options[@"tdrzEnable"] &&
                [self->recordState.options[@"tdrzEnable"] boolValue] &&
                whisper_full_get_segment_speaker_turn_next(self->ctx, i)) {
                [mutable_ns_text appendString:@" [SPEAKER_TURN]"];
            }

            // Append the text to the overall text
            text = [text stringByAppendingString:mutable_ns_text];

            const int64_t t0 = whisper_full_get_segment_t0(self->ctx, i);
            const int64_t t1 = whisper_full_get_segment_t1(self->ctx, i);

            NSDictionary *segment = @{
                @"text": [NSString stringWithString:mutable_ns_text],
                @"t0": [NSNumber numberWithLongLong:t0],
                @"t1": [NSNumber numberWithLongLong:t1]
            };
            [segments addObject:segment];

            NSLog(@"[custom-RNWhisper] Final ns_text for segment %d: \"%@\"", i, ns_text);
        } else {
            // If not valid yet, remove the null terminator we added and wait for more segments.
            // We'll try again at the next iteration once we add more data.
            [tempData setLength:[tempData length] - 1];
            NSLog(@"[custom-RNWhisper] Current buffer not valid UTF-8 yet, waiting for next segment.");
        }
    }

    // If we exit the loop and tempData still contains bytes, it means it never formed a valid UTF-8 sequence.
    // Handle that gracefully if needed, or just discard.

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
