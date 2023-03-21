
#import "RNWhisper.h"
#include <stdlib.h>
#include <string>

@interface WhisperContext : NSObject {
}

@property struct whisper_context * ctx;

@end

@implementation WhisperContext

- (void)invalidate {
    whisper_free(self.ctx);
}

@end

@implementation RNWhisper

NSMutableDictionary *contexts;

RCT_EXPORT_MODULE()

RCT_REMAP_METHOD(initContext,
                 withPath:(NSString *)modelPath
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    if (contexts == nil) {
        contexts = [[NSMutableDictionary alloc] init];
    }

    WhisperContext *context = [[WhisperContext alloc] init];
    context.ctx = whisper_init_from_file([modelPath UTF8String]);

    if (context.ctx == NULL) {
        reject(@"whisper_cpp_error", @"Failed to load the model", nil);
        return;
    }

    int contextId = arc4random_uniform(1000000);
    [contexts setObject:context forKey:[NSNumber numberWithInt:contextId]];

    resolve([NSNumber numberWithInt:contextId]);
}

RCT_REMAP_METHOD(transcribe,
                 withContextId:(int)contextId
                 withWaveFile:(NSString *)waveFilePath
                 withOptions:(NSDictionary *)options
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    WhisperContext *context = contexts[[NSNumber numberWithInt:contextId]];

    if (context == nil) {
        reject(@"whisper_error", @"Context not found", nil);
        return;
    }

    NSURL *url = [NSURL fileURLWithPath:waveFilePath];

    int count = 0;
    float *waveFile = [self decodeWaveFile:url count:&count];

    if (waveFile == nil) {
        reject(@"whisper_error", @"Invalid file", nil);
        return;
    }

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    const int max_threads = options[@"maxThreads"] != nil ?
      [options[@"maxThreads"] intValue] :
      MIN(8, (int)[[NSProcessInfo processInfo] processorCount]) - 2;

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
    params.n_threads        = max_threads;
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
        std::string *prompt = new std::string([options[@"prompt"] UTF8String]);
        rn_whisper_convert_prompt(
            context.ctx,
            params,
            prompt
        );
    }

    whisper_reset_timings(context.ctx);
    int code = whisper_full(context.ctx, params, waveFile, count);
    if (code != 0) {
        NSLog(@"Failed to run the model");
        free(waveFile);
        reject(@"whisper_cpp_error", [NSString stringWithFormat:@"Failed to run the model. Code: %d", code], nil);
        return;
    }

    // whisper_print_timings(context.ctx);
    free(waveFile);

    NSString *result = @"";
    int n_segments = whisper_full_n_segments(context.ctx);

    NSMutableArray *segments = [[NSMutableArray alloc] init];
    for (int i = 0; i < n_segments; i++) {
        const char * text_cur = whisper_full_get_segment_text(context.ctx, i);
        result = [result stringByAppendingString:[NSString stringWithUTF8String:text_cur]];

        const int64_t t0 = whisper_full_get_segment_t0(context.ctx, i);
        const int64_t t1 = whisper_full_get_segment_t1(context.ctx, i);
        NSDictionary *segment = @{
            @"text": [NSString stringWithUTF8String:text_cur],
            @"t0": [NSNumber numberWithLongLong:t0],
            @"t1": [NSNumber numberWithLongLong:t1]
        };
        [segments addObject:segment];
    }
    resolve(@{
        @"result": result,
        @"segments": segments
    });
}

RCT_REMAP_METHOD(releaseContext,
                 withContextId:(int)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    WhisperContext *context = contexts[[NSNumber numberWithInt:contextId]];
    if (context == nil) {
        reject(@"whisper_error", @"Context not found", nil);
        return;
    }
    [context invalidate];
    [contexts removeObjectForKey:[NSNumber numberWithInt:contextId]];
    resolve(nil);
}

RCT_REMAP_METHOD(releaseAllContexts,
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    [self invalidate];
    resolve(nil);
}

- (float *)decodeWaveFile:(NSURL*)fileURL count:(int *)count {
    NSData *fileData = [NSData dataWithContentsOfURL:fileURL];
    if (fileData == nil) {
        return nil;
    }
    NSMutableData *waveData = [[NSMutableData alloc] init];
    [waveData appendData:[fileData subdataWithRange:NSMakeRange(44, [fileData length]-44)]];
    const short *shortArray = (const short *)[waveData bytes];
    int shortCount = (int) ([waveData length] / sizeof(short));
    float *floatArray = (float *) malloc(shortCount * sizeof(float));
    for (NSInteger i = 0; i < shortCount; i++) {
        float floatValue = ((float)shortArray[i]) / 32767.0;
        floatValue = MAX(floatValue, -1.0);
        floatValue = MIN(floatValue, 1.0);
        floatArray[i] = floatValue;
    }
    *count = shortCount;
    return floatArray;
}

- (void)invalidate {
    if (contexts == nil) {
        return;
    }

    for (NSNumber *contextId in contexts) {
        WhisperContext *context = contexts[contextId];
        [context invalidate];
    }

    [contexts removeAllObjects];
    contexts = nil;
}

@end
