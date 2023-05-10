#import "RNWhisper.h"
#import "RNWhisperContext.h"
#include <stdlib.h>
#include <string>

@implementation RNWhisper

NSMutableDictionary *contexts;

RCT_EXPORT_MODULE()

- (NSDictionary *)constantsToExport
{
  return @{
#if WHISPER_USE_COREML
    @"WHISPER_USE_COREML": @YES,
#endif
#if WHISPER_COREML_ALLOW_FALLBACK
    @"WHISPER_COREML_ALLOW_FALLBACK": @YES,
#endif
  };
}

RCT_REMAP_METHOD(initContext,
                 withPath:(NSString *)modelPath
                 withBundleResource:(BOOL)isBundleAsset
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    if (contexts == nil) {
        contexts = [[NSMutableDictionary alloc] init];
    }

    NSString *path = modelPath;
    if (isBundleAsset) {
        path = [[NSBundle mainBundle] pathForResource:modelPath ofType:nil];
    }

    RNWhisperContext *context = [RNWhisperContext initWithModelPath:path];
    if ([context getContext] == NULL) {
        reject(@"whisper_cpp_error", @"Failed to load the model", nil);
        return;
    }

    int contextId = arc4random_uniform(1000000);
    [contexts setObject:context forKey:[NSNumber numberWithInt:contextId]];

    resolve([NSNumber numberWithInt:contextId]);
}

RCT_REMAP_METHOD(transcribeFile,
                 withContextId:(int)contextId
                 withJobId:(int)jobId
                 withWaveFile:(NSString *)waveFilePath
                 withOptions:(NSDictionary *)options
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNWhisperContext *context = contexts[[NSNumber numberWithInt:contextId]];

    if (context == nil) {
        reject(@"whisper_error", @"Context not found", nil);
        return;
    }
    if ([context isCapturing]) {
        reject(@"whisper_error", @"The context is in realtime transcribe mode", nil);
        return;
    }
    if ([context isTranscribing]) {
        reject(@"whisper_error", @"Context is already transcribing", nil);
        return;
    }

    NSURL *url = [NSURL fileURLWithPath:waveFilePath];

    int count = 0;
    float *waveFile = [self decodeWaveFile:url count:&count];
    if (waveFile == nil) {
        reject(@"whisper_error", @"Invalid file", nil);
        return;
    }
    int code = [context transcribeFile:jobId audioData:waveFile audioDataCount:count options:options];
    if (code != 0) {
        free(waveFile);
        reject(@"whisper_cpp_error", [NSString stringWithFormat:@"Failed to transcribe the file. Code: %d", code], nil);
        return;
    }
    free(waveFile);
    resolve([context getTextSegments]);
}

- (NSArray *)supportedEvents {
  return@[
    @"@RNWhisper_onRealtimeTranscribe",
    @"@RNWhisper_onRealtimeTranscribeEnd",
  ];
}

RCT_REMAP_METHOD(startRealtimeTranscribe,
                 withContextId:(int)contextId
                 withJobId:(int)jobId
                 withOptions:(NSDictionary *)options
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNWhisperContext *context = contexts[[NSNumber numberWithInt:contextId]];

    if (context == nil) {
        reject(@"whisper_error", @"Context not found", nil);
        return;
    }
    if ([context isCapturing]) {
        reject(@"whisper_error", @"The context is already capturing", nil);
        return;
    }

    OSStatus status = [context transcribeRealtime:jobId
        options:options
        onTranscribe:^(int _jobId, NSString *type, NSDictionary *payload) {
            NSString *eventName = nil;
            if ([type isEqual:@"transcribe"]) {
                eventName = @"@RNWhisper_onRealtimeTranscribe";
            } else if ([type isEqual:@"end"]) {
                eventName = @"@RNWhisper_onRealtimeTranscribeEnd";
            }
            if (eventName == nil) {
                return;
            }
            [self sendEventWithName:eventName
                body:@{
                    @"contextId": [NSNumber numberWithInt:contextId],
                    @"jobId": [NSNumber numberWithInt:jobId],
                    @"payload": payload
                }
            ];
        }
    ];
    if (status == 0) {
        resolve(nil);
        return;
    }
    reject(@"whisper_error", [NSString stringWithFormat:@"Failed to start realtime transcribe. Status: %d", status], nil);
}
RCT_REMAP_METHOD(abortTranscribe,
                 withContextId:(int)contextId
                 withJobId:(int)jobId)
{
    RNWhisperContext *context = contexts[[NSNumber numberWithInt:contextId]];
    [context stopTranscribe:jobId];
}

RCT_REMAP_METHOD(releaseContext,
                 withContextId:(int)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNWhisperContext *context = contexts[[NSNumber numberWithInt:contextId]];
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
    rn_whisper_abort_all_transcribe();

    if (contexts == nil) {
        return;
    }

    for (NSNumber *contextId in contexts) {
        RNWhisperContext *context = contexts[contextId];
        [context invalidate];
    }

    [contexts removeAllObjects];
    contexts = nil;
}

@end
