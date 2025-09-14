#import "RNWhisper.h"
#import "RNWhisperContext.h"
#import "RNWhisperVadContext.h"
#import "RNWhisperDownloader.h"
#import "RNWhisperAudioUtils.h"
#import "RNWhisperAudioSessionUtils.h"
#import "RNWhisperJSI.h"
#include <stdlib.h>
#include <string>

#ifdef RCT_NEW_ARCH_ENABLED
#import <RNWhisperSpec/RNWhisperSpec.h>
#endif

@implementation RNWhisper

NSMutableDictionary *contexts;
NSMutableDictionary *vadContexts;

RCT_EXPORT_MODULE()

+ (BOOL)requiresMainQueueSetup
{
  return NO;
}

RCT_EXPORT_METHOD(toggleNativeLog:(BOOL)enabled) {
    void (^onEmitLog)(NSString *level, NSString *text) = nil;
    if (enabled) {
        onEmitLog = ^(NSString *level, NSString *text) {
            [self sendEventWithName:@"@RNWhisper_onNativeLog" body:@{ @"level": level, @"text": text }];
        };
    }
    [RNWhisperContext toggleNativeLog:enabled onEmitLog:onEmitLog];
}

- (NSDictionary *)constantsToExport
{
  return @{
#if WHISPER_USE_COREML
    @"useCoreML": @YES,
#else
    @"useCoreML": @NO,
#endif
#if WHISPER_COREML_ALLOW_FALLBACK
    @"coreMLAllowFallback": @YES,
#else
    @"coreMLAllowFallback": @NO,
#endif
  };
}

RCT_REMAP_METHOD(initContext,
                 withOptions:(NSDictionary *)modelOptions
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    if (contexts == nil) {
        contexts = [[NSMutableDictionary alloc] init];
    }

    NSString *modelPath = [modelOptions objectForKey:@"filePath"];
    BOOL isBundleAsset = [[modelOptions objectForKey:@"isBundleAsset"] boolValue];
    BOOL useGpu = [[modelOptions objectForKey:@"useGpu"] boolValue];
    BOOL useCoreMLIos = [[modelOptions objectForKey:@"useCoreMLIos"] boolValue];
    BOOL useFlashAttn = [[modelOptions objectForKey:@"useFlashAttn"] boolValue];

    // For support debug assets in development mode
    BOOL downloadCoreMLAssets = [[modelOptions objectForKey:@"downloadCoreMLAssets"] boolValue];
    if (downloadCoreMLAssets) {
        NSArray *coreMLAssets = [modelOptions objectForKey:@"coreMLAssets"];
        // Download coreMLAssets ([{ uri, filepath }])
        for (NSDictionary *coreMLAsset in coreMLAssets) {
            NSString *path = coreMLAsset[@"uri"];
            if ([path hasPrefix:@"http://"] || [path hasPrefix:@"https://"]) {
                [RNWhisperDownloader downloadFile:path toFile:coreMLAsset[@"filepath"]];
            }
        }
    }

    NSString *path = modelPath;
    if ([path hasPrefix:@"http://"] || [path hasPrefix:@"https://"]) {
        path = [RNWhisperDownloader downloadFile:path toFile:nil];
    }
    if (isBundleAsset) {
        path = [[NSBundle mainBundle] pathForResource:modelPath ofType:nil];
    }

    int contextId = arc4random_uniform(1000000);

    RNWhisperContext *context = [RNWhisperContext
        initWithModelPath:path
        contextId:contextId
        noCoreML:!useCoreMLIos
        noMetal:!useGpu
        useFlashAttn:useFlashAttn
    ];
    if ([context getContext] == NULL) {
        reject(@"whisper_cpp_error", @"Failed to load the model", nil);
        return;
    }

    [contexts setObject:context forKey:[NSNumber numberWithInt:contextId]];
    // Also add to unified context management - store raw context pointer like Android
    rnwhisper_jsi::addContext(contextId, reinterpret_cast<long>([context getContext]));

    resolve(@{
        @"contextId": @(contextId),
        @"gpu": @([context isMetalEnabled]),
        @"reasonNoGPU": [context reasonNoMetal],
    });
}

- (NSArray *)supportedEvents {
  return@[
    @"@RNWhisper_onTranscribeProgress",
    @"@RNWhisper_onTranscribeNewSegments",
    @"@RNWhisper_onRealtimeTranscribe",
    @"@RNWhisper_onRealtimeTranscribeEnd",
    @"@RNWhisper_onNativeLog",
  ];
}

- (void)transcribeData:(RNWhisperContext *)context
    withContextId:(int)contextId
    withJobId:(int)jobId
    withData:(float *)data
    withDataCount:(int)count
    withOptions:(NSDictionary *)options
    withResolver:(RCTPromiseResolveBlock)resolve
    withRejecter:(RCTPromiseRejectBlock)reject
{
    [context transcribeData:jobId
        audioData:data
        audioDataCount:count
        options:options
        onProgress: ^(int progress) {
            rnwhisper::job* job = rnwhisper::job_get(jobId);
            if (job && job->is_aborted()) return;

            dispatch_async(dispatch_get_main_queue(), ^{
                [self sendEventWithName:@"@RNWhisper_onTranscribeProgress"
                    body:@{
                        @"contextId": [NSNumber numberWithInt:contextId],
                        @"jobId": [NSNumber numberWithInt:jobId],
                        @"progress": [NSNumber numberWithInt:progress]
                    }
                ];
            });
        }
        onNewSegments: ^(NSDictionary *result) {
            rnwhisper::job* job = rnwhisper::job_get(jobId);
            if (job && job->is_aborted()) return;

            dispatch_async(dispatch_get_main_queue(), ^{
                [self sendEventWithName:@"@RNWhisper_onTranscribeNewSegments"
                    body:@{
                        @"contextId": [NSNumber numberWithInt:contextId],
                        @"jobId": [NSNumber numberWithInt:jobId],
                        @"result": result
                    }
                ];
            });
        }
        onEnd: ^(int code) {
            if (code != 0 && code != 999) {
                reject(@"whisper_cpp_error", [NSString stringWithFormat:@"Failed to transcribe the file. Code: %d", code], nil);
                return;
            }
            NSMutableDictionary *result = [context getTextSegments];
            result[@"isAborted"] = @([context isStoppedByAction]);
            resolve(result);
        }
    ];
}

RCT_REMAP_METHOD(transcribeFile,
                 withContextId:(int)contextId
                 withJobId:(int)jobId
                 withWaveFile:(NSString *)waveFilePathOrDataBase64
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

    float *data = nil;
    int count = 0;
    if ([waveFilePathOrDataBase64 hasPrefix:@"http://"] || [waveFilePathOrDataBase64 hasPrefix:@"https://"]) {
        NSString *path = [RNWhisperDownloader downloadFile:waveFilePathOrDataBase64 toFile:nil];
        data = [RNWhisperAudioUtils decodeWaveFile:path count:&count];
    } else if ([waveFilePathOrDataBase64 hasPrefix:@"data:audio/wav;base64,"]) {
        NSData *waveData = [[NSData alloc] initWithBase64EncodedString:[waveFilePathOrDataBase64 substringFromIndex:22] options:0];
        data = [RNWhisperAudioUtils decodeWaveData:waveData count:&count cutHeader:YES];
    } else {
        data = [RNWhisperAudioUtils decodeWaveFile:waveFilePathOrDataBase64 count:&count];
    }
    if (data == nil) {
        reject(@"whisper_error", @"Invalid file", nil);
        return;
    }

    [self transcribeData:context
        withContextId:contextId
        withJobId:jobId
        withData:data
        withDataCount:count
        withOptions:options
        withResolver:resolve
        withRejecter:reject
    ];
}

RCT_REMAP_METHOD(transcribeData,
                 withContextId:(int)contextId
                 withJobId:(int)jobId
                 withData:(NSString *)dataBase64 // pcm data base64 encoded
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

  NSData *pcmData = [[NSData alloc] initWithBase64EncodedString:dataBase64 options:0];
  int count = 0;
  float *data = [RNWhisperAudioUtils decodeWaveData:pcmData count:&count cutHeader:NO];

  if (data == nil) {
      reject(@"whisper_error", @"Invalid data", nil);
      return;
  }

  [self transcribeData:context
      withContextId:contextId
      withJobId:jobId
      withData:data
      withDataCount:count
      withOptions:options
      withResolver:resolve
      withRejecter:reject
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
                 withJobId:(int)jobId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNWhisperContext *context = contexts[[NSNumber numberWithInt:contextId]];
    if (context == nil) {
        reject(@"whisper_error", @"Context not found", nil);
        return;
    }
    [context stopTranscribe:jobId];
    resolve(nil);
}

RCT_REMAP_METHOD(bench,
                 withContextId:(int)contextId
                 withMaxThreads:(int)maxThreads
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNWhisperContext *context = contexts[[NSNumber numberWithInt:contextId]];
    if (context == nil) {
        reject(@"whisper_error", @"Context not found", nil);
        return;
    }
    if ([context isTranscribing]) {
        reject(@"whisper_error", @"The context is transcribing", nil);
        return;
    }
    NSString *result = [context bench:maxThreads];
    resolve(result);
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
    rnwhisper_jsi::removeContext(contextId);
    [contexts removeObjectForKey:[NSNumber numberWithInt:contextId]];
    [context invalidate];
    resolve(nil);
}

RCT_REMAP_METHOD(releaseAllContexts,
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    [self releaseAllContexts];
    resolve(nil);
}

// MARK: - AudioSessionUtils

RCT_EXPORT_METHOD(getAudioSessionCurrentCategory:(RCTPromiseResolveBlock)resolve
                  withRejecter:(RCTPromiseRejectBlock)reject)
{
    NSString *category = [RNWhisperAudioSessionUtils getCurrentCategory];
    NSArray *options = [RNWhisperAudioSessionUtils getCurrentOptions];
    resolve(@{
        @"category": category,
        @"options": options
    });
}

RCT_EXPORT_METHOD(getAudioSessionCurrentMode:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    NSString *mode = [RNWhisperAudioSessionUtils getCurrentMode];
    resolve(mode);
}

RCT_REMAP_METHOD(setAudioSessionCategory,
                 withCategory:(NSString *)category
                 withOptions:(NSArray *)options
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    NSError *error = nil;
    [RNWhisperAudioSessionUtils setCategory:category options:options error:&error];
    if (error != nil) {
        reject(@"whisper_error", [NSString stringWithFormat:@"Failed to set category. Error: %@", error], nil);
        return;
    }
    resolve(nil);
}

RCT_REMAP_METHOD(setAudioSessionMode,
                 withMode:(NSString *)mode
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    NSError *error = nil;
    [RNWhisperAudioSessionUtils setMode:mode error:&error];
    if (error != nil) {
        reject(@"whisper_error", [NSString stringWithFormat:@"Failed to set mode. Error: %@", error], nil);
        return;
    }
    resolve(nil);
}

RCT_REMAP_METHOD(setAudioSessionActive,
                 withActive:(BOOL)active
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    NSError *error = nil;
    [RNWhisperAudioSessionUtils setActive:active error:&error];
    if (error != nil) {
        reject(@"whisper_error", [NSString stringWithFormat:@"Failed to set active. Error: %@", error], nil);
        return;
    }
    resolve(nil);
}

RCT_REMAP_METHOD(initVadContext,
                 withVadOptions:(NSDictionary *)vadOptions
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    if (vadContexts == nil) {
        vadContexts = [[NSMutableDictionary alloc] init];
    }

    NSString *modelPath = [vadOptions objectForKey:@"filePath"];
    BOOL isBundleAsset = [[vadOptions objectForKey:@"isBundleAsset"] boolValue];
    BOOL useGpu = [[vadOptions objectForKey:@"useGpu"] boolValue];
    NSNumber *nThreads = [vadOptions objectForKey:@"nThreads"];

    NSString *path = modelPath;
    if ([path hasPrefix:@"http://"] || [path hasPrefix:@"https://"]) {
        path = [RNWhisperDownloader downloadFile:path toFile:nil];
    }
    if (isBundleAsset) {
        path = [[NSBundle mainBundle] pathForResource:modelPath ofType:nil];
    }

    int contextId = arc4random_uniform(1000000);

    RNWhisperVadContext *vadContext = [RNWhisperVadContext
        initWithModelPath:path
        contextId:contextId
        noMetal:!useGpu
        nThreads:nThreads
    ];
    if ([vadContext getVadContext] == NULL) {
        reject(@"whisper_vad_error", @"Failed to load the VAD model", nil);
        return;
    }

    [vadContexts setObject:vadContext forKey:[NSNumber numberWithInt:contextId]];
    // Also add to unified context management - store raw VAD context pointer like Android
    rnwhisper_jsi::addVadContext(contextId, reinterpret_cast<long>([vadContext getVadContext]));

    resolve(@{
        @"contextId": @(contextId),
        @"gpu": @([vadContext isMetalEnabled]),
        @"reasonNoGPU": [vadContext reasonNoMetal],
    });
}

RCT_REMAP_METHOD(vadDetectSpeech,
                 withContextId:(int)contextId
                 withAudioData:(NSString *)audioDataBase64
                 withOptions:(NSDictionary *)options
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNWhisperVadContext *vadContext = vadContexts[[NSNumber numberWithInt:contextId]];

    if (vadContext == nil) {
        reject(@"whisper_vad_error", @"VAD context not found", nil);
        return;
    }

    // Decode base64 audio data
    NSData *pcmData = [[NSData alloc] initWithBase64EncodedString:audioDataBase64 options:0];
    if (pcmData == nil) {
        reject(@"whisper_vad_error", @"Invalid audio data", nil);
        return;
    }

    int count = 0;
    float *data = [RNWhisperAudioUtils decodeWaveData:pcmData count:&count cutHeader:NO];

    NSArray *segments = [vadContext detectSpeech:data samplesCount:count options:options];
    resolve(segments);
}

RCT_REMAP_METHOD(vadDetectSpeechFile,
                 withVadContextId:(int)contextId
                 withFilePath:(NSString *)filePath
                 withOptions:(NSDictionary *)options
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNWhisperVadContext *vadContext = vadContexts[[NSNumber numberWithInt:contextId]];

    if (vadContext == nil) {
        reject(@"whisper_vad_error", @"VAD context not found", nil);
        return;
    }

    // Handle different input types like transcribeFile does
    float *data = nil;
    int count = 0;
    if ([filePath hasPrefix:@"http://"] || [filePath hasPrefix:@"https://"]) {
        NSString *path = [RNWhisperDownloader downloadFile:filePath toFile:nil];
        data = [RNWhisperAudioUtils decodeWaveFile:path count:&count];
    } else if ([filePath hasPrefix:@"data:audio/wav;base64,"]) {
        NSData *waveData = [[NSData alloc] initWithBase64EncodedString:[filePath substringFromIndex:22] options:0];
        data = [RNWhisperAudioUtils decodeWaveData:waveData count:&count cutHeader:YES];
    } else {
        data = [RNWhisperAudioUtils decodeWaveFile:filePath count:&count];
    }

    if (data == nil) {
        reject(@"whisper_vad_error", @"Failed to load or decode audio file", nil);
        return;
    }

    NSArray *segments = [vadContext detectSpeech:data samplesCount:count options:options];
    resolve(segments);
}

RCT_REMAP_METHOD(releaseVadContext,
                 withVadContextId:(int)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNWhisperVadContext *vadContext = vadContexts[[NSNumber numberWithInt:contextId]];
    if (vadContext == nil) {
        reject(@"whisper_vad_error", @"VAD context not found", nil);
        return;
    }
    rnwhisper_jsi::removeVadContext(contextId);
    [vadContexts removeObjectForKey:[NSNumber numberWithInt:contextId]];
    [vadContext invalidate];
    resolve(nil);
}

RCT_EXPORT_METHOD(releaseAllVadContexts:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    [self releaseAllVadContexts];
    resolve(nil);
}

- (void)releaseAllContexts {
    rnwhisper::job_abort_all(); // graceful abort
    if (contexts != nil) {
        for (NSNumber *contextId in contexts) {
            RNWhisperContext *context = contexts[contextId];
            rnwhisper_jsi::removeContext([contextId intValue]);
            [context invalidate];
        }
        [contexts removeAllObjects];
        contexts = nil;
    }
}

- (void)releaseAllVadContexts {
    if (vadContexts != nil) {
        for (NSNumber *contextId in vadContexts) {
            RNWhisperVadContext *vadContext = vadContexts[contextId];
            rnwhisper_jsi::removeVadContext([contextId intValue]);
            [vadContext invalidate];
        }
        [vadContexts removeAllObjects];
        vadContexts = nil;
    }
}

RCT_EXPORT_METHOD(installJSIBindings:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RCTBridge *bridge = [RCTBridge currentBridge];
    if (bridge == nil) {
        reject(@"whisper_jsi_error", @"Bridge not available", nil);
        return;
    }

    RCTCxxBridge *cxxBridge = (RCTCxxBridge *)bridge;
    auto callInvoker = bridge.jsCallInvoker;
    if (cxxBridge.runtime) {
        facebook::jsi::Runtime *runtime = static_cast<facebook::jsi::Runtime *>(cxxBridge.runtime);

        if (callInvoker) {
          callInvoker->invokeAsync([runtime, callInvoker]() {
            rnwhisper_jsi::installJSIBindings(*runtime, callInvoker);
          });
        } else {
          reject(@"whisper_jsi_error", @"CallInvoker not available", nil);
          return;
        }

        resolve(@{});
    } else {
        reject(@"whisper_jsi_error", @"Runtime not available", nil);
    }
}

- (void)invalidate {
    [super invalidate];

    [self releaseAllContexts];
    [self releaseAllVadContexts];

    [RNWhisperDownloader clearCache];
}

#ifdef RCT_NEW_ARCH_ENABLED
- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params
{
    return std::make_shared<facebook::react::NativeRNWhisperSpecJSI>(params);
}
#endif

@end
