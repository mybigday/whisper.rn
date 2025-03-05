#import "RNWhisper.h"
#import "RNWhisperContext.h"
#import "RNWhisperDownloader.h"
#import "RNWhisperAudioUtils.h"
#import "RNWhisperAudioSessionUtils.h"
#include <stdlib.h>
#include <string>

#ifdef RCT_NEW_ARCH_ENABLED
#import <RNWhisperSpec/RNWhisperSpec.h>
#endif

@implementation RNWhisper

NSMutableDictionary *contexts;

RCT_EXPORT_MODULE()

+ (BOOL)requiresMainQueueSetup
{
  return NO;
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
    @"@RNWhisper_onRealtimeTranscribeVolumeChange",
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
            } else if ([type isEqualToString:@"volumeChange"]) { // Corrected line
                eventName = @"@RNWhisper_onRealtimeTranscribeVolumeChange";
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

- (void)invalidate {
    [super invalidate];

    if (contexts == nil) {
        return;
    }

    for (NSNumber *contextId in contexts) {
        RNWhisperContext *context = contexts[contextId];
        [context invalidate];
    }

    rnwhisper::job_abort_all(); // graceful abort

    [contexts removeAllObjects];
    contexts = nil;

    [RNWhisperDownloader clearCache];
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


RCT_EXPORT_METHOD(pauseRealtimeTranscribe:(int)contextId
                  withResolver:(RCTPromiseResolveBlock)resolve
                  withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNWhisperContext *context = contexts[@(contextId)];
    if (context == nil) {
        reject(@"whisper_error", @"Context not found", nil);
        return;
    }
    [context pauseAudio];
    resolve(nil);
}

RCT_EXPORT_METHOD(resumeRealtimeTranscribe:(int)contextId
                  withResolver:(RCTPromiseResolveBlock)resolve
                  withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNWhisperContext *context = contexts[@(contextId)];
    if (context == nil) {
        reject(@"whisper_error", @"Context not found", nil);
        return;
    }
    [context resumeAudio];
    resolve(nil);
}

RCT_EXPORT_METHOD(finalizeWavFile:(NSString *)filePath
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
//   std::string cppPath = [filePath UTF8String];
  
  try {
    // rnaudioutils::WavWriter::finalizeExternalWav(cppPath);
    resolve(@(YES));
  } catch (const std::exception& e) {
    reject(@"file_error", @(e.what()), nil);
  }
}

#ifdef RCT_NEW_ARCH_ENABLED
- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params
{
    return std::make_shared<facebook::react::NativeRNWhisperSpecJSI>(params);
}
#endif

@end
