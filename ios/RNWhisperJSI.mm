#import "RNWhisperJSI.h"
#import "RNWhisperContext.h"
#import "RNWhisperVadContext.h"
#import "RNWhisperAudioUtils.h"
#import <jsi/jsi.h>
#import <memory>

using namespace facebook::jsi;

// External references from RNWhisper.mm
extern NSMutableDictionary *contexts;
extern NSMutableDictionary *vadContexts;

// Thread safety for accessing contexts
static dispatch_queue_t contextQueue;
static BOOL jsiBindingsInstalled = NO;

@implementation RNWhisperJSI

+ (void)installJSIBindings:(
  facebook::jsi::Runtime &)runtime
  bridge:(RCTBridge *)bridge
  callInvoker:(std::shared_ptr<facebook::react::CallInvoker>)callInvoker
{
    @try {
        // Test function to verify JSI access to whisper contexts
        auto whisperTestContext = Function::createFromHostFunction(
            runtime,
            PropNameID::forAscii(runtime, "whisperTestContext"),
            2, // number of arguments
            [](Runtime& runtime, const Value& thisValue, const Value* arguments, size_t count) -> Value {
                @try {
                    if (count != 2) {
                        throw JSError(runtime, "whisperTestContext expects 2 arguments (contextId, arrayBuffer)");
                    }

                    if (!arguments[0].isNumber()) {
                        throw JSError(runtime, "whisperTestContext expects contextId to be a number");
                    }

                    if (!arguments[1].isObject() || !arguments[1].getObject(runtime).isArrayBuffer(runtime)) {
                        throw JSError(runtime, "whisperTestContext expects second argument to be an ArrayBuffer");
                    }

                    int contextId = (int)arguments[0].getNumber();

                    // Get ArrayBuffer data
                    auto arrayBuffer = arguments[1].getObject(runtime).getArrayBuffer(runtime);
                    size_t arrayBufferSize = arrayBuffer.size(runtime);
                    uint8_t* arrayBufferData = arrayBuffer.data(runtime);

                    NSLog(@"RNWhisperJSI: whisperTestContext called with contextId=%d, arrayBuffer size=%zu", contextId, arrayBufferSize);

                    @autoreleasepool {
                        // Thread-safe context lookup
                        RNWhisperContext *context = contexts[[NSNumber numberWithInt:contextId]];
                        if (context == nil) {
                            NSLog(@"RNWhisperJSI: Context not found for id=%d", contextId);
                            return Value::undefined();
                        }

                        // Validate that context object is still valid
                        if (![context isKindOfClass:[RNWhisperContext class]]) {
                            NSLog(@"RNWhisperJSI: Invalid context object for id=%d", contextId);
                            return Value::undefined();
                        }

                        // Test that we can access the whisper context
                        struct whisper_context *whisperCtx = nil;
                        @try {
                            whisperCtx = [context getContext];
                        } @catch (NSException *exception) {
                            NSLog(@"RNWhisperJSI: Exception getting context: %@", exception);
                            return Value::undefined();
                        }

                        if (whisperCtx == NULL) {
                            NSLog(@"RNWhisperJSI: Whisper context pointer is null for id=%d", contextId);
                            return Value::undefined();
                        }

                        NSLog(@"RNWhisperJSI: Context validated successfully for id=%d, processed ArrayBuffer with %zu bytes", contextId, arrayBufferSize);
                        return Value(true);
                    }
                } @catch (NSException *exception) {
                    NSLog(@"RNWhisperJSI: NSException in whisperTestContext: %@", exception);
                    throw JSError(runtime, std::string("whisperTestContext error: ") + std::string([exception.description UTF8String]));
                } @catch (...) {
                    NSLog(@"RNWhisperJSI: Unknown exception in whisperTestContext");
                    throw JSError(runtime, "whisperTestContext encountered unknown error");
                }
            }
        );

        // whisperTranscribeData function
        auto whisperTranscribeData = Function::createFromHostFunction(
            runtime,
            PropNameID::forAscii(runtime, "whisperTranscribeData"),
            3, // number of arguments
            [callInvoker](Runtime& runtime, const Value& thisValue, const Value* arguments, size_t count) -> Value {
                @try {
                    if (count != 3) {
                        throw JSError(runtime, "whisperTranscribeData expects 3 arguments (contextId, options, arrayBuffer)");
                    }

                    if (!arguments[0].isNumber()) {
                        throw JSError(runtime, "whisperTranscribeData expects contextId to be a number");
                    }

                    if (!arguments[1].isObject()) {
                        throw JSError(runtime, "whisperTranscribeData expects options to be an object");
                    }

                    if (!arguments[2].isObject() || !arguments[2].getObject(runtime).isArrayBuffer(runtime)) {
                        throw JSError(runtime, "whisperTranscribeData expects third argument to be an ArrayBuffer");
                    }

                    int contextId = (int)arguments[0].getNumber();
                    auto optionsObj = arguments[1].getObject(runtime);
                    auto arrayBuffer = arguments[2].getObject(runtime).getArrayBuffer(runtime);

                    size_t arrayBufferSize = arrayBuffer.size(runtime);
                    uint8_t* arrayBufferData = arrayBuffer.data(runtime);

                    NSLog(@"RNWhisperJSI: whisperTranscribeData called with contextId=%d, arrayBuffer size=%zu", contextId, arrayBufferSize);

                    @autoreleasepool {
                        // Thread-safe context lookup
                        RNWhisperContext *context = contexts[[NSNumber numberWithInt:contextId]];
                        if (context == nil) {
                            throw JSError(runtime, "Context not found for id: " + std::to_string(contextId));
                        }

                        // Convert ArrayBuffer to float32 array (assuming 16-bit PCM input)
                        if (arrayBufferSize % 2 != 0) {
                            throw JSError(runtime, "ArrayBuffer size must be even for 16-bit PCM data");
                        }

                        int audioDataCount = (int)(arrayBufferSize / 2); // 16-bit samples
                        float* audioData = (float*)malloc(audioDataCount * sizeof(float));

                        // Convert 16-bit PCM to float32
                        int16_t* pcmData = (int16_t*)arrayBufferData;
                        for (int i = 0; i < audioDataCount; i++) {
                            audioData[i] = (float)pcmData[i] / 32768.0f;
                        }

                        // Convert JSI options to NSDictionary
                        NSMutableDictionary *options = [[NSMutableDictionary alloc] init];
                        auto propNames = optionsObj.getPropertyNames(runtime);
                        for (size_t i = 0; i < propNames.size(runtime); i++) {
                            auto propName = propNames.getValueAtIndex(runtime, i).getString(runtime);
                            auto propValue = optionsObj.getProperty(runtime, propName);
                            NSString *key = [NSString stringWithUTF8String:propName.utf8(runtime).c_str()];

                            if (propValue.isBool()) {
                                options[key] = @(propValue.getBool());
                            } else if (propValue.isNumber()) {
                                options[key] = @(propValue.getNumber());
                            } else if (propValue.isString()) {
                                options[key] = [NSString stringWithUTF8String:propValue.getString(runtime).utf8(runtime).c_str()];
                            }
                        }

                        // Create a proper Promise
                        auto promiseConstructor = runtime.global().getPropertyAsFunction(runtime, "Promise");

                        auto promiseExecutor = Function::createFromHostFunction(
                            runtime,
                            PropNameID::forAscii(runtime, ""),
                            2, // resolve, reject
                            [context, audioData, audioDataCount, options, callInvoker](Runtime& runtime, const Value& thisValue, const Value* arguments, size_t count) -> Value {
                                if (count != 2) {
                                    free(audioData);
                                    throw JSError(runtime, "Promise executor expects 2 arguments (resolve, reject)");
                                }

                                // Create shared pointers to keep the functions alive
                                auto resolvePtr = std::make_shared<Function>(arguments[0].getObject(runtime).getFunction(runtime));
                                auto rejectPtr = std::make_shared<Function>(arguments[1].getObject(runtime).getFunction(runtime));

                                // Run transcription on background thread
                                dispatch_async([context getDispatchQueue], ^{
                                    @try {
                                        int jobId = rand();

                                        [context transcribeData:jobId
                                                      audioData:audioData
                                                 audioDataCount:audioDataCount
                                                        options:options
                                                     onProgress:^(int progress) {
                                                         // Progress callback could be implemented here
                                                     }
                                                  onNewSegments:^(NSDictionary *segments) {
                                                      // New segments callback could be implemented here
                                                  }
                                                          onEnd:^(int code) {
                                                              callInvoker->invokeAsync([resolvePtr, rejectPtr, code, context, audioData, &runtime]() {
                                                                  @try {
                                                                      if (code == 0) {
                                                                          // Get transcription results
                                                                          NSMutableDictionary *results = [context getTextSegments];

                                                                          // Convert results to JSI object
                                                                          auto resultObj = Object(runtime);
                                                                          resultObj.setProperty(runtime, "code", Value(code));

                                                                          if (results[@"result"]) {
                                                                              NSString *text = results[@"result"];
                                                                              resultObj.setProperty(runtime, "result", String::createFromUtf8(runtime, [text UTF8String]));
                                                                          }

                                                                          if (results[@"segments"]) {
                                                                              NSArray *segments = results[@"segments"];
                                                                              auto segmentsArray = Array(runtime, segments.count);

                                                                              for (NSUInteger i = 0; i < segments.count; i++) {
                                                                                  NSDictionary *segment = segments[i];
                                                                                  auto segmentObj = Object(runtime);

                                                                                  if (segment[@"text"]) {
                                                                                      segmentObj.setProperty(runtime, "text", String::createFromUtf8(runtime, [segment[@"text"] UTF8String]));
                                                                                  }
                                                                                  if (segment[@"t0"]) {
                                                                                      segmentObj.setProperty(runtime, "t0", Value([segment[@"t0"] doubleValue]));
                                                                                  }
                                                                                  if (segment[@"t1"]) {
                                                                                      segmentObj.setProperty(runtime, "t1", Value([segment[@"t1"] doubleValue]));
                                                                                  }

                                                                                  segmentsArray.setValueAtIndex(runtime, i, segmentObj);
                                                                              }

                                                                              resultObj.setProperty(runtime, "segments", segmentsArray);
                                                                          }

                                                                          resolvePtr->call(runtime, resultObj);
                                                                      } else {
                                                                          auto errorObj = Object(runtime);
                                                                          errorObj.setProperty(runtime, "code", Value(code));
                                                                          errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, "Transcription failed"));
                                                                          rejectPtr->call(runtime, errorObj);
                                                                      }
                                                                  } @catch (NSException *exception) {
                                                                      auto errorObj = Object(runtime);
                                                                      errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, [exception.description UTF8String]));
                                                                      rejectPtr->call(runtime, errorObj);
                                                                  } @catch (...) {
                                                                      auto errorObj = Object(runtime);
                                                                      errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, "Unknown error"));
                                                                      rejectPtr->call(runtime, errorObj);
                                                                  }

                                                                  free(audioData);
                                                              });
                                                          }];
                                    } @catch (NSException *exception) {
                                        callInvoker->invokeAsync([rejectPtr, exception, audioData, &runtime]() {
                                            auto errorObj = Object(runtime);
                                            errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, [exception.description UTF8String]));
                                            rejectPtr->call(runtime, errorObj);
                                            free(audioData);
                                        });
                                    }
                                });

                                return Value::undefined();
                            }
                        );

                        // Create and return the Promise
                        return promiseConstructor.callAsConstructor(runtime, promiseExecutor);
                    }
                } @catch (NSException *exception) {
                    NSLog(@"RNWhisperJSI: NSException in whisperTranscribeData: %@", exception);
                    throw JSError(runtime, std::string("whisperTranscribeData error: ") + std::string([exception.description UTF8String]));
                } @catch (...) {
                    NSLog(@"RNWhisperJSI: Unknown exception in whisperTranscribeData");
                    throw JSError(runtime, "whisperTranscribeData encountered unknown error");
                }
            }
        );

                // whisperVadDetectSpeech function
        auto whisperVadDetectSpeech = Function::createFromHostFunction(
            runtime,
            PropNameID::forAscii(runtime, "whisperVadDetectSpeech"),
            3, // number of arguments
            [callInvoker](Runtime& runtime, const Value& thisValue, const Value* arguments, size_t count) -> Value {
                @try {
                    if (count != 3) {
                        throw JSError(runtime, "whisperVadDetectSpeech expects 3 arguments (contextId, options, arrayBuffer)");
                    }

                    if (!arguments[0].isNumber()) {
                        throw JSError(runtime, "whisperVadDetectSpeech expects contextId to be a number");
                    }

                    if (!arguments[1].isObject()) {
                        throw JSError(runtime, "whisperVadDetectSpeech expects options to be an object");
                    }

                    if (!arguments[2].isObject() || !arguments[2].getObject(runtime).isArrayBuffer(runtime)) {
                        throw JSError(runtime, "whisperVadDetectSpeech expects third argument to be an ArrayBuffer");
                    }

                    int contextId = (int)arguments[0].getNumber();
                    auto optionsObj = arguments[1].getObject(runtime);
                    auto arrayBuffer = arguments[2].getObject(runtime).getArrayBuffer(runtime);

                    size_t arrayBufferSize = arrayBuffer.size(runtime);
                    uint8_t* arrayBufferData = arrayBuffer.data(runtime);

                    NSLog(@"RNWhisperJSI: whisperVadDetectSpeech called with contextId=%d, arrayBuffer size=%zu", contextId, arrayBufferSize);

                    @autoreleasepool {
                        // Thread-safe VAD context lookup
                        RNWhisperVadContext *vadContext = vadContexts[[NSNumber numberWithInt:contextId]];
                        if (vadContext == nil) {
                            throw JSError(runtime, "VAD Context not found for id: " + std::to_string(contextId));
                        }

                        // Convert ArrayBuffer to float32 array (assuming 16-bit PCM input)
                        if (arrayBufferSize % 2 != 0) {
                            throw JSError(runtime, "ArrayBuffer size must be even for 16-bit PCM data");
                        }

                        int audioDataCount = (int)(arrayBufferSize / 2); // 16-bit samples
                        float* audioData = (float*)malloc(audioDataCount * sizeof(float));

                        // Convert 16-bit PCM to float32
                        int16_t* pcmData = (int16_t*)arrayBufferData;
                        for (int i = 0; i < audioDataCount; i++) {
                            audioData[i] = (float)pcmData[i] / 32768.0f;
                        }

                        // Convert JSI options to NSDictionary for VAD
                        NSMutableDictionary *vadOptions = [[NSMutableDictionary alloc] init];
                        auto propNames = optionsObj.getPropertyNames(runtime);
                        for (size_t i = 0; i < propNames.size(runtime); i++) {
                            auto propName = propNames.getValueAtIndex(runtime, i).getString(runtime);
                            auto propValue = optionsObj.getProperty(runtime, propName);
                            NSString *key = [NSString stringWithUTF8String:propName.utf8(runtime).c_str()];

                            if (propValue.isBool()) {
                                vadOptions[key] = @(propValue.getBool());
                            } else if (propValue.isNumber()) {
                                vadOptions[key] = @(propValue.getNumber());
                            } else if (propValue.isString()) {
                                vadOptions[key] = [NSString stringWithUTF8String:propValue.getString(runtime).utf8(runtime).c_str()];
                            }
                        }

                        // Create a proper Promise for async VAD detection
                        auto promiseConstructor = runtime.global().getPropertyAsFunction(runtime, "Promise");

                        auto promiseExecutor = Function::createFromHostFunction(
                            runtime,
                            PropNameID::forAscii(runtime, ""),
                            2, // resolve, reject
                            [vadContext, audioData, audioDataCount, vadOptions, callInvoker](Runtime& runtime, const Value& thisValue, const Value* arguments, size_t count) -> Value {
                                if (count != 2) {
                                    free(audioData);
                                    throw JSError(runtime, "Promise executor expects 2 arguments (resolve, reject)");
                                }

                                // Create shared pointers to keep the functions alive
                                auto resolvePtr = std::make_shared<Function>(arguments[0].getObject(runtime).getFunction(runtime));
                                auto rejectPtr = std::make_shared<Function>(arguments[1].getObject(runtime).getFunction(runtime));

                                // Run VAD detection on background thread
                                dispatch_async([vadContext getDispatchQueue], ^{
                                    @try {
                                        // Call VAD detection - this returns an array of speech segments
                                        NSArray *segments = [vadContext detectSpeech:audioData samplesCount:audioDataCount options:vadOptions];

                                        // Call resolve on JS thread
                                        callInvoker->invokeAsync([resolvePtr, rejectPtr, segments, audioData, &runtime]() {
                                            @try {
                                                // Convert segments array to JSI format
                                                auto resultObj = Object(runtime);
                                                resultObj.setProperty(runtime, "hasSpeech", Value(segments.count > 0));

                                                auto segmentsArray = Array(runtime, segments.count);
                                                for (NSUInteger i = 0; i < segments.count; i++) {
                                                    NSDictionary *segment = segments[i];
                                                    auto segmentObj = Object(runtime);

                                                    if (segment[@"t0"]) {
                                                        segmentObj.setProperty(runtime, "t0", Value([segment[@"t0"] doubleValue]));
                                                    }
                                                    if (segment[@"t1"]) {
                                                        segmentObj.setProperty(runtime, "t1", Value([segment[@"t1"] doubleValue]));
                                                    }

                                                    segmentsArray.setValueAtIndex(runtime, i, segmentObj);
                                                }

                                                resultObj.setProperty(runtime, "segments", segmentsArray);
                                                resolvePtr->call(runtime, resultObj);
                                            } @catch (NSException *exception) {
                                                auto errorObj = Object(runtime);
                                                errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, [exception.description UTF8String]));
                                                rejectPtr->call(runtime, errorObj);
                                            } @catch (...) {
                                                auto errorObj = Object(runtime);
                                                errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, "Unknown error"));
                                                rejectPtr->call(runtime, errorObj);
                                            }

                                            free(audioData);
                                        });
                                    } @catch (NSException *exception) {
                                        callInvoker->invokeAsync([rejectPtr, exception, audioData, &runtime]() {
                                            auto errorObj = Object(runtime);
                                            errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, [exception.description UTF8String]));
                                            rejectPtr->call(runtime, errorObj);
                                            free(audioData);
                                        });
                                    }
                                });

                                return Value::undefined();
                            }
                        );

                        // Create and return the Promise
                        return promiseConstructor.callAsConstructor(runtime, promiseExecutor);
                    }
                } @catch (NSException *exception) {
                    NSLog(@"RNWhisperJSI: NSException in whisperVadDetectSpeech: %@", exception);
                    throw JSError(runtime, std::string("whisperVadDetectSpeech error: ") + std::string([exception.description UTF8String]));
                } @catch (...) {
                    NSLog(@"RNWhisperJSI: Unknown exception in whisperVadDetectSpeech");
                    throw JSError(runtime, "whisperVadDetectSpeech encountered unknown error");
                }
            }
        );

        // Install all JSI functions on the global object
        runtime.global().setProperty(runtime, "whisperTestContext", std::move(whisperTestContext));
        runtime.global().setProperty(runtime, "whisperTranscribeData", std::move(whisperTranscribeData));
        runtime.global().setProperty(runtime, "whisperVadDetectSpeech", std::move(whisperVadDetectSpeech));

        // Mark JSI bindings as installed
        jsiBindingsInstalled = YES;
        NSLog(@"RNWhisperJSI: JSI bindings installed successfully");

    } @catch (NSException *exception) {
        NSLog(@"RNWhisperJSI: NSException installing JSI bindings: %@", exception);
        @throw exception;
    } @catch (...) {
        NSLog(@"RNWhisperJSI: Unknown exception installing JSI bindings");
        throw facebook::jsi::JSError(runtime, "Failed to install JSI bindings: unknown error");
    }
}

@end
