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

+ (void)initialize {
    if (self == [RNWhisperJSI class]) {
        contextQueue = dispatch_queue_create("com.rnwhisper.jsi.context", DISPATCH_QUEUE_CONCURRENT);
    }
}

// Helper method to safely get context
+ (RNWhisperContext *)safelyGetContext:(int)contextId {
    __block RNWhisperContext *context = nil;
    dispatch_sync(contextQueue, ^{
        if (contexts != nil) {
            context = contexts[[NSNumber numberWithInt:contextId]];
        }
    });
    return context;
}

+ (void)installJSIBindings:(facebook::jsi::Runtime &)runtime bridge:(RCTBridge *)bridge {
    @try {
        // Test function to verify JSI access to whisper contexts
        auto whisperTestContext = Function::createFromHostFunction(
            runtime,
            PropNameID::forAscii(runtime, "whisperTestContext"),
            1, // number of arguments
            [](Runtime& runtime, const Value& thisValue, const Value* arguments, size_t count) -> Value {
                @try {
                    if (count != 1) {
                        throw JSError(runtime, "whisperTestContext expects 1 argument (contextId)");
                    }

                    if (!arguments[0].isNumber()) {
                        throw JSError(runtime, "whisperTestContext expects contextId to be a number");
                    }

                    int contextId = (int)arguments[0].getNumber();

                    NSLog(@"RNWhisperJSI: whisperTestContext called with contextId=%d", contextId);

                    @autoreleasepool {
                        // Thread-safe context lookup
                        RNWhisperContext *context = [RNWhisperJSI safelyGetContext:contextId];
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

                        NSLog(@"RNWhisperJSI: Context validated successfully for id=%d", contextId);
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

        // Install all JSI functions on the global object
        runtime.global().setProperty(runtime, "whisperTestContext", std::move(whisperTestContext));

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
