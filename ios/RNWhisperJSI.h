#ifdef __cplusplus
#if RNWHISPER_BUILD_FROM_SOURCE
#import "whisper.h"
#import "rn-whisper.h"
#else
#import <rnwhisper/whisper.h>
#import <rnwhisper/rn-whisper.h>
#endif
#endif

#import <Foundation/Foundation.h>
#import <React/RCTBridge+Private.h>
#import <ReactCommon/RCTTurboModule.h>
#import <jsi/jsi.h>

@interface RNWhisperJSI : NSObject

+ (void)installJSIBindings:(facebook::jsi::Runtime &)runtime bridge:(RCTBridge *)bridge callInvoker:(std::shared_ptr<facebook::react::CallInvoker>)callInvoker;

@end
