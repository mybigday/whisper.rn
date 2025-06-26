#ifdef __cplusplus
#if RNWHISPER_BUILD_FROM_SOURCE
#import "whisper.h"
#import "rn-whisper.h"
#else
#import <rnwhisper/whisper.h>
#import <rnwhisper/rn-whisper.h>
#endif
#endif

#import <React/RCTBridgeModule.h>
#import <React/RCTEventEmitter.h>

@interface RNWhisper : RCTEventEmitter <RCTBridgeModule>

@end
