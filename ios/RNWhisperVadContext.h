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

@interface RNWhisperVadContext : NSObject {
    int contextId;
    dispatch_queue_t dQueue;
    struct whisper_vad_context * vctx;
    NSString * reasonNoMetal;
    bool isMetalEnabled;
}

+ (instancetype)initWithModelPath:(NSString *)modelPath contextId:(int)contextId noMetal:(BOOL)noMetal nThreads:(NSNumber *)nThreads;
- (bool)isMetalEnabled;
- (NSString *)reasonNoMetal;
- (struct whisper_vad_context *)getVadContext;
- (dispatch_queue_t)getDispatchQueue;
- (NSArray *)detectSpeech:(float *)samples samplesCount:(int)samplesCount options:(NSDictionary *)options;
- (void)invalidate;

@end
