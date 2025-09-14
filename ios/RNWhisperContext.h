#ifdef __cplusplus
#if RNWHISPER_BUILD_FROM_SOURCE
#import "whisper.h"
#import "rn-whisper.h"
#else
#import <rnwhisper/whisper.h>
#import <rnwhisper/rn-whisper.h>
#endif
#endif

@interface RNWhisperContext : NSObject {
    int contextId;
    dispatch_queue_t dQueue;
    struct whisper_context * ctx;
    NSString * reasonNoMetal;
    bool isMetalEnabled;
    bool isTranscribing;

    NSDictionary* options;
    struct rnwhisper::job * job;
}

+ (void)toggleNativeLog:(BOOL)enabled onEmitLog:(void (^)(NSString *level, NSString *text))onEmitLog;
+ (instancetype)initWithModelPath:(NSString *)modelPath contextId:(int)contextId noCoreML:(BOOL)noCoreML noMetal:(BOOL)noMetal useFlashAttn:(BOOL)useFlashAttn;
- (bool)isMetalEnabled;
- (NSString *)reasonNoMetal;
- (struct whisper_context *)getContext;
- (dispatch_queue_t)getDispatchQueue;
- (void)transcribeData:(int)jobId
    audioData:(float *)audioData
    audioDataCount:(int)audioDataCount
    options:(NSDictionary *)options
    onProgress:(void (^)(int))onProgress
    onNewSegments:(void (^)(NSDictionary *))onNewSegments
    onEnd:(void (^)(int))onEnd;
- (void)stopTranscribe:(int)jobId;
- (void)stopCurrentTranscribe;
- (bool)isTranscribing;
- (NSMutableDictionary *)getTextSegments:(NSDictionary *)options;
- (NSString *)bench:(int)maxThreads;
- (void)invalidate;

@end
