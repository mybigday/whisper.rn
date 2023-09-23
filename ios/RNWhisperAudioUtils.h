#import <Foundation/Foundation.h>

@interface RNWhisperAudioUtils : NSObject

+ (NSData *)concatShortBuffers:(NSMutableArray<NSValue *> *)buffers sliceNSamples:(NSMutableArray<NSNumber *> *)sliceNSamples;
+ (void)saveWavFile:(NSData *)rawData audioOutputFile:(NSString *)audioOutputFile;

@end
