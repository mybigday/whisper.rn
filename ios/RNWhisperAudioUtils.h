#import <Foundation/Foundation.h>

@interface RNWhisperAudioUtils : NSObject

+ (NSData *)concatShortBuffers:(NSMutableArray<NSValue *> *)buffers sliceSize:(int)sliceSize lastSliceSize:(int)lastSliceSize;
+ (void)saveWavFile:(NSData *)rawData audioOutputFile:(NSString *)audioOutputFile;

@end
