#import <Foundation/Foundation.h>

@interface RNWhisperAudioUtils : NSObject

+ (float *)decodeWaveData:(NSData*)data count:(int *)count cutHeader:(BOOL)cutHeader;
+ (float *)decodeWaveFile:(NSString*)filePath count:(int *)count;

@end
