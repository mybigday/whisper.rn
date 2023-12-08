#import <Foundation/Foundation.h>

@interface RNWhisperAudioUtils : NSObject

+ (float *)decodeWaveFile:(NSString*)filePath count:(int *)count;

@end
