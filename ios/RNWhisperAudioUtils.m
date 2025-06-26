#import "RNWhisperAudioUtils.h"
#if RNWHISPER_BUILD_FROM_SOURCE
#import "whisper.h"
#else
#import <rnwhisper/whisper.h>
#endif

@implementation RNWhisperAudioUtils

+ (float *)decodeWaveData:(NSData*)data count:(int *)count cutHeader:(BOOL)cutHeader {
  NSData *waveData = data;
  if (cutHeader) {
    // just cut 44 bytes from the beginning
    waveData = [data subdataWithRange:NSMakeRange(44, [data length]-44)];
  }
  const short *shortArray = (const short *)[waveData bytes];
  int shortCount = (int) ([waveData length] / sizeof(short));
  float *floatArray = (float *) malloc(shortCount * sizeof(float));
  for (NSInteger i = 0; i < shortCount; i++) {
      float floatValue = ((float)shortArray[i]) / 32767.0;
      floatValue = MAX(floatValue, -1.0);
      floatValue = MIN(floatValue, 1.0);
      floatArray[i] = floatValue;
  }
  *count = shortCount;
  return floatArray;
}

+ (float *)decodeWaveFile:(NSString*)filePath count:(int *)count {
    NSURL *url = [NSURL fileURLWithPath:filePath];
    NSData *fileData = [NSData dataWithContentsOfURL:url];
    if (fileData == nil) {
        return nil;
    }
    return [RNWhisperAudioUtils decodeWaveData:fileData count:count cutHeader:YES];
}

@end
