#import "RNWhisperAudioUtils.h"
#import "whisper.h"

@implementation RNWhisperAudioUtils

+ (float *)decodeWaveFile:(NSString*)filePath count:(int *)count {
    NSURL *url = [NSURL fileURLWithPath:filePath];
    NSData *fileData = [NSData dataWithContentsOfURL:url];
    if (fileData == nil) {
        return nil;
    }
    NSMutableData *waveData = [[NSMutableData alloc] init];
    [waveData appendData:[fileData subdataWithRange:NSMakeRange(44, [fileData length]-44)]];
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

@end
