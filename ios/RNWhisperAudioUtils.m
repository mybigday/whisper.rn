#import "RNWhisperAudioUtils.h"
#import "whisper.h"

@implementation RNWhisperAudioUtils

+ (NSData *)concatShortBuffers:(NSMutableArray<NSValue *> *)buffers sliceSize:(int)sliceSize lastSliceSize:(int)lastSliceSize {
    NSMutableData *outputData = [NSMutableData data];
    for (NSValue *buffer in buffers) {
        int size = sliceSize;
        if (buffer == buffers.lastObject) {
            size = lastSliceSize;
        }
        short *bufferPtr = buffer.pointerValue;
        [outputData appendBytes:bufferPtr length:size * sizeof(short)];
    }
    return outputData;
}

+ (void)saveWavFile:(NSData *)rawData audioOutputFile:(NSString *)audioOutputFile {
    NSMutableData *outputData = [NSMutableData data];
    
    // WAVE header
    [outputData appendData:[@"RIFF" dataUsingEncoding:NSUTF8StringEncoding]]; // chunk id
    int chunkSize = CFSwapInt32HostToLittle(36 + rawData.length);
    [outputData appendBytes:&chunkSize length:sizeof(chunkSize)];
    [outputData appendData:[@"WAVE" dataUsingEncoding:NSUTF8StringEncoding]]; // format
    [outputData appendData:[@"fmt " dataUsingEncoding:NSUTF8StringEncoding]]; // subchunk 1 id
    
    int subchunk1Size = CFSwapInt32HostToLittle(16);
    [outputData appendBytes:&subchunk1Size length:sizeof(subchunk1Size)];

    short audioFormat = CFSwapInt16HostToLittle(1); // PCM
    [outputData appendBytes:&audioFormat length:sizeof(audioFormat)];

    short numChannels = CFSwapInt16HostToLittle(1); // mono
    [outputData appendBytes:&numChannels length:sizeof(numChannels)];

    int sampleRate = CFSwapInt32HostToLittle(WHISPER_SAMPLE_RATE);
    [outputData appendBytes:&sampleRate length:sizeof(sampleRate)];

    // (bitDepth * sampleRate * channels) >> 3
    int byteRate = CFSwapInt32HostToLittle(WHISPER_SAMPLE_RATE * 1 * 16 / 8);
    [outputData appendBytes:&byteRate length:sizeof(byteRate)];

    // (bitDepth * channels) >> 3
    short blockAlign = CFSwapInt16HostToLittle(16 / 8);
    [outputData appendBytes:&blockAlign length:sizeof(blockAlign)];

    // bitDepth
    short bitsPerSample = CFSwapInt16HostToLittle(16);
    [outputData appendBytes:&bitsPerSample length:sizeof(bitsPerSample)];

    [outputData appendData:[@"data" dataUsingEncoding:NSUTF8StringEncoding]]; // subchunk 2 id
    int subchunk2Size = CFSwapInt32HostToLittle((int)rawData.length);
    [outputData appendBytes:&subchunk2Size length:sizeof(subchunk2Size)];

    // Audio data
    [outputData appendData:rawData];
    
    // Save to file
    [outputData writeToFile:audioOutputFile atomically:YES];
}

@end
