#import "SimpleFileDownloader.h"

/**
 * NOTE: This is simple FileDownloader,
 * the main purpose is supported load assets on RN Debug mode,
 * so it's a very crude implementation.
 * 
 * If you want to use file download in production to load model / audio files,
 * I would recommend using react-native-fs or expo-file-system to manage the files.
 */
@implementation SimpleFileDownloader

+ (NSString *)downloadFile:(NSURL *)url {
  NSString *dirPath = [NSTemporaryDirectory() stringByAppendingPathComponent:@"rnwhisper/"];
  if (![[NSFileManager defaultManager] fileExistsAtPath:dirPath]) {
    [[NSFileManager defaultManager] createDirectoryAtPath:dirPath withIntermediateDirectories:YES attributes:nil error:nil];
  }
  NSString *filePath = [dirPath stringByAppendingPathComponent:[url lastPathComponent]];
  if ([[NSFileManager defaultManager] fileExistsAtPath:filePath]) {
    return filePath;
  }
  NSData *urlData = [NSData dataWithContentsOfURL:url];
  [urlData writeToFile:filePath atomically:YES];
  return filePath;
}

@end
