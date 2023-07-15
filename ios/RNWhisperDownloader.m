#import "RNWhisperDownloader.h"

/**
 * NOTE: This is simple downloader,
 * the main purpose is supported load assets on RN Debug mode,
 * so it's a very crude implementation.
 * 
 * If you want to use file download in production to load model / audio files,
 * I would recommend using react-native-fs or expo-file-system to manage the files.
 */
@implementation RNWhisperDownloader

+ (NSString *)downloadFile:(NSString *)urlString toFile:(NSString *)path {
  NSURL *url = [NSURL URLWithString:urlString];
  NSString *filePath = [NSTemporaryDirectory() stringByAppendingPathComponent:@"rnwhisper_debug_assets/"];
  if (path) {
    filePath = [filePath stringByAppendingPathComponent:path];
  } else {
    filePath = [filePath stringByAppendingPathComponent:[url lastPathComponent]];
  }
  
  NSString *folderPath = [filePath stringByDeletingLastPathComponent];
  if (![[NSFileManager defaultManager] fileExistsAtPath:folderPath]) {
    [[NSFileManager defaultManager] createDirectoryAtPath:folderPath withIntermediateDirectories:YES attributes:nil error:nil];
  }
  if ([[NSFileManager defaultManager] fileExistsAtPath:filePath]) {
    return filePath;
  }
  NSData *urlData = [NSData dataWithContentsOfURL:url];
  [urlData writeToFile:filePath atomically:YES];
  return filePath;
}

+ (void)clearCache {
  NSString *filePath = [NSTemporaryDirectory() stringByAppendingPathComponent:@"rnwhisper_debug_assets/"];
  [[NSFileManager defaultManager] removeItemAtPath:filePath error:nil];
}

@end
