

@interface RNWhisperDownloader : NSObject

+ (NSString *)downloadFile:(NSString *)urlString toFile:(NSString *)path;
+ (void)clearCache;

@end
