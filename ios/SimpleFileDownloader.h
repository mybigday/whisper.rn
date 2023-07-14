

@interface SimpleFileDownloader : NSObject

+ (NSString *)downloadFile:(NSURL *)url toFile:(NSString *)path;

@end
