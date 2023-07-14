

@interface SimpleFileDownloader : NSObject

+ (NSString *)downloadFile:(NSURL *)url toFile:(NSString *)path;
+ (NSString *)saveData:(NSString *)data toFile:(NSString *)filepath;

@end
