#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>

@interface RNWhisperAudioSessionUtils : NSObject

+(NSString *)getCurrentCategory;
+(NSArray *)getCurrentOptions;
+(NSString *)getCurrentMode;
+(void)setCategory:(NSString *)category options:(NSArray *)options error:(NSError **)error;
+(void)setMode:(NSString *)mode error:(NSError **)error;
+(void)setActive:(BOOL)active error:(NSError **)error;

@end
