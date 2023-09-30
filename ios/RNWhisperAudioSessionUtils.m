#import "RNWhisperAudioSessionUtils.h"

@implementation RNWhisperAudioSessionUtils

static NSDictionary *_categories;
static NSDictionary *_options;
static NSDictionary *_modes;

+ (void)initialize {
    _categories = @{
        @"Ambient": AVAudioSessionCategoryAmbient,
        @"SoloAmbient": AVAudioSessionCategorySoloAmbient,
        @"Playback": AVAudioSessionCategoryPlayback,
        @"Record": AVAudioSessionCategoryRecord,
        @"PlayAndRecord": AVAudioSessionCategoryPlayAndRecord,
        @"MultiRoute": AVAudioSessionCategoryMultiRoute
    };
    _options = @{
        @"MixWithOthers": @(AVAudioSessionCategoryOptionMixWithOthers),
        @"DuckOthers": @(AVAudioSessionCategoryOptionDuckOthers),
        @"InterruptSpokenAudioAndMixWithOthers": @(AVAudioSessionCategoryOptionInterruptSpokenAudioAndMixWithOthers),
        @"AllowBluetooth": @(AVAudioSessionCategoryOptionAllowBluetooth),
        @"AllowBluetoothA2DP": @(AVAudioSessionCategoryOptionAllowBluetoothA2DP),
        @"AllowAirPlay": @(AVAudioSessionCategoryOptionAllowAirPlay),
        @"DefaultToSpeaker": @(AVAudioSessionCategoryOptionDefaultToSpeaker)
    };
    _modes = @{
        @"Default": AVAudioSessionModeDefault,
        @"VoiceChat": AVAudioSessionModeVoiceChat,
        @"VideoChat": AVAudioSessionModeVideoChat,
        @"GameChat": AVAudioSessionModeGameChat,
        @"VideoRecording": AVAudioSessionModeVideoRecording,
        @"Measurement": AVAudioSessionModeMeasurement,
        @"MoviePlayback": AVAudioSessionModeMoviePlayback,
        @"SpokenAudio": AVAudioSessionModeSpokenAudio
    };
}

+(NSString *)getCurrentCategory {
    AVAudioSession *session = [AVAudioSession sharedInstance];
    return session.category;
}

+(NSArray *)getCurrentOptions {
    AVAudioSession *session = [AVAudioSession sharedInstance];
    AVAudioSessionCategoryOptions options = session.categoryOptions;
    NSMutableArray *result = [NSMutableArray array];
    for (NSString *key in _options) {
        if ((options & [[_options objectForKey:key] unsignedIntegerValue]) != 0) {
            [result addObject:key];
        }
    }
    return result;
}

+(NSString *)getCurrentMode {
    AVAudioSession *session = [AVAudioSession sharedInstance];
    return session.mode;
}

+(void)setCategory:(NSString *)category options:(NSArray *)options error:(NSError **)error {
    AVAudioSession *session = [AVAudioSession sharedInstance];
    [session setCategory:[_categories objectForKey:category] withOptions:[self getOptions:options] error:error];
}

+(void)setMode:(NSString *)mode error:(NSError **)error {
    AVAudioSession *session = [AVAudioSession sharedInstance];
    [session setMode:[_modes objectForKey:mode] error:error];
}

+(void)setActive:(BOOL)active error:(NSError **)error {
    AVAudioSession *session = [AVAudioSession sharedInstance];
    [session setActive:active error:error];
}

+(AVAudioSessionCategoryOptions)getOptions:(NSArray *)options {
    AVAudioSessionCategoryOptions result = 0;
    for (NSString *option in options) {
        result |= [[_options objectForKey:option] unsignedIntegerValue];
    }
    return result;
}

@end