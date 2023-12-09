#if defined(__ANDROID__) && defined(RNWHISPER_ANDROID_ENABLE_LOGGING)
#include <android/log.h>
#define RNWHISPER_ANDROID_TAG "RNWHISPER_LOG_ANDROID"
#define RNWHISPER_LOG_INFO(...)  __android_log_print(ANDROID_LOG_INFO , WHISPER_ANDROID_TAG, __VA_ARGS__)
#define RNWHISPER_LOG_WARN(...)  __android_log_print(ANDROID_LOG_WARN , WHISPER_ANDROID_TAG, __VA_ARGS__)
#define RNWHISPER_LOG_ERROR(...) __android_log_print(ANDROID_LOG_ERROR, WHISPER_ANDROID_TAG, __VA_ARGS__)
#else
#define RNWHISPER_LOG_INFO(...)  fprintf(stderr, __VA_ARGS__)
#define RNWHISPER_LOG_WARN(...)  fprintf(stderr, __VA_ARGS__)
#define RNWHISPER_LOG_ERROR(...) fprintf(stderr, __VA_ARGS__)
#endif // __ANDROID__
