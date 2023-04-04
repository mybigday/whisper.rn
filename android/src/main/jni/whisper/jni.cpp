#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <cstdlib>
#include <sys/sysinfo.h>
#include <string>
#include <thread>
#include "whisper.h"
#include "rn-whisper.h"
#include "ggml.h"

#define UNUSED(x) (void)(x)
#define TAG "JNI"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,     TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,     TAG, __VA_ARGS__)

static inline int min(int a, int b) {
    return (a < b) ? a : b;
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_rnwhisper_WhisperContext_initContext(
        JNIEnv *env, jobject thiz, jstring model_path_str) {
    UNUSED(thiz);
    struct whisper_context *context = nullptr;
    const char *model_path_chars = env->GetStringUTFChars(model_path_str, nullptr);
    context = whisper_init_from_file(model_path_chars);
    env->ReleaseStringUTFChars(model_path_str, model_path_chars);
    return reinterpret_cast<jlong>(context);
}

JNIEXPORT jint JNICALL
Java_com_rnwhisper_WhisperContext_fullTranscribe(
    JNIEnv *env,
    jobject thiz,
    jint job_id,
    jlong context_ptr,
    jfloatArray audio_data,
    jint audio_data_len,
    jint n_threads,
    jint max_context,
    int word_thold,
    int max_len,
    jboolean token_timestamps,
    jint offset,
    jint duration,
    jfloat temperature,
    jfloat temperature_inc,
    jint beam_size,
    jint best_of,
    jboolean speed_up,
    jboolean translate,
    jstring language,
    jstring prompt
) {
    UNUSED(thiz);
    struct whisper_context *context = reinterpret_cast<struct whisper_context *>(context_ptr);
    jfloat *audio_data_arr = env->GetFloatArrayElements(audio_data, nullptr);

    int max_threads = std::thread::hardware_concurrency();
    // Use 2 threads by default on 4-core devices, 4 threads on more cores
    int default_n_threads = max_threads == 4 ? 2 : min(4, max_threads);

    LOGI("About to create params");

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    if (beam_size > -1) {
        params.strategy = WHISPER_SAMPLING_BEAM_SEARCH;
        params.beam_search.beam_size = beam_size;
    }

    params.print_realtime = false;
    params.print_progress = false;
    params.print_timestamps = false;
    params.print_special = false;
    params.translate = translate;
    const char *language_chars = env->GetStringUTFChars(language, nullptr);
    params.language = language_chars;
    params.n_threads = n_threads > 0 ? n_threads : default_n_threads;
    params.speed_up = speed_up;
    params.offset_ms = 0;
    params.no_context = true;
    params.single_segment = false;

    if (max_len > -1) {
        params.max_len = max_len;
    }
    params.token_timestamps = token_timestamps;

    if (best_of > -1) {
        params.greedy.best_of = best_of;
    }
    if (max_context > -1) {
        params.n_max_text_ctx = max_context;
    }
    if (offset > -1) {
        params.offset_ms = offset;
    }
    if (duration > -1) {
        params.duration_ms = duration;
    }
    if (word_thold > -1) {
        params.thold_pt = word_thold;
    }
    if (temperature > -1) {
        params.temperature = temperature;
    }
    if (temperature_inc > -1) {
        params.temperature_inc = temperature_inc;
    }
    if (prompt != nullptr) {
        rn_whisper_convert_prompt(
            context,
            params,
            new std::string(env->GetStringUTFChars(prompt, nullptr))
        );
    }

    params.encoder_begin_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, void * user_data) {
        bool is_aborted = *(bool*)user_data;
        return !is_aborted;
    };
    params.encoder_begin_callback_user_data = rn_whisper_assign_abort_map(job_id);

    LOGI("About to reset timings");
    whisper_reset_timings(context);

    LOGI("About to run whisper_full");
    int code = whisper_full(context, params, audio_data_arr, audio_data_len);
    if (code == 0) {
        // whisper_print_timings(context);
    }
    env->ReleaseFloatArrayElements(audio_data, audio_data_arr, JNI_ABORT);
    env->ReleaseStringUTFChars(language, language_chars);
    rn_whisper_remove_abort_map(job_id);
    return code;
}

JNIEXPORT void JNICALL
Java_com_rnwhisper_WhisperContext_abortTranscribe(
    JNIEnv *env,
    jobject thiz,
    jint job_id
) {
    UNUSED(thiz);
    rn_whisper_abort_transcribe(job_id);
}

JNIEXPORT void JNICALL
Java_com_rnwhisper_WhisperContext_abortAllTranscribe(
    JNIEnv *env,
    jobject thiz
) {
    UNUSED(thiz);
    rn_whisper_abort_all_transcribe();
}

JNIEXPORT jint JNICALL
Java_com_rnwhisper_WhisperContext_getTextSegmentCount(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env);
    UNUSED(thiz);
    struct whisper_context *context = reinterpret_cast<struct whisper_context *>(context_ptr);
    return whisper_full_n_segments(context);
}

JNIEXPORT jstring JNICALL
Java_com_rnwhisper_WhisperContext_getTextSegment(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint index) {
    UNUSED(thiz);
    struct whisper_context *context = reinterpret_cast<struct whisper_context *>(context_ptr);
    const char *text = whisper_full_get_segment_text(context, index);
    jstring string = env->NewStringUTF(text);
    return string;
}

JNIEXPORT jint JNICALL
Java_com_rnwhisper_WhisperContext_getTextSegmentT0(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint index) {
    UNUSED(env);
    UNUSED(thiz);
    struct whisper_context *context = reinterpret_cast<struct whisper_context *>(context_ptr);
    return whisper_full_get_segment_t0(context, index);
}

JNIEXPORT jint JNICALL
Java_com_rnwhisper_WhisperContext_getTextSegmentT1(
        JNIEnv *env, jobject thiz, jlong context_ptr, jint index) {
    UNUSED(env);
    UNUSED(thiz);
    struct whisper_context *context = reinterpret_cast<struct whisper_context *>(context_ptr);
    return whisper_full_get_segment_t1(context, index);
}

JNIEXPORT void JNICALL
Java_com_rnwhisper_WhisperContext_freeContext(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env);
    UNUSED(thiz);
    struct whisper_context *context = reinterpret_cast<struct whisper_context *>(context_ptr);
    whisper_free(context);
}

} // extern "C"
