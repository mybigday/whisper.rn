#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <cstdlib>
#include <sys/sysinfo.h>
#include <string>
#include <thread>
#include <vector>
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

// Load model from input stream (used for drawable / raw resources)
struct input_stream_context {
    JNIEnv *env;
    jobject input_stream;
};

static size_t input_stream_read(void *ctx, void *output, size_t read_size) {
    input_stream_context *context = (input_stream_context *)ctx;
    JNIEnv *env = context->env;
    jobject input_stream = context->input_stream;
    jclass input_stream_class = env->GetObjectClass(input_stream);

    jbyteArray buffer = env->NewByteArray(read_size);
    jint bytes_read = env->CallIntMethod(
        input_stream,
        env->GetMethodID(input_stream_class, "read", "([B)I"),
        buffer
    );

    if (bytes_read > 0) {
        env->GetByteArrayRegion(buffer, 0, bytes_read, (jbyte *) output);
    }

    env->DeleteLocalRef(buffer);

    return bytes_read;
}

static bool input_stream_is_eof(void *ctx) {
    input_stream_context *context = (input_stream_context *)ctx;
    JNIEnv *env = context->env;
    jobject input_stream = context->input_stream;

    jclass input_stream_class = env->GetObjectClass(input_stream);

    jbyteArray buffer = env->NewByteArray(1);
    jint bytes_read = env->CallIntMethod(
        input_stream,
        env->GetMethodID(input_stream_class, "read", "([B)I"),
        buffer
    );

    bool is_eof = (bytes_read == -1);
    if (!is_eof) {
        // If we successfully read a byte, "unread" it by pushing it back into the stream.
        env->CallVoidMethod(
            input_stream,
            env->GetMethodID(input_stream_class, "unread", "([BII)V"),
            buffer,
            0,
            1
        );
    }

    env->DeleteLocalRef(buffer);

    return is_eof;
}

static void input_stream_close(void *ctx) {
    input_stream_context *context = (input_stream_context *)ctx;
    JNIEnv *env = context->env;
    jobject input_stream = context->input_stream;
    jclass input_stream_class = env->GetObjectClass(input_stream);

    env->CallVoidMethod(
        input_stream,
        env->GetMethodID(input_stream_class, "close", "()V")
    );

    env->DeleteGlobalRef(input_stream);
}

static struct whisper_context *whisper_init_from_input_stream(
    JNIEnv *env,
    jobject input_stream // PushbackInputStream
) {
    input_stream_context *context = new input_stream_context;
    context->env = env;
    context->input_stream = env->NewGlobalRef(input_stream);

    whisper_model_loader loader = {
        .context = context,
        .read = &input_stream_read,
        .eof = &input_stream_is_eof,
        .close = &input_stream_close
    };
    return whisper_init(&loader);
}

// Load model from asset
static size_t asset_read(void *ctx, void *output, size_t read_size) {
    return AAsset_read((AAsset *) ctx, output, read_size);
}

static bool asset_is_eof(void *ctx) {
    return AAsset_getRemainingLength64((AAsset *) ctx) <= 0;
}

static void asset_close(void *ctx) {
    AAsset_close((AAsset *) ctx);
}

static struct whisper_context *whisper_init_from_asset(
    JNIEnv *env,
    jobject assetManager,
    const char *asset_path
) {
    LOGI("Loading model from asset '%s'\n", asset_path);
    AAssetManager *asset_manager = AAssetManager_fromJava(env, assetManager);
    AAsset *asset = AAssetManager_open(asset_manager, asset_path, AASSET_MODE_STREAMING);
    if (!asset) {
        LOGW("Failed to open '%s'\n", asset_path);
        return NULL;
    }
    whisper_model_loader loader = {
        .context = asset,
        .read = &asset_read,
        .eof = &asset_is_eof,
        .close = &asset_close
    };
    return whisper_init(&loader);
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

JNIEXPORT jlong JNICALL
Java_com_rnwhisper_WhisperContext_initContextWithAsset(
    JNIEnv *env,
    jobject thiz,
    jobject asset_manager,
    jstring model_path_str
) {
    UNUSED(thiz);
    struct whisper_context *context = nullptr;
    const char *model_path_chars = env->GetStringUTFChars(model_path_str, nullptr);
    context = whisper_init_from_asset(env, asset_manager, model_path_chars);
    env->ReleaseStringUTFChars(model_path_str, model_path_chars);
    return reinterpret_cast<jlong>(context);
}

JNIEXPORT jlong JNICALL
Java_com_rnwhisper_WhisperContext_initContextWithInputStream(
    JNIEnv *env,
    jobject thiz,
    jobject input_stream
) {
    UNUSED(thiz);
    struct whisper_context *context = nullptr;
    context = whisper_init_from_input_stream(env, input_stream);
    return reinterpret_cast<jlong>(context);
}

JNIEXPORT jboolean JNICALL
Java_com_rnwhisper_WhisperContext_vadSimple(
    JNIEnv *env,
    jobject thiz,
    jfloatArray audio_data,
    jint audio_data_len,
    jfloat vad_thold,
    jfloat vad_freq_thold
) {
    UNUSED(thiz);

    std::vector<float> samples(audio_data_len);
    jfloat *audio_data_arr = env->GetFloatArrayElements(audio_data, nullptr);
    for (int i = 0; i < audio_data_len; i++) {
        samples[i] = audio_data_arr[i];
    }
    bool is_speech = rn_whisper_vad_simple(samples, WHISPER_SAMPLE_RATE, 1000, vad_thold, vad_freq_thold, false);
    env->ReleaseFloatArrayElements(audio_data, audio_data_arr, JNI_ABORT);
    return is_speech;
}

struct callback_context {
    JNIEnv *env;
    jobject callback_instance;
};

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
    jstring prompt,
    jobject callback_instance
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
        params.initial_prompt = env->GetStringUTFChars(prompt, nullptr);
    }

    // abort handlers
    bool* abort_ptr = rn_whisper_assign_abort_map(job_id);
    params.encoder_begin_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, void * user_data) {
        bool is_aborted = *(bool*)user_data;
        return !is_aborted;
    };
    params.encoder_begin_callback_user_data = abort_ptr;
    params.abort_callback = [](void * user_data) {
        bool is_aborted = *(bool*)user_data;
        return is_aborted;
    };
    params.abort_callback_user_data = abort_ptr;

    if (callback_instance != nullptr) {
        callback_context *cb_ctx = new callback_context;
        cb_ctx->env = env;
        cb_ctx->callback_instance = env->NewGlobalRef(callback_instance);

        params.progress_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, int progress, void * user_data) {
            callback_context *cb_ctx = (callback_context *)user_data;
            JNIEnv *env = cb_ctx->env;
            jobject callback_instance = cb_ctx->callback_instance;
            jclass callback_class = env->GetObjectClass(callback_instance);
            jmethodID onProgress = env->GetMethodID(callback_class, "onProgress", "(I)V");
            env->CallVoidMethod(callback_instance, onProgress, progress);
        };
        params.progress_callback_user_data = cb_ctx;

        params.new_segment_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, int n_new, void * user_data) {
            callback_context *cb_ctx = (callback_context *)user_data;
            JNIEnv *env = cb_ctx->env;
            jobject callback_instance = cb_ctx->callback_instance;
            jclass callback_class = env->GetObjectClass(callback_instance);
            jmethodID onNewSegments = env->GetMethodID(callback_class, "onNewSegments", "(I)V");
            env->CallVoidMethod(callback_instance, onNewSegments, n_new);
        };
        params.new_segment_callback_user_data = cb_ctx;
    }

    LOGI("About to reset timings");
    whisper_reset_timings(context);

    LOGI("About to run whisper_full");
    int code = whisper_full(context, params, audio_data_arr, audio_data_len);
    if (code == 0) {
        // whisper_print_timings(context);
    }
    env->ReleaseFloatArrayElements(audio_data, audio_data_arr, JNI_ABORT);
    env->ReleaseStringUTFChars(language, language_chars);
    if (rn_whisper_transcribe_is_aborted(job_id)) {
        code = -999;
    }
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
