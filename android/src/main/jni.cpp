#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <fbjni/fbjni.h>
#include <ReactCommon/CallInvokerHolder.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "whisper.h"
#include "rn-whisper.h"
#include "RNWhisperJSI.h"

namespace {

JavaVM *g_javaVm = nullptr;
jobject g_applicationContext = nullptr;
jobject g_assetManager = nullptr;

constexpr const char *kTag = "RNWhisperJNI";

bool isRemoteUrl(const std::string &path) {
    return path.rfind("http://", 0) == 0 || path.rfind("https://", 0) == 0;
}

std::string stripAssetPrefix(const std::string &path) {
    static const std::string assetPrefix = "asset:/";
    static const std::string androidAssetPrefix = "/android_asset/";
    static const std::string fileAndroidAssetPrefix = "file:///android_asset/";

    if (path.rfind(assetPrefix, 0) == 0) {
        return path.substr(assetPrefix.size());
    }
    if (path.rfind(fileAndroidAssetPrefix, 0) == 0) {
        return path.substr(fileAndroidAssetPrefix.size());
    }
    if (path.rfind(androidAssetPrefix, 0) == 0) {
        return path.substr(androidAssetPrefix.size());
    }
    return path;
}

bool isAssetPath(const std::string &path) {
    return path.rfind("asset:/", 0) == 0 ||
           path.rfind("file:///android_asset/", 0) == 0 ||
           path.rfind("/android_asset/", 0) == 0;
}

std::string toStdString(JNIEnv *env, jstring value) {
    if (!value) {
        return {};
    }
    const char *chars = env->GetStringUTFChars(value, nullptr);
    std::string stringValue = chars ? std::string(chars) : std::string();
    if (chars) {
        env->ReleaseStringUTFChars(value, chars);
    }
    return stringValue;
}

jobject getApplicationContext() {
    return g_applicationContext;
}

jobject getAssetManager() {
    return g_assetManager;
}

JNIEnv *getEnv(bool *needsDetach = nullptr) {
    if (needsDetach) {
        *needsDetach = false;
    }
    if (!g_javaVm) {
        return nullptr;
    }

    JNIEnv *env = nullptr;
    jint status = g_javaVm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6);
    if (status == JNI_OK) {
        return env;
    }
    if (status == JNI_EDETACHED) {
        if (g_javaVm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
            if (needsDetach) {
                *needsDetach = true;
            }
            return env;
        }
    }
    return nullptr;
}

void detachThreadIfNeeded(bool needsDetach) {
    if (needsDetach && g_javaVm) {
        g_javaVm->DetachCurrentThread();
    }
}

std::vector<uint8_t> readAllBytesFromInputStream(JNIEnv *env, jobject inputStream) {
    std::vector<uint8_t> bytes;
    if (!env || !inputStream) {
        return bytes;
    }

    jclass inputStreamClass = env->GetObjectClass(inputStream);
    jmethodID readMethod = env->GetMethodID(inputStreamClass, "read", "([B)I");
    jmethodID closeMethod = env->GetMethodID(inputStreamClass, "close", "()V");
    jbyteArray buffer = env->NewByteArray(8192);

    while (true) {
        jint read = env->CallIntMethod(inputStream, readMethod, buffer);
        if (read <= 0) {
            break;
        }
        size_t start = bytes.size();
        bytes.resize(start + static_cast<size_t>(read));
        env->GetByteArrayRegion(
            buffer,
            0,
            read,
            reinterpret_cast<jbyte *>(bytes.data() + start));
    }

    env->CallVoidMethod(inputStream, closeMethod);
    env->DeleteLocalRef(buffer);
    env->DeleteLocalRef(inputStreamClass);
    return bytes;
}

struct input_stream_context {
    JNIEnv *env;
    jobject input_stream;
};

size_t inputStreamRead(void *ctx, void *output, size_t readSize) {
    auto *context = static_cast<input_stream_context *>(ctx);
    JNIEnv *env = context->env;
    jobject inputStream = context->input_stream;
    jclass inputStreamClass = env->GetObjectClass(inputStream);

    jbyteArray buffer = env->NewByteArray(static_cast<jsize>(readSize));
    jint bytesRead = env->CallIntMethod(
        inputStream,
        env->GetMethodID(inputStreamClass, "read", "([B)I"),
        buffer);

    if (bytesRead > 0) {
        env->GetByteArrayRegion(
            buffer,
            0,
            bytesRead,
            reinterpret_cast<jbyte *>(output));
    }

    env->DeleteLocalRef(buffer);
    env->DeleteLocalRef(inputStreamClass);
    return bytesRead > 0 ? static_cast<size_t>(bytesRead) : 0;
}

bool inputStreamIsEof(void *ctx) {
    auto *context = static_cast<input_stream_context *>(ctx);
    JNIEnv *env = context->env;
    jobject inputStream = context->input_stream;
    jclass inputStreamClass = env->GetObjectClass(inputStream);

    jbyteArray buffer = env->NewByteArray(1);
    jint bytesRead = env->CallIntMethod(
        inputStream,
        env->GetMethodID(inputStreamClass, "read", "([B)I"),
        buffer);

    bool isEof = bytesRead == -1;
    if (!isEof) {
        env->CallVoidMethod(
            inputStream,
            env->GetMethodID(inputStreamClass, "unread", "([BII)V"),
            buffer,
            0,
            1);
    }

    env->DeleteLocalRef(buffer);
    env->DeleteLocalRef(inputStreamClass);
    return isEof;
}

void inputStreamClose(void *ctx) {
    auto *context = static_cast<input_stream_context *>(ctx);
    JNIEnv *env = context->env;
    jobject inputStream = context->input_stream;
    jclass inputStreamClass = env->GetObjectClass(inputStream);
    env->CallVoidMethod(
        inputStream,
        env->GetMethodID(inputStreamClass, "close", "()V"));
    env->DeleteLocalRef(inputStreamClass);
    env->DeleteGlobalRef(inputStream);
    delete context;
}

whisper_context *whisperInitFromInputStream(
    JNIEnv *env,
    jobject inputStream,
    whisper_context_params params) {
    auto *context = new input_stream_context;
    context->env = env;
    context->input_stream = env->NewGlobalRef(inputStream);

    whisper_model_loader loader = {
        .context = context,
        .read = &inputStreamRead,
        .eof = &inputStreamIsEof,
        .close = &inputStreamClose,
    };
    return whisper_init_with_params(&loader, params);
}

whisper_vad_context *whisperVadInitFromInputStream(
    JNIEnv *env,
    jobject inputStream,
    whisper_vad_context_params params) {
    auto *context = new input_stream_context;
    context->env = env;
    context->input_stream = env->NewGlobalRef(inputStream);

    whisper_model_loader loader = {
        .context = context,
        .read = &inputStreamRead,
        .eof = &inputStreamIsEof,
        .close = &inputStreamClose,
    };
    return whisper_vad_init_with_params(&loader, params);
}

size_t assetRead(void *ctx, void *output, size_t readSize) {
    return AAsset_read(static_cast<AAsset *>(ctx), output, readSize);
}

bool assetIsEof(void *ctx) {
    return AAsset_getRemainingLength64(static_cast<AAsset *>(ctx)) <= 0;
}

void assetClose(void *ctx) {
    AAsset_close(static_cast<AAsset *>(ctx));
}

whisper_context *whisperInitFromAsset(
    JNIEnv *env,
    jobject assetManager,
    const std::string &assetPath,
    whisper_context_params params) {
    auto *manager = AAssetManager_fromJava(env, assetManager);
    AAsset *asset = AAssetManager_open(manager, assetPath.c_str(), AASSET_MODE_STREAMING);
    if (!asset) {
        __android_log_print(ANDROID_LOG_WARN, kTag, "Failed to open asset %s", assetPath.c_str());
        return nullptr;
    }

    whisper_model_loader loader = {
        .context = asset,
        .read = &assetRead,
        .eof = &assetIsEof,
        .close = &assetClose,
    };
    return whisper_init_with_params(&loader, params);
}

whisper_vad_context *whisperVadInitFromAsset(
    JNIEnv *env,
    jobject assetManager,
    const std::string &assetPath,
    whisper_vad_context_params params) {
    auto *manager = AAssetManager_fromJava(env, assetManager);
    AAsset *asset = AAssetManager_open(manager, assetPath.c_str(), AASSET_MODE_STREAMING);
    if (!asset) {
        __android_log_print(ANDROID_LOG_WARN, kTag, "Failed to open VAD asset %s", assetPath.c_str());
        return nullptr;
    }

    whisper_model_loader loader = {
        .context = asset,
        .read = &assetRead,
        .eof = &assetIsEof,
        .close = &assetClose,
    };
    return whisper_vad_init_with_params(&loader, params);
}

jobject openPushbackInputStreamForResource(JNIEnv *env, int resourceId) {
    jobject context = getApplicationContext();
    if (!context || resourceId == 0) {
        return nullptr;
    }

    jclass contextClass = env->GetObjectClass(context);
    jobject resources = env->CallObjectMethod(
        context,
        env->GetMethodID(contextClass, "getResources", "()Landroid/content/res/Resources;"));
    env->DeleteLocalRef(contextClass);
    if (!resources) {
        return nullptr;
    }

    jclass resourcesClass = env->GetObjectClass(resources);
    jobject inputStream = env->CallObjectMethod(
        resources,
        env->GetMethodID(resourcesClass, "openRawResource", "(I)Ljava/io/InputStream;"),
        resourceId);
    env->DeleteLocalRef(resourcesClass);
    env->DeleteLocalRef(resources);
    if (!inputStream) {
        return nullptr;
    }

    jclass pushbackClass = env->FindClass("java/io/PushbackInputStream");
    jobject pushbackStream = env->NewObject(
        pushbackClass,
        env->GetMethodID(pushbackClass, "<init>", "(Ljava/io/InputStream;)V"),
        inputStream);
    env->DeleteLocalRef(pushbackClass);
    env->DeleteLocalRef(inputStream);
    return pushbackStream;
}

std::string getCacheDirectory(JNIEnv *env) {
    jobject context = getApplicationContext();
    if (!context) {
        return {};
    }

    jclass contextClass = env->GetObjectClass(context);
    jobject cacheDir = env->CallObjectMethod(
        context,
        env->GetMethodID(contextClass, "getCacheDir", "()Ljava/io/File;"));
    env->DeleteLocalRef(contextClass);
    if (!cacheDir) {
        return {};
    }

    jclass fileClass = env->GetObjectClass(cacheDir);
    jstring absolutePath = static_cast<jstring>(env->CallObjectMethod(
        cacheDir,
        env->GetMethodID(fileClass, "getAbsolutePath", "()Ljava/lang/String;")));
    std::string path = toStdString(env, absolutePath);
    env->DeleteLocalRef(absolutePath);
    env->DeleteLocalRef(fileClass);
    env->DeleteLocalRef(cacheDir);
    return path;
}

std::string getCacheRoot(JNIEnv *env) {
    std::string cacheDir = getCacheDirectory(env);
    if (cacheDir.empty()) {
        return {};
    }
    return cacheDir + "/rnwhisper_debug_assets";
}

std::string basenameWithoutQuery(const std::string &path) {
    std::string basename = path;
    size_t slash = basename.find_last_of('/');
    if (slash != std::string::npos) {
        basename = basename.substr(slash + 1);
    }
    size_t query = basename.find('?');
    if (query != std::string::npos) {
        basename = basename.substr(0, query);
    }
    return basename;
}

int getResourceIdentifier(JNIEnv *env, const std::string &path) {
    jobject context = getApplicationContext();
    if (!context) {
        return 0;
    }

    std::vector<std::string> candidates;
    candidates.push_back(path);
    candidates.push_back(basenameWithoutQuery(path));
    std::string basename = basenameWithoutQuery(path);
    size_t extension = basename.find('.');
    if (extension != std::string::npos) {
        candidates.push_back(basename.substr(0, extension));
    }

    jclass contextClass = env->GetObjectClass(context);
    jobject resources = env->CallObjectMethod(
        context,
        env->GetMethodID(contextClass, "getResources", "()Landroid/content/res/Resources;"));
    jstring packageName = static_cast<jstring>(env->CallObjectMethod(
        context,
        env->GetMethodID(contextClass, "getPackageName", "()Ljava/lang/String;")));

    jclass resourcesClass = env->GetObjectClass(resources);
    jmethodID identifierMethod = env->GetMethodID(
        resourcesClass,
        "getIdentifier",
        "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)I");

    int identifier = 0;
    for (const auto &candidate : candidates) {
      if (candidate.empty()) continue;
      jstring name = env->NewStringUTF(candidate.c_str());
      jstring drawable = env->NewStringUTF("drawable");
      identifier = env->CallIntMethod(resources, identifierMethod, name, drawable, packageName);
      env->DeleteLocalRef(drawable);

      if (identifier == 0) {
        jstring raw = env->NewStringUTF("raw");
        identifier = env->CallIntMethod(resources, identifierMethod, name, raw, packageName);
        env->DeleteLocalRef(raw);
      }

      env->DeleteLocalRef(name);
      if (identifier != 0) {
        break;
      }
    }

    env->DeleteLocalRef(resourcesClass);
    env->DeleteLocalRef(packageName);
    env->DeleteLocalRef(resources);
    env->DeleteLocalRef(contextClass);
    return identifier;
}

std::vector<uint8_t> readFileBytes(const std::string &filePath) {
    std::ifstream stream(filePath, std::ios::binary);
    if (!stream) {
        return {};
    }
    return std::vector<uint8_t>(
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>());
}

std::vector<uint8_t> readAssetBytes(JNIEnv *env, const std::string &assetPath) {
    jobject assetManager = getAssetManager();
    if (!assetManager) {
        return {};
    }
    auto *manager = AAssetManager_fromJava(env, assetManager);
    AAsset *asset = AAssetManager_open(manager, assetPath.c_str(), AASSET_MODE_STREAMING);
    if (!asset) {
        return {};
    }

    off_t length = AAsset_getLength(asset);
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    if (length > 0) {
        AAsset_read(asset, bytes.data(), length);
    }
    AAsset_close(asset);
    return bytes;
}

std::string downloadToCache(
    JNIEnv *env,
    const std::string &url,
    const std::string &relativePath) {
    std::string cacheRoot = getCacheRoot(env);
    if (cacheRoot.empty()) {
        return {};
    }

    std::string filename = relativePath.empty()
        ? basenameWithoutQuery(url)
        : relativePath;
    if (filename.empty()) {
        filename = "download.bin";
    }

    std::filesystem::path target =
        std::filesystem::path(cacheRoot) / std::filesystem::path(filename);
    std::filesystem::create_directories(target.parent_path());
    if (std::filesystem::exists(target)) {
        return target.string();
    }

    jclass urlClass = env->FindClass("java/net/URL");
    jstring urlString = env->NewStringUTF(url.c_str());
    jobject urlObject = env->NewObject(
        urlClass,
        env->GetMethodID(urlClass, "<init>", "(Ljava/lang/String;)V"),
        urlString);
    jobject inputStream = env->CallObjectMethod(
        urlObject,
        env->GetMethodID(urlClass, "openStream", "()Ljava/io/InputStream;"));
    env->DeleteLocalRef(urlString);

    std::vector<uint8_t> bytes = readAllBytesFromInputStream(env, inputStream);
    env->DeleteLocalRef(inputStream);
    env->DeleteLocalRef(urlObject);
    env->DeleteLocalRef(urlClass);

    if (bytes.empty()) {
        return {};
    }

    std::ofstream output(target, std::ios::binary);
    output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    output.close();
    return target.string();
}

} // namespace

namespace rnwhisper_jsi {

void setAndroidContext(JNIEnv *env, jobject applicationContext, jobject assetManager) {
    if (!env) {
        return;
    }
    env->GetJavaVM(&g_javaVm);

    if (g_applicationContext) {
        env->DeleteGlobalRef(g_applicationContext);
        g_applicationContext = nullptr;
    }
    if (g_assetManager) {
        env->DeleteGlobalRef(g_assetManager);
        g_assetManager = nullptr;
    }

    g_applicationContext = env->NewGlobalRef(applicationContext);
    g_assetManager = env->NewGlobalRef(assetManager);
}

WhisperContextInitResult hostInitWhisperContext(
    const WhisperContextInitOptions &options) {
    WhisperContextInitResult result;
    bool needsDetach = false;
    JNIEnv *env = getEnv(&needsDetach);
    if (!env) {
        return result;
    }

    auto params = whisper_context_default_params();
    params.dtw_token_timestamps = false;
    params.use_gpu = false;
    params.flash_attn = options.useFlashAttn;
    params.use_coreml = false;

    if (options.useGpu) {
        result.reasonNoGPU = "Currently not supported";
    }

    std::string modelPath = options.filePath;
    if (isRemoteUrl(modelPath)) {
        modelPath = downloadToCache(env, modelPath, "");
    }

    if (options.isBundleAsset || isAssetPath(modelPath)) {
        result.context = whisperInitFromAsset(
            env,
            getAssetManager(),
            stripAssetPrefix(modelPath),
            params);
    } else {
        int resourceId = getResourceIdentifier(env, modelPath);
        if (resourceId != 0) {
            jobject pushbackStream = openPushbackInputStreamForResource(env, resourceId);
            if (pushbackStream) {
                result.context = whisperInitFromInputStream(env, pushbackStream, params);
                env->DeleteLocalRef(pushbackStream);
            }
        } else if (!modelPath.empty()) {
            result.context =
                whisper_init_from_file_with_params(modelPath.c_str(), params);
        }
    }

    detachThreadIfNeeded(needsDetach);
    return result;
}

WhisperVadContextInitResult hostInitWhisperVadContext(
    const WhisperVadContextInitOptions &options) {
    WhisperVadContextInitResult result;
    bool needsDetach = false;
    JNIEnv *env = getEnv(&needsDetach);
    if (!env) {
        return result;
    }

    auto params = whisper_vad_default_context_params();
    params.use_gpu = false;
    if (options.nThreads > 0) {
        params.n_threads = options.nThreads;
    }
    if (options.useGpu) {
        result.reasonNoGPU = "Currently not supported";
    }

    std::string modelPath = options.filePath;
    if (isRemoteUrl(modelPath)) {
        modelPath = downloadToCache(env, modelPath, "");
    }

    if (options.isBundleAsset || isAssetPath(modelPath)) {
        result.context = whisperVadInitFromAsset(
            env,
            getAssetManager(),
            stripAssetPrefix(modelPath),
            params);
    } else {
        int resourceId = getResourceIdentifier(env, modelPath);
        if (resourceId != 0) {
            jobject pushbackStream = openPushbackInputStreamForResource(env, resourceId);
            if (pushbackStream) {
                result.context = whisperVadInitFromInputStream(env, pushbackStream, params);
                env->DeleteLocalRef(pushbackStream);
            }
        } else if (!modelPath.empty()) {
            result.context =
                whisper_vad_init_from_file_with_params(modelPath.c_str(), params);
        }
    }

    detachThreadIfNeeded(needsDetach);
    return result;
}

std::vector<uint8_t> hostLoadFileBytes(const std::string &path) {
    bool needsDetach = false;
    JNIEnv *env = getEnv(&needsDetach);
    if (!env) {
        return {};
    }

    std::string resolvedPath = path;
    if (isRemoteUrl(resolvedPath)) {
        resolvedPath = downloadToCache(env, resolvedPath, "");
    }

    std::vector<uint8_t> bytes;
    if (isAssetPath(resolvedPath)) {
        bytes = readAssetBytes(env, stripAssetPrefix(resolvedPath));
    } else {
        int resourceId = getResourceIdentifier(env, resolvedPath);
        if (resourceId != 0) {
            jobject pushbackStream = openPushbackInputStreamForResource(env, resourceId);
            if (pushbackStream) {
                bytes = readAllBytesFromInputStream(env, pushbackStream);
                env->DeleteLocalRef(pushbackStream);
            }
        } else {
            bytes = readFileBytes(resolvedPath);
        }
    }

    detachThreadIfNeeded(needsDetach);
    return bytes;
}

void hostClearCache() {
    bool needsDetach = false;
    JNIEnv *env = getEnv(&needsDetach);
    if (!env) {
        return;
    }
    std::string cacheRoot = getCacheRoot(env);
    if (!cacheRoot.empty()) {
        std::filesystem::remove_all(cacheRoot);
    }
    detachThreadIfNeeded(needsDetach);
}

} // namespace rnwhisper_jsi

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *) {
    g_javaVm = vm;
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT void JNICALL
Java_com_rnwhisper_RNWhisperModule_installJSIBindings(
    JNIEnv *env,
    jobject,
    jlong runtimePtr,
    jobject callInvokerHolder,
    jobject applicationContext,
    jobject assetManager) {
    if (runtimePtr == 0 || callInvokerHolder == nullptr || applicationContext == nullptr || assetManager == nullptr) {
        return;
    }

    rnwhisper_jsi::setAndroidContext(env, applicationContext, assetManager);

    auto *runtime = reinterpret_cast<facebook::jsi::Runtime *>(runtimePtr);
    auto holder = facebook::jni::alias_ref<facebook::react::CallInvokerHolder::javaobject>{
        reinterpret_cast<facebook::react::CallInvokerHolder::javaobject>(callInvokerHolder)
    };
    auto callInvoker = holder->cthis()->getCallInvoker();
    if (!callInvoker) {
        return;
    }

    callInvoker->invokeAsync([runtime, callInvoker]() {
        rnwhisper_jsi::installJSIBindings(*runtime, callInvoker);
    });
}

extern "C" JNIEXPORT void JNICALL
Java_com_rnwhisper_RNWhisperModule_cleanupJSIBindings(JNIEnv *env, jobject) {
    rnwhisper_jsi::cleanupJSIBindings();
    if (g_applicationContext) {
        env->DeleteGlobalRef(g_applicationContext);
        g_applicationContext = nullptr;
    }
    if (g_assetManager) {
        env->DeleteGlobalRef(g_assetManager);
        g_assetManager = nullptr;
    }
}
