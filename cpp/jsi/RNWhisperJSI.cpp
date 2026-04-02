#include "RNWhisperJSI.h"
#include "ThreadPool.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__ANDROID__)
#include <android/log.h>
#endif

using namespace facebook;

namespace {

enum class LogLevel { Debug, Info, Error };

void logMessage(LogLevel level, const char *format, ...) {
    va_list args;
    va_start(args, format);

#if defined(__ANDROID__)
    int androidLevel = ANDROID_LOG_INFO;
    if (level == LogLevel::Debug) {
        androidLevel = ANDROID_LOG_DEBUG;
    } else if (level == LogLevel::Error) {
        androidLevel = ANDROID_LOG_ERROR;
    }
    __android_log_vprint(androidLevel, "RNWhisperJSI", format, args);
#else
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    const char *levelName = "INFO";
    if (level == LogLevel::Debug) {
        levelName = "DEBUG";
    } else if (level == LogLevel::Error) {
        levelName = "ERROR";
    }
    std::printf("RNWhisperJSI %s: %s\n", levelName, buffer);
#endif

    va_end(args);
}

#define LOG_INFO(format, ...) logMessage(LogLevel::Info, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) logMessage(LogLevel::Error, format, ##__VA_ARGS__)

class JsiError : public std::runtime_error {
public:
    explicit JsiError(const std::string &message, int errorCode = -1)
        : std::runtime_error(message), code(errorCode) {}

    int code;
};

std::atomic<bool> g_isShuttingDown{false};
std::unique_ptr<ThreadPool> g_threadPool;
std::mutex g_threadPoolMutex;

ThreadPool &getThreadPool() {
    std::lock_guard<std::mutex> lock(g_threadPoolMutex);
    if (!g_threadPool) {
        int maxThreads = static_cast<int>(std::thread::hardware_concurrency());
        int threadCount = std::max(2, std::min(4, maxThreads));
        g_threadPool = std::make_unique<ThreadPool>(threadCount);
        LOG_INFO("Initialized ThreadPool with %d threads", threadCount);
    }
    return *g_threadPool;
}

template <typename HolderType>
class ContextManager {
public:
    void add(int contextId, std::shared_ptr<HolderType> holder) {
        std::lock_guard<std::mutex> lock(mutex_);
        contexts_[contextId] = std::move(holder);
    }

    std::shared_ptr<HolderType> get(int contextId) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = contexts_.find(contextId);
        return it == contexts_.end() ? nullptr : it->second;
    }

    std::shared_ptr<HolderType> remove(int contextId) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = contexts_.find(contextId);
        if (it == contexts_.end()) {
            return nullptr;
        }
        auto holder = it->second;
        contexts_.erase(it);
        return holder;
    }

    std::vector<std::shared_ptr<HolderType>> removeAll() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::shared_ptr<HolderType>> holders;
        holders.reserve(contexts_.size());
        for (const auto &entry : contexts_) {
            holders.push_back(entry.second);
        }
        contexts_.clear();
        return holders;
    }

private:
    std::unordered_map<int, std::shared_ptr<HolderType>> contexts_;
    std::mutex mutex_;
};

struct SegmentData {
    std::string text;
    int t0 = 0;
    int t1 = 0;
};

struct TranscribeResultData {
    std::string language;
    std::string result;
    std::vector<SegmentData> segments;
    bool isAborted = false;
};

struct NewSegmentsData {
    int nNew = 0;
    int totalNNew = 0;
    std::string result;
    std::vector<SegmentData> segments;
};

struct VadSegmentData {
    float t0 = 0;
    float t1 = 0;
};

struct VadResultData {
    bool hasSpeech = false;
    std::vector<VadSegmentData> segments;
};

struct ContextLifecycle {
    void retainTask() {
        std::lock_guard<std::mutex> lock(mutex);
        pendingTasks += 1;
    }

    void releaseTask() {
        std::lock_guard<std::mutex> lock(mutex);
        pendingTasks -= 1;
        if (pendingTasks <= 0) {
            pendingTasks = 0;
            condition.notify_all();
        }
    }

    void waitForIdle() {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [this]() { return pendingTasks == 0; });
    }

    std::mutex mutex;
    std::condition_variable condition;
    int pendingTasks = 0;
};

struct WhisperContextHolder : public ContextLifecycle {
    explicit WhisperContextHolder(int contextId)
        : id(contextId) {}

    bool beginExclusiveOperation(int jobId) {
        std::lock_guard<std::mutex> lock(operationMutex);
        if (busy) {
            return false;
        }
        busy = true;
        activeJobId = jobId;
        return true;
    }

    void endExclusiveOperation() {
        std::lock_guard<std::mutex> lock(operationMutex);
        busy = false;
        activeJobId = -1;
    }

    void abortActiveJob() {
        std::lock_guard<std::mutex> lock(operationMutex);
        if (activeJobId < 0) {
            return;
        }
        if (auto *job = rnwhisper::job_get(activeJobId)) {
            job->abort();
        }
    }

    int id = 0;
    whisper_context *context = nullptr;
    long ptr = 0;
    bool gpu = false;
    std::string reasonNoGPU;

    std::mutex operationMutex;
    bool busy = false;
    int activeJobId = -1;
};

struct WhisperVadContextHolder : public ContextLifecycle {
    explicit WhisperVadContextHolder(int contextId)
        : id(contextId) {}

    int id = 0;
    whisper_vad_context *context = nullptr;
    long ptr = 0;
    bool gpu = false;
    std::string reasonNoGPU;
};

ContextManager<WhisperContextHolder> g_whisperContexts;
ContextManager<WhisperVadContextHolder> g_vadContexts;

using PromiseResultGenerator = std::function<jsi::Value(jsi::Runtime &)>;
using PromiseTask = std::function<PromiseResultGenerator()>;
using JsiFunctionPtr = std::shared_ptr<jsi::Function>;

struct JsiCallbackState {
    std::shared_ptr<react::CallInvoker> callInvoker;
    JsiFunctionPtr callback;
    std::shared_ptr<jsi::Runtime> runtime;
};

struct SegmentCallbackState : public JsiCallbackState {
    bool tdrzEnable = false;
    int totalNNew = 0;
};

jsi::Value createNewSegmentsValue(
    jsi::Runtime &runtime,
    const NewSegmentsData &data);

std::mutex g_logMutex;
std::weak_ptr<react::CallInvoker> g_logInvoker;
std::shared_ptr<jsi::Function> g_logHandler;
std::shared_ptr<jsi::Runtime> g_logRuntime;

class PromiseScopeGuard {
public:
    explicit PromiseScopeGuard(std::function<void()> onExit)
        : onExit_(std::move(onExit)) {}

    ~PromiseScopeGuard() {
        if (active_ && onExit_) {
            onExit_();
        }
    }

    void dismiss() {
        active_ = false;
    }

private:
    std::function<void()> onExit_;
    bool active_ = true;
};

JsiFunctionPtr makeJsiFunction(
    jsi::Runtime &runtime,
    const jsi::Value &value,
    const std::shared_ptr<react::CallInvoker> &callInvoker) {
    if (!value.isObject() || !value.asObject(runtime).isFunction(runtime)) {
        return nullptr;
    }

    auto *fn = new jsi::Function(value.asObject(runtime).asFunction(runtime));
    std::weak_ptr<react::CallInvoker> weakInvoker = callInvoker;

    return JsiFunctionPtr(fn, [weakInvoker](jsi::Function *ptr) {
        if (!ptr || g_isShuttingDown.load(std::memory_order_relaxed)) {
            return;
        }

        auto invoker = weakInvoker.lock();
        if (!invoker) {
            return;
        }

        try {
            invoker->invokeAsync([ptr]() {
                delete ptr;
            });
        } catch (...) {
            // Runtime is shutting down; leak rather than delete on a non-JS thread.
        }
    });
}

jsi::Object createErrorObject(
    jsi::Runtime &runtime,
    const std::string &message,
    int code = -1) {
    jsi::Object error(runtime);
    error.setProperty(
        runtime,
        "message",
        jsi::String::createFromUtf8(runtime, message));
    if (code != -1) {
        error.setProperty(runtime, "code", jsi::Value(code));
    }
    return error;
}

jsi::Value createPromiseTask(
    jsi::Runtime &runtime,
    const std::shared_ptr<react::CallInvoker> &callInvoker,
    PromiseTask task) {
    auto promiseCtor =
        runtime.global().getPropertyAsObject(runtime, "Promise").asFunction(runtime);
    auto runtimePtr = std::shared_ptr<jsi::Runtime>(&runtime, [](jsi::Runtime *) {});

    return promiseCtor.callAsConstructor(
        runtime,
        jsi::Function::createFromHostFunction(
            runtime,
            jsi::PropNameID::forAscii(runtime, "executor"),
            2,
            [callInvoker, task, runtimePtr](
                jsi::Runtime &runtime,
                const jsi::Value &,
                const jsi::Value *arguments,
                size_t count) -> jsi::Value {
                if (count != 2) {
                    throw jsi::JSError(runtime, "Promise executor expects resolve and reject");
                }

                auto resolve = makeJsiFunction(runtime, arguments[0], callInvoker);
                auto reject = makeJsiFunction(runtime, arguments[1], callInvoker);

                try {
                    getThreadPool().enqueue([callInvoker, task, resolve, reject, runtimePtr]() {
                        if (g_isShuttingDown.load(std::memory_order_relaxed)) {
                            return;
                        }

                        try {
                            auto resultGenerator = task();
                            if (g_isShuttingDown.load(std::memory_order_relaxed)) {
                                return;
                            }
                            callInvoker->invokeAsync([resolve, resultGenerator, runtimePtr]() {
                                if (g_isShuttingDown.load(std::memory_order_relaxed) || !resolve) {
                                    return;
                                }
                                auto &rt = *runtimePtr;
                                resolve->call(rt, resultGenerator(rt));
                            });
                        } catch (const JsiError &error) {
                            try {
                                callInvoker->invokeAsync([reject, runtimePtr, message = std::string(error.what()), code = error.code]() {
                                    if (g_isShuttingDown.load(std::memory_order_relaxed) || !reject) {
                                        return;
                                    }
                                    auto &rt = *runtimePtr;
                                    reject->call(rt, createErrorObject(rt, message, code));
                                });
                            } catch (...) {
                            }
                        } catch (const std::exception &error) {
                            try {
                                callInvoker->invokeAsync([reject, runtimePtr, message = std::string(error.what())]() {
                                    if (g_isShuttingDown.load(std::memory_order_relaxed) || !reject) {
                                        return;
                                    }
                                    auto &rt = *runtimePtr;
                                    reject->call(rt, createErrorObject(rt, message));
                                });
                            } catch (...) {
                            }
                        } catch (...) {
                            try {
                                callInvoker->invokeAsync([reject, runtimePtr]() {
                                    if (g_isShuttingDown.load(std::memory_order_relaxed) || !reject) {
                                        return;
                                    }
                                    auto &rt = *runtimePtr;
                                    reject->call(rt, createErrorObject(rt, "Unknown error"));
                                });
                            } catch (...) {
                            }
                        }
                    });
                } catch (const std::exception &error) {
                    if (reject) {
                        auto errorObject = createErrorObject(runtime, error.what());
                        reject->call(runtime, errorObject);
                    }
                }

                return jsi::Value::undefined();
            }));
}

void invokeAsync(
    const std::shared_ptr<react::CallInvoker> &callInvoker,
    std::function<void()> callback) {
    if (g_isShuttingDown.load(std::memory_order_relaxed)) {
        return;
    }

    try {
        callInvoker->invokeAsync([callback = std::move(callback)]() {
            if (g_isShuttingDown.load(std::memory_order_relaxed)) {
                return;
            }
            callback();
        });
    } catch (...) {
    }
}

void emitProgressCallback(
    const std::shared_ptr<JsiCallbackState> &state,
    int progress) {
    if (!state || !state->callInvoker || !state->callback || !state->runtime) {
        return;
    }

    invokeAsync(state->callInvoker, [state, progress]() {
        if (g_isShuttingDown.load(std::memory_order_relaxed)) {
            return;
        }
        auto &rt = *state->runtime;
        state->callback->call(rt, jsi::Value(progress));
    });
}

void emitNewSegmentsCallback(
    const std::shared_ptr<SegmentCallbackState> &state,
    NewSegmentsData payload) {
    if (!state || !state->callInvoker || !state->callback || !state->runtime) {
        return;
    }

    invokeAsync(state->callInvoker, [state, payload = std::move(payload)]() {
        if (g_isShuttingDown.load(std::memory_order_relaxed)) {
            return;
        }
        auto &rt = *state->runtime;
        state->callback->call(rt, createNewSegmentsValue(rt, payload));
    });
}

bool getBoolProperty(
    jsi::Runtime &runtime,
    const jsi::Object &object,
    const char *name,
    bool fallback) {
    if (!object.hasProperty(runtime, name)) {
        return fallback;
    }
    auto value = object.getProperty(runtime, name);
    return value.isBool() ? value.getBool() : fallback;
}

int getIntProperty(
    jsi::Runtime &runtime,
    const jsi::Object &object,
    const char *name,
    int fallback) {
    if (!object.hasProperty(runtime, name)) {
        return fallback;
    }
    auto value = object.getProperty(runtime, name);
    return value.isNumber() ? static_cast<int>(value.asNumber()) : fallback;
}

float getFloatProperty(
    jsi::Runtime &runtime,
    const jsi::Object &object,
    const char *name,
    float fallback) {
    if (!object.hasProperty(runtime, name)) {
        return fallback;
    }
    auto value = object.getProperty(runtime, name);
    return value.isNumber() ? static_cast<float>(value.asNumber()) : fallback;
}

std::string getStringProperty(
    jsi::Runtime &runtime,
    const jsi::Object &object,
    const char *name,
    const std::string &fallback = "") {
    if (!object.hasProperty(runtime, name)) {
        return fallback;
    }
    auto value = object.getProperty(runtime, name);
    return value.isString() ? value.asString(runtime).utf8(runtime) : fallback;
}

struct TranscribeConfig {
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    std::string prompt;
    std::string language;
    int nProcessors = 1;
    int jobId = 0;
    bool tdrzEnable = false;
    JsiFunctionPtr onProgress;
    JsiFunctionPtr onNewSegments;
};

TranscribeConfig createTranscribeConfig(
    jsi::Runtime &runtime,
    const jsi::Object &options,
    const std::shared_ptr<react::CallInvoker> &callInvoker) {
    TranscribeConfig config;

    config.params.print_realtime = false;
    config.params.print_progress = false;
    config.params.print_timestamps = false;
    config.params.print_special = false;

    int maxThreads = static_cast<int>(std::thread::hardware_concurrency());
    int defaultThreads = maxThreads == 4 ? 2 : std::min(4, maxThreads);
    int nThreads = getIntProperty(runtime, options, "maxThreads", defaultThreads);
    config.params.n_threads = nThreads > 0 ? nThreads : defaultThreads;
    config.params.translate = getBoolProperty(runtime, options, "translate", false);
    config.params.token_timestamps =
        getBoolProperty(runtime, options, "tokenTimestamps", false);
    config.params.tdrz_enable = getBoolProperty(runtime, options, "tdrzEnable", false);
    config.tdrzEnable = config.params.tdrz_enable;
    config.params.max_len = getIntProperty(runtime, options, "maxLen", config.params.max_len);
    config.params.n_max_text_ctx =
        getIntProperty(runtime, options, "maxContext", config.params.n_max_text_ctx);
    config.params.offset_ms =
        getIntProperty(runtime, options, "offset", config.params.offset_ms);
    config.params.duration_ms =
        getIntProperty(runtime, options, "duration", config.params.duration_ms);
    config.params.thold_pt =
        getFloatProperty(runtime, options, "wordThold", config.params.thold_pt);
    config.params.temperature =
        getFloatProperty(runtime, options, "temperature", config.params.temperature);
    config.params.temperature_inc = getFloatProperty(
        runtime,
        options,
        "temperatureInc",
        config.params.temperature_inc);
    config.params.greedy.best_of =
        getIntProperty(runtime, options, "bestOf", config.params.greedy.best_of);
    config.nProcessors = std::max(1, getIntProperty(runtime, options, "nProcessors", 1));
    config.jobId = getIntProperty(
        runtime,
        options,
        "jobId",
        static_cast<int>(std::rand()));

    int beamSize = getIntProperty(runtime, options, "beamSize", -1);
    if (beamSize > 0) {
        config.params.strategy = WHISPER_SAMPLING_BEAM_SEARCH;
        config.params.beam_search.beam_size = beamSize;
    }

    config.prompt = getStringProperty(runtime, options, "prompt");
    if (!config.prompt.empty()) {
        config.params.initial_prompt = config.prompt.c_str();
    }

    config.language = getStringProperty(runtime, options, "language");
    if (!config.language.empty()) {
        config.params.language = config.language.c_str();
    }

    config.params.no_context = true;
    config.params.single_segment = false;

    if (options.hasProperty(runtime, "onProgress")) {
        config.onProgress = makeJsiFunction(
            runtime,
            options.getProperty(runtime, "onProgress"),
            callInvoker);
    }

    if (options.hasProperty(runtime, "onNewSegments")) {
        config.onNewSegments = makeJsiFunction(
            runtime,
            options.getProperty(runtime, "onNewSegments"),
            callInvoker);
    }

    return config;
}

whisper_vad_params createVadParams(
    jsi::Runtime &runtime,
    const jsi::Object &options) {
    whisper_vad_params params = whisper_vad_default_params();
    params.threshold =
        getFloatProperty(runtime, options, "threshold", params.threshold);
    params.min_speech_duration_ms = getIntProperty(
        runtime,
        options,
        "minSpeechDurationMs",
        params.min_speech_duration_ms);
    params.min_silence_duration_ms = getIntProperty(
        runtime,
        options,
        "minSilenceDurationMs",
        params.min_silence_duration_ms);
    params.max_speech_duration_s = getFloatProperty(
        runtime,
        options,
        "maxSpeechDurationS",
        params.max_speech_duration_s);
    params.speech_pad_ms = getIntProperty(
        runtime,
        options,
        "speechPadMs",
        params.speech_pad_ms);
    params.samples_overlap = getFloatProperty(
        runtime,
        options,
        "samplesOverlap",
        params.samples_overlap);
    return params;
}

std::vector<rnwhisper_jsi::CoreMLAssetInfo> parseCoreMLAssets(
    jsi::Runtime &runtime,
    const jsi::Object &options) {
    std::vector<rnwhisper_jsi::CoreMLAssetInfo> assets;
    if (!options.hasProperty(runtime, "coreMLAssets")) {
        return assets;
    }

    auto value = options.getProperty(runtime, "coreMLAssets");
    if (!value.isObject() || !value.asObject(runtime).isArray(runtime)) {
        return assets;
    }

    auto array = value.asObject(runtime).asArray(runtime);
    size_t length = array.size(runtime);
    assets.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        auto item = array.getValueAtIndex(runtime, i);
        if (!item.isObject()) {
            continue;
        }

        auto object = item.asObject(runtime);
        assets.push_back({
            getStringProperty(runtime, object, "uri"),
            getStringProperty(runtime, object, "filepath"),
        });
    }
    return assets;
}

std::vector<float> decodePcm16(const uint8_t *bytes, size_t byteLength) {
    if (byteLength == 0 || (byteLength % 2) != 0) {
        throw JsiError("Invalid audio data", -1);
    }

    size_t sampleCount = byteLength / sizeof(int16_t);
    std::vector<float> audio(sampleCount);

    for (size_t i = 0; i < sampleCount; ++i) {
        int16_t sample = 0;
        std::memcpy(&sample, bytes + (i * sizeof(int16_t)), sizeof(int16_t));
        audio[i] = std::max(-1.0f, std::min(1.0f, static_cast<float>(sample) / 32767.0f));
    }

    return audio;
}

std::vector<float> decodeWaveBytes(const std::vector<uint8_t> &bytes) {
    constexpr size_t kWaveHeaderSize = 44;
    if (bytes.size() <= kWaveHeaderSize) {
        throw JsiError("Invalid file", -1);
    }
    return decodePcm16(bytes.data() + kWaveHeaderSize, bytes.size() - kWaveHeaderSize);
}

int decodeBase64Value(char value) {
    if (value >= 'A' && value <= 'Z') return value - 'A';
    if (value >= 'a' && value <= 'z') return value - 'a' + 26;
    if (value >= '0' && value <= '9') return value - '0' + 52;
    if (value == '+') return 62;
    if (value == '/') return 63;
    if (value == '=') return -1;
    return -2;
}

std::vector<uint8_t> decodeBase64(const std::string &encoded) {
    std::vector<uint8_t> decoded;
    decoded.reserve((encoded.size() * 3) / 4);

    int buffer = 0;
    int bits = 0;
    for (char ch : encoded) {
        int value = decodeBase64Value(ch);
        if (value == -2) {
            continue;
        }
        if (value == -1) {
            break;
        }

        buffer = (buffer << 6) | value;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            decoded.push_back(static_cast<uint8_t>((buffer >> bits) & 0xff));
        }
    }

    return decoded;
}

std::string extractBase64Payload(const std::string &value) {
    static const std::string prefix = "data:audio/wav;base64,";
    if (value.rfind(prefix, 0) == 0) {
        return value.substr(prefix.size());
    }
    return value;
}

bool isWaveBase64(const std::string &value) {
    static const std::string prefix = "data:audio/wav;base64,";
    return value.rfind(prefix, 0) == 0;
}

std::vector<float> readWaveAudio(const std::string &pathOrBase64) {
    if (isWaveBase64(pathOrBase64)) {
        return decodeWaveBytes(decodeBase64(extractBase64Payload(pathOrBase64)));
    }
    return decodeWaveBytes(rnwhisper_jsi::hostLoadFileBytes(pathOrBase64));
}

std::vector<SegmentData> readSegments(
    whisper_context *context,
    int start,
    bool tdrzEnable,
    std::string *resultText = nullptr) {
    int count = whisper_full_n_segments(context);
    std::vector<SegmentData> segments;
    if (count <= start) {
        return segments;
    }

    segments.reserve(static_cast<size_t>(count - start));
    for (int index = start; index < count; ++index) {
        std::string text = whisper_full_get_segment_text(context, index);
        if (tdrzEnable && whisper_full_get_segment_speaker_turn_next(context, index)) {
            text += " [SPEAKER_TURN]";
        }
        if (resultText) {
            resultText->append(text);
        }
        segments.push_back({
            text,
            static_cast<int>(whisper_full_get_segment_t0(context, index)),
            static_cast<int>(whisper_full_get_segment_t1(context, index)),
        });
    }

    return segments;
}

TranscribeResultData buildTranscribeResult(
    whisper_context *context,
    bool tdrzEnable,
    bool isAborted) {
    TranscribeResultData result;
    result.isAborted = isAborted;
    result.segments = readSegments(context, 0, tdrzEnable, &result.result);
    const char *language = whisper_lang_str(whisper_full_lang_id(context));
    result.language = language ? language : "";
    return result;
}

jsi::Array createSegmentsArray(
    jsi::Runtime &runtime,
    const std::vector<SegmentData> &segments) {
    jsi::Array array(runtime, segments.size());
    for (size_t index = 0; index < segments.size(); ++index) {
        const auto &segment = segments[index];
        jsi::Object item(runtime);
        item.setProperty(
            runtime,
            "text",
            jsi::String::createFromUtf8(runtime, segment.text));
        item.setProperty(runtime, "t0", jsi::Value(segment.t0));
        item.setProperty(runtime, "t1", jsi::Value(segment.t1));
        array.setValueAtIndex(runtime, index, item);
    }
    return array;
}

jsi::Value createTranscribeResultValue(
    jsi::Runtime &runtime,
    const TranscribeResultData &data) {
    jsi::Object result(runtime);
    result.setProperty(
        runtime,
        "language",
        jsi::String::createFromUtf8(runtime, data.language));
    result.setProperty(
        runtime,
        "result",
        jsi::String::createFromUtf8(runtime, data.result));
    result.setProperty(runtime, "segments", createSegmentsArray(runtime, data.segments));
    result.setProperty(runtime, "isAborted", jsi::Value(data.isAborted));
    return result;
}

jsi::Value createNewSegmentsValue(
    jsi::Runtime &runtime,
    const NewSegmentsData &data) {
    jsi::Object result(runtime);
    result.setProperty(runtime, "nNew", jsi::Value(data.nNew));
    result.setProperty(runtime, "totalNNew", jsi::Value(data.totalNNew));
    result.setProperty(
        runtime,
        "result",
        jsi::String::createFromUtf8(runtime, data.result));
    result.setProperty(runtime, "segments", createSegmentsArray(runtime, data.segments));
    return result;
}

jsi::Value createVadResultValue(
    jsi::Runtime &runtime,
    const VadResultData &data) {
    jsi::Object result(runtime);
    result.setProperty(runtime, "hasSpeech", jsi::Value(data.hasSpeech));
    jsi::Array segments(runtime, data.segments.size());
    for (size_t index = 0; index < data.segments.size(); ++index) {
        jsi::Object item(runtime);
        item.setProperty(runtime, "t0", jsi::Value(data.segments[index].t0));
        item.setProperty(runtime, "t1", jsi::Value(data.segments[index].t1));
        segments.setValueAtIndex(runtime, index, item);
    }
    result.setProperty(runtime, "segments", segments);
    return result;
}

jsi::Value createContextValue(
    jsi::Runtime &runtime,
    const std::shared_ptr<WhisperContextHolder> &holder) {
    jsi::Object result(runtime);
    result.setProperty(
        runtime,
        "contextPtr",
        jsi::Value(static_cast<double>(holder->ptr)));
    result.setProperty(runtime, "contextId", jsi::Value(holder->id));
    result.setProperty(runtime, "gpu", jsi::Value(holder->gpu));
    result.setProperty(
        runtime,
        "reasonNoGPU",
        jsi::String::createFromUtf8(runtime, holder->reasonNoGPU));
    return result;
}

jsi::Value createVadContextValue(
    jsi::Runtime &runtime,
    const std::shared_ptr<WhisperVadContextHolder> &holder) {
    jsi::Object result(runtime);
    result.setProperty(runtime, "contextId", jsi::Value(holder->id));
    result.setProperty(runtime, "gpu", jsi::Value(holder->gpu));
    result.setProperty(
        runtime,
        "reasonNoGPU",
        jsi::String::createFromUtf8(runtime, holder->reasonNoGPU));
    return result;
}

void defaultWhisperLogCallback(
    enum wsp_ggml_log_level level,
    const char *text,
    void *) {
#if defined(__ANDROID__)
    if (level == WSP_GGML_LOG_LEVEL_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, "RNWhisper", "%s", text);
    } else if (level == WSP_GGML_LOG_LEVEL_WARN) {
        __android_log_print(ANDROID_LOG_WARN, "RNWhisper", "%s", text);
    } else if (level == WSP_GGML_LOG_LEVEL_INFO) {
        __android_log_print(ANDROID_LOG_INFO, "RNWhisper", "%s", text);
    } else {
        __android_log_print(ANDROID_LOG_DEBUG, "RNWhisper", "%s", text);
    }
#else
#ifndef WHISPER_DEBUG
    if (level == WSP_GGML_LOG_LEVEL_DEBUG) {
        return;
    }
#endif
    std::fputs(text, stderr);
    std::fflush(stderr);
#endif
}

void forwardLogToJs(enum wsp_ggml_log_level level, const char *text, void *) {
    defaultWhisperLogCallback(level, text, nullptr);

    std::shared_ptr<react::CallInvoker> invoker;
    std::shared_ptr<jsi::Function> handler;
    std::shared_ptr<jsi::Runtime> runtime;
    {
        std::lock_guard<std::mutex> lock(g_logMutex);
        invoker = g_logInvoker.lock();
        handler = g_logHandler;
        runtime = g_logRuntime;
    }

    if (!invoker || !handler || !runtime || g_isShuttingDown.load(std::memory_order_relaxed)) {
        return;
    }

    std::string levelText = "info";
    if (level == WSP_GGML_LOG_LEVEL_ERROR) {
        levelText = "error";
    } else if (level == WSP_GGML_LOG_LEVEL_WARN) {
        levelText = "warn";
    }

    std::string message = text ? text : "";
    invokeAsync(invoker, [handler, runtime, levelText, message]() {
        auto &rt = *runtime;
        handler->call(
            rt,
            jsi::String::createFromUtf8(rt, levelText),
            jsi::String::createFromUtf8(rt, message));
    });
}

void releaseWhisperHolder(const std::shared_ptr<WhisperContextHolder> &holder) {
    if (!holder) {
        return;
    }
    holder->abortActiveJob();
    holder->waitForIdle();
    if (holder->context != nullptr) {
        whisper_free(holder->context);
        holder->context = nullptr;
    }
}

void releaseVadHolder(const std::shared_ptr<WhisperVadContextHolder> &holder) {
    if (!holder) {
        return;
    }
    holder->waitForIdle();
    if (holder->context != nullptr) {
        whisper_vad_free(holder->context);
        holder->context = nullptr;
    }
}

jsi::Object requireObjectArgument(
    jsi::Runtime &runtime,
    const jsi::Value *arguments,
    size_t count,
    size_t index,
    const char *message) {
    if (count <= index || !arguments[index].isObject()) {
        throw jsi::JSError(runtime, message);
    }
    return arguments[index].asObject(runtime);
}

int requireContextId(
    jsi::Runtime &runtime,
    const jsi::Value *arguments,
    size_t count,
    size_t index = 0) {
    if (count <= index || !arguments[index].isNumber()) {
        throw jsi::JSError(runtime, "First argument must be a context id");
    }
    return static_cast<int>(arguments[index].asNumber());
}

std::string requireStringArgument(
    jsi::Runtime &runtime,
    const jsi::Value *arguments,
    size_t count,
    size_t index,
    const char *message) {
    if (count <= index || !arguments[index].isString()) {
        throw jsi::JSError(runtime, message);
    }
    return arguments[index].asString(runtime).utf8(runtime);
}

std::vector<float> requireAudioBufferArgument(
    jsi::Runtime &runtime,
    const jsi::Value *arguments,
    size_t count,
    size_t index) {
    if (count <= index || !arguments[index].isObject()) {
        throw jsi::JSError(runtime, "Audio argument must be an ArrayBuffer");
    }
    auto object = arguments[index].asObject(runtime);
    if (!object.isArrayBuffer(runtime)) {
        throw jsi::JSError(runtime, "Audio argument must be an ArrayBuffer");
    }
    auto arrayBuffer = object.getArrayBuffer(runtime);
    return decodePcm16(arrayBuffer.data(runtime), arrayBuffer.size(runtime));
}

} // namespace

namespace rnwhisper_jsi {

void installJSIBindings(
    jsi::Runtime &runtime,
    std::shared_ptr<react::CallInvoker> callInvoker) {
    g_isShuttingDown.store(false, std::memory_order_relaxed);

    auto getConstants = jsi::Function::createFromHostFunction(
        runtime,
        jsi::PropNameID::forAscii(runtime, "whisperGetConstants"),
        0,
        [callInvoker](
            jsi::Runtime &runtime,
            const jsi::Value &,
            const jsi::Value *,
            size_t) -> jsi::Value {
            return createPromiseTask(runtime, callInvoker, []() -> PromiseResultGenerator {
                return [](jsi::Runtime &rt) {
                    jsi::Object result(rt);
#if defined(WHISPER_USE_COREML)
                    result.setProperty(rt, "useCoreML", jsi::Value(true));
#else
                    result.setProperty(rt, "useCoreML", jsi::Value(false));
#endif
#if defined(WHISPER_COREML_ALLOW_FALLBACK)
                    result.setProperty(rt, "coreMLAllowFallback", jsi::Value(true));
#else
                    result.setProperty(rt, "coreMLAllowFallback", jsi::Value(false));
#endif
                    return result;
                };
            });
        });

    auto initContext = jsi::Function::createFromHostFunction(
        runtime,
        jsi::PropNameID::forAscii(runtime, "whisperInitContext"),
        2,
        [callInvoker](
            jsi::Runtime &runtime,
            const jsi::Value &,
            const jsi::Value *arguments,
            size_t count) -> jsi::Value {
            int contextId = requireContextId(runtime, arguments, count);
            auto options = requireObjectArgument(
                runtime,
                arguments,
                count,
                1,
                "Context options must be an object");

            WhisperContextInitOptions hostOptions;
            hostOptions.filePath = getStringProperty(runtime, options, "filePath");
            hostOptions.isBundleAsset =
                getBoolProperty(runtime, options, "isBundleAsset", false);
            hostOptions.useFlashAttn =
                getBoolProperty(runtime, options, "useFlashAttn", false);
            hostOptions.useGpu = getBoolProperty(runtime, options, "useGpu", true);
            hostOptions.useCoreMLIos =
                getBoolProperty(runtime, options, "useCoreMLIos", true);
            hostOptions.downloadCoreMLAssets =
                getBoolProperty(runtime, options, "downloadCoreMLAssets", false);
            hostOptions.coreMLAssets = parseCoreMLAssets(runtime, options);

            return createPromiseTask(runtime, callInvoker, [contextId, hostOptions]() -> PromiseResultGenerator {
                auto result = hostInitWhisperContext(hostOptions);
                if (result.context == nullptr) {
                    throw JsiError("Failed to load the model");
                }

                auto holder = std::make_shared<WhisperContextHolder>(contextId);
                holder->context = result.context;
                holder->ptr = reinterpret_cast<long>(result.context);
                holder->gpu = result.gpu;
                holder->reasonNoGPU = result.reasonNoGPU;
                g_whisperContexts.add(contextId, holder);

                return [holder](jsi::Runtime &rt) {
                    return createContextValue(rt, holder);
                };
            });
        });

    auto releaseContext = jsi::Function::createFromHostFunction(
        runtime,
        jsi::PropNameID::forAscii(runtime, "whisperReleaseContext"),
        1,
        [callInvoker](
            jsi::Runtime &runtime,
            const jsi::Value &,
            const jsi::Value *arguments,
            size_t count) -> jsi::Value {
            int contextId = requireContextId(runtime, arguments, count);
            return createPromiseTask(runtime, callInvoker, [contextId]() -> PromiseResultGenerator {
                auto holder = g_whisperContexts.remove(contextId);
                if (!holder) {
                    throw JsiError("Context not found");
                }
                releaseWhisperHolder(holder);
                return [](jsi::Runtime &) {
                    return jsi::Value::undefined();
                };
            });
        });

    auto releaseAllContexts = jsi::Function::createFromHostFunction(
        runtime,
        jsi::PropNameID::forAscii(runtime, "whisperReleaseAllContexts"),
        0,
        [callInvoker](
            jsi::Runtime &runtime,
            const jsi::Value &,
            const jsi::Value *,
            size_t) -> jsi::Value {
            return createPromiseTask(runtime, callInvoker, []() -> PromiseResultGenerator {
                auto holders = g_whisperContexts.removeAll();
                for (const auto &holder : holders) {
                    releaseWhisperHolder(holder);
                }
                return [](jsi::Runtime &) {
                    return jsi::Value::undefined();
                };
            });
        });

    auto transcribeFile = jsi::Function::createFromHostFunction(
        runtime,
        jsi::PropNameID::forAscii(runtime, "whisperTranscribeFile"),
        3,
        [callInvoker](
            jsi::Runtime &runtime,
            const jsi::Value &,
            const jsi::Value *arguments,
            size_t count) -> jsi::Value {
            int contextId = requireContextId(runtime, arguments, count);
            std::string input = requireStringArgument(
                runtime,
                arguments,
                count,
                1,
                "Transcription input must be a string");
            auto options = requireObjectArgument(
                runtime,
                arguments,
                count,
                2,
                "Transcription options must be an object");

            auto holder = g_whisperContexts.get(contextId);
            if (!holder) {
                throw jsi::JSError(runtime, "Context not found");
            }
            auto runtimePtr = std::shared_ptr<jsi::Runtime>(&runtime, [](jsi::Runtime *) {});

            auto config = createTranscribeConfig(runtime, options, callInvoker);
            if (!holder->beginExclusiveOperation(config.jobId)) {
                throw jsi::JSError(runtime, "Context is already transcribing");
            }

            holder->retainTask();
            try {
                return createPromiseTask(runtime, callInvoker, [holder, config, input, callInvoker, runtimePtr]() mutable -> PromiseResultGenerator {
                    PromiseScopeGuard taskGuard([holder]() { holder->releaseTask(); });
                    PromiseScopeGuard exclusiveGuard([holder]() { holder->endExclusiveOperation(); });

                    auto audio = readWaveAudio(input);
                    if (audio.empty()) {
                        throw JsiError("Invalid file");
                    }

                    auto progressState = std::make_shared<JsiCallbackState>();
                    progressState->callInvoker = callInvoker;
                    progressState->callback = config.onProgress;
                    progressState->runtime = runtimePtr;
                    if (config.onProgress) {
                        config.params.progress_callback =
                            [](whisper_context *, whisper_state *, int progress, void *userData) {
                                auto *state = static_cast<std::shared_ptr<JsiCallbackState> *>(userData);
                                if (!state || !(*state)) {
                                    return;
                                }
                                emitProgressCallback(*state, progress);
                            };
                        config.params.progress_callback_user_data = &progressState;
                    }

                    auto segmentsState = std::make_shared<SegmentCallbackState>();
                    segmentsState->callInvoker = callInvoker;
                    segmentsState->callback = config.onNewSegments;
                    segmentsState->runtime = runtimePtr;
                    segmentsState->tdrzEnable = config.tdrzEnable;

                    if (config.onNewSegments) {
                        config.params.new_segment_callback =
                            [](whisper_context *ctx, whisper_state *, int nNew, void *userData) {
                                auto *state = static_cast<std::shared_ptr<SegmentCallbackState> *>(userData);
                                if (!state || !(*state)) {
                                    return;
                                }

                                (*state)->totalNNew += nNew;
                                int offset = (*state)->totalNNew - nNew;
                                std::string resultText;
                                auto segments = readSegments(ctx, offset, (*state)->tdrzEnable, &resultText);

                                NewSegmentsData payload;
                                payload.nNew = nNew;
                                payload.totalNNew = (*state)->totalNNew;
                                payload.result = std::move(resultText);
                                payload.segments = std::move(segments);

                                emitNewSegmentsCallback(*state, std::move(payload));
                            };
                        config.params.new_segment_callback_user_data = &segmentsState;
                    }

                    rnwhisper::job *job = rnwhisper::job_new(config.jobId, config.params);
                    if (job == nullptr) {
                        throw JsiError("Failed to create transcription job");
                    }

                    int code = whisper_full_parallel(
                        holder->context,
                        job->params,
                        audio.data(),
                        static_cast<int>(audio.size()),
                        config.nProcessors);
                    bool isAborted = job->is_aborted();
                    rnwhisper::job_remove(config.jobId);

                    if (code != 0 && !isAborted) {
                        throw JsiError("Transcription failed", code);
                    }

                    auto result = buildTranscribeResult(
                        holder->context,
                        config.tdrzEnable,
                        isAborted);
                    return [result](jsi::Runtime &rt) {
                        return createTranscribeResultValue(rt, result);
                    };
                });
            } catch (...) {
                holder->endExclusiveOperation();
                holder->releaseTask();
                throw;
            }
        });

    auto transcribeData = jsi::Function::createFromHostFunction(
        runtime,
        jsi::PropNameID::forAscii(runtime, "whisperTranscribeData"),
        3,
        [callInvoker](
            jsi::Runtime &runtime,
            const jsi::Value &,
            const jsi::Value *arguments,
            size_t count) -> jsi::Value {
            int contextId = requireContextId(runtime, arguments, count);
            auto options = requireObjectArgument(
                runtime,
                arguments,
                count,
                1,
                "Transcription options must be an object");
            auto audio = requireAudioBufferArgument(runtime, arguments, count, 2);

            auto holder = g_whisperContexts.get(contextId);
            if (!holder) {
                throw jsi::JSError(runtime, "Context not found");
            }
            auto runtimePtr = std::shared_ptr<jsi::Runtime>(&runtime, [](jsi::Runtime *) {});

            auto config = createTranscribeConfig(runtime, options, callInvoker);
            if (!holder->beginExclusiveOperation(config.jobId)) {
                throw jsi::JSError(runtime, "Context is already transcribing");
            }

            holder->retainTask();
            try {
                return createPromiseTask(runtime, callInvoker, [holder, config, audio, callInvoker, runtimePtr]() mutable -> PromiseResultGenerator {
                    PromiseScopeGuard taskGuard([holder]() { holder->releaseTask(); });
                    PromiseScopeGuard exclusiveGuard([holder]() { holder->endExclusiveOperation(); });

                    auto progressState = std::make_shared<JsiCallbackState>();
                    progressState->callInvoker = callInvoker;
                    progressState->callback = config.onProgress;
                    progressState->runtime = runtimePtr;
                    if (config.onProgress) {
                        config.params.progress_callback =
                            [](whisper_context *, whisper_state *, int progress, void *userData) {
                                auto *state = static_cast<std::shared_ptr<JsiCallbackState> *>(userData);
                                if (!state || !(*state)) {
                                    return;
                                }
                                emitProgressCallback(*state, progress);
                            };
                        config.params.progress_callback_user_data = &progressState;
                    }

                    auto segmentsState = std::make_shared<SegmentCallbackState>();
                    segmentsState->callInvoker = callInvoker;
                    segmentsState->callback = config.onNewSegments;
                    segmentsState->runtime = runtimePtr;
                    segmentsState->tdrzEnable = config.tdrzEnable;

                    if (config.onNewSegments) {
                        config.params.new_segment_callback =
                            [](whisper_context *ctx, whisper_state *, int nNew, void *userData) {
                                auto *state = static_cast<std::shared_ptr<SegmentCallbackState> *>(userData);
                                if (!state || !(*state)) {
                                    return;
                                }

                                (*state)->totalNNew += nNew;
                                int offset = (*state)->totalNNew - nNew;
                                std::string resultText;
                                auto segments = readSegments(ctx, offset, (*state)->tdrzEnable, &resultText);

                                NewSegmentsData payload;
                                payload.nNew = nNew;
                                payload.totalNNew = (*state)->totalNNew;
                                payload.result = std::move(resultText);
                                payload.segments = std::move(segments);

                                emitNewSegmentsCallback(*state, std::move(payload));
                            };
                        config.params.new_segment_callback_user_data = &segmentsState;
                    }

                    rnwhisper::job *job = rnwhisper::job_new(config.jobId, config.params);
                    if (job == nullptr) {
                        throw JsiError("Failed to create transcription job");
                    }

                    int code = whisper_full_parallel(
                        holder->context,
                        job->params,
                        audio.data(),
                        static_cast<int>(audio.size()),
                        config.nProcessors);
                    bool isAborted = job->is_aborted();
                    rnwhisper::job_remove(config.jobId);

                    if (code != 0 && !isAborted) {
                        throw JsiError("Transcription failed", code);
                    }

                    auto result = buildTranscribeResult(
                        holder->context,
                        config.tdrzEnable,
                        isAborted);
                    return [result](jsi::Runtime &rt) {
                        return createTranscribeResultValue(rt, result);
                    };
                });
            } catch (...) {
                holder->endExclusiveOperation();
                holder->releaseTask();
                throw;
            }
        });

    auto abortTranscribe = jsi::Function::createFromHostFunction(
        runtime,
        jsi::PropNameID::forAscii(runtime, "whisperAbortTranscribe"),
        2,
        [callInvoker](
            jsi::Runtime &runtime,
            const jsi::Value &,
            const jsi::Value *arguments,
            size_t count) -> jsi::Value {
            int contextId = requireContextId(runtime, arguments, count);
            int jobId = count > 1 && arguments[1].isNumber()
                ? static_cast<int>(arguments[1].asNumber())
                : -1;
            return createPromiseTask(runtime, callInvoker, [contextId, jobId]() -> PromiseResultGenerator {
                auto holder = g_whisperContexts.get(contextId);
                if (holder) {
                    holder->abortActiveJob();
                }
                if (jobId >= 0) {
                    if (auto *job = rnwhisper::job_get(jobId)) {
                        job->abort();
                    }
                }
                return [](jsi::Runtime &) {
                    return jsi::Value::undefined();
                };
            });
        });

    auto bench = jsi::Function::createFromHostFunction(
        runtime,
        jsi::PropNameID::forAscii(runtime, "whisperBench"),
        2,
        [callInvoker](
            jsi::Runtime &runtime,
            const jsi::Value &,
            const jsi::Value *arguments,
            size_t count) -> jsi::Value {
            int contextId = requireContextId(runtime, arguments, count);
            int maxThreads =
                count > 1 && arguments[1].isNumber()
                    ? static_cast<int>(arguments[1].asNumber())
                    : 0;

            auto holder = g_whisperContexts.get(contextId);
            if (!holder) {
                throw jsi::JSError(runtime, "Context not found");
            }

            if (!holder->beginExclusiveOperation(-1)) {
                throw jsi::JSError(runtime, "The context is transcribing");
            }

            holder->retainTask();
            try {
                return createPromiseTask(runtime, callInvoker, [holder, maxThreads]() -> PromiseResultGenerator {
                    PromiseScopeGuard taskGuard([holder]() { holder->releaseTask(); });
                    PromiseScopeGuard exclusiveGuard([holder]() { holder->endExclusiveOperation(); });

                    std::string result = rnwhisper::bench(holder->context, maxThreads);
                    return [result](jsi::Runtime &rt) {
                        return jsi::String::createFromUtf8(rt, result);
                    };
                });
            } catch (...) {
                holder->endExclusiveOperation();
                holder->releaseTask();
                throw;
            }
        });

    auto initVadContext = jsi::Function::createFromHostFunction(
        runtime,
        jsi::PropNameID::forAscii(runtime, "whisperInitVadContext"),
        2,
        [callInvoker](
            jsi::Runtime &runtime,
            const jsi::Value &,
            const jsi::Value *arguments,
            size_t count) -> jsi::Value {
            int contextId = requireContextId(runtime, arguments, count);
            auto options = requireObjectArgument(
                runtime,
                arguments,
                count,
                1,
                "VAD options must be an object");

            WhisperVadContextInitOptions hostOptions;
            hostOptions.filePath = getStringProperty(runtime, options, "filePath");
            hostOptions.isBundleAsset =
                getBoolProperty(runtime, options, "isBundleAsset", false);
            hostOptions.useGpu = getBoolProperty(runtime, options, "useGpu", true);
            hostOptions.nThreads = getIntProperty(runtime, options, "nThreads", 0);

            return createPromiseTask(runtime, callInvoker, [contextId, hostOptions]() -> PromiseResultGenerator {
                auto result = hostInitWhisperVadContext(hostOptions);
                if (result.context == nullptr) {
                    throw JsiError("Failed to load the VAD model");
                }

                auto holder = std::make_shared<WhisperVadContextHolder>(contextId);
                holder->context = result.context;
                holder->ptr = reinterpret_cast<long>(result.context);
                holder->gpu = result.gpu;
                holder->reasonNoGPU = result.reasonNoGPU;
                g_vadContexts.add(contextId, holder);

                return [holder](jsi::Runtime &rt) {
                    return createVadContextValue(rt, holder);
                };
            });
        });

    auto releaseVadContext = jsi::Function::createFromHostFunction(
        runtime,
        jsi::PropNameID::forAscii(runtime, "whisperReleaseVadContext"),
        1,
        [callInvoker](
            jsi::Runtime &runtime,
            const jsi::Value &,
            const jsi::Value *arguments,
            size_t count) -> jsi::Value {
            int contextId = requireContextId(runtime, arguments, count);
            return createPromiseTask(runtime, callInvoker, [contextId]() -> PromiseResultGenerator {
                auto holder = g_vadContexts.remove(contextId);
                if (!holder) {
                    throw JsiError("VAD context not found");
                }
                releaseVadHolder(holder);
                return [](jsi::Runtime &) {
                    return jsi::Value::undefined();
                };
            });
        });

    auto releaseAllVadContexts = jsi::Function::createFromHostFunction(
        runtime,
        jsi::PropNameID::forAscii(runtime, "whisperReleaseAllVadContexts"),
        0,
        [callInvoker](
            jsi::Runtime &runtime,
            const jsi::Value &,
            const jsi::Value *,
            size_t) -> jsi::Value {
            return createPromiseTask(runtime, callInvoker, []() -> PromiseResultGenerator {
                auto holders = g_vadContexts.removeAll();
                for (const auto &holder : holders) {
                    releaseVadHolder(holder);
                }
                return [](jsi::Runtime &) {
                    return jsi::Value::undefined();
                };
            });
        });

    auto vadDetectSpeech = jsi::Function::createFromHostFunction(
        runtime,
        jsi::PropNameID::forAscii(runtime, "whisperVadDetectSpeech"),
        3,
        [callInvoker](
            jsi::Runtime &runtime,
            const jsi::Value &,
            const jsi::Value *arguments,
            size_t count) -> jsi::Value {
            int contextId = requireContextId(runtime, arguments, count);
            auto options = requireObjectArgument(
                runtime,
                arguments,
                count,
                1,
                "VAD options must be an object");
            auto audio = requireAudioBufferArgument(runtime, arguments, count, 2);

            auto holder = g_vadContexts.get(contextId);
            if (!holder) {
                throw jsi::JSError(runtime, "VAD context not found");
            }

            auto vadOptions = createVadParams(runtime, options);
            holder->retainTask();
            try {
                return createPromiseTask(runtime, callInvoker, [holder, audio, vadOptions]() -> PromiseResultGenerator {
                    PromiseScopeGuard taskGuard([holder]() { holder->releaseTask(); });

                    bool detected = whisper_vad_detect_speech(
                        holder->context,
                        audio.data(),
                        static_cast<int>(audio.size()));

                    VadResultData result;
                    result.hasSpeech = detected;
                    if (detected) {
                        auto *segments = whisper_vad_segments_from_probs(holder->context, vadOptions);
                        if (segments != nullptr) {
                            int count = whisper_vad_segments_n_segments(segments);
                            result.segments.reserve(static_cast<size_t>(count));
                            for (int index = 0; index < count; ++index) {
                                result.segments.push_back({
                                    whisper_vad_segments_get_segment_t0(segments, index),
                                    whisper_vad_segments_get_segment_t1(segments, index),
                                });
                            }
                            whisper_vad_free_segments(segments);
                            result.hasSpeech = !result.segments.empty();
                        } else {
                            result.hasSpeech = false;
                        }
                    }

                    return [result](jsi::Runtime &rt) {
                        return createVadResultValue(rt, result);
                    };
                });
            } catch (...) {
                holder->releaseTask();
                throw;
            }
        });

    auto vadDetectSpeechFile = jsi::Function::createFromHostFunction(
        runtime,
        jsi::PropNameID::forAscii(runtime, "whisperVadDetectSpeechFile"),
        3,
        [callInvoker](
            jsi::Runtime &runtime,
            const jsi::Value &,
            const jsi::Value *arguments,
            size_t count) -> jsi::Value {
            int contextId = requireContextId(runtime, arguments, count);
            std::string input = requireStringArgument(
                runtime,
                arguments,
                count,
                1,
                "VAD input must be a string");
            auto options = requireObjectArgument(
                runtime,
                arguments,
                count,
                2,
                "VAD options must be an object");

            auto holder = g_vadContexts.get(contextId);
            if (!holder) {
                throw jsi::JSError(runtime, "VAD context not found");
            }

            auto vadOptions = createVadParams(runtime, options);
            holder->retainTask();
            try {
                return createPromiseTask(runtime, callInvoker, [holder, input, vadOptions]() -> PromiseResultGenerator {
                    PromiseScopeGuard taskGuard([holder]() { holder->releaseTask(); });

                    auto audio = readWaveAudio(input);
                    bool detected = whisper_vad_detect_speech(
                        holder->context,
                        audio.data(),
                        static_cast<int>(audio.size()));

                    VadResultData result;
                    result.hasSpeech = detected;
                    if (detected) {
                        auto *segments = whisper_vad_segments_from_probs(holder->context, vadOptions);
                        if (segments != nullptr) {
                            int count = whisper_vad_segments_n_segments(segments);
                            result.segments.reserve(static_cast<size_t>(count));
                            for (int index = 0; index < count; ++index) {
                                result.segments.push_back({
                                    whisper_vad_segments_get_segment_t0(segments, index),
                                    whisper_vad_segments_get_segment_t1(segments, index),
                                });
                            }
                            whisper_vad_free_segments(segments);
                            result.hasSpeech = !result.segments.empty();
                        } else {
                            result.hasSpeech = false;
                        }
                    }

                    return [result](jsi::Runtime &rt) {
                        return createVadResultValue(rt, result);
                    };
                });
            } catch (...) {
                holder->releaseTask();
                throw;
            }
        });

    auto toggleNativeLog = jsi::Function::createFromHostFunction(
        runtime,
        jsi::PropNameID::forAscii(runtime, "whisperToggleNativeLog"),
        2,
        [callInvoker](
            jsi::Runtime &runtime,
            const jsi::Value &,
            const jsi::Value *arguments,
            size_t count) -> jsi::Value {
            bool enabled = count > 0 && arguments[0].isBool() ? arguments[0].getBool() : false;
            auto callback =
                enabled && count > 1 ? makeJsiFunction(runtime, arguments[1], callInvoker) : nullptr;
            auto runtimePtr = std::shared_ptr<jsi::Runtime>(&runtime, [](jsi::Runtime *) {});

            return createPromiseTask(runtime, callInvoker, [enabled, callback, callInvoker, runtimePtr]() -> PromiseResultGenerator {
                std::lock_guard<std::mutex> lock(g_logMutex);
                if (enabled && callback) {
                    g_logInvoker = callInvoker;
                    g_logHandler = callback;
                    g_logRuntime = runtimePtr;
                    whisper_log_set(forwardLogToJs, nullptr);
                } else {
                    g_logInvoker.reset();
                    g_logHandler.reset();
                    g_logRuntime.reset();
                    whisper_log_set(defaultWhisperLogCallback, nullptr);
                }
                return [](jsi::Runtime &) {
                    return jsi::Value::undefined();
                };
            });
        });

    runtime.global().setProperty(runtime, "whisperGetConstants", std::move(getConstants));
    runtime.global().setProperty(runtime, "whisperInitContext", std::move(initContext));
    runtime.global().setProperty(runtime, "whisperReleaseContext", std::move(releaseContext));
    runtime.global().setProperty(runtime, "whisperReleaseAllContexts", std::move(releaseAllContexts));
    runtime.global().setProperty(runtime, "whisperTranscribeFile", std::move(transcribeFile));
    runtime.global().setProperty(runtime, "whisperTranscribeData", std::move(transcribeData));
    runtime.global().setProperty(runtime, "whisperAbortTranscribe", std::move(abortTranscribe));
    runtime.global().setProperty(runtime, "whisperBench", std::move(bench));
    runtime.global().setProperty(runtime, "whisperInitVadContext", std::move(initVadContext));
    runtime.global().setProperty(runtime, "whisperReleaseVadContext", std::move(releaseVadContext));
    runtime.global().setProperty(runtime, "whisperReleaseAllVadContexts", std::move(releaseAllVadContexts));
    runtime.global().setProperty(runtime, "whisperVadDetectSpeech", std::move(vadDetectSpeech));
    runtime.global().setProperty(runtime, "whisperVadDetectSpeechFile", std::move(vadDetectSpeechFile));
    runtime.global().setProperty(runtime, "whisperToggleNativeLog", std::move(toggleNativeLog));
}

void cleanupJSIBindings() {
    g_isShuttingDown.store(true, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lock(g_logMutex);
        g_logInvoker.reset();
        g_logHandler.reset();
        g_logRuntime.reset();
    }
    whisper_log_set(defaultWhisperLogCallback, nullptr);

    auto whisperHolders = g_whisperContexts.removeAll();
    auto vadHolders = g_vadContexts.removeAll();

    for (const auto &holder : whisperHolders) {
        releaseWhisperHolder(holder);
    }
    for (const auto &holder : vadHolders) {
        releaseVadHolder(holder);
    }

    hostClearCache();

    std::lock_guard<std::mutex> lock(g_threadPoolMutex);
    g_threadPool.reset();
}

} // namespace rnwhisper_jsi
