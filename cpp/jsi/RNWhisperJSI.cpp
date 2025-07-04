#include "RNWhisperJSI.h"
#include "ThreadPool.h"
#include <jsi/jsi.h>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <atomic>

#if defined(__ANDROID__)
#include <android/log.h>
#endif

using namespace facebook::jsi;

namespace rnwhisper_jsi {

// Platform-specific logging
static void logInfo(const char* format, ...) {
  va_list args;
  va_start(args, format);

#if defined(__ANDROID__)
  __android_log_vprint(ANDROID_LOG_INFO, "RNWhisperJSI", format, args);
#else
  char buffer[1024];
  vsnprintf(buffer, sizeof(buffer), format, args);
  printf("RNWhisperJSI: %s\n", buffer);
#endif

  va_end(args);
}

static void logError(const char* format, ...) {
  va_list args;
  va_start(args, format);

#if defined(__ANDROID__)
  __android_log_vprint(ANDROID_LOG_ERROR, "RNWhisperJSI", format, args);
#else
  char buffer[1024];
  vsnprintf(buffer, sizeof(buffer), format, args);
  printf("RNWhisperJSI ERROR: %s\n", buffer);
#endif

  va_end(args);
}

static void logDebug(const char* format, ...) {
  va_list args;
  va_start(args, format);

#if defined(__ANDROID__)
  __android_log_vprint(ANDROID_LOG_DEBUG, "RNWhisperJSI", format, args);
#else
  char buffer[1024];
  vsnprintf(buffer, sizeof(buffer), format, args);
  printf("RNWhisperJSI DEBUG: %s\n", buffer);
#endif

  va_end(args);
}

static std::unique_ptr<ThreadPool> whisperThreadPool = nullptr;
static std::mutex threadPoolMutex;

// Initialize ThreadPool with optimal thread count
ThreadPool& getWhisperThreadPool() {
    std::lock_guard<std::mutex> lock(threadPoolMutex);
    if (!whisperThreadPool) {
        int max_threads = std::thread::hardware_concurrency();
        int thread_count = std::max(2, std::min(4, max_threads)); // Use 2-4 threads
        whisperThreadPool = std::make_unique<ThreadPool>(thread_count);
        logInfo("Initialized ThreadPool with %d threads", thread_count);
    }
    return *whisperThreadPool;
}

static std::unordered_map<int, long> contextMap;
static std::mutex contextMapMutex;
static std::unordered_map<int, long> vadContextMap;
static std::mutex vadContextMapMutex;

void addContext(int contextId, long contextPtr) {
    std::lock_guard<std::mutex> lock(contextMapMutex);
    contextMap[contextId] = contextPtr;
    logDebug("Context added: id=%d, ptr=%ld", contextId, contextPtr);
}

void removeContext(int contextId) {
    std::lock_guard<std::mutex> lock(contextMapMutex);
    auto it = contextMap.find(contextId);
    if (it != contextMap.end()) {
        logDebug("Context removed: id=%d", contextId);
        contextMap.erase(it);
    }
}

void addVadContext(int contextId, long vadContextPtr) {
    std::lock_guard<std::mutex> lock(vadContextMapMutex);
    vadContextMap[contextId] = vadContextPtr;
    logDebug("VAD Context added: id=%d, ptr=%ld", contextId, vadContextPtr);
}

void removeVadContext(int contextId) {
    std::lock_guard<std::mutex> lock(vadContextMapMutex);
    auto it = vadContextMap.find(contextId);
    if (it != vadContextMap.end()) {
        logDebug("VAD Context removed: id=%d", contextId);
        vadContextMap.erase(it);
    }
}

long getContextPtr(int contextId) {
    std::lock_guard<std::mutex> lock(contextMapMutex);
    auto it = contextMap.find(contextId);
    return (it != contextMap.end()) ? it->second : 0;
}

long getVadContextPtr(int contextId) {
    std::lock_guard<std::mutex> lock(vadContextMapMutex);
    auto it = vadContextMap.find(contextId);
    return (it != vadContextMap.end()) ? it->second : 0;
}

// Helper function to convert JSI object to whisper_full_params
whisper_full_params createFullParamsFromJSI(Runtime& runtime, const Object& optionsObj) {
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    params.print_realtime = false;
    params.print_progress = false;
    params.print_timestamps = false;
    params.print_special = false;

    int max_threads = std::thread::hardware_concurrency();
    int default_n_threads = max_threads == 4 ? 2 : std::min(4, max_threads);

    try {
        auto propNames = optionsObj.getPropertyNames(runtime);
        for (size_t i = 0; i < propNames.size(runtime); i++) {
            auto propNameValue = propNames.getValueAtIndex(runtime, i);
            std::string propName = propNameValue.getString(runtime).utf8(runtime);
            Value propValue = optionsObj.getProperty(runtime, propNameValue.getString(runtime));

            if (propName == "maxThreads" && propValue.isNumber()) {
                int n_threads = (int)propValue.getNumber();
                params.n_threads = n_threads > 0 ? n_threads : default_n_threads;
            } else if (propName == "translate" && propValue.isBool()) {
                params.translate = propValue.getBool();
            } else if (propName == "tokenTimestamps" && propValue.isBool()) {
                params.token_timestamps = propValue.getBool();
            } else if (propName == "tdrzEnable" && propValue.isBool()) {
                params.tdrz_enable = propValue.getBool();
            } else if (propName == "beamSize" && propValue.isNumber()) {
                int beam_size = (int)propValue.getNumber();
                if (beam_size > 0) {
                    params.strategy = WHISPER_SAMPLING_BEAM_SEARCH;
                    params.beam_search.beam_size = beam_size;
                }
            } else if (propName == "bestOf" && propValue.isNumber()) {
                params.greedy.best_of = (int)propValue.getNumber();
            } else if (propName == "maxLen" && propValue.isNumber()) {
                params.max_len = (int)propValue.getNumber();
            } else if (propName == "maxContext" && propValue.isNumber()) {
                params.n_max_text_ctx = (int)propValue.getNumber();
            } else if (propName == "offset" && propValue.isNumber()) {
                params.offset_ms = (int)propValue.getNumber();
            } else if (propName == "duration" && propValue.isNumber()) {
                params.duration_ms = (int)propValue.getNumber();
            } else if (propName == "wordThold" && propValue.isNumber()) {
                params.thold_pt = (int)propValue.getNumber();
            } else if (propName == "temperature" && propValue.isNumber()) {
                params.temperature = (float)propValue.getNumber();
            } else if (propName == "temperatureInc" && propValue.isNumber()) {
                params.temperature_inc = (float)propValue.getNumber();
            } else if (propName == "prompt" && propValue.isString()) {
                std::string prompt = propValue.getString(runtime).utf8(runtime);
                params.initial_prompt = strdup(prompt.c_str());
            } else if (propName == "language" && propValue.isString()) {
                std::string language = propValue.getString(runtime).utf8(runtime);
                params.language = strdup(language.c_str());
            }
        }
    } catch (...) {
        // Use default values if parsing fails
    }

    params.offset_ms = 0;
    params.no_context = true;
    params.single_segment = false;

    return params;
}

// Helper function to convert ArrayBuffer to float32 audio data
struct AudioData {
    std::vector<float> data;
    int count;
};

AudioData convertArrayBufferToAudioData(Runtime& runtime, size_t arrayBufferSize, uint8_t* arrayBufferData) {
    // Convert ArrayBuffer to float32 array (assuming 16-bit PCM input)
    if (arrayBufferSize % 2 != 0) {
        throw JSError(runtime, "ArrayBuffer size must be even for 16-bit PCM data");
    }

    int audioDataCount = (int)(arrayBufferSize / 2); // 16-bit samples
    std::vector<float> audioData(audioDataCount);

    // Convert 16-bit PCM to float32
    int16_t* pcmData = (int16_t*)arrayBufferData;
    for (int i = 0; i < audioDataCount; i++) {
        audioData[i] = (float)pcmData[i] / 32768.0f;
    }

    return {std::move(audioData), audioDataCount};
}

void installJSIBindings(
    facebook::jsi::Runtime& runtime,
    std::shared_ptr<facebook::react::CallInvoker> callInvoker
) {
    try {
        // whisperTranscribeData function
        auto whisperTranscribeData = Function::createFromHostFunction(
            runtime,
            PropNameID::forAscii(runtime, "whisperTranscribeData"),
            3, // number of arguments
            [callInvoker](Runtime& runtime, const Value& thisValue, const Value* arguments, size_t count) -> Value {
                try {
                    if (count != 3) {
                        throw JSError(runtime, "whisperTranscribeData expects 3 arguments (contextId, options, arrayBuffer)");
                    }

                    if (!arguments[0].isNumber()) {
                        throw JSError(runtime, "whisperTranscribeData expects contextId to be a number");
                    }

                    if (!arguments[1].isObject()) {
                        throw JSError(runtime, "whisperTranscribeData expects options to be an object");
                    }

                    if (!arguments[2].isObject() || !arguments[2].getObject(runtime).isArrayBuffer(runtime)) {
                        throw JSError(runtime, "whisperTranscribeData expects third argument to be an ArrayBuffer");
                    }

                    int contextId = (int)arguments[0].getNumber();
                    auto optionsObj = arguments[1].getObject(runtime);
                    auto arrayBuffer = arguments[2].getObject(runtime).getArrayBuffer(runtime);

                    size_t arrayBufferSize = arrayBuffer.size(runtime);
                    uint8_t* arrayBufferData = arrayBuffer.data(runtime);

                    logInfo("whisperTranscribeData called with contextId=%d, arrayBuffer size=%zu", contextId, arrayBufferSize);
                    long contextPtr = getContextPtr(contextId);
                    if (contextPtr == 0) {
                        throw JSError(runtime, "Context not found for id: " + std::to_string(contextId));
                    }

                    struct whisper_context *context = reinterpret_cast<struct whisper_context *>(contextPtr);
                    if (context == nullptr) {
                        throw JSError(runtime, "Invalid context pointer for id: " + std::to_string(contextId));
                    }

                    // Convert ArrayBuffer to audio data
                    AudioData audioResult = convertArrayBufferToAudioData(runtime, arrayBufferSize, arrayBufferData);
                    auto& audioData = audioResult.data;
                    int audioDataCount = audioResult.count;

                    // Create whisper_full_params from JSI options
                    whisper_full_params params = createFullParamsFromJSI(runtime, optionsObj);

                    // Check for callbacks and jobId
                    std::shared_ptr<Function> onProgressCallback = nullptr;
                    std::shared_ptr<Function> onNewSegmentsCallback = nullptr;
                    int jobId = rand(); // Default fallback jobId

                    try {
                        auto propNames = optionsObj.getPropertyNames(runtime);
                        for (size_t i = 0; i < propNames.size(runtime); i++) {
                            auto propNameValue = propNames.getValueAtIndex(runtime, i);
                            std::string propName = propNameValue.getString(runtime).utf8(runtime);
                            Value propValue = optionsObj.getProperty(runtime, propNameValue.getString(runtime));

                            if (propName == "onProgress" && propValue.isObject() && propValue.getObject(runtime).isFunction(runtime)) {
                                onProgressCallback = std::make_shared<Function>(propValue.getObject(runtime).getFunction(runtime));
                            } else if (propName == "onNewSegments" && propValue.isObject() && propValue.getObject(runtime).isFunction(runtime)) {
                                onNewSegmentsCallback = std::make_shared<Function>(propValue.getObject(runtime).getFunction(runtime));
                            } else if (propName == "jobId" && propValue.isNumber()) {
                                jobId = (int)propValue.getNumber();
                            }
                        }
                    } catch (...) {
                        // Ignore callback detection errors
                    }

                    // Create a promise for async transcription
                    auto promiseConstructor = runtime.global().getPropertyAsFunction(runtime, "Promise");

                    auto promiseExecutor = Function::createFromHostFunction(
                        runtime,
                        PropNameID::forAscii(runtime, ""),
                        2, // resolve, reject
                        [context, audioData, audioDataCount, params, onProgressCallback, onNewSegmentsCallback, jobId, callInvoker](Runtime& runtime, const Value& thisValue, const Value* arguments, size_t count) -> Value {
                            if (count != 2) {
                                throw JSError(runtime, "Promise executor expects 2 arguments (resolve, reject)");
                            }

                            auto resolvePtr = std::make_shared<Function>(arguments[0].getObject(runtime).getFunction(runtime));
                            auto rejectPtr = std::make_shared<Function>(arguments[1].getObject(runtime).getFunction(runtime));

                            // Capture a safe reference to runtime
                            auto safeRuntime = std::shared_ptr<Runtime>(&runtime, [](Runtime*){});

                            // Use ThreadPool for CPU-intensive whisper operations
                            auto future = getWhisperThreadPool().enqueue([
                                context,
                                audioData, audioDataCount, params, onProgressCallback, onNewSegmentsCallback,
                                resolvePtr, rejectPtr, callInvoker, jobId, safeRuntime]() {

                                try {
                                    // Validate context is still alive before processing
                                    if (context == nullptr) {
                                        logError("Context is null in ThreadPool task");
                                        callInvoker->invokeAsync([rejectPtr, safeRuntime]() {
                                            auto& runtime = *safeRuntime;
                                            auto errorObj = Object(runtime);
                                            errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, "Context is null"));
                                            rejectPtr->call(runtime, errorObj);
                                        });
                                        return;
                                    }

                                    // Validate audio buffer
                                    if (audioData.empty() || audioDataCount <= 0) {
                                        logError("Invalid audio data: size=%zu, count=%d", audioData.size(), audioDataCount);
                                        callInvoker->invokeAsync([rejectPtr, safeRuntime]() {
                                            auto& runtime = *safeRuntime;
                                            auto errorObj = Object(runtime);
                                            errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, "Invalid audio data"));
                                            rejectPtr->call(runtime, errorObj);
                                        });
                                        return;
                                    }

                                    logInfo("Starting whisper_full: context=%p, audioDataCount=%d, jobId=%d",
                                           context, audioDataCount, jobId);
                                    whisper_reset_timings(context);

                                    // Setup callbacks for progress and new segments
                                    whisper_full_params mutable_params = params;

                                    // Progress callback setup
                                    struct progress_callback_data {
                                        std::shared_ptr<facebook::react::CallInvoker> callInvoker;
                                        std::shared_ptr<Function> callback;
                                        std::shared_ptr<Runtime> safeRuntime;
                                    };

                                    auto progress_data = std::make_shared<progress_callback_data>();
                                    progress_data->callInvoker = callInvoker;
                                    progress_data->callback = onProgressCallback;
                                    progress_data->safeRuntime = safeRuntime;

                                    if (onProgressCallback) {
                                        mutable_params.progress_callback = [](struct whisper_context* /*ctx*/, struct whisper_state* /*state*/, int progress, void* user_data) {
                                            auto* data_ptr = static_cast<std::shared_ptr<progress_callback_data>*>(user_data);
                                            if (data_ptr && *data_ptr) {
                                                auto data = *data_ptr;
                                                if (data->callInvoker && data->callback && data->safeRuntime) {
                                                    data->callInvoker->invokeAsync([progress, callback = data->callback, safeRuntime = data->safeRuntime]() {
                                                        try {
                                                            logInfo("Progress: %d%%", progress);
                                                            auto& runtime = *safeRuntime;
                                                            Value progressValue = Value(progress);
                                                            callback->call(runtime, progressValue);
                                                        } catch (...) {
                                                            logError("Error in progress callback");
                                                        }
                                                    });
                                                }
                                            }
                                        };
                                        mutable_params.progress_callback_user_data = &progress_data;
                                    }

                                    // New segments callback setup
                                    struct new_segments_callback_data {
                                        std::shared_ptr<facebook::react::CallInvoker> callInvoker;
                                        std::shared_ptr<Function> callback;
                                        std::shared_ptr<Runtime> safeRuntime;
                                        std::atomic<int> total_n_new;
                                    };

                                    auto segments_data = std::make_shared<new_segments_callback_data>();
                                    segments_data->callInvoker = callInvoker;
                                    segments_data->callback = onNewSegmentsCallback;
                                    segments_data->safeRuntime = safeRuntime;
                                    segments_data->total_n_new = 0;

                                    if (onNewSegmentsCallback) {
                                        mutable_params.new_segment_callback = [](struct whisper_context* ctx, struct whisper_state* /*state*/, int n_new, void* user_data) {
                                            auto* data_ptr = static_cast<std::shared_ptr<new_segments_callback_data>*>(user_data);
                                            if (data_ptr && *data_ptr) {
                                                auto data = *data_ptr;
                                                if (data->callInvoker && data->callback && data->safeRuntime && ctx) {
                                                    int current_total = data->total_n_new.fetch_add(n_new) + n_new;
                                                    data->callInvoker->invokeAsync([ctx, n_new, current_total, callback = data->callback, safeRuntime = data->safeRuntime]() {
                                                        try {
                                                            logInfo("New segments: %d (total: %d)", n_new, current_total);

                                                            auto& runtime = *safeRuntime;
                                                            auto resultObj = Object(runtime);
                                                            resultObj.setProperty(runtime, "nNew", Value(n_new));
                                                            resultObj.setProperty(runtime, "totalNNew", Value(current_total));
                                                            auto n_segments = whisper_full_n_segments(ctx);
                                                            auto segmentsArray = Array(runtime, n_segments);
                                                            std::string fullText = "";
                                                            for (int i = 0; i < n_segments; i++) {
                                                                const char* text = whisper_full_get_segment_text(ctx, i);
                                                                std::string segmentText(text);
                                                                fullText = fullText + segmentText;
                                                                auto segmentObj = Object(runtime);
                                                                segmentObj.setProperty(runtime, "text", String::createFromUtf8(runtime, segmentText));
                                                                segmentObj.setProperty(runtime, "t0", Value((double)whisper_full_get_segment_t0(ctx, i)));
                                                                segmentObj.setProperty(runtime, "t1", Value((double)whisper_full_get_segment_t1(ctx, i)));
                                                                segmentsArray.setValueAtIndex(runtime, i, segmentObj);
                                                            }
                                                            resultObj.setProperty(runtime, "segments", segmentsArray);
                                                            resultObj.setProperty(runtime, "result", String::createFromUtf8(runtime, fullText));
                                                            callback->call(runtime, resultObj);
                                                        } catch (...) {
                                                            logError("Error in new segments callback");
                                                        }
                                                    });
                                                }
                                            }
                                        };
                                        mutable_params.new_segment_callback_user_data = &segments_data;
                                    }

                                    // Create job for transcription with abort support
                                    rnwhisper::job* job = rnwhisper::job_new(jobId, mutable_params);
                                    int code = -1;

                                    if (job == nullptr) {
                                        logError("Failed to create job for transcription");
                                        code = -2;
                                    } else {
                                        code = whisper_full(context, job->params, audioData.data(), audioDataCount);
                                        if (job->is_aborted()) {
                                            code = -999;
                                        }
                                        rnwhisper::job_remove(jobId);
                                    }

                                    // Resolve with results on JS thread
                                    callInvoker->invokeAsync([resolvePtr, rejectPtr, code, context, safeRuntime]() {
                                        try {
                                            auto& runtime = *safeRuntime;
                                            if (code == 0) {
                                                int n_segments = whisper_full_n_segments(context);
                                                auto resultObj = Object(runtime);
                                                resultObj.setProperty(runtime, "code", Value(code));

                                                std::string fullText = "";
                                                auto segmentsArray = Array(runtime, n_segments);

                                                for (int i = 0; i < n_segments; i++) {
                                                    const char* text = whisper_full_get_segment_text(context, i);
                                                    std::string segmentText(text);
                                                    fullText += segmentText;

                                                    auto segmentObj = Object(runtime);
                                                    segmentObj.setProperty(runtime, "text", String::createFromUtf8(runtime, segmentText));
                                                    segmentObj.setProperty(runtime, "t0", Value((double)whisper_full_get_segment_t0(context, i)));
                                                    segmentObj.setProperty(runtime, "t1", Value((double)whisper_full_get_segment_t1(context, i)));
                                                    segmentsArray.setValueAtIndex(runtime, i, segmentObj);
                                                }

                                                resultObj.setProperty(runtime, "result", String::createFromUtf8(runtime, fullText));
                                                resultObj.setProperty(runtime, "segments", segmentsArray);
                                                resolvePtr->call(runtime, resultObj);
                                            } else {
                                                auto errorObj = Object(runtime);
                                                errorObj.setProperty(runtime, "code", Value(code));
                                                std::string errorMsg = (code == -2) ? "Failed to create transcription job" :
                                                                      (code == -999) ? "Transcription was aborted" :
                                                                      "Transcription failed";
                                                errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, errorMsg));
                                                rejectPtr->call(runtime, errorObj);
                                            }
                                        } catch (...) {
                                            auto& runtime = *safeRuntime;
                                            auto errorObj = Object(runtime);
                                            errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, "Unknown error"));
                                            rejectPtr->call(runtime, errorObj);
                                        }
                                    });
                                } catch (...) {
                                    callInvoker->invokeAsync([rejectPtr, safeRuntime]() {
                                        auto& runtime = *safeRuntime;
                                        auto errorObj = Object(runtime);
                                        errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, "Transcription error"));
                                        rejectPtr->call(runtime, errorObj);
                                    });
                                }
                            });

                            return Value::undefined();
                        }
                    );

                    return promiseConstructor.callAsConstructor(runtime, promiseExecutor);
                } catch (const JSError& e) {
                    throw;
                } catch (const std::exception& e) {
                    logError("Exception in whisperTranscribeData: %s", e.what());
                    throw JSError(runtime, std::string("whisperTranscribeData error: ") + e.what());
                } catch (...) {
                    logError("Unknown exception in whisperTranscribeData");
                    throw JSError(runtime, "whisperTranscribeData encountered unknown error");
                }
            }
        );

        // whisperVadDetectSpeech function
        auto whisperVadDetectSpeech = Function::createFromHostFunction(
            runtime,
            PropNameID::forAscii(runtime, "whisperVadDetectSpeech"),
            3, // number of arguments
            [callInvoker](Runtime& runtime, const Value& thisValue, const Value* arguments, size_t count) -> Value {
                try {
                    if (count != 3) {
                        throw JSError(runtime, "whisperVadDetectSpeech expects 3 arguments (contextId, options, arrayBuffer)");
                    }

                    if (!arguments[0].isNumber()) {
                        throw JSError(runtime, "whisperVadDetectSpeech expects contextId to be a number");
                    }

                    if (!arguments[1].isObject()) {
                        throw JSError(runtime, "whisperVadDetectSpeech expects options to be an object");
                    }

                    if (!arguments[2].isObject() || !arguments[2].getObject(runtime).isArrayBuffer(runtime)) {
                        throw JSError(runtime, "whisperVadDetectSpeech expects third argument to be an ArrayBuffer");
                    }

                    int contextId = (int)arguments[0].getNumber();
                    auto optionsObj = arguments[1].getObject(runtime);
                    auto arrayBuffer = arguments[2].getObject(runtime).getArrayBuffer(runtime);

                    size_t arrayBufferSize = arrayBuffer.size(runtime);
                    uint8_t* arrayBufferData = arrayBuffer.data(runtime);

                    logInfo("whisperVadDetectSpeech called with contextId=%d, arrayBuffer size=%zu", contextId, arrayBufferSize);

                    long vadContextPtr = getVadContextPtr(contextId);
                    if (vadContextPtr == 0) {
                        throw JSError(runtime, "VAD Context not found for id: " + std::to_string(contextId));
                    }

                    struct whisper_vad_context *vadContext = reinterpret_cast<struct whisper_vad_context *>(vadContextPtr);
                    if (vadContext == nullptr) {
                        throw JSError(runtime, "Invalid VAD context pointer for id: " + std::to_string(contextId));
                    }

                    // Convert ArrayBuffer to audio data
                    AudioData audioResult = convertArrayBufferToAudioData(runtime, arrayBufferSize, arrayBufferData);
                    auto& audioData = audioResult.data;
                    int audioDataCount = audioResult.count;

                    auto promiseConstructor = runtime.global().getPropertyAsFunction(runtime, "Promise");

                    auto promiseExecutor = Function::createFromHostFunction(
                        runtime,
                        PropNameID::forAscii(runtime, ""),
                        2, // resolve, reject
                        [vadContext, audioData, audioDataCount, callInvoker](Runtime& runtime, const Value& thisValue, const Value* arguments, size_t count) -> Value {
                            if (count != 2) {
                                throw JSError(runtime, "Promise executor expects 2 arguments (resolve, reject)");
                            }

                            auto resolvePtr = std::make_shared<Function>(arguments[0].getObject(runtime).getFunction(runtime));
                            auto rejectPtr = std::make_shared<Function>(arguments[1].getObject(runtime).getFunction(runtime));

                            // Capture a safe reference to runtime
                            auto safeRuntime = std::shared_ptr<Runtime>(&runtime, [](Runtime*){});

                            // Use ThreadPool for CPU-intensive VAD operations
                            auto future = getWhisperThreadPool().enqueue([
                                vadContext,
                                audioData, audioDataCount, resolvePtr, rejectPtr, callInvoker, safeRuntime]() {

                                try {
                                    // Validate VAD context is still alive before processing
                                    if (vadContext == nullptr) {
                                        logError("VAD context is null in ThreadPool task");
                                        callInvoker->invokeAsync([rejectPtr, safeRuntime]() {
                                            auto& runtime = *safeRuntime;
                                            auto errorObj = Object(runtime);
                                            errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, "VAD context is null"));
                                            rejectPtr->call(runtime, errorObj);
                                        });
                                        return;
                                    }

                                    // Validate audio buffer
                                    if (audioData.empty() || audioDataCount <= 0) {
                                        logError("Invalid audio data: size=%zu, count=%d", audioData.size(), audioDataCount);
                                        callInvoker->invokeAsync([rejectPtr, safeRuntime]() {
                                            auto& runtime = *safeRuntime;
                                            auto errorObj = Object(runtime);
                                            errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, "Invalid audio data"));
                                            rejectPtr->call(runtime, errorObj);
                                        });
                                        return;
                                    }

                                    logInfo("Starting whisper_vad_detect_speech: vadContext=%p, audioDataCount=%d",
                                           vadContext, audioDataCount);
                                    bool isSpeech = whisper_vad_detect_speech(vadContext, audioData.data(), audioDataCount);
                                    logInfo("VAD detection result: %s", isSpeech ? "speech" : "no speech");

                                    struct whisper_vad_params vad_params = whisper_vad_default_params();
                                    struct whisper_vad_segments* segments = nullptr;
                                    if (isSpeech) {
                                        segments = whisper_vad_segments_from_probs(vadContext, vad_params);
                                    }

                                    // Process results on JS thread
                                    callInvoker->invokeAsync([resolvePtr, rejectPtr, segments, safeRuntime]() {
                                        try {
                                            auto& runtime = *safeRuntime;
                                            auto resultObj = Object(runtime);

                                            if (segments) {
                                                int n_segments = whisper_vad_segments_n_segments(segments);
                                                resultObj.setProperty(runtime, "hasSpeech", Value(n_segments > 0));
                                                auto segmentsArray = Array(runtime, n_segments);

                                                for (int i = 0; i < n_segments; i++) {
                                                    auto segmentObj = Object(runtime);
                                                    segmentObj.setProperty(runtime, "t0", Value((double)whisper_vad_segments_get_segment_t0(segments, i)));
                                                    segmentObj.setProperty(runtime, "t1", Value((double)whisper_vad_segments_get_segment_t1(segments, i)));
                                                    segmentsArray.setValueAtIndex(runtime, i, segmentObj);
                                                }

                                                resultObj.setProperty(runtime, "segments", segmentsArray);
                                                whisper_vad_free_segments(segments);
                                            } else {
                                                resultObj.setProperty(runtime, "hasSpeech", Value(false));
                                                resultObj.setProperty(runtime, "segments", Array(runtime, 0));
                                            }

                                            resolvePtr->call(runtime, resultObj);
                                        } catch (...) {
                                            auto& runtime = *safeRuntime;
                                            auto errorObj = Object(runtime);
                                            errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, "VAD result processing error"));
                                            rejectPtr->call(runtime, errorObj);
                                        }
                                    });
                                } catch (...) {
                                    callInvoker->invokeAsync([rejectPtr, safeRuntime]() {
                                        auto& runtime = *safeRuntime;
                                        auto errorObj = Object(runtime);
                                        errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, "VAD detection error"));
                                        rejectPtr->call(runtime, errorObj);
                                    });
                                }
                            });

                            return Value::undefined();
                        }
                    );

                    return promiseConstructor.callAsConstructor(runtime, promiseExecutor);
                } catch (const JSError& e) {
                    throw;
                } catch (const std::exception& e) {
                    logError("Exception in whisperVadDetectSpeech: %s", e.what());
                    throw JSError(runtime, std::string("whisperVadDetectSpeech error: ") + e.what());
                } catch (...) {
                    logError("Unknown exception in whisperVadDetectSpeech");
                    throw JSError(runtime, "whisperVadDetectSpeech encountered unknown error");
                }
            }
        );

        // Install the JSI functions on the global object
        runtime.global().setProperty(runtime, "whisperTranscribeData", std::move(whisperTranscribeData));
        runtime.global().setProperty(runtime, "whisperVadDetectSpeech", std::move(whisperVadDetectSpeech));

        logInfo("JSI bindings installed successfully");
    } catch (const JSError& e) {
        logError("JSError installing JSI bindings: %s", e.getMessage().c_str());
        throw;
    } catch (const std::exception& e) {
        logError("Exception installing JSI bindings: %s", e.what());
        throw JSError(runtime, std::string("Failed to install JSI bindings: ") + e.what());
    } catch (...) {
        logError("Unknown exception installing JSI bindings");
        throw JSError(runtime, "Failed to install JSI bindings: unknown error");
    }
}

// Cleanup function to dispose of ThreadPool
void cleanupJSIBindings() {
    std::lock_guard<std::mutex> lock(threadPoolMutex);
    if (whisperThreadPool) {
        logInfo("Cleaning up ThreadPool");
        whisperThreadPool.reset();
    }
}

} // namespace rnwhisper_jsi
