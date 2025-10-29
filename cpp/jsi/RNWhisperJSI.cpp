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

using namespace facebook::jsi;

// Consolidated logging function
enum class LogLevel { LOG_DEBUG, LOG_INFO, LOG_ERROR };

static void log(LogLevel level, const char* format, ...) {
    va_list args;
    va_start(args, format);

#if defined(__ANDROID__)
    int androidLevel = (level == LogLevel::LOG_DEBUG) ? ANDROID_LOG_DEBUG :
                      (level == LogLevel::LOG_INFO) ? ANDROID_LOG_INFO : ANDROID_LOG_ERROR;
    __android_log_vprint(androidLevel, "RNWhisperJSI", format, args);
#else
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    const char* levelStr = (level == LogLevel::LOG_DEBUG) ? "DEBUG" :
                          (level == LogLevel::LOG_INFO) ? "INFO" : "ERROR";
    printf("RNWhisperJSI %s: %s\n", levelStr, buffer);
#endif

    va_end(args);
}

#define logInfo(format, ...) log(LogLevel::LOG_INFO, format, ##__VA_ARGS__)
#define logError(format, ...) log(LogLevel::LOG_ERROR, format, ##__VA_ARGS__)
#define logDebug(format, ...) log(LogLevel::LOG_DEBUG, format, ##__VA_ARGS__)

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

// Template-based context management
template<typename T>
class ContextManager {
private:
    std::unordered_map<int, long> contextMap;
    std::mutex contextMutex;
    const char* contextType;

public:
    ContextManager(const char* type) : contextType(type) {}

    void add(int contextId, long contextPtr) {
        std::lock_guard<std::mutex> lock(contextMutex);
        contextMap[contextId] = contextPtr;
        logDebug("%s Context added: id=%d, ptr=%ld", contextType, contextId, contextPtr);
    }

    void remove(int contextId) {
        std::lock_guard<std::mutex> lock(contextMutex);
        auto it = contextMap.find(contextId);
        if (it != contextMap.end()) {
            logDebug("%s Context removed: id=%d", contextType, contextId);
            contextMap.erase(it);
        }
    }

    long get(int contextId) {
        std::lock_guard<std::mutex> lock(contextMutex);
        auto it = contextMap.find(contextId);
        return (it != contextMap.end()) ? it->second : 0;
    }

    T* getTyped(int contextId) {
        long ptr = get(contextId);
        return ptr ? reinterpret_cast<T*>(ptr) : nullptr;
    }
};

static ContextManager<whisper_context> contextManager("Whisper");
static ContextManager<whisper_vad_context> vadContextManager("VAD");

// Context management functions
void addContext(int contextId, long contextPtr) {
    contextManager.add(contextId, contextPtr);
}

void removeContext(int contextId) {
    contextManager.remove(contextId);
}

void addVadContext(int contextId, long vadContextPtr) {
    vadContextManager.add(contextId, vadContextPtr);
}

void removeVadContext(int contextId) {
    vadContextManager.remove(contextId);
}

long getContextPtr(int contextId) {
    return contextManager.get(contextId);
}

long getVadContextPtr(int contextId) {
    return vadContextManager.get(contextId);
}

// Helper function to validate JSI function arguments
struct JSIValidationResult {
    bool isValid;
    std::string errorMessage;
};

JSIValidationResult validateJSIArguments(Runtime& runtime, const Value* arguments, size_t count, size_t expectedCount) {
    if (count != expectedCount) {
        return {false, "Expected " + std::to_string(expectedCount) + " arguments, got " + std::to_string(count)};
    }

    if (!arguments[0].isNumber()) {
        return {false, "First argument (contextId) must be a number"};
    }

    if (!arguments[1].isObject()) {
        return {false, "Second argument (options) must be an object"};
    }

    if (!arguments[2].isObject() || !arguments[2].getObject(runtime).isArrayBuffer(runtime)) {
        return {false, "Third argument must be an ArrayBuffer"};
    }

    return {true, ""};
}

// Helper function to create error objects
Object createErrorObject(Runtime& runtime, const std::string& message, int code = -1) {
    auto errorObj = Object(runtime);
    errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, message));
    if (code != -1) {
        errorObj.setProperty(runtime, "code", Value(code));
    }
    return errorObj;
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

// Common callback data structure
template<typename CallbackType>
struct CallbackData {
    std::shared_ptr<facebook::react::CallInvoker> callInvoker;
    std::shared_ptr<Function> callback;
    std::shared_ptr<Runtime> safeRuntime;
    std::atomic<int> counter{0};
};

// Helper function to extract callbacks from options
struct CallbackInfo {
    std::shared_ptr<Function> onProgressCallback;
    std::shared_ptr<Function> onNewSegmentsCallback;
    int jobId;
    int nProcessors;
};

CallbackInfo extractCallbacks(Runtime& runtime, const Object& optionsObj) {
    CallbackInfo info;
    info.jobId = rand(); // Default fallback jobId
    info.nProcessors = 1; // Default to 1 processor

    try {
        auto propNames = optionsObj.getPropertyNames(runtime);
        for (size_t i = 0; i < propNames.size(runtime); i++) {
            auto propNameValue = propNames.getValueAtIndex(runtime, i);
            std::string propName = propNameValue.getString(runtime).utf8(runtime);
            Value propValue = optionsObj.getProperty(runtime, propNameValue.getString(runtime));

            if (propName == "onProgress" && propValue.isObject() && propValue.getObject(runtime).isFunction(runtime)) {
                info.onProgressCallback = std::make_shared<Function>(propValue.getObject(runtime).getFunction(runtime));
            } else if (propName == "onNewSegments" && propValue.isObject() && propValue.getObject(runtime).isFunction(runtime)) {
                info.onNewSegmentsCallback = std::make_shared<Function>(propValue.getObject(runtime).getFunction(runtime));
            } else if (propName == "jobId" && propValue.isNumber()) {
                info.jobId = (int)propValue.getNumber();
            } else if (propName == "nProcessors" && propValue.isNumber()) {
                info.nProcessors = (int)propValue.getNumber();
            }
        }
    } catch (...) {
        // Ignore callback detection errors
    }

    return info;
}

// Helper function to extract VAD parameters from options
whisper_vad_params extractVadParams(Runtime& runtime, const Object& optionsObj) {
    whisper_vad_params vadParams = whisper_vad_default_params();

    try {
        auto propNames = optionsObj.getPropertyNames(runtime);
        for (size_t i = 0; i < propNames.size(runtime); i++) {
            auto propNameValue = propNames.getValueAtIndex(runtime, i);
            std::string propName = propNameValue.getString(runtime).utf8(runtime);
            Value propValue = optionsObj.getProperty(runtime, propNameValue.getString(runtime));

            if (propName == "threshold" && propValue.isNumber()) {
                vadParams.threshold = (float)propValue.getNumber();
            } else if (propName == "minSpeechDurationMs" && propValue.isNumber()) {
                vadParams.min_speech_duration_ms = (int)propValue.getNumber();
            } else if (propName == "minSilenceDurationMs" && propValue.isNumber()) {
                vadParams.min_silence_duration_ms = (int)propValue.getNumber();
            } else if (propName == "maxSpeechDurationS" && propValue.isNumber()) {
                vadParams.max_speech_duration_s = (float)propValue.getNumber();
            } else if (propName == "speechPadMs" && propValue.isNumber()) {
                vadParams.speech_pad_ms = (int)propValue.getNumber();
            } else if (propName == "samplesOverlap" && propValue.isNumber()) {
                vadParams.samples_overlap = (float)propValue.getNumber();
            }
        }
    } catch (...) {
        // Ignore parameter extraction errors
    }

    return vadParams;
}

// Helper function to create segments array
Array createSegmentsArray(Runtime& runtime, struct whisper_context* ctx, int offset) {
    int n_segments = whisper_full_n_segments(ctx);
    auto segmentsArray = Array(runtime, n_segments);

    for (int i = offset; i < n_segments; i++) {
        const char* text = whisper_full_get_segment_text(ctx, i);
        auto segmentObj = Object(runtime);
        segmentObj.setProperty(runtime, "text", String::createFromUtf8(runtime, text));
        segmentObj.setProperty(runtime, "t0", Value((double)whisper_full_get_segment_t0(ctx, i)));
        segmentObj.setProperty(runtime, "t1", Value((double)whisper_full_get_segment_t1(ctx, i)));
        segmentsArray.setValueAtIndex(runtime, i, segmentObj);
    }

    return segmentsArray;
}

// Helper function to create full text from segments
std::string createFullTextFromSegments(struct whisper_context* ctx, int offset) {
    int n_segments = whisper_full_n_segments(ctx);
    std::string fullText = "";

    for (int i = offset; i < n_segments; i++) {
        const char* text = whisper_full_get_segment_text(ctx, i);
        fullText += text;
    }

    return fullText;
}

// Helper function to create and execute promise-based operations
template<typename ContextType, typename TaskFunc>
Value createPromiseTask(
    Runtime& runtime,
    const std::string& functionName,
    std::shared_ptr<facebook::react::CallInvoker> callInvoker,
    const Value* arguments,
    size_t count,
    TaskFunc task
) {
    // Validate arguments
    auto validation = validateJSIArguments(runtime, arguments, count, 3);
    if (!validation.isValid) {
        throw JSError(runtime, functionName + " " + validation.errorMessage);
    }

    int contextId = (int)arguments[0].getNumber();
    auto optionsObj = arguments[1].getObject(runtime);
    auto arrayBuffer = arguments[2].getObject(runtime).getArrayBuffer(runtime);

    size_t arrayBufferSize = arrayBuffer.size(runtime);
    uint8_t* arrayBufferData = arrayBuffer.data(runtime);

    logInfo("%s called with contextId=%d, arrayBuffer size=%zu", functionName.c_str(), contextId, arrayBufferSize);

    // Convert ArrayBuffer to audio data
    AudioData audioResult = convertArrayBufferToAudioData(runtime, arrayBufferSize, arrayBufferData);

    whisper_full_params params = {};
    CallbackInfo callbackInfo = {};
    whisper_vad_params vadParams = {};
    if (functionName == "whisperTranscribeData") {
        params = createFullParamsFromJSI(runtime, optionsObj);
        // Extract data from optionsObj before lambda capture
        callbackInfo = extractCallbacks(runtime, optionsObj);
    } else if (functionName == "whisperVadDetectSpeech") {
        vadParams = extractVadParams(runtime, optionsObj);
    }

    // Create promise
    auto promiseConstructor = runtime.global().getPropertyAsFunction(runtime, "Promise");

    auto promiseExecutor = Function::createFromHostFunction(
        runtime,
        PropNameID::forAscii(runtime, ""),
        2, // resolve, reject
        [contextId, audioResult, params, callbackInfo, vadParams, task, callInvoker, functionName](Runtime& runtime, const Value& thisValue, const Value* arguments, size_t count) -> Value {
            if (count != 2) {
                throw JSError(runtime, "Promise executor expects 2 arguments (resolve, reject)");
            }

            auto resolvePtr = std::make_shared<Function>(arguments[0].getObject(runtime).getFunction(runtime));
            auto rejectPtr = std::make_shared<Function>(arguments[1].getObject(runtime).getFunction(runtime));
            auto safeRuntime = std::shared_ptr<Runtime>(&runtime, [](Runtime*){});

            // Execute task in ThreadPool
            auto future = getWhisperThreadPool().enqueue([
                contextId, audioResult, params, callbackInfo, vadParams, task, resolvePtr, rejectPtr, callInvoker, safeRuntime, functionName]() {

                try {
                    task(contextId, audioResult, params, callbackInfo, vadParams, resolvePtr, rejectPtr, callInvoker, safeRuntime);
                } catch (...) {
                    callInvoker->invokeAsync([rejectPtr, safeRuntime, functionName]() {
                        auto& runtime = *safeRuntime;
                        auto errorObj = createErrorObject(runtime, functionName + " processing error");
                        rejectPtr->call(runtime, errorObj);
                    });
                }
            });

            return Value::undefined();
        }
    );

    return promiseConstructor.callAsConstructor(runtime, promiseExecutor);
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
                    return createPromiseTask<whisper_context>(
                        runtime, "whisperTranscribeData", callInvoker, arguments, count,
                        [](int contextId, const AudioData& audioResult, const whisper_full_params& params, const CallbackInfo& callbackInfo, const whisper_vad_params& vadParams,
                           std::shared_ptr<Function> resolvePtr, std::shared_ptr<Function> rejectPtr,
                           std::shared_ptr<facebook::react::CallInvoker> callInvoker,
                           std::shared_ptr<Runtime> safeRuntime) {

                            // Get context
                            auto context = contextManager.getTyped(contextId);
                            if (!context) {
                                callInvoker->invokeAsync([rejectPtr, safeRuntime, contextId]() {
                                    auto& runtime = *safeRuntime;
                                    auto errorObj = createErrorObject(runtime, "Context not found for id: " + std::to_string(contextId));
                                    rejectPtr->call(runtime, errorObj);
                                });
                                return;
                            }

                            // Validate audio data
                            if (audioResult.data.empty() || audioResult.count <= 0) {
                                logError("Invalid audio data: size=%zu, count=%d", audioResult.data.size(), audioResult.count);
                                callInvoker->invokeAsync([rejectPtr, safeRuntime]() {
                                    auto& runtime = *safeRuntime;
                                    auto errorObj = createErrorObject(runtime, "Invalid audio data");
                                    rejectPtr->call(runtime, errorObj);
                                });
                                return;
                            }

                            logInfo("Starting whisper_full: context=%p, audioDataCount=%d, jobId=%d",
                                   context, audioResult.count, callbackInfo.jobId);
                            whisper_reset_timings(context);

                            // Setup callbacks
                            whisper_full_params mutable_params = params;
                            auto progress_data = std::make_shared<CallbackData<Function>>();
                            progress_data->callInvoker = callInvoker;
                            progress_data->callback = callbackInfo.onProgressCallback;
                            progress_data->safeRuntime = safeRuntime;

                            if (callbackInfo.onProgressCallback) {
                                mutable_params.progress_callback = [](struct whisper_context* /*ctx*/, struct whisper_state* /*state*/, int progress, void* user_data) {
                                    auto* data_ptr = static_cast<std::shared_ptr<CallbackData<Function>>*>(user_data);
                                    if (data_ptr && *data_ptr) {
                                        auto data = *data_ptr;
                                        if (data->callInvoker && data->callback && data->safeRuntime) {
                                            data->callInvoker->invokeAsync([progress, callback = data->callback, safeRuntime = data->safeRuntime]() {
                                                try {
                                                    logInfo("Progress: %d%%", progress);
                                                    auto& runtime = *safeRuntime;
                                                    callback->call(runtime, Value(progress));
                                                } catch (...) {
                                                    logError("Error in progress callback");
                                                }
                                            });
                                        }
                                    }
                                };
                                mutable_params.progress_callback_user_data = &progress_data;
                            }

                            auto segments_data = std::make_shared<CallbackData<Function>>();
                            segments_data->callInvoker = callInvoker;
                            segments_data->callback = callbackInfo.onNewSegmentsCallback;
                            segments_data->safeRuntime = safeRuntime;

                            if (callbackInfo.onNewSegmentsCallback) {
                                mutable_params.new_segment_callback = [](struct whisper_context* ctx, struct whisper_state* /*state*/, int n_new, void* user_data) {
                                    auto* data_ptr = static_cast<std::shared_ptr<CallbackData<Function>>*>(user_data);
                                    if (data_ptr && *data_ptr) {
                                        auto data = *data_ptr;
                                        if (data->callInvoker && data->callback && data->safeRuntime && ctx) {
                                            int current_total = data->counter.fetch_add(n_new) + n_new;
                                            data->callInvoker->invokeAsync([ctx, n_new, current_total, callback = data->callback, safeRuntime = data->safeRuntime]() {
                                                try {
                                                    logInfo("New segments: %d (total: %d)", n_new, current_total);
                                                    auto& runtime = *safeRuntime;
                                                    auto resultObj = Object(runtime);
                                                    resultObj.setProperty(runtime, "nNew", Value(n_new));
                                                    resultObj.setProperty(runtime, "totalNNew", Value(current_total));
                                                    auto offset = current_total - n_new;
                                                    resultObj.setProperty(runtime, "segments", createSegmentsArray(runtime, ctx, offset));
                                                    resultObj.setProperty(runtime, "result", String::createFromUtf8(runtime, createFullTextFromSegments(ctx, offset)));
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

                            // Execute transcription
                            rnwhisper::job* job = rnwhisper::job_new(callbackInfo.jobId, mutable_params);
                            int code = -1;

                            if (job == nullptr) {
                                logError("Failed to create job for transcription");
                                code = -2;
                            } else {
                                try {
                                    job->n_processors = callbackInfo.nProcessors;
                                    code = whisper_full_parallel(context, job->params, audioResult.data.data(), audioResult.count, job->n_processors);
                                    if (job->is_aborted()) {
                                        code = -999;
                                    }
                                } catch (...) {
                                    logError("Exception during whisper_full_parallel transcription");
                                    code = -3;
                                }
                                rnwhisper::job_remove(callbackInfo.jobId);
                            }

                            // Resolve with results
                            callInvoker->invokeAsync([resolvePtr, rejectPtr, code, context, safeRuntime]() {
                                try {
                                    auto& runtime = *safeRuntime;
                                    if (code == 0) {
                                        auto resultObj = Object(runtime);
                                        resultObj.setProperty(runtime, "code", Value(code));
                                        resultObj.setProperty(runtime, "result", String::createFromUtf8(runtime, createFullTextFromSegments(context, 0)));
                                        resultObj.setProperty(runtime, "segments", createSegmentsArray(runtime, context, 0));
                                        resolvePtr->call(runtime, resultObj);
                                    } else {
                                        std::string errorMsg = (code == -2) ? "Failed to create transcription job" :
                                                              (code == -3) ? "Transcription failed with exception" :
                                                              (code == -999) ? "Transcription was aborted" :
                                                              "Transcription failed";
                                        auto errorObj = createErrorObject(runtime, errorMsg, code);
                                        rejectPtr->call(runtime, errorObj);
                                    }
                                } catch (...) {
                                    auto& runtime = *safeRuntime;
                                    auto errorObj = createErrorObject(runtime, "Unknown error");
                                    rejectPtr->call(runtime, errorObj);
                                }
                            });
                        }
                    );
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
                    return createPromiseTask<whisper_vad_context>(
                        runtime, "whisperVadDetectSpeech", callInvoker, arguments, count,
                        [](int contextId, const AudioData& audioResult, const whisper_full_params& params, const CallbackInfo& callbackInfo, const whisper_vad_params& vadParams,
                           std::shared_ptr<Function> resolvePtr, std::shared_ptr<Function> rejectPtr,
                           std::shared_ptr<facebook::react::CallInvoker> callInvoker,
                           std::shared_ptr<Runtime> safeRuntime) {

                            // Get VAD context
                            auto vadContext = vadContextManager.getTyped(contextId);
                            if (!vadContext) {
                                callInvoker->invokeAsync([rejectPtr, safeRuntime, contextId]() {
                                    auto& runtime = *safeRuntime;
                                    auto errorObj = createErrorObject(runtime, "VAD Context not found for id: " + std::to_string(contextId));
                                    rejectPtr->call(runtime, errorObj);
                                });
                                return;
                            }

                            // Validate audio data
                            if (audioResult.data.empty() || audioResult.count <= 0) {
                                logError("Invalid audio data: size=%zu, count=%d", audioResult.data.size(), audioResult.count);
                                callInvoker->invokeAsync([rejectPtr, safeRuntime]() {
                                    auto& runtime = *safeRuntime;
                                    auto errorObj = createErrorObject(runtime, "Invalid audio data");
                                    rejectPtr->call(runtime, errorObj);
                                });
                                return;
                            }

                            logInfo("Starting whisper_vad_detect_speech: vadContext=%p, audioDataCount=%d",
                                   vadContext, audioResult.count);

                            // Perform VAD detection with error handling
                            bool isSpeech = false;
                            try {
                                isSpeech = whisper_vad_detect_speech(vadContext, audioResult.data.data(), audioResult.count);
                                logInfo("VAD detection result: %s", isSpeech ? "speech" : "no speech");
                            } catch (...) {
                                logError("Exception during whisper_vad_detect_speech");
                                callInvoker->invokeAsync([rejectPtr, safeRuntime]() {
                                    auto& runtime = *safeRuntime;
                                    auto errorObj = createErrorObject(runtime, "VAD detection failed with exception");
                                    rejectPtr->call(runtime, errorObj);
                                });
                                return;
                            }

                            struct whisper_vad_params vad_params = vadParams;

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
                                    auto errorObj = createErrorObject(runtime, "VAD result processing error");
                                    rejectPtr->call(runtime, errorObj);
                                }
                            });
                        }
                    );
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
