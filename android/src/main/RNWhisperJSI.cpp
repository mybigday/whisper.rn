#include "RNWhisperJSI.h"
#include <jsi/jsi.h>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <android/log.h>
#include "whisper.h"

using namespace facebook::jsi;

namespace rnwhisper {

// Contexts map (contextId -> contextPtr) with thread safety
static std::unordered_map<int, long> contextMap;
static std::mutex contextMapMutex;

// VAD Contexts map (contextId -> vadContextPtr) with thread safety
static std::unordered_map<int, long> vadContextMap;
static std::mutex vadContextMapMutex;

void addContext(int contextId, long contextPtr) {
    std::lock_guard<std::mutex> lock(contextMapMutex);
    contextMap[contextId] = contextPtr;
    __android_log_print(ANDROID_LOG_DEBUG, "RNWhisperJSI",
        "Context added: id=%d, ptr=%ld", contextId, contextPtr);
}

void removeContext(int contextId) {
    std::lock_guard<std::mutex> lock(contextMapMutex);
    auto it = contextMap.find(contextId);
    if (it != contextMap.end()) {
        __android_log_print(ANDROID_LOG_DEBUG, "RNWhisperJSI",
            "Context removed: id=%d", contextId);
        contextMap.erase(it);
    }
}

void addVadContext(int contextId, long vadContextPtr) {
    std::lock_guard<std::mutex> lock(vadContextMapMutex);
    vadContextMap[contextId] = vadContextPtr;
    __android_log_print(ANDROID_LOG_DEBUG, "RNWhisperJSI",
        "VAD Context added: id=%d, ptr=%ld", contextId, vadContextPtr);
}

void removeVadContext(int contextId) {
    std::lock_guard<std::mutex> lock(vadContextMapMutex);
    auto it = vadContextMap.find(contextId);
    if (it != vadContextMap.end()) {
        __android_log_print(ANDROID_LOG_DEBUG, "RNWhisperJSI",
            "VAD Context removed: id=%d", contextId);
        vadContextMap.erase(it);
    }
}

// Helper function to safely get context pointer
static long getContextPtr(int contextId) {
    std::lock_guard<std::mutex> lock(contextMapMutex);
    auto it = contextMap.find(contextId);
    return (it != contextMap.end()) ? it->second : 0;
}

// Helper function to safely get VAD context pointer
static long getVadContextPtr(int contextId) {
    std::lock_guard<std::mutex> lock(vadContextMapMutex);
    auto it = vadContextMap.find(contextId);
    return (it != vadContextMap.end()) ? it->second : 0;
}

// Helper function to convert JSI object to whisper_full_params
static whisper_full_params createFullParamsFromJSI(Runtime& runtime, const Object& optionsObj) {
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

void installJSIBindings(
    facebook::jsi::Runtime& runtime,
    std::shared_ptr<facebook::react::CallInvoker> callInvoker,
    JNIEnv* env,
    jobject javaModule
) {
    try {
        // Test function to verify JSI bindings are working
        auto whisperTestContext = Function::createFromHostFunction(
            runtime,
            PropNameID::forAscii(runtime, "whisperTestContext"),
            2, // number of arguments
            [](Runtime& runtime, const Value& thisValue, const Value* arguments, size_t count) -> Value {
                try {
                    if (count != 2) {
                        throw JSError(runtime, "whisperTestContext expects 2 arguments (contextId, arrayBuffer)");
                    }

                    if (!arguments[0].isNumber()) {
                        throw JSError(runtime, "whisperTestContext expects contextId to be a number");
                    }

                    if (!arguments[1].isObject() || !arguments[1].getObject(runtime).isArrayBuffer(runtime)) {
                        throw JSError(runtime, "whisperTestContext expects second argument to be an ArrayBuffer");
                    }

                    int contextId = (int)arguments[0].getNumber();

                    // Get ArrayBuffer data
                    auto arrayBuffer = arguments[1].getObject(runtime).getArrayBuffer(runtime);
                    size_t arrayBufferSize = arrayBuffer.size(runtime);
                    uint8_t* arrayBufferData = arrayBuffer.data(runtime);

                    __android_log_print(ANDROID_LOG_INFO, "RNWhisperJSI",
                        "whisperTestContext called with contextId=%d, arrayBuffer size=%zu", contextId, arrayBufferSize);

                    // Thread-safe context lookup
                    long contextPtr = getContextPtr(contextId);

                    if (contextPtr == 0) {
                        __android_log_print(ANDROID_LOG_DEBUG, "RNWhisperJSI",
                            "Context not found for id=%d", contextId);
                        return Value::undefined();
                    }

                    // Validate context pointer before casting
                    struct whisper_context *context = reinterpret_cast<struct whisper_context *>(contextPtr);
                    if (context == nullptr) {
                        __android_log_print(ANDROID_LOG_WARN, "RNWhisperJSI",
                            "Context pointer is null for id=%d", contextId);
                        return Value::undefined();
                    }

                    __android_log_print(ANDROID_LOG_INFO, "RNWhisperJSI",
                        "Context validated successfully for id=%d, processed ArrayBuffer with %zu bytes", contextId, arrayBufferSize);
                    return Value(true);
                } catch (const JSError& e) {
                    // Re-throw JSError
                    throw;
                } catch (const std::exception& e) {
                    __android_log_print(ANDROID_LOG_ERROR, "RNWhisperJSI",
                        "Exception in whisperTestContext: %s", e.what());
                    throw JSError(runtime, std::string("whisperTestContext error: ") + e.what());
                } catch (...) {
                    __android_log_print(ANDROID_LOG_ERROR, "RNWhisperJSI",
                        "Unknown exception in whisperTestContext");
                    throw JSError(runtime, "whisperTestContext encountered unknown error");
                }
            }
        );

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

                    __android_log_print(ANDROID_LOG_INFO, "RNWhisperJSI",
                        "whisperTranscribeData called with contextId=%d, arrayBuffer size=%zu", contextId, arrayBufferSize);

                    // Thread-safe context lookup
                    long contextPtr = getContextPtr(contextId);
                    if (contextPtr == 0) {
                        throw JSError(runtime, "Context not found for id: " + std::to_string(contextId));
                    }

                    struct whisper_context *context = reinterpret_cast<struct whisper_context *>(contextPtr);
                    if (context == nullptr) {
                        throw JSError(runtime, "Invalid context pointer for id: " + std::to_string(contextId));
                    }

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

                    // Create whisper_full_params from JSI options
                    whisper_full_params params = createFullParamsFromJSI(runtime, optionsObj);

                    // Create a promise for async transcription
                    auto promiseConstructor = runtime.global().getPropertyAsFunction(runtime, "Promise");

                    auto promiseExecutor = Function::createFromHostFunction(
                        runtime,
                        PropNameID::forAscii(runtime, ""),
                        2, // resolve, reject
                        [context, audioData, audioDataCount, params, callInvoker](Runtime& runtime, const Value& thisValue, const Value* arguments, size_t count) -> Value {
                            if (count != 2) {
                                throw JSError(runtime, "Promise executor expects 2 arguments (resolve, reject)");
                            }

                            // Create shared pointers to keep the functions alive
                            auto resolvePtr = std::make_shared<Function>(arguments[0].getObject(runtime).getFunction(runtime));
                            auto rejectPtr = std::make_shared<Function>(arguments[1].getObject(runtime).getFunction(runtime));

                            // Run transcription on background thread
                            if (callInvoker) {
                                callInvoker->invokeAsync([context, audioData, audioDataCount, params, resolvePtr, rejectPtr, &runtime]() {
                                    try {
                                        // Reset timings before transcription
                                        whisper_reset_timings(context);

                                        // Run transcription
                                        int code = whisper_full(context, params, audioData.data(), audioDataCount);

                                        // Call resolve/reject on JS thread
                                        if (code == 0) {
                                            // Get results
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
                                            errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, "Transcription failed"));
                                            rejectPtr->call(runtime, errorObj);
                                        }
                                    } catch (const JSError& e) {
                                        throw;
                                    } catch (const std::exception& e) {
                                        auto errorObj = Object(runtime);
                                        errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, e.what()));
                                        rejectPtr->call(runtime, errorObj);
                                    } catch (...) {
                                        auto errorObj = Object(runtime);
                                        errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, "Unknown error"));
                                        rejectPtr->call(runtime, errorObj);
                                    }
                                });
                            } else {
                                // Fallback: run synchronously
                                try {
                                    whisper_reset_timings(context);
                                    int code = whisper_full(context, params, audioData.data(), audioDataCount);

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
                                        errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, "Transcription failed"));
                                        rejectPtr->call(runtime, errorObj);
                                    }
                                } catch (const std::exception& e) {
                                    auto errorObj = Object(runtime);
                                    errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, e.what()));
                                    rejectPtr->call(runtime, errorObj);
                                }
                            }

                            return Value::undefined();
                        }
                    );

                    // Create and return the Promise
                    return promiseConstructor.callAsConstructor(runtime, promiseExecutor);
                } catch (const JSError& e) {
                    throw;
                } catch (const std::exception& e) {
                    __android_log_print(ANDROID_LOG_ERROR, "RNWhisperJSI",
                        "Exception in whisperTranscribeData: %s", e.what());
                    throw JSError(runtime, std::string("whisperTranscribeData error: ") + e.what());
                } catch (...) {
                    __android_log_print(ANDROID_LOG_ERROR, "RNWhisperJSI",
                        "Unknown exception in whisperTranscribeData");
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

                                        __android_log_print(ANDROID_LOG_INFO, "RNWhisperJSI",
                        "whisperVadDetectSpeech called with contextId=%d, arrayBuffer size=%zu", contextId, arrayBufferSize);

                    // Thread-safe VAD context lookup
                    long vadContextPtr = getVadContextPtr(contextId);
                    if (vadContextPtr == 0) {
                        throw JSError(runtime, "VAD Context not found for id: " + std::to_string(contextId));
                    }

                    struct whisper_vad_context *vadContext = reinterpret_cast<struct whisper_vad_context *>(vadContextPtr);
                    if (vadContext == nullptr) {
                        throw JSError(runtime, "Invalid VAD context pointer for id: " + std::to_string(contextId));
                    }

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

                                        // Create a promise for async VAD detection
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

                            // Run VAD detection on background thread
                            if (callInvoker) {
                                callInvoker->invokeAsync([vadContext, audioData, audioDataCount, resolvePtr, rejectPtr, &runtime]() {
                                    try {
                                        // Call whisper_vad_detect_speech
                                        bool isSpeech = whisper_vad_detect_speech(vadContext, audioData.data(), audioDataCount);

                                        __android_log_print(ANDROID_LOG_INFO, "RNWhisperJSI",
                                            "VAD detection result: %s", isSpeech ? "speech" : "no speech");

                                        // Get VAD parameters for segments detection
                                        struct whisper_vad_params vad_params = whisper_vad_default_params();

                                        // Get segments from VAD probabilities if speech detected
                                        struct whisper_vad_segments* segments = nullptr;
                                        if (isSpeech) {
                                            segments = whisper_vad_segments_from_probs(vadContext, vad_params);
                                        }

                                        // Create result object
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
                                    } catch (const JSError& e) {
                                        throw;
                                    } catch (const std::exception& e) {
                                        auto errorObj = Object(runtime);
                                        errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, e.what()));
                                        rejectPtr->call(runtime, errorObj);
                                    } catch (...) {
                                        auto errorObj = Object(runtime);
                                        errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, "Unknown error"));
                                        rejectPtr->call(runtime, errorObj);
                                    }
                                });
                            } else {
                                // Fallback: run synchronously
                                try {
                                    bool isSpeech = whisper_vad_detect_speech(vadContext, audioData.data(), audioDataCount);

                                    __android_log_print(ANDROID_LOG_INFO, "RNWhisperJSI",
                                        "VAD detection result: %s", isSpeech ? "speech" : "no speech");

                                    auto resultObj = Object(runtime);
                                    resultObj.setProperty(runtime, "hasSpeech", Value(isSpeech));
                                    resultObj.setProperty(runtime, "segments", Array(runtime, 0));

                                    resolvePtr->call(runtime, resultObj);
                                } catch (const std::exception& e) {
                                    auto errorObj = Object(runtime);
                                    errorObj.setProperty(runtime, "message", String::createFromUtf8(runtime, e.what()));
                                    rejectPtr->call(runtime, errorObj);
                                }
                            }

                            return Value::undefined();
                        }
                    );

                    // Create and return the Promise
                    return promiseConstructor.callAsConstructor(runtime, promiseExecutor);
                } catch (const JSError& e) {
                    throw;
                } catch (const std::exception& e) {
                    __android_log_print(ANDROID_LOG_ERROR, "RNWhisperJSI",
                        "Exception in whisperVadDetectSpeech: %s", e.what());
                    throw JSError(runtime, std::string("whisperVadDetectSpeech error: ") + e.what());
                } catch (...) {
                    __android_log_print(ANDROID_LOG_ERROR, "RNWhisperJSI",
                        "Unknown exception in whisperVadDetectSpeech");
                    throw JSError(runtime, "whisperVadDetectSpeech encountered unknown error");
                }
            }
        );

        // Install the function on the global object
        runtime.global().setProperty(runtime, "whisperTestContext", std::move(whisperTestContext));
        runtime.global().setProperty(runtime, "whisperTranscribeData", std::move(whisperTranscribeData));
        runtime.global().setProperty(runtime, "whisperVadDetectSpeech", std::move(whisperVadDetectSpeech));

        __android_log_print(ANDROID_LOG_INFO, "RNWhisperJSI", "JSI bindings installed successfully");
    } catch (const JSError& e) {
        __android_log_print(ANDROID_LOG_ERROR, "RNWhisperJSI",
            "JSError installing JSI bindings: %s", e.getMessage().c_str());
        throw;
    } catch (const std::exception& e) {
        __android_log_print(ANDROID_LOG_ERROR, "RNWhisperJSI",
            "Exception installing JSI bindings: %s", e.what());
        throw JSError(runtime, std::string("Failed to install JSI bindings: ") + e.what());
    } catch (...) {
        __android_log_print(ANDROID_LOG_ERROR, "RNWhisperJSI",
            "Unknown exception installing JSI bindings");
        throw JSError(runtime, "Failed to install JSI bindings: unknown error");
    }
}

} // namespace rnwhisper
