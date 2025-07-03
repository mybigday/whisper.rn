#include "RNWhisperJSI.h"
#include <jsi/jsi.h>
#include <memory>
#include <mutex>
#include <android/log.h>
#include "whisper.h"

using namespace facebook::jsi;

namespace rnwhisper {

// Contexts map (contextId -> contextPtr) with thread safety
static std::unordered_map<int, long> contextMap;
static std::mutex contextMapMutex;

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

// Helper function to safely get context pointer
static long getContextPtr(int contextId) {
    std::lock_guard<std::mutex> lock(contextMapMutex);
    auto it = contextMap.find(contextId);
    return (it != contextMap.end()) ? it->second : 0;
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
            1, // number of arguments
            [](Runtime& runtime, const Value& thisValue, const Value* arguments, size_t count) -> Value {
                try {
                    if (count != 1) {
                        throw JSError(runtime, "whisperTestContext expects 1 argument (contextId)");
                    }

                    if (!arguments[0].isNumber()) {
                        throw JSError(runtime, "whisperTestContext expects contextId to be a number");
                    }

                    int contextId = (int)arguments[0].getNumber();

                    __android_log_print(ANDROID_LOG_INFO, "RNWhisperJSI",
                        "whisperTestContext called with contextId=%d", contextId);

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
                        "Context validated successfully for id=%d", contextId);
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

        // Install the function on the global object
        runtime.global().setProperty(runtime, "whisperTestContext", std::move(whisperTestContext));

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
