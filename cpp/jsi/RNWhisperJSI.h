#pragma once

#include <ReactCommon/CallInvoker.h>
#include <jsi/jsi.h>
#include <string>
#include <vector>

#if defined(__ANDROID__)
#include <jni.h>
#include "whisper.h"
#include "rn-whisper.h"
#endif

#if defined(__APPLE__)
#if RNWHISPER_BUILD_FROM_SOURCE
#include "whisper.h"
#include "rn-whisper.h"
#else
#include <rnwhisper/whisper.h>
#include <rnwhisper/rn-whisper.h>
#endif
#endif

namespace rnwhisper_jsi {

struct CoreMLAssetInfo {
    std::string uri;
    std::string filepath;
};

struct WhisperContextInitOptions {
    std::string filePath;
    bool isBundleAsset = false;
    bool useFlashAttn = false;
    bool useGpu = true;
    bool useCoreMLIos = true;
    bool downloadCoreMLAssets = false;
    std::vector<CoreMLAssetInfo> coreMLAssets;
};

struct WhisperContextInitResult {
    whisper_context *context = nullptr;
    bool gpu = false;
    std::string reasonNoGPU;
};

struct WhisperVadContextInitOptions {
    std::string filePath;
    bool isBundleAsset = false;
    bool useGpu = true;
    int nThreads = 0;
};

struct WhisperVadContextInitResult {
    whisper_vad_context *context = nullptr;
    bool gpu = false;
    std::string reasonNoGPU;
};

WhisperContextInitResult hostInitWhisperContext(
    const WhisperContextInitOptions &options);
WhisperVadContextInitResult hostInitWhisperVadContext(
    const WhisperVadContextInitOptions &options);
std::vector<uint8_t> hostLoadFileBytes(const std::string &path);
void hostClearCache();

#if defined(__ANDROID__)
void setAndroidContext(JNIEnv *env, jobject applicationContext, jobject assetManager);
#elif defined(__APPLE__)
struct MetalAvailability {
    bool available = false;
    std::string reason;
};

std::string resolveIosAssetPath(const std::string &path, bool isBundleAsset);
std::string downloadIosFile(const std::string &url, const std::string &relativePath);
MetalAvailability getMetalAvailability(bool requestedGpu);
#endif

void installJSIBindings(
    facebook::jsi::Runtime &runtime,
    std::shared_ptr<facebook::react::CallInvoker> callInvoker);
void cleanupJSIBindings();

} // namespace rnwhisper_jsi
