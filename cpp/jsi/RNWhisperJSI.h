#pragma once

#include <jsi/jsi.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <ReactCommon/CallInvoker.h>

#if defined(__ANDROID__)
#include <android/log.h>
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

// Context management functions
void addContext(int contextId, long contextPtr);
void removeContext(int contextId);
void addVadContext(int contextId, long vadContextPtr);
void removeVadContext(int contextId);

// Main JSI installation function
void installJSIBindings(
    facebook::jsi::Runtime& runtime,
    std::shared_ptr<facebook::react::CallInvoker> callInvoker
);

// Cleanup function to dispose of ThreadPool
void cleanupJSIBindings();

} // namespace rnwhisper_jsi
