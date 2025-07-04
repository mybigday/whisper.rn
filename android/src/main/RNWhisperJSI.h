#pragma once

#include <jni.h>
#include <jsi/jsi.h>
#include <ReactCommon/CallInvoker.h>

namespace rnwhisper {
    void addContext(int contextId, long contextPtr);
    void removeContext(int contextId);
    void addVadContext(int contextId, long vadContextPtr);
    void removeVadContext(int contextId);
    void installJSIBindings(
        facebook::jsi::Runtime& runtime,
        std::shared_ptr<facebook::react::CallInvoker> callInvoker
    );
}
