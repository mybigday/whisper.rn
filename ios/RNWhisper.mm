#import "RNWhisper.h"
#import "RNWhisperJSI.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <TargetConditionals.h>

#include <cstring>
#include <mutex>

#ifdef RCT_NEW_ARCH_ENABLED
#import <RNWhisperSpec/RNWhisperSpec.h>
#endif

namespace {

NSString *cacheDirectoryPath() {
    return [NSTemporaryDirectory() stringByAppendingPathComponent:@"rnwhisper_debug_assets"];
}

NSString *ensureDirectoryForFile(NSString *filePath) {
    NSString *directory = [filePath stringByDeletingLastPathComponent];
    if (![[NSFileManager defaultManager] fileExistsAtPath:directory]) {
        [[NSFileManager defaultManager]
            createDirectoryAtPath:directory
      withIntermediateDirectories:YES
                       attributes:nil
                            error:nil];
    }
    return filePath;
}

std::mutex &cacheMutex() {
    static std::mutex mutex;
    return mutex;
}

NSString *sourceMarkerPath(NSString *baseDirectory, NSString *relativePath) {
    NSString *markerPath = [[baseDirectory stringByAppendingPathComponent:@".sources"]
        stringByAppendingPathComponent:relativePath];
    return [markerPath stringByAppendingString:@".source"];
}

bool isRemoteUrl(const std::string &path) {
    return path.rfind("http://", 0) == 0 || path.rfind("https://", 0) == 0;
}

std::string toStdString(NSString *value) {
    return value ? std::string([value UTF8String]) : std::string();
}

NSString *toNSString(const std::string &value) {
    return [NSString stringWithUTF8String:value.c_str()];
}

std::string downloadToCache(const std::string &url, const std::string &relativePath) {
    std::lock_guard<std::mutex> lock(cacheMutex());

    NSString *urlString = toNSString(url);
    NSURL *nsUrl = [NSURL URLWithString:urlString];
    NSString *baseDirectory = cacheDirectoryPath();
    NSString *targetRelativePath = nil;

    if (!relativePath.empty()) {
        targetRelativePath = toNSString(relativePath);
    } else {
        targetRelativePath = nsUrl.lastPathComponent ?: @"download.bin";
    }

    NSString *targetPath = [baseDirectory stringByAppendingPathComponent:targetRelativePath];
    NSString *markerPath = sourceMarkerPath(baseDirectory, targetRelativePath);
    ensureDirectoryForFile(targetPath);
    ensureDirectoryForFile(markerPath);

    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSData *source = [urlString dataUsingEncoding:NSUTF8StringEncoding];
    NSData *cachedSource = [NSData dataWithContentsOfFile:markerPath];
    if ([fileManager fileExistsAtPath:targetPath] &&
        cachedSource &&
        [cachedSource isEqualToData:source]) {
        return toStdString(targetPath);
    }

    // Metro asset URLs include a content hash. Record the source URL in cache
    // metadata so an updated asset cannot reuse stale bytes with the same name.
    [fileManager removeItemAtPath:markerPath error:nil];
    [fileManager removeItemAtPath:targetPath error:nil];

    NSData *data = [NSData dataWithContentsOfURL:nsUrl];
    if (!data || ![data writeToFile:targetPath options:NSDataWritingAtomic error:nil]) {
        return std::string();
    }

    [source writeToFile:markerPath options:NSDataWritingAtomic error:nil];
    return toStdString(targetPath);
}

} // namespace

namespace rnwhisper_jsi {

std::string resolveIosAssetPath(const std::string &path, bool isBundleAsset) {
    if (!isBundleAsset) {
        return path;
    }

    NSString *resourcePath = [[NSBundle mainBundle] pathForResource:toNSString(path) ofType:nil];
    return resourcePath ? toStdString(resourcePath) : path;
}

std::string downloadIosFile(const std::string &url, const std::string &relativePath) {
    return downloadToCache(url, relativePath);
}

MetalAvailability getMetalAvailability(bool requestedGpu) {
    MetalAvailability availability;
    if (!requestedGpu) {
        availability.available = false;
        availability.reason = "GPU disabled by user";
        return availability;
    }

#if defined(WSP_GGML_USE_METAL)
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    bool supportsMetal = false;

    if (device) {
        supportsMetal = [device supportsFamily:MTLGPUFamilyApple7];
        if (@available(iOS 16.0, tvOS 16.0, *)) {
            supportsMetal = supportsMetal && [device supportsFamily:MTLGPUFamilyMetal3];
        }
    }

#if TARGET_OS_SIMULATOR
    supportsMetal = false;
#endif

    availability.available = supportsMetal;
    if (!supportsMetal) {
#if TARGET_OS_SIMULATOR
        availability.reason = "Metal is not supported in simulator";
#else
        availability.reason = "Metal is not supported in this device";
#endif
    }
#else
    availability.available = false;
    availability.reason = "Metal is not enabled in this build";
#endif

    return availability;
}

WhisperContextInitResult hostInitWhisperContext(
    const WhisperContextInitOptions &options) {
    WhisperContextInitResult result;

    if (options.downloadCoreMLAssets) {
        for (const auto &asset : options.coreMLAssets) {
            if (isRemoteUrl(asset.uri)) {
                downloadToCache(asset.uri, asset.filepath);
            }
        }
    }

    std::string modelPath = options.filePath;
    if (options.isBundleAsset) {
        modelPath = resolveIosAssetPath(modelPath, true);
    } else if (isRemoteUrl(modelPath)) {
        modelPath = downloadToCache(modelPath, "");
    }

    if (modelPath.empty()) {
        return result;
    }

    auto params = whisper_context_default_params();
    params.use_gpu = options.useGpu;
    params.flash_attn = options.useFlashAttn;
    params.dtw_token_timestamps = false;
    params.use_coreml = options.useCoreMLIos;

#if !defined(WHISPER_USE_COREML)
    if (params.use_coreml) {
        params.use_coreml = false;
    }
#endif

    NSString *reasonNoGPU = @"";
    auto metalAvailability = getMetalAvailability(params.use_gpu);
    if (!metalAvailability.available) {
        params.use_gpu = false;
        reasonNoGPU = toNSString(metalAvailability.reason);
    } else {
        params.gpu_device = 0;
    }

    if (params.use_gpu && params.use_coreml) {
        params.use_coreml = false;
    }

    result.context =
        whisper_init_from_file_with_params(modelPath.c_str(), params);
    result.gpu = params.use_gpu;
    result.reasonNoGPU = toStdString(reasonNoGPU);
    return result;
}

WhisperVadContextInitResult hostInitWhisperVadContext(
    const WhisperVadContextInitOptions &options) {
    WhisperVadContextInitResult result;

    std::string modelPath = options.filePath;
    if (options.isBundleAsset) {
        modelPath = resolveIosAssetPath(modelPath, true);
    } else if (isRemoteUrl(modelPath)) {
        modelPath = downloadToCache(modelPath, "");
    }

    if (modelPath.empty()) {
        return result;
    }

    auto params = whisper_vad_default_context_params();
    if (options.nThreads > 0) {
        params.n_threads = options.nThreads;
    }
    params.use_gpu = false;

    result.context =
        whisper_vad_init_from_file_with_params(modelPath.c_str(), params);
    result.gpu = false;
    if (options.useGpu) {
        result.reasonNoGPU = "GPU VAD is not supported";
    }
    return result;
}

ParakeetContextInitResult hostInitParakeetContext(
    const ParakeetContextInitOptions &options) {
    ParakeetContextInitResult result;

    std::string modelPath = options.filePath;
    if (options.isBundleAsset) {
        modelPath = resolveIosAssetPath(modelPath, true);
    } else if (isRemoteUrl(modelPath)) {
        modelPath = downloadToCache(modelPath, "");
    }

    if (modelPath.empty()) {
        return result;
    }

    auto params = parakeet_context_default_params();
    params.use_gpu = options.useGpu;

    auto metalAvailability = getMetalAvailability(params.use_gpu);
    if (!metalAvailability.available) {
        params.use_gpu = false;
        result.reasonNoGPU = metalAvailability.reason;
    } else {
        params.gpu_device = 0;
    }

    result.context =
        parakeet_init_from_file_with_params(modelPath.c_str(), params);
    result.gpu = params.use_gpu;
    return result;
}

std::vector<uint8_t> hostLoadFileBytes(const std::string &path) {
    std::string resolvedPath = path;
    if (isRemoteUrl(resolvedPath)) {
        resolvedPath = downloadToCache(resolvedPath, "");
    }
    if (resolvedPath.empty()) {
        return {};
    }

    NSData *data = [NSData dataWithContentsOfFile:toNSString(resolvedPath)];
    if (!data) {
        return {};
    }

    std::vector<uint8_t> bytes([data length]);
    if (!bytes.empty()) {
        std::memcpy(bytes.data(), [data bytes], [data length]);
    }
    return bytes;
}

void hostClearCache() {
    std::lock_guard<std::mutex> lock(cacheMutex());
    [[NSFileManager defaultManager] removeItemAtPath:cacheDirectoryPath() error:nil];
}

} // namespace rnwhisper_jsi

@protocol RNWhisperBridgeRuntimeAccess <NSObject>
@property (nonatomic, readonly) void *runtime;
@property (nonatomic, readonly) std::shared_ptr<facebook::react::CallInvoker> jsCallInvoker;
@end

@implementation RNWhisper {
    __unsafe_unretained RCTBridge *_bridge;
}

RCT_EXPORT_MODULE()

+ (BOOL)requiresMainQueueSetup
{
  return NO;
}

- (RCTBridge *)bridge {
    return _bridge;
}

- (void)setBridge:(RCTBridge *)bridge {
    _bridge = bridge;
}

- (NSDictionary *)constantsToExport
{
  return @{
#if WHISPER_USE_COREML
    @"useCoreML": @YES,
#else
    @"useCoreML": @NO,
#endif
#if WHISPER_COREML_ALLOW_FALLBACK
    @"coreMLAllowFallback": @YES,
#else
    @"coreMLAllowFallback": @NO,
#endif
  };
}

RCT_EXPORT_METHOD(install:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RCTBridge *bridge = self.bridge ?: [RCTBridge currentBridge];
    if (!bridge) {
        resolve(@false);
        return;
    }

    id<RNWhisperBridgeRuntimeAccess> cxxBridge = (id<RNWhisperBridgeRuntimeAccess>)bridge.batchedBridge;
    if (!cxxBridge) {
        cxxBridge = (id<RNWhisperBridgeRuntimeAccess>)bridge;
    }

    auto callInvoker = cxxBridge.jsCallInvoker;
    if (!cxxBridge.runtime) {
        resolve(@false);
        return;
    }

    auto *runtime = static_cast<facebook::jsi::Runtime *>(cxxBridge.runtime);
    RCTPromiseResolveBlock resolveBlock = [resolve copy];

    if (callInvoker) {
        try {
            callInvoker->invokeAsync([runtime, callInvoker, resolveBlock]() {
                rnwhisper_jsi::installJSIBindings(*runtime, callInvoker);
                dispatch_async(dispatch_get_main_queue(), ^{
                    resolveBlock(@true);
                });
            });
        } catch (...) {
            resolveBlock(@false);
        }
    } else {
        resolveBlock(@false);
        return;
    }
}

- (void)invalidate {
    rnwhisper_jsi::cleanupJSIBindings();
}

#ifdef RCT_NEW_ARCH_ENABLED
- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params
{
    return std::make_shared<facebook::react::NativeRNWhisperSpecJSI>(params);
}
#endif

@end
