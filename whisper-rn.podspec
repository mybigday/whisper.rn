require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))
# Written by scripts/sync-vendor.sh; whisper.cpp passes these as compile
# definitions from its own CMake project (see cmake/rnwhisper-sources.cmake).
version_info = JSON.parse(File.read(File.join(__dir__, "src", "version.json")))
version_flags = {
  "WHISPER_VERSION" => version_info["version"],
  "PARAKEET_VERSION" => version_info["version"],
  "GGML_VERSION" => version_info["ggmlVersion"],
  "GGML_COMMIT" => version_info["commit"],
}.map { |name, value| "-D#{name}=\\\"#{value}\\\"" }.join(" ")

base_ld_flags = "-framework Accelerate -framework Foundation -framework Metal -framework MetalKit"
base_compiler_flags = "-DGGML_USE_CPU -DGGML_USE_ACCELERATE -pthread -Wno-shorten-64-to-32 #{version_flags}"
folly_compiler_flags = "-DFOLLY_NO_CONFIG -DFOLLY_MOBILE=1 -DFOLLY_USE_LIBCPP=1 -Wno-comma"

# Use base_optimizer_flags = "" for debug builds
# base_optimizer_flags = ""
base_optimizer_flags = "-O3 -DNDEBUG" +
 " -fvisibility=hidden -fvisibility-inlines-hidden" +
 " -ffunction-sections -fdata-sections"

if ENV['RNWHISPER_DISABLE_COREML'] != '1' then
  base_ld_flags += " -framework CoreML"
  base_compiler_flags += " -DWHISPER_USE_COREML -DWHISPER_COREML_ALLOW_FALLBACK"
end

if ENV["RNWHISPER_DISABLE_METAL"] != "1" then
  # GGMLMetalClass is an Objective-C class (process-global name); rename it per
  # library so whisper-rn and llama-rn can both be built from source in one app.
  base_compiler_flags += " -DGGML_USE_METAL -DGGMLMetalClass=RNWhisperGGMLMetalClass" # -DGGML_METAL_NDEBUG
end

Pod::Spec.new do |s|
  s.name         = "whisper-rn"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]

  s.platforms    = { :ios => "13.0", :tvos => "13.0" }
  s.source       = { :git => "https://github.com/mybigday/whisper.rn.git", :tag => "#{s.version}" }

  s.requires_arc = true

  header_search_paths = ['$(inherited)']

  whisper_cpp = "vendor/whisper.cpp"
  ggml_metal = "#{whisper_cpp}/ggml/src/ggml-metal"

  if ENV["RNWHISPER_BUILD_FROM_SOURCE"] == "1"
    # ios/*: not ios/**, or the prebuilt xcframework's flat Headers/ would join
    # the header map and shadow the vendored headers with the same basenames.
    s.source_files = "ios/*.{h,m,mm}", "cpp/**/*.{h,cpp,hpp,c}",
      "#{whisper_cpp}/include/*.h",
      "#{whisper_cpp}/src/*.{h,cpp}",
      "#{whisper_cpp}/src/coreml/*.{h,m,mm}",
      "#{whisper_cpp}/ggml/include/*.h",
      "#{whisper_cpp}/ggml/src/*.{h,c,cpp}",
      "#{whisper_cpp}/ggml/src/ggml-cpu/**/*.{h,c,cpp}",
      "#{ggml_metal}/*.{h,cpp}"
    # Metal kernels are compiled at runtime from these sources.
    s.resources = [
      "#{ggml_metal}/kernels",
      "#{ggml_metal}/ggml-metal-impl.h",
      "#{whisper_cpp}/ggml/src/ggml-common.h"
    ]
    base_compiler_flags += " -DRNWHISPER_BUILD_FROM_SOURCE"
    # Same list as cmake/rnwhisper-sources.cmake.
    [
      "cpp",
      "#{whisper_cpp}/include",
      "#{whisper_cpp}/ggml/include",
      "#{whisper_cpp}/ggml/src",
      "#{whisper_cpp}/ggml/src/ggml-cpu",
      "#{whisper_cpp}/src",
    ].each { |dir| header_search_paths << "\"$(PODS_TARGET_SRCROOT)/#{dir}\"" }

    # ggml-metal's Objective-C sources do not support ARC.
    s.subspec "no-require-arc" do |ss|
      ss.requires_arc = false
      ss.source_files = "#{ggml_metal}/*.m"
    end
  else
    # JSI bindings always compiled from source (must match RN version)
    s.source_files = "ios/*.{h,m,mm}", "cpp/jsi/*.{h,cpp}"
    s.vendored_frameworks = "ios/rnwhisper.xcframework"
  end

  s.compiler_flags = base_compiler_flags
  s.pod_target_xcconfig = {
    "OTHER_LDFLAGS" => base_ld_flags,
    "OTHER_CFLAGS" => base_optimizer_flags,
    "OTHER_CPLUSPLUSFLAGS" => base_optimizer_flags + " -std=c++20",
    "HEADER_SEARCH_PATHS" => header_search_paths.join(" ")
  }

  s.dependency "React-callinvoker"
  s.dependency "React"

  install_modules_dependencies(s)
end
