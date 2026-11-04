require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))
# Written by scripts/sync-vendor.sh; whisper.cpp passes these as compile
# definitions from its own CMake project (see cmake/rnwhisper-sources.cmake).
# ggml's GGML_VERSION / GGML_COMMIT come from the generated ggml-version.h.
version_info = JSON.parse(File.read(File.join(__dir__, "src", "version.json")))
version_flags = {
  "WHISPER_VERSION" => version_info["version"],
  "PARAKEET_VERSION" => version_info["version"],
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
  vad_cpp = "vendor/vad.cpp"
  ggml_metal = "#{whisper_cpp}/ggml/src/ggml-metal"

  if ENV["RNWHISPER_BUILD_FROM_SOURCE"] == "1"
    # ios/*: not ios/**, or the prebuilt xcframework's flat Headers/ would join
    # the header map and shadow the vendored headers with the same basenames.
    s.source_files = "ios/*.{h,m,mm}", "cpp/**/*.{h,cpp,hpp,c}",
      "#{whisper_cpp}/include/*.h",
      "#{whisper_cpp}/src/*.{h,cpp}",
      "#{whisper_cpp}/src/coreml/*.{h,m,mm}",
      "#{whisper_cpp}/src/aneforge/*.{h,cpp}",
      "#{whisper_cpp}/ggml/include/*.h",
      "#{whisper_cpp}/ggml/src/*.{h,c,cpp}",
      "#{whisper_cpp}/ggml/src/ggml-cpu/**/*.{h,c,cpp}",
      "#{ggml_metal}/*.{h,cpp}",
      "#{vad_cpp}/include/*.h",
      # vad.cpp: CocoaPods has no per-file flags, so unlike
      # rnwhisper_vad_configure_sources this skips -ffp-contract=off on
      # models/frame_dft.cpp (FireRed/smart-turn stream vs offline rounding
      # only; Silero does not use it) and -fno-math-errno on the kernels
      # (Apple clang's default already).
      "#{vad_cpp}/src/**/*.{h,cpp}"
    # kernels_impl.inc is #included by the kernel variants, not compiled itself.
    s.preserve_paths = "#{vad_cpp}/src/ggml-vad-cpu/*.inc"
    # Exclude what no whisper.rn build compiles (same set as the filters in
    # cmake/rnwhisper-sources.cmake): the parts of vendor/whisper.cpp that are
    # only there for upstream's CMake project (see vendor/README.md).
    s.exclude_files = "#{whisper_cpp}/ggml/src/ggml-cpu/hbm.cpp",
      "#{whisper_cpp}/ggml/src/ggml-cpu/{kleidiai,llamafile,arch/wasm}/*",
      # vad.cpp's x86 kernel variants (never an Apple device target); the
      # ARMv8.2 one is built by the vad-armv82 subspec below.
      "#{vad_cpp}/src/ggml-vad-cpu/kernels_{avx2,avx512,avx512vnni,armv82}.cpp"
    # Metal kernels are compiled at runtime from these sources.
    s.resources = [
      "#{ggml_metal}/kernels",
      "#{ggml_metal}/ggml-metal-impl.h",
      "#{whisper_cpp}/ggml/src/ggml-common.h"
    ]
    # VADK_ARM_VARIANTS: dispatch to the ARMv8.2 kernels when the sysctls report
    # fp16 + dotprod (false on x86_64 simulators, so that slice stays baseline)
    base_compiler_flags += " -DRNWHISPER_BUILD_FROM_SOURCE -DVAD_WITH_CPU_KERNELS -DVADK_ARM_VARIANTS"
    # Same list as cmake/rnwhisper-sources.cmake.
    [
      "cpp",
      "#{whisper_cpp}/include",
      "#{whisper_cpp}/ggml/include",
      "#{whisper_cpp}/ggml/src",
      "#{whisper_cpp}/ggml/src/ggml-cpu",
      "#{whisper_cpp}/src",
      "#{vad_cpp}/include",
      "#{vad_cpp}/src",
      "#{vad_cpp}/src/ggml-vad-cpu",
    ].each { |dir| header_search_paths << "\"$(PODS_TARGET_SRCROOT)/#{dir}\"" }

    # vad.cpp's ARMv8.2 (fp16 + dotprod) kernels, picked at runtime. A subspec
    # is the only way to give one file its own flags; -Xarch_arm64 keeps
    # -march off x86_64 simulator compiles.
    s.subspec "vad-armv82" do |ss|
      ss.source_files = "#{vad_cpp}/src/ggml-vad-cpu/kernels_armv82.cpp"
      ss.compiler_flags = "-Xarch_arm64 -march=armv8.2-a+fp16+dotprod"
    end

    # ggml-metal's Objective-C sources do not support ARC.
    s.subspec "no-require-arc" do |ss|
      ss.requires_arc = false
      ss.source_files = "#{ggml_metal}/*.m"
    end
  else
    # The prebuilt xcframework is a GitHub release asset that the package's
    # postinstall downloads (install/download-native-artifacts.js).
    xcframework = File.join(__dir__, "ios", "rnwhisper.xcframework")
    unless Dir.exist?(File.join(xcframework, "ios-arm64"))
      raise "[whisper.rn] ios/rnwhisper.xcframework is missing. Run `npx whisper-rn-download-artifacts` " \
            "(the package's postinstall) to download it, or set RNWHISPER_BUILD_FROM_SOURCE=1 to build from source."
    end
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
