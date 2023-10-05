require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))
base_ld_flags = "-framework Accelerate -framework Foundation -framework Metal -framework MetalKit"
base_compiler_flags = "-DWSP_GGML_USE_ACCELERATE -Wno-shorten-64-to-32"
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

# TODO: Enable Metal
# if ENV["RNWHISPER_DISABLE_METAL"] != "1" then
#   base_compiler_flags += " -DWSP_GGML_USE_METAL" # -DWSP_GGML_METAL_NDEBUG
# end

Pod::Spec.new do |s|
  s.name         = "whisper-rn"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]

  s.platforms    = { :ios => "11.0", :tvos => "11.0" }
  s.source       = { :git => "https://github.com/mybigday/whisper.rn.git", :tag => "#{s.version}" }

  s.source_files = "ios/**/*.{h,m,mm}", "cpp/**/*.{h,cpp,c,m,mm}"
  s.resources = "cpp/**/*.{metal}"

  s.dependency "React-Core"

  s.compiler_flags = base_compiler_flags
  s.pod_target_xcconfig = {
    "OTHER_LDFLAGS" => base_ld_flags,
    "OTHER_CFLAGS" => base_optimizer_flags,
    "OTHER_CPLUSPLUSFLAGS" => base_optimizer_flags
  }

  # Don't install the dependencies when we run `pod install` in the old architecture.
  if ENV['RCT_NEW_ARCH_ENABLED'] == '1' then
    s.compiler_flags = base_compiler_flags + " " + folly_compiler_flags + " -DRCT_NEW_ARCH_ENABLED=1"
    new_arch_cpp_flags = "-DFOLLY_NO_CONFIG -DFOLLY_MOBILE=1 -DFOLLY_USE_LIBCPP=1"
    s.pod_target_xcconfig = {
      "CLANG_CXX_LANGUAGE_STANDARD" => "c++17",
      "HEADER_SEARCH_PATHS" => "\"$(PODS_ROOT)/boost\"",
      "OTHER_LDFLAGS" => base_ld_flags,
      "OTHER_CFLAGS" => base_optimizer_flags,
      "OTHER_CPLUSPLUSFLAGS" => new_arch_cpp_flags + " " + base_optimizer_flags
    }
    s.dependency "React-Codegen"
    s.dependency "RCT-Folly"
    s.dependency "RCTRequired"
    s.dependency "RCTTypeSafety"
    s.dependency "ReactCommon/turbomodule/core"
  end
end
