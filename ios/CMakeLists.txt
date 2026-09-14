cmake_minimum_required(VERSION 3.16)
project(rnwhisper VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# iOS specific settings
set(CMAKE_OSX_DEPLOYMENT_TARGET 15.0)
set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE NO)

# Dependencies and compile options
add_definitions(
    -DNDEBUG
    -DO3
    -DGGML_USE_CPU
    -DGGML_USE_ACCELERATE
    -DGGML_USE_METAL
    -DGGML_METAL_USE_BF16
    # ggml-metal's bundle-lookup Objective-C class is process-global; rename it
    # per library so this framework can coexist with other ggml-based libraries
    # (e.g. llama.rn) in the same app.
    -DGGMLMetalClass=RNWhisperGGMLMetalClass
    -DWHISPER_USE_COREML
    -DWHISPER_COREML_ALLOW_FALLBACK
)

if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64;x86_64")
    add_definitions(-DGGML_CPU_GENERIC)
endif ()

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/rnwhisper-sources.cmake)

if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
    set(SOURCE_FILES_ARCH
        ${RNWHISPER_GGML_CPU_ARCH_DIR}/arm/quants.c
        ${RNWHISPER_GGML_CPU_ARCH_DIR}/arm/repack.cpp
    )
endif ()

# Define public headers
set(PUBLIC_HEADERS
    ${RNWHISPER_CPP_DIR}/rn-whisper.h
    ${RNWHISPER_WHISPER_CPP_DIR}/include/whisper.h
    ${RNWHISPER_WHISPER_CPP_DIR}/ggml/include/ggml.h
)

# Create library target
add_library(rnwhisper SHARED
    ${RNWHISPER_CORE_SOURCES}
    ${RNWHISPER_GGML_METAL_SOURCES}
    ${RNWHISPER_COREML_SOURCES}
    ${SOURCE_FILES_ARCH}
)

target_compile_definitions(rnwhisper PRIVATE ${RNWHISPER_VERSION_DEFINITIONS})

# Setup include directories
target_include_directories(rnwhisper
    PUBLIC
        "$<BUILD_INTERFACE:${RNWHISPER_INCLUDE_DIRS}>"
        $<INSTALL_INTERFACE:include>
)

# Keep ggml/gguf out of the framework's export table so a second ggml copy in
# the app (e.g. llama.rn) can never be resolved through this image. Hidden
# inline visibility also stops dyld from coalescing header-inline functions
# across the two frameworks.
target_compile_options(rnwhisper PRIVATE $<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-fvisibility-inlines-hidden>)
target_link_options(rnwhisper PRIVATE
    "LINKER:-unexported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/unexported-symbols.txt")

# Link required frameworks
target_link_libraries(rnwhisper PRIVATE
    "-framework Accelerate"
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalKit"
    "-framework CoreML"
)

# Set properties for framework
set_target_properties(rnwhisper PROPERTIES
    MACOSX_FRAMEWORK_IDENTIFIER "com.rnwhisper"
    MACOSX_FRAMEWORK_BUNDLE_VERSION 1.0.0
    MACOSX_FRAMEWORK_SHORT_VERSION_STRING 1.0.0
    FRAMEWORK TRUE
    FRAMEWORK_VERSION 1.0.0
    VERSION 1.0.0
    PUBLIC_HEADER "${PUBLIC_HEADERS}"
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
    # Generate dSYMs so packaged xcframework slices can be symbolicated in App Store crash logs.
    XCODE_ATTRIBUTE_DEBUG_INFORMATION_FORMAT "dwarf-with-dsym"
    XCODE_ATTRIBUTE_GCC_GENERATE_DEBUGGING_SYMBOLS YES
)

# Disable ARC for Metal Objective-C files (they don't support ARC)
set_source_files_properties(
    ${RNWHISPER_GGML_METAL_OBJC_SOURCES}
    PROPERTIES COMPILE_FLAGS "-fno-objc-arc"
)

# Xcode applies CLANG_ENABLE_OBJC_ARC at the target level. Other generators
# need ARC enabled explicitly for the Core ML Objective-C sources.
set_source_files_properties(
    ${RNWHISPER_COREML_SOURCES}
    PROPERTIES COMPILE_FLAGS "-fobjc-arc"
)
