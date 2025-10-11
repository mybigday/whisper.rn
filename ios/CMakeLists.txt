cmake_minimum_required(VERSION 3.16)
project(rnwhisper VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# iOS specific settings
set(CMAKE_OSX_DEPLOYMENT_TARGET 13.0)
set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE NO)

# Dependencies and compile options
add_definitions(
    -DNDEBUG
    -DO3
    -DWSP_GGML_USE_CPU
    -DWSP_GGML_USE_ACCELERATE
    -DWSP_GGML_USE_METAL
    -DWSP_GGML_METAL_USE_BF16
)

if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64;x86_64")
    add_definitions(-DWSP_GGML_CPU_GENERIC)
endif ()

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cpp)

if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
    set(SOURCE_FILES_ARCH
        ${SOURCE_DIR}/ggml-cpu/arch/arm/quants.c
        ${SOURCE_DIR}/ggml-cpu/arch/arm/repack.cpp
    )
endif ()

# Define public headers
set(PUBLIC_HEADERS
    ${SOURCE_DIR}/rn-whisper.h
    ${SOURCE_DIR}/whisper.h
    ${SOURCE_DIR}/ggml.h
)

# Create library target
add_library(rnwhisper SHARED
    ${SOURCE_DIR}/ggml.c
    ${SOURCE_DIR}/ggml-alloc.c
    ${SOURCE_DIR}/ggml-backend.cpp
    ${SOURCE_DIR}/ggml-backend-reg.cpp
    ${SOURCE_DIR}/ggml-cpu/amx/amx.cpp
    ${SOURCE_DIR}/ggml-cpu/amx/mmq.cpp
    ${SOURCE_DIR}/ggml-cpu/ggml-cpu.c
    ${SOURCE_DIR}/ggml-cpu/ggml-cpu.cpp
    ${SOURCE_DIR}/ggml-cpu/quants.c
    ${SOURCE_DIR}/ggml-cpu/traits.cpp
    ${SOURCE_DIR}/ggml-cpu/repack.cpp
    ${SOURCE_DIR}/ggml-cpu/unary-ops.cpp
    ${SOURCE_DIR}/ggml-cpu/binary-ops.cpp
    ${SOURCE_DIR}/ggml-cpu/vec.cpp
    ${SOURCE_DIR}/ggml-cpu/ops.cpp
    ${SOURCE_DIR}/ggml-metal/ggml-metal.cpp
    ${SOURCE_DIR}/ggml-metal/ggml-metal-common.cpp
    ${SOURCE_DIR}/ggml-metal/ggml-metal-device.cpp
    ${SOURCE_DIR}/ggml-metal/ggml-metal-context.m
    ${SOURCE_DIR}/ggml-metal/ggml-metal-device.m
    ${SOURCE_DIR}/ggml-metal/ggml-metal-ops.cpp
    ${SOURCE_DIR}/ggml-opt.cpp
    ${SOURCE_DIR}/ggml-threading.cpp
    ${SOURCE_DIR}/ggml-quants.c
    ${SOURCE_DIR}/gguf.cpp
    ${SOURCE_DIR}/whisper.cpp
    ${SOURCE_DIR}/rn-whisper.cpp
    ${SOURCE_DIR}/rn-audioutils.cpp
    ${SOURCE_FILES_ARCH}
)

# Setup include directories
target_include_directories(rnwhisper
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../cpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../cpp/ggml-cpu>
        $<INSTALL_INTERFACE:include>
)

# Link required frameworks
target_link_libraries(rnwhisper PRIVATE
    "-framework Accelerate"
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalKit"
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
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC NO
)
