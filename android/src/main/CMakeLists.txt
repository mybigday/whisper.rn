cmake_minimum_required(VERSION 3.10)

project(whisper.rn)

set(CMAKE_CXX_STANDARD 17)
set(RNWHISPER_LIB_DIR ${CMAKE_SOURCE_DIR}/../../../cpp)

# Configure STL to be compatible with ReactAndroid JSI libraries
set(CMAKE_ANDROID_STL_TYPE c++_shared)

# Find ReactAndroid package for JSI
find_package(ReactAndroid REQUIRED CONFIG)
find_package(fbjni REQUIRED CONFIG)

include_directories(
    ${RNWHISPER_LIB_DIR}
    ${RNWHISPER_LIB_DIR}/ggml-cpu
    ${RNWHISPER_LIB_DIR}/jsi
)

set(
    SOURCE_FILES
    ${RNWHISPER_LIB_DIR}/ggml.c
    ${RNWHISPER_LIB_DIR}/ggml-alloc.c
    ${RNWHISPER_LIB_DIR}/ggml-backend.cpp
    ${RNWHISPER_LIB_DIR}/ggml-backend-reg.cpp
    ${RNWHISPER_LIB_DIR}/ggml-cpu/amx/amx.cpp
    ${RNWHISPER_LIB_DIR}/ggml-cpu/amx/mmq.cpp
    ${RNWHISPER_LIB_DIR}/ggml-cpu/ggml-cpu.c
    ${RNWHISPER_LIB_DIR}/ggml-cpu/ggml-cpu.cpp
    ${RNWHISPER_LIB_DIR}/ggml-cpu/quants.c
    ${RNWHISPER_LIB_DIR}/ggml-cpu/traits.cpp
    ${RNWHISPER_LIB_DIR}/ggml-cpu/repack.cpp
    ${RNWHISPER_LIB_DIR}/ggml-cpu/unary-ops.cpp
    ${RNWHISPER_LIB_DIR}/ggml-cpu/binary-ops.cpp
    ${RNWHISPER_LIB_DIR}/ggml-cpu/vec.cpp
    ${RNWHISPER_LIB_DIR}/ggml-cpu/ops.cpp
    ${RNWHISPER_LIB_DIR}/ggml-opt.cpp
    ${RNWHISPER_LIB_DIR}/ggml-threading.cpp
    ${RNWHISPER_LIB_DIR}/ggml-quants.c
    ${RNWHISPER_LIB_DIR}/gguf.cpp
    ${RNWHISPER_LIB_DIR}/whisper.cpp
    ${RNWHISPER_LIB_DIR}/rn-audioutils.cpp
    ${RNWHISPER_LIB_DIR}/rn-whisper.cpp
    ${RNWHISPER_LIB_DIR}/jsi/RNWhisperJSI.cpp
    ${CMAKE_SOURCE_DIR}/jni.cpp
)

find_library(LOG_LIB log)

function(build_library target_name arch cpu_flags)
    if (NOT ${arch} STREQUAL "generic")
        set(SOURCE_FILES_ARCH
            ${RNWHISPER_LIB_DIR}/ggml-cpu/arch/${arch}/quants.c
            ${RNWHISPER_LIB_DIR}/ggml-cpu/arch/${arch}/repack.cpp
        )
    endif ()

    add_library(
        ${target_name}
        SHARED
        ${SOURCE_FILES}
        ${SOURCE_FILES_ARCH}
    )

    # Link JSI libraries
    if(${REACT_NATIVE_MINOR_VERSION} GREATER_EQUAL 76)
        target_link_libraries(${target_name}
            ${LOG_LIB}
            android
            fbjni::fbjni
            ReactAndroid::jsi
            ReactAndroid::reactnative
        )
    else ()
        target_link_libraries(${target_name}
            ${LOG_LIB}
            android
            fbjni::fbjni
            ReactAndroid::jsi
            ReactAndroid::turbomodulejsijni
            ReactAndroid::react_nativemodule_core
        )
    endif ()

    if (${arch} STREQUAL "generic")
        target_compile_options(${target_name} PRIVATE -DWSP_GGML_CPU_GENERIC)
    endif ()

    target_compile_options(${target_name} PRIVATE -DWSP_GGML_USE_CPU -DWSP_GGML_USE_CPU_REPACK -pthread ${cpu_flags})

    if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE -DRNWHISPER_ANDROID_ENABLE_LOGGING)
    endif ()

    # NOTE: If you want to debug the native code, you can uncomment if and endif
    # Note that it will be extremely slow
    # if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE -O3 -DNDEBUG)
        target_compile_options(${target_name} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
        target_compile_options(${target_name} PRIVATE -ffunction-sections -fdata-sections)

        target_link_options(${target_name} PRIVATE -Wl,--gc-sections)
        target_link_options(${target_name} PRIVATE -Wl,--exclude-libs,ALL)
        target_link_options(${target_name} PRIVATE -flto)
    # endif ()
endfunction()

build_library("rnwhisper" "generic" "")

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    build_library("rnwhisper_v8fp16_va_2" "arm" "-march=armv8.2-a+fp16")
    build_library("rnwhisper_v8" "arm" "-march=armv8-a")
elseif (${ANDROID_ABI} STREQUAL "armeabi-v7a")
    build_library("rnwhisper_vfpv4" "arm" "-mfpu=neon-vfpv4")
elseif (${ANDROID_ABI} STREQUAL "x86_64")
    # x86_64 target
    build_library("rnwhisper_x86_64" "x86" "-march=x86-64" "-mtune=intel" "-msse4.2" "-mpopcnt")
endif ()


include_directories(${RNWHISPER_LIB_DIR})
