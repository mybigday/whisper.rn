cmake_minimum_required(VERSION 3.16)

project(whisper.rn)

set(CMAKE_CXX_STANDARD 17)

# Configure STL to be compatible with ReactAndroid JSI libraries
set(CMAKE_ANDROID_STL_TYPE c++_shared)

# Find ReactAndroid package for JSI
find_package(ReactAndroid REQUIRED CONFIG)
find_package(fbjni REQUIRED CONFIG)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/rnwhisper-sources.cmake)

# Hexagon (Qualcomm NPU). The rnwhisper_*_hexagon variant is only built when
# the SDK is present: its host side needs the SDK's FastRPC headers and the
# stub that scripts/build-hexagon-htp.sh generates into htp/v73/ (the driver
# itself dlopens the device's libcdsprpc.so, see htp-drv.cpp). The DSP-side
# libggml-htp-*.so come from the same script and are shipped as app assets
# (see ../../build.gradle).
set(HEXAGON_SDK_ROOT "" CACHE PATH "Hexagon SDK root directory")
set(RNWHISPER_HEXAGON_STUB_DIR "${RNWHISPER_GGML_HEXAGON_DIR}/htp/v73")
if (HEXAGON_SDK_ROOT AND IS_DIRECTORY "${HEXAGON_SDK_ROOT}/incs/stddef")
    set(RNWHISPER_HEXAGON_AVAILABLE ON)
    message(STATUS "Hexagon SDK found at ${HEXAGON_SDK_ROOT}")
else ()
    set(RNWHISPER_HEXAGON_AVAILABLE OFF)
    message(STATUS "Hexagon SDK not found, skipping the rnwhisper_*_hexagon variant")
endif ()

include_directories(
    ${RNWHISPER_INCLUDE_DIRS}
    ${RNWHISPER_CPP_DIR}/jsi
)

set(
    SOURCE_FILES
    ${RNWHISPER_CORE_SOURCES}
    ${RNWHISPER_CPP_DIR}/jsi/RNWhisperJSI.cpp
    ${CMAKE_SOURCE_DIR}/jni.cpp
)

find_library(LOG_LIB log)

function(build_library target_name arch cpu_flags)
    if (NOT ${arch} STREQUAL "generic")
        set(SOURCE_FILES_ARCH
            ${RNWHISPER_GGML_CPU_ARCH_DIR}/${arch}/quants.c
            ${RNWHISPER_GGML_CPU_ARCH_DIR}/${arch}/repack.cpp
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
        target_compile_options(${target_name} PRIVATE -DGGML_CPU_GENERIC)
    endif ()

    # Expose Linux CPU affinity APIs used by ggml on Android.
    target_compile_definitions(${target_name} PRIVATE _GNU_SOURCE ${RNWHISPER_VERSION_DEFINITIONS})
    target_compile_options(${target_name} PRIVATE -DGGML_USE_CPU -DGGML_USE_CPU_REPACK -pthread ${cpu_flags})

    if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE -DRNWHISPER_ANDROID_ENABLE_LOGGING)
    endif ()

    if (${target_name} MATCHES "_hexagon$")
        if (NOT EXISTS "${RNWHISPER_HEXAGON_STUB_DIR}/htp_iface_stub.c" OR NOT EXISTS "${RNWHISPER_HEXAGON_STUB_DIR}/htp_iface.h")
            message(FATAL_ERROR
                "Hexagon FastRPC stub is missing; run scripts/build-hexagon-htp.sh to generate ${RNWHISPER_HEXAGON_STUB_DIR}")
        endif ()

        target_sources(${target_name} PRIVATE
            ${RNWHISPER_GGML_HEXAGON_DIR}/ggml-hexagon.cpp
            ${RNWHISPER_GGML_HEXAGON_DIR}/htp-drv.cpp
            ${RNWHISPER_HEXAGON_STUB_DIR}/htp_iface_stub.c
        )
        target_include_directories(${target_name} PRIVATE
            ${HEXAGON_SDK_ROOT}/incs
            ${HEXAGON_SDK_ROOT}/incs/stddef
            ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc
            ${HEXAGON_SDK_ROOT}/utils/examples
            ${RNWHISPER_GGML_HEXAGON_DIR}
            ${RNWHISPER_GGML_HEXAGON_DIR}/htp
            ${RNWHISPER_HEXAGON_STUB_DIR}
        )
        target_link_libraries(${target_name} dl)
        target_compile_options(${target_name} PRIVATE -DGGML_USE_HEXAGON)
    endif ()

    # NOTE: If you want to debug the native code, you can uncomment if and endif
    # Note that it will be extremely slow
    # if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE -O3 -DNDEBUG)
        target_compile_options(${target_name} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
        target_compile_options(${target_name} PRIVATE -ffunction-sections -fdata-sections)

        target_link_options(${target_name} PRIVATE -Wl,--gc-sections)
        target_link_options(${target_name} PRIVATE -Wl,--exclude-libs,ALL)
        # ggml/whisper symbols keep upstream names. Bind this library's own
        # references at link time so a second ggml copy in the process (e.g.
        # llama.rn) can never be picked up via the dynamic linker's global lookup.
        target_link_options(${target_name} PRIVATE -Wl,-Bsymbolic)
        target_link_options(${target_name} PRIVATE -flto)
    # endif ()
endfunction()

build_library("rnwhisper" "generic" "")

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    if (RNWHISPER_HEXAGON_AVAILABLE)
        build_library("rnwhisper_v8fp16_va_2_hexagon" "arm" "-march=armv8.2-a+fp16")
    endif ()
    build_library("rnwhisper_v8fp16_va_2" "arm" "-march=armv8.2-a+fp16")
    build_library("rnwhisper_v8" "arm" "-march=armv8-a")
elseif (${ANDROID_ABI} STREQUAL "armeabi-v7a")
    build_library("rnwhisper_vfpv4" "arm" "-mfpu=neon-vfpv4")
elseif (${ANDROID_ABI} STREQUAL "x86_64")
    # x86_64 target
    build_library("rnwhisper_x86_64" "x86" "-march=x86-64" "-mtune=intel" "-msse4.2" "-mpopcnt")
endif ()
