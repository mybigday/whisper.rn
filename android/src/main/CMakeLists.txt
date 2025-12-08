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

# Include core library builds
add_subdirectory(rnwhisper)

# JNI source files
set(
    JNI_SOURCE_FILES
    ${RNWHISPER_LIB_DIR}/jsi/RNWhisperJSI.cpp
    ${CMAKE_SOURCE_DIR}/jni.cpp
)

find_library(LOG_LIB log)

# Build JNI wrapper library function
function(build_rnwhisper_jni jni_name rnwhisper_name cpu_flags)
    # Check if Hexagon is enabled (name contains "_hexagon")
    string(REGEX MATCH ".*_hexagon.*" ENABLE_HEXAGON ${rnwhisper_name})

    add_library(
        ${jni_name}
        SHARED
        ${JNI_SOURCE_FILES}
    )

    # JNI wrapper links to core library and ReactAndroid
    if(${REACT_NATIVE_MINOR_VERSION} GREATER_EQUAL 76)
        target_link_libraries(${jni_name}
            ${LOG_LIB}
            android
            ${rnwhisper_name}
            fbjni::fbjni
            ReactAndroid::jsi
            ReactAndroid::reactnative
        )
    else ()
        target_link_libraries(${jni_name}
            ${LOG_LIB}
            android
            ${rnwhisper_name}
            fbjni::fbjni
            ReactAndroid::jsi
            ReactAndroid::turbomodulejsijni
            ReactAndroid::react_nativemodule_core
        )
    endif ()

    target_compile_options(${jni_name} PRIVATE -pthread ${cpu_flags})

    if (ENABLE_HEXAGON)
        target_compile_options(${jni_name} PRIVATE -DWSP_GGML_USE_HEXAGON)
    endif ()

    if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        target_compile_options(${jni_name} PRIVATE -DRNWHISPER_ANDROID_ENABLE_LOGGING)
    endif ()

    target_compile_options(${jni_name} PRIVATE -O3 -DNDEBUG)
    target_compile_options(${jni_name} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
    target_compile_options(${jni_name} PRIVATE -ffunction-sections -fdata-sections)

    target_link_options(${jni_name} PRIVATE -Wl,--gc-sections)
    target_link_options(${jni_name} PRIVATE -Wl,--exclude-libs,ALL)
    target_link_options(${jni_name} PRIVATE -flto)
endfunction()

# Build JNI wrapper variants
build_rnwhisper_jni("rnwhisper_jni" "rnwhisper" "")

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    build_rnwhisper_jni("rnwhisper_jni_v8fp16_va_2" "rnwhisper_v8fp16_va_2" "-march=armv8.2-a+fp16")
    build_rnwhisper_jni("rnwhisper_jni_v8" "rnwhisper_v8" "-march=armv8-a")
    # Hexagon HTP variant (experimental)
    build_rnwhisper_jni("rnwhisper_jni_v8_2_hexagon" "rnwhisper_v8_2_hexagon" "-march=armv8.2-a+fp16+dotprod")
elseif (${ANDROID_ABI} STREQUAL "armeabi-v7a")
    build_rnwhisper_jni("rnwhisper_jni_vfpv4" "rnwhisper_vfpv4" "-mfpu=neon-vfpv4")
elseif (${ANDROID_ABI} STREQUAL "x86_64")
    build_rnwhisper_jni("rnwhisper_jni_x86_64" "rnwhisper_x86_64" "-march=x86-64" "-mtune=intel" "-msse4.2" "-mpopcnt")
endif ()

include_directories(${RNWHISPER_LIB_DIR})
