cmake_minimum_required(VERSION 3.10)

project(whisper.rn)

set(CMAKE_CXX_STANDARD 11)
set(RNWHISPER_LIB_DIR ${CMAKE_SOURCE_DIR}/../../../cpp)

set(
    SOURCE_FILES
    ${RNWHISPER_LIB_DIR}/ggml.c
    ${RNWHISPER_LIB_DIR}/whisper.cpp
    ${RNWHISPER_LIB_DIR}/rn-whisper.cpp
    ${CMAKE_SOURCE_DIR}/jni.cpp
)

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    set(RNWHISPER_LIBRARY_NAME whisper_v8fp16_va)
elseif (${ANDROID_ABI} STREQUAL "armeabi-v7a")
    set(RNWHISPER_LIBRARY_NAME whisper_vfpv4)
endif ()

add_library(
    ${RNWHISPER_LIBRARY_NAME}
    SHARED
    ${SOURCE_FILES}
)

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    target_compile_options(${RNWHISPER_LIBRARY_NAME} PRIVATE -march=armv8.2-a+fp16)
elseif (${ANDROID_ABI} STREQUAL "armeabi-v7a")
    target_compile_options(${RNWHISPER_LIBRARY_NAME} PRIVATE -mfpu=neon-vfpv4)
endif ()

target_link_libraries(${RNWHISPER_LIBRARY_NAME} log android)
include_directories(${RNWHISPER_LIB_DIR})

target_compile_options(${RNWHISPER_LIBRARY_NAME} PRIVATE -DLM_GGML_USE_K_QUANTS -pthread)

# NOTE: If you want to debug the native code, you can uncomment if and endif
# if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")

target_compile_options(${RNWHISPER_LIBRARY_NAME} PRIVATE -O3 -DNDEBUG)
target_compile_options(${RNWHISPER_LIBRARY_NAME} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
target_compile_options(${RNWHISPER_LIBRARY_NAME} PRIVATE -ffunction-sections -fdata-sections)

target_link_options(${RNWHISPER_LIBRARY_NAME} PRIVATE -Wl,--gc-sections)
target_link_options(${RNWHISPER_LIBRARY_NAME} PRIVATE -Wl,--exclude-libs,ALL)
target_link_options(${RNWHISPER_LIBRARY_NAME} PRIVATE -flto)

# endif ()
