cmake_minimum_required(VERSION 3.16)

# AGP entry point. Builds the JNI/JSI wrapper (jni.cpp + cpp/jsi) from source
# for every CPU-feature variant, because it links against the host app's React
# Native (fbjni/jsi/reactnative prefab targets) and therefore depends on the RN
# version. Each wrapper links one rnwhisper core library (rnwhisper/CMakeLists.txt):
#   - prebuilt (default): jniLibs/<abi>/lib<core>.so, downloaded by the package's
#     postinstall from the matching GitHub release (install/download-native-artifacts.js)
#   - from source (-DRNWHISPER_BUILD_FROM_SOURCE=ON, gradle rnwhisperBuildFromSource=true):
#     the core is built here as a subdirectory
project(whisper.rn)

set(CMAKE_CXX_STANDARD 17)

# Configure STL to be compatible with ReactAndroid JSI libraries
set(CMAKE_ANDROID_STL_TYPE c++_shared)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/rnwhisper-build-options.cmake)

# Find ReactAndroid package for JSI
find_package(ReactAndroid REQUIRED CONFIG)
find_package(fbjni REQUIRED CONFIG)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/rnwhisper-sources.cmake)

option(RNWHISPER_BUILD_FROM_SOURCE "Build the rnwhisper core libraries from source instead of using jniLibs prebuilts" OFF)

set(RNWHISPER_PREBUILT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/jniLibs/${ANDROID_ABI}")

if (RNWHISPER_BUILD_FROM_SOURCE)
    message(STATUS "whisper.rn: building rnwhisper core libraries from source")
    add_subdirectory(rnwhisper)
else ()
    message(STATUS "whisper.rn: using prebuilt rnwhisper core libraries from ${RNWHISPER_PREBUILT_DIR}")
    if (NOT EXISTS "${RNWHISPER_PREBUILT_DIR}/librnwhisper.so")
        message(FATAL_ERROR
            "whisper.rn: no prebuilt libraries for ${ANDROID_ABI} at ${RNWHISPER_PREBUILT_DIR}. "
            "Run `npx whisper-rn-download-artifacts` (the package's postinstall) to fetch them, "
            "or set `rnwhisperBuildFromSource=true` in android/gradle.properties to build from source.")
    endif ()
endif ()

find_library(LOG_LIB log)

set(JNI_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/jni.cpp
    ${RNWHISPER_CPP_DIR}/jsi/RNWhisperJSI.cpp
)

# The wrapper for core library ${core_name} is lib${jni_name}.so; RNWhisper.java
# loads the wrapper and the dynamic linker pulls the core in through DT_NEEDED.
function(build_rnwhisper_jni jni_name core_name)
    rnwhisper_variant_enabled(${core_name} VARIANT_ENABLED)
    if (NOT VARIANT_ENABLED)
        message(STATUS "Skipping ${jni_name} - not in RNWHISPER_ANDROID_VARIANTS")
        return()
    endif ()

    if (RNWHISPER_BUILD_FROM_SOURCE)
        if (NOT TARGET ${core_name})
            # e.g. the Hexagon variant without the SDK on the build host
            message(STATUS "Skipping ${jni_name} - ${core_name} is not built")
            return()
        endif ()
    else ()
        set(PREBUILT_LIB "${RNWHISPER_PREBUILT_DIR}/lib${core_name}.so")
        if (NOT EXISTS "${PREBUILT_LIB}")
            message(STATUS "Skipping ${jni_name} - no prebuilt lib${core_name}.so for ${ANDROID_ABI}")
            return()
        endif ()
    endif ()

    add_library(${jni_name} SHARED ${JNI_SOURCE_FILES})

    target_include_directories(${jni_name} PRIVATE
        ${RNWHISPER_INCLUDE_DIRS}
        ${RNWHISPER_CPP_DIR}/jsi
    )

    # Link JSI libraries
    if (${REACT_NATIVE_MINOR_VERSION} GREATER_EQUAL 76)
        target_link_libraries(${jni_name} PRIVATE
            ${LOG_LIB}
            android
            fbjni::fbjni
            ReactAndroid::jsi
            ReactAndroid::reactnative
        )
    else ()
        target_link_libraries(${jni_name} PRIVATE
            ${LOG_LIB}
            android
            fbjni::fbjni
            ReactAndroid::jsi
            ReactAndroid::turbomodulejsijni
            ReactAndroid::react_nativemodule_core
        )
    endif ()

    if (RNWHISPER_BUILD_FROM_SOURCE)
        target_link_libraries(${jni_name} PRIVATE ${core_name})
    else ()
        # Linking the .so by path records its SONAME (lib${core_name}.so) as
        # DT_NEEDED; AGP packages the file itself from src/main/jniLibs.
        target_link_libraries(${jni_name} PRIVATE ${PREBUILT_LIB})
    endif ()

    # The core exports its API with visibility("default") under these macros;
    # the wrapper only consumes it.
    target_compile_definitions(${jni_name} PRIVATE
        GGML_SHARED WHISPER_SHARED PARAKEET_SHARED RNWHISPER_SHARED
    )

    # jni.cpp reports the Hexagon device only in the variant that carries it.
    if (${core_name} MATCHES "_hexagon$")
        target_compile_options(${jni_name} PRIVATE -DGGML_USE_HEXAGON)
    endif ()

    if (CMAKE_BUILD_TYPE AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${jni_name} PRIVATE -DRNWHISPER_ANDROID_ENABLE_LOGGING)
    endif ()

    target_compile_options(${jni_name} PRIVATE -pthread)
    target_compile_options(${jni_name} PRIVATE -O3 -DNDEBUG)
    target_compile_options(${jni_name} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
    target_compile_options(${jni_name} PRIVATE -ffunction-sections -fdata-sections)

    target_link_options(${jni_name} PRIVATE -Wl,--gc-sections)
    target_link_options(${jni_name} PRIVATE -Wl,--exclude-libs,ALL)
endfunction()

rnwhisper_android_variants(RNWHISPER_VARIANTS)
foreach (entry IN LISTS RNWHISPER_VARIANTS)
    rnwhisper_split_variant("${entry}" core arch flags)
    # rnwhisper -> rnwhisper_jni, rnwhisper_v8 -> rnwhisper_jni_v8, ...
    string(REGEX REPLACE "^rnwhisper" "rnwhisper_jni" jni "${core}")
    build_rnwhisper_jni(${jni} ${core})
endforeach ()
