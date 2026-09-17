cmake_minimum_required(VERSION 3.16)

# The rnwhisper core: whisper.cpp + ggml + cpp/rn-whisper.cpp, one shared
# library per CPU-feature variant. It has no React Native dependency, so it can
# be prebuilt once (scripts/build-android.sh -> android/src/main/jniLibs, shipped
# as a GitHub release asset) and reused across React Native versions. The
# JNI/JSI wrapper in ../CMakeLists.txt links against it, either as a sibling
# target (rnwhisperBuildFromSource=true) or as the prebuilt .so.
project(rnwhisper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_ANDROID_STL_TYPE c++_shared)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/rnwhisper-build-options.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/rnwhisper-sources.cmake)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(RNWHISPER_STANDALONE_BUILD ON)
    message(STATUS "rnwhisper: standalone build (prebuilt distribution)")
else ()
    set(RNWHISPER_STANDALONE_BUILD OFF)
    message(STATUS "rnwhisper: building as a subdirectory of the JNI wrapper")
endif ()

# Hexagon (Qualcomm NPU). The rnwhisper_*_hexagon variant is only built when
# the SDK is present: its host side needs the SDK's FastRPC headers and the
# stub that scripts/build-hexagon-htp.sh generates into htp/v73/ (the driver
# itself dlopens the device's libcdsprpc.so, see htp-drv.cpp). The DSP-side
# libggml-htp-*.so come from the same script and are shipped as app assets
# (see ../../../build.gradle).
set(HEXAGON_SDK_ROOT "" CACHE PATH "Hexagon SDK root directory")
set(RNWHISPER_HEXAGON_STUB_DIR "${RNWHISPER_GGML_HEXAGON_DIR}/htp/v73")
if (HEXAGON_SDK_ROOT AND IS_DIRECTORY "${HEXAGON_SDK_ROOT}/incs/stddef")
    set(RNWHISPER_HEXAGON_AVAILABLE ON)
    message(STATUS "rnwhisper: Hexagon SDK found at ${HEXAGON_SDK_ROOT}")
else ()
    set(RNWHISPER_HEXAGON_AVAILABLE OFF)
    message(STATUS "rnwhisper: Hexagon SDK not found, skipping the rnwhisper_*_hexagon variant")
endif ()

find_library(LOG_LIB log)

function(build_rnwhisper_library target_name arch cpu_flags)
    rnwhisper_variant_enabled(${target_name} VARIANT_ENABLED)
    if (NOT VARIANT_ENABLED)
        message(STATUS "Skipping ${target_name} - not in RNWHISPER_ANDROID_VARIANTS")
        return()
    endif ()

    set(ENABLE_HEXAGON OFF)
    if (${target_name} MATCHES "_hexagon$")
        if (NOT RNWHISPER_HEXAGON_AVAILABLE)
            message(STATUS "Skipping ${target_name} - Hexagon SDK not available")
            return()
        endif ()
        set(ENABLE_HEXAGON ON)
    endif ()

    if (NOT ${arch} STREQUAL "generic")
        set(SOURCE_FILES_ARCH
            ${RNWHISPER_GGML_CPU_ARCH_DIR}/${arch}/quants.c
            ${RNWHISPER_GGML_CPU_ARCH_DIR}/${arch}/repack.cpp
        )
    endif ()

    add_library(
        ${target_name}
        SHARED
        ${RNWHISPER_CORE_SOURCES}
        ${SOURCE_FILES_ARCH}
    )

    target_include_directories(${target_name} PRIVATE ${RNWHISPER_INCLUDE_DIRS})
    target_link_libraries(${target_name} PRIVATE ${LOG_LIB} android)

    if (${arch} STREQUAL "generic")
        target_compile_options(${target_name} PRIVATE -DGGML_CPU_GENERIC)
    endif ()

    # Only the public whisper/ggml/parakeet/rnwhisper APIs are exported; the
    # *_SHARED/*_BUILD macros switch their API annotations to
    # visibility("default") while everything else stays hidden.
    target_compile_definitions(${target_name} PRIVATE
        _GNU_SOURCE  # Linux CPU affinity APIs used by ggml on Android
        GGML_SHARED GGML_BUILD
        WHISPER_SHARED WHISPER_BUILD
        PARAKEET_SHARED PARAKEET_BUILD
        RNWHISPER_SHARED RNWHISPER_BUILD
        ${RNWHISPER_VERSION_DEFINITIONS}
    )
    target_compile_options(${target_name} PRIVATE -DGGML_USE_CPU -DGGML_USE_CPU_REPACK -pthread ${cpu_flags})

    if (CMAKE_BUILD_TYPE AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE -DRNWHISPER_ANDROID_ENABLE_LOGGING)
    endif ()

    if (ENABLE_HEXAGON)
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
        target_link_libraries(${target_name} PRIVATE dl)
        target_compile_options(${target_name} PRIVATE -DGGML_USE_HEXAGON)
    endif ()

    target_compile_options(${target_name} PRIVATE -O3 -DNDEBUG)
    target_compile_options(${target_name} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
    target_compile_options(${target_name} PRIVATE -ffunction-sections -fdata-sections)

    target_link_options(${target_name} PRIVATE -Wl,--gc-sections)
    target_link_options(${target_name} PRIVATE -Wl,--exclude-libs,ALL)
    # ggml/whisper symbols keep upstream names. Bind this library's own
    # references at link time so a second ggml copy in the process (e.g.
    # llama.rn) can never be picked up via the dynamic linker's global lookup.
    target_link_options(${target_name} PRIVATE -Wl,-Bsymbolic)
    # ThinLTO rather than monolithic LTO: linking is parallel and incremental,
    # and peak memory stays low enough for several variants to link at once.
    target_compile_options(${target_name} PRIVATE -flto=thin)
    target_link_options(${target_name} PRIVATE -flto=thin)
endfunction()

rnwhisper_android_variants(RNWHISPER_VARIANTS)
foreach (entry IN LISTS RNWHISPER_VARIANTS)
    rnwhisper_split_variant("${entry}" core arch flags)
    build_rnwhisper_library(${core} ${arch} "${flags}")
endforeach ()
