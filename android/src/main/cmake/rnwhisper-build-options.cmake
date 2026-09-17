# Build knobs shared by the two Android CMake entry points:
#   android/src/main/CMakeLists.txt            (AGP: JNI wrappers, optionally the core)
#   android/src/main/rnwhisper/CMakeLists.txt  (standalone core, scripts/build-android.sh)
include_guard(GLOBAL)

# --- ccache ------------------------------------------------------------------
# Every CPU-feature variant compiles the whole whisper.cpp tree again, so a warm
# compiler cache is worth a lot on CI and on rebuilds.
option(RNWHISPER_CCACHE "Use ccache to speed up recompilation" ON)

if (RNWHISPER_CCACHE AND NOT CMAKE_C_COMPILER_LAUNCHER)
    find_program(RNWHISPER_CCACHE_BIN NAMES ccache sccache)
    if (RNWHISPER_CCACHE_BIN)
        # include() runs in the calling scope, so these reach the targets defined
        # there and any add_subdirectory() below it.
        set(CMAKE_C_COMPILER_LAUNCHER   ${RNWHISPER_CCACHE_BIN})
        set(CMAKE_CXX_COMPILER_LAUNCHER ${RNWHISPER_CCACHE_BIN})
        message(STATUS "rnwhisper: using compiler cache ${RNWHISPER_CCACHE_BIN}")
    else ()
        message(STATUS "rnwhisper: ccache not found, compiling without a compiler cache")
    endif ()
endif ()

# --- variant selection -------------------------------------------------------
# An empty value (the default) builds every variant; CI narrows this down to the
# ones that exercise distinct code paths.
set(RNWHISPER_ANDROID_VARIANTS "" CACHE STRING
    "Comma/semicolon-separated subset of rnwhisper library variants to build (empty = all)")

if (RNWHISPER_ANDROID_VARIANTS)
    message(STATUS "rnwhisper: restricted to variants ${RNWHISPER_ANDROID_VARIANTS}")
endif ()

# `name` is the core library variant (e.g. rnwhisper_v8), which the JNI wrappers
# are keyed off as well.
function(rnwhisper_variant_enabled name result)
    if (RNWHISPER_ANDROID_VARIANTS STREQUAL "")
        set(${result} TRUE PARENT_SCOPE)
        return()
    endif ()

    string(REPLACE "," ";" wanted "${RNWHISPER_ANDROID_VARIANTS}")
    if ("${name}" IN_LIST wanted)
        set(${result} TRUE PARENT_SCOPE)
    else ()
        set(${result} FALSE PARENT_SCOPE)
    endif ()
endfunction()

# --- variants ----------------------------------------------------------------
# Same list for both entry points so the wrapper built by AGP always has a
# matching core library. `rnwhisper` (generic, no CPU-feature flags) is built
# for every ABI and is the runtime fallback (see RNWhisper.java).
#
# Each entry is "core_name|arch|cpu_flags" (flags space-separated); arch selects the ggml-cpu SIMD
# kernels (arm, x86, or generic for GGML_CPU_GENERIC). A core name ending in
# _hexagon also carries ggml's Hexagon backend (arm64-v8a only).
function(rnwhisper_android_variants out_var)
    set(variants "rnwhisper|generic|")
    if (ANDROID_ABI STREQUAL "arm64-v8a")
        list(APPEND variants
            "rnwhisper_v8fp16_va_2_hexagon|arm|-march=armv8.2-a+fp16"
            "rnwhisper_v8fp16_va_2|arm|-march=armv8.2-a+fp16"
            "rnwhisper_v8|arm|-march=armv8-a"
        )
    elseif (ANDROID_ABI STREQUAL "armeabi-v7a")
        list(APPEND variants "rnwhisper_vfpv4|arm|-mfpu=neon-vfpv4")
    elseif (ANDROID_ABI STREQUAL "x86_64")
        list(APPEND variants "rnwhisper_x86_64|x86|-march=x86-64 -mtune=intel -msse4.2 -mpopcnt")
    endif ()
    set(${out_var} "${variants}" PARENT_SCOPE)
endfunction()

function(rnwhisper_split_variant entry core_var arch_var flags_var)
    string(REPLACE "|" ";" parts "${entry}")
    list(GET parts 0 core)
    list(GET parts 1 arch)
    list(LENGTH parts n)
    set(flags "")
    if (n GREATER 2)
        list(GET parts 2 flags)
        separate_arguments(flags)
    endif ()
    set(${core_var} "${core}" PARENT_SCOPE)
    set(${arch_var} "${arch}" PARENT_SCOPE)
    set(${flags_var} "${flags}" PARENT_SCOPE)
endfunction()
