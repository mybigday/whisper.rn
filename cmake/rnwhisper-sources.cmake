# Source and include lists shared by every native build of whisper.rn:
#   ios/CMakeLists.txt               (xcframework)
#   android/src/main/CMakeLists.txt  (per-CPU-variant .so)
#
# vendor/whisper.cpp keeps its upstream layout, so the paths below match
# upstream's own CMake targets (see vendor/README.md). Backends that only some
# builds enable (Metal, Core ML) are exposed as separate variables for the
# caller to add. The CocoaPods equivalent lives in whisper-rn.podspec.
include_guard(GLOBAL)

get_filename_component(RNWHISPER_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(RNWHISPER_CPP_DIR         "${RNWHISPER_ROOT_DIR}/cpp")
set(RNWHISPER_WHISPER_CPP_DIR "${RNWHISPER_ROOT_DIR}/vendor/whisper.cpp")

set(_ggml    "${RNWHISPER_WHISPER_CPP_DIR}/ggml/src")
set(_whisper "${RNWHISPER_WHISPER_CPP_DIR}/src")

set(RNWHISPER_INCLUDE_DIRS
    ${RNWHISPER_CPP_DIR}
    ${RNWHISPER_WHISPER_CPP_DIR}/include
    ${RNWHISPER_WHISPER_CPP_DIR}/ggml/include
    ${_ggml}                                    # ggml-impl.h, "ggml-cpu/ggml-cpu-impl.h"
    ${_ggml}/ggml-cpu                           # ggml-cpu/amx includes its siblings bare
    ${_whisper}                                 # whisper-arch.h, "coreml/whisper-encoder.h"
)

# --- version definitions ------------------------------------------------------
# whisper.cpp passes these from its own CMake project. The vendor sync records
# them in src/version.json (scripts/sync-vendor.sh); ggml's own version macros
# come from the generated ggml/src/ggml-version.h.
function(_rnwhisper_version_field key out_var)
    file(READ "${RNWHISPER_ROOT_DIR}/src/version.json" _json)
    if (NOT _json MATCHES "\"${key}\":\"([^\"]+)\"")
        message(FATAL_ERROR "src/version.json has no \"${key}\"; run scripts/sync-vendor.sh")
    endif ()
    set(${out_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()
_rnwhisper_version_field(version RNWHISPER_WHISPER_VERSION)
set(RNWHISPER_VERSION_DEFINITIONS
    WHISPER_VERSION="${RNWHISPER_WHISPER_VERSION}"
    PARAKEET_VERSION="${RNWHISPER_WHISPER_VERSION}"
)

# --- ggml ---------------------------------------------------------------------
# vendor/whisper.cpp also carries what upstream's CMake project needs for
# whisper.node (see vendor/README.md); the filters below leave out the parts
# no whisper.rn build uses. Keep them in sync with whisper-rn.podspec.
file(GLOB RNWHISPER_GGML_CPU_SOURCES CONFIGURE_DEPENDS
    ${_ggml}/ggml-cpu/*.c
    ${_ggml}/ggml-cpu/*.cpp
    ${_ggml}/ggml-cpu/amx/*.cpp
)
# Intel HBM allocator (GGML_USE_CPU_HBM); never enabled here.
list(FILTER RNWHISPER_GGML_CPU_SOURCES EXCLUDE REGEX "/hbm\\.cpp$")
set(RNWHISPER_GGML_SOURCES
    ${_ggml}/ggml.c
    ${_ggml}/ggml.cpp
    ${_ggml}/ggml-alloc.c
    ${_ggml}/ggml-backend.cpp
    ${_ggml}/ggml-backend-dl.cpp
    ${_ggml}/ggml-backend-meta.cpp
    ${_ggml}/ggml-backend-reg.cpp
    ${_ggml}/ggml-opt.cpp
    ${_ggml}/ggml-threading.cpp
    ${_ggml}/ggml-quants.c
    ${_ggml}/gguf.cpp
    ${RNWHISPER_GGML_CPU_SOURCES}
)
# Per-arch SIMD kernels; callers add ${RNWHISPER_GGML_CPU_ARCH_DIR}/<arm|x86>/{quants.c,repack.cpp}
# when not building with GGML_CPU_GENERIC.
set(RNWHISPER_GGML_CPU_ARCH_DIR "${_ggml}/ggml-cpu/arch")

# Metal kernels are compiled at runtime from the .metal sources in
# ${RNWHISPER_GGML_METAL_DIR}/kernels, which the Apple builds ship as resources
# together with ggml-metal-impl.h and ggml-common.h.
set(RNWHISPER_GGML_METAL_DIR "${_ggml}/ggml-metal")
file(GLOB RNWHISPER_GGML_METAL_SOURCES      CONFIGURE_DEPENDS ${_ggml}/ggml-metal/*.cpp ${_ggml}/ggml-metal/*.m)
file(GLOB RNWHISPER_GGML_METAL_OBJC_SOURCES CONFIGURE_DEPENDS ${_ggml}/ggml-metal/*.m)  # need -fno-objc-arc

# Hexagon (Android only). The host side is ggml-hexagon.cpp + htp-drv.cpp plus
# the FastRPC stub that scripts/build-hexagon-htp.sh generates into htp/v73/;
# the DSP side (htp/) is cross-compiled by that script into libggml-htp-*.so.
set(RNWHISPER_GGML_HEXAGON_DIR "${_ggml}/ggml-hexagon")

# --- whisper ------------------------------------------------------------------
set(RNWHISPER_WHISPER_SOURCES
    ${_whisper}/whisper.cpp
    ${_whisper}/parakeet.cpp
)
file(GLOB RNWHISPER_COREML_SOURCES CONFIGURE_DEPENDS ${_whisper}/coreml/*.m ${_whisper}/coreml/*.mm)  # need -fobjc-arc

# --- whisper.rn ---------------------------------------------------------------
set(RNWHISPER_RN_SOURCES
    ${RNWHISPER_CPP_DIR}/rn-whisper.cpp
)

# Everything a CPU-only build of the library needs.
set(RNWHISPER_CORE_SOURCES
    ${RNWHISPER_GGML_SOURCES}
    ${RNWHISPER_WHISPER_SOURCES}
    ${RNWHISPER_RN_SOURCES}
)

unset(_ggml)
unset(_whisper)
