# Core whisper library build (without JNI/ReactAndroid dependencies)
# This is included via add_subdirectory() from the main CMakeLists.txt

set(RNWHISPER_LIB_DIR ${CMAKE_SOURCE_DIR}/../../../cpp)

include_directories(
    ${RNWHISPER_LIB_DIR}
    ${RNWHISPER_LIB_DIR}/ggml-cpu
)

# Core source files (without JNI)
set(
    RNWHISPER_CORE_SOURCE_FILES
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
)

find_library(LOG_LIB log)

# Build core library function
function(build_rnwhisper_library target_name arch cpu_flags)
    # Check if Hexagon is enabled (name contains "_hexagon")
    string(REGEX MATCH ".*_hexagon.*" ENABLE_HEXAGON ${target_name})

    if (NOT ${arch} STREQUAL "generic")
        set(SOURCE_FILES_ARCH
            ${RNWHISPER_LIB_DIR}/ggml-cpu/arch/${arch}/quants.c
            ${RNWHISPER_LIB_DIR}/ggml-cpu/arch/${arch}/repack.cpp
        )
    endif ()

    # Hexagon source files
    if (ENABLE_HEXAGON)
        set(HEXAGON_PREBUILT_HTP_DIR "${RNWHISPER_LIB_DIR}/ggml-hexagon/htp/v73/")
        set(HTP_IFACE_STUB ${HEXAGON_PREBUILT_HTP_DIR}/htp_iface_stub.c)

        if (NOT EXISTS ${HTP_IFACE_STUB})
            message(WARNING "Hexagon HTP interface stub not found at ${HTP_IFACE_STUB}. Run ./scripts/build-hexagon-htp.sh first.")
            return()
        endif()

        set(CDSPRPC_LIB "${HEXAGON_SDK_ROOT}/ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so")
        if (NOT EXISTS ${CDSPRPC_LIB})
            message(WARNING "libcdsprpc.so not found at ${CDSPRPC_LIB}. Check HEXAGON_SDK_ROOT.")
            return()
        endif()

        set(SOURCE_FILES_HEXAGON
            ${RNWHISPER_LIB_DIR}/ggml-hexagon/ggml-hexagon.cpp
            ${RNWHISPER_LIB_DIR}/ggml-hexagon/htp-utils.c
            ${HTP_IFACE_STUB}
        )
    endif ()

    add_library(
        ${target_name}
        SHARED
        ${RNWHISPER_CORE_SOURCE_FILES}
        ${SOURCE_FILES_ARCH}
        ${SOURCE_FILES_HEXAGON}
    )

    # Core library only links to basic Android libs - NO ReactAndroid/fbjni
    if (ENABLE_HEXAGON)
        target_link_libraries(${target_name} PRIVATE ${LOG_LIB} android ${CDSPRPC_LIB})

        target_include_directories(${target_name} PRIVATE
            ${HEXAGON_SDK_ROOT}/incs
            ${HEXAGON_SDK_ROOT}/incs/stddef
            ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc
            ${HEXAGON_SDK_ROOT}/utils/examples
            ${RNWHISPER_LIB_DIR}/ggml-hexagon
            ${RNWHISPER_LIB_DIR}/ggml-hexagon/htp
            ${HEXAGON_PREBUILT_HTP_DIR}
        )

        target_compile_options(${target_name} PRIVATE -DWSP_GGML_USE_HEXAGON)
        target_link_options(${target_name} PRIVATE -ldl)
    else ()
        target_link_libraries(${target_name} PRIVATE ${LOG_LIB} android)
    endif ()

    if (${arch} STREQUAL "generic")
        target_compile_options(${target_name} PRIVATE -DWSP_GGML_CPU_GENERIC)
    endif ()

    target_compile_options(${target_name} PRIVATE -DWSP_GGML_USE_CPU -DWSP_GGML_USE_CPU_REPACK -pthread ${cpu_flags} -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE)

    if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE -DRNWHISPER_ANDROID_ENABLE_LOGGING)
    endif ()

    target_compile_options(${target_name} PRIVATE -O3 -DNDEBUG)
    target_compile_options(${target_name} PRIVATE -ffunction-sections -fdata-sections)

    target_link_options(${target_name} PRIVATE -Wl,--gc-sections)
    target_link_options(${target_name} PRIVATE -flto)
endfunction()

# Build core library variants
build_rnwhisper_library("rnwhisper" "generic" "")

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    build_rnwhisper_library("rnwhisper_v8fp16_va_2" "arm" "-march=armv8.2-a+fp16")
    build_rnwhisper_library("rnwhisper_v8" "arm" "-march=armv8-a")
    # Hexagon HTP variant (experimental)
    build_rnwhisper_library("rnwhisper_v8_2_hexagon" "arm" "-march=armv8.2-a+fp16+dotprod")
elseif (${ANDROID_ABI} STREQUAL "armeabi-v7a")
    build_rnwhisper_library("rnwhisper_vfpv4" "arm" "-mfpu=neon-vfpv4")
elseif (${ANDROID_ABI} STREQUAL "x86_64")
    build_rnwhisper_library("rnwhisper_x86_64" "x86" "-march=x86-64" "-mtune=intel" "-msse4.2" "-mpopcnt")
endif ()
