cmake_minimum_required(VERSION 3.16)
project(vad VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(VAD_STANDALONE OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(VAD_STANDALONE ON)
endif()

option(VAD_CPU_KERNELS     "Run CPU graphs on vad.cpp's own kernels (vad-cpu backend)" ON)
option(VAD_CPU_NATIVE      "Build the CPU kernels only for the host CPU (-march=native)" OFF)
option(VAD_BUILD_TESTS     "Build tests"      ${VAD_STANDALONE})
option(VAD_BUILD_EXAMPLES  "Build examples"   ${VAD_STANDALONE})
option(VAD_BUILD_BENCH     "Build benchmarks" ${VAD_STANDALONE})
option(VAD_WASM_SIMD       "Emscripten: build the CPU kernels with WASM SIMD128" ON)
option(VAD_WASM_THREADS    "Emscripten: pthreads (browsers then need SharedArrayBuffer: COOP/COEP headers)" OFF)

# ------------------------------------------------------------------ ggml
# Reuse a ggml target provided by a parent project, otherwise build the pinned
# submodule. Accelerator backends use the usual GGML_* options (GGML_VULKAN, ...).
if(NOT TARGET ggml)
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/CMakeLists.txt)
        message(FATAL_ERROR "third_party/ggml missing: run git submodule update --init")
    endif()
    add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)
endif()
get_target_property(GGML_SOURCE_DIR ggml SOURCE_DIR)   # ggml/src: backend implementation headers

# ------------------------------------------------------------------ vad-cpu backend
if(VAD_CPU_KERNELS)
    set(VADCPU_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/ggml-vad-cpu)
    add_library(ggml-vad-cpu STATIC
        ${VADCPU_DIR}/vad-cpu.cpp
        ${VADCPU_DIR}/kernels_dispatch.cpp
        ${VADCPU_DIR}/kernels_generic.cpp)
    target_include_directories(ggml-vad-cpu PUBLIC ${VADCPU_DIR} PRIVATE ${GGML_SOURCE_DIR})
    target_link_libraries(ggml-vad-cpu PUBLIC ggml)
    set_target_properties(ggml-vad-cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(NOT MSVC)
        # errno/FP-exception semantics of libm calls block vectorization; the kernels never rely on them
        target_compile_options(ggml-vad-cpu PRIVATE -fno-math-errno -fno-trapping-math)
    endif()

    if(VAD_CPU_NATIVE)
        if(NOT MSVC)
            set_source_files_properties(${VADCPU_DIR}/kernels_generic.cpp PROPERTIES COMPILE_OPTIONS "-march=native")
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        # x86: one build per ISA level, chosen at runtime
        target_sources(ggml-vad-cpu PRIVATE ${VADCPU_DIR}/kernels_avx2.cpp ${VADCPU_DIR}/kernels_avx512.cpp ${VADCPU_DIR}/kernels_avx512vnni.cpp)
        target_compile_definitions(ggml-vad-cpu PRIVATE VADK_X86_VARIANTS)
        if(MSVC)
            set_source_files_properties(${VADCPU_DIR}/kernels_avx2.cpp   PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
            set_source_files_properties(${VADCPU_DIR}/kernels_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
            set_source_files_properties(${VADCPU_DIR}/kernels_avx512vnni.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
        else()
            set_source_files_properties(${VADCPU_DIR}/kernels_avx2.cpp   PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
            set_source_files_properties(${VADCPU_DIR}/kernels_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx2;-mfma;-mf16c")
            set_source_files_properties(${VADCPU_DIR}/kernels_avx512vnni.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx2;-mfma;-mf16c")
        endif()
    endif()
    if(NOT VAD_CPU_NATIVE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$" AND CMAKE_SYSTEM_NAME MATCHES "^(Linux|Android)$" AND NOT MSVC)
        # aarch64 Linux/Android: NEON baseline + an ARMv8.2 fp16/dotprod build, chosen at runtime (HWCAP)
        target_sources(ggml-vad-cpu PRIVATE ${VADCPU_DIR}/kernels_armv82.cpp)
        target_compile_definitions(ggml-vad-cpu PRIVATE VADK_ARM_VARIANTS)
        set_source_files_properties(${VADCPU_DIR}/kernels_armv82.cpp PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+fp16+dotprod")
    endif()
    if(EMSCRIPTEN AND VAD_WASM_SIMD)
        target_compile_options(ggml-vad-cpu PRIVATE -msimd128)
    endif()
endif()

if(EMSCRIPTEN)
    # node runs tests/examples against local files; the browser build is examples/web
    add_link_options(-sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=4MB)
    if(VAD_WASM_THREADS)
        add_compile_options(-pthread)
        add_link_options(-pthread -sPTHREAD_POOL_SIZE=8)
    endif()
    set(VAD_NODE_LINK_OPTIONS -sNODERAWFS=1 -sEXIT_RUNTIME=1)
endif()

# ------------------------------------------------------------------ libvad
add_library(vad
    src/vad.cpp
    src/segments.cpp
    src/weights.cpp
    src/loader/loaders.cpp
    src/models/silero.cpp
    src/models/firered.cpp
    src/models/smart_turn.cpp
    src/models/frame_dft.cpp
    src/models/sortformer.cpp)
target_include_directories(vad PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
    # one frame (streaming) and 8 frames at once (offline) must give identical spectra
    set_source_files_properties(src/models/frame_dft.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()
target_link_libraries(vad PUBLIC ggml)
if(VAD_CPU_KERNELS)
    target_link_libraries(vad PRIVATE ggml-vad-cpu)
    target_compile_definitions(vad PRIVATE VAD_WITH_CPU_KERNELS)
endif()
if(BUILD_SHARED_LIBS)
    target_compile_definitions(vad PUBLIC VAD_SHARED PRIVATE VAD_BUILD)
endif()

# ------------------------------------------------------------------ extras
if(VAD_BUILD_EXAMPLES OR VAD_BUILD_TESTS OR VAD_BUILD_BENCH)
    add_library(vad-common STATIC examples/common/wav.cpp)
    target_include_directories(vad-common PUBLIC examples/common)
endif()

if(VAD_BUILD_EXAMPLES)
    add_executable(vad-cli examples/vad-cli/main.cpp)
    target_link_libraries(vad-cli PRIVATE vad vad-common)
    if(EMSCRIPTEN)
        target_link_options(vad-cli PRIVATE ${VAD_NODE_LINK_OPTIONS})
    endif()
endif()

# browser module: the public C API as an ES module (examples/web)
if(EMSCRIPTEN)
    set(VAD_WEB_EXPORTS
        _vad_device_count _vad_device_name _vad_device_description
        _vad_model_default_params _vad_model_load _vad_model_free _vad_model_output _vad_model_n_outputs _vad_model_arch _vad_model_device
        _vad_model_sample_rate _vad_model_hop_samples _vad_model_warmup_samples _vad_model_max_lookahead_samples
        _vad_stream_default_params _vad_stream_new _vad_stream_new_with_params _vad_stream_free _vad_stream_reset
        _vad_stream_push _vad_stream_n_pending _vad_stream_flush _vad_stream_n_flush _vad_stream_n_buffered
        _vad_streams_push _vad_detect _vad_detect_with_params
        _vad_segment_default_params _vad_segments_from_probs _vad_web_model_load _vad_web_detect)
    list(JOIN VAD_WEB_EXPORTS "," VAD_WEB_EXPORTS_STR)
    # JSPI_EXPORTS matches wasm export names, which have no leading underscore
    string(REPLACE "_vad_" "vad_" VAD_WEB_JSPI_STR "${VAD_WEB_EXPORTS_STR}")
    add_executable(vad-web examples/web/vad_web.cpp)
    target_link_libraries(vad-web PRIVATE vad)
    set_target_properties(vad-web PROPERTIES SUFFIX ".mjs")
    target_link_options(vad-web PRIVATE
        --no-entry -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createVad -sENVIRONMENT=web,worker,node
        "-sEXPORTED_FUNCTIONS=[${VAD_WEB_EXPORTS_STR},_malloc,_free]"
        "-sEXPORTED_RUNTIME_METHODS=[FS,HEAPF32,HEAPU8,UTF8ToString,stringToNewUTF8]"
        -sFORCE_FILESYSTEM=1)
    if(GGML_WEBGPU AND GGML_WEBGPU_JSPI)
        # calls that may reach WebGPU suspend: they return Promises in JS
        target_link_options(vad-web PRIVATE "-sJSPI_EXPORTS=[${VAD_WEB_JSPI_STR}]")
    endif()
    # debugging: per-node comparison of a device against the CPU, in the browser
    add_executable(vad-web-debug tools/debug/compare_backends.cpp)
    target_compile_definitions(vad-web-debug PRIVATE VAD_COMPARE_NO_MAIN)
    target_include_directories(vad-web-debug PRIVATE src)
    target_link_libraries(vad-web-debug PRIVATE vad ggml)
    set_target_properties(vad-web-debug PROPERTIES SUFFIX ".mjs")
    target_link_options(vad-web-debug PRIVATE --no-entry -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createVadDebug
        -sENVIRONMENT=web "-sEXPORTED_FUNCTIONS=[_vad_compare_backends,_malloc,_free]"
        "-sEXPORTED_RUNTIME_METHODS=[FS,stringToNewUTF8,HEAPU32]" -sFORCE_FILESYSTEM=1)
    if(GGML_WEBGPU AND GGML_WEBGPU_JSPI)
        target_link_options(vad-web-debug PRIVATE "-sJSPI_EXPORTS=[vad_compare_backends]")
    endif()
endif()

if(VAD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(VAD_BUILD_BENCH)
    add_executable(vad-bench bench/vad_bench.cpp)
    target_include_directories(vad-bench PRIVATE bench)
    if(EMSCRIPTEN)
        target_link_options(vad-bench PRIVATE ${VAD_NODE_LINK_OPTIONS})
    endif()
    target_link_libraries(vad-bench PRIVATE vad vad-common)
    if(VAD_CPU_KERNELS)
        target_link_libraries(vad-bench PRIVATE ggml-vad-cpu)
        target_compile_definitions(vad-bench PRIVATE VAD_WITH_CPU_KERNELS)
    endif()
endif()

# debugging tools (not installed)
if(VAD_BUILD_TESTS)
    add_executable(compare-backends tools/debug/compare_backends.cpp)
    target_link_libraries(compare-backends PRIVATE vad ggml)
    target_include_directories(compare-backends PRIVATE src)
    if(EMSCRIPTEN)
        target_link_options(compare-backends PRIVATE ${VAD_NODE_LINK_OPTIONS})
    endif()
endif()
