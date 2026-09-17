#ifndef RNWHISPER_H
#define RNWHISPER_H

#include <string>
#include <vector>
#include "whisper.h"
#include "rn-whisper-log.h"

// The Android core library (android/src/main/rnwhisper) is built with hidden
// visibility and exports only its public API; the JNI/JSI wrapper links against
// it as a separate .so. Elsewhere the macro is a no-op.
#if defined(RNWHISPER_SHARED) && !defined(_WIN32)
#    define RNWHISPER_API __attribute__ ((visibility ("default")))
#else
#    define RNWHISPER_API
#endif

namespace rnwhisper {

RNWHISPER_API std::string bench(whisper_context * ctx, int n_threads);

struct vad_params {
    bool use_vad = false;
    float vad_thold = 0.6f;
    float freq_thold = 100.0f;
    int vad_ms = 2000;
    int last_ms = 1000;
    bool verbose = false;
};

struct RNWHISPER_API job {
    int job_id;
    bool aborted = false;
    whisper_full_params params;
    int n_processors = 1;

    ~job();
    bool is_aborted();
    void abort();
};

// Installs ggml's abort handler. The JSI glue is compiled into the host app and
// linked against this library as a dynamic framework; if another ggml-based
// framework (e.g. llama.rn) is in the same app, an app-level reference to a ggml
// symbol binds to whichever framework the linker sees first. The glue therefore
// goes through this rnwhisper-namespaced wrapper instead of calling ggml directly.
RNWHISPER_API ggml_abort_callback_t set_ggml_abort_callback(ggml_abort_callback_t callback);

RNWHISPER_API void job_abort_all();
RNWHISPER_API job* job_new(int job_id, struct whisper_full_params params);
RNWHISPER_API void job_remove(int job_id);
RNWHISPER_API job* job_get(int job_id);

} // namespace rnwhisper

#endif // RNWHISPER_H
