#ifndef RNWHISPER_H
#define RNWHISPER_H

#include <functional>
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

// Placement of a job's ggml CPU compute threads (cpu_mask / cpu_strict).
struct cpu_params {
    bool mask[GGML_MAX_N_THREADS] = {};
    bool strict = false;           // one thread per CPU of the mask, in order

    // At least one CPU set. Inline: the JSI glue links against the core
    // library, which exports only RNWHISPER_API symbols on Android.
    bool has_mask() const {
        for (bool set : mask) {
            if (set) {
                return true;
            }
        }
        return false;
    }
};

// cpu_mask syntax shared with llama.rn / llama.node, i.e. llama.cpp common's:
// a hex mask with an optional 0x prefix ("0xC0", "C0": CPUs 6 and 7), or, as
// with llama.cpp's --cpu-range, an inclusive "lo-hi" range where either end
// may be omitted ("6-7", "4-"). Bits are OR-ed into boolmask. Returns false
// for invalid input.
RNWHISPER_API bool parse_cpu_mask(const std::string & mask, bool (&boolmask)[GGML_MAX_N_THREADS]);

struct RNWHISPER_API job {
    int job_id;
    bool aborted = false;
    whisper_full_params params;
    int n_processors = 1;

    // Owned pool behind params.threadpool, only when the job has a cpu mask.
    ggml_threadpool_t threadpool = nullptr;
    // Undoes the pinning ggml applies to the thread that created the pool.
    std::function<void()> restore_affinity;

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
// With a cpu mask, the job gets a ggml threadpool of params.n_threads threads
// placed on that mask (not on Apple platforms, where ggml has no affinity). Call job_new and job_remove on the thread that runs
// whisper_full*: ggml pins the creating thread too, and the job restores its
// affinity when removed.
RNWHISPER_API job* job_new(int job_id, struct whisper_full_params params, const cpu_params * cpu = nullptr);
RNWHISPER_API void job_remove(int job_id);
RNWHISPER_API job* job_get(int job_id);

} // namespace rnwhisper

#endif // RNWHISPER_H
