// vad.cpp — voice activity detection on ggml.
//
// Threading: a vad_model may be shared by streams on several threads; compute
// calls on one model are serialized internally. A vad_stream must not be used
// from two threads at once.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef VAD_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef VAD_BUILD
#            define VAD_API __declspec(dllexport)
#        else
#            define VAD_API __declspec(dllimport)
#        endif
#    else
#        define VAD_API __attribute__((visibility("default")))
#    endif
#else
#    define VAD_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vad_model  vad_model;
typedef struct vad_stream vad_stream;

// ------------------------------------------------------------------ devices

// Devices usable for vad_model_params.device:
//   "cpu"      CPU with vad.cpp's own kernels (falls back to "ggml-cpu" if built without them)
//   "ggml-cpu" CPU with the stock ggml kernels
//   any ggml device name (e.g. "Vulkan0", "CUDA0", "Metal") as listed below
VAD_API int          vad_device_count(void);
VAD_API const char * vad_device_name(int i);
VAD_API const char * vad_device_description(int i);

// ------------------------------------------------------------------ model
//
// Every model, whatever its architecture, is used through the same calls. A
// model turns audio into a sequence of output frames of vad_model_n_outputs()
// probabilities each; what they mean, how far apart they are and how much audio
// they need are model properties below. Every count named "probs" in this API
// (max_probs, return values) counts frames: a buffer for n frames holds
// n * vad_model_n_outputs() floats, frame-major.

struct vad_model_params {
    const char * device;     // NULL or "auto" = "cpu" (accelerators only pay off with many concurrent streams)
    int          n_threads;  // CPU threads, <= 0 = 1
};

VAD_API struct vad_model_params vad_model_default_params(void);

// The single entry point for every model: the file header selects the loader
// (vad.cpp GGUF, official GGUF files of supported architectures, or whisper.cpp
// ggml-silero-*.bin). Weights are used at the
// precision stored in the file (F32, F16, BF16 or quantized, see
// scripts/quantize-gguf.py); only the original precision is guaranteed to match
// the reference implementation. NULL on failure (reason printed to stderr).
VAD_API vad_model * vad_model_load(const char * path, struct vad_model_params params);
VAD_API void        vad_model_free(vad_model * model);

enum vad_output {
    VAD_OUTPUT_SPEECH   = 0,   // probability that the audio at that point is speech
    VAD_OUTPUT_TURN_END = 1,   // probability that the speaker has finished their turn at that point
    VAD_OUTPUT_SPEAKERS = 2,   // per-speaker activity (diarization): value s of a frame is the probability that
                               // speaker s talks, speakers numbered by their first appearance in the stream
};

VAD_API enum vad_output vad_model_output (const vad_model * model);
VAD_API const char *    vad_model_arch   (const vad_model * model);   // informational, e.g. for logs
VAD_API const char *    vad_model_device (const vad_model * model);
VAD_API int             vad_model_sample_rate(const vad_model * model);   // Hz
// Probabilities per output frame: 1, or the number of speaker channels for VAD_OUTPUT_SPEAKERS.
VAD_API int             vad_model_n_outputs(const vad_model * model);
// Samples between consecutive output frames with default stream parameters.
// Frame i refers to the audio ending at sample warmup + (i + 1) * hop; a turn-end
// probability produced by vad_stream_flush refers to the end of the audio.
VAD_API int             vad_model_hop_samples(const vad_model * model);
// Samples a fresh stream needs before its first probability (analysis window longer than the hop).
VAD_API int             vad_model_warmup_samples(const vad_model * model);
// Future audio a probability may depend on (0 = causal). Streams can trade it
// for latency (vad_stream_params.lookahead_samples) at some accuracy cost.
VAD_API int             vad_model_max_lookahead_samples(const vad_model * model);

// ------------------------------------------------------------------ streaming

// A stream holds the state of one audio stream; every model can stream.
// Causal models produce one exact probability per hop. Models with lookahead
// re-run a sliding window every slice and emit a probability once lookahead
// samples after it are available: with the maximum lookahead the output equals
// vad_detect, less lookahead lowers latency at some accuracy cost, and a larger
// slice lowers the compute per probability (each window adds a fixed context
// margin) but adds latency. Models whose probability covers a long window
// (e.g. turn-end) emit one probability per slice, over the audio ending at the
// slice boundary. Chunked streaming models (e.g. diarization with a speaker
// cache) process a chunk of slice samples plus lookahead samples per step and
// emit the chunk's frames; with lookahead < 0 they use the model's recommended
// lookahead for the slice, and vad_detect uses the model's offline configuration
// (identical to a stream with that configuration except for the last frames).
struct vad_stream_params {
    int slice_samples;       // 0 = model default; ignored by causal per-hop models
    int lookahead_samples;   // < 0 = maximum (exact) / chunked models: recommended; ignored by causal models
};

VAD_API struct vad_stream_params vad_stream_default_params(void);

// NULL (with a message on stderr) for invalid params, e.g. lookahead beyond the maximum.
VAD_API vad_stream * vad_stream_new_with_params(vad_model * model, struct vad_stream_params params);
VAD_API vad_stream * vad_stream_new  (vad_model * model);   // default params
VAD_API void         vad_stream_free (vad_stream * stream);
VAD_API void         vad_stream_reset(vad_stream * stream);

// Feeds any number of samples (mono float, model sample rate). Writes the
// probabilities that became available (at most max_probs) and returns their
// count, or -1 on error. Use vad_stream_n_pending() to size probs.
VAD_API int vad_stream_push(vad_stream * stream, const float * samples, int n_samples, float * probs, int max_probs);
// Number of probabilities vad_stream_push would produce for n_samples more samples.
VAD_API int vad_stream_n_pending(const vad_stream * stream, int n_samples);

// End of audio: emits what is still pending (a partial hop zero padded, the
// remaining probabilities with the audio end as their future, or the
// probability for the audio ending now). Returns the count or -1.
VAD_API int vad_stream_flush(vad_stream * stream, float * probs, int max_probs);
VAD_API int vad_stream_n_flush(const vad_stream * stream);

// Samples pushed but not yet reflected in any probability.
VAD_API int vad_stream_n_buffered(const vad_stream * stream);

// vad_stream_push for several streams of one model at once: work that can be
// batched (e.g. many streams advancing by one hop, or several windows) runs in
// one compute. probs[i] receives up to max_probs[i] values for streams[i] and
// n_out[i] their count. Returns 0 on success.
VAD_API int vad_streams_push(vad_stream ** streams, int n_streams, const float * const * samples, const int * n_samples,
                             float * const * probs, const int * max_probs, int * n_out);

// ------------------------------------------------------------------ offline

// Probabilities for a whole buffer: a fresh stream, push, flush. Writes at most
// max_probs and returns the total count (which may exceed max_probs), or -1 on
// error. n_samples / vad_model_hop_samples + 1 values (with default params)
// always suffice.
VAD_API int vad_detect(vad_model * model, const float * samples, int n_samples, float * probs, int max_probs);
VAD_API int vad_detect_with_params(vad_model * model, struct vad_stream_params params, const float * samples, int n_samples,
                                   float * probs, int max_probs);

// ------------------------------------------------------------------ segments

struct vad_segment_params {
    float threshold;             // speech if prob >= threshold (default 0.5)
    float neg_threshold;         // silence if prob < neg_threshold; < 0 = max(threshold - 0.15, 0.01)
    int   min_speech_ms;         // drop shorter speech (default 250)
    int   min_silence_ms;        // silence needed to end speech (default 100)
    int   speech_pad_ms;         // padding added to each side (default 30)
    float max_speech_s;          // split longer speech (default: no limit, <= 0)
};

struct vad_segment {
    int64_t start;               // in samples
    int64_t end;
    int     channel;             // output channel (speaker for VAD_OUTPUT_SPEAKERS, else 0)
};

VAD_API struct vad_segment_params vad_segment_default_params(void);

// Segments from VAD_OUTPUT_SPEECH or VAD_OUTPUT_SPEAKERS frames (n_frames frames
// of n_outputs values, one frame per hop), each channel segmented like silero-vad's
// get_speech_timestamps_from_probs (v6.2.x). Sorted by start, then channel.
// n_samples is the audio length (<= 0: n_frames * hop). Writes at most
// max_segments and returns the total number of segments found.
VAD_API int vad_segments_from_probs(const float * probs, int n_frames, int n_outputs, int hop_samples, int sample_rate,
                                    int64_t n_samples, struct vad_segment_params params, struct vad_segment * segments,
                                    int max_segments);

#ifdef __cplusplus
}
#endif
