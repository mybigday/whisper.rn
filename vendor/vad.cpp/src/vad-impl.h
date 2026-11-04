// Internal interfaces shared by the loaders, the model architectures and the runtime.
#pragma once

#include "ggml.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vad {

// ------------------------------------------------------------------ model files

// A tensor as stored in the model file. ne follows ggml order (ne[0] fastest),
// so the layout equals a row-major PyTorch tensor with the dimensions reversed.
// Data is kept in the stored type: the model runs at the file's precision.
struct host_tensor {
    std::vector<int64_t> ne;
    ggml_type type = GGML_TYPE_F32;
    std::vector<uint8_t> data;   // raw bytes of `type`
    int64_t nelements() const { int64_t n = 1; for (auto v : ne) n *= v; return n; }
    bool is_float() const { return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16; }
    std::vector<float> to_f32() const;   // any type ggml can convert (float or quantized)
    static host_tensor from_f32(std::vector<int64_t> ne, const std::vector<float> & v);
};

struct model_file {
    std::string arch;                               // "silero", ...
    std::map<std::string, std::string> kv_str;      // metadata: "<arch>." stripped; general.* string keys kept as is
    std::map<std::string, int64_t>     kv_int;        // integers and booleans
    std::map<std::string, double>      kv_float;
    std::map<std::string, host_tensor> tensors;     // graph-ready names (see docs/gguf.md)
};

bool load_gguf       (const std::string & path, model_file & out, std::string & err);
bool load_whisper_bin(const std::string & path, model_file & out, std::string & err);   // whisper.cpp ggml-silero-*.bin
bool silero_to_graph_layout(model_file & m, std::string & err);                        // PyTorch names -> graph-ready
// Conv1d weight (Cout, Cin, K) -> (Cout, K, Cin), the order of the graphs' [K * Cin]
// window view; float types only (element-wise, exact). false if not float or the size differs.
bool conv_to_window_layout(const host_tensor & w, int64_t cout, int64_t cin, int64_t k, host_tensor & out);

// ------------------------------------------------------------------ architectures

// Weights created by an architecture with their bytes, uploaded by the runtime.
struct weight_uploads {
    std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> items;
};

// Creates model weights from the file at the file's precision, with checks:
//  - mat():   used only as a mul_mat weight; any type the backends support
//  - f32():   used by element-wise ops / convs (biases, norms, conv taps): must be
//             a float type in the file, converted to F32 (no precision lost for F32;
//             F16/BF16 are widened exactly)
//  - constant(): values computed at load time (F32)
// The first problem is reported through err and the call returns nullptr.
class weight_builder {
public:
    weight_builder(ggml_context * ctx, weight_uploads & up, model_file & f, std::string & err)
        : ctx_(ctx), up_(up), f_(f), err_(err) {}
    ggml_tensor * mat(const std::string & name, int64_t ne0, int64_t ne1);
    ggml_tensor * f32(const std::string & name, std::vector<int64_t> ne);
    ggml_tensor * constant(const std::string & name, std::vector<int64_t> ne, const std::vector<float> & v);
    bool ok() const { return err_.empty(); }
private:
    const host_tensor * find(const std::string & name, int64_t n);
    ggml_tensor * make(const std::string & name, ggml_type type, const std::vector<int64_t> & ne);
    ggml_context * ctx_;
    weight_uploads & up_;
    model_file & f_;
    std::string & err_;
};

// Tensors of one built graph. B = streams (columns), T = consecutive chunks per stream.
struct graph_io {
    ggml_tensor * audio      = nullptr;   // input  [audio_samples(T), B]
    ggml_tensor * probs      = nullptr;   // output [T, B]
    ggml_tensor * frame_mask = nullptr;   // optional input [T, B]: 1 for real frames, 0 for padding
};

// How the runtime will execute a graph; lets architectures pick graph forms.
struct graph_opts {
    bool host = true;   // CPU backend: strided views into mul_mat are fine; accelerators want contiguous inputs
    // ops whose inputs exceed F16 range or need exact F32 math (e.g. power spectra):
    // the runtime runs them on the CPU when the compute device is an accelerator
    std::vector<ggml_tensor *> * pin_cpu = nullptr;
};

// ------------------------------------------------------------------ host-driven models
//
// Models whose streaming algorithm needs host-side logic between graph runs
// (e.g. a speaker cache updated by top-k selection) implement a session and run
// graphs of their own through a graph_runner provided by the runtime.

// A graph built by an architecture: inputs are uploaded and outputs read back
// whole, in the order listed.
struct custom_graph {
    std::vector<ggml_tensor *> inputs, outputs;
};
using custom_builder = std::function<custom_graph(ggml_context *, ggml_cgraph *, const graph_opts &)>;

class graph_runner {
public:
    virtual ~graph_runner() = default;
    // Builds the graph for `key` on first use (at most max_nodes nodes), uploads
    // in[i] (ggml_nbytes of inputs[i]), computes, and reads outputs[i] into out[i].
    virtual bool run(const std::string & key, size_t max_nodes, const custom_builder & build,
                     const std::vector<const void *> & in, const std::vector<void *> & out) = 0;
};

// Streaming state of one audio stream of a host-driven model. Counts are output
// frames of n_outputs() floats each.
struct session {
    virtual ~session() = default;
    virtual int64_t n_pending(int64_t n_samples) const = 0;   // frames push() of n_samples more would emit
    virtual int64_t n_flush() const = 0;                      // frames flush() would emit
    virtual int64_t n_buffered() const = 0;                   // samples pushed but not yet reflected in any frame
    virtual bool push(graph_runner & g, const float * x, int64_t n, float * out) = 0;
    virtual bool flush(graph_runner & g, float * out) = 0;
    virtual void reset() = 0;
};

struct arch {
    virtual ~arch() = default;
    virtual const char * name() const = 0;
    virtual int sample_rate() const = 0;
    virtual int chunk_size() const = 0;
    virtual int state_size() const = 0;                 // floats of recurrent state per stream (zeros = initial)
    // Samples a fresh stream copies into state[0, warmup) before the first chunk
    // (models whose analysis window is longer than the hop). 0 = none.
    virtual int warmup_samples() const { return 0; }
    // turn detectors classify a whole window (chunk_size samples) instead of
    // producing per-chunk speech probabilities; they have no stream state
    virtual bool is_turn() const { return false; }
    // frame models that need future context (lookahead) cannot stream: they run
    // offline over a whole utterance, T frames at once, with frame_mask set
    virtual bool streaming() const { return true; }
    // offline-only frame models: receptive field in frames (past, future). A frame's
    // output is exact when the window holds this much real context on both sides.
    virtual int lookback_frames() const { return 0; }
    virtual int lookahead_frames() const { return 0; }
    // audio samples per stream fed to a graph of T chunks
    virtual int64_t audio_samples(int T) const { return (int64_t) chunk_size() * T; }
    // false (with err set) if a tensor is missing, has the wrong shape, or has a type its op cannot use
    virtual bool create_weights(weight_builder & wb) = 0;
    // state: persistent [state_size, B] tensor owned by the runtime (null when state_size == 0). The graph reads
    // the current state from it and writes the new state back in place (ggml_cpy
    // into views, expanded into gf); every write must depend on the reads of the
    // same region.
    virtual graph_io build_graph(ggml_context * ctx, ggml_cgraph * gf, int B, int T, const graph_opts & opts,
                                 ggml_tensor * state) = 0;
    virtual size_t max_nodes(int B, int T) const = 0;

    // values per output frame (e.g. one per speaker)
    virtual int n_outputs() const { return 1; }
    // diarization models: n_outputs() speaker activities ordered by first appearance
    virtual bool is_speakers() const { return false; }
    // Host-driven models (see session): a new stream state. slice and lookahead are
    // in samples (0 / < 0 = model default); offline = the model's whole-recording mode
    // (vad_detect). nullptr with err set for invalid parameters.
    virtual bool host_driven() const { return false; }
    virtual std::unique_ptr<session> new_session(int slice_samples, int lookahead_samples, bool offline,
                                                 std::string & err) const { (void) slice_samples; (void) lookahead_samples; (void) offline; err = "not a host-driven model"; return nullptr; }
    // Host-driven models: pushes x[i] (len[i] samples) to the sessions s[i] of this model, each writing its
    // n_pending frames to out[i]; models may batch the graph work across sessions. Default: one by one.
    virtual bool streams_push(graph_runner & g, session * const * s, int n, const float * const * x, const int64_t * len,
                              float * const * out) const {
        for (int i = 0; i < n; ++i) if (!s[i]->push(g, x[i], len[i], out[i])) return false;
        return true;
    }
};

std::unique_ptr<arch> make_silero    (const model_file & file, std::string & err);
std::unique_ptr<arch> make_firered   (const model_file & file, std::string & err);
std::unique_ptr<arch> make_smart_turn(const model_file & file, std::string & err);
std::unique_ptr<arch> make_sortformer (model_file & file, std::string & err);   // re-lays out its conv weight

} // namespace vad
