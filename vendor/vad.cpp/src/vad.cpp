#include "vad.h"
#include "vad-impl.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef VAD_WITH_CPU_KERNELS
#include "vad-cpu.h"
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

using namespace vad;

// ------------------------------------------------------------------ devices

namespace {

struct device_entry { std::string name, desc; ggml_backend_dev_t dev; };

const std::vector<device_entry> & devices() {
    static std::vector<device_entry> list = [] {
#ifdef GGML_BACKEND_DL
        ggml_backend_load_all();
#endif
        std::vector<device_entry> v;
#ifdef VAD_WITH_CPU_KERNELS
        v.push_back({ "cpu", "CPU (vad.cpp kernels)", nullptr });
#else
        v.push_back({ "cpu", "CPU (ggml kernels)", nullptr });
#endif
        v.push_back({ "ggml-cpu", "CPU (ggml kernels)", nullptr });
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) continue;
            v.push_back({ ggml_backend_dev_name(d), ggml_backend_dev_description(d), d });
        }
        return v;
    }();
    return list;
}

// whisper.rn: route errors through ggml's logger (whisper_log_set installs it),
// stderr is not visible on Android.
extern "C" void ggml_log_internal(enum ggml_log_level level, const char * format, ...);
void log_err(const std::string & msg) { ggml_log_internal(GGML_LOG_LEVEL_ERROR, "vad: %s\n", msg.c_str()); }

} // namespace

int vad_device_count(void) { return (int) devices().size(); }
const char * vad_device_name(int i) { return i >= 0 && i < vad_device_count() ? devices()[i].name.c_str() : nullptr; }
const char * vad_device_description(int i) { return i >= 0 && i < vad_device_count() ? devices()[i].desc.c_str() : nullptr; }

// ------------------------------------------------------------------ model

struct vad_stream;

namespace {

// One built graph for (B streams, T chunks) with its own persistent state
// columns. A stream's authoritative state is either on the host (vad_stream::state)
// or resident in one column of one graph instance (vad_stream::home); streams
// that keep using the same column need no state copies at all.
struct graph_instance {
    int B = 0, T = 0;
    std::vector<uint8_t> meta;
    ggml_context * ctx = nullptr, * sctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_backend_buffer_t sbuf = nullptr;
    ggml_tensor * state = nullptr;          // [state_size, B]
    graph_io io;
    std::vector<float> audio, probs, mask;
    std::vector<vad_stream *> resident;     // per column

    ~graph_instance() {
        if (sched) ggml_backend_sched_free(sched);
        if (ctx) ggml_free(ctx);
        if (sbuf) ggml_backend_buffer_free(sbuf);
        if (sctx) ggml_free(sctx);
    }
};

int pow2_ceil(int x) { int p = 1; while (p < x) p <<= 1; return p; }

// A graph built by a host-driven architecture (vad::custom_graph), cached by key.
// All of them share one scheduler, so compute buffers are reused between graphs.
struct custom_instance {
    std::vector<uint8_t> meta;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    custom_graph io;
    std::vector<ggml_tensor *> pin;
    ~custom_instance() { if (ctx) ggml_free(ctx); }
};

} // namespace

enum class stream_kind { NATIVE, WINDOW, TURN, HOST };

struct vad_stream {
    vad_model * model;
    stream_kind kind = stream_kind::NATIVE;
    // NATIVE
    std::vector<float> state;              // valid when home == nullptr
    std::vector<float> pending;            // < chunk_size buffered samples
    int warm = 0;                          // warmup samples already in state
    graph_instance * home = nullptr;       // graph holding the live state
    int col = -1;
    // WINDOW (frames) / TURN (samples)
    std::vector<float> hist;               // retained audio, hist[0] is absolute sample hist0
    int64_t hist0 = 0, total = 0;          // samples pushed
    int64_t emitted = 0;                   // WINDOW: frames emitted; TURN: slices emitted
    int slice = 0, lookahead = 0;          // WINDOW: frames; TURN: slice in samples
    // HOST
    std::unique_ptr<session> sess;
};

static bool VAD_DEBUG_ENABLED() { static const bool on = getenv("VAD_DEBUG") != nullptr; return on; }

struct vad_model {
    std::unique_ptr<arch> a;
    std::string device;
    ggml_backend_t backend = nullptr;   // compute backend
    ggml_backend_t cpu = nullptr;       // fallback for ops the accelerator lacks (null when backend is a CPU)
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    std::mutex mu;
    std::map<std::pair<int, int>, std::unique_ptr<graph_instance>> graphs;
    // host-driven models
    std::map<std::string, std::unique_ptr<custom_instance>> custom;
    ggml_backend_sched_t custom_sched = nullptr;
    size_t custom_sched_nodes = 0;

    ~vad_model() {
        graphs.clear();
        custom.clear();
        if (custom_sched) ggml_backend_sched_free(custom_sched);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
        if (backend) ggml_backend_free(backend);
        if (cpu) ggml_backend_free(cpu);
    }

    graph_instance * graph(int B, int T) {
        auto & slot = graphs[{B, T}];
        if (slot) return slot.get();
        auto g = std::make_unique<graph_instance>();
        g->B = B; g->T = T;
        const size_t n_max = a->max_nodes(B, T);
        g->meta.resize(ggml_tensor_overhead() * n_max * 2 + ggml_graph_overhead_custom(n_max, false));
        ggml_init_params ip = { g->meta.size(), g->meta.data(), true };
        g->ctx = ggml_init(ip);
        g->gf = ggml_new_graph_custom(g->ctx, n_max, false);
        // persistent state columns on the compute device
        g->resident.assign(B, nullptr);
        if (a->state_size() > 0) {
            ggml_init_params sp = { ggml_tensor_overhead() * 2, nullptr, true };
            g->sctx = ggml_init(sp);
            g->state = ggml_new_tensor_2d(g->sctx, GGML_TYPE_F32, a->state_size(), B);
            ggml_set_name(g->state, "state");
            g->sbuf = ggml_backend_alloc_ctx_tensors(g->sctx, backend);
            if (!g->sbuf) { log_err("failed to allocate state"); slot.reset(); return nullptr; }
            ggml_backend_buffer_clear(g->sbuf, 0);
        }

        graph_opts opts;
        opts.host = cpu == nullptr;
        std::vector<ggml_tensor *> pin;
        opts.pin_cpu = &pin;
        g->io = a->build_graph(g->ctx, g->gf, B, T, opts, g->state);
        ggml_backend_t bes[2] = { backend, cpu };
        g->sched = ggml_backend_sched_new(bes, nullptr, cpu ? 2 : 1, n_max, false, true);
        for (ggml_tensor * t : pin) ggml_backend_sched_set_tensor_backend(g->sched, t, cpu);
        if (!ggml_backend_sched_alloc_graph(g->sched, g->gf)) { log_err("failed to allocate compute graph"); slot.reset(); return nullptr; }
        if (getenv("VAD_DEBUG")) {
            fprintf(stderr, "vad: graph B=%d T=%d: %d nodes, %d splits, %d copies\n", B, T, ggml_graph_n_nodes(g->gf),
                    ggml_backend_sched_get_n_splits(g->sched), ggml_backend_sched_get_n_copies(g->sched));
            for (int i = 0; i < ggml_graph_n_nodes(g->gf); ++i) {
                ggml_tensor * t = ggml_graph_node(g->gf, i);
                ggml_backend_t b = ggml_backend_sched_get_tensor_backend(g->sched, t);
                if (b != backend) fprintf(stderr, "  node %d %s (%s) on %s\n", i, ggml_op_desc(t), t->name, b ? ggml_backend_name(b) : "?");
            }
        }
        g->audio.assign(ggml_nelements(g->io.audio), 0.0f);
        g->probs.assign(ggml_nelements(g->io.probs), 0.0f);
        slot = std::move(g);
        return slot.get();
    }

    size_t state_bytes() const { return (size_t) a->state_size() * sizeof(float); }

    // graph_runner::run for host-driven models. Caller holds mu. The graph is
    // rebuilt in its cached metadata buffer on every run: the shared scheduler may
    // reallocate its buffers between graphs, so tensors of an earlier allocation
    // must not be reused (vad-cpu keeps packed weights per weight tensor, and
    // the rebuilt nodes land at the same addresses, so its plan is reused too).
    bool run_custom(const std::string & key, size_t max_nodes, const custom_builder & build,
                    const std::vector<const void *> & in, const std::vector<void *> & out) {
        auto & slot = custom[key];
        if (!slot) {
            slot = std::make_unique<custom_instance>();
            slot->meta.resize(ggml_tensor_overhead() * max_nodes * 2 + ggml_graph_overhead_custom(max_nodes, false));
        }
        custom_instance * g = slot.get();
        if (g->ctx) ggml_free(g->ctx);
        ggml_init_params ip = { g->meta.size(), g->meta.data(), true };
        g->ctx = ggml_init(ip);
        g->gf = ggml_new_graph_custom(g->ctx, max_nodes, false);
        g->pin.clear();
        graph_opts opts;
        opts.host = cpu == nullptr;
        opts.pin_cpu = &g->pin;
        g->io = build(g->ctx, g->gf, opts);
        if (g->io.inputs.size() != in.size() || g->io.outputs.size() != out.size()) { log_err("custom graph: wrong number of buffers"); return false; }
        const size_t nodes = std::max<size_t>(max_nodes, 8192);
        if (!custom_sched || custom_sched_nodes < nodes) {
            if (custom_sched) ggml_backend_sched_free(custom_sched);
            ggml_backend_t bes[2] = { backend, cpu };
            custom_sched = ggml_backend_sched_new(bes, nullptr, cpu ? 2 : 1, nodes, false, true);
            custom_sched_nodes = nodes;
        }
        ggml_backend_sched_reset(custom_sched);
        for (ggml_tensor * t : g->pin) ggml_backend_sched_set_tensor_backend(custom_sched, t, cpu);
        if (!ggml_backend_sched_alloc_graph(custom_sched, g->gf)) { log_err("failed to allocate compute graph " + key); return false; }
        for (size_t i = 0; i < in.size(); ++i) ggml_backend_tensor_set(g->io.inputs[i], in[i], 0, ggml_nbytes(g->io.inputs[i]));
        if (ggml_backend_sched_graph_compute(custom_sched, g->gf) != GGML_STATUS_SUCCESS) { log_err("compute failed"); return false; }
        for (size_t i = 0; i < out.size(); ++i) ggml_backend_tensor_get(g->io.outputs[i], out[i], 0, ggml_nbytes(g->io.outputs[i]));
        return true;
    }

    // brings a stream's state back to the host copy
    void evict(vad_stream * s) {
        if (!s->home) return;
        ggml_backend_tensor_get(s->home->state, s->state.data(), s->col * state_bytes(), state_bytes());
        s->home->resident[s->col] = nullptr;
        s->home = nullptr;
        s->col = -1;
    }

    // Runs n streams (n <= B) for T chunks each. audio[i] points at T*chunk
    // samples, probs[i] receives T values. Caller holds mu.
    bool compute(int n, int T, vad_stream * const * streams, const float * const * audio, float * const * probs,
                 const int * n_valid = nullptr) {
        const int B = pow2_ceil(n);
        graph_instance * g = graph(B, T);
        if (!g) return false;
        // place stream i in column i (no copy when it is already there)
        for (int i = 0; i < n && g->state; ++i) {
            vad_stream * s = streams[i];
            if (s->home == g && s->col == i) continue;
            evict(s);
            if (g->resident[i]) evict(g->resident[i]);
            ggml_backend_tensor_set(g->state, s->state.data(), i * state_bytes(), state_bytes());
            g->resident[i] = s; s->home = g; s->col = i;
        }
        // padding columns compute garbage: they must not hold a live stream
        for (int i = n; i < B && g->state; ++i) if (g->resident[i]) evict(g->resident[i]);

        const size_t na = (size_t) a->audio_samples(T);
        for (int i = 0; i < n; ++i) std::memcpy(g->audio.data() + i * na, audio[i], na * sizeof(float));
        if (n < B) std::fill(g->audio.begin() + n * na, g->audio.end(), 0.0f);
        ggml_backend_tensor_set(g->io.audio, g->audio.data(), 0, ggml_nbytes(g->io.audio));
        if (g->io.frame_mask) {
            g->mask.assign((size_t) T * B, 0.0f);
            for (int i = 0; i < n; ++i)
                for (int t = 0; t < (n_valid ? std::min(n_valid[i], T) : T); ++t) g->mask[(size_t) i * T + t] = 1.0f;
            ggml_backend_tensor_set(g->io.frame_mask, g->mask.data(), 0, ggml_nbytes(g->io.frame_mask));
        }
        if (ggml_backend_sched_graph_compute(g->sched, g->gf) != GGML_STATUS_SUCCESS) { log_err("compute failed"); return false; }
        ggml_backend_tensor_get(g->io.probs, g->probs.data(), 0, (size_t) n * T * sizeof(float));
        for (int i = 0; i < n; ++i) std::memcpy(probs[i], g->probs.data() + (size_t) i * T, T * sizeof(float));
        return true;
    }
};


vad_model_params vad_model_default_params(void) {
    vad_model_params p;
    p.device = nullptr;
    p.n_threads = 1;
    return p;
}

static bool read_magic(const char * path, uint32_t & magic) {
    std::ifstream f(path, std::ios::binary);
    return f && f.read((char *) &magic, 4);
}

vad_model * vad_model_load(const char * path, vad_model_params params) {
    uint32_t magic = 0;
    if (!read_magic(path, magic)) { log_err(std::string("cannot open ") + path); return nullptr; }
    model_file mf;
    std::string err;
    bool ok = false;
    if (magic == 0x46554747u) ok = load_gguf(path, mf, err);              // "GGUF"
    else if (magic == 0x67676d6cu) ok = load_whisper_bin(path, mf, err);  // whisper.cpp "ggml"
    else err = "unknown file format";
    if (!ok) { log_err(err); return nullptr; }

    auto m = std::make_unique<vad_model>();
    if (mf.arch == "silero") m->a = make_silero(mf, err);
    else if (mf.arch == "firered") m->a = make_firered(mf, err);
    else if (mf.arch == "smart_turn") m->a = make_smart_turn(mf, err);
    else if (mf.arch == "sortformer") m->a = make_sortformer(mf, err);
    else err = "unsupported architecture: " + mf.arch;
    if (!m->a) { log_err(err); return nullptr; }

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    const int n_threads = 1;   // WASM without pthreads (build with VAD_WASM_THREADS for threads)
#else
    const int n_threads = params.n_threads > 0 ? params.n_threads : 1;
#endif
    std::string dev = params.device && *params.device && std::string(params.device) != "auto" ? params.device : "cpu";
    if (dev == "cpu") {
#ifdef VAD_WITH_CPU_KERNELS
        m->backend = ggml_backend_vad_cpu_init(n_threads);
#else
        m->backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(m->backend, n_threads);
#endif
    } else if (dev == "ggml-cpu") {
        m->backend = ggml_backend_cpu_init();
        if (m->backend) ggml_backend_cpu_set_n_threads(m->backend, n_threads);
    } else {
        for (auto & d : devices())
            if (d.dev && d.name == dev) m->backend = ggml_backend_dev_init(d.dev, nullptr);
        if (!m->backend) { log_err("unknown device " + dev); return nullptr; }
        m->cpu = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(m->cpu, n_threads);
    }
    if (!m->backend) { log_err("failed to initialize device " + dev); return nullptr; }
    m->device = dev;

    weight_uploads up;
    ggml_init_params ip = { ggml_tensor_overhead() * 2048, nullptr, true };
    m->wctx = ggml_init(ip);
    {
        weight_builder wb(m->wctx, up, mf, err);
        if (!m->a->create_weights(wb)) { log_err(err); return nullptr; }
    }
    m->wbuf = ggml_backend_alloc_ctx_tensors(m->wctx, m->backend);
    if (!m->wbuf) { log_err("failed to allocate weights"); return nullptr; }
    ggml_backend_buffer_set_usage(m->wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (auto & [t, d] : up.items) {
        GGML_ASSERT(d.size() == ggml_nbytes(t));
        ggml_backend_tensor_set(t, d.data(), 0, d.size());
    }
    return m.release();
}

void vad_model_free(vad_model * model) { delete model; }
const char * vad_model_arch(const vad_model * m) { return m->a->name(); }
vad_output vad_model_output(const vad_model * m) {
    return m->a->is_speakers() ? VAD_OUTPUT_SPEAKERS : m->a->is_turn() ? VAD_OUTPUT_TURN_END : VAD_OUTPUT_SPEECH;
}
int vad_model_n_outputs(const vad_model * m) { return m->a->n_outputs(); }
const char * vad_model_device(const vad_model * m) { return m->device.c_str(); }
int vad_model_sample_rate(const vad_model * m) { return m->a->sample_rate(); }
static constexpr int DEFAULT_SLICE = 8000;   // window/turn streams: 0.5 s at 16 kHz
int vad_model_hop_samples(const vad_model * m) { return m->a->is_turn() ? DEFAULT_SLICE : m->a->chunk_size(); }
int vad_model_warmup_samples(const vad_model * m) { return m->a->warmup_samples(); }

// ------------------------------------------------------------------ streams

vad_stream_params vad_stream_default_params(void) {
    vad_stream_params p;
    p.slice_samples = 0;
    p.lookahead_samples = -1;
    return p;
}

int vad_model_max_lookahead_samples(const vad_model * m) {
    return m->a->is_turn() ? 0 : m->a->lookahead_frames() * m->a->chunk_size();
}

namespace {

// graph_runner over a model's custom graphs (takes the model lock per run)
struct model_runner : graph_runner {
    vad_model * m;
    explicit model_runner(vad_model * model) : m(model) {}
    bool run(const std::string & key, size_t max_nodes, const custom_builder & build,
             const std::vector<const void *> & in, const std::vector<void *> & out) override {
        std::lock_guard<std::mutex> lock(m->mu);
        return m->run_custom(key, max_nodes, build, in, out);
    }
};

vad_stream * stream_new(vad_model * model, vad_stream_params p, bool offline) {
    auto * s = new vad_stream();
    s->model = model;
    const int cs = model->a->chunk_size();
    if (model->a->host_driven()) {
        s->kind = stream_kind::HOST;
        std::string err;
        s->sess = model->a->new_session(p.slice_samples, p.lookahead_samples, offline, err);
        if (!s->sess) { log_err(err); delete s; return nullptr; }
    } else if (model->a->is_turn()) {
        s->kind = stream_kind::TURN;
        s->slice = p.slice_samples > 0 ? p.slice_samples : DEFAULT_SLICE;
    } else if (!model->a->streaming()) {
        s->kind = stream_kind::WINDOW;
        const int max_la = model->a->lookahead_frames();
        s->lookahead = p.lookahead_samples < 0 ? max_la : (p.lookahead_samples + cs - 1) / cs;
        if (s->lookahead > max_la) {
            log_err("lookahead_samples exceeds the model maximum of " + std::to_string(max_la * cs));
            delete s;
            return nullptr;
        }
        s->slice = std::max(1, (p.slice_samples > 0 ? p.slice_samples : DEFAULT_SLICE) / cs);
    } else {
        s->state.assign(model->a->state_size(), 0.0f);
    }
    return s;
}

} // namespace

vad_stream * vad_stream_new_with_params(vad_model * model, vad_stream_params p) { return stream_new(model, p, false); }

vad_stream * vad_stream_new(vad_model * model) { return vad_stream_new_with_params(model, vad_stream_default_params()); }

void vad_stream_free(vad_stream * s) {
    if (!s) return;
    {
        std::lock_guard<std::mutex> lock(s->model->mu);
        if (s->home) { s->home->resident[s->col] = nullptr; s->home = nullptr; }
    }
    delete s;
}

void vad_stream_reset(vad_stream * s) {
    std::lock_guard<std::mutex> lock(s->model->mu);
    if (s->home) { s->home->resident[s->col] = nullptr; s->home = nullptr; }
    std::fill(s->state.begin(), s->state.end(), 0.0f);
    s->pending.clear();
    s->warm = 0;
    s->hist.clear();
    s->hist0 = s->total = s->emitted = 0;
    if (s->sess) s->sess->reset();
}

// ---- WINDOW streams (offline-only frame models)

// complete frames in n samples (kaldi snip_edges)
static int64_t frames_in(const vad_stream * s, int64_t n) {
    const int cs = s->model->a->chunk_size(), wu = s->model->a->warmup_samples();
    return n >= wu + cs ? 1 + (n - wu - cs) / cs : 0;
}

static int64_t window_ready(const vad_stream * s, int64_t total, bool flush) {
    const int64_t avail = frames_in(s, total);
    const int64_t ready = avail - (flush ? 0 : s->lookahead) - s->emitted;
    if (ready <= 0) return 0;
    return flush || ready >= s->slice ? ready : 0;
}

// emits `ready` frames: one offline run over [emitted - lookback, emitted + ready + lookahead)
// (clipped to the audio), so emitted frames see their full past and the chosen future
struct window_job {
    vad_stream * s = nullptr;
    int64_t ready = 0, f0 = 0;
    int nf = 0, T = 0;
    std::vector<float> audio, out;
    float * probs = nullptr;
};

static void window_prepare(vad_stream * s, int64_t ready, float * probs, window_job & j) {
    vad_model * m = s->model;
    const int cs = m->a->chunk_size();
    const int64_t avail = frames_in(s, s->total);
    j.s = s; j.ready = ready; j.probs = probs;
    j.f0 = std::max<int64_t>(0, s->emitted - m->a->lookback_frames());
    const int64_t f1 = std::min<int64_t>(avail, s->emitted + ready + m->a->lookahead_frames());
    j.nf = (int) (f1 - j.f0);
    j.T = (j.nf + 63) / 64 * 64;
    j.audio.assign(m->a->audio_samples(j.T), 0.0f);
    const int64_t s0 = j.f0 * cs;
    const int64_t n = std::min<int64_t>((int64_t) j.audio.size(), s->total - s0);
    std::copy(s->hist.begin() + (s0 - s->hist0), s->hist.begin() + (s0 - s->hist0 + n), j.audio.begin());
    j.out.assign(j.T, 0.0f);
}

static void window_finish(window_job & j) {
    vad_stream * s = j.s;
    vad_model * m = s->model;
    std::copy(j.out.begin() + (s->emitted - j.f0), j.out.begin() + (s->emitted - j.f0 + j.ready), j.probs);
    s->emitted += j.ready;
    // keep only what later windows can need
    const int64_t keep = std::max<int64_t>(0, s->emitted - m->a->lookback_frames()) * m->a->chunk_size();
    if (keep > s->hist0) {
        s->hist.erase(s->hist.begin(), s->hist.begin() + (keep - s->hist0));
        s->hist0 = keep;
    }
}

// windows of the same length (T) run as batched computes; padding frames are masked
static bool window_run_all(vad_model * m, std::vector<window_job> & jobs) {
    std::vector<size_t> order(jobs.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) { return jobs[x].T < jobs[y].T; });
    for (size_t g0 = 0; g0 < order.size();) {
        size_t g1 = g0;
        // at most 8 windows per compute: larger batches stop fitting in cache (FireRed VAD, 64 streams:
        // 1 thread 69 -> 67 us at 8, 88 us at 64; 8 threads 16.4 -> 11.8 us at 8)
        constexpr size_t MAX_BATCH = 8;
        while (g1 < order.size() && g1 - g0 < MAX_BATCH && jobs[order[g1]].T == jobs[order[g0]].T) ++g1;
        std::vector<const float *> a;
        std::vector<float *> p;
        std::vector<int> nv;
        for (size_t k = g0; k < g1; ++k) {
            window_job & j = jobs[order[k]];
            a.push_back(j.audio.data()); p.push_back(j.out.data()); nv.push_back(j.nf);
        }
        {
            std::lock_guard<std::mutex> lock(m->mu);
            if (!m->compute((int) a.size(), jobs[order[g0]].T, nullptr, a.data(), p.data(), nv.data())) return false;
        }
        g0 = g1;
    }
    for (window_job & j : jobs) window_finish(j);
    return true;
}

static bool window_run(vad_stream * s, int64_t ready, float * probs) {
    std::vector<window_job> jobs(1);
    window_prepare(s, ready, probs, jobs[0]);
    return window_run_all(s->model, jobs);
}

// ---- TURN streams: one probability per slice, window ending at the slice boundary

// one batched compute over n windows: the last W samples of each input, zero padded at the start
static bool turn_windows(vad_model * model, const float * const * samples, const int * n_samples, int n, float * probs) {
    const int W = model->a->chunk_size();
    std::vector<std::vector<float>> win(n, std::vector<float>(W, 0.0f));
    std::vector<const float *> ptr(n);
    std::vector<float *> out(n);
    for (int i = 0; i < n; ++i) {
        const int k = std::min(W, n_samples[i]);
        std::copy(samples[i] + n_samples[i] - k, samples[i] + n_samples[i], win[i].begin() + (W - k));
        ptr[i] = win[i].data();
        out[i] = probs + i;
    }
    std::lock_guard<std::mutex> lock(model->mu);
    return model->compute(n, 1, nullptr, ptr.data(), out.data());
}

static int turn_window(vad_stream * s, int64_t end, std::vector<float> & w) {
    const int W = s->model->a->chunk_size();
    const int64_t b0 = std::max<int64_t>(s->hist0, end - W);
    w.assign(s->hist.begin() + (b0 - s->hist0), s->hist.begin() + (end - s->hist0));
    return (int) w.size();
}

static bool turn_predict(vad_stream * s, const std::vector<int64_t> & ends, float * probs) {
    std::vector<std::vector<float>> wins(ends.size());
    std::vector<const float *> ptr;
    std::vector<int> len;
    for (size_t i = 0; i < ends.size(); ++i) {
        len.push_back(turn_window(s, ends[i], wins[i]));
        ptr.push_back(wins[i].data());
    }
    return ends.empty() || turn_windows(s->model, ptr.data(), len.data(), (int) ends.size(), probs);
}

int vad_stream_n_pending(const vad_stream * s, int n_samples) {
    switch (s->kind) {
        case stream_kind::WINDOW: return (int) window_ready(s, s->total + n_samples, false);
        case stream_kind::TURN:   return (int) ((s->total + n_samples) / s->slice - s->emitted);
        case stream_kind::HOST:   return (int) s->sess->n_pending(n_samples);
        case stream_kind::NATIVE: break;
    }
    const int need = s->model->a->warmup_samples() - s->warm;
    const int n = std::max(0, n_samples - need);
    return (int) ((s->pending.size() + n) / s->model->a->chunk_size());
}

int vad_stream_n_buffered(const vad_stream * s) {
    switch (s->kind) {
        case stream_kind::WINDOW: return (int) (s->total - std::min<int64_t>(s->total, s->emitted * s->model->a->chunk_size()));
        case stream_kind::TURN:   return (int) (s->total - s->emitted * s->slice);
        case stream_kind::HOST:   return (int) s->sess->n_buffered();
        case stream_kind::NATIVE: break;
    }
    return (int) s->pending.size();
}

int vad_stream_n_flush(const vad_stream * s) {
    switch (s->kind) {
        case stream_kind::WINDOW: return (int) window_ready(s, s->total, true);
        case stream_kind::TURN:   return s->total > 0 ? 1 : 0;
        case stream_kind::HOST:   return (int) s->sess->n_flush();
        case stream_kind::NATIVE: break;
    }
    return !s->pending.empty() && s->warm == s->model->a->warmup_samples() ? 1 : 0;
}

// Runs n_chunks consecutive chunks of one stream, in blocks of T from a fixed
// set so only a few graphs are ever built.
static bool run_chunks(vad_stream * s, const float * x, int n_chunks, float * probs) {
    static const int blocks[] = { 64, 16, 4, 1 };
    const int cs = s->model->a->chunk_size();
    std::lock_guard<std::mutex> lock(s->model->mu);
    int done = 0;
    while (done < n_chunks) {
        int T = 1;
        for (int b : blocks) if (n_chunks - done >= b) { T = b; break; }
        const float * a = x + (size_t) done * cs;
        float * p = probs + done;
        if (!s->model->compute(1, T, &s, &a, &p)) return false;
        done += T;
    }
    return true;
}

int vad_stream_push(vad_stream * s, const float * samples, int n_samples, float * probs, int max_probs) {
    if (n_samples < 0) return -1;
    const int n_out = vad_stream_n_pending(s, n_samples);
    if (n_out > max_probs) return -1;
    if (s->kind == stream_kind::HOST) {
        model_runner r(s->model);
        return s->sess->push(r, samples, n_samples, probs) ? n_out : -1;
    }
    if (s->kind != stream_kind::NATIVE) {
        s->hist.insert(s->hist.end(), samples, samples + n_samples);
        const int64_t old_total = s->total;
        s->total += n_samples;
        if (s->kind == stream_kind::WINDOW) {
            const int64_t ready = window_ready(s, s->total, false);
            if (ready > 0 && !window_run(s, ready, probs)) return -1;
            return (int) ready;
        }
        // TURN
        std::vector<int64_t> ends;
        for (int64_t b = (old_total / s->slice + 1) * s->slice; b <= s->total; b += s->slice) ends.push_back(b);
        if (!turn_predict(s, ends, probs)) return -1;
        s->emitted += (int64_t) ends.size();
        const int64_t keep = std::max<int64_t>(0, s->total - s->model->a->chunk_size());
        if (keep > s->hist0) { s->hist.erase(s->hist.begin(), s->hist.begin() + (keep - s->hist0)); s->hist0 = keep; }
        (void) old_total;
        return (int) ends.size();
    }
    const int cs = s->model->a->chunk_size();
    // warmup: the first samples only fill the analysis window kept in state[0, warmup)
    const int need = s->model->a->warmup_samples() - s->warm;
    if (need > 0) {
        const int k = std::min(need, n_samples);
        {
            std::lock_guard<std::mutex> lock(s->model->mu);
            s->model->evict(s);   // (a fresh stream is never resident; kept for safety)
        }
        std::copy(samples, samples + k, s->state.begin() + s->warm);
        s->warm += k;
        samples += k;
        n_samples -= k;
    }
    int used = 0;
    int produced = 0;
    if (!s->pending.empty() && n_out > 0) {
        used = cs - (int) s->pending.size();
        s->pending.insert(s->pending.end(), samples, samples + used);
        if (!run_chunks(s, s->pending.data(), 1, probs)) return -1;
        s->pending.clear();
        produced = 1;
    }
    const int n_direct = (n_samples - used) / cs;
    if (n_direct > 0) {
        if (!run_chunks(s, samples + used, n_direct, probs + produced)) return -1;
        produced += n_direct;
        used += n_direct * cs;
    }
    s->pending.insert(s->pending.end(), samples + used, samples + n_samples);
    return produced;
}

int vad_stream_flush(vad_stream * s, float * probs, int max_probs) {
    const int n_out = vad_stream_n_flush(s);
    if (n_out > max_probs) return -1;
    if (n_out == 0) return 0;
    switch (s->kind) {
        case stream_kind::WINDOW:
            return window_run(s, n_out, probs) ? n_out : -1;
        case stream_kind::TURN:
            return turn_predict(s, { s->total }, probs) ? 1 : -1;
        case stream_kind::HOST: {
            model_runner r(s->model);
            return s->sess->flush(r, probs) ? n_out : -1;
        }
        case stream_kind::NATIVE: {
            std::vector<float> tail(s->model->a->chunk_size(), 0.0f);
            std::copy(s->pending.begin(), s->pending.end(), tail.begin());
            s->pending.clear();
            return run_chunks(s, tail.data(), 1, probs) ? 1 : -1;
        }
    }
    return -1;
}

int vad_streams_push(vad_stream ** streams, int n, const float * const * samples, const int * n_samples,
                     float * const * probs, const int * max_probs, int * n_out) {
    if (n <= 0) return 0;
    vad_model * m = streams[0]->model;
    for (int i = 0; i < n; ++i) {
        if (streams[i]->model != m || n_samples[i] < 0) return -1;
        for (int j = 0; j < i; ++j) if (streams[j] == streams[i]) return -1;
        n_out[i] = 0;
    }
    const int cs = m->a->chunk_size();
    std::vector<int> rest;
    // causal per-hop streams advancing by exactly one hop: one batched compute
    std::vector<vad_stream *> bs;
    std::vector<const float *> ba;
    std::vector<float *> bp;
    std::vector<int> bi;
    for (int i = 0; i < n; ++i) {
        vad_stream * s = streams[i];
        if (s->kind == stream_kind::NATIVE && s->pending.empty() && s->warm == m->a->warmup_samples() &&
            n_samples[i] == cs && max_probs[i] >= 1) {
            bs.push_back(s); ba.push_back(samples[i]); bp.push_back(probs[i]); bi.push_back(i);
        } else {
            rest.push_back(i);
        }
    }
    if (bs.size() > 1) {
        std::lock_guard<std::mutex> lock(m->mu);
        if (!m->compute((int) bs.size(), 1, bs.data(), ba.data(), bp.data())) return -1;
        for (int i : bi) n_out[i] = 1;
    } else {
        rest.insert(rest.end(), bi.begin(), bi.end());
    }
    // turn streams: all windows of all streams in one batched compute
    std::vector<int> turn;
    std::vector<int> other;
    for (int i : rest) (streams[i]->kind == stream_kind::TURN ? turn : other).push_back(i);
    if (!turn.empty()) {
        std::vector<std::vector<float>> wins;
        std::vector<std::pair<int, int>> owner;   // (stream index, output slot)
        for (int i : turn) {
            vad_stream * s = streams[i];
            if (vad_stream_n_pending(s, n_samples[i]) > max_probs[i]) return -1;
            s->hist.insert(s->hist.end(), samples[i], samples[i] + n_samples[i]);
            const int64_t old_total = s->total;
            s->total += n_samples[i];
            int k = 0;
            for (int64_t b = (old_total / s->slice + 1) * s->slice; b <= s->total; b += s->slice) {
                wins.emplace_back();
                turn_window(s, b, wins.back());
                owner.emplace_back(i, k++);
            }
        }
        std::vector<const float *> ptr;
        std::vector<int> len;
        for (auto & w : wins) { ptr.push_back(w.data()); len.push_back((int) w.size()); }
        std::vector<float> out(wins.size());
        if (!wins.empty() && !turn_windows(m, ptr.data(), len.data(), (int) wins.size(), out.data())) return -1;
        for (size_t w = 0; w < wins.size(); ++w) probs[owner[w].first][owner[w].second] = out[w];
        for (int i : turn) {
            vad_stream * s = streams[i];
            const int64_t produced = s->total / s->slice - s->emitted;
            n_out[i] = (int) produced;
            s->emitted += produced;
            const int64_t keep = std::max<int64_t>(0, s->total - cs);
            if (keep > s->hist0) { s->hist.erase(s->hist.begin(), s->hist.begin() + (keep - s->hist0)); s->hist0 = keep; }
        }
    }
    // sliding-window streams: the windows of all streams, batched by length
    std::vector<window_job> jobs;
    for (int i : other) {
        vad_stream * s = streams[i];
        if (s->kind != stream_kind::WINDOW) continue;
        if (vad_stream_n_pending(s, n_samples[i]) > max_probs[i]) return -1;
        s->hist.insert(s->hist.end(), samples[i], samples[i] + n_samples[i]);
        s->total += n_samples[i];
        const int64_t ready = window_ready(s, s->total, false);
        n_out[i] = (int) ready;
        if (ready > 0) { jobs.emplace_back(); window_prepare(s, ready, probs[i], jobs.back()); }
    }
    if (!jobs.empty() && !window_run_all(m, jobs)) return -1;
    // host-driven streams: the model may batch their graph work across streams
    {
        std::vector<session *> hs;
        std::vector<const float *> hx;
        std::vector<int64_t> hl;
        std::vector<float *> ho;
        for (int i : other) {
            vad_stream * s = streams[i];
            if (s->kind != stream_kind::HOST) continue;
            const int k = vad_stream_n_pending(s, n_samples[i]);
            if (k > max_probs[i]) return -1;
            n_out[i] = k;
            hs.push_back(s->sess.get()); hx.push_back(samples[i]); hl.push_back(n_samples[i]); ho.push_back(probs[i]);
        }
        if (!hs.empty()) {
            model_runner r(m);
            if (!m->a->streams_push(r, hs.data(), (int) hs.size(), hx.data(), hl.data(), ho.data())) return -1;
        }
    }
    for (int i : other) {
        if (streams[i]->kind == stream_kind::WINDOW || streams[i]->kind == stream_kind::HOST) continue;
        const int r = vad_stream_push(streams[i], samples[i], n_samples[i], probs[i], max_probs[i]);
        if (r < 0) return -1;
        n_out[i] = r;
    }
    return 0;
}

// offline frame models (lookahead): official chunking of at most MAX_FRAMES
// frames, each chunk an independent utterance; T is rounded up to a bucket so
// graphs are reused, and frame_mask hides the padding
static int detect_offline(vad_model * model, const float * samples, int n_samples, float * probs, int max_probs) {
    constexpr int MAX_FRAMES = 30000, BUCKET = 512;
    const int cs = model->a->chunk_size(), wu = model->a->warmup_samples();
    const int n_frames = n_samples > wu + cs - 1 ? 1 + (n_samples - wu - cs) / cs : 0;   // kaldi snip_edges
    if (n_frames > max_probs) return -1;
    std::lock_guard<std::mutex> lock(model->mu);
    std::vector<float> buf, out;
    for (int f0 = 0; f0 < n_frames; f0 += MAX_FRAMES) {
        const int nf = std::min(MAX_FRAMES, n_frames - f0);
        const int T = (nf + BUCKET - 1) / BUCKET * BUCKET;
        buf.assign(model->a->audio_samples(T), 0.0f);
        const int64_t s0 = (int64_t) f0 * cs;
        const int64_t avail = std::min<int64_t>((int64_t) buf.size(), n_samples - s0);
        std::copy(samples + s0, samples + s0 + avail, buf.begin());
        out.resize(T);
        const float * a = buf.data();
        float * p = out.data();
        if (!model->compute(1, T, nullptr, &a, &p, &nf)) return -1;
        std::copy(out.begin(), out.begin() + nf, probs + f0);
    }
    return n_frames;
}

// fresh stream, push, flush; results beyond max_probs are counted but not written
int vad_detect_with_params(vad_model * model, vad_stream_params params, const float * samples, int n_samples,
                           float * probs, int max_probs) {
    if (n_samples < 0 || max_probs < 0) return -1;
    // offline-only frame models with full lookahead: one pass over the utterance (official chunking)
    if (!model->a->is_turn() && !model->a->streaming() && params.lookahead_samples < 0) {
        const int cs = model->a->chunk_size(), wu = model->a->warmup_samples();
        const int n_frames = n_samples >= wu + cs ? 1 + (n_samples - wu - cs) / cs : 0;
        std::vector<float> all(n_frames);
        if (detect_offline(model, samples, n_samples, all.data(), n_frames) < 0) return -1;
        std::copy(all.begin(), all.begin() + std::min(n_frames, max_probs), probs);
        return n_frames;
    }
    // host-driven models without explicit stream parameters: their offline mode
    const bool offline = model->a->host_driven() && params.slice_samples == 0 && params.lookahead_samples < 0;
    vad_stream * s = stream_new(model, params, offline);
    if (!s) return -1;
    const size_t no = (size_t) model->a->n_outputs();
    const int n_push = vad_stream_n_pending(s, n_samples);
    std::vector<float> all((size_t) n_push * no);
    int got = vad_stream_push(s, samples, n_samples, all.data(), n_push);
    if (got >= 0) {
        const int nf = vad_stream_n_flush(s);
        all.resize((size_t) (got + nf) * no);
        const int f = vad_stream_flush(s, all.data() + (size_t) got * no, nf);
        got = f < 0 ? -1 : got + f;
    }
    vad_stream_free(s);
    if (got < 0) return -1;
    std::copy(all.begin(), all.begin() + (size_t) std::min(got, max_probs) * no, probs);
    return got;
}

int vad_detect(vad_model * model, const float * samples, int n_samples, float * probs, int max_probs) {
    return vad_detect_with_params(model, vad_stream_default_params(), samples, n_samples, probs, max_probs);
}
