// Weight file loaders: vad.cpp GGUF and whisper.cpp's legacy ggml-silero-*.bin.
#include "../vad-impl.h"

#include "ggml.h"
#include "gguf.h"

#include <cstring>
#include <fstream>

namespace vad {

namespace {

bool supported(ggml_type type) {
    const ggml_type_traits * tt = ggml_get_type_traits(type);
    return type == GGML_TYPE_F32 || (tt && tt->to_float);
}

} // namespace

// GGUF written by tools/convert_*.py (docs/gguf.md): general.* kept with full names,
// "<arch>.<key>" stored as "<key>". Each architecture checks its own keys and tensors.
bool load_gguf(const std::string & path, model_file & out, std::string & err) {
    ggml_context * ctx = nullptr;
    gguf_init_params ip = { /*no_alloc*/ false, &ctx };
    gguf_context * g = gguf_init_from_file(path.c_str(), ip);
    if (!g) { err = "not a valid GGUF file: " + path; return false; }

    const int64_t ia = gguf_find_key(g, "general.architecture");
    if (ia < 0 || gguf_get_kv_type(g, ia) != GGUF_TYPE_STRING) {
        err = "GGUF has no general.architecture";
        gguf_free(g); ggml_free(ctx);
        return false;
    }
    out.arch = gguf_get_val_str(g, ia);
    const std::string prefix = out.arch + ".";
    for (int64_t i = 0; i < gguf_get_n_kv(g); ++i) {
        std::string key = gguf_get_key(g, i);
        if (key.rfind("general.", 0) == 0) {           // kept with their full name
            if (gguf_get_kv_type(g, i) == GGUF_TYPE_STRING) out.kv_str[key] = gguf_get_val_str(g, i);
            continue;
        }
        if (key.rfind(prefix, 0) != 0) continue;
        key = key.substr(prefix.size());
        switch (gguf_get_kv_type(g, i)) {
            case GGUF_TYPE_STRING: out.kv_str[key] = gguf_get_val_str(g, i); break;
            case GGUF_TYPE_UINT32: out.kv_int[key] = gguf_get_val_u32(g, i); break;
            case GGUF_TYPE_INT32:  out.kv_int[key] = gguf_get_val_i32(g, i); break;
            case GGUF_TYPE_UINT64: out.kv_int[key] = (int64_t) gguf_get_val_u64(g, i); break;
            case GGUF_TYPE_INT64:  out.kv_int[key] = gguf_get_val_i64(g, i); break;
            case GGUF_TYPE_BOOL:   out.kv_int[key] = gguf_get_val_bool(g, i) ? 1 : 0; break;
            case GGUF_TYPE_FLOAT32: out.kv_float[key] = gguf_get_val_f32(g, i); break;
            case GGUF_TYPE_FLOAT64: out.kv_float[key] = gguf_get_val_f64(g, i); break;
            default: break;
        }
    }
    bool ok = true;
    // iterate the GGUF tensor list: the ggml context also holds the raw data blob
    for (int64_t i = 0; i < gguf_get_n_tensors(g) && ok; ++i) {
        ggml_tensor * t = ggml_get_tensor(ctx, gguf_get_tensor_name(g, i));
        if (!t) { err = "missing tensor data"; ok = false; break; }
        host_tensor ht;
        for (int d = 0; d < ggml_n_dims(t); ++d) ht.ne.push_back(t->ne[d]);
        ht.type = t->type;
        if (!supported(t->type)) { err = std::string("unsupported tensor type ") + ggml_type_name(t->type); ok = false; break; }
        ht.data.assign((const uint8_t *) t->data, (const uint8_t *) t->data + ggml_nbytes(t));
        out.tensors[ggml_get_name(t)] = std::move(ht);
    }
    gguf_free(g);
    ggml_free(ctx);
    return ok;
}

// PyTorch-named Silero tensors (whisper.cpp files) -> the graph-ready names of
// docs/gguf.md, at the stored precision: conv weights (Cout, Cin, 3) are re-laid
// out as (Cout, 3, Cin) element-wise, the two LSTM biases are summed.
bool conv_to_window_layout(const host_tensor & w, int64_t cout, int64_t cin, int64_t k, host_tensor & out) {
    const size_t es = ggml_type_size(w.type);
    if (!w.is_float() || ggml_blck_size(w.type) != 1 || w.nelements() != cout * cin * k) return false;
    out.type = w.type;
    out.ne = { k * cin, cout };
    out.data.resize(w.data.size());
    for (int64_t o = 0; o < cout; ++o)
        for (int64_t ci = 0; ci < cin; ++ci)
            for (int64_t j = 0; j < k; ++j)
                std::memcpy(&out.data[((o * k + j) * cin + ci) * es], &w.data[((o * cin + ci) * k + j) * es], es);
    return true;
}

static bool permute_conv(const host_tensor & w, int cout, int cin, host_tensor & out) {
    return conv_to_window_layout(w, cout, cin, 3, out);
}

bool silero_to_graph_layout(model_file & m, std::string & err) {
    auto & T = m.tensors;
    auto need = [&](const char * n) { if (!T.count(n)) { err = std::string("silero: missing tensor ") + n; return false; } return true; };
    const int cin[4] = {129, 128, 64, 64}, cout[4] = {128, 64, 64, 128};
    std::map<std::string, host_tensor> g;
    if (!need("stft.forward_basis_buffer")) return false;
    g["stft.basis"] = T["stft.forward_basis_buffer"];
    g["stft.basis"].ne = { 256, 258 };
    for (int l = 0; l < 4; ++l) {
        const std::string p = "encoder." + std::to_string(l) + ".reparam_conv.", q = "enc" + std::to_string(l) + ".";
        if (!need((p + "weight").c_str()) || !need((p + "bias").c_str())) return false;
        if (!permute_conv(T[p + "weight"], cout[l], cin[l], g[q + "w"])) { err = "silero: unsupported conv weight type"; return false; }
        g[q + "b"] = T[p + "bias"];
    }
    for (const char * n : {"decoder.rnn.weight_ih", "decoder.rnn.weight_hh", "decoder.rnn.bias_ih", "decoder.rnn.bias_hh",
                           "decoder.decoder.2.weight", "decoder.decoder.2.bias"}) if (!need(n)) return false;
    g["lstm.w_ih"] = T["decoder.rnn.weight_ih"];
    g["lstm.w_hh"] = T["decoder.rnn.weight_hh"];
    std::vector<float> b = T["decoder.rnn.bias_ih"].to_f32(), bh = T["decoder.rnn.bias_hh"].to_f32();
    for (size_t i = 0; i < b.size(); ++i) b[i] += bh[i];
    g["lstm.b"] = host_tensor::from_f32({ (int64_t) b.size() }, b);
    g["dec.w"] = T["decoder.decoder.2.weight"];
    g["dec.w"].ne = { 128, 1 };
    g["dec.b"] = T["decoder.decoder.2.bias"];
    T.swap(g);
    return true;
}

// whisper.cpp models/convert-silero-vad-to-ggml.py format (little endian int32 fields):
//   magic 0x67676d6c, len + "silero-16k", major, minor, patch, n_window, n_context,
//   n_encoder_layers, per layer (in, out, kernel), lstm_in, lstm_hidden, final_in, final_out,
//   then tensors: n_dims, name_len, ftype (0 f32, 1 f16), dims[n_dims], name, data
bool load_whisper_bin(const std::string & path, model_file & out, std::string & err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    auto rd = [&](int32_t & v) { return (bool) f.read((char *) &v, 4); };
    int32_t magic = 0, len = 0;
    if (!rd(magic) || magic != 0x67676d6c) { err = "not a whisper.cpp ggml file"; return false; }
    rd(len);
    if (len <= 0 || len > 64) { err = "bad model type"; return false; }
    std::string type(len, '\0');
    f.read(type.data(), len);
    if (type != "silero-16k") { err = "unsupported whisper.cpp VAD model type: " + type; return false; }

    int32_t major, minor, patch, n_window, n_context, n_layers;
    rd(major); rd(minor); rd(patch); rd(n_window); rd(n_context); rd(n_layers);
    if (!f || n_layers <= 0 || n_layers > 16) { err = "bad header"; return false; }
    for (int i = 0; i < n_layers * 3 + 4; ++i) { int32_t skip; rd(skip); }

    out.arch = "silero";
    out.kv_str["general.version"] = std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    out.kv_int["preprocessor.sample_rate"] = 16000;
    out.kv_int["chunk_size"] = n_window;
    out.kv_int["context_size"] = n_context;

    while (true) {
        int32_t n_dims, name_len, ftype;
        if (!rd(n_dims)) break;   // EOF
        rd(name_len); rd(ftype);
        // n_dims may be 0: the converter squeezes 1-element tensors to scalars
        if (!f || n_dims < 0 || n_dims > 4 || name_len <= 0 || name_len > 256 || (ftype != 0 && ftype != 1)) {
            err = "bad tensor header"; return false;
        }
        host_tensor ht;
        int64_t n = 1;
        for (int d = 0; d < n_dims; ++d) { int32_t v; rd(v); ht.ne.push_back(v); n *= v; }
        if (n_dims == 0) ht.ne.push_back(1);
        std::string name(name_len, '\0');
        f.read(name.data(), name_len);
        ht.type = ftype ? GGML_TYPE_F16 : GGML_TYPE_F32;
        ht.data.resize(n * (ftype ? 2 : 4));
        f.read((char *) ht.data.data(), ht.data.size());
        if (!f) { err = "truncated tensor " + name; return false; }
        if (name.rfind("_model.", 0) == 0) name = name.substr(7);
        out.tensors[name] = std::move(ht);
    }
    return silero_to_graph_layout(out, err);
}

} // namespace vad
