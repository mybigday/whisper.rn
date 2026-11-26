// Speech probabilities of a 16 kHz mono PCM16 WAV, one per hop, streamed
// through vad.cpp as WhisperVadStream does (stateful, chunk by chunk).
//
//   vad-probs <model> <wav>   ->  {"hopSamples":512,"probs":[...]}
#include "vad.h"

#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: vad-probs <model> <wav>\n");
        return 1;
    }
    auto params = vad_model_default_params();
    params.device = "cpu";
    params.n_threads = 1;
    vad_model *model = vad_model_load(argv[1], params);
    if (!model) return 1;
    std::ifstream f(argv[2], std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(f)), {});
    // Find the data chunk
    size_t off = 12, data = 0, size = 0;
    while (off + 8 <= bytes.size()) {
        const unsigned n = *(const unsigned *) &bytes[off + 4];
        if (std::string(&bytes[off], 4) == "data") { data = off + 8; size = n; break; }
        off += 8 + n + (n % 2);
    }
    std::vector<float> pcm(size / 2);
    for (size_t i = 0; i < pcm.size(); ++i) pcm[i] = *(const int16_t *) &bytes[data + 2 * i] / 32768.0f;

    vad_stream *stream = vad_stream_new(model);
    std::vector<float> probs, out(4096);
    const size_t chunk = 1600;  // 100 ms, as the transcriber pushes
    for (size_t i = 0; i < pcm.size(); i += chunk) {
        const int n = (int) std::min(chunk, pcm.size() - i);
        const int got = vad_stream_push(stream, pcm.data() + i, n, out.data(), (int) out.size());
        probs.insert(probs.end(), out.begin(), out.begin() + std::max(got, 0));
    }
    const int got = vad_stream_flush(stream, out.data(), (int) out.size());
    probs.insert(probs.end(), out.begin(), out.begin() + std::max(got, 0));
    printf("{\"hopSamples\":%d,\"probs\":[", vad_model_hop_samples(model));
    for (size_t i = 0; i < probs.size(); ++i) printf("%s%.4f", i ? "," : "", probs[i]);
    printf("]}\n");
    vad_stream_free(stream);
    vad_model_free(model);
}
