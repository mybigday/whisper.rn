// Parakeet over stdin/stdout for the realtime evaluation harness (run.ts
// --engine parakeet), with the same protocol as whisper-server.
//
//   parakeet-server <model> [threads]
//
// Request:  "<bytes> key=value ...\n" then <bytes> of mono 16 kHz PCM16.
//           Keys: audio_ctx (number; others are ignored, as in the JSI layer).
// Response: one JSON line shaped like whisper.rn's TranscribeResult for
//           Parakeet (segments with text, t0, t1), plus "ms".
#include "parakeet.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string jsonString(const std::string &s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += (char) c; }
        else if (c < 0x20) { char buf[8]; snprintf(buf, sizeof buf, "\\u%04x", c); out += buf; }
        else out += (char) c;
    }
    return out + "\"";
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: parakeet-server <model> [threads]\n");
        return 1;
    }
    parakeet_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    auto cp = parakeet_context_default_params();
    cp.use_gpu = false;
    const int threads = argc > 2 ? atoi(argv[2]) : 4;
    parakeet_context *ctx = parakeet_init_from_file_with_params(argv[1], cp);
    if (!ctx) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    std::cout << "{\"ready\":true,\"dtw\":false}" << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream header(line);
        size_t bytes = 0;
        header >> bytes;
        std::map<std::string, std::string> opts;
        std::string kv;
        while (header >> kv) {
            const auto eq = kv.find('=');
            if (eq != std::string::npos) opts[kv.substr(0, eq)] = kv.substr(eq + 1);
        }
        std::vector<int16_t> pcm16(bytes / 2);
        std::cin.read(reinterpret_cast<char *>(pcm16.data()), (std::streamsize) bytes);
        std::vector<float> pcm(pcm16.size());
        for (size_t i = 0; i < pcm16.size(); ++i) pcm[i] = pcm16[i] / 32768.0f;

        // As createParakeetTranscribeConfig in the JSI layer
        auto p = parakeet_full_default_params(PARAKEET_SAMPLING_GREEDY);
        p.n_threads = threads;
        p.no_context = true;
        if (opts.count("audio_ctx") && opts["audio_ctx"] != "auto") p.audio_ctx = atoi(opts["audio_ctx"].c_str());

        const auto t0 = std::chrono::steady_clock::now();
        const int rc = parakeet_full(ctx, p, pcm.data(), (int) pcm.size());
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        std::ostringstream out;
        std::string text;
        out << "{\"isAborted\":false,\"ms\":" << ms << ",\"rc\":" << rc << ",\"segments\":[";
        const int ns = rc == 0 ? parakeet_full_n_segments(ctx) : 0;
        for (int s = 0; s < ns; ++s) {
            const char *segText = parakeet_full_get_segment_text(ctx, s);
            const std::string t = segText ? segText : "";
            text += t;
            out << (s ? "," : "") << "{\"text\":" << jsonString(t) << ",\"t0\":" << parakeet_full_get_segment_t0(ctx, s)
                << ",\"t1\":" << parakeet_full_get_segment_t1(ctx, s) << "}";
        }
        out << "],\"result\":" << jsonString(text) << ",\"language\":\"\"}";
        std::cout << out.str() << std::endl;
    }
    parakeet_free(ctx);
    return 0;
}
