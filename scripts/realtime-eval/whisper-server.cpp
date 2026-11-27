// Whisper over stdin/stdout for the realtime evaluation harness (run.ts).
//
//   whisper-server <model> [dtw-preset|-] [threads]
//
// Request:  "<bytes> key=value ...\n" then <bytes> of mono 16 kHz PCM16.
//           Keys: language, prompt_hex, audio_ctx (number|auto), max_tokens,
//           single_segment, temperature_inc, include_tokens, token_timestamps.
// Response: one JSON line shaped like whisper.rn's TranscribeResult, plus
//           "ms" (inference time). Token texts are regrouped into valid UTF-8
//           as the JSI layer does.
#include "AudioCtx.h"
#include "whisper.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

size_t completeUtf8Length(const std::string &text) {
    const size_t size = text.size();
    for (size_t back = 1; back <= 4 && back <= size; ++back) {
        const unsigned char c = static_cast<unsigned char>(text[size - back]);
        if ((c & 0xC0) == 0x80) continue;
        const size_t length = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
        return back >= length ? size : size - back;
    }
    return size;
}

std::string jsonString(const std::string &s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += (char) c; }
        else if (c < 0x20) { char buf[8]; snprintf(buf, sizeof buf, "\\u%04x", c); out += buf; }
        else out += (char) c;
    }
    return out + "\"";
}

std::string fromHex(const std::string &hex) {
    std::string out;
    for (size_t i = 0; i + 1 < hex.size(); i += 2) out += (char) std::stoi(hex.substr(i, 2), nullptr, 16);
    return out;
}

whisper_alignment_heads_preset presetOf(const std::string &name) {
    static const std::map<std::string, whisper_alignment_heads_preset> presets = {
        {"tiny.en", WHISPER_AHEADS_TINY_EN}, {"tiny", WHISPER_AHEADS_TINY},
        {"base.en", WHISPER_AHEADS_BASE_EN}, {"base", WHISPER_AHEADS_BASE},
        {"small.en", WHISPER_AHEADS_SMALL_EN}, {"small", WHISPER_AHEADS_SMALL},
        {"medium.en", WHISPER_AHEADS_MEDIUM_EN}, {"medium", WHISPER_AHEADS_MEDIUM},
        {"large-v1", WHISPER_AHEADS_LARGE_V1}, {"large-v2", WHISPER_AHEADS_LARGE_V2},
        {"large-v3", WHISPER_AHEADS_LARGE_V3}, {"large-v3-turbo", WHISPER_AHEADS_LARGE_V3_TURBO},
    };
    auto it = presets.find(name);
    return it == presets.end() ? WHISPER_AHEADS_NONE : it->second;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: whisper-server <model> [dtw-preset|-] [threads]\n");
        return 1;
    }
    whisper_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    auto cp = whisper_context_default_params();
    cp.use_gpu = false;
    cp.flash_attn = false;
    const std::string dtw = argc > 2 ? argv[2] : "-";
    if (dtw != "-" && presetOf(dtw) != WHISPER_AHEADS_NONE) {
        cp.dtw_token_timestamps = true;
        cp.dtw_aheads_preset = presetOf(dtw);
    }
    const int threads = argc > 3 ? atoi(argv[3]) : 4;
    whisper_context *ctx = whisper_init_from_file_with_params(argv[1], cp);
    if (!ctx) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    std::cout << "{\"ready\":true,\"dtw\":" << (cp.dtw_token_timestamps ? "true" : "false") << "}" << std::endl;

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

        auto p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        p.n_threads = threads;
        p.no_context = true;
        p.print_progress = false;
        p.print_realtime = false;
        p.print_timestamps = false;
        const std::string language = opts.count("language") ? opts["language"] : "auto";
        p.language = language.c_str();
        const std::string prompt = opts.count("prompt_hex") ? fromHex(opts["prompt_hex"]) : "";
        if (!prompt.empty()) p.initial_prompt = prompt.c_str();
        if (opts.count("audio_ctx")) {
            p.audio_ctx = opts["audio_ctx"] == "auto"
                ? rnwhisper_jsi::fitAudioCtx((int64_t) pcm.size(), whisper_model_n_audio_ctx(ctx))
                : atoi(opts["audio_ctx"].c_str());
        }
        if (opts.count("max_tokens")) p.max_tokens = atoi(opts["max_tokens"].c_str());
        if (opts.count("single_segment")) p.single_segment = opts["single_segment"] == "1";
        if (opts.count("temperature_inc")) p.temperature_inc = (float) atof(opts["temperature_inc"].c_str());
        p.token_timestamps = opts["token_timestamps"] == "1";
        const bool includeTokens = opts["include_tokens"] == "1";
        // whisper needs at least 1 s
        if (pcm.size() < 16000 + 1600) pcm.resize(16000 + 1600, 0.0f);

        const auto t0 = std::chrono::steady_clock::now();
        const int rc = whisper_full(ctx, p, pcm.data(), (int) pcm.size());
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        std::ostringstream out;
        std::string text;
        const whisper_token eot = whisper_token_eot(ctx);
        out << "{\"isAborted\":false,\"ms\":" << ms << ",\"rc\":" << rc;
        out << ",\"segments\":[";
        const int ns = rc == 0 ? whisper_full_n_segments(ctx) : 0;
        for (int s = 0; s < ns; ++s) {
            const std::string segText = whisper_full_get_segment_text(ctx, s);
            text += segText;
            double plogSum = 0;
            int n = 0;
            std::ostringstream toks;
            std::string pending;
            bool firstTok = true;
            for (int t = 0; t < whisper_full_n_tokens(ctx, s); ++t) {
                const auto d = whisper_full_get_token_data(ctx, s, t);
                if (d.id >= eot) continue;
                plogSum += d.plog;
                ++n;
                if (!includeTokens) continue;
                pending += whisper_full_get_token_text(ctx, s, t);
                const size_t complete = completeUtf8Length(pending);
                const std::string tokText = pending.substr(0, complete);
                pending.erase(0, complete);
                toks << (firstTok ? "" : ",") << "{\"id\":" << d.id << ",\"text\":" << jsonString(tokText)
                     << ",\"p\":" << d.p << ",\"plog\":" << d.plog << ",\"t0\":" << d.t0 << ",\"t1\":" << d.t1
                     << ",\"tDtw\":" << d.t_dtw << "}";
                firstTok = false;
            }
            out << (s ? "," : "") << "{\"text\":" << jsonString(segText) << ",\"t0\":" << whisper_full_get_segment_t0(ctx, s)
                << ",\"t1\":" << whisper_full_get_segment_t1(ctx, s)
                << ",\"noSpeechProb\":" << whisper_full_get_segment_no_speech_prob(ctx, s)
                << ",\"avgLogprob\":" << (n ? plogSum / n : 0.0);
            if (includeTokens) out << ",\"tokens\":[" << toks.str() << "]";
            out << "}";
        }
        const char *lang = whisper_lang_str(whisper_full_lang_id(ctx));
        out << "],\"result\":" << jsonString(text) << ",\"language\":" << jsonString(lang ? lang : "");
        if (whisper_full_lang_prob(ctx) >= 0) out << ",\"languageProb\":" << whisper_full_lang_prob(ctx);
        out << "}";
        std::cout << out.str() << std::endl;
    }
    whisper_free(ctx);
    return 0;
}
