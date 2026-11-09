// Spectrum of audio frames on the CPU: optional int16 scale, DC removal and
// pre-emphasis, a window, zero padding to nfft and a real FFT (an nfft/2-point
// complex Stockham FFT, radix 2/3/4/5, plus the real-input split).
// Replaces a [win -> 2 * nbin] DFT-matrix mul_mat that reads the whole matrix for
// every frame. Output layout matches that matmul: [Re(0..nbin-1), Im(0..nbin-1)].
#pragma once

#include "../vad-impl.h"

#include <vector>

namespace vad {

struct frame_dft {
    int win = 0, nfft = 0, nbin = 0;
    float scale = 1.0f;          // applied first (e.g. 32768: [-1, 1] -> int16 range)
    bool remove_dc = false;      // subtract the frame mean
    float preemph = 0.0f;        // y[n] = x[n] - p x[n-1], y[0] = (1 - p) x[0]
    std::vector<float> window;   // win values

    // nfft must be even, nfft / 2 must factor into 2, 3, 4 and 5
    bool init(int win, int nfft, std::vector<float> window, float scale = 1.0f, bool remove_dc = false, float preemph = 0.0f);
    // x: win samples -> re, im: nbin values each. work: at least work_size() floats.
    void run(const float * x, float * re, float * im, float * work) const;
    // 8 frames at once (same results as run); work: 8 * work_size() floats
    void run8(const float * const * x, float * const * re, float * const * im, float * work) const;
    int work_size() const { return 4 * (nfft / 2); }

    // graph op: frames [win, T, B] (any strides) -> [2 * nbin, T, B] F32, computed on the CPU
    ggml_tensor * build(ggml_context * c, ggml_tensor * frames, const graph_opts & o) const;

private:
    template <int L> void run_lanes(const float * const * x, float * const * re, float * const * im, float * work) const;
    struct stage { int p, m, s; std::vector<float> wr, wi; };   // twiddles w^(j r), j < m, r < p: [(j * p + r)]
    std::vector<stage> stages_;
    std::vector<float> sr_, si_;                                  // real split: exp(-2 pi i k / nfft), k <= nfft / 2
};

} // namespace vad
