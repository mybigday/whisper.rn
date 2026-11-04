// Graph helpers shared by the model architectures.
#pragma once

#include "../vad-impl.h"

namespace vad {

// Sliding windows over dim 0: n windows of `win` elements, `hop` elements apart,
// for each of ne2 rows (row stride nb2 bytes), starting at byte offset offs.
// Result [win, n, ne2].
//  - host (CPU backends): one overlapping strided view, no copy. ggml's view
//    constructor rejects views whose element count exceeds the source even when
//    the strided span fits, so a one-row view is created and widened.
//  - accelerators: such a view breaks the scheduler's copies between backends,
//    so each window is assembled from non-overlapping views (pieces of at most
//    `hop` elements) joined with ggml_concat.
inline ggml_tensor * sliding_windows(ggml_context * c, ggml_tensor * src, int64_t win, int64_t n, int64_t hop,
                                     int64_t ne2, size_t nb2, const graph_opts & o, size_t offs = 0) {
    const size_t es = ggml_element_size(src), nb1 = hop * es;
    if (o.host) {
        ggml_tensor * v = ggml_view_3d(c, src, win, 1, 1, nb1, nb2, offs);
        v->ne[1] = n; v->ne[2] = ne2;
        v->nb[1] = nb1; v->nb[2] = nb2; v->nb[3] = nb2 * ne2;
        GGML_ASSERT(offs + ggml_nbytes(v) <= ggml_nbytes(src));
        return v;
    }
    ggml_tensor * out = nullptr;
    for (int64_t at = 0; at < win; at += hop) {
        const int64_t len = std::min(hop, win - at);
        ggml_tensor * piece = ggml_view_3d(c, src, len, n, ne2, nb1, nb2, offs + at * es);
        out = out ? ggml_concat(c, out, piece, 0) : ggml_cont(c, piece);
    }
    return out;
}

} // namespace vad
