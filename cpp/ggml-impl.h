#pragma once

// GGML internal header

#include "ggml.h"

#include <assert.h>
#include <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#undef MIN
#undef MAX

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// required for mmap as gguf only guarantees 32-byte alignment
#define TENSOR_ALIGNMENT 32

// static_assert should be a #define, but if it's not,
// fall back to the _Static_assert C11 keyword.
// if C99 - static_assert is noop
// ref: https://stackoverflow.com/a/53923785/4039976
#ifndef __cplusplus
#ifndef static_assert
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
#define static_assert(cond, msg) _Static_assert(cond, msg)
#else
#define static_assert(cond, msg) struct global_scope_noop_trick
#endif
#endif
#endif

//
// logging
//

WSP_GGML_ATTRIBUTE_FORMAT(2, 3)
void wsp_ggml_log_internal        (enum wsp_ggml_log_level level, const char * format, ...);
void wsp_ggml_log_callback_default(enum wsp_ggml_log_level level, const char * text, void * user_data);

#define WSP_GGML_LOG(...)       wsp_ggml_log_internal(WSP_GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define WSP_GGML_LOG_INFO(...)  wsp_ggml_log_internal(WSP_GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define WSP_GGML_LOG_WARN(...)  wsp_ggml_log_internal(WSP_GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define WSP_GGML_LOG_ERROR(...) wsp_ggml_log_internal(WSP_GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define WSP_GGML_LOG_DEBUG(...) wsp_ggml_log_internal(WSP_GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define WSP_GGML_LOG_CONT(...)  wsp_ggml_log_internal(WSP_GGML_LOG_LEVEL_CONT , __VA_ARGS__)

// bitset

typedef uint32_t wsp_ggml_bitset_t;

static_assert(sizeof(wsp_ggml_bitset_t) == 4, "bitset_t constants must be updated");
#define BITSET_SHR 5 // log2(sizeof(wsp_ggml_bitset_t)*8)
#define BITSET_MASK (sizeof(wsp_ggml_bitset_t)*8 - 1)

static size_t wsp_ggml_bitset_size(size_t n) {
    return (n + BITSET_MASK) >> BITSET_SHR;
}

static inline bool wsp_ggml_bitset_get(const wsp_ggml_bitset_t * bitset, size_t i) {
    return !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_MASK)));
}

static inline void wsp_ggml_bitset_set(wsp_ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_MASK));
}

static inline void wsp_ggml_bitset_clear(wsp_ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_MASK));
}

// hash set

#define WSP_GGML_HASHSET_FULL ((size_t)-1)
#define WSP_GGML_HASHSET_ALREADY_EXISTS ((size_t)-2)

struct wsp_ggml_hash_set {
    size_t size;
    wsp_ggml_bitset_t * used;       // whether or not the keys are in use i.e. set
    struct wsp_ggml_tensor ** keys; // actual tensors in the set, keys[i] is only defined if wsp_ggml_bitset_get(used, i)
};

struct wsp_ggml_hash_set wsp_ggml_hash_set_new(size_t size);
void                 wsp_ggml_hash_set_free(struct wsp_ggml_hash_set * hash_set);

// returns the minimum size for a hash set that can hold min_sz elements
size_t wsp_ggml_hash_size(size_t min_sz);

// remove all elements from the hash set
void wsp_ggml_hash_set_reset(struct wsp_ggml_hash_set * hash_set);

// returns true if key is in the hash set
static bool wsp_ggml_hash_contains(const struct wsp_ggml_hash_set * hash_set, struct wsp_ggml_tensor * key);

// returns WSP_GGML_HASHSET_FULL if table is full, otherwise the current index of the key or where it should be inserted
static size_t wsp_ggml_hash_find(const struct wsp_ggml_hash_set * hash_set, struct wsp_ggml_tensor * key);

// returns WSP_GGML_HASHSET_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
static size_t wsp_ggml_hash_insert(struct wsp_ggml_hash_set * hash_set, struct wsp_ggml_tensor * key);

// return index, asserts if table is full
static size_t wsp_ggml_hash_find_or_insert(struct wsp_ggml_hash_set * hash_set, struct wsp_ggml_tensor * key);

// hash function for wsp_ggml_tensor
static inline size_t wsp_ggml_hash(const struct wsp_ggml_tensor * p) {
    // the last 4 bits are always zero due to alignment
    return (size_t)(uintptr_t)p >> 4;
}

static size_t wsp_ggml_hash_find(const struct wsp_ggml_hash_set * hash_set, struct wsp_ggml_tensor * key) {
    size_t h = wsp_ggml_hash(key) % hash_set->size;

    // linear probing
    size_t i = h;
    while (wsp_ggml_bitset_get(hash_set->used, i) && hash_set->keys[i] != key) {
        i = (i + 1) % hash_set->size;
        if (i == h) {
            // visited all hash table entries -> not found
            return WSP_GGML_HASHSET_FULL;
        }
    }
    return i;
}

static bool wsp_ggml_hash_contains(const struct wsp_ggml_hash_set * hash_set, struct wsp_ggml_tensor * key) {
    size_t i = wsp_ggml_hash_find(hash_set, key);
    return i != WSP_GGML_HASHSET_FULL && wsp_ggml_bitset_get(hash_set->used, i);
}

static size_t wsp_ggml_hash_insert(struct wsp_ggml_hash_set * hash_set, struct wsp_ggml_tensor * key) {
    size_t h = wsp_ggml_hash(key) % hash_set->size;

    // linear probing
    size_t i = h;
    do {
        if (!wsp_ggml_bitset_get(hash_set->used, i)) {
            wsp_ggml_bitset_set(hash_set->used, i);
            hash_set->keys[i] = key;
            return i;
        }
        if (hash_set->keys[i] == key) {
            return WSP_GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_set->size;
    } while (i != h);

    // visited all hash table entries -> not found
    WSP_GGML_ABORT("fatal error");
}

static size_t wsp_ggml_hash_find_or_insert(struct wsp_ggml_hash_set * hash_set, struct wsp_ggml_tensor * key) {
    size_t h = wsp_ggml_hash(key) % hash_set->size;

    // linear probing
    size_t i = h;
    do {
        if (!wsp_ggml_bitset_get(hash_set->used, i)) {
            wsp_ggml_bitset_set(hash_set->used, i);
            hash_set->keys[i] = key;
            return i;
        }
        if (hash_set->keys[i] == key) {
            return i;
        }
        i = (i + 1) % hash_set->size;
    } while (i != h);

    // visited all hash table entries -> not found
    WSP_GGML_ABORT("fatal error");
}

// computation graph

enum wsp_ggml_cgraph_eval_order {
    WSP_GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
    WSP_GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
    WSP_GGML_CGRAPH_EVAL_ORDER_COUNT
};

struct wsp_ggml_cgraph {
    int size;
    int n_nodes;
    int n_leafs;

    struct wsp_ggml_tensor ** nodes;
    struct wsp_ggml_tensor ** grads;
    struct wsp_ggml_tensor ** leafs;

    struct wsp_ggml_hash_set visited_hash_set;

    enum wsp_ggml_cgraph_eval_order order;
};

struct wsp_ggml_cgraph wsp_ggml_graph_view(struct wsp_ggml_cgraph * cgraph, int i0, int i1);

// Memory allocation

void * wsp_ggml_aligned_malloc(size_t size);
void wsp_ggml_aligned_free(void * ptr, size_t size);

#ifdef __cplusplus
}
#endif
