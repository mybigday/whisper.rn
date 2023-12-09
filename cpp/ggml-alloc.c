#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-impl.h"
#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX_FREE_BLOCKS 256

//#define WSP_GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) fprintf(stderr, __VA_ARGS__)
#define AT_PRINTF(...)

// TODO: WSP_GGML_PAD ?
static size_t aligned_offset(const void * buffer, size_t offset, size_t alignment) {
    assert(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
    return offset + align;
}

struct free_block {
    void * addr;
    size_t size;
};

struct wsp_ggml_tallocr {
    struct wsp_ggml_backend_buffer * buffer;
    bool buffer_owned;
    void * base;
    size_t alignment;

    int n_free_blocks;
    struct free_block free_blocks[MAX_FREE_BLOCKS];

    size_t max_size;

    bool measure;

#ifdef WSP_GGML_ALLOCATOR_DEBUG
    struct wsp_ggml_tensor * allocated_tensors[1024];
#endif
};

#ifdef WSP_GGML_ALLOCATOR_DEBUG
static void add_allocated_tensor(wsp_ggml_tallocr_t alloc, struct wsp_ggml_tensor * tensor) {
    for (int i = 0; i < 1024; i++) {
        if (alloc->allocated_tensors[i] == NULL) {
            alloc->allocated_tensors[i] = tensor;
            return;
        }
    }
    WSP_GGML_ASSERT(!"out of allocated_tensors");
}
static void remove_allocated_tensor(wsp_ggml_tallocr_t alloc, struct wsp_ggml_tensor * tensor) {
    for (int i = 0; i < 1024; i++) {
        if (alloc->allocated_tensors[i] == tensor ||
            (alloc->allocated_tensors[i] != NULL && alloc->allocated_tensors[i]->data == tensor->data)) {
            alloc->allocated_tensors[i] = NULL;
            return;
        }
    }
    printf("tried to free tensor %s not found\n", tensor->name);
    WSP_GGML_ASSERT(!"tensor not found");
}
#endif

// check if a tensor is allocated by this buffer
static bool wsp_ggml_tallocr_is_own(wsp_ggml_tallocr_t alloc, const struct wsp_ggml_tensor * tensor) {
    return tensor->buffer == alloc->buffer;
}

static bool wsp_ggml_is_view(struct wsp_ggml_tensor * t) {
    return t->view_src != NULL;
}

void wsp_ggml_tallocr_alloc(wsp_ggml_tallocr_t alloc, struct wsp_ggml_tensor * tensor) {
    WSP_GGML_ASSERT(!wsp_ggml_is_view(tensor)); // views generally get data pointer from one of their sources
    WSP_GGML_ASSERT(tensor->data == NULL); // avoid allocating tensor which already has memory allocated

    size_t size = wsp_ggml_backend_buffer_get_alloc_size(alloc->buffer, tensor);
    size = aligned_offset(NULL, size, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

    size_t max_avail = 0;

    // find the best fitting free block besides the last block
    int best_fit_block = -1;
    size_t best_fit_size = SIZE_MAX;
    for (int i = 0; i < alloc->n_free_blocks - 1; i++) {
        struct free_block * block = &alloc->free_blocks[i];
        max_avail = MAX(max_avail, block->size);
        if (block->size >= size && block->size <= best_fit_size) {
            best_fit_block = i;
            best_fit_size = block->size;
        }
    }

    AT_PRINTF("block %d\n", best_fit_block);

    if (best_fit_block == -1) {
        // the last block is our last resort
        struct free_block * block = &alloc->free_blocks[alloc->n_free_blocks - 1];
        max_avail = MAX(max_avail, block->size);
        if (block->size >= size) {
            best_fit_block = alloc->n_free_blocks - 1;
        } else {
            fprintf(stderr, "%s: not enough space in the buffer (needed %zu, largest block available %zu)\n",
                    __func__, size, max_avail);
            WSP_GGML_ASSERT(!"not enough space in the buffer");
            return;
        }
    }
    struct free_block * block = &alloc->free_blocks[best_fit_block];
    void * addr = block->addr;
    block->addr = (char*)block->addr + size;
    block->size -= size;
    if (block->size == 0) {
        // remove block if empty
        alloc->n_free_blocks--;
        for (int j = best_fit_block; j < alloc->n_free_blocks; j++) {
            alloc->free_blocks[j] = alloc->free_blocks[j+1];
        }
    }

    tensor->data = addr;
    tensor->buffer = alloc->buffer;
    if (!alloc->measure) {
        wsp_ggml_backend_buffer_init_tensor(alloc->buffer, tensor);
    }

#ifdef WSP_GGML_ALLOCATOR_DEBUG
    add_allocated_tensor(alloc, tensor);
    size_t cur_max = (char*)addr - (char*)alloc->base + size;
    if (cur_max > alloc->max_size) {
        printf("max_size = %.2f MB: tensors: ", cur_max / 1024.0 / 1024.0);
        for (int i = 0; i < 1024; i++) {
            if (alloc->allocated_tensors[i]) {
                printf("%s (%.2f MB) ", alloc->allocated_tensors[i]->name, wsp_ggml_nbytes(alloc->allocated_tensors[i]) / 1024.0 / 1024.0);
            }
        }
        printf("\n");
    }
#endif

    alloc->max_size = MAX(alloc->max_size, (char*)addr - (char*)alloc->base + size);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
static void wsp_ggml_tallocr_free_tensor(wsp_ggml_tallocr_t alloc, struct wsp_ggml_tensor * tensor) {
    if (wsp_ggml_tallocr_is_own(alloc, tensor) == false) {
        // the tensor was not allocated in this buffer
        // this can happen because the graph allocator will try to free weights and other tensors from different buffers
        // the easiest way to deal with this is just to ignore it
        // AT_PRINTF("ignoring %s (their buffer: %p, our buffer: %p)\n", tensor->name, (void *)tensor->buffer, (void *)alloc->buffer);
        return;
    }

    void * ptr = tensor->data;

    size_t size = wsp_ggml_backend_buffer_get_alloc_size(alloc->buffer, tensor);
    size = aligned_offset(NULL, size, alloc->alignment);
    AT_PRINTF("%s: freeing %s at %p (%zu bytes) - n_free_blocks = %d\n", __func__, tensor->name, ptr, size, alloc->n_free_blocks);

#ifdef WSP_GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, tensor);
#endif

    // see if we can merge with an existing block
    for (int i = 0; i < alloc->n_free_blocks; i++) {
        struct free_block * block = &alloc->free_blocks[i];
        // check if ptr is at the end of the block
        if ((char*)block->addr + block->size == ptr) {
            block->size += size;
            // check if we can merge with the next block
            if (i < alloc->n_free_blocks - 1 && (char*)block->addr + block->size == alloc->free_blocks[i+1].addr) {
                block->size += alloc->free_blocks[i+1].size;
                alloc->n_free_blocks--;
                for (int j = i+1; j < alloc->n_free_blocks; j++) {
                    alloc->free_blocks[j] = alloc->free_blocks[j+1];
                }
            }
            return;
        }
        // check if ptr is at the beginning of the block
        if ((char*)ptr + size == block->addr) {
            block->addr = ptr;
            block->size += size;
            // check if we can merge with the previous block
            if (i > 0 && (char*)alloc->free_blocks[i-1].addr + alloc->free_blocks[i-1].size == block->addr) {
                alloc->free_blocks[i-1].size += block->size;
                alloc->n_free_blocks--;
                for (int j = i; j < alloc->n_free_blocks; j++) {
                    alloc->free_blocks[j] = alloc->free_blocks[j+1];
                }
            }
            return;
        }
    }
    // otherwise, add a new block
    WSP_GGML_ASSERT(alloc->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    int insert_pos = 0;
    while (insert_pos < alloc->n_free_blocks && alloc->free_blocks[insert_pos].addr < ptr) {
        insert_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    for (int i = alloc->n_free_blocks; i > insert_pos; i--) {
        alloc->free_blocks[i] = alloc->free_blocks[i-1];
    }
    // insert the new block
    alloc->free_blocks[insert_pos].addr = ptr;
    alloc->free_blocks[insert_pos].size = size;
    alloc->n_free_blocks++;
}

void wsp_ggml_tallocr_reset(wsp_ggml_tallocr_t alloc) {
    alloc->n_free_blocks = 1;
    size_t align_offset = aligned_offset(alloc->base, 0, alloc->alignment);
    alloc->free_blocks[0].addr = (char *)alloc->base + align_offset;

    if (alloc->measure) {
        alloc->free_blocks[0].size = SIZE_MAX/2; // restrict maximum size of a measure allocator to half size_t max to avoid overflows
    } else {
        alloc->free_blocks[0].size = wsp_ggml_backend_buffer_get_size(alloc->buffer) - align_offset;
    }
}

wsp_ggml_tallocr_t wsp_ggml_tallocr_new(void * data, size_t size, size_t alignment) {
    struct wsp_ggml_backend_buffer * buffer = wsp_ggml_backend_cpu_buffer_from_ptr(data, size);

    wsp_ggml_tallocr_t alloc = (wsp_ggml_tallocr_t)malloc(sizeof(struct wsp_ggml_tallocr));

    *alloc = (struct wsp_ggml_tallocr) {
        /*.buffer        = */ buffer,
        /*.buffer_owned  = */ true,
        /*.base          = */ wsp_ggml_backend_buffer_get_base(buffer),
        /*.alignment     = */ alignment,
        /*.n_free_blocks = */ 0,
        /*.free_blocks   = */ {{0}},
        /*.max_size      = */ 0,
        /*.measure       = */ false,
#ifdef WSP_GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {0},
#endif
    };

    wsp_ggml_tallocr_reset(alloc);

    return alloc;
}

wsp_ggml_tallocr_t wsp_ggml_tallocr_new_measure(size_t alignment) {
    wsp_ggml_tallocr_t alloc = wsp_ggml_tallocr_new((void *)0x1000, SIZE_MAX/2, alignment);
    alloc->measure = true;

    return alloc;
}

wsp_ggml_tallocr_t wsp_ggml_tallocr_new_measure_from_backend(struct wsp_ggml_backend * backend) {
    // create a backend buffer to get the correct tensor allocation sizes
    wsp_ggml_backend_buffer_t buffer = wsp_ggml_backend_alloc_buffer(backend, 1);

    // TODO: move alloc initialization to a common wsp_ggml_tallocr_new_impl function
    wsp_ggml_tallocr_t alloc = wsp_ggml_tallocr_new_from_buffer(buffer);
    alloc->buffer_owned = true;
    alloc->measure = true;
    wsp_ggml_tallocr_reset(alloc);
    return alloc;
}

wsp_ggml_tallocr_t wsp_ggml_tallocr_new_from_backend(struct wsp_ggml_backend * backend, size_t size) {
    wsp_ggml_backend_buffer_t buffer = wsp_ggml_backend_alloc_buffer(backend, size);
    wsp_ggml_tallocr_t alloc = wsp_ggml_tallocr_new_from_buffer(buffer);
    alloc->buffer_owned = true;
    return alloc;
}

wsp_ggml_tallocr_t wsp_ggml_tallocr_new_from_buffer(struct wsp_ggml_backend_buffer * buffer) {
    wsp_ggml_tallocr_t alloc = (wsp_ggml_tallocr_t)malloc(sizeof(struct wsp_ggml_tallocr));

    *alloc = (struct wsp_ggml_tallocr) {
        /*.buffer        = */ buffer,
        /*.buffer_owned  = */ false,
        /*.base          = */ wsp_ggml_backend_buffer_get_base(buffer),
        /*.alignment     = */ wsp_ggml_backend_buffer_get_alignment(buffer),
        /*.n_free_blocks = */ 0,
        /*.free_blocks   = */ {{0}},
        /*.max_size      = */ 0,
        /*.measure       = */ false,
#ifdef WSP_GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {0},
#endif
    };

    wsp_ggml_tallocr_reset(alloc);

    return alloc;
}

struct wsp_ggml_backend_buffer * wsp_ggml_tallocr_get_buffer(wsp_ggml_tallocr_t alloc) {
    return alloc->buffer;
}

void wsp_ggml_tallocr_free(wsp_ggml_tallocr_t alloc) {
    if (alloc == NULL) {
        return;
    }

    if (alloc->buffer_owned) {
        wsp_ggml_backend_buffer_free(alloc->buffer);
    }
    free(alloc);
}

bool wsp_ggml_tallocr_is_measure(wsp_ggml_tallocr_t alloc) {
    return alloc->measure;
}

size_t wsp_ggml_tallocr_max_size(wsp_ggml_tallocr_t alloc) {
    return alloc->max_size;
}

// graph allocator

struct hash_node {
    int n_children;
    int n_views;
};

struct wsp_ggml_gallocr {
    wsp_ggml_tallocr_t talloc;
    struct wsp_ggml_hash_set hash_set;
    struct hash_node * hash_values;
    size_t hash_values_size;
    wsp_ggml_tallocr_t * hash_allocs;
    int * parse_seq;
    int parse_seq_len;
};

wsp_ggml_gallocr_t wsp_ggml_gallocr_new(void) {
    wsp_ggml_gallocr_t galloc = (wsp_ggml_gallocr_t)malloc(sizeof(struct wsp_ggml_gallocr));

    *galloc = (struct wsp_ggml_gallocr) {
        /*.talloc           = */ NULL,
        /*.hash_set         = */ {0},
        /*.hash_values      = */ NULL,
        /*.hash_values_size = */ 0,
        /*.hash_allocs      = */ NULL,
        /*.parse_seq        = */ NULL,
        /*.parse_seq_len    = */ 0,
    };

    return galloc;
}

void wsp_ggml_gallocr_free(wsp_ggml_gallocr_t galloc) {
    if (galloc == NULL) {
        return;
    }

    if (galloc->hash_set.keys != NULL) {
        free(galloc->hash_set.keys);
    }
    if (galloc->hash_values != NULL) {
        free(galloc->hash_values);
    }
    if (galloc->hash_allocs != NULL) {
        free(galloc->hash_allocs);
    }
    if (galloc->parse_seq != NULL) {
        free(galloc->parse_seq);
    }
    free(galloc);
}

void wsp_ggml_gallocr_set_parse_seq(wsp_ggml_gallocr_t galloc, const int * list, int n) {
    free(galloc->parse_seq);
    galloc->parse_seq = malloc(sizeof(int) * n);

    for (int i = 0; i < n; i++) {
        galloc->parse_seq[i] = list[i];
    }
    galloc->parse_seq_len = n;
}

static struct hash_node * hash_get(wsp_ggml_gallocr_t galloc, struct wsp_ggml_tensor * t) {
    size_t i = wsp_ggml_hash_find_or_insert(galloc->hash_set, t);
    return &galloc->hash_values[i];
}

static bool wsp_ggml_are_same_layout(const struct wsp_ggml_tensor * a, const struct wsp_ggml_tensor * b) {
    if (a->type != b->type) {
        return false;
    }
    for (int i = 0; i < WSP_GGML_MAX_DIMS; i++) {
        if (a->ne[i] != b->ne[i]) {
            return false;
        }
        if (a->nb[i] != b->nb[i]) {
            return false;
        }
    }
    return true;
}

static bool wsp_ggml_op_can_inplace(enum wsp_ggml_op op) {
    switch (op) {
        case WSP_GGML_OP_SCALE:
        case WSP_GGML_OP_DIAG_MASK_ZERO:
        case WSP_GGML_OP_DIAG_MASK_INF:
        case WSP_GGML_OP_ADD:
        case WSP_GGML_OP_ADD1:
        case WSP_GGML_OP_SUB:
        case WSP_GGML_OP_MUL:
        case WSP_GGML_OP_DIV:
        case WSP_GGML_OP_SQR:
        case WSP_GGML_OP_SQRT:
        case WSP_GGML_OP_LOG:
        case WSP_GGML_OP_UNARY:
        case WSP_GGML_OP_ROPE:
        case WSP_GGML_OP_RMS_NORM:
        case WSP_GGML_OP_SOFT_MAX:
            return true;

        default:
            return false;
    }
}

static wsp_ggml_tallocr_t node_tallocr(wsp_ggml_gallocr_t galloc, struct wsp_ggml_tensor * node) {
    if (galloc->talloc != NULL) {
        return galloc->talloc;
    }

    return galloc->hash_allocs[wsp_ggml_hash_find_or_insert(galloc->hash_set, node)];
}

static void init_view(wsp_ggml_gallocr_t galloc, struct wsp_ggml_tensor * view, bool update_backend) {
    wsp_ggml_tallocr_t alloc = node_tallocr(galloc, view);

    WSP_GGML_ASSERT(view->view_src != NULL && view->view_src->data != NULL);
    if (update_backend) {
        view->backend = view->view_src->backend;
    }
    view->buffer  = view->view_src->buffer;
    view->data    = (char *)view->view_src->data + view->view_offs;

    // FIXME: the view should be initialized by the owning buffer, but currently this breaks the CUDA backend
    // due to the wsp_ggml_tensor_extra_gpu ring buffer overwriting the KV cache extras
    assert(wsp_ggml_tallocr_is_measure(alloc) || !view->buffer || view->buffer->buft == alloc->buffer->buft);

    if (!alloc->measure) {
        wsp_ggml_backend_buffer_init_tensor(alloc->buffer, view);
    }
}

static void allocate_node(wsp_ggml_gallocr_t galloc, struct wsp_ggml_tensor * node) {
    wsp_ggml_tallocr_t alloc = node_tallocr(galloc, node);

    if (node->data == NULL) {
        if (wsp_ggml_is_view(node)) {
            init_view(galloc, node, true);
        } else {
            // see if we can reuse a parent's buffer (inplace)
            if (wsp_ggml_op_can_inplace(node->op)) {
                for (int i = 0; i < WSP_GGML_MAX_SRC; i++) {
                    struct wsp_ggml_tensor * parent = node->src[i];
                    if (parent == NULL) {
                        break;
                    }

                    // if the node's data is external, then we cannot re-use it
                    if (wsp_ggml_tallocr_is_own(alloc, parent) == false) {
                        AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, node->name, parent->data);
                        continue;
                    }

                    struct hash_node * p_hn = hash_get(galloc, parent);
                    if (parent->data != NULL && p_hn->n_children == 1 && p_hn->n_views == 0 && wsp_ggml_are_same_layout(node, parent)) {
                        if (wsp_ggml_is_view(parent)) {
                            struct wsp_ggml_tensor * view_src = parent->view_src;
                            struct hash_node * view_src_hn = hash_get(galloc, view_src);
                            if (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
                                // TODO: the offset of the view parent must be kept to ensure that the op doesn't overwrite
                                // the parent's data that it will need later (same layout requirement). the problem is that then
                                // we cannot free the tensor because the original address of the allocation is lost.
                                // adding a view_src pointer to the tensor would solve this and simplify the code dealing with views
                                // for now, we only reuse the parent's data if the offset is zero (view_src->data == parent->data)
                                AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, node->name);
                                node->view_src = view_src;
                                view_src_hn->n_views += 1;
                                init_view(galloc, node, false);
                                return;
                            }
                        } else {
                            AT_PRINTF("reusing parent %s for %s\n", parent->name, node->name);
                            node->view_src = parent;
                            p_hn->n_views += 1;
                            init_view(galloc, node, false);
                            return;
                        }
                    }
                }
            }
            wsp_ggml_tallocr_alloc(alloc, node);
        }
    }
}

static void free_node(wsp_ggml_gallocr_t galloc, struct wsp_ggml_tensor * node) {
    wsp_ggml_tallocr_t alloc = node_tallocr(galloc, node);

    wsp_ggml_tallocr_free_tensor(alloc, node);
}

static void wsp_ggml_tallocr_alloc_graph_impl(wsp_ggml_gallocr_t galloc, struct wsp_ggml_cgraph * gf) {
    const int * parse_seq     = galloc->parse_seq;
    int         parse_seq_len = galloc->parse_seq_len;

    // count number of children and views
    for (int i = 0; i < gf->n_nodes; i++) {
        struct wsp_ggml_tensor * node = gf->nodes[i];

        if (wsp_ggml_is_view(node)) {
            struct wsp_ggml_tensor * view_src = node->view_src;
            hash_get(galloc, view_src)->n_views += 1;
            if (node->buffer == NULL && node->data != NULL) {
                // view of a pre-allocated tensor, didn't call init_view() yet
                init_view(galloc, node, true);
            }
        }

        for (int j = 0; j < WSP_GGML_MAX_SRC; j++) {
            struct wsp_ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                break;
            }
            hash_get(galloc, parent)->n_children += 1;
            if (wsp_ggml_is_view(parent) && parent->buffer == NULL && parent->data != NULL) {
                init_view(galloc, parent, true);
            }
        }
   }

    // allocate tensors
    // if we have parse_seq then we allocate nodes following the list, and we only free nodes at barriers
    int last_barrier_pos = 0;
    int n_nodes = parse_seq_len ? parse_seq_len : gf->n_nodes;

    for (int ind = 0; ind < n_nodes; ind++) {
        // allocate a node if there is no parse_seq or this is not a barrier
        if (parse_seq_len == 0 || parse_seq[ind] != -1) {
            int i = parse_seq_len ? parse_seq[ind] : ind;
            struct wsp_ggml_tensor * node = gf->nodes[i];

            // allocate parents (leafs)
            for (int j = 0; j < WSP_GGML_MAX_SRC; j++) {
                struct wsp_ggml_tensor * parent = node->src[j];
                if (parent == NULL) {
                    break;
                }
                allocate_node(galloc, parent);
            }

            // allocate node
            allocate_node(galloc, node);

            AT_PRINTF("exec: %s (%s) <= ", wsp_ggml_op_name(node->op), node->name);
            for (int j = 0; j < WSP_GGML_MAX_SRC; j++) {
                struct wsp_ggml_tensor * parent = node->src[j];
                if (parent == NULL) {
                    break;
                }
                AT_PRINTF("%s", parent->name);
                if (j < WSP_GGML_MAX_SRC - 1 && node->src[j + 1] != NULL) {
                    AT_PRINTF(", ");
                }
            }
            AT_PRINTF("\n");
        }

        // update parents
        // update immediately if there is no parse_seq
        // update only at barriers if there is parse_seq
        if ((parse_seq_len == 0) || parse_seq[ind] == -1) {
            int update_start = parse_seq_len ? last_barrier_pos : ind;
            int update_end   = parse_seq_len ? ind              : ind + 1;
            for (int i = update_start; i < update_end; i++) {
                int node_i = parse_seq_len ? parse_seq[i] : i;
                struct wsp_ggml_tensor * node = gf->nodes[node_i];

                for (int j = 0; j < WSP_GGML_MAX_SRC; j++) {
                    struct wsp_ggml_tensor * parent = node->src[j];
                    if (parent == NULL) {
                        break;
                    }
                    struct hash_node * p_hn = hash_get(galloc, parent);
                    p_hn->n_children -= 1;

                    //AT_PRINTF("parent %s: %d children, %d views\n", parent->name, parent->n_children, parent->n_views);

                    if (p_hn->n_children == 0 && p_hn->n_views == 0) {
                        if (wsp_ggml_is_view(parent)) {
                            struct wsp_ggml_tensor * view_src = parent->view_src;
                            struct hash_node * view_src_hn = hash_get(galloc, view_src);
                            view_src_hn->n_views -= 1;
                            AT_PRINTF("view_src %s: %d children, %d views\n", view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                            if (view_src_hn->n_views == 0 && view_src_hn->n_children == 0) {
                                free_node(galloc, view_src);
                            }
                        }
                        else {
                            free_node(galloc, parent);
                        }
                    }
                }
            }
            AT_PRINTF("\n");
            if (parse_seq_len) {
                last_barrier_pos = ind + 1;
            }
        }
    }
}

size_t wsp_ggml_gallocr_alloc_graph(wsp_ggml_gallocr_t galloc, wsp_ggml_tallocr_t talloc, struct wsp_ggml_cgraph * graph) {
    size_t hash_size = graph->visited_hash_table.size;

    // check if the hash table is initialized and large enough
    if (galloc->hash_set.size < hash_size) {
        if (galloc->hash_set.keys != NULL) {
            free(galloc->hash_set.keys);
        }
        if (galloc->hash_values != NULL) {
            free(galloc->hash_values);
        }
        galloc->hash_set.keys = malloc(sizeof(struct wsp_ggml_tensor *) * hash_size);
        galloc->hash_set.size = hash_size;
        galloc->hash_values = malloc(sizeof(struct hash_node) * hash_size);
    }

    // reset hash table
    memset(galloc->hash_set.keys, 0, sizeof(struct wsp_ggml_tensor *) * hash_size);
    memset(galloc->hash_values,   0, sizeof(struct hash_node) * hash_size);

    galloc->talloc = talloc;
    wsp_ggml_tallocr_alloc_graph_impl(galloc, graph);
    galloc->talloc = NULL;

    size_t max_size = wsp_ggml_tallocr_max_size(talloc);

    return max_size;
}

void wsp_ggml_gallocr_alloc_graph_n(wsp_ggml_gallocr_t galloc, struct wsp_ggml_cgraph * graph, struct wsp_ggml_hash_set hash_set, wsp_ggml_tallocr_t * hash_node_talloc) {
    const size_t hash_size = hash_set.size;

    WSP_GGML_ASSERT(hash_size >= (size_t)(graph->n_nodes + graph->n_leafs));

    galloc->talloc = NULL;

    // alloc hash_values if needed
    if (galloc->hash_values == NULL || galloc->hash_values_size < hash_size) {
        free(galloc->hash_values);
        galloc->hash_values      = malloc(sizeof(struct hash_node) * hash_size);
        galloc->hash_values_size = hash_size;
    }

    // free hash_set.keys if needed
    if (galloc->hash_set.keys != NULL) {
        free(galloc->hash_set.keys);
    }
    galloc->hash_set = hash_set;

    // reset hash values
    memset(galloc->hash_values, 0, sizeof(struct hash_node) * hash_size);

    galloc->hash_allocs = hash_node_talloc;

    wsp_ggml_tallocr_alloc_graph_impl(galloc, graph);

    // remove unowned resources
    galloc->hash_set.keys = NULL;
    galloc->hash_allocs = NULL;
}

// legacy API wrapper

struct wsp_ggml_allocr {
    wsp_ggml_tallocr_t talloc;
    wsp_ggml_gallocr_t galloc;
};

static wsp_ggml_allocr_t wsp_ggml_allocr_new_impl(wsp_ggml_tallocr_t talloc) {
    wsp_ggml_allocr_t alloc = (wsp_ggml_allocr_t)malloc(sizeof(struct wsp_ggml_allocr));
    *alloc = (struct wsp_ggml_allocr) {
        /*.talloc = */ talloc,
        /*.galloc = */ wsp_ggml_gallocr_new(),
    };
    return alloc;
}

wsp_ggml_allocr_t wsp_ggml_allocr_new(void * data, size_t size, size_t alignment) {
    return wsp_ggml_allocr_new_impl(wsp_ggml_tallocr_new(data, size, alignment));
}

wsp_ggml_allocr_t wsp_ggml_allocr_new_measure(size_t alignment) {
    return wsp_ggml_allocr_new_impl(wsp_ggml_tallocr_new_measure(alignment));
}

wsp_ggml_allocr_t wsp_ggml_allocr_new_from_buffer(struct wsp_ggml_backend_buffer * buffer) {
    return wsp_ggml_allocr_new_impl(wsp_ggml_tallocr_new_from_buffer(buffer));
}

wsp_ggml_allocr_t wsp_ggml_allocr_new_from_backend(struct wsp_ggml_backend * backend, size_t size) {
    return wsp_ggml_allocr_new_impl(wsp_ggml_tallocr_new_from_backend(backend, size));
}

wsp_ggml_allocr_t wsp_ggml_allocr_new_measure_from_backend(struct wsp_ggml_backend * backend) {
    return wsp_ggml_allocr_new_impl(wsp_ggml_tallocr_new_measure_from_backend(backend));
}

struct wsp_ggml_backend_buffer * wsp_ggml_allocr_get_buffer(wsp_ggml_allocr_t alloc) {
    return wsp_ggml_tallocr_get_buffer(alloc->talloc);
}

void wsp_ggml_allocr_set_parse_seq(wsp_ggml_allocr_t alloc, const int * list, int n) {
    wsp_ggml_gallocr_set_parse_seq(alloc->galloc, list, n);
}

void wsp_ggml_allocr_free(wsp_ggml_allocr_t alloc) {
    wsp_ggml_gallocr_free(alloc->galloc);
    wsp_ggml_tallocr_free(alloc->talloc);
    free(alloc);
}

bool wsp_ggml_allocr_is_measure(wsp_ggml_allocr_t alloc) {
    return wsp_ggml_tallocr_is_measure(alloc->talloc);
}

void wsp_ggml_allocr_reset(wsp_ggml_allocr_t alloc) {
    wsp_ggml_tallocr_reset(alloc->talloc);
}

void wsp_ggml_allocr_alloc(wsp_ggml_allocr_t alloc, struct wsp_ggml_tensor * tensor) {
    wsp_ggml_tallocr_alloc(alloc->talloc, tensor);
}

size_t wsp_ggml_allocr_max_size(wsp_ggml_allocr_t alloc) {
    return wsp_ggml_tallocr_max_size(alloc->talloc);
}

size_t wsp_ggml_allocr_alloc_graph(wsp_ggml_allocr_t alloc, struct wsp_ggml_cgraph * graph) {
    return wsp_ggml_gallocr_alloc_graph(alloc->galloc, alloc->talloc, graph);
}

// utils
wsp_ggml_backend_buffer_t wsp_ggml_backend_alloc_ctx_tensors_from_buft(struct wsp_ggml_context * ctx, wsp_ggml_backend_buffer_type_t buft) {
    WSP_GGML_ASSERT(wsp_ggml_get_no_alloc(ctx) == true);

    size_t alignment = wsp_ggml_backend_buft_get_alignment(buft);

    size_t nbytes = 0;
    for (struct wsp_ggml_tensor * t = wsp_ggml_get_first_tensor(ctx); t != NULL; t = wsp_ggml_get_next_tensor(ctx, t)) {
        if (t->data == NULL && t->view_src == NULL) {
            nbytes += WSP_GGML_PAD(wsp_ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }
    }

    if (nbytes == 0) {
        fprintf(stderr, "%s: no tensors to allocate\n", __func__);
        return NULL;
    }

    wsp_ggml_backend_buffer_t buffer = wsp_ggml_backend_buft_alloc_buffer(buft, nbytes);
    wsp_ggml_tallocr_t tallocr = wsp_ggml_tallocr_new_from_buffer(buffer);

    for (struct wsp_ggml_tensor * t = wsp_ggml_get_first_tensor(ctx); t != NULL; t = wsp_ggml_get_next_tensor(ctx, t)) {
        if (t->data == NULL) {
            if (t->view_src == NULL) {
                wsp_ggml_tallocr_alloc(tallocr, t);
            } else {
                wsp_ggml_backend_view_init(buffer, t);
            }
        }
    }

    wsp_ggml_tallocr_free(tallocr);

    return buffer;
}

wsp_ggml_backend_buffer_t wsp_ggml_backend_alloc_ctx_tensors(struct wsp_ggml_context * ctx, wsp_ggml_backend_t backend) {
    return wsp_ggml_backend_alloc_ctx_tensors_from_buft(ctx, wsp_ggml_backend_get_default_buffer_type(backend));
}
