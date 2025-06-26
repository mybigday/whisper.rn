#include "ggml-opt.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cinttypes>
#include <map>
#include <random>
#include <vector>

struct wsp_ggml_opt_dataset {
    struct wsp_ggml_context   * ctx    = nullptr;
    wsp_ggml_backend_buffer_t   buf    = nullptr;
    struct wsp_ggml_tensor    * data   = nullptr;
    struct wsp_ggml_tensor    * labels = nullptr;

    int64_t ndata       = -1;
    int64_t ndata_shard = -1;
    size_t  nbs_data    = -1;
    size_t  nbs_labels  = -1;

    std::vector<int64_t> permutation;
};

struct wsp_ggml_opt_context {
    wsp_ggml_backend_sched_t       backend_sched        = nullptr;
    wsp_ggml_cgraph              * allocated_graph      = nullptr;
    wsp_ggml_cgraph              * allocated_graph_copy = nullptr;
    struct wsp_ggml_context      * ctx_static           = nullptr;
    struct wsp_ggml_context      * ctx_cpu              = nullptr;
    struct wsp_ggml_context      * ctx_compute          = nullptr;
    struct wsp_ggml_context      * ctx_copy             = nullptr;
    wsp_ggml_backend_buffer_t      buf_static           = nullptr;
    wsp_ggml_backend_buffer_t      buf_cpu              = nullptr;
    std::mt19937               rng;
    enum wsp_ggml_opt_loss_type    loss_type;
    enum wsp_ggml_opt_build_type   build_type;
    enum wsp_ggml_opt_build_type   build_type_alloc;

    struct wsp_ggml_tensor * inputs  = nullptr;
    struct wsp_ggml_tensor * outputs = nullptr;
    struct wsp_ggml_tensor * labels  = nullptr;

    struct wsp_ggml_tensor * loss     = nullptr;
    struct wsp_ggml_tensor * pred     = nullptr;
    struct wsp_ggml_tensor * ncorrect = nullptr;

    struct wsp_ggml_cgraph * gf      = nullptr;
    struct wsp_ggml_cgraph * gb_grad = nullptr;
    struct wsp_ggml_cgraph * gb_opt  = nullptr;
    bool static_graphs           = false;
    bool eval_ready              = false;
    std::vector<struct wsp_ggml_tensor *> grad_accs;
    std::vector<struct wsp_ggml_tensor *> grad_m;
    std::vector<struct wsp_ggml_tensor *> grad_v;

    int64_t iter               = 1;
    int32_t opt_period         = 1;
    int32_t opt_i              = 0;
    bool    loss_per_datapoint = false;

    wsp_ggml_opt_get_optimizer_params get_opt_pars = nullptr;
    void * get_opt_pars_ud                     = nullptr;
    struct wsp_ggml_tensor * adamw_params          = nullptr;
};

struct wsp_ggml_opt_result {
    int64_t              ndata    = 0;
    std::vector<float>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;

    int64_t opt_period         = -1;
    bool    loss_per_datapoint = false;
};

// ====== Dataset ======

wsp_ggml_opt_dataset_t wsp_ggml_opt_dataset_init(
        enum wsp_ggml_type type_data,
        enum wsp_ggml_type type_label,
        int64_t        ne_datapoint,
        int64_t        ne_label,
        int64_t        ndata,
        int64_t        ndata_shard) {
    WSP_GGML_ASSERT(ne_datapoint >  0);
    WSP_GGML_ASSERT(ne_label     >= 0);
    WSP_GGML_ASSERT(ndata        >  0);
    WSP_GGML_ASSERT(ndata_shard  >  0);

    wsp_ggml_opt_dataset_t result = new wsp_ggml_opt_dataset;
    result->ndata       = ndata;
    result->ndata_shard = ndata_shard;

    {
        struct wsp_ggml_init_params params = {
            /*.mem_size   =*/ 2*wsp_ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        result->ctx = wsp_ggml_init(params);
    }

    result->data = wsp_ggml_new_tensor_2d(result->ctx, type_data, ne_datapoint, ndata);
    result->nbs_data = wsp_ggml_nbytes(result->data) * ndata_shard/ndata;

    if (ne_label > 0) {
        result->labels = wsp_ggml_new_tensor_2d(result->ctx, type_label, ne_label, ndata);
        result->nbs_labels = wsp_ggml_nbytes(result->labels) * ndata_shard/ndata;
    } else {
        result->labels = nullptr;
        result->nbs_labels = 0;
    }

    result->buf = wsp_ggml_backend_alloc_ctx_tensors_from_buft(result->ctx, wsp_ggml_backend_cpu_buffer_type());

    const int64_t nshards = ndata/ndata_shard;
    result->permutation.resize(nshards);
    for (int64_t i = 0; i < nshards; ++i) {
        result->permutation[i] = i;
    }
    return result;
}

void wsp_ggml_opt_dataset_free(wsp_ggml_opt_dataset_t dataset) {
    wsp_ggml_backend_buffer_free(dataset->buf);
    wsp_ggml_free(dataset->ctx);
    delete dataset;
}

int64_t wsp_ggml_opt_dataset_ndata(wsp_ggml_opt_dataset_t dataset) {
    return dataset->ndata;
}

struct wsp_ggml_tensor * wsp_ggml_opt_dataset_data(wsp_ggml_opt_dataset_t dataset) {
    return dataset->data;
}

struct wsp_ggml_tensor * wsp_ggml_opt_dataset_labels(wsp_ggml_opt_dataset_t dataset) {
    return dataset->labels;
}

void wsp_ggml_opt_dataset_shuffle(wsp_ggml_opt_context_t opt_ctx, wsp_ggml_opt_dataset_t dataset, int64_t idata) {
    WSP_GGML_ASSERT(idata <= dataset->ndata);

    if (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        return;
    }

    WSP_GGML_ASSERT(idata % dataset->ndata_shard == 0);
    const int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}

void wsp_ggml_opt_dataset_get_batch(wsp_ggml_opt_dataset_t dataset, struct wsp_ggml_tensor * data_batch, struct wsp_ggml_tensor * labels_batch, int64_t ibatch) {
    WSP_GGML_ASSERT(   data_batch && wsp_ggml_is_contiguous(data_batch));
    WSP_GGML_ASSERT(!labels_batch || wsp_ggml_is_contiguous(labels_batch));
    WSP_GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
    WSP_GGML_ASSERT(                   data_batch->type == dataset->data->type);
    WSP_GGML_ASSERT(!labels_batch || labels_batch->type == dataset->labels->type);

    const size_t nb_data_batch = wsp_ggml_nbytes(data_batch);
    WSP_GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);
    const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    if (labels_batch) {
        const size_t nb_labels_batch = wsp_ggml_nbytes(labels_batch);
        WSP_GGML_ASSERT(nb_labels_batch == shards_per_batch*dataset->nbs_labels);
    }

    WSP_GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        const char * ptr_data = (const char *) dataset->data->data + ishard*dataset->nbs_data;
        wsp_ggml_backend_tensor_set(data_batch, ptr_data, ishard_batch*dataset->nbs_data, dataset->nbs_data);

        if (!labels_batch) {
            continue;
        }

        const char * ptr_labels = (const char *) dataset->labels->data + ishard*dataset->nbs_labels;
        wsp_ggml_backend_tensor_set(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels);
    }
}

void wsp_ggml_opt_dataset_get_batch_host(wsp_ggml_opt_dataset_t dataset, void * data_batch, size_t nb_data_batch, void * labels_batch, int64_t ibatch) {
    WSP_GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
    WSP_GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);

    const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    WSP_GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        const char * ptr_data       = (const char *) dataset->data->data + ishard      *dataset->nbs_data;
        char       * ptr_data_batch = (char       *) data_batch          + ishard_batch*dataset->nbs_data;
        memcpy(ptr_data_batch, ptr_data, dataset->nbs_data);

        if (!labels_batch) {
            continue;
        }

        const char * ptr_labels       = (const char *) dataset->labels->data + ishard      *dataset->nbs_labels;
        char       * ptr_labels_batch = (char       *) labels_batch          + ishard_batch*dataset->nbs_labels;
        memcpy(ptr_labels_batch, ptr_labels, dataset->nbs_labels);
    }
}

// ====== Model / Context ======

struct wsp_ggml_opt_optimizer_params wsp_ggml_opt_get_default_optimizer_params(void * userdata) {
    WSP_GGML_UNUSED(userdata);

    wsp_ggml_opt_optimizer_params result;

    result.adamw.alpha = 0.001f;
    result.adamw.beta1 = 0.9f;
    result.adamw.beta2 = 0.999f;
    result.adamw.eps   = 1e-8f;
    result.adamw.wd    = 0.0f;

    return result;
}

struct wsp_ggml_opt_optimizer_params wsp_ggml_opt_get_constant_optimizer_params(void * userdata) {
    return *((struct wsp_ggml_opt_optimizer_params *) userdata);
}

struct wsp_ggml_opt_params wsp_ggml_opt_default_params(
        wsp_ggml_backend_sched_t      backend_sched,
        enum wsp_ggml_opt_loss_type   loss_type) {
    return {
        /*backend_sched   =*/ backend_sched,
        /*ctx_compute     =*/ nullptr,
        /*inputs          =*/ nullptr,
        /*logits          =*/ nullptr,
        /*loss_type       =*/ loss_type,
        /*build_type      =*/ WSP_GGML_OPT_BUILD_TYPE_OPT,
        /*opt_period      =*/ 1,
        /*get_opt_pars    =*/ wsp_ggml_opt_get_default_optimizer_params,
        /*get_opt_pars_ud =*/ nullptr,
    };
}

static wsp_ggml_tensor * map_tensor(std::map<wsp_ggml_tensor *, wsp_ggml_tensor *> & tensor_map, wsp_ggml_context * ctx, wsp_ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }

    if (tensor_map.find(tensor) != tensor_map.end()) {
        return tensor_map[tensor];
    }

    wsp_ggml_tensor * new_tensor = wsp_ggml_dup_tensor(ctx, tensor);
    tensor_map[tensor] = new_tensor;

    new_tensor->op = tensor->op;
    for (int i = 0; i < WSP_GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = tensor->nb[i];
    }
    new_tensor->flags = tensor->flags;
    memcpy(new_tensor->op_params, tensor->op_params, sizeof(tensor->op_params));
    strcpy(new_tensor->name, tensor->name);
    new_tensor->data = tensor->data;
    new_tensor->buffer = tensor->buffer;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    for (int i = 0; i < WSP_GGML_MAX_SRC; i++) {
        new_tensor->src[i] = map_tensor(tensor_map, ctx, tensor->src[i]);
    }

    return new_tensor;
}

static wsp_ggml_cgraph * dup_graph(wsp_ggml_context * ctx, wsp_ggml_cgraph * src) {
    std::map<wsp_ggml_tensor *, wsp_ggml_tensor *> tensor_map;

    wsp_ggml_cgraph * dst = wsp_ggml_new_graph_custom(ctx, src->size, /*grads =*/ true);

    for (int i = 0; i < src->n_leafs; i++) {
        wsp_ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->leafs[i]));
    }
    WSP_GGML_ASSERT(dst->n_leafs == src->n_leafs);
    for (int i = 0; i < src->n_nodes; i++) {
        wsp_ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->nodes[i]));
    }
    WSP_GGML_ASSERT(dst->n_nodes == src->n_nodes);
    for (int i = 0; i < src->n_nodes; ++i) {
        const size_t igrad_src = wsp_ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
        const size_t igrad_dst = wsp_ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

        WSP_GGML_ASSERT(igrad_src != WSP_GGML_HASHSET_FULL);
        WSP_GGML_ASSERT(wsp_ggml_bitset_get(src->visited_hash_set.used, igrad_src));
        WSP_GGML_ASSERT(igrad_dst != WSP_GGML_HASHSET_FULL);
        WSP_GGML_ASSERT(wsp_ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

        dst->grads[igrad_dst]     = src->grads[igrad_src];
        dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }

    return dst;
}

static void wsp_ggml_opt_build(wsp_ggml_opt_context_t opt_ctx) {
    WSP_GGML_ASSERT(opt_ctx->ctx_compute && "no compute context set, either use static graphs or set one with wsp_ggml_opt_prepare_alloc");
    WSP_GGML_ASSERT((!opt_ctx->static_graphs || opt_ctx->inputs->data) && "when using static graphs the inputs must be allocated statically");

    const bool accumulate = opt_ctx->build_type_alloc >= WSP_GGML_OPT_BUILD_TYPE_GRAD &&
        !(opt_ctx->static_graphs && opt_ctx->build_type_alloc == WSP_GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period == 1);

    wsp_ggml_set_input(opt_ctx->inputs);
    wsp_ggml_set_output(opt_ctx->outputs);

    int n_param = 0;
    for (int i = 0; i < opt_ctx->gf->n_nodes; ++i) {
        const struct wsp_ggml_tensor * node = opt_ctx->gf->nodes[i];
        if (node->flags & WSP_GGML_TENSOR_FLAG_PARAM) {
            n_param++;
        }
        WSP_GGML_ASSERT(!(node->flags & WSP_GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
    }

    if (!opt_ctx->ctx_static) {
        // The static context is used for:
        //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
        //   - optimizer momenta (2 tensors per param)
        //   - labels (if using static graphs)
        //   - loss (if using static graphs, up to 5 tensors)
        //   - pred (if using static graphs)
        //   - ncorrect (if using static graphs, 2 tensors).
        constexpr size_t n_loss = 1;
        const size_t tensors_per_param = (accumulate ? 1 : 0) +
            (opt_ctx->build_type_alloc == WSP_GGML_OPT_BUILD_TYPE_OPT ? 2 : 0);
        const size_t tensors_const = opt_ctx->static_graphs ? 9 : 0;
        const size_t size_meta = (n_loss + tensors_per_param*n_param + tensors_const) * wsp_ggml_tensor_overhead();
        struct wsp_ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        opt_ctx->ctx_static = wsp_ggml_init(params);
    }
    WSP_GGML_ASSERT(opt_ctx->build_type <= opt_ctx->build_type_alloc);

    {
        // The cpu context is allocated statically if using static graphs, dynamically otherwise.
        // It is used for:
        //   - optimizer parameters (1 shared for all optimizer invocations)
        const size_t size_meta = 1 * wsp_ggml_tensor_overhead();
        struct wsp_ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        wsp_ggml_free(opt_ctx->ctx_cpu);
        opt_ctx->ctx_cpu = wsp_ggml_init(params);

        wsp_ggml_backend_buffer_free(opt_ctx->buf_cpu);
        opt_ctx->buf_cpu = nullptr;
    }

    struct wsp_ggml_context * ctx_results = opt_ctx->static_graphs ? opt_ctx->ctx_static : opt_ctx->ctx_compute;

    switch (opt_ctx->loss_type) {
        case WSP_GGML_OPT_LOSS_TYPE_MEAN: {
            opt_ctx->loss = wsp_ggml_sum(ctx_results, opt_ctx->outputs);
            wsp_ggml_set_name(opt_ctx->loss, "loss_sum");
            const float scale = 1.0f / (opt_ctx->opt_period * wsp_ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = wsp_ggml_scale(ctx_results, opt_ctx->loss, scale);
            wsp_ggml_set_name(opt_ctx->loss, "loss_mean");
            opt_ctx->loss_per_datapoint = true;
            break;
        }
        case WSP_GGML_OPT_LOSS_TYPE_SUM: {
            opt_ctx->loss = wsp_ggml_sum(ctx_results, opt_ctx->outputs);
            wsp_ggml_set_name(opt_ctx->loss, "loss_sum");
            opt_ctx->loss_per_datapoint = false;
            break;
        }
        case WSP_GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
            opt_ctx->labels = wsp_ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            wsp_ggml_set_input(opt_ctx->labels);
            wsp_ggml_set_name(opt_ctx->labels, "labels");
            opt_ctx->loss = wsp_ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            wsp_ggml_set_name(opt_ctx->loss, "loss_cross_entropy");
            if (opt_ctx->opt_period > 1) {
                opt_ctx->loss = wsp_ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
                wsp_ggml_set_name(opt_ctx->loss, "loss_cross_entropy_scaled");
            }
            opt_ctx->loss_per_datapoint = true;
            break;
        }
        case WSP_GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR: {
            opt_ctx->labels = wsp_ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            wsp_ggml_set_input(opt_ctx->labels);
            wsp_ggml_set_name(opt_ctx->labels, "labels");
            opt_ctx->loss = wsp_ggml_sub(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            wsp_ggml_set_name(opt_ctx->loss, "loss_error");
            opt_ctx->loss = wsp_ggml_sqr(ctx_results, opt_ctx->loss);
            wsp_ggml_set_name(opt_ctx->loss, "loss_squared_error");
            opt_ctx->loss = wsp_ggml_sum(ctx_results, opt_ctx->loss);
            wsp_ggml_set_name(opt_ctx->loss, "loss_sum_squared_error");
            const float scale = 1.0f / (opt_ctx->opt_period * wsp_ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = wsp_ggml_scale(ctx_results, opt_ctx->loss, scale);
            wsp_ggml_set_name(opt_ctx->loss, "loss_mean_squared_error");
            opt_ctx->loss_per_datapoint = true;
            break;
        }
    }
    wsp_ggml_set_output(opt_ctx->loss);
    wsp_ggml_set_loss(opt_ctx->loss);
    wsp_ggml_build_forward_expand(opt_ctx->gf, opt_ctx->loss);

    if (opt_ctx->loss_type == WSP_GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
        opt_ctx->pred = wsp_ggml_argmax(ctx_results, opt_ctx->outputs);
        wsp_ggml_set_name(opt_ctx->pred, "pred");
        wsp_ggml_set_output(opt_ctx->pred);
        wsp_ggml_build_forward_expand(opt_ctx->gf, opt_ctx->pred);

        opt_ctx->ncorrect = wsp_ggml_count_equal(ctx_results, opt_ctx->pred, wsp_ggml_argmax(ctx_results, opt_ctx->labels));
        wsp_ggml_set_name(opt_ctx->ncorrect, "ncorrect");
        wsp_ggml_set_output(opt_ctx->ncorrect);
        wsp_ggml_build_forward_expand(opt_ctx->gf, opt_ctx->ncorrect);
    }

    if (opt_ctx->buf_static) {
        if (opt_ctx->build_type == WSP_GGML_OPT_BUILD_TYPE_FORWARD) {
            return;
        }
    } else if (opt_ctx->build_type_alloc == WSP_GGML_OPT_BUILD_TYPE_FORWARD) {
        opt_ctx->buf_static = wsp_ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, wsp_ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        return;
    }

    if (opt_ctx->grad_accs.empty()) {
        WSP_GGML_ASSERT(opt_ctx->build_type_alloc >= WSP_GGML_OPT_BUILD_TYPE_GRAD);

        const int n_nodes = opt_ctx->gf->n_nodes;
        opt_ctx->grad_accs.resize(n_nodes);
        for (int i = 0; i < n_nodes; ++i) {
            wsp_ggml_tensor * node = opt_ctx->gf->nodes[i];
            if ((accumulate && (node->flags & WSP_GGML_TENSOR_FLAG_PARAM)) || (node->flags & WSP_GGML_TENSOR_FLAG_LOSS)) {
                opt_ctx->grad_accs[i] = wsp_ggml_new_tensor(opt_ctx->ctx_static, WSP_GGML_TYPE_F32, WSP_GGML_MAX_DIMS, node->ne);
            } else {
                opt_ctx->grad_accs[i] = nullptr;
            }
        }

        if (opt_ctx->build_type_alloc >= WSP_GGML_OPT_BUILD_TYPE_OPT) {
            opt_ctx->grad_m.resize(n_nodes);
            opt_ctx->grad_v.resize(n_nodes);
            for (int i = 0; i < n_nodes; ++i) {
                wsp_ggml_tensor * node = opt_ctx->gf->nodes[i];
                if (node->flags & WSP_GGML_TENSOR_FLAG_PARAM) {
                    opt_ctx->grad_m[i] = wsp_ggml_new_tensor(opt_ctx->ctx_static, WSP_GGML_TYPE_F32, WSP_GGML_MAX_DIMS, node->ne);
                    opt_ctx->grad_v[i] = wsp_ggml_new_tensor(opt_ctx->ctx_static, WSP_GGML_TYPE_F32, WSP_GGML_MAX_DIMS, node->ne);
                } else {
                    opt_ctx->grad_m[i] = nullptr;
                    opt_ctx->grad_v[i] = nullptr;
                }
            }
        }
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    opt_ctx->gb_grad = wsp_ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gf, /*force_grads =*/ true);
    wsp_ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.data());

    if (opt_ctx->buf_static) {
        if (opt_ctx->build_type == WSP_GGML_OPT_BUILD_TYPE_GRAD) {
            return;
        }
    } else if (opt_ctx->build_type_alloc == WSP_GGML_OPT_BUILD_TYPE_GRAD) {
        opt_ctx->buf_static = wsp_ggml_backend_alloc_ctx_tensors(opt_ctx->ctx_static, wsp_ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        wsp_ggml_graph_reset(opt_ctx->gb_grad);
    }

    WSP_GGML_ASSERT(opt_ctx->build_type_alloc == WSP_GGML_OPT_BUILD_TYPE_OPT);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    opt_ctx->gb_opt = wsp_ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gb_grad, /*force_grads =*/ true);

    opt_ctx->adamw_params = wsp_ggml_new_tensor_1d(opt_ctx->ctx_cpu, WSP_GGML_TYPE_F32, 7);
    wsp_ggml_set_input(opt_ctx->adamw_params);
    wsp_ggml_set_name(opt_ctx->adamw_params, "adamw_params");

    for (int i = opt_ctx->gf->n_nodes-1; i >= 0; --i) {
        struct wsp_ggml_tensor * node = opt_ctx->gb_opt->nodes[i];
        struct wsp_ggml_tensor * grad = wsp_ggml_graph_get_grad(opt_ctx->gb_opt, node);

        if (grad && (node->flags & WSP_GGML_TENSOR_FLAG_PARAM)) {
            struct wsp_ggml_tensor * m        = opt_ctx->grad_m[i];
            struct wsp_ggml_tensor * v        = opt_ctx->grad_v[i];
            struct wsp_ggml_tensor * opt_step = wsp_ggml_opt_step_adamw(opt_ctx->ctx_compute, node, grad, m, v, opt_ctx->adamw_params);

            wsp_ggml_set_name(m,        (std::string("AdamW m for ")    + std::string(node->name)).c_str());
            wsp_ggml_set_name(v,        (std::string("AdamW v for ")    + std::string(node->name)).c_str());
            wsp_ggml_set_name(opt_step, (std::string("AdamW step for ") + std::string(node->name)).c_str());

            wsp_ggml_build_forward_expand(opt_ctx->gb_opt, opt_step);
        }
    }

    if (!opt_ctx->buf_static) {
        opt_ctx->buf_static = wsp_ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, wsp_ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        wsp_ggml_graph_reset(opt_ctx->gb_opt);
    }

    opt_ctx->buf_cpu = wsp_ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_cpu, wsp_ggml_backend_cpu_buffer_type());
}

wsp_ggml_opt_context_t wsp_ggml_opt_init(struct wsp_ggml_opt_params params) {
    wsp_ggml_opt_context_t result = new struct wsp_ggml_opt_context;
    result->backend_sched    = params.backend_sched;
    result->ctx_compute      = params.ctx_compute;
    result->loss_type        = params.loss_type;
    result->build_type       = params.build_type;
    result->build_type_alloc = params.build_type;
    result->inputs           = params.inputs;
    result->outputs          = params.outputs;
    result->opt_period       = params.opt_period;
    result->get_opt_pars     = params.get_opt_pars;
    result->get_opt_pars_ud  = params.get_opt_pars_ud;

    WSP_GGML_ASSERT(result->opt_period >= 1);

    result->static_graphs = result->ctx_compute;

    if (!result->static_graphs) {
        WSP_GGML_ASSERT(!result->inputs);
        WSP_GGML_ASSERT(!result->outputs);
        return result;
    }

    WSP_GGML_ASSERT(result->inputs);
    WSP_GGML_ASSERT(result->outputs);

    result->gf = wsp_ggml_new_graph_custom(result->ctx_compute, WSP_GGML_DEFAULT_GRAPH_SIZE, /*grads =*/ true); // Forward pass.
    wsp_ggml_build_forward_expand(result->gf, result->outputs);

    wsp_ggml_opt_build(result);

    return result;
}

void wsp_ggml_opt_free(wsp_ggml_opt_context_t opt_ctx) {
    if (opt_ctx == nullptr) {
        return;
    }
    wsp_ggml_backend_buffer_free(opt_ctx->buf_static);
    wsp_ggml_backend_buffer_free(opt_ctx->buf_cpu);
    wsp_ggml_free(opt_ctx->ctx_static);
    wsp_ggml_free(opt_ctx->ctx_cpu);
    delete opt_ctx;
}

void wsp_ggml_opt_reset(wsp_ggml_opt_context_t opt_ctx, bool optimizer) {
    if (optimizer) {
        wsp_ggml_graph_reset(opt_ctx->gb_opt);
        opt_ctx->iter = 1;
    } else {
        wsp_ggml_graph_reset(opt_ctx->gb_grad);
    }
}

bool wsp_ggml_opt_static_graphs(wsp_ggml_opt_context_t opt_ctx) {
    return opt_ctx->static_graphs;
}

struct wsp_ggml_tensor * wsp_ggml_opt_inputs(wsp_ggml_opt_context_t opt_ctx) {
    return opt_ctx->inputs;
}

struct wsp_ggml_tensor * wsp_ggml_opt_outputs(wsp_ggml_opt_context_t opt_ctx) {
    return opt_ctx->outputs;
}

struct wsp_ggml_tensor * wsp_ggml_opt_labels(wsp_ggml_opt_context_t opt_ctx) {
    return opt_ctx->labels;
}

struct wsp_ggml_tensor * wsp_ggml_opt_loss(wsp_ggml_opt_context_t opt_ctx) {
    return opt_ctx->loss;
}

struct wsp_ggml_tensor * wsp_ggml_opt_pred(wsp_ggml_opt_context_t opt_ctx) {
    return opt_ctx->pred;
}

struct wsp_ggml_tensor * wsp_ggml_opt_ncorrect(wsp_ggml_opt_context_t opt_ctx) {
    return opt_ctx->ncorrect;
}

struct wsp_ggml_tensor * wsp_ggml_opt_grad_acc(wsp_ggml_opt_context_t opt_ctx, struct wsp_ggml_tensor * node) {
    return wsp_ggml_graph_get_grad_acc(opt_ctx->gb_opt, node);
}

// ====== Optimization Result ======

wsp_ggml_opt_result_t wsp_ggml_opt_result_init() {
    return new wsp_ggml_opt_result;
}

void wsp_ggml_opt_result_free(wsp_ggml_opt_result_t result) {
    delete result;
}

void wsp_ggml_opt_result_reset(wsp_ggml_opt_result_t result) {
    result->ndata = 0;
    result->loss.clear();
    result->pred.clear();
    result->ncorrect = 0;
}

void wsp_ggml_opt_result_ndata(wsp_ggml_opt_result_t result, int64_t * ndata) {
    *ndata = result->ndata;
}

void wsp_ggml_opt_result_loss(wsp_ggml_opt_result_t result, double * loss, double * unc) {
    const int64_t nbatches = result->loss.size(); // Number of physical batches.

    if (nbatches == 0) {
        *loss = 0.0;
        *unc  = NAN;
        return;
    }

    double sum         = 0.0;
    double sum_squared = 0.0;

    for (const float & loss : result->loss) {
        // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
        const float loss_scaled = result->loss_per_datapoint ? loss*result->opt_period : loss;
        sum         += loss_scaled;
        sum_squared += loss_scaled*loss_scaled;
    }

    const double mean = sum/nbatches;
    *loss = result->loss_per_datapoint ? mean : sum;

    if (!unc) {
        return;
    }

    if (nbatches < 2) {
        *unc = NAN;
        return;
    }

    const double var_sum = sum_squared/nbatches - mean*mean; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    *unc = result->loss_per_datapoint ? sqrt(var_sum / (nbatches - 1)) : sqrt(var_sum * nbatches/(nbatches - 1));
}

void wsp_ggml_opt_result_pred(wsp_ggml_opt_result_t result, int32_t * pred) {
    for (size_t i = 0; i < result->pred.size(); ++i) {
        pred[i] = result->pred[i];
    }
}

void wsp_ggml_opt_result_accuracy(wsp_ggml_opt_result_t result, double * accuracy, double * unc) {
    *accuracy = result->ncorrect >= 0 ? double(result->ncorrect) / double(result->ndata) : NAN;

    if (!unc) {
        return;
    }

    *unc = result->ncorrect >= 0 && result->ndata >= 2 ?
        sqrt((*accuracy) * (1.0 - (*accuracy)) / double(result->ndata - 1)) : NAN;
}

// ====== Computation ======

void wsp_ggml_opt_prepare_alloc(
        wsp_ggml_opt_context_t    opt_ctx,
        struct wsp_ggml_context * ctx_compute,
        struct wsp_ggml_cgraph  * gf,
        struct wsp_ggml_tensor  * inputs,
        struct wsp_ggml_tensor  * outputs) {
    WSP_GGML_ASSERT(!opt_ctx->static_graphs);
    opt_ctx->ctx_compute = ctx_compute;
    opt_ctx->gf          = gf;
    opt_ctx->inputs      = inputs;
    opt_ctx->outputs     = outputs;
}

void wsp_ggml_opt_alloc(wsp_ggml_opt_context_t opt_ctx, bool backward) {
    WSP_GGML_ASSERT(!opt_ctx->eval_ready);
    if (opt_ctx->build_type == WSP_GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period > 1 && opt_ctx->opt_i == 0) {
        wsp_ggml_graph_reset(opt_ctx->gb_grad);
    }
    if (backward) {
        const int32_t opt_i_next = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
        opt_ctx->build_type = opt_i_next == 0 ? WSP_GGML_OPT_BUILD_TYPE_OPT : WSP_GGML_OPT_BUILD_TYPE_GRAD;
    } else {
        opt_ctx->build_type = WSP_GGML_OPT_BUILD_TYPE_FORWARD;
    }

    if (!opt_ctx->static_graphs) {
        wsp_ggml_opt_build(opt_ctx);
    }

    struct wsp_ggml_cgraph * graph = nullptr;
    switch (opt_ctx->build_type) {
        case WSP_GGML_OPT_BUILD_TYPE_FORWARD: {
            graph = opt_ctx->gf;
        } break;
        case WSP_GGML_OPT_BUILD_TYPE_GRAD: {
            graph = opt_ctx->gb_grad;
        } break;
        case WSP_GGML_OPT_BUILD_TYPE_OPT: {
            graph = opt_ctx->gb_opt;
        } break;
    }
    WSP_GGML_ASSERT(graph);

    if (opt_ctx->allocated_graph == graph) {
        opt_ctx->eval_ready = true;
        return;
    }

    wsp_ggml_backend_sched_reset(opt_ctx->backend_sched); // clear allocation of previous graph

    if (opt_ctx->static_graphs) {
        wsp_ggml_init_params params = {
            /*.mem_size   =*/ graph->size*wsp_ggml_tensor_overhead() + wsp_ggml_graph_overhead_custom(graph->size, graph->grads),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        wsp_ggml_free(opt_ctx->ctx_copy);
        opt_ctx->ctx_copy = wsp_ggml_init(params);

        opt_ctx->allocated_graph_copy = dup_graph(opt_ctx->ctx_copy, graph);
    } else {
        opt_ctx->allocated_graph_copy = graph;
    }

    wsp_ggml_backend_sched_alloc_graph(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->allocated_graph = graph;

    opt_ctx->eval_ready = true;
}

void wsp_ggml_opt_eval(wsp_ggml_opt_context_t opt_ctx, wsp_ggml_opt_result_t result) {
    WSP_GGML_ASSERT(opt_ctx->eval_ready);
    if (opt_ctx->allocated_graph == opt_ctx->gb_opt) {
        struct wsp_ggml_opt_optimizer_params opt_pars = opt_ctx->get_opt_pars(opt_ctx->get_opt_pars_ud);

        WSP_GGML_ASSERT(opt_pars.adamw.alpha >  0.0f);
        WSP_GGML_ASSERT(opt_pars.adamw.beta1 >= 0.0f);
        WSP_GGML_ASSERT(opt_pars.adamw.beta1 <= 1.0f);
        WSP_GGML_ASSERT(opt_pars.adamw.beta2 >= 0.0f);
        WSP_GGML_ASSERT(opt_pars.adamw.beta2 <= 1.0f);
        WSP_GGML_ASSERT(opt_pars.adamw.eps   >= 0.0f);
        WSP_GGML_ASSERT(opt_pars.adamw.wd    >= 0.0f);
        WSP_GGML_ASSERT(opt_pars.adamw.wd    <= 1.0f);

        // beta1, beta2 after applying warmup
        const float beta1h = 1.0f/(1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
        const float beta2h = 1.0f/(1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));

        float * adamw_par_data = wsp_ggml_get_data_f32(opt_ctx->adamw_params);
        adamw_par_data[0] = opt_pars.adamw.alpha;
        adamw_par_data[1] = opt_pars.adamw.beta1;
        adamw_par_data[2] = opt_pars.adamw.beta2;
        adamw_par_data[3] = opt_pars.adamw.eps;
        adamw_par_data[4] = opt_pars.adamw.wd;
        adamw_par_data[5] = beta1h;
        adamw_par_data[6] = beta2h;
    }

    wsp_ggml_backend_sched_graph_compute(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->iter += opt_ctx->allocated_graph == opt_ctx->gb_opt;
    opt_ctx->opt_i = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;

    if (!opt_ctx->static_graphs) {
        opt_ctx->gf                   = nullptr;
        opt_ctx->gb_grad              = nullptr;
        opt_ctx->gb_opt               = nullptr;
        opt_ctx->allocated_graph      = nullptr;
        opt_ctx->allocated_graph_copy = nullptr;
    }

    opt_ctx->eval_ready = false;

    if (!result) {
        return;
    }

    if (result->ndata == 0) {
        result->loss_per_datapoint = opt_ctx->loss_per_datapoint;
        result->opt_period         = opt_ctx->opt_period;
    } else {
        WSP_GGML_ASSERT(result->loss_per_datapoint == opt_ctx->loss_per_datapoint);
        WSP_GGML_ASSERT(result->opt_period         == opt_ctx->opt_period);
    }

    const int64_t ndata = opt_ctx->outputs->ne[1];
    WSP_GGML_ASSERT(result->ndata == ndata*int64_t(result->loss.size()) && "varying batch size not supported");
    result->ndata += ndata;

    WSP_GGML_ASSERT(wsp_ggml_is_scalar(opt_ctx->loss));
    WSP_GGML_ASSERT(opt_ctx->loss->type == WSP_GGML_TYPE_F32);
    float loss;
    wsp_ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, wsp_ggml_nbytes(opt_ctx->loss));
    result->loss.push_back(loss);

    if (opt_ctx->pred) {
        WSP_GGML_ASSERT(opt_ctx->pred->type == WSP_GGML_TYPE_I32);
        std::vector<int32_t> pred(ndata);
        wsp_ggml_backend_tensor_get(opt_ctx->pred, pred.data(), 0, wsp_ggml_nbytes(opt_ctx->pred));
        result->pred.insert(result->pred.end(), pred.begin(), pred.end());
    }

    if (!opt_ctx->ncorrect || result->ncorrect < 0) {
        result->ncorrect = -1;
        return;
    }

    WSP_GGML_ASSERT(wsp_ggml_is_scalar(opt_ctx->ncorrect));
    WSP_GGML_ASSERT(opt_ctx->ncorrect->type == WSP_GGML_TYPE_I64);
    int64_t ncorrect;
    wsp_ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, wsp_ggml_nbytes(opt_ctx->ncorrect));
    result->ncorrect += ncorrect;
}

// ====== High-Level Functions ======

void wsp_ggml_opt_epoch(
        wsp_ggml_opt_context_t      opt_ctx,
        wsp_ggml_opt_dataset_t      dataset,
        wsp_ggml_opt_result_t       result_train,
        wsp_ggml_opt_result_t       result_eval,
        int64_t                 idata_split,
        wsp_ggml_opt_epoch_callback callback_train,
        wsp_ggml_opt_epoch_callback callback_eval) {
    WSP_GGML_ASSERT(wsp_ggml_opt_static_graphs(opt_ctx) && "wsp_ggml_opt_epoch requires static graphs");
    struct wsp_ggml_tensor * inputs = wsp_ggml_opt_inputs(opt_ctx);
    struct wsp_ggml_tensor * labels = wsp_ggml_opt_labels(opt_ctx);
    struct wsp_ggml_tensor * data   = wsp_ggml_opt_dataset_data(dataset);
    WSP_GGML_ASSERT(data->ne[0] == inputs->ne[0]);

    const int64_t ndata       =   data->ne[1];
    const int64_t ndata_batch = inputs->ne[1];

    WSP_GGML_ASSERT(data->ne[1] % inputs->ne[1] == 0);
    const int64_t nbatches = ndata/ndata_batch;

    idata_split = idata_split < 0 ? ndata : idata_split;
    WSP_GGML_ASSERT(idata_split % ndata_batch == 0);
    const int64_t ibatch_split = idata_split / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_start = wsp_ggml_time_us();
    for (; ibatch < ibatch_split; ++ibatch) {
        wsp_ggml_opt_alloc(opt_ctx, /*backward =*/ true);
        wsp_ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        wsp_ggml_opt_eval(opt_ctx, result_train);
        if (callback_train) {
            callback_train(true, opt_ctx, dataset, result_train, ibatch+1, ibatch_split, t_loop_start);
        }
    }
    t_loop_start = wsp_ggml_time_us();
    for (; ibatch < nbatches; ++ibatch) {
        wsp_ggml_opt_alloc(opt_ctx, /*backward =*/ false);
        wsp_ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        wsp_ggml_opt_eval(opt_ctx, result_eval);
        if (callback_eval) {
            callback_eval(false, opt_ctx, dataset, result_eval, ibatch+1-ibatch_split, nbatches-ibatch_split, t_loop_start);
        }
    }
}

void wsp_ggml_opt_epoch_callback_progress_bar(
        bool               train,
        wsp_ggml_opt_context_t opt_ctx,
        wsp_ggml_opt_dataset_t dataset,
        wsp_ggml_opt_result_t  result,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    fprintf(stderr, "%s[", train ? "train: " : "val:   ");

    // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
    constexpr int64_t bar_length = 8;
    const int64_t ibatch8 = 8 * ibatch;
    for (int64_t j = 0; j < bar_length; ++j) {
        if        (ibatch_max * (8*j + 8) / bar_length < ibatch8) {
            fprintf(stderr, "\u2588"); // full block
        } else if (ibatch_max * (8*j + 7) / bar_length < ibatch8) {
            fprintf(stderr, "\u2589"); // 7/8 filled
        } else if (ibatch_max * (8*j + 6) / bar_length < ibatch8) {
            fprintf(stderr, "\u258A"); // 6/8 filled
        } else if (ibatch_max * (8*j + 5) / bar_length < ibatch8) {
            fprintf(stderr, "\u258B"); // 5/8 filled
        } else if (ibatch_max * (8*j + 4) / bar_length < ibatch8) {
            fprintf(stderr, "\u258C"); // 4/8 filled
        } else if (ibatch_max * (8*j + 3) / bar_length < ibatch8) {
            fprintf(stderr, "\u258D"); // 3/8 filled
        } else if (ibatch_max * (8*j + 2) / bar_length < ibatch8) {
            fprintf(stderr, "\u258E"); // 2/8 filled
        } else if (ibatch_max * (8*j + 1) / bar_length < ibatch8) {
            fprintf(stderr, "\u258F"); // 1/8 filled
        } else {
            fprintf(stderr, " ");
        }
    }

    const int64_t batch_size = wsp_ggml_opt_inputs(opt_ctx)->ne[1];
    const int64_t idata      = ibatch*batch_size;
    const int64_t idata_max  = ibatch_max*batch_size;

    double loss;
    double loss_unc;
    wsp_ggml_opt_result_loss(result, &loss, &loss_unc);

    double accuracy;
    double accuracy_unc;
    wsp_ggml_opt_result_accuracy(result, &accuracy, &accuracy_unc);

    const int64_t t_ibatch_us = wsp_ggml_time_us() - t_start_us;
    int64_t t_ibatch_s = t_ibatch_us / 1000000;
    const int64_t t_ibatch_h = t_ibatch_s / 3600;
    t_ibatch_s -= t_ibatch_h * 3600;
    const int64_t t_ibatch_m = t_ibatch_s / 60;
    t_ibatch_s -= t_ibatch_m * 60;

    const int64_t t_eta_us = t_ibatch_us * (ibatch_max - ibatch)/ibatch;
    int64_t t_eta_s = t_eta_us / 1000000;
    const int64_t t_eta_h = t_eta_s / 3600;
    t_eta_s -= t_eta_h * 3600;
    const int64_t t_eta_m = t_eta_s / 60;
    t_eta_s -= t_eta_m * 60;

    fprintf(stderr, "] data=%07" PRId64 "/%07" PRId64 " loss=%.5lf±%.5lf acc=%.2lf±%.2lf%% "
            "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
            idata, idata_max, loss, loss_unc, 100.0*accuracy, 100.0*accuracy_unc,
            t_ibatch_h, t_ibatch_m, t_ibatch_s, t_eta_h, t_eta_m, t_eta_s);
    if (ibatch == ibatch_max) {
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    WSP_GGML_UNUSED(dataset);
}

void wsp_ggml_opt_fit(
        wsp_ggml_backend_sched_t            backend_sched,
        wsp_ggml_context                  * ctx_compute,
        wsp_ggml_tensor                   * inputs,
        wsp_ggml_tensor                   * outputs,
        wsp_ggml_opt_dataset_t              dataset,
        enum wsp_ggml_opt_loss_type         loss_type,
        wsp_ggml_opt_get_optimizer_params   get_opt_pars,
        int64_t                         nepoch,
        int64_t                         nbatch_logical,
        float                           val_split,
        bool                            silent) {
    wsp_ggml_time_init();
    const int64_t t_start_us = wsp_ggml_time_us();

    const int64_t ndata           = wsp_ggml_opt_dataset_data(dataset)->ne[1];
    const int64_t nbatch_physical = inputs->ne[1];
    WSP_GGML_ASSERT(ndata          % nbatch_logical  == 0);
    WSP_GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    const int64_t opt_period       = nbatch_logical / nbatch_physical;
    const int64_t nbatches_logical = ndata / nbatch_logical;

    WSP_GGML_ASSERT(val_split >= 0.0f);
    WSP_GGML_ASSERT(val_split <  1.0f);
    const int64_t ibatch_split = int64_t(((1.0f - val_split) * nbatches_logical)) * opt_period; // train <-> val split index (physical)
    const int64_t idata_split  = ibatch_split * nbatch_physical;

    int64_t epoch = 1;

    wsp_ggml_opt_params params = wsp_ggml_opt_default_params(backend_sched, loss_type);
    params.ctx_compute     = ctx_compute;
    params.inputs          = inputs;
    params.outputs         = outputs;
    params.opt_period      = opt_period;
    params.get_opt_pars    = get_opt_pars;
    params.get_opt_pars_ud = &epoch;
    wsp_ggml_opt_context_t opt_ctx = wsp_ggml_opt_init(params);

    // Shuffling the data is generally useful but there is only a point if not all data is used in a single batch.
    if (nbatch_logical < ndata) {
        wsp_ggml_opt_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all data (train + validation).
    }

    wsp_ggml_opt_result_t result_train = wsp_ggml_opt_result_init();
    wsp_ggml_opt_result_t result_val   = wsp_ggml_opt_result_init();

    wsp_ggml_opt_epoch_callback epoch_callback = silent ? nullptr : wsp_ggml_opt_epoch_callback_progress_bar;

    for (; epoch <= nepoch; ++epoch) {
        if (nbatch_logical < idata_split) {
            wsp_ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split);
        }

        wsp_ggml_opt_result_reset(result_train);
        wsp_ggml_opt_result_reset(result_val);

        if (!silent) {
            fprintf(stderr, "%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch);
        }
        wsp_ggml_opt_epoch(opt_ctx, dataset, result_train, result_val, idata_split, epoch_callback, epoch_callback);
        if (!silent) {
            fprintf(stderr, "\n");
        }
    }

    if (!silent) {
        int64_t t_total_s = (wsp_ggml_time_us() - t_start_us) / 1000000;
        const int64_t t_total_h = t_total_s / 3600;
        t_total_s -= t_total_h * 3600;
        const int64_t t_total_m = t_total_s / 60;
        t_total_s -= t_total_m * 60;
        fprintf(stderr, "%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n", __func__, t_total_h, t_total_m, t_total_s);
    }

    wsp_ggml_opt_free(opt_ctx);
    wsp_ggml_opt_result_free(result_train);
    wsp_ggml_opt_result_free(result_val);
}
