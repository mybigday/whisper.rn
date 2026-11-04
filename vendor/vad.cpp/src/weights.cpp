#include "vad-impl.h"

#include <cstring>

namespace vad {

std::vector<float> host_tensor::to_f32() const {
    std::vector<float> v(nelements());
    if (type == GGML_TYPE_F32) {
        std::memcpy(v.data(), data.data(), v.size() * sizeof(float));
    } else {
        const ggml_type_traits * tt = ggml_get_type_traits(type);
        GGML_ASSERT(tt->to_float && "type cannot be converted to F32");
        tt->to_float(data.data(), v.data(), (int64_t) v.size());
    }
    return v;
}

host_tensor host_tensor::from_f32(std::vector<int64_t> ne, const std::vector<float> & v) {
    host_tensor t;
    t.ne = std::move(ne);
    t.type = GGML_TYPE_F32;
    t.data.resize(v.size() * sizeof(float));
    std::memcpy(t.data.data(), v.data(), t.data.size());
    return t;
}

const host_tensor * weight_builder::find(const std::string & name, int64_t n) {
    if (!err_.empty()) return nullptr;
    auto it = f_.tensors.find(name);
    if (it == f_.tensors.end()) { err_ = f_.arch + ": missing tensor " + name; return nullptr; }
    if (it->second.nelements() != n) {
        err_ = f_.arch + ": tensor " + name + " has " + std::to_string(it->second.nelements()) + " elements, expected " + std::to_string(n);
        return nullptr;
    }
    return &it->second;
}

ggml_tensor * weight_builder::make(const std::string & name, ggml_type type, const std::vector<int64_t> & ne) {
    ggml_tensor * t = nullptr;
    switch (ne.size()) {
        case 1: t = ggml_new_tensor_1d(ctx_, type, ne[0]); break;
        case 2: t = ggml_new_tensor_2d(ctx_, type, ne[0], ne[1]); break;
        case 3: t = ggml_new_tensor_3d(ctx_, type, ne[0], ne[1], ne[2]); break;
        default: GGML_ABORT("unsupported rank");
    }
    ggml_set_name(t, name.c_str());
    return t;
}

ggml_tensor * weight_builder::mat(const std::string & name, int64_t ne0, int64_t ne1) {
    const host_tensor * h = find(name, ne0 * ne1);
    if (!h) return nullptr;
    if (ne0 % ggml_blck_size(h->type) != 0) {
        err_ = f_.arch + ": tensor " + name + " is " + ggml_type_name(h->type) + ", whose block size does not divide its rows (" +
               std::to_string(ne0) + ")";
        return nullptr;
    }
    ggml_tensor * t = make(name, h->type, { ne0, ne1 });
    up_.items.emplace_back(t, h->data);
    return t;
}

ggml_tensor * weight_builder::f32(const std::string & name, std::vector<int64_t> ne) {
    int64_t n = 1;
    for (auto v : ne) n *= v;
    const host_tensor * h = find(name, n);
    if (!h) return nullptr;
    if (!h->is_float()) {
        err_ = f_.arch + ": tensor " + name + " is " + ggml_type_name(h->type) +
               " but is used outside mul_mat and must be F32/F16/BF16 in the model file";
        return nullptr;
    }
    std::vector<float> v = h->to_f32();
    std::vector<uint8_t> bytes(v.size() * sizeof(float));
    std::memcpy(bytes.data(), v.data(), bytes.size());
    ggml_tensor * t = make(name, GGML_TYPE_F32, ne);
    up_.items.emplace_back(t, std::move(bytes));
    return t;
}

ggml_tensor * weight_builder::constant(const std::string & name, std::vector<int64_t> ne, const std::vector<float> & v) {
    if (!err_.empty()) return nullptr;
    ggml_tensor * t = make(name, GGML_TYPE_F32, ne);
    GGML_ASSERT((size_t) ggml_nelements(t) == v.size());
    std::vector<uint8_t> bytes(v.size() * sizeof(float));
    std::memcpy(bytes.data(), v.data(), bytes.size());
    up_.items.emplace_back(t, std::move(bytes));
    return t;
}

} // namespace vad
