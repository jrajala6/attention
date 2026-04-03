#pragma once
#include <vector>
#include <cstdint>

enum class CachePrecision { FP16, INT8 };

struct KVCache {
    struct LayerCache {
        std::vector<uint8_t> keys;
        std::vector<uint8_t> values;
        std::vector<float> key_scales;
        std::vector<float> value_scales;
    };

    std::vector<LayerCache> layers;

    size_t num_layers, num_kv_heads, head_dim;
    size_t max_seq_len;
    size_t current_seq_len = 0;
    size_t total_seq_len = 0;
    size_t quant_group_size = 0;
    CachePrecision precision;

    size_t num_quant_groups() const {
        size_t gs = (quant_group_size == 0) ? head_dim : quant_group_size;
        return (head_dim + gs - 1) / gs;
    }
    size_t scale_index(size_t token, size_t head) const {
        return (token * num_kv_heads + head) * num_quant_groups();
    }

    size_t window_size = 1024;
    size_t sink_size = 4;

    void init(size_t layers, size_t max_seq, size_t heads, size_t dim, CachePrecision prec, size_t group_size = 0);
    void reset();

    void append(size_t layer, const __fp16* new_keys, const __fp16* new_values,
                size_t num_new_tokens);

    void append_int8(size_t layer, const int8_t* new_keys, const int8_t* new_values,
                    const float* key_scales, const float* value_scales, size_t num_new_tokens);

    void evict_if_needed(size_t additional_tokens = 0);

    void get_keys_fp16(size_t layer, __fp16* out) const;
    void get_values_fp16(size_t layer, __fp16* out) const;

    void get_keys_int8(size_t layer, int8_t* out) const;
    void get_values_int8(size_t layer, int8_t* out) const;
    const float* get_key_scales(size_t layer) const;
    const float* get_value_scales(size_t layer) const;

    const int8_t* get_keys_direct(size_t layer) const;
    const int8_t* get_values_direct(size_t layer) const;

    void set_window(size_t window, size_t sinks);
    size_t effective_len() const { return current_seq_len; }
};