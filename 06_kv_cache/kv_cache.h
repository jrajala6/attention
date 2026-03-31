#include <vector>
#include <cstdint>

enum class CachePrecision { FP16, INT8 };

struct KVCache {
    struct LayerCache {
        std::vector<uint8_t> keys;      // raw bytes (fp16 or int8)
        std::vector<uint8_t> values;
        std::vector<float> key_scales;  // only used for INT8
        std::vector<float> value_scales;
    };

    std::vector<LayerCache> layers;

    size_t num_layers, num_kv_heads, head_dim;
    size_t max_seq_len;
    size_t current_seq_len = 0;
    size_t total_seq_len = 0;
    CachePrecision precision;

    size_t window_size = 1024;
    size_t sink_size = 4;

    // Core operations
    void init(size_t layers, size_t max_seq, size_t heads, size_t dim, CachePrecision prec);
    void reset();

    // Append new KV data (fp16 input → stored as fp16 or quantized to int8)
    void append(size_t layer, const __fp16* new_keys, const __fp16* new_values,
                size_t num_new_tokens);

    // Sliding window eviction — keep sinks + recent window
    void evict_if_needed();

    // Read back — returns contiguous fp16 (dequantizes if int8)
    void get_keys_fp16(size_t layer, __fp16* out) const;
    void get_values_fp16(size_t layer, __fp16* out) const;

    // Direct int8 access (for hybrid attention)
    const int8_t* get_keys_int8(size_t layer) const;
    const float* get_key_scales(size_t layer) const;

    void set_window(size_t window, size_t sinks);
    size_t effective_len() const { return current_seq_len; }
};