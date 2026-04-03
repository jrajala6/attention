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

    // Append FP16 KV data (for FP16 caches only)
    void append(size_t layer, const __fp16* new_keys, const __fp16* new_values,
                size_t num_new_tokens);

    // Direct INT8 append with quantization scales
    void append_int8(size_t layer, const int8_t* new_keys, const int8_t* new_values,
                    const float* key_scales, const float* value_scales, size_t num_new_tokens);

    // Sliding window eviction — keep sinks + recent window
    void evict_if_needed(size_t additional_tokens = 0);

    // Read back — returns contiguous fp16 (dequantizes if int8)
    void get_keys_fp16(size_t layer, __fp16* out) const;
    void get_values_fp16(size_t layer, __fp16* out) const;

    // Direct int8 access (for hybrid attention)
    void get_keys_int8(size_t layer, int8_t* out) const;
    void get_values_int8(size_t layer, int8_t* out) const;
    const float* get_key_scales(size_t layer) const;
    const float* get_value_scales(size_t layer) const;

    // Zero-copy direct access for performance (inference only)
    const int8_t* get_keys_direct(size_t layer) const;
    const int8_t* get_values_direct(size_t layer) const;

    void set_window(size_t window, size_t sinks);
    size_t effective_len() const { return current_seq_len; }
};