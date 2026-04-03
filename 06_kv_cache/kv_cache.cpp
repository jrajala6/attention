#include "kv_cache.h"
#include <cstring>
#include <algorithm>
#include <stdexcept>

void KVCache::init(size_t layers, size_t max_seq, size_t heads, size_t dim, CachePrecision prec) {
    this->layers.resize(layers);
    this->num_layers = layers;
    this->max_seq_len = max_seq;
    this->num_kv_heads = heads;
    this->head_dim = dim;
    this->precision = prec;

    size_t bytes_per_element = (prec == CachePrecision::FP16) ? sizeof(__fp16) : sizeof(int8_t);
    size_t total_bytes = max_seq * heads * dim * bytes_per_element;

    for (auto& layer : this->layers) {
        layer.keys.resize(total_bytes);
        layer.values.resize(total_bytes);

        if (prec == CachePrecision::INT8) {
            layer.key_scales.resize(max_seq);
            layer.value_scales.resize(max_seq);
        }
    }
}

void KVCache::reset() {
    this->total_seq_len = 0;
    this->current_seq_len = 0;
}

void KVCache::append(size_t layer, const __fp16* new_keys, const __fp16* new_values, size_t num_new_tokens) {
    if (layer >= this->num_layers) {
        throw std::invalid_argument("Invalid layer index");
    }
    if (new_keys == nullptr || new_values == nullptr) {
        throw std::invalid_argument("Null key or value data");
    }
    if (num_new_tokens == 0) {
        return;
    }

    if (layer == 0) {
        this->evict_if_needed(num_new_tokens);
    }

    if (this->precision != CachePrecision::FP16) {
        throw std::runtime_error("Use append_int8() for INT8 caches");
    }

    __fp16* keys_data = (__fp16*)this->layers[layer].keys.data();
    __fp16* values_data = (__fp16*)this->layers[layer].values.data();

    for (size_t head = 0; head < this->num_kv_heads; ++head) {
        size_t src_offset = head * num_new_tokens * this->head_dim;

        size_t head_base = head * this->max_seq_len * this->head_dim;
        size_t dst_offset = head_base + this->current_seq_len * this->head_dim;

        memcpy(keys_data + dst_offset, new_keys + src_offset, num_new_tokens * this->head_dim * sizeof(__fp16));
        memcpy(values_data + dst_offset, new_values + src_offset, num_new_tokens * this->head_dim * sizeof(__fp16));
    }

    if (layer == this->num_layers - 1) {
        this->current_seq_len += num_new_tokens;
        this->total_seq_len += num_new_tokens;
    }
}

void KVCache::append_int8(size_t layer, const int8_t* new_keys, const int8_t* new_values,
                         const float* key_scales, const float* value_scales, size_t num_new_tokens) {
    if (layer >= this->num_layers) {
        throw std::invalid_argument("Invalid layer index");
    }
    if (new_keys == nullptr || new_values == nullptr) {
        throw std::invalid_argument("Null key or value data");
    }
    if (this->precision != CachePrecision::INT8) {
        throw std::runtime_error("Cannot append INT8 data to FP16 cache");
    }
    if ((key_scales == nullptr || value_scales == nullptr) && this->precision == CachePrecision::INT8) {
        throw std::invalid_argument("INT8 cache requires quantization scales");
    }
    if (num_new_tokens == 0) {
        return; 
    }

    if (layer == 0) {
        this->evict_if_needed(num_new_tokens);
    }

    int8_t* keys_data = (int8_t*)this->layers[layer].keys.data();
    int8_t* values_data = (int8_t*)this->layers[layer].values.data();

    for (size_t head = 0; head < this->num_kv_heads; ++head) {
        size_t src_offset = head * num_new_tokens * this->head_dim;

        size_t head_base = head * this->max_seq_len * this->head_dim;
        size_t dst_offset = head_base + this->current_seq_len * this->head_dim;

        memcpy(keys_data + dst_offset, new_keys + src_offset, num_new_tokens * this->head_dim);
        memcpy(values_data + dst_offset, new_values + src_offset, num_new_tokens * this->head_dim);
    }

    memcpy(this->layers[layer].key_scales.data() + this->current_seq_len, key_scales, num_new_tokens * sizeof(float));
    memcpy(this->layers[layer].value_scales.data() + this->current_seq_len, value_scales, num_new_tokens * sizeof(float));

    if (layer == this->num_layers - 1) {
        this->current_seq_len += num_new_tokens;
        this->total_seq_len += num_new_tokens;
    }
}

void KVCache::evict_if_needed(size_t additional_tokens) {
    size_t future_seq_len = this->current_seq_len + additional_tokens;                                                                         
    if (future_seq_len <= this->window_size) return;
                                                                                                                                                
    size_t bytes_per_element = (this->precision == CachePrecision::FP16) ? sizeof(__fp16) : sizeof(int8_t);                                                                                                                                                
    size_t keep = this->window_size - this->sink_size;
                                                                                                                                                
    size_t offset_position = future_seq_len - keep;
                
    size_t tokens_to_copy = (offset_position >= this->current_seq_len) ?
                            0 : (this->current_seq_len - offset_position);                                                                                                                                                    

    for (size_t layer = 0; layer < this->num_layers; ++layer) {  
        for (size_t head = 0; head < this->num_kv_heads; ++head) {     
            size_t head_offset = head * this->max_seq_len * this->head_dim * bytes_per_element;   
            size_t src_offset = head_offset + offset_position * this->head_dim * bytes_per_element;   
            size_t dst_offset = head_offset + this->sink_size * this->head_dim * bytes_per_element;
            size_t copy_bytes = tokens_to_copy * this->head_dim * bytes_per_element;                                                                    
            std::memmove(this->layers[layer].keys.data() + dst_offset, this->layers[layer].keys.data() + src_offset, copy_bytes);                                                                    
            std::memmove(this->layers[layer].values.data() + dst_offset, this->layers[layer].values.data() + src_offset, copy_bytes);   
            
        } 
        if (this->precision == CachePrecision::INT8) {
            std::memmove(this->layers[layer].key_scales.data() + this->sink_size, this->layers[layer].key_scales.data() + offset_position, tokens_to_copy * sizeof(float));
            std::memmove(this->layers[layer].value_scales.data() + this->sink_size, this->layers[layer].value_scales.data() + offset_position, tokens_to_copy * sizeof(float));
        }                                                           
    }

    this->current_seq_len = this->sink_size + tokens_to_copy;
} 

void KVCache::get_keys_fp16(size_t layer, __fp16* out) const {
    if (layer >= this->num_layers) {
        throw std::invalid_argument("Invalid layer index");
    }
    if (out == nullptr) {
        throw std::invalid_argument("Null output buffer");
    }
    if (this->precision != CachePrecision::FP16) {
        throw std::runtime_error("Cannot get FP16 keys from INT8 cache");
    }

    const __fp16* keys_data = (const __fp16*)this->layers[layer].keys.data();
    for (size_t head = 0; head < this->num_kv_heads; ++head) {
        size_t src_head_offset = head * this->max_seq_len * this->head_dim;
        size_t dst_head_offset = head * this->current_seq_len * this->head_dim;
        size_t copy_bytes = this->current_seq_len * this->head_dim * sizeof(__fp16);
        std::memcpy(out + dst_head_offset, keys_data + src_head_offset, copy_bytes);
    }
}

void KVCache::get_values_fp16(size_t layer, __fp16* out) const {
    if (layer >= this->num_layers) {
        throw std::invalid_argument("Invalid layer index");
    }
    if (out == nullptr) {
        throw std::invalid_argument("Null output buffer");
    }
    if (this->precision != CachePrecision::FP16) {
        throw std::runtime_error("Cannot get FP16 values from INT8 cache");
    }

    const __fp16* values_data = (const __fp16*)this->layers[layer].values.data();
    for (size_t head = 0; head < this->num_kv_heads; ++head) {
        size_t src_head_offset = head * this->max_seq_len * this->head_dim;
        size_t dst_head_offset = head * this->current_seq_len * this->head_dim;
        size_t copy_bytes = this->current_seq_len * this->head_dim * sizeof(__fp16);
        std::memcpy(out + dst_head_offset, values_data + src_head_offset, copy_bytes);
    }
}

void KVCache::get_keys_int8(size_t layer, int8_t* out) const {
    if (layer >= this->num_layers) throw std::invalid_argument("Invalid layer index");
    if (out == nullptr) throw std::invalid_argument("Null output buffer");
    if (this->precision != CachePrecision::INT8) throw std::runtime_error("Cannot get INT8 keys from FP16 cache");

    const int8_t* keys_data = (const int8_t*)this->layers[layer].keys.data();
    for (size_t head = 0; head < this->num_kv_heads; ++head) {
        size_t src_head_offset = head * this->max_seq_len * this->head_dim;
        size_t dst_head_offset = head * this->current_seq_len * this->head_dim;
        size_t copy_bytes = this->current_seq_len * this->head_dim;

        memcpy(out + dst_head_offset, keys_data + src_head_offset, copy_bytes);
    }
}

void KVCache::get_values_int8(size_t layer, int8_t* out) const {
    if (layer >= this->num_layers) throw std::invalid_argument("Invalid layer index");
    if (out == nullptr) throw std::invalid_argument("Null output buffer");
    if (this->precision != CachePrecision::INT8) throw std::runtime_error("Cannot get INT8 values from FP16 cache");

    const int8_t* values_data = (const int8_t*)this->layers[layer].values.data();
    for (size_t head = 0; head < this->num_kv_heads; ++head) {
        size_t src_head_offset = head * this->max_seq_len * this->head_dim;
        size_t dst_head_offset = head * this->current_seq_len * this->head_dim;
        size_t copy_bytes = this->current_seq_len * this->head_dim;

        memcpy(out + dst_head_offset, values_data + src_head_offset, copy_bytes);
    }
}

const float* KVCache::get_key_scales(size_t layer) const {
    if (layer >= this->num_layers) {
        throw std::invalid_argument("Invalid layer index");
    }
    if (this->precision != CachePrecision::INT8) {
        throw std::runtime_error("Key scales only available for INT8 caches");
    }
    return this->layers[layer].key_scales.data();
}

const float* KVCache::get_value_scales(size_t layer) const {
    if (layer >= this->num_layers) {
        throw std::invalid_argument("Invalid layer index");
    }
    if (this->precision != CachePrecision::INT8) {
        throw std::runtime_error("Value scales only available for INT8 caches");
    }
    return this->layers[layer].value_scales.data();
}

void KVCache::set_window(size_t window, size_t sinks) {
    if (sinks >= window) {
        throw std::invalid_argument("Sink size must be less than window size");
    }
    if (window == 0) {
        throw std::invalid_argument("Window size cannot be zero");
    }
    this->window_size = window;
    this->sink_size = sinks;
}