#include <cstddef>
#include <cstdint>

void hybrid_attention_f16(
    const __fp16* Q,
    const int8_t* K_cached, const int8_t* V_cached,  
    const float* k_scales, const float* v_scales,    
    const __fp16* K_new, const __fp16* V_new,           
    __fp16* O,
    size_t batch_size, size_t seq_len,
    size_t cache_len, size_t new_len,
    size_t num_q_heads, size_t num_kv_heads, size_t head_dim,
    float scale, size_t position_offset, bool is_causal,
    size_t window_size, size_t quant_group_size
);