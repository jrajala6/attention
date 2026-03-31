#pragma once

#include <cstddef>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

void flash_attention_f16(
    const __fp16* Q, const __fp16* K, const __fp16* V, __fp16* O,
    size_t batch_size, size_t seq_len, size_t kv_seq_len,
    size_t num_q_heads, size_t num_kv_heads, size_t head_dim,
    float scale, bool is_causal, size_t position_offset = 0,
    size_t window_size = 0
);

// Auto-vectorized version using simple loops
void flash_attention_f16_simple(
    const __fp16* Q, const __fp16* K, const __fp16* V, __fp16* O,
    size_t batch_size, size_t seq_len, size_t kv_seq_len,
    size_t num_q_heads, size_t num_kv_heads, size_t head_dim,
    float scale, bool is_causal, size_t position_offset = 0,
    size_t window_size = 0
);
