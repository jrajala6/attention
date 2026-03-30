#include "flash_attention.h"
#include <vector>
#include <cmath>
#include <cfloat>

void flash_attention_f16(const __fp16* Q, const __fp16* K, const __fp16* V, __fp16* O, size_t batch_size, size_t seq_len, size_t kv_seq_len,
    size_t num_q_heads, size_t num_kv_heads, size_t head_dim, float scale, bool is_causal, size_t window_size) 
{
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float* O_temp = (float*) malloc(head_dim * sizeof(float));
    const size_t BLOCK_SIZE = 128;
    size_t qkv_ratio = num_q_heads / num_kv_heads;

    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t q_head = 0; q_head < num_q_heads; ++q_head)
        {
            for (size_t q_idx = 0; q_idx < seq_len; ++q_idx)
            {
                size_t end_idx = is_causal ? (q_idx + 1) : kv_seq_len;
                size_t start_idx = (window_size > 0) ? (q_idx - window_size) : 0;
                
                for (size_t block_start = start_idx; block_start < end_idx; block_start += BLOCK_SIZE)
                {
                    size_t block_end = std::min(block_start + BLOCK_SIZE, end_idx);
                    for (size_t kv_idx = block_start; kv_idx < block_end; ++kv_idx)
                    {
                        size_t q_offset = b * num_q_heads * seq_len * head_dim + q_head * seq_len * head_dim + q_idx * head_dim;
                        size_t k_offset = b * num_kv_heads * kv_seq_len * head_dim + (q_head / qkv_ratio) * kv_seq_len * head_dim + kv_idx * head_dim;
                        size_t v_offset = b * num_kv_heads * kv_seq_len * head_dim + (q_head / qkv_ratio) * kv_seq_len * head_dim + kv_idx * head_dim;
                        O_temp[kv_idx] = 0.0f;
                        O_temp[kv_idx] += Q[q_offset] * K[k_offset];
                        
                    }
                    
                }
            }
        }
    }
    

}