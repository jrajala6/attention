#include "flash_attention.h"
#include <vector>
#include <cmath>
#include <cfloat>
#include "../01_neon_basics/neon_ops.h"
#include "fast_math.h"

void flash_attention_f16(const __fp16* Q, const __fp16* K, const __fp16* V, __fp16* O, size_t batch_size, size_t seq_len, size_t kv_seq_len,
    size_t num_q_heads, size_t num_kv_heads, size_t head_dim, float scale, bool is_causal, size_t position_offset, size_t window_size) 
{
    float* O_temp = (float*) malloc(head_dim * sizeof(float));
    const size_t BLOCK_SIZE = 128;
    size_t qkv_ratio = num_q_heads / num_kv_heads;

    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t q_head = 0; q_head < num_q_heads; ++q_head)
        {
            for (size_t q_idx = 0; q_idx < seq_len; ++q_idx)
            {
                float running_max = -FLT_MAX;
                float running_sum = 0.0f;

                for (size_t i = 0; i < head_dim; ++i)
                    O_temp[i] = 0;

                size_t end_idx = is_causal ? (q_idx + 1) : kv_seq_len;
                size_t start_idx = 0;
                if (window_size > 0 && q_idx > window_size) {
                    start_idx = q_idx - window_size;
                }
                
                for (size_t block_start = start_idx; block_start < end_idx; block_start += BLOCK_SIZE)
                {
                    size_t block_end = std::min(block_start + BLOCK_SIZE, end_idx);
                    for (size_t kv_idx = block_start; kv_idx < block_end; ++kv_idx)
                    {
                        size_t q_offset = b * num_q_heads * seq_len * head_dim + q_head * seq_len * head_dim + q_idx * head_dim;
                        size_t k_offset = b * num_kv_heads * kv_seq_len * head_dim + (q_head / qkv_ratio) * kv_seq_len * head_dim + kv_idx * head_dim;
                        size_t v_offset = b * num_kv_heads * kv_seq_len * head_dim + (q_head / qkv_ratio) * kv_seq_len * head_dim + kv_idx * head_dim;
                        float score = 0; 

                        score = dot_product_f16(Q + q_offset, K + k_offset, head_dim);
                        score *= scale;

                        float prev_max = running_max;
                        running_max = std::max(running_max, score);
                        
                        running_sum = running_sum * exp(prev_max - running_max) + exp(score - running_max);

                        if (running_max != prev_max) {
                            float scale = exp(prev_max - running_max);
                            for (size_t i = 0; i < head_dim; ++i)
                                O_temp[i] *= scale;
                        }
                        
                        float current_weight = exp(score - running_max);
                        weighted_accumulate(O_temp, V + v_offset, current_weight, head_dim);
                    }
                    
                }

                size_t out_offset = b * num_q_heads * seq_len * head_dim + q_head * seq_len * head_dim + q_idx * head_dim;
                for (size_t i = 0; i < head_dim; ++i) {
                    O[out_offset + i] = (__fp16)(O_temp[i] / running_sum);
                }
            }
        }
    }
    
    free(O_temp);
}