#include "naive_attention.h"
#include <vector>
#include <cmath>
#include <cfloat>

// Q, K, V, O --> (batch, head, seq_len, head_dim)
void naive_attention(const __fp16* Q, const __fp16* K, const __fp16* V, __fp16* O, size_t batch_size, size_t seq_len, size_t kv_seq_len,
    size_t num_q_heads, size_t num_kv_heads, size_t head_dim, float scale, bool is_causal, size_t window_size) 
{
    size_t qkv_ratio = num_q_heads / num_kv_heads;
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t q_head = 0; q_head < num_q_heads; ++q_head)
        {            
            for (size_t q_idx = 0; q_idx < seq_len; ++q_idx)
            {
                size_t q_offset = b * num_q_heads * seq_len * head_dim + q_head * seq_len * head_dim + q_idx * head_dim;

                // set Output vector to 0
                for (size_t hd_idx = 0; hd_idx < head_dim; ++hd_idx)
                    O[q_offset + hd_idx] = 0.0f;


                std::vector<float> scores(kv_seq_len, 0.0f);
                float sum_exp_scores = 0;
                float max_score = -FLT_MAX;
                size_t end_idx = kv_seq_len;

                if (is_causal)
                    end_idx = q_idx + 1;

                size_t start_idx = 0;
                if (window_size > 0 && q_idx > window_size) {
                    start_idx = q_idx - window_size;
                }
                
                for (size_t kv_idx = start_idx; kv_idx < end_idx; ++kv_idx)
                {
                    size_t k_offset = b * num_kv_heads * kv_seq_len * head_dim + (q_head / qkv_ratio) * kv_seq_len * head_dim + kv_idx * head_dim;

                    for (size_t hd_idx = 0; hd_idx < head_dim; ++hd_idx)
                        scores[kv_idx] += (float)Q[q_offset + hd_idx] * (float)K[k_offset + hd_idx];
                
                    scores[kv_idx] *= scale;
                    max_score = std::max(max_score, scores[kv_idx]);
                }

                for (size_t kv_idx = end_idx; kv_idx < kv_seq_len; ++kv_idx)
                    scores[kv_idx] = -FLT_MAX;

                for (size_t kv_idx = 0; kv_idx < start_idx; ++kv_idx)
                    scores[kv_idx] = -FLT_MAX;

                for (size_t kv_idx = 0; kv_idx < kv_seq_len; ++kv_idx)
                {
                    scores[kv_idx] = exp(scores[kv_idx] - max_score);
                    sum_exp_scores += scores[kv_idx];
                }

                for (size_t kv_idx = 0; kv_idx < kv_seq_len; ++kv_idx)
                    scores[kv_idx] /= sum_exp_scores;
                
                for (size_t kv_idx = 0; kv_idx < kv_seq_len; ++kv_idx)
                {
                    if (scores[kv_idx] == 0.0f) continue;
                    size_t v_offset = b * num_kv_heads * kv_seq_len * head_dim + (q_head / qkv_ratio) * kv_seq_len * head_dim + kv_idx * head_dim;
                    for (size_t hd_idx = 0; hd_idx < head_dim; ++hd_idx)
                        O[q_offset + hd_idx] += scores[kv_idx] * V[v_offset + hd_idx];
                }
            }
        }
    }
    
}