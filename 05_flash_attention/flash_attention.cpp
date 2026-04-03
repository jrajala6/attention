#include "flash_attention.h"
#include <vector>
#include <cmath>
#include <cfloat>
#include "../01_neon_basics/neon_ops.h"
#include "../02_threading/thread_pool.h"
#include "fast_math.h"

void flash_attention_f16(const __fp16* Q, const __fp16* K, const __fp16* V, __fp16* O, size_t batch_size, size_t seq_len, size_t kv_seq_len,
    size_t num_q_heads, size_t num_kv_heads, size_t head_dim, float scale, bool is_causal, size_t position_offset, size_t window_size) 
{
    const size_t BLOCK_SIZE = 128;
    size_t qkv_ratio = num_q_heads / num_kv_heads;
    size_t total_work = batch_size * num_q_heads * seq_len;

    parallel_for(total_work, 1, [&](size_t idx) {
        float O_temp[head_dim];
        float block_scores[BLOCK_SIZE];
        size_t q_idx = idx % seq_len;
        size_t q_head = (idx / seq_len) % num_q_heads;
        size_t b = (idx / (seq_len * num_q_heads)) % batch_size;
        
        float running_max = -FLT_MAX;
        float running_sum = 0.0f;

        for (size_t i = 0; i < head_dim; ++i)
            O_temp[i] = 0;

        size_t end_idx = is_causal ? std::min(q_idx + 1, kv_seq_len) : kv_seq_len;
        size_t start_idx = 0;
        if (window_size > 0 && q_idx > window_size) {
            start_idx = q_idx - window_size;
        }
        
        for (size_t block_start = start_idx; block_start < end_idx; block_start += BLOCK_SIZE)
        {
            size_t block_end = std::min(block_start + BLOCK_SIZE, end_idx);
            float block_scores[BLOCK_SIZE];
            float block_max = -FLT_MAX;
            for (size_t kv_idx = block_start; kv_idx < block_end; ++kv_idx)
            {
                if (kv_idx + 1 < block_end) {
                    size_t next_k_offset = b * num_kv_heads * kv_seq_len * head_dim + (q_head / qkv_ratio) * kv_seq_len * head_dim + (kv_idx + 1) * head_dim;
                    __builtin_prefetch(K + next_k_offset, 0, 1); // read-only and low temporal locality

                    size_t v_offset_prefetch = b * num_kv_heads * kv_seq_len * head_dim + (q_head / qkv_ratio) * kv_seq_len * head_dim + kv_idx * head_dim;
                    __builtin_prefetch(V + v_offset_prefetch, 0, 1);  // read-only and low temporal locality
                }
                
                size_t q_offset = b * num_q_heads * seq_len * head_dim + q_head * seq_len * head_dim + q_idx * head_dim;
                size_t k_offset = b * num_kv_heads * kv_seq_len * head_dim + (q_head / qkv_ratio) * kv_seq_len * head_dim + kv_idx * head_dim;
                
                float score = dot_product_f16(Q + q_offset, K + k_offset, head_dim) * scale;

                block_scores[kv_idx - block_start] = score;
                block_max = std::max(block_max, score);
            }

            if (block_max > running_max){
                float shrink_factor = fast_expf(running_max - block_max);
                running_sum *= shrink_factor;
                for (size_t i = 0; i < head_dim; ++i)
                    O_temp[i] *= shrink_factor;
                running_max = block_max;
            }

            size_t block_end_vec = ((block_end - block_start) / 4) * 4 + block_start;
            for (size_t kv_idx = block_start; kv_idx < block_end_vec; kv_idx += 4) {
                size_t v_offset = b * num_kv_heads * kv_seq_len * head_dim + (q_head / qkv_ratio) * kv_seq_len * head_dim + kv_idx * head_dim;
                float32x4_t curr = vld1q_f32(&block_scores[kv_idx - block_start]);
                float32x4_t shifted = vsubq_f32(curr, vdupq_n_f32(running_max));
                float32x4_t scores = fast_exp_f32x4(shifted);
                running_sum += vaddvq_f32(scores);
                weighted_accumulate(O_temp, V + v_offset, vgetq_lane_f32(scores, 0), head_dim);
                weighted_accumulate(O_temp, V + v_offset + head_dim, vgetq_lane_f32(scores, 1), head_dim);
                weighted_accumulate(O_temp, V + v_offset + 2 * head_dim, vgetq_lane_f32(scores, 2), head_dim);
                weighted_accumulate(O_temp, V + v_offset + 3 * head_dim, vgetq_lane_f32(scores, 3), head_dim);
            }

            for (size_t kv_idx = block_end_vec; kv_idx < block_end; ++kv_idx)
            {
                size_t v_offset = b * num_kv_heads * kv_seq_len * head_dim + (q_head / qkv_ratio) * kv_seq_len * head_dim + kv_idx * head_dim;
                float score = block_scores[kv_idx - block_start];
                score = fast_expf(score - running_max);
                running_sum += score;
                weighted_accumulate(O_temp, V + v_offset, score, head_dim);
            }
        }

        size_t out_offset = b * num_q_heads * seq_len * head_dim + q_head * seq_len * head_dim + q_idx * head_dim;
        for (size_t i = 0; i < head_dim; ++i) 
            O[out_offset + i] = (__fp16)(O_temp[i] / running_sum);
    });
}
