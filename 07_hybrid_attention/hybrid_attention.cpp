#include "hybrid_attention.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <limits>
#include <arm_neon.h>
#include "../01_neon_basics/neon_ops.h"
#include "../02_threading/thread_pool.h"
#include "../05_flash_attention/fast_math.h"

void hybrid_attention_f16(const __fp16* Q, const int8_t* K_cached, const int8_t* V_cached, const float* k_scales, const float* v_scales, const __fp16* K_new, const __fp16* V_new, __fp16* O, size_t batch_size,
    size_t seq_len, size_t cache_len, size_t new_len, size_t num_q_heads, size_t num_kv_heads, size_t head_dim, float scale, size_t position_offset, bool is_causal, size_t window_size, size_t quant_group_size
) {
    const size_t BLOCK_SIZE = 128;
    size_t qkv_ratio = num_q_heads / num_kv_heads;
    size_t total_work = batch_size * num_q_heads * seq_len;
    size_t num_quant_groups = (head_dim + quant_group_size - 1) / quant_group_size;

    parallel_for_range(total_work, 1, [&](size_t start_idx, size_t end_idx) {
        std::vector<float> O_temp(head_dim);
        std::vector<float> block_scores(BLOCK_SIZE);

        for (size_t idx = start_idx; idx < end_idx; ++idx) {
            std::fill(O_temp.begin(), O_temp.end(), 0.0f);

            size_t q_idx = idx % seq_len;
            size_t q_head = (idx / seq_len) % num_q_heads;
            size_t b = (idx / (seq_len * num_q_heads)) % batch_size;
            size_t kv_head = q_head / qkv_ratio;

            float running_max = -std::numeric_limits<float>::infinity();
            float running_sum = 0.0f;

            size_t kv_seq_len = cache_len + new_len;

            size_t absolute_q_pos = position_offset + q_idx;
            size_t end_kv = is_causal ? std::min(absolute_q_pos + 1, kv_seq_len) : kv_seq_len;
            size_t start_kv = 0;
            if (window_size > 0 && absolute_q_pos > window_size) {
                start_kv = absolute_q_pos - window_size;
            }

            size_t q_offset = b * num_q_heads * seq_len * head_dim + q_head * seq_len * head_dim + q_idx * head_dim;

            for (size_t block_start = start_kv; block_start < end_kv; block_start += BLOCK_SIZE)
            {
                size_t block_end = std::min(block_start + BLOCK_SIZE, end_kv);
                float block_max = -std::numeric_limits<float>::infinity();
                for (size_t kv_idx = block_start; kv_idx < block_end; ++kv_idx)
                {
                   if (kv_idx < cache_len){
                        size_t cache_offset = kv_head * cache_len * head_dim + kv_idx * head_dim;
                        size_t scale_base = (kv_idx * num_kv_heads + kv_head) * num_quant_groups;
                        float score = dot_product_int8_fp16_dequant_grouped(
                            K_cached + cache_offset, Q + q_offset,
                            k_scales + scale_base, quant_group_size, head_dim);
                        score *= scale;
                        block_scores[kv_idx - block_start] = score;
                   }
                   else {
                        size_t new_token_idx = kv_idx - cache_len;
                        if (kv_idx + 1 < block_end && new_token_idx + 1 < new_len) {
                            size_t next_k_offset = b * num_kv_heads * new_len * head_dim + kv_head * new_len * head_dim + (new_token_idx + 1) * head_dim;
                            __builtin_prefetch(K_new + next_k_offset, 0, 1);

                            size_t v_offset_prefetch = b * num_kv_heads * new_len * head_dim + kv_head * new_len * head_dim + new_token_idx * head_dim;
                            __builtin_prefetch(V_new + v_offset_prefetch, 0, 1);
                        }

                        size_t k_offset = b * num_kv_heads * new_len * head_dim + kv_head * new_len * head_dim + new_token_idx * head_dim;
                        float score = dot_product_f16(Q + q_offset, K_new + k_offset, head_dim) * scale;
                        block_scores[kv_idx - block_start] = score;
                   }
                    block_max = std::max(block_max, block_scores[kv_idx - block_start]);
                }

                if (block_max > running_max){
                    float shrink_factor = expf(running_max - block_max);
                    running_sum *= shrink_factor;
                    for (size_t i = 0; i < head_dim; ++i)
                        O_temp[i] *= shrink_factor;
                    running_max = block_max;
                }

                size_t block_end_vec = ((block_end - block_start) / 4) * 4 + block_start;
                for (size_t kv_idx = block_start; kv_idx < block_end_vec; kv_idx += 4) {
                    float32x4_t curr = vld1q_f32(&block_scores[kv_idx - block_start]);
                    float32x4_t shifted = vsubq_f32(curr, vdupq_n_f32(running_max));
                    float32x4_t scores = fast_exp_f32x4(shifted);
                    running_sum += vaddvq_f32(scores);
                    float score_vals[4];
                    vst1q_f32(score_vals, scores);

                    for (int i = 0; i < 4 && kv_idx + i < block_end; ++i) {
                        if (kv_idx + i < cache_len) {
                            size_t cache_offset = kv_head * cache_len * head_dim + (kv_idx + i) * head_dim;
                            size_t scale_base = ((kv_idx + i) * num_kv_heads + kv_head) * num_quant_groups;
                            weighted_accumulate_int8_fp16_dequant_grouped(
                                V_cached + cache_offset, v_scales + scale_base,
                                O_temp.data(), score_vals[i], quant_group_size, head_dim);
                        } else if (kv_idx + i < cache_len + new_len) {
                            size_t new_token_idx = (kv_idx + i) - cache_len;
                            size_t v_offset = b * num_kv_heads * new_len * head_dim + kv_head * new_len * head_dim + new_token_idx * head_dim;
                            weighted_accumulate(O_temp.data(), V_new + v_offset, score_vals[i], head_dim);
                        }
                    }
                }

                for (size_t kv_idx = block_end_vec; kv_idx < block_end; ++kv_idx)
                {
                    float score = block_scores[kv_idx - block_start];
                    score = fast_expf(score - running_max);
                    running_sum += score;

                    if (kv_idx < cache_len) {
                        size_t cache_offset = kv_head * cache_len * head_dim + kv_idx * head_dim;
                        size_t scale_base = (kv_idx * num_kv_heads + kv_head) * num_quant_groups;
                        weighted_accumulate_int8_fp16_dequant_grouped(
                            V_cached + cache_offset, v_scales + scale_base,
                            O_temp.data(), score, quant_group_size, head_dim);
                    } else if (kv_idx < cache_len + new_len) {
                        size_t new_token_idx = kv_idx - cache_len;
                        size_t v_offset = b * num_kv_heads * new_len * head_dim + kv_head * new_len * head_dim + new_token_idx * head_dim;
                        weighted_accumulate(O_temp.data(), V_new + v_offset, score, head_dim);
                    }
                }
            }

            size_t out_offset = b * num_q_heads * seq_len * head_dim + q_head * seq_len * head_dim + q_idx * head_dim;
            if (running_sum > 0.0f) {
                float inv_sum = 1.0f / running_sum;
                for (size_t i = 0; i < head_dim; ++i)
                    O[out_offset + i] = (__fp16)(O_temp[i] * inv_sum);
            } else {
                for (size_t i = 0; i < head_dim; ++i)
                    O[out_offset + i] = (__fp16)0.0f;
            }
        }
    });
}