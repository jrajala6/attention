#include "quant.h"
#include "../02_threading/thread_pool.h"
#include <algorithm>
#include <cmath>

void fp16_to_fp32(const __fp16 *src, float *dst, size_t count) {
  for (size_t i = 0; i < (count / 8) * 8; i += 8) {
    float16x8_t src_chunk = vld1q_f16(src + i);
    vst1q_f32(dst + i, vcvt_f32_f16(vget_low_f16(src_chunk)));
    vst1q_f32(dst + i + 4, vcvt_f32_f16(vget_high_f16(src_chunk)));
  }

  for (size_t i = (count / 8) * 8; i < count; ++i)
    dst[i] = static_cast<float>(src[i]);
}

void fp32_to_fp16(const float *src, __fp16 *dst, size_t count) {
  for (size_t i = 0; i < (count / 8) * 8; i += 8) {
    float32x4_t src_lower = vld1q_f32(src + i);
    float32x4_t src_upper = vld1q_f32(src + i + 4);
    vst1q_f16(dst + i,
              vcombine_f16(vcvt_f16_f32(src_lower), vcvt_f16_f32(src_upper)));
  }

  for (size_t i = (count / 8) * 8; i < count; ++i)
    dst[i] = static_cast<__fp16>(src[i]);
}

void int8_to_fp32(const int8_t *src, float *dst, size_t count, float scale) {
  float32x4_t scale_vec = vdupq_n_f32(scale);
  for (size_t i = 0; i < (count / 8) * 8; i += 8) {
    int8x8_t src_chunk = vld1_s8(src + i);
    int16x8_t src_chunk_16 = vmovl_s8(src_chunk);
    vst1q_f32(dst + i,
              vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(src_chunk_16))),
                        scale_vec));
    vst1q_f32(dst + i + 4,
              vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(src_chunk_16))),
                        scale_vec));
  }

  for (size_t i = (count / 8) * 8; i < count; ++i)
    dst[i] = static_cast<float>(src[i]) * scale;
}

void fp16_to_int8(const __fp16 *src, int8_t *dst, size_t count, float scale) {
  float16x8_t scale_vec = vdupq_n_f16(scale);
  for (size_t i = 0; i < (count / 8) * 8; i += 8) {
    float16x8_t src_chunk = vld1q_f16(src + i);
    vst1_s8(dst + i,
            vqmovn_s16(vcvtaq_s16_f16(vdivq_f16(src_chunk, scale_vec))));
  }

  for (size_t i = (count / 8) * 8; i < count; ++i)
    dst[i] = static_cast<int8_t>(
        std::max(-128, std::min(127, (int)std::round(src[i] / scale))));
}

float fp16_max_abs(const __fp16 *src, size_t count) {
  return parallel_reduce(
      count, [src](size_t i) { return std::abs(static_cast<float>(src[i])); },
      0.0f, [](float a, float b) { return std::max(a, b); });
}

void quantize_fp16_to_int8_grouped(const __fp16 *src, int8_t *dst,
                                   float *scales, size_t total_elements,
                                   size_t group_size) {
  size_t num_groups = (total_elements + group_size - 1) / group_size;

  parallel_for(
      num_groups, 1,
      [src, dst, scales, total_elements, group_size](size_t group_idx) {
        size_t group_start = group_idx * group_size;
        size_t current_group_size =
            std::min(group_size, total_elements - group_start);

        float16x8_t max_vec = vdupq_n_f16(0.0f);
        size_t j = 0;
        for (; j + 8 <= current_group_size; j += 8) {
          float16x8_t vals = vld1q_f16(src + group_start + j);
          max_vec = vmaxq_f16(max_vec, vabsq_f16(vals));
        }

        __fp16 max_arr[8];
        vst1q_f16(max_arr, max_vec);
        float max_val = 0.0f;
        for (int k = 0; k < 8; ++k) {
          max_val = std::max(max_val, static_cast<float>(max_arr[k]));
        }

        for (; j < current_group_size; ++j) {
          max_val = std::max(
              max_val, std::abs(static_cast<float>(src[group_start + j])));
        }

        scales[group_idx] = max_val / 127.0f;

        fp16_to_int8(src + group_start, dst + group_start, current_group_size,
                     scales[group_idx]);
      });
}

void dequantize_int8_to_fp32(const int8_t *src, float *dst, float scale,
                             size_t count) {
  size_t elements_per_chunk = 1024;
  size_t num_chunks = (count + elements_per_chunk - 1) / elements_per_chunk;

  parallel_for(num_chunks, 1,
               [src, dst, scale, count, elements_per_chunk](size_t chunk_idx) {
                 size_t start = chunk_idx * elements_per_chunk;
                 size_t current_size =
                     std::min(elements_per_chunk, count - start);

                 int8_to_fp32(src + start, dst + start, current_size, scale);
               });
}