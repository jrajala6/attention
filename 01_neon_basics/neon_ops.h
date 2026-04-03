#pragma once
#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

float dot_product_f16(const __fp16 *a, const __fp16 *b, size_t len);

void vec_add_f16(const __fp16 *a, const __fp16 *b, __fp16 *out, size_t len);

void vec_scale_f16(const __fp16 *a, __fp16 *out, size_t len, float scalar);

void weighted_accumulate(float *accum, const __fp16 *vec, float weight,
                         size_t len);

float dot_product_int8_fp16_dequant(const int8_t* a_int8, const __fp16* b_fp16,
                                    float scale, size_t len);

void weighted_accumulate_int8_fp16_dequant(const int8_t* a_int8, float scale,
                                           float* O_temp, float attention_score, size_t len);

float dot_product_int8_fp16_dequant_grouped(const int8_t* a_int8, const __fp16* b_fp16,
                                            const float* scales, size_t group_size, size_t len);

void weighted_accumulate_int8_fp16_dequant_grouped(const int8_t* a_int8, const float* scales,
                                                   float* O_temp, float attention_score,
                                                   size_t group_size, size_t len);