#pragma once
#include <cstddef>

// Simple loop versions for auto-vectorization
float dot_product_f16_simple(const __fp16* a, const __fp16* b, size_t len);
void weighted_accumulate_simple(float* accum, const __fp16* vec, float weight, size_t len);