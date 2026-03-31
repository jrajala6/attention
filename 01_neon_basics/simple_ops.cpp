#include "simple_ops.h"

// Simple dot product - compiler can auto-vectorize this
float dot_product_f16_simple(const __fp16* a, const __fp16* b, size_t len) {
    float sum = 0.0f;

    // Simple loop - perfect for compiler auto-vectorization
    for (size_t i = 0; i < len; ++i) {
        sum += static_cast<float>(a[i]) * static_cast<float>(b[i]);
    }

    return sum;
}

// Simple weighted accumulate - compiler can auto-vectorize this
void weighted_accumulate_simple(float* accum, const __fp16* vec, float weight, size_t len) {
    // Simple loop - perfect for compiler auto-vectorization
    for (size_t i = 0; i < len; ++i) {
        accum[i] += weight * static_cast<float>(vec[i]);
    }
}