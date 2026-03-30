#include <cstddef>
#include <arm_neon.h>

// Basic conversions (NEON-vectorized)
void fp16_to_fp32(const __fp16* src, float* dst, size_t count);
void fp32_to_fp16(const float* src, __fp16* dst, size_t count);
void int8_to_fp32(const int8_t* src, float* dst, size_t count, float scale);
void fp16_to_int8(const __fp16* src, int8_t* dst, size_t count, float scale);

// Find max absolute value (for computing quantization scale)
float fp16_max_abs(const __fp16* src, size_t count);

// Group quantization: quantize a fp16 vector to int8 in groups
//   Each group of `group_size` elements gets its own scale factor
//   scale = max_abs(group) / 127.0
//   quantized = round(value / scale)
void quantize_fp16_to_int8_grouped(
    const __fp16* src, int8_t* dst, float* scales,
    size_t total_elements, size_t group_size
);

// Dequantize: recover fp32 from int8 + scale
void dequantize_int8_to_fp32(
    const int8_t* src, float* dst, float scale, size_t count
);