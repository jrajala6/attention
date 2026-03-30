#include <arm_neon.h>
#include <cstddef>

// 1. Dot product of two fp16 vectors (the core of Q·K)
//    Load fp16 → convert to fp32 → FMA → horizontal sum
float dot_product_f16(const __fp16 *a, const __fp16 *b, size_t len);

// 2. Element-wise vector add (fp16 in, fp16 out, fp32 compute)
void vec_add_f16(const __fp16 *a, const __fp16 *b, __fp16 *out, size_t len);

// 3. Scale a vector: out = a * scalar
void vec_scale_f16(const __fp16 *a, __fp16 *out, size_t len, float scalar);

// 4. Weighted accumulate: accum += weight * vec (fp32 accum, fp16 vec)
//    This is the V accumulation pattern from attention
void weighted_accumulate(float *accum, const __fp16 *vec, float weight,
                         size_t len);