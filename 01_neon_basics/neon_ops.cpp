#include "neon_ops.h"

float dot_product_f16(const __fp16 *a, const __fp16 *b, size_t len) {
    float32x4_t accum = vdupq_n_f32(0);

    for (size_t i = 0; i < (len / 8) * 8; i += 8) {
        float16x8_t curr_a = vld1q_f16(a + i);
        float16x8_t curr_b = vld1q_f16(b + i);

        float32x4_t a_low = vcvt_f32_f16(vget_low_f16(curr_a));
        float32x4_t a_high = vcvt_f32_f16(vget_high_f16(curr_a));
        float32x4_t b_low = vcvt_f32_f16(vget_low_f16(curr_b));
        float32x4_t b_high = vcvt_f32_f16(vget_high_f16(curr_b));

        accum = vfmaq_f32(accum, a_low, b_low); // multiply and accumulate 
        accum = vfmaq_f32(accum, a_high, b_high);
    }

    float dot_prod = vaddvq_f32(accum);

    // Handle remaining elements
    for (size_t i = (len / 8) * 8; i < len; ++i)
        dot_prod += a[i] * b[i];

    return dot_prod;
}

void vec_add_f16(const __fp16 *a, const __fp16 *b, __fp16 *out, size_t len)
{
    for (size_t i = 0; i < (len / 8) * 8; i += 8)
    {
        float16x8_t a_curr = vld1q_f16(a + i);
        float16x8_t b_curr = vld1q_f16(b + i);

        float32x4_t a_low = vcvt_f32_f16(vget_low_f16(a_curr));
        float32x4_t a_high = vcvt_f32_f16(vget_high_f16(a_curr));
        float32x4_t b_low = vcvt_f32_f16(vget_low_f16(b_curr));
        float32x4_t b_high = vcvt_f32_f16(vget_high_f16(b_curr));

        float32x4_t low_sum = vaddq_f32(a_low, b_low);
        float32x4_t high_sum = vaddq_f32(a_high, b_high);

        vst1q_f16(out + i, vcombine_f16(vcvt_f16_f32(low_sum), vcvt_f16_f32(high_sum)));
    }

    for (size_t i = (len / 8) * 8; i < len; ++i)
        out[i] = static_cast<float>(a[i]) + static_cast<float>(b[i]);
}

void vec_scale_f16(const __fp16* a, __fp16* out, size_t len, float scalar)
{
    float32x4_t scalar_vec = vdupq_n_f32(scalar);
    for (size_t i = 0; i < (len / 8) * 8; i += 8)
    {
        float16x8_t a_curr = vld1q_f16(a + i);
        float32x4_t a_low = vcvt_f32_f16(vget_low_f16(a_curr));
        float32x4_t a_high = vcvt_f32_f16(vget_high_f16(a_curr));

        float32x4_t low_scale = vmulq_f32(a_low, scalar_vec);
        float32x4_t high_scale = vmulq_f32(a_high, scalar_vec);


        vst1q_f16(out + i, vcombine_f16(vcvt_f16_f32(low_scale), vcvt_f16_f32(high_scale)));
    }

    for (size_t i = (len / 8) * 8; i < len; ++i)
        out[i] = static_cast<float>(a[i]) * scalar;
}


void weighted_accumulate(float* accum, const __fp16* vec, float weight, size_t len)
{
    float32x4_t scalar_vec = vdupq_n_f32(weight);

    for (size_t i = 0; i < (len / 8) * 8; i += 8)
    {
        float16x8_t vec_curr = vld1q_f16(vec + i);
        float32x4_t vec_low = vcvt_f32_f16(vget_low_f16(vec_curr));
        float32x4_t vec_high = vcvt_f32_f16(vget_high_f16(vec_curr));

        float32x4_t accum_low = vld1q_f32(accum + i);
        float32x4_t accum_high = vld1q_f32(accum + i + 4);

        accum_low = vfmaq_f32(accum_low, vec_low, scalar_vec);
        accum_high = vfmaq_f32(accum_high, vec_high, scalar_vec);

        vst1q_f32(accum + i, accum_low);
        vst1q_f32(accum + i + 4, accum_high);
    }

    for (size_t i = (len / 8) * 8; i < len; ++i)
        accum[i] += weight * vec[i];
}