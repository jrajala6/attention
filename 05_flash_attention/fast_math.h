#include <arm_neon.h>

inline float32x4_t fast_exp_f32x4(float32x4_t x) {
    const float32x4_t log2e = vdupq_n_f32(1.4426950408889634f);
    
    x = vmaxq_f32(x, vdupq_n_f32(-87.0f));
    x = vminq_f32(x, vdupq_n_f32(87.0f)); // clamping input to prevent overflow 

    float32x4_t z = vmulq_f32(x, log2e);
    
    int32x4_t zi = vcvtq_s32_f32(z);
    float32x4_t zf = vsubq_f32(z, vcvtq_f32_s32(zi));

    uint32x4_t neg_mask = vcltq_f32(zf, vdupq_n_f32(0.0f));
    zi = vsubq_s32(zi, vandq_s32(vreinterpretq_s32_u32(neg_mask), vdupq_n_s32(1)));
    zf = vaddq_f32(zf, vreinterpretq_f32_u32(vandq_u32(neg_mask, vreinterpretq_u32_f32(vdupq_n_f32(1.0f)))));
    
    // Taylor series constants/ Polynomial estimate for fractional exponent
    const float32x4_t ln2 = vdupq_n_f32(0.69314718f);
    const float32x4_t c0 = vdupq_n_f32(1.0f);
    const float32x4_t c1 = vdupq_n_f32(0.69314718f); 
    const float32x4_t c2 = vdupq_n_f32(0.24022650f);  
    const float32x4_t c3 = vdupq_n_f32(0.05550410f);
    const float32x4_t c4 = vdupq_n_f32(0.00961812f);

    float32x4_t zf_ln2 = vmulq_f32(zf, ln2);
    float32x4_t p = c4;
    p = vfmaq_f32(c3, p, zf_ln2);
    p = vfmaq_f32(c2, p, zf_ln2);
    p = vfmaq_f32(c1, p, zf_ln2);
    p = vfmaq_f32(c0, p, zf_ln2);

    int32x4_t exp_bits = vshlq_n_s32(vaddq_s32(zi, vdupq_n_s32(127)), 23);
    float32x4_t exp_val = vreinterpretq_f32_s32(exp_bits);
    
    return vmulq_f32(exp_val, p);
}