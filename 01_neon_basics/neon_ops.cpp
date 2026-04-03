#include "neon_ops.h"

float dot_product_f16(const __fp16 *a, const __fp16 *b, size_t len) {
    float32x4_t accum1 = vdupq_n_f32(0);
    float32x4_t accum2 = vdupq_n_f32(0);

    size_t vec_len = (len / 8) * 8;
    for (size_t i = 0; i < vec_len; i += 8) {
        float16x8_t curr_a = vld1q_f16(a + i);
        float16x8_t curr_b = vld1q_f16(b + i);

        float32x4_t a_low = vcvt_f32_f16(vget_low_f16(curr_a));
        float32x4_t a_high = vcvt_f32_f16(vget_high_f16(curr_a));
        float32x4_t b_low = vcvt_f32_f16(vget_low_f16(curr_b));
        float32x4_t b_high = vcvt_f32_f16(vget_high_f16(curr_b));

        accum1 = vfmaq_f32(accum1, a_low, b_low);
        accum2 = vfmaq_f32(accum2, a_high, b_high);
    }

    float32x4_t final_accum = vaddq_f32(accum1, accum2);
    float dot_prod = vaddvq_f32(final_accum);


    for (size_t i = vec_len; i < len; ++i)
        dot_prod += static_cast<float>(a[i] * b[i]);

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

float dot_product_int8_fp16_dequant(const int8_t* a_int8, const __fp16* b_fp16, float scale, size_t len) {
      float32x4_t accum1 = vdupq_n_f32(0);                                                                             
      float32x4_t accum2 = vdupq_n_f32(0);                                                                             
      float32x4_t scale_vec = vdupq_n_f32(scale);                                                                      
                                                                                                                       
      size_t vec_len = (len / 8) * 8;
      for (size_t i = 0; i < vec_len; i += 8) {                                                                        
          int8x8_t a_i8 = vld1_s8(a_int8 + i);                                                                         
   
          int16x8_t a_i16 = vmovl_s8(a_i8);
          int32x4_t a_i32_low = vmovl_s16(vget_low_s16(a_i16));                                                        
          int32x4_t a_i32_high = vmovl_s16(vget_high_s16(a_i16));
          float32x4_t a_low = vcvtq_f32_s32(a_i32_low);                                                                
          float32x4_t a_high = vcvtq_f32_s32(a_i32_high);                                                              
                                                                                                                       
          a_low = vmulq_f32(a_low, scale_vec);
          a_high = vmulq_f32(a_high, scale_vec);                                                                       
   
          float16x8_t curr_b = vld1q_f16(b_fp16 + i);
          float32x4_t b_low = vcvt_f32_f16(vget_low_f16(curr_b));                                                      
          float32x4_t b_high = vcvt_f32_f16(vget_high_f16(curr_b));
                                                                                                                       
          accum1 = vfmaq_f32(accum1, a_low, b_low);                                                                    
          accum2 = vfmaq_f32(accum2, a_high, b_high);
      }                                                                                                                
   
      float32x4_t final_accum = vaddq_f32(accum1, accum2);                                                             
      float dot_prod = vaddvq_f32(final_accum);
                                                                                                                       
      for (size_t i = vec_len; i < len; ++i)                                                                           
          dot_prod += (float)a_int8[i] * scale * (float)b_fp16[i];                                                     
                                                                                                                       
      return dot_prod;
  }     


void weighted_accumulate_int8_fp16_dequant(const int8_t* a_int8, float scale, float* O_temp, float attention_score, size_t len)
{
    float32x4_t scale_vec = vdupq_n_f32(scale); 
    float32x4_t attention_score_vec = vdupq_n_f32(attention_score);

    size_t vec_len = (len / 8) * 8;
    for (size_t i = 0; i < vec_len; i += 8) {                                                                        
        int8x8_t a_i8 = vld1_s8(a_int8 + i);                                                                         
 
        int16x8_t a_i16 = vmovl_s8(a_i8);
        int32x4_t a_i32_low = vmovl_s16(vget_low_s16(a_i16));                                                        
        int32x4_t a_i32_high = vmovl_s16(vget_high_s16(a_i16));
        float32x4_t a_low = vcvtq_f32_s32(a_i32_low);                                                                
        float32x4_t a_high = vcvtq_f32_s32(a_i32_high);                                                              
                                                                                                                      
        a_low = vmulq_f32(a_low, scale_vec);
        a_high = vmulq_f32(a_high, scale_vec); 

        float32x4_t accum_low = vld1q_f32(O_temp + i);
        float32x4_t accum_high = vld1q_f32(O_temp + i + 4);

        accum_low = vfmaq_f32(accum_low, a_low, attention_score_vec);
        accum_high = vfmaq_f32(accum_high, a_high, attention_score_vec);

        vst1q_f32(O_temp + i, accum_low);
        vst1q_f32(O_temp + i + 4, accum_high);
    }

    for (size_t i = (len / 8) * 8; i < len; ++i)
        O_temp[i] += scale * (float)a_int8[i] * attention_score;
}