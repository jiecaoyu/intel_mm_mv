#include"opt_sgemv.hpp"
#include<iostream>
#include <xmmintrin.h> // Contain the SSE compiler intrinsics
#include <x86intrin.h>
#include <pmmintrin.h>
void opt_sgemv(const int m, const int n,
        float* const A,  float* const x, float* y) {
    int s = 0;
    int t = 0;
    float* weight = A;
    __m128 weight_tmp;
    __m128 input_tmp;
    __m128 sum_tmp;
    __m128 mul_tmp;
    const int simd_len = n>>3;
    for (s=0; s < m; ++s) {
        float* input = x;
        sum_tmp = _mm_set_ps(0.0, 0.0, 0.0, 0.0);
        // SIMD part
        for (int i = 0; i < simd_len; ++i) {
            weight_tmp = _mm_load_ps(weight);
            input_tmp = _mm_load_ps(input);
            mul_tmp = _mm_mul_ps(weight_tmp, input_tmp);
            sum_tmp = _mm_add_ps(sum_tmp, mul_tmp);
            weight += 4;
            input += 4;
            weight_tmp = _mm_load_ps(weight);
            input_tmp = _mm_load_ps(input);
            mul_tmp = _mm_mul_ps(weight_tmp, input_tmp);
            sum_tmp = _mm_add_ps(sum_tmp, mul_tmp);
            weight += 4;
            input += 4;
        }
        sum_tmp = _mm_hadd_ps(sum_tmp, sum_tmp);
        sum_tmp = _mm_hadd_ps(sum_tmp, sum_tmp);
        y[s] = *((float*)(&sum_tmp));
    }
    return;
}
