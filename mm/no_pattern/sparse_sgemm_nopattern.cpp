#include"sparse_sgemm_nopattern.hpp"
#include<iostream>
#include<xmmintrin.h>
#include <iterator>  // std::ostream_iterator
#include <algorithm> // std::swap (until C++11)
#include <vector>
#include<x86intrin.h>
#include<pmmintrin.h>
void sparse_sgemm_nopattern(const int m, const int n, const int k,
        const float* val, const int* indx, const int* pntrb, const int* pntre,
        const float* B, float* C) {
    float weight_tmp_single;
    __m128 weight_tmp_simd;
    const float* weight = val;
    for(int s=0; s<m; ++s) {
        int begin = *(pntrb+s);
        int end = *(pntre+s);
        if (begin==end) continue;
        const int* indx_ptr = (indx+begin);
        __m128* output_tmp = (__m128*) (C+n*s);
        for(int i=begin; i< end; ++i) {
            weight_tmp_single = *(weight++);
            int index_tmp = *(indx_ptr++);
            __m128* input_tmp = (__m128*)(B+n*index_tmp);
            weight_tmp_simd = _mm_set1_ps(weight_tmp_single);
            if (i!=begin) {
                for(int t=0; t<n/4; t+=1) {
                    __m128 mul_tmp = _mm_mul_ps(weight_tmp_simd, input_tmp[t]);
                    output_tmp[t] = _mm_add_ps(mul_tmp, output_tmp[t]);
                }
            }
            else {
                for(int t=0; t<n/4; t+=1) {
                    output_tmp[t] = _mm_mul_ps(weight_tmp_simd, input_tmp[t]);
                }
            }
        }
    }
    return;
}
