#include"sparse_sgemm_nopattern.hpp"
#include<iostream>
#include<xmmintrin.h>
#include <iterator>  // std::ostream_iterator
#include <algorithm> // std::swap (until C++11)
#include <vector>
#include<x86intrin.h>
#include<pmmintrin.h>
#include<assert.h>
void sparse_sgemm_nopattern(const int m, const int n, const int k,
        const float* val, const int* indx, const int* pntrb, const int* pntre,
        const float* B, float* C) {
    float weight_tmp_single;
    __m256 weight_tmp_simd;
    assert(n%8==0);
    const float* weight = val;
    for(int s=0; s<m; ++s) {
        int begin = *(pntrb+s);
        int end = *(pntre+s);
        if (begin==end) continue;
        const int* indx_ptr = (indx+begin);
        __m256* output_tmp = (__m256*) (C+n*s);
        for(int i=begin; i< end; ++i) {
            weight_tmp_single = *(weight++);
            int index_tmp = *(indx_ptr++);
            __m256* input_tmp = (__m256*)(B+n*index_tmp);
            weight_tmp_simd = _mm256_set1_ps(weight_tmp_single);
            if (i!=begin) {
                for(int t=0; t<n/8; t+=1) {
                    __m256 mul_tmp = _mm256_mul_ps(weight_tmp_simd, input_tmp[t]);
                    _mm256_storeu_ps((float*)(output_tmp+t), _mm256_add_ps(mul_tmp, output_tmp[t]));
                }
            }
            else {
                for(int t=0; t<n/8; t+=1) {
                    _mm256_storeu_ps((float*)(output_tmp+t), _mm256_mul_ps(weight_tmp_simd, input_tmp[t]));
                }
            }
        }
    }
    return;
}
