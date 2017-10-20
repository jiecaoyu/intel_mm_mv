#include"sparse_sgemm_pattern.hpp"
#include<iostream>
#include<xmmintrin.h>
#include <iterator>  // std::ostream_iterator
#include <algorithm> // std::swap (until C++11)
#include <vector>
#include<x86intrin.h>
#include<pmmintrin.h>

static inline void local_transpose_4x4(float* data, const int k, const int n) {
    for(int i=0; i<k; i+=4) {
        for(int j=0; j<n; j+=4) {
            __m128* I0 = (__m128*) (data+i*n+j);
            __m128* I1 = (__m128*) (data+(i+1)*n+j);
            __m128* I2 = (__m128*) (data+(i+2)*n+j);
            __m128* I3 = (__m128*) (data+(i+3)*n+j);
            _MM_TRANSPOSE4_PS(*I0, *I1, *I2, *I3);
        }
    }
    return;
}

void sparse_sgemm_pattern(const int m, const int n, const int k,
        const float* val, const int* indx, const int* pntrb, const int* pntre,
        float* B, float* C) {
    __m128 weight_tmp;
    __m128 input_tmp;
    __m128 mul_tmp;
    const float* weight = val;
    const int* indx_ptr = indx;
    int index;
    local_transpose_4x4(B, k, n);
    __m128* output_tmp = (__m128*)malloc(n*sizeof(__m128));
    for(int s=0; s<m; s++) {
        int begin = *(pntrb+s);
        int end = *(pntre+s);
        if(begin == end) continue;
        int simd_length = (end-begin);
        for(int i=0; i< simd_length; ++i) {
            weight_tmp = _mm_load_ps(weight);
            weight+=4;
            index = *(indx_ptr++);
            if (i!=0){
                for(int t=0; t<n; t+=4) {
                    input_tmp = _mm_load_ps(&B[(index+0)*n+t]);
                    mul_tmp = _mm_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t] = _mm_add_ps(mul_tmp, output_tmp[t]);

                    input_tmp = _mm_load_ps(&B[(index+1)*n+t]);
                    mul_tmp = _mm_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t+1] = _mm_add_ps(mul_tmp, output_tmp[t+1]);

                    input_tmp = _mm_load_ps(&B[(index+2)*n+t]);
                    mul_tmp = _mm_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t+2] = _mm_add_ps(mul_tmp, output_tmp[t+2]);

                    input_tmp = _mm_load_ps(&B[(index+3)*n+t]);
                    mul_tmp = _mm_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t+3] = _mm_add_ps(mul_tmp, output_tmp[t+3]);
                }
            }
            else {
                for(int t=0; t<n; t+=4) {
                    input_tmp = _mm_load_ps(&B[(index+0)*n+t]);
                    mul_tmp = _mm_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t] = mul_tmp;

                    input_tmp = _mm_load_ps(&B[(index+1)*n+t]);
                    mul_tmp = _mm_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t+1] = mul_tmp;

                    input_tmp = _mm_load_ps(&B[(index+2)*n+t]);
                    mul_tmp = _mm_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t+2] = mul_tmp;

                    input_tmp = _mm_load_ps(&B[(index+3)*n+t]);
                    mul_tmp = _mm_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t+3] = mul_tmp;
                }
            }
        }
        for (int t=0; t<n; ++t) {
            output_tmp[t] = _mm_hadd_ps(output_tmp[t], output_tmp[t]);
            output_tmp[t] = _mm_hadd_ps(output_tmp[t], output_tmp[t]);
            C[s*n+t] = *((float*)(&output_tmp[t]));
        }
    }
    free(output_tmp);
    return;
}

void sd2csr_pattern(float* const data, const int m, const int n,
        float* val, int* indx, int* pntrb, int* pntre) {
    float* val_ptr = val;
    int* indx_ptr = indx;
    int* pntrb_ptr = pntrb;
    int* pntre_ptr = pntre;
    int non_zero_count = 0;
    for(int s=0; s<m; ++s) {
        *(pntrb_ptr++) = non_zero_count;
        for(int t=0; t<n; t+=4) {
            if(data[s*n + t]!=0) {
                *(val_ptr++) = data[s*n + t + 0];
                *(val_ptr++) = data[s*n + t + 1];
                *(val_ptr++) = data[s*n + t + 2];
                *(val_ptr++) = data[s*n + t + 3];
                *(indx_ptr++) = t;
                non_zero_count++;
            }
        }
        *(pntre_ptr++) = non_zero_count;
    }
    return;
}
