#include"sparse_sgemm_pattern.hpp"
#include<iostream>
#include<xmmintrin.h>
#include <iterator>  // std::ostream_iterator
#include <algorithm> // std::swap (until C++11)
#include <vector>
#include<x86intrin.h>
#include<pmmintrin.h>

inline void local_transpose_4x4(float* data, const int k, const int n) {
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
        float* const val, int* const indx, int* const pntrb, int* const pntre,
        float* const B, float* C) {
    __m128 weight_tmp;
    __m128 input_tmp;
    __m128 mul_tmp;
    __m128* output = (__m128*) C;
    for(int i=0; i<(m*n/4); ++i) {
        output[i] = _mm_set1_ps(0.);
    }
    int* pntrb_ptr = pntrb;
    int* pntre_ptr = pntre;
    int* indx_ptr = indx;
    int index;
    float* weight = val;
    local_transpose_4x4(B, k, n);
    const int index_gran = 256;
    const int n_gran = 256;
    std::vector<__m128> output_tmp(n_gran, _mm_set1_ps(0.));
    // create a vector to store the begin position
    for(int n_start = 0;n_start<n; n_start+=n_gran) {
        std::vector<int> begin_position(m, 0);
        int n_length = std::min(n_gran, n-n_start);
        for(int index_start=0; index_start<k; index_start+=index_gran) {
            for(int s=0; s<m; s++) {
                int begin = std::max(*(pntrb+s), begin_position[s]);
                int end = *(pntre+s);
                if(begin == end) continue;
                int simd_length = (end-begin);
                int index_end = index_start+index_gran;
                int i =0;
                for(i=0; i< simd_length; ++i) {
                    weight_tmp = _mm_load_ps(weight+begin*4+i*4);
                    index = *(indx+begin+i);
                    if (index>=index_end) {
                        break;
                    }
                    if (i!=0) {
                        for(int t=0; t<n_length; t+=4) {
                            input_tmp = _mm_load_ps(&B[(index+0)*n+t+n_start]);
                            mul_tmp = _mm_mul_ps(weight_tmp, input_tmp);
                            output_tmp[t] = _mm_add_ps(mul_tmp, output_tmp[t]);

                            input_tmp = _mm_load_ps(&B[(index+1)*n+t+n_start]);
                            mul_tmp = _mm_mul_ps(weight_tmp, input_tmp);
                            output_tmp[t+1] = _mm_add_ps(mul_tmp, output_tmp[t+1]);

                            input_tmp = _mm_load_ps(&B[(index+2)*n+t+n_start]);
                            mul_tmp = _mm_mul_ps(weight_tmp, input_tmp);
                            output_tmp[t+2] = _mm_add_ps(mul_tmp, output_tmp[t+2]);

                            input_tmp = _mm_load_ps(&B[(index+3)*n+t+n_start]);
                            mul_tmp = _mm_mul_ps(weight_tmp, input_tmp);
                            output_tmp[t+3] = _mm_add_ps(mul_tmp, output_tmp[t+3]);
                        }
                    }
                    else {
                        for(int t=0; t<n_length; t+=4) {
                            input_tmp = _mm_load_ps(&B[(index+0)*n+t+n_start]);
                            mul_tmp = _mm_mul_ps(weight_tmp, input_tmp);
                            output_tmp[t] = mul_tmp;

                            input_tmp = _mm_load_ps(&B[(index+1)*n+t+n_start]);
                            mul_tmp = _mm_mul_ps(weight_tmp, input_tmp);
                            output_tmp[t+1] = mul_tmp;

                            input_tmp = _mm_load_ps(&B[(index+2)*n+t+n_start]);
                            mul_tmp = _mm_mul_ps(weight_tmp, input_tmp);
                            output_tmp[t+2] = mul_tmp;

                            input_tmp = _mm_load_ps(&B[(index+3)*n+t+n_start]);
                            mul_tmp = _mm_mul_ps(weight_tmp, input_tmp);
                            output_tmp[t+3] = mul_tmp;
                        }
                    }
                }
                begin_position[s]= std::min(begin+i, end);
                for (int t=0; t<n_length; ++t) {
                    output_tmp[t] = _mm_hadd_ps(output_tmp[t], output_tmp[t]);
                    output_tmp[t] = _mm_hadd_ps(output_tmp[t], output_tmp[t]);
                    C[s*n+t+n_start] += *((float*)(&output_tmp[t]));
                }
            }
        }
    }
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
