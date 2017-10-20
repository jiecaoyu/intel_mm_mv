#include"sparse_sgemm_pattern.hpp"
#include<assert.h>
#include<iostream>
#include<xmmintrin.h>
#include <iterator>  // std::ostream_iterator
#include <algorithm> // std::swap (until C++11)
#include <vector>
#include<x86intrin.h>
#include<pmmintrin.h>

inline void transpose8_ps(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7) {
    __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
    __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
    __t0 = _mm256_unpacklo_ps(row0, row1);
    __t1 = _mm256_unpackhi_ps(row0, row1);
    __t2 = _mm256_unpacklo_ps(row2, row3);
    __t3 = _mm256_unpackhi_ps(row2, row3);
    __t4 = _mm256_unpacklo_ps(row4, row5);
    __t5 = _mm256_unpackhi_ps(row4, row5);
    __t6 = _mm256_unpacklo_ps(row6, row7);
    __t7 = _mm256_unpackhi_ps(row6, row7);
    __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
    __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
    __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
    __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
    __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
    __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
    __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
    __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
    row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
    row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
    row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
    row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
    row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
    row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
    row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
    row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}

inline void local_transpose_8x8(float* data, const int k, const int n) {
    assert(k%8==0);
    assert(n%8==0);
    for(int i=0; i<k; i+=8) {
        for(int j=0; j<n; j+=8) {
            __m256 I0 = _mm256_loadu_ps(data+i*n+j);
            __m256 I1 = _mm256_loadu_ps(data+(i+1)*n+j);
            __m256 I2 = _mm256_loadu_ps(data+(i+2)*n+j);
            __m256 I3 = _mm256_loadu_ps(data+(i+3)*n+j);
            __m256 I4 = _mm256_loadu_ps(data+(i+4)*n+j);
            __m256 I5 = _mm256_loadu_ps(data+(i+5)*n+j);
            __m256 I6 = _mm256_loadu_ps(data+(i+6)*n+j);
            __m256 I7 = _mm256_loadu_ps(data+(i+7)*n+j);
            transpose8_ps(I0, I1, I2, I3, I4, I5, I6, I7);
            _mm256_storeu_ps(data+(i+0)*n+j,I0);
            _mm256_storeu_ps(data+(i+1)*n+j,I1);
            _mm256_storeu_ps(data+(i+2)*n+j,I2);
            _mm256_storeu_ps(data+(i+3)*n+j,I3);
            _mm256_storeu_ps(data+(i+4)*n+j,I4);
            _mm256_storeu_ps(data+(i+5)*n+j,I5);
            _mm256_storeu_ps(data+(i+6)*n+j,I6);
            _mm256_storeu_ps(data+(i+7)*n+j,I7);
        }
    }
    return;
}

void sparse_sgemm_pattern(const int m, const int n, const int k,
        float* const val, int* const indx, int* const pntrb, int* const pntre,
        float* const B, float* C) {
    __m256 weight_tmp;
    __m256 input_tmp;
    __m256 mul_tmp;
    __m256* output = (__m256*) C;
    for(int i=0; i<(m*n/8); ++i) {
        output[i] = _mm256_set1_ps(0.);
    }
    int* pntrb_ptr = pntrb;
    int* pntre_ptr = pntre;
    int* indx_ptr = indx;
    int index;
    float* weight = val;
    local_transpose_8x8(B, k, n);
    const int index_gran = 2048;
    __m256 output_tmp[n];
    // create a vector to store the begin position
    std::vector<int> begin_position(m, 0);
    for(int s=0; s<m; s++) {
        int begin = std::max(*(pntrb+s), begin_position[s]);
        int end = *(pntre+s);
        if(begin == end) continue;
        int simd_length = (end-begin);
        int index_end = index_gran;
        int i =0;
        for(i=0; i< simd_length; ++i) {
            weight_tmp = _mm256_loadu_ps(weight+begin*8+i*8);
            index = *(indx+begin+i);
            if (i!=0) {
                for(int t=0; t<n; t+=8) {
                    input_tmp = _mm256_loadu_ps(&B[(index+0)*n+t]);
                    mul_tmp = _mm256_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t] = _mm256_add_ps(mul_tmp, output_tmp[t]);

                    input_tmp = _mm256_loadu_ps(&B[(index+1)*n+t]);
                    mul_tmp = _mm256_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t+1] = _mm256_add_ps(mul_tmp, output_tmp[t+1]);

                    input_tmp = _mm256_loadu_ps(&B[(index+2)*n+t]);
                    mul_tmp = _mm256_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t+2] = _mm256_add_ps(mul_tmp, output_tmp[t+2]);

                    input_tmp = _mm256_loadu_ps(&B[(index+3)*n+t]);
                    mul_tmp = _mm256_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t+3] = _mm256_add_ps(mul_tmp, output_tmp[t+3]);

                    input_tmp = _mm256_loadu_ps(&B[(index+4)*n+t]);
                    mul_tmp = _mm256_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t+4] = _mm256_add_ps(mul_tmp, output_tmp[t+4]);

                    input_tmp = _mm256_loadu_ps(&B[(index+5)*n+t]);
                    mul_tmp = _mm256_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t+5] = _mm256_add_ps(mul_tmp, output_tmp[t+5]);

                    input_tmp = _mm256_loadu_ps(&B[(index+6)*n+t]);
                    mul_tmp = _mm256_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t+6] = _mm256_add_ps(mul_tmp, output_tmp[t+6]);

                    input_tmp = _mm256_loadu_ps(&B[(index+7)*n+t]);
                    mul_tmp = _mm256_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t+7] = _mm256_add_ps(mul_tmp, output_tmp[t+7]);
                }
            }
            else {
                for(int t=0; t<n; t+=8) {
                    input_tmp = _mm256_loadu_ps(&B[(index+0)*n+t]);
                    mul_tmp = _mm256_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t+0] = mul_tmp;

                    input_tmp = _mm256_loadu_ps(&B[(index+1)*n+t]);
                    mul_tmp = _mm256_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t+1] = mul_tmp;

                    input_tmp = _mm256_loadu_ps(&B[(index+2)*n+t]);
                    mul_tmp = _mm256_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t+2] = mul_tmp;

                    input_tmp = _mm256_loadu_ps(&B[(index+3)*n+t]);
                    mul_tmp = _mm256_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t+3] = mul_tmp;

                    input_tmp = _mm256_loadu_ps(&B[(index+4)*n+t]);
                    mul_tmp = _mm256_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t+4] = mul_tmp;

                    input_tmp = _mm256_loadu_ps(&B[(index+5)*n+t]);
                    mul_tmp = _mm256_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t+5] = mul_tmp;

                    input_tmp = _mm256_loadu_ps(&B[(index+6)*n+t]);
                    mul_tmp = _mm256_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t+6] = mul_tmp;

                    input_tmp = _mm256_loadu_ps(&B[(index+7)*n+t]);
                    mul_tmp = _mm256_mul_ps(weight_tmp, input_tmp);
                    output_tmp[t+7] = mul_tmp;
                }
            }
        }
        begin_position[s]= std::min(begin+i, end);
        for (int t=0; t<n; ++t) {
            output_tmp[t] = _mm256_hadd_ps(output_tmp[t], output_tmp[t]);
            output_tmp[t] = _mm256_hadd_ps(output_tmp[t], output_tmp[t]);
            C[s*n+t] += *((float*)(&output_tmp[t]))+*((float*)(&output_tmp[t])+4);
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
        for(int t=0; t<n; t+=8) {
            if(data[s*n + t]!=0) {
                *(val_ptr++) = data[s*n + t + 0];
                *(val_ptr++) = data[s*n + t + 1];
                *(val_ptr++) = data[s*n + t + 2];
                *(val_ptr++) = data[s*n + t + 3];
                *(val_ptr++) = data[s*n + t + 4];
                *(val_ptr++) = data[s*n + t + 5];
                *(val_ptr++) = data[s*n + t + 6];
                *(val_ptr++) = data[s*n + t + 7];
                *(indx_ptr++) = t;
                non_zero_count++;
            }
        }
        *(pntre_ptr++) = non_zero_count;
    }
    return;
}
