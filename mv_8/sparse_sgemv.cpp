#include<iostream>
#include<cstdlib>
#include<assert.h>
#include"sparse_sgemv.hpp"
#include<xmmintrin.h> // Contain the SSE compiler intrinsics
#include<x86intrin.h>
#include<pmmintrin.h>
void sparse_transfer(const int m, const int n,
        float * const dense_weight,
        float * sparse_weight,
        int * sparse_index,
        int * sparse_base) {
    int zero_count = 0;
    int * index = sparse_index;
    float * weight = sparse_weight;
    for (int s=0; s<m; ++s) {
        int base_row = 0;
        for(int t=0; t<n; ++t) {
            if(dense_weight[s*n + t] != 0) {
                *index = t;
                index++;
                *weight = dense_weight[s*n+t];
                weight++;
                assert((float*)index!=weight);
                base_row++;
            }
            else {
                zero_count++;
            }
        }
        sparse_base[s] = base_row;
    }
    return;
}

void sparse_sgemv(const int m, const int n,
        float * const sparse_weight, 
        int * const sparse_index,
        int * const sparse_base,
        float * const x,
        float * y) {
    float* weight = sparse_weight;
    int* index = sparse_index;
    int* base = sparse_base;
    for (int s=0; s<m ; ++s) {
        int base_row = *(base++);
        __m256 output_tmp = _mm256_set_ps(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
        for (; base_row>0; base_row-=8) {
            __m256 weight_tmp = _mm256_loadu_ps(weight);
            weight +=8;
            int index_tmp0 = *(index++);
            int index_tmp1 = *(index++);
            int index_tmp2 = *(index++);
            int index_tmp3 = *(index++);
            int index_tmp4 = *(index++);
            int index_tmp5 = *(index++);
            int index_tmp6 = *(index++);
            int index_tmp7 = *(index++);
            __m256 input_tmp = _mm256_set_ps(x[index_tmp7], x[index_tmp6], x[index_tmp5], x[index_tmp4],
                    x[index_tmp3],x[index_tmp2],x[index_tmp1],x[index_tmp0]);
            //output_tmp += weight_tmp * input_tmp;

            __m256 mul_tmp = _mm256_mul_ps(weight_tmp, input_tmp);

            output_tmp = _mm256_add_ps(mul_tmp, output_tmp);
        }
        output_tmp = _mm256_hadd_ps(output_tmp, output_tmp);
        output_tmp = _mm256_hadd_ps(output_tmp, output_tmp);
        y[s] = *((float*)(&output_tmp))+*((float*)(&output_tmp)+4);
        // y[s] = output_tmp;
    }
    return;
}
