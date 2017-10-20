#include<iostream>
#include<cstdlib>
#include<assert.h>
#include"sparse_pattern_sgemv.hpp"
#include<xmmintrin.h> // Contain the SSE compiler intrinsics
#include<x86intrin.h>
#include<pmmintrin.h>
void sparse_pattern_transfer(const int m, const int n,
        float * const dense_weight,
        float * sparse_weight,
        int * sparse_index,
        int * sparse_base) {
    int zero_count = 0;
    int * index = sparse_index;
    float * weight = sparse_weight;
    assert(n%4==0);
    for (int s=0; s<m; ++s) {
        int base_row = 0;
        for(int t=0; t<(n/4); ++t) {
            if(dense_weight[s*n + 4*t] != 0) {
                *index = 4*t;
                index++;
                base_row++;
                *(weight++) = dense_weight[s*n+4*t+0];
                *(weight++) = dense_weight[s*n+4*t+1];
                *(weight++) = dense_weight[s*n+4*t+2];
                *(weight++) = dense_weight[s*n+4*t+3];
            }
            else {
                zero_count++;
            }
        }
        sparse_base[s] = base_row;
    }
    return;
}

void sparse_pattern_sgemv(const int m, const int n,
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
        float output_tmp = 0.0;
        __m128 sum_tmp = _mm_set_ps(0.0, 0.0, 0.0, 0.0);
        for (; base_row>0; --base_row) {
            __m128 weight_tmp = _mm_load_ps(weight);
            weight += 4;
            int index_tmp = *(index++);
            __m128 input_tmp = _mm_load_ps(&x[index_tmp]);
            __m128 mul_tmp = _mm_mul_ps(weight_tmp, input_tmp);
            sum_tmp = _mm_add_ps(sum_tmp, mul_tmp);
        }
        sum_tmp = _mm_hadd_ps(sum_tmp, sum_tmp);
        sum_tmp = _mm_hadd_ps(sum_tmp, sum_tmp);
        y[s] = *((float*)(&sum_tmp));
    }
    return;
}
