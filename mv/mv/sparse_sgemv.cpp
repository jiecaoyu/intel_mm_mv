#include<iostream>
#include<cstdlib>
#include<assert.h>
#include"sparse_sgemv.hpp"
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
        float output_tmp = 0.0;
        for (; base_row>0; --base_row) {
            float weight_tmp = *(weight++);
            int index_tmp = *(index++);
            float input_tmp = x[index_tmp];
            output_tmp += weight_tmp * input_tmp;
        }
        y[s] = output_tmp;
    }
    return;
}
