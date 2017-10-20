#define BILLION 1000000000
#include<iostream>
#include<stdlib.h>
#include<stdint.h>
#include<limits.h>
#include<assert.h>

#include"data.hpp"


void fill_rand(float* data, const int size) {
    int i = 0;
    for(i=0; i<size; ++i) {
        data[i] = (float)rand() / INT_MAX;
    }
    return;
}

void fill_zero(float* data, const int size) {
    int i = 0;
    for(i=0; i<size; ++i) {
        data[i] = 0;
    }
    return;
}

long print_time(timespec begin, timespec end) {
    long sec = end.tv_sec - begin.tv_sec;
    long nsec = end.tv_nsec - begin.tv_nsec;
    if(nsec<0) {
        nsec += BILLION;
        sec--;
    }
    return (sec*BILLION+nsec);
}

void fill_rand_sparse(float* data, const int size, const float sparse_rate) {
    int i = 0;
    int zero_count = 0;
    int zero_num = size*sparse_rate;
    for(i = 0; i< size; ++i) {
        float indicate = (float)rand() / INT_MAX;
        if (((indicate>sparse_rate)&((zero_num-zero_count)<=(size-i-1)))|(zero_count == zero_num)) {
            data[i] = (float)rand() / INT_MAX;
        }
        else {
            data[i] = 0.0;
            zero_count++;
        }
    }
    return;
}

void sd2csr(float* const data, const int m, const int n,
        float* val, int* indx, int* pntrb, int* pntre) {
    float* val_ptr = val;
    int* indx_ptr = indx;
    int* pntrb_ptr = pntrb;
    int* pntre_ptr = pntre;
    int non_zero_count = 0;
    for(int s=0; s<m; ++s) {
        *(pntrb_ptr++) = non_zero_count;
        for(int t=0; t<n; ++t) {
            if(data[s*n + t]!=0) {
                *(val_ptr++) = data[s*n + t];
                *(indx_ptr++) = t;
                non_zero_count++;
            }
        }
        *(pntre_ptr++) = non_zero_count;
    }
    return;
}

void fill_rand_sparse_pattern(float* data, const int size, const float sparse_rate) {
    assert(size%8 == 0);
    int i = 0;
    int zero_count = 0;
    int zero_num = ((int)(size*sparse_rate/8))*8;
    for(i = 0; i< (size/8); ++i) {
        float indicate = (float)rand() / INT_MAX;
        if (((indicate>sparse_rate)&((zero_num-zero_count)<=(size-8*i-1)))|(zero_count == zero_num)) {
            data[8*i+0] = (float)rand() / INT_MAX;
            data[8*i+1] = (float)rand() / INT_MAX;
            data[8*i+2] = (float)rand() / INT_MAX;
            data[8*i+3] = (float)rand() / INT_MAX;
            data[8*i+4] = (float)rand() / INT_MAX;
            data[8*i+5] = (float)rand() / INT_MAX;
            data[8*i+6] = (float)rand() / INT_MAX;
            data[8*i+7] = (float)rand() / INT_MAX;
        }
        else {
            data[8*i+0] = 0.0;
            data[8*i+1] = 0.0;
            data[8*i+2] = 0.0;
            data[8*i+3] = 0.0;
            data[8*i+4] = 0.0;
            data[8*i+5] = 0.0;
            data[8*i+6] = 0.0;
            data[8*i+7] = 0.0;
            zero_count+=8;
        }
    }
    return;
}
