#include"naive_sgemv.hpp"
#include<iostream>
void naive_sgemv(const int m, const int n,
        float* const A,  float* const x, float* y) {
    int s = 0;
    int t = 0;
    float* weight = A;
    for (s=0; s < m; ++s) {
        y[s] = 0;
        float* input = x;
        for (t=0; t < n; ++t) {
            y[s] += (*(weight++)) * (*(input++));
        }
    }
    return;
}
