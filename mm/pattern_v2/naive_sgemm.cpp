#include"naive_sgemm.hpp"

void naive_sgemm(const int m, const int n, const int k,
        float* const A, float* const B, float* C) {

    for (int s=0; s<m ;++s) {
        for (int t=0; t<n; ++t) {
            C[s*n+t] = 0;
            for (int i=0; i<k; ++i) {
                C[s*n+t] += A[s*k+i]*B[i*n+t];
            }
        }
    }
    return;
}
