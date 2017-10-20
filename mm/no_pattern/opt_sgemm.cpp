#include"opt_sgemm.hpp"
#include<xmmintrin.h>
#include <iterator>  // std::ostream_iterator
#include <algorithm> // std::swap (until C++11)
#include <vector>

void transpose(float* first, float* last, const int n) {
    const int mn1 = (last-first-1);
    const int m = (last-first) / n;
    std::vector<bool> visited(last-first);
    float* cycle = first;
    while(++cycle != last) {
        if(visited[cycle-first]) continue;
        int a = cycle-first;
        do {
            a = a==mn1? mn1:(m*a)%mn1;
            std::swap(*(first+a), *cycle);
            visited[a] = true;
        }while((first+a)!=cycle);
    }
    return;
}

void opt_sgemm(const int m, const int n, const int k,
        float* A, float* B, float* C) {
    __m128* output = (__m128*) C;
    for(int i=0; i<(m*n/4); ++i) {
        output[i] = _mm_set1_ps(0.);
    }
    return;
}
