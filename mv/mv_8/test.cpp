#include <iostream>
#include <math.h>
#include <assert.h>
#include <iomanip>
#include <ctime>
#include <stdlib.h>
#include <xmmintrin.h> // Contain the SSE compiler intrinsics
#include <malloc.h>
#include <sys/time.h>
#include "mkl.h"

#include"data.hpp"
#include"naive_sgemv.hpp"
#include"opt_sgemv.hpp"
#include"sparse_sgemv.hpp"
#include"sparse_pattern_sgemv.hpp"


int main() {
    // __m128 f;

    timespec begin;
    timespec end;
    long time_exe;
    const int M_ = 4096;
    const int N_ = 9216;
    float *A = (float*) malloc (M_ * N_ * sizeof(float));
    float *B = (float*) malloc (     N_ * sizeof(float));
    float *C = (float*) malloc (     M_ * sizeof(float));
    float *C_ref = (float*) malloc (     M_ * sizeof(float));
    if ((A==NULL) | (B==NULL) | (C==NULL)) exit(1);

    srand(time(NULL));
    const float matrix_sparse_rate = 35104400.0/37748736;
    fill_rand_sparse_pattern(A, M_ * N_, matrix_sparse_rate);
    fill_rand(B, N_);
    fill_zero(C, M_);
    fill_zero(C_ref, M_);
    // warm up
    for (int i =0; i<1; ++i) {
        naive_sgemv(M_, N_, A, B, C_ref);
    }
    
    clock_gettime(CLOCK_REALTIME, &begin);
    for (int i =0; i<1; ++i) {
        naive_sgemv(M_, N_, A, B, C_ref);
    }
    clock_gettime(CLOCK_REALTIME, &end);
    time_exe = print_time(begin, end);
    std::cout<<"Naive execution:\t"<<time_exe<<std::endl;


    fill_zero(C, M_);
    for (int i =0; i<1000; ++i) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, M_, N_, (float)1.,
                A, N_, B, 1, (float)0., C, 1);
    }
    clock_gettime(CLOCK_REALTIME, &begin);
    for (int i =0; i<1000; ++i) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, M_, N_, (float)1.,
                A, N_, B, 1, (float)0., C, 1);
    }
    clock_gettime(CLOCK_REALTIME, &end);
    time_exe = print_time(begin, end);
    for(int i=0; i< M_; ++i) {
        assert(fabs(C_ref[i] - C[i])<0.01);
    }
    std::cout<<"MKL Dense execution:\t"<<time_exe<<std::endl;


    fill_zero(C, M_);
    int non_zero = M_ * N_ * (1-matrix_sparse_rate)+8;
    float *val = (float*) malloc(non_zero*sizeof(float));
    int *indx = (int*) malloc(non_zero * sizeof(int));
    int *pntrb = (int*) malloc(std::max(M_, N_) * sizeof(int));
    int *pntre = (int*) malloc(std::max(M_, N_) * sizeof(int));
    sd2csr(A, M_, N_, val, indx, pntrb, pntre);
    char matdescra[6];
    matdescra[0] = 'G';
    matdescra[3] = 'C';
    float alpha = 1.;
    float beta = 0.;
    char transa = 'N';
    for (int i =0; i<1000; ++i) {
        mkl_scsrmv(&transa, &M_, &N_,
                &alpha, matdescra,
                val, indx, pntrb, pntre, B, &beta, C);
    }
    clock_gettime(CLOCK_REALTIME, &begin);
    for (int i =0; i<1000; ++i) {
        mkl_scsrmv(&transa, &M_, &N_,
                &alpha, matdescra,
                val, indx, pntrb, pntre, B, &beta, C);
    }
    clock_gettime(CLOCK_REALTIME, &end);
    time_exe = print_time(begin, end);
    for(int i=0; i< M_; ++i) {
        assert(fabs(C_ref[i] - C[i])<0.01);
    }
    std::cout<<"MKL Sparse execution:\t"<<time_exe<<std::endl;



    // opt_sgemv has error
    //fill_zero(C, M_);
    //for (int i =0; i<1000; ++i) {
    //    opt_sgemv(M_, N_, A, B, C);
    //}
    //clock_gettime(CLOCK_REALTIME, &begin);
    //for (int i =0; i<1000; ++i) {
    //    opt_sgemv(M_, N_, A, B, C);
    //}
    //clock_gettime(CLOCK_REALTIME, &end);
    //time_exe = print_time(begin, end);
    //for(int i=0; i< M_; ++i) {
    //    assert(fabs(C_ref[i] - C[i])<0.00001);
    //}
    //std::cout<<"Opt execution:\t\t"<<time_exe<<std::endl;



    fill_zero(C, M_);
    float *A_sparse_weight = (float*) malloc (non_zero * sizeof(float));
    int *A_sparse_index = (int*) malloc (non_zero * sizeof(int));
    int *A_sparse_base = (int*) malloc (M_ * sizeof(int));
    sparse_transfer(M_, N_, A, A_sparse_weight, A_sparse_index, A_sparse_base);
    for (int i =0; i<1000; ++i) {
        sparse_sgemv(M_, N_, A_sparse_weight, A_sparse_index, A_sparse_base, B, C);
    }
    clock_gettime(CLOCK_REALTIME, &begin);
    for (int i =0; i<1000; ++i) {
        sparse_sgemv(M_, N_, A_sparse_weight, A_sparse_index, A_sparse_base, B, C);
    }
    clock_gettime(CLOCK_REALTIME, &end);
    time_exe = print_time(begin, end);
    for(int i=0; i< M_; ++i) {
        assert(fabs(C_ref[i] - C[i])<0.01);
    }
    std::cout<<"Sparse execution:\t"<<time_exe<<std::endl;



    fill_zero(C, M_);
    sparse_pattern_transfer(M_, N_, A, A_sparse_weight, A_sparse_index, A_sparse_base);
    for (int i =0; i<1000; ++i) {
        sparse_pattern_sgemv(M_, N_, A_sparse_weight, A_sparse_index, A_sparse_base, B, C);
    }
    clock_gettime(CLOCK_REALTIME, &begin);
    for (int i =0; i<1000; ++i) {
        sparse_pattern_sgemv(M_, N_, A_sparse_weight, A_sparse_index, A_sparse_base, B, C);
    }
    clock_gettime(CLOCK_REALTIME, &end);
    time_exe = print_time(begin, end);
    for(int i=0; i< M_; ++i) {
        assert(fabs(C_ref[i] - C[i])<0.01);
    }
    std::cout<<"Sparse P execution:\t"<<time_exe<<std::endl;
    return 0;
}
