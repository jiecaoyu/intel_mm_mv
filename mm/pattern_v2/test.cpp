#include<iostream>
#include<assert.h>
#include<cmath>
#include<ctime>
#include<stdlib.h>
#include<sys/time.h>
#include"mkl.h"

#include"data.hpp"
#include"naive_sgemm.hpp"
#include"opt_sgemm.hpp"
#include"sparse_sgemm_pattern.hpp"

int main() {
    timespec begin;
    timespec end;
    //const int M_ = 8;
    //const int N_ = 8;
    //const int K_ = 16;
    const int M_ = 192;
    const int N_ = 256;
    const int K_ = 2400;

    float *A = (float*) malloc(M_ * K_ * sizeof(float));
    float *B = (float*) malloc(K_ * N_ * sizeof(float));
    float *C = (float*) malloc(M_ * N_ * sizeof(float));
    float *C_ref = (float*) malloc(M_ * N_ * sizeof(float));

    srand(time(NULL));
    const float sparse_rate = 0.7;
    const int non_zero = M_*K_*(1.-sparse_rate);
    fill_rand_sparse_pattern(A, M_ * K_, sparse_rate);
    fill_rand(B, K_ * N_);
    fill_zero(C, M_ * N_);

    //clock_gettime(CLOCK_REALTIME, &begin);
    //for(int i=0; i<1000; ++i) {
    //    naive_sgemm(M_, N_, K_, A, B, C);
    //}
    //clock_gettime(CLOCK_REALTIME, &end);
    //std::cout<< "Naive sgemm:\t\t"<<print_time(begin, end)<<std::endl;
    //print_data(C, M_*N_);

    fill_zero(C, M_ * N_);
    fill_zero(C_ref, M_ * N_);
    for(int i=0; i<100; ++i) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M_, N_, K_,
                (float)1., A, K_, B, N_,
                (float)0., C_ref, N_);
    }
    clock_gettime(CLOCK_REALTIME, &begin);
    for(int i=0; i<1000; ++i) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M_, N_, K_,
                (float)1., A, K_, B, N_,
                (float)0., C_ref, N_);
    }
    clock_gettime(CLOCK_REALTIME, &end);
    std::cout<< "Openblas sgemm:\t\t"<<print_time(begin, end)<<std::endl;
    //print_data(C, M_*N_);

    fill_zero(C, M_ * N_);
    float *val = (float*) malloc(non_zero * sizeof(float));
    int *indx = (int*) malloc((non_zero+2) * sizeof(int));
    int *pntrb = (int*) malloc(std::max(M_, N_) * sizeof(int));
    int *pntre = (int*) malloc(std::max(M_, N_) * sizeof(int));
    sd2csr(A, M_, K_, val, indx, pntrb, pntre);
    char matdescra[6];
    matdescra[0] = 'G';
    matdescra[3] = 'C';
    float alpha = 1.;
    float beta = 0.;
    char transa = 'N';
    for(int i=0; i<100; ++i) {
    mkl_scsrmm(&transa, &M_, &N_, &K_,
            &alpha, matdescra,
            val, indx, pntrb, pntre, B, &N_,
            &beta, C, &N_);
    }
    clock_gettime(CLOCK_REALTIME, &begin);
    for(int i=0; i<1000; ++i) {
    mkl_scsrmm(&transa, &M_, &N_, &K_,
            &alpha, matdescra,
            val, indx, pntrb, pntre, B, &N_,
            &beta, C, &N_);
    }
    clock_gettime(CLOCK_REALTIME, &end);
    for(int i=0; i< M_*N_; ++i) {
        assert(fabs(C_ref[i] - C[i])<0.001);
    }
    std::cout<< "Sparse mkl sgemm:\t"<<print_time(begin, end)<<std::endl;
    //print_data(C, M_*N_);

    fill_zero(C, M_ * N_);
    sd2csr_pattern(A, M_, K_, val, indx, pntrb, pntre);
    sparse_sgemm_pattern(M_, N_, K_,
            val, indx, pntrb, pntre,
            B, C);
    for(int i=0; i<100; ++i) {
        sparse_sgemm_pattern(M_, N_, K_,
                val, indx, pntrb, pntre,
                B, C);
    }
    long long int time_sp=0;
    for(int i=0; i<1000; ++i) {
        sd2csr_pattern(A, M_, K_, val, indx, pntrb, pntre);
        clock_gettime(CLOCK_REALTIME, &begin);
        sparse_sgemm_pattern(M_, N_, K_,
                val, indx, pntrb, pntre,
                B, C);
        clock_gettime(CLOCK_REALTIME, &end);
        time_sp += print_time(begin, end);
    }
    for(int i=0; i< M_*N_; ++i) {
        assert(fabs(C_ref[i] - C[i])<0.001);
    }
    std::cout<< "Sparse pattern sgemm:\t"<<time_sp<<std::endl;
    //print_data(C, M_*N_);
    return 0;
}
