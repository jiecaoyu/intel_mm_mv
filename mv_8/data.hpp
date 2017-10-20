#include <sys/time.h>
void fill_rand(float* data, const int size);
void fill_rand_sparse(float* data, const int size, const float sparse_rate);
void fill_rand_sparse_pattern(float* data, const int size, const float sparse_rate);
void fill_zero(float* data, const int size);
void print_data(float* const data, const int size);
long print_time(timespec begin, timespec end);
void sd2csr(float* const data, const int m, const int n,
        float* val, int* indx, int* pntrb, int* pntre);
