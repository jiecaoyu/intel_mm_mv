void sparse_sgemm_pattern(const int m, const int n, const int k,
        float* const val, int* const indx, int* const pntrb, int* const pntre,
        float* const B, float* C);

void sd2csr_pattern(float* const data, const int m, const int n,
        float* val, int* indx, int* pntrb, int* pntre);
