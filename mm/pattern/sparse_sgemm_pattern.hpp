void sparse_sgemm_pattern(const int m, const int n, const int k,
        const float* val, const int* indx, const int* pntrb, const int* pntre,
        float* B, float* C);

void sd2csr_pattern(float* const data, const int m, const int n,
        float* val, int* indx, int* pntrb, int* pntre);
