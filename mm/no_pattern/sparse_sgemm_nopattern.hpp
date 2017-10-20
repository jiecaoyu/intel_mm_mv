void sparse_sgemm_nopattern(const int m, const int n, const int k,
        const float* val, const int* indx, const int* pntrb, const int* pntre,
        const float* B, float* C);
