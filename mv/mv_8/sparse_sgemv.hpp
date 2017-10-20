void sparse_transfer(const int m, const int n,
        float * const dense_weight,
        float * sparse_weight,
        int * sparse_index,
        int * sparse_base);

void sparse_sgemv(const int m, const int n,
        float * const sparse_weight, 
        int * const sparse_index,
        int * const sparse_base,
        float * const x,
        float * y);
