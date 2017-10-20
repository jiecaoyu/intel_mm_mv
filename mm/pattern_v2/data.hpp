void fill_rand(float* data, const int size);
void fill_zero(float* data, const int size);
void fill_rand_sparse(float* data, const int size, const float sparse_rate);
void fill_rand_sparse_pattern(float* data, const int size, const float sparse_rate);
long print_time(timespec begin, timespec end);

template<typename Dtype>
void print_data(Dtype* data, const int size) {
    for(int i=0; i<size; ++i) {
        std::cout<<data[i];
        if(i!=(size-1)) std::cout<<", ";
    }
    std::cout<<"\n======================="<<std::endl;
    return;
}

template<typename Dtype>
void print_data(Dtype* data, const int size1, const int size2) {
    for (int i=0; i<size1; ++i ){
        for(int j=0; j<size2; ++j) {
            std::cout<<data[i*size2+j]<<",";
        }
        std::cout<<std::endl;
    }
    std::cout<<"\n======================="<<std::endl;
    return;
}

void sd2csr(float* const data, const int m, const int n,
        float* val, int* indx, int* pntrb, int* pntre);
