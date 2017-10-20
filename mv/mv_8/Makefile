FLAG=-msse3 -lopenblas -O3 -lrt

all:
	g++ *.cpp -msse3 -lmkl_rt -O3 -lrt -o test -L/opt/intel/mkl/lib/intel64_lin -I/opt/intel/mkl/include -mavx

clean:
	rm test
