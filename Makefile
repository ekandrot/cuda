all : memcpy_test.app bandwidthtest.app matmul.app matmul2.app sortk.app
clean:
	-rm *.app

%.app : %.cu
	nvcc $< -o $@ -O3 -D_FORCE_INLINES -arch=sm_61 -lcuda -lcublas -D_CRT_SECURE_NO_DEPRECATE -I $(CUDA_HOME)/samples/common/inc -I ../../ekclib -std=c++11 --compiler-options='-Wall -pthread -O3'

