all : memcpy_test.app bandwidthtest.app matmul.app matmul2.app sortk.app
clean:
	-rm *.app

%.app : %.cu
	nvcc $< -o $@ -O3 -D_FORCE_INLINES -arch=sm_52 -lcuda -lcublas -D_CRT_SECURE_NO_DEPRECATE -I $(CUDA_HOME)/samples/common/inc

