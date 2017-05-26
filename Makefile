all : memcpy_test.app bandwidthtest.app matmul.app matmul2.app
clean:
	-rm *.app

memcpy_test.app : memcpy_test.cu
	nvcc memcpy_test.cu -O3 -D_FORCE_INLINES -arch=sm_52 -lcuda -D_CRT_SECURE_NO_DEPRECATE -o memcpy_test.app -I $(CUDA_HOME)/samples/common/inc

bandwidthtest.app : bandwidthtest.cu
	nvcc bandwidthtest.cu -O3 -D_FORCE_INLINES -arch=sm_52 -lcuda -D_CRT_SECURE_NO_DEPRECATE -o bandwidthtest.app -I $(CUDA_HOME)/samples/common/inc

matmul.app : matmul.cu
	nvcc matmul.cu -O3 -D_FORCE_INLINES -arch=sm_52 -lcuda -D_CRT_SECURE_NO_DEPRECATE -maxrregcount 20 -o matmul.app -I $(CUDA_HOME)/samples/common/inc

matmul2.app : matmul2.cu
	nvcc matmul2.cu -O3 -D_FORCE_INLINES -arch=sm_52 -lcuda -lcublas -D_CRT_SECURE_NO_DEPRECATE -o matmul2.app -I $(CUDA_HOME)/samples/common/inc

