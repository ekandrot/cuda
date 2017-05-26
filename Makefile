all : memcpy_test.app bandwidthtest.app matmul.app

memcpy_test.app : memcpy_test.cu
	nvcc memcpy_test.cu -O3 -D_FORCE_INLINES -arch=sm_52 -lcuda -D_CRT_SECURE_NO_DEPRECATE -o memcpy_test.app -I $(CUDA_HOME)/samples/common/inc

bandwidthtest.app : bandwidthtest.cu
	nvcc bandwidthtest.cu -O3 -D_FORCE_INLINES -arch=sm_52 -lcuda -D_CRT_SECURE_NO_DEPRECATE -o bandwidthtest.app -I $(CUDA_HOME)/samples/common/inc

matmul.app : matmul.cu
	nvcc matmul.cu -O3 -D_FORCE_INLINES -arch=sm_52 -lcuda -D_CRT_SECURE_NO_DEPRECATE -o matmul.app -I $(CUDA_HOME)/samples/common/inc

