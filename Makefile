test1 : test1.cu
	nvcc test1.cu -O3 -D_FORCE_INLINES -arch=sm_52 -lcuda -D_CRT_SECURE_NO_DEPRECATE -o test1.app -I $(CUDA_HOME)/samples/common/inc

