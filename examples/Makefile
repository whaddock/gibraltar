all:
	nvcc -I../inc -I/usr/local/cuda/include/ --default-stream per-thread pthread_test.cu -o pthread_test -L../src/ -lgibraltar --gpu-architecture=sm_35 -L/usr/local/cuda/lib64/ -lcuda

.PHONY:  all
