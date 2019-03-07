#include <pthread.h>

#include "../inc/gibraltar.h"
#include "../inc/gib_context.h"
#include "../inc/gib_galois.h"
#include "../inc/gib_cpu_funcs.h"
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>


#define N 120
#define M 24
#define S 4194304

/* gib_cuda_checksum.cu: CUDA kernels for Reed-Solomon coding.
 *
 * Copyright (C) University of Alabama at Birmingham and Sandia
 * National Laboratories, 2010, written by Matthew L. Curry
 * <mlcurry@sandia.gov>
 *
 * Changes:
 *
 */

typedef unsigned char byte;
__device__ unsigned char gf_log_d[256];
__device__ unsigned char gf_ilog_d[256];
__constant__ byte F_d[M*N];
__constant__ byte inv_d[N*N];

/* The "fetch" datatype is the unit for performing data copies between areas of
 * memory on the GPU.  While today's wisdom says that 32-bit types are optimal
 * for this, I want to easily experiment with the others.
 */
typedef int fetch;
#define nthreadsPerBlock 64

/* These quantities must be hand-recalculated, as the compiler doesn't seem to
 * always do such things at compile time.
 */
/* fetchsize = nthreadsPerBlock * sizeof(fetch) */
#define fetchsize 512
/* size of fetch, i.e. sizeof(fetch)*/
#define SOF 4
#define nbytesPerThread SOF 

#define ROUNDUPDIV(x,y) ((x + y - 1)/y)

/* We're pulling buffers from main memory based on the fetch type, but want
 * to index into it at the byte level.
 */
union shmem_bytes {
  fetch f;
  byte b[SOF];
};

/* Shared memory copies of pertinent data */
__shared__ byte sh_log[256];
__shared__ byte sh_ilog[256];

__device__ __inline__ void load_tables(uint3 threadIdx, const dim3 blockDim) {
  /* Fully arbitrary routine for any blocksize and fetch size to load
   * the log and ilog tables into shared memory.
   */
  int iters = ROUNDUPDIV(256,fetchsize);
  for (int i = 0; i < iters; i++) {
    if (i*fetchsize/SOF+threadIdx.x < 256/SOF) {
      int fetchit = threadIdx.x + i*fetchsize/SOF;
      ((fetch *)sh_log)[fetchit] = *(fetch *)(&gf_log_d[fetchit*SOF]);
      ((fetch *)sh_ilog)[fetchit] = *(fetch *)(&gf_ilog_d[fetchit*SOF]);
    }
  }
}

__global__ void gib_recover_d(shmem_bytes *bufs, int buf_size,
			      int recover_last) {
  /* Requirement: 
     buf_size % SOF == 0.  This prevents expensive divide operations. */
  int rank = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
  load_tables(threadIdx, blockDim);
	
  /* Load the data to shared memory */
  shmem_bytes out[M];
  shmem_bytes in;
	
  for (int i = 0; i < M; i++) 
    out[i].f = 0;

  __syncthreads();
  for (int i = 0; i < N; ++i) {
    /* Fetch the in-disk */
    in.f = bufs[rank+buf_size/SOF*i].f;
    for (int j = 0; j < recover_last; ++j) {
      /* Unless this is due to a drive bug, this conditional really
	 helps/helped on the 8000-series parts, but it hurts performance on 
	 the 260+.
      */
      //if (F_d[j*N+i] != 0) {
      int F_tmp = sh_log[F_d[j*N+i]]; /* No load conflicts */
      for (int b = 0; b < SOF; ++b) {
	if (in.b[b] != 0) {
	  int sum_log = F_tmp + sh_log[(in.b)[b]];
	  if (sum_log >= 255) sum_log -= 255;
	  (out[j].b)[b] ^= sh_ilog[sum_log];
	}
      }
      //}
    }
  }
  /* This works as long as buf_size % blocksize == 0 
   * TODO:  Ensure that allocation does this. */
  for (int i = 0; i < recover_last; i++) 
    bufs[rank+buf_size/SOF*(i+N)].f = out[i].f;
}

/* There is a bug affecting CUDA compilers from version 2.3 onward that causes
   this kernel to miscompile for M=2. For this case, there is some preprocessor
   trickiness that allows this kernel to generate M=3, but only store for M=2.
*/
__global__ void gib_checksum_d(shmem_bytes *bufs, int buf_size) {
  /* Requirement: 
     buf_size % SOF == 0.  This prevents expensive divide operations. */
  int rank = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
  load_tables(threadIdx, blockDim);
	
  /* Load the data to shared memory */
  shmem_bytes out[M];
  shmem_bytes in;
	
  for (int i = 0; i < M; i++) 
    out[i].f = 0;

  __syncthreads();
  for (int i = 0; i < N; ++i) {
    /* Fetch the in-disk */
    in.f = bufs[rank+buf_size/SOF*i].f;
    for (int j = 0; j < M; ++j) {
      /* If I'm not hallucinating, this conditional really
	 helps on the 8800 stuff, but it hurts on the 260.
      */
      int F_tmp = sh_log[F_d[j*N+i]]; /* No load conflicts */
      for (int b = 0; b < SOF; ++b) {
	if (in.b[b] != 0) {
	  int sum_log = F_tmp + sh_log[(in.b)[b]];
	  if (sum_log >= 255) sum_log -= 255;
	  (out[j].b)[b] ^= sh_ilog[sum_log];
	}
      }
    }
  }
  /* This works as long as buf_size % blocksize == 0 */
  for (int i = 0; i < M; i++) 
    bufs[rank+buf_size/SOF*(i+N)].f = out[i].f;
}

int cudaInitialized = 0;

struct gpu_context_t {
	CUdevice dev;
	CUmodule module;
	CUcontext pCtx;
	CUfunction checksum;
	CUfunction recover_sparse;
	CUfunction recover;
	CUdeviceptr buffers;
};

typedef struct gpu_context_t * gpu_context;

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

/*********************************************/
const int NW = 1 << 20;

__global__ void kernel(void *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

void *launch_kernel(void *dummy)
{
    void *data;
    cudaMalloc(&data, N * M * S);

    kernel<<<1, 64>>>(data, NW);

    cudaStreamSynchronize(0);

    return NULL;
}

int main(int argc, char* argv[])
{
    int n = N;
    int m = M;
    int size = S;

    int devId = 0;
    if (argc > 1) devId = atoi(argv[1]);

    cudaDeviceProp prop;
    checkCuda( cudaGetDeviceProperties(&prop, devId));
    printf("Device : %s\n", prop.name);
    checkCuda( cudaSetDevice(devId) );
    
    // allocate pinned host memory and device memory
    float *a, *d_a;
    checkCuda( cudaMallocHost((void**)&a, bytes) );      // host pinned
    checkCuda( cudaMalloc((void**)&d_a, bytes) ); // device

    /* Initialize the math libraries */
    gib_galois_init();
    unsigned char F[256*256];
    gib_galois_gen_F(F, m, n);

    /* Initialize/Allocate GPU-side structures */
    CUdeviceptr log_d, ilog_d, F_d;
    cudaMalloc(&log_d, sizeof(F));
    cudaMalloc(&ilog_d, sizeof(F));
    cudaMalloc(&F_d, sizeof(F));

    checkCuda(cuMemcpyHtoD(log_d, gib_gf_log, 256));
    checkCuda(cuMemcpyHtoD(ilog_d, gib_gf_ilog, 256));
    checkCuda(cuMemcpyHtoD(F_d, F, m*n));

    const int num_threads = 2;

    pthread_t threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        if (pthread_create(&threads[i], NULL, launch_kernel, 0)) {
            fprintf(stderr, "Error creating threadn");
            return 1;
        }
    }

    for (int i = 0; i < num_threads; i++) {
        if(pthread_join(threads[i], NULL)) {
            fprintf(stderr, "Error joining threadn");
            return 2;
        }
    }

    cudaDeviceReset();

    return 0;
}