/* gib_cuda_driver.c: Host logic for CUDA
 *
 * Copyright (C) University of Alabama at Birmingham and Sandia
 * National Laboratories, 2010, written by Matthew L. Curry
 * <mlcurry@sandia.gov>, Rodrigo Sardinas <ras0054@tigermail.auburn.edu>
 * under contract to Sandia National Laboratories.
 *
 * Changes:
 * Initial version, Matthew L. Curry
 * Dec 16, 2014, Rodrigo Sardinas; revised to enable dynamic use.
 *
 */

/* TODO:
   - Noncontiguous only occurs on CPU!
*/

/* If compute capability 1.3 or higher is available, this should be set.
 * If it's set by the user at compile time, respect it.
 */
#ifndef GIB_USE_MMAP
#define GIB_USE_MMAP 1
#endif
//#define DEF_STREAM
#define OLD_STREAM
#ifndef NSTREAMS
#define NSTREAMS 1
#endif

/* Size of each GPU buffer; n+m will be allocated */
int gib_buf_size = 1024*1024*8;

const char env_error_str[] =
	"Your environment is not completely set. Please indicate a directory "
	"where\n Gibraltar kernel sources can be found. This should not be a "
	"publicly\naccessible directory.\n";

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
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>

int cudaInitialized = 0;
int stream = 0;

struct gpu_context_t {
	CUdevice dev;
	CUmodule module;
	CUcontext pCtx;
	CUfunction checksum;
	CUfunction recover_sparse;
	CUfunction recover;
	CUdeviceptr buffers;
  //CUdeviceptr stride;
#ifndef DEF_STREAM
        CUstream streams[NSTREAMS];
#endif
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

/* This macro checks for an error in the command given.  If it fails, the
 * entire program is killed.
 * TODO:  Fail over to CPU code if an error occurs.
 */
#define ERROR_CHECK_FAIL(cmd) {						\
		CUresult rc = cmd;					\
		if (rc != CUDA_SUCCESS) {				\
			fprintf(stderr, "%s failed with %i at "		\
				"%i in %s\n", #cmd, rc,                 \
				__LINE__,  __FILE__);			\
			exit(EXIT_FAILURE);				\
		}							\
	}

/* Massive performance increases come from compiling the CUDA kernels
   specifically for the coding process at hand.  This does so with the
   following command line:
      nvcc --ptx -DN=n -DM=m src/gib_cuda_checksum.cu -o gib_cuda_n+m.ptx
   This is called in a separate process fork'd from the original.  This
   function should never return, and the parent process should wait on the
   return code from the compiler before resuming operation.
*/
void
gib_cuda_compile(int n, int m, char *filename)
{
	/* Never returns */
	char *executable = "nvcc";

	if (getenv("PATH") == NULL) {
		fprintf(stderr, "Your path is not set.  Please set it, and "
			"include the path to nvcc.");
		exit(1);
	}

	char argv1[100], argv2[100];
	sprintf(argv1, "-DN=%i", n);
	sprintf(argv2, "-DM=%i", m);
	if (getenv("GIB_SRC_DIR") == NULL) {
		fprintf(stderr, "%s", env_error_str);
		exit(1);
	}

	char src_filename[100];
	sprintf(src_filename, "%s/gib_cuda_checksum.cu",
		getenv("GIB_SRC_DIR"));
	char *const argv[] = {
		executable,
		"--ptx",
		"--default-stream",
		"per-thread",
		"--gpu-architecture=sm_35",
		argv1,
		argv2,
		src_filename,
		"-o",
		filename,
		NULL
	};

	execvp(argv[0], argv);
	perror("execve(nvcc)");
	fflush(0);
	exit(-1);
}

int
gib_init_cuda(int n, int m, gib_context *c)
{

	/* Initializes the CPU and GPU runtimes. */
	static CUcontext pCtx;
	static CUdevice dev;
	if (m < 2 || n < 2) {
		fprintf(stderr, "It makes little sense to use Reed-Solomon "
			"coding when n or m is less than\ntwo. Use XOR or "
			"replication instead.\n");
		exit(1);
	}
	int rc_i = gib_cpu_init(n,m,c);
	if (rc_i != GIB_SUC) {
		fprintf(stderr, "gib_cpu_init returned %i\n", rc_i);
		exit(EXIT_FAILURE);
	}

	int gpu_id = 0;
	if (!cudaInitialized) {
		/* Initialize the CUDA runtime */
		int device_count;
		ERROR_CHECK_FAIL(cuInit(0));
		ERROR_CHECK_FAIL(cuDeviceGetCount(&device_count));
		if (getenv("GIB_GPU_ID") != NULL) {
			gpu_id = atoi(getenv("GIB_GPU_ID"));
			if (device_count <= gpu_id) {
				fprintf(stderr, "GIB_GPU_ID is set to an "
					"invalid value (%i).  There are only "
					"%i GPUs in the\n system.  Please "
					"specify another value.\n", gpu_id,
					device_count);
				exit(-1);
			}
		}
		cudaInitialized = 1;
	}
	ERROR_CHECK_FAIL(cuDeviceGet(&dev, gpu_id));
#if GIB_USE_MMAP
	ERROR_CHECK_FAIL(cuCtxCreate(&pCtx, CU_CTX_MAP_HOST, dev));
#else
	ERROR_CHECK_FAIL(cuCtxCreate(&pCtx, 0, dev));
#endif

	/* Initialize the Gibraltar context */
	gpu_context gpu_c = (gpu_context)malloc(sizeof(struct gpu_context_t));
	gpu_c->dev = dev;
	gpu_c->pCtx = pCtx;
	(*c)->acc_context = (void *)gpu_c;

	/* Determine whether the PTX has been generated or not by
	 * attempting to open it read-only.
	 */
	if (getenv("GIB_CACHE_DIR") == NULL) {
		fprintf(stderr, "%s", env_error_str);
		exit(-1);
	}

	/* Try to open the appropriate ptx file.  If it doesn't exist, compile a
	 * new one.
	 */
	int filename_len = strlen(getenv("GIB_CACHE_DIR")) +
		strlen("/gib_cuda_+.ptx") + log10(n)+1 + log10(m)+1 + 1;
	char *filename = (char *)malloc(filename_len);
	sprintf(filename, "%s/gib_cuda_%i+%i.ptx", getenv("GIB_CACHE_DIR"), n, m);

	FILE *fp = fopen(filename, "r");
	if (fp == NULL) {
		/* Compile the ptx and open it */
		int pid = fork();
		if (pid == -1) {
			perror("Forking for nvcc");
			exit(-1);
		}
		if (pid == 0) {
			gib_cuda_compile(n, m, filename); /* never returns */
		}
		int status;
		wait(&status);
		if (status != 0) {
			printf("Waiting for the compiler failed.\n");
			printf("The exit status was %i\n",
			       WEXITSTATUS(status));
			printf("The child did%s exit normally.\n",
			       (WIFEXITED(status)) ? "" : " NOT");

			exit(-1);
		}
		fp = fopen(filename, "r");
		if (fp == NULL) {
			perror(filename);
			exit(-1);
		}
	}
	fclose(fp);

	/* If we got here, the ptx file exists.  Use it. */
	ERROR_CHECK_FAIL(cuModuleLoad(&(gpu_c->module), filename));
	ERROR_CHECK_FAIL(
		cuModuleGetFunction(&(gpu_c->checksum),
				    (gpu_c->module),
				    "_Z14gib_checksum_dP11shmem_bytesi"));
	ERROR_CHECK_FAIL(
		cuModuleGetFunction(&(gpu_c->recover),
				    (gpu_c->module),
				    "_Z13gib_recover_dP11shmem_bytesii"));

	/* Initialize the math libraries */
	gib_galois_init();
	unsigned char F[256*256];
	gib_galois_gen_F(F, m, n);

	/* Initialize/Allocate GPU-side structures */
	CUdeviceptr log_d, ilog_d, F_d;
	ERROR_CHECK_FAIL(cuModuleGetGlobal(&log_d, NULL, gpu_c->module,
					   "gf_log_d"));
	ERROR_CHECK_FAIL(cuMemcpyHtoD(log_d, gib_gf_log, 256));
	ERROR_CHECK_FAIL(cuModuleGetGlobal(&ilog_d, NULL, gpu_c->module,
					   "gf_ilog_d"));
	ERROR_CHECK_FAIL(cuMemcpyHtoD(ilog_d, gib_gf_ilog, 256));
	ERROR_CHECK_FAIL(cuModuleGetGlobal(&F_d, NULL, gpu_c->module, "F_d"));
	ERROR_CHECK_FAIL(cuMemcpyHtoD(F_d, F, m*n));
#if !GIB_USE_MMAP
	ERROR_CHECK_FAIL(cuMemAlloc(&(gpu_c->buffers), (n+m)*gib_buf_size));
#endif
	ERROR_CHECK_FAIL(cuCtxPopCurrent((&gpu_c->pCtx)));
	free(filename);

	//set strategy for other functions
	(*c)->strategy = &cuda;

	return GIB_SUC;
}

static int
_gib_destroy(gib_context c)
{
	/* TODO:  Make sure everything created in gib_init is destroyed
	   here. */
	ERROR_CHECK_FAIL(
		cuCtxPushCurrent(((gpu_context)(c->acc_context))->pCtx));
	int rc_i = gib_cpu_destroy(c);
	if (rc_i != GIB_SUC) {
		printf("gib_cpu_destroy returned %i\n", rc_i);
		exit(EXIT_FAILURE);
	}
	gpu_context gpu_c = (gpu_context) c->acc_context;
#ifndef DEF_STREAM
	while(0) for (int i = 0; i < NSTREAMS; ++i)
	  ERROR_CHECK_FAIL( cuStreamDestroy(&gpu_c->streams[i]) );
#endif
	ERROR_CHECK_FAIL(cuModuleUnload(gpu_c->module));
	ERROR_CHECK_FAIL(cuMemFree(gpu_c->buffers));
	ERROR_CHECK_FAIL(cuCtxDestroy(gpu_c->pCtx));
	return GIB_SUC;
}

static int
_gib_alloc(void **buffers, size_t buf_size, size_t *ld, gib_context c)
{
	ERROR_CHECK_FAIL(
		cuCtxPushCurrent(((gpu_context)(c->acc_context))->pCtx));
	gpu_context gpu_c = (gpu_context) c->acc_context;
	while(0) fprintf(stderr,"allocating buffers. %i, %i, %i, %i\n", c->n,c->m,buf_size,NSTREAMS);
	ERROR_CHECK_FAIL(cuMemAllocHost(buffers, (c->n+c->m)*buf_size*NSTREAMS));
	ERROR_CHECK_FAIL(cuMemAlloc(&gpu_c->buffers, (c->n+c->m)*buf_size*NSTREAMS));
	//	ERROR_CHECK_FAIL(cudaMalloc(gpu_c->buffers, (c->n+c->m)*buf_size));

	*ld = buf_size;
	//ERROR_CHECK_FAIL(cuMemAlloc(&gpu_c->stride, sizeof(int)));
	//ERROR_CHECK_FAIL(cuMemcpyHtoD(gpu_c->stride,ld, sizeof(int)));
	// Create NSTREAMS CUDA streams
#ifndef DEF_STREAM
	for (int i = 0; i < NSTREAMS; ++i)
	  ERROR_CHECK_FAIL( cuStreamCreate(&gpu_c->streams[i], CU_STREAM_NON_BLOCKING) );
#endif
	ERROR_CHECK_FAIL(
		cuCtxPopCurrent(&((gpu_context)(c->acc_context))->pCtx));
	return GIB_SUC;
}

static int
_gib_free(void *buffers, gib_context c)
{
	ERROR_CHECK_FAIL(
		cuCtxPushCurrent(((gpu_context)(c->acc_context))->pCtx));
	gpu_context gpu_c = (gpu_context) c->acc_context;
	ERROR_CHECK_FAIL(cuMemFreeHost(buffers));
	ERROR_CHECK_FAIL(
		cuCtxPopCurrent(&((gpu_context)(c->acc_context))->pCtx));
	return GIB_SUC;
}

static int
_gib_free_gpu(gib_context c)
{
	ERROR_CHECK_FAIL(
		cuCtxPushCurrent(((gpu_context)(c->acc_context))->pCtx));
	gpu_context gpu_c = (gpu_context) c->acc_context;
	ERROR_CHECK_FAIL(cuMemFree(gpu_c->buffers));
	ERROR_CHECK_FAIL(
		cuCtxPopCurrent(&((gpu_context)(c->acc_context))->pCtx));
	return GIB_SUC;
}

static int
_gib_generate(void *buffers, size_t buf_size, int stream, gib_context c)
{
	ERROR_CHECK_FAIL(
		cuCtxPushCurrent(((gpu_context)(c->acc_context))->pCtx));
	/* Do it all at once if the buffers are small enough */

	/* This is too large to do at once in the GPU memory we have
	 * allocated.  Split it into several noncontiguous jobs.
	 */
	if (buf_size > gib_buf_size) {
		int rc = gib_generate_nc(buffers, buf_size, buf_size, c);
		ERROR_CHECK_FAIL(
			cuCtxPopCurrent(&((gpu_context)(c->acc_context))->pCtx));
		return rc;
	}

	int nthreads_per_block = 128;
	int fetch_size = sizeof(int)*nthreads_per_block;
	int nblocks = (buf_size + fetch_size - 1)/fetch_size;
	gpu_context gpu_c = (gpu_context) c->acc_context;

	// unsigned char F[256*256];
	// gib_galois_gen_F(F, c->m, c->n);
	// CUdeviceptr F_d;
	// ERROR_CHECK_FAIL(cuModuleGetGlobal(&F_d, NULL, gpu_c->module, "F_d"));
	// ERROR_CHECK_FAIL(cuMemcpyHtoD(F_d, F, (c->m)*(c->n)));
#ifdef DEF_STREAM
	/* Copy the buffers to memory */
	while(0) fprintf(stderr,"copying buffers. %i, %i, %i\n", c->n,c->m,buf_size);
	ERROR_CHECK_FAIL(
		cuMemcpyHtoD(gpu_c->buffers, buffers, (c->n)*buf_size));
	/* Configure and launch */
	ERROR_CHECK_FAIL(
		cuFuncSetBlockShape(gpu_c->checksum, nthreads_per_block, 1,
				    1));
	int offset = 0;
	void *ptr;
	ptr = (void *)(gpu_c->buffers);
	ERROR_CHECK_FAIL(
		cuParamSetv(gpu_c->checksum, offset, &ptr, sizeof(ptr)));
	offset += sizeof(ptr);
	ERROR_CHECK_FAIL(
		cuParamSetv(gpu_c->checksum, offset, &buf_size,
			    sizeof(buf_size)));
	offset += sizeof(buf_size);
	ERROR_CHECK_FAIL(cuParamSetSize(gpu_c->checksum, offset));
	while(0) fprintf(stderr,"launching checksum. %i, %i, %i\n", c->n,c->m,buf_size);
	ERROR_CHECK_FAIL(cuLaunchGrid(gpu_c->checksum, nblocks, 1));

	/* Get the results back */
	CUdeviceptr tmp_d = gpu_c->buffers + c->n*buf_size;
	void *tmp_h = (void *)((unsigned char *)(buffers) + c->n*buf_size);
	ERROR_CHECK_FAIL(cuMemcpyDtoH(tmp_h, tmp_d, (c->m)*buf_size));
#endif
#ifndef DEF_STREAM
#ifdef OLD_STREAM
	size_t stripe_offset = (size_t)stream * (c->n+c->m)*buf_size;
	/* Copy the buffers to memory */
	while(0) fprintf(stderr,"copying buffers. %i, %i, %i\n", c->n,c->m,buf_size);
	ERROR_CHECK_FAIL(
			 cuMemcpyAsync((void*)gpu_c->buffers + stripe_offset,
				       (void*)buffers + stripe_offset,
				       (c->n)*buf_size,
				       gpu_c->streams[stream]));
	/* Configure and launch */
	ERROR_CHECK_FAIL(
			 cuFuncSetBlockShape(gpu_c->checksum, nthreads_per_block, 1,
					     1));
	int offset = 0;
	void *ptr;
	ptr = (void *)(gpu_c->buffers + stripe_offset);
	ERROR_CHECK_FAIL(
			 cuParamSetv(gpu_c->checksum, offset, &ptr, sizeof(ptr)));
	offset += sizeof(ptr);
	ERROR_CHECK_FAIL(
			 cuParamSetv(gpu_c->checksum, offset, &buf_size,
				     sizeof(buf_size)));
	offset += sizeof(buf_size);
	ERROR_CHECK_FAIL(cuParamSetSize(gpu_c->checksum, offset));
	while(0) fprintf(stderr,"launching checksum. %i, %i, %i\n", c->n,c->m,buf_size);
	ERROR_CHECK_FAIL(cuLaunchGridAsync(gpu_c->checksum, nblocks, 1, gpu_c->streams[stream]));

	/* Get the results back */
	CUdeviceptr tmp_d = gpu_c->buffers + c->n*buf_size + stripe_offset;
	void *tmp_h = (void *)((unsigned char *)(buffers) + c->n*buf_size + stripe_offset);
	ERROR_CHECK_FAIL(
			 cuMemcpyAsync((void*)tmp_h,
				       (void*)tmp_d,
				       (c->m)*buf_size,
				       gpu_c->streams[stream]));
#else
	// int stripe_offset = i * (c->n+c->m)*buf_size;
	/* Copy the buffers to memory */
	ERROR_CHECK_FAIL(
			 cuMemcpyAsync((void*)gpu_c->buffers_offset,
				       (void*)buffers,
				       (c->n)*buf_size,
				       gpu_c->streams[stream]));
	while(0) fprintf(stderr,"Launching checksum kernel on GPU. nblocks: %d, nthreads/block: %d\n",
		nblocks,nthreads_per_block);
	/* Configure and launch */
	int *stride;
	stride = (int*)malloc(sizeof(int));
	while(0) fprintf(stderr,"buf_size: %d\n",buf_size);
	*stride = buf_size;
	void *kernelArgs[] = { &gpu_c->buffers, stride };
	ERROR_CHECK_FAIL(
			 cuLaunchKernel(gpu_c->checksum, nblocks, 1, 1, /* grid dim */
					nthreads_per_block, 1, 1, /* block dim */
					1, gpu_c->streams[stream],  /* shared mem, stream */
					kernelArgs,  /* arguments */
					0));
	/* Get the results back */
	CUdeviceptr tmp_d = gpu_c->buffers + c->n*buf_size;
	void *tmp_h = (void *)((unsigned char *)(buffers) + c->n*buf_size);
	ERROR_CHECK_FAIL(
			 cuMemcpyAsync((void*)tmp_h,
				       (void*)tmp_d,
				       (c->m)*buf_size,
				       gpu_c->streams[stream]));
	}
#endif
#endif
	stream = (stream + 1) % NSTREAMS;
	ERROR_CHECK_FAIL(
		cuCtxPopCurrent(&((gpu_context)(c->acc_context))->pCtx));
	return GIB_SUC;
}

static int
_gib_recover(void *buffers, size_t buf_size, int *buf_ids, int recover_last,
	    gib_context c)
{
	ERROR_CHECK_FAIL(
		cuCtxPushCurrent(((gpu_context)(c->acc_context))->pCtx));
	int i, j;
	int n = c->n;
	int m = c->m;
	unsigned char A[256*256], inv[256*256], modA[256*256];
	for (i = n; i < n+recover_last; i++)
		if (buf_ids[i] >= n) {
			fprintf(stderr, "Attempting to recover a parity "
				"buffer, not allowed\n");
			return GIB_ERR;
		}

	gib_galois_gen_A(A, m+n, n);

	/* Modify the matrix to have the failed drives reflected */
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			modA[i*n+j] = A[buf_ids[i]*n+j];

	gib_galois_gaussian_elim(modA, inv, n, n);

	/* Copy row buf_ids[i] into row i */
	for (i = n; i < n+recover_last; i++)
		for (j = 0; j < n; j++)
			modA[i*n+j] = inv[buf_ids[i]*n+j];

	while(0) fprintf(stderr,"finished GF. %i, %i, %i\n", c->n,c->m,buf_size);

	int nthreads_per_block = 128;
	int fetch_size = sizeof(int)*nthreads_per_block;
	int nblocks = (buf_size + fetch_size - 1)/fetch_size;
	gpu_context gpu_c = (gpu_context) c->acc_context;

	CUdeviceptr F_d;
	ERROR_CHECK_FAIL(cuModuleGetGlobal(&F_d, NULL, gpu_c->module, "F_d"));
	while(0) fprintf(stderr,"got F_d pointer. %i, %i, %i\n", c->n,c->m,buf_size);
	ERROR_CHECK_FAIL(cuMemcpyHtoD(F_d, modA+n*n, (c->m)*(c->n)));

	while(0) fprintf(stderr,"finished copying GF. %i, %i, %i\n", c->n,c->m,buf_size);
	ERROR_CHECK_FAIL(cuFuncSetBlockShape(gpu_c->recover,
					     nthreads_per_block, 1, 1));
	//size_t stripe_offset = (size_t)stream * (c->n+c->m)*buf_size;
	/* Copy the buffers to memory */
	while(0) fprintf(stderr,"copying buffers. %i, %i, %i\n", c->n,c->m,buf_size);
	ERROR_CHECK_FAIL(
			 cuMemcpyAsync((void*)gpu_c->buffers,
				       (void*)buffers,
				       (c->n)*buf_size,
				       gpu_c->streams[stream]));

	int offset = 0;
	void *ptr;
	ptr = (void *)(gpu_c->buffers);

	ERROR_CHECK_FAIL(cuParamSetv(gpu_c->recover, offset, &ptr,
				     sizeof(ptr)));
	offset += sizeof(ptr);
	ERROR_CHECK_FAIL(cuParamSetv(gpu_c->recover, offset, &buf_size,
				     sizeof(buf_size)));
	offset += sizeof(buf_size);
	ERROR_CHECK_FAIL(cuParamSetv(gpu_c->recover, offset, &recover_last,
				     sizeof(recover_last)));
	offset += sizeof(recover_last);
	ERROR_CHECK_FAIL(cuParamSetSize(gpu_c->recover, offset));
	while(0) fprintf(stderr,"Launching gib_recover kernel\n.");
	ERROR_CHECK_FAIL(cuLaunchGridAsync(gpu_c->recover, nblocks, 1, gpu_c->streams[stream]));

	CUdeviceptr tmp_d = gpu_c->buffers + c->n*buf_size;
	void *tmp_h = (void *)((unsigned char *)(buffers) + c->n*buf_size);
	while(0) fprintf(stderr,"Copying back from gib_recover\n.");
	ERROR_CHECK_FAIL(
			 cuMemcpyAsync((void*)tmp_h,
				       (void*)tmp_d,
				       recover_last*buf_size,
				       gpu_c->streams[stream]));
	while(0) fprintf(stderr,"Stream: %d\n.",stream);
	stream = (stream + 1) % NSTREAMS;

	ERROR_CHECK_FAIL(
		cuCtxPopCurrent(&((gpu_context)(c->acc_context))->pCtx));
	return GIB_SUC;
}

/* The inclusion of memory mapping has obviated the need for this before
   it was implemented.  It's done in the CPU to make it work, but there is
   no attempt to make it fast as there appears to be little need.  A GPU
   upgrade fixes this.

   TODO:  The MMapped version can benefit from this if the buffer isn't full.
   Bring this to life for that implementation only.
 */
static int
_gib_generate_nc(void *buffers, size_t buf_size, int work_size,
		gib_context c)
{
	return gib_cpu_generate_nc(buffers, buf_size, work_size, c);
}

static int
_gib_recover_nc(void *buffers, size_t buf_size, int work_size, int *buf_ids,
		int recover_last, gib_context c)
{
	return gib_cpu_recover_nc(buffers, buf_size, work_size, buf_ids,
				  recover_last, c);
}


struct dynamic_fp cuda = {
		.gib_alloc = &_gib_alloc,
		.gib_destroy = &_gib_destroy,
		.gib_free_gpu = &_gib_free_gpu,
		.gib_free = &_gib_free,
		.gib_generate = &_gib_generate,
		.gib_generate_nc = &_gib_generate_nc,
		.gib_recover = &_gib_recover,
		.gib_recover_nc = &_gib_recover_nc,
};

