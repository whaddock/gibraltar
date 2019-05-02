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

#include "../inc/gib_context.h"
#include "../inc/gib_galois.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include "AES_key_schedule.h"


int cudaInitialized = 0;

const int P_LENGTH = 16;
const unsigned char *P = (unsigned char *)"00112233445566778899aabbccddeeff";

const int C_LENGTH = 16;
const unsigned char *C = (unsigned char *)"8ea2b7ca516745bfeafc49904b496089";
//                                       0xA9F720C1E2DC28C1A5BE5881BB90E9BD
//                                       0xA756F64C995E4927CF755C7795F478F0
const int KEY_LENGTH = 32;
const unsigned char *key = (unsigned char *)"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
/*
const unsigned char key[32] = {
      0x60U, 0x3dU, 0xebU, 0x10U, 0x15U, 0xcaU, 0x71U, 0xbeU,
      0x2bU, 0x73U, 0xaeU, 0xf0U, 0x85U, 0x7dU, 0x77U, 0x81U,
      0x1fU, 0x35U, 0x2cU, 0x07U, 0x3bU, 0x61U, 0x08U, 0xd7U,
      0x2dU, 0x98U, 0x10U, 0xa3U, 0x09U, 0x14U, 0xdfU, 0xf4U
  };
*/
const int IV_LENGTH = 12;
const unsigned char iv[12] = {
  0x00U, 0x01U, 0x02U, 0x03,
  0x04U, 0x05U, 0x06U, 0x07,
  0x08U, 0x09U, 0x0AU, 0x0B
};

struct gpu_context_t {
  CUdevice dev;
  CUmodule module;
  CUcontext pCtx;
  CUfunction aes_gcm_encrypt;
  CUfunction checksum;
  CUfunction recover_sparse;
  CUfunction recover;
  CUdeviceptr buffers;
  AES_KEY * enRoundKeys;
  //CUstream streams[NSTREAMS];
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
#define ERROR_CHECK_FAIL(cmd) {			\
    CUresult rc = cmd;				\
    if (rc != CUDA_SUCCESS) {			\
      fprintf(stderr, "%s failed with %i at "	\
	      "%i in %s\n", #cmd, rc,		\
	      __LINE__,  __FILE__);		\
      exit(EXIT_FAILURE);			\
    }						\
  }

char* be2le(char* input, int length)
{
  int llength = length;
  // Caller must free memory after use, i.e. free(tmp)
  char *result = malloc(sizeof(char)*llength+1);
  // Should always be even, just in case it is not.
  if(! (length % 2))
    llength--;
  // Reverse each two bytes
  char tmp;
  for(int i=0;i<llength;i += 2) {
    tmp = input[i];
    result[i] = input[i+1];
    result[i+1] = tmp;
  }
  // Reverse each 4 bytes
  char tmp2;
  for(int i=0;i<llength;i += 4) {
    tmp = result[i];
    tmp2 = result[i+1];
    result[i] = result[i+2];
    result[i+1] = result[i+3];
    result[i+2] = tmp;
    result[i+3] = tmp2;
  }
  result[llength] = 0;
  return result;
}
int print_hex(char* input, int length)
{
  char *out = malloc(2*length*sizeof(char)+1);
  int pos = 0;
  for (;pos<length;pos++) {
    for (int j=0;j<2;j++) {
      out[2*pos+j] = input[pos] >> 4*((j+1)%2) & 0x0f;
      if ((int)out[2*pos+j] > 9) {
	out[2*pos+j] = (char)((int)out[2*pos+j] - 0x09);
	out[2*pos+j] = (char)((int)out[2*pos+j] | 0x40);
      }
      else
	out[2*pos+j] = (char)((int)out[2*pos+j] + 0x30);
    }
  } 
  out[2*pos] = 0x00;
  fprintf(stderr,"0x%s\n",out);
  free(out);
  return 0;
}

int my_set_encrypt_key(const unsigned char *userKey, gib_context c)
{
  ERROR_CHECK_FAIL(
		   cuCtxPushCurrent(((gpu_context)(c->acc_context))->pCtx));
  gpu_context gpu_c = (gpu_context) c->acc_context;
  fprintf(stderr,"Size of AES_Key: %lu\n",sizeof(AES_KEY));
  gpu_c->enRoundKeys = malloc(sizeof(AES_KEY));
  int r = AES_set_encrypt_key(userKey, 256,
			      gpu_c->enRoundKeys);

  CUdeviceptr aes_key_d, iv_d;
  ERROR_CHECK_FAIL(cuModuleGetGlobal(&aes_key_d, NULL, gpu_c->module, "aes_key"));
  ERROR_CHECK_FAIL(cuMemcpy(aes_key_d, gpu_c->enRoundKeys, sizeof(AES_KEY)));
  ERROR_CHECK_FAIL(cuModuleGetGlobal(&iv_d, NULL, gpu_c->module, "iv_d"));
  ERROR_CHECK_FAIL(cuMemcpy(iv_d, iv, IV_LENGTH));
  ERROR_CHECK_FAIL(
		   cuCtxPopCurrent(&((gpu_context)(c->acc_context))->pCtx));
  return r | 0;
}

int
main(int argc, char **argv)
{
  gib_context * gc;
  int rc;

  /* Initializes the CPU and GPU runtimes. */
  static CUcontext pCtx;
  static CUdevice dev;

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
  ERROR_CHECK_FAIL(cuCtxCreate(&pCtx, 0, dev));

  // For reporting the memory allocation on the GPU
  unsigned long long int bytes = 0;

  /* Initialize the Gibraltar context */
  gpu_context gpu_c = (gpu_context)malloc(sizeof(struct gpu_context_t));
  gpu_c->dev = dev;
  gpu_c->pCtx = pCtx;
  //  (*c)->acc_context = (void *)gpu_c;

  /* Try to open the appropriate ptx file.  If it doesn't exist, compile a
   * new one.
   */
  int filename_len = strlen(getenv("PWD")) +
    strlen("/gib_cuda_crypto.ptx") +1;
  char *filename = (char *)malloc(filename_len);
  sprintf(filename, "%s/gib_cuda_crypto.ptx", getenv("PWD"));

  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Open of gib_cuda_crypto.ptx failed.\n");
    exit(-1);
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
  ERROR_CHECK_FAIL(
		   cuModuleGetFunction(&(gpu_c->aes_gcm_encrypt),
				       (gpu_c->module),
				       "_Z24__cuda_aes_16b_encrypt__jPKjPjS1_S1_iS1_S1_S1_S1_"));
  
  void *buffers_h = NULL;
  ERROR_CHECK_FAIL(cuMemAllocHost(&buffers_h, C_LENGTH));

  /* Initialize the math libraries */
#ifdef GF
  gib_galois_init();
  unsigned char F[256*256];
  gib_galois_gen_F(F, m, n);

  /* Initialize/Allocate GPU-side structures */
  CUdeviceptr log_d, ilog_d, F_d;
  ERROR_CHECK_FAIL(cuModuleGetGlobal(&log_d, &bytes, gpu_c->module,
				     "gf_log_d"));
  fprintf(stderr,"Size of gf_log_d: %u\n",bytes);
  ERROR_CHECK_FAIL(cuMemcpy(log_d, gib_gf_log, 256));
  ERROR_CHECK_FAIL(cuModuleGetGlobal(&ilog_d, &bytes, gpu_c->module,
				     "gf_ilog_d"));
  fprintf(stderr,"Size of gf_ilog_d: %u\n",bytes);
  ERROR_CHECK_FAIL(cuMemcpy(ilog_d, gib_gf_ilog, 256));
  ERROR_CHECK_FAIL(cuModuleGetGlobal(&F_d, &bytes, gpu_c->module, "F_d"));
  fprintf(stderr,"Size of F_d: %u\n",bytes);
  ERROR_CHECK_FAIL(cuMemcpy(F_d, F, m*n));
  //ERROR_CHECK_FAIL(cuCtxPopCurrent((&gpu_c->pCtx)));
#endif
  free(filename);

  /* AES configuration */
  int nthreads_per_block = 128;
  int fetch_size = sizeof(int)*nthreads_per_block;
  int nblocks = (P_LENGTH + fetch_size - 1)/fetch_size;
  int key_bits = 256;
  size_t aes_blocks = P_LENGTH/(4*4); // blocks of 4 32-bit words
  //size_t stripe_offset = (size_t)stream * (c->n+c->m)*buf_size;
  void *ptr_h = (void*)((char *)P);

  // int stripe_offset = i * (c->n+c->m)*buf_size;
  /* Copy the buffers to memory */
  fprintf(stderr,"Starting AES configuration\nAES Key:\n");
  print_hex((char*)key,KEY_LENGTH);

  ERROR_CHECK_FAIL(cuMemAlloc(&gpu_c->buffers, P_LENGTH));
  fprintf(stderr,"gpu_c->buffers: %p\n",gpu_c->buffers);
  fprintf(stderr,"gpu_c->buffers size: %u\n",P_LENGTH);

  CUdeviceptr out_d;
  ERROR_CHECK_FAIL(cuMemAlloc(&out_d, C_LENGTH));
  fprintf(stderr,"out_d: %p\n",out_d);
  fprintf(stderr,"out_d size: %u\n",C_LENGTH);

  fprintf(stderr,"Starting AES cuMemcpy\n\n");
  ERROR_CHECK_FAIL(cuMemcpy(gpu_c->buffers,
			    ptr_h,
			    P_LENGTH));

  fprintf(stderr,"%s\n",ptr_h);
  print_hex((char*)ptr_h,P_LENGTH);


  fprintf(stderr,"Size of AES Round Key: %lu\n",sizeof(AES_KEY));
  gpu_c->enRoundKeys = malloc(sizeof(AES_KEY));
  int r = AES_set_encrypt_key(key, 256,
			      gpu_c->enRoundKeys);
  char *tmp = be2le((char*)gpu_c->enRoundKeys,sizeof(AES_KEY));
  print_hex(tmp,sizeof(AES_KEY));
  free(tmp);

  CUdeviceptr aes_key_d, iv_d, T0_d, T1_d, T2_d, T3_d;
  ERROR_CHECK_FAIL(cuModuleGetGlobal(&aes_key_d, &bytes, gpu_c->module,
				     "aes_key_d"));
  fprintf(stderr,"aes_key_d: %p\n",aes_key_d);
  fprintf(stderr,"Size of aes_key_d: %u\n",bytes);
  ERROR_CHECK_FAIL(cuMemcpy(aes_key_d, gpu_c->enRoundKeys, sizeof(AES_KEY)));
  ERROR_CHECK_FAIL(cuModuleGetGlobal(&iv_d, &bytes, gpu_c->module,
				     "iv_d"));
  fprintf(stderr,"iv_d: %p\n",iv_d);
  fprintf(stderr,"Size of iv_d: %u\n",bytes);
  ERROR_CHECK_FAIL(cuMemcpy(iv_d, iv, 16));

  ERROR_CHECK_FAIL(cuModuleGetGlobal(&T0_d, &bytes, gpu_c->module, "aes_Te0"));
  fprintf(stderr,"aes_Te0: %p\n",T0_d);
  fprintf(stderr,"Size of aes_Te0: %u\n",bytes);
  ERROR_CHECK_FAIL(cuModuleGetGlobal(&T1_d, &bytes, gpu_c->module, "aes_Te1"));
  fprintf(stderr,"aes_Te1: %p\n",T1_d);
  fprintf(stderr,"Size of aes_Te1: %u\n",bytes);
  ERROR_CHECK_FAIL(cuModuleGetGlobal(&T2_d, &bytes, gpu_c->module, "aes_Te2"));
  fprintf(stderr,"aes_Te2: %p\n",T2_d);
  fprintf(stderr,"Size of aes_Te2: %u\n",bytes);
  ERROR_CHECK_FAIL(cuModuleGetGlobal(&T3_d, &bytes, gpu_c->module, "aes_Te3"));
  fprintf(stderr,"aes_Te3: %p\n",T3_d);
  fprintf(stderr,"Size of aes_Te3: %u\n",bytes);
  void * args[10] = { &aes_blocks, &gpu_c->buffers, &out_d, &iv_d, &aes_key_d, &key_bits,
		      &T0_d, &T1_d, &T2_d, &T3_d };

  fprintf(stderr,"Number of blocks: %u\n",aes_blocks);
  ERROR_CHECK_FAIL(cuLaunchKernel(gpu_c->aes_gcm_encrypt, nblocks, 1, 1, /* grid dim */
				  nthreads_per_block, 1, 1, /* block dim */
				  16, 0, /* gpu_c->streams[stream], ** shared mem, stream */
				  args,  /* arguments */
				  0));

  /* Copy the encrypted data back to host. */

  ERROR_CHECK_FAIL(cuMemcpy(buffers_h,
			    out_d,
			    C_LENGTH));
  print_hex((char*)buffers_h,C_LENGTH);

  ERROR_CHECK_FAIL(cuModuleUnload(gpu_c->module));
  ERROR_CHECK_FAIL(cuCtxDestroy(gpu_c->pCtx));

  return 0;
}
