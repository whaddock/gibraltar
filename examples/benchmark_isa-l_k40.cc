/* benchmark.cc: A simple Gibraltar benchmark
 *
 * Copyright (C) University of Alabama at Birmingham and Sandia
 * National Laboratories, 2010, written by Matthew L. Curry
 * <mlcurry@sandia.gov>, Rodrigo Sardinas <ras0054@tigermail.auburn.edu>
 * under contract to Sandia National Laboratories.
 *
 * Changes:
 * Initial version, Matthew L. Curry
 * Dec 16, 2014, Rodrigo Sardinas; revised to use new dynamic api.
 * Sep 5, 2020, Walker Haddock; streams with AES 256 GCM encryption
 */

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <sys/time.h>
#include <cstring>
#include <string>
#include <cstdio>
#include <thread>
#include <vector>
#include "isa-l.h"
#include "isa-l/types.h"

using namespace std;

typedef unsigned char u8;

#define MMAX 255
#define KMAX 255

#ifndef NSTREAMS
#define NSTREAMS 1
#endif
#ifndef min_test
#define min_test 120
#endif
#ifndef max_test
#define max_test 120
#endif
#ifndef SHARDS
#define SHARDS 24
#endif
#define ITERS 1000

const unsigned char *key = (unsigned char *)"F19142998DC13512706DADB657029C2AFF3FFB1901FC0D667E2294C66A2FBC2";

double
etime(void)
{
  /* Return time since epoch (in seconds) */
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + 1.e-6*t.tv_usec;
}

#define time_iters(var, cmd, iters) do {	\
    var = -1*etime();				\
    for (int iter = 0; iter < iters; iter++)	\
      cmd;					\
    var = (var + etime()) / iters;		\
  } while(0)

static int gf_gen_decode_matrix_simple(u8 * encode_matrix,
				       u8 * decode_matrix,
				       u8 * invert_matrix,
				       u8 * temp_matrix,
				       u8 * decode_index,
				       u8 * frag_err_list, int nerrs, int k, int m);

int
main(int argc, char **argv)
{
  /* The ISA-L examples use k for the number of data "fragments", p for the
   * number of parity "fragments" and m for k + p. We'll use that convention,
   * too so that I don't make any errors translating.
   */

  int iters = ITERS;
  int p = SHARDS;
  int k = min_test;
  size_t buf_size = 1024 * 1024 * 8;
  /* If k, p or iters is provided on the command line, they will be used. */
  if (argc >= 1) k = atoi(argv[1]);
  if (argc >= 2) p = atoi(argv[2]);
  if (argc >= 3) buf_size = atoi(argv[3]);
  if (argc >= 4) iters = atoi(argv[4]);
  int stripe_size = buf_size * (k + p);

  // Fragment buffer pointers
  u8 *frag_ptrs[MMAX];
  u8 *recover_srcs[KMAX];
  u8 *recover_outp[KMAX];
  u8 frag_err_list[MMAX];

  // Coefficient matrices
  u8 *encode_matrix, *decode_matrix;
  u8 *invert_matrix, *temp_matrix;
  u8 *g_tbls;
  u8 decode_index[MMAX];
  int m = k + p;

  // Allocate coding matrices
  encode_matrix = (u8*)malloc(m * k);
  decode_matrix = (u8*)malloc(m * k);
  invert_matrix = (u8*)malloc(m * k);
  temp_matrix = (u8*)malloc(m * k);
  g_tbls = (u8*)malloc(k * p * 32);

  if (encode_matrix == NULL || decode_matrix == NULL
      || invert_matrix == NULL || temp_matrix == NULL || g_tbls == NULL) {
    printf("Test failure! Error with malloc\n");
    return -1;
  }
  // Allocate the src & parity buffers
  for (int i = 0; i < m; i++) {
    if (NULL == (frag_ptrs[i] = (u8*)malloc(buf_size))) {
      printf("alloc error: Fail\n");
      return -1;
    }
  }

  // Fill sources with A-Z data
  for (int i = 0; i < k; i++)
    for (int j = 0; j < (int)buf_size; j++)
      ((char **)frag_ptrs)[i][j] = (char) ((i+j)%26 + 61);

  printf(" encode (m,k,p)=(%d,%d,%d) buf_size=%d iters=%d\n", m, k, p, buf_size, iters);
  printf("%% Speed test with correctness checks\n");
  printf("%% datasize is n*bufsize, or the total size of all data buffers\n");
  printf("%%                          cuda     cuda     cpu      cpu      jerasure jerasure\n");
  printf("%%      k        p datasize chk_tput rec_tput chk_tput rec_tput chk_tput rec_tput\n");

  //for (; p <= p; p++) {
  //for (; k <= n; k = k + 5) {
  printf("%8i %8i ", k, p);
  double chk_time, prep_time;
  prep_time = -1*etime();

  // Pick an encode matrix. A Cauchy matrix is a good choice as even
  // large k are always invertable keeping the recovery rule simple.
  gf_gen_cauchy1_matrix(encode_matrix, m, k);

  // Initialize g_tbls from encode matrix
  ec_init_tables(k, p, &encode_matrix[k * k], g_tbls);

  // Generate EC parity blocks from sources
  chk_time = -1*etime();
  for (int i = 0; i < iters; i++) {
    ec_encode_data(buf_size, k, p, g_tbls, frag_ptrs, &frag_ptrs[k]);
  }
  chk_time = (chk_time + etime());

  // Finished creating the buffers
  prep_time = prep_time + etime();	

  double size_mb = buf_size * k / 1024.0 / 1024.0;
  printf("%8lu ", buf_size * k);

  printf("%8.3lf GiB/s, %8.3lf total seconds \n", size_mb * iters / (chk_time * 1024), chk_time);
  printf("Create buffers time: %8.3lf\n", prep_time);

  prep_time = -1*etime();
  return 0;
}

