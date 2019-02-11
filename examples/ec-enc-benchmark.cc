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
 */

#include "Paracrypt.hpp"
#include <gibraltar.h>
#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <cstring>
#include <cstdio>
#include <assert.h>

using namespace std;

#ifndef LARGE_ENOUGH
#define LARGE_ENOUGH 1024 * 1024
#endif
#ifndef min_test
#define min_test 2
#endif
#ifndef max_test
#define max_test 16
#endif

/*
 * --cipher aes-256-ctr
 * --Key 000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f 
 * --iv 000102030405060708090A0B0C0D0E0F
 * --encrypt / --decrypt
 */
#define KEY "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f" 
#define IV "000102030405060708090A0B0C0D0E0F"
#define CIPHER "aes-256-ctr"
#define KEY_SIZE 256

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

// WARNING: caller have to use delete
// WARNING: hexstring length has to be odd
unsigned char* hexstring2array(string hexstring)
{
	size_t length = hexstring.size();
	assert(length % 2 == 0);
	const char *pos = hexstring.c_str();

	size_t bytes = length/2;
	unsigned char* val = new unsigned char[bytes];
	size_t count = 0;

	/* WARNING: no sanitization or error-checking whatsoever */
	for(count = 0; count < bytes; count++) {
	     sscanf(pos, "%2hhx", &val[count]);
	     pos += 2;
	}

	return val;
}

int
main(int argc, char **argv)
{
  int iters = 5;
  printf("%% Encryption and erasure coding Speed test with correctness checks\n");
  printf("%% datasize is n*bufsize, or the total size of all data buffers\n");
  printf("%%                          cuda     cuda     cpu      cpu      jerasure jerasure\n");
  printf("%%      n        m datasize chk_tput rec_tput chk_tput rec_tput chk_tput rec_tput\n");

  // Load the encryption functions
  paracrypt::cipher_t c;
  paracrypt::operation_t op;
  string inFile, outFile;
  unsigned char* key;
  int key_bits;
  paracrypt::mode_t m;
  paracrypt::verbosity_t verbosity = paracrypt::WARNING;

  c = paracrypt::AES16B;
  key = hexstring2array(KEY);
  key_bits = KEY_SIZE;
  m = paracrypt::CTR;
  op = paracrypt::ENCRYPT;
  inFile = "/dev/zero";
  outFile = "/dev/null";
  paracrypt::config conf(c, op, inFile, outFile, key, key_bits, m);
  conf.setVerbosity(verbosity);
  unsigned char* iv = hexstring2array(IV);
  conf.setIV(iv, 128);
  conf.setStagingLimit(8388608);
  conf.streamLimit(4);

  // Call a gib_cuda function.
  for (int m = min_test; m <= min_test; m++) {
    for (int n = min_test; n <= min_test; n++) {
      printf("%8i %8i ", n, m);
      for (int j = 0; j < 3; j++) {
	double chk_time, dns_time;
	gib_context_t * gc;

	int rc;

	if (j == 0)
	  rc = gib_init_cuda(n, m, &gc);
	else if (j == 1)
	  rc = gib_init_cpu(n, m, &gc);
	else if (j == 2)
	  rc = gib_init_jerasure(n, m, &gc);

	if (rc) {
	  printf("Error:  %i\n", rc);
	  exit(EXIT_FAILURE);
	}

	int size = LARGE_ENOUGH;
	void *data;
	gib_alloc(&data, size, &size, gc);

	for (int i = 0; i < size * n; i++)
	  ((char *) data)[i] = (unsigned char) rand() % 256;

	time_iters(chk_time, gib_generate(data, size, gc), iters);

	unsigned char *backup_data = (unsigned char *)
	  malloc(size * (n + m));

	memcpy(backup_data, data, size * (n + m));

	char failed[256];
	for (int i = 0; i < n + m; i++)
	  failed[i] = 0;
	for (int i = 0; i < ((m < n) ? m : n); i++) {
	  int probe;
	  do {
	    probe = rand() % n;
	  } while (failed[probe] == 1);
	  failed[probe] = 1;

	  /* Destroy the buffer */
	  memset((char *) data + size * probe, 0, size);
	}

	int buf_ids[256];
	int index = 0;
	int f_index = n;
	for (int i = 0; i < n; i++) {
	  while (failed[index]) {
	    buf_ids[f_index++] = index;
	    index++;
	  }
	  buf_ids[i] = index;
	  index++;
	}
	while (f_index != n + m) {
	  buf_ids[f_index] = f_index;
	  f_index++;
	}

	void *dense_data;
	gib_alloc((void **) &dense_data, size, &size, gc);
	for (int i = 0; i < m + n; i++) {
	  memcpy((unsigned char *) dense_data + i * size,
		 (unsigned char *) data + buf_ids[i] * size, size);
	}

	int nfailed = (m < n) ? m : n;
	memset((unsigned char *) dense_data + n * size, 0,
	       size * nfailed);
	time_iters(dns_time,
		   gib_recover(dense_data, size, buf_ids, nfailed, gc),
		   iters);

	for (int i = 0; i < m + n; i++) {
	  if (memcmp((unsigned char *) dense_data + i * size,
		     backup_data + buf_ids[i] * size, size)) {
	    printf("Dense test failed on buffer %i/%i.\n", i,
		   buf_ids[i]);
	    exit(1);
	  }
	}

	double size_mb = size * n / 1024.0 / 1024.0;

	if(j==0) printf("%8i ", size * n);

	printf("%8.3lf %8.3lf ", size_mb / chk_time,
	       size_mb / dns_time);

	gib_free(data, gc);
	gib_free(dense_data, gc);
	free(backup_data);
	gib_destroy(gc);
      }
      printf("\n");
    }
  }
  return 0;
}
