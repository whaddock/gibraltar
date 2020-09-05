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

#include <gibraltar.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <sys/time.h>
#include <cstring>
#include <cstdio>
#include <thread>
#include <vector>

using namespace std;

#include <cuda_runtime_api.h>
#include <cuda.h>

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

const unsigned char *key = (unsigned char *)"F19142998DC13512706DADB657029C2AFF3FFB1901FC0D667E2294C66A2FBC24";

double
etime(void)
{
	/* Return time since epoch (in seconds) */
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec + 1.e-6*t.tv_usec;
}

#define time_iters(var, cmd, iters) do {				\
		var = -1*etime();					\
		for (int iter = 0; iter < iters; iter++)		\
			cmd;						\
		var = (var + etime()) / iters;				\
	} while(0)

void
checksumThread(void *ptr, size_t size, int stream, gib_context_t * gc, int count)
{
  for (int i = 0; i < count; i++) {
    gib_generate(ptr, size, stream, gc);
  }
}

int
main(int argc, char **argv)
{
	int iters = 100;
	printf("%% Speed test with correctness checks\n");
	printf("%% datasize is n*bufsize, or the total size of all data buffers\n");
	printf("%%                          cuda     cuda     cpu      cpu      jerasure jerasure\n");
	printf("%%      n        m datasize chk_tput rec_tput chk_tput rec_tput chk_tput rec_tput\n");

	for (int m = SHARDS; m <= SHARDS; m++) {
		for (int n = min_test; n <= max_test; n++) {
			printf("%8i %8i ", n, m);
				double chk_time, prep_time;
				prep_time = -1*etime();
				gib_context_t * gc;

				int rc;

				rc = gib_init_cuda(n, m, &gc);
				rc |= set_encrypt_key(key,gc);

				if (rc) {
					printf("Error:  %i\n", rc);
					exit(EXIT_FAILURE);
				}

				size_t buf_size = 1024 * 1024 * 8;
				int stripe_size = buf_size * (n + m);
				void *data;
				// So, how do we know that the amount of memory that will be 
				// allocated is m * n * NSTREAMS? What should the interface
				// look like?
				// gib_alloc creates a buffer of buf_size * n * m * threads bytes
				gib_alloc(&data, buf_size, &buf_size, gc);

				// Create the buffers to be encrypted and encoded
				for (int j = 0; j < NSTREAMS; j++) {
				  for (int i = j * stripe_size; i < stripe_size * (j + 1); i++)
				    ((char *) data)[i] = (unsigned char) (i%26 + 61);
				}
				// Finished creating the buffers
				prep_time = prep_time + etime();	

				do {
				  chk_time = -1*etime();
				  //checksumThread(void *ptr, size_t size, int stream, gib_context_t * gc, int count)
				  std::vector<std::thread> v_ct;
				  for (int thread = 0; thread < NSTREAMS; thread++) {
				    v_ct.push_back(std::thread (checksumThread, data, buf_size, 
								thread,
								gc, iters/NSTREAMS));
				  }
				  std::vector<std::thread>::iterator ct;
				  for (ct=v_ct.begin();ct!=v_ct.end();ct++)
				    ct->join(); // wait for the v_ct threads to finish.
				  chk_time = (chk_time + etime());
				} while(0);

				double size_mb = buf_size * n / 1024.0 / 1024.0;
				printf("%8i ", buf_size * n);

				printf("%8.3lf %8.3lf \n", size_mb * iters / (chk_time * 1000), chk_time);
				printf("Create buffers time: %8.3lf\n", prep_time);

				prep_time = -1*etime();
				gib_destroy(gc);
				printf("gib_desroy time: %8.3lf\n", prep_time + etime());	
		}
	}
	return 0;
}

