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
			for (int j = 0; j < 1; j++) {
				double chk_time, dns_time;
				gib_context_t * gc;

				int rc;

				if (j == 0)
					rc = gib_init_cuda(n, m, &gc);
				else if (j == 1)
					rc = gib_init_cpu(n, m, &gc);
				else if (j == 2)
					rc = gib_init_jerasure(n, m, &gc);
				rc |= set_encrypt_key(key,gc);

				if (rc) {
					printf("Error:  %i\n", rc);
					exit(EXIT_FAILURE);
				}

				size_t size = 1024 * 1024 * 8;
				void *data;
				// So, how do we know that the amount of memory that will be 
				// allocated is m * n * NSTREAMS? What should the interface
				// look like?
				gib_alloc(&data, size, &size, gc);

				int stripe_size = size * (n + m);
				for (int j = 0; j < NSTREAMS; j++) {
				  for (int i = j * stripe_size; i < stripe_size * (j + 1); i++)
				    ((char *) data)[i] = (unsigned char) (i%26 + 61);
				  //((char *) data)[i] = (unsigned char) rand() % 256;
				}
				//gib_free(data, gc);
				//return 0;
				//#define time_iters(var, cmd, iters) do {
				//time_iters(chk_time, gib_generate(data, size, gc), iters);	
				// unsigned int *F;

				do {
				  chk_time = -1*etime();
				  //checksumThread(void *ptr, size_t size, int stream, gib_context_t * gc, int count)
				  std::vector<std::thread> v_ct;
				  for (int thread = 0; thread < NSTREAMS; thread++) {
				    v_ct.push_back(std::thread (checksumThread, data, size, 
								thread,
								gc, iters/NSTREAMS));
				  }
				  std::vector<std::thread>::iterator ct;
				  for (ct=v_ct.begin();ct!=v_ct.end();ct++)
				    ct->join(); // wait for the v_ct threads to finish.
				  chk_time = (chk_time + etime());
				} while(0);
				while(0) std::cerr << "Finished checksum. %lf\n" 
					  << etime() << std::endl << std::flush;

				// gib_free_gpu(gc); // finished with this

				std::cerr << std::hex << std::endl
					  << (unsigned int *)((char *)data)[0] << ","
					  << (unsigned int *)((char *)data)[1] << ","
					  << (unsigned int *)((char *)data)[2] << ","
					  << (unsigned int *)((char *)data)[3] << ","
					  << (unsigned int *)((char *)data)[4] << ","
					  << (unsigned int *)((char *)data)[5] << ","
					  << (unsigned int *)((char *)data)[6] << ","
					  << (unsigned int *)((char *)data)[7]
					  << std::endl;
				char *ptr = (char *)data + n*size;
				std::cerr << std::hex << std::endl
					  << (( int *)ptr[0]) << ","
					  << (( int *)ptr[1]) << ","
					  << (( int *)ptr[2]) << ","
					  << (( int *)ptr[3]) << ","
					  << (( int *)ptr[4]) << ","
					  << (( int *)ptr[5]) << ","
					  << (( int *)ptr[6]) << ","
					  << (( int *)ptr[7])
					  << std::endl;

				if (0) {
				void *dense_data;
				unsigned char *backup_data = (unsigned char *)
								malloc(size * (n + m));

				std::cout << "Starting decode." << std::endl << std::flush;

				memcpy(backup_data, data, size * (n + m));

				while(0) std::cerr << "Just released checksum memory." 
					  << std::endl << std::flush;
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

				unsigned int buf_ids[256];
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

				gib_alloc((void **) &dense_data, size, &size, gc);
				while(0) std::cerr << "Just allocated dense_data.\n" << std::endl << std::flush;
				for (int i = 0; i < m + n; i++) {
					memcpy((unsigned char *) dense_data + i * size,
							(unsigned char *) data + buf_ids[i] * size, size);
				}

				gib_free(data, gc); // finished with this now.

				int nfailed = (m < n) ? m : n;
				memset((unsigned char *) dense_data + n * size, 0,
						size * nfailed);
				while(0) std::cerr << "Calling gib_recover.\n" << std::endl << std::flush;
				time_iters(dns_time,
					   gib_recover2(dense_data, size, buf_ids, nfailed, gc),
					   iters);

				while(0) for (int i = 0; i < m + n; i++) {
					if (memcmp((unsigned char *) dense_data + i * size,
							backup_data + buf_ids[i] * size, size)) {
						printf("Dense test failed on buffer %i/%i.\n", i,
								buf_ids[i]);
						exit(1);
					}
				}
				gib_free(dense_data, gc);
				free(backup_data);
				}
				double size_mb = size * n / 1024.0 / 1024.0;

				if(j==0) printf("%8i ", size * n);

				printf("%8.3lf %8.3lf %8.3lf ", size_mb * iters / (chk_time * 3600), chk_time,
						size_mb / dns_time);

				gib_destroy(gc);
			}
			printf("\n");
		}
	}
	return 0;
}
