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
#include <cstdlib>
#include <sys/time.h>
#include <cstring>
#include <cstdio>
using namespace std;

#ifndef min_test
#define min_test 120
#endif
#ifndef max_test
#define max_test 24
#endif

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

int
main(int argc, char **argv)
{
	int iters = 5;
	printf("%% Speed test with correctness checks\n");
	printf("%% datasize is n*bufsize, or the total size of all data buffers\n");
	printf("%%                          cuda     cuda     cpu      cpu      jerasure jerasure\n");
	printf("%%      n        m datasize chk_tput rec_tput chk_tput rec_tput chk_tput rec_tput\n");

	for (int m = max_test; m <= max_test; m++) {
		for (int n = min_test; n <= min_test; n++) {
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

				if (rc) {
					printf("Error:  %i\n", rc);
					exit(EXIT_FAILURE);
				}

				int size = 1024 * 1024 * 8;
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
				// gib_free(dense_data, gc);
				// free(backup_data);
				gib_destroy(gc);
			}
			printf("\n");
		}
	}
	return 0;
}
