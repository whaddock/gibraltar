/* dynamic_gibraltar.c: Implementation of public interface for the
 * Gibraltar library
 *
 * Copyright (C) University of Alabama at Birmingham and Sandia
 * National Laboratories, 2010, written by and Rodrigo A. Sardinas
 * <ras0054@tigermail.auburn.edu>, under contract to Sandia National
 * Laboratories.
 *
 * Changes:
 * Initial version, Matthew L. Curry
 *
 */

#include "../inc/gibraltar.h"
#include "../inc/gib_context.h"
#include "../inc/dynamic_fp.h"

/* Functions */

int
set_encrypt_key(const unsigned char *userKey, gib_context c)
{
  return c->strategy->set_encrypt_key(userKey, c);
}

int
gib_destroy(gib_context c)
{
	return c->strategy->gib_destroy(c);
}

gib_free_gpu(gib_context c)
{
	return c->strategy->gib_free_gpu(c);
}

int
gib_alloc(void **buffers, size_t buf_size, size_t *ld, gib_context c)
{
	return c->strategy->gib_alloc(buffers, buf_size, ld, c);
}

int
gib_free(void *buffers, gib_context c)
{
	return c->strategy->gib_free(buffers, c);
}

int
gib_generate(void *buffers, size_t buf_size, int stream, gib_context c)
{
  return c->strategy->gib_generate(buffers, buf_size, stream, c);
}

int
gib_generate_nc(void *buffers, size_t buf_size, int work_size,
		    gib_context c)
{
	return c->strategy->gib_generate_nc(buffers, buf_size, work_size, c);
}

int
gib_recover(void *buffers, size_t buf_size, int *buf_ids, int recover_last,
		gib_context c)
{
	return c->strategy->gib_recover(buffers, buf_size, buf_ids,
					recover_last, c);
}

int
gib_recover_nc(void *buffers, size_t buf_size, int work_size, int *buf_ids,
		   int recover_last, gib_context c)
{
	return c->strategy->gib_recover_nc(buffers, buf_size, work_size,
					   buf_ids, recover_last, c);
}
