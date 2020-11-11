SRC=\
	src/gib_cpu_funcs.c		\
	src/gib_cuda_driver.c 		\
	src/gibraltar.c			\
	src/gib_galois.c		\
	src/gibraltar_cpu.c		\
	src/gibraltar_jerasure.c	\
	openssl/aes_core.c	\
	openssl/AES_key_schedule.c \


TESTS=\
	examples/benchmark		\
	examples/benchmark_aes_k40	\
	examples/benchmark_isa-l_k40		\
	examples/sweeping_test		\

# Expect CUDA library include directive to already be in CPPFLAGS,
# e.g. -I/usr/local/cuda/include
# nvcc -I../inc -I/usr/local/cuda/include/ --default-stream per-thread pthread_test.cu -o pthread_test -L../src/ -lgibraltar --gpu-architecture=sm_35 -L/usr/local/cuda/lib64/ -lcuda
# Use newer version of C++
# scl enable devtoolset-7 bash
CPPFLAGS += -Iinc/ -I/usr/local/cuda/include/ -std=c++14
CPPFLAGS += -I/usr/include/ -lisal
CPPFLAGS += -DNSTREAMS=2 -Wpedantic -Wall -Wextra
# CPPFLAGS += -DDEF_STREAM
# Expect CUDA library link directive to already be in LDFLAGS,
# .e.g. -L/usr/local/cuda/lib
LDFLAGS += -Llib/ -L/usr/local/cuda/lib64 -L../src/ -L/usr/lib64 
#-fsanitize=address,undefined 

CFLAGS += -Wall -Wextra -Wpedantic -std=c11 -Iopenssl 
#-fsanitize=address,undefined 
CFLAGS += -DNSTREAMS=2 -Iinc/ -I/usr/local/cuda/include/
LDLIBS=-lcuda -lcudart -ljerasure -lpthread

all: lib/libjerasure.a src/libgibraltar.a $(TESTS)

src/libgibraltar.a: src/libgibraltar.a($(SRC:.c=.o))

$(TESTS): src/libgibraltar.a

lib/libjerasure.a:
	cd lib/Jerasure-1.2 && make
	ar rus lib/libjerasure.a lib/Jerasure-1.2/*.o

clean:
	rm -f lib/libjerasure.a src/libgibraltar.a
	rm -f $(TESTS)
