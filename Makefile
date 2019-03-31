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
	examples/sweeping_test		\

# Expect CUDA library include directive to already be in CPPFLAGS,
# e.g. -I/usr/local/cuda/include
# nvcc -I../inc -I/usr/local/cuda/include/ --default-stream per-thread pthread_test.cu -o pthread_test -L../src/ -lgibraltar --gpu-architecture=sm_35 -L/usr/local/cuda/lib64/ -lcuda
CPPFLAGS += -Iinc/ -I/usr/local/cuda/include/ -std=c++11
CPPFLAGS += -DNSTREAMS=2
# Expect CUDA library link directive to already be in LDFLAGS,
# .e.g. -L/usr/local/cuda/lib
LDFLAGS += -Llib/ -L/usr/local/cuda/lib64 -L../src/

CFLAGS += -Wall -std=c11 -Iopenssl
CFLAGS += -DNSTREAMS=2
LDLIBS=-lcuda -lcudart -ljerasure 

all: lib/libjerasure.a src/libgibraltar.a $(TESTS)

src/libgibraltar.a: src/libgibraltar.a($(SRC:.c=.o))

$(TESTS): src/libgibraltar.a

lib/libjerasure.a:
	cd lib/Jerasure-1.2 && make
	ar rus lib/libjerasure.a lib/Jerasure-1.2/*.o

clean:
	rm -f lib/libjerasure.a src/libgibraltar.a
	rm -f $(TESTS)
