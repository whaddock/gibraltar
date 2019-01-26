BASE_DIR := $(HOME)
CEPH_DIR := $(BASE_DIR)/ceph/src
SOURCE := $(BASE_DIR)/gibraltar/src
EXAMPLES := $(BASE_DIR)/examples
LIB := $(BASE_DIR)/gibraltar/lib

SRC=\
	$(SOURCE)/gib_cpu_funcs.c		\
	$(SOURCE)/gib_cuda_driver.c 		\
	$(SOURCE)/gibraltar.c			\
	$(SOURCE)/gib_galois.c		\
	$(SOURCE)/gibraltar_cpu.c		\
	$(SOURCE)/gibraltar_jerasure.c	\

TESTS=\
	$(EXAMPLES)/benchmark		\
	$(EXAMPLES)/benchmark-2		\
	$(EXAMPLES)/sweeping_test		
#	$(EXAMPLES)/GibraltarTest		\
#	$(EXAMPLES)/GibraltarTest-2	\
#	$(EXAMPLES)/GibraltarCephTest

PARACRYPT_OBJS := \
	$(SOURCE)/CudaConstant.cu.o \
	$(SOURCE)/CudaAes16B.cu.o

# Expect CUDA library include directive to already be in CPPFLAGS,
# e.g. -I/usr/local/cuda/include
INCL := -I/usr/local/cuda/include
INCL += -Iinclude
INCL += -I$(CEPH_DIR)

CPPFLAGS := -DGIB_USE_MMAP=1
CPPFLAGS += -DLARGE_ENOUGH=1024*4
CPPFLAGS += -pg

# Expect CUDA library link directive to already be in LDFLAGS,
# .e.g. -L/usr/local/cuda/lib
NVCC := /usr/local/cuda/bin/nvcc
NVCC_FLAGS_ := 
LDFLAGS := -L/usr/local/cuda/lib
LDFLAGS += -L$(LIB)/
LDFLAGS += -L$(CEPH_DIR)/.libs

STATICLIBS := $(CEPH_DIR)/.libs/libosd.a
STATICLIBS += $(CEPH_DIR)/.libs/libosdc.a
STATICLIBS += $(CEPH_DIR)/.libs/libcommon.a
STATICLIBS += $(CEPH_DIR)/.libs/libglobal.a

CFLAGS += -Wall $(INCL)
LDLIBS=-lcuda -ljerasure -lrados -lprofiler

all: $(LIB)/libjerasure.a $(SOURCE)/libgibraltar.a $(TESTS) paracrypt

$(SOURCE)/libgibraltar.a: $(SOURCE)/libgibraltar.a($(SRC:.c=.o))

$(TESTS): $(SOURCE)/libgibraltar.a

$(EXAMPLES)/GibraltarCephTest: $(SOURCE)/libgibraltar.a
	g++ $(CPPFLAGS) $(LDFLAGS) $(STATICLIBS) $(INCL) \
		$(EXAMPLES)/GibraltarCephTest.cc $(SOURCE)/libgibraltar.a  $(LDLIBS) \
		 -o $(EXAMPLES)/GibraltarCephTest

$(LIB)/libjerasure.a:
	cd $(LIB)/Jerasure-1.2 && make
	ar rus $(LIB)/libjerasure.a $(LIB)/Jerasure-1.2/*.o

# Paracrypt code
$(SOURCE)/CudaConstant.cu.o: $(SOURCE)/CudaConstant.cu
	$(NVCC) $(NVCC_FLAGS_) -c $< -o $@ $(INCL)

$(SOURCE)/CudaAes16B.cu.o: $(SOURCE)/CudaAes16B.cu
	$(NVCC) $(NVCC_FLAGS_) -c $< -o $@ $(INCL)

paracrypt: $(PARACRYPT_OBJS) $(SOURCE)/libgibraltar.a
	ar rv $(SOURCE)/libgibraltar.a $(PARACRYPT_OBJS)
clean:
	rm -f $(LIB)/libjerasure.a $(SOURCE)/libgibraltar.a
	rm -f $(TESTS)
