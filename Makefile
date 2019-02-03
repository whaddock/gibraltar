BASE_DIR := $(HOME)/git
CEPH_DIR := $(BASE_DIR)/ceph/src
SOURCE := $(BASE_DIR)/gibraltar/src
EXAMPLES := $(BASE_DIR)/gibraltar/examples
LIB := $(BASE_DIR)/gibraltar/lib

BOOST_LIB =-lboost_program_options -lboost_system -lboost_thread \
           -lpthread -lboost_system-mt
BOOST_LIB_DIR = -L$(HOME)/boost/lib
BOOST_INC = -I$(HOME)/boost/include

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
	$(SOURCE)/CUDACipherDevice.o \
	$(SOURCE)/BlockCipher.o \
	$(SOURCE)/CUDABlockCipher.o \
	$(SOURCE)/CudaAesVersions.o \
	$(SOURCE)/AES.o \
	$(SOURCE)/AES_key_schedule.o \
	$(SOURCE)/endianess.o \
	$(SOURCE)/IO.o \
	$(SOURCE)/BlockIO.o \
	$(SOURCE)/SimpleIO.o \
	$(SOURCE)/SimpleCudaIO.o \
	$(SOURCE)/SharedIO.o \
	$(SOURCE)/CudaSharedIO.o \
	$(SOURCE)/Pinned.o \
	$(SOURCE)/CudaPinned.o \
	$(SOURCE)/CudaAES.o \
	$(SOURCE)/Launcher.o \
	$(SOURCE)/Paracrypt.o

PARACRYPT_CUDA_OBJS := \
	$(SOURCE)/CudaConstant.cu.o \
	$(SOURCE)/CudaAes16B.cu.o

# Expect CUDA library include directive to already be in CPPFLAGS,
# e.g. -I/usr/local/cuda/include
INCL := -I/usr/local/cuda/include
INCL += -I$(BASE_DIR)/gibraltar/include
INCL += -I$(CEPH_DIR)

CPPFLAGS := -DGIB_USE_MMAP=1
CPPFLAGS += -DLARGE_ENOUGH=1024*4
CPPFLAGS += -pg -std=c++11

# Expect CUDA library link directive to already be in LDFLAGS,
# .e.g. -L/usr/local/cuda/lib
NVCC := /usr/local/cuda/bin/nvcc
NVCC_FLAGS_ := 
LDFLAGS := -L/usr/local/cuda/lib64
LDFLAGS += -L$(LIB)/
LDFLAGS += -L$(CEPH_DIR)/.libs

STATICLIBS := $(CEPH_DIR)/libosd.a
STATICLIBS += $(CEPH_DIR)/.libs/libosdc.a
STATICLIBS += $(CEPH_DIR)/.libs/libcommon.a
STATICLIBS += $(CEPH_DIR)/.libs/libglobal.a

CFLAGS += -Wall $(INCL)
LDLIBS=-lcuda -lcudart -lrados -lprofiler

all: $(LIB)/libjerasure.a $(SOURCE)/libgibraltar.a paracrypt benchmark

$(SOURCE)/libgibraltar.a: $(SOURCE)/libgibraltar.a($(SRC:.c=.o))

$(TESTS): $(SOURCE)/libgibraltar.a

benchmark: $(EXAMPLES)/benchmark.cc $(SOURCE)/libgibraltar.a
	g++ $(CPPFLAGS) $(LDFLAGS) $(STATICLIBS) $(INCL) \
	$(EXAMPLES)/benchmark.cc $(SOURCE)/libgibraltar.a  $(LDLIBS) \
	$(LIB)/libjerasure.a -o $(EXAMPLES)/benchmark 

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

$(SOURCE)/CUDACipherDevice.o: $(SOURCE)/CUDACipherDevice.cpp
	g++ $(CPPFLAGS) $(LDFLAGS) $(INCL) -c $< -o $@ $(INCL)

$(SOURCE)/BlockCipher.o: $(SOURCE)/BlockCipher.cpp
	g++ $(CPPFLAGS) $(LDFLAGS) $(INCL) -c $< -o $@ $(INCL)

$(SOURCE)/CUDABlockCipher.o: $(SOURCE)/CUDABlockCipher.cpp
	g++ $(CPPFLAGS) $(LDFLAGS) $(INCL) -c $< -o $@ $(INCL)

$(SOURCE)/CudaAesVersions.o: $(SOURCE)/CudaAesVersions.cpp
	g++ $(CPPFLAGS) $(LDFLAGS) $(INCL) -c $< -o $@ $(INCL)

$(SOURCE)/AES.o: $(SOURCE)/AES.cpp
	g++ $(CPPFLAGS) $(LDFLAGS) $(INCL) -c $< -o $@ $(INCL)

$(SOURCE)/endianess.o: $(SOURCE)/endianess.c
	g++ $(CPPFLAGS) $(LDFLAGS) $(INCL) -c $< -o $@ $(INCL)

$(SOURCE)/AES_key_schedule.o: $(SOURCE)/AES_key_schedule.c
	g++ $(CPPFLAGS) $(LDFLAGS) $(INCL) -c $< -o $@ $(INCL)

$(SOURCE)/IO.o: $(SOURCE)/IO.cpp
	g++ $(CPPFLAGS) $(LDFLAGS) $(INCL) -c $< -o $@ $(INCL)

$(SOURCE)/BlockIO.o: $(SOURCE)/BlockIO.cpp
	g++ $(CPPFLAGS) $(LDFLAGS) $(INCL) -c $< -o $@ $(INCL)

$(SOURCE)/SimpleIO.o: $(SOURCE)/SimpleIO.cpp
	g++ $(CPPFLAGS) $(LDFLAGS) $(INCL) -c $< -o $@ $(INCL)

$(SOURCE)/SimpleCudaIO.o: $(SOURCE)/SimpleCudaIO.cpp
	g++ $(CPPFLAGS) $(LDFLAGS) $(INCL) -c $< -o $@ $(INCL)

$(SOURCE)/SharedIO.o: $(SOURCE)/SharedIO.cpp
	g++ $(CPPFLAGS) $(LDFLAGS) $(INCL) -c $< -o $@ $(INCL)

$(SOURCE)/CudaSharedIO.o: $(SOURCE)/CudaSharedIO.cpp
	g++ $(CPPFLAGS) $(LDFLAGS) $(INCL) -c $< -o $@ $(INCL)

$(SOURCE)/Pinned.o: $(SOURCE)/Pinned.cpp
	g++ $(CPPFLAGS) $(LDFLAGS) $(INCL) -c $< -o $@ $(INCL)

$(SOURCE)/CudaPinned.o: $(SOURCE)/CudaPinned.cpp
	g++ $(CPPFLAGS) $(LDFLAGS) $(INCL) -c $< -o $@ $(INCL)

$(SOURCE)/CudaAES.o: $(SOURCE)/CudaAES.cpp
	g++ $(CPPFLAGS) $(LDFLAGS) $(INCL) -c $< -o $@ $(INCL)

$(SOURCE)/Launcher.o:
	g++  $(SOURCE)/Launcher.cpp $(CPPFLAGS) $(LDFLAGS) $(INCL) -c $< -o $@ $(INCL)

$(SOURCE)/Paracrypt.o:
	g++  $(SOURCE)/Paracrypt.cpp $(CPPFLAGS) $(LDFLAGS) $(INCL) -c $< -o $@ $(INCL)

paracrypt: $(PARACRYPT_OBJS) $(PARACRYPT_CUDA_OBJS) $(SOURCE)/libgibraltar.a
	ar rv $(SOURCE)/libgibraltar.a $(PARACRYPT_OBJS) $(PARACRYPT_CUDA_OBJS)
	$(CXX) $(CXX_FLAGS_) -o $(EXAMPLES)/paracrypt$(OUT_TAG) $(EXAMPLES)/main.cpp \
	$(INCL) $(SOURCE)/libgibraltar.a $(BOOST_LIB_DIR) $(LIBS) \
	$(BOOST_INC) $(BOOST_LIB) $(LDLIBS) $(LIB)/libjerasure.a $(LDFLAGS)

clean:
	rm -f $(LIB)/libjerasure.a $(SOURCE)/libgibraltar.a
	rm -f $(SOURCE)/*.o
	rm -f $(TESTS)
	rm -f $(PARACRYPT_OBJS) $(PARACRYPT_CUDA_OBJS) 
