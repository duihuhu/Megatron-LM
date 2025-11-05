# Path variables
CUDA_INSTALL_PATH ?= /usr/local/cuda-12.3

# Compler and Linker
CXX := g++
CC := gcc
LINK := nvcc
NVCC  := nvcc

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_INSTALL_PATH)/samples/common/inc

# Common flags
COMMONFLAGS += $(INCLUDES) -DPERASURE_CUDA_GM -DPERASURE_CUDA_SM
CFLAGS += -O3 $(COMMONFLAGS) -fPIC
CXXFLAGS += -m64 $(COMMONFLAGS) -fPIC
NVCCFLAGS += -ccbin g++ -m64 $(COMMONFLAGS) -Xptxas -v --maxrregcount=32 -Xcompiler -fPIC

LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcuda



LINKLINE = $(LINK) -o $(TARGET) $(LIB_CUDA) $(OBJS)

TARGET = PCIEMeasure
TEST_TARGET = test_gpu_tensor

OBJS = galois.o jerasure.o GCRSKernel.o GCRSMatrix.o GCRSCoding.o PErasureUtilities.o  PErasureWorker.o PCIEMeasure.o
TEST_OBJS = galois.o jerasure.o GCRSKernel.o GCRSMatrix.o GCRSCoding.o PErasureUtilities.o  PErasureWorker.o test_gpu_tensor.o

LIBTARGET = libgcrspcie.so
ALL = $(OBJS) $(TARGET) $(LIBTARGET)
ALL_WITH_TEST = $(ALL) $(TEST_TARGET)

all: $(ALL)

test: $(ALL_WITH_TEST)

$(LIBTARGET): $(OBJS)
	$(LINK) -shared -o $(LIBTARGET) $(OBJS) $(LIB_CUDA)

clean:
	rm -rf *.o
	rm -f $(TARGET) test_gpu_tensor


.SUFFIXES: .c .cpp .cu .o

%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

jerasure.o: jerasure.h galois.h
galois.o: galois.h
GCRSKernel.o: GCRSCoding.h GCRSCommon.h
GCRSMatrix.o: GCRSMatrix.h jerasure.h galois.h
GCRSCoding.o: GCRSCoding.h
PErasureUtilities.o: PErasureUtilities.h
PErasureWorker.o: PErasureWorker.h PErasureUtilities.h
PCIEMeasure.o: PErasureWorker.h PErasureUtilities.h GCRSCoding.h GCRSMatrix.h
test_gpu_tensor.o: PErasureWorker.h PErasureUtilities.h GCRSCoding.h GCRSMatrix.h GCRSCommon.h

$(TARGET): $(OBJS)
	$(LINKLINE)

$(TEST_TARGET): $(TEST_OBJS)
	$(LINK) -o $(TEST_TARGET) $(LIB_CUDA) $(TEST_OBJS)
