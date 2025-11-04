# Path variables
CUDA_INSTALL_PATH ?= /usr/local/cuda-7.5

# Compler and Linker
CXX := nvcc
CC := nvcc
LINK := nvcc
NVCC  := nvcc

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_INSTALL_PATH)/samples/common/inc

# Common flags
COMMONFLAGS += $(INCLUDES) -DPERASURE_CUDA_GM -DPERASURE_CUDA_SM
NVCCFLAGS += -ccbin g++ -m64 $(COMMONFLAGS) -Xptxas -v --maxrregcount=32
CXXFLAGS += -m64 $(COMMONFLAGS)
CFLAGS += -O3 $(COMMONFLAGS)

LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcuda



LINKLINE = $(LINK) -o $(TARGET) $(LIB_CUDA) $(OBJS)

TARGET = GCRSMeasureWorkload

OBJS = galois.o jerasure.o GCRSKernel.o GCRSMatrix.o GCRSCoding.o MeasureGCRS.o GCRSMeasureWorkload.o
ALL = $(OBJS) $(TARGET)
all: $(ALL)

clean:
	rm -rf *.o
	rm $(TARGET)


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
MeasureGCRS.o: MeasureGCRS.h GCRSCommon.h GCRSCoding.h
GCRSMeasureWorkload.o: MeasureGCRS.h GCRSCommon.h
$(TARGET): $(OBJS)
	$(LINKLINE)
