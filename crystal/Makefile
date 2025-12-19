CUDA_PATH       ?= /opt/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
NVCC = $(CUDA_BIN_PATH)/nvcc
# SM_TARGETS   = -gencode=arch=compute_86,code=\"sm_86,compute_86\" 
SM_TARGETS   = "--generate-code=arch=compute_86,code=[compute_86,sm_86]" \
	"--generate-code=arch=compute_90,code=[compute_90,sm_90]" \
	"--generate-code=arch=compute_120,code=[compute_120,sm_120]"
NVCCFLAGS += -Xptxas="-dlcm=cg -v" -lineinfo -Xcudafe -\# 

SRC = src
BIN = bin
OBJ = obj
INCLUDES = -I. -I$(INC)

$(OBJ)/%.o: $(SRC)/%.cu
	$(NVCC) -lcurand $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INCLUDES) $(LIBS) -O3 -dc $< -o $@

$(BIN)/%: $(OBJ)/%.o
	$(NVCC) $(SM_TARGETS) -lcurand $^ -o $@

all: setup $(BIN)/ssb/q11 $(BIN)/ssb/q12 $(BIN)/ssb/q13 $(BIN)/ssb/q21 $(BIN)/ssb/q22 $(BIN)/ssb/q23 $(BIN)/ssb/q31 $(BIN)/ssb/q32 $(BIN)/ssb/q33 $(BIN)/ssb/q34 $(BIN)/ssb/q41 $(BIN)/ssb/q42 $(BIN)/ssb/q43

setup:
	mkdir -p bin/ssb obj/ssb
	mkdir -p bin/ops obj/ops

clean:
	rm -rf bin/* obj/*
