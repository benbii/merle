CUDA_PATH       ?= /opt/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin

NVCC = nvcc

# SM_TARGETS   = -gencode=arch=compute_52,code=\"sm_52,compute_52\" 
# SM_DEF     = -DSM520

SM_TARGETS   = -gencode=arch=compute_86,code=\"sm_86,compute_86\" 
SM_DEF     = -DSM860

# GENCODE_SM50    := -gencode arch=compute_52,code=sm_52
#GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM86    := -gencode arch=compute_86,code=sm_86
GENCODE_FLAGS   := $(GENCODE_SM86)

NVCCFLAGS += $(SM_DEF) -Xptxas="-dlcm=cg -v" -lineinfo -Xcudafe -\# 

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
