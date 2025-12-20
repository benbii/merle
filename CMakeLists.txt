cmake_minimum_required(VERSION 3.18)
project(mydemo LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Option for datacenter GPUs with large shared memory (e.g., A100, H100)
option(LARGE_SMEM "Build for datacenter GPUs with large shared memory (sm_90)" OFF)

if(LARGE_SMEM)
  set(CMAKE_CUDA_ARCHITECTURES 100 90 80)
  add_compile_definitions(LARGE_SMEM)
else()
  set(CMAKE_CUDA_ARCHITECTURES 86 89 120)
endif()

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(CUB REQUIRED)
find_package(OpenMP REQUIRED)

# Add executable
add_executable(mydemo
    main.c bmpidx_create.cu
    ssb/cpuref.c ssb/joindemo.cu ssb/bmpidxdemo.cu ssb/bmpidxdemo_abla.cu
    ssb/index_creation_demo.cu ssb/wah_roast.cu
    synth/cpuref.c synth/synthdemo.cu synth/window_zipf.c synth/wah_roast.cu
    merle/wahGpuCmprs.cu merle/wahGpuDirect.cu
)
# Link libraries
target_link_libraries(mydemo PRIVATE CUDA::cudart CUB::CUB OpenMP::OpenMP_C)

set(C_FLAGS -Wall -Wextra -march=x86-64-v3)
set(CUDA_FLAGS
  -Wall -Wextra --extended-lambda
  --expt-relaxed-constexpr --use_fast_math
  $<$<CXX_COMPILER_ID:GNU>:-O3> # Host optimize-level
  --ptxas-options -v
  # --debug # Host debug
  # --device-debug # Device debug
)
target_compile_options(mydemo PUBLIC
  $<$<COMPILE_LANGUAGE:C>:${C_FLAGS}>
  $<$<COMPILE_LANGUAGE:CXX>:${C_FLAGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
)

