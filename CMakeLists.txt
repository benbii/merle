cmake_minimum_required(VERSION 3.18)
project(mydemo LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90 120)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(CUB REQUIRED)
find_package(OpenMP REQUIRED)

# Add executable
add_executable(mydemo
    main.c bmpidx_create.cu
    ssb/cpuref.c ssb/controlled-demo.cu
    ssb/fixedlayout.cu ssb/bmpidx-create.cu ssb/wah-demo.cu
    ssb/inst/1.cu ssb/inst/2.cu ssb/inst/3.cu ssb/inst/4.cu
    ssb/inst/1c.cu ssb/inst/2c.cu ssb/inst/3c.cu ssb/inst/4c.cu
    synth/cpuref.c synth/synthdemo.cu synth/wah_roast.cu synth/window_zipf.c
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
  --debug # Host debug
  # --device-debug # Device debug
)
target_compile_options(mydemo PUBLIC
  $<$<COMPILE_LANGUAGE:C>:${C_FLAGS}>
  $<$<COMPILE_LANGUAGE:CXX>:${C_FLAGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
)
