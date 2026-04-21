#include "../merle/wahGpu.cuh"
#include "synthdemo.h"
static constexpr size_t ndup = 100;

float synth_wah(const struct synth_schema *dat, uint32_t factsz, uint32_t fa1lo,
                uint32_t fa1hi, uint32_t fa2lo, uint32_t fa2hi, uint32_t da1lo,
                uint32_t da1hi) {
  dumb_pool_t ctx(2ull << 30, false);
  auto b1f = dbjoinFlatWah((uint32_t *)dat->factattr1, factsz, nullptr, nullptr,
                          fa1lo, fa1hi, ctx);
  auto b2f = dbjoinFlatWah((uint32_t *)dat->factattr2, factsz, nullptr, nullptr,
                          fa2lo, fa2hi, ctx);
  auto b3f = dbjoinFlatWah(dat->fkey, factsz, nullptr, nullptr, da1lo, da1hi, ctx);
  auto b1 = wahCompress((int*)b1f.data(), b1f.size(), ctx);
  auto b2 = wahCompress((int*)b2f.data(), b2f.size(), ctx);
  auto b3 = wahCompress((int*)b3f.data(), b3f.size(), ctx);
  auto b1s = wahCntExcScan(b1.data(), b1.size(), ctx);
  auto b2s = wahCntExcScan(b2.data(), b2.size(), ctx);
  auto b3s = wahCntExcScan(b3.data(), b3.size(), ctx);
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (size_t i = 0; i < ndup; ++i) {
    auto t1 = wahOr(b1.data(), b1s.data(), b1.size(),
                    b2.data(), b2s.data(), b2.size(), ctx);
    auto t1s = wahCntExcScan(t1.data(), t1.size(), ctx);
    (void)wahAndNo1(t1.data(), t1s.data(), t1.size(),
                    b3.data(), b3s.data(), b3.size(), ctx);
  }

  cudaEventRecord(stop); cudaEventSynchronize(stop);
  float totalTime; cudaEventElapsedTime(&totalTime, start, stop);
  cudaEventDestroy(start); cudaEventDestroy(stop);
  return totalTime / ndup;
}
