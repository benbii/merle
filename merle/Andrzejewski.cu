#include "wahGpu.cuh"
#include "moderngpu/kernel_scan.cuh"

mgpu::mem_t<int> andrzejewski(const int *C, size_t m, mgpu::context_t &ctx) {
  using namespace mgpu;
  mem_t<int> S(m, ctx), SS(m + 1, ctx);
  transform([] MGPU_DEVICE (int idx, int* sData, const int* c) {
    int w = c[idx];
    sData[idx] = w & 0x80000000 ? w & 0x3fffffff : 1;
  }, m, ctx, S.data(), C);
  scan(S.data(), m, SS.data(), plus_t<int>(), SS.data() + m, ctx);
  int n; dtoh(&n, SS.data() + m, 1);
  // auto bruh = from_mem(SS);

  #define F S // S is no longer used
  F = fill(0, n, ctx);
  transform([] MGPU_DEVICE (int idx, int* fData, int* ssData) {
    if (idx == 0) return;
    fData[ssData[idx] - 1] = 1;
  }, m, ctx, F.data(), SS.data());
  #define SF SS
  SF = mem_t<int>(n, ctx);
  scan(F.data(), n, SF.data(), plus_t<int>(), discard_iterator_t<int>(), ctx);
  #undef F // F is no longer used

  #define E S
  E = mem_t<int>(n, ctx);
  transform([] MGPU_DEVICE (int idx, int* eData, const int* c, int* sfData) {
    int D = c[sfData[idx]];
    if (D & 0x80000000)
      eData[idx] = D & 0x40000000 ? 0x7fffffff : 0;
    else
      eData[idx] = D;
  }, n, ctx, E.data(), C, SF.data());
  return E;
  #undef E
  #undef SF
}

mgpu::mem_t<int> ours(const int *C, size_t m, mgpu::context_t &ctx) {
  auto scan = wahCntExcScan(C, m, ctx);
  return wahDecomp(C, scan.data(), m, ctx);
}
