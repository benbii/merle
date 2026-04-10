// ssb/bmpidx_create.cu - for creating and collecting stats on SSB dataset;
// uses ./bmpidx_create.cu on SSB column data
#include "ssbdemo.h"
#include "../primitive.cuh"
#include "../merle/wahGpu.cuh"
#include <cuda_runtime_api.h>
using namespace mybmpidx;

// returns fused bmp creation time and WAH bmp idx creation time
struct foo {
  float bmp_msec, wah_msec;
  size_t bmp_sz, wah_sz;
  void print(const char* a) const {
    printf("%s\t%.2f\t%.2f\t%.2f\t%.2f\n", a, bmp_msec, wah_msec,
           bmp_sz / (1024.0 * 1024.0), wah_sz / (1024.0 * 1024.0));
  }
  foo& operator+=(const foo& rhs) {
    bmp_msec += rhs.bmp_msec, wah_msec += rhs.wah_msec;
    bmp_sz += rhs.bmp_sz, wah_sz += rhs.wah_sz;
    return *this;
  }
  // For high bin counts, prefer managed memory and prefetch, and chunking the
  // fact table into say SF=10, 60M-row parts. Prefetch the next part while
  // processing the previous, thereby hiding all but 1 chunk's HToD latencies.
  // However that's such a hassle for this prototype. Create useful bins only!
  foo& operator*=(float n) {
    bmp_msec *= n, wah_msec *= n;
    bmp_sz *= n, wah_sz *= n;
    return *this;
  }
};

static foo _helper(const uint *__restrict fk, const void *__restrict attr,
                   size_t nbit, uint factSz, uint dimsz, const uint64_t *mins,
                   const uint64_t *maxes, size_t ncol, uint **devColOut) {
  foo bar = {0.0, 0.0, 0, 0};
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);

  // Time fused bitmap creation
  cudaEventRecord(start);
  size_t single_bmp_sz =
      create_bin(fk, attr, nbit, factSz, dimsz, mins, maxes, ncol, devColOut);
  cudaEventRecord(stop); cudaEventSynchronize(stop);
  cudaEventElapsedTime(&bar.bmp_msec, start, stop);
  bar.bmp_sz = ncol * single_bmp_sz;
  // cudaFree(devColOut[0]);

  // WAH compression timing
  mgpu::standard_context_t ctx(false);
  size_t total_wah_sz = 0;
  // Convert uint16/uint8 to uint32 if needed (not timed)
  mgpu::mem_t<uint32_t> converted_attr;
  const uint32_t* attr_u32 = nullptr;
  if (nbit == 32) {
    attr_u32 = (const uint32_t*)attr;
  } else if (nbit == 16) {
    converted_attr = mgpu::mem_t<uint32_t>(dimsz ? dimsz : factSz, ctx);
    mgpu::transform([=] __device__ (int idx, uint32_t *o) {
      o[idx] = ((const uint16_t*)attr)[idx];
    }, dimsz ? dimsz : factSz, ctx, converted_attr.data());
    attr_u32 = converted_attr.data();
  } else if (nbit == 8) {
    converted_attr = mgpu::mem_t<uint32_t>(dimsz ? dimsz : factSz, ctx);
    mgpu::transform([=] __device__ (int idx, uint32_t *o) {
      o[idx] = ((const uint8_t*)attr)[idx];
    }, dimsz ? dimsz : factSz, ctx, converted_attr.data());
    attr_u32 = converted_attr.data();
  }

  // Now time the WAH creation on converted uint32 data
  cudaEventRecord(start);
  for (size_t i = 0; i < ncol; ++i) {
    mgpu::mem_t<uint32_t> bitmap31;
    if (fk == nullptr) {
      // Direct attribute filter
      bitmap31 = dbjoinFlatWah(attr_u32, factSz, nullptr, nullptr,
                               (uint32_t)mins[i], (uint32_t)maxes[i], ctx);
    } else {
      // Join filter
      bitmap31 = dbjoinFlatWah(fk, factSz, attr_u32, nullptr,
                               (uint32_t)mins[i], (uint32_t)maxes[i], ctx);
    }
    mgpu::mem_t<int> wah = wahCompress((const int*)bitmap31.data(), bitmap31.size(), ctx);
    total_wah_sz += wah.size() * sizeof(int);
  }
  cudaEventRecord(stop); cudaEventSynchronize(stop);
  cudaEventElapsedTime(&bar.wah_msec, start, stop);
  bar.wah_sz = total_wah_sz;

  cudaEventDestroy(start); cudaEventDestroy(stop);
  return bar;
}

void ssb_bmpcreate(const struct ssb_schema *host, const struct ssb_schema *dat,
                   size_t factSz, struct ssb_bmp *devOut) {
  // Find actual dimension sizes by scanning for maximum keys (like in merledemo.cu)
  uint32_t maxPartKey = 0, maxCustKey = 0;
  for (size_t i = 0; i < factSz; ++i) {
    if (host->loPartKey[i] > maxPartKey)
      maxPartKey = host->loPartKey[i];
    if (host->loCustKey[i] > maxCustKey)
      maxCustKey = host->loCustKey[i];
  }
  maxPartKey++, maxCustKey++;
  printf("\nColumn\tMy(ms)\tWAH(ms)\tMy(MB)\tWAH(MB)\n");
  foo total = {};

  // SPARSE COLUMNS
  // 1. Order Date: 1 year * 7
  uint64_t bin[85];
  bin[0] = SSBDATE_920101, bin[1] = SSBDATE_930101, bin[2] = SSBDATE_940101,
  bin[3] = SSBDATE_950101, bin[4] = SSBDATE_960101, bin[5] = SSBDATE_970101,
  bin[6] = SSBDATE_980101, bin[7] = SSBDATE_990101;
  foo result = _helper(nullptr, dat->loOrderDate, 16, factSz, 0, bin, bin + 1,
                       7, devOut->dateSparse);
  result.print("loOrderDate_Sparse");
  total += result;

  // 2. Discount: 4 bins 01, 234, 567, 8910
  bin[0] = 0, bin[1] = 2, bin[2] = 5, bin[3] = 8, bin[4] = 11;
  result = _helper(nullptr, dat->loDiscount, 8, factSz, 0, bin, bin + 1, 4,
                   devOut->discntSparse);
  result.print("loDiscount_Sparse");
  total += result;

  // 3. Quantity: 5 bins 12345678910, 1112.....50
  for (size_t i = 0; i <= 5; i++) bin[i] = i * 10 + 1;
  result = _helper(nullptr, dat->loQuantity, 8, factSz, 0, bin, bin + 1, 5,
                   devOut->qtySparse);
  result.print("loQuantity_Sparse");
  total += result;

  // Shared city bins for both supplier and customer
  for (size_t i = 0; i <= 5; i++) bin[i] = i * 50;
  // 4. Supplier City: 5 bins; one each SSB region.
  result = _helper(nullptr, dat->loSuppCity, 8, factSz, 0, bin, bin + 1, 5,
                   devOut->sCitySparse);
  result.print("loSuppCity_Sparse");
  total += result;
  // 5. Customer City: 5 bins; one each SSB region.
  result = _helper(dat->loCustKey, dat->custCity, 8, factSz, maxCustKey, bin,
                   bin + 1, 5, devOut->cCitySparse);
  result.print("custCity_Sparse");
  total += result;

  // 6. Part Manufacturer: 5 bins; one each SSB Category
  for (size_t i = 0; i <= 5; i++) bin[i] = i * 200;
  result = _helper(dat->loPartKey, dat->partMfgr, 16, factSz, maxPartKey, bin,
                   bin + 1, 5, devOut->pMfgrSparse);
  result.print("partMfgr_Sparse");
  total += result;
  total.print("Total_Sparse");
  // END OF SPARSE INDEX

  total = {};
  // DENSE index built *on top of* sparse index
  // 1. Order Date: 1 month * (12*7), faked with only 3 bins
  uint64_t fakeDate[3] = {SSBDATE_940101, SSBDATE_940201, SSBDATE_971201};
  bin[0] = fakeDate[0] + 31, bin[1] = fakeDate[1] + 28, bin[2] = fakeDate[2] + 31;
  result = _helper(nullptr, dat->loOrderDate, 16, factSz, 0, fakeDate, bin, 3,
                   devOut->dateDense);
  result *= 28;
  result.print("loOrderDate_Dense");
  total += result;

  // 2. Discount: 11 bins 0,1,2,3,4,5,6,7,8,9,10
  for (size_t i = 0; i <= 11; ++i) bin[i] = i + 1;
  result = _helper(nullptr, dat->loDiscount, 8, factSz, 0, bin, bin + 1, 11,
                   devOut->discntDense);
  result.print("loDiscount_Dense");
  total += result;

  // 3. Quantity: 10 bins 12345, 678910, ...
  for (size_t i = 0; i <= 10; i++) bin[i] = i * 5 + 1;
  result = _helper(nullptr, dat->loQuantity, 8, factSz, 0, bin, bin + 1, 10,
                   devOut->qtyDense);
  result.print("loQuantity_Dense");
  total += result;

  // Shared city bins for both supplier and customer
  for (size_t i = 0; i <= 25; i++) bin[i] = i * 10;
  // 4. Supplier City: 25 bins; one each SSB Nation (10 cities)
  result = _helper(nullptr, dat->loSuppCity, 8, factSz, 0, bin, bin + 1, 25,
                   devOut->sCityDense);
  result.print("loSuppCity_Dense");
  total += result;
  // 5. Customer City: 25 bins; one each SSB Nation (10 cities)
  result = _helper(dat->loCustKey, dat->custCity, 8, factSz, maxCustKey, bin,
                   bin + 1, 25, devOut->cCityDense);
  result.print("custCity_Dense");
  total += result;

  // 6. Part Manufacturer: 25 bins; one each SSB category (40 brands).
  // HACKY: only category 2 (40~80) gets used in query
  bin[0] = 40, bin[1] = 80;
  result = _helper(dat->loPartKey, dat->partMfgr, 16, factSz, maxPartKey, bin,
                   bin + 1, 1, devOut->pMfgrDense);
  result *= 25; // actual time shourter since 1 coulmn read generates multi bins
  result.print("partMfgr_Dense");
  total += result;
  total.print("Total_Dense");
  // END OF DENSE INDEX

  // See the RTScan log file
  printf("RTScanSieve\t0.00\t0.00\t24695.7\t3420\n"
         "RTScanRays\t0.00\t0.00\t2505.57\t5396\n");
}

void ssb_bmpfree(struct ssb_bmp *devOut) {
  cudaFree(devOut->pMfgrSparse[0]); cudaFree(devOut->pMfgrDense[0]);
  cudaFree(devOut->sCitySparse[0]); cudaFree(devOut->sCityDense[0]);
  cudaFree(devOut->cCitySparse[0]); cudaFree(devOut->cCityDense[0]);
  cudaFree(devOut->dateSparse[0]); cudaFree(devOut->dateDense[0]);
  cudaFree(devOut->discntSparse[0]); cudaFree(devOut->discntDense[0]);
  cudaFree(devOut->qtySparse[0]); cudaFree(devOut->qtyDense[0]);
}
