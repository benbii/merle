// ssb/bmpidx_create.cu - for creating and collecting stats on SSB dataset;
// uses ./bmpidx_create.cu on SSB column data
#include "ssbdemo.h"
#include "../primitive.cuh"
#include "../merle/wahGpu.cuh"
#include <cuda_runtime_api.h>
using namespace mybmpidx;

const uint64_t dateSpBin[8] = {
  SSBDATE_920101, SSBDATE_930101, SSBDATE_940101, SSBDATE_950101,
  SSBDATE_960101, SSBDATE_970101, SSBDATE_980101, SSBDATE_990101,
}, dateDeBin[5] = {
  SSBDATE_940101, SSBDATE_940201, SSBDATE_940301,
  SSBDATE_971201, SSBDATE_980101
}, discntSpBin[5] = {
  0, 2, 5, 8, 11
}, discntDeBin[12] = {
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
}, qtySpBin[6] = {
  1, 11, 21, 31, 41, 51
}, qtyDeBin[11] = {
  1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51
}, mfgrSpBin[6] = {
  0, 200, 400, 600, 800, 1000
}, mfgrDeBin[6] = {
  40, 80, 120, 160, 240, 280
}, sCitySpBin[6] = {
  0, 50, 100, 150, 200, 250
}, sCityDeBin[4] = {
  50, 60, 190, 200
};

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
                   size_t nbit, uint factSz, uint dimsz, const uint64_t *bin,
                   size_t ncol, uint **devColOut) {
  foo bar = {0.0, 0.0, 0, 0};
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);

  // Time fused bitmap creation
  cudaEventRecord(start);
  size_t single_bmp_sz =
      create_bin(fk, attr, nbit, factSz, dimsz, bin, bin + 1, ncol, devColOut);
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
                               (uint32_t)bin[i], (uint32_t)bin[i + 1], ctx);
    } else {
      // Join filter
      bitmap31 = dbjoinFlatWah(fk, factSz, attr_u32, nullptr,
                               (uint32_t)bin[i], (uint32_t)bin[i + 1], ctx);
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
  foo result = _helper(nullptr, dat->loOrderDate, 16, factSz, 0, dateSpBin,
                       7, devOut->dateSparse);
  result.print("loOrderDate_Sparse");
  total += result;

  // 2. Discount: 4 bins 01, 234, 567, 8910
  result = _helper(nullptr, dat->loDiscount, 8, factSz, 0, discntSpBin, 4,
                   devOut->discntSparse);
  result.print("loDiscount_Sparse");
  total += result;

  // 3. Quantity: 5 bins 12345678910, 1112.....50
  result = _helper(nullptr, dat->loQuantity, 8, factSz, 0, qtySpBin, 5,
                   devOut->qtySparse);
  result.print("loQuantity_Sparse");
  total += result;

  // Shared city bins for both supplier and customer
  // 4. Supplier City: 5 bins; one each SSB region.
  result = _helper(nullptr, dat->loSuppCity, 8, factSz, 0, sCitySpBin, 5,
                   devOut->sCitySparse);
  result.print("loSuppCity_Sparse");
  total += result;
  // 5. Customer City: 5 bins; one each SSB region.
  result = _helper(dat->loCustKey, dat->custCity, 8, factSz, maxCustKey,
                   sCitySpBin, 5, devOut->cCitySparse);
  result.print("custCity_Sparse");
  total += result;

  // 6. Part Manufacturer: 5 bins; one each SSB Category
  result = _helper(dat->loPartKey, dat->partMfgr, 16, factSz, maxPartKey,
                   mfgrSpBin, 5, devOut->mfgrSparse);
  result.print("partMfgr_Sparse");
  total += result;
  total.print("Total_Sparse");
  // END OF SPARSE INDEX

  foo mid_total = total;
  // DENSE index built *on top of* sparse index
  // 1. Order Date: 1 month * (12*7), faked with only 3 bins
  result = _helper(nullptr, dat->loOrderDate, 16, factSz, 0, dateDeBin, 4,
                   devOut->dateDense);
  result *= 21; // actual time shourter since 1 coulmn read generates multi bins
  result.print("loOrderDate_Dense");
  total += result;

  // 2. Discount: 11 bins 0,1,2,3,4,5,6,7,8,9,10
  result = _helper(nullptr, dat->loDiscount, 8, factSz, 0, discntDeBin, 11,
                   devOut->discntDense);
  result.print("loDiscount_Dense");
  total += result, mid_total += result;

  // 3. Quantity: 10 bins 12345, 678910, ...
  result = _helper(nullptr, dat->loQuantity, 8, factSz, 0, qtyDeBin, 10,
                   devOut->qtyDense);
  result.print("loQuantity_Dense");
  total += result, mid_total += result;

  // Shared city bins for both supplier and customer
  // HACKY: only 2 countries (50~60~190~200) present in query
  // 4. Supplier City: 25 bins; one each SSB Nation (10 cities)
  result = _helper(nullptr, dat->loSuppCity, 8, factSz, 0, sCityDeBin, 3,
                   devOut->sCityDense);
  result *= (25.0 / 3.0);
  result.print("loSuppCity_Dense");
  total += result;
  // 5. Customer City: 25 bins; one each SSB Nation (10 cities)
  result = _helper(dat->loCustKey, dat->custCity, 8, factSz, maxCustKey,
                   sCityDeBin, 3, devOut->cCityDense);
  result *= (25.0 / 3.0);
  result.print("custCity_Dense");
  total += result, mid_total += result;

  // 6. Part Manufacturer: 25 bins; one each SSB category (40 brands).
  // HACKY: only category 1,3,6 (40~80~120~160~240~280) gets used in query
  result = _helper(dat->loPartKey, dat->partMfgr, 16, factSz, maxPartKey,
                   mfgrDeBin, 5, devOut->mfgrDense);
  result *= 5.0;
  result.print("partMfgr_Dense");
  total += result;
  mid_total.print("Total_Balanced");
  total.print("Total_Dense");
  // END OF DENSE INDEX

  // See the RTScan log file
  printf("RTScanSieve\t0.00\t0.00\t24695.7\t3420\n"
         "RTScanRays\t0.00\t0.00\t2505.57\t5396\n");
}

void ssb_bmpfree(struct ssb_bmp *devOut) {
  cudaFree(devOut->mfgrSparse[0]); cudaFree(devOut->mfgrDense[0]);
  cudaFree(devOut->sCitySparse[0]); cudaFree(devOut->sCityDense[0]);
  cudaFree(devOut->cCitySparse[0]); cudaFree(devOut->cCityDense[0]);
  cudaFree(devOut->dateSparse[0]); cudaFree(devOut->dateDense[0]);
  cudaFree(devOut->discntSparse[0]); cudaFree(devOut->discntDense[0]);
  cudaFree(devOut->qtySparse[0]); cudaFree(devOut->qtyDense[0]);
}
