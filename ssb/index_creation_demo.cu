#include "ssbdemo.h"
#include "../primitive.cuh"
#include "../merle/wahGpu.cuh"
using namespace mybmpidx;

// returns fused bmp creation time and WAH bmp idx creation time
struct foo {
  float bmp_msec, wah_msec;
  size_t bmp_sz, wah_sz;
};

static foo _helper(const uint *__restrict fk, const void *__restrict attr,
                   size_t nbit, uint factSz, uint dimsz, const uint64_t *mins,
                   const uint64_t *maxes, size_t ncol) {
  foo bar = {0.0, 0.0, 0, 0};
  uint *out[ncol];
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);

  // Time fused bitmap creation
  cudaEventRecord(start);
  size_t single_bmp_sz =
      create_bin(fk, attr, nbit, factSz, dimsz, mins, maxes, ncol, out);
  cudaEventRecord(stop); cudaEventSynchronize(stop);
  cudaEventElapsedTime(&bar.bmp_msec, start, stop);
  bar.bmp_sz = ncol * single_bmp_sz;
  cudaFree(out[0]);

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
    mgpu::transform([=] __device__ (int idx, uint32_t *o) -> uint32_t {
      o[idx] = ((const uint16_t*)attr)[idx];
    }, dimsz ? dimsz : factSz, ctx, converted_attr.data());
    attr_u32 = converted_attr.data();
  } else if (nbit == 8) {
    converted_attr = mgpu::mem_t<uint32_t>(dimsz ? dimsz : factSz, ctx);
    mgpu::transform([=] __device__ (int idx, uint32_t *o) -> uint32_t {
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

void ssb_democreate(const struct ssb_schema *hostdat,
                    const struct ssb_schema *dat, size_t factSz) {
  printf("\nColumn\tMy(ms)\tWAH(ms)\tMy(MB)\tWAH(MB)\n");

  // Find actual dimension sizes by scanning for maximum keys (like in merledemo.cu)
  uint32_t maxPartKey = 0, maxCustKey = 0;
  for (size_t i = 0; i < factSz; ++i) {
    if (hostdat->loPartKey[i] > maxPartKey)
      maxPartKey = hostdat->loPartKey[i];
    if (hostdat->loCustKey[i] > maxCustKey)
      maxCustKey = hostdat->loCustKey[i];
  }
  size_t custDimSize = maxCustKey + 1;
  size_t partDimSize = maxPartKey + 1;

  size_t mybyte = 0, total_wah_bytes = 0;
  float myms = 0.0, wahms = 0.0;

  // 1. Order Date: 4 months per bin, for a total of 18 bins
  {
    uint64_t mins[] = {SSBDATE_920101, SSBDATE_920501, SSBDATE_920901, SSBDATE_930101,
                       SSBDATE_930501, SSBDATE_930901, SSBDATE_940101, SSBDATE_940501,
                       SSBDATE_940901, SSBDATE_950101, SSBDATE_950501, SSBDATE_950901,
                       SSBDATE_960101, SSBDATE_960501, SSBDATE_960901, SSBDATE_970101,
                       SSBDATE_970501, SSBDATE_970901};
    uint64_t maxes[] = {SSBDATE_920501, SSBDATE_920901, SSBDATE_930101, SSBDATE_930501,
                        SSBDATE_930901, SSBDATE_940101, SSBDATE_940501, SSBDATE_940901,
                        SSBDATE_950101, SSBDATE_950501, SSBDATE_950901, SSBDATE_960101,
                        SSBDATE_960501, SSBDATE_960901, SSBDATE_970101, SSBDATE_970501,
                        SSBDATE_970901, SSBDATE_980101};
    foo result = _helper(nullptr, dat->loOrderDate, 16, factSz, 0, mins, maxes, 18);
    printf("loOrderDate\t%.2f\t%.2f\t%.2f\t%.2f\n", result.bmp_msec, result.wah_msec,
           result.bmp_sz / (1024.0 * 1024.0), result.wah_sz / (1024.0 * 1024.0));
    mybyte += result.bmp_sz;
    total_wah_bytes += result.wah_sz;
    myms += result.bmp_msec;
    wahms += result.wah_msec;
  }

  // 2. Discount: 3 bins
  {
    uint64_t mins[3] = {1, 5, 9};
    uint64_t maxes[3] = {5, 9, 13};
    foo result = _helper(nullptr, dat->loDiscount, 8, factSz, 0, mins, maxes, 3);
    printf("loDiscount\t%.2f\t%.2f\t%.2f\t%.2f\n", result.bmp_msec, result.wah_msec,
           result.bmp_sz / (1024.0 * 1024.0), result.wah_sz / (1024.0 * 1024.0));
    mybyte += result.bmp_sz;
    total_wah_bytes += result.wah_sz;
    myms += result.bmp_msec;
    wahms += result.wah_msec;
  }

  // 3. Quantity: 10 bins
  {
    uint64_t mins[10], maxes[10];
    for (size_t i = 0; i < 10; i++) {
      mins[i] = i * 5 + 1;
      maxes[i] = (i + 1) * 5 + 1;
    }
    foo result = _helper(nullptr, dat->loQuantity, 8, factSz, 0, mins, maxes, 10);
    printf("loQuantity\t%.2f\t%.2f\t%.2f\t%.2f\n", result.bmp_msec, result.wah_msec,
           result.bmp_sz / (1024.0 * 1024.0), result.wah_sz / (1024.0 * 1024.0));
    mybyte += result.bmp_sz;
    total_wah_bytes += result.wah_sz;
    myms += result.bmp_msec;
    wahms += result.wah_msec;
  }

  // Shared city bins for both supplier and customer
  uint64_t city_mins[5], city_maxes[5];
  for (size_t i = 0; i < 5; i++) {
    city_mins[i] = i * 50;
    city_maxes[i] = (i + 1) * 50;
  }

  // 4. Supplier City: 5 bins
  {
    foo result = _helper(nullptr, dat->loSuppCity, 8, factSz, 0, city_mins,
                         city_maxes, 5);
    printf("loSuppCity\t%.2f\t%.2f\t%.2f\t%.2f\n", result.bmp_msec,
           result.wah_msec, result.bmp_sz / 1048576.0, result.wah_sz / 1048576.0);
    mybyte += result.bmp_sz;
    total_wah_bytes += result.wah_sz;
    myms += result.bmp_msec;
    wahms += result.wah_msec;
  }

  // 5. Customer City: 5 bins
  {
    foo result = _helper(dat->loCustKey, dat->custCity, 8, factSz, custDimSize,
                         city_mins, city_maxes, 5);
    printf("custCity\t%.2f\t%.2f\t%.2f\t%.2f\n", result.bmp_msec, result.wah_msec,
           result.bmp_sz / 1048576.0, result.wah_sz / 1048576.0);
    mybyte += result.bmp_sz;
    total_wah_bytes += result.wah_sz;
    myms += result.bmp_msec;
    wahms += result.wah_msec;
  }

  // 6. Part Manufacturer: 20 bins
  {
    uint64_t mins[20], maxes[20];
    for (size_t i = 0; i < 20; i++) {
      mins[i] = i * 50;
      maxes[i] = (i + 1) * 50;
    }
    foo result = _helper(dat->loPartKey, dat->partMfgr, 16, factSz, partDimSize,
                         mins, maxes, 20);
    printf("partMfgr\t%.2f\t%.2f\t%.2f\t%.2f\n", result.bmp_msec, result.wah_msec,
           result.bmp_sz / 1048576.0, result.wah_sz / 1048576.0);
    mybyte += result.bmp_sz;
    total_wah_bytes += result.wah_sz;
    myms += result.bmp_msec;
    wahms += result.wah_msec;
  }

  // See the RTScan log file
  printf("Total\t%.2f\t%.2f\t%.2f\t%.2f\n"
         "RTScanSieve\t0.00\t0.00\t24695.7\t3420\n"
         "RTScanRays\t0.00\t0.00\t2505.57\t5396\n",
         myms, wahms, mybyte / 1048576.0, total_wah_bytes / 1048576.0);
}
