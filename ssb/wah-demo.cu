#include "../merle/wahGpu.cuh"
#include "ssbdemo.h"
static constexpr size_t ndup = 100;
size_t poolSz = 1ul << 30;

// Convert uint8_t array to uint32_t for dbjoinFlatWah
static mgpu::mem_t<uint32_t> convertToU32(const uint8_t *src, size_t factSz,
                                          mgpu::context_t &ctx) {
  mgpu::mem_t<uint32_t> dst(factSz, ctx);
  mgpu::transform([=] __device__(int idx, uint32_t *o) { o[idx] = src[idx]; },
                  factSz, ctx, dst.data());
  return dst;
}

// Convert uint16_t array to uint32_t for dbjoinFlatWah
static mgpu::mem_t<uint32_t> convertU16ToU32(const uint16_t *src, size_t factSz,
                                             mgpu::context_t &ctx) {
  mgpu::mem_t<uint32_t> dst(factSz, ctx);
  mgpu::transform([=] __device__(int idx, uint32_t *o) { o[idx] = src[idx]; },
                  factSz, ctx, dst.data());
  return dst;
}

float s1wah(const struct ssb_schema *dat, uint factsz, uint16_t dateMin,
            uint16_t dateMax, uint8_t discntMin, uint8_t discntMax,
            uint8_t qtyMin, uint8_t qtyMax) {
  dumb_pool_t ctx(poolSz, false);
  // Convert uint8_t columns to uint32_t for dbjoinFlatWah
  auto dateU32 = convertU16ToU32(dat->loOrderDate, factsz, ctx);
  auto discntU32 = convertToU32(dat->loDiscount, factsz, ctx);
  auto qtyU32 = convertToU32(dat->loQuantity, factsz, ctx);
  // Create 31-bit bitmaps using dbjoinFlatWah (for direct column filters, use
  // dim1=nullptr, dim2=nullptr)
  auto dateBitmap = dbjoinFlatWah(dateU32.data(), factsz, nullptr, nullptr,
                                  dateMin, dateMax, ctx);
  auto discntBitmap = dbjoinFlatWah(discntU32.data(), factsz, nullptr, nullptr,
                                    discntMin, discntMax, ctx);
  auto qtyBitmap = dbjoinFlatWah(qtyU32.data(), factsz, nullptr, nullptr,
                                 qtyMin, qtyMax, ctx);
  // Compress 31-bit bitmaps to WAH format
  mgpu::mem_t<int> dateWah =
      wahCompress((const int *)dateBitmap.data(), dateBitmap.size(), ctx);
  mgpu::mem_t<int> discntWah =
      wahCompress((const int *)discntBitmap.data(), discntBitmap.size(), ctx);
  mgpu::mem_t<int> qtyWah =
      wahCompress((const int *)qtyBitmap.data(), qtyBitmap.size(), ctx);
  // Scan run lengths for each WAH bitmap (excluded from timing as this is
  // "prefix summing")
  mgpu::mem_t<int> dateScan =
      wahCntExcScan(dateWah.data(), dateWah.size(), ctx);
  mgpu::mem_t<int> discntScan =
      wahCntExcScan(discntWah.data(), discntWah.size(), ctx);
  mgpu::mem_t<int> qtyScan = wahCntExcScan(qtyWah.data(), qtyWah.size(), ctx);
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start);

  for (size_t i = 0; i < ndup; ++i) {
    // Perform AND operations between the WAH bitmaps
    mgpu::mem_t<int> intermediate =
        wahAndNo1(discntWah.data(), discntScan.data(), discntWah.size(),
                  dateWah.data(), dateScan.data(), dateWah.size(), ctx);
    mgpu::mem_t<int> result =
        wahAndNo1(intermediate.data(), dateScan.data(), intermediate.size(),
                  qtyWah.data(), qtyScan.data(), qtyWah.size(), ctx);
    // Decompress the final result
    mgpu::mem_t<int> resultScan =
        wahCntExcScan(result.data(), result.size(), ctx);
    mgpu::mem_t<int> decompressed =
        wahDecomp(result.data(), resultScan.data(), result.size(), ctx);
  }

  cudaEventRecord(stop); cudaEventSynchronize(stop);
  float totalTime = 0.0f; cudaEventElapsedTime(&totalTime, start, stop);
  cudaEventDestroy(start); cudaEventDestroy(stop);
  return totalTime / ndup;
}

void s2wah(const struct ssb_schema *dat, uint factsz, uint32_t pMfgrMin,
           uint32_t pMfgrMax, uint8_t sCityMin, uint8_t sCityMax,
           const char *query_name, size_t partDimSize) {
  dumb_pool_t ctx(poolSz, false);
  printf("%s\t", query_name);
  // prepare
  auto sCityU32 = convertToU32(dat->loSuppCity, factsz, ctx);
  auto partMfgrU32 = convertU16ToU32(dat->partMfgr, partDimSize, ctx);
  auto pMfgrBitmap = dbjoinFlatWah(dat->loPartKey, factsz, partMfgrU32.data(),
                                   nullptr, pMfgrMin, pMfgrMax, ctx);
  auto sCityBitmap = dbjoinFlatWah(sCityU32.data(), factsz, nullptr, nullptr,
                                   sCityMin, sCityMax, ctx);
  mgpu::mem_t<int> pMfgrWah =
      wahCompress((const int *)pMfgrBitmap.data(), pMfgrBitmap.size(), ctx);
  mgpu::mem_t<int> sCityWah =
      wahCompress((const int *)sCityBitmap.data(), sCityBitmap.size(), ctx);
  mgpu::mem_t<int> pMfgrScan =
      wahCntExcScan(pMfgrWah.data(), pMfgrWah.size(), ctx);
  mgpu::mem_t<int> sCityScan =
      wahCntExcScan(sCityWah.data(), sCityWah.size(), ctx);
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (size_t i = 0; i < ndup; ++i) {
    mgpu::mem_t<int> result =
        wahAndNo1(sCityWah.data(), sCityScan.data(), sCityWah.size(),
                  pMfgrWah.data(), pMfgrScan.data(), pMfgrWah.size(), ctx);
    mgpu::mem_t<int> resultScan =
        wahCntExcScan(result.data(), result.size(), ctx);
    mgpu::mem_t<int> decompressed =
        wahDecomp(result.data(), resultScan.data(), result.size(), ctx);
  }

  cudaEventRecord(stop); cudaEventSynchronize(stop);
  float totalTime; cudaEventElapsedTime(&totalTime, start, stop);
  cudaEventDestroy(start); cudaEventDestroy(stop);
  // RTScan supports conjunctive selection (no joins) only. How silly 😅.
  printf("%.4f\t99999.9\n", totalTime / ndup);
}

void s3wah(const struct ssb_schema *dat, uint factsz, uint8_t cCityMin,
           uint8_t cCityMax, uint8_t sCityMin, uint8_t sCityMax,
           uint16_t dateMin, uint16_t dateMax, const char *query_name,
           size_t custDimSize) {
  dumb_pool_t ctx(poolSz, false);
  printf("%s\t", query_name);
  // prepare
  auto dateU32 = convertU16ToU32(dat->loOrderDate, factsz, ctx);
  auto sCityU32 = convertToU32(dat->loSuppCity, factsz, ctx);
  auto custCityU32 = convertToU32(dat->custCity, custDimSize, ctx);
  auto cCityBitmap = dbjoinFlatWah(dat->loCustKey, factsz, custCityU32.data(),
                                   nullptr, cCityMin, cCityMax, ctx);
  auto sCityBitmap = dbjoinFlatWah(sCityU32.data(), factsz, nullptr, nullptr,
                                   sCityMin, sCityMax, ctx);
  auto dateBitmap = dbjoinFlatWah(dateU32.data(), factsz, nullptr, nullptr,
                                  dateMin, dateMax, ctx);
  mgpu::mem_t<int> cCityWah =
      wahCompress((const int *)cCityBitmap.data(), cCityBitmap.size(), ctx);
  mgpu::mem_t<int> sCityWah =
      wahCompress((const int *)sCityBitmap.data(), sCityBitmap.size(), ctx);
  mgpu::mem_t<int> dateWah =
      wahCompress((const int *)dateBitmap.data(), dateBitmap.size(), ctx);
  mgpu::mem_t<int> cCityScan =
      wahCntExcScan(cCityWah.data(), cCityWah.size(), ctx);
  mgpu::mem_t<int> sCityScan =
      wahCntExcScan(sCityWah.data(), sCityWah.size(), ctx);
  mgpu::mem_t<int> dateScan =
      wahCntExcScan(dateWah.data(), dateWah.size(), ctx);
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (size_t i = 0; i < ndup; ++i) {
    // Chain 3 AND operations: cCity AND sCity AND date
    mgpu::mem_t<int> intermediate1 =
        wahAndNo1(cCityWah.data(), cCityScan.data(), cCityWah.size(),
                  sCityWah.data(), sCityScan.data(), sCityWah.size(), ctx);
    // Result has same structure as sCityWah, so reuse sCityScan
    mgpu::mem_t<int> result =
        wahAndNo1(intermediate1.data(), sCityScan.data(), intermediate1.size(),
                  dateWah.data(), dateScan.data(), dateWah.size(), ctx);
    mgpu::mem_t<int> resultScan =
        wahCntExcScan(result.data(), result.size(), ctx);
    mgpu::mem_t<int> decompressed =
        wahDecomp(result.data(), resultScan.data(), result.size(), ctx);
  }

  cudaEventRecord(stop); cudaEventSynchronize(stop);
  float totalTime; cudaEventElapsedTime(&totalTime, start, stop);
  cudaEventDestroy(start); cudaEventDestroy(stop);
  printf("%.4f\t99999.9\n", totalTime / ndup);
}

void s4wah(const struct ssb_schema *dat, uint factsz, uint8_t cCityMin,
           uint8_t cCityMax, uint8_t sCityMin, uint8_t sCityMax,
           uint16_t pMfgrMin, uint16_t pMfgrMax, uint16_t dateMin,
           uint16_t dateMax, const char *query_name, size_t custDimSize,
           size_t partDimSize) {
  dumb_pool_t ctx(poolSz, false);
  printf("%s\t", query_name);
  // prepare
  auto dateU32 = convertU16ToU32(dat->loOrderDate, factsz, ctx);
  auto sCityU32 = convertToU32(dat->loSuppCity, factsz, ctx);
  auto custCityU32 = convertToU32(dat->custCity, custDimSize, ctx);
  auto partMfgrU32 = convertU16ToU32(dat->partMfgr, partDimSize, ctx);
  auto cCityBitmap = dbjoinFlatWah(dat->loCustKey, factsz, custCityU32.data(),
                                   nullptr, cCityMin, cCityMax, ctx);
  auto sCityBitmap = dbjoinFlatWah(sCityU32.data(), factsz, nullptr, nullptr,
                                   sCityMin, sCityMax, ctx);
  auto pMfgrBitmap = dbjoinFlatWah(dat->loPartKey, factsz, partMfgrU32.data(),
                                   nullptr, pMfgrMin, pMfgrMax, ctx);
  auto dateBitmap = dbjoinFlatWah(dateU32.data(), factsz, nullptr, nullptr,
                                  dateMin, dateMax, ctx);
  mgpu::mem_t<int> cCityWah =
      wahCompress((const int *)cCityBitmap.data(), cCityBitmap.size(), ctx);
  mgpu::mem_t<int> sCityWah =
      wahCompress((const int *)sCityBitmap.data(), sCityBitmap.size(), ctx);
  mgpu::mem_t<int> pMfgrWah =
      wahCompress((const int *)pMfgrBitmap.data(), pMfgrBitmap.size(), ctx);
  mgpu::mem_t<int> dateWah =
      wahCompress((const int *)dateBitmap.data(), dateBitmap.size(), ctx);
  mgpu::mem_t<int> cCityScan =
      wahCntExcScan(cCityWah.data(), cCityWah.size(), ctx);
  mgpu::mem_t<int> sCityScan =
      wahCntExcScan(sCityWah.data(), sCityWah.size(), ctx);
  mgpu::mem_t<int> pMfgrScan =
      wahCntExcScan(pMfgrWah.data(), pMfgrWah.size(), ctx);
  mgpu::mem_t<int> dateScan =
      wahCntExcScan(dateWah.data(), dateWah.size(), ctx);
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (size_t i = 0; i < ndup; ++i) {
    // Chain 4 AND operations: cCity AND sCity AND pMfgr AND date
    mgpu::mem_t<int> intermediate1 =
        wahAndNo1(cCityWah.data(), cCityScan.data(), cCityWah.size(),
                  sCityWah.data(), sCityScan.data(), sCityWah.size(), ctx);
    // Result has same structure as sCityWah, so reuse sCityScan
    mgpu::mem_t<int> intermediate2 =
        wahAndNo1(intermediate1.data(), sCityScan.data(), intermediate1.size(),
                  pMfgrWah.data(), pMfgrScan.data(), pMfgrWah.size(), ctx);
    // Result has same structure as pMfgrWah, so reuse pMfgrScan
    mgpu::mem_t<int> result =
        wahAndNo1(intermediate2.data(), pMfgrScan.data(), intermediate2.size(),
                  dateWah.data(), dateScan.data(), dateWah.size(), ctx);
    mgpu::mem_t<int> resultScan =
        wahCntExcScan(result.data(), result.size(), ctx);
    mgpu::mem_t<int> decompressed =
        wahDecomp(result.data(), resultScan.data(), result.size(), ctx);
  }

  cudaEventRecord(stop); cudaEventSynchronize(stop);
  float totalTime; cudaEventElapsedTime(&totalTime, start, stop);
  cudaEventDestroy(start); cudaEventDestroy(stop);
  printf("%.4f\t99999.9\n", totalTime / ndup);
}

void ssb_wah(const struct ssb_schema *hostdat, const ssb_schema *dat,
                 size_t factSz) {
  // Find dimension sizes by scanning for maximum keys
  uint32_t maxPartKey = 0, maxCustKey = 0;
  for (size_t i = 0; i < factSz; ++i) {
    if (hostdat->loPartKey[i] > maxPartKey)
      maxPartKey = hostdat->loPartKey[i];
    if (hostdat->loCustKey[i] > maxCustKey)
      maxCustKey = hostdat->loCustKey[i];
  }
  ++maxPartKey; ++maxCustKey;

  try {
    float a;
    // SSB Q1 -- Q11 should be the most memory intensive one
    a = s1wah(dat, factSz, SSBDATE_930101, SSBDATE_940101, 1, 4, 0, 25);
    // RTScan running time extracted from the log, selected from 3 queries with
    printf("\nCase\tWAH\tRTScan\nSSB11\t%.4f\t0.9978\n", a);
    a = s1wah(dat, factSz, SSBDATE_940101, SSBDATE_940201, 4, 7, 26, 36);
    // similar selectivities as SSB Q1*. Why no SSB? Because it supports
    printf("SSB12\t%.4f\t1.1918\n", a);
    a = s1wah(dat, factSz, SSBDATE_940204, SSBDATE_940211, 5, 8, 26, 36);
    // specific unrealistic integer columns only: distributions between 0~1e6!
    printf("SSB13\t%.4f\t1.7910\n", a); // Pathetic 🙄.
  } catch (std::runtime_error &e) {
    poolSz += 1ul << 30;
    fprintf(stderr, "trying %zu GiB VRAM\n", poolSz >> 30);
    return ssb_wah(hostdat, dat, factSz);
  } catch (mgpu::cuda_exception_t &e) {
    fprintf(stderr, "Skipping WAH cause it sucks: %s\n", e.what());
    return;
  }

  // SSB Q2
  s2wah(dat, factSz, 40, 80, 150, 200, "SSB21", maxPartKey);
  s2wah(dat, factSz, 260, 268, 200, 250, "SSB22", maxPartKey);
  s2wah(dat, factSz, 260, 261, 50, 100, "SSB23", maxPartKey);

  // SSB Q3
  s3wah(dat, factSz, 200, 250, 200, 250, SSBDATE_920101, SSBDATE_980101,
        "SSB31", maxCustKey);
  s3wah(dat, factSz, 190, 200, 190, 200, SSBDATE_920101, SSBDATE_980101,
        "SSB32", maxCustKey);
  s3wah(dat, factSz, 51, 55, 51, 55, SSBDATE_920101, SSBDATE_980101, "SSB33",
        maxCustKey);
  s3wah(dat, factSz, 51, 55, 51, 55, SSBDATE_971201, SSBDATE_980101, "SSB34",
        maxCustKey);

  // SSB Q4
  s4wah(dat, factSz, 150, 200, 150, 200, 0, 400, SSBDATE_920101, SSBDATE_990101,
        "SSB41", maxCustKey, maxPartKey);
  s4wah(dat, factSz, 150, 200, 150, 200, 0, 400, SSBDATE_970101, SSBDATE_990101,
        "SSB42", maxCustKey, maxPartKey);
  s4wah(dat, factSz, 150, 200, 190, 200, 120, 160, SSBDATE_970101,
        SSBDATE_990101, "SSB43", maxCustKey, maxPartKey);
}
