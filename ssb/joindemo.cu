#include "ssbdemo.h"
#include "../primitive.cuh"
using namespace mybmpidx;
using cuda::ceil_div;

void s1dev(const struct ssb_schema *dat, size_t factsz, uint32_t dateMin,
           uint32_t dateMax, uint8_t discntMin, uint8_t discntMax,
           uint8_t qtyMin, uint8_t qtyMax, uint32_t *red_out) {
  // must bring pointers in `*dat` to __constant__ memory
  auto op = [=, dat = *dat] __device__ (uint i) {
    uint2 ret; ret.y = ELIMINATED;
    // TODO: use __ldcs?
    if (dat.loOrderDate[i] < dateMin || dat.loOrderDate[i] >= dateMax)
      return ret;
    const uint8_t discnt = dat.loDiscount[i];
    if (discnt < discntMin || discnt >= discntMax)
      return ret;
    if (dat.loQuantity[i] < qtyMin || dat.loQuantity[i] >= qtyMax)
      return ret;
    ret.x = dat.loExtendedPrice[i] * discnt;
    ret.y = 0;
    return ret;
  };

  cudaMemset(red_out, 0, sizeof(uint32_t));
  // Reduction -> group by with only one group
  grpby_small<256, 8, 1>
      <<<ceil_div(factsz, 256 * 8), 256>>>(factsz, red_out, op);
}

// We need nr_groups for CUDA kernels
void s2dev(const struct ssb_schema *dat, size_t factsz, uint32_t pMfgrMin,
           uint32_t pMfgrMax, uint8_t sCityMin, uint8_t sCityMax,
           uint32_t *grpby_out, size_t nr_grp) {
  auto op = [=, dat = *dat] __device__ (uint i) {
    uint2 ret; ret.y = ELIMINATED;
    // Filter by supplier city range
    if (dat.loSuppCity[i] < sCityMin || dat.loSuppCity[i] >= sCityMax)
      return ret;
    // Join with part dimension to get manufacturer
    uint32_t partKey = dat.loPartKey[i];
    uint16_t mfgr = dat.partMfgr[partKey];  // Keys are zero-padded
    // Filter by manufacturer range
    if (mfgr < pMfgrMin || mfgr >= pMfgrMax)
      return ret;
    // Extract year from date (YYYYMMDD format)
    uint32_t year = dat.loOrderDate[i] / 10000;
    // GROUP BY group position calculation:
    // (mfgr - mfgrMin) + (date / 10000 - 1992) * (mfgrMax - mfgrMin)
    ret.y = (mfgr - pMfgrMin) + (year - 1992) * (pMfgrMax - pMfgrMin);
    ret.x = dat.loRevenue[i];
    return ret;
  };

  cudaMemset(grpby_out, 0, nr_grp * sizeof(uint32_t));
  grpby<256, 4><<<ceil_div(factsz, 256 * 4), 256, nr_grp * sizeof(uint)>>>(
      factsz, grpby_out, nr_grp, op);
}

void s3dev(const struct ssb_schema *dat, size_t factsz, uint8_t cCityMin,
           uint8_t cCityMax, uint8_t sCityMin, uint8_t sCityMax,
           uint32_t dateMin, uint32_t dateMax, uint32_t *grpby_out, size_t nr_grp) {
  auto op = [=, dat = *dat] __device__ (uint i) {
    uint2 ret; ret.y = ELIMINATED;
    // Filter by supplier city range
    uint8_t sCity = dat.loSuppCity[i];
    if (sCity < sCityMin || sCity >= sCityMax)
      return ret;
    // Filter by date range
    uint32_t date = dat.loOrderDate[i];
    if (date < dateMin || date >= dateMax)
      return ret;
    // Join with customer dimension to get customer city
    uint32_t custKey = dat.loCustKey[i];
    uint8_t cCity = dat.custCity[custKey];
    if (cCity < cCityMin || cCity >= cCityMax)
      return ret;

    // Apply downscaling if needed (for Q3.1)
    uint8_t cCityMinScaled = cCityMin, cCityMaxScaled = cCityMax;
    uint8_t sCityMinScaled = sCityMin, sCityMaxScaled = sCityMax;
    if (cCityMax - cCityMin >= 50) {
      cCity /= 10; cCityMinScaled /= 10; cCityMaxScaled /= 10;
      sCity /= 10; sCityMinScaled /= 10; sCityMaxScaled /= 10;
    }
    const uint32_t year = date / 10000;
    const uint32_t yearMin = dateMin / 10000;
    const uint a = cCityMaxScaled - cCityMinScaled;
    const uint b = sCityMaxScaled - sCityMinScaled;
    ret.y = a * b * (year - yearMin) + a * (cCity - cCityMinScaled) +
            (sCity - sCityMinScaled);
    ret.x = dat.loRevenue[i];
    return ret;
  };

  cudaMemset(grpby_out, 0, nr_grp * sizeof(uint32_t));
  grpby<256, 4><<<ceil_div(factsz, 256 * 4), 256, nr_grp * sizeof(uint)>>>(
      factsz, grpby_out, nr_grp, op);
}

void s4dev(const struct ssb_schema *dat, size_t factsz, uint8_t cCityMin,
           uint8_t cCityMax, uint8_t sCityMin, uint8_t sCityMax,
           uint16_t pMfgrMin, uint16_t pMfgrMax, uint32_t dateMin,
           uint32_t dateMax, uint32_t *grpby_out, size_t nr_grp) {
  auto op = [=, dat = *dat] __device__ (uint i) {
    uint2 ret; ret.y = ELIMINATED;
    // Filter by supplier city range  
    uint16_t sCity = dat.loSuppCity[i];
    if (sCity < sCityMin || sCity >= sCityMax)
      return ret;
    // Filter by date range
    uint32_t date = dat.loOrderDate[i];
    if (date < dateMin || date >= dateMax)
      return ret;
    // Join with customer dimension to get customer city
    uint32_t custKey = dat.loCustKey[i];
    uint16_t cCity = dat.custCity[custKey];
    if (cCity < cCityMin || cCity >= cCityMax)
      return ret;
    // Join with part dimension to get manufacturer
    uint32_t partKey = dat.loPartKey[i];
    uint16_t pMfgr = dat.partMfgr[partKey];
    if (pMfgr < pMfgrMin || pMfgr >= pMfgrMax)
      return ret;

    // Apply downscaling based on ranges
    uint8_t sCityMinScaled = sCityMin, sCityMaxScaled = sCityMax;
    if (sCityMax - sCityMin >= 50) {
      sCity /= 10; sCityMinScaled /= 10; sCityMaxScaled /= 10;
    }
    uint16_t pMfgrMinScaled = pMfgrMin, pMfgrMaxScaled = pMfgrMax;
    if (pMfgrMax - pMfgrMin >= 200) {
      pMfgr /= 40; pMfgrMinScaled /= 40; pMfgrMaxScaled /= 40;
    }
    const uint32_t year = date / 10000;
    const uint32_t yearMin = dateMin / 10000;
    const uint a = sCityMaxScaled - sCityMinScaled;
    const uint b = pMfgrMaxScaled - pMfgrMinScaled;
    ret.y = a * b * (year - yearMin) + a * (sCity - sCityMinScaled) + (pMfgr - pMfgrMinScaled);
    ret.x = dat.loRevenue[i] - dat.loSupplyCost[i];
    return ret;
  };

  cudaMemset(grpby_out, 0, nr_grp * sizeof(uint32_t));
  grpby<256, 4><<<ceil_div(factsz, 256 * 4), 256, nr_grp * sizeof(uint)>>>(
      factsz, grpby_out, nr_grp, op);
}

uint32_t *ssb_demojoin(const struct ssb_schema *dat, size_t factSz) {
  uint32_t *res;
  cudaMalloc(&res, (SUMGRP_ALL + 1) * sizeof(uint32_t));
  cudaEvent_t start, stop; float msec;
  cudaEventCreate(&start); cudaEventCreate(&stop);

  #define T(x, name) cudaEventRecord(start); \
  for (size_t i = 0; i < 100; ++i) { x; } \
  cudaDeviceSynchronize(); \
  cudaEventRecord(stop); \
  cudaEventSynchronize(stop); \
  cudaEventElapsedTime(&msec, start, stop); \
  printf(#name"\t%.4f\n", msec / 100);

  printf("\nCase\tJoin\n");
  T(s1dev(dat, factSz, 19930000, 19940000, 1, 4, 0, 25, res), SSB11);
  T(s1dev(dat, factSz, 19940100, 19940200, 4, 7, 26, 36, res + 1), SSB12);
  T(s1dev(dat, factSz, 19940204, 19940211, 5, 8, 26, 36, res + 2), SSB13);

  T(s2dev(dat, factSz, 40, 80, 150, 200, res + SUMGRP_S1, NGRP_S21), SSB21);
  T(s2dev(dat, factSz, 260, 268, 200, 250, res + SUMGRP_S1 + NGRP_S21,
          NGRP_S22), SSB22);
  T(s2dev(dat, factSz, 260, 261, 50, 100, res + SUMGRP_S1 + NGRP_S21 + NGRP_S22,
          NGRP_S23), SSB23);

  T(s3dev(dat, factSz, 200, 250, 200, 250, 19920000, 19980000, res + SUMGRP_S2,
          NGRP_S31), SSB31);
  T(s3dev(dat, factSz, 190, 200, 190, 200, 19920000, 19980000,
          res + SUMGRP_S2 + NGRP_S31, NGRP_S32), SSB32);
  T(s3dev(dat, factSz, 51, 55, 51, 55, 19920000, 19980000,
          res + SUMGRP_S2 + NGRP_S31 + NGRP_S32, NGRP_S33), SSB33);
  T(s3dev(dat, factSz, 51, 55, 51, 55, 19971200, 19980000,
          res + SUMGRP_S2 + NGRP_S31 + NGRP_S32 + NGRP_S33, NGRP_S34), SSB34);

  T(s4dev(dat, factSz, 150, 200, 150, 200, 0, 400, 19920000, 19990000,
          res + SUMGRP_S3, NGRP_S41), SSB41);
  T(s4dev(dat, factSz, 150, 200, 150, 200, 0, 400, 19970000, 19990000,
          res + SUMGRP_S3 + NGRP_S41, NGRP_S42), SSB42);
  T(s4dev(dat, factSz, 150, 200, 190, 200, 120, 160, 19970000, 19990000,
          res + SUMGRP_S3 + NGRP_S41 + NGRP_S42, NGRP_S43), SSB43);
  cudaEventDestroy(start); cudaEventDestroy(stop);
  return res;
}

