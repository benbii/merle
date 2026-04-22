#pragma once
#include "ssbdemo.h"
#include "../primitive_abla.cuh"
using namespace mybmpidx;
using cuda::ceil_div;

// vt0=8 is sufficient for SSB
static constexpr size_t nt = 256, vt = 3, vt0 = 8, ndup = 100;
static constexpr size_t nv_ = nt * vt, nv32 = nv_ * 32;
#ifdef LARGE_SMEM
static constexpr size_t cp_vt = vt, cp_nv32 = nv32;
#else
// Program + candchk is highly shmem intensive.
// Go for a less aggressive `vt` choice.
static constexpr size_t cp_vt = 2, cp_nv = nt * cp_vt, cp_nv32 = cp_nv * 32;
#endif

// lambdas used across files
struct s1op { // not slop I swear :D
  uint16_t dateMin, dateMax, *loOrderDate;
  uint8_t discntMin, discntMax, *loDiscount;
  uint8_t qtyMin, qtyMax, *loQuantity;
  uint32_t *extendedPrice;
  uint2 __device__ operator()(uint i, bool chk = true) const {
    uint2 ret; ret.y = ELIMINATED;
    // TODO: use __ldcs?
    if (chk && (loOrderDate[i] < dateMin || loOrderDate[i] >= dateMax))
      return ret;
    const uint8_t discnt = loDiscount[i];
    if (chk && (discnt < discntMin || discnt >= discntMax))
      return ret;
    if (chk && (loQuantity[i] < qtyMin || loQuantity[i] >= qtyMax))
      return ret;
    ret.x = extendedPrice[i] * discnt;
    ret.y = 0;
    return ret;
  }
};

struct s2op {
  uint32_t *loPartKey;
  uint16_t pMfgrMin, pMfgrMax, *partMfgr;
  uint8_t sCityMin, sCityMax, *loSuppCity;
  uint16_t *loOrderDate;
  uint32_t *loRevenue;
  uint2 __device__ operator()(uint i, bool chk = true) const {
    uint2 ret; ret.y = ELIMINATED;
    // Filter by supplier city range
    if (chk && (loSuppCity[i] < sCityMin || loSuppCity[i] >= sCityMax))
      return ret;
    // Join with part dimension to get manufacturer
    uint32_t partKey = loPartKey[i];
    uint16_t mfgr = partMfgr[partKey];  // Keys are zero-padded
    // Filter by manufacturer range
    if (chk && (mfgr < pMfgrMin || mfgr >= pMfgrMax))
      return ret;
    uint32_t year = ssbDateToYear(loOrderDate[i]);
    ret.y = (mfgr - pMfgrMin) + year * (pMfgrMax - pMfgrMin);
    ret.x = loRevenue[i];
    return ret;
  }
};

struct s3op {
  uint32_t *loCustKey;
  uint8_t cCityMin, cCityMax, *custCity;
  uint8_t sCityMin, sCityMax, *loSuppCity;
  uint16_t dateMin, dateMax, *loOrderDate;
  uint32_t *loRevenue;
  uint2 __device__ operator()(uint i, bool chk = true) const {
    uint2 ret; ret.y = ELIMINATED;
    // Filter by supplier city range
    uint8_t sCity = loSuppCity[i];
    if (chk && (sCity < sCityMin || sCity >= sCityMax))
      return ret;
    // Filter by date range
    uint16_t date = loOrderDate[i];
    if (chk && (date < dateMin || date >= dateMax))
      return ret;
    // Join with customer dimension to get customer city
    uint32_t custKey = loCustKey[i];
    uint8_t cCity = custCity[custKey];
    if (chk && (cCity < cCityMin || cCity >= cCityMax))
      return ret;

    // Apply downscaling if needed (for Q3.1)
    uint8_t cCityMinScaled = cCityMin, cCityMaxScaled = cCityMax;
    uint8_t sCityMinScaled = sCityMin, sCityMaxScaled = sCityMax;
    if (cCityMax - cCityMin >= 50) {
      cCity /= 10; cCityMinScaled /= 10; cCityMaxScaled /= 10;
      sCity /= 10; sCityMinScaled /= 10; sCityMaxScaled /= 10;
    }
    const uint32_t year = ssbDateToYear(date);
    const uint32_t yearMin = ssbDateToYear(dateMin);
    const uint a = cCityMaxScaled - cCityMinScaled;
    const uint b = sCityMaxScaled - sCityMinScaled;
    ret.y = a * b * (year - yearMin) + a * (cCity - cCityMinScaled) +
            (sCity - sCityMinScaled);
    ret.x = loRevenue[i];
    return ret;
  }
};

struct s4op {
  uint32_t *loCustKey, *loPartKey;
  uint8_t cCityMin, cCityMax, *custCity;
  uint8_t sCityMin, sCityMax, *loSuppCity;
  uint16_t pMfgrMin, pMfgrMax, *partMfgr;
  uint16_t dateMin, dateMax, *loOrderDate;
  uint32_t *loRevenue, *loSupplyCost;

  uint2 __device__ operator()(uint i, bool chk = true) const {
    uint2 ret; ret.y = ELIMINATED;
    // Filter by supplier city range
    uint8_t sCity = loSuppCity[i];
    if (chk && (sCity < sCityMin || sCity >= sCityMax))
      return ret;
    // Filter by date range
    uint16_t date = loOrderDate[i];
    if (chk && (date < dateMin || date >= dateMax))
      return ret;
    // Join with customer dimension to get customer city
    if (chk) {
      uint32_t custKey = loCustKey[i];
      uint8_t cCity = custCity[custKey];
      if (cCity < cCityMin || cCity >= cCityMax)
        return ret;
    }
    // Join with part dimension to get manufacturer
    uint32_t partKey = loPartKey[i];
    uint16_t pMfgr = partMfgr[partKey];
    if (chk && (pMfgr < pMfgrMin || pMfgr >= pMfgrMax))
      return ret;

    uint8_t sCityMinScaled = sCityMin, sCityMaxScaled = sCityMax;
    if (sCityMax - sCityMin >= 50) {
      sCity /= 10; sCityMinScaled /= 10; sCityMaxScaled /= 10;
    }
    uint16_t pMfgrMinScaled = pMfgrMin, pMfgrMaxScaled = pMfgrMax;
    if (pMfgrMax - pMfgrMin >= 200) {
      pMfgr /= 40; pMfgrMinScaled /= 40; pMfgrMaxScaled /= 40;
    }
    const uint32_t year = ssbDateToYear(date);
    const uint32_t yearMin = ssbDateToYear(dateMin);
    const uint a = sCityMaxScaled - sCityMinScaled;
    const uint b = pMfgrMaxScaled - pMfgrMinScaled;
    ret.y = a * b * (year - yearMin) + a * (sCity - sCityMinScaled) +
            (pMfgr - pMfgrMinScaled);
    ret.x = loRevenue[i] - loSupplyCost[i];
    return ret;
  };
};

template <typename op_t>
void DOWORK(const char *a, const char *b, mybmpidx::vprg &p, uint32_t *grp_out,
            size_t nr_grp, op_t &op, uint factsz, bool join = false) {
  printf(join ? "\n%s\t%s" : "\n%s\t%s\t0", a, b);
  cudaEvent_t start, stop; float msec;
  cudaEventCreate(&start), cudaEventCreate(&stop);
  uint *idxbuf, *possi, *uncert, *onelist, h_count = 6666;
  cudaMalloc(&idxbuf, sizeof(uint) * 6 * ceil_div(factsz, 32));
  cudaMalloc(&possi, ceil_div(factsz, 32) * sizeof(uint));
  cudaMalloc(&uncert, ceil_div(factsz, 32) * sizeof(uint));
  cudaMalloc(&onelist, ceil_div(factsz, 10) * sizeof(uint));
  if (!onelist || !uncert || !possi || !idxbuf)
    exit(fprintf(stderr, __FILE__ " CUDA OOM\n"));

  for (size_t i = join ? 0 : 2; i < 14; i += 2) {
    cudaEventRecord(start);
    for (size_t d = 0; d < ndup; ++d) {
      switch (i + (size_t)p.is_nochk()) {
      case 0: case 1: // join
        cudaMemset(grp_out, 0, nr_grp * sizeof(uint32_t));
        grpby<256, 4>
            <<<ceil_div(factsz, 256 * 4), 256, nr_grp * sizeof(uint)>>>(
                factsz, grp_out, nr_grp, op); break;
      case 2: // OUR WORKS; PLEASE BE THE BEST PERFORMING ONE
        cudaMemset(grp_out + SUMGRP_ALL, 0, nr_grp * sizeof(uint32_t));
        vprg_grpby<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
            p, grp_out + SUMGRP_ALL, nr_grp, op); break;
      case 3:
        cudaMemset(grp_out + SUMGRP_ALL, 0, nr_grp * sizeof(uint32_t));
        vprg_grpby<nt, vt, vt0, false><<<ceil_div(factsz, nv32), nt>>>(
            p, grp_out + SUMGRP_ALL, nr_grp, op); break;
      case 4: // baseline fusion; bins go to gmem, naively pasted on the rest
        cudaMemset(grp_out + SUMGRP_ALL * 2, 0, nr_grp * sizeof(uint32_t));
        cudaMemset(idxbuf, 0, sizeof(uint) * 6 * ceil_div(factsz, 32));
        abla::base_fuse<nt, cp_vt, true><<<ceil_div(factsz, cp_nv32), nt>>>(
            p, grp_out + SUMGRP_ALL * 2, nr_grp, op, idxbuf); break;
      case 5:
        cudaMemset(grp_out + SUMGRP_ALL * 2, 0, nr_grp * sizeof(uint32_t));
        cudaMemset(idxbuf, 0, sizeof(uint) * 3 * ceil_div(factsz, 32));
        abla::base_fuse<nt, vt, false><<<ceil_div(factsz, nv32), nt>>>(
            p, grp_out + SUMGRP_ALL * 2, nr_grp, op, idxbuf); break;
      case 6: // basically no fusion at all; bins and intermediates in gmem
        cudaMemset(idxbuf, 0, sizeof(uint) * 6 * ceil_div(factsz, 32));
        abla::_stg1<nt, cp_vt, true>
            <<<ceil_div(factsz, cp_nv32), nt>>>(p, possi, uncert, idxbuf);
        break;
      case 7:
        cudaMemset(idxbuf, 0, sizeof(uint) * 3 * ceil_div(factsz, 32));
        abla::_stg1<nt, vt, false>
            <<<ceil_div(factsz, nv32), nt>>>(p, possi, uncert, idxbuf);
        break;
      case 8: // stage 1 of virtual-program-only fusion; bins go to smem
        abla::_stg1<nt, cp_vt, true>
            <<<ceil_div(factsz, cp_nv32), nt>>>(p, possi, uncert, nullptr);
        break;
      case 9: // stage 1 of virtual-program-only fusion; no checking
        abla::_stg1<nt, vt, false>
            <<<ceil_div(factsz, nv32), nt>>>(p, possi, uncert, nullptr);
        break;
      case 10: // stage 2 of separate bitmap & join
        cudaMemset(idxbuf, 0, sizeof(uint));
        // idxbuf no longer used; reuse idxbuf[0] as popcount
        abla::_stg2<nt, vt, true><<<ceil_div(factsz, nv32), nt>>>(
            possi, factsz, idxbuf, onelist, uncert); break;
      case 11:
        cudaMemset(idxbuf, 0, sizeof(uint));
        abla::_stg2<nt, vt, false><<<ceil_div(factsz, nv32), nt>>>(
            possi, factsz, idxbuf, onelist, uncert); break;
      case 12:
        cudaMemcpy(&h_count, idxbuf, sizeof(uint), cudaMemcpyDeviceToHost);
        cudaMemset(grp_out + SUMGRP_ALL * 3, 0, nr_grp * sizeof(uint32_t));
        abla::_stg3<nt, vt, true><<<ceil_div(h_count, nv_), nt, nr_grp * 4>>>(
            onelist, h_count, grp_out + SUMGRP_ALL * 3, nr_grp, op); break;
      case 13:
        cudaMemcpy(&h_count, idxbuf, sizeof(uint), cudaMemcpyDeviceToHost);
        cudaMemset(grp_out + SUMGRP_ALL * 3, 0, nr_grp * sizeof(uint32_t));
        abla::_stg3<nt, vt, false><<<ceil_div(h_count, nv_), nt, nr_grp * 4>>>(
            onelist, h_count, grp_out + SUMGRP_ALL * 3, nr_grp, op); break;
      }
      if (cudaGetLastError() != cudaSuccess)
        exit(fprintf(stderr, "CUDA ERROR i=%zu d=%zu h=%u\n", i, d, h_count));
    }

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msec, start, stop);
    printf("\t%.4f", msec / ndup);
  }
  cudaEventDestroy(start), cudaEventDestroy(stop);
  cudaFree(idxbuf), cudaFree(possi), cudaFree(uncert), cudaFree(onelist);
}

extern template void DOWORK<s1op>(const char *a, const char *b, vprg &p,
                                  uint32_t *grp_out, size_t nr_grp, s1op &op,
                                  uint factsz, bool join);
extern template void DOWORK<s2op>(const char *a, const char *b, vprg &p,
                                  uint32_t *grp_out, size_t nr_grp, s2op &op,
                                  uint factsz, bool join);
extern template void DOWORK<s3op>(const char *a, const char *b, vprg &p,
                                  uint32_t *grp_out, size_t nr_grp, s3op &op,
                                  uint factsz, bool join);
extern template void DOWORK<s4op>(const char *a, const char *b, vprg &p,
                                  uint32_t *grp_out, size_t nr_grp, s4op &op,
                                  uint factsz, bool join);

namespace c {
// compiled functions dedicated for a specific query flavor
// commented ones falls back to lower density ones
void s11s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s12s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s13s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s21s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s22s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s23s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s31s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s32s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s33s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s34s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s41s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s42s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s43s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s11d(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s12d(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s13d(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s21d(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s22d(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s23d(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s31d(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s32d(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s33d(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s34d(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
// void s41d(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s42d(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s43d(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s11m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s12m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s13m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s21m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s22m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s23m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s31m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s32m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s33m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s34m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
// void s41m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s42m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
void s43m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out);
}
