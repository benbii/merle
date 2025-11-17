#include "ssbdemo.h"
#include "../primitive.cuh"
using namespace mybmpidx;
using cuda::ceil_div;
static constexpr auto nodim = recipe::nodim;

static constexpr size_t nt = 256, vt = 3, vt0 = 8, ndup = 100;
static constexpr size_t nv_ = nt * vt, nv32 = nv_ * 32;
// program + candchk is highly register and shmem intensive. Go for a less
// aggressive `vt` choice.
static constexpr size_t cp_vt = 2, cp_nv = nt * cp_vt, cp_nv32 = cp_nv * 32;

#define DOWORK \
  cudaEvent_t start, stop; float msec; \
  cudaEventCreate(&start); cudaEventCreate(&stop);\
  for (size_t i = 0; i < 3; ++i) { \
    vprg p = i == 0 ? r.perfect(instrs) \
                    : (i == 1 ? r.many_or(instrs) : r.candchk(instrs)); \
    cudaEventRecord(start); \
    for (size_t d = 0; d < ndup; ++d) { \
      if (i == 2) { \
        cudaMemset(grp_out, 0, nr_grp * sizeof(uint)); \
        vprg_grpby<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>( \
          p, grp_out, nr_grp, op); \
      } else { \
        vprg_grpby<nt, vt, vt0, false><<<ceil_div(factsz, nv32), nt>>>( \
          p, grp_out, nr_grp, op); \
      } \
    } \
    cudaEventRecord(stop); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&msec, start, stop); \
    printf("\t%.4f", msec / ndup); \
    p.release(); \
  } cudaEventDestroy(start); cudaEventDestroy(stop);

void s1bmp(const struct ssb_schema *dat, uint factsz, uint32_t dateMin,
           uint32_t dateMax, uint8_t discntMin, uint8_t discntMax,
           uint8_t qtyMin, uint8_t qtyMax, uint32_t *grp_out) {
  auto op = [=, dat = *dat] __device__ (uint i, bool chk) {
    uint2 ret; ret.y = ELIMINATED;
    // TODO: use __ldcs?
    if (chk && (dat.loOrderDate[i] < dateMin || dat.loOrderDate[i] >= dateMax))
      return ret;
    const uint8_t discnt = dat.loDiscount[i];
    if (chk && (discnt < discntMin || discnt >= discntMax))
      return ret;
    if (chk && (dat.loQuantity[i] < qtyMin || dat.loQuantity[i] >= qtyMax))
      return ret;
    ret.x = dat.loExtendedPrice[i] * discnt;
    ret.y = 0;
    return ret;
  };
  recipe r = {.factsz = factsz, .nbit = {32, 8, 8},
              .min = {dateMin, discntMin, qtyMin},
              .max = {dateMax, discntMax, qtyMax},
              .dimsz = {nodim, nodim, nodim},
              .fk = {nullptr, nullptr, nullptr},
              .attr = {dat->loOrderDate, dat->loDiscount, dat->loQuantity}};
  constexpr size_t nr_grp = 1;
  vprg::instr instrs[MAXNINSTR] = {
      {vprg::ORM, 0, 0}, {vprg::ANDM, 0, 1}, {vprg::ANDM, 0, 2},
      {vprg::END, 0, 0}, {vprg::END, 0, 0},  {vprg::END, 0, 0},
  };
  DOWORK
}

// We need nr_groups for CUDA kernels
void s2bmp(const struct ssb_schema *dat, uint factsz, uint32_t pMfgrMin,
           uint32_t pMfgrMax, uint8_t sCityMin, uint8_t sCityMax,
           uint *grp_out, size_t nr_grp) {
  auto op = [=, dat = *dat] __device__ (uint i, bool chk = false) {
    uint2 ret; ret.y = ELIMINATED;
    // Filter by supplier city range
    if (chk && (dat.loSuppCity[i] < sCityMin || dat.loSuppCity[i] >= sCityMax))
      return ret;
    // Join with part dimension to get manufacturer
    uint32_t partKey = dat.loPartKey[i];
    uint16_t mfgr = dat.partMfgr[partKey];  // Keys are zero-padded
    // Filter by manufacturer range
    if (chk && (mfgr < pMfgrMin || mfgr >= pMfgrMax))
      return ret;
    // Extract year from date (YYYYMMDD format)
    uint32_t year = dat.loOrderDate[i] / 10000;
    // GROUP BY group position calculation:
    // (mfgr - mfgrMin) + (date / 10000 - 1992) * (mfgrMax - mfgrMin)
    ret.y = (mfgr - pMfgrMin) + (year - 1992) * (pMfgrMax - pMfgrMin);
    ret.x = dat.loRevenue[i];
    return ret;
  };
  recipe r = {.factsz = factsz, .nbit = {16, 8},
              .min = {pMfgrMin, sCityMin},
              .max = {pMfgrMax, sCityMax},
              // No dimension bound check; ssb guarantees in bound
              .dimsz = {nodim, nodim},
              .fk = {dat->loPartKey, nullptr},
              .attr = {dat->partMfgr, dat->loSuppCity}};
  vprg::instr instrs[MAXNINSTR] = {
      {vprg::ORM, 0, 0}, {vprg::ANDM, 0, 1}, {vprg::END, 0, 2},
      {vprg::END, 0, 0}, {vprg::END, 0, 0},  {vprg::END, 0, 0},
  };
  DOWORK
}

void s3bmp(const struct ssb_schema *dat, uint factsz, uint8_t cCityMin,
           uint8_t cCityMax, uint8_t sCityMin, uint8_t sCityMax,
           uint32_t dateMin, uint32_t dateMax, uint32_t *grp_out, size_t nr_grp) {
  auto op = [=, dat = *dat] __device__ (uint i, bool chk = false) {
    uint2 ret; ret.y = ELIMINATED;
    // Filter by supplier city range
    uint8_t sCity = dat.loSuppCity[i];
    if (chk && (sCity < sCityMin || sCity >= sCityMax))
      return ret;
    // Filter by date range
    uint32_t date = dat.loOrderDate[i];
    if (chk && (date < dateMin || date >= dateMax))
      return ret;
    // Join with customer dimension to get customer city
    uint32_t custKey = dat.loCustKey[i];
    uint8_t cCity = dat.custCity[custKey];
    if (chk && (cCity < cCityMin || cCity >= cCityMax))
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

  recipe r = {.factsz = factsz, .nbit = {8, 8, 32},
              .min = {cCityMin, sCityMin, dateMin},
              .max = {cCityMax, sCityMax, dateMax},
              .dimsz = {nodim, nodim, nodim},
              .fk = {dat->loCustKey, nullptr, nullptr},
              .attr = {dat->custCity, dat->loSuppCity, dat->loOrderDate}};
  vprg::instr instrs[MAXNINSTR] = {
      {vprg::ORM, 0, 0}, {vprg::ANDM, 0, 1}, {vprg::ANDM, 0, 2},
      {vprg::END, 0, 0}, {vprg::END, 0, 0},  {vprg::END, 0, 0},
  };
  DOWORK
}

void s4bmp(const struct ssb_schema *dat, uint factsz, uint8_t cCityMin,
           uint8_t cCityMax, uint8_t sCityMin, uint8_t sCityMax,
           uint16_t pMfgrMin, uint16_t pMfgrMax, uint32_t dateMin,
           uint32_t dateMax, uint32_t *grp_out, size_t nr_grp) {
  auto op = [=, dat = *dat] __device__ (uint i, bool chk = false) {
    uint2 ret; ret.y = ELIMINATED;
    // Filter by supplier city range  
    uint8_t sCity = dat.loSuppCity[i];
    if (chk && (sCity < sCityMin || sCity >= sCityMax))
      return ret;
    // Filter by date range
    uint32_t date = dat.loOrderDate[i];
    if (chk && (date < dateMin || date >= dateMax))
      return ret;
    // Join with customer dimension to get customer city
    if (chk) {
      uint32_t custKey = dat.loCustKey[i];
      uint8_t cCity = dat.custCity[custKey];
      if (cCity < cCityMin || cCity >= cCityMax)
        return ret;
    }
    // Join with part dimension to get manufacturer
    uint32_t partKey = dat.loPartKey[i];
    uint16_t pMfgr = dat.partMfgr[partKey];
    if (chk && (pMfgr < pMfgrMin || pMfgr >= pMfgrMax))
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

  recipe r = {.factsz = factsz, .nbit = {8, 8, 16, 32},
              .min = {cCityMin, sCityMin, pMfgrMin, dateMin},
              .max = {cCityMax, sCityMax, pMfgrMax, dateMax},
              .dimsz = {nodim, nodim, nodim, nodim},
              .fk = {dat->loCustKey, nullptr, dat->loPartKey, nullptr},
              .attr = {dat->custCity, dat->loSuppCity, dat->partMfgr, dat->loOrderDate}};
  vprg::instr instrs[MAXNINSTR] = {
      {vprg::ORM, 0, 0}, {vprg::ANDM, 0, 1}, {vprg::ANDM, 0, 2},
      {vprg::ANDM, 0, 3}, {vprg::END, 0, 0},  {vprg::END, 0, 0},
  };
  if (dateMin <= 19920000 && dateMax >= 19990000)
    r.attr[3] = nullptr, instrs[3].opcode = vprg::END;
  DOWORK
}

uint32_t *ssb_demobmp(const struct ssb_schema *dat, size_t factSz) {
  uint32_t *res;
  cudaMalloc(&res, (SUMGRP_ALL + 1) * sizeof(uint32_t));

  printf("\n\nCase\tPerfect\tManyOrs\tCandchk\nSSB11");
  s1bmp(dat, factSz, 19930000, 19940000, 1, 4, 0, 25, res);
  printf("\nSSB12");
  s1bmp(dat, factSz, 19940100, 19940200, 4, 7, 26, 36, res + 1);
  printf("\nSSB13");
  s1bmp(dat, factSz, 19940204, 19940211, 5, 8, 26, 36, res + 2);

  printf("\nSSB21");
  s2bmp(dat, factSz, 40, 80, 150, 200, res + SUMGRP_S1, NGRP_S21);
  printf("\nSSB22");
  s2bmp(dat, factSz, 260, 268, 200, 250, res + SUMGRP_S1 + NGRP_S21, NGRP_S22);
  printf("\nSSB23");
  s2bmp(dat, factSz, 260, 261, 50, 100, res + SUMGRP_S1 + NGRP_S21 + NGRP_S22, NGRP_S23);

  printf("\nSSB31");
  s3bmp(dat, factSz, 200, 250, 200, 250, 19920000, 19980000,
        res + SUMGRP_S2, NGRP_S31);
  printf("\nSSB32");
  s3bmp(dat, factSz, 190, 200, 190, 200, 19920000, 19980000,
        res + SUMGRP_S2 + NGRP_S31, NGRP_S32);
  printf("\nSSB33");
  s3bmp(dat, factSz, 51, 55, 51, 55, 19920000, 19980000,
        res + SUMGRP_S2 + NGRP_S31 + NGRP_S32, NGRP_S33);
  printf("\nSSB34");
  s3bmp(dat, factSz, 51, 55, 51, 55, 19971200, 19980000,
        res + SUMGRP_S2 + NGRP_S31 + NGRP_S32 + NGRP_S33, NGRP_S34);

  printf("\nSSB41");
  s4bmp(dat, factSz, 150, 200, 150, 200, 0, 400, 19920000, 19990000,
        res + SUMGRP_S3, NGRP_S41);
  printf("\nSSB42");
  s4bmp(dat, factSz, 150, 200, 150, 200, 0, 400, 19970000, 19990000,
        res + SUMGRP_S3 + NGRP_S41, NGRP_S42);
  printf("\nSSB43");
  s4bmp(dat, factSz, 150, 200, 190, 200, 120, 160, 19970000, 19990000,
        res + SUMGRP_S3 + NGRP_S41 + NGRP_S42, NGRP_S43);
  return res;
}

