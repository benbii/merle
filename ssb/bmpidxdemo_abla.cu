#include "ssbdemo.h"
#include "../primitive_abla.cuh"
using namespace mybmpidx;
using namespace mybmpidx::abla;
using cuda::ceil_div;
static constexpr auto nodim = recipe::nodim;

// lazy
static uint *d_possi, *d_uncert, *d_onelist;

static constexpr size_t nt = 256, vt = 3, vt0 = 8, ndup = 100;
static constexpr size_t nv_ = nt * vt, nv32 = nv_ * 32;
#ifdef LARGE_SMEM
static constexpr size_t cp_vt = vt, cp_nv32 = nv32;
#else
// Program + candchk is highly shmem intensive.
// Go for a less aggressive `vt` choice.
static constexpr size_t cp_vt = 2, cp_nv = nt * cp_vt, cp_nv32 = cp_nv * 32;
#endif

#define PLEASE \
  cudaEvent_t start, stop; float msec; \
  cudaEventCreate(&start); cudaEventCreate(&stop);\
  for (size_t i = 0; i < 3; ++i) { \
    vprg p = i == 0 ? r.perfect(instrs) \
                    : (i == 1 ? r.many_or(instrs) : r.candchk(instrs)); \
    ands a; memcpy(&a, &p, sizeof(ands)); \
    cudaEventRecord(start); \
    for (size_t d = 0; d < ndup; ++d) { \
      if (i == 2) \
        ands_grpby<nt, vt, vt0, true><<<ceil_div(factsz, nv32), nt>>>( \
          a, grp_out, nr_grp, op); \
      else \
        ands_grpby<nt, vt, vt0, false><<<ceil_div(factsz, nv32), nt>>>( \
          a, grp_out, nr_grp, op); \
    } \
    cudaEventRecord(stop); cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&msec, start, stop); \
    printf("\t%.4f", msec / ndup); \
    p.release(); \
  } cudaEventDestroy(start); cudaEventDestroy(stop); \
  for (size_t i = 0; i < 3; ++i) { \
    vprg p = i == 0 ? r.perfect(instrs) \
                    : (i == 1 ? r.many_or(instrs) : r.candchk(instrs)); \
    float4 foo, bar = {0.0,0.0,0.0,0.0};\
    for (size_t d = 0; d < ndup; ++d) { \
      if (i == 2) \
        foo = nofuse_abla<nt, cp_vt, vt, vt, true>( \
          p, nr_grp, op, grp_out, d_possi, d_uncert, d_onelist); \
      else \
        foo = nofuse_abla<nt, cp_vt, vt, vt, false>( \
          p, nr_grp, op, grp_out, d_possi, d_uncert, d_onelist); \
      bar.x += foo.x; bar.y += foo.y; bar.z += foo.z; bar.w += foo.w; \
    } \
    bar.x /= ndup; bar.y /= ndup; bar.z /= ndup; bar.w /= ndup; \
    printf("\t%.4f\t%.4f\t%.4f", bar.x, bar.y, bar.z); \
    p.release(); \
  }

void s1abl(const struct ssb_schema *dat, uint factsz, uint16_t dateMin,
           uint16_t dateMax, uint8_t discntMin, uint8_t discntMax,
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
  recipe r = {.factsz = factsz, .nbit = {16, 8, 8},
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
  PLEASE

}

// We need nr_groups for CUDA kernels
void s2abl(const struct ssb_schema *dat, uint factsz, uint32_t pMfgrMin,
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
    uint32_t year = ssbDateToYear(dat.loOrderDate[i]);
    ret.y = (mfgr - pMfgrMin) + year * (pMfgrMax - pMfgrMin);
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
  PLEASE
}

void s3abl(const struct ssb_schema *dat, uint factsz, uint8_t cCityMin,
           uint8_t cCityMax, uint8_t sCityMin, uint8_t sCityMax,
           uint16_t dateMin, uint16_t dateMax, uint32_t *grp_out, size_t nr_grp) {
  auto op = [=, dat = *dat] __device__ (uint i, bool chk = false) {
    uint2 ret; ret.y = ELIMINATED;
    // Filter by supplier city range
    uint8_t sCity = dat.loSuppCity[i];
    if (chk && (sCity < sCityMin || sCity >= sCityMax))
      return ret;
    // Filter by date range
    uint16_t date = dat.loOrderDate[i];
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
    const uint32_t year = ssbDateToYear(date);
    const uint32_t yearMin = ssbDateToYear(dateMin);
    const uint a = cCityMaxScaled - cCityMinScaled;
    const uint b = sCityMaxScaled - sCityMinScaled;
    ret.y = a * b * (year - yearMin) + a * (cCity - cCityMinScaled) +
            (sCity - sCityMinScaled);
    ret.x = dat.loRevenue[i];
    return ret;
  };

  recipe r = {.factsz = factsz, .nbit = {8, 8, 16},
              .min = {cCityMin, sCityMin, dateMin},
              .max = {cCityMax, sCityMax, dateMax},
              .dimsz = {nodim, nodim, nodim},
              .fk = {dat->loCustKey, nullptr, nullptr},
              .attr = {dat->custCity, dat->loSuppCity, dat->loOrderDate}};
  vprg::instr instrs[MAXNINSTR] = {
      {vprg::ORM, 0, 0}, {vprg::ANDM, 0, 1}, {vprg::ANDM, 0, 2},
      {vprg::END, 0, 0}, {vprg::END, 0, 0},  {vprg::END, 0, 0},
  };
  PLEASE
}

void s4abl(const struct ssb_schema *dat, uint factsz, uint8_t cCityMin,
           uint8_t cCityMax, uint8_t sCityMin, uint8_t sCityMax,
           uint16_t pMfgrMin, uint16_t pMfgrMax, uint16_t dateMin,
           uint16_t dateMax, uint32_t *grp_out, size_t nr_grp) {
  auto op = [=, dat = *dat] __device__ (uint i, bool chk = false) {
    uint2 ret; ret.y = ELIMINATED;
    // Filter by supplier city range  
    uint8_t sCity = dat.loSuppCity[i];
    if (chk && (sCity < sCityMin || sCity >= sCityMax))
      return ret;
    // Filter by date range
    uint16_t date = dat.loOrderDate[i];
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
    const uint32_t year = ssbDateToYear(date);
    const uint32_t yearMin = ssbDateToYear(dateMin);
    const uint a = sCityMaxScaled - sCityMinScaled;
    const uint b = pMfgrMaxScaled - pMfgrMinScaled;
    ret.y = a * b * (year - yearMin) + a * (sCity - sCityMinScaled) + (pMfgr - pMfgrMinScaled);
    ret.x = dat.loRevenue[i] - dat.loSupplyCost[i];
    return ret;
  };

  recipe r = {.factsz = factsz, .nbit = {8, 8, 16, 16},
              .min = {cCityMin, sCityMin, pMfgrMin, dateMin},
              .max = {cCityMax, sCityMax, pMfgrMax, dateMax},
              .dimsz = {nodim, nodim, nodim, nodim},
              .fk = {dat->loCustKey, nullptr, dat->loPartKey, nullptr},
              .attr = {dat->custCity, dat->loSuppCity, dat->partMfgr, dat->loOrderDate}};
  vprg::instr instrs[MAXNINSTR] = {
      {vprg::ORM, 0, 0}, {vprg::ANDM, 0, 1}, {vprg::ANDM, 0, 2},
      {vprg::ANDM, 0, 3}, {vprg::END, 0, 0},  {vprg::END, 0, 0},
  };
  if (dateMin <= SSBDATE_920101 && dateMax >= SSBDATE_990101)
    r.attr[3] = nullptr, instrs[3].opcode = vprg::END;
  PLEASE
}

uint32_t *ssb_demobmp_abl(const struct ssb_schema *dat, size_t factSz) {
  uint32_t *res;
  cudaMalloc(&res, (SUMGRP_ALL + 1) * sizeof(uint32_t));
  cudaMalloc(&d_possi, ceil_div(factSz, 32) * sizeof(uint32_t));
  cudaMalloc(&d_uncert, ceil_div(factSz, 32) * sizeof(uint32_t));
  cudaMalloc(&d_onelist, ceil_div(factSz, 10) * sizeof(uint32_t));

  printf("\nCase\tPftNprg\tOrsNprg\tChkNprg\tPftStg1\tPftStg2\tPftStg3"
         "\tOrsStg1\tOrsStg2\tOrsStg3\tChkStg1\tChkStg2\tChkStg3\nSSB11");
  s1abl(dat, factSz, SSBDATE_930101, SSBDATE_940101, 1, 4, 0, 25, res);
  printf("\nSSB12");
  s1abl(dat, factSz, SSBDATE_940101, SSBDATE_940201, 4, 7, 26, 36, res + 1);
  printf("\nSSB13");
  s1abl(dat, factSz, SSBDATE_940204, SSBDATE_940211, 5, 8, 26, 36, res + 2);

  printf("\nSSB21");
  s2abl(dat, factSz, 40, 80, 150, 200, res + SUMGRP_S1, NGRP_S21);
  printf("\nSSB22");
  s2abl(dat, factSz, 260, 268, 200, 250, res + SUMGRP_S1 + NGRP_S21, NGRP_S22);
  printf("\nSSB23");
  s2abl(dat, factSz, 260, 261, 50, 100, res + SUMGRP_S1 + NGRP_S21 + NGRP_S22, NGRP_S23);

  printf("\nSSB31");
  s3abl(dat, factSz, 200, 250, 200, 250, SSBDATE_920101, SSBDATE_980101,
        res + SUMGRP_S2, NGRP_S31);
  printf("\nSSB32");
  s3abl(dat, factSz, 190, 200, 190, 200, SSBDATE_920101, SSBDATE_980101,
        res + SUMGRP_S2 + NGRP_S31, NGRP_S32);
  printf("\nSSB33");
  s3abl(dat, factSz, 51, 55, 51, 55, SSBDATE_920101, SSBDATE_980101,
        res + SUMGRP_S2 + NGRP_S31 + NGRP_S32, NGRP_S33);
  printf("\nSSB34");
  s3abl(dat, factSz, 51, 55, 51, 55, SSBDATE_971201, SSBDATE_980101,
        res + SUMGRP_S2 + NGRP_S31 + NGRP_S32 + NGRP_S33, NGRP_S34);

  printf("\nSSB41");
  s4abl(dat, factSz, 150, 200, 150, 200, 0, 400, SSBDATE_920101, SSBDATE_990101,
        res + SUMGRP_S3, NGRP_S41);
  printf("\nSSB42");
  s4abl(dat, factSz, 150, 200, 150, 200, 0, 400, SSBDATE_970101, SSBDATE_990101,
        res + SUMGRP_S3 + NGRP_S41, NGRP_S42);
  printf("\nSSB43");
  s4abl(dat, factSz, 150, 200, 190, 200, 120, 160, SSBDATE_970101, SSBDATE_990101,
        res + SUMGRP_S3 + NGRP_S41 + NGRP_S42, NGRP_S43);

  cudaFree(d_possi); cudaFree(d_uncert); cudaFree(d_onelist);
  return res;
}
