#include "hardcoded_frontend.cuh"
using namespace mybmpidx;
using cuda::ceil_div;

void s1dev(const struct ssb_schema *dat, size_t factsz, uint16_t dateMin,
           uint16_t dateMax, uint8_t discntMin, uint8_t discntMax,
           uint8_t qtyMin, uint8_t qtyMax, uint32_t *red_out) {
  s1op op {
      dateMin, dateMax, dat->loOrderDate,
      discntMin, discntMax, dat->loDiscount,
      qtyMin, qtyMax, dat->loQuantity, dat->loExtendedPrice,
  };
  cudaMemset(red_out, 0, sizeof(uint32_t));
  // Reduction -> group by with only one group
  grpby_small<256, 8, 1>
      <<<ceil_div(factsz, 256 * 8), 256>>>(factsz, red_out, op);
}

// We need nr_groups for CUDA kernels
void s2dev(const struct ssb_schema *dat, size_t factsz, uint16_t pMfgrMin,
           uint16_t pMfgrMax, uint8_t sCityMin, uint8_t sCityMax,
           uint32_t *grpby_out, size_t nr_grp) {
  s2op op {
      dat->loPartKey, pMfgrMin, pMfgrMax, dat->partMfgr,
      sCityMin, sCityMax, dat->loSuppCity,
      dat->loOrderDate, dat->loRevenue,
  };
  cudaMemset(grpby_out, 0, nr_grp * sizeof(uint32_t));
  grpby<256, 4><<<ceil_div(factsz, 256 * 4), 256, nr_grp * sizeof(uint)>>>(
      factsz, grpby_out, nr_grp, op);
}

void s3dev(const struct ssb_schema *dat, size_t factsz, uint8_t cCityMin,
           uint8_t cCityMax, uint8_t sCityMin, uint8_t sCityMax,
           uint16_t dateMin, uint16_t dateMax, uint32_t *grpby_out, size_t nr_grp) {
  s3op op {
    dat->loCustKey, cCityMin, cCityMax, dat->custCity,
    sCityMin, sCityMax, dat->loSuppCity,
    dateMin, dateMax, dat->loOrderDate, dat->loRevenue
  };
  cudaMemset(grpby_out, 0, nr_grp * sizeof(uint32_t));
  grpby<256, 4><<<ceil_div(factsz, 256 * 4), 256, nr_grp * sizeof(uint)>>>(
      factsz, grpby_out, nr_grp, op);
}

void s4dev(const struct ssb_schema *dat, size_t factsz, uint8_t cCityMin,
           uint8_t cCityMax, uint8_t sCityMin, uint8_t sCityMax,
           uint16_t pMfgrMin, uint16_t pMfgrMax, uint16_t dateMin,
           uint16_t dateMax, uint32_t *grpby_out, size_t nr_grp) {
  s4op op{dat->loCustKey, dat->loPartKey,   cCityMin,       cCityMax,
          dat->custCity,  sCityMin,         sCityMax,       dat->loSuppCity,
          pMfgrMin,       pMfgrMax,         dat->partMfgr,  dateMin,
          dateMax,        dat->loOrderDate, dat->loRevenue, dat->loSupplyCost};
  cudaMemset(grpby_out, 0, nr_grp * sizeof(uint32_t));
  grpby<256, 4><<<ceil_div(factsz, 256 * 4), 256, nr_grp * sizeof(uint)>>>(
      factsz, grpby_out, nr_grp, op);
}

uint32_t *ssb_gpujoin(const struct ssb_schema *dat, size_t factSz) {
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
  T(s1dev(dat, factSz, SSBDATE_930101, SSBDATE_940101, 1, 4, 1, 25, res), SSB11);
  T(s1dev(dat, factSz, SSBDATE_940101, SSBDATE_940201, 4, 7, 26, 36, res + 1), SSB12);
  T(s1dev(dat, factSz, SSBDATE_940204, SSBDATE_940211, 5, 8, 26, 36, res + 2), SSB13);

  T(s2dev(dat, factSz, 40, 80, 150, 200, res + SUMGRP_S1, NGRP_S21), SSB21);
  T(s2dev(dat, factSz, 260, 268, 200, 250, res + SUMGRP_S1 + NGRP_S21,
          NGRP_S22), SSB22);
  T(s2dev(dat, factSz, 260, 261, 50, 100, res + SUMGRP_S1 + NGRP_S21 + NGRP_S22,
          NGRP_S23), SSB23);

  T(s3dev(dat, factSz, 200, 250, 200, 250, SSBDATE_920101, SSBDATE_980101, res + SUMGRP_S2,
          NGRP_S31), SSB31);
  T(s3dev(dat, factSz, 190, 200, 190, 200, SSBDATE_920101, SSBDATE_980101,
          res + SUMGRP_S2 + NGRP_S31, NGRP_S32), SSB32);
  T(s3dev(dat, factSz, 51, 55, 51, 55, SSBDATE_920101, SSBDATE_980101,
          res + SUMGRP_S2 + NGRP_S31 + NGRP_S32, NGRP_S33), SSB33);
  T(s3dev(dat, factSz, 51, 55, 51, 55, SSBDATE_971201, SSBDATE_980101,
          res + SUMGRP_S2 + NGRP_S31 + NGRP_S32 + NGRP_S33, NGRP_S34), SSB34);

  T(s4dev(dat, factSz, 150, 200, 150, 200, 0, 400, SSBDATE_920101, SSBDATE_990101,
          res + SUMGRP_S3, NGRP_S41), SSB41);
  T(s4dev(dat, factSz, 150, 200, 150, 200, 0, 400, SSBDATE_970101, SSBDATE_990101,
          res + SUMGRP_S3 + NGRP_S41, NGRP_S42), SSB42);
  T(s4dev(dat, factSz, 150, 200, 190, 200, 120, 160, SSBDATE_970101, SSBDATE_990101,
          res + SUMGRP_S3 + NGRP_S41 + NGRP_S42, NGRP_S43), SSB43);
  cudaEventDestroy(start); cudaEventDestroy(stop);
  return res;
}
