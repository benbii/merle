#include "lambdas.cuh"
using namespace mybmpidx;
static constexpr auto ANDM = vprg::ANDM, LOAD = vprg::ORM, AND = vprg::AND, NOT = vprg::NOT;
static constexpr const uint64_t *cCitySpBin = sCitySpBin, *cCityDeBin = sCityDeBin;

#define SPARSE(c)                                                              \
  col::from_bin(sizeof(bmp->c##Sparse) / sizeof(intptr_t), bmp->c##Sparse,     \
                c##SpBin, c##Min, c##Max)
#define DENSE(c)                                                               \
  col::from_bin2(sizeof(bmp->c##Sparse) / sizeof(intptr_t), bmp->c##Sparse,    \
                 c##SpBin, sizeof(bmp->c##Dense) / sizeof(intptr_t),           \
                 bmp->c##Dense, c##DeBin, c##Min, c##Max)

using ded_fn_t = void (*)(const ssb_schema*, ssb_bmp*, size_t, uint*);
struct ded_set {
  ded_fn_t sparse = nullptr, medium = nullptr, dense = nullptr;
};

float DEDICATE(ded_fn_t fun, const ssb_schema *dat, ssb_bmp *bmp,
               size_t factsz, uint *grp_out) {
  if (fun == nullptr)
    return 0.0f;
  cudaEvent_t start, stop; float msec;
  cudaEventCreate(&start), cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (size_t i = 0; i < ndup; ++i)
    fun(dat, bmp, factsz, grp_out + SUMGRP_ALL * 4);
  cudaEventRecord(stop), cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msec, start, stop);
  cudaEventDestroy(start), cudaEventDestroy(stop);
  return msec / ndup;
}

void s1fix(const char *a, const ssb_schema *dat, ssb_bmp *bmp, uint factsz,
           uint16_t dateMin, uint16_t dateMax, uint8_t discntMin,
           uint8_t discntMax, uint8_t qtyMin, uint8_t qtyMax, uint32_t *grp_out,
           ded_set ded = {}) {
  s1op op {
      dateMin, dateMax, dat->loOrderDate,
      discntMin, discntMax, dat->loDiscount,
      qtyMin, qtyMax, dat->loQuantity, dat->loExtendedPrice,
  };
  vprg p = {.cols = {SPARSE(date), SPARSE(discnt), SPARSE(qty)},
            .factsz = factsz,
            .instrs = {{LOAD, 0, 0}, {ANDM, 0, 1}, {ANDM, 0, 2}}};
  static constexpr size_t nr_grp = 1;
  DOWORK(a, "Sparse", p, grp_out, nr_grp, op, factsz, true);
  printf("\t%.4f", DEDICATE(ded.sparse, dat, bmp, factsz, grp_out));
  p.cols[1] = DENSE(discnt);
  p.cols[2] = DENSE(qty);
  DOWORK(a, "Medium", p, grp_out, nr_grp, op, factsz);
  printf("\t%.4f", DEDICATE(ded.medium, dat, bmp, factsz, grp_out));
  p.cols[0] = DENSE(date);
  DOWORK(a, "Dense", p, grp_out, nr_grp, op, factsz);
  printf("\t%.4f", DEDICATE(ded.dense, dat, bmp, factsz, grp_out));
  // DO NOT CALL `p.release()`! Bitmaps are owned by `bmp`, not `p`!
}

void s2fix(const char *a, const struct ssb_schema *dat, ssb_bmp *bmp, uint factsz,
           uint16_t mfgrMin, uint16_t mfgrMax, uint8_t sCityMin,
           uint8_t sCityMax, uint *grp_out, size_t nr_grp,
           ded_set ded = {}) {
  s2op op {
      dat->loPartKey, mfgrMin, mfgrMax, dat->partMfgr,
      sCityMin, sCityMax, dat->loSuppCity,
      dat->loOrderDate, dat->loRevenue,
  };
  vprg p = {.cols = {SPARSE(mfgr), SPARSE(sCity)},
            .factsz = factsz,
            .instrs = {{LOAD, 0, 0}, {ANDM, 0, 1}}};
  DOWORK(a, "Sparse", p, grp_out, nr_grp, op, factsz, true);
  printf("\t%.4f", DEDICATE(ded.sparse, dat, bmp, factsz, grp_out));
  p.cols[0] = DENSE(mfgr);
  DOWORK(a, "Medium", p, grp_out, nr_grp, op, factsz);
  printf("\t%.4f", DEDICATE(ded.medium, dat, bmp, factsz, grp_out));
  p.cols[1] = DENSE(sCity);
  DOWORK(a, "Dense", p, grp_out, nr_grp, op, factsz);
  printf("\t%.4f", DEDICATE(ded.dense, dat, bmp, factsz, grp_out));
}

void s3fix(const char *a, const struct ssb_schema *dat, ssb_bmp *bmp,
           uint factsz, uint8_t cCityMin, uint8_t cCityMax, uint8_t sCityMin,
           uint8_t sCityMax, uint16_t dateMin, uint16_t dateMax,
           uint32_t *grp_out, size_t nr_grp, ded_set ded = {}) {
  s3op op {
    dat->loCustKey, cCityMin, cCityMax, dat->custCity,
    sCityMin, sCityMax, dat->loSuppCity,
    dateMin, dateMax, dat->loOrderDate, dat->loRevenue
  };
  vprg p = {.cols = {SPARSE(cCity), SPARSE(sCity)},
            .factsz = factsz,
            .instrs = {{LOAD, 0, 0}, {ANDM, 0, 1}}};
  if (dateMin == SSBDATE_920101) {
    dateMin = dateMax, dateMax = SSBDATE_990101;
    p.instrs[2] = {LOAD, 1, 2};
    p.instrs[3] = {NOT, 1, 1};
    p.instrs[4] = {AND, 0, 1};
  } else {
    p.instrs[2] = {ANDM, 0, 2};
  }

  p.cols[2] = SPARSE(date);
  // `op` was already initialized to the original dateMin and dateMax
  DOWORK(a, "Sparse", p, grp_out, nr_grp, op, factsz, true);
  printf("\t%.4f", DEDICATE(ded.sparse, dat, bmp, factsz, grp_out));
  p.cols[0] = DENSE(cCity);
  DOWORK(a, "Medium", p, grp_out, nr_grp, op, factsz);
  printf("\t%.4f", DEDICATE(ded.medium, dat, bmp, factsz, grp_out));
  p.cols[1] = DENSE(sCity);
  p.cols[2] = DENSE(date);
  DOWORK(a, "Dense", p, grp_out, nr_grp, op, factsz);
  printf("\t%.4f", DEDICATE(ded.dense, dat, bmp, factsz, grp_out));
}

void s4fix(const char *a, const struct ssb_schema *dat, ssb_bmp *bmp,
           uint factsz, uint8_t cCityMin, uint8_t cCityMax, uint8_t sCityMin,
           uint8_t sCityMax, uint16_t mfgrMin, uint16_t mfgrMax,
           uint16_t dateMin, uint16_t dateMax, uint32_t *grp_out,
           size_t nr_grp, ded_set ded = {}) {
  s4op op{dat->loCustKey, dat->loPartKey,   cCityMin,       cCityMax,
          dat->custCity,  sCityMin,         sCityMax,       dat->loSuppCity,
          mfgrMin,        mfgrMax,          dat->partMfgr,  dateMin,
          dateMax,        dat->loOrderDate, dat->loRevenue, dat->loSupplyCost};
  vprg p = {{SPARSE(cCity), SPARSE(sCity), SPARSE(mfgr)},
            factsz,
            {{LOAD, 0, 0}, {ANDM, 0, 1}, {ANDM, 0, 2}}};
  if (dateMax < SSBDATE_990101 || dateMin > SSBDATE_920101) {
    p.cols[3] = SPARSE(date);
    p.instrs[3] = {ANDM, 0, 3};
  }
  DOWORK(a, "Sparse", p, grp_out, nr_grp, op, factsz, true);
  printf("\t%.4f", DEDICATE(ded.sparse, dat, bmp, factsz, grp_out));
  // Q4X do not have month-bounded queries
  p.cols[0] = DENSE(cCity);
  p.cols[2] = DENSE(mfgr);
  DOWORK(a, "Medium", p, grp_out, nr_grp, op, factsz);
  printf("\t%.4f", DEDICATE(ded.medium, dat, bmp, factsz, grp_out));
  p.cols[1] = DENSE(sCity);
  DOWORK(a, "Dense", p, grp_out, nr_grp, op, factsz);
  printf("\t%.4f", DEDICATE(ded.dense, dat, bmp, factsz, grp_out));
}

uint32_t *ssb_bmp_fixed(const ssb_schema *dat, ssb_bmp* bmp, size_t factSz) {
  uint32_t *res;
  cudaMalloc(&res, SUMGRP_ALL * sizeof(uint32_t) * 5);
  printf("\nCase\tBinType\tJoin\tOurs\tBasefus\tNfuStg1\tPrgStg1\tStg2\tStg3\tDedicat");
  s1fix("SSB11", dat, bmp, factSz, SSBDATE_930101, SSBDATE_940101, 1, 4, 1, 25,
        res, {.sparse = c::s11s, .medium = c::s11m, .dense = c::s11m});
  s1fix("SSB12", dat, bmp, factSz, SSBDATE_940101, SSBDATE_940201, 4, 7, 26, 36,
        res + 1, {.sparse = c::s12s, .medium = c::s12m, .dense = c::s12d});
  s1fix("SSB13", dat, bmp, factSz, SSBDATE_940204, SSBDATE_940211, 5, 8, 26, 36,
        res + 2, {.sparse = c::s13s, .medium = c::s13m, .dense = c::s13d});

  s2fix("SSB21", dat, bmp, factSz, 40, 80, 150, 200, res + SUMGRP_S1, NGRP_S21,
        {.sparse = c::s21s, .medium = c::s21m, .dense = c::s21m});
  s2fix("SSB22", dat, bmp, factSz, 260, 268, 200, 250,
        res + SUMGRP_S1 + NGRP_S21, NGRP_S22,
        {.sparse = c::s22s, .medium = c::s22m, .dense = c::s22m});
  s2fix("SSB23", dat, bmp, factSz, 260, 261, 50, 100,
        res + SUMGRP_S1 + NGRP_S21 + NGRP_S22, NGRP_S23,
        {.sparse = c::s23s, .medium = c::s23m, .dense = c::s23m});

  s3fix("SSB31", dat, bmp, factSz, 200, 250, 200, 250, SSBDATE_920101,
        SSBDATE_980101, res + SUMGRP_S2, NGRP_S31,
        {.sparse = c::s31s, .medium = c::s31s, .dense = c::s31s});
  s3fix("SSB32", dat, bmp, factSz, 190, 200, 190, 200, SSBDATE_920101,
        SSBDATE_980101, res + SUMGRP_S2 + NGRP_S31, NGRP_S32,
        {.sparse = c::s32s, .medium = c::s32m, .dense = c::s32d});
  s3fix("SSB33", dat, bmp, factSz, 51, 55, 51, 55, SSBDATE_920101,
        SSBDATE_980101, res + SUMGRP_S2 + NGRP_S31 + NGRP_S32, NGRP_S33,
        {.sparse = c::s33s, .medium = c::s33m, .dense = c::s33d});
  s3fix("SSB34", dat, bmp, factSz, 51, 55, 51, 55, SSBDATE_971201,
        SSBDATE_980101, res + SUMGRP_S2 + NGRP_S31 + NGRP_S32 + NGRP_S33, NGRP_S34,
        {.sparse = c::s34s, .medium = c::s34m, .dense = c::s34d});

  s4fix("SSB41", dat, bmp, factSz, 150, 200, 150, 200, 0, 400, SSBDATE_920101,
        SSBDATE_990101, res + SUMGRP_S3, NGRP_S41,
        {.sparse = c::s41s, .medium = c::s41s, .dense = c::s41s});
  s4fix("SSB42", dat, bmp, factSz, 150, 200, 150, 200, 0, 400, SSBDATE_970101,
        SSBDATE_990101, res + SUMGRP_S3 + NGRP_S41, NGRP_S42,
        {.sparse = c::s42s, .medium = c::s42s, .dense = c::s42s});
  s4fix("SSB43", dat, bmp, factSz, 150, 200, 190, 200, 120, 160, SSBDATE_970101,
        SSBDATE_990101, res + SUMGRP_S3 + NGRP_S41 + NGRP_S42, NGRP_S43,
        {.sparse = c::s43s, .medium = c::s43m, .dense = c::s43d});
  return res;
}
