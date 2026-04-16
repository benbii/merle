#include "lambdas.cuh"
using namespace mybmpidx;
using cuda::ceil_div;
static constexpr auto ANDM = vprg::ANDM, LOAD = vprg::ORM, AND = vprg::AND, NOT = vprg::NOT;
static constexpr const uint64_t *cCitySpBin = sCitySpBin, *cCityDeBin = sCityDeBin;

#define SPARSE(c)                                                              \
  col::from_bin(sizeof(bmp->c##Sparse) / sizeof(intptr_t), bmp->c##Sparse,     \
                c##SpBin, c##Min, c##Max)
#define DENSE(c)                                                               \
  col::from_bin2(sizeof(bmp->c##Sparse) / sizeof(intptr_t), bmp->c##Sparse,    \
                 c##SpBin, sizeof(bmp->c##Dense) / sizeof(intptr_t),           \
                 bmp->c##Dense, c##DeBin, c##Min, c##Max)

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

template<typename op_t> static void
DOWORK(vprg &p, uint32_t *grp_out, size_t nr_grp, op_t &op, uint factsz) {
  cudaEvent_t start, stop; float msec;
  cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start);
  for (size_t d = 0; d < ndup; ++d) {
    cudaMemset(grp_out, 0, nr_grp * sizeof(uint));
    if (p.is_nochk())
      vprg_grpby<nt, vt, vt0, false>
          <<<ceil_div(factsz, nv32), nt>>>(p, grp_out, nr_grp, op);
    else
      vprg_grpby<nt, cp_vt, vt0, true>
          <<<ceil_div(factsz, cp_nv32), nt>>>(p, grp_out, nr_grp, op);
  }
  cudaEventRecord(stop); cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msec, start, stop);
  printf("\t%.4f", msec / ndup);
  cudaEventDestroy(start); cudaEventDestroy(stop);
}

void s1fix(const ssb_schema *dat, ssb_bmp *bmp, uint factsz, uint16_t dateMin,
           uint16_t dateMax, uint8_t discntMin, uint8_t discntMax,
           uint8_t qtyMin, uint8_t qtyMax, uint32_t *grp_out) {
  s1op op {
      dateMin, dateMax, dat->loOrderDate,
      discntMin, discntMax, dat->loDiscount,
      qtyMin, qtyMax, dat->loQuantity, dat->loExtendedPrice,
  };
  vprg p = {.cols = {SPARSE(date), SPARSE(discnt), SPARSE(qty)},
            .factsz = factsz,
            .instrs = {{LOAD, 0, 0}, {ANDM, 0, 1}, {ANDM, 0, 2}}};
  static constexpr size_t nr_grp = 1;
  DOWORK(p, grp_out, nr_grp, op, factsz); // sparse
  p.cols[1] = DENSE(discnt);
  p.cols[2] = DENSE(qty);
  DOWORK(p, grp_out, nr_grp, op, factsz); // balanced
  p.cols[0] = DENSE(date);
  DOWORK(p, grp_out, nr_grp, op, factsz); // dense
  // DO NOT CALL `p.release()`! Bitmaps are owned by `bmp`, not `p`!
}

void s2fix(const struct ssb_schema *dat, ssb_bmp *bmp, uint factsz,
           uint16_t mfgrMin, uint16_t mfgrMax, uint8_t sCityMin,
           uint8_t sCityMax, uint *grp_out, size_t nr_grp) {
  s2op op {
      dat->loPartKey, mfgrMin, mfgrMax, dat->partMfgr,
      sCityMin, sCityMax, dat->loSuppCity,
      dat->loOrderDate, dat->loRevenue,
  };
  vprg p = {.cols = {SPARSE(mfgr), SPARSE(sCity)},
            .factsz = factsz,
            .instrs = {{LOAD, 0, 0}, {ANDM, 0, 1}}};
  DOWORK(p, grp_out, nr_grp, op, factsz);
  p.cols[0] = DENSE(mfgr);
  DOWORK(p, grp_out, nr_grp, op, factsz);
  p.cols[1] = DENSE(sCity);
  DOWORK(p, grp_out, nr_grp, op, factsz);
}

void s3fix(const struct ssb_schema *dat, ssb_bmp *bmp, uint factsz,
           uint8_t cCityMin, uint8_t cCityMax, uint8_t sCityMin,
           uint8_t sCityMax, uint16_t dateMin, uint16_t dateMax,
           uint32_t *grp_out, size_t nr_grp) {
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
  DOWORK(p, grp_out, nr_grp, op, factsz);
  p.cols[0] = DENSE(cCity);
  DOWORK(p, grp_out, nr_grp, op, factsz);
  p.cols[1] = DENSE(sCity);
  p.cols[2] = DENSE(date);
  DOWORK(p, grp_out, nr_grp, op, factsz);
}

void s4fix(const struct ssb_schema *dat, ssb_bmp *bmp, uint factsz,
           uint8_t cCityMin, uint8_t cCityMax, uint8_t sCityMin,
           uint8_t sCityMax, uint16_t mfgrMin, uint16_t mfgrMax,
           uint16_t dateMin, uint16_t dateMax, uint32_t *grp_out, size_t nr_grp) {
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
  DOWORK(p, grp_out, nr_grp, op, factsz);
  // Q4X do not have month-bounded queries
  p.cols[0] = DENSE(cCity);
  p.cols[2] = DENSE(mfgr);
  DOWORK(p, grp_out, nr_grp, op, factsz);
  p.cols[1] = DENSE(sCity);
  DOWORK(p, grp_out, nr_grp, op, factsz);
}

uint32_t *ssb_bmp_fixed(const ssb_schema *dat, ssb_bmp* bmp, size_t factSz) {
  uint32_t *res;
  cudaMalloc(&res, (SUMGRP_ALL + 1) * sizeof(uint32_t));
  printf("\nCase\tSparse\tMedium\tDense\nSSB11");
  s1fix(dat, bmp, factSz, SSBDATE_930101, SSBDATE_940101, 1, 4, 1, 25, res);
  printf("\nSSB12");
  s1fix(dat, bmp, factSz, SSBDATE_940101, SSBDATE_940201, 4, 7, 26, 36, res + 1);
  printf("\nSSB13");
  s1fix(dat, bmp, factSz, SSBDATE_940204, SSBDATE_940211, 5, 8, 26, 36, res + 2);

  printf("\nSSB21");
  s2fix(dat, bmp, factSz, 40, 80, 150, 200, res + SUMGRP_S1, NGRP_S21);
  printf("\nSSB22");
  s2fix(dat, bmp, factSz, 260, 268, 200, 250, res + SUMGRP_S1 + NGRP_S21, NGRP_S22);
  printf("\nSSB23");
  s2fix(dat, bmp, factSz, 260, 261, 50, 100, res + SUMGRP_S1 + NGRP_S21 + NGRP_S22, NGRP_S23);

  printf("\nSSB31");
  s3fix(dat, bmp, factSz, 200, 250, 200, 250, SSBDATE_920101, SSBDATE_980101,
        res + SUMGRP_S2, NGRP_S31);
  printf("\nSSB32");
  s3fix(dat, bmp, factSz, 190, 200, 190, 200, SSBDATE_920101, SSBDATE_980101,
        res + SUMGRP_S2 + NGRP_S31, NGRP_S32);
  printf("\nSSB33");
  s3fix(dat, bmp, factSz, 51, 55, 51, 55, SSBDATE_920101, SSBDATE_980101,
        res + SUMGRP_S2 + NGRP_S31 + NGRP_S32, NGRP_S33);
  printf("\nSSB34");
  s3fix(dat, bmp, factSz, 51, 55, 51, 55, SSBDATE_971201, SSBDATE_980101,
        res + SUMGRP_S2 + NGRP_S31 + NGRP_S32 + NGRP_S33, NGRP_S34);

  printf("\nSSB41");
  s4fix(dat, bmp, factSz, 150, 200, 150, 200, 0, 400, SSBDATE_920101,
        SSBDATE_990101, res + SUMGRP_S3, NGRP_S41);
  printf("\nSSB42");
  s4fix(dat, bmp, factSz, 150, 200, 150, 200, 0, 400, SSBDATE_970101,
        SSBDATE_990101, res + SUMGRP_S3 + NGRP_S41, NGRP_S42);
  printf("\nSSB43");
  s4fix(dat, bmp, factSz, 150, 200, 190, 200, 120, 160, SSBDATE_970101,
        SSBDATE_990101, res + SUMGRP_S3 + NGRP_S41 + NGRP_S42, NGRP_S43);
  return res;
}
