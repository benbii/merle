#include "lambdas.cuh"
#include "../primitive_abla.cuh"
using namespace mybmpidx;
using namespace mybmpidx::abla;
using cuda::ceil_div;
static constexpr auto nodim = recipe::nodim;
static constexpr auto AND = vprg::AND, OR = vprg::OR, ANDM = vprg::ANDM,
                      LOAD = vprg::ORM, END = vprg::END;

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
  cudaEvent_t start, stop; float msec; uint *idxbuf; \
  cudaMalloc(&idxbuf, sizeof(uint) * 6 * ceil_div(factsz, 32)); \
  cudaEventCreate(&start); cudaEventCreate(&stop);\
  for (size_t i = 0; i < 6; ++i) { \
    vprg p; \
    switch (i) { \
    case 0: p = r.perfect(igood); break; \
    case 1: p = r.many_or(igood); break; \
    case 2: p = r.candchk(igood); break; \
    case 3: p = r.perfect(ibad); break; \
    case 4: p = r.many_or(ibad); break; \
    case 5: p = r.candchk(ibad); break; \
    }\
    ands a; memcpy(&a, &p, sizeof(ands)); \
    cudaEventRecord(start); \
    for (size_t d = 0; d < ndup; ++d) { \
      switch (i) { \
      case 0: case 1: \
        noprg_abla<nt, vt, vt0, false><<<ceil_div(factsz, nv32), nt>>>( \
          a, grp_out, nr_grp, op); break; \
      case 2: \
        noprg_abla<nt, vt, vt0, true><<<ceil_div(factsz, nv32), nt>>>( \
          a, grp_out, nr_grp, op); break; \
      case 3: case 4: \
        cudaMemset(idxbuf, 0, sizeof(uint) * 3 * ceil_div(factsz, 32)); \
        base_fuse<nt, vt, false><<<ceil_div(factsz, nv32), nt>>>( \
          p, grp_out, nr_grp, op, idxbuf); break; \
      case 5: \
        cudaMemset(grp_out, 0, sizeof(uint) * nr_grp); \
        cudaMemset(idxbuf, 0, sizeof(uint) * 6 * ceil_div(factsz, 32)); \
        base_fuse<nt, vt, true><<<ceil_div(factsz, nv32), nt>>>( \
          p, grp_out, nr_grp, op, idxbuf); break; \
      } \
    } \
    cudaEventRecord(stop); cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&msec, start, stop); \
    printf("\t%.4f", msec / ndup); \
    p.release(); \
  } cudaFree(idxbuf); cudaEventDestroy(start); cudaEventDestroy(stop);  /* \
  for (size_t i = 0; i < 3; ++i) { \
    vprg p = i == 0 ? r.perfect(igood) \
                    : (i == 1 ? r.many_or(igood) : r.candchk(igood)); \
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
  } */

void s1fus(const struct ssb_schema *dat, uint factsz, uint16_t dateMin,
           uint16_t dateMax, uint8_t discntMin, uint8_t discntMax,
           uint8_t qtyMin, uint8_t qtyMax, uint32_t *grp_out) {
  s1op op {
      dateMin, dateMax, dat->loOrderDate,
      discntMin, discntMax, dat->loDiscount,
      qtyMin, qtyMax, dat->loQuantity, dat->loExtendedPrice,
  };
  recipe r = {.factsz = factsz, .nbit = {16, 8, 8},
              .min = {dateMin, discntMin, qtyMin},
              .max = {dateMax, discntMax, qtyMax},
              .dimsz = {nodim, nodim, nodim},
              .fk = {nullptr, nullptr, nullptr},
              .attr = {dat->loOrderDate, dat->loDiscount, dat->loQuantity}};
  constexpr size_t nr_grp = 1;
  vprg::instr igood[MAXNINSTR] = {{LOAD, 0, 0}, {ANDM, 0, 1}, {ANDM, 0, 2}};
  vprg::instr ibad[MAXNINSTR] = {
      {LOAD, 0, 0}, {LOAD, 1, 1}, {AND, 0, 1}, {LOAD, 2, 2}, {AND, 0, 2}};
  PLEASE
}

// We need nr_groups for CUDA kernels
void s2fus(const struct ssb_schema *dat, uint factsz, uint16_t pMfgrMin,
           uint16_t pMfgrMax, uint8_t sCityMin, uint8_t sCityMax,
           uint *grp_out, size_t nr_grp) {
  s2op op {
      dat->loPartKey, pMfgrMin, pMfgrMax, dat->partMfgr,
      sCityMin, sCityMax, dat->loSuppCity,
      dat->loOrderDate, dat->loRevenue,
  };
  recipe r = {.factsz = factsz, .nbit = {16, 8},
              .min = {pMfgrMin, sCityMin},
              .max = {pMfgrMax, sCityMax},
              // No dimension bound check; ssb guarantees in bound
              .dimsz = {nodim, nodim},
              .fk = {dat->loPartKey, nullptr},
              .attr = {dat->partMfgr, dat->loSuppCity}};
  vprg::instr igood[MAXNINSTR] = {{LOAD, 0, 0}, {ANDM, 0, 1}};
  vprg::instr ibad[MAXNINSTR] = {{LOAD, 0, 0}, {LOAD, 1, 1}, {AND, 0, 1}};
  PLEASE
}

void s3fus(const struct ssb_schema *dat, uint factsz, uint8_t cCityMin,
           uint8_t cCityMax, uint8_t sCityMin, uint8_t sCityMax,
           uint16_t dateMin, uint16_t dateMax, uint32_t *grp_out, size_t nr_grp) {
  s3op op {
    dat->loCustKey, cCityMin, cCityMax, dat->custCity,
    sCityMin, sCityMax, dat->loSuppCity,
    dateMin, dateMax, dat->loOrderDate, dat->loRevenue
  };
  recipe r = {.factsz = factsz, .nbit = {8, 8, 16},
              .min = {cCityMin, sCityMin, dateMin},
              .max = {cCityMax, sCityMax, dateMax},
              .dimsz = {nodim, nodim, nodim},
              .fk = {dat->loCustKey, nullptr, nullptr},
              .attr = {dat->custCity, dat->loSuppCity, dat->loOrderDate}};
  vprg::instr igood[MAXNINSTR] = {{LOAD, 0, 0}, {ANDM, 0, 1}, {ANDM, 0, 2}};
  vprg::instr ibad[MAXNINSTR] = {
      {LOAD, 0, 0}, {LOAD, 1, 1}, {AND, 0, 1}, {LOAD, 2, 2}, {AND, 0, 2}};
  PLEASE
}

void s4fus(const struct ssb_schema *dat, uint factsz, uint8_t cCityMin,
           uint8_t cCityMax, uint8_t sCityMin, uint8_t sCityMax,
           uint16_t pMfgrMin, uint16_t pMfgrMax, uint16_t dateMin,
           uint16_t dateMax, uint32_t *grp_out, size_t nr_grp) {
  s4op op{dat->loCustKey, dat->loPartKey,   cCityMin,       cCityMax,
          dat->custCity,  sCityMin,         sCityMax,       dat->loSuppCity,
          pMfgrMin,       pMfgrMax,         dat->partMfgr,  dateMin,
          dateMax,        dat->loOrderDate, dat->loRevenue, dat->loSupplyCost};
  recipe r = {.factsz = factsz, .nbit = {8, 8, 16, 16},
              .min = {cCityMin, sCityMin, pMfgrMin, dateMin},
              .max = {cCityMax, sCityMax, pMfgrMax, dateMax},
              .dimsz = {nodim, nodim, nodim, nodim},
              .fk = {dat->loCustKey, nullptr, dat->loPartKey, nullptr},
              .attr = {dat->custCity, dat->loSuppCity, dat->partMfgr, dat->loOrderDate}};
  vprg::instr igood[MAXNINSTR] = {{LOAD, 0, 0}, {ANDM, 0, 1}, {ANDM, 0, 2}, {ANDM, 0, 3}};
  vprg::instr ibad[MAXNINSTR] = {{LOAD, 0, 0}, {LOAD, 1, 1}, {AND, 0, 1},
                                 {LOAD, 2, 2}, {AND, 0, 2}, {ANDM, 0, 3}};
  if (dateMin <= SSBDATE_920101 && dateMax >= SSBDATE_990101) {
    r.attr[3] = nullptr;
    igood[3].opcode = vprg::END;
    ibad[5].opcode = vprg::END;
  }
  PLEASE
}

uint32_t *ssb_bmp_control_abl_fuse(const struct ssb_schema *dat, size_t factSz) {
  uint32_t *res;
  cudaMalloc(&res, (SUMGRP_ALL + 1) * sizeof(uint32_t));
  // printf("\nCase\tPftNprg\tOrsNprg\tChkNprg\tPftStg1\tPftStg2\tPftStg3"
  //        "\tOrsStg1\tOrsStg2\tOrsStg3\tChkStg1\tChkStg2\tChkStg3\nSSB11");
  printf("\nCase\tPftNprg\tOrsNprg\tChkNprg\tPftBfus\tOrsBfus\tChkBfus\nSSB11");
  s1fus(dat, factSz, SSBDATE_930101, SSBDATE_940101, 1, 4, 1, 25, res);
  printf("\nSSB12");
  s1fus(dat, factSz, SSBDATE_940101, SSBDATE_940201, 4, 7, 26, 36, res + 1);
  printf("\nSSB13");
  s1fus(dat, factSz, SSBDATE_940204, SSBDATE_940211, 5, 8, 26, 36, res + 2);

  printf("\nSSB21");
  s2fus(dat, factSz, 40, 80, 150, 200, res + SUMGRP_S1, NGRP_S21);
  printf("\nSSB22");
  s2fus(dat, factSz, 260, 268, 200, 250, res + SUMGRP_S1 + NGRP_S21, NGRP_S22);
  printf("\nSSB23");
  s2fus(dat, factSz, 260, 261, 50, 100, res + SUMGRP_S1 + NGRP_S21 + NGRP_S22, NGRP_S23);

  printf("\nSSB31");
  s3fus(dat, factSz, 200, 250, 200, 250, SSBDATE_920101, SSBDATE_980101,
        res + SUMGRP_S2, NGRP_S31);
  printf("\nSSB32");
  s3fus(dat, factSz, 190, 200, 190, 200, SSBDATE_920101, SSBDATE_980101,
        res + SUMGRP_S2 + NGRP_S31, NGRP_S32);
  printf("\nSSB33");
  s3fus(dat, factSz, 51, 55, 51, 55, SSBDATE_920101, SSBDATE_980101,
        res + SUMGRP_S2 + NGRP_S31 + NGRP_S32, NGRP_S33);
  printf("\nSSB34");
  s3fus(dat, factSz, 51, 55, 51, 55, SSBDATE_971201, SSBDATE_980101,
        res + SUMGRP_S2 + NGRP_S31 + NGRP_S32 + NGRP_S33, NGRP_S34);

  printf("\nSSB41");
  s4fus(dat, factSz, 150, 200, 150, 200, 0, 400, SSBDATE_920101, SSBDATE_990101,
        res + SUMGRP_S3, NGRP_S41);
  printf("\nSSB42");
  s4fus(dat, factSz, 150, 200, 150, 200, 0, 400, SSBDATE_970101, SSBDATE_990101,
        res + SUMGRP_S3 + NGRP_S41, NGRP_S42);
  printf("\nSSB43");
  s4fus(dat, factSz, 150, 200, 190, 200, 120, 160, SSBDATE_970101, SSBDATE_990101,
        res + SUMGRP_S3 + NGRP_S41 + NGRP_S42, NGRP_S43);
  return res;
}
