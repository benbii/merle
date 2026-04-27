#include "lambdas.cuh"
using namespace mybmpidx;
static constexpr auto ANDM = vprg::ANDM, LOAD = vprg::ORM, END = vprg::END;
constexpr auto nodim = recipe::nodim;

template <typename op_t>
void MOREWORK(const char *a, recipe &r, uint32_t *grp_out, size_t nr_grp,
              op_t &op, uint factsz, const vprg::instr *instrs) {
  vprg p = r.perfect(instrs);
  DOWORK(a, "Perfect", p, grp_out, nr_grp, op, factsz);
  p.release();
  p = r.many_or(instrs);
  DOWORK(a, "AllAlig", p, grp_out, nr_grp, op, factsz);
  p.release();
  p = r.candchk(instrs);
  DOWORK(a, "MisAlig", p, grp_out, nr_grp, op, factsz);
  p.release();
}

void s1bmp(const char *a, const struct ssb_schema *dat, uint factsz,
           uint16_t dateMin, uint16_t dateMax, uint8_t discntMin,
           uint8_t discntMax, uint8_t qtyMin, uint8_t qtyMax, uint32_t *grp_out) {
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
  vprg::instr instrs[MAXNINSTR] = {{LOAD, 0, 0}, {ANDM, 0, 1}, {ANDM, 0, 2}};
  MOREWORK(a, r, grp_out, nr_grp, op, factsz, instrs);
}

// We need nr_groups for CUDA kernels
void s2bmp(const char *a, const struct ssb_schema *dat, uint factsz,
           uint16_t pMfgrMin, uint16_t pMfgrMax, uint8_t sCityMin,
           uint8_t sCityMax, uint *grp_out, size_t nr_grp) {
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
  vprg::instr instrs[MAXNINSTR] = {{LOAD, 0, 0}, {ANDM, 0, 1}};
  MOREWORK(a, r, grp_out, nr_grp, op, factsz, instrs);
}

void s3bmp(const char *a, const struct ssb_schema *dat, uint factsz,
           uint8_t cCityMin, uint8_t cCityMax, uint8_t sCityMin,
           uint8_t sCityMax, uint16_t dateMin, uint16_t dateMax,
           uint32_t *grp_out, size_t nr_grp) {
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
  vprg::instr instrs[MAXNINSTR] = {{LOAD, 0, 0}, {ANDM, 0, 1}, {ANDM, 0, 2}};
  MOREWORK(a, r, grp_out, nr_grp, op, factsz, instrs);
}

void s4bmp(const char *a, const struct ssb_schema *dat, uint factsz,
           uint8_t cCityMin, uint8_t cCityMax, uint8_t sCityMin,
           uint8_t sCityMax, uint16_t pMfgrMin, uint16_t pMfgrMax,
           uint16_t dateMin, uint16_t dateMax, uint32_t *grp_out, size_t nr_grp) {
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
  vprg::instr instrs[MAXNINSTR] = {
      {LOAD, 0, 0}, {ANDM, 0, 1}, {ANDM, 0, 2}, {ANDM, 0, 3}};
  if (dateMin <= SSBDATE_920101 && dateMax >= SSBDATE_990101)
    r.attr[3] = nullptr, instrs[3].opcode = END;
  MOREWORK(a, r, grp_out, nr_grp, op, factsz, instrs);
}

uint32_t *ssb_bmp_control(const struct ssb_schema *dat, size_t factSz) {
  uint32_t *res;
  cudaMalloc(&res, SUMGRP_ALL * 4 * sizeof(uint32_t));

  s1bmp("SSB11", dat, factSz, SSBDATE_930101, SSBDATE_940101, 1, 4, 1, 25, res);
  s1bmp("SSB12", dat, factSz, SSBDATE_940101, SSBDATE_940201, 4, 7, 26, 36, res + 1);
  s1bmp("SSB13", dat, factSz, SSBDATE_940204, SSBDATE_940211, 5, 8, 26, 36, res + 2);

  s2bmp("SSB21", dat, factSz, 40, 80, 150, 200, res + SUMGRP_S1, NGRP_S21);
  s2bmp("SSB22", dat, factSz, 260, 268, 200, 250, res + SUMGRP_S1 + NGRP_S21,
        NGRP_S22);
  s2bmp("SSB23", dat, factSz, 260, 261, 50, 100,
        res + SUMGRP_S1 + NGRP_S21 + NGRP_S22, NGRP_S23);

  s3bmp("SSB31", dat, factSz, 200, 250, 200, 250, SSBDATE_920101, SSBDATE_980101,
        res + SUMGRP_S2, NGRP_S31);
  s3bmp("SSB32", dat, factSz, 190, 200, 190, 200, SSBDATE_920101, SSBDATE_980101,
        res + SUMGRP_S2 + NGRP_S31, NGRP_S32);
  s3bmp("SSB33", dat, factSz, 51, 55, 51, 55, SSBDATE_920101, SSBDATE_980101,
        res + SUMGRP_S2 + NGRP_S31 + NGRP_S32, NGRP_S33);
  s3bmp("SSB34", dat, factSz, 51, 55, 51, 55, SSBDATE_971201, SSBDATE_980101,
        res + SUMGRP_S2 + NGRP_S31 + NGRP_S32 + NGRP_S33, NGRP_S34);

  s4bmp("SSB41", dat, factSz, 150, 200, 150, 200, 0, 400, SSBDATE_920101,
        SSBDATE_990101, res + SUMGRP_S3, NGRP_S41);
  s4bmp("SSB42", dat, factSz, 150, 200, 150, 200, 0, 400, SSBDATE_970101,
        SSBDATE_990101, res + SUMGRP_S3 + NGRP_S41, NGRP_S42);
  s4bmp("SSB43", dat, factSz, 150, 200, 190, 200, 120, 160, SSBDATE_970101,
        SSBDATE_990101, res + SUMGRP_S3 + NGRP_S41 + NGRP_S42, NGRP_S43);
  return res;
}
