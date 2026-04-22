#include "lambdas.cuh"
using namespace mybmpidx;
static constexpr auto nodim = 0xffffffff;
static constexpr auto ANDM = vprg::ANDM, LOAD = vprg::ORM, END = vprg::END;

// Helper type to create a bitmap index for a specific query
struct recipe {
  uint factsz; // size of fact table
  // arguments to create_join or select (if corresponding foreign key is NULL)
  uint nbit[MAXCOLS], min[MAXCOLS], max[MAXCOLS], dimsz[MAXCOLS];
  uint32_t* fk[MAXCOLS];
  void* attr[MAXCOLS];

  // Perfect: all bitmaps index bin boundary "perfectly align" with query. If
  // the query is 3<=attr1<11 AND 4<=attr2<13, then we create the 2 exact bitmaps:
  // col_bmps[0].middle[0] = create_select(..., 3, 11)
  // col_bmps[1].middle[0] = create_select(..., 4, 13)
  // all other fields are NULL.
  vprg perfect(const vprg::instr instrs[MAXNINSTR]) const;

  // aligned: the query includes 3 bins, but the lower bound of
  // leftmost bin and upper bound of rightmost bin align with the query. If max
  // - min is not divisible by 3 then remainder goes to final bin. If max - min
  // < 3, then fall back to "perfect" creation.
  // Ex: query 3<=attr1<11, bins are {3,4}, {5,6}, {7,8,9,10}.
  vprg aligned(const vprg::instr instrs[MAXNINSTR]) const;

  // misaligned: bin boundaries must be a multiple of the given interval. Place
  // the bin into leftmost or rightmost if the bin is not fully included in the
  // bin. For now if >MAXBIN_PERCOL middle bins, only keep the first MAXBIN_PERCOL.
  vprg misaligned(const vprg::instr instrs[MAXNINSTR]) const;
};

vprg recipe::perfect(const vprg::instr *instrs) const {
  vprg ret;
  ret.factsz = factsz;
  memcpy(ret.instrs, instrs, sizeof(vprg::instr) * MAXNINSTR);
  ret.factsz = factsz;
  for (size_t i = 0; i < MAXCOLS; ++i) {
    if (attr[i] == NULL) continue;
    ret.cols[i].middle[0] =
        create_join(fk[i], attr[i], nbit[i], factsz, dimsz[i], min[i], max[i]);
  }
  return ret;
}

vprg recipe::aligned(const vprg::instr *instrs) const {
  vprg ret;
  ret.factsz = factsz;
  memcpy(ret.instrs, instrs, sizeof(vprg::instr) * MAXNINSTR);
  ret.factsz = factsz;
  for (size_t i = 0; i < MAXCOLS; ++i) {
    if (attr[i] == NULL) continue;
    uint step = (max[i] - min[i]) / 3, **middle = ret.cols[i].middle;
    if (step == 0) {
      middle[0] = create_join(fk[i], attr[i], nbit[i], factsz, dimsz[i], min[i], max[i]);
      continue;
    }

    middle[0] = create_join(fk[i], attr[i], nbit[i], factsz, dimsz[i], min[i],
                            min[i] + step);
    middle[1] = create_join(fk[i], attr[i], nbit[i], factsz, dimsz[i],
                            min[i] + step, min[i] + 2 * step);
    middle[2] = create_join(fk[i], attr[i], nbit[i], factsz, dimsz[i],
                            min[i] + 2 * step, max[i]);
  }
  return ret;
}

vprg recipe::misaligned(const vprg::instr *instrs) const {
  vprg ret;
  ret.factsz = factsz;
  memcpy(ret.instrs, instrs, sizeof(vprg::instr) * MAXNINSTR);
  // column 0 - left bin left range -= step / 2
  if (attr[0] == nullptr) return ret;
  uint step = (max[0] - min[0]) / 3, **middle = ret.cols[0].middle;
  if (step == 0)
    ret.cols[0].leftmost = create_join(fk[0], attr[0], nbit[0], factsz,
                                           dimsz[0], min[0], max[0] + 1);
  else {
    auto l = std::max(min[0], step / 2) - step / 2;
    ret.cols[0].leftmost =
      create_join(fk[0], attr[0], nbit[0], factsz, dimsz[0], l, min[0] + step);
    middle[0] = create_join(fk[0], attr[0], nbit[0], factsz, dimsz[0],
                            min[0] + step, min[0] + 2 * step);
    middle[1] = create_join(fk[0], attr[0], nbit[0], factsz, dimsz[0],
                            min[0] + 2 * step, max[0]);
  }

  // column 1 - right bin right range += step / 2
  if (attr[1] == nullptr) return ret;
  step = (max[1] - min[1]) / 3, middle = ret.cols[1].middle;
  if (step == 0)
    ret.cols[1].rightmost = create_join(fk[1], attr[1], nbit[1], factsz,
                                            dimsz[1], min[1] - 1, max[1]);
  else {
    ret.cols[1].rightmost =
      create_join(fk[1], attr[1], nbit[1], factsz, dimsz[1], min[1] + 2 * step,
                  max[1] + step / 2);
    middle[0] = create_join(fk[1], attr[1], nbit[1], factsz, dimsz[1],
                            min[1], min[1] + 1 * step);
    middle[1] = create_join(fk[1], attr[1], nbit[1], factsz, dimsz[1],
                            min[1] + step, min[1] + 2 * step);
  }

  // rest columns same as many or
  for (size_t i = 2; i < MAXCOLS; ++i) {
    if (attr[i] == NULL) return ret;
    uint step = (max[i] - min[i]) / 3, **middle = ret.cols[i].middle;
    if (step == 0) {
      middle[0] = create_join(fk[i], attr[i], nbit[i], factsz, dimsz[i], min[i], max[i]);
      continue;
    }
    middle[0] = create_join(fk[i], attr[i], nbit[i], factsz, dimsz[i], min[i],
                            min[i] + step);
    middle[1] = create_join(fk[i], attr[i], nbit[i], factsz, dimsz[i],
                            min[i] + step, min[i] + 2 * step);
    middle[2] = create_join(fk[i], attr[i], nbit[i], factsz, dimsz[i],
                            min[i] + 2 * step, max[i]);
  }
  return ret;
}

template <typename op_t>
void MOREWORK(const char *a, recipe &r, uint32_t *grp_out, size_t nr_grp,
              op_t &op, uint factsz, const vprg::instr *instrs) {
  vprg p = r.perfect(instrs);
  DOWORK(a, "Perfect", p, grp_out, nr_grp, op, factsz);
  p.release();
  p = r.aligned(instrs);
  DOWORK(a, "AllAlig", p, grp_out, nr_grp, op, factsz);
  p.release();
  p = r.misaligned(instrs);
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
