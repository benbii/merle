#include "lambdas.cuh"
#include "../primitive_abla.cuh"
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

template <typename op_t>
static void DOWORK(const char *a, const char *b, vprg &p, uint32_t *grp_out,
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

void s1fix(const char *a, const ssb_schema *dat, ssb_bmp *bmp, uint factsz,
           uint16_t dateMin, uint16_t dateMax, uint8_t discntMin,
           uint8_t discntMax, uint8_t qtyMin, uint8_t qtyMax, uint32_t *grp_out) {
  s1op op {
      dateMin, dateMax, dat->loOrderDate,
      discntMin, discntMax, dat->loDiscount,
      qtyMin, qtyMax, dat->loQuantity, dat->loExtendedPrice,
  };
  vprg p = {.cols = {SPARSE(date), SPARSE(discnt), SPARSE(qty)},
            .factsz = factsz,
            .instrs = {{LOAD, 0, 0}, {ANDM, 0, 1}, {ANDM, 0, 2}}};
  static constexpr size_t nr_grp = 1;
  DOWORK(a, "Sparse", p, grp_out, nr_grp, op, factsz, true); // sparse
  p.cols[1] = DENSE(discnt);
  p.cols[2] = DENSE(qty);
  DOWORK(a, "Medium", p, grp_out, nr_grp, op, factsz); // balanced
  p.cols[0] = DENSE(date);
  DOWORK(a, "Dense", p, grp_out, nr_grp, op, factsz); // dense
  // DO NOT CALL `p.release()`! Bitmaps are owned by `bmp`, not `p`!
}

void s2fix(const char *a, const struct ssb_schema *dat, ssb_bmp *bmp, uint factsz,
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
  DOWORK(a, "Sparse", p, grp_out, nr_grp, op, factsz, true);
  p.cols[0] = DENSE(mfgr);
  DOWORK(a, "Medium", p, grp_out, nr_grp, op, factsz);
  p.cols[1] = DENSE(sCity);
  DOWORK(a, "Dense", p, grp_out, nr_grp, op, factsz);
}

void s3fix(const char *a, const struct ssb_schema *dat, ssb_bmp *bmp,
           uint factsz, uint8_t cCityMin, uint8_t cCityMax, uint8_t sCityMin,
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
  DOWORK(a, "Sparse", p, grp_out, nr_grp, op, factsz, true);
  p.cols[0] = DENSE(cCity);
  DOWORK(a, "Medium", p, grp_out, nr_grp, op, factsz);
  p.cols[1] = DENSE(sCity);
  p.cols[2] = DENSE(date);
  DOWORK(a, "Dense", p, grp_out, nr_grp, op, factsz);
}

void s4fix(const char *a, const struct ssb_schema *dat, ssb_bmp *bmp,
           uint factsz, uint8_t cCityMin, uint8_t cCityMax, uint8_t sCityMin,
           uint8_t sCityMax, uint16_t mfgrMin, uint16_t mfgrMax,
           uint16_t dateMin, uint16_t dateMax, uint32_t *grp_out,
           size_t nr_grp) {
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
  // Q4X do not have month-bounded queries
  p.cols[0] = DENSE(cCity);
  p.cols[2] = DENSE(mfgr);
  DOWORK(a, "Medium", p, grp_out, nr_grp, op, factsz);
  p.cols[1] = DENSE(sCity);
  DOWORK(a, "Dense", p, grp_out, nr_grp, op, factsz);
}

uint32_t *ssb_bmp_fixed(const ssb_schema *dat, ssb_bmp* bmp, size_t factSz) {
  uint32_t *res;
  cudaMalloc(&res, SUMGRP_ALL * sizeof(uint32_t) * 4);
  printf("\nCase\tBinType\tJoin\tOurs\tBasefus\tNfuStg1\tPrgStg1\tStg2\tStg3");
  s1fix("SSB11", dat, bmp, factSz, SSBDATE_930101, SSBDATE_940101, 1, 4, 1, 25,
        res);
  s1fix("SSB12", dat, bmp, factSz, SSBDATE_940101, SSBDATE_940201, 4, 7, 26, 36,
        res + 1);
  s1fix("SSB13", dat, bmp, factSz, SSBDATE_940204, SSBDATE_940211, 5, 8, 26, 36,
        res + 2);

  s2fix("SSB21", dat, bmp, factSz, 40, 80, 150, 200, res + SUMGRP_S1, NGRP_S21);
  s2fix("SSB22", dat, bmp, factSz, 260, 268, 200, 250,
        res + SUMGRP_S1 + NGRP_S21, NGRP_S22);
  s2fix("SSB23", dat, bmp, factSz, 260, 261, 50, 100,
        res + SUMGRP_S1 + NGRP_S21 + NGRP_S22, NGRP_S23);

  s3fix("SSB31", dat, bmp, factSz, 200, 250, 200, 250, SSBDATE_920101,
        SSBDATE_980101, res + SUMGRP_S2, NGRP_S31);
  s3fix("SSB32", dat, bmp, factSz, 190, 200, 190, 200, SSBDATE_920101,
        SSBDATE_980101, res + SUMGRP_S2 + NGRP_S31, NGRP_S32);
  s3fix("SSB33", dat, bmp, factSz, 51, 55, 51, 55, SSBDATE_920101,
        SSBDATE_980101, res + SUMGRP_S2 + NGRP_S31 + NGRP_S32, NGRP_S33);
  s3fix("SSB34", dat, bmp, factSz, 51, 55, 51, 55, SSBDATE_971201,
        SSBDATE_980101, res + SUMGRP_S2 + NGRP_S31 + NGRP_S32 + NGRP_S33, NGRP_S34);

  s4fix("SSB41", dat, bmp, factSz, 150, 200, 150, 200, 0, 400, SSBDATE_920101,
        SSBDATE_990101, res + SUMGRP_S3, NGRP_S41);
  s4fix("SSB42", dat, bmp, factSz, 150, 200, 150, 200, 0, 400, SSBDATE_970101,
        SSBDATE_990101, res + SUMGRP_S3 + NGRP_S41, NGRP_S42);
  s4fix("SSB43", dat, bmp, factSz, 150, 200, 190, 200, 120, 160, SSBDATE_970101,
        SSBDATE_990101, res + SUMGRP_S3 + NGRP_S41 + NGRP_S42, NGRP_S43);
  return res;
}
