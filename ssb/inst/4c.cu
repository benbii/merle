#include "../../primitive_abla.cuh"
#include "../lambdas.cuh"

namespace c {
// s41 is the most incidental: even the sparse bitmap aligns with the query
// also it does NOT select on date column
void s41s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint) * NGRP_S41);
  s4op op{dat->loCustKey, dat->loPartKey, 150, 200, dat->custCity, 150, 200,
          dat->loSuppCity, 0, 400, dat->partMfgr, SSBDATE_920101,
          SSBDATE_990101, dat->loOrderDate,
          dat->loRevenue, dat->loSupplyCost};
  auto c3 = b->cCitySparse[3], s3 = b->sCitySparse[3],
       m0 = b->mfgrSparse[0], m1 = b->mfgrSparse[1];
  abla::hardcoded<nt, vt, vt0, false><<<ceil_div(factsz, nv32), nt>>>(
      factsz, grp_out, NGRP_S41,
      [=] __device__ (uint i) {
        uint x = c3[i];
        if (x != 0) x &= s3[i];
        if (x != 0) x &= (m0[i] | m1[i]);
        return make_uint2(x, 0);
      }, op);
}

void s42s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint) * NGRP_S42);
  s4op op{dat->loCustKey, dat->loPartKey, 150, 200, dat->custCity, 150, 200,
          dat->loSuppCity, 0, 400, dat->partMfgr, SSBDATE_970101,
          SSBDATE_990101, dat->loOrderDate,
          dat->loRevenue, dat->loSupplyCost};
  auto c3 = b->cCitySparse[3], s3 = b->sCitySparse[3],
       m0 = b->mfgrSparse[0], m1 = b->mfgrSparse[1],
       d5 = b->dateSparse[5], d6 = b->dateSparse[6];
  abla::hardcoded<nt, vt, vt0, false><<<ceil_div(factsz, nv32), nt>>>(
      factsz, grp_out, NGRP_S42,
      [=] __device__ (uint i) {
        uint x = c3[i];
        if (x != 0) x &= s3[i];
        if (x != 0) x &= (m0[i] | m1[i]);
        if (x != 0) x &= (d5[i] | d6[i]);
        return make_uint2(x, 0);
      }, op);
}

void s43s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint) * NGRP_S43);
  s4op op{dat->loCustKey, dat->loPartKey, 150, 200, dat->custCity, 190, 200,
          dat->loSuppCity, 120, 160, dat->partMfgr, SSBDATE_970101,
          SSBDATE_990101, dat->loOrderDate,
          dat->loRevenue, dat->loSupplyCost};
  auto c3 = b->cCitySparse[3], s3 = b->sCitySparse[3],
       m0 = b->mfgrSparse[0], d5 = b->dateSparse[5], d6 = b->dateSparse[6];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, NGRP_S43,
      [=] __device__ (uint i) {
        uint x = c3[i], y = 0;
        if (x != 0) x &= s3[i], y |= s3[i];
        if (x != 0) x &= m0[i], y |= m0[i];
        if (x != 0) x &= (d5[i] | d6[i]);
        return make_uint2(x, y);
      }, op);
}

void s43m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint) * NGRP_S43);
  s4op op{dat->loCustKey, dat->loPartKey, 150, 200, dat->custCity, 190, 200,
          dat->loSuppCity, 120, 160, dat->partMfgr, SSBDATE_970101,
          SSBDATE_990101, dat->loOrderDate,
          dat->loRevenue, dat->loSupplyCost};
  auto c3 = b->cCitySparse[3], s3 = b->sCitySparse[3],
       m2 = b->mfgrDense[2], d5 = b->dateSparse[5], d6 = b->dateSparse[6];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, NGRP_S43,
      [=] __device__ (uint i) {
        uint x = c3[i], y = 0;
        if (x != 0) x &= s3[i], y |= s3[i];
        if (x != 0) x &= m2[i];
        if (x != 0) x &= (d5[i] | d6[i]);
        return make_uint2(x, y);
      }, op);
}

void s43d(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint) * NGRP_S43);
  s4op op{dat->loCustKey, dat->loPartKey, 150, 200, dat->custCity, 190, 200,
          dat->loSuppCity, 120, 160, dat->partMfgr, SSBDATE_970101,
          SSBDATE_990101, dat->loOrderDate,
          dat->loRevenue, dat->loSupplyCost};
  auto c3 = b->cCitySparse[3], s2 = b->sCityDense[2],
       m2 = b->mfgrDense[2], d5 = b->dateSparse[5], d6 = b->dateSparse[6];
  abla::hardcoded<nt, vt, vt0, false><<<ceil_div(factsz, nv32), nt>>>(
      factsz, grp_out, NGRP_S43,
      [=] __device__ (uint i) {
        uint x = c3[i];
        if (x != 0) x &= s2[i];
        if (x != 0) x &= m2[i];
        if (x != 0) x &= (d5[i] | d6[i]);
        return make_uint2(x, 0);
      }, op);
}
}
