#include "../../primitive_abla.cuh"
#include "../lambdas.cuh"

namespace c {
void s31s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint) * NGRP_S31);
  s3op op{
      dat->loCustKey, 200, 250, dat->custCity,
      200, 250, dat->loSuppCity,
      SSBDATE_920101, SSBDATE_980101, dat->loOrderDate, dat->loRevenue,
  };
  auto c4 = b->cCitySparse[4], s4 = b->sCitySparse[4], d6 = b->dateSparse[6];
  abla::hardcoded<nt, vt, vt0, false><<<ceil_div(factsz, nv32), nt>>>(
      factsz, grp_out, NGRP_S31,
      [=] __device__ (uint i) {
        uint x = c4[i];
        if (x != 0) x &= s4[i];
        if (x != 0) x &= ~d6[i];
        return make_uint2(x, 0);
      }, op);
}

void s32s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint) * NGRP_S32);
  s3op op{
      dat->loCustKey, 190, 200, dat->custCity,
      190, 200, dat->loSuppCity,
      SSBDATE_920101, SSBDATE_980101, dat->loOrderDate, dat->loRevenue,
  };
  auto c3 = b->cCitySparse[3], s3 = b->sCitySparse[3], d6 = b->dateSparse[6];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, NGRP_S32,
      [=] __device__ (uint i) {
        uint x = c3[i], y = x;
        if (x != 0) x &= s3[i], y |= s3[i];
        if (x != 0) x &= ~d6[i];
        return make_uint2(x, y);
      }, op);
}

void s32m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint) * NGRP_S32);
  s3op op{
      dat->loCustKey, 190, 200, dat->custCity,
      190, 200, dat->loSuppCity,
      SSBDATE_920101, SSBDATE_980101, dat->loOrderDate, dat->loRevenue,
  };
  auto c2 = b->cCityDense[2], s3 = b->sCitySparse[3], d6 = b->dateSparse[6];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, NGRP_S32,
      [=] __device__ (uint i) {
        uint x = c2[i], y = 0;
        if (x != 0) x &= s3[i], y = x;
        if (x != 0) x &= ~d6[i];
        return make_uint2(x, y);
      }, op);
}

void s32d(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint) * NGRP_S32);
  s3op op{
      dat->loCustKey, 190, 200, dat->custCity,
      190, 200, dat->loSuppCity,
      SSBDATE_920101, SSBDATE_980101, dat->loOrderDate, dat->loRevenue,
  };
  auto c2 = b->cCityDense[2], s2 = b->sCityDense[2], d6 = b->dateSparse[6];
  abla::hardcoded<nt, vt, vt0, false><<<ceil_div(factsz, nv32), nt>>>(
      factsz, grp_out, NGRP_S32,
      [=] __device__ (uint i) {
        uint x = c2[i];
        if (x != 0) x &= s2[i];
        if (x != 0) x &= ~d6[i];
        return make_uint2(x, 0);
      }, op);
}

void s33s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint) * NGRP_S33);
  s3op op{
      dat->loCustKey, 51, 55, dat->custCity,
      51, 55, dat->loSuppCity,
      SSBDATE_920101, SSBDATE_980101, dat->loOrderDate, dat->loRevenue,
  };
  auto c1 = b->cCitySparse[1], s1 = b->sCitySparse[1], d6 = b->dateSparse[6];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, NGRP_S33,
      [=] __device__ (uint i) {
        uint x = c1[i], y = x;
        if (x != 0) x &= s1[i], y |= s1[i];
        if (x != 0) x &= ~d6[i];
        return make_uint2(x, y);
      }, op);
}

void s33m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint) * NGRP_S33);
  s3op op{
      dat->loCustKey, 51, 55, dat->custCity,
      51, 55, dat->loSuppCity,
      SSBDATE_920101, SSBDATE_980101, dat->loOrderDate, dat->loRevenue,
  };
  auto c0 = b->cCityDense[0], s1 = b->sCitySparse[1], d6 = b->dateSparse[6];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, NGRP_S33,
      [=] __device__ (uint i) {
        uint x = c0[i], y = x;
        if (x != 0) x &= s1[i], y |= s1[i];
        if (x != 0) x &= ~d6[i];
        return make_uint2(x, y);
      }, op);
}

void s33d(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint) * NGRP_S33);
  s3op op{
      dat->loCustKey, 51, 55, dat->custCity,
      51, 55, dat->loSuppCity,
      SSBDATE_920101, SSBDATE_980101, dat->loOrderDate, dat->loRevenue,
  };
  auto c0 = b->cCityDense[0], s0 = b->sCityDense[0], d6 = b->dateSparse[6];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, NGRP_S33,
      [=] __device__ (uint i) {
        uint x = c0[i], y = x;
        if (x != 0) x &= s0[i], y |= s0[i];
        if (x != 0) x &= ~d6[i];
        return make_uint2(x, y);
      }, op);
}

void s34s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint) * NGRP_S34);
  s3op op{
      dat->loCustKey, 51, 55, dat->custCity,
      51, 55, dat->loSuppCity,
      SSBDATE_971201, SSBDATE_980101, dat->loOrderDate, dat->loRevenue,
  };
  auto c1 = b->cCitySparse[1], s1 = b->sCitySparse[1], d6 = b->dateSparse[6];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, NGRP_S34,
      [=] __device__ (uint i) {
        uint x = c1[i], y = 0;
        if (x != 0) x &= s1[i], y |= s1[i];
        if (x != 0) x &= d6[i], y |= d6[i];
        return make_uint2(x, y);
      }, op);
}

void s34m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint) * NGRP_S34);
  s3op op{
      dat->loCustKey, 51, 55, dat->custCity,
      51, 55, dat->loSuppCity,
      SSBDATE_971201, SSBDATE_980101, dat->loOrderDate, dat->loRevenue,
  };
  auto c0 = b->cCityDense[0], s1 = b->sCitySparse[1], d6 = b->dateSparse[6];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, NGRP_S34,
      [=] __device__ (uint i) {
        uint x = c0[i], y = 0;
        if (x != 0) x &= s1[i], y |= s1[i];
        if (x != 0) x &= d6[i], y |= d6[i];
        return make_uint2(x, y);
      }, op);
}

void s34d(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint) * NGRP_S34);
  s3op op{
      dat->loCustKey, 51, 55, dat->custCity,
      51, 55, dat->loSuppCity,
      SSBDATE_971201, SSBDATE_980101, dat->loOrderDate, dat->loRevenue,
  };
  auto c0 = b->cCityDense[0], s0 = b->sCityDense[0], d3 = b->dateDense[3];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, NGRP_S34,
      [=] __device__ (uint i) {
        uint x = c0[i], y = 0;
        if (x != 0) x &= s0[i], y |= s0[i];
        if (x != 0) x &= d3[i], y |= d3[i];
        return make_uint2(x, y);
      }, op);
}
}
