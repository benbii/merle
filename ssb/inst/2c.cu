#include "../../primitive_abla.cuh"
#include "../lambdas.cuh"

namespace c {
void s21s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint) * NGRP_S21);
  s2op op{
      dat->loPartKey, 40, 80, dat->partMfgr,
      150, 200, dat->loSuppCity,
      dat->loOrderDate, dat->loRevenue,
  };
  auto m0 = b->mfgrSparse[0], s3 = b->sCitySparse[3];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, NGRP_S21,
      [=] __device__ (uint i) {
        uint x = m0[i], y = x;
        if (x != 0) x &= s3[i];
        return make_uint2(x, y);
      }, op);
}

void s21m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint) * NGRP_S21);
  s2op op{
      dat->loPartKey, 40, 80, dat->partMfgr,
      150, 200, dat->loSuppCity,
      dat->loOrderDate, dat->loRevenue,
  };
  auto m0 = b->mfgrDense[0], s3 = b->sCitySparse[3];
  abla::hardcoded<nt, vt, vt0, false><<<ceil_div(factsz, nv32), nt>>>(
      factsz, grp_out, NGRP_S21,
      [=] __device__ (uint i) {
        uint x = m0[i];
        if (x != 0) x &= s3[i];
        return make_uint2(x, 0);
      }, op);
}

void s22s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint) * NGRP_S22);
  s2op op{
      dat->loPartKey, 260, 268, dat->partMfgr,
      200, 250, dat->loSuppCity,
      dat->loOrderDate, dat->loRevenue,
  };
  auto m1 = b->mfgrSparse[1], s4 = b->sCitySparse[4];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, NGRP_S22,
      [=] __device__ (uint i) {
        uint x = m1[i], y = x;
        if (x != 0) x &= s4[i];
        return make_uint2(x, y);
      }, op);
}

void s22m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint) * NGRP_S22);
  s2op op{
      dat->loPartKey, 260, 268, dat->partMfgr,
      200, 250, dat->loSuppCity,
      dat->loOrderDate, dat->loRevenue,
  };
  auto m4 = b->mfgrDense[4], s4 = b->sCitySparse[4];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, NGRP_S22,
      [=] __device__ (uint i) {
        uint x = s4[i], y = 0;
        if (x != 0) x &= m4[i], y = x;
        return make_uint2(x, y);
      }, op);
}

void s23s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint) * NGRP_S23);
  s2op op{
      dat->loPartKey, 260, 261, dat->partMfgr,
      50, 100, dat->loSuppCity,
      dat->loOrderDate, dat->loRevenue,
  };
  auto m1 = b->mfgrSparse[1], s1 = b->sCitySparse[1];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, NGRP_S23,
      [=] __device__ (uint i) {
        uint x = m1[i], y = x;
        if (x != 0) x &= s1[i];
        return make_uint2(x, y);
      }, op);
}

void s23m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint) * NGRP_S23);
  s2op op{
      dat->loPartKey, 260, 261, dat->partMfgr,
      50, 100, dat->loSuppCity,
      dat->loOrderDate, dat->loRevenue,
  };
  auto m4 = b->mfgrDense[4], s1 = b->sCitySparse[1];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, NGRP_S23,
      [=] __device__ (uint i) {
        uint x = m4[i], y = x;
        if (x != 0) x &= s1[i];
        return make_uint2(x, y);
      }, op);
}

}
