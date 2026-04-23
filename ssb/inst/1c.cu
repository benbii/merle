#include "../../primitive_abla.cuh"
#include "../lambdas.cuh"

namespace c {
void s11s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint));
  s1op op { // We assume that op.operator() gets constant-propagated here
      SSBDATE_930101, SSBDATE_940101, dat->loOrderDate,
      1, 4, dat->loDiscount,
      1, 25, dat->loQuantity, dat->loExtendedPrice,
  };
  auto d1 = b->dateSparse[1], di0 = b->discntSparse[0], di1 = b->discntSparse[1],
       q0 = b->qtySparse[0], q1 = b->qtySparse[1], q2 = b->qtySparse[2];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, 1,
      [=] __device__ (uint i) {
        uint x = d1[i], y = 0;
        if (x != 0) x &= (q0[i] | q1[i] | q2[i]), y |= q2[i];
        if (x != 0) x &= (di0[i] | di1[i]), y |= (di0[i] | di1[i]);
        return make_uint2(x, y);
      }, op);
}

void s11m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint));
  s1op op{
      SSBDATE_930101, SSBDATE_940101, dat->loOrderDate,
      1, 4, dat->loDiscount,
      1, 25, dat->loQuantity, dat->loExtendedPrice,
  };
  // Medium keeps date sparse/aligned and discount dense/aligned, but quantity
  // still has an unaligned upper boundary in [21, 26).
  auto d1 = b->dateSparse[1];
  auto di1 = b->discntDense[1], di2 = b->discntDense[2], di3 = b->discntDense[3];
  auto q0 = b->qtySparse[0], q1 = b->qtySparse[1], q4 = b->qtyDense[4];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, 1,
      [=] __device__ (uint i) {
        uint x = d1[i], y = 0;
        if (x != 0) x &= (di1[i] | di2[i] | di3[i]);
        if (x != 0) x &= (q0[i] | q1[i] | q4[i]), y |= q4[i];
        return make_uint2(x, y);
      }, op);
}

void s12s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint));
  s1op op{
      SSBDATE_940101, SSBDATE_940201, dat->loOrderDate,
      4, 7, dat->loDiscount,
      26, 36, dat->loQuantity, dat->loExtendedPrice,
  };
  auto d2 = b->dateSparse[2], di1 = b->discntSparse[1], di2 = b->discntSparse[2],
       q2 = b->qtySparse[2], q3 = b->qtySparse[3];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, 1,
      [=] __device__ (uint i) {
        uint x = d2[i], y = 0;
        if (x != 0) x &= (di1[i] | di2[i]), y |= (di1[i] | di2[i]);
        if (x != 0) x &= (q2[i] | q3[i]), y |= (q2[i] | q3[i]);
        return make_uint2(x, y);
      }, op);
}

void s12m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint));
  s1op op{
      SSBDATE_940101, SSBDATE_940201, dat->loOrderDate,
      4, 7, dat->loDiscount,
      26, 36, dat->loQuantity, dat->loExtendedPrice,
  };
  // Medium uses aligned fine bins for discount/quantity, but date remains in
  // the sparse 1994 year bin and therefore still requires candidate checking.
  auto d2 = b->dateSparse[2];
  auto di4 = b->discntDense[4], di5 = b->discntDense[5], di6 = b->discntDense[6];
  auto q5 = b->qtyDense[5], q6 = b->qtyDense[6];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, 1,
      [=] __device__ (uint i) {
        uint x = d2[i], y = x;
        if (x != 0) x &= (q5[i] | q6[i]);
        if (x != 0) x &= (di4[i] | di5[i] | di6[i]);
        return make_uint2(x, y);
      }, op);
}

void s12d(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint));
  s1op op{
      SSBDATE_940101, SSBDATE_940201, dat->loOrderDate,
      4, 7, dat->loDiscount,
      26, 36, dat->loQuantity, dat->loExtendedPrice,
  };
  auto d0 = b->dateDense[0];
  auto di4 = b->discntDense[4], di5 = b->discntDense[5], di6 = b->discntDense[6];
  auto q5 = b->qtyDense[5], q6 = b->qtyDense[6];
  abla::hardcoded<nt, vt, vt0, false><<<ceil_div(factsz, nv32), nt>>>(
      factsz, grp_out, 1,
      [=] __device__(uint i) {
        uint x = d0[i];
        if (x != 0) x &= (q5[i] | q6[i]);
        if (x != 0) x &= (di4[i] | di5[i] | di6[i]);
        return make_uint2(x, 0);
      }, op);
}

void s13s(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint));
  s1op op{
      SSBDATE_940204, SSBDATE_940211, dat->loOrderDate,
      5, 8, dat->loDiscount,
      26, 36, dat->loQuantity, dat->loExtendedPrice,
  };
  auto d2 = b->dateSparse[2], di2 = b->discntSparse[2],
       q2 = b->qtySparse[2], q3 = b->qtySparse[3];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, 1,
      [=] __device__ (uint i) {
        uint x = d2[i], y = x;
        if (x != 0) x &= di2[i];
        if (x != 0) x &= (q2[i] | q3[i]), y |= (q2[i] | q3[i]);
        return make_uint2(x, y);
      }, op);
}

void s13m(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint));
  s1op op{
      SSBDATE_940204, SSBDATE_940211, dat->loOrderDate,
      5, 8, dat->loDiscount,
      26, 36, dat->loQuantity, dat->loExtendedPrice,
  };
  // Discount remains on the exact sparse [5, 8) bin; only quantity refines.
  auto d2 = b->dateSparse[2], di2 = b->discntSparse[2];
  auto q5 = b->qtyDense[5], q6 = b->qtyDense[6];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, 1,
      [=] __device__ (uint i) {
        uint x = d2[i], y = x;
        if (x != 0) x &= di2[i];
        if (x != 0) x &= (q5[i] | q6[i]);
        return make_uint2(x, y);
      }, op);
}

void s13d(const ssb_schema *dat, ssb_bmp *b, size_t factsz, uint *grp_out) {
  cudaMemset(grp_out, 0, sizeof(uint));
  s1op op{
      SSBDATE_940204, SSBDATE_940211, dat->loOrderDate,
      5, 8, dat->loDiscount,
      26, 36, dat->loQuantity, dat->loExtendedPrice,
  };
  // Dense date only narrows to the 1994-02 month bin, so candidate checking
  // still remains enabled.
  auto d1 = b->dateDense[1], di2 = b->discntSparse[2];
  auto q5 = b->qtyDense[5], q6 = b->qtyDense[6];
  abla::hardcoded<nt, cp_vt, vt0, true><<<ceil_div(factsz, cp_nv32), nt>>>(
      factsz, grp_out, 1,
      [=] __device__ (uint i) {
        uint x = d1[i], y = x;
        if (x != 0) x &= di2[i];
        if (x != 0) x &= (q5[i] | q6[i]);
        return make_uint2(x, y);
      }, op);
}
}
