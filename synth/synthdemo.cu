#include "synthdemo.h"
#include "../primitive_abla.cuh"
#include <cstdlib>
using namespace mybmpidx;
using cuda::ceil_div;

static __device__ uint2 _op(uint i, synth_schema dat, uint64_t fa1low,
                            uint64_t fa1hi, uint64_t fa2low, uint64_t fa2hi,
                            uint64_t da1low, uint64_t da1hi, bool chk) {
  // factattr support 8 16 32 64b but we only use 16 and 32
  uint2 ret = {1, ELIMINATED};
  if (chk) {
    uint64_t factAttr1Val = 0, factAttr2Val = 0;
    // Extract fact attribute values based on bitwidth
    switch (dat.bitwidth) {
    case 8:
      factAttr1Val = ((uint8_t *)dat.factattr1)[i];
      factAttr2Val = ((uint8_t *)dat.factattr2)[i];
      break;
    case 16:
      factAttr1Val = ((uint16_t *)dat.factattr1)[i];
      factAttr2Val = ((uint16_t *)dat.factattr2)[i];
      break;
    case 32:
      factAttr1Val = ((uint32_t *)dat.factattr1)[i];
      factAttr2Val = ((uint32_t *)dat.factattr2)[i];
      break;
    case 64:
      factAttr1Val = ((uint64_t *)dat.factattr1)[i];
      factAttr2Val = ((uint64_t *)dat.factattr2)[i];
      break;
    default:
      __builtin_unreachable();
    }
    // Check fact table predicates (OR condition)
    if (!((factAttr1Val >= fa1low && factAttr1Val < fa1hi) ||
          (factAttr2Val >= fa2low && factAttr2Val < fa2hi)))
      return ret;
  }

  // Join with dimension table using foreign key
  uint32_t dimKey = dat.fkey[i];
  // Check dimension table predicate
  // dimattr1 is the dimension key (0, 1, 2, 3, ...)
  if (chk) {
    auto da1val = dat.dimattr1[dimKey];
    if (da1val < da1low || da1val >= da1hi)
      return ret;
  }
  // GROUP BY dimattr2 and count qualifying rows
  ret.y = dat.dimattr2[dimKey]; // Group key (0-255)
  return ret;
}

#ifdef LARGE_SMEM
static constexpr size_t nt = 256, vt = 3, vt0 = 19, ndup = 100;
static constexpr size_t nv_ = nt * vt, nv32 = nv_ * 32;
static constexpr size_t cp_vt = vt, cp_nv32 = nv32;
#else
static constexpr size_t nt = 256, vt = 3, vt0 = 12, ndup = 100;
static constexpr size_t nv_ = nt * vt, nv32 = nv_ * 32;
// Program + candchk is highly shmem intensive.
// Go for a less aggressive `vt` choice.
static constexpr size_t cp_vt = 2, cp_nv = nt * cp_vt, cp_nv32 = cp_nv * 32;
#endif

float synth_join(const synth_schema *dat, size_t factsz, uint64_t fa1lo,
               uint64_t fa1hi, uint64_t fa2lo, uint64_t fa2hi, uint64_t da1lo,
               uint64_t da1hi, uint32_t *grpout) {
  float foo;
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (size_t d = 0; d < ndup; ++d) {
    cudaMemset(grpout, 0, 256 * sizeof(uint32_t));
    grpby<nt, 4><<<ceil_div(factsz, nt * 4), nt, 256 * sizeof(uint)>>>(
        factsz, grpout, 256, [=, dat = *dat] __device__(uint i) {
          return _op(i, dat, fa1lo, fa1hi, fa2lo, fa2hi, da1lo, da1hi, true);
        });
  }
  cudaEventRecord(stop); cudaEventSynchronize(stop);
  cudaEventElapsedTime(&foo, start, stop);
  cudaEventDestroy(start); cudaEventDestroy(stop);
  foo /= ndup;
  return foo;
}

// ours, baseFuse, vprgFuse, noFuse
float4 synth_bmpdemo(const synth_schema *dat, struct synth_bmp *bmp,
                     uint factsz, uint f1lo, uint f1hi, uint f2lo,
                     uint f2hi, uint d1lo, uint d1hi, uint *grp_out) {
  auto op = [=, dat = *dat] __device__(uint i, bool c) {
    return _op(i, dat, f1lo, f1hi, f2lo, f2hi, d1lo, d1hi, c);
  };
#define BIN(c, n) col::from_bin(NBIN / 2, bmp->c##n, bmp->c##Bound, c##n##lo, c##n##hi)

  float4 foo;
  cudaEvent_t start, stop;
  cudaEventCreate(&start), cudaEventCreate(&stop);
  vprg p = {{BIN(f, 1), BIN(f, 2), BIN(d, 1)},
            factsz,
            {{vprg::ORM, 0, 0}, {vprg::ORM, 0, 1}, {vprg::ANDM, 0, 2}}};
  constexpr size_t nr_grp = 256;

  cudaEventRecord(start);
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
  cudaEventElapsedTime(&foo.w, start, stop);
  foo.x = foo.w / ndup;

  p.instrs[2] = {vprg::ORM, 1, 2};
  p.instrs[3] = {vprg::AND, 0, 1};
  grp_out += nr_grp;
  uint *idxbuf;
  cudaMalloc(&idxbuf, sizeof(uint) * 6 * ceil_div(factsz, 32));
  if (!idxbuf)
    exit(fprintf(stderr, __FILE__ " CUDA OOM\n"));
  cudaEventRecord(start);
  for (size_t d = 0; d < ndup; ++d) {
    cudaMemset(grp_out, 0, nr_grp * sizeof(uint));
    if (p.is_nochk()) {
      cudaMemset(idxbuf, 0, sizeof(uint) * 3 * ceil_div(factsz, 32));
      abla::base_fuse<nt, vt, false>
          <<<ceil_div(factsz, nv32), nt>>>(p, grp_out, nr_grp, op, idxbuf);
    } else {
      cudaMemset(idxbuf, 0, sizeof(uint) * 6 * ceil_div(factsz, 32));
      abla::base_fuse<nt, cp_vt, true>
          <<<ceil_div(factsz, cp_nv32), nt>>>(p, grp_out, nr_grp, op, idxbuf);
    }
  }
  cudaEventRecord(stop); cudaEventSynchronize(stop);
  cudaEventElapsedTime(&foo.w, start, stop);
  foo.y = foo.w / ndup;
  cudaFree(idxbuf);

  // uint *d_possi, *d_uncert, *d_onelist;
  // cudaMalloc(&d_possi, ceil_div(factsz, 32) * sizeof(uint32_t));
  // cudaMalloc(&d_uncert, ceil_div(factsz, 32) * sizeof(uint32_t));
  // cudaMalloc(&d_onelist, ceil_div(factsz, 4) * sizeof(uint32_t));
  // if (!d_possi || !d_uncert || !d_onelist)
  //   exit(fprintf(stderr, __FILE__ " CUDA OOM\n"));

  // } else if (ty == 3 || ty == 4) {
  //   foo = abla::nofuse_abla<nt, cp_vt, vt, vt, false>(
  //       p, 256, op, grp_out, d_possi, d_uncert, d_onelist);
  // } else if (ty == 5) {
  //   foo = abla::nofuse_abla<nt, cp_vt, vt, vt, true>(
  //       p, 256, op, grp_out, d_possi, d_uncert, d_onelist);
  cudaEventDestroy(start); cudaEventDestroy(stop);
  return foo;
}

/* float2 synth_method(const synth_schema *dat, uint factsz, uint fa1lo,
    uint fa1hi, uint fa2lo, uint fa2hi, uint da1lo, uint da1hi) {
  static constexpr auto nodim = recipe::nodim;
  float2 ret = {0.0f, 0.0f};
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  uint *grpout;
  cudaMalloc(&grpout, 512 * sizeof(uint32_t));

  recipe r = {.factsz = factsz, .nbit = {dat->bitwidth, dat->bitwidth, 32},
              .min = {fa1lo, fa2lo, da1lo}, .max = {fa1hi, fa2hi, da1hi},
              .dimsz = {nodim, nodim, factsz / 100},
              .fk = {nullptr, nullptr, dat->fkey},
              .attr = {dat->factattr1, dat->factattr2, dat->dimattr1}};
  const vprg::instr instrs[MAXNINSTR] = {
      {vprg::ORM, 0, 0, 0}, {vprg::ORM, 0, 1, 0}, {vprg::ANDM, 0, 2, 0}};
  vprg p = r.perfect(instrs);
  auto op = [=, dat = *dat] __device__(uint i, bool c) {
    return _op(i, dat, fa1lo, fa1hi, fa2lo, fa2hi, da1lo, da1hi, c);
  };

  // Method 1: direct while loop -> writes to grpout[0:256]
  cudaEventRecord(start);
  for (size_t d = 0; d < ndup; ++d) {
    cudaMemset(grpout, 0, 256 * sizeof(uint32_t));
    abla::method1<nt, vt><<<ceil_div(factsz, nv32), nt>>>(p, grpout, 256, op);
  }
  cudaEventRecord(stop); cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ret.x, start, stop);

  // Method 2: block scan + parallel -> writes to grpout[256:512]
  cudaEventRecord(start);
  for (size_t d = 0; d < ndup; ++d) {
    cudaMemset(grpout + 256, 0, 256 * sizeof(uint32_t));
    abla::method2<nt, vt, vt0, false>
        <<<ceil_div(factsz, nv32), nt>>>(p, grpout + 256, 256, op);
  }
  cudaEventRecord(stop); cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ret.y, start, stop);

  uint32_t hostbuf[512];
  cudaMemcpy(hostbuf, grpout, 512 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 256; ++i)
    if (hostbuf[i] != hostbuf[i + 256])
      exit(fprintf(stderr, "Method 1 vs 2 mismatch at %d: %u != %u\n",
              i, hostbuf[i], hostbuf[i + 256]));
  cudaFree(grpout);
  p.release();
  cudaEventDestroy(start); cudaEventDestroy(stop);
  return ret;
} */
