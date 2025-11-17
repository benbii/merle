#include "primitive.cuh"
// make clangd happy
#ifdef __CLANG__CUDA_MATH_FORWARD_DECLARES_H__
#define __ldcs(x) *(x)
#endif

static void* __cum(size_t sz) {
  void *bruh;
  cudaError_t e = cudaMalloc(&bruh, sz);
  if (e != cudaSuccess) exit(fputs(__FILE__" cuda oom\n", stderr));
  return bruh;
}

namespace mybmpidx {

// ONLY FREE OUT[0]!!!
size_t create_bin(const uint *__restrict__ fk, const void *__restrict__ attr,
                  size_t nbit, uint factsz, uint dimsz, const uint64_t *min,
                  const uint64_t *max, size_t ncol, uint **out) {
  auto fetch = [=] __device__ (uint i) -> uint64_t {
    if (fk == nullptr) {
      if (nbit <= 8) return reinterpret_cast<const uint8_t*>(attr)[i];
      if (nbit <= 16) return reinterpret_cast<const uint16_t*>(attr)[i];
      if (nbit <= 32) return reinterpret_cast<const uint32_t*>(attr)[i];
      return reinterpret_cast<const uint64_t*>(attr)[i];
    }

    // Join
    uint dimk = fk[i];
    if (dimk >= dimsz) return INT64_MAX;
    if (nbit <= 8)
      return reinterpret_cast<const uint8_t *>(attr)[dimk];
    if (nbit <= 16)
      return reinterpret_cast<const uint16_t *>(attr)[dimk];
    if (nbit <= 32)
      return reinterpret_cast<const uint32_t *>(attr)[dimk];
    return reinterpret_cast<const uint64_t *>(attr)[dimk];
  };
  auto compar = [] __device__(uint64_t a, uint64_t l, uint64_t h) {
    return a >= l && a < h;
  };

  size_t bmpsz_words = cuda::ceil_div(factsz, 32);
  size_t bmpsz_bytes = bmpsz_words * sizeof(uint);
  out[0] = (uint*)__cum(bmpsz_bytes * ncol);
  for (size_t i = 1; i < ncol; ++i)
    out[i] = out[0] + bmpsz_words * i;
  uint64_t *devmin = (uint64_t *)__cum(ncol * sizeof(uint64_t) * 3);
  uint64_t *devmax = devmin + ncol;
  uint **devout = (uint **)(devmax + ncol);
  cudaMemcpy(devmin, min, ncol * sizeof(uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(devmax, max, ncol * sizeof(uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(devout, out, ncol * sizeof(intptr_t), cudaMemcpyHostToDevice);
  bmpcreate<256, 4><<<cuda::ceil_div(factsz, 256 * 4), 256>>>(
      factsz, devout, fetch, compar, devmin, devmax, ncol);
  cudaMemcpy(out, devout, ncol * sizeof(intptr_t), cudaMemcpyDeviceToHost);
  cudaFree(devmin);
  return bmpsz_bytes;
}

uint *create_select(const void *__restrict__ src, size_t nbit, uint factsz,
                    uint min, uint max) {
  auto op8 = [=] __device__ (uint idx) {
    uint8_t val = __ldcs(static_cast<const uint8_t*>(src) + idx);
    return val >= min && val < max;
  };
  auto op16 = [=] __device__ (uint idx) {
    uint16_t val = __ldcs(static_cast<const uint16_t*>(src) + idx);
    return val >= min && val < max;
  };
  auto op32 = [=] __device__ (uint idx) {
    uint32_t val = __ldcs(static_cast<const uint32_t*>(src) + idx);
    return val >= min && val < max;
  };
  auto op64 = [=] __device__ (uint idx) {
    uint64_t val = __ldcs(static_cast<const uint64_t*>(src) + idx);
    return val >= min && val < max;
  };

  uint *out, ceildivd = cuda::ceil_div(factsz, 32) * 4;
  if (cudaMalloc(&out, ceildivd) != cudaSuccess)
    return NULL;
  ceildivd = cuda::ceil_div(factsz, 256*4);
  if (nbit <= 8)
    bmpcreate<256, 4><<<ceildivd, 256>>>(factsz, out, op8);
  else if (nbit <= 16)
    bmpcreate<256, 4><<<ceildivd, 256>>>(factsz, out, op16);
  else if (nbit <= 32)
    bmpcreate<256, 4><<<ceildivd, 256>>>(factsz, out, op32);
  else if (nbit <= 64)
    bmpcreate<256, 4><<<ceildivd, 256>>>(factsz, out, op64);
  return out;
}

uint *create_join(const uint32_t *__restrict__ foreignkey,
                  const void *__restrict__ dimattr, size_t nbit, uint factsz,
                  uint dimsz, uint min, uint max) {
  if (foreignkey == NULL)
    return create_select(dimattr, nbit, factsz, min, max);
  auto op8 = [=] __device__ (uint idx) {
    uint32_t key = __ldcs(foreignkey + idx);
    if (key >= dimsz) return false;
    // DO NOT stream load in dim tables!
    uint8_t val = static_cast<const uint8_t*>(dimattr)[key];
    return val >= min && val < max;
  };
  auto op16 = [=] __device__ (uint idx) {
    uint32_t key = __ldcs(foreignkey + idx);
    if (key >= dimsz) return false;
    uint16_t val = static_cast<const uint16_t*>(dimattr)[key];
    return val >= min && val < max;
  };
  auto op32 = [=] __device__ (uint idx) {
    uint32_t key = __ldcs(foreignkey + idx);
    if (key >= dimsz) return false;
    uint32_t val = static_cast<const uint32_t*>(dimattr)[key];
    return val >= min && val < max;
  };
  auto op64 = [=] __device__ (uint idx) {
    uint32_t key = __ldcs(foreignkey + idx);
    if (key >= dimsz) return false;
    uint64_t val = static_cast<const uint64_t*>(dimattr)[key];
    return val >= min && val < max;
  };

  uint *out, ceildivd = cuda::ceil_div(factsz, 32) * 4;
  if (cudaMalloc(&out, ceildivd) != cudaSuccess)
    return NULL;
  ceildivd = cuda::ceil_div(factsz, 256*4);
  if (nbit <= 8)
    bmpcreate<256, 4><<<ceildivd, 256>>>(factsz, out, op8);
  else if (nbit <= 16)
    bmpcreate<256, 4><<<ceildivd, 256>>>(factsz, out, op16);
  else if (nbit <= 32)
    bmpcreate<256, 4><<<ceildivd, 256>>>(factsz, out, op32);
  else if (nbit <= 64)
    bmpcreate<256, 4><<<ceildivd, 256>>>(factsz, out, op64);
  return out;
}

vprg recipe::perfect(const vprg::instr *instrs) const {
  vprg ret;
  ret.factsz = factsz;
  memcpy(ret.instrs, instrs, sizeof(vprg::instr) * MAXNINSTR);
  ret.factsz = factsz;
  for (size_t i = 0; i < MAXCOLS; ++i) {
    if (attr[i] == NULL) continue;
    ret.col_bmps[i].middle[0] =
        create_join(fk[i], attr[i], nbit[i], factsz, dimsz[i], min[i], max[i]);
  }
  return ret;
}

vprg recipe::many_or(const vprg::instr *instrs) const {
  vprg ret;
  ret.factsz = factsz;
  memcpy(ret.instrs, instrs, sizeof(vprg::instr) * MAXNINSTR);
  ret.factsz = factsz;
  for (size_t i = 0; i < MAXCOLS; ++i) {
    if (attr[i] == NULL) continue;
    uint step = (max[i] - min[i]) / 3, **middle = ret.col_bmps[i].middle;
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

vprg recipe::candchk(const vprg::instr *instrs) const {
  vprg ret;
  ret.factsz = factsz;
  memcpy(ret.instrs, instrs, sizeof(vprg::instr) * MAXNINSTR);
  // column 0 - left bin left range -= step / 2
  if (attr[0] == nullptr) return ret;
  uint step = (max[0] - min[0]) / 3, **middle = ret.col_bmps[0].middle;
  if (step == 0)
    ret.col_bmps[0].leftmost = create_join(fk[0], attr[0], nbit[0], factsz,
                                           dimsz[0], min[0], max[0] + 1);
  else {
    auto l = std::max(min[0], step / 2) - step / 2;
    ret.col_bmps[0].leftmost =
      create_join(fk[0], attr[0], nbit[0], factsz, dimsz[0], l, min[0] + step);
    middle[0] = create_join(fk[0], attr[0], nbit[0], factsz, dimsz[0],
                            min[0] + step, min[0] + 2 * step);
    middle[1] = create_join(fk[0], attr[0], nbit[0], factsz, dimsz[0],
                            min[0] + 2 * step, max[0]);
  }

  // column 1 - right bin right range += step / 2
  if (attr[1] == nullptr) return ret;
  step = (max[1] - min[1]) / 3, middle = ret.col_bmps[1].middle;
  if (step == 0)
    ret.col_bmps[1].rightmost = create_join(fk[1], attr[1], nbit[1], factsz,
                                            dimsz[1], min[1] - 1, max[1]);
  else {
    ret.col_bmps[1].rightmost =
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
    uint step = (max[i] - min[i]) / 3, **middle = ret.col_bmps[i].middle;
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

} // namespace mybmpidx
