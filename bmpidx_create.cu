#include "primitive.cuh"
#include <cassert>

static void* __cum(size_t sz) {
  void *bruh;
  cudaError_t e = cudaMalloc(&bruh, sz);
  if (e != cudaSuccess) exit(fputs(__FILE__" cuda oom\n", stderr));
  return bruh;
}

namespace mybmpidx {
// ONLY cudaFree out[0]!!!
// `out` is a host array of pointers to device addresses
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

col col::from_bin(size_t nbin, uint** bins, const size_t *bounds, size_t min, size_t max) {
  col res;
  if (nbin == 0 || min >= max || *bounds > min || bounds[nbin] < max)
    return res;
  while (bounds[1] <= min) // bins[0] is the leftmost bin
    --nbin, ++bins, ++bounds;
  while (bounds[nbin - 1] >= max) // bins[nbin-1] is the rightmost bin
    --nbin;
  assert(nbin > 0);

  if (nbin == 1) {
    res.middle[0] = bounds[0] == min && bounds[1] == max ? bins[0] : nullptr;
    res.leftmost = bounds[0] == min && bounds[1] == max ? nullptr : bins[0];
    return res;
  }

  if (bounds[0] != min) {
    res.leftmost = bins[0];
    ++bounds, ++bins, --nbin;
  }
  if (bounds[nbin] != max)
    res.rightmost = bins[--nbin];
  // the rest are middle bins, assuming count < MAXBIN_PERCOL
  memcpy(res.middle, bins, nbin * sizeof(intptr_t));
  return res;
}

col col::from_bin2(size_t nbin, uint *bins[], const size_t bounds[],
                   size_t nf, uint *fine_bins[], const size_t fbounds[],
                   size_t min, size_t max) {
  // Nah for SSB go for secondary when primary requires candidate checking
  // happens to yield optimal plan on all cases.
  col res = from_bin(nbin, bins, bounds, min, max);
  if (res.leftmost || res.rightmost)
    return from_bin(nf, fine_bins, fbounds, min, max);
  return res;
  // Full code:

  /* col res;
  if (nbin == 0 || min >= max || *bounds > min || bounds[nbin] < max)
    return res;
  while (bounds[1] <= min) // bins[0] is the leftmost bin
    --nbin, ++bins, ++bounds;
  while (bounds[nbin - 1] >= max) // bins[nbin-1] is the rightmost bin
    --nbin;
  assert(nbin > 0);
  if (nbin == 1) {
    if (bounds[0] != min || bounds[1] != max) {
      res = from_bin(nf, fine_bins, fbounds, min, max);
      if (res.leftmost || res.middle[0]) // succeeded
        return res;
    }
    res.middle[0] = bins[0];
    return res;
  }

  uint **middle = res.middle, **smid;
  while (*middle)
    ++middle;
  col scratch = from_bin(nf, fine_bins, fbounds, min, bounds[1]);
  if (scratch.rightmost == nullptr) {
    res.leftmost = scratch.leftmost;
    smid = scratch.middle;
    while (*smid)
      *middle++ = *smid++;
    min = bounds[1];
  }
  scratch = from_bin(nf, fine_bins, fbounds, bounds[nbin - 1], max);
  if (scratch.leftmost == nullptr) {
    res.rightmost = scratch.rightmost;
    smid = scratch.middle;
    while (*smid)
      *middle++ = *smid++;
    max = bounds[nbin - 1];
  }
  scratch = from_bin(nbin, bins, bounds, min, max);
  smid = scratch.middle;
  while (*smid)
    *middle++ = *smid++;
  if (scratch.leftmost) {
    assert(res.leftmost == nullptr);
    res.leftmost = scratch.leftmost;
  }
  if (scratch.rightmost) {
    assert(res.rightmost == nullptr);
    res.rightmost = scratch.rightmost;
  }
  return res; */
}
} // namespace mybmpidx
