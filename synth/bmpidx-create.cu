#include "synthdemo.h"
#include "../primitive.cuh"
using namespace mybmpidx;

void synth_bmpcreate(size_t factSz, const struct synth_schema *dat,
                     struct synth_bmp *out) {
  const uint32_t maxFkey = factSz / 100;
  uint64_t factBounds[NBIN / 2 + 1], dimBounds[NBIN / 2 + 1];
  factBounds[0] = dimBounds[0] = 0;
  for (size_t i = 0; i < NBIN / 2 - 1; ++i) {
    dimBounds[i+1] = dimBounds[i] + maxFkey / NBIN;
    switch (dat->bitwidth) {
    case 64: default: abort();
    case 8: factBounds[i+1] = factBounds[i] + 256 / NBIN; break;
    case 16: factBounds[i+1] = factBounds[i] + 65536 / NBIN; break;
    case 32: factBounds[i+1] = factBounds[i] + 4294967296 / NBIN;
    }
  }

  create_bin(nullptr, dat->factattr1, dat->bitwidth, factSz, 0, factBounds,
             factBounds + 1, NBIN / 2, out->f1);
  create_bin(nullptr, dat->factattr2, dat->bitwidth, factSz, 0, factBounds,
             factBounds + 1, NBIN / 2, out->f2);
  create_bin(dat->fkey, dat->dimattr1, 32, factSz, maxFkey, dimBounds,
             dimBounds + 1, NBIN / 2, out->d1);

  memcpy(out->fBound, factBounds, sizeof(factBounds));
  memcpy(out->dBound, dimBounds, sizeof(dimBounds));
}

void synth_bmpfree(struct synth_bmp *dev) {
  cudaFree(dev->f1[0]);
  cudaFree(dev->f2[0]);
  cudaFree(dev->d1[0]);
}
