#include "nbitRandGen.h"
#include <byteswap.h>
#include <stdlib.h>

uint32_t nbitRand32Dumb(size_t nBit) {
  uint32_t res = 0;
  size_t curBit = 0;
  while (curBit < nBit) {
    uint32_t b = 1 << (rand() & 31);
    curBit += ((b & res) == 0);
    res |= b;
  }
  return res;
}

static inline uint32_t rotl32(uint32_t n, uint32_t c) {
  c &= 31;
  return (n << c) | (n >> ((-c) & 31));
}

static inline uint32_t rotr32(uint32_t n, uint32_t c) {
  c &= 31;
  return (n >> c) | (n << ((-c) & 31));
}

nbitRand32Gen_t nbitRand32Next(uint32_t* out, size_t n, nbitRand32Gen_t gen) {
  while (n--) {
    gen.lSeed = rotl32(gen.lSeed, gen.rolStep);
    gen.rSeed = rotr32(gen.rSeed, gen.rorStep);
    gen.lMask = rotl32(gen.lMask, gen.maskStep);
    gen.rolStep += 3; gen.rolStep += 5; gen.maskStep += 7;
    gen.lSeed = bswap_32(gen.lSeed);
    gen.rSeed = bswap_32(gen.rSeed);
    gen.lMask = bswap_32(gen.lMask);

    uint32_t rMask = ~gen.lMask;
    uint32_t lPart = gen.lMask & gen.lSeed, rPart = rMask & gen.rSeed;
    uint32_t lRem = rMask & gen.lSeed, rRem = gen.lMask & gen.rSeed;
    gen.rSeed = lPart | rPart;
    gen.lSeed = lRem | rRem;
    *out++ = gen.rSeed;
  }
  return gen;
}

// int main() {
//   srand(time(NULL));
//   nbitRand32Gen_t gen = nbitRand32Init(25);
//   uint32_t *out = malloc(4 * 10000000);
//   gen = nbitRand32Next(out, 10000000, gen);
//   for (size_t i = 0; i < 128; ++i)
//     printf("%08x%c", out[i], i % 16 == 15 ? '\n' : ' ');
//   size_t nBit = 0;
//   for (size_t i = 0; i < 10000000; ++i)
//     nBit += __builtin_popcount(out[i]);
//   printf("%zu bits\n", nBit);
// }
