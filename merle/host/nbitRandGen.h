#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct nbitRand32Gen_s {
  uint32_t lSeed, rSeed;
  uint32_t rolStep, rorStep;
  uint32_t lMask, maskStep;
};
typedef struct nbitRand32Gen_s nbitRand32Gen_t;

// Uses `rand()`
uint32_t nbitRand32Dumb(size_t nBit);
// Does not use `rand()`
nbitRand32Gen_t nbitRand32Next(uint32_t* out, size_t n, nbitRand32Gen_t gen);

static inline nbitRand32Gen_t
nbitRand32SeedInit(uint32_t lSeed, uint32_t rSeed, uint32_t lMask) {
  nbitRand32Gen_t res = {.lSeed = lSeed, .rSeed = rSeed,
                         .rolStep = 3, .rorStep = 5,
                         .lMask = lMask, .maskStep = 7};
  return res;
}
static inline nbitRand32Gen_t nbitRand32Init(size_t nBit) {
  return nbitRand32SeedInit(nbitRand32Dumb(nBit), nbitRand32Dumb(nBit),
                            nbitRand32Dumb(16));
}

#ifdef __cplusplus
}
#endif
