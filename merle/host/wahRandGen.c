#include "nbitRandGen.h"
#include "wahCpu.h"
#include <stdio.h>
#include <stdlib.h>

void wahGen(uint32_t *out, uint32_t *outEnd, size_t maxTail, size_t maxFill,
            size_t tailDens, size_t fillBitDens) {
  _Static_assert(RAND_MAX >= (1 << 20), "RAND_MAX too small!");
  nbitRand32Gen_t tailGen = nbitRand32Init(tailDens);
  while (out < outEnd) {
    size_t r = rand();
    size_t nTail = (r % maxTail) + 1; r >>= 5;
    size_t nFill = (r % maxFill) + 1; r >>= 10;

    tailGen = nbitRand32Next(out, nTail, tailGen);
    for (size_t i = 0; i < nTail; ++i)
      out[i] &= 0x7fffffff;
    out += nTail;

    uint32_t fillPart = (r & 31) < fillBitDens ? 0xc0000000 : 0x80000000;
    // while (nFill > 0x3fffffff) {
    //   *out++ = fillPart + 0x3fffffff;
    //   nFill -= 0x3fffffff;
    // }
    *out++ = fillPart + nFill;
  }
}

void wahPrint(const uint32_t* bah, size_t n32) {
  for (size_t i = 0; i < n32; ++i) {
    uint32_t val = bah[i];
    switch (val >> 30) {
      case 0: case 1:
        printf("T%08x ", val & 0x7fffffff); break;
      case 2:
        printf("0%08x ", val & 0x3fffffff); break;
      case 3:
        printf("1%08x ", val & 0x3fffffff); break;
    }
  }
  putchar('\n');
}
