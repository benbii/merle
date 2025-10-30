#include <stdint.h>
#include <stdbool.h>

#define likely_po(expr)   __builtin_expect(!!(expr), 1)
#define unlikely_po(expr) __builtin_expect(!!(expr), 0)
#define unpredictable_po  __builtin_unpredictable

const unsigned long fillTbl31[32] = {
    0b1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0b1100,
};
const unsigned long propTbl31[32] = {
  // Set previous fill count to 0 if two segment cannot merge
  0,0,0,0,0,0,0,0,0,0,0x3fffffff,0,0,0,0,0x3fffffff,
  // Set current output byte to a maximum-length fill if two segs can merge
  // 0,0,0,0,0,0,0,0,0,0,0x3fffffff,0,0,0,0,0x3fffffff,
  // Increment output pointer by 1 if two segments cannot merge
  1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0
};

#define FILL_BIT     0x80000000
#define FILLVAL_BIT  0x40000000
#define FILL_PART    (FILL_BIT | FILLVAL_BIT)
#define TAIL_MASK    0x7fffffff
#define FILLCNT_MASK 0x3fffffff

uint32_t *wahAndCPU(const uint32_t *upp, const uint32_t *upEnd,
                    const uint32_t *downp, const uint32_t *downEnd,
                    uint32_t *out, uint32_t *outEnd) {
  unsigned long up = (unsigned long)*upp, down = (unsigned long)*downp,
                prevFil = 0, prevCnt = 0;
  --out;

  while (upp < upEnd && downp < downEnd && out < outEnd) {
    unsigned long val = up & FILL_BIT ? 0 - ((up & FILLVAL_BIT) != 0) : up;
    unsigned long upCnt = up & FILL_BIT ? up & FILLCNT_MASK : 1;
    unsigned long downVal = down & FILL_BIT ? 0 - ((down & FILLVAL_BIT) != 0) : down;
    unsigned long downCnt = down & FILL_BIT ? down & FILLCNT_MASK : 1;
    val &= downVal;
    val &= TAIL_MASK;

    bool ge = downCnt >= upCnt;
    unsigned long outCnt = ge ? upCnt : downCnt;
    upp += ge;
    // For some reason, if the next line is written as:
    // unsigned long upNext = *(unsigned long*)upp;
    // which is the same damn thing, then clang compiles it to branched code:(
    // (short* or int* would not emit branch code either)
    unsigned long upNext = (unsigned long)*upp;
    up = ge ? upNext : up - downCnt;
    ge = upCnt >= downCnt;
    downp += ge;
    unsigned long downNext = (unsigned long)*downp;
    down = ge ? downNext : down - upCnt;
    // if (outCnt == 0)
    //   continue;

    prevFil >>= 2;
#ifdef __WAH_POPCNT
    prevFil |= fillTbl31[__builtin_popcount(val)];
#else
    (void)fillTbl31;
    unsigned long prevFilOr = val == 0 ? 0b1000 : 0;
    prevFilOr = val == 0x7fffffff ? 0b1100 : prevFilOr;
    prevFil |= prevFilOr;
#endif
    __auto_type addr = propTbl31 + prevFil;
    prevCnt &= addr[0];
    prevCnt += outCnt;
    if (unlikely_po(prevCnt & FILLVAL_BIT)) {
      *out++ |= FILLCNT_MASK;
      prevCnt -= FILLCNT_MASK;
    }
    out += addr[16];
    val = prevFil & 8 ? FILL_BIT + prevCnt + (val & FILLVAL_BIT) : val;
    *out = (uint32_t)val;
  }
  return (*out & FILL_BIT) && !(*out & FILLVAL_BIT) ? out : out + 1;
}

int *wahAndCPUint(const int *upp, const int *upEnd, const int *downp,
                  const int *downEnd, int *out, int *outEnd) {
  return (int *)wahAndCPU((const uint32_t *)upp, (const uint32_t *)upEnd,
                          (const uint32_t *)downp, (const uint32_t *)downEnd,
                          (uint32_t *)out, (uint32_t *)outEnd);
}
