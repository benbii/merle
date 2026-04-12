#include "ssbdemo.h"
#include "../primitive.cuh"
using namespace mybmpidx;

// 6 supported columns whose min max boundary packed inside a 12-element array,
// plus a "sparsity" parameter (0 sparse, 1 balanced, 2 dense)
static vprg hardcoded_frontend(const ssb_bmp* bmpidx, const size_t bounds[13]) {
}
/* void s1fix(const ssb_schema *dat, const ssb_bmp *bmp, uint factsz,
           uint16_t dateMin, uint16_t dateMax, uint8_t discntMin,
           uint8_t discntMax, uint8_t qtyMin, uint8_t qtyMax, uint32_t *grp_out) {
  auto op = [=, dat = *dat] __device__ (uint i, bool chk) {
    uint2 ret; ret.y = ELIMINATED;
    // TODO: use __ldcs?
    if (chk && (dat.loOrderDate[i] < dateMin || dat.loOrderDate[i] >= dateMax))
      return ret;
    const uint8_t discnt = dat.loDiscount[i];
    if (chk && (discnt < discntMin || discnt >= discntMax))
      return ret;
    if (chk && (dat.loQuantity[i] < qtyMin || dat.loQuantity[i] >= qtyMax))
      return ret;
    ret.x = dat.loExtendedPrice[i] * discnt;
    ret.y = 0;
    return ret;
  };

  vprg s11 = {
      {
          {.middle = {bmp->dateSparse[1]}},
          {.leftmost = bmp->discntSparse[0], .rightmost = bmp->discntSparse[1]},
          {.middle = {bmp->qtySparse[0], bmp->qtySparse[1]}, .rightmost = bmp->qtySparse[2]},
      },
      factsz,
      {{ORM, 0, 1}, {ANDM, 0, 2}, {ANDM, 0, 3}}};
} */
