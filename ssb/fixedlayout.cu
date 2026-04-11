#include "ssbdemo.h"
#include "../primitive.cuh"
using namespace mybmpidx;
static constexpr auto nodim = recipe::nodim;
static constexpr auto AND = vprg::AND, OR = vprg::OR, ANDM = vprg::ANDM,
                      ORM = vprg::ORM, END = vprg::END;

// vt0=8 is sufficient for SSB
static constexpr size_t nt = 256, vt = 3, vt0 = 8, ndup = 100;
static constexpr size_t nv_ = nt * vt, nv32 = nv_ * 32;
#ifdef LARGE_SMEM
static constexpr size_t cp_vt = vt, cp_nv32 = nv32;
#else
// Program + candchk is highly shmem intensive.
// Go for a less aggressive `vt` choice.
static constexpr size_t cp_vt = 2, cp_nv = nt * cp_vt, cp_nv32 = cp_nv * 32;
#endif

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
