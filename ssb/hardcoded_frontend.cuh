#include "ssbdemo.h"
#include "../primitive.cuh"
using namespace mybmpidx;

// lambdas used across files
struct s1op { // not slop I swear :D
  uint16_t dateMin, dateMax, *loOrderDate;
  uint8_t discntMin, discntMax, *loDiscount;
  uint8_t qtyMin, qtyMax, *loQuantity;
  uint32_t *extendedPrice;
  uint2 __device__ operator()(uint i, bool chk = true) {
    uint2 ret; ret.y = ELIMINATED;
    // TODO: use __ldcs?
    if (chk && (loOrderDate[i] < dateMin || loOrderDate[i] >= dateMax))
      return ret;
    const uint8_t discnt = loDiscount[i];
    if (chk && (discnt < discntMin || discnt >= discntMax))
      return ret;
    if (chk && (loQuantity[i] < qtyMin || loQuantity[i] >= qtyMax))
      return ret;
    ret.x = extendedPrice[i] * discnt;
    ret.y = 0;
    return ret;
  }
};

struct s2op {
  uint32_t *loPartKey;
  uint16_t pMfgrMin, pMfgrMax, *partMfgr;
  uint8_t sCityMin, sCityMax, *loSuppCity;
  uint16_t *loOrderDate;
  uint32_t *loRevenue;
  uint2 __device__ operator()(uint i, bool chk = true) {
    uint2 ret; ret.y = ELIMINATED;
    // Filter by supplier city range
    if (chk && (loSuppCity[i] < sCityMin || loSuppCity[i] >= sCityMax))
      return ret;
    // Join with part dimension to get manufacturer
    uint32_t partKey = loPartKey[i];
    uint16_t mfgr = partMfgr[partKey];  // Keys are zero-padded
    // Filter by manufacturer range
    if (chk && (mfgr < pMfgrMin || mfgr >= pMfgrMax))
      return ret;
    uint32_t year = ssbDateToYear(loOrderDate[i]);
    ret.y = (mfgr - pMfgrMin) + year * (pMfgrMax - pMfgrMin);
    ret.x = loRevenue[i];
    return ret;
  }
};

struct s3op {
  uint32_t *loCustKey;
  uint8_t cCityMin, cCityMax, *custCity;
  uint8_t sCityMin, sCityMax, *loSuppCity;
  uint16_t dateMin, dateMax, *loOrderDate;
  uint32_t *loRevenue;
  uint2 __device__ operator()(uint i, bool chk = true) {
    uint2 ret; ret.y = ELIMINATED;
    // Filter by supplier city range
    uint8_t sCity = loSuppCity[i];
    if (chk && (sCity < sCityMin || sCity >= sCityMax))
      return ret;
    // Filter by date range
    uint16_t date = loOrderDate[i];
    if (chk && (date < dateMin || date >= dateMax))
      return ret;
    // Join with customer dimension to get customer city
    uint32_t custKey = loCustKey[i];
    uint8_t cCity = custCity[custKey];
    if (chk && (cCity < cCityMin || cCity >= cCityMax))
      return ret;

    // Apply downscaling if needed (for Q3.1)
    uint8_t cCityMinScaled = cCityMin, cCityMaxScaled = cCityMax;
    uint8_t sCityMinScaled = sCityMin, sCityMaxScaled = sCityMax;
    if (cCityMax - cCityMin >= 50) {
      cCity /= 10; cCityMinScaled /= 10; cCityMaxScaled /= 10;
      sCity /= 10; sCityMinScaled /= 10; sCityMaxScaled /= 10;
    }
    const uint32_t year = ssbDateToYear(date);
    const uint32_t yearMin = ssbDateToYear(dateMin);
    const uint a = cCityMaxScaled - cCityMinScaled;
    const uint b = sCityMaxScaled - sCityMinScaled;
    ret.y = a * b * (year - yearMin) + a * (cCity - cCityMinScaled) +
            (sCity - sCityMinScaled);
    ret.x = loRevenue[i];
    return ret;
  }
};

struct s4op {
  uint32_t *loCustKey, *loPartKey;
  uint8_t cCityMin, cCityMax, *custCity;
  uint8_t sCityMin, sCityMax, *loSuppCity;
  uint16_t pMfgrMin, pMfgrMax, *partMfgr;
  uint16_t dateMin, dateMax, *loOrderDate;
  uint32_t *loRevenue, *loSupplyCost;

  uint2 __device__ operator()(uint i, bool chk = true) {
    uint2 ret; ret.y = ELIMINATED;
    // Filter by supplier city range
    uint8_t sCity = loSuppCity[i];
    if (chk && (sCity < sCityMin || sCity >= sCityMax))
      return ret;
    // Filter by date range
    uint16_t date = loOrderDate[i];
    if (chk && (date < dateMin || date >= dateMax))
      return ret;
    // Join with customer dimension to get customer city
    if (chk) {
      uint32_t custKey = loCustKey[i];
      uint8_t cCity = custCity[custKey];
      if (cCity < cCityMin || cCity >= cCityMax)
        return ret;
    }
    // Join with part dimension to get manufacturer
    uint32_t partKey = loPartKey[i];
    uint16_t pMfgr = partMfgr[partKey];
    if (chk && (pMfgr < pMfgrMin || pMfgr >= pMfgrMax))
      return ret;

    // Apply downscaling based on ranges
    uint8_t sCityMinScaled = sCityMin, sCityMaxScaled = sCityMax;
    if (sCityMax - sCityMin >= 50) {
      sCity /= 10; sCityMinScaled /= 10; sCityMaxScaled /= 10;
    }
    uint16_t pMfgrMinScaled = pMfgrMin, pMfgrMaxScaled = pMfgrMax;
    if (pMfgrMax - pMfgrMin >= 200) {
      pMfgr /= 40; pMfgrMinScaled /= 40; pMfgrMaxScaled /= 40;
    }
    const uint32_t year = ssbDateToYear(date);
    const uint32_t yearMin = ssbDateToYear(dateMin);
    const uint a = sCityMaxScaled - sCityMinScaled;
    const uint b = pMfgrMaxScaled - pMfgrMinScaled;
    ret.y = a * b * (year - yearMin) + a * (sCity - sCityMinScaled) + (pMfgr - pMfgrMinScaled);
    ret.x = loRevenue[i] - loSupplyCost[i];
    return ret;
  };
};
