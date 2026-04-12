#include "ssbdemo.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static size_t __ldssb(void **ptr, const char *filename, const char *dirname,
                      size_t sz) {
  // sz = 0 => use ftell (minus 4-byte header)
  char filepath[256];
  sprintf(filepath, "%s/%s", dirname, filename);
  FILE *fp = fopen(filepath, "rb");
  assert(fp != NULL);
  if (sz == 0) {
    fseek(fp, 0, SEEK_END);
    sz = ftell(fp) - 4;  // exclude 4-byte element size header
    rewind(fp);
  }
  fseek(fp, 4, SEEK_SET);  // skip 4-byte element size header
  *ptr = malloc(sz);
  if (ptr == NULL) exit(fputs(__FILE__"host oom\n", stderr));
  size_t read_sz = fread(*ptr, 1, sz, fp);
  assert(read_sz == sz);
  fclose(fp);
  return sz;
}

static void* __cum(size_t sz) {
  void *bruh;
  cudaError_t e = cudaMalloc(&bruh, sz);
  if (e != cudaSuccess) exit(fputs(__FILE__" cuda oom\n", stderr));
  return bruh;
}

// Returns size of fact table (lo)
size_t ssb_load(struct ssb_schema *host, struct ssb_schema *dev,
                const char *ssbDirname) {
  // Load first column to determine fact table size
  size_t factsz =
      __ldssb((void **)&host->loCustKey, "loCustKey", ssbDirname, 0) /
      sizeof(uint32_t);
  // Load remaining lineorder fact table columns
  __ldssb((void **)&host->loPartKey, "loPartKey", ssbDirname,
          factsz * sizeof(uint32_t));
  __ldssb((void **)&host->loOrderDate, "loOrderDate", ssbDirname,
          factsz * sizeof(uint16_t));
  __ldssb((void **)&host->loExtendedPrice, "loExtendedPrice", ssbDirname,
          factsz * sizeof(uint32_t));
  __ldssb((void **)&host->loRevenue, "loRevenue", ssbDirname,
          factsz * sizeof(uint32_t));
  __ldssb((void **)&host->loSupplyCost, "loSupplyCost", ssbDirname,
          factsz * sizeof(uint32_t));
  __ldssb((void **)&host->loSuppCity, "loSuppCity", ssbDirname,
          factsz * sizeof(uint8_t));
  __ldssb((void **)&host->loQuantity, "loQuantity", ssbDirname,
          factsz * sizeof(uint8_t));
  __ldssb((void **)&host->loDiscount, "loDiscount", ssbDirname,
          factsz * sizeof(uint8_t));

  // Load dimension table columns and calculate table sizes
  size_t custCity_sz = __ldssb((void **)&host->custCity, "custCity", ssbDirname, 0);
  size_t custsz = custCity_sz / sizeof(uint8_t);  // Number of customers
  __ldssb((void **)&host->custMktSegment, "custMktSegment", ssbDirname, custsz * sizeof(uint8_t));
  
  size_t partName_sz = __ldssb((void **)&host->partName, "partName", ssbDirname, 0);
  size_t partsz = partName_sz / sizeof(uint16_t);  // Number of parts
  __ldssb((void **)&host->partMfgr, "partMfgr", ssbDirname, partsz * sizeof(uint16_t));
  __ldssb((void **)&host->partColor, "partColor", ssbDirname, partsz * sizeof(uint8_t));
  __ldssb((void **)&host->partType, "partType", ssbDirname, partsz * sizeof(uint8_t));
  __ldssb((void **)&host->partSize, "partSize", ssbDirname, partsz * sizeof(uint8_t));
  __ldssb((void **)&host->partContainer, "partContainer", ssbDirname, partsz * sizeof(uint8_t));

  if (dev) {
    // Allocate device memory
    dev->loCustKey = __cum(factsz * sizeof(uint32_t));
    dev->loPartKey = __cum(factsz * sizeof(uint32_t));
    dev->loOrderDate = __cum(factsz * sizeof(uint16_t));
    dev->loExtendedPrice = __cum(factsz * sizeof(uint32_t));
    dev->loRevenue = __cum(factsz * sizeof(uint32_t));
    dev->loSupplyCost = __cum(factsz * sizeof(uint32_t));
    dev->loSuppCity = __cum(factsz * sizeof(uint8_t));
    dev->loQuantity = __cum(factsz * sizeof(uint8_t));
    dev->loDiscount = __cum(factsz * sizeof(uint8_t));
    dev->custCity = __cum(custsz * sizeof(uint8_t));
    dev->custMktSegment = __cum(custsz * sizeof(uint8_t));
    dev->partName = __cum(partsz * sizeof(uint16_t));
    dev->partMfgr = __cum(partsz * sizeof(uint16_t));
    dev->partColor = __cum(partsz * sizeof(uint8_t));
    dev->partType = __cum(partsz * sizeof(uint8_t));
    dev->partSize = __cum(partsz * sizeof(uint8_t));
    dev->partContainer = __cum(partsz * sizeof(uint8_t));

    // Copy fact table data to device
    cudaMemcpy(dev->loCustKey, host->loCustKey, factsz * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev->loPartKey, host->loPartKey, factsz * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev->loOrderDate, host->loOrderDate, factsz * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev->loExtendedPrice, host->loExtendedPrice, factsz * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev->loRevenue, host->loRevenue, factsz * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev->loSupplyCost, host->loSupplyCost, factsz * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev->loSuppCity, host->loSuppCity, factsz * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev->loQuantity, host->loQuantity, factsz * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev->loDiscount, host->loDiscount, factsz * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev->custCity, host->custCity, custsz * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev->custMktSegment, host->custMktSegment, custsz * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev->partName, host->partName, partsz * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev->partMfgr, host->partMfgr, partsz * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev->partColor, host->partColor, partsz * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev->partType, host->partType, partsz * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev->partSize, host->partSize, partsz * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev->partContainer, host->partContainer, partsz * sizeof(uint8_t), cudaMemcpyHostToDevice);
  }
  return factsz;
}

void ssb_free(struct ssb_schema* host, struct ssb_schema* dev) {
  if (host) {
    // Free host memory
    free(host->loCustKey); free(host->loPartKey); free(host->loOrderDate);
    free(host->loExtendedPrice); free(host->loRevenue); free(host->loSupplyCost);
    free(host->loSuppCity); free(host->loQuantity); free(host->loDiscount);
    free(host->custCity); free(host->custMktSegment); free(host->partName);
    free(host->partMfgr); free(host->partColor); free(host->partType);
    free(host->partSize); free(host->partContainer);
  }
  if (dev) {
    // Free device memory
    cudaFree(dev->loCustKey); cudaFree(dev->loPartKey);
    cudaFree(dev->loOrderDate); cudaFree(dev->loExtendedPrice);
    cudaFree(dev->loRevenue); cudaFree(dev->loSupplyCost);
    cudaFree(dev->loSuppCity); cudaFree(dev->loQuantity);
    cudaFree(dev->loDiscount); cudaFree(dev->custCity);
    cudaFree(dev->custMktSegment); cudaFree(dev->partName);
    cudaFree(dev->partMfgr); cudaFree(dev->partColor);
    cudaFree(dev->partType); cudaFree(dev->partSize);
    cudaFree(dev->partContainer);
  }
}

// All ranges are left inclusive, right exclusive.
// SSB Q1 is aggregating without GROUP BY so it returns a value.
uint64_t s1ref(const struct ssb_schema *dat, size_t factsz, uint16_t dateMin,
               uint16_t dateMax, uint8_t discntMin, uint8_t discntMax,
               uint8_t qtyMin, uint8_t qtyMax) {
  uint64_t res = 0;
  for (size_t i = 0; i < factsz; ++i) {
    if (dat->loOrderDate[i] < dateMin || dat->loOrderDate[i] >= dateMax)
      continue;
    if (dat->loDiscount[i] < discntMin || dat->loDiscount[i] >= discntMax)
      continue;
    if (dat->loQuantity[i] < qtyMin || dat->loQuantity[i] >= qtyMax)
      continue;
    res += dat->loExtendedPrice[i] * dat->loDiscount[i];
  }
  return res;
}

void s2ref(const struct ssb_schema *dat, size_t factsz, uint32_t pMfgrMin,
           uint32_t pMfgrMax, uint8_t sCityMin, uint8_t sCityMax,
           uint32_t *grpby_out) {
  for (size_t i = 0; i < factsz; ++i) {
    // Filter by supplier city range
    if (dat->loSuppCity[i] < sCityMin || dat->loSuppCity[i] >= sCityMax)
      continue;
    // Join with part dimension to get manufacturer
    uint32_t partKey = dat->loPartKey[i];
    uint16_t mfgr = dat->partMfgr[partKey];  // Keys are zero-padded
    // Filter by manufacturer range
    if (mfgr < pMfgrMin || mfgr >= pMfgrMax) continue;
    uint32_t year = ssbDateToYear(dat->loOrderDate[i]);
    uint32_t grp = (mfgr - pMfgrMin) + year * (pMfgrMax - pMfgrMin);
    // Aggregate revenue into the group
    grpby_out[grp] += dat->loRevenue[i];
  }
}

void s3ref(const struct ssb_schema *dat, size_t factsz, uint8_t cCityMin,
           uint8_t cCityMax, uint8_t sCityMin, uint8_t sCityMax,
           uint16_t dateMin, uint16_t dateMax, uint32_t *grpby_out) {
  for (size_t i = 0; i < factsz; ++i) {
    uint16_t sCity = dat->loSuppCity[i];
    if (sCity < sCityMin || sCity >= sCityMax) continue;
    uint16_t date = dat->loOrderDate[i];
    if (date < dateMin || date >= dateMax) continue;
    // Join with customer dimension to get customer city
    uint32_t custKey = dat->loCustKey[i];
    uint16_t cCity = dat->custCity[custKey]; // zero-padded
    if (cCity < cCityMin || cCity >= cCityMax) continue;

    // GROUP BY group position calculation:
    uint8_t cCityMinScaled = cCityMin, cCityMaxScaled = cCityMax;
    uint8_t sCityMinScaled = sCityMin, sCityMaxScaled = sCityMax;
    if (cCityMax - cCityMin >= 50) {
      cCity /= 10; cCityMinScaled /= 10; cCityMaxScaled /= 10;
      sCity /= 10; sCityMinScaled /= 10; sCityMaxScaled /= 10;
    }
    const uint32_t year = ssbDateToYear(date);
    const uint32_t yearMin = ssbDateToYear(dateMin);
    const size_t a = cCityMaxScaled - cCityMinScaled;
    const size_t b = sCityMaxScaled - sCityMinScaled;
    const size_t grp = a * b * (year - yearMin) + a * (cCity - cCityMinScaled) +
                       (sCity - sCityMinScaled);
    // Aggregate revenue into the group
    grpby_out[grp] += dat->loRevenue[i];
  }
}

void s4ref(const struct ssb_schema *dat, size_t factsz, uint8_t cCityMin,
           uint8_t cCityMax, uint8_t sCityMin, uint8_t sCityMax,
           uint16_t pMfgrMin, uint16_t pMfgrMax, uint16_t dateMin,
           uint16_t dateMax, uint32_t *grpby_out) {
  for (size_t i = 0; i < factsz; ++i) {
    uint16_t sCity = dat->loSuppCity[i];
    if (sCity < sCityMin || sCity >= sCityMax) continue;
    uint16_t date = dat->loOrderDate[i];
    if (date < dateMin || date >= dateMax) continue;
    // Join with customer dimension to get customer city
    uint32_t custKey = dat->loCustKey[i];
    uint16_t cCity = dat->custCity[custKey];  // Zero-padded
    if (cCity < cCityMin || cCity >= cCityMax) continue;
    // Join with part dimension to get manufacturer
    uint32_t partKey = dat->loPartKey[i];
    uint16_t pMfgr = dat->partMfgr[partKey];
    if (pMfgr < pMfgrMin || pMfgr >= pMfgrMax) continue;

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
    const size_t a = sCityMaxScaled - sCityMinScaled;
    const size_t b = pMfgrMaxScaled - pMfgrMinScaled;
    const size_t grp = a * b * (year - yearMin) + a * (sCity - sCityMinScaled) +
                       (pMfgr - pMfgrMinScaled);
    grpby_out[grp] += dat->loRevenue[i] - dat->loSupplyCost[i];
  }
}

uint32_t *ssb_cpujoin(const struct ssb_schema *dat, size_t factSz) {
  uint32_t *res = calloc(SUMGRP_ALL + 1, sizeof(uint32_t));
  assert(res != NULL);

  #pragma omp parallel num_threads(13)
  {
    switch (omp_get_thread_num()) {
    case 0:
      res[0] = (uint32_t)s1ref(dat, factSz, SSBDATE_930101, SSBDATE_940101, 1, 4, 0, 25);
      break;
    case 1:
      res[1] = (uint32_t)s1ref(dat, factSz, SSBDATE_940101, SSBDATE_940201, 4, 7, 26, 36);
      break;
    case 2:
      res[2] = (uint32_t)s1ref(dat, factSz, SSBDATE_940204, SSBDATE_940211, 5, 8, 26, 36);
      break;

    case 3:
      s2ref(dat, factSz, 40, 80, 150, 200, res + SUMGRP_S1);
      break;
    case 4:
      s2ref(dat, factSz, 260, 268, 200, 250, res + SUMGRP_S1 + NGRP_S21);
      break;
    case 5:
      s2ref(dat, factSz, 260, 261, 50, 100,
            res + SUMGRP_S1 + NGRP_S21 + NGRP_S22);
      break;

    case 6:
      s3ref(dat, factSz, 200, 250, 200, 250, SSBDATE_920101, SSBDATE_980101,
            &res[SUMGRP_S2]);
      break;
    case 7:
      s3ref(dat, factSz, 190, 200, 190, 200, SSBDATE_920101, SSBDATE_980101,
            &res[SUMGRP_S2 + NGRP_S31]);
      break;
    case 8:
      s3ref(dat, factSz, 51, 55, 51, 55, SSBDATE_920101, SSBDATE_980101,
            &res[SUMGRP_S2 + NGRP_S31 + NGRP_S32]);
      break;
    case 9:
      s3ref(dat, factSz, 51, 55, 51, 55, SSBDATE_971201, SSBDATE_980101,
            &res[SUMGRP_S2 + NGRP_S31 + NGRP_S32 + NGRP_S33]);
      break;

    case 10:
      s4ref(dat, factSz, 150, 200, 150, 200, 0, 400, SSBDATE_920101, SSBDATE_990101,
            &res[SUMGRP_S3]);
      break;
    case 11:
      s4ref(dat, factSz, 150, 200, 150, 200, 0, 400, SSBDATE_970101, SSBDATE_990101,
            &res[SUMGRP_S3 + NGRP_S41]);
      break;
    case 12:
      s4ref(dat, factSz, 150, 200, 190, 200, 120, 160, SSBDATE_970101, SSBDATE_990101,
            &res[SUMGRP_S3 + NGRP_S41 + NGRP_S42]);
    }
  }
  return res;
}

bool ssb_demoall(const char *ssbDirname) {
  struct ssb_schema host_dat, dev_dat;
  size_t factSz = ssb_load(&host_dat, &dev_dat, ssbDirname);

  // Run the index creation and WAH baseline demo first
  ssb_wah(&host_dat, &dev_dat, factSz);
  struct ssb_bmp *bmp = malloc(sizeof(struct ssb_bmp));
  ssb_bmpcreate(&host_dat, &dev_dat, factSz, bmp);

  uint32_t *host_res = ssb_cpujoin(&host_dat, factSz), *dev_res;
  uint32_t *xfertmp = malloc(SUMGRP_ALL * sizeof(uint32_t));
  bool matches = true;

  dev_res = ssb_gpujoin(&dev_dat, factSz);
  fflush(stdout);
  cudaMemcpy(xfertmp, dev_res, SUMGRP_ALL * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < SUMGRP_ALL; i++) {
    if (host_res[i] != xfertmp[i]) {
      fprintf(stderr, "Join mismatch at %zu: host=%u, dev=%u\n", i, host_res[i],
              xfertmp[i]);
      matches = false; break;
    }
  }
  cudaFree(dev_res);

  // TODO: fixed-layout here
  ssb_bmpfree(bmp);
  free(bmp);

  dev_res = ssb_bmp_control_abl_fuse(&dev_dat, factSz);
  fflush(stdout);
  cudaMemcpy(xfertmp, dev_res, SUMGRP_ALL * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < SUMGRP_ALL; i++) {
    if (host_res[i] != xfertmp[i]) {
      fprintf(stderr, "BmpAbl mismatch at %zu: host=%u, dev=%u\n", i,
              host_res[i], xfertmp[i]);
      matches = false;
    }
  }
  cudaFree(dev_res);

  dev_res = ssb_bmp_control_abl_nofuse(&dev_dat, factSz);
  fflush(stdout);
  cudaMemcpy(xfertmp, dev_res, SUMGRP_ALL * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < SUMGRP_ALL; i++) {
    if (host_res[i] != xfertmp[i]) {
      fprintf(stderr, "BmpAbl mismatch at %zu: host=%u, dev=%u\n", i,
              host_res[i], xfertmp[i]);
      matches = false;
    }
  }
  cudaFree(dev_res);

  dev_res = ssb_bmp_control(&dev_dat, factSz);
  fflush(stdout);
  cudaMemcpy(xfertmp, dev_res, SUMGRP_ALL * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < SUMGRP_ALL; i++) {
    if (host_res[i] != xfertmp[i]) {
      fprintf(stderr, "Bmp mismatch at %zu: host=%u, dev=%u\n", i, host_res[i],
              xfertmp[i]);
      matches = false; break;
    }
  }
  cudaFree(dev_res);

  free(host_res);
  free(xfertmp);
  ssb_free(&host_dat, &dev_dat);
  return matches;
}
