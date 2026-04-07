#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  SSBDATE_920101 = 0,
  SSBDATE_920501 = 121,
  SSBDATE_920901 = 244,
  SSBDATE_930101 = 366,
  SSBDATE_930501 = 486,
  SSBDATE_930901 = 609,
  SSBDATE_940101 = 731,
  SSBDATE_940201 = 762,
  SSBDATE_940204 = 765,
  SSBDATE_940211 = 772,
  SSBDATE_940501 = 851,
  SSBDATE_940901 = 974,
  SSBDATE_950101 = 1096,
  SSBDATE_950501 = 1216,
  SSBDATE_950901 = 1339,
  SSBDATE_960101 = 1461,
  SSBDATE_960501 = 1582,
  SSBDATE_960901 = 1705,
  SSBDATE_970101 = 1827,
  SSBDATE_970501 = 1947,
  SSBDATE_970901 = 2070,
  SSBDATE_971201 = 2161,
  SSBDATE_980101 = 2192,
  SSBDATE_980501 = 2312,
  SSBDATE_980901 = 2435,
  SSBDATE_990101 = 2557,
};

#ifdef __CUDACC__
static inline __host__ __device__ uint32_t ssbDateToYear(uint16_t date) {
#else
static inline uint32_t ssbDateToYear(uint16_t date) {
#endif
  return date >= SSBDATE_980101 ? 6 :
         date >= SSBDATE_970101 ? 5 :
         date >= SSBDATE_960101 ? 4 :
         date >= SSBDATE_950101 ? 3 :
         date >= SSBDATE_940101 ? 2 :
         date >= SSBDATE_930101 ? 1 : 0;
}

struct ssb_schema {
  uint32_t *loCustKey, *loPartKey;
  uint16_t *loOrderDate;
  uint32_t *loExtendedPrice, *loRevenue, *loSupplyCost;
  uint8_t *loQuantity, *loDiscount, *loSuppCity;
  uint8_t *custMktSegment, *custCity;
  uint16_t *partName, *partMfgr;
  uint8_t *partColor, *partType, *partSize, *partContainer;
};

// Q1.{1,2,3} - 1 group each
// Q2.1 7*40=280, Q2.2 7*8=56, Q2.3 7*1=7
// Q3.1 5*5*6=150, Q3.2 10*10*6=600, Q3.3 4*4*6=96, Q3.4 4*4*1=16
// Q4.1 7*5*10=350, Q4.2 2*5*10=100, Q4.3 2*10*40=800
enum {
  NGRP_S11 = 1, NGRP_S12 = 1, NGRP_S13 = 1,
  NGRP_S21 = 280, NGRP_S22 = 56, NGRP_S23 = 7,
  NGRP_S31 = 150, NGRP_S32 = 600, NGRP_S33 = 96, NGRP_S34 = 16,
  NGRP_S41 = 350, NGRP_S42 = 100, NGRP_S43 = 800,
  SUMGRP_S1 = 3, SUMGRP_S2 = 3 + 280 + 56 + 7,
  SUMGRP_S3 = SUMGRP_S2 + 150 + 600 + 96 + 16,
  SUMGRP_ALL = SUMGRP_S3 + 350 + 100 + 800
};

size_t ssb_load(struct ssb_schema *host, struct ssb_schema *dev,
                const char *ssbDirname);
void ssb_free(struct ssb_schema* host, struct ssb_schema* dev);

uint32_t *ssb_demoref(const struct ssb_schema *dat, size_t factSz);
uint32_t *ssb_demojoin(const struct ssb_schema *dat, size_t factSz);
uint32_t *ssb_demobmp(const struct ssb_schema *dat, size_t factSz);
uint32_t *ssb_demobmp_abl(const struct ssb_schema *dat, size_t factSz);
void ssb_demowah(const struct ssb_schema *host, const struct ssb_schema *dat,
                 size_t factSz);
void ssb_democreate(const struct ssb_schema *host, const struct ssb_schema *dat,
                    size_t factSz);
bool ssb_demoall(const char *ssbDirname);

#ifdef __cplusplus
} /* extern "C" */
#endif
