#include <assert.h>
#include <getopt.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
_Static_assert(2147483647 == RAND_MAX, "2147483647 != RAND_MAX");
#define NTHREADS 16

void cust(size_t nr, uint8_t* city, uint8_t* mktSgmt, uint64_t* phone, uint32_t seed) {
  #pragma omp parallel for num_threads(NTHREADS)
  for (size_t i = 0; i < nr; ++i) {
    uint32_t myseed = seed + i;
    uint64_t x = ((uint64_t)rand_r(&myseed) << 31) + rand_r(&myseed);
    phone[i] = x % 10000000000ull; x /= 10000000000ull;
    assert(x < 1ull << 32ull);
    uint32_t x32 = (uint32_t)x;
    city[i] = x32 % 250; x32 /= 250;
    mktSgmt[i] = x32 % 5;
    // phone = nation_code (10-34) prepended to 10-digit random
    phone[i] += (city[i] / 10 + 10) * 10000000000ull;
  }
}
void supp(size_t nr, uint8_t* city, uint64_t* phone, uint32_t seed) {
  #pragma omp parallel for num_threads(NTHREADS)
  for (size_t i = 0; i < nr; ++i) {
    uint32_t myseed = seed + i;
    uint64_t x = ((uint64_t)rand_r(&myseed) << 31) + rand_r(&myseed);
    phone[i] = x % 10000000000ull; x /= 10000000000ull;
    assert(x < 1ull << 32ull);
    city[i] = (uint32_t)x % 250;
    // phone = nation_code (10-34) prepended to 10-digit random
    phone[i] += (city[i] / 10 + 10) * 10000000000ull;
  }
}
void part(size_t nr, uint16_t *name, uint16_t *mfgr, uint8_t *color,
          uint8_t *type, uint8_t *size, uint8_t *container, uint32_t seed) {
  #pragma omp parallel for num_threads(NTHREADS)
  for (size_t i = 0; i < nr; ++i) {
    uint32_t myseed = seed + i;
    int x = rand_r(&myseed);
    name[i] = x % 8836; x /= 8836;
    mfgr[i] = x % 1000; x /= 1000;
    color[i] = x % 94;
    x = rand_r(&myseed);
    type[i] = x % 215; x /= 215;
    size[i] = x % 50; x /= 50;
    container[i] = x % 40;
  }
}

struct ssb {
  uint32_t *loCustKey, *loPartKey, *loOrderDate;
  uint32_t *loExtendedPrice, *loRevenue, *loSupplyCost;
  uint8_t *loQuantity, *loDiscount, *loSuppCity;
  uint8_t *custMktSegment, *custCity;
  uint16_t *partName, *partMfgr;
  uint8_t *partColor, *partType, *partSize, *partContainer;
  // somehow not needed in ALL 13 queries lol
  uint32_t *loOrderKey, *loOrderPrice, *loCommitDate;
  uint32_t *loSuppKey, *loOrdTotalPrice;
  uint8_t *loLineNumber, *loOrderPriority, *loShipPriority;
  uint8_t *loTax, *loShipMode, *suppCity;
  uint64_t *custPhone, *suppPhone;
  size_t loNrRow, partNrRow, custNrRow, suppNrRow;
};

struct ssb ssb_create(size_t sf) {
  struct ssb res = {
    .loNrRow = 6000000 * sf, .partNrRow = 200000 * (1 + (size_t)log2(sf)),
    .suppNrRow = 2000 * sf, .custNrRow = 30000 * sf
  };
  // allocate dim tables: customer
  res.custCity = malloc(sizeof(*res.custCity) * res.custNrRow);
  res.custMktSegment = malloc(sizeof(*res.custMktSegment) * res.custNrRow);
  res.custPhone = malloc(sizeof(*res.custPhone) * res.custNrRow);
  // part
  res.partName = malloc(sizeof(*res.partName) * res.partNrRow);
  res.partMfgr = malloc(sizeof(*res.partMfgr) * res.partNrRow);
  res.partColor = malloc(sizeof(*res.partColor) * res.partNrRow);
  res.partType = malloc(sizeof(*res.partType) * res.partNrRow);
  res.partSize = malloc(sizeof(*res.partSize) * res.partNrRow);
  res.partContainer = malloc(sizeof(*res.partContainer) * res.partNrRow);
  // supplier
  res.suppCity = malloc(sizeof(*res.suppCity) * res.suppNrRow);
  res.suppPhone = malloc(sizeof(*res.suppPhone) * res.suppNrRow);

  // allocate 1 SF of line order table (reused for each SF chunk)
  res.loCustKey = malloc(sizeof(*res.loCustKey) * 6000000);
  res.loPartKey = malloc(sizeof(*res.loPartKey) * 6000000);
  res.loSuppKey = malloc(sizeof(*res.loSuppKey) * 6000000);
  res.loOrderDate = malloc(sizeof(*res.loOrderDate) * 6000000);
  res.loOrderKey = malloc(sizeof(*res.loOrderKey) * 6000000);
  res.loExtendedPrice = malloc(sizeof(*res.loExtendedPrice) * 6000000);
  res.loRevenue = malloc(sizeof(*res.loRevenue) * 6000000);
  res.loSupplyCost = malloc(sizeof(*res.loSupplyCost) * 6000000);
  res.loOrdTotalPrice = malloc(sizeof(*res.loOrdTotalPrice) * 6000000);
  res.loCommitDate = malloc(sizeof(*res.loCommitDate) * 6000000);
  res.loQuantity = malloc(sizeof(*res.loQuantity) * 6000000);
  res.loDiscount = malloc(sizeof(*res.loDiscount) * 6000000);
  res.loTax = malloc(sizeof(*res.loTax) * 6000000);
  res.loSuppCity = malloc(sizeof(*res.loSuppCity) * 6000000);
  res.loLineNumber = malloc(sizeof(*res.loLineNumber) * 6000000);
  res.loOrderPriority = malloc(sizeof(*res.loOrderPriority) * 6000000);
  res.loShipPriority = malloc(sizeof(*res.loShipPriority) * 6000000);
  res.loShipMode = malloc(sizeof(*res.loShipMode) * 6000000);

  if (res.loShipMode == NULL)
    exit(10);
  return res;
}

static uint32_t _ssbymd(uint32_t rng) {
  uint32_t m = rng % 12; rng /= 12;
  uint32_t y = rng % 7; rng /= 7;
  uint32_t r = 19920101 + m * 100 + y * 10000;
  switch (m) {
  case 0: case 2: case 4: case 6: case 7: case 9: case 11:
    return r + rng % 31;
  case 3: case 5: case 8: case 10:
    return r + rng % 30;
  case 1:
    if (y % 4 == 0) return r + rng % 29;
    return r + rng % 28;
  }
  __builtin_unreachable();
}
static uint32_t _ssbmd(uint32_t rng) {
  uint32_t m = rng % 12; rng /= 12;
  uint32_t r = 19920101 + m * 100;
  switch (m) {
  case 0: case 2: case 4: case 6: case 7: case 9: case 11:
    return r + rng % 31;
  case 3: case 5: case 8: case 10:
    return r + rng % 30;
  case 1:
    return r + rng % 28;
  }
  __builtin_unreachable();
}

// partKey, orderDate, orderPriority, shipPriority, ordTotalPrice: shared by order;
// custKey, suppkey, quantity, extendedPrice, discount, revenue, supplyCost,
// tax, shipMode, suppCity: exclusive to each row.
// commitDate: delta to orderDate exclusive to row, but orderDate isn't.
// Always generate a SF worth of data
size_t lo(const struct ssb *s, size_t id_start, uint32_t seed) {
  // Step 1, generate independent portions (easy to parallelize)
  #pragma omp parallel for num_threads(NTHREADS)
  for (size_t i = 0; i < 6000000; ++i) {
    uint32_t myseed = seed + i;
    int x = rand_r(&myseed);
    s->loExtendedPrice[i] = x % 55450;
    s->loDiscount[i] = x % 11; x /= 11;  // 0-10 per SSB spec
    s->loTax[i] = x % 9; x /= 9;
    s->loCommitDate[i] = _ssbmd(x);
    s->loRevenue[i] = s->loExtendedPrice[i] * (100 - s->loDiscount[i]) / 100;
    x = rand_r(&myseed);
    s->loSupplyCost[i] = x % 131072; x /= 131072;
    s->loShipMode[i] = x % 7;  // 7 ship modes per SSB spec
    s->loQuantity[i] = (x / 7) % 50 + 1;  // 1-50 per SSB spec
    s->loPartKey[i] = rand_r(&myseed) % s->partNrRow;
    s->loSuppKey[i] = rand_r(&myseed) % s->suppNrRow;
    // Pre-join lineorder with suppCity is a decent denormalization choice
    // and benefits 2 out of 4 query flights significantly (but less significant
    // than one might imagine since supp table is small)
    s->loSuppCity[i] = s->suppCity[s->loSuppKey[i]];
  }

  // Step 2, generate orders (parallelized with non-contiguous order keys)
  // Each thread gets 375K rows and 375K order ID space
  const size_t chunk = 6000000 / NTHREADS;  // 375000
  #pragma omp parallel num_threads(NTHREADS)
  {
    int tid = omp_get_thread_num();
    size_t row_start = tid * chunk;
    size_t row_end = row_start + chunk;
    size_t myid = id_start + tid * chunk - 1;
    size_t myrow = row_start;
    uint32_t myseed = seed + 0x9E3779B9 * tid;  // different seed per thread

    while (myrow < row_end) {
      int x = rand_r(&myseed), nLine = x % 8;
      x /= 8;
      ++myid;
      if (myrow + nLine > row_end)
        nLine = row_end - myrow;
      if (nLine == 0) continue;
      const uint32_t custkey = x % s->custNrRow;
      x = rand_r(&myseed);
      const uint8_t shipPriority = x % 4; x /= 4;
      const uint8_t orderPriority = x % 5 + 1; x /= 5;
      const uint32_t orderDate = _ssbymd(x);
      size_t ordTotalPrice = 0;
      for (int i = 0; i < nLine; ++i) {
        s->loCustKey[myrow + i] = custkey;
        s->loShipPriority[myrow + i] = shipPriority;
        s->loOrderPriority[myrow + i] = orderPriority;
        s->loOrderDate[myrow + i] = orderDate;
        s->loLineNumber[myrow + i] = i;
        s->loOrderKey[myrow + i] = myid;
        ordTotalPrice += s->loExtendedPrice[myrow + i];
      }
      for (int i = 0; i < nLine; ++i, ++myrow) {
        s->loOrdTotalPrice[myrow] = ordTotalPrice;
        while (s->loCommitDate[myrow] < orderDate)
          s->loCommitDate[myrow] += 10000;
      }
    }
  }
  // Return max possible order ID (conservative upper bound for next SF chunk)
  return id_start + 6000000;
}

void _dump_helper(uint32_t elemSz, size_t nrElem, const void *col, const char *d,
                  size_t dirlen, const char *colName, bool append) {
  char p[dirlen + 32];
  sprintf(p, "%s/%s", d, colName);
  FILE* f = fopen(p, append ? "ab" : "wb");
  if (!append) fwrite(&elemSz, 4, 1, f);
  // TODO: some error check? Disk might be full. Panic on error.
  fwrite(col, elemSz, nrElem, f);
  fclose(f);
}
#define D(col, nr) _dump_helper(sizeof(*s->col), nr, s->col, p, dirlen, #col, false);
void ssb_dim_dump(const struct ssb *s, const char *p, size_t dirlen) {
  D(custCity, s->custNrRow); D(custMktSegment, s->custNrRow);
  D(partName, s->partNrRow); D(partMfgr, s->partNrRow); D(partColor, s->partNrRow);
  D(partType, s->partNrRow); D(partSize, s->partNrRow);
  D(partContainer, s->partNrRow); D(suppCity, s->suppNrRow);
}
#undef D
#define D(col) _dump_helper(sizeof(*s->col), 6000000, s->col, p, dirlen, #col, app)
void ssb_fact_dump(const struct ssb *s, const char *p, size_t dirlen, bool app) {
  #pragma omp parallel num_threads(11)
  {
    switch (omp_get_thread_num()) {
      case 0: D(loCustKey); break;
      case 1: D(loPartKey); break;
      case 2: D(loSuppKey); break;
      case 3: D(loOrderDate); break;
      case 4: D(loOrderKey); break;
      case 5: D(loExtendedPrice); break;
      case 6: D(loRevenue); break;
      case 7: D(loSupplyCost); break;
      case 8: D(loQuantity); break;
      case 9: D(loDiscount); break;
      case 10: D(loSuppCity); break;
    }
  }
}
#undef D

// Apparently toad and toadette do not exist in vanilla SSB, yet the PDF
// says there are 94 colors
const char *ssbstr_color[94] = {
    "turquoise", "deep",       "frosted",   "seashell",  "blanched",
    "almond",    "lace",       "red",       "violet",    "cream",
    "drab",      "midnight",   "pale",      "blush",     "medium",
    "coral",     "tomato",     "bisque",    "beige",     "powder",
    "indian",    "cyan",       "honeydew",  "orange",    "sandy",
    "mint",      "aquamarine", "azure",     "lime",      "grey",
    "puff",      "burlywood",  "tan",       "sky",       "cornflower",
    "blue",      "cornsilk",   "burnished", "chiffon",   "white",
    "forest",    "wheat",      "gainsboro", "hot",       "antique",
    "saddle",    "linen",      "moccasin",  "rosy",      "light",
    "green",     "olive",      "black",     "ghost",     "ivory",
    "peach",     "smoke",      "royal",     "pink",      "slate",
    "navajo",    "dodger",     "yellow",    "steel",     "magenta",
    "lavender",  "rose",       "misty",     "spring",    "navy",
    "purple",    "khaki",      "thistle",   "firebrick", "lawn",
    "dim",       "chocolate",  "snow",      "metallic",  "lemon",
    "plum",      "brown",      "sienna",    "papaya",    "chartreuse",
    "dark",      "floral",     "peru",      "salmon",    "orchid",
    "maroon",    "goldenrod",  "toad",      "toadette"};
const char *ssbstr_city[250] = {
    "MOROCCO  0", "MOROCCO  1", "MOROCCO  2", "MOROCCO  3", "MOROCCO  4",
    "MOROCCO  5", "MOROCCO  6", "MOROCCO  7", "MOROCCO  8", "MOROCCO  9",
    "ETHIOPIA 0", "ETHIOPIA 1", "ETHIOPIA 2", "ETHIOPIA 3", "ETHIOPIA 4",
    "ETHIOPIA 5", "ETHIOPIA 6", "ETHIOPIA 7", "ETHIOPIA 8", "ETHIOPIA 9",
    "ALGERIA  0", "ALGERIA  1", "ALGERIA  2", "ALGERIA  3", "ALGERIA  4",
    "ALGERIA  5", "ALGERIA  6", "ALGERIA  7", "ALGERIA  8", "ALGERIA  9",
    "MOZAMBIQU0", "MOZAMBIQU1", "MOZAMBIQU2", "MOZAMBIQU3", "MOZAMBIQU4",
    "MOZAMBIQU5", "MOZAMBIQU6", "MOZAMBIQU7", "MOZAMBIQU8", "MOZAMBIQU9",
    "KENYA    0", "KENYA    1", "KENYA    2", "KENYA    3", "KENYA    4",
    "KENYA    5", "KENYA    6", "KENYA    7", "KENYA    8", "KENYA    9",
    "UNITED KI0", "UNITED KI1", "UNITED KI2", "UNITED KI3", "UNITED KI4",
    "UNITED KI5", "UNITED KI6", "UNITED KI7", "UNITED KI8", "UNITED KI9",
    "FRANCE   0", "FRANCE   1", "FRANCE   2", "FRANCE   3", "FRANCE   4",
    "FRANCE   5", "FRANCE   6", "FRANCE   7", "FRANCE   8", "FRANCE   9",
    "GERMANY  0", "GERMANY  1", "GERMANY  2", "GERMANY  3", "GERMANY  4",
    "GERMANY  5", "GERMANY  6", "GERMANY  7", "GERMANY  8", "GERMANY  9",
    "RUSSIA   0", "RUSSIA   1", "RUSSIA   2", "RUSSIA   3", "RUSSIA   4",
    "RUSSIA   5", "RUSSIA   6", "RUSSIA   7", "RUSSIA   8", "RUSSIA   9",
    "ROMANIA  0", "ROMANIA  1", "ROMANIA  2", "ROMANIA  3", "ROMANIA  4",
    "ROMANIA  5", "ROMANIA  6", "ROMANIA  7", "ROMANIA  8", "ROMANIA  9",
    "JORDAN   0", "JORDAN   1", "JORDAN   2", "JORDAN   3", "JORDAN   4",
    "JORDAN   5", "JORDAN   6", "JORDAN   7", "JORDAN   8", "JORDAN   9",
    "EGYPT    0", "EGYPT    1", "EGYPT    2", "EGYPT    3", "EGYPT    4",
    "EGYPT    5", "EGYPT    6", "EGYPT    7", "EGYPT    8", "EGYPT    9",
    "SAUDI ARA0", "SAUDI ARA1", "SAUDI ARA2", "SAUDI ARA3", "SAUDI ARA4",
    "SAUDI ARA5", "SAUDI ARA6", "SAUDI ARA7", "SAUDI ARA8", "SAUDI ARA9",
    "IRAN     0", "IRAN     1", "IRAN     2", "IRAN     3", "IRAN     4",
    "IRAN     5", "IRAN     6", "IRAN     7", "IRAN     8", "IRAN     9",
    "IRAQ     0", "IRAQ     1", "IRAQ     2", "IRAQ     3", "IRAQ     4",
    "IRAQ     5", "IRAQ     6", "IRAQ     7", "IRAQ     8", "IRAQ     9",
    "ARGENTINA0", "ARGENTINA1", "ARGENTINA2", "ARGENTINA3", "ARGENTINA4",
    "ARGENTINA5", "ARGENTINA6", "ARGENTINA7", "ARGENTINA8", "ARGENTINA9",
    "CANADA   0", "CANADA   1", "CANADA   2", "CANADA   3", "CANADA   4",
    "CANADA   5", "CANADA   6", "CANADA   7", "CANADA   8", "CANADA   9",
    "PERU     0", "PERU     1", "PERU     2", "PERU     3", "PERU     4",
    "PERU     5", "PERU     6", "PERU     7", "PERU     8", "PERU     9",
    "BRAZIL   0", "BRAZIL   1", "BRAZIL   2", "BRAZIL   3", "BRAZIL   4",
    "BRAZIL   5", "BRAZIL   6", "BRAZIL   7", "BRAZIL   8", "BRAZIL   9",
    "UNITED ST0", "UNITED ST1", "UNITED ST2", "UNITED ST3", "UNITED ST4",
    "UNITED ST5", "UNITED ST6", "UNITED ST7", "UNITED ST8", "UNITED ST9",
    "CHINA    0", "CHINA    1", "CHINA    2", "CHINA    3", "CHINA    4",
    "CHINA    5", "CHINA    6", "CHINA    7", "CHINA    8", "CHINA    9",
    "INDIA    0", "INDIA    1", "INDIA    2", "INDIA    3", "INDIA    4",
    "INDIA    5", "INDIA    6", "INDIA    7", "INDIA    8", "INDIA    9",
    "JAPAN    0", "JAPAN    1", "JAPAN    2", "JAPAN    3", "JAPAN    4",
    "JAPAN    5", "JAPAN    6", "JAPAN    7", "JAPAN    8", "JAPAN    9",
    "VIETNAM  0", "VIETNAM  1", "VIETNAM  2", "VIETNAM  3", "VIETNAM  4",
    "VIETNAM  5", "VIETNAM  6", "VIETNAM  7", "VIETNAM  8", "VIETNAM  9",
    "INDONESIA0", "INDONESIA1", "INDONESIA2", "INDONESIA3", "INDONESIA4",
    "INDONESIA5", "INDONESIA6", "INDONESIA7", "INDONESIA8", "INDONESIA9"};
const char *ssbstr_nation[250 / 10] = {
    "MOROCCO",   "ETHIOPIA", "ALGERIA",   "MOZAMBIQU", "KENYA",
    "UNITED KI", "FRANCE",   "GERMANY",   "RUSSIA",    "ROMANIA",
    "JORDAN",    "EGYPT",    "SAUDI ARA", "IRAN",      "IRAQ",
    "ARGENTINA", "CANADA",   "PERU",      "BRAZIL",    "UNITED ST",
    "CHINA",     "INDIA",    "JAPAN",     "VIETNAM",   "INDONESIA"};
const char *ssbstr_region[250 / 50] = {
  "AFRICA", "EUROPE", "MIDDLE EAST", "AMERICA", "ASIA"
};
const char *ssbstr_mktsgmt[5] = {
  "AUTOMOBILE", "HOUSEHOLD", "BUILDING", "MACHINERY", "FURNITURE"
};
const char *ssbstr_container0[5] = {"JUMBO", "LG", "WRAP", "MED", "SM"};
const char *ssbstr_container1[8] = {"PKG",  "CASE", "DRUM", "BAG",
                                    "PACK", "CAN",  "BOX",  "JAR"};
const char *ssbstr_type0[6] = {"PROMO", "LARGE",  "STANDARD",
                               "SMALL", "MEDIUM", "ECONOMY"};
const char *ssbstr_type1[6] = {"POLISHED", "PLATED",    "ANODIZED",
                               "BRUSHED",  "BURNISHED", "MATTE"};
const char *ssbstr_type2[6] = {"COPPER", "BRASS",  "TIN",
                               "STEEL",  "NICKEL", "CHROME"};
const char *ssbstr_shipmode[7] = {"TRUCK", "MAIL", "REG AIR", "AIR",
                                  "FOB",   "RAIL", "SHIP"};
const char *ssbstr_ordprio[6] = {"0-ERROR",  "1-URGENT",    "2-HIGH",
                                 "3-MEDIUM", "4-NOT SPECI", "5-LOW"};

void ssb_dim_txtdump(const struct ssb *s, const char *d, size_t dirlen) {
  char p[dirlen + 32]; sprintf(p, "%s/%s", d, "supplier.tbl");
  FILE* f = fopen(p, "w"); assert(f);
  for (size_t i = 0; i < s->suppNrRow; ++i) {
    uint8_t c = s->suppCity[i];
    uint64_t p = s->suppPhone[i];
    fprintf(f, "%zu|Supplier#%09zu|x|%s|%s|%s|%02zu-%03zu-%03zu-%04zu|\n", i, i,
            ssbstr_city[c], ssbstr_nation[c / 10], ssbstr_region[c / 50],
            p / 10000000000, p / 10000000 % 1000, p / 10000 % 1000, p % 10000);
  }
  fclose(f);

  sprintf(p, "%s/%s", d, "customer.tbl");
  f = fopen(p, "w"); assert(f);
  for (size_t i = 0; i < s->custNrRow; ++i) {
    uint8_t c = s->custCity[i];
    uint64_t p = s->custPhone[i];
    fprintf(f, "%zu|Customer#%09zu|x|%s|%s|%s|%02zu-%03zu-%03zu-%04zu|%s|\n", i,
            i, ssbstr_city[c], ssbstr_nation[c / 10], ssbstr_region[c / 50],
            p / 10000000000, p / 10000000 % 1000, p / 10000 % 1000, p % 10000,
            ssbstr_mktsgmt[s->custMktSegment[i]]);
  }
  fclose(f);

  sprintf(p, "%s/%s", d, "part.tbl");
  f = fopen(p, "w"); assert(f);
  for (size_t i = 0; i < s->partNrRow; ++i) {
    uint16_t name = s->partName[i], mfgr = s->partMfgr[i];
    uint8_t ty = s->partType[i], cn = s->partContainer[i];
    // mfgr encodes MFGR#(1-5), Category (1-5), Brand (1-40)
    // mfgr = mfgr1 * 200 + cat * 40 + brand
    uint8_t m1 = mfgr / 200 + 1, m2 = (mfgr / 40) % 5 + 1, m3 = mfgr % 40 + 1;
    fprintf(
        f, "%zu|%s %s|MFGR#%u|MFGR#%u%u|MFGR#%u%u%02u|%s|%s %s %s|%u|%s %s|\n",
        i, ssbstr_color[name / 94], ssbstr_color[name % 94], m1, m1, m2,
        m1, m2, m3, ssbstr_color[s->partColor[i]], ssbstr_type0[ty / 36],
        ssbstr_type1[(ty / 6) % 6], ssbstr_type2[ty % 6], s->partSize[i] + 1,
        ssbstr_container0[cn % 5], ssbstr_container1[cn / 5]);
  }
  fclose(f);
}

void ssb_fact_txtdump(const struct ssb *s, const char *d, size_t dirlen) {
  char p[dirlen + 32];
  sprintf(p, "%s/%s", d, "lineorder.tbl");
  FILE* f = fopen(p, "a"); assert(f);
  for (size_t i = 0; i < 6000000; ++i) {
    // orderkey|linenumber|custkey|partkey|suppkey|orderdate|orderpriority|
    // shippriority|quantity|extendedprice|ordtotalprice|discount|revenue|
    // supplycost|tax|commitdate|shipmode|
    fprintf(f, "%u|%u|%u|%u|%u|%u|%s|%u|%u|%u|%u|%u|%u|%u|%u|%u|%s|\n",
            s->loOrderKey[i], s->loLineNumber[i], s->loCustKey[i],
            s->loPartKey[i], s->loSuppKey[i], s->loOrderDate[i],
            ssbstr_ordprio[s->loOrderPriority[i]], s->loShipPriority[i],
            s->loQuantity[i], s->loExtendedPrice[i], s->loOrdTotalPrice[i],
            s->loDiscount[i], s->loRevenue[i], s->loSupplyCost[i],
            s->loTax[i], s->loCommitDate[i], ssbstr_shipmode[s->loShipMode[i]]);
  }
  fclose(f);
}

int main(int argc, char** argv) {
  size_t sf = 1;
  uint32_t seed = time(NULL);
  const char *outdir = ".";
  bool dump_text = false, dump_bin = false;

  int opt;
  while ((opt = getopt(argc, argv, "s:d:r:tb")) != -1) {
    switch (opt) {
    case 's': sf = atoll(optarg); break;
    case 'd': outdir = optarg; break;
    case 'r': seed = atoi(optarg); break;
    case 't': dump_text = true; break;
    case 'b': dump_bin = true; break;
    default:
      fprintf(stderr,
              "Usage: %s -s <SF> -d <outdir> [-r seed] [-t] [-b]\n"
              "  -s SF      Scale factor (default: 1)\n"
              "  -d outdir  Output directory (default: current dir)\n"
              "  -r seed    RNG seed (default: time)\n"
              "  -t         Dump text format (.tbl files)\n"
              "  -b         Dump binary format (columnar files)\n"
              "  If neither -t nor -b specified, defaults to -t\n", argv[0]);
      exit(1);
    }
  }
  if (!dump_text && !dump_bin)
    dump_text = true;

  (void)mkdir(outdir, 0755);
  size_t dirlen = strlen(outdir) + 1;
  struct ssb s = ssb_create(sf);

  // Generate dimensions
  cust(s.custNrRow, s.custCity, s.custMktSegment, s.custPhone, seed);
  seed *= 16807;
  part(s.partNrRow, s.partName, s.partMfgr, s.partColor, s.partType,
       s.partSize, s.partContainer, seed);
  seed *= 16807;
  supp(s.suppNrRow, s.suppCity, s.suppPhone, seed);
  seed *= 16807;
  if (dump_bin) ssb_dim_dump(&s, outdir, dirlen);
  if (dump_text) ssb_dim_txtdump(&s, outdir, dirlen);

  // Generate and dump lineorder fact table (1 SF at a time)
  size_t myid = lo(&s, 0, seed);
  if (dump_bin) ssb_fact_dump(&s, outdir, dirlen, false);
  if (dump_text) ssb_fact_txtdump(&s, outdir, dirlen);
  for (size_t i = 1; i < sf; ++i) {
    seed *= 16807;
    myid = lo(&s, myid, seed);
    if (dump_bin) ssb_fact_dump(&s, outdir, dirlen, true);
    if (dump_text) ssb_fact_txtdump(&s, outdir, dirlen);
  }
  return 0;
}
