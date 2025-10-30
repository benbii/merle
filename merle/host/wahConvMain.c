#include "wahCpu.h"
#include "roaring.h"

uint32_t *loadColFile(const char *path, size_t *sz) {
  if (path == NULL) return NULL;
  FILE *f = fopen(path, "rb");
  if (f == NULL) return NULL;
  fseek(f, 0, SEEK_END);
  size_t sz_ = ftell(f) / sizeof(uint32_t);
  fseek(f, 0, SEEK_SET);
  uint32_t *dat = malloc(sz_ * sizeof(uint32_t));
  if (dat == NULL) { perror("malloc"); exit(1); }
  if (sz_ != fread(dat, sizeof(uint32_t), sz_, f))
    exit(fprintf(stderr, "partial read?\n"));
  fclose(f);
  if (sz != NULL) *sz = sz_;
  return dat;
}

typedef enum {
  BitsetToWah = 1,
  ArrToBitset = 2,
  ArrToWah = 3,
  Scan = 4
} Subcmd;

int main(int argc, const char* const* argv) {
  // const char* inFmt = "dataset/census-income/census-income.csv%zu.txt";
  // const char* inFmt = "dataset/census-income_srt/census-income_srt.csv%zu.txt";
  // const char* inFmt = "dataset/census1881/census1881.csv%zu.txt";
  // const char* inFmt = "dataset/census1881_srt/census1881_srt.csv%zu.txt";
  // const char* inFmt = "dataset/weather_sept_85/weather_sept_85.csv%zu.txt";
  const char* inFmt = "dataset/weather_sept_85_srt/weather_sept_85_srt.csv%zu.txt";
  const char* dim1ColFile = NULL, *dim2ColFile = NULL;
  // const char* inFmt = "dataset/wikileaks-noquotes/wikileaks-noquotes.csv%zu.txt";
  // const char* inFmt = "dataset/wikileaks-noquotes_srt/wikileaks-noquotes_srt.csv%zu.txt";
  const char* outFmt = "wahData/weaSrt/%zu.wah";
  Subcmd subcmd = ArrToWah;
  size_t unroll1Fill = 8, nFile = 1, nDup = 1;
  uint32_t scanMin = 0, scanMax = UINT32_MAX;

  if (0 == strcasecmp(argv[1], "arrToBitset"))
    subcmd = ArrToBitset;
  else if (0 == strcasecmp(argv[1], "BitsetToWah"))
    subcmd = BitsetToWah;
  else if (0 == strcasecmp(argv[1], "arrToWah"))
    subcmd = ArrToWah;
  else if (0 == strcasecmp(argv[1], "scanIntoBitset"))
    subcmd = Scan;
  else if (0 == strcasecmp(argv[1], "scanIntoWah"))
    subcmd = Scan | BitsetToWah;
  else return fputs("Invalid conversion type", stderr);

  for (int i = 2; i < argc; i += 2) {
    if (0 == strcasecmp(argv[i], "--inFmt") ||
        0 == strcasecmp(argv[i], "--factColFile"))
      inFmt = argv[1 + i];
    else if (0 == strcasecmp(argv[i], "--outFmt") ||
             0 == strcasecmp(argv[i], "--scanOut"))
      outFmt = argv[1 + i];
    else if (0 == strcasecmp(argv[i], "--dim1ColFile"))
      dim1ColFile = argv[1 + i];
    else if (0 == strcasecmp(argv[i], "--dim2ColFile"))
      dim2ColFile = argv[1 + i];
    else if (0 == strcasecmp(argv[i], "--scanMin"))
      scanMin = atoi(argv[1 + i]);
    else if (0 == strcasecmp(argv[i], "--scanMax"))
      scanMax = atoi(argv[1 + i]);
    else if (0 == strcasecmp(argv[i], "--nFile"))
      nFile = atol(argv[1 + i]);
    else if (0 == strcasecmp(argv[i], "--nDup"))
      nDup = atol(argv[1 + i]);
    else if (0 == strcasecmp(argv[i], "--unroll"))
      unroll1Fill = atol(argv[1 + i]);
    else return fputs("Invalid Argument", stderr);
  }
  if (strlen(inFmt) > 240)
    return fputs("Name too long!", stderr);
  if (strlen(outFmt) > 240)
    return fputs("Name too long!", stderr);

  struct wahHost_s *res = NULL;
  if (subcmd & 4) {
    size_t factSize;
    uint32_t *fact = loadColFile(inFmt, &factSize),
             *dim1 = loadColFile(dim1ColFile, NULL),
             *dim2 = loadColFile(dim2ColFile, NULL);
    if (fact == NULL)
      return fprintf(stderr, "Fact column is required.\n");

    size_t resSize = (factSize + 30) / 31;
    res = malloc(sizeof(*res) + resSize * sizeof(uint32_t) * 8);
    if (res == NULL) { perror("malloc"); return 1; }
    uint32_t *resDat = res->dat;
    res->magic = 0x44f8a1ef;
    res->has1Fill = false;
    res->cmprsNrWord = res->decmprsNrWord = resSize;
    #pragma omp parallel for num_threads(8)
    for (size_t i = 0; i < nDup; ++i)
      simpleJoin(fact, factSize, dim1, dim2, scanMin, scanMax,
                 resDat + (i % 8) * resSize);
    nFile = 1, nDup = 0;
  }

  for (size_t i = 0; i < nFile; ++i) {
    char path[256];
    sprintf(path, inFmt, i);
    if (res == NULL)
      res = subcmd & 2 ? loadArrayFile(path) : loadWahFile(path);

    uint32_t *arr = malloc(sizeof(uint32_t) * res->decmprsNrWord * 31);
    size_t arrNr = bitset_extract_setbits((uint64_t *)res->dat,
                                          res->decmprsNrWord / 2, arr, 0);
    #pragma omp parallel for num_threads(16)
    for (size_t i = 0; i < nDup; ++i) {
      roaring_bitmap_t *ra = roaring_bitmap_of_ptr(arrNr, arr);
      roaring_bitmap_run_optimize(ra);
      roaring_bitmap_free(ra);
    }

    roaring_bitmap_t *ra = roaring_bitmap_of_ptr(arrNr, arr);
    roaring_bitmap_run_optimize(ra);
    free(arr);
    size_t raSz = roaring_bitmap_size_in_bytes(ra);
    char *rabuf = malloc(raSz);
    roaring_bitmap_serialize(ra, rabuf);
    sprintf(path, outFmt, i, ".ra");
    FILE* outStream = fopen(path, "wb");
    fwrite(rabuf, raSz, 1, outStream);
    fclose(outStream);
    free(rabuf);

    if (subcmd & 1) {
      size_t len1Fill = wahLongest1Fill(res->dat, res->cmprsNrWord);
      if (len1Fill > unroll1Fill) {
        res->cmprsNrWord = wahCmprsCPU(res->dat, res->cmprsNrWord, res->dat);
        res->has1Fill = true;
        printf("%zu: %lu bits~\n", i, res->cmprsNrWord * 32);
      } else {
        res->cmprsNrWord = wahCmprsNo1CPU(res->dat, res->cmprsNrWord, res->dat);
        res->has1Fill = false;
        printf("%zu: %lu bits \n", i, res->cmprsNrWord * 32);
      }
    }

    // TODO: Function?
    sprintf(path, outFmt, i, ".wah");
    outStream = fopen(path, "wb");
    fwrite(res, sizeof(*res) + res->cmprsNrWord * 4, 1, outStream);
    fclose(outStream);
    free(res);
    res = NULL;
  }
  return 0;
}
