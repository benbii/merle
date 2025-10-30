#include "wahCpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t wahLongest1Fill(const uint32_t* b31, size_t nrWord) {
  size_t res = 0, cur = 0;
  for (size_t i = 0; i < nrWord; ++i) {
    uint32_t wah = b31[i];
    if (wah == 0x7fffffff)
      wah = 0xc0000001;
    if (wah >> 30 == 3) {
      cur += wah & 0x3fffffff;
      if (res < cur)
        res = cur;
    } else
      cur = 0;
  }
  return res;
}

size_t wahCmprsNo1CPU(const uint32_t *dec, size_t nrWord, uint32_t *out) {
  size_t i = 0;
  while (i < nrWord) {
    while (i < nrWord && dec[i] != 0)
      *out++ = dec[i++];
    uint32_t fillLen = 0;
    while (i < nrWord && dec[i] == 0)
      ++fillLen, ++i;
    *out++ = 0x80000000 + fillLen;
  }
  // The final 0-fill is discarded in `-1`
  return out - dec - 1;
}

size_t wahCmprsCPU(const uint32_t *dec, size_t nrWord, uint32_t *out) {
  size_t i = 0;
  while (i < nrWord) {
    while (i < nrWord && dec[i] != 0 && dec[i] != 0x7fffffff)
      *out++ = dec[i++];

    // 0 fills
    uint32_t fillLen = 0;
    while (i < nrWord && dec[i] == 0)
      ++fillLen, ++i;
    if (fillLen != 0)
      *out++ = 0x80000000 + fillLen;

    // 1 fills
    fillLen = 0;
    while (i < nrWord && dec[i] == 0x7fffffff)
      ++fillLen, ++i;
    if (fillLen != 0)
      *out++ = 0xc0000000 + fillLen;
  }
  return out - dec;
}

struct wahHost_s* loadArrayFile(const char* path) {
  FILE *file = fopen(path, "r");
  if (file == NULL) {
    (void)fprintf(stderr, "Cannot open %s\n", path);
    exit(1);
  }

  size_t maxBit = 0;
  while (!feof(file)) {
    size_t a;
    (void)fscanf(file, "%zu", &a);
    (void)fgetc(file);
    if (a > maxBit)
      maxBit = a;
  }

  maxBit = maxBit / 31 + 1; // maxBit means maxWord now
  (void)fseek(file, 0, SEEK_SET);
  struct wahHost_s* ret = calloc(sizeof(*ret) + 4 * maxBit, 1);
  ret->has1Fill = false; ret->magic = 0x44f8a1ef;
  ret->cmprsNrWord = maxBit; ret->decmprsNrWord = maxBit;

  while (!feof(file)) {
    size_t nthBit;
    (void)fscanf(file, "%zu", &nthBit);
    (void)fgetc(file);
    size_t inTile = nthBit % 31, tile = nthBit / 31;
    inTile = 1 << inTile;
    ret->dat[tile] |= inTile;
  }
  fclose(file);
  return ret;
}

struct wahHost_s* loadWahFile(const char* path) {
  FILE *file = fopen(path, "rb");
  if (file == NULL) {
    (void)fprintf(stderr, "Cannot open %s\n", path);
    exit(1);
  }

  (void)fseek(file, 0, SEEK_END);
  __auto_type retSz = ftell(file);
  struct wahHost_s* ret = malloc(retSz);
  (void)fseek(file, 0, SEEK_SET);
  (void)fread(ret, retSz, 1, file);
  if (ret->magic != 0x44f8a1ef)
    exit(fprintf(stderr, "Bad WAH file: %s\n", path));
  (void)fclose(file);
  return ret;
}

struct wahHost_s* wahDup(const struct wahHost_s* src, size_t nDup) {
  struct wahHost_s* res = malloc(sizeof(*src) + src->cmprsNrWord * nDup * 4 + 1);
  res->cmprsNrWord = src->cmprsNrWord * nDup;
  // res->cmprsNrWord = src->cmprsNrWord * nDup + 1;
  res->decmprsNrWord = src->decmprsNrWord * nDup;
  res->magic = src->magic;
  res->has1Fill = src->has1Fill;
  for (size_t i = 0; i < nDup; ++i) {
    size_t off = src->cmprsNrWord * i;
    memcpy(res->dat + off, src->dat, src->cmprsNrWord * 4);
  }
  // res->dat[src->cmprsNrWord * nDup] = 0x80010000;
  return res;
}

size_t simpleJoin(const uint32_t *fact, size_t factSize, const uint32_t *dimA,
                  const uint32_t *dimB, uint32_t min, uint32_t max, uint32_t *result) {
  size_t numWords = (factSize + 30) / 31;
  size_t wordIdx = 0, bitIdx = 0, curWord = 0;
  for (size_t i = 0; i < factSize; i++, bitIdx++) {
    if (31 == bitIdx) {
      bitIdx = 0;
      result[wordIdx++] = curWord;
      curWord = 0;
    }
    uint32_t dimValue = fact[i];
    if (dimA != NULL) {
      dimValue = dimA[dimValue];
      if (dimB != NULL)
        dimValue = dimB[dimValue];
    }
    curWord <<= 1;
    if (dimValue >= min && dimValue < max)
      curWord |= 1;
  }
  result[wordIdx] = curWord;
  return numWords;
}
