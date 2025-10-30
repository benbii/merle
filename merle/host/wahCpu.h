#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct wahHost_s {
  bool has1Fill;
  uint32_t magic;
  uint64_t cmprsNrWord, decmprsNrWord;
  uint32_t dat[0];
};

struct wahHost_s* loadArrayFile(const char* path);
struct wahHost_s* loadWahFile(const char* path);
size_t simpleJoin(const uint32_t *fact, size_t factSize, const uint32_t *dimA,
                  const uint32_t *dimB, uint32_t min, uint32_t max, uint32_t *result);

size_t wahLongest1Fill(const uint32_t* b31, size_t nrWord);
size_t wahCmprsNo1CPU(const uint32_t *b31, size_t nrWord, uint32_t *out);
size_t wahCmprsCPU(const uint32_t *b31, size_t nrWord, uint32_t *out);
struct wahHost_s* wahDup(const struct wahHost_s* src, size_t nDup);
void wahPrint(const uint32_t* bah, size_t n32);

uint32_t *wahAndCPU(const uint32_t *upp, const uint32_t *upEnd,
                    const uint32_t *downp, const uint32_t *downEnd,
                    uint32_t *out, uint32_t *outEnd);
// GPU uses `int`
int *wahAndCPUint(const int *upp, const int *upEnd, const int *downp,
                  const int *downEnd, int *out, int *outEnd);

void wahGen(uint32_t *out, uint32_t *outEnd, size_t maxTail, size_t maxFill,
            size_t tailDens, size_t fillBitDens);
void wahPrint(const uint32_t* bah, size_t n32);
#ifdef __cplusplus
}
#endif