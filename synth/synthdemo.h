#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <vector_types.h> // cuda's float4

#ifdef __cplusplus
extern "C" {
#endif

// Fixed synthetic query:
// SELECT count(*) FROM fact, dim WHERE fact.fk = dim.key AND
// (fact.attr1 BETWEEN fa1low AND fa1hi-1 OR
//  fact.attr2 BETWEEN fa2low AND fa2hi-1) AND
// dim.attr1 BETWEEN da1low AND da1hi-1
// GROUP BY dim.attr2;
struct synth_schema {
  uint32_t bitwidth; // 8, 16, 32, 64
  void *factattr1, *factattr2;
  uint32_t *fkey;
  uint32_t *dimattr1; // 0,1,2,3,4... for easy selevtivity estimates
  uint8_t *dimattr2; // uniformly distributed between 0~255
  // Dimension key is implicitly self-incrementing
};

size_t synth_load(struct synth_schema *host, struct synth_schema *dev,
                  const char *synthDirname);
void synth_free(struct synth_schema* host, struct synth_schema* dev);

void synth_ref(const struct synth_schema *host, size_t factsz, uint64_t fa1low,
               uint64_t fa1hi, uint64_t fa2low, uint64_t fa2hi, uint64_t da1low,
               uint64_t da1hi, uint32_t grpout[256]);
void synth_join(const struct synth_schema *dev, size_t factsz, uint64_t fa1low,
               uint64_t fa1hi, uint64_t fa2low, uint64_t fa2hi, uint64_t da1low,
               uint64_t da1hi, uint32_t grpout[256]);
float4 synth_bmp(const struct synth_schema *dat, uint32_t factsz,
                 uint32_t fa1lo, uint32_t fa1hi, uint32_t fa2lo, uint32_t fa2hi,
                 uint32_t da1lo, uint32_t da1hi, uint32_t grpout[256], int ty);
float2 synth_method(const struct synth_schema *dat, uint32_t factsz,
                    uint32_t fa1lo, uint32_t fa1hi, uint32_t fa2lo,
                    uint32_t fa2hi, uint32_t da1lo, uint32_t da1hi);
void synth_wah(const struct synth_schema *dat, uint32_t factsz, uint32_t fa1lo,
               uint32_t fa1hi, uint32_t fa2lo, uint32_t fa2hi, uint32_t da1lo,
               uint32_t da1hi);


bool synth_demoall(const char *synthDirname);

/**
 * Calculate window probabilities for Zipf distribution
 * @param skewness The skewness parameter for Zipf distribution
 * @param n The number of windows to calculate
 * @param winsize The size of each window
 * @return Array of probabilities for windows [x, x+winsize) where x in [0, n)
 */
double *window_zipf(double skewness, size_t n, size_t winsize);

#ifdef __cplusplus
} /* extern "C" */
#endif
