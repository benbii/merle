#include "synthdemo.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static size_t __ldsynth(void **ptr, const char *filename, const char *dirname,
                        size_t sz) {
  // NOTE: synthetic fact table columns has variable element bit width,
  // identified by its first 4 bytes (8, 16, 32, 64)
  char filepath[256];
  sprintf(filepath, "%s/%s", dirname, filename);
  FILE *fp = fopen(filepath, "rb");
  assert(fp != NULL);

  if (sz == 0) {
    fseek(fp, 0, SEEK_END);
    sz = ftell(fp);
    rewind(fp);
  }

  // All column files have a 4-byte header that must be skipped
  fseek(fp, 4, SEEK_SET); // Skip the 4-byte header
  sz -= sizeof(uint32_t); // Adjust size to exclude the header

  *ptr = malloc(sz);
  assert(*ptr != NULL);
  size_t read_sz = fread(*ptr, 1, sz, fp);
  if (read_sz != sz) exit(66);
  fclose(fp);
  return sz;
}

static void* __cum(size_t sz) {
  void *bruh;
  cudaError_t e = cudaMalloc(&bruh, sz);
  if (e != cudaSuccess)
    exit(fprintf(stderr, __FILE__ " cuda oom"));
  return bruh;
}

size_t synth_load(struct synth_schema *host, struct synth_schema *dev,
                  const char *dir) {
  size_t factFk_sz, factAttr_sz;
  #pragma omp parallel for num_threads(3)
  for (size_t i = 0; i < 3; ++i) {
    switch (i) {
    case 0:
      // Load foreign key column to determine fact table size
      factFk_sz = __ldsynth((void **)&host->fkey, "fk", dir, 0);
      break;
    case 1:
      // Load fact table attribute columns
      factAttr_sz = __ldsynth((void **)&host->factattr1, "a1", dir, 0);
      break;
    case 2:
      __ldsynth((void **)&host->factattr2, "a2", dir, 0);
      break;
    default: __builtin_unreachable();
    }
  }

  size_t factsz = factFk_sz / sizeof(uint32_t);
  // Calculate bitwidth from the size of fact attributes
  host->bitwidth = (factAttr_sz * 8) / factsz; // Convert bytes to bits per element

  // Dim attrs are generated on-the-fly. Both have size (factsz / 100)
  size_t dimsz = factsz / 100;
  host->dimattr1 = malloc(dimsz * sizeof(uint32_t));
  host->dimattr2 = malloc(dimsz * sizeof(uint8_t));
  assert(host->dimattr1 && host->dimattr2);

  // Initialize dimattr1 as sequential 0,1,2,3,4... for easy selectivity estimates
  // Initialize dimattr2 with uniformly distributed values 0~255
  srand(13458786);
  for (size_t i = 0; i < dimsz; i++) {
    host->dimattr1[i] = i;
    host->dimattr2[i] = rand();
  }

  if (dev) {
    // Allocate device memory
    dev->fkey = __cum(factFk_sz);
    dev->factattr1 = __cum(factAttr_sz);
    dev->factattr2 = __cum(factAttr_sz);
    dev->dimattr1 = __cum(dimsz * sizeof(uint32_t));
    dev->dimattr2 = __cum(dimsz * sizeof(uint8_t));

    // Copy fact table data to device
    cudaMemcpy(dev->fkey, host->fkey, factFk_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(dev->factattr1, host->factattr1, factAttr_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(dev->factattr2, host->factattr2, factAttr_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(dev->dimattr1, host->dimattr1, dimsz * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev->dimattr2, host->dimattr2, dimsz * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Set bitwidth for device schema
    dev->bitwidth = host->bitwidth;
  }

  return factsz;
}

void synth_free(struct synth_schema* host, struct synth_schema* dev) {
  if (host) {
    // Free host memory
    free(host->factattr1);
    free(host->factattr2);
    free(host->fkey);
    free(host->dimattr1);
    free(host->dimattr2);
  }
  if (dev) {
    // Free device memory
    cudaFree(dev->factattr1);
    cudaFree(dev->factattr2);
    cudaFree(dev->fkey);
    cudaFree(dev->dimattr1);
    cudaFree(dev->dimattr2);
  }
}

void synth_ref(const struct synth_schema *dat, size_t factsz, uint64_t fa1low,
               uint64_t fa1hi, uint64_t fa2low, uint64_t fa2hi, uint64_t da1low,
               uint64_t da1hi, uint32_t grpout[256]) {
  memset(grpout, 0, 256 * sizeof(uint32_t));
  // Based on bitwidth, cast the fact attributes appropriately
  #pragma omp parallel for
  for (size_t i = 0; i < factsz; ++i) {
    uint64_t factAttr1Val = 0, factAttr2Val = 0;

    // Extract fact attribute values based on bitwidth
    switch (dat->bitwidth) {
      case 8:
        factAttr1Val = ((uint8_t*)dat->factattr1)[i];
        factAttr2Val = ((uint8_t*)dat->factattr2)[i];
        break;
      case 16:
        factAttr1Val = ((uint16_t*)dat->factattr1)[i];
        factAttr2Val = ((uint16_t*)dat->factattr2)[i];
        break;
      case 32:
        factAttr1Val = ((uint32_t*)dat->factattr1)[i];
        factAttr2Val = ((uint32_t*)dat->factattr2)[i];
        break;
      case 64:
        factAttr1Val = ((uint64_t*)dat->factattr1)[i];
        factAttr2Val = ((uint64_t*)dat->factattr2)[i];
        break;
      default: __builtin_unreachable();
    }
    // Check fact table predicates (OR condition)
    if (!((factAttr1Val >= fa1low && factAttr1Val < fa1hi) ||
          (factAttr2Val >= fa2low && factAttr2Val < fa2hi)))
      continue;

    // Join with dimension table using foreign key
    uint32_t dimKey = dat->fkey[i];
    // Check dimension table predicate
    // dimattr1 is the dimension key (0, 1, 2, 3, ...)
    if (dat->dimattr1[dimKey] < da1low || dat->dimattr1[dimKey] >= da1hi)
      continue;
    // GROUP BY dimattr2 and count qualifying rows
    #pragma omp atomic
    grpout[dat->dimattr2[dimKey]]++;
  }
}

static bool _bruh(const uint32_t* hostres, const uint32_t* xfertmp, int a) {
  uint32_t devres[256];
  cudaMemcpy(devres, xfertmp, 256 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  for (size_t j = 0; j < 256; ++j) {
    if (hostres[j] != devres[j]) {
      fprintf(stderr, "Mismatch at %zu, h=%u, d=%u, ty=%d\n",
              j, hostres[j], devres[j], a);
      return false;
    }
  }
  return true;
}

bool synth_demoall(const char *synthDirname) {
  struct synth_schema host_dat, dev_dat;
  const char* cases[] = {
    "04_16", "04_32", "08_16", "08_32",
    "12_16", "12_32", "16_16", "16_32", "20_16", "20_32",
  };
  const double skews[] = {1.04, 1.04, 1.08, 1.08, 1.12,
                          1.12, 1.16, 1.16, 1.2,  1.2};
  const double colsel[11] = {.02, .04, .06, .09, .12, .18,
                             .25, .39, .51, .67, .82};
  const double fulsel[11] = {1 / 512.0, 1 / 256.0, 1 / 128.0, 1 / 64.0,
                            1 / 32.0,  1 / 16.0,  1 / 8.0,   1 / 4.0,
                            .4,        .6,        .8};
  char case_dir[512];

  puts("\n\nSeletiv\tMethod1\tMethod2");
  // Method 1 vs. 2; skewness 1.04, selectivity 1/{128,64,32,16}
  sprintf(case_dir, "%s/%s", synthDirname, cases[1]);
  size_t factSz = synth_load(&host_dat, &dev_dat, case_dir);
  double *win = window_zipf(skews[1], 1024, 60);
  for (size_t i = 0; i < 7; ++i) {
    uint64_t low = 0, hi = 60;
    while (win[low] > colsel[i]) ++low, ++hi;
    float2 times = synth_method(&dev_dat, factSz, low, hi, low, hi, low, hi);
    printf("%.5f\t%.4f\t%.4f\n", fulsel[i], times.x, times.y);
    fflush(stdout);
  }
  free(win);
  synth_free(&host_dat, &dev_dat);

  puts("\nSkew\tBitW\tSeletiv\tJoin\tPerfect\tManyOrs\tCandChk"
       "\tPftStg1\tPftStg2\tPftStg3"
       "\tOrsStg1\tOrsStg2\tOrsStg3"
       "\tChkStg1\tChkStg2\tChkStg3\tWAH");
  for (size_t i = 0; i < 10; ++i) {
    // Construct directory path for this case
    sprintf(case_dir, "%s/%s", synthDirname, cases[i]);
    factSz = synth_load(&host_dat, &dev_dat, case_dir);
    win = window_zipf(skews[i], 1024, 60);

    // Run on selectivity 1/{256,128,64,32,16,8}
    for (size_t o = 1; o < 11; o++) {
      // Find the first x at which [x, x+60) has prob <= sel
      uint64_t low = 0, hi = 60;
      while (win[low] > colsel[o]) ++low, ++hi;
      printf("%.2f\t%s\t%.5f", skews[i], cases[i] + 3, fulsel[o]);

      uint32_t hostres[256], *xfertmp = __cum(sizeof(hostres) + 8);
      // No memset: synth_* functions do this
      // Run join on host once and save result to hostres (no timing)
      synth_ref(&host_dat, factSz, low, hi, low, hi, low, hi, hostres);

      for (int ty = -1; ty < 6; ++ty) {
        float4 bruh =
            synth_bmp(&dev_dat, factSz, low, hi, low, hi, low, hi, xfertmp, ty);
        printf(ty >= 3 ? "\t%.4f\t%.4f\t%.4f" : "\t%.4f", bruh.x, bruh.y, bruh.z);
        if (!_bruh(hostres, xfertmp, ty)) return false;
      }

      cudaFree(xfertmp);
      if (host_dat.bitwidth == 32)
        synth_wah(&dev_dat, factSz, low, hi, low, hi, low, hi);
      else puts("\t0.0");
      fflush(stdout);
    }

    free(win);
    synth_free(&host_dat, &dev_dat);
  }
  return true;
}
