#include "ssb/ssbdemo.h"
#include "synth/synthdemo.h"
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
  const char *ssbDir = (argc > 1) ? argv[1] : "./ssb/ssbCols";
  const char *synthDir = (argc > 2) ? argv[2] : "./synth/zfcols";
  if (argc > 3)
    omp_set_num_threads(atoi(argv[3]));
  if (!ssb_demoall(ssbDir))
    return 2;
  return 0;
  // return !synth_demoall(synthDir);
}
