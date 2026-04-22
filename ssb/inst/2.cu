#include "../lambdas.cuh"

template void DOWORK<s2op>(const char *a, const char *b, vprg &p,
                           uint32_t *grp_out, size_t nr_grp, s2op &op,
                           uint factsz, bool join);
