#include "host/wahCpu.h"
#include "wahGpu.cuh"

int main(int argc, char** argv) {
  dumb_pool_t context(768 * 1024 * 1024);
  // mgpu::standard_context_t context;
  srand(time(0));
  // srand(12345681);
  std::vector<int> lhsHost(1 << 22);
  std::vector<int> rhsHost(lhsHost.size() - (1 << 15)); // 16 MiB each
  size_t maxTail = 4, maxFill = 15, tailDens = 4, fillDens = 1;
  FILE *outf = stdout;
  using explFun_t = decltype(&wahAndExpl);
  using directFun_t = decltype(&wahAnd);
  explFun_t explFun = nullptr;
  directFun_t directFun = nullptr;

  for (int i = 1; i < argc; i += 2) {
    if (0 == strcasecmp(argv[i], "--maxTail"))
      maxTail = atoi(argv[i + 1]);
    else if (0 == strcasecmp(argv[i], "--maxFill"))
      maxFill = atoi(argv[i + 1]);
    else if (0 == strcasecmp(argv[i], "--tailDens"))
      tailDens = atoi(argv[i + 1]);
    else if (0 == strcasecmp(argv[i], "--fillDens"))
      fillDens = atoi(argv[i + 1]);
    else if (0 == strcasecmp(argv[i], "--out")) {
      outf = fopen(argv[i + 1], "w");
      if (outf == NULL)
        return fprintf(stderr, "Fail to open %s\n", argv[i + 1]);
    } else if (0 == strcasecmp(argv[i], "--op")) {
      switch(toupper(argv[i + 1][0])) {
      case 'A':
        explFun = wahAndExpl;
        directFun = fillDens == 0 ? wahAndNo1 : wahAnd;
        break;
      case 'O': explFun = wahOrExpl; directFun = wahOr; break;
      case 'X': explFun = wahXorExpl; directFun = wahXor; break;
      case 'D': explFun = nullptr; directFun = nullptr; break;
      default: fputs("Invalid Operation\n", stderr); return 2;
      }
    }
    else return fputs("Invalid Argument\n", stderr);
  }

  wahGen((uint32_t*)lhsHost.data(), (uint32_t*)lhsHost.data() + lhsHost.size(),
         maxTail, maxFill, tailDens, fillDens);
  wahGen((uint32_t*)rhsHost.data(), (uint32_t*)rhsHost.data() + rhsHost.size(),
         maxTail, maxFill, tailDens, fillDens);
  mgpu::mem_t<int> lhsDev = mgpu::to_mem(lhsHost, context);
  mgpu::mem_t<int> rhsDev = mgpu::to_mem(rhsHost, context);
  mgpu::mem_t<int> lhsScan = wahCntExcScan(lhsDev.data(), lhsDev.size(), context);
  mgpu::mem_t<int> rhsScan = wahCntExcScan(rhsDev.data(), rhsDev.size(), context);

  if (explFun == nullptr) {
    // Profile decompression only
    mgpu::mem_t<int> andrzejewski(const int *C, size_t m, mgpu::context_t &ctx);
    mgpu::mem_t<int> ours(const int *C, size_t m, mgpu::context_t &ctx);

    fputs("encByte,decByte,andr,ours,andrThput,oursThput\n", outf);
    for (size_t sz = 3000; sz <= lhsHost.size(); sz += (sz >> 3)) {
      context.timer_begin();
      mgpu::mem_t<int> res = andrzejewski(lhsDev.data(), sz, context);
      double badTime = context.timer_end(); context.timer_begin();
      res = ours(lhsDev.data(), sz, context);
      double goodTime = context.timer_end();
      fprintf(outf, "%zu,%zu,%f,%f,%g,%g\n", sz * 4, res.size() * 4, badTime,
              goodTime, sz * 4 / badTime, sz * 4 / goodTime);
    }
    return 0;
  }

  fputs("encByte,decByte,explTime,directTime,explThput,directThput\n", outf);
  for (size_t rsz = 3000; rsz <= rhsHost.size(); rsz += (rsz >> 3)) {
    size_t lsz = rsz + (1 << 13);
    if (explFun != wahAndExpl)
      std::swap(lsz, rsz);
    double directTime = 0.0, explTime = 0.0;
    size_t decSz;
    for (size_t i = 0; i < 50; ++i) {
      context.timer_begin();
      auto lDec = wahDecomp(lhsDev.data(), lhsScan.data(), lsz, context);
      auto rDec = wahDecomp(rhsDev.data(), rhsScan.data(), rsz, context);
      decSz = rDec.size() * 8;
      explFun(lDec.data(), rDec.data(), rDec.size(), context);
      explTime += context.timer_end();

      context.timer_begin();
      auto directRes = directFun(lhsDev.data(), lhsScan.data(), lsz,
                                 rhsDev.data(), rhsScan.data(), rsz, context);
      directTime += context.timer_end();
    }

    directTime /= 50.0; explTime /= 50.0;
    size_t nbyte = (lsz + rsz) * 4;
    fprintf(outf, "%zu,%zu,%f,%f,%g,%g\n", nbyte, decSz, explTime,
            directTime, nbyte / explTime, nbyte / directTime);
    if (explFun != wahAndExpl)
      std::swap(lsz, rsz);
  }
  fclose(outf);
}
