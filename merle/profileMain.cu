#include "host/wahCpu.h"
#include "moderngpu/memory.cuh"
#include "wahGpu.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <thread>
#include <vector>
using merle = mgpu::mem_t<int>;

enum class BenchWah {
  Join,
  Copmression,
  BitsetToWah,
  Collection,
  STZ, // ssb tpch zipf
};

mgpu::mem_t<uint32_t> loadColFile(const char* path, mgpu::context_t& ctx) {
  if (path == NULL)
    return mgpu::mem_t<uint32_t>(mgpu::memory_space_host);
  FILE *f = fopen(path, "rb");
  if (f == NULL)
    return mgpu::mem_t<uint32_t>(mgpu::memory_space_host);
  fseek(f, 0, SEEK_END);
  size_t sz_ = ftell(f) / sizeof(uint32_t);
  fseek(f, 0, SEEK_SET);
  auto ret = mgpu::mem_t<uint32_t>(sz_, ctx, mgpu::memory_space_host);
  if (sz_ != fread(ret.data(), sizeof(uint32_t), sz_, f))
    exit(fprintf(stderr, "partial read?\n"));
  fclose(f);
  return ret;
}

int main(int argc, char** argv) {
  dumb_pool_t ctx(1200 * 1024 * 1024, false);
  // mgpu::standard_context_t context;
  BenchWah subcmd = BenchWah::STZ;
  if (argv[1] == NULL || 0 == strcasecmp(argv[1], "benchStz"))
    subcmd = BenchWah::STZ;
  else if (0 == strcasecmp(argv[1], "benchJoin"))
    subcmd = BenchWah::Join;
  else if (0 == strcasecmp(argv[1], "benchCompression"))
    subcmd = BenchWah::Copmression;
  else if (0 == strcasecmp(argv[1], "benchCollection"))
    subcmd = BenchWah::Collection;
  else exit(fprintf(stderr, "Unknown command %s\n", argv[1]));
  if (subcmd == BenchWah::STZ) {
    // Each runTestCase call run 1 GPU on ssb and tpch segments
    // 1000 iterations each testcase and total time is reported in secs
    void runTestCase(size_t n);

    int deviceCnt = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCnt);
    if (err != cudaSuccess || deviceCnt < 1)
      return fputs(cudaGetErrorString(err), stderr);
    std::vector<std::thread> thrds;
    for (size_t i = 0; i < (size_t)deviceCnt; ++i) {
      thrds.emplace_back(runTestCase, i);
    }
    for (auto& t : thrds)
      t.join();
    return 0;
  }

  const char *inFmt = nullptr, *stashOut = "stash.csv", *opOut = "op.csv";
  const char *dim1Path = nullptr, *dim2Path = nullptr;
  size_t nFile = 200, nDup = 200, scanMin = 0, scanMax = (size_t)-1;
  wahOp op = wahOp::AND;
  bool semi = false;
  for (int i = 2; i < argc; i += 2) {
    if (0 == strcasecmp(argv[i], "--inFmt"))
      inFmt = argv[1 + i];
    else if (0 == strcasecmp(argv[i], "--stashOut") ||
             0 == strcasecmp(argv[i], "--factColFile"))
      stashOut = argv[1 + i];
    else if (0 == strcasecmp(argv[i], "--nFile"))
      nFile = atoi(argv[1 + i]);
    else if (0 == strcasecmp(argv[i], "--nDup"))
      nDup = atoi(argv[1 + i]);
    else if (0 == strcasecmp(argv[i], "--scanOut") ||
             0 == strcasecmp(argv[i], "--opOut") ||
             0 == strcasecmp(argv[i], "--outFmt"))
      opOut = argv[1 + i];
    else if (0 == strcasecmp(argv[i], "--dim1ColFile"))
      dim1Path = argv[1 + i];
    else if (0 == strcasecmp(argv[i], "--dim2ColFile"))
      dim2Path = argv[1 + i];
    else if (0 == strcasecmp(argv[i], "--scanMin"))
      scanMin = atoi(argv[1 + i]);
    else if (0 == strcasecmp(argv[i], "--scanMax"))
      scanMax = atoi(argv[1 + i]);
    else if (0 == strcasecmp(argv[i], "--semi")) {
      semi = true; --i;
    } else if (0 == strcasecmp(argv[i], "--op")) {
      if (argv[1 + i][0] == 'o' || argv[1 + i][0] == 'O')
        op = wahOp::OR;
      else if (argv[1 + i][0] == 'x' || argv[1 + i][0] == 'X')
        op = wahOp::XOR;
      else if (argv[1 + i][0] == 'd' || argv[1 + i][0] == 'D')
        stashOut = nullptr;
    }
    else return fputs("Invalid Argument", stderr);
  }

  if (subcmd == BenchWah::Join) {
    mgpu::mem_t<uint32_t> fact = loadColFile(stashOut, ctx),
                          dim1 = loadColFile(dim1Path, ctx),
                          dim2 = loadColFile(dim2Path, ctx);
    mgpu::mem_t<uint32_t> factD(fact.size(), ctx), dim1D(dim1.size(), ctx),
        dim2D(dim2.size(), ctx);
    mgpu::htod(factD.data(), fact.data(), fact.size());
    mgpu::htod(dim1D.data(), dim1.data(), dim1.size());
    mgpu::htod(dim2D.data(), dim2.data(), dim2.size());

    // Is the join correct?
    // auto joined = dbjoinFlatWah(factD.data(), factD.size(), dim1D.data(),
    //                             dim2D.data(), scanMin, scanMax, ctx);
    // auto joinedH = mgpu::from_mem(joined);
    // auto joinVerify = std::vector<uint32_t>(joinedH.size(), 12345);
    // simpleJoin(fact.data(), fact.size(), dim1.data(), dim2.data(), scanMin,
    //            scanMax, joinVerify.data());
    // ctx.synchronize();
    // for (size_t i = 0; i < joinedH.size(); ++i)
    //   if (joinedH[i] != joinVerify[i])
    //     return fprintf(stderr, "Error at word %zu\n", i);

    for (size_t i = 1; i < nDup; ++i)
      (void)dbjoinFlatWah(factD.data(), factD.size(), dim1D.data(),
                          dim2D.data(), scanMin, scanMax, ctx);
    return 0;
  }

  if (nullptr == inFmt || strlen(inFmt) > 240)
    return fputs("inFmt must be specified and <240 chars", stderr);
  std::vector<wahHost_s*> wahHosts;
  for (size_t i = 0; i < nFile; ++i) {
    char path[256];
    sprintf(path, inFmt, i);
    wahHosts.push_back(loadWahFile(path));
  }
  if (subcmd == BenchWah::Copmression) {
    for (wahHost_s *h : wahHosts) {
      assert(h->decmprsNrWord == h->cmprsNrWord);
      merle d(h->decmprsNrWord, ctx);
      mgpu::htod((uint32_t *)d.data(), h->dat, d.size());
      for (size_t i = 0; i < nDup; ++i)
        (void)wahCompress(d.data(), d.size(), ctx);
    }
    return 0;
  }

  for (size_t i = 0; i < wahHosts.size(); ++i) {
    wahHost_s *orig = wahHosts[i];
    wahHosts[i] = wahDup(orig, nDup);
    free(orig);
  }
  // Heat up & correctness check
  wahOpProfile profile;
  srand(time(0));
  for (size_t i = 0; i < 10; ++i) {
    size_t lId = rand() % nFile, rId = rand() % nFile;
    profile.setLhs(ctx, wahHosts[lId], wahStashDat_s());
    profile.setRhs(ctx, wahHosts[rId], wahStashDat_s());
    profile.profileAnd(true);
    profile.profileOr(true);
    profile.profileXor(true);
    profile.clear();
  }

  FILE *opOutf = fopen(opOut, "w");
  if (opOutf == nullptr)
    return fputs("Cannot open output file!\n", stderr);
  if (stashOut == nullptr) {
    // Profile decompression only
    merle andrzejewski(const int *C, size_t m, mgpu::context_t &ctx);
    merle ours(const int *C, size_t m, mgpu::context_t &ctx);
    fputs("id,encSz,decSz,andrzejewski,ours\n", opOutf);
    for (size_t i = 0; i < nFile; ++i) {
      wahHost_s *encHost = wahHosts[i];
      size_t sz = encHost->cmprsNrWord;
      merle encDev = merle(sz, ctx);
      mgpu::htod(encDev.data(), reinterpret_cast<const int *>(encHost->dat), sz);
      ctx.timer_begin();
      auto resD = ours(encDev.data(), sz, ctx);
      double goodTime = ctx.timer_end();
      auto goodResH = mgpu::from_mem(resD);
      resD = merle();
      ctx.timer_begin();
      resD = andrzejewski((const int *)encDev.data(), sz, ctx);
      double badTime = ctx.timer_end();
      auto badResH = mgpu::from_mem(resD);
      fprintf(opOutf, "%zu,%zu,%zu,%f,%f\n", i, sz, resD.size(),
              badTime, goodTime);
      if (badResH != goodResH)
        exit(1);
    }
    fclose(opOutf);
    return 0;
  }

  // stash data
  FILE *stashOutf = fopen(stashOut, "w");
  if (stashOutf == nullptr)
    return fputs("Cannot open stash output!", stderr);
  fputs("id,excScanTime,decTime,orig+scanSz,decSz\n", stashOutf);
  std::vector<wahStashDat_s> stashDats;
  for (size_t i = 0; i < nFile; ++i) {
    wahDev_s wah(ctx, wahHosts[i]);
    stashDats.emplace_back(i, wah, 4);
    stashDats.back().print(stashOutf);
  }
  fclose(stashOutf);

  fputs(semi ? "lhs,rhs,expl,explXfer,explCptXfer,direct,directXfer,"
               "directCptXfer,semi,semiXfer,semiCptXfer,dummy\n"
             : "lhs,rhs,expl,explXfer,explCptXfer,direct,directXfer,"
               "directCptXfer,dummy\n",
        opOutf);
  for (size_t i = 0; i < nFile; ++i) {
    profile.clear();
    profile.setLhs(ctx, wahHosts[i], stashDats[i]);
    for (size_t j = i+1; j < nFile; ++j) {
      profile.setRhs(ctx, wahHosts[j], stashDats[j]);
      profile.profile(op, semi);
      profile.print(opOutf, semi);
    }
  }

  fclose(opOutf);
  return 0;
}
