#include "wahGpu.cuh"

wahStashDat_s::wahStashDat_s(size_t id_, wahDev_s &wah, size_t nIter)
    : id(id_), execScanTime(0), decTime(0) {
  for (size_t i = 0; i < nIter; ++i) {
    execScanTime += wah.timeExcScan();
    decTime += wah.timeDecompress();
  }
  execScanTime /= nIter;
  decTime /= nIter;
  cmprsNrWord = wah._wah.size();
  decmNrWord = wah._decomp.size();
}

void wahOpProfile::profileAnd(bool semi) {
  explOpTime = lhsDev.timeExplAnd(rhsDev);
  explXferTime = lhsDev.timeXfer();
  // explCrsTime = lhsDev.timeCompress();
  // explCrsXferTime = lhsDev.timeXfer();

  // lhsDev.context().timer_begin();
  // uint32_t* out = new uint32_t[lhsDev._opResH.size() + 60];
  // wahAndCPU(lhsHost->dat, lhsHost->dat + lhsHost->cmprsNrWord, rhsHost->dat,
  //           rhsHost->dat + rhsHost->cmprsNrWord, out,
  //           out + lhsDev._opResH.size() + 50);
  // cpuTime = lhsDev.context().timer_end();
  // if (memcmp(lhsDev._opResH.data(), out, lhsDev._opResH.size()) != 0)
  //   exit(fputs("WAH explicit AND error!\n", stderr));

  directOpTime = rhsDev.timeDirectAnd(lhsDev);
  directXferTime = rhsDev.timeXfer();
  // directCptTime = rhsDev.timeCompact();
  // directCptXferTime = rhsDev.timeXfer();

  // if (memcmp(rhsDev._opResH.data(), out, rhsDev._opResH.size()) != 0)
  //   exit(fputs("WAH direct AND error!\n", stderr));
  // delete[] out;
  lhsDev._opResD = mgpu::mem_t<int>();
  rhsDev._opResD = mgpu::mem_t<int>();

  if (!semi) return;
  auto hostRes = std::move(lhsDev._opResH);
  wahDev_s& larg = lhsDev._wah.size() > rhsDev._wah.size() ? lhsDev : rhsDev;
  wahDev_s& smal = lhsDev._wah.size() > rhsDev._wah.size() ? rhsDev : lhsDev;
  semiOpTime = larg.timeSemiExplAnd(smal);
  semiXferTime = larg.timeXfer();
  // semiCrsTime = larg.timeCompact();
  // semiCrsXferTime = larg.timeXfer();
  // if (hostRes != larg._opResH)
  //   exit(fputs("WAH semi AND error!\n", stderr));
  larg._opResD = mgpu::mem_t<int>();
}

void wahOpProfile::profileOr(bool semi) {
  explOpTime = lhsDev.timeExplOr(rhsDev);
  explXferTime = lhsDev.timeXfer();
  // explCrsTime = lhsDev.timeCompress();
  // explCrsXferTime = lhsDev.timeXfer();

  // TODO: Currently the CPU AND also serves as a baseline for GPU OR since in
  // branchless CPU implementation, all logics in AND, OR, XOR are virtually
  // identical except that the extra part of longer sequence is discarded.
  // lhsDev.context().timer_begin();
  // uint32_t* out = new uint32_t[lhsDev._opResH.size() + 60];
  // wahAndCPU(lhsHost->dat, lhsHost->dat + lhsHost->cmprsNrWord, rhsHost->dat,
  //           rhsHost->dat + rhsHost->cmprsNrWord, out,
  //           out + lhsDev._opResH.size() + 50);
  // cpuTime = lhsDev.context().timer_end() * 1.1;
  // delete[] out;

  directOpTime = rhsDev.timeDirectOr(lhsDev);
  directXferTime = rhsDev.timeXfer();
  // directCptTime = rhsDev.timeCompact();
  // directCptXferTime = rhsDev.timeXfer();
  // if (lhsDev._opResH != rhsDev._opResH)
  //   exit(fputs("WAH OR error!\n", stderr));
  lhsDev._opResD = mgpu::mem_t<int>();
  rhsDev._opResD = mgpu::mem_t<int>();

  if (!semi) return;
  auto hostRes = std::move(lhsDev._opResH);
  wahDev_s& larg = lhsDev._wah.size() > rhsDev._wah.size() ? lhsDev : rhsDev;
  wahDev_s& smal = lhsDev._wah.size() > rhsDev._wah.size() ? rhsDev : lhsDev;
  semiOpTime = larg.timeSemiExplOr(smal);
  semiXferTime = larg.timeXfer();
  // semiCrsTime = larg.timeCompress();
  // semiCrsXferTime = larg.timeXfer();
  // if (hostRes != larg._opResH)
  //   exit(fputs("WAH semi OR error!\n", stderr));
  larg._opResD = mgpu::mem_t<int>();
}

void wahOpProfile::profileXor(bool semi) {
  explOpTime = lhsDev.timeExplXor(rhsDev);
  explXferTime = lhsDev.timeXfer();
  // explCrsTime = lhsDev.timeCompress();
  // explCrsXferTime = lhsDev.timeXfer();

  // lhsDev.context().timer_begin();
  // uint32_t* out = new uint32_t[lhsDev._opResH.size() + 60];
  // wahAndCPU(lhsHost->dat, lhsHost->dat + lhsHost->cmprsNrWord, rhsHost->dat,
  //           rhsHost->dat + rhsHost->cmprsNrWord, out,
  //           out + lhsDev._opResH.size() + 50);
  // cpuTime = lhsDev.context().timer_end() * 1.1;
  // delete[] out;

  directOpTime = rhsDev.timeDirectXor(lhsDev);
  directXferTime = rhsDev.timeXfer();
  // directCptTime = rhsDev.timeCompact();
  // directCptXferTime = rhsDev.timeXfer();
  // if (lhsDev._opResH != rhsDev._opResH)
  //   exit(fputs("WAH XOR error!\n", stderr));
  lhsDev._opResD = mgpu::mem_t<int>();
  rhsDev._opResD = mgpu::mem_t<int>();

  if (!semi) return;
  auto hostRes = std::move(lhsDev._opResH);
  wahDev_s& larg = lhsDev._wah.size() > rhsDev._wah.size() ? lhsDev : rhsDev;
  wahDev_s& smal = lhsDev._wah.size() > rhsDev._wah.size() ? rhsDev : lhsDev;
  semiOpTime = larg.timeSemiExplXor(smal);
  semiXferTime = larg.timeXfer();
  // semiCrsTime = larg.timeCompress();
  // semiCrsXferTime = larg.timeXfer();
  // if (hostRes != larg._opResH)
  //   exit(fputs("WAH semi XOR error!\n", stderr));
  larg._opResD = mgpu::mem_t<int>();
}

void wahOpProfile::profile(wahOp op, bool semi) {
  if (op == wahOp::AND)
    profileAnd(semi);
  else if (op == wahOp::OR)
    profileOr(semi);
  else profileXor(semi);
}

double wahOpProfile::explTotalTime(int lStash, int rStash) const noexcept {
  double r = mgpu::min(explXferTime, explCrsTime + explCrsXferTime) + explOpTime;
  if (lStash < 3) r += lhsDat.decTime;
  if (rStash < 3) r += rhsDat.decTime;
  if (lStash < 2) r += lhsDat.execScanTime;
  if (rStash < 2) r += rhsDat.execScanTime;
  return r;
}

double wahOpProfile::directTotalTime(int lStash, int rStash) const noexcept {
  double r = mgpu::min(directXferTime, directCptTime + directCptXferTime) +
             directOpTime;
  if (lStash < 2) r += lhsDat.execScanTime;
  if (rStash < 2) r += rhsDat.execScanTime;
  return r;
}

double wahOpProfile::semiTotalTime() const noexcept {
  return mgpu::min(semiXferTime, semiCrsTime + semiCrsXferTime) + semiOpTime;
}

void wahOpProfile::setLhs(mgpu::context_t &ctx, const wahHost_s *lhsH,
                          wahStashDat_s dat) {
  lhsDat = dat;
  lhsHost = lhsH;
  lhsDev.~wahDev_s();
  new (&lhsDev) wahDev_s(ctx, lhsH);
}

void wahOpProfile::setRhs(mgpu::context_t &ctx, const wahHost_s *rhsH,
                          wahStashDat_s dat) {
  rhsDat = dat;
  rhsHost = rhsH;
  rhsDev.~wahDev_s();
  new (&rhsDev) wahDev_s(ctx, rhsH);
}

void wahOpProfile::print(FILE* stream, bool semi) const noexcept {
  if (semi)
    fprintf(stream,
        "%zu,%zu,%.06f,%.06f,%.06f,%.06f,%.06f,%.06f,%.06f,%.06f,%.06f,%.06f\n",
        lhsDat.id, rhsDat.id, explOpTime, explXferTime,
        explCrsTime + explCrsXferTime, directOpTime, directXferTime,
        directCptTime + directCptXferTime, semiOpTime, semiXferTime,
        semiCrsTime + semiCrsXferTime, cpuTime);
  else
    fprintf(stream, "%zu,%zu,%.06f,%.06f,%.06f,%.06f,%.06f,%.06f,%.06f\n",
            lhsDat.id, rhsDat.id, explOpTime, explXferTime,
            explCrsTime + explCrsXferTime, directOpTime, directXferTime,
            directCptTime + directCptXferTime, cpuTime);
}

void wahStashDat_s::print(FILE* stream) const noexcept {
  fprintf(stream, "%zu,%.06f,%.06f,%zu,%zu\n", id, stash2Time(), stash3Time(),
          stash2Sz(), stash3Sz());
}

void wahOpProfile::clear() {
  lhsDev = wahDev_s();
  rhsDev = wahDev_s();
  lhsHost = nullptr;
  rhsHost = nullptr;
}
