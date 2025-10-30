#include "dumb_pool.cuh"
#include "moderngpu/memory.cuh"
#include "host/wahCpu.h"

mgpu::mem_t<int> wahCntExcScan(const int *wah, size_t sz, mgpu::context_t &ctx);
mgpu::mem_t<int> wahDecomp(const int *wah, const int *excScan, size_t sz,
                           mgpu::context_t &ctx, int decompSz = 0);
mgpu::mem_t<int> wahCompact(const int *wah, size_t wahSz, mgpu::context_t &context);

void wahAndExpl(int *upDecomp, const int *downDecomp, size_t minSz,
                mgpu::context_t &ctx);
mgpu::mem_t<int> wahAndNo1(const int *up, const int *upIncScan, size_t upSz,
                           const int *down, const int *downIncScan,
                           size_t downSz, mgpu::context_t &context);
mgpu::mem_t<int> wahAnd(const int *up, const int *upExcScan, size_t upSz,
                        const int *down, const int *downExcScan, size_t downSz,
                        mgpu::context_t &context);

void wahOrExpl(int *upDecomp, const int *downDecomp, size_t minSz,
                mgpu::context_t &ctx);
mgpu::mem_t<int> wahOr(const int *up, const int *upExcScan, size_t upSz,
                       const int *down, const int *downExcScan, size_t downSz,
                       mgpu::context_t &context);

void wahXorExpl(int *upDecomp, const int *downDecomp, size_t minSz,
                mgpu::context_t &ctx);
mgpu::mem_t<int> wahXor(const int *up, const int *upExcScan, size_t upSz,
                        const int *down, const int *downExcScan, size_t downSz,
                        mgpu::context_t &context);

enum class wahOp { AND, OR, XOR };
mgpu::mem_t<int> wahEncOpDec(const int *enc, const int *excScan, size_t encSz,
                             const int *dec, size_t decSz, wahOp op,
                             mgpu::context_t &ctx, int encDecompSz = 0);
mgpu::mem_t<int> wahEncAndDec(const int *enc, const int *excScan, size_t encSz,
                              const int *dec, size_t decSz, mgpu::context_t &ctx);
void wahEncNo1AndDec(int *enc, const int *excScan, size_t encSz, const int *dec,
                     size_t decSz, mgpu::context_t &ctx);
mgpu::mem_t<int> wahCompress(const int *dec, size_t wahSz,
                             mgpu::context_t &context);

mgpu::mem_t<uint32_t>
dbjoinFlatWah(const uint32_t *fact, size_t factSz, const uint32_t* dim1,
              const uint32_t *dim2, uint32_t min, uint32_t max, mgpu::context_t &ctx);

struct wahDev_s {
  mgpu::mem_t<int> _wah;
  mgpu::mem_t<int> _excScan;
  mgpu::mem_t<int> _decomp;
  mgpu::mem_t<int> _opResD;

  bool has1Fill;
  uint64_t decmprsNrWord;
  std::vector<int> _opResH;
  mgpu::context_t& context() noexcept { return _wah.context(); }
  wahDev_s() = default;

  wahDev_s(mgpu::context_t &ctx, const wahHost_s *h)
      : _wah(h ? h->cmprsNrWord : 1, ctx), has1Fill(h ? h->has1Fill : false),
        decmprsNrWord(h ? h->decmprsNrWord : 0) {
    if (h)
      mgpu::htod(_wah.data(), (const int *)h->dat, _wah.size());
  }

  mgpu::mem_t<int>& excScan() {
    if (_excScan.data() == nullptr)
      _excScan = wahCntExcScan(_wah.data(), _wah.size(), _wah.context());
    return _excScan;
  }
  double timeExcScan() {
    _excScan = mgpu::mem_t<int>();
    context().timer_begin();
    _excScan = wahCntExcScan(_wah.data(), _wah.size(), _wah.context());
    return context().timer_end();
  }

  mgpu::mem_t<int>& decompress() {
    if (_decomp.data() == nullptr)
      _decomp = wahDecomp(_wah.data(), excScan().data(), _wah.size(), context());
    return _decomp;
  }
  double timeDecompress() {
    (void)excScan();
    context().timer_begin();
    _decomp = mgpu::mem_t<int>();
    _decomp = wahDecomp(_wah.data(), _excScan.data(), _wah.size(), context());
    return context().timer_end();
  }

  mgpu::mem_t<int>& explAnd(wahDev_s &rhs);
  double timeExplAnd(wahDev_s &rhs);
  mgpu::mem_t<int>& explOr(wahDev_s &rhs);
  double timeExplOr(wahDev_s &rhs);
  mgpu::mem_t<int>& explXor(wahDev_s &rhs);
  double timeExplXor(wahDev_s &rhs);

  mgpu::mem_t<int>& directAnd(wahDev_s &rhs);
  double timeDirectAnd(wahDev_s &rhs);
  mgpu::mem_t<int>& directOr(wahDev_s &rhs);
  double timeDirectOr(wahDev_s &rhs);
  mgpu::mem_t<int>& directXor(wahDev_s &rhs);
  double timeDirectXor(wahDev_s &rhs);

  // decompress lhs (*this), then rhs operates on decompressed lhs
  mgpu::mem_t<int>& semiExplAnd(wahDev_s &rhs);
  // only times rhs operating on decompressed lhs
  double timeSemiExplAnd(wahDev_s &rhs);
  mgpu::mem_t<int>& semiExplOr(wahDev_s &rhs);
  double timeSemiExplOr(wahDev_s &rhs);
  mgpu::mem_t<int>& semiExplXor(wahDev_s &rhs);
  double timeSemiExplXor(wahDev_s &rhs);

  mgpu::mem_t<int>& compact() {
    if (_opResD.size() == 0) return _opResD;
    _opResD = wahCompact(_opResD.data(), _opResD.size(), context());
    return _opResD;
  }
  double timeCompact() {
    if (_opResD.size() == 0) return 0.0;
    context().timer_begin();
    _opResD = wahCompact(_opResD.data(), _opResD.size(), context());
    return context().timer_end();
  }

  mgpu::mem_t<int>& compress() {
    if (_opResD.size() == 0) return _opResD;
    _opResD = wahCompress(_opResD.data(), _opResD.size(), context());
    return _opResD;
  }
  double timeCompress() {
    if (_opResD.size() == 0) return 0.0;
    context().timer_begin();
    _opResD = wahCompress(_opResD.data(), _opResD.size(), context());
    return context().timer_end();
  }

  std::vector<int>& xfer() { return (_opResH = mgpu::from_mem(_opResD)); }
  double timeXfer() {
    context().timer_begin();
    _opResH = mgpu::from_mem(_opResD);
    return context().timer_end();
  }
};

struct wahStashDat_s {
  size_t id;
  // double hToDTime;
  // Time spent on computing execlusive scan and decompression
  double execScanTime, decTime;
  // Size of compressed wah and decompressed size
  size_t cmprsNrWord, decmNrWord;

  wahStashDat_s()
      : id(0), execScanTime(0.0), decTime(0.0), cmprsNrWord(0), decmNrWord(0) {}
  wahStashDat_s(size_t id, wahDev_s &wah, size_t nIter);
  // wahStashDat_s(wahHost_s *wah); // TODO

  size_t stash2Sz() const noexcept { return 2 * cmprsNrWord + 1; }
  size_t stash3Sz() const noexcept { return decmNrWord; }
  double stash2Time() const noexcept { return execScanTime /*+hToDTime*/; }
  double stash3Time() const noexcept { return decTime + stash2Time(); }

  void print(FILE* stream) const noexcept;
};

struct wahOpProfile {
  const wahHost_s *lhsHost = nullptr, *rhsHost = nullptr;
  wahDev_s lhsDev, rhsDev;
  wahStashDat_s lhsDat, rhsDat;
  double explOpTime, explXferTime, explCrsTime, explCrsXferTime;
  double directOpTime, directXferTime, directCptTime, directCptXferTime;
  double semiOpTime, semiXferTime, semiCrsTime, semiCrsXferTime;
  double cpuTime;

  void setLhs(mgpu::context_t &ctx, const wahHost_s* lhsH, wahStashDat_s dat);
  void setRhs(mgpu::context_t &ctx, const wahHost_s* rhsH, wahStashDat_s dat);

  void profileAnd(bool semi);
  void profileOr(bool semi);
  void profileXor(bool semi);
  void profile(wahOp op, bool semi);
  // TODO: populate wahStashDat_s as well
  // void setLhs(mgpu::context_t &ctx, const wahHost_s* lhsH);

  double explTotalTime(int lStash, int rStash) const noexcept;
  double directTotalTime(int lStash, int rStash) const noexcept;
  double semiTotalTime() const noexcept;

  void print(FILE* stream, bool semi) const noexcept;
  void clear();
};
