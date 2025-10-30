#include "wahGpu.cuh"

mgpu::mem_t<int> &wahDev_s::explAnd(wahDev_s &rhs) {
  _opResD = mgpu::mem_t<int>();
  if (decompress().size() < rhs.decompress().size()) {
    _opResD = _decomp.clone();
    wahAndExpl(_opResD.data(), rhs._decomp.data(), _opResD.size(), context());
  } else {
    _opResD = rhs._decomp.clone();
    wahAndExpl(_opResD.data(), _decomp.data(), _opResD.size(), context());
  }
  return _opResD;
}
double wahDev_s::timeExplAnd(wahDev_s &rhs) {
  decompress(); rhs.decompress();
  _opResD = mgpu::mem_t<int>();
  if (_decomp.size() < rhs._decomp.size()) {
    _opResD = _decomp.clone();
    context().timer_begin();
    wahAndExpl(_opResD.data(), rhs._decomp.data(), _opResD.size(), context());
  } else {
    _opResD = rhs._decomp.clone();
    context().timer_begin();
    wahAndExpl(_opResD.data(), _decomp.data(), _opResD.size(), context());
  }
  return context().timer_end();
}

mgpu::mem_t<int> &wahDev_s::explOr(wahDev_s &rhs) {
  _opResD = mgpu::mem_t<int>();
  if (decompress().size() > rhs.decompress().size()) {
    _opResD = _decomp.clone();
    wahOrExpl(_opResD.data(), rhs._decomp.data(), rhs._decomp.size(), context());
  } else {
    _opResD = rhs._decomp.clone();
    wahOrExpl(_opResD.data(), _decomp.data(), _decomp.size(), context());
  }
  return _opResD;
}
double wahDev_s::timeExplOr(wahDev_s &rhs) {
  decompress(); rhs.decompress();
  _opResD = mgpu::mem_t<int>();
  if (_decomp.size() > rhs._decomp.size()) {
    _opResD = _decomp.clone(); // rhs smaller, this larger
    context().timer_begin();
    wahOrExpl(_opResD.data(), rhs._decomp.data(), rhs._decomp.size(), context());
  } else {
    _opResD = rhs._decomp.clone(); // this smaller, rhs larger
    context().timer_begin();
    wahOrExpl(_opResD.data(), _decomp.data(), _decomp.size(), context());
  }
  return context().timer_end();
}

mgpu::mem_t<int> &wahDev_s::explXor(wahDev_s &rhs) {
  _opResD = mgpu::mem_t<int>();
  if (decompress().size() > rhs.decompress().size()) {
    _opResD = _decomp.clone();
    wahXorExpl(_opResD.data(), rhs._decomp.data(), rhs._decomp.size(), context());
  } else {
    _opResD = rhs._decomp.clone();
    wahXorExpl(_opResD.data(), _decomp.data(), _decomp.size(), context());
  }
  return _opResD;
}
double wahDev_s::timeExplXor(wahDev_s &rhs) {
  decompress(); rhs.decompress();
  _opResD = mgpu::mem_t<int>();
  if (_decomp.size() > rhs._decomp.size()) {
    _opResD = _decomp.clone();
    context().timer_begin();
    wahXorExpl(_opResD.data(), rhs._decomp.data(), rhs._decomp.size(), context());
  } else {
    _opResD = rhs._decomp.clone();
    context().timer_begin();
    wahXorExpl(_opResD.data(), _decomp.data(), _decomp.size(), context());
  }
  return context().timer_end();
}

mgpu::mem_t<int> &wahDev_s::directAnd(wahDev_s &rhs) {
  excScan(); rhs.excScan();
  _opResD = mgpu::mem_t<int>();
  bool slow = rhs.has1Fill && has1Fill;
  bool rhsDown =
      has1Fill == rhs.has1Fill ? rhs._wah.size() < _wah.size() : rhs.has1Fill;
  if (rhsDown) {
    if (slow)
      _opResD =
          wahAnd(_wah.data(), _excScan.data(), _wah.size(), rhs._wah.data(),
                 rhs._excScan.data(), rhs._wah.size(), context());
    else
      _opResD = wahAndNo1(_wah.data(), _excScan.data() + 1, _wah.size(),
                          rhs._wah.data(), rhs._excScan.data() + 1,
                          rhs._wah.size(), context());
  } else {
    if (slow)
      _opResD = wahAnd(rhs._wah.data(), rhs._excScan.data(), rhs._wah.size(),
                       _wah.data(), _excScan.data(), _wah.size(), context());
    else
      _opResD =
          wahAndNo1(rhs._wah.data(), rhs._excScan.data() + 1, rhs._wah.size(),
                    _wah.data(), _excScan.data() + 1, _wah.size(), context());
  }
  return _opResD;
}
double wahDev_s::timeDirectAnd(wahDev_s &rhs) {
  excScan(); rhs.excScan();
  _opResD = mgpu::mem_t<int>();
  bool slow = rhs.has1Fill && has1Fill;
  bool rhsDown =
      has1Fill == rhs.has1Fill ? rhs._wah.size() < _wah.size() : has1Fill;
  context().timer_begin();
  if (rhsDown) {
    if (slow)
      _opResD =
          wahAnd(_wah.data(), _excScan.data(), _wah.size(), rhs._wah.data(),
                 rhs._excScan.data(), rhs._wah.size(), context());
    else
      _opResD = wahAndNo1(_wah.data(), _excScan.data() + 1, _wah.size(),
                          rhs._wah.data(), rhs._excScan.data() + 1,
                          rhs._wah.size(), context());
  } else {
    if (slow)
      _opResD = wahAnd(rhs._wah.data(), rhs._excScan.data(), rhs._wah.size(),
                       _wah.data(), _excScan.data(), _wah.size(), context());
    else
      _opResD =
          wahAndNo1(rhs._wah.data(), rhs._excScan.data() + 1, rhs._wah.size(),
                    _wah.data(), _excScan.data() + 1, _wah.size(), context());
  }
  return context().timer_end();
}

mgpu::mem_t<int> &wahDev_s::directOr(wahDev_s &rhs) {
  excScan(); rhs.excScan();
  _opResD = mgpu::mem_t<int>();
  bool rhsDown = decmprsNrWord <= rhs.decmprsNrWord;
  if (rhsDown) {
    _opResD = wahOr(_wah.data(), _excScan.data(), _wah.size(), rhs._wah.data(),
                    rhs._excScan.data(), rhs._wah.size(), context());
  } else {
    _opResD = wahOr(rhs._wah.data(), rhs._excScan.data(), rhs._wah.size(),
                    _wah.data(), _excScan.data(), _wah.size(), context());
  }
  return _opResD;
}
double wahDev_s::timeDirectOr(wahDev_s &rhs) {
  excScan(); rhs.excScan();
  _opResD = mgpu::mem_t<int>();
  bool rhsDown = decmprsNrWord <= rhs.decmprsNrWord;
  context().timer_begin();
  if (rhsDown) {
    _opResD = wahOr(_wah.data(), _excScan.data(), _wah.size(), rhs._wah.data(),
                    rhs._excScan.data(), rhs._wah.size(), context());
  } else {
    _opResD = wahOr(rhs._wah.data(), rhs._excScan.data(), rhs._wah.size(),
                    _wah.data(), _excScan.data(), _wah.size(), context());
  }
  return context().timer_end();
}

mgpu::mem_t<int> &wahDev_s::directXor(wahDev_s &rhs) {
  excScan(); rhs.excScan();
  _opResD = mgpu::mem_t<int>();
  bool rhsDown = decmprsNrWord <= rhs.decmprsNrWord;
  if (rhsDown) {
    _opResD = wahXor(_wah.data(), _excScan.data(), _wah.size(), rhs._wah.data(),
                     rhs._excScan.data(), rhs._wah.size(), context());
  } else {
    _opResD = wahXor(rhs._wah.data(), rhs._excScan.data(), rhs._wah.size(),
                     _wah.data(), _excScan.data(), _wah.size(), context());
  }
  return _opResD;
}
double wahDev_s::timeDirectXor(wahDev_s &rhs) {
  excScan(); rhs.excScan();
  _opResD = mgpu::mem_t<int>();
  bool rhsDown = decmprsNrWord <= rhs.decmprsNrWord;
  context().timer_begin();
  if (rhsDown) {
    _opResD = wahXor(_wah.data(), _excScan.data(), _wah.size(), rhs._wah.data(),
                    rhs._excScan.data(), rhs._wah.size(), context());
  } else {
    _opResD = wahXor(rhs._wah.data(), rhs._excScan.data(), rhs._wah.size(),
                    _wah.data(), _excScan.data(), _wah.size(), context());
  }
  return context().timer_end();
}

mgpu::mem_t<int>& wahDev_s::semiExplAnd(wahDev_s &rhs) {
  decompress(); rhs.excScan();
  if (rhs.has1Fill)
    _opResD =
        wahEncAndDec(rhs._wah.data(), rhs._excScan.data(), rhs._wah.size(),
                     _decomp.data(), _decomp.size(), context());
  else {
    _opResD = rhs._wah.clone();
    wahEncNo1AndDec(_opResD.data(), rhs._excScan.data(), _opResD.size(),
                    _decomp.data(), _decomp.size(), context());
  }
  return _opResD;
}
double wahDev_s::timeSemiExplAnd(wahDev_s &rhs) {
  decompress(); rhs.excScan();
  if (rhs.has1Fill) {
    context().timer_begin();
    _opResD =
        wahEncAndDec(rhs._wah.data(), rhs._excScan.data(), rhs._wah.size(),
                     _decomp.data(), _decomp.size(), context());
  } else {
    _opResD = rhs._wah.clone();
    context().timer_begin();
    wahEncNo1AndDec(_opResD.data(), rhs._excScan.data(), _opResD.size(),
                    _decomp.data(), _decomp.size(), context());
  }
  return context().timer_end();
}

mgpu::mem_t<int>& wahDev_s::semiExplOr(wahDev_s &rhs) {
  decompress(); rhs.excScan();
  _opResD = wahEncOpDec(rhs._wah.data(), rhs._excScan.data(), rhs._wah.size(),
                        _decomp.data(), _decomp.size(), wahOp::OR, context(),
                        rhs.decmprsNrWord);
  return _opResD;
}
double wahDev_s::timeSemiExplOr(wahDev_s &rhs) {
  decompress(); rhs.excScan(); context().timer_begin();
  _opResD = wahEncOpDec(rhs._wah.data(), rhs._excScan.data(), rhs._wah.size(),
                        _decomp.data(), _decomp.size(), wahOp::OR, context(),
                        rhs.decmprsNrWord);
  return context().timer_end();
}

mgpu::mem_t<int>& wahDev_s::semiExplXor(wahDev_s &rhs) {
  decompress(); rhs.excScan();
  _opResD = wahEncOpDec(rhs._wah.data(), rhs._excScan.data(), rhs._wah.size(),
                        _decomp.data(), _decomp.size(), wahOp::XOR, context(),
                        rhs.decmprsNrWord);
  return _opResD;
}
double wahDev_s::timeSemiExplXor(wahDev_s &rhs) {
  decompress(); rhs.excScan(); context().timer_begin();
  _opResD = wahEncOpDec(rhs._wah.data(), rhs._excScan.data(), rhs._wah.size(),
                        _decomp.data(), _decomp.size(), wahOp::XOR, context(),
                        rhs.decmprsNrWord);
  return context().timer_end();
}
