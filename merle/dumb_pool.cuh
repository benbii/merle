#pragma once
#include "moderngpu/context.cuh"
#include <cstddef>
#define NEXT256(x) ((x + 255) / 256 * 256)

class verbose_context_t : public mgpu::context_t {
  size_t _al = 0;
  mgpu::context_t *_ctx;

public:
  verbose_context_t(mgpu::context_t& ctx) : _ctx(&ctx) {}
  void* alloc(size_t size, mgpu::memory_space_t space) override {
    ++_al;
    void* alloced = _ctx->alloc(size, space);
    printf("Alloc: %zu %p %s %zu\n", size, alloced,
           space == mgpu::memory_space_host ? "host" : "dev", _al);
    fflush(stdout);
    return alloced;
  }

  void free(void* p, mgpu::memory_space_t space) override {
    --_al;
    printf("Free: %p %s %zu\n", p,
           space == mgpu::memory_space_host ? "host" : "dev", _al);
    fflush(stdout);
    _ctx->free(p, space);
  }

  const cudaDeviceProp& props() const override { return _ctx->props(); }
  int ptx_version() const override { return _ctx->ptx_version(); }
  cudaStream_t stream() override { return _ctx->stream(); }
  virtual void synchronize() override { _ctx->synchronize(); }

  virtual cudaEvent_t event() override { return _ctx->event(); }
  virtual void timer_begin() override { _ctx->timer_begin(); }
  virtual double timer_end() override { return _ctx->timer_end(); }
};

class dumb_pool_t : public mgpu::standard_context_t {
protected:
  // std::array<ptrdiff_t, 48> _alloc_offs;
  ptrdiff_t _alloc_offs[48];
  ptrdiff_t _pool_sz;
  ptrdiff_t _at;
  void* _ptr;

public:
  dumb_pool_t(size_t pool_sz, bool print_prop = true,
              cudaStream_t stream = nullptr)
      : mgpu::standard_context_t(print_prop, stream),
        _pool_sz(NEXT256(pool_sz)), _at(0) {
    auto res = cudaMalloc(&_ptr, _pool_sz);
    if (cudaSuccess != res)
      throw mgpu::cuda_exception_t(res);
    _alloc_offs[0] = 1;
  }

  void* alloc(size_t size, mgpu::memory_space_t space) override {
    if (space != mgpu::memory_space_device)
      return mgpu::standard_context_t::alloc(size, space);
    if (_at >= 47)
      throw std::out_of_range("Too much allocation on dumb pool - max 48");
    assert(_alloc_offs[_at] & 1);
    ptrdiff_t start = _alloc_offs[_at] & ~0xffll;
    ptrdiff_t finish = NEXT256(start + size);
    if (finish > _pool_sz)
      throw std::runtime_error("Dumb pool exhausted");
    _alloc_offs[_at] = start;
    ++_at;
    _alloc_offs[_at] = finish + 1;
    return (uint8_t*)_ptr + start;
  }

  void free(void* p, mgpu::memory_space_t space) override {
    if (space != mgpu::memory_space_device)
      return mgpu::standard_context_t::free(p, space);
    ptrdiff_t toFree = 0;
    while (toFree < _at) {
      if (_alloc_offs[toFree] == (uint8_t*)p - (uint8_t*)_ptr)
        break;
      ++toFree;
    }
    if (toFree == _at)
      throw std::runtime_error("Not allocated on dumb pool");

    ++_alloc_offs[toFree];
    if (toFree == _at - 1) {
      while (_at >= 0 && (_alloc_offs[_at] & 1))
        --_at;
      ++_at;
    }
  }

  ~dumb_pool_t() { cudaFree(_ptr); }
};

#undef NEXT256
