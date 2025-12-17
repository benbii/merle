#pragma once
#include "primitive.cuh"
#include <cub/block/block_load.cuh>
#include <cub/device/device_transform.cuh>
#include <cub/device/device_select.cuh>
#include <thrust/iterator/transform_iterator.h>

namespace mybmpidx {
namespace abla {

template <int nt, int vt, typename op_t>
__global__ void
method1(vprg data, uint *__restrict__ out, uint nr_grp, op_t op) {
  static_assert(nt % 32 == 0, "must have thrd count a multiple of 32");
  __shared__ union {
    uint idxbuf[3 * nt * vt];
    uint cta_grp[256]; // expand with dyn shmem lol
  } shared;

  uint idxwords[vt];
  const uint base = blockIdx.x * (nt * vt);
  #pragma unroll
  for (uint k = 0; k < vt; ++k) {
    const uint off = threadIdx.x + k * nt;
    idxwords[k] = base + off < cuda::ceil_div(data.factsz, 32)
                      ? data.nochk(base + off, &shared.idxbuf[off * 3]).x : 0;
  }
  __syncthreads();
  for (uint i = threadIdx.x; i < nr_grp; i += nt)
    shared.cta_grp[i] = 0;
  __syncthreads();

  #pragma unroll
  for (uint k = 0; k < vt; ++k) {
    while (idxwords[k] != 0) { // diverges
      // not coalesced due to `32*` in `off`
      const uint pos = 32 * (base + threadIdx.x + k * nt) + __ffs(idxwords[k]) - 1;
      idxwords[k] &= (idxwords[k] - 1);
      // for check, test the __ffs th bit in idxword.y
      auto [val, grpidx] = op(pos, false);
      if (grpidx >= nr_grp) continue;
      unsigned peers = __match_any_sync(__activemask(), grpidx);
      val = __reduce_add_sync(peers, val);
      if ((__ffs(peers) - 1) == (threadIdx.x & 31))
        atomicAdd(&shared.cta_grp[grpidx], val);
    }
  }
  __syncthreads();

  // put value back into global memory
  for (uint i = threadIdx.x; i < nr_grp; i += nt)
    if (shared.cta_grp[i] != 0)
      atomicAdd(out + i, shared.cta_grp[i]);
}

// An AND query between several columns
struct ands {
  col col_bmps[MAXCOLS];
  uint factsz = 0;
  void release() noexcept {
    for (size_t i = 0; i < MAXCOLS; ++i)
      col_bmps[i].release();
  }

  __device__ uint2 nochk(uint idxword_pos) const {
    uint idxword = 0xffffffff; // start from all 1, AND each column
    #pragma unroll
    for (uint c = 0; c < MAXCOLS; ++c) {
      uint *const *middle = this->col_bmps[c].middle;
      if (middle[0] == nullptr)
        break; // No more columns! `idxword` is the final result!

      // OR all bins in this column
      uint thecol = __ldcs(&middle[0][idxword_pos]);
      #pragma unroll
      for (uint b = 1; b < MAXBIN_PERCOL; ++b) {
        if (middle[b] == nullptr)
          break; // No more bins in this column
        thecol |= __ldcs(&middle[b][idxword_pos]);
        // all 1 is basically nonexistent so no early bailout
      }
      idxword &= thecol;
      if (idxword == 0)
        break; // Early bailout; 0 is quite frequent
    }
    return make_uint2(idxword, 0);
  }

  __device__ uint2 chk(uint idxword_pos) const {
    uint possible = 0xffffffff, uncertain = 0;
    #pragma unroll
    for (uint c = 0; c < MAXCOLS; ++c) {
      uint *const *middle = col_bmps[c].middle;
      const uint* le = col_bmps[c].leftmost, *ri = col_bmps[c].rightmost;
      if (middle[0] == nullptr && le == nullptr && ri == nullptr)
        break; // No more columns!

      uint p_cur = 0;
      #pragma unroll
      for (uint b = 0; b < MAXBIN_PERCOL; ++b) {
        if (middle[b] == nullptr)
          break; // No more bins in this column
        p_cur |= __ldcs(&middle[b][idxword_pos]);
        // all 1 is basically nonexistent so no early bailout
      }
      uint u_cur = (le ? __ldcs(&le[idxword_pos]) : 0) |
                   (ri ? __ldcs(&ri[idxword_pos]) : 0);
      p_cur |= u_cur;
      possible &= p_cur;
      uncertain = (uncertain | u_cur) & possible;
      if (possible == 0)
        break; // Early bailout; 0 is quite frequent
    }
    return make_uint2(possible, uncertain);
  }
};

template <int nt, int vt, int vt0, bool chk, typename op_t>
__global__ void
ands_grpby(ands data, uint *__restrict__ out, uint nr_grp, op_t op) {
  static_assert(nt % 32 == 0, "must have thrd count a multiple of 32");
  static_assert(vt0 < 32, "must have <32 vt");
  __shared__ union {
    typename cub::BlockScan<uint, nt>::TempStorage scan_temp;
    uint onelist[nt * vt0];
    uint cta_grp[2048]; // assuming nr_grp <= 2048
  } shared;

  uint2 idxwords[vt];
  const uint base = blockIdx.x * (nt * vt);
  #pragma unroll
  for (uint k = 0; k < vt; ++k) {
    const uint idxword_pos = base + threadIdx.x + k * nt;
    idxwords[k] = idxword_pos < cuda::ceil_div(data.factsz, 32)
                      ? (chk ? data.chk(idxword_pos) : data.nochk(idxword_pos))
                      : make_uint2(0, 0);
  }

  uint my_off, tot_1, idxword_popc = 0;
  #pragma unroll
  for (uint k = 0; k < vt; ++k)
    idxword_popc += __popc(idxwords[k].x);
  cub::BlockScan<uint, nt>(shared.scan_temp)
    .ExclusiveSum(idxword_popc, my_off, tot_1);
  __syncthreads();

  #pragma unroll
  for (uint k = 0; k < vt; ++k) {
    while (idxwords[k].x != 0) {
      const uint idxword_pos = base + threadIdx.x + k * nt;
      const uint bpos = __ffs(idxwords[k].x) - 1, row_id = idxword_pos * 32 + bpos;
      if constexpr (chk)
        shared.onelist[my_off++] =
            (idxwords[k].y & (1 << bpos)) ? (0x80000000 | row_id) : row_id;
      else
        shared.onelist[my_off++] = row_id;
      idxwords[k].x &= idxwords[k].x - 1;
    }
  } // `idxwords` retire here
  __syncthreads();

  // All threads refer to `vt` values in the onelist
  uint my_1[vt0];
  #pragma unroll
  for (uint i = 0, j = threadIdx.x; i < vt0; j += nt, i += 1) {
    if (j >= tot_1) break;
    my_1[i] = shared.onelist[j];
  }
  __syncthreads();

  // 1 values now in registers. Do actual group-by now.
  for (uint i = threadIdx.x; i < nr_grp; i += nt)
    shared.cta_grp[i] = 0;
  __syncthreads();
  #pragma unroll
  for (uint i = 0, j = threadIdx.x; i < vt0; j += nt, i += 1) {
    if (j >= tot_1) break;
    auto [val, grpidx] =
        op(my_1[i] & 0x7fffffff, chk ? bool(my_1[i] & 0x80000000) : false);
    if (grpidx >= nr_grp) continue;
    unsigned peers = __match_any_sync(__activemask(), grpidx);
    val = __reduce_add_sync(peers, val);
    if ((__ffs(peers) - 1) == (threadIdx.x & 31))
      atomicAdd(&shared.cta_grp[grpidx], val);
  }
  __syncthreads();

  // put value back into global memory
  for (uint i = threadIdx.x; i < nr_grp; i += nt)
    if (shared.cta_grp[i] != 0)
      atomicAdd(out + i, shared.cta_grp[i]);
}

namespace {
template <int nt, int vt, bool chk>
__global__ void _stg1(vprg data, uint *__restrict__ poss,
                      uint *__restrict__ uncert) {
  static_assert(nt % 32 == 0, "nt must be a multiple of 32");
  static_assert(vt > 0,        "vt must be > 0");
  // One 32-bit word per 32 rows
  const uint nwords = (data.factsz + 31u) >> 5;
  __shared__ uint s_buf[(chk ? 6 : 3) * vt * nt];
  const uint t    = threadIdx.x;
  const uint base = blockIdx.x * (nt * vt);

  #pragma unroll
  for (int k = 0; k < vt; ++k) {
    const uint wpos  = base + t + k * nt;
    uint2 res        = make_uint2(0u, 0u);
    if (wpos < nwords) {
      uint* mybuf = &s_buf[(t + k * nt) * (chk ? 6 : 3)];
      // Compute masks via virtual program (coalesced loads inside)
      res = chk ? data.chk(wpos, mybuf) : data.nochk(wpos, mybuf);
      // Coalesced stores (one 32b word per lane to consecutive addresses)
      poss[wpos] = res.x;
      if constexpr (chk) uncert[wpos] = res.y;
    }
  }
}

template <int nt, int vt, bool chk>
__global__ void _stg2(const uint *__restrict__ possible, uint factsz,
                       uint *__restrict__ out_count, uint *__restrict__ onelist,
                       const uint *__restrict__ uncertain) {
  static_assert(nt % 32 == 0, "nt must be a multiple of 32");
  static_assert(vt > 0,       "vt must be > 0");

  using BlockScan = cub::BlockScan<uint, nt>;
  __shared__ typename BlockScan::TempStorage scan_temp;
  __shared__ uint block_base;

  const uint nwords   = (factsz + 31u) >> 5;
  const uint t        = threadIdx.x;
  const uint baseword = blockIdx.x * (nt * vt);

  // Load my vt words (and optional uncertain) with tail guard
  uint poss[vt];
  uint unct[vt];  // only used if chk==true
  #pragma unroll
  for (int k = 0; k < vt; ++k) {
    const uint wpos = baseword + t + k * nt;
    const bool in   = (wpos < nwords);
    poss[k] = in ? possible[wpos] : 0u;
    if constexpr (chk) unct[k] = (in && uncertain) ? uncertain[wpos] : 0u;
  }

  // Per-thread popc over my vt words
  uint my_ones = 0;
  #pragma unroll
  for (int k = 0; k < vt; ++k) my_ones += __popc(poss[k]);

  // Block-exclusive scan: my offset in block and total ones
  uint my_off = 0, tot_ones = 0;
  BlockScan(scan_temp).ExclusiveSum(my_ones, my_off, tot_ones);
  __syncthreads();

  // One atomic to reserve a contiguous global segment; broadcast base
  if (t == 0) block_base = atomicAdd(out_count, tot_ones);
  __syncthreads();

  // Emit my set bits into global onelist at [block_base + my_off, ...)
  #pragma unroll
  for (int k = 0; k < vt; ++k) {
    uint m = poss[k];
    if (m == 0) continue;

    const uint wpos   = baseword + t + k * nt;
    const uint row_lo = wpos << 5;                // wpos * 32

    while (m) {
      const uint bpos   = __ffs(m) - 1;           // 0..31
      const uint row_id = row_lo + bpos;
      if (row_id < factsz) {
        uint out_val = row_id;
        if constexpr (chk) {
          const uint ubit = (unct[k] >> bpos) & 1u;
          if (ubit) out_val |= 0x80000000u;       // tag uncertain rows with MSB
        }
        onelist[block_base + my_off] = out_val;
        ++my_off;
      }
      m &= (m - 1);  // clear lowest 1-bit
    }
  }
}

template <int nt, int vt, bool chk, typename op_t>
__global__ void _stg3(const uint *__restrict__ onelist, uint sz,
                      uint *__restrict__ out, uint nr_grp, op_t op) {
  static_assert(nt % 32 == 0, "nt must be a multiple of 32");
  static_assert(vt > 0,       "vt must be > 0");
  extern __shared__ uint cta_grp[];  // size at launch: nr_grp * sizeof(uint)
  // zero CTA histogram
  for (uint i = threadIdx.x; i < nr_grp; i += nt) cta_grp[i] = 0;
  __syncthreads();
  const uint lane = threadIdx.x & 31u;
  const uint base = blockIdx.x * (nt * vt) + threadIdx.x;

  #pragma unroll
  for (int k = 0; k < vt; ++k) {
    const uint idx = base + k * nt;
    if (idx >= sz) break;
    uint packed = onelist[idx];
    const uint row_id = packed & 0x7FFFFFFFu;
    const bool unct   = chk ? (packed >> 31) : false;
    // user op: scattered loads likely; this stage isolates them
    auto [val, grpidx] = op(row_id, unct);
    if (grpidx >= nr_grp) continue;
    // warp-aggregated update: one atomic per unique key in the warp
    const unsigned act   = __activemask();
    const unsigned peers = __match_any_sync(act, grpidx);
    val = __reduce_add_sync(peers, val);
    const int leader = __ffs(peers) - 1;
    if (int(lane) == leader) {
      atomicAdd(&cta_grp[grpidx], val);
    }
  }
  __syncthreads();

  // flush CTA histogram to global
  for (uint i = threadIdx.x; i < nr_grp; i += nt) {
    const uint x = cta_grp[i];
    if (x) atomicAdd(out + i, x);
  }
}
} // namespace

template <int nt, int vt1, int vt2, int vt3, bool chk, typename Op>
float4 nofuse_abla(const vprg &data, uint nr_grp, Op op, uint *d_grpout,
                   uint *d_possi, uint *d_uncert, uint *d_onelist,
                   cudaStream_t stream = 0) {
  const uint factsz = data.factsz, nwords = (factsz + 31u) >> 5;
  uint *d_count = &d_grpout[nr_grp];
  cudaError_t err;
  float4 bruh = {0.0, 0.0, 0.0, 999.99};
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);

  // Index access
  cudaEventRecord(start, stream);
  err = cudaMemsetAsync(d_grpout, 0, sizeof(uint) * (1 + nr_grp), stream);
  if ((err = cudaGetLastError()) != cudaSuccess) return bruh;
  // for (size_t i = 0; i < 100; ++i) {
    _stg1<nt, vt1, chk><<<cuda::ceil_div(nwords, nt * vt1), nt, 0, stream>>>(
      data, d_possi, d_uncert);
    if ((err = cudaGetLastError()) != cudaSuccess) return bruh;
  // }
  cudaEventRecord(stop, stream); cudaEventSynchronize(stop);
  cudaEventElapsedTime(&bruh.x, start, stop);
  // printf("\t%.4f", bruh / 100);

  // Collect set bits
  cudaEventRecord(start, stream);
  // for (size_t i = 0; i < 100; ++i) {
    err = cudaMemsetAsync(d_count, 0, sizeof(uint), stream);
    if ((err = cudaGetLastError()) != cudaSuccess) return bruh;
    _stg2<nt, vt2, chk><<<cuda::ceil_div(nwords, nt * vt2), nt, 0, stream>>>(
      d_possi, factsz, d_count, d_onelist, d_uncert);
    if ((err = cudaGetLastError()) != cudaSuccess) return bruh;
  // }
  cudaEventRecord(stop, stream); cudaEventSynchronize(stop);
  cudaEventElapsedTime(&bruh.y, start, stop);
  // printf("\t%.4f", bruh / 100);

  // Do query
  uint h_count;
  cudaEventRecord(start, stream);
  // for (size_t i = 0; i < 100; ++i) {
    err = cudaMemcpyAsync(&h_count, d_count, sizeof(uint),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return bruh;
    err = cudaMemsetAsync(d_grpout, 0, sizeof(uint) * nr_grp, stream);
    if (err != cudaSuccess) return bruh;
    _stg3<nt, vt3, chk>
        <<<cuda::ceil_div(h_count, nt * vt3), nt, nr_grp * sizeof(uint),
           stream>>>(d_onelist, h_count, d_grpout, nr_grp, op);
  // }
  cudaEventRecord(stop, stream); cudaEventSynchronize(stop);
  cudaEventElapsedTime(&bruh.z, start, stop);
  // printf("\t%.4f", bruh / 100);

  cudaEventDestroy(start); cudaEventDestroy(stop);
  bruh.w = bruh.x + bruh.y + bruh.z;
  return bruh;
}

} // namespace abla
} // namespace mybmpidx
//
