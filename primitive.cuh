#pragma once
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda/std/array>
#ifdef __CLANG__CUDA_MATH_FORWARD_DECLARES_H__
#define __ldcs(x) *(x)
#endif

namespace mybmpidx {
static constexpr uint ELIMINATED = 0x44f8a1ef;
static_assert(sizeof(uint) == 4, "Need 4B word!");

// Group by, using atomics on shared memory. Must have few enough groups.
// `out` is nr. of groups elements long.
// Each op function returns [result, position-to-be-grouped-by] on success (i.e.
// the row should be kept); [undefined, ELIMINATED] on failure. The `result` can
// be the selected row, value to be aggregated, etc. Only one select/aggregate
// result per row is supported in scalar op functions.
template <int nt, int vt, typename op_fun_t>
__global__ void grpby(uint maxid, uint *__restrict__ out, uint nr_grp,
                      op_fun_t fun) {
  static_assert(nt % 32 == 0, "must have thrd count a multiple of 32");
  extern __shared__ uint cta_grp[];
  for (uint i = threadIdx.x; i < nr_grp; i += nt)
    cta_grp[i] = 0;
  __syncthreads();

  // do works
  const uint base = blockIdx.x * nt * vt + threadIdx.x;
  #pragma unroll
  for (uint i = 0; i < vt; ++i) {
    // strides mem accesses in `fun`
    // (`fun` is most of the time a lambda capture)
    const uint my_idx = i * nt + base;
    if (my_idx >= maxid) break;
    auto [redval, grpidx] = fun(my_idx);
    if (grpidx >= nr_grp) // ELIMINATED should > nr_grp
      continue;
    unsigned peers = __match_any_sync(__activemask(), grpidx);
    redval = __reduce_add_sync(peers, redval);
    int leader = __ffs(peers) - 1;
    if ((threadIdx.x & 31) == leader)
      atomicAdd(cta_grp + grpidx, redval);
  }
  __syncthreads();

  // put value back into global memory
  for (uint i = threadIdx.x; i < nr_grp; i += nt)
    if (cta_grp[i] != 0)
      atomicAdd(out + i, cta_grp[i]);
}

template <int nt, int vt, int nr_grp, typename op_fun_t>
__global__ void grpby_small(uint maxid, uint *__restrict__ out, op_fun_t fun) {
  using Item = cuda::std::array<uint, nr_grp>;
  using BlockReduce = cub::BlockReduce<Item, nt>;
  __shared__ typename BlockReduce::TempStorage temp;
  auto add_item = [] __device__(const Item &a, const Item &b) {
    Item c;
    #pragma unroll
    for (int i = 0; i < nr_grp; ++i)
      c[i] = a[i] + b[i];
    return c;
  };

  Item reses{};
  const uint base = blockIdx.x * nt * vt + threadIdx.x;
  #pragma unroll
  for (uint i = 0; i < vt; ++i) {
    const uint my_idx = i * nt + base;
    if (my_idx >= maxid) break;
    auto [redval, grpidx] = fun(my_idx);
    if (grpidx >= nr_grp) continue;
    #pragma unroll
    for (uint g = 0; g < nr_grp; ++g) {
      if (grpidx == g) reses[g] += redval;
    }
  }

  reses = BlockReduce(temp).Reduce(reses, add_item);
  if (threadIdx.x == 0) {
    #pragma unroll
    for (uint i = 0; i < nr_grp; ++i)
      atomicAdd(&out[i], reses[i]);
  }
}

// Store results of a function into a dense bitmap.
// This is a generic kernel useful for, but not necessarily bound to, creating
// bitmap indices.
template <int nt, int vt, typename op_fun_t>
__global__ void bmpcreate(uint maxid, uint *__restrict__ out, op_fun_t fun) {
  static_assert(nt % 32 == 0, "must have thrd count a multiple of 32");
  const uint base = blockIdx.x * nt * vt + threadIdx.x;
  #pragma unroll
  for (uint i = 0; i < vt; ++i) {
    uint idx = base + nt * i;
    int ok = (idx < maxid) ? (int)fun(idx) : 0;
    ok = __ballot_sync(0xffffffff, ok);
    if ((idx & 31) == 0)
      out[idx / 32] = ok;
  }
}

template <int nt, int vt, typename T, typename Fetch, typename Compar>
__global__ void bmpcreate(uint maxid, uint **out, Fetch fetch, Compar compar,
                          const T *mins, const T* maxes, uint nrcol) {
  static_assert(nt % 32 == 0, "must have thrd count a multiple of 32");
  const uint base = blockIdx.x * nt * vt + threadIdx.x;
  // no unroll is 15% faster. no idea how.
  for (uint i = 0; i < vt; ++i) {
    uint idx = base + nt * i;
    if (idx >= maxid) break;
    T val = fetch(idx);
    for (uint j = 0; j < nrcol; ++j) {
      int ok = __ballot_sync(0xffffffff, compar(val, mins[j], maxes[j]));
      if (threadIdx.x % 32 == 0)
        out[j][idx / 32] = ok;
    }
  }
}

// Routines and types for creating and querying with bitmap index
static constexpr size_t MAXBIN_PERCOL = 4, MAXCOLS = 5, MAXNINSTR = 6;

// Create a bitmap based on column values. Calls `bmpcreate`.
// ALL ranges are left inclusive 
uint *create_select(const void *__restrict__ src, size_t nbit, uint factsz,
                    uint min, uint max);
uint *create_join(const uint32_t *fk, const void *__restrict__ dimattr,
                  size_t nbit, uint factsz, uint dimsz, uint min, uint max);
// ONLY FREE OUT[0]!!!
size_t create_bin(const uint *__restrict__ fk, const void *__restrict__ attr,
                  size_t nbit, uint factsz, uint dimsz, const uint64_t *min,
                  const uint64_t *max, size_t ncol, uint **out);

// Bitmap index of a column
struct col {
  // At most 2 bins may not entirely be included inside the query range,
  // potentially requiring candidate check depending on other columns.
  uint* leftmost, *rightmost;
  // Bins inside the query range
  uint* middle[MAXBIN_PERCOL];

  col() noexcept { memset(this, 0, sizeof(col)); }
  // NOT AN AUTO FREE DTOR
  void release() noexcept {
    if (leftmost) cudaFree(leftmost);
    if (rightmost) cudaFree(rightmost);
    for (size_t i = 0; i < MAXBIN_PERCOL; ++i)
      if (middle[i]) cudaFree(middle[i]);
  }
};

// Virtual bmp query program
struct vprg {
  col col_bmps[MAXCOLS];
  uint factsz;

  enum ops: uint8_t {
    END, // no more programs
    AND, OR, // dest reg &= (|=) src reg
    ANDM, ORM, // dest reg &= (|=) col_bmps[src]
  };
  // Assembly-like bitmap query "instruction". It loads and operates on 3
  // "virtual registers", which maps to 3*(vt*nt*4B) shared memory.
  struct instr {
    ops opcode;
    uint8_t dstreg;
    uint8_t src; // also index to memory operand in ASSIGN
  };
  // In the end, result must be placed in r0
  // The first instruction is usually ORM to read a memory operand.
  instr instrs[MAXNINSTR];

  vprg() noexcept { memset(this, 0, sizeof(vprg)); }
  void release() noexcept {
    for (size_t i = 0; i < MAXCOLS; ++i)
      col_bmps[i].release();
  }
  static constexpr uint shmsz(uint vt, uint nt, uint nr_grp, bool chk) {
    return std::max((chk ? 6 : 3) * vt * nt, nr_grp) * sizeof(uint);
  }

  // Each position uses 3 words of shmem
  __device__ uint2 nochk(uint wordpos, uint* myshm) {
    myshm[0] = myshm[1] = myshm[2] = 0;
    #pragma unroll
    for (uint i = 0; i < MAXNINSTR; ++i) {
      const instr &instr_ = this->instrs[i];
      if (instr_.opcode == vprg::END) break;
      // NOTE: in real datasets there are virtually no column reuse
      switch (instr_.opcode) {
      case vprg::AND: myshm[instr_.dstreg] &= myshm[instr_.src]; break;
      case vprg::OR:  myshm[instr_.dstreg] |= myshm[instr_.src]; break;

      case vprg::ANDM: {
        uint dst = myshm[instr_.dstreg], thecol = 0;
        if (dst == 0) break; // 0 words are fairly common
        const col &memop = col_bmps[instr_.src];
        #pragma unroll
        for (uint k = 0; k < MAXBIN_PERCOL && memop.middle[k]; ++k)
          thecol |= __ldcs(&memop.middle[k][wordpos]); // full 1 words are nonexistent
        dst &= thecol;
        myshm[instr_.dstreg] = dst; break;
      }

      case vprg::ORM: {
        uint dst = myshm[instr_.dstreg];
        const col &memop = col_bmps[instr_.src];
        #pragma unroll
        for (uint k = 0; k < MAXBIN_PERCOL && memop.middle[k]; ++k)
          dst |= __ldcs(&memop.middle[k][wordpos]);
        myshm[instr_.dstreg] = dst; break;
      }
      default: __builtin_unreachable();
      }
    }
    return make_uint2(myshm[0], 0);
  }

  // Each position uses 6 words of shmem
  __device__ uint2 chk(uint wordpos, uint* myshm) {
    myshm[0] = myshm[1] = myshm[2] = 0; // possible
    myshm[3] = myshm[4] = myshm[5] = 0; // uncertain
    #pragma unroll
    for (uint i = 0; i < MAXNINSTR; ++i) {
      const instr &instr_ = this->instrs[i];
      if (instr_.opcode == vprg::END) break;

      switch (instr_.opcode) {
      case vprg::AND:
        // p_dst = p_src & p_prev
        myshm[instr_.dstreg] &= myshm[instr_.src];
        // u_dst = (u_src | u_prev) & p_dst
        (myshm[instr_.dstreg + 3] |= myshm[instr_.src + 3]) &= myshm[instr_.dstreg];
        break;
      case vprg::OR: {
        const uint psrc = myshm[instr_.src], pprev = myshm[instr_.dstreg];
        const uint usrc = myshm[instr_.src + 3], csrc = psrc & ~usrc;
        // Uncertain rows after this column operation are either previously
        // uncertain and remain unconfirmed this column (~csrc), or unconfirmed
        // this column and not included beforehand.
        // u_dst = (u_src & ~p_prev) | (u_prev & ~c_src)
        (myshm[instr_.dstreg + 3] &= ~csrc) |= (usrc & ~pprev);
        // Possible rows are the union of those from previous and this column
        // p_dst = p_src | p_prev
        myshm[instr_.dstreg] = pprev | psrc;
        break;
      }

      case vprg::ANDM: {
        uint p_dst = myshm[instr_.dstreg];
        if (p_dst == 0) break; // nothing is possible
        const col &memop = this->col_bmps[instr_.src];
        uint c_src = 0,
             u_src = (memop.leftmost ? __ldcs(&memop.leftmost[wordpos]) : 0) |
                     (memop.rightmost ? __ldcs(&memop.rightmost[wordpos]) : 0);
        // full 1 words do not exist; no early bailout
        #pragma unroll
        for (uint k = 0; k < MAXBIN_PERCOL && memop.middle[k]; ++k)
          c_src |= __ldcs(&memop.middle[k][wordpos]);
        uint u_dst = myshm[instr_.dstreg + 3];
        // p_dst = (c_src | u_src) & p_prev
        p_dst &= (u_src | c_src);
        // u_dst = (u_src | u_prev) & p_dst
        (u_dst |= u_src) &= p_dst;
        myshm[instr_.dstreg] = p_dst;
        myshm[instr_.dstreg + 3] = u_dst;
        break;
      }

      case vprg::ORM: {
        const col &memop = this->col_bmps[instr_.src];
        uint c_src = 0,
             u_src = (memop.leftmost ? __ldcs(&memop.leftmost[wordpos]) : 0) |
                     (memop.rightmost ? __ldcs(&memop.rightmost[wordpos]) : 0);
        #pragma unroll
        for (uint k = 0; k < MAXBIN_PERCOL && memop.middle[k]; ++k)
          c_src |= __ldcs(&memop.middle[k][wordpos]);
        uint p_dst = myshm[instr_.dstreg], u_dst = myshm[instr_.dstreg + 3];
        // u_dst = (u_src & ~p_prev) | (u_prev & ~c_src)
        (u_dst &= ~c_src) |= (u_src & ~p_dst);
        // p_dst = p_src | p_prev
        p_dst |= u_src | c_src;
        myshm[instr_.dstreg] = p_dst;
        myshm[instr_.dstreg + 3] = u_dst;
        break;
      }
      default: __builtin_unreachable();
      }
    }
    return make_uint2(myshm[0], myshm[3]);
  }

  // Only used in adapting compressed operations.
  // These are host functions that calls kernels internally, because decoding
  // and operating on compressed bitmaps are multi-phase and hence not fused.
  uint* _cmprs_adapt_run() const;
};

template <int nt, int vt, int vt0, bool chk, typename op_t>
__global__ void
vprg_grpby(vprg data, uint *__restrict__ out, uint nr_grp, op_t op) {
  static_assert(nt % 32 == 0, "must have thrd count a multiple of 32");
  static_assert(vt0 < 32, "must have <32 vt0");
  __shared__ union {
    uint idxbuf[(chk ? 6 : 3) * nt * vt];
    typename cub::BlockScan<uint, nt>::TempStorage scan_temp;
    uint onelist[nt * vt0];
    uint cta_grp[0]; // expand with dyn shmem lol
  } shared;

  uint2 idxwords[vt];
  const uint base = blockIdx.x * (nt * vt);
  #pragma unroll
  for (uint k = 0; k < vt; ++k) {
    const uint off = threadIdx.x + k * nt;
    idxwords[k] = base + off < cuda::ceil_div(data.factsz, 32)
                      ? (chk ? data.chk(base + off, &shared.idxbuf[off * 6])
                             : data.nochk(base + off, &shared.idxbuf[off * 3]))
                      : make_uint2(0, 0);
    // compiler should be smart enough to eliminate idxwords[*].y on nochk
  }
  __syncthreads();

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

// Helper type to create a bitmap index for a specific query
struct recipe {
  uint factsz; // size of fact table
  // arguments to create_join or select (if corresponding foreign key is NULL)
  uint nbit[MAXCOLS], min[MAXCOLS], max[MAXCOLS], dimsz[MAXCOLS];
  uint32_t* fk[MAXCOLS];
  void* attr[MAXCOLS];
  static constexpr uint nodim = 0xffffffff;

  // Perfect: all bitmaps index bin boundary "perfectly align" with query. If
  // the query is 3<=attr1<11 AND 4<=attr2<13, then we create the 2 exact bitmaps:
  // col_bmps[0].middle[0] = create_select(..., 3, 11)
  // col_bmps[1].middle[0] = create_select(..., 4, 13)
  // all other fields are NULL.
  vprg perfect(const vprg::instr instrs[MAXNINSTR]) const;

  // ManyOr: the query includes 3 bins, but the lower bound of
  // leftmost bin and upper bound of rightmost bin align with the query. If max
  // - min is not divisible by 3 then remainder goes to final bin. If max - min
  // < 3, then fall back to "perfect" creation.
  // Ex: query 3<=attr1<11, bins are {3,4}, {5,6}, {7,8,9,10}.
  vprg many_or(const vprg::instr instrs[MAXNINSTR]) const;

  // CandChk: bin boundaries must be a multiple of the given interval. Place the
  // bin into leftmost or rightmost if the bin is not fully included in the bin.
  // For now if >MAXBIN_PERCOL middle bins, only keep the first MAXBIN_PERCOL.
  vprg candchk(const vprg::instr instrs[MAXNINSTR]) const;

  // Only used in adapting operation between compressed bit vectors
  vprg _cmprs_adapt_perfect(const vprg::instr instrs[MAXNINSTR]) const;
  vprg _cmprs_adapt_manyor(const vprg::instr instrs[MAXNINSTR]) const;
};

} // namespace mybmpidx
#ifdef __CLANG__CUDA_MATH_FORWARD_DECLARES_H__
#undef __ldcs
#endif
