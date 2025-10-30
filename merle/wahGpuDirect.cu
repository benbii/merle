#include "moderngpu/loadstore.cuh"
#include "wahGpu.cuh"
#include "moderngpu/kernel_scan.cuh"
#include "moderngpu/kernel_sortedsearch.cuh"
#include "moderngpu/kernel_load_balance.cuh"
using tplInt2 = mgpu::tuple<int, int>;

mgpu::mem_t<int> wahAndNo1(const int *up, const int *upIncScan, size_t upSz,
                           const int *down, const int *downIncScan,
                           size_t downSz, mgpu::context_t &context) {
  using namespace mgpu;
  using launch_t = mgpu::launch_box_t<mgpu::arch_20_cta<256, 10>>;
  mem_t<int> res(downSz, context); int* resData = res.data();
  auto lb = make_store_iterator<int>([=] MGPU_DEVICE (int upAt, int downAt) {
    int downWah = down[downAt], out;
    if (downWah == 0xc0000001) downWah = 0x7fffffff;
    if (upAt >= upSz) {
      out = 0x80000001;
    } else if (downWah & 0x80000000) { // 0-fill
      out = downWah;
    } else { // Tail: downVal == downWah
      int upVal = up[upAt];
      upVal = upVal & 0x80000000 ? int32_t(upVal) << 1 >> 31 : upVal;
      out = downWah & upVal;
      if (out == 0)
        out = 0x80000001;
    }
    resData[downAt] = out;
  });

  sorted_search<bounds_lower, launch_t>(downIncScan, downSz, upIncScan, upSz,
                                        lb, less_t<int>(), context);
  return res;
}

mgpu::mem_t<int> wahAnd(const int *up, const int *upExScan, size_t upSz,
                        const int *down, const int *downExScan, size_t downSz,
                        mgpu::context_t &context)
{
  using namespace mgpu;
  mem_t<int> lb(downSz + 1, context); int* lbData = lb.data();
  sorted_search_flag<bounds_lower>(downExScan, downSz + 1, upExScan, upSz + 1,
                                   lbData, mgpu::less_t<int>(), context);

  mem_t<int> workCntScan(downSz + 1, context);
  transform_scan<int, scan_type_inc>([=] MGPU_DEVICE (int thId) {
    if (thId == 0) return 0;
    if (((uint32_t)down[thId - 1] >> 30) != 3)
      return 1;
    int lbId = lbData[thId], prevLbId = lbData[thId - 1];
    return 0x7fffffff & int(lbId - prevLbId + 1 - (unsigned(prevLbId) >> 31));
  }, downSz + 1, workCntScan.data(), plus_t<int>(),
    discard_iterator_t<int>(), context);
  int nRes; dtoh(&nRes, workCntScan.data() + downSz, 1);

  mem_t<int> res(nRes, context); int* resData = res.data();
  transform_lbs([=] MGPU_DEVICE(int idx, int downAt /*seg*/, int rank, tplInt2 c) {
    int downWah = get<1>(c), out; // down[seg]
    uint32_t ty = (uint32_t)downWah >> 30;
    if (ty == 2)
      out = downWah;
    else {
      int upAt = get<0>(c), found = uint32_t(upAt) >> 31;
      upAt &= 0x7fffffff;
      upAt = upAt + rank - 1 + found;
      if (upAt >= upSz) { resData[idx] = 0x80000001; return; }
      int upWah = up[upAt];
      if (ty == 3) {  // down is 1-fill
        if (!(upWah & 0x80000000)) { // up is tail
          out = upWah;
        } else {
          int downBeg = downExScan[downAt],
              downEnd = downBeg + (downWah & 0x3fffffff),
              upBeg = upExScan[upAt], upEnd = upBeg + (upWah & 0x3fffffff),
              resBeg = rank == 0 ? downBeg : upBeg,
              resEnd = mgpu::min(upEnd, downEnd), outCnt = resEnd - resBeg;
          out = outCnt + (upWah & 0xc0000000);
        }
      } else { // down is tail
        int upVal = upWah & 0x80000000 ? int32_t(upWah) << 1 >> 31 : upWah;
        out = upVal & downWah;
      }
    }
    resData[idx] = out;
  },
  nRes, workCntScan.data(), downSz + 1,
  mgpu::make_tuple(lbData, down), context);
  return res;
}


template <int nt, int vt>
struct wahShm_t {
  int values[nt * vt];
  typename mgpu::cta_scan_t<nt, int>::storage_t scan;
};

template <int nt, int vt, int vt0, int vtmax>
MGPU_DEVICE void
innerDwnswp(int tid, int cta, mgpu::range_t cta_rg, const int *spine_red,
            const int *lb_dat, int *cntscan_dat, wahShm_t<nt, vtmax> &shared) {
  using namespace mgpu;
  // Load a tile to register in thread order.
  array_t<int, vt> x = mem_to_reg_strided<nt, vt, vt0, true>(
      lb_dat + cta_rg.begin, tid, cta_rg.count());
  reg_to_shared_strided<nt, vt>(x, tid, shared.values);
  thread_iterate<vt>([&](int i, int j) {
    x[i] = 1 - (unsigned(shared.values[j]) >> 31);
  }, tid);

  // Scan the unmatched flag with carry-in from the partials.
  // (clangd dies here for some reason and mistakenly emits errors)
  x = cta_scan_t<nt, int>().scan<vt, vt0>(
      tid, x, shared.scan, spine_red[cta], cta > 0, cta_rg.count(),
      plus_t<int>(), 0, scan_type_exc).scan;

  // Add scan result in threads to lb_dat in shared memory
  thread_iterate<vt>(
      [&](int i, int j) { (shared.values[j] += x[i]) &= 0x7fffffff; }, tid);
  __syncthreads();

  x = shared_to_reg_strided<nt, vt>(shared.values, tid, false);
  reg_to_mem_strided<nt, vt, vt0, true>(x, tid, cta_rg.count(),
                                        cntscan_dat + cta_rg.begin);

  // strided_iterate<nt, vt, vt0, true>( [&](int i, int j) { 
  //   cntscan_dat[j + cta_rg.begin] = shared.values[j]; 
  // }, tid, cta_rg.count());

  // shared_to_mem<nt, vt>(shared.values, tid, cta_rg.count(),
  //                       cntscan_dat + cta_rg.begin);
}

static std::pair<mgpu::mem_t<int>, mgpu::mem_t<int>>
pleaseWork(const int* needles, int num_needles, const int* haystack,
           int num_haystack, mgpu::context_t &context)
{
  using namespace mgpu;
  typedef launch_box_t<arch_20_cta<256, 11>> launch_t;
  typedef typename launch_t::sm_ptx params_t;
  enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };

  // Partition the needles and haystacks into tiles.
  mem_t<int> partitions = merge_path_partitions<bounds_lower>(needles, num_needles,
    haystack, num_haystack, launch_t::nv(context), less_t<int>(), context);
  const int *mp_data = partitions.data();

  // Scan of unmatched counts in each cta
  cta_dim_t cta = launch_t::cta_dim(context.ptx_version());
  int num_ctas = div_up(num_haystack + num_needles, cta.nv());
  mem_t<int> spine_match(num_ctas + 1, context);
  int *spine_match_dat = spine_match.data();

  // Return value
  std::pair<mgpu::mem_t<int>, mgpu::mem_t<int>> ret = std::make_pair(
    mem_t<int>(num_needles, context),
    mem_t<int>(num_needles, context)
  );
  int* lb_dat = ret.first.data(), *cntscan_dat = ret.second.data();

  // Upsweep: produce both the lb (sorted_search_index) and reduces unmatch count
  // the first parts of this func is basically sorted_search_flag<bounds_lower>
  auto upswp_f = [=]MGPU_DEVICE(int tid, int cta) {
    __shared__ union {
      int keys[nt * (vt + 1)];
      int indices[nv];
      typename cta_reduce_t<nt, int>::storage_t reduce;
    } shared;

    // The range this CTA works on. Corresponds to a0 a1 b0 b1 in mgpu1
    merge_range_t cta_rg = compute_merge_range(
        num_needles, num_haystack, cta, nv, mp_data[cta], mp_data[cta + 1]);
    // aCount2 bCount2 in mgpu1
    int cta_nr_needle = cta_rg.a_count(), cta_nr_haystack = cta_rg.b_count(), out;
    if (cta_nr_needle != 0)
  { // don't want to indent this huge single branch

    // Equivalent to bounds_lower in sorted_search_flg
    bool // left_a = false,
        // left_b = bounds == bounds_upper && cta_rg.b_begin > 0,
        right_a = cta_rg.a_end < num_needles,
        right_b = cta_rg.b_end < num_haystack,
        extended = right_a && right_b; // && (bounds == bounds_lower || left_b);

    // Range of effective elements in shared memory including halos;
    merge_range_t shmem_rg = {
      .a_begin = 0 /*left_a*/, .a_end = cta_nr_needle + right_a,
      .b_begin = cta_nr_needle + right_a /*+ left_b*/,
      .b_end = cta_nr_needle + right_a /*+ left_b*/ + cta_nr_haystack + right_b
    };

    array_t<int, vt + 1> x = load_two_streams_reg_halo<nt, vt, int>(
      needles + cta_rg.a_begin /*-left_a*/, shmem_rg.a_end /*-a_begin*/,
      haystack + cta_rg.b_begin /*- left_b*/, shmem_rg.b_end - shmem_rg.a_end, tid
    );
    reg_to_shared_strided<nt, vt + 1>(x, tid, shared.keys);

    int diag = vt * tid;
    int mp = merge_path<bounds_lower>(shared.keys, cta_nr_needle,
        shared.keys + shmem_rg.b_begin, cta_nr_haystack, diag, less_t<int>());
    merge_range_t thread_rg = shmem_rg.partition(mp, diag);

    array_t<int, vt + 1> thrd_idx =
      extended ? serial_search_flg<vt, bounds_lower, false, true>(
          shared.keys, thread_rg, cta_rg.b_begin - shmem_rg.b_begin, less_t<int>()
      ) : serial_search_flg<vt, bounds_lower, true, true>(
          shared.keys, thread_rg, cta_rg.b_begin - shmem_rg.b_begin, less_t<int>()
      );
    // if (cta == 0)
    //   printf("tid %d acc %d\n", tid, thrd_idx[vt]);

    iterate<vt>([&] (int i) {
      if (thrd_idx[i] != -1)
        shared.indices[mp++] = thrd_idx[i];
    }); // lower bound register to shared
    __syncthreads();
    shared_to_mem<nt, vt>(shared.indices, tid, cta_nr_needle,
                          lb_dat + cta_rg.a_begin, true);

    int all_reduce = cta_reduce_t<nt, int>().reduce(tid, thrd_idx[vt],
        shared.reduce, nt, plus_t<int>(), false);
    out = cta_nr_needle - all_reduce;
  } else out = 0;
    if (tid == 0)
      spine_match_dat[cta] = out;
  };

  cta_launch<launch_t>(upswp_f, num_ctas, context);
  // Spine phase: Scan unmatch count
  scan<scan_type_exc>(spine_match_dat, num_ctas, spine_match_dat, plus_t<int>(),
                      spine_match_dat + num_ctas, context);

  // Downsweep phase
  auto downswp_f = [=]MGPU_DEVICE(int tid, int cta) {
    __shared__ wahShm_t<nt, vt> shared;
    range_t cta_rg = compute_merge_range(num_needles, num_haystack,
        cta, nv, mp_data[cta], mp_data[cta + 1]).a_range();
    if (cta_rg.count() <= 0)
      return;

    if (cta_rg.count() < nt * 4) {
      innerDwnswp<nt, 4, 0, vt>(tid, cta, cta_rg, spine_match_dat, lb_dat,
                                 cntscan_dat, shared);
    } else if (cta_rg.count() < nt * 6) {
      innerDwnswp<nt, 6, 4, vt>(tid, cta, cta_rg, spine_match_dat, lb_dat,
                                 cntscan_dat, shared);
    } else if (cta_rg.count() < nt * 8) {
      innerDwnswp<nt, 8, 6, vt>(tid, cta, cta_rg, spine_match_dat, lb_dat,
                                 cntscan_dat, shared);
    } else {
      innerDwnswp<nt, vt, 8, vt>(tid, cta, cta_rg, spine_match_dat, lb_dat,
                                  cntscan_dat, shared);
    }
  };
  cta_launch<launch_t>(downswp_f, num_ctas, context);
  int nRes; dtoh(&nRes, cntscan_dat + num_needles - 1, 1);
  // std::cout << "pleaseWork reduction " << nRes << std::endl;
  return ret;
}


mgpu::mem_t<int> wahOr(const int *up, const int *upExScan, size_t upSz,
                       const int *down, const int *downExScan, size_t downSz,
                       mgpu::context_t &context)
{
  using namespace mgpu;
  auto [lb, workCntScan] =
      pleaseWork(downExScan, downSz + 1, upExScan, upSz + 1, context);
  int* lbData = lb.data();
  int nRes; dtoh(&nRes, workCntScan.data() + downSz, 1);
  // std::cout << nRes << std::endl;

  mem_t<int> res(nRes, context); int* resData = res.data();
  transform_lbs([=] MGPU_DEVICE(int idx, int downAt, int rank, tplInt2 c) {
    int upAt = get<0>(c), found = uint32_t(upAt) >> 31;
    int downWah = get<1>(c), out; // down[seg]
    upAt = (upAt & 0x7fffffff) + rank - 1 + found;
    if (upAt >= upSz) {
      if (rank != 0) // will only run at most once
        resData[idx] =
            downExScan[downAt + 1] - upExScan[upSz] + (downWah & 0xc0000000);
      else resData[idx] = downWah;
      return;
    }
    int upWah = up[upAt];
    if (((uint32_t)downWah >> 31) == 0) { // down is tail
      int upVal = upWah & 0x80000000 ? int32_t(upWah) << 1 >> 31 : upWah;
      out = (upVal | downWah) & 0x7fffffff;
    } else if (((uint32_t)upWah >> 31) == 0) { // up is tail, down is fill
      int downVal = downWah << 1 >> 31;
      out = (downVal | upWah) & 0x7fffffff;
    } else { // both are fills
      int downBeg = downExScan[downAt],
          downEnd = downBeg + (downWah & 0x3fffffff), upBeg = upExScan[upAt],
          upEnd = upBeg + (upWah & 0x3fffffff),
          resBeg = rank == 0 ? downBeg : upBeg,
          resEnd = mgpu::min(upEnd, downEnd), outCnt = resEnd - resBeg;
      out = outCnt + ((upWah | downWah) & 0xc0000000);
    }
    resData[idx] = out;
  },
  nRes, workCntScan.data(), downSz + 1,
  mgpu::make_tuple(lbData, down), context);
  return res;
}

mgpu::mem_t<int> wahXor(const int *up, const int *upExScan, size_t upSz,
                        const int *down, const int *downExScan, size_t downSz,
                        mgpu::context_t &context)
{
  using namespace mgpu;
  auto [lb, workCntScan] =
      pleaseWork(downExScan, downSz + 1, upExScan, upSz + 1, context);
  int* lbData = lb.data();
  int nRes; dtoh(&nRes, workCntScan.data() + downSz, 1);

  mem_t<int> res(nRes, context); int* resData = res.data();
  transform_lbs([=] MGPU_DEVICE(int idx, int downAt, int rank, tplInt2 c) {
    int upAt = get<0>(c), found = uint32_t(upAt) >> 31;
    int downWah = get<1>(c), out; // down[seg]
    upAt = (upAt & 0x7fffffff) + rank - 1 + found;
    if (upAt >= upSz) {
      if (rank != 0) // will only run at most once
        resData[idx] =
            downExScan[downAt + 1] - upExScan[upSz] + (downWah & 0xc0000000);
      else resData[idx] = downWah;
      return;
    }
    int upWah = up[upAt];
    if (((uint32_t)downWah >> 31) == 0) { // down is tail
      int upVal = upWah & 0x80000000 ? int32_t(upWah) << 1 >> 31 : upWah;
      out = (upVal ^ downWah) & 0x7fffffff;
    } else if (((uint32_t)upWah >> 31) == 0) { // up is tail, down is fill
      int downVal = downWah << 1 >> 31;
      out = (downVal ^ upWah) & 0x7fffffff;
    } else { // both are fills
      int downBeg = downExScan[downAt],
          downEnd = downBeg + (downWah & 0x3fffffff), upBeg = upExScan[upAt],
          upEnd = upBeg + (upWah & 0x3fffffff),
          resBeg = rank == 0 ? downBeg : upBeg,
          resEnd = mgpu::min(upEnd, downEnd), outCnt = resEnd - resBeg;
      out = outCnt + ((upWah ^ downWah) & 0x40000000) + 0x80000000;
    }
    resData[idx] = out;
  },
  nRes, workCntScan.data(), downSz + 1,
  mgpu::make_tuple(lbData, down), context);
  return res;
}
