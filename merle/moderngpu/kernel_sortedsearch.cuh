#pragma once
#include "cta_reduce.cuh"
#include "kernel_scan.cuh"
#include "cta_search.cuh"
#include "search.cuh"
#include "transform.cuh"

BEGIN_MGPU_NAMESPACE

template<bounds_t bounds, typename launch_arg_t = empty_t,
  typename needles_it, typename haystack_it, typename indices_it,
  typename comp_it>
void sorted_search(needles_it needles, int num_needles, haystack_it haystack,
  int num_haystack, indices_it indices, comp_it comp, context_t& context) {

  typedef typename conditional_typedef_t<launch_arg_t, 
    launch_box_t<
      arch_20_cta<128, 15>,
      arch_35_cta<128, 11>,
      arch_52_cta<128, 15>,
      arch_86_cta<256, 11>
    >
  >::type_t launch_t;

  typedef typename std::iterator_traits<needles_it>::value_type type_t;

  // Partition the needles and haystacks into tiles.
  mem_t<int> partitions = merge_path_partitions<bounds>(needles, num_needles,
    haystack, num_haystack, launch_t::nv(context), comp, context);
  const int* mp_data = partitions.data();

  auto k = [=]MGPU_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };
    
    __shared__ union {
      type_t keys[nv + 1];
      int indices[nv];
    } shared;

    // Load the range for this CTA and merge the values into register.
    int mp0 = mp_data[cta + 0];
    int mp1 = mp_data[cta + 1];
    merge_range_t range = compute_merge_range(num_needles, num_haystack, cta,
      nv, mp0, mp1);

    // Merge the values needles and haystack.
    merge_pair_t<type_t, vt> merge = cta_merge_from_mem<bounds, nt, vt>(
      needles, haystack, range, tid, comp, shared.keys);

    // Store the needle indices to shared memory.
    iterate<vt>([&](int i) {
      if(merge.indices[i] < range.a_count()) {
        int needle = merge.indices[i];
        int haystack = range.b_begin + vt * tid + i - needle;
        shared.indices[needle] = haystack;
      }
    });
    __syncthreads();

    shared_to_mem<nt, vt>(shared.indices, tid, range.a_count(), 
      indices + range.a_begin);
  };

  cta_transform<launch_t>(k, num_needles + num_haystack, context);
}

template <bounds_t bounds, typename launch_arg_t = empty_t, typename needles_it,
          typename haystack_it, typename indices_it, typename comp_it>
void sorted_search_flag(needles_it needles, int num_needles,
                        haystack_it haystack, int num_haystack,
                        indices_it indices, comp_it comp, context_t &context) {
  // Host part is identical to sorted search without flags
  typedef typename conditional_typedef_t<launch_arg_t, 
    launch_box_t<
      arch_20_cta<128, 15>,
      arch_35_cta<128, 11>,
      arch_52_cta<128, 15>,
      arch_86_cta<256, 11>
    >
  >::type_t launch_t;
  typedef typename std::iterator_traits<needles_it>::value_type type_t;
  // Partition the needles and haystacks into tiles.
  mem_t<int> partitions = merge_path_partitions<bounds>(needles, num_needles,
    haystack, num_haystack, launch_t::nv(context), comp, context);
  const int *mp_data = partitions.data();

  // Device part is identical to non-flag version until merge
  auto k = [=]MGPU_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };
    __shared__ union {
      type_t keys[nt * (vt + 1)];
      int indices[nv];
      typename cta_reduce_t<nt, int>::storage_t reduce;
    } shared;

    // The range this CTA works on. Corresponds to a0 a1 b0 b1 in mgpu1
    merge_range_t cta_rg = compute_merge_range(
        num_needles, num_haystack, cta, nv, mp_data[cta], mp_data[cta + 1]);
    // aCount2 bCount2 in mgpu1
    int cta_nr_needle = cta_rg.a_count(), cta_nr_haystack = cta_rg.b_count();
    if (cta_nr_needle == 0) return;

    // Equivalent to
    // !MatchB && !IndexB && MatchA && IndexA in mgpu1
    bool // left_a = false,
        left_b = bounds == bounds_upper && cta_rg.b_begin > 0,
        right_a = cta_rg.a_end < num_needles,
        right_b = cta_rg.b_end < num_haystack,
        extended = right_a && right_b && (bounds == bounds_lower || left_b);

    // Range of effective elements in shared memory including halos;
    // corresponds to aStart(always 0 here) aEnd bStart bEnd in mgpu1
    merge_range_t shmem_rg = {
      .a_begin = 0 /*left_a*/, .a_end = cta_nr_needle + right_a,
      .b_begin = cta_nr_needle + right_a + left_b,
      .b_end = cta_nr_needle + right_a + left_b + cta_nr_haystack + right_b
    };

    // DeviceLoad2ToShared<NT, VT, VT + 1>(a_global + a0 - leftA, aEnd, 
    // b_global + b0 - leftB, bEnd - aEnd, tid, keys_shared);
    array_t<type_t, vt + 1> x = load_two_streams_reg_halo<nt, vt, type_t>(
      needles + cta_rg.a_begin /*-left_a*/, shmem_rg.a_end /*-a_begin*/,
      haystack + cta_rg.b_begin - left_b, shmem_rg.b_end - shmem_rg.a_end, tid
    );
    reg_to_shared_strided<nt, vt + 1>(x, tid, shared.keys);

    int diag = vt * tid;
    int mp = merge_path<bounds>(shared.keys, cta_nr_needle,
                                shared.keys + shmem_rg.b_begin,
                                cta_nr_haystack, diag, comp);
    // a0tid aEnd b0tid+bStart bEnd
    merge_range_t thread_rg = shmem_rg.partition(mp, diag);

    array_t<type_t, vt> thrd_idx = extended ? serial_search_flg<vt, bounds, false>(
        shared.keys, thread_rg, cta_rg.b_begin - shmem_rg.b_begin, comp
    ) : serial_search_flg<vt, bounds, true>(
        shared.keys, thread_rg, cta_rg.b_begin - shmem_rg.b_begin, comp
    );

    iterate<vt>([&] (int i) {
      if (thrd_idx[i] != -1)
        shared.indices[mp++] = thrd_idx[i];
    }); // Register to shared
    __syncthreads();
    shared_to_mem<nt, vt>(shared.indices, tid, cta_nr_needle,
                          indices + cta_rg.a_begin, false);
  };

  cta_transform<launch_t>(k, num_needles + num_haystack, context);

  // __host__ Downsweep:
  // load and compute cta_rg.a_count()
  // calls dswp_dtl based on vt_need == cta_rg.a_count() / nt
  // easy: call dswp_dtl<nt, vt, 0>
  // __device__ dwsp_dtl<int nt, int vt, int vt0>:
  // mem_to_shared_strided
  // shared_to_reg_thread loads booleans (highest bit)
  // cta_scan_t<nt, bool>().scan<vt>(...)
  // reg_to_shared_thread is read-write using += :(
}

END_MGPU_NAMESPACE
