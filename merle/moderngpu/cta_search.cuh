// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "cta_merge.cuh"
#include "meta.cuh"

BEGIN_MGPU_NAMESPACE

template <int vt, bounds_t bounds, bool range_check, bool retsize = false,
          typename type_t, typename comp_t>
MGPU_DEVICE array_t<int, vt + int(retsize)>
serial_search_flg(const type_t *keys_shared, merge_range_t range,
                  int b_offset, comp_t comp) {
  type_t a_key = keys_shared[range.a_begin];
  type_t b_key = keys_shared[range.b_begin];
  [[maybe_unused]] type_t b_prev;
  // b_keys start right after the end of the a_keys.
  const int b_start = range.a_end;
  if constexpr (bounds == bounds_t::bounds_upper)
    b_prev = (range.b_begin <= b_start) ? type_t() : keys_shared[range.b_begin - 1];

  /* register */ array_t<int, vt + int(retsize)> result;
  if constexpr (retsize)
    result[vt] = 0;

  iterate<vt>([&](int i) {
    // This is almost the same merge predicate as serial_merge, except for the
    // "range_check" template parameter
    bool p = merge_predicate<bounds, range_check>(a_key, b_key, range, comp);

    if(p) {
      int match = bounds_upper == bounds
          ? (!range_check || range.b_begin > b_start) && !comp(b_prev, a_key)
          : (!range_check || range.b_valid()) && !comp(a_key, b_key);
      if constexpr (retsize)
        result[vt] += match;
      match <<= 31;
      match |= b_offset + range.b_begin;
      result[i] = match;
      a_key = keys_shared[++range.a_begin];

    } else {
      if constexpr (bounds == bounds_t::bounds_upper)
        b_prev = b_key;
      b_key = keys_shared[++range.b_begin];
      result[i] = -1;
    }
  });

  __syncthreads();
  return result;
}

END_MGPU_NAMESPACE
