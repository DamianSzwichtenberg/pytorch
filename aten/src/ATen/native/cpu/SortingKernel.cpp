#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/Sorting.h>

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <limits>
#include <utility>

#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/NumericUtils.h>
#include <ATen/TensorIterator.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <ATen/native/CompositeRandomAccessor.h>
#include <ATen/native/TopKImpl.h>
#include <ATen/native/cpu/radix_sort.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/util/irange.h>

namespace at { namespace native {

namespace {

template <typename func_t>
void _dim_apply(
    const TensorBase &values,
    const TensorBase &indices,
    int64_t dim,
    const std::string& method_name,
    const func_t& f) {
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .declare_static_shape(values.sizes(), /*squash_dims=*/dim)
    .add_output(values)
    .add_output(indices)
    .build();

  auto values_dim_stride = values.stride(dim);
  auto indices_dim_stride = indices.stride(dim);
  auto dim_size = values.size(dim);

  AT_DISPATCH_ALL_TYPES_AND3(
    ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, iter.dtype(),
    "sorting_kernel_method_name", [&] {
      auto loop = [&](char** data, const int64_t* strides, int64_t n) {
        auto* values_data_bytes = data[0];
        auto* indices_data_bytes = data[1];

        if(values_data_bytes==nullptr || indices_data_bytes==nullptr){
          return;
        }

        for (const auto i C10_UNUSED : c10::irange(n)) {
          f(
            reinterpret_cast<scalar_t*>(values_data_bytes),
            values_dim_stride,
            reinterpret_cast<int64_t*>(indices_data_bytes),
            indices_dim_stride,
            dim_size
          );

          values_data_bytes += strides[0];
          indices_data_bytes += strides[1];
        }
      };

      int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, dim_size);
      iter.for_each(loop, /*grain_size=*/grain_size);
    }
  );
}

template <typename scalar_t>
struct KeyValueCompAsc {
  template <typename LHS, typename RHS>
  constexpr bool operator()(LHS lhs, RHS rhs) const {
    return (!_isnan<scalar_t>(get<0>(lhs)) && _isnan<scalar_t>(get<0>(rhs)))
      || (get<0>(lhs) < get<0>(rhs));
  }
};

template <typename scalar_t>
struct KeyValueCompDesc {
  template <typename LHS, typename RHS>
  constexpr bool operator()(LHS lhs, RHS rhs) const {
    return (_isnan<scalar_t>(get<0>(lhs)) && !_isnan<scalar_t>(get<0>(rhs)))
      || (get<0>(lhs) > get<0>(rhs));
  }
};

template <typename scalar_t>
std::pair<scalar_t, scalar_t> get_min_max(const scalar_t* data, int64_t size) {
  using Vec = vec::Vectorized<vec::vec_scalar_t<scalar_t>>;
  int num_threads = at::get_num_threads();
  std::vector<scalar_t> min_per_thread(num_threads, std::numeric_limits<scalar_t>::max());
  std::vector<scalar_t> max_per_thread(num_threads, std::numeric_limits<scalar_t>::lowest());

  at::parallel_for(0, size, at::internal::GRAIN_SIZE / num_threads, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    const scalar_t* local_data = data + begin;
    const int64_t local_size = end - begin;
    const auto local_min_max = at::vec::reduce2_all(
            [](Vec x, Vec y) { return at::vec::minimum(x, y); },
            [](Vec x, Vec y) { return at::vec::maximum(x, y); },
            local_data,
            local_size);
    min_per_thread[tid] = local_min_max.first;
    max_per_thread[tid] = local_min_max.second;
  });

  scalar_t global_min = min_per_thread[0];
  scalar_t global_max = max_per_thread[0];
  for (const auto i : c10::irange(1, num_threads)) {
    if (min_per_thread[i] < global_min) {
      global_min = min_per_thread[i];
    }
    if (max_per_thread[i] > global_max) {
      global_max = max_per_thread[i];
    }
  }

  return std::pair<scalar_t, scalar_t>(global_min, global_max);
}

static void parallel_sort1d_kernel(
    const TensorBase& values,
    const TensorBase& indices,
    bool descending) {
  // this kernel does not care about `stable` parameter as radix sort
  // used here is a stable sorting algorithm
  AT_DISPATCH_INTEGRAL_TYPES(values.scalar_type(), "parallel_sort1d_kernel", [&] {
    scalar_t min = std::numeric_limits<scalar_t>::max();
    scalar_t max = std::numeric_limits<scalar_t>::lowest();
    std::tie(min, max) =
        get_min_max(values.data_ptr<scalar_t>(), values.size(0));

    int64_t elements = values.numel();
    scalar_t* keys = values.data_ptr<scalar_t>();
    int64_t* vals = indices.data_ptr<int64_t>();
    std::vector<scalar_t> tmp_keys(elements);
    std::vector<int64_t> tmp_vals(elements);
    scalar_t* sorted_keys = nullptr;
    int64_t* sorted_vals = nullptr;
    std::tie(sorted_keys, sorted_vals) = radix_sort_parallel(
        keys,
        vals,
        tmp_keys.data(),
        tmp_vals.data(),
        elements,
        std::max(std::abs(min), std::abs(max)));

    const bool sorted_in_place = keys == sorted_keys;
    if (!sorted_in_place) {
      std::memcpy(keys, sorted_keys, values.numel() * values.itemsize());
      std::memcpy(vals, sorted_vals, indices.numel() * indices.itemsize());
    }
    if (min < 0) {
      // values like [2, 4, 1, -3, 8, -9, -5, 5] will be sorted as follows
      // (because of how negative values are stored): [1, 2, 4, 5, 8, -9, -5, -3]
      int64_t neg_val_offset = 0;
      for (const auto i : c10::irange(0, elements)) {
        if (keys[i] < 0) {
          neg_val_offset = i;
          break;
        }
      }
      // use tmp_keys and tmp_vals as a temporary buffer
      std::memcpy(tmp_keys.data(), keys, neg_val_offset * values.itemsize());
      std::memcpy(keys, keys + neg_val_offset, (elements - neg_val_offset) * values.itemsize());
      std::memcpy(keys + elements - neg_val_offset, tmp_keys.data(), neg_val_offset * values.itemsize());
      std::memcpy(tmp_vals.data(), vals, neg_val_offset * indices.itemsize());
      std::memcpy(vals, vals + neg_val_offset, (elements - neg_val_offset) * indices.itemsize());
      std::memcpy(vals + elements - neg_val_offset, tmp_vals.data(), neg_val_offset * indices.itemsize());
    }
    if (descending) {
      int num_threads = at::get_num_threads();
      at::parallel_for(0, elements / 2, at::internal::GRAIN_SIZE / num_threads, [&](int64_t begin, int64_t end) {
        int64_t end_pos = elements - begin - 1;
        for (const auto i : c10::irange(begin, end)) {
          std::swap(keys[i], keys[end_pos]);
          std::swap(vals[i], vals[end_pos]);
          --end_pos;
        }
      });
    }
  });
}

static void sort_kernel(
    const TensorBase& self,
    const TensorBase& values,
    const TensorBase& indices,
    int64_t dim,
    bool descending,
    bool stable) {
  dim = maybe_wrap_dim(dim, values.dim());
  _fill_indices(indices, dim);
  if (values.dim() == 1 && values.numel() >= at::internal::GRAIN_SIZE &&
      at::isIntegralType(values.scalar_type(), /*includeBool=*/false) &&
      is_radix_sort_available()) {
    parallel_sort1d_kernel(values, indices, descending);
    return;
  }
  _dim_apply(
    values, indices, dim,
    "sort_cpu", [&](
      auto* values, int64_t values_dim_stride,
      auto* indices, int64_t indices_dim_stride,
      int64_t dim_size
    ) {
      using scalar_t = typename std::remove_pointer<decltype(values)>::type;
      auto values_accessor = StridedRandomAccessor<scalar_t>(
        values, values_dim_stride);
      auto indices_accessor = StridedRandomAccessor<int64_t>(
        indices, indices_dim_stride);
      auto composite_accessor = CompositeRandomAccessorCPU<
        decltype(values_accessor), decltype(indices_accessor)
      >(values_accessor, indices_accessor);

      if (descending) {
        if (stable) {
          std::stable_sort(composite_accessor, composite_accessor + dim_size,
            KeyValueCompDesc<scalar_t>());
        }
        else {
          std::sort(composite_accessor, composite_accessor + dim_size,
            KeyValueCompDesc<scalar_t>());
        }
      }
      else {
        if (stable) {
          std::stable_sort(composite_accessor, composite_accessor + dim_size,
            KeyValueCompAsc<scalar_t>());
        }
        else {
          std::sort(composite_accessor, composite_accessor + dim_size,
            KeyValueCompAsc<scalar_t>());
        }
      }
    }
  );
}

static void topk_kernel(
    const TensorBase &values,
    const TensorBase &indices,
    const TensorBase &self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  auto sizes = self.sizes();
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .declare_static_shape(sizes, /*squash_dims=*/dim)
    .add_output(values)
    .add_output(indices)
    .add_input(self)
    .build();

  auto mode_values_stride = values.strides()[dim];
  auto mode_indices_stride = indices.strides()[dim];
  auto tmp_values_stride = self.strides()[dim];

  AT_DISPATCH_ALL_TYPES_AND(ScalarType::BFloat16, self.scalar_type(), "topk_cpu", [&] {
    auto loop = [&](char** data, const int64_t* strides, int64_t n) {
      if (self.scalar_type() == ScalarType::BFloat16) {
        return topk_impl_loop<scalar_t, float>(
            mode_values_stride, mode_indices_stride, tmp_values_stride,
            k, sizes[dim], largest, sorted, data, strides, n);
      } else {
        return topk_impl_loop<scalar_t, scalar_t>(
            mode_values_stride, mode_indices_stride, tmp_values_stride,
            k, sizes[dim], largest, sorted, data, strides, n);
      }
    };

    int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, sizes[dim]);
    iter.for_each(loop, /*grain_size=*/grain_size);
  });
}

} // anonymous namespace

REGISTER_DISPATCH(sort_stub, &sort_kernel);
REGISTER_DISPATCH(topk_stub, &topk_kernel);

}} //at::native
