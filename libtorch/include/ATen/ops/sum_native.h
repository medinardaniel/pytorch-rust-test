#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>
#include <ATen/ops/sum_meta.h>

namespace at {
namespace native {
TORCH_API at::Tensor sum(const at::Tensor & self, ::std::optional<at::ScalarType> dtype=::std::nullopt);
TORCH_API at::Tensor & sum_out(const at::Tensor & self, ::std::optional<at::ScalarType> dtype, at::Tensor & out);
TORCH_API at::Tensor sum_coo(const at::Tensor & self, ::std::optional<at::ScalarType> dtype=::std::nullopt);
TORCH_API at::Tensor sum_csr(const at::Tensor & self, ::std::optional<at::ScalarType> dtype=::std::nullopt);
struct TORCH_API structured_sum_out : public at::meta::structured_sum_dim_IntList {
void impl(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype, const at::Tensor & out);
};
TORCH_API at::Tensor NestedTensor_sum_dim_CPU(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim=false, ::std::optional<at::ScalarType> dtype=::std::nullopt);
TORCH_API at::Tensor sum_sparse_coo(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim=false, ::std::optional<at::ScalarType> dtype=::std::nullopt);
TORCH_API at::Tensor sum_sparse_compressed(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim=false, ::std::optional<at::ScalarType> dtype=::std::nullopt);
TORCH_API at::Tensor sum(const at::Tensor & self, at::DimnameList dim, bool keepdim=false, ::std::optional<at::ScalarType> dtype=::std::nullopt);
TORCH_API at::Tensor & sum_out(const at::Tensor & self, at::DimnameList dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out);
} // namespace native
} // namespace at
