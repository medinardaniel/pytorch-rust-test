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


namespace at {
namespace native {
TORCH_API at::Tensor var(const at::Tensor & self, bool unbiased=true);
TORCH_API at::Tensor var(const at::Tensor & self, at::OptionalIntArrayRef dim, bool unbiased=true, bool keepdim=false);
TORCH_API at::Tensor & var_out(const at::Tensor & self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim, at::Tensor & out);
TORCH_API at::Tensor var(const at::Tensor & self, at::OptionalIntArrayRef dim=::std::nullopt, const ::std::optional<at::Scalar> & correction=::std::nullopt, bool keepdim=false);
TORCH_API at::Tensor & var_out(const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim, at::Tensor & out);
TORCH_API at::Tensor var(const at::Tensor & self, at::DimnameList dim, bool unbiased=true, bool keepdim=false);
TORCH_API at::Tensor & var_out(const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim, at::Tensor & out);
TORCH_API at::Tensor var(const at::Tensor & self, at::DimnameList dim, const ::std::optional<at::Scalar> & correction=::std::nullopt, bool keepdim=false);
TORCH_API at::Tensor & var_out(const at::Tensor & self, at::DimnameList dim, const ::std::optional<at::Scalar> & correction, bool keepdim, at::Tensor & out);
} // namespace native
} // namespace at
