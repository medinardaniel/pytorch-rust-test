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
TORCH_API at::Tensor & _efficientzerotensor_out_symint(c10::SymIntArrayRef size, at::Tensor & out);
TORCH_API at::Tensor _efficientzerotensor(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype={}, ::std::optional<at::Layout> layout={}, ::std::optional<at::Device> device={}, ::std::optional<bool> pin_memory={});
TORCH_API at::Tensor _efficientzerotensor_cuda(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype={}, ::std::optional<at::Layout> layout={}, ::std::optional<at::Device> device={}, ::std::optional<bool> pin_memory={});
TORCH_API at::Tensor _efficientzerotensor_meta_symint(c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype={}, ::std::optional<at::Layout> layout={}, ::std::optional<at::Device> device={}, ::std::optional<bool> pin_memory={});
} // namespace native
} // namespace at
