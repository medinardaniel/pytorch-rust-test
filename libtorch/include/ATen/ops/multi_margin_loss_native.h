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
TORCH_API at::Tensor multi_margin_loss_cpu(const at::Tensor & self, const at::Tensor & target, const at::Scalar & p=1, const at::Scalar & margin=1, const ::std::optional<at::Tensor> & weight={}, int64_t reduction=at::Reduction::Mean);
TORCH_API at::Tensor & multi_margin_loss_cpu_out(const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const ::std::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & out);
TORCH_API at::Tensor multi_margin_loss_cuda(const at::Tensor & self, const at::Tensor & target, const at::Scalar & p=1, const at::Scalar & margin=1, const ::std::optional<at::Tensor> & weight={}, int64_t reduction=at::Reduction::Mean);
TORCH_API at::Tensor & multi_margin_loss_cuda_out(const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const ::std::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & out);
} // namespace native
} // namespace at