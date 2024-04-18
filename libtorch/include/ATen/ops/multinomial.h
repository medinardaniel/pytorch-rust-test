#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/multinomial_ops.h>

namespace at {


// aten::multinomial.out(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & multinomial_out(at::Tensor & out, const at::Tensor & self, int64_t num_samples, bool replacement=false, ::std::optional<at::Generator> generator=::std::nullopt) {
    return at::_ops::multinomial_out::call(self, num_samples, replacement, generator, out);
}
// aten::multinomial.out(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & multinomial_outf(const at::Tensor & self, int64_t num_samples, bool replacement, ::std::optional<at::Generator> generator, at::Tensor & out) {
    return at::_ops::multinomial_out::call(self, num_samples, replacement, generator, out);
}

// aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor
inline at::Tensor multinomial(const at::Tensor & self, int64_t num_samples, bool replacement=false, ::std::optional<at::Generator> generator=::std::nullopt) {
    return at::_ops::multinomial::call(self, num_samples, replacement, generator);
}

}
