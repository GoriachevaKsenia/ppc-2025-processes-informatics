#include "goriacheva_k_violation_order_elem_vec/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "goriacheva_k_violation_order_elem_vec/common/include/common.hpp"

namespace goriacheva_k_violation_order_elem_vec {

GoriachevaKViolationOrderElemVecSEQ::GoriachevaKViolationOrderElemVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GoriachevaKViolationOrderElemVecSEQ::ValidationImpl() {
  return true;
}

bool GoriachevaKViolationOrderElemVecSEQ::PreProcessingImpl() {
  input_vec_ = GetInput();
  result_ = 0;
  return true;
}

bool GoriachevaKViolationOrderElemVecSEQ::RunImpl() {
  input_vec_ = GetInput();
  result_ = 0;

  if (input_vec_.size() <= 1) {
    result_ = 0;
  }

  for (size_t i = 0; i + 1 < input_vec_.size(); ++i) {
    if (input_vec_[i] > input_vec_[i + 1]) {
      ++result_;
    }
  }

  return true;
}

bool GoriachevaKViolationOrderElemVecSEQ::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace goriacheva_k_violation_order_elem_vec
