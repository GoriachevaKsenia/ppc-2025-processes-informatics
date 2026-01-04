#include "savva_d_min_elem_vec/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "savva_d_min_elem_vec/common/include/common.hpp"

namespace savva_d_min_elem_vec {

SavvaDMinElemVecSEQ::SavvaDMinElemVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SavvaDMinElemVecSEQ::ValidationImpl() {
  return (!GetInput().empty()) && (GetOutput() == 0);
}

bool SavvaDMinElemVecSEQ::PreProcessingImpl() {
  return true;
}

bool SavvaDMinElemVecSEQ::RunImpl() {
  if (GetInput().empty()) {
    return false;
  }

  int min_val = GetInput()[0];
  for (size_t i = 1; i < GetInput().size(); ++i) {
    min_val = std::min(GetInput()[i], min_val);
  }
  GetOutput() = min_val;

  return true;
}

bool SavvaDMinElemVecSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace savva_d_min_elem_vec
