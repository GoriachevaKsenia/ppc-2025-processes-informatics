#include "nikolaev_d_most_dif_vec_neighbors/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

#include "nikolaev_d_most_dif_vec_neighbors/common/include/common.hpp"

namespace nikolaev_d_most_dif_vec_neighbors {

NikolaevDMostDifVecNeighborsSEQ::NikolaevDMostDifVecNeighborsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool NikolaevDMostDifVecNeighborsSEQ::ValidationImpl() {
  return GetInput().size() >= 2;
}

bool NikolaevDMostDifVecNeighborsSEQ::PreProcessingImpl() {
  return true;
}

bool NikolaevDMostDifVecNeighborsSEQ::RunImpl() {
  if (GetInput().size() < 2) {
    return false;
  }

  const auto &vec = GetInput();
  std::pair<int, int> result_elements;
  int64_t max_diff = -1;

  for (std::vector<int>::size_type i = 0; i < GetInput().size() - 1; i++) {
    int64_t diff = std::llabs(static_cast<int64_t>(vec[i + 1]) - static_cast<int64_t>(vec[i]));
    if (diff > max_diff) {
      max_diff = diff;
      result_elements = {vec[i], vec[i + 1]};
    }
  }

  GetOutput() = result_elements;
  return true;
}

bool NikolaevDMostDifVecNeighborsSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace nikolaev_d_most_dif_vec_neighbors
