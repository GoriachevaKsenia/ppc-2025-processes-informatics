#include "kurpiakov_a_shellsort/seq/include/ops_seq.hpp"

#include <vector>

#include "kurpiakov_a_shellsort/common/include/common.hpp"

namespace kurpiakov_a_shellsort {

KurpiakovAShellsortSEQ::KurpiakovAShellsortSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool KurpiakovAShellsortSEQ::ValidationImpl() {
  int input_size = std::get<0>(GetInput());
  auto vector_size = static_cast<int>(std::get<1>(GetInput()).size());
  return input_size == vector_size && input_size >= 0;
}

bool KurpiakovAShellsortSEQ::PreProcessingImpl() {
  data_ = std::get<1>(GetInput());

  int n = static_cast<int>(data_.size());
  gaps_.clear();
  for (int gap = n / 2; gap > 0; gap /= 2) {
    gaps_.push_back(gap);
  }

  return true;
}

bool KurpiakovAShellsortSEQ::RunImpl() {
  int n = static_cast<int>(data_.size());

  for (int gap : gaps_) {
    for (int i = gap; i < n; i++) {
      int temp = data_[i];
      int j = i;
      while (j >= gap && data_[j - gap] > temp) {
        data_[j] = data_[j - gap];
        j -= gap;
      }
      data_[j] = temp;
    }
  }

  return true;
}

bool KurpiakovAShellsortSEQ::PostProcessingImpl() {
  GetOutput() = data_;
  return true;
}

}  // namespace kurpiakov_a_shellsort
