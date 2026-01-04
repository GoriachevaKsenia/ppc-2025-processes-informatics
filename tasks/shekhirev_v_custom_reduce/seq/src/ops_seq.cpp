#include "../include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "../../common/include/common.hpp"

namespace shekhirev_v_custom_reduce_seq {

CustomReduceSequential::CustomReduceSequential(const shekhirev_v_custom_reduce::InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool CustomReduceSequential::ValidationImpl() {
  return GetInput() >= 0;
}

bool CustomReduceSequential::PreProcessingImpl() {
  GetOutput() = GetInput() + 1;
  return true;
}

bool CustomReduceSequential::RunImpl() {
  const std::vector<int> vec(GetInput(), 1);
  const int sum = std::accumulate(vec.begin(), vec.end(), 0);
  GetOutput() = sum;
  return true;
}

bool CustomReduceSequential::PostProcessingImpl() {
  return true;
}

}  // namespace shekhirev_v_custom_reduce_seq
