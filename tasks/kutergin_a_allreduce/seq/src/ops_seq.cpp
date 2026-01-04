#include "../include/ops_seq.hpp"

#include <numeric>

#include "../../common/include/common.hpp"

namespace kutergin_a_allreduce {

AllreduceSequential::AllreduceSequential(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool AllreduceSequential::ValidationImpl() {
  return true;
}

bool AllreduceSequential::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool AllreduceSequential::RunImpl() {
  const InType &input_data = GetInput();

  if (input_data.elements.empty()) {
    GetOutput() = 0;
  } else {
    GetOutput() = std::accumulate(input_data.elements.begin(), input_data.elements.end(), 0);
  }
  return true;
}

bool AllreduceSequential::PostProcessingImpl() {
  return true;
}

}  // namespace kutergin_a_allreduce
