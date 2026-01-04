#include "eremin_v_hypercube/seq/include/ops_seq.hpp"

#include <cmath>
#include <tuple>

#include "eremin_v_hypercube/common/include/common.hpp"

namespace eremin_v_hypercube {

EreminVHypercubeSEQ::EreminVHypercubeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool EreminVHypercubeSEQ::ValidationImpl() {
  auto &input = GetInput();
  return (std::get<0>(input) < std::get<1>(input)) && (std::get<2>(input) > 0) && (std::get<2>(input) <= 100000000) &&
         (std::get<0>(input) >= -1e9) && (std::get<0>(input) <= 1e9) && (std::get<1>(input) >= -1e9) &&
         (std::get<1>(input) <= 1e9) && (GetOutput() == 0);
}

bool EreminVHypercubeSEQ::PreProcessingImpl() {
  return true;
}

bool EreminVHypercubeSEQ::RunImpl() {
  auto &input = GetInput();
  const double lower_bound = std::get<0>(input);
  const double upper_bound = std::get<1>(input);
  const int steps = std::get<2>(input);
  const auto &in_function = std::get<3>(input);
  double step_size = (upper_bound - lower_bound) / steps;
  double result = 0.0;

  for (int i = 0; i < steps; i++) {
    result += in_function(lower_bound + ((static_cast<double>(i) + 0.5) * step_size));
  }

  result *= step_size;
  GetOutput() = result;
  return true;
}

bool EreminVHypercubeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace eremin_v_hypercube
