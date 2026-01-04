#include "lifanov_k_allreduce/seq/include/ops_seq.hpp"

#include <vector>

#include "lifanov_k_allreduce/common/include/common.hpp"

namespace lifanov_k_allreduce {

LifanovKAllreduceSEQ::LifanovKAllreduceSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool LifanovKAllreduceSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool LifanovKAllreduceSEQ::PreProcessingImpl() {
  return true;
}

bool LifanovKAllreduceSEQ::RunImpl() {
  std::vector<int> current_values = GetInput();
  GetOutput().resize(1);
  int global_sum = 0;
  for (int current_value : current_values) {
    global_sum += current_value;
  }
  GetOutput()[0] = global_sum;

  return true;
}

bool LifanovKAllreduceSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace lifanov_k_allreduce
