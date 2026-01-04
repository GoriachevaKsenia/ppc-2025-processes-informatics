#include "yurkin_counting_number/seq/include/ops_seq.hpp"

#include <cctype>
#include <cstddef>

#include "yurkin_counting_number/common/include/common.hpp"

namespace yurkin_counting_number {

YurkinCountingNumberSEQ::YurkinCountingNumberSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YurkinCountingNumberSEQ::ValidationImpl() {
  return GetOutput() == 0;
}

bool YurkinCountingNumberSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool YurkinCountingNumberSEQ::RunImpl() {
  const InType &input = GetInput();
  std::size_t total_size = input.size();

  int local_count = 0;
  for (std::size_t i = 0; i < total_size; ++i) {
    if (std::isalpha(static_cast<unsigned char>(input[i])) != 0) {
      ++local_count;
    }
  }

  GetOutput() = local_count;
  return true;
}

bool YurkinCountingNumberSEQ::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace yurkin_counting_number
