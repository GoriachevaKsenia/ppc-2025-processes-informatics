#include "zavyalov_a_qsort_simple_merge/seq/include/ops_seq.hpp"

#include <cstdlib>
#include <vector>

#include "zavyalov_a_qsort_simple_merge/common/include/common.hpp"

namespace zavyalov_a_qsort_simple_merge {

ZavyalovAQsortSEQ::ZavyalovAQsortSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ZavyalovAQsortSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool ZavyalovAQsortSEQ::PreProcessingImpl() {
  return true;
}

bool ZavyalovAQsortSEQ::RunImpl() {
  std::vector<double> input_copy = GetInput();
  if (input_copy.data() != nullptr) {
    MyQsort(input_copy.data(), 0, static_cast<int>(input_copy.size()) - 1);
    if (!input_copy.empty()) {
      GetOutput() = input_copy;
    }
  }
  return true;
}

bool ZavyalovAQsortSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zavyalov_a_qsort_simple_merge
