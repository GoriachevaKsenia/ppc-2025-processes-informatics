#include "scalar_product/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "scalar_product/common/include/common.hpp"

namespace scalar_product {

ScalarProductSEQ::ScalarProductSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ScalarProductSEQ::ValidationImpl() {
  const auto &vector_a = GetInput().first;
  const auto &vector_b = GetInput().second;
  return vector_a.size() == vector_b.size() && !vector_a.empty();
}

bool ScalarProductSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool ScalarProductSEQ::RunImpl() {
  const auto &vector_a = GetInput().first;
  const auto &vector_b = GetInput().second;

  if (vector_a.empty() || vector_b.empty()) {
    return false;
  }

  int dot_product = 0;
  size_t size = vector_a.size();

  for (size_t i = 0; i < size; ++i) {
    dot_product += vector_a[i] * vector_b[i];
  }

  GetOutput() = dot_product;

  return true;
}

bool ScalarProductSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace scalar_product
