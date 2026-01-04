#include "rozenberg_a_bubble_odd_even_sort/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "rozenberg_a_bubble_odd_even_sort/common/include/common.hpp"

namespace rozenberg_a_bubble_odd_even_sort {

RozenbergABubbleOddEvenSortSEQ::RozenbergABubbleOddEvenSortSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  InType empty;
  GetInput().swap(empty);

  for (const auto &elem : in) {
    GetInput().push_back(elem);
  }

  GetOutput().clear();
}

bool RozenbergABubbleOddEvenSortSEQ::ValidationImpl() {
  return (!(GetInput().empty())) && (GetOutput().empty());
}

bool RozenbergABubbleOddEvenSortSEQ::PreProcessingImpl() {
  GetOutput().resize(GetInput().size());
  return GetOutput().size() == GetInput().size();
}

bool RozenbergABubbleOddEvenSortSEQ::RunImpl() {
  GetOutput() = GetInput();
  size_t n = GetOutput().size();

  for (size_t i = 0; i < n; i++) {
    for (size_t j = i % 2 == 0 ? 0 : 1; j < n - 1; j += 2) {
      if (GetOutput()[j] > GetOutput()[j + 1]) {
        std::swap(GetOutput()[j + 1], GetOutput()[j]);
      }
    }
  }

  return true;
}

bool RozenbergABubbleOddEvenSortSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace rozenberg_a_bubble_odd_even_sort
