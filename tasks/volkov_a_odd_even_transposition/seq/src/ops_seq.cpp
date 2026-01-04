#include "volkov_a_odd_even_transposition/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "volkov_a_odd_even_transposition/common/include/common.hpp"

namespace volkov_a_odd_even_transposition {

OddEvenSortSeq::OddEvenSortSeq(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool OddEvenSortSeq::ValidationImpl() {
  return GetOutput().empty();
}

bool OddEvenSortSeq::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

bool OddEvenSortSeq::RunImpl() {
  auto &arr = GetOutput();
  size_t n = arr.size();

  if (n < 2) {
    return true;
  }

  bool is_sorted = false;

  while (!is_sorted) {
    is_sorted = true;

    // Нечетная фаза
    for (size_t i = 1; i < n - 1; i += 2) {
      if (arr[i] > arr[i + 1]) {
        std::swap(arr[i], arr[i + 1]);
        is_sorted = false;
      }
    }

    // Четная фаза
    for (size_t i = 0; i < n - 1; i += 2) {
      if (arr[i] > arr[i + 1]) {
        std::swap(arr[i], arr[i + 1]);
        is_sorted = false;
      }
    }
  }

  return true;
}

bool OddEvenSortSeq::PostProcessingImpl() {
  return true;
}

}  // namespace volkov_a_odd_even_transposition
