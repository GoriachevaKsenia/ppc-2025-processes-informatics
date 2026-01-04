#include "dolov_v_qsort_batcher/seq/include/ops_seq.hpp"

#include <algorithm>
#include <stack>
#include <utility>
#include <vector>

#include "dolov_v_qsort_batcher/common/include/common.hpp"

namespace dolov_v_qsort_batcher {

DolovVQsortBatcherSEQ::DolovVQsortBatcherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool DolovVQsortBatcherSEQ::ValidationImpl() {
  return true;
}

bool DolovVQsortBatcherSEQ::PreProcessingImpl() {
  res_array_ = GetInput();
  return true;
}

bool DolovVQsortBatcherSEQ::RunImpl() {
  if (res_array_.empty()) {
    return true;
  }
  ApplyQuicksort(res_array_.data(), 0, static_cast<int>(res_array_.size()) - 1);
  return true;
}

bool DolovVQsortBatcherSEQ::PostProcessingImpl() {
  GetOutput() = std::move(res_array_);
  return true;
}

int DolovVQsortBatcherSEQ::GetHoarePartition(double *array, int low, int high) {
  double pivot = array[low + ((high - low) / 2)];
  int i = low - 1;
  int j = high + 1;

  while (true) {
    i++;
    while (array[i] < pivot) {
      i++;
    }
    j--;
    while (array[j] > pivot) {
      j--;
    }
    if (i >= j) {
      return j;
    }
    std::swap(array[i], array[j]);
  }
}

void DolovVQsortBatcherSEQ::ApplyQuicksort(double *array, int low, int high) {
  std::stack<std::pair<int, int>> stack;
  stack.emplace(low, high);

  while (!stack.empty()) {
    std::pair<int, int> range = stack.top();
    stack.pop();

    if (range.first < range.second) {
      int p = GetHoarePartition(array, range.first, range.second);
      if (p - range.first < range.second - p) {
        stack.emplace(p + 1, range.second);
        stack.emplace(range.first, p);
      } else {
        stack.emplace(range.first, p);
        stack.emplace(p + 1, range.second);
      }
    }
  }
}

}  // namespace dolov_v_qsort_batcher
