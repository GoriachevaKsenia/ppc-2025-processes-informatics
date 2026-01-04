#include "votincev_d_qsort_batcher/seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

#include "votincev_d_qsort_batcher/common/include/common.hpp"

namespace votincev_d_qsort_batcher {

VotincevDQsortBatcherSEQ::VotincevDQsortBatcherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

// входные данные — просто std::vector<double>, проверяем, что не пустой
bool VotincevDQsortBatcherSEQ::ValidationImpl() {
  const auto &vec = GetInput();
  return !vec.empty();
}

bool VotincevDQsortBatcherSEQ::PreProcessingImpl() {
  return true;
}

bool VotincevDQsortBatcherSEQ::RunImpl() {
  std::vector<double> data = GetInput();

  if (!data.empty()) {
    QuickSort(data.data(), 0, static_cast<int>(data.size()) - 1);
  }

  GetOutput() = data;
  return true;
}

// partition (для qsort)
int VotincevDQsortBatcherSEQ::Partition(double *arr, int l, int h) {
  int i = l;
  int j = h;
  double pivot = arr[(l + h) / 2];

  while (i <= j) {
    while (arr[i] < pivot) {
      i++;
    }
    while (arr[j] > pivot) {
      j--;
    }

    if (i <= j) {
      std::swap(arr[i], arr[j]);
      i++;
      j--;
    }
  }
  // i — это граница следующего правого подмассива
  return i;
}

// итеративная qsort
void VotincevDQsortBatcherSEQ::QuickSort(double *arr, int left, int right) {
  std::vector<int> stack;

  stack.push_back(left);
  stack.push_back(right);

  while (!stack.empty()) {
    int h = stack.back();
    stack.pop_back();
    int l = stack.back();
    stack.pop_back();

    if (l >= h) {
      continue;
    }

    // вызываю Partition для разделения массива
    int p = Partition(arr, l, h);
    // p - это i после Partition. j находится на p-1 или p-2.

    // p - начало правого подмассива (i)
    // j - конец левого подмассива (j после Partition)

    // пересчитываю l и h для стека, используя внутренние границы Partition
    int l_end = p - 1;  // конец левого подмассива
    int r_start = p;    // начало правого подмассива

    // если левый подмассив существует
    if (l < l_end) {
      stack.push_back(l);
      stack.push_back(l_end);
    }

    // если правый подмассив существует
    if (r_start < h) {
      stack.push_back(r_start);
      stack.push_back(h);
    }
  }
}

bool VotincevDQsortBatcherSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_qsort_batcher
