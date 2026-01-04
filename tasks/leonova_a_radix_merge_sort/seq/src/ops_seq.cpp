#include "leonova_a_radix_merge_sort/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "leonova_a_radix_merge_sort/common/include/common.hpp"

namespace leonova_a_radix_merge_sort {

LeonovaARadixMergeSortSEQ::LeonovaARadixMergeSortSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>(GetInput().size());
}

bool LeonovaARadixMergeSortSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool LeonovaARadixMergeSortSEQ::PreProcessingImpl() {
  return true;
}

bool LeonovaARadixMergeSortSEQ::RunImpl() {
  if (!ValidationImpl()) {
    return false;
  }
  GetOutput() = GetInput();
  if (GetOutput().size() > 1) {
    RadixMergeSort(GetOutput(), 0, GetOutput().size());
  }
  return true;
}

bool LeonovaARadixMergeSortSEQ::PostProcessingImpl() {
  return true;
}

uint64_t LeonovaARadixMergeSortSEQ::TransformDoubleToKey(double value) {
  uint64_t int_value = 0;
  std::memcpy(&int_value, &value, sizeof(double));

  constexpr uint64_t kSignBitMask = 0x8000000000000000ULL;
  // Преобразование для корректной сортировки чисел с плавающей точкой
  if ((int_value & kSignBitMask) != 0U) {
    return ~int_value;
  }
  return int_value | kSignBitMask;
}

void LeonovaARadixMergeSortSEQ::RadixSort(std::vector<double> &arr, size_t left, size_t right) {
  size_t size = right - left;
  if (size <= 1) {
    return;
  }

  std::vector<uint64_t> keys(size);
  for (size_t index = 0; index < size; ++index) {
    keys[index] = TransformDoubleToKey(arr[left + index]);
  }

  std::vector<double> temp_arr(size);
  std::vector<uint64_t> temp_keys(size);

  for (int byte_pos = 0; byte_pos < kNumBytes; ++byte_pos) {
    std::vector<size_t> count(kNumCounters, 0);

    for (size_t index = 0; index < size; ++index) {
      uint8_t byte_val = (keys[index] >> (byte_pos * kByteSize)) & 0xFF;
      ++count[byte_val];
    }

    size_t total = 0;
    for (size_t &elem : count) {
      size_t old_count = elem;
      elem = total;
      total += old_count;
    }

    for (size_t index = 0; index < size; ++index) {
      uint8_t byte_val = (keys[index] >> (byte_pos * kByteSize)) & 0xFF;
      size_t pos = count[byte_val]++;
      temp_arr[pos] = arr[left + index];
      temp_keys[pos] = keys[index];
    }

    auto left_it = arr.begin() + static_cast<typename std::vector<double>::difference_type>(left);
    std::ranges::copy(temp_arr, left_it);
    keys.swap(temp_keys);
  }
}

void LeonovaARadixMergeSortSEQ::SimpleRadixMerge(std::vector<double> &arr, size_t left, size_t mid, size_t right) {
  size_t left_size = mid - left;
  size_t right_size = right - mid;

  std::vector<double> merged(left_size + right_size);

  size_t index = 0;
  size_t jndex = 0;
  size_t kndex = 0;

  uint64_t key_left = 0;
  uint64_t key_right = 0;
  if (left_size > 0) {
    key_left = TransformDoubleToKey(arr[left]);
  }
  if (right_size > 0) {
    key_right = TransformDoubleToKey(arr[mid]);
  }

  while (index < left_size && jndex < right_size) {
    if (key_left < key_right) {
      merged[kndex++] = arr[left + index++];
      if (index < left_size) {
        key_left = TransformDoubleToKey(arr[left + index]);
      }
    } else {
      merged[kndex++] = arr[mid + jndex++];
      if (jndex < right_size) {
        key_right = TransformDoubleToKey(arr[mid + jndex]);
      }
    }
  }

  while (index < left_size) {
    merged[kndex++] = arr[left + index++];
  }
  while (jndex < right_size) {
    merged[kndex++] = arr[mid + jndex++];
  }

  auto left_it = arr.begin() + static_cast<typename std::vector<double>::difference_type>(left);
  std::ranges::copy(merged, left_it);
}

void LeonovaARadixMergeSortSEQ::RadixMergeSort(std::vector<double> &arr, size_t left, size_t right) {
  struct SortTask {
    size_t left;
    size_t right;
    bool sorted;
  };

  std::vector<SortTask> stack;
  stack.reserve(128);
  stack.push_back({left, right, false});

  while (!stack.empty()) {
    SortTask current = stack.back();
    stack.pop_back();

    size_t size = current.right - current.left;

    if (size <= 1) {
      continue;
    }

    if (size <= kRadixThreshold) {
      RadixSort(arr, current.left, current.right);
      continue;
    }

    if (!current.sorted) {
      size_t mid = current.left + (size / 2);

      stack.push_back({current.left, current.right, true});
      stack.push_back({mid, current.right, false});
      stack.push_back({current.left, mid, false});
    } else {
      size_t mid = current.left + (size / 2);
      SimpleRadixMerge(arr, current.left, mid, current.right);
    }
  }
}

}  // namespace leonova_a_radix_merge_sort
