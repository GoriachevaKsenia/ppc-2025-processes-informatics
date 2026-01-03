#include "leonova_a_radix_merge_sort/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "leonova_a_radix_merge_sort/common/include/common.hpp"

namespace leonova_a_radix_merge_sort {

LeonovaARadixMergeSortMPI::LeonovaARadixMergeSortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();
}

bool LeonovaARadixMergeSortMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  int is_valid = 1;

  if (rank_ == 0) {
    if (GetInput().empty()) {
      is_valid = 0;
    }
  }

  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return is_valid == 1;
}

bool LeonovaARadixMergeSortMPI::PreProcessingImpl() {
  return true;
}

bool LeonovaARadixMergeSortMPI::RunImpl() {
  if (!ValidationImpl()) {
    return false;
  }

  const auto &full_input = GetInput();
  size_t total_size = full_input.size();

  std::vector<int> send_counts;
  std::vector<int> displacements;
  send_counts.reserve(world_size_);
  displacements.reserve(world_size_);

  send_counts.resize(world_size_, 0);
  displacements.resize(world_size_, 0);

  if (rank_ == 0) {
    auto base_size = static_cast<int>(total_size / world_size_);
    auto remainder = static_cast<int>(total_size % world_size_);

    for (int index = 0; index < world_size_; ++index) {
      send_counts[index] = base_size + (index < remainder ? 1 : 0);
      if (index > 0) {
        displacements[index] = displacements[index - 1] + send_counts[index - 1];
      }
    }
  }

  MPI_Bcast(send_counts.data(), world_size_, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displacements.data(), world_size_, MPI_INT, 0, MPI_COMM_WORLD);

  int local_count = send_counts[rank_];
  local_data_.resize(local_count);

  if (world_size_ == 1) {
    local_data_ = full_input;
  } else {
    MPI_Scatterv(full_input.data(), send_counts.data(), displacements.data(), MPI_DOUBLE, local_data_.data(),
                 local_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  if (!local_data_.empty()) {
    RadixMergeSort(local_data_, 0, local_data_.size());
  }

  HierarchicalMerge();

  BroadcastResult();
  return true;
}

bool LeonovaARadixMergeSortMPI::PostProcessingImpl() {
  return true;
}

inline uint64_t LeonovaARadixMergeSortMPI::TransformDoubleToKey(double value) {
  uint64_t int_value = 0;
  std::memcpy(&int_value, &value, sizeof(double));

  constexpr uint64_t kSignBitMask = 0x8000000000000000ULL;
  return ((int_value & kSignBitMask) != 0U) ? ~int_value : (int_value | kSignBitMask);
}

void LeonovaARadixMergeSortMPI::ComputeKeysForVector(const std::vector<double> &vec, std::vector<uint64_t> &keys) {
  size_t size = vec.size();
  keys.resize(size);

  size_t index = 0;
  for (; index + 3 < size; index += 4) {
    keys[index] = TransformDoubleToKey(vec[index]);
    keys[index + 1] = TransformDoubleToKey(vec[index + 1]);
    keys[index + 2] = TransformDoubleToKey(vec[index + 2]);
    keys[index + 3] = TransformDoubleToKey(vec[index + 3]);
  }
  for (; index < size; ++index) {
    keys[index] = TransformDoubleToKey(vec[index]);
  }
}

void LeonovaARadixMergeSortMPI::MergeWithKeys(std::vector<double> &result, const std::vector<double> &vec1,
                                              const std::vector<uint64_t> &keys1, const std::vector<double> &vec2,
                                              const std::vector<uint64_t> &keys2) {
  size_t total_size = vec1.size() + vec2.size();
  result.clear();
  result.reserve(total_size);

  auto index = static_cast<typename std::vector<double>::difference_type>(0);
  auto jndex = static_cast<typename std::vector<double>::difference_type>(0);

  while (static_cast<size_t>(index) < vec1.size() && static_cast<size_t>(jndex) < vec2.size()) {
    const uint64_t key1 = keys1[static_cast<size_t>(index)];
    const uint64_t key2 = keys2[static_cast<size_t>(jndex)];

    if (key1 < key2) {
      result.push_back(vec1[static_cast<size_t>(index++)]);
    } else {
      result.push_back(vec2[static_cast<size_t>(jndex++)]);
    }
  }

  if (static_cast<size_t>(index) < vec1.size()) {
    result.insert(result.end(), vec1.begin() + index, vec1.end());
  }
  if (static_cast<size_t>(jndex) < vec2.size()) {
    result.insert(result.end(), vec2.begin() + jndex, vec2.end());
  }
}

bool LeonovaARadixMergeSortMPI::ShouldMergeWithPartner(int step, int &partner_rank, bool &is_sender) const {
  int mask = step * 2;

  if ((rank_ & (mask - 1)) == 0) {  // rank_ % mask == 0
    partner_rank = rank_ + step;
    if (partner_rank >= world_size_) {
      partner_rank = -1;
    }
    return partner_rank != -1;
  }

  if ((rank_ & (mask - 1)) == step) {  // rank_ % mask == step
    partner_rank = rank_ - step;
    is_sender = true;
    return true;
  }

  return false;
}

void LeonovaARadixMergeSortMPI::SendLocalData(int partner_rank) {
  int data_size = static_cast<int>(local_data_.size());
  MPI_Send(&data_size, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);

  if (data_size > 0) {
    MPI_Send(local_data_.data(), data_size, MPI_DOUBLE, partner_rank, 0, MPI_COMM_WORLD);
  }

  local_data_.clear();
  local_keys_.clear();
  local_data_.shrink_to_fit();
  local_keys_.shrink_to_fit();
}

void LeonovaARadixMergeSortMPI::ReceiveAndMergeData(int partner_rank, std::vector<double> &partner_data_buffer,
                                                    std::vector<uint64_t> &partner_keys_buffer) {
  int partner_size = 0;
  MPI_Recv(&partner_size, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (partner_size <= 0) {
    return;
  }

  partner_data_buffer.resize(partner_size);
  MPI_Recv(partner_data_buffer.data(), partner_size, MPI_DOUBLE, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  partner_keys_buffer.resize(partner_size);
  for (int index = 0; index < partner_size; ++index) {
    partner_keys_buffer[index] = TransformDoubleToKey(partner_data_buffer[index]);
  }

  if (!local_data_.empty()) {
    std::vector<double> merged;
    merged.reserve(local_data_.size() + partner_size);

    MergeWithKeys(merged, local_data_, local_keys_, partner_data_buffer, partner_keys_buffer);

    local_data_.swap(merged);
    ComputeKeysForVector(local_data_, local_keys_);
  } else {
    local_data_.swap(partner_data_buffer);
    local_keys_.swap(partner_keys_buffer);
  }
}

void LeonovaARadixMergeSortMPI::HierarchicalMerge() {
  int step = 1;

  if (!local_data_.empty()) {
    ComputeKeysForVector(local_data_, local_keys_);
  }

  std::vector<double> partner_data_buffer;
  std::vector<uint64_t> partner_keys_buffer;
  partner_data_buffer.reserve(local_data_.capacity());
  partner_keys_buffer.reserve(local_keys_.capacity());

  while (step < world_size_) {
    int partner_rank = -1;
    bool is_sender = false;

    if (ShouldMergeWithPartner(step, partner_rank, is_sender)) {
      if (is_sender) {
        SendLocalData(partner_rank);
      } else {
        ReceiveAndMergeData(partner_rank, partner_data_buffer, partner_keys_buffer);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    step *= 2;
  }
}

void LeonovaARadixMergeSortMPI::BroadcastResult() {
  int result_size = 0;
  if (rank_ == 0) {
    result_size = static_cast<int>(local_data_.size());
  }

  MPI_Bcast(&result_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (result_size <= 0) {
    GetOutput().clear();
    return;
  }

  if (rank_ != 0) {
    GetOutput().resize(result_size);
  } else {
    GetOutput() = std::move(local_data_);
  }

  if (result_size > 0) {
    MPI_Bcast(GetOutput().data(), result_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
}

void LeonovaARadixMergeSortMPI::CountFrequencies(const std::vector<uint64_t> &keys, int shift,
                                                 std::array<size_t, 256> &count) {
  for (uint64_t key : keys) {
    auto byte_val = static_cast<uint8_t>((key >> shift) & 0xFF);
    auto byte_index = static_cast<size_t>(byte_val);
    auto *count_ptr = count.data() + byte_index;
    ++(*count_ptr);
  }
}

void LeonovaARadixMergeSortMPI::ComputePrefixSums(std::array<size_t, 256> &count) {
  size_t total = 0;
  for (size_t &elem : count) {
    size_t old_count = elem;
    elem = total;
    total += old_count;
  }
}

void LeonovaARadixMergeSortMPI::DistributeElements(const std::vector<double> &arr, const std::vector<uint64_t> &keys,
                                                   size_t left, int shift, const std::array<size_t, 256> &count,
                                                   std::vector<double> &temp_arr, std::vector<uint64_t> &temp_keys) {
  std::array<size_t, 256> local_count = count;

  for (size_t index = 0; index < keys.size(); ++index) {
    auto byte_val = static_cast<uint8_t>((keys[index] >> shift) & 0xFF);
    auto byte_index = static_cast<size_t>(byte_val);
    auto *local_count_ptr = local_count.data() + byte_index;
    size_t pos = (*local_count_ptr)++;
    temp_arr[pos] = arr[left + index];
    temp_keys[pos] = keys[index];
  }
}

void LeonovaARadixMergeSortMPI::RadixSort(std::vector<double> &arr, size_t left, size_t right) {
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
    const int shift = byte_pos * kByteSize;
    std::array<size_t, 256> count = {};

    CountFrequencies(keys, shift, count);
    ComputePrefixSums(count);
    DistributeElements(arr, keys, left, shift, count, temp_arr, temp_keys);

    auto left_it = arr.begin() + static_cast<typename std::vector<double>::difference_type>(left);
    std::ranges::copy(temp_arr, left_it);
    keys.swap(temp_keys);
  }
}

void LeonovaARadixMergeSortMPI::CopyRemainingElements(const std::vector<double> &arr, size_t src_offset,
                                                      size_t src_size, std::vector<double> &merged,
                                                      size_t dest_offset) {
  if (src_size > 0) {
    auto src_begin = arr.begin() + static_cast<typename std::vector<double>::difference_type>(src_offset);
    auto src_end = arr.begin() + static_cast<typename std::vector<double>::difference_type>(src_offset + src_size);
    auto dest_begin = merged.begin() + static_cast<typename std::vector<double>::difference_type>(dest_offset);
    std::copy(src_begin, src_end, dest_begin);
  }
}

void LeonovaARadixMergeSortMPI::MergeTwoParts(const std::vector<double> &arr, size_t left, size_t mid, size_t right,
                                              std::vector<double> &merged) {
  const size_t left_size = mid - left;
  const size_t right_size = right - mid;

  size_t index = 0;
  size_t jndex = 0;
  size_t kndex = 0;

  double left_val = NAN;
  double right_val = NAN;
  uint64_t left_key = 0;
  uint64_t right_key = 0;

  if (left_size > 0) {
    left_val = arr[left];
    left_key = TransformDoubleToKey(left_val);
  }
  if (right_size > 0) {
    right_val = arr[mid];
    right_key = TransformDoubleToKey(right_val);
  }

  while (index < left_size && jndex < right_size) {
    if (left_key < right_key) {
      merged[kndex++] = left_val;
      ++index;
      if (index < left_size) {
        left_val = arr[left + index];
        left_key = TransformDoubleToKey(left_val);
      }
    } else {
      merged[kndex++] = right_val;
      ++jndex;
      if (jndex < right_size) {
        right_val = arr[mid + jndex];
        right_key = TransformDoubleToKey(right_val);
      }
    }
  }

  CopyRemainingElements(arr, left + index, left_size - index, merged, kndex);
  kndex += left_size - index;
  CopyRemainingElements(arr, mid + jndex, right_size - jndex, merged, kndex);
}

void LeonovaARadixMergeSortMPI::RadixMergeSort(std::vector<double> &arr, size_t left, size_t right) {
  struct SortTask {
    size_t left;
    size_t right;
    bool sorted;
  };

  std::vector<SortTask> stack;
  stack.reserve(128);

  // Начинаем с исходной задачи
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
      size_t mid = current.left + (size >> 1);

      stack.push_back({current.left, current.right, true});
      stack.push_back({mid, current.right, false});
      stack.push_back({current.left, mid, false});
    } else {
      // Обе части отсортированы, нужно их слить
      size_t mid = current.left + (size >> 1);
      std::vector<double> merged(size);
      MergeTwoParts(arr, current.left, mid, current.right, merged);
      auto left_it = arr.begin() + static_cast<typename std::vector<double>::difference_type>(current.left);
      std::ranges::copy(merged, left_it);
    }
  }
}

}  // namespace leonova_a_radix_merge_sort
