#include "dolov_v_qsort_batcher/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <stack>
#include <utility>
#include <vector>

#include "dolov_v_qsort_batcher/common/include/common.hpp"

namespace dolov_v_qsort_batcher {

DolovVQsortBatcherMPI::DolovVQsortBatcherMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool DolovVQsortBatcherMPI::ValidationImpl() {
  return true;
}

bool DolovVQsortBatcherMPI::PreProcessingImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (world_rank == 0) {
    total_count_ = static_cast<int>(GetInput().size());
  }

  MPI_Bcast(&total_count_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_count_ < 0) {
    return false;
  }

  part_sizes_.assign(world_size, 0);
  part_offsets_.assign(world_size, 0);

  int base_size = total_count_ / world_size;
  int extra = total_count_ % world_size;
  int current_offset = 0;

  for (int i = 0; i < world_size; ++i) {
    part_sizes_[i] = base_size + (i < extra ? 1 : 0);
    part_offsets_[i] = current_offset;
    current_offset += part_sizes_[i];
  }

  local_buffer_.resize(part_sizes_[world_rank]);
  return true;
}

bool DolovVQsortBatcherMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (total_count_ <= 0) {
    return true;
  }

  DistributeData(world_rank, world_size);

  if (!local_buffer_.empty()) {
    FastSort(local_buffer_.data(), 0, static_cast<int>(local_buffer_.size()) - 1);
  }

  ExecuteBatcherParallel(world_rank, world_size);

  CollectData(world_rank, world_size);

  return true;
}

void DolovVQsortBatcherMPI::DistributeData(int world_rank, int /*world_size*/) {
  const double *send_ptr = (world_rank == 0) ? GetInput().data() : nullptr;
  MPI_Scatterv(send_ptr, part_sizes_.data(), part_offsets_.data(), MPI_DOUBLE, local_buffer_.data(),
               part_sizes_[world_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void DolovVQsortBatcherMPI::BatcherStep(int p_step, int k, int j, int i, int world_rank, int world_size) {
  int p1 = i + j;
  int p2 = i + j + k;

  if ((p1 / (p_step * 2)) == (p2 / (p_step * 2))) {
    if (world_rank == p1 || world_rank == p2) {
      int partner = (world_rank == p1) ? p2 : p1;
      if (partner < world_size) {
        std::vector<double> neighbor_data(part_sizes_[partner]);

        MPI_Sendrecv(local_buffer_.data(), static_cast<int>(local_buffer_.size()), MPI_DOUBLE, partner, 0,
                     neighbor_data.data(), part_sizes_[partner], MPI_DOUBLE, partner, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

        std::vector<double> merged;
        MergeSequences(local_buffer_, neighbor_data, merged, world_rank == p1);
        local_buffer_ = std::move(merged);
      }
    }
  }
}

void DolovVQsortBatcherMPI::ExecuteBatcherParallel(int world_rank, int world_size) {
  if (world_size <= 1 || total_count_ == 0) {
    return;
  }
  for (int p_step = 1; p_step < world_size; p_step <<= 1) {
    for (int k = p_step; k >= 1; k >>= 1) {
      for (int j = k % p_step; j <= world_size - 1 - k; j += (k << 1)) {
        for (int i = 0; i < k; ++i) {
          BatcherStep(p_step, k, j, i, world_rank, world_size);
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
}

void DolovVQsortBatcherMPI::MergeSequences(const std::vector<double> &first, const std::vector<double> &second,
                                           std::vector<double> &result, bool take_low) {
  if (first.empty() && second.empty()) {
    return;
  }

  size_t n1 = first.size();
  size_t n2 = second.size();
  result.resize(n1);
  std::vector<double> combined(n1 + n2);
  size_t i = 0;
  size_t j = 0;
  size_t k = 0;

  while (i < n1 && j < n2) {
    if (first[i] <= second[j]) {
      combined[k++] = first[i++];
    } else {
      combined[k++] = second[j++];
    }
  }
  while (i < n1) {
    combined[k++] = first[i++];
  }
  while (j < n2) {
    combined[k++] = second[j++];
  }

  auto diff_n1 = static_cast<std::ptrdiff_t>(n1);
  if (take_low) {
    std::copy(combined.begin(), combined.begin() + diff_n1, result.begin());
  } else {
    std::copy(combined.end() - diff_n1, combined.end(), result.begin());
  }
}

void DolovVQsortBatcherMPI::CollectData(int world_rank, int /*world_size*/) {
  std::vector<double> global_array;
  if (world_rank == 0) {
    global_array.resize(total_count_);
  }

  MPI_Gatherv(local_buffer_.data(), static_cast<int>(local_buffer_.size()), MPI_DOUBLE, global_array.data(),
              part_sizes_.data(), part_offsets_.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    GetOutput() = std::move(global_array);
  }
}

int DolovVQsortBatcherMPI::GetSplitIndex(double *data, int low, int high) {
  double pivot = data[low + ((high - low) / 2)];
  int i = low - 1;
  int j = high + 1;
  while (true) {
    i++;
    while (data[i] < pivot) {
      i++;
    }
    j--;
    while (data[j] > pivot) {
      j--;
    }
    if (i >= j) {
      return j;
    }
    std::swap(data[i], data[j]);
  }
}

void DolovVQsortBatcherMPI::FastSort(double *data, int low, int high) {
  std::stack<std::pair<int, int>> s;
  s.emplace(low, high);
  while (!s.empty()) {
    std::pair<int, int> range = s.top();
    s.pop();
    if (range.first < range.second) {
      int split_point = GetSplitIndex(data, range.first, range.second);
      s.emplace(range.first, split_point);
      s.emplace(split_point + 1, range.second);
    }
  }
}

bool DolovVQsortBatcherMPI::PostProcessingImpl() {
  return true;
}

}  // namespace dolov_v_qsort_batcher
