#include "kurpiakov_a_shellsort/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "kurpiakov_a_shellsort/common/include/common.hpp"

namespace kurpiakov_a_shellsort {

KurpiakovAShellsortMPI::KurpiakovAShellsortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool KurpiakovAShellsortMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    int input_size = std::get<0>(GetInput());
    int vector_size = static_cast<int>(std::get<1>(GetInput()).size());
    return input_size == vector_size && input_size >= 0;
  }
  return true;
}

bool KurpiakovAShellsortMPI::PreProcessingImpl() {
  world_size_ = 0;
  rank_ = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

  if (rank_ == 0) {
    data_ = std::get<1>(GetInput());
  }

  int n = 0;
  if (rank_ == 0) {
    n = static_cast<int>(data_.size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

void KurpiakovAShellsortMPI::ShellSortLocal(OutType &arr) {
  int n = static_cast<int>(arr.size());
  for (int gap = n / 2; gap > 0; gap /= 2) {
    for (int i = gap; i < n; ++i) {
      int temp = arr[i];
      int j = i;
      while (j >= gap && arr[j - gap] > temp) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = temp;
    }
  }
}

void KurpiakovAShellsortMPI::MergeSortedArrays(const OutType &send_counts, const OutType &gathered_data,
                                               const OutType &displs, const int &n) {
  data_.clear();
  data_.reserve(n);

  std::vector<int> indices(world_size_, 0);

  for (int k = 0; k < n; k++) {
    int min_val = 0;
    int min_part = -1;

    for (int i = 0; i < world_size_; i++) {
      if (indices[i] < send_counts[i]) {
        int val = gathered_data[displs[i] + indices[i]];
        if (min_part == -1 || val < min_val) {
          min_val = val;
          min_part = i;
        }
      }
    }

    data_.push_back(min_val);
    indices[min_part]++;
  }
}

bool KurpiakovAShellsortMPI::RunImpl() {
  int n = 0;
  if (rank_ == 0) {
    n = static_cast<int>(data_.size());
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n == 0) {
    data_.clear();
    return true;
  }

  int base_size = n / world_size_;
  int remainder = n % world_size_;

  std::vector<int> send_counts(world_size_);
  std::vector<int> displs(world_size_);

  int offset = 0;
  for (int i = 0; i < world_size_; i++) {
    send_counts[i] = base_size + (i < remainder ? 1 : 0);
    displs[i] = offset;
    offset += send_counts[i];
  }

  int local_size = send_counts[rank_];
  std::vector<int> local_data(local_size);

  MPI_Scatterv(rank_ == 0 ? data_.data() : nullptr, send_counts.data(), displs.data(), MPI_INT, local_data.data(),
               local_size, MPI_INT, 0, MPI_COMM_WORLD);

  ShellSortLocal(local_data);

  std::vector<int> gathered_data(n);

  MPI_Gatherv(local_data.data(), local_size, MPI_INT, gathered_data.data(), send_counts.data(), displs.data(), MPI_INT,
              0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    MergeSortedArrays(send_counts, gathered_data, displs, n);
  }

  if (rank_ != 0) {
    data_.resize(n);
  }
  MPI_Bcast(data_.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool KurpiakovAShellsortMPI::PostProcessingImpl() {
  GetOutput() = data_;
  return true;
}

}  // namespace kurpiakov_a_shellsort
