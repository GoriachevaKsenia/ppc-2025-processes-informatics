#include "shkrebko_m_shell_sort_batcher_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "shkrebko_m_shell_sort_batcher_merge/common/include/mpi_utils.hpp"
#include "shkrebko_m_shell_sort_batcher_merge/common/include/utils.hpp"

namespace shkrebko_m_shell_sort_batcher_merge {

bool ShkrebkoMShellSortBatcherMergeMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  return true;
}

bool ShkrebkoMShellSortBatcherMergeMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  int global_size = 0;
  if (world_rank_ == 0) {
    global_size = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  counts_.assign(world_size_, 0);
  displs_.assign(world_size_, 0);

  const int base = (world_size_ > 0) ? (global_size / world_size_) : 0;
  const int rem = (world_size_ > 0) ? (global_size % world_size_) : 0;

  for (int rank_index = 0; rank_index < world_size_; ++rank_index) {
    counts_[rank_index] = base + ((rank_index < rem) ? 1 : 0);
  }
  for (int rank_index = 1; rank_index < world_size_; ++rank_index) {
    displs_[rank_index] = displs_[rank_index - 1] + counts_[rank_index - 1];
  }

  local_.assign(static_cast<std::size_t>(counts_[world_rank_]), 0);

  int *send_buf = nullptr;
  if (world_rank_ == 0 && global_size > 0) {
    send_buf = GetInput().data();
  }

  MPI_Scatterv(send_buf, counts_.data(), displs_.data(), MPI_INT, local_.data(), counts_[world_rank_], MPI_INT, 0,
               MPI_COMM_WORLD);

  GetOutput().clear();
  return true;
}

bool ShkrebkoMShellSortBatcherMergeMPI::RunImpl() {
  ShellSort(&local_);

  for (int step = 1; step < world_size_; step <<= 1) {
    if ((world_rank_ % (2 * step)) == 0) {
      const int partner = world_rank_ + step;
      if (partner < world_size_) {
        std::vector<int> other = RecvVector(partner, 1000 + step, MPI_COMM_WORLD);
        local_ = BatcherOddEvenMerge(local_, other);
      }
    } else {
      const int partner = world_rank_ - step;
      SendVector(partner, 1000 + step, local_, MPI_COMM_WORLD);
      break;
    }
  }
  return true;
}

bool ShkrebkoMShellSortBatcherMergeMPI::PostProcessingImpl() {
  int out_size = 0;
  if (world_rank_ == 0) {
    GetOutput() = local_;
    out_size = static_cast<int>(GetOutput().size());
  }

  MPI_Bcast(&out_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (world_rank_ != 0) {
    GetOutput().assign(static_cast<std::size_t>(out_size), 0);
  }

  if (out_size > 0) {
    MPI_Bcast(GetOutput().data(), out_size, MPI_INT, 0, MPI_COMM_WORLD);
  }
  return true;
}

}  // namespace shkrebko_m_shell_sort_batcher_merge
