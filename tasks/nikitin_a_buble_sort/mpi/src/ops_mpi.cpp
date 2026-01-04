#include "nikitin_a_buble_sort/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "nikitin_a_buble_sort/common/include/common.hpp"

namespace nikitin_a_buble_sort {

NikitinABubleSortMPI::NikitinABubleSortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool NikitinABubleSortMPI::ValidationImpl() {
  return true;
}

bool NikitinABubleSortMPI::PreProcessingImpl() {
  data_ = GetInput();
  return true;
}

bool NikitinABubleSortMPI::RunImpl() {
  int rank = 0;
  int comm_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if (rank == 0) {
    n_ = static_cast<int>(data_.size());
  }
  MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n_ <= 1) {
    std::vector<double> result;
    if (n_ == 1) {
      result.resize(1);
      if (rank == 0) {
        result[0] = data_[0];
      }
      MPI_Bcast(result.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    GetOutput() = std::move(result);
    return true;
  }

  const int base = n_ / comm_size;
  const int rem = n_ % comm_size;
  const int count = base + (rank < rem ? 1 : 0);
  const int offset = (rank * base) + std::min(rank, rem);

  std::vector<int> counts(comm_size);
  std::vector<int> displs(comm_size);
  int current_offset = 0;
  for (int i = 0; i < comm_size; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = current_offset;
    current_offset += counts[i];
  }

  std::vector<double> local(count);
  MPI_Scatterv(rank == 0 ? data_.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, local.data(), count,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  for (int phase = 0; phase < n_; ++phase) {
    const int parity = phase & 1;
    const int tag = phase;

    LocalSort(local, offset, parity);
    ExchangeRight(local, counts, displs, rank, comm_size, parity, tag);
    ExchangeLeft(local, counts, displs, rank, parity, tag);
  }

  std::vector<double> result(n_);
  MPI_Allgatherv(local.data(), count, MPI_DOUBLE, result.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 MPI_COMM_WORLD);

  GetOutput() = std::move(result);
  return true;
}

bool NikitinABubleSortMPI::PostProcessingImpl() {
  return true;
}

void NikitinABubleSortMPI::LocalSort(std::vector<double> &local, int global_offset, int phase_parity) {
  const int local_n = static_cast<int>(local.size());
  if (local_n < 2) {
    return;
  }

  int start = ((global_offset & 1) == phase_parity) ? 0 : 1;
  for (int i = start; i + 1 < local_n; i += 2) {
    if (local[i] > local[i + 1]) {
      std::swap(local[i], local[i + 1]);
    }
  }
}

void NikitinABubleSortMPI::ExchangeRight(std::vector<double> &local, const std::vector<int> &counts,
                                         const std::vector<int> &displs, int rank, int comm_size, int phase_parity,
                                         int tag) {
  if (local.empty()) {
    return;
  }

  if (rank + 1 >= comm_size || counts[rank + 1] == 0) {
    return;
  }

  const int last_global = displs[rank] + static_cast<int>(local.size()) - 1;
  if ((last_global & 1) != phase_parity) {
    return;
  }

  double send_val = local.back();
  double recv_val = 0;

  MPI_Sendrecv(&send_val, 1, MPI_DOUBLE, rank + 1, tag, &recv_val, 1, MPI_DOUBLE, rank + 1, tag, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  local.back() = std::min(send_val, recv_val);
}

void NikitinABubleSortMPI::ExchangeLeft(std::vector<double> &local, const std::vector<int> &counts,
                                        const std::vector<int> &displs, int rank, int phase_parity, int tag) {
  if (local.empty()) {
    return;
  }
  if (rank - 1 < 0 || counts[rank - 1] == 0) {
    return;
  }

  const int first_global = displs[rank];
  const int boundary_left_global = first_global - 1;
  if ((boundary_left_global & 1) != phase_parity) {
    return;
  }

  double send_val = local.front();
  double recv_val = 0;

  MPI_Sendrecv(&send_val, 1, MPI_DOUBLE, rank - 1, tag, &recv_val, 1, MPI_DOUBLE, rank - 1, tag, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  local.front() = std::max(send_val, recv_val);
}

}  // namespace nikitin_a_buble_sort
