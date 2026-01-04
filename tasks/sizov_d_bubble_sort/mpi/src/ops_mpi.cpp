#include "sizov_d_bubble_sort/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "sizov_d_bubble_sort/common/include/common.hpp"

namespace sizov_d_bubble_sort {

namespace {

struct ScatterPlan {
  std::vector<int> counts;
  std::vector<int> displs;
};

ScatterPlan MakeScatterPlan(int n, int comm_size) {
  ScatterPlan plan;
  plan.counts.assign(comm_size, 0);
  plan.displs.assign(comm_size, 0);

  const int base = n / comm_size;
  const int rem = n % comm_size;

  int offset = 0;
  for (int rank_idx = 0; rank_idx < comm_size; ++rank_idx) {
    plan.counts[rank_idx] = base + ((rank_idx < rem) ? 1 : 0);
    plan.displs[rank_idx] = offset;
    offset += plan.counts[rank_idx];
  }

  return plan;
}

inline void CompareSwap(int &a, int &b) {
  if (a > b) {
    std::swap(a, b);
  }
}

void LocalOddEvenPhase(std::vector<int> &local, int global_offset, int phase_parity) {
  const int local_n = static_cast<int>(local.size());
  if (local_n < 2) {
    return;
  }

  int idx = ((global_offset & 1) == phase_parity) ? 0 : 1;
  for (; idx + 1 < local_n; idx += 2) {
    CompareSwap(local[idx], local[idx + 1]);
  }
}

void ExchangeRightIfNeeded(std::vector<int> &local, const ScatterPlan &plan, int rank, int comm_size, int phase_parity,
                           int tag) {
  if (local.empty()) {
    return;
  }
  if (rank + 1 >= comm_size || plan.counts[rank + 1] == 0) {
    return;
  }

  const int last_global = plan.displs[rank] + static_cast<int>(local.size()) - 1;
  if ((last_global & 1) != phase_parity) {
    return;
  }

  int send_val = local.back();
  int recv_val = 0;

  MPI_Sendrecv(&send_val, 1, MPI_INT, rank + 1, tag, &recv_val, 1, MPI_INT, rank + 1, tag, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  local.back() = std::min(send_val, recv_val);
}

void ExchangeLeftIfNeeded(std::vector<int> &local, const ScatterPlan &plan, int rank, int phase_parity, int tag) {
  if (local.empty()) {
    return;
  }
  if (rank - 1 < 0 || plan.counts[rank - 1] == 0) {
    return;
  }

  const int first_global = plan.displs[rank];
  const int boundary_left_global = first_global - 1;
  if ((boundary_left_global & 1) != phase_parity) {
    return;
  }

  int send_val = local.front();
  int recv_val = 0;

  MPI_Sendrecv(&send_val, 1, MPI_INT, rank - 1, tag, &recv_val, 1, MPI_INT, rank - 1, tag, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  local.front() = std::max(send_val, recv_val);
}

}  // namespace

SizovDBubbleSortMPI::SizovDBubbleSortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool SizovDBubbleSortMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return !GetInput().empty();
  }
  return true;
}

bool SizovDBubbleSortMPI::PreProcessingImpl() {
  data_ = GetInput();
  return true;
}

bool SizovDBubbleSortMPI::RunImpl() {
  int rank = 0;
  int comm_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(data_.size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n <= 1) {
    if (rank != 0) {
      data_.assign(n, 0);
    }
    if (n == 1) {
      MPI_Bcast(data_.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    GetOutput() = data_;
    return true;
  }

  const ScatterPlan plan = MakeScatterPlan(n, comm_size);
  const int local_n = plan.counts[rank];

  std::vector<int> local(local_n);
  MPI_Scatterv(rank == 0 ? data_.data() : nullptr, plan.counts.data(), plan.displs.data(), MPI_INT, local.data(),
               local_n, MPI_INT, 0, MPI_COMM_WORLD);

  for (int phase = 0; phase < n; ++phase) {
    const int parity = phase & 1;
    const int tag = phase;

    LocalOddEvenPhase(local, plan.displs[rank], parity);
    ExchangeRightIfNeeded(local, plan, rank, comm_size, parity, tag);
    ExchangeLeftIfNeeded(local, plan, rank, parity, tag);
  }

  std::vector<int> result(n);
  MPI_Allgatherv(local.data(), local_n, MPI_INT, result.data(), plan.counts.data(), plan.displs.data(), MPI_INT,
                 MPI_COMM_WORLD);

  GetOutput() = std::move(result);
  return true;
}

bool SizovDBubbleSortMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sizov_d_bubble_sort
