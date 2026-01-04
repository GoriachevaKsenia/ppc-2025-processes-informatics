#include "goriacheva_k_violation_order_elem_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "goriacheva_k_violation_order_elem_vec/common/include/common.hpp"

namespace goriacheva_k_violation_order_elem_vec {

GoriachevaKViolationOrderElemVecMPI::GoriachevaKViolationOrderElemVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GoriachevaKViolationOrderElemVecMPI::ValidationImpl() {
  return true;
}

bool GoriachevaKViolationOrderElemVecMPI::PreProcessingImpl() {
  input_vec_ = GetInput();
  result_ = 0;
  return true;
}

void GoriachevaKViolationOrderElemVecMPI::ScatterInput(int rank, int size, int n, std::vector<int> &local) const {
  const int base = n / size;
  const int rem = n % size;
  const int local_size = static_cast<int>(local.size());

  if (rank == 0) {
    if (local_size > 0) {
      std::copy(input_vec_.begin(), input_vec_.begin() + local_size, local.begin());
    }

    for (int rank_iter = 1; rank_iter < size; ++rank_iter) {
      const int sz = base + (rank_iter < rem ? 1 : 0);
      if (sz > 0) {
        const int offset = (rank_iter * base) + std::min(rank_iter, rem);
        MPI_Send(input_vec_.data() + offset, sz, MPI_INT, rank_iter, 0, MPI_COMM_WORLD);
      }
    }
  } else if (local_size > 0) {
    MPI_Recv(local.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

int GoriachevaKViolationOrderElemVecMPI::CountLocalViolations(const std::vector<int> &local) {
  int count = 0;
  for (std::size_t i = 0; i + 1 < local.size(); ++i) {
    if (local[i] > local[i + 1]) {
      ++count;
    }
  }
  return count;
}

int GoriachevaKViolationOrderElemVecMPI::CheckBoundaryViolation(int rank, int size, const std::vector<int> &local) {
  int send_val = 0;
  int left_last = 0;

  if (!local.empty()) {
    send_val = local.back();
  }

  MPI_Sendrecv(&send_val, 1, MPI_INT, (rank + 1 < size) ? rank + 1 : MPI_PROC_NULL, 1, &left_last, 1, MPI_INT,
               (rank > 0) ? rank - 1 : MPI_PROC_NULL, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (rank > 0 && !local.empty() && left_last > local.front()) {
    return 1;
  }

  return 0;
}

bool GoriachevaKViolationOrderElemVecMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = static_cast<int>(input_vec_.size());
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n <= 1) {
    result_ = 0;
    MPI_Bcast(&result_, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return true;
  }

  const int base = n / size;
  const int rem = n % size;
  const int local_size = base + (rank < rem ? 1 : 0);

  std::vector<int> local(local_size);
  ScatterInput(rank, size, n, local);

  int local_count = CountLocalViolations(local);
  local_count += CheckBoundaryViolation(rank, size, local);

  MPI_Reduce(&local_count, &result_, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&result_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool GoriachevaKViolationOrderElemVecMPI::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace goriacheva_k_violation_order_elem_vec
