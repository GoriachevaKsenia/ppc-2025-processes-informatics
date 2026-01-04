#include "goriacheva_k_reduce/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "goriacheva_k_reduce/common/include/common.hpp"

namespace goriacheva_k_reduce {

GoriachevaKReduceMPI::GoriachevaKReduceMPI(const InType &in) {
  InType tmp(in);
  GetInput().swap(tmp);
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool GoriachevaKReduceMPI::ValidationImpl() {
  return true;
}

bool GoriachevaKReduceMPI::PreProcessingImpl() {
  GetOutput().resize(1);
  return true;
}

bool GoriachevaKReduceMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<int>(GetInput().size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_size == 0) {
    GetOutput()[0] = 0;
    return true;
  }

  int base = total_size / size;
  int rem = total_size % size;

  int local_size = base + (rank < rem ? 1 : 0);
  local_input_.resize(local_size);

  std::vector<int> sendcounts;
  std::vector<int> displs;

  if (rank == 0) {
    sendcounts.resize(size);
    displs.resize(size);
    int offset = 0;
    for (int i = 0; i < size; ++i) {
      sendcounts[i] = base + (i < rem ? 1 : 0);
      displs[i] = offset;
      offset += sendcounts[i];
    }
  }

  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, sendcounts.data(), displs.data(), MPI_INT, local_input_.data(),
               local_size, MPI_INT, 0, MPI_COMM_WORLD);

  int local_sum = std::accumulate(local_input_.begin(), local_input_.end(), 0);
  int global_sum = 0;

  if (rank == 0) {
    global_sum = local_sum;
    for (int i = 1; i < size; ++i) {
      int tmp = 0;
      MPI_Recv(&tmp, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      global_sum += tmp;
    }
  } else {
    MPI_Send(&local_sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  MPI_Bcast(&global_sum, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput()[0] = global_sum;

  return true;
}

bool GoriachevaKReduceMPI::PostProcessingImpl() {
  return true;
}

}  // namespace goriacheva_k_reduce
