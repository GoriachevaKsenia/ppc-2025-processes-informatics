#include "../include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "../../common/include/common.hpp"

namespace shekhirev_v_custom_reduce_mpi {

CustomReduceMPI::CustomReduceMPI(const shekhirev_v_custom_reduce::InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool CustomReduceMPI::ValidationImpl() {
  return GetInput() > 0;
}

bool CustomReduceMPI::PreProcessingImpl() {
  shekhirev_v_custom_reduce::InType input_val = 0;
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    input_val = GetInput();
  }

  MPI_Bcast(&input_val, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetInput() = input_val;
  GetOutput() = GetInput() + 1;
  return true;
}

bool CustomReduceMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int count_per_process = GetInput() / size;
  const int remainder = GetInput() % size;
  const int local_count = count_per_process + (rank < remainder ? 1 : 0);

  const std::vector<int> local_vec(local_count, 1);
  const int local_sum = std::accumulate(local_vec.begin(), local_vec.end(), 0);

  if (rank == 0) {
    int global_sum = local_sum;

    for (int i = 1; i < size; ++i) {
      int recv_val = 0;
      MPI_Status status;
      MPI_Recv(&recv_val, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
      global_sum += recv_val;
    }

    GetOutput() = global_sum;
  } else {
    MPI_Send(&local_sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool CustomReduceMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shekhirev_v_custom_reduce_mpi
