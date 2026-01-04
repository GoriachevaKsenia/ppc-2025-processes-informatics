#include "lifanov_k_allreduce/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "lifanov_k_allreduce/common/include/common.hpp"

namespace lifanov_k_allreduce {

LifanovKAllReduceMPI::LifanovKAllReduceMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool LifanovKAllReduceMPI::ValidationImpl() {
  return (!GetInput().empty());
}

bool LifanovKAllReduceMPI::PreProcessingImpl() {
  GetOutput().resize(1);
  return true;
}

bool LifanovKAllReduceMPI::RunImpl() {
  int current_rank = 0;
  int num_processes = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  int input_size = 0;
  if (current_rank == 0) {
    input_size = static_cast<int>(GetInput().size());
  }

  MPI_Bcast(&input_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (input_size == 0) {
    GetOutput()[0] = 0;
    return true;
  }

  int base_chunk_size = input_size / num_processes;
  int extra_elements = input_size % num_processes;

  int local_chunk_size = base_chunk_size + (current_rank < extra_elements ? 1 : 0);
  local_input_.resize(local_chunk_size);

  std::vector<int> send_counts(num_processes, 0);
  std::vector<int> displacements(num_processes, 0);

  if (current_rank == 0) {
    int offset = 0;
    for (int i = 0; i < num_processes; ++i) {
      send_counts[i] = base_chunk_size + (i < extra_elements ? 1 : 0);
      displacements[i] = offset;
      offset += send_counts[i];
    }
  }

  MPI_Scatterv(current_rank == 0 ? GetInput().data() : nullptr, send_counts.data(), displacements.data(), MPI_INT,
               local_input_.data(), local_chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

  int local_sum = 0;
  for (int i = 0; i < local_chunk_size; ++i) {
    local_sum += local_input_[i];
  }

  int total_sum = 0;

  if (current_rank == 0) {
    total_sum = local_sum;
    for (int i = 1; i < num_processes; ++i) {
      int received_sum = 0;
      MPI_Recv(&received_sum, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      total_sum += received_sum;
    }
  } else {
    MPI_Send(&local_sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  MPI_Bcast(&total_sum, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput()[0] = total_sum;

  return true;
}

bool LifanovKAllReduceMPI::PostProcessingImpl() {
  return true;
}

}  // namespace lifanov_k_allreduce
