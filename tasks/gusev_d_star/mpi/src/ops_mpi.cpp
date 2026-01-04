#include "gusev_d_star/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>

#include "gusev_d_star/common/include/common.hpp"

namespace gusev_d_star {

constexpr int kTagTask = 0;
constexpr int kTagResult = 1;
constexpr int kHubRank = 0;

GusevDStarMPI::GusevDStarMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GusevDStarMPI::ValidationImpl() {
  return (GetInput() > 0);
}

bool GusevDStarMPI::PreProcessingImpl() {
  GetOutput() = 2 * GetInput();
  return true;
}

bool GusevDStarMPI::RunImpl() {
  auto input = GetInput();
  int rank = 0;
  int size = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size == 1) {
    volatile int local_res = 0;
    for (int i = 0; i < input; i++) {
      for (int j = 0; j < input; j++) {
        for (int k = 0; k < input; k++) {
          local_res += (i + j + k);
          local_res -= (i + j + k);
        }
      }
    }
    GetOutput() += local_res;
    return true;
  }

  if (rank == kHubRank) {
    RunAsMaster(size, input);
  } else {
    RunAsWorker(input);
  }

  return true;
}

bool GusevDStarMPI::PostProcessingImpl() {
  GetOutput() -= GetInput();
  return true;
}

void GusevDStarMPI::RunAsMaster(int size, int input) {
  int workers_count = size - 1;
  int chunk = input / workers_count;
  int remainder = input % workers_count;
  int current_start_index = 0;

  for (int dst_rank = 1; dst_rank < size; dst_rank++) {
    int count = chunk + (dst_rank <= remainder ? 1 : 0);

    std::array<int, 2> task_data = {current_start_index, count};

    MPI_Send(task_data.data(), 2, MPI_INT, dst_rank, kTagTask, MPI_COMM_WORLD);
    current_start_index += count;
  }

  for (int src_rank = 1; src_rank < size; src_rank++) {
    int worker_result = 0;
    MPI_Status status;

    MPI_Recv(&worker_result, 1, MPI_INT, src_rank, kTagResult, MPI_COMM_WORLD, &status);
    GetOutput() += worker_result;
  }
}

void GusevDStarMPI::RunAsWorker(int input) {
  std::array<int, 2> task_data = {};
  MPI_Status status;

  MPI_Recv(task_data.data(), 2, MPI_INT, kHubRank, kTagTask, MPI_COMM_WORLD, &status);

  int start_i = task_data[0];
  int count_i = task_data[1];
  int end_i = start_i + count_i;

  volatile int local_res = 0;

  for (int i = start_i; i < end_i; i++) {
    for (int j = 0; j < input; j++) {
      for (int k = 0; k < input; k++) {
        local_res += (i + j + k);
        local_res -= (i + j + k);
      }
    }
  }

  int res_to_send = local_res;

  MPI_Send(static_cast<const void *>(&res_to_send), 1, MPI_INT, kHubRank, kTagResult, MPI_COMM_WORLD);
}

}  // namespace gusev_d_star
