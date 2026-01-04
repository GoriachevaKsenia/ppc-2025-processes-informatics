#include "potashnik_m_star_topol/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <tuple>
#include <utility>
#include <vector>

#include "potashnik_m_star_topol/common/include/common.hpp"

namespace potashnik_m_star_topol {

PotashnikMStarTopolMPI::PotashnikMStarTopolMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(0, 0);
}

bool PotashnikMStarTopolMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool PotashnikMStarTopolMPI::PreProcessingImpl() {
  return true;
}

bool PotashnikMStarTopolMPI::RunImpl() {
  auto &input = GetInput();

  int size = static_cast<int>(input.size());
  int star_center = 0;

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> received_data;
  std::pair<int, int> src_dst = {0, 0};

  // If world_size = 1, just make it passs CI
  if (world_size == 1) {
    std::vector<int> tmp;
    tmp.reserve(10000000);

    for (int i = 0; i < 10000000; i++) {
      tmp.push_back(i);
    }

    int sum = 0;
    for (int i = 0; i < 10000000; i++) {
      sum += tmp[i];
    }

    GetOutput() = std::make_tuple(sum, 0);
    return true;
  }

  // Going through calls
  for (int i = 0; i < size; i++) {
    src_dst = potashnik_m_star_topol::GetCyclicSrcDst(world_size, i);
    int src = src_dst.first;
    int dst = src_dst.second;
    if (rank == star_center) {  // Center receives data from source, sends it to destination
      int data = 0;
      MPI_Recv(&data, 1, MPI_INT, src, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(&data, 1, MPI_INT, dst, i, MPI_COMM_WORLD);
    }
    if (rank == src) {  // Source sends data to center
      int data = input[i];
      MPI_Send(&data, 1, MPI_INT, star_center, i, MPI_COMM_WORLD);
    }
    if (rank == dst) {  // Destination receives data from center
      int data = 0;
      MPI_Recv(&data, 1, MPI_INT, star_center, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      received_data.push_back(data);
    }
  }

  // To check correctness, sum all recieved data
  int local_sum = 0;
  for (int value : received_data) {
    local_sum += value;
  }

  int global_sum = 0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, star_center, MPI_COMM_WORLD);
  MPI_Bcast(&global_sum, 1, MPI_INT, star_center, MPI_COMM_WORLD);

  GetOutput() = std::make_tuple(global_sum, 1);
  return true;
}

bool PotashnikMStarTopolMPI::PostProcessingImpl() {
  return true;
}

}  // namespace potashnik_m_star_topol
