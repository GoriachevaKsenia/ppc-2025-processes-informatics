#include "nikolaev_d_most_dif_vec_neighbors/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

#include "nikolaev_d_most_dif_vec_neighbors/common/include/common.hpp"

namespace nikolaev_d_most_dif_vec_neighbors {

NikolaevDMostDifVecNeighborsMPI::NikolaevDMostDifVecNeighborsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool NikolaevDMostDifVecNeighborsMPI::ValidationImpl() {
  return GetInput().size() >= 2;
}

bool NikolaevDMostDifVecNeighborsMPI::PreProcessingImpl() {
  return true;
}

bool NikolaevDMostDifVecNeighborsMPI::RunImpl() {
  const auto &input = GetInput();
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (input.size() < 2) {
    return false;
  }

  int n = static_cast<int>(input.size());
  int actual_processes = std::min(size, n);
  int elements_per_process = n / actual_processes;
  int remainder = n % actual_processes;

  std::vector<int> send_counts(size, 0);
  std::vector<int> displacements(size, 0);
  int current_displacement = 0;

  for (int i = 0; i < actual_processes; i++) {
    send_counts[i] = elements_per_process + (i < remainder ? 1 : 0);
    displacements[i] = current_displacement;
    current_displacement += send_counts[i];
  }

  int local_size = send_counts[rank];
  std::vector<int> local_data(local_size);

  MPI_Scatterv(input.data(), send_counts.data(), displacements.data(), MPI_INT, local_data.data(), local_size, MPI_INT,
               0, MPI_COMM_WORLD);

  FindLocalDiff(rank, size, actual_processes, local_data, local_size);

  return true;
}

void NikolaevDMostDifVecNeighborsMPI::FindLocalDiff(int rank, int size, int actual_processes,
                                                    std::vector<int> &local_data, int local_size) {
  int64_t local_max_diff = -1;
  std::pair<int, int> local_result = {0, 0};

  if (local_size >= 2) {
    for (int i = 0; i < local_size - 1; i++) {
      int64_t diff = std::llabs(static_cast<int64_t>(local_data[i + 1]) - static_cast<int64_t>(local_data[i]));
      if (diff > local_max_diff) {
        local_max_diff = diff;
        local_result = {local_data[i], local_data[i + 1]};
      }
    }
  }

  int first_element = local_data.empty() ? 0 : local_data[0];
  int last_element = local_data.empty() ? 0 : local_data[local_size - 1];

  LocalMaxInfo local_info;
  local_info.diff = local_max_diff;
  local_info.pair_first = local_result.first;
  local_info.pair_second = local_result.second;
  local_info.first_elem = first_element;
  local_info.last_elem = last_element;

  std::vector<LocalMaxInfo> all_info;
  if (rank == 0) {
    all_info.resize(size);
  }

  MPI_Gather(&local_info, sizeof(LocalMaxInfo), MPI_BYTE, all_info.data(), sizeof(LocalMaxInfo), MPI_BYTE, 0,
             MPI_COMM_WORLD);

  ProcessLocalData(rank, actual_processes, all_info);
}

void NikolaevDMostDifVecNeighborsMPI::ProcessLocalData(int rank, int actual_processes,
                                                       std::vector<LocalMaxInfo> &all_info) {
  std::pair<int, int> global_result;
  if (rank == 0) {
    int64_t global_max_diff = -1;

    // обработка локальных максимумов из каждого сегмента
    for (int i = 0; i < actual_processes; i++) {
      if (all_info[i].diff > global_max_diff) {
        global_max_diff = all_info[i].diff;
        global_result = {all_info[i].pair_first, all_info[i].pair_second};
      }
    }

    // обработка граничных пар между сегментами
    for (int i = 0; i < actual_processes - 1; i++) {
      int64_t diff =
          std::llabs(static_cast<int64_t>(all_info[i + 1].first_elem) - static_cast<int64_t>(all_info[i].last_elem));
      if (diff > global_max_diff) {
        global_max_diff = diff;
        global_result = {all_info[i].last_elem, all_info[i + 1].first_elem};
      }
    }
  }

  MPI_Bcast(&global_result, 2, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_result;
}

bool NikolaevDMostDifVecNeighborsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace nikolaev_d_most_dif_vec_neighbors
