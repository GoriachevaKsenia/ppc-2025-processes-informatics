#include "maslova_u_fast_sort_simple/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

#include "maslova_u_fast_sort_simple/common/include/common.hpp"

namespace maslova_u_fast_sort_simple {

MaslovaUFastSortSimpleMPI::MaslovaUFastSortSimpleMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MaslovaUFastSortSimpleMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int flag = 0;
  if (rank == 0) {
    if (GetInput().size() > static_cast<size_t>(2147483647)) {
      flag = 1;
    }
  }
  MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return flag == 0;
}

bool MaslovaUFastSortSimpleMPI::PreProcessingImpl() {
  return true;
}

void MaslovaUFastSortSimpleMPI::TreeMerge(std::vector<int> &local_vec, int rank, int size) {
  for (int step = 1; step < size; step *= 2) {
    if ((rank % (2 * step)) == 0) {
      if ((rank + step) < size) {
        int recv_size = 0;
        MPI_Status status;
        MPI_Probe(rank + step, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &recv_size);
        std::vector<int> received(recv_size);
        MPI_Recv(received.data(), recv_size, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<int> merged;
        merged.reserve(local_vec.size() + received.size());
        std::ranges::merge(local_vec, received, std::back_inserter(merged));
        local_vec = std::move(merged);
      }
    } else {
      int target = rank - step;
      MPI_Send(local_vec.data(), static_cast<int>(local_vec.size()), MPI_INT, target, 0, MPI_COMM_WORLD);
      break;
    }
  }
}

bool MaslovaUFastSortSimpleMPI::RunImpl() {
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
    return true;
  }

  std::vector<int> send_counts(size);
  std::vector<int> displs(size);
  int part = total_size / size;
  int rem = total_size % size;
  for (int i = 0; i < size; ++i) {
    send_counts[i] = part + (i < rem ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + send_counts[i - 1];
  }

  std::vector<int> local_vec(send_counts[rank]);
  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, send_counts.data(), displs.data(), MPI_INT, local_vec.data(),
               send_counts[rank], MPI_INT, 0, MPI_COMM_WORLD);

  if (!local_vec.empty()) {
    QuickSort(local_vec.data(), 0, static_cast<int>(local_vec.size()) - 1);
  }

  TreeMerge(local_vec, rank, size);

  if (rank == 0) {
    GetOutput() = std::move(local_vec);
  }
  return true;
}

bool MaslovaUFastSortSimpleMPI::PostProcessingImpl() {
  return true;
}

}  // namespace maslova_u_fast_sort_simple
