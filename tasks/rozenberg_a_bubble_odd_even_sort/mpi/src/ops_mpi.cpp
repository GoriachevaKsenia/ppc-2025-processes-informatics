#include "rozenberg_a_bubble_odd_even_sort/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "rozenberg_a_bubble_odd_even_sort/common/include/common.hpp"

namespace rozenberg_a_bubble_odd_even_sort {

void RozenbergABubbleOddEvenSortMPI::LocalBubbleSort(InType &local_buf) {
  int chunk = static_cast<int>(local_buf.size());
  for (int i = 0; i < chunk; i++) {
    for (int j = 0; j < chunk - 1; j++) {
      if (local_buf[j] > local_buf[j + 1]) {
        std::swap(local_buf[j], local_buf[j + 1]);
      }
    }
  }
}

void RozenbergABubbleOddEvenSortMPI::ExchangeAndMerge(InType &local_buf, int neighbor, int chunk, int neighbor_n,
                                                      int rank) {
  std::vector<int> neighbor_data(static_cast<size_t>(neighbor_n));
  std::vector<int> merged(static_cast<size_t>(chunk + neighbor_n));

  MPI_Sendrecv(local_buf.data(), chunk, MPI_INT, neighbor, 0, neighbor_data.data(), neighbor_n, MPI_INT, neighbor, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::ranges::merge(local_buf, neighbor_data, merged.begin());

  if (rank < neighbor) {
    std::copy(merged.begin(), merged.begin() + chunk, local_buf.begin());
  } else {
    std::copy(merged.end() - chunk, merged.end(), local_buf.begin());
  }
}

RozenbergABubbleOddEvenSortMPI::RozenbergABubbleOddEvenSortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  InType empty;
  GetInput().swap(empty);

  for (const auto &elem : in) {
    GetInput().push_back(elem);
  }

  GetOutput().clear();
}

bool RozenbergABubbleOddEvenSortMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return (!(GetInput().empty())) && (GetOutput().empty());
  }
  return true;
}

bool RozenbergABubbleOddEvenSortMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetOutput().resize(GetInput().size());
  }
  return true;
}

bool RozenbergABubbleOddEvenSortMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);

  int sum = 0;
  for (int i = 0; i < size; i++) {
    sendcounts[i] = (n / size) + (i < (n % size) ? 1 : 0);
    displs[i] = sum;
    sum += sendcounts[i];
  }

  int chunk = sendcounts[rank];
  InType local_buf(static_cast<size_t>(chunk));
  MPI_Scatterv(GetInput().data(), sendcounts.data(), displs.data(), MPI_INT, local_buf.data(), chunk, MPI_INT, 0,
               MPI_COMM_WORLD);

  LocalBubbleSort(local_buf);

  int iterations = size + n;
  for (int i = 0; i < iterations; i++) {
    bool local_changed = false;
    bool is_even_step = (i % 2 == 0);
    bool is_even_rank = (rank % 2 == 0);
    int neighbor = (is_even_step == is_even_rank) ? rank + 1 : rank - 1;

    if (neighbor >= 0 && neighbor < size) {
      InType init_data = local_buf;

      int neighbor_n = sendcounts[neighbor];
      ExchangeAndMerge(local_buf, neighbor, chunk, neighbor_n, rank);

      if (init_data != local_buf) {
        local_changed = true;
      }
    }

    bool any_changed = false;
    MPI_Allreduce(&local_changed, &any_changed, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
    if (!any_changed) {
      break;
    }
  }

  MPI_Gatherv(local_buf.data(), chunk, MPI_INT, GetOutput().data(), sendcounts.data(), displs.data(), MPI_INT, 0,
              MPI_COMM_WORLD);

  return true;
}

bool RozenbergABubbleOddEvenSortMPI::PostProcessingImpl() {
  return true;
}

}  // namespace rozenberg_a_bubble_odd_even_sort
