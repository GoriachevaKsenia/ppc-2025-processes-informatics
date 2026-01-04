#include "volkov_a_odd_even_transposition/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "volkov_a_odd_even_transposition/common/include/common.hpp"

namespace volkov_a_odd_even_transposition {

OddEvenSortMPI::OddEvenSortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool OddEvenSortMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return GetOutput().empty();
  }
  return true;
}

bool OddEvenSortMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetOutput().resize(GetInput().size());
  }
  return true;
}

void OddEvenSortMPI::CalculateDistribution(int n, int size, std::vector<int> &counts, std::vector<int> &displs) {
  int rem = n % size;
  int base = n / size;
  int offset = 0;

  for (int i = 0; i < size; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = offset;
    offset += counts[i];
  }
}

void OddEvenSortMPI::PerformCompareSplit(InType &local_data, int partner_rank, int my_rank) {
  int my_count = static_cast<int>(local_data.size());
  int partner_count = 0;

  MPI_Sendrecv(&my_count, 1, MPI_INT, partner_rank, 0, &partner_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  InType partner_data(partner_count);

  MPI_Sendrecv(local_data.data(), my_count, MPI_INT, partner_rank, 1, partner_data.data(), partner_count, MPI_INT,
               partner_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  InType merged(my_count + partner_count);

  std::ranges::merge(local_data, partner_data, merged.begin());

  if (my_rank < partner_rank) {
    local_data.assign(merged.begin(), merged.begin() + my_count);
  } else {
    local_data.assign(merged.end() - my_count, merged.end());
  }
}

int OddEvenSortMPI::GetNeighbor(int phase, int rank, int size) {
  int partner = -1;
  if (phase % 2 == 0) {
    if (rank % 2 == 0) {
      partner = rank + 1;
    } else {
      partner = rank - 1;
    }
  } else {
    if (rank % 2 != 0) {
      partner = rank + 1;
    } else {
      partner = rank - 1;
    }
  }

  if (partner < 0 || partner >= size) {
    return -1;
  }
  return partner;
}

bool OddEvenSortMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  CalculateDistribution(n, size, counts, displs);

  int local_n = counts[rank];
  InType local_vec(local_n);

  MPI_Scatterv(GetInput().data(), counts.data(), displs.data(), MPI_INT, local_vec.data(), local_n, MPI_INT, 0,
               MPI_COMM_WORLD);

  std::ranges::sort(local_vec);

  for (int i = 0; i < size; ++i) {
    int partner = GetNeighbor(i, rank, size);
    if (partner != -1) {
      PerformCompareSplit(local_vec, partner, rank);
    }
  }

  MPI_Gatherv(local_vec.data(), local_n, MPI_INT, GetOutput().data(), counts.data(), displs.data(), MPI_INT, 0,
              MPI_COMM_WORLD);

  return true;
}

bool OddEvenSortMPI::PostProcessingImpl() {
  return true;
}

}  // namespace volkov_a_odd_even_transposition
