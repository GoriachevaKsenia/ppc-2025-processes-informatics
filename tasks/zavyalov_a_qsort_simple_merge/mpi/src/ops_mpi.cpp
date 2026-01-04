#include "zavyalov_a_qsort_simple_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "zavyalov_a_qsort_simple_merge/common/include/common.hpp"

namespace zavyalov_a_qsort_simple_merge {

ZavyalovAQsortMPI::ZavyalovAQsortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ZavyalovAQsortMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool ZavyalovAQsortMPI::PreProcessingImpl() {
  return true;
}

bool ZavyalovAQsortMPI::RunImpl() {
  int world_size = 0;
  int rank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int vec_size = static_cast<int>(GetInput().size());

  MPI_Bcast(&vec_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> sendcounts(world_size);
  std::vector<int> displs(world_size);

  int blocksize = vec_size / world_size;
  int elements_left = vec_size - (world_size * blocksize);
  int elements_processed = 0;

  for (int i = 0; i < world_size; i++) {
    sendcounts[i] = blocksize + (i < elements_left ? 1 : 0);
    displs[i] = elements_processed;
    elements_processed += sendcounts[i];
  }

  int elements_count = sendcounts[rank];
  std::vector<double> local_vector(elements_count);

  MPI_Scatterv(GetInput().data(), sendcounts.data(), displs.data(), MPI_DOUBLE, local_vector.data(), elements_count,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MyQsort(local_vector.data(), 0, elements_count - 1);

  if (rank == 0) {
    std::vector<double> res(vec_size);
    MPI_Gatherv(local_vector.data(), elements_count, MPI_DOUBLE, res.data(), sendcounts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 1; i < world_size; i++) {
      std::inplace_merge(res.data(), res.data() + displs[i], res.data() + displs[i] + sendcounts[i]);
    }
    if (!res.empty()) {
      GetOutput() = res;
    }
  } else {
    MPI_Gatherv(local_vector.data(), elements_count, MPI_DOUBLE, nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
  }

  return true;
}

bool ZavyalovAQsortMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zavyalov_a_qsort_simple_merge
