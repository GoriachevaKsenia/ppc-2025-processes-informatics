#include "guseva_a_hypercube/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <guseva_a_hypercube/mpi/include/hypercube.hpp>

#include "guseva_a_hypercube/common/include/common.hpp"

namespace guseva_a_hypercube {

GusevaAHypercubeMine::GusevaAHypercubeMine(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput();
}

bool GusevaAHypercubeMine::ValidationImpl() {
  return std::get<0>(GetInput()) < std::get<1>(GetInput());
}

bool GusevaAHypercubeMine::PreProcessingImpl() {
  return true;
}

bool GusevaAHypercubeMine::RunImpl() {
  hypercube_handler_ = Hypercube::Create();
  const auto &[a, b, func, unused] = GetInput();

  if (hypercube_handler_) {
    const auto &rank = hypercube_handler_->GetRank();
    const auto &size = hypercube_handler_->GetSize();

    const auto &length = b - a + 1;
    const auto &base = length / size;
    const auto &remainder = length % size;
    const auto &offset = (rank * base) + std::min(rank, remainder);
    const auto &local_size = base + (rank < remainder ? 1 : 0);

    const auto &local_start = a + offset;
    const auto &local_end = local_start + local_size;

    double local_result = 0;

    for (int xx = local_start; xx < local_end; xx++) {
      local_result += func(xx);
    }

    double result = 0;

    hypercube_handler_->ReduceSum(local_result, MPI_DOUBLE, result);
    GetOutput() = result;
  }

  MPI_Bcast(&GetOutput(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return true;
}

bool GusevaAHypercubeMine::PostProcessingImpl() {
  return true;
}

}  // namespace guseva_a_hypercube
