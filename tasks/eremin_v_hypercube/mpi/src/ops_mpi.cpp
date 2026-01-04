#include "eremin_v_hypercube/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <tuple>

#include "eremin_v_hypercube/common/include/common.hpp"

namespace eremin_v_hypercube {

EreminVHypercubeMPI::EreminVHypercubeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool EreminVHypercubeMPI::ValidationImpl() {
  auto &input = GetInput();
  return (std::get<0>(input) < std::get<1>(input)) && (std::get<2>(input) > 0) && (std::get<2>(input) <= 100000000) &&
         (std::get<0>(input) >= -1e9) && (std::get<0>(input) <= 1e9) && (std::get<1>(input) >= -1e9) &&
         (std::get<1>(input) <= 1e9) && (GetOutput() == 0);
}

bool EreminVHypercubeMPI::PreProcessingImpl() {
  return true;
}

bool EreminVHypercubeMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int ndims = static_cast<int>(std::floor(std::log2(world_size)));
  int hypercube_size = (1 << ndims);

  double lower_bound = 0.0;
  double upper_bound = 0.0;
  double final_result = 0.0;
  int steps = 0;

  if (world_rank == 0) {
    auto &input = GetInput();
    lower_bound = std::get<0>(input);
    upper_bound = std::get<1>(input);
    steps = std::get<2>(input);
  }

  if (world_rank < hypercube_size) {
    int cube_rank = world_rank;

    BroadcastBoundsOverHypercube(cube_rank, ndims, lower_bound, upper_bound, steps);

    const auto in_function = std::get<3>(GetInput());
    double step_size = (upper_bound - lower_bound) / static_cast<double>(steps);
    double local_result = 0.0;

    for (int i = cube_rank; i < steps; i += hypercube_size) {
      local_result += in_function(lower_bound + ((static_cast<double>(i) + 0.5) * step_size));
    }
    local_result *= step_size;

    double current_sum = local_result;
    for (int dim = 0; dim < ndims; ++dim) {
      int neighbor = cube_rank ^ (1 << dim);
      double received_sum = 0.0;
      MPI_Status status;
      MPI_Sendrecv(&current_sum, 1, MPI_DOUBLE, neighbor, 10, &received_sum, 1, MPI_DOUBLE, neighbor, 10,
                   MPI_COMM_WORLD, &status);
      current_sum += received_sum;
    }
    final_result = current_sum;
  }

  MPI_Bcast(&final_result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = final_result;
  return true;
}

bool EreminVHypercubeMPI::PostProcessingImpl() {
  return true;
}

void EreminVHypercubeMPI::BroadcastBoundsOverHypercube(int cube_rank, int ndims, double &lower_bound,
                                                       double &upper_bound, int &steps) {
  for (int dim = ndims - 1; dim >= 0; --dim) {
    int neighbor = cube_rank ^ (1 << dim);
    const bool is_active = (cube_rank & ((1 << dim) - 1)) == 0;
    const bool is_sender = (cube_rank & (1 << dim)) == 0;

    if (!is_active) {
      continue;
    }

    if (is_sender) {
      MPI_Send(&lower_bound, 1, MPI_DOUBLE, neighbor, 0, MPI_COMM_WORLD);
      MPI_Send(&upper_bound, 1, MPI_DOUBLE, neighbor, 1, MPI_COMM_WORLD);
      MPI_Send(&steps, 1, MPI_INT, neighbor, 2, MPI_COMM_WORLD);
    } else {
      MPI_Recv(&lower_bound, 1, MPI_DOUBLE, neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&upper_bound, 1, MPI_DOUBLE, neighbor, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&steps, 1, MPI_INT, neighbor, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

}  // namespace eremin_v_hypercube
