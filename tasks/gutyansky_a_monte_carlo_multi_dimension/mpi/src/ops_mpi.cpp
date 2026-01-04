#include "gutyansky_a_monte_carlo_multi_dimension/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "gutyansky_a_monte_carlo_multi_dimension/common/include/common.hpp"
#include "gutyansky_a_monte_carlo_multi_dimension/common/include/function_registry.hpp"

namespace gutyansky_a_monte_carlo_multi_dimension {

namespace {}

GutyanskyAMonteCarloMultiDimensionMPI::GutyanskyAMonteCarloMultiDimensionMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  GetInput() = in;
  GetOutput() = {};
}

bool GutyanskyAMonteCarloMultiDimensionMPI::ValidationImpl() {
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    return GetInput().IsValid();
  }

  return true;
}

bool GutyanskyAMonteCarloMultiDimensionMPI::PreProcessingImpl() {
  return true;
}

int GutyanskyAMonteCarloMultiDimensionMPI::ComputePackedTaskSize(size_t n_dims) {
  int size = 0;
  int tmp = 0;

  MPI_Pack_size(1, MPI_UINT64_T, MPI_COMM_WORLD, &tmp);  // func_id
  size += tmp;

  MPI_Pack_size(1, MPI_UINT64_T, MPI_COMM_WORLD, &tmp);  // n_dims
  size += tmp;

  MPI_Pack_size(1, MPI_UINT64_T, MPI_COMM_WORLD, &tmp);  // n_points
  size += tmp;

  MPI_Pack_size(static_cast<int>(n_dims), MPI_DOUBLE, MPI_COMM_WORLD, &tmp);  // lower_bounds
  size += tmp;

  MPI_Pack_size(static_cast<int>(n_dims), MPI_DOUBLE, MPI_COMM_WORLD, &tmp);  // upper_bounds
  size += tmp;

  return size;
}

void GutyanskyAMonteCarloMultiDimensionMPI::PackTaskData(const IntegrationTask &task, std::vector<uint8_t> &buffer) {
  int position = 0;

  MPI_Pack(&task.func_id, 1, MPI_UINT64_T, buffer.data(), static_cast<int>(buffer.size()), &position, MPI_COMM_WORLD);
  MPI_Pack(&task.n_dims, 1, MPI_UINT64_T, buffer.data(), static_cast<int>(buffer.size()), &position, MPI_COMM_WORLD);
  MPI_Pack(&task.n_points, 1, MPI_UINT64_T, buffer.data(), static_cast<int>(buffer.size()), &position, MPI_COMM_WORLD);
  MPI_Pack(task.lower_bounds.data(), static_cast<int>(task.n_dims), MPI_DOUBLE, buffer.data(),
           static_cast<int>(buffer.size()), &position, MPI_COMM_WORLD);
  MPI_Pack(task.upper_bounds.data(), static_cast<int>(task.n_dims), MPI_DOUBLE, buffer.data(),
           static_cast<int>(buffer.size()), &position, MPI_COMM_WORLD);
}

void GutyanskyAMonteCarloMultiDimensionMPI::UnpackTaskData(const std::vector<uint8_t> &buffer, IntegrationTask &task) {
  int position = 0;
  MPI_Unpack(buffer.data(), static_cast<int>(buffer.size()), &position, &task.func_id, 1, MPI_UINT64_T, MPI_COMM_WORLD);
  MPI_Unpack(buffer.data(), static_cast<int>(buffer.size()), &position, &task.n_dims, 1, MPI_UINT64_T, MPI_COMM_WORLD);
  MPI_Unpack(buffer.data(), static_cast<int>(buffer.size()), &position, &task.n_points, 1, MPI_UINT64_T,
             MPI_COMM_WORLD);
  task.lower_bounds.resize(task.n_dims);
  MPI_Unpack(buffer.data(), static_cast<int>(buffer.size()), &position, task.lower_bounds.data(),
             static_cast<int>(task.n_dims), MPI_DOUBLE, MPI_COMM_WORLD);
  task.upper_bounds.resize(task.n_dims);
  MPI_Unpack(buffer.data(), static_cast<int>(buffer.size()), &position, task.upper_bounds.data(),
             static_cast<int>(task.n_dims), MPI_DOUBLE, MPI_COMM_WORLD);
}

bool GutyanskyAMonteCarloMultiDimensionMPI::RunImpl() {
  int world_size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Bcast(&GetInput().n_dims, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  std::vector<uint8_t> buffer(ComputePackedTaskSize(GetInput().n_dims));

  if (rank == 0) {
    PackTaskData(GetInput(), buffer);
  }

  MPI_Bcast(buffer.data(), static_cast<int>(buffer.size()), MPI_PACKED, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    UnpackTaskData(buffer, GetInput());
  }

  size_t n_points = GetInput().n_points;
  size_t n_dims = GetInput().n_dims;

  size_t chunk_size = n_points / static_cast<size_t>(world_size);
  size_t remainder_size = n_points % static_cast<size_t>(world_size);

  size_t size = 0;

  if (std::cmp_less(rank, remainder_size)) {
    size = chunk_size + 1;
  } else {
    size = chunk_size;
  }

  FunctionRegistry::IntegralFunction function = GetInput().GetFunction();

  std::random_device rd;
  std::mt19937 gen(rd() + static_cast<unsigned>(rank));
  std::uniform_real_distribution<double> distr(0.0, 1.0);
  std::vector<double> random_point(n_dims);

  double sum = 0.0;

  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < n_dims; j++) {
      double lb = GetInput().lower_bounds[j];
      double rb = GetInput().upper_bounds[j];

      random_point[j] = lb + (distr(gen) * (rb - lb));
    }

    sum += function(random_point);
  }

  double total_sum = 0.0;
  MPI_Reduce(&sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    double volume = 1.0;

    for (size_t i = 0; i < n_dims; i++) {
      volume *= GetInput().upper_bounds[i] - GetInput().lower_bounds[i];
    }

    GetOutput() = volume * (total_sum / static_cast<double>(n_points));
  }

  return true;
}

bool GutyanskyAMonteCarloMultiDimensionMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gutyansky_a_monte_carlo_multi_dimension
