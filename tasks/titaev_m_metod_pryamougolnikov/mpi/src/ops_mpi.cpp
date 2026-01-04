#include "titaev_m_metod_pryamougolnikov/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "titaev_m_metod_pryamougolnikov/common/include/common.hpp"
namespace titaev_m_metod_pryamougolnikov {

namespace {

double Function(const std::vector<double> &coords) {
  double sum = 0.0;
  for (double x : coords) {
    sum += x;
  }
  return sum;
}

double ComputeLocalSum(int start_index, int end_index, int partitions, const std::vector<double> &step_sizes,
                       const std::vector<double> &left_bounds) {
  double local_sum = 0.0;
  const int dimensions = static_cast<int>(left_bounds.size());

  std::vector<int> indices(dimensions, 0);
  int total_points = 1;
  for (int dimension = 0; dimension < dimensions; ++dimension) {
    total_points *= partitions;
  }

  for (int point_idx = 0; point_idx < total_points; ++point_idx) {
    int temp = point_idx;
    for (int dimension = 0; dimension < dimensions; ++dimension) {
      indices[dimension] = temp % partitions;
      temp /= partitions;
    }
    if (indices[0] < start_index || indices[0] > end_index) {
      continue;
    }

    std::vector<double> point(dimensions);
    for (int dimension = 0; dimension < dimensions; ++dimension) {
      point[dimension] = left_bounds[dimension] + ((indices[dimension] + 0.5) * step_sizes[dimension]);
    }

    local_sum += Function(point);
  }

  return local_sum;
}

}  // namespace

TitaevMMetodPryamougolnikovMPI::TitaevMMetodPryamougolnikovMPI(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  GetOutput() = 0.0;
}

bool TitaevMMetodPryamougolnikovMPI::ValidationImpl() {
  const auto &input = GetInput();
  if (input.left_bounds.size() != input.right_bounds.size()) {
    return false;
  }
  if (input.partitions <= 0) {
    return false;
  }
  for (std::size_t i = 0; i < input.left_bounds.size(); ++i) {
    if (input.right_bounds[i] <= input.left_bounds[i]) {
      return false;
    }
  }
  return true;
}

bool TitaevMMetodPryamougolnikovMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

double TitaevMMetodPryamougolnikovMPI::IntegrandFunction(const std::vector<double> &coords) {
  return Function(coords);
}

bool TitaevMMetodPryamougolnikovMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  const int partitions = input.partitions;
  const int dimensions = static_cast<int>(input.left_bounds.size());

  if (dimensions == 0) {
    GetOutput() = 0.0;
    return true;
  }

  std::vector<double> step_sizes(dimensions);
  for (int dimension = 0; dimension < dimensions; ++dimension) {
    step_sizes[dimension] = (input.right_bounds[dimension] - input.left_bounds[dimension]) / partitions;
  }

  const int chunk_size = partitions / size;
  const int remainder = partitions % size;
  const int start_index = (rank * chunk_size) + std::min(rank, remainder);
  const int end_index = start_index + chunk_size - 1 + (rank < remainder ? 1 : 0);

  double local_sum = ComputeLocalSum(start_index, end_index, partitions, step_sizes, input.left_bounds);

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  double volume_element = 1.0;
  for (double h : step_sizes) {
    volume_element *= h;
  }

  if (rank == 0) {
    GetOutput() = global_sum * volume_element;
  }

  MPI_Bcast(&GetOutput(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return true;
}

bool TitaevMMetodPryamougolnikovMPI::PostProcessingImpl() {
  return true;
}

}  // namespace titaev_m_metod_pryamougolnikov
