#include "dolov_v_monte_carlo_integration/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "dolov_v_monte_carlo_integration/common/include/common.hpp"

namespace dolov_v_monte_carlo_integration {

DolovVMonteCarloIntegrationMPI::DolovVMonteCarloIntegrationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool DolovVMonteCarloIntegrationMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  bool is_valid = true;
  if (rank == 0) {
    const auto &in = GetInput();
    is_valid = in.func && (in.samples_count > 0) && (in.dimension > 0) &&
               (in.center.size() == static_cast<size_t>(in.dimension)) && (in.radius > 0.0);
  }
  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return is_valid;
}

bool DolovVMonteCarloIntegrationMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool DolovVMonteCarloIntegrationMPI::RunImpl() {
  int current_rank = 0;
  int total_procs = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &total_procs);

  InType params = GetInput();
  int dim = params.dimension;
  int total_samples = params.samples_count;
  double rad = params.radius;
  int domain_type_int = static_cast<int>(params.domain_type);

  MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&total_samples, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rad, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&domain_type_int, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (current_rank != 0) {
    params.dimension = dim;
    params.samples_count = total_samples;
    params.radius = rad;
    params.domain_type = static_cast<IntegrationDomain>(domain_type_int);
    params.center.resize(dim);
  }

  if (dim > 0) {
    MPI_Bcast(params.center.data(), dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  int local_samples = total_samples / total_procs;
  if (current_rank == total_procs - 1) {
    local_samples += total_samples % total_procs;
  }

  const double r_sq = params.radius * params.radius;
  std::random_device rd;
  std::mt19937 random_generator(rd());
  std::uniform_real_distribution<double> value_distributor(-params.radius, params.radius);

  double local_sum_of_f = 0.0;
  std::vector<double> sample_point(params.dimension);

  for (int i = 0; i < local_samples; ++i) {
    for (int dim_idx = 0; dim_idx < params.dimension; ++dim_idx) {
      sample_point[dim_idx] = params.center[dim_idx] + value_distributor(random_generator);
    }

    bool is_valid_point = true;
    if (params.domain_type == IntegrationDomain::kHyperSphere) {
      double distance_sq = 0.0;
      for (int dim_idx = 0; dim_idx < params.dimension; ++dim_idx) {
        double diff = sample_point[dim_idx] - params.center[dim_idx];
        distance_sq += diff * diff;
      }
      is_valid_point = (distance_sq <= r_sq);
    }

    if (is_valid_point) {
      local_sum_of_f += params.func(sample_point);
    }
  }

  double global_sum_of_f = 0.0;
  MPI_Reduce(&local_sum_of_f, &global_sum_of_f, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  double final_result = 0.0;
  if (current_rank == 0) {
    const double volume = std::pow(2.0 * params.radius, params.dimension);
    final_result = volume * (global_sum_of_f / total_samples);
  }

  MPI_Bcast(&final_result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = final_result;
  return std::isfinite(GetOutput());
}

bool DolovVMonteCarloIntegrationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace dolov_v_monte_carlo_integration
