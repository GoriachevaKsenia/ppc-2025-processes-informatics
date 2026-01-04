#include "titaev_m_metod_pryamougolnikov/seq/include/ops_seq.hpp"

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

}  // namespace

TitaevMMetodPryamougolnikovSEQ::TitaevMMetodPryamougolnikovSEQ(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  GetOutput() = 0.0;
}

bool TitaevMMetodPryamougolnikovSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.left_bounds.size() != input.right_bounds.size()) {
    return false;
  }
  if (input.partitions <= 0) {
    return false;
  }
  for (std::size_t i = 0; i < input.left_bounds.size(); ++i) {  // Исправлено на std::size_t
    if (input.right_bounds[i] <= input.left_bounds[i]) {
      return false;
    }
  }
  return true;
}

bool TitaevMMetodPryamougolnikovSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

double TitaevMMetodPryamougolnikovSEQ::IntegrandFunction(const std::vector<double> &coords) {
  return Function(coords);
}

bool TitaevMMetodPryamougolnikovSEQ::RunImpl() {
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

  std::vector<int> indices(dimensions, 0);
  int total_points = 1;
  for (int dimension = 0; dimension < dimensions; ++dimension) {
    total_points *= partitions;
  }

  double total_sum = 0.0;
  for (int point_idx = 0; point_idx < total_points; ++point_idx) {
    int temp = point_idx;
    for (int dimension = 0; dimension < dimensions; ++dimension) {
      indices[dimension] = temp % partitions;
      temp /= partitions;
    }

    std::vector<double> point(dimensions);
    for (int dimension = 0; dimension < dimensions; ++dimension) {
      point[dimension] = input.left_bounds[dimension] + ((indices[dimension] + 0.5) * step_sizes[dimension]);
    }

    total_sum += Function(point);
  }

  double volume_element = 1.0;
  for (double h : step_sizes) {
    volume_element *= h;
  }

  GetOutput() = total_sum * volume_element;
  return true;
}

bool TitaevMMetodPryamougolnikovSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace titaev_m_metod_pryamougolnikov
