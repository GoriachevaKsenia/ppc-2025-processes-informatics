#include "kutuzov_i_simpson_integration/seq/include/ops_seq.hpp"

#include <tuple>

#include "kutuzov_i_simpson_integration/common/include/common.hpp"

namespace kutuzov_i_simpson_integration {

KutuzovISimpsonIntegrationSEQ::KutuzovISimpsonIntegrationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool KutuzovISimpsonIntegrationSEQ::ValidationImpl() {
  int n = std::get<0>(GetInput());
  double x_min = std::get<1>(GetInput()).first;
  double x_max = std::get<1>(GetInput()).second;
  double y_min = std::get<2>(GetInput()).first;
  double y_max = std::get<2>(GetInput()).second;
  int function_id = std::get<3>(GetInput());

  if (x_min >= x_max) {
    return false;
  }
  if (y_min >= y_max) {
    return false;
  }
  if (n <= 0 || n % 2 != 0) {
    return false;
  }
  if (function_id <= 0 || function_id > 4) {
    return false;
  }
  return true;
}

bool KutuzovISimpsonIntegrationSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool KutuzovISimpsonIntegrationSEQ::RunImpl() {
  int n = std::get<0>(GetInput());
  double x_min = std::get<0>(std::get<1>(GetInput()));
  double x_max = std::get<1>(std::get<1>(GetInput()));
  double y_min = std::get<0>(std::get<2>(GetInput()));
  double y_max = std::get<1>(std::get<2>(GetInput()));
  int function_id = std::get<3>(GetInput());

  double step_x = (x_max - x_min) / n;
  double step_y = (y_max - y_min) / n;

  double sum = 0;
  for (int i = 0; i <= n; i++) {
    double x = x_min + (step_x * i);

    for (int j = 0; j <= n; j++) {
      double y = y_min + (step_y * j);
      sum += GetWeight(i, n) * GetWeight(j, n) * CallFunction(function_id, x, y);
    }
  }
  sum *= step_x * step_y / 9;
  GetOutput() = sum;
  return true;
}

bool KutuzovISimpsonIntegrationSEQ::PostProcessingImpl() {
  return true;
}

double KutuzovISimpsonIntegrationSEQ::GetWeight(int i, int n) {
  if (i == 0 || i == n) {
    return 1.0;
  }
  if (i % 2 == 1) {
    return 4.0;
  }
  return 2.0;
}

}  // namespace kutuzov_i_simpson_integration
