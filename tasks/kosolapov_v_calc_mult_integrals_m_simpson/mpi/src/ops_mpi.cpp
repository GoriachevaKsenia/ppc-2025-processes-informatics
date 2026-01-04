#include "kosolapov_v_calc_mult_integrals_m_simpson/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <tuple>

#include "kosolapov_v_calc_mult_integrals_m_simpson/common/include/common.hpp"

namespace kosolapov_v_calc_mult_integrals_m_simpson {

KosolapovVCalcMultIntegralsMSimpsonMPI::KosolapovVCalcMultIntegralsMSimpsonMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = 0.0;
}

bool KosolapovVCalcMultIntegralsMSimpsonMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    int steps = std::get<0>(GetInput());
    int func_id = std::get<1>(GetInput());
    return steps > 0 && (steps % 2 == 0) && func_id >= 1 && func_id <= 4;
  }
  return true;
}

bool KosolapovVCalcMultIntegralsMSimpsonMPI::PreProcessingImpl() {
  return true;
}

bool KosolapovVCalcMultIntegralsMSimpsonMPI::RunImpl() {
  int processes_count = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &processes_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int steps = 0;
  int func_id = 0;
  if (rank == 0) {
    steps = std::get<0>(GetInput());
    func_id = std::get<1>(GetInput());
  }
  MPI_Bcast(&steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&func_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int rows_per_process = (steps + 1) / processes_count;
  int remainder = (steps + 1) % processes_count;

  int start_i = 0;
  int end_i = 0;
  if (rank < remainder) {
    start_i = rank * (rows_per_process + 1);
    end_i = start_i + rows_per_process;
  } else {
    start_i = (remainder * (rows_per_process + 1)) + ((rank - remainder) * rows_per_process);
    end_i = start_i + rows_per_process - 1;
  }
  if (rank == processes_count - 1) {
    end_i = steps;
  }
  std::tuple<double, double, double, double> temp = GetBounds(func_id);
  double a = std::get<0>(temp);
  double b = std::get<1>(temp);
  double c = std::get<2>(temp);
  double d = std::get<3>(temp);
  double local_sum = ComputePartialSimpsonIntegral(func_id, steps, a, b, c, d, start_i, end_i);
  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  double result = 0.0;
  if (rank == 0) {
    double hx = (b - a) / steps;
    double hy = (d - c) / steps;
    result = global_sum * (hx * hy) / 9.0;
  }
  MPI_Bcast(&result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = result;
  return true;
}

bool KosolapovVCalcMultIntegralsMSimpsonMPI::PostProcessingImpl() {
  return true;
}

double KosolapovVCalcMultIntegralsMSimpsonMPI::Function1(double x, double y) {
  // f(x,y) = x^2 + y^2
  return (x * x) + (y * y);
}
double KosolapovVCalcMultIntegralsMSimpsonMPI::Function2(double x, double y) {
  // f(x,y) = sin(x) * cos(y)
  return std::sin(x) * std::cos(y);
}
double KosolapovVCalcMultIntegralsMSimpsonMPI::Function3(double x, double y) {
  // f(x,y) = exp(-(x^2 + y^2))
  return std::exp(-((x * x) + (y * y)));
}
double KosolapovVCalcMultIntegralsMSimpsonMPI::Function4(double x, double y) {
  // f(x,y) = sin(x + y)
  return std::sin(x + y);
}
double KosolapovVCalcMultIntegralsMSimpsonMPI::CallFunction(int func_id, double x, double y) {
  switch (func_id) {
    case 1:
      return Function1(x, y);
    case 2:
      return Function2(x, y);
    case 3:
      return Function3(x, y);
    case 4:
      return Function4(x, y);
    default:
      return Function1(x, y);
  }
}
std::tuple<double, double, double, double> KosolapovVCalcMultIntegralsMSimpsonMPI::GetBounds(int func_id) {
  switch (func_id) {
    case 1:
      return {0.0, 1.0, 0.0, 1.0};
    case 2:
      return {0.0, kPi, 0.0, kPi / 2.0};
    case 3:
      return {-1.0, 1.0, -1.0, 1.0};
    case 4:
      return {0.0, kPi, 0.0, kPi};
    default:
      return {0.0, 1.0, 0.0, 1.0};
  }
}
double KosolapovVCalcMultIntegralsMSimpsonMPI::ComputePartialSimpsonIntegral(int func_id, int steps, double a, double b,
                                                                             double c, double d, int start_i,
                                                                             int end_i) {
  double hx = (b - a) / steps;
  double hy = (d - c) / steps;
  double local_sum = 0.0;
  for (int i = start_i; i <= end_i; i++) {
    double x = a + (i * hx);
    double wx = GetSimpsonWeight(i, steps);
    for (int j = 0; j <= steps; j++) {
      double y = c + (j * hy);
      double wy = GetSimpsonWeight(j, steps);
      local_sum += (wx * wy) * CallFunction(func_id, x, y);
    }
  }
  return local_sum;
}
double KosolapovVCalcMultIntegralsMSimpsonMPI::GetSimpsonWeight(int index, int steps) {
  if (index == 0 || index == steps) {
    return 1.0;
  }
  return (index % 2 == 0) ? 2.0 : 4.0;
}

}  // namespace kosolapov_v_calc_mult_integrals_m_simpson
