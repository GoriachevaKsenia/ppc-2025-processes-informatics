#include "luzan_e_simps_int/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <tuple>
#include <vector>

#include "luzan_e_simps_int/common/include/common.hpp"

namespace luzan_e_simps_int {

LuzanESimpsIntMPI::LuzanESimpsIntMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetOutput() = {};

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetInput() = in;
  } else {
    GetInput() = {};
  }
}

bool LuzanESimpsIntMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return true;
  }

  int n = std::get<0>(GetInput());
  double a = std::get<0>(std::get<1>(GetInput()));
  double b = std::get<1>(std::get<1>(GetInput()));
  double c = std::get<0>(std::get<2>(GetInput()));
  double d = std::get<1>(std::get<2>(GetInput()));
  int func_num = std::get<3>(GetInput());
  bool flag = (a < b) && (c < d) && (n % 2 == 0) && (n > 0) && (func_num >= 0);
  return flag;
}

bool LuzanESimpsIntMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return true;
  }

  GetOutput() = 0.0;
  return true;
}

bool LuzanESimpsIntMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double a = 0.0;
  double b = 0.0;
  double c = 0.0;
  double d = 0.0;
  int n = 0;  // кол-во отрезков
  int func_num = 0;

  // sharing data
  std::vector<double> tmp(4, 0);
  if (rank == 0) {
    n = std::get<0>(GetInput());
    a = std::get<0>(std::get<1>(GetInput()));
    b = std::get<1>(std::get<1>(GetInput()));
    c = std::get<0>(std::get<2>(GetInput()));
    d = std::get<1>(std::get<2>(GetInput()));
    func_num = std::get<3>(GetInput());
    tmp = {a, b, c, d};
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&func_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(tmp.data(), 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  a = tmp[0];
  b = tmp[1];
  c = tmp[2];
  d = tmp[3];
  tmp.resize(0);

  // mpi things
  int rest = (n + 1) % size;
  int per_proc = (n + 1) / size;

  int i_start = (rank * per_proc) + (rest > rank ? rank : rest);
  int i_end = i_start + per_proc + (rest > rank ? 1 : 0);

  // calculations
  double (*fp)(double, double) = GetFunc(func_num);
  double hx = (b - a) / n;
  double hy = (d - c) / n;

  double wx = 1.0;
  double wy = 1.0;
  double x = 0.0;
  double y = 0.0;

  double local_sum = 0.0;
  /// printf_s("rnak: %i,    st = %i,  end = %i\n", rank, i_start, i_end);

  for (int i = i_start; i < i_end; i++) {
    x = a + (hx * i);
    wx = GetWeight(i, n);

    for (int j = 0; j < n + 1; j++) {
      y = c + (hy * j);
      wy = GetWeight(j, n);
      local_sum += wy * wx * fp(x, y);
    }
  }
  local_sum *= hx * hy / (3 * 3);
  double sum = 0;
  MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // finalize
  MPI_Bcast(&sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = sum;
  return true;
}

bool LuzanESimpsIntMPI::PostProcessingImpl() {
  return true;
}

}  // namespace luzan_e_simps_int
