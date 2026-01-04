#include "gonozov_l_global_search/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "gonozov_l_global_search/common/include/common.hpp"

namespace gonozov_l_global_search {

GonozovLGlobalSearchMPI::GonozovLGlobalSearchMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool GonozovLGlobalSearchMPI::ValidationImpl() {
  return (std::get<1>(GetInput()) > 1.0) && (std::get<2>(GetInput()) < std::get<3>(GetInput())) &&
         (std::get<4>(GetInput()) > 0);
}

bool GonozovLGlobalSearchMPI::PreProcessingImpl() {
  return true;
}

namespace {

double CountM(int t, double highm, const std::vector<double> &x, const std::function<double(double)> &f) {
  if (highm == -std::numeric_limits<double>::infinity()) {
    return std::abs((f(x[1]) - f(x[0])) / (x[1] - x[0]));
  }

  double highm1 = std::abs((f(x.back()) - f(x[t - 1])) / (x.back() - x[t - 1]));
  double highm2 = std::abs((f(x[t]) - f(x.back())) / (x[t] - x.back()));

  double high = std::max(highm, highm1);
  return std::max(high, highm2);
}

double IntervalCharacteristic(double x1, double x2, double f1, double f2, double m) {
  double dx = x2 - x1;
  return (m * dx) + ((f2 - f1) * (f2 - f1) / (m * dx)) - (2.0 * (f1 + f2));
}

void InizialzationStartParameters(int proc_rank, std::vector<double> &test_sequence,
                                  const std::function<double(double)> &function, double &global_min_x,
                                  double &global_min_value, double a, double b) {
  if (proc_rank == 0) {
    test_sequence = {a, b};
    if (function(b) < global_min_value) {
      global_min_x = b;
      global_min_value = function(b);
    }
  }
}

void FormNewtestSequence(int proc_rank, std::vector<double> &test_sequence, int *n) {
  if (proc_rank == 0) {
    *n = static_cast<int>(test_sequence.size());
  }
  MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (proc_rank != 0) {
    test_sequence.resize(*n);
  }

  MPI_Bcast(test_sequence.data(), *n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::ranges::sort(test_sequence.begin(), test_sequence.end());
}

void FormNewParameters(int proc_rank, double &m, double &highm, double r, int t,
                       const std::vector<double> &test_sequence, const std::function<double(double)> &function) {
  if (proc_rank == 0) {
    highm = CountM(t, highm, test_sequence, function);
    m = (highm == 0.0) ? 1.0 : r * highm;
  }
  MPI_Bcast(&m, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void CountingNewCoordinateContinueIteration(int proc_rank, int glob, bool &continue_iteration,
                                            std::vector<double> &test_sequence, int &t,
                                            const std::function<double(double)> &function, double &global_min_x,
                                            double &global_min_value, double m, double eps) {
  if (proc_rank == 0) {
    if (glob <= 0 || std::cmp_greater_equal(glob, test_sequence.size())) {
      continue_iteration = false;
    } else {
      t = glob;
      if (t < 1) {
        continue_iteration = false;
      }

      double x_new = (0.5 * (test_sequence[t] + test_sequence[t - 1])) -
                     ((function(test_sequence[t]) - function(test_sequence[t - 1])) / (2.0 * m));

      double fx = function(x_new);
      if (fx < global_min_value) {
        global_min_value = fx;
        global_min_x = x_new;
      }
      test_sequence.push_back(x_new);
    }

    continue_iteration = std::abs(test_sequence[t] - test_sequence[t - 1]) > eps;
  }

  MPI_Bcast(&continue_iteration, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
}
}  // namespace
bool GonozovLGlobalSearchMPI::RunImpl() {
  auto function = std::get<0>(GetInput());
  double r = std::get<1>(GetInput());
  double a = std::get<2>(GetInput());
  double b = std::get<3>(GetInput());
  double eps = std::get<4>(GetInput());

  int proc_num = 0;
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

  std::vector<double> test_sequence;
  double global_min_x = a;
  double global_min_value = function(a);

  int t = 1;
  double highm = -std::numeric_limits<double>::infinity();

  InizialzationStartParameters(proc_rank, test_sequence, function, global_min_x, global_min_value, a, b);

  bool continue_iteration = true;

  while (continue_iteration) {
    int n = 0;
    FormNewtestSequence(proc_rank, test_sequence, &n);

    double m = 0.0;

    FormNewParameters(proc_rank, m, highm, r, t, test_sequence, function);

    int intervals = n - 1;
    int per_proc = intervals / proc_num;
    int rem = intervals % proc_num;

    int l = (proc_rank * per_proc) + std::min(proc_rank, rem);
    int r_i = l + per_proc + (proc_rank < rem ? 1 : 0);

    double local_max = -std::numeric_limits<double>::infinity();
    int local_idx = -1;

    if (l < r_i) {
      for (int i = l + 1; i <= r_i; ++i) {
        double highr = IntervalCharacteristic(test_sequence[i - 1], test_sequence[i], function(test_sequence[i - 1]),
                                              function(test_sequence[i]), m);
        if (highr > local_max) {
          local_max = highr;
          local_idx = i;
        }
      }
    }

    struct {
      double val;
      int idx;
    } loc{}, glob{};
    loc.val = local_max;
    loc.idx = local_idx;

    MPI_Reduce(&loc, &glob, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);

    CountingNewCoordinateContinueIteration(proc_rank, glob.idx, continue_iteration, test_sequence, t, function,
                                           global_min_x, global_min_value, m, eps);
  }

  // рассылаем результат всем процессам
  MPI_Bcast(&global_min_x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = global_min_x;

  return true;
}

bool GonozovLGlobalSearchMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gonozov_l_global_search
