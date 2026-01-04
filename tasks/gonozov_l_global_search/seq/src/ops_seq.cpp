#include "gonozov_l_global_search/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <tuple>
#include <vector>

#include "gonozov_l_global_search/common/include/common.hpp"

namespace gonozov_l_global_search {
GonozovLGlobalSearchSEQ::GonozovLGlobalSearchSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool GonozovLGlobalSearchSEQ::ValidationImpl() {
  return (std::get<1>(GetInput()) > 1.0) && (std::get<2>(GetInput()) < std::get<3>(GetInput())) &&
         (std::get<4>(GetInput()) > 0);
}

bool GonozovLGlobalSearchSEQ::PreProcessingImpl() {
  return true;
}

namespace {

double CountingM(int t, double &highm, const std::vector<double> &test_sequence, const auto &function) {
  if (highm == -std::numeric_limits<double>::infinity()) {
    return std::abs((function(test_sequence[1]) - function(test_sequence[0])) / (test_sequence[1] - test_sequence[0]));
  }
  double highm1 = std::abs(function(test_sequence.back()) - function(test_sequence[t - 1])) /
                  (test_sequence.back() - test_sequence[t - 1]);

  double highm2 =
      std::abs(function(test_sequence[t]) - function(test_sequence.back())) / (test_sequence[t] - test_sequence.back());

  double high = std::max(highm, highm1);
  return std::max(high, highm2);
}

int Countingt(double m, std::vector<double> &test_sequence, const auto &function) {
  int t = 1;
  double value_rt = -std::numeric_limits<double>::infinity();
  for (unsigned int i = 1; i < test_sequence.size(); i++) {
    double sub_x = (test_sequence[i] - test_sequence[i - 1]);
    double sub_z = (function(test_sequence[i]) - function(test_sequence[i - 1]));
    double sum_z = (function(test_sequence[i]) + function(test_sequence[i - 1]));
    double inter_value_rt = (m * sub_x) + (sub_z * sub_z / m / sub_x) - (2 * sum_z);
    if (inter_value_rt > value_rt) {
      value_rt = inter_value_rt;
      t = static_cast<int>(i);
    }
  }
  return t;
}

void InizialzationStartParameters(std::vector<double> &test_sequence, const std::function<double(double)> &function,
                                  double &global_min_x, double &global_min_value, double a, double b) {
  test_sequence.push_back(a);
  test_sequence.push_back(b);
  if (function(b) < global_min_value) {
    global_min_x = b;
    global_min_value = function(b);
  }
}
void FormNewtestSequence(std::vector<double> &test_sequence) {
  std::ranges::sort(test_sequence.begin(), test_sequence.end());
}

void FormNewParameters(double &m, double &highm, double r, int t, const std::vector<double> &test_sequence,
                       const std::function<double(double)> &function) {
  highm = CountingM(t, highm, test_sequence, function);
  m = (highm == 0.0) ? 1.0 : r * highm;
}

void CountingNewCoordinateContinueIteration(bool &continue_iteration, std::vector<double> &test_sequence, int &t,
                                            const std::function<double(double)> &function, double &global_min_x,
                                            double &global_min_value, double m, double epsilon) {
  double new_elem_sequence = (0.5 * (test_sequence[t] + test_sequence[t - 1])) -
                             ((0.5 / m) * (function(test_sequence[t]) - function(test_sequence[t - 1])));
  if (function(new_elem_sequence) < global_min_value) {
    global_min_x = new_elem_sequence;
    global_min_value = function(new_elem_sequence);
  }
  test_sequence.push_back(new_elem_sequence);

  continue_iteration = abs(test_sequence[t] - test_sequence[t - 1]) > epsilon;
}

}  // namespace

bool GonozovLGlobalSearchSEQ::RunImpl() {
  std::vector<double> test_sequence;
  const std::function<double(double)> &function = std::get<0>(GetInput());
  double r = std::get<1>(GetInput());
  double a = std::get<2>(GetInput());
  double b = std::get<3>(GetInput());
  double epsilon = std::get<4>(GetInput());

  int t = 1;
  double highm = -std::numeric_limits<double>::infinity();

  double global_min_x = a;
  double global_min_value = function(a);

  InizialzationStartParameters(test_sequence, function, global_min_x, global_min_value, a, b);
  bool continue_iteration = true;
  while (continue_iteration) {
    FormNewtestSequence(test_sequence);

    double m = 0.0;
    FormNewParameters(m, highm, r, t, test_sequence, function);

    t = Countingt(m, test_sequence, function);

    CountingNewCoordinateContinueIteration(continue_iteration, test_sequence, t, function, global_min_x,
                                           global_min_value, m, epsilon);
  }

  GetOutput() = global_min_x;

  return true;
}

bool GonozovLGlobalSearchSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace gonozov_l_global_search
