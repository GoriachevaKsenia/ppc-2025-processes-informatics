#include "gonozov_l_simple_iteration_method/seq/include/ops_seq.hpp"

#include <cmath>
#include <tuple>
#include <vector>

#include "gonozov_l_simple_iteration_method/common/include/common.hpp"
// #include "util/include/util.hpp"

namespace gonozov_l_simple_iteration_method {

GonozovLSimpleIterationMethodSEQ::GonozovLSimpleIterationMethodSEQ(const InType &in)
    : number_unknowns_(static_cast<int>(std::get<0>(in))) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  std::vector g_outp(number_unknowns_, 0.0);
  GetOutput() = g_outp;
}

bool GonozovLSimpleIterationMethodSEQ::ValidationImpl() {
  // д.б. |a11| > |a12|+|a13|, |a22| > |a21|+|a23|, |a33| > |a31|+|a32|
  for (int i = 0; i < number_unknowns_; i++) {
    double sum = 0.0;
    for (int j = 0; j < number_unknowns_; j++) {
      if (j != i) {
        sum += std::get<1>(GetInput())[(i * number_unknowns_) + j];
      }
    }
    if (std::get<1>(GetInput())[(i * number_unknowns_) + i] < sum) {
      return false;
    }
  }
  return (static_cast<int>(std::get<0>(GetInput())) > 0) && (static_cast<int>(std::get<1>(GetInput()).size()) > 0) &&
         (static_cast<int>(std::get<2>(GetInput()).size()) > 0);
}

bool GonozovLSimpleIterationMethodSEQ::PreProcessingImpl() {
  return true;
}

namespace {
void CalculatingNewApproximations(std::vector<double> &matrix, std::vector<double> &previous_approximations,
                                  std::vector<double> &current_approximations, int number_unknowns,
                                  std::vector<double> &b) {
  for (int i = 0; i < number_unknowns; i++) {
    double sum = 0.0;
    // Суммируем все недиагональные элементы
    for (int j = 0; j < number_unknowns; j++) {
      if (j != i) {
        sum += matrix[(i * number_unknowns) + j] * previous_approximations[j];
      }
    }

    // Строим новое приближение
    current_approximations[i] = (b[i] - sum) / matrix[(i * number_unknowns) + i];
  }
}

int ConvergenceCheck(std::vector<double> &current_approximations, std::vector<double> &previous_approximations,
                     int number_unknowns) {
  int converged = 0;
  for (int i = 0; i < number_unknowns; i++) {
    double diff = fabs(current_approximations[i] - previous_approximations[i]);
    double norm = fabs(current_approximations[i]);
    if (diff < 0.00001 * (norm + 1e-10)) {
      converged++;
    }
  }
  return converged;
}
}  // namespace
bool GonozovLSimpleIterationMethodSEQ::RunImpl() {
  int max_number_iteration = 10000;

  std::vector<double> matrix = std::get<1>(GetInput());
  std::vector<double> b = std::get<2>(GetInput());

  std::vector<double> previous_approximations(number_unknowns_, 0.0);
  std::vector<double> current_approximations(number_unknowns_, 0.0);

  // Нулевое приближение
  for (int i = 0; i < number_unknowns_; i++) {
    previous_approximations[i] = b[i] / matrix[(i * number_unknowns_) + i];
  }

  // Основной цикл
  for (int iter = 0; iter < max_number_iteration; iter++) {
    // Для каждой переменной вычисляем новое приближение
    CalculatingNewApproximations(matrix, previous_approximations, current_approximations, number_unknowns_, b);

    // Проверка сходимости
    int converged = ConvergenceCheck(current_approximations, previous_approximations, number_unknowns_);

    // Если все переменные сошлись
    if (converged == number_unknowns_) {
      break;
    }

    previous_approximations = current_approximations;
  }

  for (int i = 0; i < number_unknowns_; i++) {
    GetOutput()[i] = current_approximations[i];
  }

  return true;
}

bool GonozovLSimpleIterationMethodSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace gonozov_l_simple_iteration_method
