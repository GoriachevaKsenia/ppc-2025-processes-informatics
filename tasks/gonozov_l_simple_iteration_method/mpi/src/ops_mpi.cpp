#include "gonozov_l_simple_iteration_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "gonozov_l_simple_iteration_method/common/include/common.hpp"
// #include "util/include/util.hpp"

namespace gonozov_l_simple_iteration_method {

GonozovLSimpleIterationMethodMPI::GonozovLSimpleIterationMethodMPI(const InType &in)
    : number_unknowns_(static_cast<int>(std::get<0>(in))) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();
}

bool GonozovLSimpleIterationMethodMPI::ValidationImpl() {
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

bool GonozovLSimpleIterationMethodMPI::PreProcessingImpl() {
  return true;
}
namespace {
void CalculatingNewApproximations(int my_first_row, std::vector<double> &local_matrix,
                                  std::vector<double> &previous_approximations, std::vector<double> &local_current,
                                  int local_size, int number_unknowns, std::vector<double> &local_b) {
  for (int i = 0; i < local_size; i++) {
    int global_row = my_first_row + i;
    double sum = 0.0;

    // Суммируем все недиагональные элементы
    for (int j = 0; j < number_unknowns; j++) {
      if (j != global_row) {
        sum += local_matrix[(i * number_unknowns) + j] * previous_approximations[j];
      }
    }

    local_current[i] = (local_b[i] - sum) / local_matrix[(i * number_unknowns) + global_row];
  }
}

int CountingMyFirstRow(int proc_rank, int number_processed, int remainder) {
  int my_first_row = 0;
  for (int pl = 0; pl < proc_rank; pl++) {
    my_first_row += number_processed + (pl < remainder ? 1 : 0);
  }
  return my_first_row;
}

int CountingLocalSize(int number_processed, int proc_rank, int remainder) {
  return number_processed + (proc_rank < remainder ? 1 : 0);
}

int ConvergenceCheck(int my_first_row, std::vector<double> current_approximations,
                     std::vector<double> previous_approximations, int local_size) {
  int local_converged = 0;
  for (int i = 0; i < local_size; i++) {
    int global_row = my_first_row + i;
    double diff = abs(current_approximations[global_row] - previous_approximations[global_row]);
    double norm = abs(current_approximations[global_row]);
    if (diff < 0.00001 * (norm + 1e-10)) {
      local_converged++;
    }
  }
  return local_converged;
}
}  // namespace

bool GonozovLSimpleIterationMethodMPI::RunImpl() {
  int max_number_iteration = 10000;
  int proc_num = 0;
  int proc_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  std::vector<double> matrix;
  std::vector<double> b;

  if (proc_rank == 0) {
    matrix = std::get<1>(GetInput());
    b = std::get<2>(GetInput());
    number_unknowns_ = std::get<0>(GetInput());
  }

  // Распределение строк по процессам
  int number_processed = number_unknowns_ / proc_num;
  int remainder = number_unknowns_ % proc_num;
  int local_size = CountingLocalSize(number_processed, proc_rank, remainder);

  // Определяем локально, какие строки обрабатывает какой из процессов
  int my_first_row = CountingMyFirstRow(proc_rank, number_processed, remainder);

  // Данные для Scatterv
  std::vector<int> sendcounts_for_matrix(proc_num, 0);
  std::vector<int> sendcounts_for_b(proc_num, 0);
  std::vector<int> displs_for_matrix(proc_num, 0);
  std::vector<int> displs_for_b(proc_num, 0);

  if (proc_rank == 0) {
    int offset_matrix = 0;
    int offset_b = 0;
    for (int pl = 0; pl < proc_num; pl++) {
      int p_size = number_processed + (pl < remainder ? 1 : 0);
      sendcounts_for_matrix[pl] = p_size * number_unknowns_;
      sendcounts_for_b[pl] = p_size;

      displs_for_matrix[pl] = offset_matrix;
      displs_for_b[pl] = offset_b;

      offset_matrix += sendcounts_for_matrix[pl];
      offset_b += sendcounts_for_b[pl];
    }
  }

  // Информация требующаяся для работы MPI_Allgatherv
  MPI_Bcast(sendcounts_for_b.data(), proc_num, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs_for_b.data(), proc_num, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<double> local_matrix(static_cast<size_t>(local_size * number_unknowns_));
  std::vector<double> local_b(local_size);

  MPI_Scatterv((proc_rank == 0) ? matrix.data() : nullptr, sendcounts_for_matrix.data(), displs_for_matrix.data(),
               MPI_DOUBLE, local_matrix.data(), local_size * number_unknowns_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv((proc_rank == 0) ? b.data() : nullptr, sendcounts_for_b.data(), displs_for_b.data(), MPI_DOUBLE,
               local_b.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> previous_approximations(number_unknowns_, 0.0);
  std::vector<double> current_approximations(number_unknowns_, 0.0);
  std::vector<double> local_previous(local_size, 0.0);
  std::vector<double> local_current(local_size, 0.0);

  // Подсчёт нулевого приближени
  for (int i = 0; i < local_size; i++) {
    int global_row = my_first_row + i;
    local_previous[i] = local_b[i] / local_matrix[(i * number_unknowns_) + global_row];
  }

  // Отправка начальных  приближений
  MPI_Allgatherv(local_previous.data(), local_size, MPI_DOUBLE, previous_approximations.data(), sendcounts_for_b.data(),
                 displs_for_b.data(), MPI_DOUBLE, MPI_COMM_WORLD);

  // Основной цикл
  for (int iter = 0; iter < max_number_iteration; iter++) {
    // Каждый процесс вычисляет новые приближения для строк, которые у него имеются
    CalculatingNewApproximations(my_first_row, local_matrix, previous_approximations, local_current, local_size,
                                 number_unknowns_, local_b);

    MPI_Allgatherv(local_current.data(), local_size, MPI_DOUBLE, current_approximations.data(), sendcounts_for_b.data(),
                   displs_for_b.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    // Проверка сходимости
    int local_converged = ConvergenceCheck(my_first_row, current_approximations, previous_approximations, local_size);

    int global_converged = 0;
    MPI_Allreduce(&local_converged, &global_converged, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (global_converged == number_unknowns_) {
      break;
    }

    previous_approximations.swap(current_approximations);
    local_previous = local_current;
  }

  GetOutput() = current_approximations;

  return true;
}

bool GonozovLSimpleIterationMethodMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gonozov_l_simple_iteration_method
