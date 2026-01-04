#include "nikitin_a_fox_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "nikitin_a_fox_algorithm/common/include/common.hpp"

namespace nikitin_a_fox_algorithm {

NikitinAFoxAlgorithmMPI::NikitinAFoxAlgorithmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool NikitinAFoxAlgorithmMPI::ValidateMatricesOnRoot(const std::vector<std::vector<double>> &matrix_a,
                                                     const std::vector<std::vector<double>> &matrix_b) {
  if (matrix_a.empty() || matrix_b.empty()) {
    return false;
  }

  const auto n = static_cast<int>(matrix_a.size());
  for (int i = 0; i < n; ++i) {
    if (matrix_a[i].size() != static_cast<std::size_t>(n)) {
      return false;
    }
  }

  if (matrix_b.size() != static_cast<std::size_t>(n)) {
    return false;
  }

  for (int i = 0; i < n; ++i) {
    if (matrix_b[i].size() != static_cast<std::size_t>(n)) {
      return false;
    }
  }

  return true;
}

bool NikitinAFoxAlgorithmMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &matrix_a = GetInput().first;
    const auto &matrix_b = GetInput().second;
    return ValidateMatricesOnRoot(matrix_a, matrix_b);
  }

  return true;
}

bool NikitinAFoxAlgorithmMPI::PreProcessingImpl() {
  return true;
}

void NikitinAFoxAlgorithmMPI::DistributeMatrixB(int n, std::vector<double> &local_b) {
  const auto &matrix_b = GetInput().second;
  // Заполняем локальный буфер матрицей B
  for (int i = 0; i < n; ++i) {
    const auto i_offset = static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
    for (int j = 0; j < n; ++j) {
      local_b[i_offset + static_cast<std::size_t>(j)] = matrix_b[i][j];
    }
  }
}

void NikitinAFoxAlgorithmMPI::SendMatrixAToProcess(int dest, int rows_per_proc, int remainder, int n,
                                                   int &current_row) {
  const auto &matrix_a = GetInput().first;
  const int dest_rows = (dest < remainder) ? (rows_per_proc + 1) : rows_per_proc;
  const auto dest_elements = static_cast<std::size_t>(dest_rows) * static_cast<std::size_t>(n);
  std::vector<double> send_buffer(dest_elements);

  for (int i = 0; i < dest_rows; ++i) {
    const auto i_offset = static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
    for (int j = 0; j < n; ++j) {
      send_buffer[i_offset + static_cast<std::size_t>(j)] = matrix_a[current_row][j];
    }
    current_row++;
  }

  MPI_Send(send_buffer.data(), static_cast<int>(dest_elements), MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
}

void NikitinAFoxAlgorithmMPI::DistributeMatrixA(int rank, int size, int n, int local_rows,
                                                std::vector<double> &local_a) {
  if (rank == 0) {
    const auto &matrix_a = GetInput().first;

    // Сначала копируем строки для процесса 0
    int current_row = 0;
    for (int i = 0; i < local_rows; ++i) {
      const auto i_offset = static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
      for (int j = 0; j < n; ++j) {
        local_a[i_offset + static_cast<std::size_t>(j)] = matrix_a[current_row][j];
      }
      current_row++;
    }

    // Отправляем строки остальным процессам
    const int rows_per_proc = n / size;
    const int remainder = n % size;
    for (int dest = 1; dest < size; ++dest) {
      SendMatrixAToProcess(dest, rows_per_proc, remainder, n, current_row);
    }
  } else {
    // Получаем свои строки от процесса 0
    const auto local_elements = static_cast<std::size_t>(local_rows) * static_cast<std::size_t>(n);
    MPI_Recv(local_a.data(), static_cast<int>(local_elements), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void NikitinAFoxAlgorithmMPI::LocalMatrixMultiply(int n, int local_rows, const std::vector<double> &local_a,
                                                  const std::vector<double> &local_b, std::vector<double> &local_c) {
  for (int i = 0; i < local_rows; ++i) {
    const auto i_idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        const auto k_idx = static_cast<std::size_t>(k) * static_cast<std::size_t>(n);
        sum += local_a[i_idx + static_cast<std::size_t>(k)] * local_b[k_idx + static_cast<std::size_t>(j)];
      }
      local_c[i_idx + static_cast<std::size_t>(j)] = sum;
    }
  }
}

void NikitinAFoxAlgorithmMPI::CopyLocalResultsToOutput(int local_rows, int n, const std::vector<double> &local_c) {
  int current_row = 0;
  for (int i = 0; i < local_rows; ++i) {
    const auto i_idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
    for (int j = 0; j < n; ++j) {
      GetOutput()[current_row][j] = local_c[i_idx + static_cast<std::size_t>(j)];
    }
    current_row++;
  }
}

void NikitinAFoxAlgorithmMPI::ReceiveResultsFromProcess(int src, int rows_per_proc, int remainder, int n,
                                                        int &current_row) {
  const int src_rows = (src < remainder) ? (rows_per_proc + 1) : rows_per_proc;
  const auto src_elements = static_cast<std::size_t>(src_rows) * static_cast<std::size_t>(n);
  std::vector<double> recv_buffer(src_elements);

  MPI_Recv(recv_buffer.data(), static_cast<int>(src_elements), MPI_DOUBLE, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  for (int i = 0; i < src_rows; ++i) {
    const auto i_idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
    for (int j = 0; j < n; ++j) {
      GetOutput()[current_row][j] = recv_buffer[i_idx + static_cast<std::size_t>(j)];
    }
    current_row++;
  }
}

void NikitinAFoxAlgorithmMPI::GatherResults(int rank, int size, int n, int rows_per_proc, int remainder, int local_rows,
                                            const std::vector<double> &local_c) {
  if (rank == 0) {
    // Создаем результирующую матрицу
    GetOutput() = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));

    // Копируем свои результаты
    CopyLocalResultsToOutput(local_rows, n, local_c);

    // Получаем результаты от других процессов
    int current_row = local_rows;
    for (int src = 1; src < size; ++src) {
      ReceiveResultsFromProcess(src, rows_per_proc, remainder, n, current_row);
    }
  } else {
    // Отправляем свои результаты процессу 0
    const auto local_elements = static_cast<std::size_t>(local_rows) * static_cast<std::size_t>(n);
    MPI_Send(local_c.data(), static_cast<int>(local_elements), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
  }
}

void NikitinAFoxAlgorithmMPI::CreateAndSendFlatResult(int n) {
  const auto total_elements = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  std::vector<double> flat_result(total_elements);

  for (int i = 0; i < n; ++i) {
    const auto i_offset = static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
    for (int j = 0; j < n; ++j) {
      flat_result[i_offset + static_cast<std::size_t>(j)] = GetOutput()[i][j];
    }
  }

  MPI_Bcast(flat_result.data(), static_cast<int>(total_elements), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  for (int i = 0; i < n; ++i) {
    const auto i_offset = static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
    for (int j = 0; j < n; ++j) {
      GetOutput()[i][j] = flat_result[i_offset + static_cast<std::size_t>(j)];
    }
  }
}

void NikitinAFoxAlgorithmMPI::ReceiveFlatResultAndCreateMatrix(int n) {
  const auto total_elements = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  GetOutput() = std::vector<std::vector<double>>(n, std::vector<double>(n));

  std::vector<double> flat_result(total_elements);
  MPI_Bcast(flat_result.data(), static_cast<int>(total_elements), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  for (int i = 0; i < n; ++i) {
    const auto i_offset = static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
    for (int j = 0; j < n; ++j) {
      GetOutput()[i][j] = flat_result[i_offset + static_cast<std::size_t>(j)];
    }
  }
}

void NikitinAFoxAlgorithmMPI::BroadcastResultToAll(int rank, int n) {
  if (rank == 0) {
    CreateAndSendFlatResult(n);
  } else {
    ReceiveFlatResultAndCreateMatrix(n);
  }
}

int NikitinAFoxAlgorithmMPI::GetMatrixSize(int rank) {
  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(GetInput().first.size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return n;
}

std::pair<int, int> NikitinAFoxAlgorithmMPI::CalculateRowDistribution(int n, int size, int rank) {
  const int rows_per_proc = n / size;
  const int remainder = n % size;
  const int local_rows = (rank < remainder) ? (rows_per_proc + 1) : rows_per_proc;
  return {rows_per_proc, local_rows};
}

bool NikitinAFoxAlgorithmMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Получаем размер матрицы
  const int n = GetMatrixSize(rank);

  if (n == 0) {
    return false;
  }

  // Вычисляем распределение строк
  const auto [rows_per_proc, local_rows] = CalculateRowDistribution(n, size, rank);
  const auto local_elements = static_cast<std::size_t>(local_rows) * static_cast<std::size_t>(n);

  // 1. Создаем и распределяем матрицу B всем процессам
  const auto total_elements = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  std::vector<double> local_b(total_elements);

  if (rank == 0) {
    DistributeMatrixB(n, local_b);
  }
  MPI_Bcast(local_b.data(), static_cast<int>(total_elements), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // 2. Распределяем матрицу A по строкам
  std::vector<double> local_a(local_elements);
  DistributeMatrixA(rank, size, n, local_rows, local_a);

  // 3. Локальное умножение
  std::vector<double> local_c(local_elements, 0.0);
  LocalMatrixMultiply(n, local_rows, local_a, local_b, local_c);

  // 4. Сбор результатов на процессе 0
  const int remainder = n % size;
  GatherResults(rank, size, n, rows_per_proc, remainder, local_rows, local_c);

  // 5. Рассылаем результат всем процессам (для проверки в тестах)
  BroadcastResultToAll(rank, n);

  return true;
}

bool NikitinAFoxAlgorithmMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const auto &matrix_c = GetOutput();
  return !matrix_c.empty();
}

}  // namespace nikitin_a_fox_algorithm
