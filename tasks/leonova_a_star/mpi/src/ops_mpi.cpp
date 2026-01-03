#include "leonova_a_star/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include "leonova_a_star/common/include/common.hpp"

namespace leonova_a_star {

namespace {

constexpr size_t kMaxMatrixSize = 10000;
constexpr size_t kMaxVectorSize = 1000000;
constexpr int kMaxProcesses = 1024;

bool CheckMatricesNotEmpty(const std::vector<std::vector<int>> &matrix_a,
                           const std::vector<std::vector<int>> &matrix_b) {
  return !matrix_a.empty() && !matrix_b.empty() && !matrix_a[0].empty() && !matrix_b[0].empty();
}

void FillMatrixBRow(int row_idx, int cols, const std::vector<std::vector<int>> &matrix_b, std::vector<int> &output) {
  if (static_cast<size_t>(row_idx) >= matrix_b.size()) {
    output.insert(output.end(), static_cast<size_t>(cols), 0);
    return;
  }

  const auto &row = matrix_b[static_cast<size_t>(row_idx)];
  size_t row_size = row.size();
  for (int col_idx = 0; col_idx < cols; ++col_idx) {
    output.push_back(std::cmp_less(col_idx, row_size) ? row[static_cast<size_t>(col_idx)] : 0);
  }
}

void FillMatrixBFlat(int cols_a_int, int cols_b_int, const std::vector<std::vector<int>> &matrix_b,
                     std::vector<int> &matrix_b_flat) {
  for (int index = 0; index < cols_a_int; ++index) {
    FillMatrixBRow(index, cols_b_int, matrix_b, matrix_b_flat);
  }
}

void DistributeMatrixB(int rank, int cols_a_int, int cols_b_int, const std::vector<std::vector<int>> &matrix_b,
                       std::vector<int> &matrix_b_flat) {
  if (rank == 0) {
    size_t reserve_size = static_cast<size_t>(cols_a_int) * static_cast<size_t>(cols_b_int);
    if (reserve_size <= kMaxVectorSize) {
      matrix_b_flat.reserve(reserve_size);
    }
    FillMatrixBFlat(cols_a_int, cols_b_int, matrix_b, matrix_b_flat);
  } else {
    size_t new_size = static_cast<size_t>(cols_a_int) * static_cast<size_t>(cols_b_int);
    if (new_size <= kMaxVectorSize) {
      matrix_b_flat.resize(new_size);
    }
  }

  int total_elements = cols_a_int * cols_b_int;
  // Проверка на переполнение
  if (total_elements > 0 && std::cmp_less_equal(total_elements, static_cast<int>(kMaxVectorSize)) &&
      static_cast<size_t>(total_elements) <= matrix_b_flat.size()) {
    MPI_Bcast(matrix_b_flat.data(), total_elements, MPI_INT, 0, MPI_COMM_WORLD);
  }
}

bool SafeVectorResize(std::vector<int> &vec, size_t new_size) {
  if (new_size == 0 || new_size > kMaxVectorSize) {
    return false;
  }

  try {
    vec.resize(new_size);
    return true;
  } catch (const std::bad_alloc &) {
    return false;
  }
}

bool SafeVectorResize(std::vector<int> &vec, size_t new_size, int value) {
  if (new_size == 0 || new_size > kMaxVectorSize) {
    return false;
  }

  try {
    vec.assign(new_size, value);
    return true;
  } catch (const std::bad_alloc &) {
    return false;
  }
}

bool PrepareRowDistribution(int size, int rows_a_int, int cols_a_int, std::vector<int> &rows_per_rank,
                            std::vector<int> &displacements, std::vector<int> &elements_per_rank) {
  if (rows_a_int <= 0 || cols_a_int <= 0 || size <= 0 || size > kMaxProcesses ||
      std::cmp_greater(static_cast<size_t>(rows_a_int), kMaxMatrixSize)) {
    rows_per_rank.clear();
    displacements.clear();
    elements_per_rank.clear();
    return false;
  }

  // Проверка на переполнение при умножении
  if (static_cast<size_t>(rows_a_int) > SIZE_MAX / static_cast<size_t>(cols_a_int) ||
      static_cast<size_t>(rows_a_int) * static_cast<size_t>(cols_a_int) > kMaxVectorSize) {
    rows_per_rank.clear();
    displacements.clear();
    elements_per_rank.clear();
    return false;
  }

  int rows_per_process = rows_a_int / size;
  int extra_rows = rows_a_int % size;

  rows_per_rank.clear();
  displacements.clear();
  elements_per_rank.clear();

  rows_per_rank.reserve(static_cast<size_t>(size));
  displacements.reserve(static_cast<size_t>(size));
  elements_per_rank.reserve(static_cast<size_t>(size));

  for (int index = 0; index < size; ++index) {
    int rows = rows_per_process;
    if (index < extra_rows) {
      rows += 1;
    }
    rows_per_rank.push_back(rows);
  }

  int offset = 0;
  for (int index = 0; index < size; ++index) {
    // Проверка на переполнение
    if (offset > INT_MAX / cols_a_int) {
      rows_per_rank.clear();
      displacements.clear();
      elements_per_rank.clear();
      return false;
    }
    displacements.push_back(offset * cols_a_int);
    offset += rows_per_rank[static_cast<size_t>(index)];
  }

  for (int index = 0; index < size; ++index) {
    elements_per_rank.push_back(rows_per_rank[static_cast<size_t>(index)] * cols_a_int);
  }

  return true;
}

void FlattenMatrixA(const std::vector<std::vector<int>> &matrix_a, int cols_a_int, std::vector<int> &matrix_a_flat) {
  size_t rows = matrix_a.size();

  // Проверка на переполнение перед умножением
  if (static_cast<size_t>(cols_a_int) > 0 && rows > SIZE_MAX / static_cast<size_t>(cols_a_int)) {
    return;
  }

  size_t reserve_size = rows * static_cast<size_t>(cols_a_int);

  if (reserve_size == 0 || reserve_size > kMaxVectorSize) {
    return;
  }

  matrix_a_flat.reserve(reserve_size);

  for (size_t i = 0; i < rows; ++i) {
    const auto &row = matrix_a[i];
    size_t row_size = row.size();
    size_t copy_size = std::min(row_size, static_cast<size_t>(cols_a_int));

    if (copy_size > 0) {
      // Безопасное преобразование с проверкой
      if (copy_size > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return;
      }
      auto copy_offset = static_cast<int64_t>(copy_size);
      if (row.begin() + copy_offset > row.end()) {
        return;
      }
      matrix_a_flat.insert(matrix_a_flat.end(), row.begin(), row.begin() + copy_offset);
    }

    if (std::cmp_greater(static_cast<size_t>(cols_a_int), row_size)) {
      size_t padding = static_cast<size_t>(cols_a_int) - row_size;
      matrix_a_flat.insert(matrix_a_flat.end(), padding, 0);
    }
  }
}

void ScatterMatrixA(int rank, int cols_a_int, const std::vector<std::vector<int>> &matrix_a,
                    const std::vector<int> &displacements, const std::vector<int> &elements_per_rank,
                    std::vector<int> &local_rows_flat) {
  if (rank == 0) {
    std::vector<int> matrix_a_flat;
    FlattenMatrixA(matrix_a, cols_a_int, matrix_a_flat);

    if (matrix_a_flat.size() > kMaxVectorSize) {
      if (!local_rows_flat.empty()) {
        std::ranges::fill(local_rows_flat, 0);
      }
      return;
    }

    MPI_Scatterv(matrix_a_flat.data(), elements_per_rank.data(), displacements.data(), MPI_INT, local_rows_flat.data(),
                 elements_per_rank[static_cast<size_t>(rank)], MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, elements_per_rank.data(), displacements.data(), MPI_INT, local_rows_flat.data(),
                 elements_per_rank[static_cast<size_t>(rank)], MPI_INT, 0, MPI_COMM_WORLD);
  }
}

void MultiplyRowByElement(int value, const std::vector<int> &b_row, int *result_row, int cols) {
  if (result_row == nullptr || b_row.empty() || cols <= 0) {
    return;
  }

  size_t b_size = b_row.size();
  size_t limit = std::min(b_size, static_cast<size_t>(cols));

  // Проверка, что result_row имеет достаточно места
  if (limit == 0) {
    return;
  }

  for (size_t j = 0; j < limit; ++j) {
    result_row[j] += value * b_row[j];
  }
}

void MultiplyRow(const std::vector<int> &row_a, const std::vector<std::vector<int>> &local_b, int *result_row,
                 int cols_a, int cols_b) {
  if (result_row == nullptr || row_a.empty() || local_b.empty() || cols_a <= 0 || cols_b <= 0) {
    return;
  }

  size_t row_a_size = row_a.size();
  size_t local_b_size = local_b.size();
  size_t limit = std::min(row_a_size, local_b_size);
  limit = std::min(limit, static_cast<size_t>(cols_a));

  for (size_t k = 0; k < limit; ++k) {
    // Проверка, что local_b[k] существует
    if (k >= local_b.size()) {
      break;
    }
    int aik = row_a[k];
    const auto &b_row = local_b[k];
    MultiplyRowByElement(aik, b_row, result_row, cols_b);
  }
}

void ComputeLocalMultiplication(int local_rows_count, int cols_a_int, int cols_b_int,
                                const std::vector<std::vector<int>> &local_a,
                                const std::vector<std::vector<int>> &local_b, std::vector<int> &local_result_flat) {
  if (local_result_flat.empty() || cols_b_int <= 0) {
    return;
  }

  std::ranges::fill(local_result_flat, 0);

  size_t rows = std::min(static_cast<size_t>(local_rows_count), local_a.size());
  auto cols_b = static_cast<size_t>(cols_b_int);

  for (size_t i = 0; i < rows; ++i) {
    // Проверка, что local_a[i] существует
    if (i >= local_a.size()) {
      break;
    }

    // Проверка на переполнение при вычислении offset
    if (i > SIZE_MAX / cols_b) {
      break;
    }

    size_t offset = i * cols_b;

    // Проверка границ
    if (offset > local_result_flat.size() || local_result_flat.size() - offset < cols_b) {
      break;
    }

    int *result_row = local_result_flat.data() + offset;
    MultiplyRow(local_a[i], local_b, result_row, cols_a_int, cols_b_int);
  }
}

bool PrepareGatherParameters(const std::vector<int> &rows_per_rank, int cols_b_int,
                             std::vector<int> &result_elements_per_rank, std::vector<int> &result_displacements) {
  if (rows_per_rank.empty()) {
    result_elements_per_rank.clear();
    result_displacements.clear();
    return true;
  }

  size_t size = rows_per_rank.size();
  if (size > kMaxVectorSize) {
    result_elements_per_rank.clear();
    result_displacements.clear();
    return false;
  }

  result_elements_per_rank.clear();
  result_displacements.clear();

  result_elements_per_rank.reserve(size);
  result_displacements.reserve(size);

  int displacement = 0;
  for (size_t index = 0; index < size; ++index) {
    int rows = rows_per_rank[index];
    // Проверка на переполнение
    if (rows < 0 || cols_b_int < 0 || (rows > 0 && cols_b_int > INT_MAX / rows)) {
      result_elements_per_rank.clear();
      result_displacements.clear();
      return false;
    }

    int elements = rows * cols_b_int;
    result_elements_per_rank.push_back(elements);
    result_displacements.push_back(displacement);

    // Проверка на переполнение displacement
    if (displacement > INT_MAX - elements) {
      result_elements_per_rank.clear();
      result_displacements.clear();
      return false;
    }
    displacement += elements;
  }

  return true;
}

void GatherResults(int local_rows_count, int cols_b_int, const std::vector<int> &rows_per_rank,
                   const std::vector<int> &local_result_flat, std::vector<int> &full_result_flat) {
  std::vector<int> result_elements_per_rank;
  std::vector<int> result_displacements;

  if (!PrepareGatherParameters(rows_per_rank, cols_b_int, result_elements_per_rank, result_displacements)) {
    return;
  }

  if (result_elements_per_rank.empty() || result_displacements.empty()) {
    return;
  }

  // Проверка на переполнение
  if (local_rows_count > 0 && cols_b_int > 0 && local_rows_count <= INT_MAX / cols_b_int) {
    int send_count = local_rows_count * cols_b_int;
    if (send_count > 0 && static_cast<size_t>(send_count) <= local_result_flat.size()) {
      MPI_Gatherv(local_result_flat.data(), send_count, MPI_INT, full_result_flat.data(),
                  result_elements_per_rank.data(), result_displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);
    }
  }
}

std::vector<std::vector<int>> ConvertFlatToMatrix(const std::vector<int> &flat_data, int rows, int cols) {
  if (rows <= 0 || cols <= 0 || flat_data.empty()) {
    return {};
  }

  // Проверка на переполнение
  if (rows > INT_MAX / cols) {
    return {};
  }

  // Проверка на переполнение size_t
  if (static_cast<size_t>(rows) > SIZE_MAX / static_cast<size_t>(cols)) {
    return {};
  }

  size_t expected_size = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  if (flat_data.size() < expected_size) {
    return {};
  }

  std::vector<std::vector<int>> result;
  try {
    result.resize(static_cast<size_t>(rows));
    for (auto &row : result) {
      row.resize(static_cast<size_t>(cols), 0);
    }
  } catch (const std::bad_alloc &) {
    return {};
  }

  for (int i = 0; i < rows; ++i) {
    // Проверка на переполнение при вычислении индекса
    size_t row_offset = 0;
    if (static_cast<size_t>(i) > SIZE_MAX / static_cast<size_t>(cols)) {
      return {};
    }
    row_offset = static_cast<size_t>(i) * static_cast<size_t>(cols);

    for (int j = 0; j < cols; ++j) {
      size_t idx = row_offset + static_cast<size_t>(j);
      if (idx < flat_data.size()) {
        result[static_cast<size_t>(i)][static_cast<size_t>(j)] = flat_data[idx];
      }
    }
  }

  return result;
}

std::pair<bool, std::array<int, 3>> ValidateAndGetDimensions(int rank, int size,
                                                             const std::vector<std::vector<int>> &matrix_a,
                                                             const std::vector<std::vector<int>> &matrix_b) {
  if (size <= 0 || size > kMaxProcesses) {
    return {false, {}};
  }

  if (!CheckMatricesNotEmpty(matrix_a, matrix_b)) {
    return {false, {}};
  }

  size_t rows_a = matrix_a.size();
  size_t cols_a = matrix_a[0].size();
  size_t cols_b = matrix_b[0].size();

  if (rows_a > kMaxMatrixSize || cols_a > kMaxMatrixSize || cols_b > kMaxMatrixSize) {
    return {false, {}};
  }

  std::array<int, 3> dims{0, 0, 0};
  if (rank == 0) {
    dims[0] = static_cast<int>(rows_a);
    dims[1] = static_cast<int>(cols_a);
    dims[2] = static_cast<int>(cols_b);
  }

  MPI_Bcast(dims.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
  return {true, dims};
}

bool ValidateDimensions(int rows_a_int, int cols_a_int, int cols_b_int) {
  if (rows_a_int <= 0 || cols_a_int <= 0 || cols_b_int <= 0) {
    return false;
  }

  if (std::cmp_greater(static_cast<size_t>(rows_a_int), kMaxMatrixSize) ||
      std::cmp_greater(static_cast<size_t>(cols_a_int), kMaxMatrixSize) ||
      std::cmp_greater(static_cast<size_t>(cols_b_int), kMaxMatrixSize)) {
    return false;
  }

  // Проверка на возможное переполнение при умножении
  if (static_cast<size_t>(rows_a_int) > SIZE_MAX / static_cast<size_t>(cols_a_int) ||
      static_cast<size_t>(cols_a_int) > SIZE_MAX / static_cast<size_t>(cols_b_int) ||
      static_cast<size_t>(rows_a_int) > SIZE_MAX / static_cast<size_t>(cols_b_int)) {
    return false;
  }

  if (static_cast<size_t>(rows_a_int) * static_cast<size_t>(cols_a_int) > kMaxVectorSize ||
      static_cast<size_t>(cols_a_int) * static_cast<size_t>(cols_b_int) > kMaxVectorSize ||
      static_cast<size_t>(rows_a_int) * static_cast<size_t>(cols_b_int) > kMaxVectorSize) {
    return false;
  }

  return true;
}

bool PrepareLocalData(int rank, int cols_a_int, int cols_b_int, const std::vector<std::vector<int>> &matrix_b,
                      std::vector<int> &matrix_b_flat, const std::vector<int> &elements_per_rank,
                      std::vector<int> &local_rows_flat) {
  DistributeMatrixB(rank, cols_a_int, cols_b_int, matrix_b, matrix_b_flat);

  auto rank_idx = static_cast<size_t>(rank);
  if (rank_idx >= elements_per_rank.size()) {
    return false;
  }

  int elements = elements_per_rank[rank_idx];
  if (elements <= 0) {
    return false;
  }

  try {
    local_rows_flat.reserve(static_cast<size_t>(elements));
  } catch (const std::bad_alloc &) {
    return false;
  }

  return SafeVectorResize(local_rows_flat, static_cast<size_t>(elements));
}

// Вспомогательная функция для заполнения матрицы из плоского массива
bool FillMatrixFromFlat(std::vector<std::vector<int>> &matrix, int rows, int cols, const std::vector<int> &flat_data,
                        size_t start_idx) {
  if (rows <= 0 || cols <= 0 || matrix.empty() || flat_data.empty()) {
    return false;
  }

  // Проверка на переполнение при вычислении индексов
  if (static_cast<size_t>(rows) > SIZE_MAX / static_cast<size_t>(cols)) {
    return false;
  }

  for (int i = 0; i < rows; ++i) {
    // Проверка на переполнение
    if (static_cast<size_t>(i) > SIZE_MAX / static_cast<size_t>(cols)) {
      return false;
    }

    size_t row_offset = static_cast<size_t>(i) * static_cast<size_t>(cols);
    size_t base_idx = 0;

    // Проверка на переполнение при сложении
    if (start_idx > SIZE_MAX - row_offset) {
      return false;
    }

    base_idx = start_idx + row_offset;

    for (int j = 0; j < cols; ++j) {
      size_t idx = base_idx + static_cast<size_t>(j);
      if (idx >= flat_data.size()) {
        return false;
      }
      matrix[static_cast<size_t>(i)][static_cast<size_t>(j)] = flat_data[idx];
    }
  }

  return true;
}

// Разделенная на две функции для уменьшения cognitive complexity
std::vector<std::vector<int>> CreateLocalMatrixA(int local_rows_count, int cols_a_int,
                                                 const std::vector<int> &local_rows_flat) {
  if (cols_a_int <= 0 || local_rows_count <= 0) {
    return {};
  }

  // Проверка на переполнение
  if (cols_a_int > 0 && local_rows_count > INT_MAX / cols_a_int) {
    return {};
  }

  // Проверка на переполнение size_t
  if (static_cast<size_t>(local_rows_count) > SIZE_MAX / static_cast<size_t>(cols_a_int)) {
    return {};
  }

  std::vector<std::vector<int>> local_a;
  try {
    local_a.resize(static_cast<size_t>(local_rows_count));
    for (auto &row : local_a) {
      row.resize(static_cast<size_t>(cols_a_int), 0);
    }
  } catch (const std::bad_alloc &) {
    return {};
  }

  if (!FillMatrixFromFlat(local_a, local_rows_count, cols_a_int, local_rows_flat, 0)) {
    return {};
  }

  return local_a;
}

std::vector<std::vector<int>> CreateLocalMatrixB(int cols_a_int, int cols_b_int,
                                                 const std::vector<int> &matrix_b_flat) {
  if (cols_a_int <= 0 || cols_b_int <= 0) {
    return {};
  }

  // Проверка на переполнение
  if (cols_b_int > 0 && cols_a_int > INT_MAX / cols_b_int) {
    return {};
  }

  // Проверка на переполнение size_t
  if (static_cast<size_t>(cols_a_int) > SIZE_MAX / static_cast<size_t>(cols_b_int)) {
    return {};
  }

  std::vector<std::vector<int>> local_b;
  try {
    local_b.resize(static_cast<size_t>(cols_a_int));
    for (auto &row : local_b) {
      row.resize(static_cast<size_t>(cols_b_int), 0);
    }
  } catch (const std::bad_alloc &) {
    return {};
  }

  if (!FillMatrixFromFlat(local_b, cols_a_int, cols_b_int, matrix_b_flat, 0)) {
    return {};
  }

  return local_b;
}

// Основная функция теперь просто объединяет результаты
std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> CreateLocalMatrices(
    int local_rows_count, int cols_a_int, int cols_b_int, const std::vector<int> &local_rows_flat,
    const std::vector<int> &matrix_b_flat) {
  auto local_a = CreateLocalMatrixA(local_rows_count, cols_a_int, local_rows_flat);
  if (local_a.empty()) {
    return {{}, {}};
  }

  auto local_b = CreateLocalMatrixB(cols_a_int, cols_b_int, matrix_b_flat);
  if (local_b.empty()) {
    return {{}, {}};
  }

  return {std::move(local_a), std::move(local_b)};
}

std::vector<int> ComputeLocalResult(int local_rows_count, int cols_a_int, int cols_b_int,
                                    const std::vector<std::vector<int>> &local_a,
                                    const std::vector<std::vector<int>> &local_b) {
  std::vector<int> local_result_flat;

  // Проверка на переполнение
  if (local_rows_count > 0 && cols_b_int > 0 && local_rows_count > INT_MAX / cols_b_int) {
    return {};
  }

  // Проверка на переполнение size_t
  if (static_cast<size_t>(local_rows_count) > SIZE_MAX / static_cast<size_t>(cols_b_int)) {
    return {};
  }

  size_t local_result_size = static_cast<size_t>(local_rows_count) * static_cast<size_t>(cols_b_int);

  if (local_result_size == 0 || local_result_size > kMaxVectorSize) {
    return {};
  }

  if (!SafeVectorResize(local_result_flat, local_result_size, 0)) {
    return {};
  }

  ComputeLocalMultiplication(local_rows_count, cols_a_int, cols_b_int, local_a, local_b, local_result_flat);
  return local_result_flat;
}

}  // namespace

LeonovaAStarMPI::LeonovaAStarMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool LeonovaAStarMPI::ValidateMatricesOnMaster() {
  const auto &matrix_a = std::get<0>(GetInput());
  const auto &matrix_b = std::get<1>(GetInput());

  if (!CheckMatricesNotEmpty(matrix_a, matrix_b)) {
    return false;
  }

  if (matrix_a.size() > kMaxMatrixSize || matrix_b.size() > kMaxMatrixSize) {
    return false;
  }

  size_t rows_a = matrix_a.size();
  size_t cols_a = matrix_a[0].size();

  for (size_t index = 1; index < rows_a; ++index) {
    if (index >= matrix_a.size() || matrix_a[index].size() != cols_a) {
      return false;
    }
  }

  size_t rows_b = matrix_b.size();
  size_t cols_b = matrix_b[0].size();

  for (size_t index = 1; index < rows_b; ++index) {
    if (index >= matrix_b.size() || matrix_b[index].size() != cols_b) {
      return false;
    }
  }

  return cols_a == rows_b;
}

bool LeonovaAStarMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool is_valid_local = true;
  if (rank == 0) {
    is_valid_local = ValidateMatricesOnMaster();
  }

  int is_valid_int = is_valid_local ? 1 : 0;
  MPI_Bcast(&is_valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return (is_valid_int == 1);
}

bool LeonovaAStarMPI::PreProcessingImpl() {
  return true;
}

std::vector<std::vector<int>> LeonovaAStarMPI::MultiplyMatricesMpi(const std::vector<std::vector<int>> &matrix_a,
                                                                   const std::vector<std::vector<int>> &matrix_b) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto [valid, dims] = ValidateAndGetDimensions(rank, size, matrix_a, matrix_b);
  if (!valid) {
    return {};
  }

  int rows_a_int = dims[0];
  int cols_a_int = dims[1];
  int cols_b_int = dims[2];

  if (!::leonova_a_star::ValidateDimensions(rows_a_int, cols_a_int, cols_b_int)) {
    return {};
  }

  std::vector<int> rows_per_rank;
  std::vector<int> displacements;
  std::vector<int> elements_per_rank;
  if (!PrepareRowDistribution(size, rows_a_int, cols_a_int, rows_per_rank, displacements, elements_per_rank)) {
    return {};
  }

  if (rows_per_rank.empty()) {
    return {};
  }

  int local_rows_count = rows_per_rank[static_cast<size_t>(rank)];
  if (local_rows_count < 0) {
    return {};
  }

  std::vector<int> matrix_b_flat;
  std::vector<int> local_rows_flat;
  if (!PrepareLocalData(rank, cols_a_int, cols_b_int, matrix_b, matrix_b_flat, elements_per_rank, local_rows_flat)) {
    return {};
  }

  ScatterMatrixA(rank, cols_a_int, matrix_a, displacements, elements_per_rank, local_rows_flat);
  auto [local_a, local_b] =
      CreateLocalMatrices(local_rows_count, cols_a_int, cols_b_int, local_rows_flat, matrix_b_flat);
  if (local_a.empty() || local_b.empty()) {
    return {};
  }

  std::vector<int> local_result_flat = ComputeLocalResult(local_rows_count, cols_a_int, cols_b_int, local_a, local_b);
  if (local_result_flat.empty()) {
    return {};
  }

  std::vector<int> full_result_flat;
  if (rank == 0) {
    // Проверка на переполнение size_t
    if (static_cast<size_t>(rows_a_int) > SIZE_MAX / static_cast<size_t>(cols_b_int)) {
      return {};
    }

    size_t full_result_size = static_cast<size_t>(rows_a_int) * static_cast<size_t>(cols_b_int);
    if (!SafeVectorResize(full_result_flat, full_result_size)) {
      return {};
    }
  }

  GatherResults(local_rows_count, cols_b_int, rows_per_rank, local_result_flat, full_result_flat);

  if (rank == 0) {
    return ConvertFlatToMatrix(full_result_flat, rows_a_int, cols_b_int);
  }

  return {};
}

bool LeonovaAStarMPI::ResizeOutputMatrix(int rows, int cols) {
  if (rows < 0 || cols < 0) {
    GetOutput().clear();
    return false;
  }

  if (std::cmp_greater(static_cast<size_t>(rows), kMaxMatrixSize) ||
      std::cmp_greater(static_cast<size_t>(cols), kMaxMatrixSize)) {
    GetOutput().clear();
    return false;
  }

  try {
    GetOutput().resize(static_cast<size_t>(rows));
    for (auto &row : GetOutput()) {
      row.resize(static_cast<size_t>(cols), 0);
    }
    return true;
  } catch (const std::bad_alloc &) {
    GetOutput().clear();
    return false;
  }
}

std::pair<int, int> LeonovaAStarMPI::GetResultDimensions(int rank) {
  int rows = 0;
  int cols = 0;

  if (rank == 0) {
    rows = static_cast<int>(GetOutput().size());
    if (rows > 0 && !GetOutput().empty()) {
      cols = static_cast<int>(GetOutput()[0].size());
    }
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return {rows, cols};
}

bool LeonovaAStarMPI::ValidateDimensions(int rows, int cols) {
  return rows >= 0 && cols >= 0;
}

void LeonovaAStarMPI::BroadcastFromMaster(int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    if (static_cast<size_t>(i) < GetOutput().size() && !GetOutput()[static_cast<size_t>(i)].empty() && cols > 0) {
      MPI_Bcast(GetOutput()[static_cast<size_t>(i)].data(), cols, MPI_INT, 0, MPI_COMM_WORLD);
    }
  }
}

void LeonovaAStarMPI::ReceiveFromMaster(int rows, int cols) {
  GetOutput().resize(static_cast<size_t>(rows));
  for (auto &row : GetOutput()) {
    row.resize(static_cast<size_t>(cols));
  }

  for (int i = 0; i < rows; ++i) {
    if (static_cast<size_t>(i) < GetOutput().size() && cols > 0) {
      MPI_Bcast(GetOutput()[static_cast<size_t>(i)].data(), cols, MPI_INT, 0, MPI_COMM_WORLD);
    }
  }
}

void LeonovaAStarMPI::BroadcastResult(int rank) {
  auto [result_rows, result_cols] = GetResultDimensions(rank);

  if (!ValidateDimensions(result_rows, result_cols)) {
    GetOutput().clear();
    return;
  }

  if (!ResizeOutputMatrix(result_rows, result_cols)) {
    return;
  }

  if (rank == 0) {
    BroadcastFromMaster(result_rows, result_cols);
  } else {
    ReceiveFromMaster(result_rows, result_cols);
  }
}

bool LeonovaAStarMPI::RunImpl() {
  if (!ValidationImpl()) {
    return false;
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const auto &matrix_a = std::get<0>(GetInput());
  const auto &matrix_b = std::get<1>(GetInput());

  std::vector<std::vector<int>> result = MultiplyMatricesMpi(matrix_a, matrix_b);

  if (rank == 0) {
    GetOutput() = std::move(result);
  } else {
    GetOutput().clear();
  }

  BroadcastResult(rank);
  return true;
}

bool LeonovaAStarMPI::PostProcessingImpl() {
  return true;
}

}  // namespace leonova_a_star
