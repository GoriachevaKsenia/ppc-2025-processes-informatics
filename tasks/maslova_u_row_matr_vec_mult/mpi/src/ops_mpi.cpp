#include "maslova_u_row_matr_vec_mult/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "maslova_u_row_matr_vec_mult/common/include/common.hpp"

namespace maslova_u_row_matr_vec_mult {

MaslovaURowMatrVecMultMPI::MaslovaURowMatrVecMultMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MaslovaURowMatrVecMultMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int flag = 0;
  if (rank == 0) {
    if (GetInput().first.cols != GetInput().second.size()) {
      flag = 1;
    }
  }
  MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return flag == 0;
}

bool MaslovaURowMatrVecMultMPI::PreProcessingImpl() {
  return true;
}

bool MaslovaURowMatrVecMultMPI::RunImpl() {
  int rank = 0;
  int proc_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_size);

  uint64_t rows = 0;
  uint64_t cols = 0;
  if (rank == 0) {
    rows = static_cast<uint64_t>(GetInput().first.rows);
    cols = static_cast<uint64_t>(GetInput().first.cols);
  }
  MPI_Bcast(&rows, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);  // рассылаем размеры матрицы

  MPI_Datatype row_type = MPI_DATATYPE_NULL;  // для удобства создадим новый тип строка
  MPI_Type_contiguous(static_cast<int>(cols), MPI_DOUBLE, &row_type);
  MPI_Type_commit(&row_type);

  std::vector<double> vec(cols);
  if (rank == 0) {
    vec = GetInput().second;
  }
  MPI_Bcast(vec.data(), static_cast<int>(cols), MPI_DOUBLE, 0, MPI_COMM_WORLD);  // рассылаем вектор

  std::vector<int> row_cnt(proc_size);    // сколько срок получит каждый процесс
  std::vector<int> row_start(proc_size);  // с какой строки начинается порция процесса

  int q = static_cast<int>(rows) / proc_size;
  int r = static_cast<int>(rows) % proc_size;

  for (int i = 0; i < proc_size; ++i) {
    row_cnt[i] = q + (i < r ? 1 : 0);
    row_start[i] = (i == 0) ? 0 : row_start[i - 1] + row_cnt[i - 1];
  }

  int local_rows = row_cnt[rank];
  std::vector<double> local_matrix(local_rows * cols);

  MPI_Scatterv(rank == 0 ? GetInput().first.data.data() : nullptr, row_cnt.data(), row_start.data(), row_type,
               local_matrix.data(), static_cast<int>(local_rows * cols), MPI_DOUBLE, 0,
               MPI_COMM_WORLD);  // рассылаем строки матрицы

  std::vector<double> local_res(local_rows, 0.0);
  for (int i = 0; i < local_rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      local_res[i] += local_matrix[(i * cols) + j] * vec[j];  // вычисляем локально
    }
  }

  if (rank == 0) {
    GetOutput().resize(rows);
  }

  MPI_Gatherv(local_res.data(), local_rows, MPI_DOUBLE, rank == 0 ? GetOutput().data() : nullptr, row_cnt.data(),
              row_start.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);  // собираем результаты

  MPI_Type_free(&row_type);  // очищаем созданный ип

  return true;
}

bool MaslovaURowMatrVecMultMPI::PostProcessingImpl() {
  return true;
}

}  // namespace maslova_u_row_matr_vec_mult
