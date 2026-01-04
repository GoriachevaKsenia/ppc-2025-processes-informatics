#include "savva_d_min_elem_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "savva_d_min_elem_vec/common/include/common.hpp"

namespace savva_d_min_elem_vec {

SavvaDMinElemVecMPI::SavvaDMinElemVecMPI(const InType &in) {  // эта функция не изменяется в задачах
  SetTypeOfTask(GetStaticTypeOfTask());                       // конструктор правильная постановка задачи
  GetInput() = in;  // GetInput() нужен чтобы больше не использовать сами данные in
  GetOutput() = 0;
}

bool SavvaDMinElemVecMPI::ValidationImpl() {
  return (!GetInput().empty()) && (GetOutput() == 0);
}

bool SavvaDMinElemVecMPI::PreProcessingImpl() {
  return true;
}

bool SavvaDMinElemVecMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int local_n = 0;
  int global_n = 0;
  int *local_data = nullptr;
  const int *global_data = nullptr;
  int *counts = new int[size]();
  int *displacements = new int[size]();

  // если вектор - пустой, то false
  if (rank == 0) {
    global_data = GetInput().data();
    global_n = static_cast<int>(GetInput().size());
    int elements_per_proc = global_n / size;
    int remainder = global_n % size;
    int offset = 0;

    for (int i = 0; i < size; ++i) {
      counts[i] = elements_per_proc + (i < remainder ? 1 : 0);
      displacements[i] = offset;
      offset += counts[i];
    }
  }

  MPI_Bcast(&global_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (global_n == 0) {
    delete[] counts;
    delete[] displacements;
    return false;
  }
  MPI_Bcast(counts, size, MPI_INT, 0, MPI_COMM_WORLD);
  local_data = new int[counts[rank]];
  local_n = counts[rank];
  MPI_Scatterv(global_data, counts, displacements, MPI_INT, local_data, local_n, MPI_INT, 0, MPI_COMM_WORLD);

  int local_min = std::numeric_limits<int>::max();
  for (int i = 0; i < local_n; ++i) {
    local_min = std::min(local_data[i], local_min);
  }

  int global_min = 0;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = global_min;

  delete[] counts;
  delete[] displacements;
  delete[] local_data;

  // Синхронизация
  MPI_Barrier(MPI_COMM_WORLD);

  return true;
}

bool SavvaDMinElemVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace savva_d_min_elem_vec
