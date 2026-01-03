#include "frolova_s_sum_elem_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <limits>
#include <tuple>
#include <vector>

#include "frolova_s_sum_elem_matrix/common/include/common.hpp"

namespace frolova_s_sum_elem_matrix {

FrolovaSSumElemMatrixMPI::FrolovaSSumElemMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool FrolovaSSumElemMatrixMPI::ValidationImpl() {
  const auto &in = GetInput();

  const auto &my_matrix = std::get<0>(in);
  int param_dim1 = std::get<1>(in);
  int param_dim2 = std::get<2>(in);

  return (param_dim1 > 0 && param_dim2 > 0 && static_cast<int>(my_matrix.size()) == (param_dim1 * param_dim2));
}

bool FrolovaSSumElemMatrixMPI::PreProcessingImpl() {
  return true;
}

bool FrolovaSSumElemMatrixMPI::RunImpl() {
  int process_n = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &process_n);

  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  int total_size = 0;
  std::vector<double> vect_data;

  if (proc_rank == 0) {
    vect_data = std::get<0>(GetInput());
    total_size = static_cast<int>(vect_data.size());

    if (process_n > total_size && total_size > 0) {
      process_n = total_size;
    }
  }

  MPI_Bcast(&process_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_size == 0) {
    if (proc_rank == 0) {
      GetOutput() = 0.0;
    }
    return true;
  }

  if (proc_rank == 0) {
    GetOutput() = ProcessMaster(process_n, vect_data);
  } else if (proc_rank < process_n) {
    ProcessWorker();
    GetOutput() = std::numeric_limits<double>::max();
  } else {
    GetOutput() = std::numeric_limits<double>::max();
  }

  return true;
}

bool FrolovaSSumElemMatrixMPI::PostProcessingImpl() {
  return true;
}

double FrolovaSSumElemMatrixMPI::ProcessMaster(int process_n, const std::vector<double> &vect_data) {
  const int n = static_cast<int>(vect_data.size());
  const int base = n / process_n;
  int remain = n % process_n;

  int start_id = 0;

  for (int worker = 1; worker < process_n; worker++) {
    int part_size = base + (remain > 0 ? 1 : 0);
    if (remain > 0) {
      remain--;
    }

    MPI_Send(&part_size, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
    MPI_Send(vect_data.data() + start_id, part_size, MPI_DOUBLE, worker, 0, MPI_COMM_WORLD);

    start_id += part_size;
  }

  double total_sum = 0.0;
  for (int i = start_id; i < n; ++i) {
    total_sum += vect_data[i];
  }

  for (int worker = 1; worker < process_n; worker++) {
    double worker_sum = 0.0;
    MPI_Recv(&worker_sum, 1, MPI_DOUBLE, worker, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    total_sum += worker_sum;
  }

  return total_sum;
}

void FrolovaSSumElemMatrixMPI::ProcessWorker() {
  int part_size = 0;
  MPI_Recv(&part_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::vector<double> local_data(part_size);
  MPI_Recv(local_data.data(), part_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  double local_sum = 0.0;
  for (double val : local_data) {
    local_sum += val;
  }

  MPI_Send(&local_sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
}

}  // namespace frolova_s_sum_elem_matrix
