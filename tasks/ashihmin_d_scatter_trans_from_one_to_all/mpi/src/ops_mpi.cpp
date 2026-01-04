#include "ashihmin_d_scatter_trans_from_one_to_all/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <type_traits>
#include <vector>

#include "ashihmin_d_scatter_trans_from_one_to_all/common/include/common.hpp"

namespace ashihmin_d_scatter_trans_from_one_to_all {

template <typename T>
AshihminDScatterTransFromOneToAllMPI<T>::AshihminDScatterTransFromOneToAllMPI(const InType &in) {
  this->SetTypeOfTask(GetStaticTypeOfTask());
  this->GetInput() = in;
  this->GetOutput().clear();
}

template <typename T>
bool AshihminDScatterTransFromOneToAllMPI<T>::ValidationImpl() {
  const auto &params = this->GetInput();
  return (params.elements_per_process > 0) && (params.root >= 0) && (this->GetOutput().empty());
}

template <typename T>
bool AshihminDScatterTransFromOneToAllMPI<T>::PreProcessingImpl() {
  return true;
}

namespace {

template <typename T>
MPI_Datatype GetMPIDataType() {
  if (std::is_same_v<T, int>) {
    return MPI_INT;
  }
  if (std::is_same_v<T, float>) {
    return MPI_FLOAT;
  }
  if (std::is_same_v<T, double>) {
    return MPI_DOUBLE;
  }
  return MPI_DATATYPE_NULL;
}

inline int VirtualToRealRank(int virtual_rank, int root, int size) {
  return (virtual_rank + root) % size;
}

template <typename T>
void SendBlock(const ScatterParams &params, const std::vector<T> &local_data, int dest_virtual, int dest_real,
               int elements_per_proc, int rank, int root, MPI_Datatype mpi_type) {
  if (rank == root) {
    const int offset = dest_virtual * elements_per_proc;
    if (offset + elements_per_proc <= static_cast<int>(params.data.size())) {
      MPI_Send(params.data.data() + offset, elements_per_proc, mpi_type, dest_real, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Send(local_data.data(), elements_per_proc, mpi_type, dest_real, 0, MPI_COMM_WORLD);
  }
}

template <typename T>
void ReceiveBlock(std::vector<T> &local_data, int src_real, int elements_per_proc, MPI_Datatype mpi_type) {
  MPI_Recv(local_data.data(), elements_per_proc, mpi_type, src_real, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

}  // namespace

template <typename T>
bool AshihminDScatterTransFromOneToAllMPI<T>::RunImpl() {
  const auto &params = this->GetInput();

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Datatype mpi_type = GetMPIDataType<T>();
  const int elements_per_proc = params.elements_per_process;
  const int root = params.root % size;

  const int virtual_rank = (rank - root + size) % size;

  std::vector<T> local_data(elements_per_proc);

  int mask = 1;
  while (mask < size) {
    if ((virtual_rank & mask) == 0) {
      const int dest_virtual = virtual_rank | mask;
      if (dest_virtual < size) {
        const int dest_real = VirtualToRealRank(dest_virtual, root, size);
        SendBlock(params, local_data, dest_virtual, dest_real, elements_per_proc, rank, root, mpi_type);
      }
    } else {
      const int src_virtual = virtual_rank & ~mask;
      const int src_real = VirtualToRealRank(src_virtual, root, size);
      ReceiveBlock(local_data, src_real, elements_per_proc, mpi_type);
      break;
    }
    mask <<= 1;
  }

  if (rank == root) {
    const int offset = virtual_rank * elements_per_proc;
    if (offset + elements_per_proc <= static_cast<int>(params.data.size())) {
      std::copy(params.data.begin() + offset, params.data.begin() + offset + elements_per_proc, local_data.begin());
    }
  }

  this->GetOutput() = local_data;
  return true;
}

template <typename T>
bool AshihminDScatterTransFromOneToAllMPI<T>::PostProcessingImpl() {
  return !this->GetOutput().empty();
}

template class AshihminDScatterTransFromOneToAllMPI<int>;
template class AshihminDScatterTransFromOneToAllMPI<float>;
template class AshihminDScatterTransFromOneToAllMPI<double>;

}  // namespace ashihmin_d_scatter_trans_from_one_to_all
