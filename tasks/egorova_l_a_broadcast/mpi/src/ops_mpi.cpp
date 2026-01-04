#include "egorova_l_a_broadcast/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstring>
#include <vector>

#include "egorova_l_a_broadcast/common/include/common.hpp"

namespace egorova_l_a_broadcast {

EgorovaLBroadcastMPI::EgorovaLBroadcastMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool EgorovaLBroadcastMPI::ValidationImpl() {
  return true;
}
bool EgorovaLBroadcastMPI::PreProcessingImpl() {
  return true;
}

void EgorovaLBroadcastMPI::TreeBroadcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int v_rank = (rank - root + size) % size;

  for (int mask = 1; mask < size; mask <<= 1) {
    if (v_rank < mask) {
      int v_dest = v_rank | mask;
      if (v_dest < size) {
        int dest = (v_dest + root) % size;
        MPI_Send(buffer, count, datatype, dest, 0, comm);
      }
    } else if (v_rank < (mask << 1)) {
      int v_src = v_rank & ~mask;
      int src = (v_src + root) % size;
      MPI_Recv(buffer, count, datatype, src, 0, comm, MPI_STATUS_IGNORE);
    }
  }
}

bool EgorovaLBroadcastMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  auto &in = GetInput();
  int count = 0;
  MPI_Datatype type = MPI_DATATYPE_NULL;

  if (in.type_indicator == 0) {
    count = static_cast<int>(in.data_int.size());
    type = MPI_INT;
    GetOutput().resize(count * sizeof(int));
    if (rank == in.root) {
      std::memcpy(GetOutput().data(), in.data_int.data(), GetOutput().size());
    }
  } else if (in.type_indicator == 1) {
    count = static_cast<int>(in.data_float.size());
    type = MPI_FLOAT;
    GetOutput().resize(count * sizeof(float));
    if (rank == in.root) {
      std::memcpy(GetOutput().data(), in.data_float.data(), GetOutput().size());
    }
  } else {
    count = static_cast<int>(in.data_double.size());
    type = MPI_DOUBLE;
    GetOutput().resize(count * sizeof(double));
    if (rank == in.root) {
      std::memcpy(GetOutput().data(), in.data_double.data(), GetOutput().size());
    }
  }

  TreeBroadcast(GetOutput().data(), count, type, in.root, MPI_COMM_WORLD);
  return true;
}

bool EgorovaLBroadcastMPI::PostProcessingImpl() {
  return true;
}

}  // namespace egorova_l_a_broadcast
