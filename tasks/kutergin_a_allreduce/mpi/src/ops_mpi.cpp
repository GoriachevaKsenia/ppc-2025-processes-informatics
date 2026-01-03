#include "../include/ops_mpi.hpp"

#include <mpi.h>

#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "../../common/include/common.hpp"

namespace kutergin_a_allreduce {

namespace {

void ApplyOp(void *a, const void *b, int count, MPI_Datatype datatype, MPI_Op op) {
  if (op == MPI_SUM && datatype == MPI_INT) {
    for (int i = 0; i < count; ++i) {
      reinterpret_cast<int *>(a)[i] += reinterpret_cast<const int *>(b)[i];
    }
  }
}

}  // namespace

AllreduceMPI::AllreduceMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

int Allreduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int type_size = 0;
  MPI_Type_size(datatype, &type_size);

  std::memcpy(recvbuf, sendbuf, static_cast<size_t>(count) * type_size);

  for (int mask = 1; mask < size; mask <<= 1) {
    if ((rank & mask) != 0) {
      MPI_Send(recvbuf, count, datatype, rank - mask, 0, comm);
      break;
    }

    if (rank + mask < size) {
      std::vector<uint8_t> tmp(static_cast<size_t>(count) * type_size);
      MPI_Recv(tmp.data(), count, datatype, rank + mask, 0, comm, MPI_STATUS_IGNORE);
      ApplyOp(recvbuf, tmp.data(), count, datatype, op);
    }
  }

  for (int mask = 1; mask < size; mask <<= 1) {
    if (rank < mask && rank + mask < size) {
      MPI_Send(recvbuf, count, datatype, rank + mask, 0, comm);
    } else if (rank >= mask && rank < 2 * mask) {
      MPI_Recv(recvbuf, count, datatype, rank - mask, 0, comm, MPI_STATUS_IGNORE);
    }
  }

  return MPI_SUCCESS;
}

bool AllreduceMPI::ValidationImpl() {
  return true;
}

bool AllreduceMPI::PreProcessingImpl() {
  return true;
}

bool AllreduceMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const auto &input_struct = GetInput();

  int local_sum = 0;
  if (!input_struct.elements.empty()) {
    local_sum = std::accumulate(input_struct.elements.begin(), input_struct.elements.end(), 0);
  }

  int global_sum = 0;

  Allreduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_sum;

  return true;
}

bool AllreduceMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kutergin_a_allreduce
