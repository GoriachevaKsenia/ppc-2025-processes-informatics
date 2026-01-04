#include "ovsyannikov_n_star/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "ovsyannikov_n_star/common/include/common.hpp"

namespace ovsyannikov_n_star {

constexpr int kTerm = -1;

OvsyannikovNStarMPI::OvsyannikovNStarMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool OvsyannikovNStarMPI::ValidationImpl() {
  return true;
}

bool OvsyannikovNStarMPI::PreProcessingImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank != 0) {
    int data_size = GetInput();
    dest_ = 1 + (rank % (size - 1));
    if (dest_ == rank) {
      dest_ = (dest_ + 1) % size;
    }
    if (dest_ == 0) {
      dest_ = 1;
    }

    data_.resize(data_size);
    output_.resize(data_size);

    for (int i = 0; i < data_size; ++i) {
      data_[i] = i;
    }
  }

  return true;
}

bool OvsyannikovNStarMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    int nodes = size - 1;
    std::vector<int> buf;
    MPI_Status status;

    for (int i = 0; i < nodes; ++i) {
      int dst = 0;
      MPI_Recv(&dst, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
      int src = status.MPI_SOURCE;

      MPI_Probe(src, 0, MPI_COMM_WORLD, &status);
      int buf_size = 0;
      MPI_Get_count(&status, MPI_INT, &buf_size);
      buf.resize(buf_size);
      MPI_Recv(buf.data(), buf_size, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      MPI_Send(&src, 1, MPI_INT, dst, 0, MPI_COMM_WORLD);
      MPI_Send(buf.data(), buf_size, MPI_INT, dst, 0, MPI_COMM_WORLD);
    }

    for (int i = 0; i < nodes; ++i) {
      MPI_Send(&kTerm, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
    }

  } else {
    MPI_Send(&dest_, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    int send_size = data_.empty() ? 0 : static_cast<int>(data_.size());
    MPI_Send(data_.data(), send_size, MPI_INT, 0, 0, MPI_COMM_WORLD);

    while (true) {
      int src = 0;
      MPI_Recv(&src, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (src == kTerm) {
        break;
      }

      MPI_Status status;
      MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
      int buf_size = 0;
      MPI_Get_count(&status, MPI_INT, &buf_size);
      output_.resize(buf_size);
      MPI_Recv(output_.data(), buf_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  return true;
}

bool OvsyannikovNStarMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank != 0) {
    GetOutput() = static_cast<OutType>(output_.size());
  } else {
    GetOutput() = 0;
  }

  return true;
}

}  // namespace ovsyannikov_n_star
