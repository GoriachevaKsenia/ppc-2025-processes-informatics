#include "shkrebko_m_hypercube/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "shkrebko_m_hypercube/common/include/common.hpp"

namespace shkrebko_m_hypercube {

namespace {

int CalcNextRank(int current_rank, int destination_rank, int world_size) {
  if (current_rank == destination_rank) {
    return current_rank;
  }

  int ndims = static_cast<int>(log2(world_size));

  for (int dim = 0; dim < ndims; dim++) {
    int current_bit = (current_rank >> dim) & 1;
    int dest_bit = (destination_rank >> dim) & 1;

    if (current_bit != dest_bit) {
      int next_rank = current_rank ^ (1 << dim);
      return next_rank;
    }
  }

  return current_rank;
}

void SendHypercubeData(const HypercubeData &data, int dest_rank, int tag) {
  MPI_Send(&data.value, 1, MPI_INT, dest_rank, tag, MPI_COMM_WORLD);
  MPI_Send(&data.destination, 1, MPI_INT, dest_rank, tag + 1, MPI_COMM_WORLD);
  MPI_Send(&data.finish, 1, MPI_C_BOOL, dest_rank, tag + 2, MPI_COMM_WORLD);
  int path_size = static_cast<int>(data.path.size());
  MPI_Send(&path_size, 1, MPI_INT, dest_rank, tag + 3, MPI_COMM_WORLD);
  if (path_size > 0) {
    MPI_Send(data.path.data(), path_size, MPI_INT, dest_rank, tag + 4, MPI_COMM_WORLD);
  }
}

void RecvHypercubeData(HypercubeData &data, int src_rank, int tag, MPI_Status *status = MPI_STATUS_IGNORE) {
  MPI_Recv(&data.value, 1, MPI_INT, src_rank, tag, MPI_COMM_WORLD, status);
  MPI_Recv(&data.destination, 1, MPI_INT, src_rank, tag + 1, MPI_COMM_WORLD, status);
  MPI_Recv(&data.finish, 1, MPI_C_BOOL, src_rank, tag + 2, MPI_COMM_WORLD, status);
  int path_size = 0;
  MPI_Recv(&path_size, 1, MPI_INT, src_rank, tag + 3, MPI_COMM_WORLD, status);
  data.path.resize(path_size);
  if (path_size > 0) {
    MPI_Recv(data.path.data(), path_size, MPI_INT, src_rank, tag + 4, MPI_COMM_WORLD, status);
  }
}

void ArtificialDelay() {
  for (int i = 0; i < 10000; i++) {
    volatile int dummy = 0;
    for (int j = 0; j < 10000; j++) {
      dummy += i * j;
    }
  }
}

HypercubeData HandleSingleProcessCase(int rank, const HypercubeData &init_data) {
  HypercubeData result = init_data;
  result.path.push_back(rank);
  result.finish = true;
  return result;
}

HypercubeData HandleRootProcess(int rank, int world_size, const HypercubeData &init_data) {
  HypercubeData local_data = init_data;
  local_data.path.push_back(rank);
  local_data.finish = false;

  if (local_data.destination != rank) {
    int next_rank = CalcNextRank(rank, local_data.destination, world_size);
    SendHypercubeData(local_data, next_rank, 0);
    RecvHypercubeData(local_data, MPI_ANY_SOURCE, 1);
  } else {
    local_data.finish = true;
  }

  for (int i = 1; i < world_size; i++) {
    SendHypercubeData(local_data, i, 0);
  }

  return local_data;
}

HypercubeData HandleNonRootProcess(int rank, int world_size) {
  HypercubeData local_data;
  MPI_Status status;
  RecvHypercubeData(local_data, MPI_ANY_SOURCE, 0, &status);

  if (!local_data.finish) {
    local_data.path.push_back(rank);

    if (rank == local_data.destination) {
      local_data.finish = true;
      SendHypercubeData(local_data, 0, 1);
      RecvHypercubeData(local_data, 0, 0, &status);
    } else {
      int next_rank = CalcNextRank(rank, local_data.destination, world_size);
      SendHypercubeData(local_data, next_rank, 0);
      RecvHypercubeData(local_data, 0, 0, &status);
    }
  }

  return local_data;
}

}  // namespace

ShkrebkoMHypercubeMPI::ShkrebkoMHypercubeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = HypercubeData();
}

bool ShkrebkoMHypercubeMPI::ValidationImpl() {
  int world_size = 0;
  int world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  bool is_valid = true;

  if (world_rank == 0) {
    if (GetInput().size() < 2) {
      is_valid = false;
    }

    if (is_valid) {
      int destination = GetInput()[1];
      if (destination < 0 || destination >= world_size) {
        is_valid = false;
      }

      if (GetInput()[0] <= 0) {
        is_valid = false;
      }
    }
  }

  int valid_int = is_valid ? 1 : 0;
  MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return valid_int == 1;
}

bool ShkrebkoMHypercubeMPI::PreProcessingImpl() {
  if (GetInput().size() >= 2) {
    GetOutput().value = GetInput()[0];
    GetOutput().destination = GetInput()[1];
  } else {
    GetOutput().value = 1;
    GetOutput().destination = 0;
  }
  return true;
}

bool ShkrebkoMHypercubeMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  ArtificialDelay();

  if (world_size == 1) {
    GetOutput() = HandleSingleProcessCase(world_rank, GetOutput());
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  HypercubeData result;

  if (world_rank == 0) {
    result = HandleRootProcess(world_rank, world_size, GetOutput());
  } else {
    result = HandleNonRootProcess(world_rank, world_size);
  }

  GetOutput() = result;
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool ShkrebkoMHypercubeMPI::PostProcessingImpl() {
  for (std::size_t i = 1; i < GetOutput().path.size(); i++) {
    int prev = GetOutput().path[i - 1];
    int curr = GetOutput().path[i];

    int diff = prev ^ curr;
    if ((diff & (diff - 1)) != 0) {
      return false;
    }
  }

  return true;
}

}  // namespace shkrebko_m_hypercube
