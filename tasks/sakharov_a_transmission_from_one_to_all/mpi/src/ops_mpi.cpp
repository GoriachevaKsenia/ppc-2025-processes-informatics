#include "sakharov_a_transmission_from_one_to_all/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "sakharov_a_transmission_from_one_to_all/common/include/common.hpp"

namespace sakharov_a_transmission_from_one_to_all {

int MyBcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int v_rank = (rank - root + size) % size;

  if (v_rank != 0) {
    int v_parent = (v_rank - 1) / 2;
    int real_parent = (v_parent + root) % size;
    MPI_Recv(buffer, count, datatype, real_parent, 0, comm, MPI_STATUS_IGNORE);
  }

  for (int child_offset = 1; child_offset <= 2; ++child_offset) {
    int v_child = (2 * v_rank) + child_offset;
    if (v_child < size) {
      int real_child = (v_child + root) % size;
      MPI_Send(buffer, count, datatype, real_child, 0, comm);
    }
  }

  return MPI_SUCCESS;
}

namespace {

void BroadcastInts(int root, int rank, const InType &input, std::vector<int> &data_out) {
  std::vector<int> data;
  int count = 0;

  if (rank == root) {
    data = std::get<1>(input);
    count = static_cast<int>(data.size());
  }

  MyBcast(&count, 1, MPI_INT, root, MPI_COMM_WORLD);

  if (rank != root) {
    data.resize(count);
  }

  MyBcast(data.data(), count, MPI_INT, root, MPI_COMM_WORLD);
  data_out = data;
}

void BroadcastFloats(int root, int rank) {
  std::vector<float> float_data(100);
  constexpr int kFloatCount = 100;

  if (rank == root) {
    for (int i = 0; i < kFloatCount; ++i) {
      float_data[i] = static_cast<float>(i) + 0.5F;
    }
  }

  MyBcast(float_data.data(), kFloatCount, MPI_FLOAT, root, MPI_COMM_WORLD);
}

void BroadcastDoubles(int root, int rank) {
  std::vector<double> double_data(100);
  constexpr int kDoubleCount = 100;

  if (rank == root) {
    for (int i = 0; i < kDoubleCount; ++i) {
      double_data[i] = static_cast<double>(i) + 0.123;
    }
  }

  MyBcast(double_data.data(), kDoubleCount, MPI_DOUBLE, root, MPI_COMM_WORLD);
}

}  // namespace

SakharovATransmissionFromOneToAllMPI::SakharovATransmissionFromOneToAllMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SakharovATransmissionFromOneToAllMPI::ValidationImpl() {
  int root = std::get<0>(GetInput());
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  return root >= 0 && root < world_size;
}

bool SakharovATransmissionFromOneToAllMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool SakharovATransmissionFromOneToAllMPI::RunImpl() {
  const int root = std::get<0>(GetInput());
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> data;
  BroadcastInts(root, rank, GetInput(), data);

  BroadcastFloats(root, rank);
  BroadcastDoubles(root, rank);

  GetOutput() = data;
  return true;
}

bool SakharovATransmissionFromOneToAllMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sakharov_a_transmission_from_one_to_all
