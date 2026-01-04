#include "romanov_a_scatter/seq/include/ops_seq.hpp"

// #include <mpi.h>

#include <cmath>
#include <utility>
#include <vector>

#include "romanov_a_scatter/common/include/common.hpp"

namespace romanov_a_scatter {

RomanovAScatterSEQ::RomanovAScatterSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>{};
}

bool RomanovAScatterSEQ::ValidationImpl() {
  return true;
}

bool RomanovAScatterSEQ::PreProcessingImpl() {
  return true;
}

bool RomanovAScatterSEQ::RunImpl() {
  const auto &[sendbuf, sendcount, root] = GetInput();

  std::vector<int> recvbuf(sendcount);

  for (int i = 0; (i < sendcount) && std::cmp_less(i, static_cast<int>(sendbuf.size())); ++i) {
    recvbuf[i] = sendbuf[i];
  }

  GetOutput() = recvbuf;

  // Код для сравнения с MPI-версией
  // int rank = 0;
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // int num_processes = 0;
  // MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  // int root = std::get<2>(GetInput());
  // int sendcount = std::get<1>(GetInput());

  // std::vector<int> sendbuf;
  // if (rank == root) {
  //   sendbuf = std::get<0>(GetInput());
  //   sendbuf.resize(num_processes * sendcount);
  // }

  // std::vector<int> recvbuf(sendcount);

  // MPI_Scatter(sendbuf.data(), sendcount, MPI_INT, recvbuf.data(), sendcount, MPI_INT, root, MPI_COMM_WORLD);

  // GetOutput() = recvbuf;

  return true;
}

bool RomanovAScatterSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace romanov_a_scatter
