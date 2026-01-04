#include "rozenberg_a_radix_simple_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "rozenberg_a_radix_simple_merge/common/include/common.hpp"

namespace rozenberg_a_radix_simple_merge {

void RozenbergARadixSimpleMergeMPI::LocalRadixSort(InType &data) {
  if (data.empty()) {
    return;
  }
  size_t n = data.size();
  InType buffer(n);

  for (int shift = 0; shift < 64; shift += 8) {
    std::array<size_t, 256> count = {0};

    for (double val : data) {
      uint64_t u = 0;
      std::memcpy(&u, &val, sizeof(double));

      u = (u >> 63 != 0U) ? ~u : (u ^ 0x8000000000000000);

      auto byte = static_cast<uint8_t>((u >> shift) & 0xFF);
      count.at(byte)++;
    }

    for (int i = 1; i < 256; ++i) {
      count.at(i) += count.at(i - 1);
    }

    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
      uint64_t u = 0;
      std::memcpy(&u, &data[i], 8);

      u = (u >> 63 != 0U) ? ~u : (u ^ 0x8000000000000000);

      auto byte = static_cast<uint8_t>((u >> shift) & 0xFF);
      buffer[--count.at(byte)] = data[i];
    }
    data = buffer;
  }
}

void RozenbergARadixSimpleMergeMPI::ExchangeAndMerge(InType &local_buf, int rank, int size) {
  for (int step = 1; step < size; step *= 2) {
    if (rank % (2 * step) == 0) {
      int neighbor = rank + step;
      if (neighbor < size) {
        int neighbor_size = 0;
        MPI_Recv(&neighbor_size, 1, MPI_INT, neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<double> neighbor_data(neighbor_size);
        MPI_Recv(neighbor_data.data(), neighbor_size, MPI_DOUBLE, neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        InType merged(local_buf.size() + neighbor_size);
        std::ranges::merge(local_buf, neighbor_data, merged.begin());
        local_buf = std::move(merged);
      }
    } else {
      int target = rank - step;
      int my_size = static_cast<int>(local_buf.size());
      MPI_Send(&my_size, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
      MPI_Send(local_buf.data(), my_size, MPI_DOUBLE, target, 0, MPI_COMM_WORLD);
      local_buf.clear();
      break;
    }
  }
}

RozenbergARadixSimpleMergeMPI::RozenbergARadixSimpleMergeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  InType empty;
  GetInput().swap(empty);

  for (const auto &elem : in) {
    GetInput().push_back(elem);
  }

  GetOutput().clear();
}

bool RozenbergARadixSimpleMergeMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return (!(GetInput().empty())) && (GetOutput().empty());
  }
  return true;
}

bool RozenbergARadixSimpleMergeMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetOutput().resize(GetInput().size());
  }
  return true;
}

bool RozenbergARadixSimpleMergeMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);

  int sum = 0;
  for (int i = 0; i < size; i++) {
    sendcounts[i] = (n / size) + (i < (n % size) ? 1 : 0);
    displs[i] = sum;
    sum += sendcounts[i];
  }

  int chunk = sendcounts[rank];
  InType local_buf(static_cast<size_t>(chunk));
  MPI_Scatterv(GetInput().data(), sendcounts.data(), displs.data(), MPI_DOUBLE, local_buf.data(), chunk, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  LocalRadixSort(local_buf);

  ExchangeAndMerge(local_buf, rank, size);

  if (rank == 0) {
    GetOutput() = std::move(local_buf);
  }
  return true;
}

bool RozenbergARadixSimpleMergeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace rozenberg_a_radix_simple_merge
