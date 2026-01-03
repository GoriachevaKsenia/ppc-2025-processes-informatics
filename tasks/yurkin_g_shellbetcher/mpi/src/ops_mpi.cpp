#include "yurkin_g_shellbetcher/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "yurkin_g_shellbetcher/common/include/common.hpp"

namespace yurkin_g_shellbetcher {
namespace {

void ShellSort(std::vector<int> &a) {
  const std::size_t n = a.size();
  if (n < 2) {
    return;
  }

  std::size_t gap = 1;
  while (gap < n / 3) {
    gap = (gap * 3) + 1;
  }

  while (gap > 0) {
    for (std::size_t i = gap; i < n; ++i) {
      const int tmp = a[i];
      std::size_t j = i;
      while (j >= gap && a[j - gap] > tmp) {
        a[j] = a[j - gap];
        j -= gap;
      }
      a[j] = tmp;
    }
    gap = (gap - 1) / 3;
  }
}

void OddEvenBatcherMergeLocal(const std::vector<int> &a, const std::vector<int> &b, std::vector<int> &out) {
  out.clear();
  out.reserve(a.size() + b.size());
  out.insert(out.end(), a.begin(), a.end());
  out.insert(out.end(), b.begin(), b.end());

  const std::size_t n = out.size();
  if (n < 2) {
    return;
  }

  for (std::size_t phase = 0; phase < n; ++phase) {
    const std::size_t start = phase % 2;
    for (std::size_t i = start; i + 1 < n; i += 2) {
      if (out[i] > out[i + 1]) {
        std::swap(out[i], out[i + 1]);
      }
    }
  }
}

int ComputeNeighbor(int rank, int phase, int size) {
  int neighbor = MPI_PROC_NULL;
  if ((phase % 2) == 0) {
    neighbor = (rank % 2 == 0) ? rank + 1 : rank - 1;
  } else {
    neighbor = (rank % 2 == 0) ? rank - 1 : rank + 1;
  }
  if (neighbor < 0 || neighbor >= size) {
    neighbor = MPI_PROC_NULL;
  }
  return neighbor;
}

void KeepBlockFromMerged(std::vector<int> &local_data, std::vector<int> &merged, int keep_count, int rank,
                         int partner) {
  const auto k = static_cast<std::size_t>(keep_count);
  if (merged.size() <= k) {
    local_data.swap(merged);
    return;
  }

  if (rank < partner) {
    merged.resize(k);
    local_data.swap(merged);
  } else {
    std::vector<int> tmp;
    tmp.assign(merged.end() - static_cast<std::vector<int>::difference_type>(k), merged.end());
    local_data.swap(tmp);
  }
}

void ExchangeAndMergeWithNeighbor(std::vector<int> &local_data, int neighbor, int rank, int keep_count) {
  if (neighbor == MPI_PROC_NULL) {
    return;
  }

  const int send_count = static_cast<int>(local_data.size());
  int recv_count = 0;

  MPI_Sendrecv(&send_count, 1, MPI_INT, neighbor, 100, &recv_count, 1, MPI_INT, neighbor, 100, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  std::vector<int> recv_buf(static_cast<std::size_t>(recv_count));

  MPI_Sendrecv(local_data.data(), send_count, MPI_INT, neighbor, 101, recv_buf.data(), recv_count, MPI_INT, neighbor,
               101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::vector<int> merged;
  OddEvenBatcherMergeLocal(local_data, recv_buf, merged);
  KeepBlockFromMerged(local_data, merged, keep_count, rank, neighbor);
}

void DoPowerOfTwoMergeStep(std::vector<int> &local_data, int rank, int size, int stages, int keep_count) {
  for (int stage = 0; stage < stages; ++stage) {
    for (int sub = stage; sub >= 0; --sub) {
      const int partner = rank ^ (1 << sub);
      if (partner < 0 || partner >= size) {
        continue;
      }

      const int send_count = static_cast<int>(local_data.size());
      int recv_count = 0;

      const int tag_count = 2000 + (stage * 16) + sub;
      const int tag_data = 3000 + (stage * 16) + sub;

      MPI_Sendrecv(&send_count, 1, MPI_INT, partner, tag_count, &recv_count, 1, MPI_INT, partner, tag_count,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::vector<int> recv_buf(static_cast<std::size_t>(recv_count));

      MPI_Sendrecv(local_data.data(), send_count, MPI_INT, partner, tag_data, recv_buf.data(), recv_count, MPI_INT,
                   partner, tag_data, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::vector<int> merged;
      OddEvenBatcherMergeLocal(local_data, recv_buf, merged);
      KeepBlockFromMerged(local_data, merged, keep_count, rank, partner);

      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
}

void DoOddEvenTransposition(std::vector<int> &local_data, int rank, int size, int keep_count) {
  for (int phase = 0; phase < size; ++phase) {
    const int neighbor = ComputeNeighbor(rank, phase, size);
    ExchangeAndMergeWithNeighbor(local_data, neighbor, rank, keep_count);
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

}  // namespace

YurkinGShellBetcherMPI::YurkinGShellBetcherMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YurkinGShellBetcherMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool YurkinGShellBetcherMPI::PreProcessingImpl() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  return initialized != 0 && GetInput() > 0;
}

bool YurkinGShellBetcherMPI::RunImpl() {
  const InType n = GetInput();
  if (n <= 0) {
    return false;
  }

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const InType base = n / size;
  const InType rem = n % size;
  const InType local_n = base + (rank < rem ? 1 : 0);

  std::vector<int> local_data;
  local_data.reserve(static_cast<std::size_t>(local_n));

  std::mt19937 rng(static_cast<unsigned int>(n));
  std::uniform_int_distribution<int> dist(0, 1000000);

  const InType offset = (base * static_cast<InType>(rank)) + std::min(static_cast<InType>(rank), rem);

  for (InType i = 0; i < offset; ++i) {
    (void)dist(rng);
  }
  for (InType i = 0; i < local_n; ++i) {
    local_data.push_back(dist(rng));
  }

  ShellSort(local_data);

  const int keep_count = static_cast<int>(local_n);

  auto is_power_of_two = [](int x) { return x > 0 && ((x & (x - 1)) == 0); };

  if (is_power_of_two(size)) {
    const int stages = static_cast<int>(std::log2(size));
    DoPowerOfTwoMergeStep(local_data, rank, size, stages, keep_count);
  } else {
    DoOddEvenTransposition(local_data, rank, size, keep_count);
  }

  std::int64_t local_checksum = 0;
  for (int v : local_data) {
    local_checksum += static_cast<std::int64_t>(v);
  }

  std::int64_t global_checksum = 0;
  MPI_Allreduce(&local_checksum, &global_checksum, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = static_cast<OutType>(global_checksum & 0x7FFFFFFF);
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool YurkinGShellBetcherMPI::PostProcessingImpl() {
  return GetOutput() > 0;
}

}  // namespace yurkin_g_shellbetcher
