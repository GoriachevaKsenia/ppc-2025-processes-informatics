#include "akimov_i_radix_sort_double_batcher_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "akimov_i_radix_sort_double_batcher_merge/common/include/common.hpp"

namespace akimov_i_radix_sort_double_batcher_merge {

AkimovIRadixBatcherSortMPI::AkimovIRadixBatcherSortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool AkimovIRadixBatcherSortMPI::ValidationImpl() {
  return true;
}
bool AkimovIRadixBatcherSortMPI::PreProcessingImpl() {
  return true;
}
bool AkimovIRadixBatcherSortMPI::PostProcessingImpl() {
  return true;
}

uint64_t AkimovIRadixBatcherSortMPI::PackDouble(double v) noexcept {
  uint64_t bits = 0ULL;
  std::memcpy(&bits, &v, sizeof(bits));
  // transform IEEE-754 double to lexicographically sortable integer
  if ((bits & (1ULL << 63)) != 0ULL) {  // negative numbers
    bits = ~bits;
  } else {  // positive numbers
    bits ^= (1ULL << 63);
  }
  return bits;
}

double AkimovIRadixBatcherSortMPI::UnpackDouble(uint64_t k) noexcept {
  if ((k & (1ULL << 63)) != 0ULL) {
    k ^= (1ULL << 63);
  } else {
    k = ~k;
  }
  double value = 0.0;
  std::memcpy(&value, &k, sizeof(value));
  return value;
}

void AkimovIRadixBatcherSortMPI::LsdRadixSort(std::vector<double> &arr) {
  const std::size_t n = arr.size();
  if (n <= 1U) {
    return;
  }

  constexpr int kBits = 8;
  constexpr int kBuckets = 1 << kBits;
  constexpr int kPasses = static_cast<int>((sizeof(uint64_t) * 8) / kBits);

  std::vector<uint64_t> keys;
  keys.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    keys[i] = PackDouble(arr[i]);
  }

  std::vector<uint64_t> tmp_keys;
  tmp_keys.resize(n);
  std::vector<double> tmp_vals;
  tmp_vals.resize(n);

  for (int pass = 0; pass < kPasses; ++pass) {
    int shift = pass * kBits;
    std::vector<std::size_t> count;
    count.assign(kBuckets + 1, 0U);

    for (std::size_t i = 0; i < n; ++i) {
      auto digit = static_cast<std::size_t>((keys[i] >> shift) & (kBuckets - 1));
      ++count[digit + 1];
    }

    for (int i = 0; i < kBuckets; ++i) {
      count[i + 1] += count[i];
    }

    for (std::size_t i = 0; i < n; ++i) {
      auto digit = static_cast<std::size_t>((keys[i] >> shift) & (kBuckets - 1));
      std::size_t pos = count[digit]++;
      tmp_keys[pos] = keys[i];
      tmp_vals[pos] = arr[i];
    }

    keys.swap(tmp_keys);
    arr.swap(tmp_vals);
  }

  for (std::size_t i = 0; i < n; ++i) {
    arr[i] = UnpackDouble(keys[i]);
  }
}

void AkimovIRadixBatcherSortMPI::CmpSwap(std::vector<double> &arr, int i, int j) noexcept {
  const int sz = static_cast<int>(arr.size());
  if (i < sz && j < sz && arr[i] > arr[j]) {
    std::swap(arr[i], arr[j]);
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
void AkimovIRadixBatcherSortMPI::OddEvenMergeRec(std::vector<double> &arr, int start, int len, int stride) {
  int step = stride * 2;
  if (step < len) {
    OddEvenMergeRec(arr, start, len, step);
    OddEvenMergeRec(arr, start + stride, len, step);
    for (int i = start + stride; i + stride < start + len; i += step) {
      CmpSwap(arr, i, i + stride);
    }
  } else {
    CmpSwap(arr, start, start + stride);
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
void AkimovIRadixBatcherSortMPI::OddEvenMergeSortRec(std::vector<double> &arr, int start, int len) {
  if (len > 1) {
    int mid = len / 2;
    OddEvenMergeSortRec(arr, start, mid);
    OddEvenMergeSortRec(arr, start + mid, len - mid);
    OddEvenMergeRec(arr, start, len, 1);
  }
}

std::vector<std::pair<int, int>> AkimovIRadixBatcherSortMPI::BuildOddEvenPhasePairs(int procs) {
  std::vector<std::pair<int, int>> comparators;
  if (procs <= 1) {
    return comparators;
  }

  int num_phases = procs * 2;
  for (int phase = 0; phase < num_phases; ++phase) {
    if ((phase % 2) == 0) {
      for (int i = 0; i + 1 < procs; i += 2) {
        comparators.emplace_back(i, i + 1);
      }
    } else {
      for (int i = 1; i + 1 < procs; i += 2) {
        comparators.emplace_back(i, i + 1);
      }
    }
  }

  return comparators;
}

void AkimovIRadixBatcherSortMPI::ExchangeAndSelect(std::vector<double> &local, int partner, int /*rank*/,
                                                   bool keep_lower) {
  int local_size = static_cast<int>(local.size());
  int partner_size = 0;

  MPI_Sendrecv(&local_size, 1, MPI_INT, partner, 0, &partner_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  std::vector<double> partner_data;
  partner_data.resize(static_cast<std::size_t>(partner_size));

  MPI_Sendrecv(local.data(), local_size, MPI_DOUBLE, partner, 1, partner_data.data(), partner_size, MPI_DOUBLE, partner,
               1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (local_size == 0 && partner_size == 0) {
    return;
  }

  std::vector<double> merged;
  merged.reserve(static_cast<std::size_t>(local_size) + static_cast<std::size_t>(partner_size));

  std::size_t i = 0U;
  std::size_t j = 0U;
  while (i < local.size() && j < partner_data.size()) {
    if (local[i] <= partner_data[j]) {
      merged.push_back(local[i++]);
    } else {
      merged.push_back(partner_data[j++]);
    }
  }
  while (i < local.size()) {
    merged.push_back(local[i++]);
  }
  while (j < partner_data.size()) {
    merged.push_back(partner_data[j++]);
  }

  if (keep_lower) {
    auto mid_it = merged.begin() + static_cast<std::vector<double>::difference_type>(local_size);
    local.assign(merged.begin(), mid_it);
  } else {
    auto start_it = merged.end() - static_cast<std::vector<double>::difference_type>(local_size);
    local.assign(start_it, merged.end());
  }
}

void AkimovIRadixBatcherSortMPI::ComputeCountsDispls(int total, int world, std::vector<int> &counts,
                                                     std::vector<int> &displs) {
  int base_count = total / world;
  int remainder = total % world;

  for (int i = 0; i < world; ++i) {
    counts[i] = base_count + ((i < remainder) ? 1 : 0);
    displs[i] = (i == 0) ? 0 : (displs[i - 1] + counts[i - 1]);
  }
}

void AkimovIRadixBatcherSortMPI::ScatterData(int total, int world, int rank, std::vector<double> &data,
                                             std::vector<int> &counts, std::vector<int> &displs,
                                             std::vector<double> &local_data) {
  ComputeCountsDispls(total, world, counts, displs);
  local_data.resize(static_cast<std::size_t>(counts[rank]));
  MPI_Scatterv(data.data(), counts.data(), displs.data(), MPI_DOUBLE, local_data.data(), counts[rank], MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
}

void AkimovIRadixBatcherSortMPI::PerformNetworkMerge(std::vector<double> &local_data, int world_size, int rank) {
  auto comparators = BuildOddEvenPhasePairs(world_size);

  for (const auto &pr : comparators) {
    int proc1 = pr.first;
    int proc2 = pr.second;
    if (rank == proc1) {
      ExchangeAndSelect(local_data, proc2, rank, true);
    } else if (rank == proc2) {
      ExchangeAndSelect(local_data, proc1, rank, false);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void AkimovIRadixBatcherSortMPI::GatherData(std::vector<double> &local_data, int total_size, int world_size, int rank,
                                            std::vector<double> &sorted_data) {
  if (rank == 0) {
    sorted_data.resize(static_cast<std::size_t>(total_size));
  }

  int new_local_size = static_cast<int>(local_data.size());
  std::vector<int> new_counts;
  new_counts.resize(static_cast<std::size_t>(world_size));
  MPI_Gather(&new_local_size, 1, MPI_INT, new_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> new_displs;
  new_displs.assign(static_cast<std::size_t>(world_size), 0);

  if (rank == 0) {
    for (int i = 1; i < world_size; ++i) {
      new_displs[i] = new_displs[i - 1] + new_counts[i - 1];
    }
  }

  MPI_Gatherv(local_data.data(), new_local_size, MPI_DOUBLE, sorted_data.data(), new_counts.data(), new_displs.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

bool AkimovIRadixBatcherSortMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int total_size = 0;
  std::vector<double> data;

  if (rank == 0) {
    data = GetInput();
    total_size = static_cast<int>(data.size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_size == 0) {
    if (rank == 0) {
      GetOutput() = {};
    }
    return true;
  }

  std::vector<int> counts;
  counts.resize(static_cast<std::size_t>(world_size));
  std::vector<int> displs;
  displs.resize(static_cast<std::size_t>(world_size));
  std::vector<double> local_data;

  ScatterData(total_size, world_size, rank, data, counts, displs, local_data);

  LsdRadixSort(local_data);

  PerformNetworkMerge(local_data, world_size, rank);

  std::vector<double> sorted_data;
  GatherData(local_data, total_size, world_size, rank, sorted_data);

  if (rank == 0) {
    GetOutput() = sorted_data;
  }

  return true;
}

}  // namespace akimov_i_radix_sort_double_batcher_merge
