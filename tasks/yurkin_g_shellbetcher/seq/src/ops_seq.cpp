#include "yurkin_g_shellbetcher/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "yurkin_g_shellbetcher/common/include/common.hpp"

namespace yurkin_g_shellbetcher {
namespace {

void ShellSort(std::vector<int> &data) {
  const std::size_t n = data.size();
  if (n < 2) {
    return;
  }

  std::size_t gap = 1;
  while (gap < n / 3) {
    gap = (gap * 3) + 1;
  }

  while (gap > 0) {
    for (std::size_t i = gap; i < n; ++i) {
      int tmp = data[i];
      std::size_t j = i;
      while (j >= gap && data[j - gap] > tmp) {
        data[j] = data[j - gap];
        j -= gap;
      }
      data[j] = tmp;
    }
    gap = (gap - 1) / 3;
  }
}

void OddEvenBatcherMerge(const std::vector<int> &a, const std::vector<int> &b, std::vector<int> &out) {
  out.resize(a.size() + b.size());

  std::ranges::merge(a, b, out.begin());

  for (int phase = 0; phase < 2; ++phase) {
    auto start = static_cast<std::size_t>(phase);
    for (std::size_t i = start; i + 1 < out.size(); i += 2) {
      if (out[i] > out[i + 1]) {
        std::swap(out[i], out[i + 1]);
      }
    }
  }
}

}  // namespace

YurkinGShellBetcherSEQ::YurkinGShellBetcherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YurkinGShellBetcherSEQ::ValidationImpl() {
  return GetInput() > 0;
}

bool YurkinGShellBetcherSEQ::PreProcessingImpl() {
  return true;
}

bool YurkinGShellBetcherSEQ::RunImpl() {
  const InType n = GetInput();
  if (n <= 0) {
    return false;
  }

  std::vector<int> data(static_cast<std::size_t>(n));

  std::mt19937 gen(static_cast<unsigned int>(n));
  std::uniform_int_distribution<int> dist(0, 1'000'000);
  for (int &v : data) {
    v = dist(gen);
  }

  std::vector<int> expected = data;
  std::ranges::sort(expected);

  ShellSort(data);

  std::vector<int> merged;
  OddEvenBatcherMerge(data, {}, merged);
  data.swap(merged);

  if (!std::ranges::is_sorted(data)) {
    return false;
  }

  if (data != expected) {
    return false;
  }

  std::int64_t checksum = 0;
  for (int v : data) {
    checksum += static_cast<std::int64_t>(v);
  }

  GetOutput() = static_cast<OutType>(checksum & 0x7FFFFFFF);
  return true;
}

bool YurkinGShellBetcherSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace yurkin_g_shellbetcher
