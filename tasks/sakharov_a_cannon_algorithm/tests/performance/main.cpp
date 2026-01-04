#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <vector>

#include "sakharov_a_cannon_algorithm/common/include/common.hpp"
#include "sakharov_a_cannon_algorithm/mpi/include/ops_mpi.hpp"
#include "sakharov_a_cannon_algorithm/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace ppc::util {
template <typename InType, typename OutType>
static inline void PrintTo(const PerfTestParam<InType, OutType> &param, ::std::ostream *os) {
  *os << "PerfTestParam{"
      << "name=" << std::get<static_cast<std::size_t>(GTestParamIndex::kNameTest)>(param) << "}";
}
}  // namespace ppc::util

namespace sakharov_a_cannon_algorithm {

class SakharovACannonPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_result_;

  void SetUp() override {
    const int size = SelectSize();
    input_data_ = BuildInput(size);
    expected_result_ = NaiveMultiply(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (expected_result_.size() != output_data.size()) {
      return false;
    }
    constexpr double kEps = 1e-6;
    for (std::size_t idx = 0; idx < expected_result_.size(); ++idx) {
      if (std::abs(expected_result_[idx] - output_data[idx]) > kEps) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  static int SelectSize() {
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    constexpr int kBaseSize = 1024;
    return std::max(kBaseSize, world_size * 128);
  }

  static InType BuildInput(int size) {
    InType in{};
    in.rows_a = size;
    in.cols_a = size;
    in.rows_b = size;
    in.cols_b = size;
    auto total = static_cast<std::size_t>(size) * static_cast<std::size_t>(size);
    in.a.resize(total);
    in.b.resize(total);

    for (int ii = 0; ii < size; ++ii) {
      for (int jj = 0; jj < size; ++jj) {
        in.a[Idx(size, ii, jj)] = static_cast<double>(((ii + 1) * (jj + 2)) % 100) * 0.01;
        in.b[Idx(size, ii, jj)] = (ii == jj) ? 1.0 : static_cast<double>((jj % 3) + 1) * 0.1;
      }
    }
    return in;
  }

  static OutType NaiveMultiply(const InType &input) {
    const int m = input.rows_a;
    const int k = input.cols_a;
    const int n = input.cols_b;
    OutType result(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0);

    for (int ii = 0; ii < m; ++ii) {
      for (int kk = 0; kk < k; ++kk) {
        double a_val = input.a[Idx(k, ii, kk)];
        for (int jj = 0; jj < n; ++jj) {
          result[Idx(n, ii, jj)] += a_val * input.b[Idx(n, kk, jj)];
        }
      }
    }
    return result;
  }
};

TEST_P(SakharovACannonPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SakharovACannonAlgorithmMPI, SakharovACannonAlgorithmSEQ>(
        PPC_SETTINGS_sakharov_a_cannon_algorithm);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SakharovACannonPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SakharovACannonPerfTests, kGtestValues, kPerfTestName);

}  // namespace sakharov_a_cannon_algorithm
