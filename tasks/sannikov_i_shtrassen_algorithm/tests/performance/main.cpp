#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "sannikov_i_shtrassen_algorithm/common/include/common.hpp"
#include "sannikov_i_shtrassen_algorithm/mpi/include/ops_mpi.hpp"
#include "sannikov_i_shtrassen_algorithm/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sannikov_i_shtrassen_algorithm {

class SannikovIShtrassenAlgorithmPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  using Matrix = std::vector<std::vector<double>>;
  void SetUp() override {
    std::size_t n = 256;

    Matrix a(n, std::vector<double>(n, 0.0));
    Matrix b(n, std::vector<double>(n, 0.0));

    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        a[i][j] = static_cast<double>((i * 13 + j * 7) % 100);
        b[i][j] = static_cast<double>((i * 5 + j * 11) % 100);
      }
    }

    input_data_ = std::make_tuple(a, b);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return true;
    }

    const auto &a = std::get<0>(input_data_);
    const auto &b = std::get<1>(input_data_);
    std::size_t n = 0;
    n = a.size();
    if (output_data.size() != n) {
      return false;
    }

    for (const auto &row : output_data) {
      if (row.size() != n) {
        return false;
      }
    }

    Matrix ref(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t j = 0; j < n; ++j) {
          ref[i][j] += a[i][k] * b[k][j];
        }
      }
    }

    double eps = 1e-6;
    double diff = 0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        diff = ref[i][j] - output_data[i][j];
        if (diff > eps || diff < -eps) {
          return false;
        }
      }
    }

    return true;
  }

 private:
  InType input_data_;
};

TEST_P(SannikovIShtrassenAlgorithmPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SannikovIShtrassenAlgorithmMPI, SannikovIShtrassenAlgorithmSEQ>(
        PPC_SETTINGS_sannikov_i_shtrassen_algorithm);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SannikovIShtrassenAlgorithmPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SannikovIShtrassenAlgorithmPerfTests, kGtestValues, kPerfTestName);

}  // namespace sannikov_i_shtrassen_algorithm
