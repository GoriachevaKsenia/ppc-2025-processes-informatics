#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "khruev_a_gauss_jordan/common/include/common.hpp"
#include "khruev_a_gauss_jordan/mpi/include/ops_mpi.hpp"
#include "khruev_a_gauss_jordan/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace khruev_a_gauss_jordan {

class KhruevGaussJordanPerfBase : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType input;

  void GenerateMatrix(int size, double diag_value) {
    input.assign(static_cast<size_t>(size), std::vector<double>(static_cast<size_t>(size + 1), 0.0));

    for (int i = 0; i < size; ++i) {
      double rhs = 0.0;
      for (int j = 0; j < size; ++j) {
        if (i == j) {
          input[i][j] = diag_value + (i % 7);
        } else {
          input[i][j] = (i + j) % 5 * 0.1;
        }
        rhs += std::abs(input[i][j]);
      }
      input[i][size] = rhs;
    }
  }

  bool CheckTestOutputData(OutType &output) final {
    return !output.empty();
  }

  InType GetTestInputData() final {
    return input;
  }

  void SetUp() override {
    GenerateMatrix(400, 20.0);
  }
};

TEST_P(KhruevGaussJordanPerfBase, RunPerf) {
  ExecuteTest(GetParam());
}

const auto kPerfTasks = ppc::util::MakeAllPerfTasks<InType, KhruevAGaussJordanMPI, KhruevAGaussJordanSEQ>(
    PPC_SETTINGS_khruev_a_gauss_jordan);

const auto kValues = ppc::util::TupleToGTestValues(kPerfTasks);

const auto kName = KhruevGaussJordanPerfBase::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(KhruevPerf, KhruevGaussJordanPerfBase, kValues, kName);

}  // namespace khruev_a_gauss_jordan
