#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "spichek_d_jacobi/common/include/common.hpp"
#include "spichek_d_jacobi/mpi/include/ops_mpi.hpp"
#include "spichek_d_jacobi/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace spichek_d_jacobi {

class JacobiPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_;

  void SetUp() override {
    const size_t n = 200;

    std::vector<std::vector<double>> a(n, std::vector<double>(n, 0.0));
    std::vector<double> b(n, 1.0);

    for (size_t i = 0; i < n; ++i) {
      a[i][i] = 4.0;
      if (i > 0) {
        a[i][i - 1] = 1.0;
      }
      if (i + 1 < n) {
        a[i][i + 1] = 1.0;
      }
    }

    input_data_ = {a, b, 1e-6, 300};
  }

  InType GetTestInputData() override {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output) override {
    return !output.empty();
  }
};

TEST_P(JacobiPerfTests, RunPerf) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    JacobiPerf, JacobiPerfTests,
    ppc::util::TupleToGTestValues(
        ppc::util::MakeAllPerfTasks<InType, SpichekDJacobiMPI, SpichekDJacobiSEQ>(PPC_SETTINGS_spichek_d_jacobi)));

}  // namespace spichek_d_jacobi
