#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "goriacheva_k_strassen_algorithm/common/include/common.hpp"
#include "goriacheva_k_strassen_algorithm/mpi/include/ops_mpi.hpp"
#include "goriacheva_k_strassen_algorithm/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace goriacheva_k_strassen_algorithm {

class GoriachevaKStrassenAlgorithmPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr std::size_t kMatrixSize = 256;

 protected:
  void SetUp() override {
    input_.a.assign(kMatrixSize, std::vector<double>(kMatrixSize, 1.0));
    input_.b.assign(kMatrixSize, std::vector<double>(kMatrixSize, 2.0));
  }

  InType GetTestInputData() final {
    return input_;
  }

  bool CheckTestOutputData(OutType & /*output_data*/) final {
    return true;
  }

 private:
  InType input_;
};

TEST_P(GoriachevaKStrassenAlgorithmPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

auto BuildPerfParams() {
  const auto all_tasks =
      ppc::util::MakeAllPerfTasks<InType, GoriachevaKStrassenAlgorithmMPI, GoriachevaKStrassenAlgorithmSEQ>(
          PPC_SETTINGS_goriacheva_k_strassen_algorithm);

  return ppc::util::TupleToGTestValues(all_tasks);
}

const auto kPerfParams = BuildPerfParams();

std::string MakePerfName(const ::testing::TestParamInfo<GoriachevaKStrassenAlgorithmPerfTests::ParamType> &info) {
  return GoriachevaKStrassenAlgorithmPerfTests::CustomPerfTestName(info);
}

INSTANTIATE_TEST_SUITE_P(GoriachevaKStrassenAlgorithmPerfomanceTests, GoriachevaKStrassenAlgorithmPerfTests,
                         kPerfParams, MakePerfName);

}  // namespace
}  // namespace goriacheva_k_strassen_algorithm
