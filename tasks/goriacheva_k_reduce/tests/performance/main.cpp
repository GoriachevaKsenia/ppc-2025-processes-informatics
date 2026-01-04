#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include "goriacheva_k_reduce/common/include/common.hpp"
#include "goriacheva_k_reduce/mpi/include/ops_mpi.hpp"
#include "goriacheva_k_reduce/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace goriacheva_k_reduce {

class GoriachevaKReducePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr std::size_t kPerfSize = 10'000'000;

 protected:
  void SetUp() override {
    input_.resize(kPerfSize);

    for (std::size_t i = 0; i < kPerfSize; ++i) {
      input_[i] = static_cast<int>(i % 1000);
    }

    expected_ = OutType{0};
  }

  InType GetTestInputData() final {
    return input_;
  }

  bool CheckTestOutputData(OutType & /*output_data*/) final {
    return true;
  }

 private:
  InType input_;
  OutType expected_;
};

TEST_P(GoriachevaKReducePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

auto BuildPerfParams() {
  const auto all_tasks =
      ppc::util::MakeAllPerfTasks<InType, GoriachevaKReduceMPI, GoriachevaKReduceSEQ>(PPC_SETTINGS_goriacheva_k_reduce);

  return ppc::util::TupleToGTestValues(all_tasks);
}

const auto kPerfParams = BuildPerfParams();

std::string MakePerfName(const ::testing::TestParamInfo<GoriachevaKReducePerfTests::ParamType> &info) {
  return GoriachevaKReducePerfTests::CustomPerfTestName(info);
}

INSTANTIATE_TEST_SUITE_P(GoriachevaKReducePerfomanceTests, GoriachevaKReducePerfTests, kPerfParams, MakePerfName);

}  // namespace
}  // namespace goriacheva_k_reduce
