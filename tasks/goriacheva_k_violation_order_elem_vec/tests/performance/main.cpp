#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include "goriacheva_k_violation_order_elem_vec/common/include/common.hpp"
#include "goriacheva_k_violation_order_elem_vec/mpi/include/ops_mpi.hpp"
#include "goriacheva_k_violation_order_elem_vec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace goriacheva_k_violation_order_elem_vec {

class GoriachevaKViolationOrderElemVecPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr std::size_t kPerfSize = 100'000'000;

 protected:
  void SetUp() override {
    input_.resize(kPerfSize);

    for (std::size_t i = 0; i < kPerfSize; ++i) {
      if (i % 3 == 0) {
        input_[i] = static_cast<int>(kPerfSize - i);
      } else {
        input_[i] = static_cast<int>(i % 1000);
      }
    }

    expected_ = 1;
  }

  InType GetTestInputData() final {
    return input_;
  }

  bool CheckTestOutputData(OutType & /*output_data*/) final {
    return true;
  }

 private:
  InType input_;
  OutType expected_{};
};

TEST_P(GoriachevaKViolationOrderElemVecPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {
auto BuildPerfParams() {
  const auto all_tasks =
      ppc::util::MakeAllPerfTasks<InType, GoriachevaKViolationOrderElemVecMPI, GoriachevaKViolationOrderElemVecSEQ>(
          PPC_SETTINGS_goriacheva_k_violation_order_elem_vec);

  return ppc::util::TupleToGTestValues(all_tasks);
}

const auto kPerfParams = BuildPerfParams();

std::string MakePerfName(const ::testing::TestParamInfo<GoriachevaKViolationOrderElemVecPerfTests::ParamType> &info) {
  return GoriachevaKViolationOrderElemVecPerfTests::CustomPerfTestName(info);
}

INSTANTIATE_TEST_SUITE_P(GoriachevaKViolationOrderElemVecPerfomanceTests, GoriachevaKViolationOrderElemVecPerfTests,
                         kPerfParams, MakePerfName);

}  // namespace
}  // namespace goriacheva_k_violation_order_elem_vec
