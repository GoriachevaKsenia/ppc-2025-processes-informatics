#include <gtest/gtest.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "scalar_product/common/include/common.hpp"
#include "scalar_product/mpi/include/ops_mpi.hpp"
#include "scalar_product/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace scalar_product {

class ScalarProductRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    const size_t vector_size = 100000000;

    std::vector<int> vector_a(vector_size);
    std::vector<int> vector_b(vector_size);

    for (size_t i = 0; i < vector_size; ++i) {
      vector_a[i] = static_cast<int>(i % 500) + 1;
      vector_b[i] = static_cast<int>((i * 4) % 500) + 1;
    }

    input_data_ = std::make_pair(vector_a, vector_b);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (output_data != 0);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ScalarProductRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ScalarProductMPI, ScalarProductSEQ>(PPC_SETTINGS_scalar_product);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ScalarProductRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ScalarProductRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace scalar_product
