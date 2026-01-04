#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <tuple>

#include "romanov_a_crs_product/common/include/common.hpp"
#include "romanov_a_crs_product/mpi/include/ops_mpi.hpp"
#include "romanov_a_crs_product/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace romanov_a_crs_product {

class RomanovACRSProductPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const uint64_t n_ = static_cast<uint64_t>(1000);
  const double density_ = 0.1;

  InType input_data_;
  OutType expected_;

  void SetUp() override {
    CRS a(n_);
    CRS b(n_);
    a.FillRandom(density_);
    b.FillRandom(density_);

    input_data_ = std::make_tuple(a, b);
    expected_ = a * b;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(RomanovACRSProductPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, RomanovACRSProductMPI, RomanovACRSProductSEQ>(
    PPC_SETTINGS_romanov_a_crs_product);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RomanovACRSProductPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RomanovACRSProductPerfTests, kGtestValues, kPerfTestName);

}  // namespace romanov_a_crs_product
