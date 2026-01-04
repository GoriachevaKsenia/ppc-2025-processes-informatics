#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>

#include "mityaeva_d_contrast_enhancement_histogram_stretching/common/include/common.hpp"
#include "mityaeva_d_contrast_enhancement_histogram_stretching/mpi/include/ops_mpi.hpp"
#include "mityaeva_d_contrast_enhancement_histogram_stretching/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace mityaeva_d_contrast_enhancement_histogram_stretching {

class ContrastEnhancementRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_index = std::get<0>(params);

    switch (test_index) {
      case 1: {
        input_data_ = {2, 2, 10, 50, 150, 200};
        expected_output_ = {2, 2, 0, 54, 188, 255};
        break;
      }
      case 2: {
        input_data_ = {3, 3, 100, 100, 100, 100, 100, 100, 100, 100, 100};
        expected_output_ = {3, 3, 100, 100, 100, 100, 100, 100, 100, 100, 100};
        break;
      }
      case 3: {
        input_data_ = {2, 1, 0, 255};
        expected_output_ = {2, 1, 0, 255};
        break;
      }
      case 4: {
        input_data_ = {1, 1, 128};
        expected_output_ = {1, 1, 128};
        break;
      }
      case 5: {
        input_data_ = {4, 1, 0, 85, 170, 255};
        expected_output_ = {4, 1, 0, 85, 170, 255};
        break;
      }
      case 6: {
        input_data_ = {3, 2, 100, 110, 120, 105, 115, 125};
        expected_output_ = {3, 2, 0, 102, 204, 51, 153, 255};
        break;
      }
      case 7: {
        input_data_ = {3, 3, 30, 60, 90, 120, 150, 180, 210, 240, 255};
        expected_output_ = {3, 3, 0, 34, 68, 102, 136, 170, 204, 238, 255};
        break;
      }
      case 8: {
        input_data_ = {2, 2, 0, 0, 255, 255};
        expected_output_ = {2, 2, 0, 0, 255, 255};
        break;
      }
      default:
        throw std::runtime_error("Unknown test index: " + std::to_string(test_index));
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_output_[i]) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(ContrastEnhancementRunFuncTests, ContrastEnhancement) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {
    std::make_tuple(1, "simple_2x2_contrast"),  std::make_tuple(2, "all_pixels_equal"),
    std::make_tuple(3, "already_max_contrast"), std::make_tuple(4, "single_pixel"),
    std::make_tuple(5, "sequential_values"),    std::make_tuple(6, "narrow_range"),
    std::make_tuple(7, "large_3x3_variety"),    std::make_tuple(8, "zeros_and_max")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<ContrastEnhancementMPI, InType>(
                       kTestParam, PPC_SETTINGS_mityaeva_d_contrast_enhancement_histogram_stretching),
                   ppc::util::AddFuncTask<ContrastEnhancementSEQ, InType>(
                       kTestParam, PPC_SETTINGS_mityaeva_d_contrast_enhancement_histogram_stretching));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ContrastEnhancementRunFuncTests::PrintFuncTestName<ContrastEnhancementRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(ContrastEnhancementTests, ContrastEnhancementRunFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace mityaeva_d_contrast_enhancement_histogram_stretching
