#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>

#include "mityaeva_d_striped_horizontal_matrix_vector/common/include/common.hpp"
#include "mityaeva_d_striped_horizontal_matrix_vector/mpi/include/ops_mpi.hpp"
#include "mityaeva_d_striped_horizontal_matrix_vector/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace mityaeva_d_striped_horizontal_matrix_vector {

class StripedHorizontalMatrixVectorRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
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
        input_data_ = {2, 2, 2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        expected_output_ = {2.0, 17.0, 39.0};
        break;
      }
      case 2: {
        input_data_ = {1, 3, 3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        expected_output_ = {1.0, 32.0};
        break;
      }
      case 3: {
        input_data_ = {3, 1, 1, 1.0, 2.0, 3.0, 4.0};
        expected_output_ = {3.0, 4.0, 8.0, 12.0};
        break;
      }
      case 4: {
        input_data_ = {2, 2, 2, -1.0, 2.0, 3.0, -4.0, 5.0, -6.0};
        expected_output_ = {2.0, -17.0, 39.0};
        break;
      }
      case 5: {
        input_data_ = {3, 3, 3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0};
        expected_output_ = {3.0, 1.0, 2.0, 3.0};
        break;
      }
      case 6: {
        input_data_ = {4,    4,    4,    1.0,  2.0,  3.0,  4.0,  5.0, 6.0, 7.0,  8.0, 9.0,
                       10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 1.0, 0.0, -1.0, 2.0};
        expected_output_ = {4.0, 6.0, 14.0, 22.0, 30.0};
        break;
      }
      case 7: {
        input_data_ = {2, 3, 3, 1.0, 0.5, 2.0, 3.0, 4.0, 1.5, 2.0, 4.0, 1.0};
        expected_output_ = {2.0, 6.0, 23.5};
        break;
      }
      case 8: {
        input_data_ = {2, 3, 3, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 3.0, 4.0, 5.0};
        expected_output_ = {2.0, 0.0, 13.0};
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

    const double epsilon = 1e-10;
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_output_[i]) > epsilon) {
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

TEST_P(StripedHorizontalMatrixVectorRunFuncTests, StripedHorizontalMatrixVector) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {
    std::make_tuple(1, "simple_2x2_matrix"),    std::make_tuple(2, "single_row_matrix"),
    std::make_tuple(3, "single_column_matrix"), std::make_tuple(4, "negative_numbers"),
    std::make_tuple(5, "identity_matrix"),      std::make_tuple(6, "large_4x4_matrix"),
    std::make_tuple(7, "fractional_numbers"),   std::make_tuple(8, "zero_elements")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<StripedHorizontalMatrixVectorMPI, InType>(
                                               kTestParam, PPC_SETTINGS_mityaeva_d_striped_horizontal_matrix_vector),
                                           ppc::util::AddFuncTask<StripedHorizontalMatrixVectorSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_mityaeva_d_striped_horizontal_matrix_vector));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    StripedHorizontalMatrixVectorRunFuncTests::PrintFuncTestName<StripedHorizontalMatrixVectorRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(MatrixVectorMultiplicationTests, StripedHorizontalMatrixVectorRunFuncTests, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace mityaeva_d_striped_horizontal_matrix_vector
