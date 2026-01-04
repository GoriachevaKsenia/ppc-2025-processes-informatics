#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "matrix_band_multiplication/common/include/common.hpp"
#include "matrix_band_multiplication/mpi/include/ops_mpi.hpp"
#include "matrix_band_multiplication/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace matrix_band_multiplication {

struct MatrixCase {
  Matrix a;
  Matrix b;
};

using TestType = MatrixCase;

namespace {

Matrix MultiplyReference(const Matrix &a, const Matrix &b) {
  Matrix result;
  result.rows = a.rows;
  result.cols = b.cols;
  result.values.assign(result.rows * result.cols, 0.0);
  for (std::size_t i = 0; i < a.rows; ++i) {
    for (std::size_t j = 0; j < b.cols; ++j) {
      double sum = 0.0;
      for (std::size_t k = 0; k < a.cols; ++k) {
        sum += a.values[FlattenIndex(i, k, a.cols)] * b.values[FlattenIndex(k, j, b.cols)];
      }
      result.values[FlattenIndex(i, j, result.cols)] = sum;
    }
  }
  return result;
}

}  // namespace

class MatrixBandMultiplicationFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    return std::to_string(param.a.rows) + "x" + std::to_string(param.a.cols) + "x" + std::to_string(param.b.cols);
  }

 protected:
  void SetUp() override {
    const auto &param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_.a = param.a;
    input_.b = param.b;
    expected_ = MultiplyReference(input_.a, input_.b);
  }

  InType GetTestInputData() final {
    return input_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.rows != expected_.rows || output_data.cols != expected_.cols) {
      return false;
    }
    const double eps = 1e-9;
    for (std::size_t idx = 0; idx < expected_.values.size(); ++idx) {
      if (std::fabs(expected_.values[idx] - output_data.values[idx]) > eps) {
        return false;
      }
    }
    return true;
  }

 private:
  InType input_{};
  Matrix expected_{};
};

namespace {

const std::array<TestType, 3> kTestParams = {
    TestType{Matrix{.rows = 2, .cols = 2, .values = {1.0, 2.0, 3.0, 4.0}},
             Matrix{.rows = 2, .cols = 2, .values = {5.0, 6.0, 7.0, 8.0}}},
    TestType{Matrix{.rows = 2, .cols = 3, .values = {1.0, -1.0, 2.0, 0.0, 3.0, -2.0}},
             Matrix{.rows = 3, .cols = 2, .values = {4.0, 1.0, 0.0, -3.0, 5.0, 2.0}}},
    TestType{Matrix{.rows = 3, .cols = 2, .values = {1.0, 0.0, -1.0, 2.0, 3.0, -2.0}},
             Matrix{.rows = 2, .cols = 3, .values = {2.0, 1.0, 0.0, 3.0, -1.0, 4.0}}}};

const auto kTaskList = std::tuple_cat(
    ppc::util::AddFuncTask<MatrixBandMultiplicationSeq, InType>(kTestParams, PPC_SETTINGS_matrix_band_multiplication),
    ppc::util::AddFuncTask<MatrixBandMultiplicationMpi, InType>(kTestParams, PPC_SETTINGS_matrix_band_multiplication));

const auto kGTestValues = ppc::util::ExpandToValues(kTaskList);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(MatrixProduct, MatrixBandMultiplicationFuncTests, kGTestValues,
                         MatrixBandMultiplicationFuncTests::PrintFuncTestName<MatrixBandMultiplicationFuncTests>);

TEST_P(MatrixBandMultiplicationFuncTests, Runs) {
  ExecuteTest(GetParam());
}

}  // namespace

}  // namespace matrix_band_multiplication
