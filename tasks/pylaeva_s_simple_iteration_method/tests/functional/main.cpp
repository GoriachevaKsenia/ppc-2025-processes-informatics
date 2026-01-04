#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>    // Для std::ifstream
#include <stdexcept>  // Для std::runtime_error
#include <string>
#include <tuple>
#include <vector>

#include "pylaeva_s_simple_iteration_method/common/include/common.hpp"
#include "pylaeva_s_simple_iteration_method/mpi/include/ops_mpi.hpp"
#include "pylaeva_s_simple_iteration_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace pylaeva_s_simple_iteration_method {

class PylaevaSSimpleIterationMethodFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  PylaevaSSimpleIterationMethodFuncTests() = default;

  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string filename = ppc::util::GetAbsoluteTaskPath(PPC_ID_pylaeva_s_simple_iteration_method, param + ".txt");

    std::ifstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file: " + filename);
    }

    size_t n = 0;

    file >> n;

    std::vector<double> matrix_a(n * n);
    std::vector<double> vector_b(n);
    std::vector<double> expected_result(n);

    // Читаем матрицу A
    for (size_t i = 0; i < n * n; ++i) {
      if (!(file >> matrix_a[i])) {
        throw std::runtime_error("Failed to read matrix A element " + std::to_string(i));
      }
    }

    // Читаем вектор b
    for (size_t i = 0; i < n; ++i) {
      if (!(file >> vector_b[i])) {
        throw std::runtime_error("Failed to read vector b element " + std::to_string(i));
      }
    }

    // Читаем ожидаемый результат
    for (size_t i = 0; i < n; ++i) {
      if (!(file >> expected_result[i])) {
        throw std::runtime_error("Failed to read expected result element " + std::to_string(i));
      }
    }

    input_data_ = std::make_tuple(n, matrix_a, vector_b);
    expected_data_ = expected_result;

    file.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    constexpr double kEps = 1e-6;
    if (expected_data_.size() != output_data.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); i++) {
      if (std::fabs(expected_data_[i] - output_data[i]) > kEps) {
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
  OutType expected_data_;
};

TEST_P(PylaevaSSimpleIterationMethodFuncTests, SimpleIterationsTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 11> kTestParam = {"Simple_1x1",
                                             "Identity_matrix_2x2",
                                             "Identity_matrix_5x5",
                                             "Identity_matrix_3x3",
                                             "DiagDominanceRandom_2x2",
                                             "DiagDominanceRandom_3x3",
                                             "DiagDominanceRandom_5x5",
                                             "Negative_1x1",
                                             "Negative_2x2",
                                             "Negative_3x3",
                                             "Negative_5x5"};
const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<PylaevaSSimpleIterationMethodMPI, InType>(
                                               kTestParam, PPC_SETTINGS_pylaeva_s_simple_iteration_method),
                                           ppc::util::AddFuncTask<PylaevaSSimpleIterationMethodSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_pylaeva_s_simple_iteration_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    PylaevaSSimpleIterationMethodFuncTests::PrintFuncTestName<PylaevaSSimpleIterationMethodFuncTests>;

INSTANTIATE_TEST_SUITE_P(SimpleIterationMethodTests, PylaevaSSimpleIterationMethodFuncTests, kGtestValues,
                         kPerfTestName);

}  // namespace pylaeva_s_simple_iteration_method
