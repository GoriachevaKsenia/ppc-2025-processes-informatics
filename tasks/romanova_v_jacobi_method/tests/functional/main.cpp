#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "romanova_v_jacobi_method/common/include/common.hpp"
#include "romanova_v_jacobi_method/mpi/include/ops_mpi.hpp"
#include "romanova_v_jacobi_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace romanova_v_jacobi_method {

class RomanovaVJacobiMethodFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_romanova_v_jacobi_method, params);
    std::ifstream file(abs_path + ".txt");
    if (file.is_open()) {
      int rows = 0;
      int columns = 0;
      size_t iterations = 0;
      double eps = 0.0;
      file >> rows >> columns >> iterations >> eps;
      exp_answer_ = OutType(rows);
      for (int i = 0; i < rows; i++) {
        file >> exp_answer_[i];
      }

      std::vector<double> x(rows);
      for (int i = 0; i < rows; i++) {
        file >> x[i];
      }

      std::vector<double> b(rows);
      for (int i = 0; i < rows; i++) {
        file >> b[i];
      }

      std::vector<std::vector<double>> a(rows, std::vector<double>(columns));
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
          file >> a[i][j];
        }
      }

      input_data_ = std::make_tuple(x, a, b, eps, iterations);
      eps_ = eps;

      file.close();
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != exp_answer_.size()) {
      return false;
    }
    for (size_t i = 0; i < output_data.size(); i++) {
      if (abs(output_data[i] - exp_answer_[i]) > eps_) {
        // std::cout << output_data[i] << " " << exp_answer_[i] << "\n";
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
  OutType exp_answer_;
  double eps_ = 0.0;
};

namespace {

TEST_P(RomanovaVJacobiMethodFuncTestsProcesses, Jacobi) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 11> kTestParam = {"trivialTest",          "reallySmallTest",      "simpleTest",
                                             "fastConvergeTest",     "slowConvergeTest",     "sparseMatrixTest",
                                             "alternatingSignsTest", "matrix5x5Test",        "blockMatrixTest",
                                             "matrix8x8Test",        "tridiagonalMatrixTest"};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<RomanovaVJacobiMethodMPI, InType>(kTestParam, PPC_SETTINGS_romanova_v_jacobi_method),
    ppc::util::AddFuncTask<RomanovaVJacobiMethodSEQ, InType>(kTestParam, PPC_SETTINGS_romanova_v_jacobi_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    RomanovaVJacobiMethodFuncTestsProcesses::PrintFuncTestName<RomanovaVJacobiMethodFuncTestsProcesses>;
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(Tests, RomanovaVJacobiMethodFuncTestsProcesses, kGtestValues, kPerfTestName);
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, modernize-type-traits)
}  // namespace

}  // namespace romanova_v_jacobi_method
