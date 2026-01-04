#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "kosolapov_v_gauss_method_tape_hor_scheme/common/include/common.hpp"
#include "kosolapov_v_gauss_method_tape_hor_scheme/mpi/include/ops_mpi.hpp"
#include "kosolapov_v_gauss_method_tape_hor_scheme/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kosolapov_v_gauss_method_tape_hor_scheme {

class KosolapovVGaussMethodTapeHorSchemeFuncTestsProcesses
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string data_source =
        ppc::util::GetAbsoluteTaskPath(PPC_ID_kosolapov_v_gauss_method_tape_hor_scheme, params + ".txt");
    std::ifstream file(data_source);
    int n = 0;
    file >> n;
    InType system;
    system.matrix.resize(n, std::vector<double>(n));
    system.r_side.resize(n);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        file >> system.matrix[i][j];
      }
    }
    for (int i = 0; i < n; i++) {
      file >> system.r_side[i];
    }
    input_data_ = system;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &matrix = input_data_.matrix;
    const auto &r_side = input_data_.r_side;
    int n = static_cast<int>(matrix.size());
    double epsilon = 1e-6;
    bool solution_correct = true;

    for (int i = 0; i < n; i++) {
      double sum = 0.0;
      for (int j = 0; j < n; j++) {
        sum += matrix[i][j] * output_data[j];
      }
      double diff = std::abs(sum - r_side[i]);
      if (diff > epsilon) {
        solution_correct = false;
      }
    }
    return solution_correct;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(KosolapovVGaussMethodTapeHorSchemeFuncTestsProcesses, GaussMethodTapeHorScheme) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 9> kTestParam = {"simple_test", "diag_test", "symetric7x7", "band7x7", "small_number",
                                            "call_swap",   "lower_tri", "upper_tri",   "identity"};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KosolapovVGaussMethodTapeHorSchemeMPI, InType>(
                                               kTestParam, PPC_SETTINGS_kosolapov_v_gauss_method_tape_hor_scheme),
                                           ppc::util::AddFuncTask<KosolapovVGaussMethodTapeHorSchemeSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_kosolapov_v_gauss_method_tape_hor_scheme));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KosolapovVGaussMethodTapeHorSchemeFuncTestsProcesses::PrintFuncTestName<
    KosolapovVGaussMethodTapeHorSchemeFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(GaussMethodTapeHorScheme, KosolapovVGaussMethodTapeHorSchemeFuncTestsProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace kosolapov_v_gauss_method_tape_hor_scheme
