#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "../../common/include/common.hpp"
#include "../../mpi/include/ops_mpi.hpp"
#include "../../seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shekhirev_v_cg_method {

class ShekhirevVCGFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<2>(test_param);
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    if (rank == 0) {
      if (output_data.size() != expected_output_.size()) {
        return false;
      }
      for (size_t i = 0; i < output_data.size(); ++i) {
        if (std::abs(output_data[i] - expected_output_[i]) > 1e-4) {
          return false;
        }
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

TEST_P(ShekhirevVCGFuncTests, Test) {
  ExecuteTest(GetParam());
}

const std::vector<double> kA1 = {1.0, 0.0, 0.0, 1.0};
const std::vector<double> kB1 = {5.0, -3.0};
const std::vector<double> kX1 = {5.0, -3.0};

const std::vector<double> kA2 = {4.0, 1.0, 1.0, 3.0};
const std::vector<double> kB2 = {1.0, 2.0};
const std::vector<double> kX2 = {0.090909, 0.636363};

const std::vector<double> kA3 = {2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0};
const std::vector<double> kB3 = {2.0, 9.0, 8.0};
const std::vector<double> kX3 = {1.0, 3.0, 2.0};

const std::array<TestType, 3> kTestCases = {std::make_tuple(InputData(kA1, kB1, 2), kX1, "Identity_2x2"),
                                            std::make_tuple(InputData(kA2, kB2, 2), kX2, "SPD_System_2x2"),
                                            std::make_tuple(InputData(kA3, kB3, 3), kX3, "Diagonal_3x3")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<shekhirev_v_cg_method_seq::ConjugateGradientSeq, InType>(
                       kTestCases, PPC_SETTINGS_shekhirev_v_cg_method),
                   ppc::util::AddFuncTask<shekhirev_v_cg_method_mpi::ConjugateGradientMPI, InType>(
                       kTestCases, PPC_SETTINGS_shekhirev_v_cg_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = ShekhirevVCGFuncTests::PrintFuncTestName<ShekhirevVCGFuncTests>;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables, modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(CGTests, ShekhirevVCGFuncTests, kGtestValues, kTestName);

}  // namespace
}  // namespace shekhirev_v_cg_method
