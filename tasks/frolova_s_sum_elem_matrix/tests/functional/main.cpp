#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "frolova_s_sum_elem_matrix/common/include/common.hpp"
#include "frolova_s_sum_elem_matrix/mpi/include/ops_mpi.hpp"
#include "frolova_s_sum_elem_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace frolova_s_sum_elem_matrix {

class FrolovaSSumElemMatrixRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string input_data_source = ppc::util::GetAbsoluteTaskPath(PPC_ID_frolova_s_sum_elem_matrix, param + ".txt");

    std::ifstream file(input_data_source);

    int data_count = 0;
    file >> data_count;

    double expect_res = 0;
    file >> expect_res;

    expected_res_ = expect_res;

    std::vector<double> vect_data;
    for (int i = 0; i < data_count; i++) {
      double elem = 0.0;
      file >> elem;
      vect_data.push_back(elem);
    }

    input_data_ = std::make_tuple(vect_data, data_count, 1);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data == std::numeric_limits<double>::max()) {
      return true;
    }
    return output_data == expected_res_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_res_ = -1;
};

namespace {

TEST_P(FrolovaSSumElemMatrixRunFuncTestsProcesses, CountSumInMatrix) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {"test1", "test2", "test3", "test4", "test5",
                                             "test6", "test7", "test8", "test9", "test10"};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<FrolovaSSumElemMatrixMPI, InType>(kTestParam, PPC_SETTINGS_frolova_s_sum_elem_matrix),
    ppc::util::AddFuncTask<FrolovaSSumElemMatrixSEQ, InType>(kTestParam, PPC_SETTINGS_frolova_s_sum_elem_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    FrolovaSSumElemMatrixRunFuncTestsProcesses::PrintFuncTestName<FrolovaSSumElemMatrixRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(CountSumInMatrixr, FrolovaSSumElemMatrixRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace frolova_s_sum_elem_matrix
