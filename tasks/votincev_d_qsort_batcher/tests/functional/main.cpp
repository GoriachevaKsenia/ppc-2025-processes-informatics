#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "votincev_d_qsort_batcher/common/include/common.hpp"
#include "votincev_d_qsort_batcher/mpi/include/ops_mpi.hpp"
#include "votincev_d_qsort_batcher/seq/include/ops_seq.hpp"

namespace votincev_d_qsort_batcher {

class VotincevDQsortBatcherRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  InType input_data;
  OutType expected_res;
  void SetUp() override {
    TestType param = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    std::string input_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_votincev_d_qsort_batcher, param + ".txt");

    std::ifstream file(input_path);
    if (!file.is_open()) {
      return;
    }

    size_t vect_sz = 0;
    file >> vect_sz;

    std::vector<double> vect_data(vect_sz);

    for (double &v : vect_data) {
      file >> v;
    }

    input_data = vect_data;

    // получаю ожидаемый результат
    expected_res = vect_data;
    std::ranges::sort(expected_res);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // 1,2... процессы не владеют нужным результатом
    if (output_data.size() != expected_res.size()) {
      return true;
    }
    // 0й процесс должен иметь отсортированный массив
    return output_data == expected_res;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

namespace {

TEST_P(VotincevDQsortBatcherRunFuncTestsProcesses, QsortBatcherTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {"test1", "test2", "test3", "test4", "test5",
                                             "test6", "test7", "test8", "test9", "test10"};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<VotincevDQsortBatcherMPI, InType>(kTestParam, PPC_SETTINGS_votincev_d_qsort_batcher),
    ppc::util::AddFuncTask<VotincevDQsortBatcherSEQ, InType>(kTestParam, PPC_SETTINGS_votincev_d_qsort_batcher));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    VotincevDQsortBatcherRunFuncTestsProcesses::PrintFuncTestName<VotincevDQsortBatcherRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixMultTests, VotincevDQsortBatcherRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace votincev_d_qsort_batcher
