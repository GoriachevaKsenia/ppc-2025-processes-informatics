#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "dolov_v_qsort_batcher/common/include/common.hpp"
#include "dolov_v_qsort_batcher/mpi/include/ops_mpi.hpp"
#include "dolov_v_qsort_batcher/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace dolov_v_qsort_batcher {

class DolovVQsortBatcherFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    size_t underscore_pos = params.find('_');
    if (underscore_pos == std::string::npos) {
      return;
    }

    std::string type = params.substr(0, underscore_pos);
    int size = std::stoi(params.substr(underscore_pos + 1));

    input_data_.resize(size);
    if (type == "Random") {
      for (int i = 0; i < size; ++i) {
        input_data_[i] = static_cast<double>((i * 7) % 100);
      }
    } else if (type == "Reverse") {
      for (int i = 0; i < size; ++i) {
        input_data_[i] = static_cast<double>(size - i);
      }
    } else if (type == "Sorted") {
      for (int i = 0; i < size; ++i) {
        input_data_[i] = static_cast<double>(i);
      }
    } else if (type == "Single") {
      if (size > 0) {
        input_data_[0] = 42.0;
      }
    } else if (type == "Empty") {
      input_data_.clear();
    }

    expected_res_ = input_data_;
    if (!expected_res_.empty()) {
      std::sort(expected_res_.data(), expected_res_.data() + expected_res_.size());
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty() && !expected_res_.empty()) {
      return true;
    }

    if (output_data.size() != expected_res_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_res_[i]) > 1e-9) {
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
  OutType expected_res_;
};

namespace {

TEST_P(DolovVQsortBatcherFuncTests, QsortBatcherTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {"Empty_0",   "Single_1",    "Sorted_20",
                                            "Random_50", "Reverse_100", "Random_250"};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<DolovVQsortBatcherMPI, InType>(kTestParam, PPC_SETTINGS_dolov_v_qsort_batcher),
    ppc::util::AddFuncTask<DolovVQsortBatcherSEQ, InType>(kTestParam, PPC_SETTINGS_dolov_v_qsort_batcher));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = DolovVQsortBatcherFuncTests::PrintFuncTestName<DolovVQsortBatcherFuncTests>;

INSTANTIATE_TEST_SUITE_P(QsortBatcherTests, DolovVQsortBatcherFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace dolov_v_qsort_batcher
