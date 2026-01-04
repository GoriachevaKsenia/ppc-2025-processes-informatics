#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "agafonov_i_sentence_count/common/include/common.hpp"
#include "agafonov_i_sentence_count/mpi/include/ops_mpi.hpp"
#include "agafonov_i_sentence_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace agafonov_i_sentence_count {
namespace {

class SentenceCountFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  SentenceCountFuncTests() = default;

  static std::string PrintTestParam(
      const testing::TestParamInfo<ppc::util::FuncTestParam<InType, OutType, TestType>> &info) {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(info.param);
    return std::to_string(std::get<0>(params)) + "_" + std::get<1>(params);
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_case_id = std::get<0>(params);

    switch (test_case_id) {
      case 1:
        input_data_ = "Hello. Yes!";
        expected_output_ = 2;
        break;
      default:
        input_data_ = "Default.";
        expected_output_ = 1;
        break;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }
  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_{0};
};

TEST_P(SentenceCountFuncTests, RunSentenceCountFuncTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 2> kTestParams = {std::make_tuple(1, "test_1"), std::make_tuple(2, "test_2")};

const auto kMpiTasks =
    ppc::util::AddFuncTask<SentenceCountMPI, InType>(kTestParams, PPC_SETTINGS_agafonov_i_sentence_count);
const auto kSeqTasks =
    ppc::util::AddFuncTask<SentenceCountSEQ, InType>(kTestParams, PPC_SETTINGS_agafonov_i_sentence_count);

const auto kMpiValues = ppc::util::ExpandToValues(kMpiTasks);
const auto kSeqValues = ppc::util::ExpandToValues(kSeqTasks);

const auto kPerfTestName = SentenceCountFuncTests::PrintTestParam;
// NOLINTNEXTLINE
INSTANTIATE_TEST_SUITE_P(MPI, SentenceCountFuncTests, kMpiValues, kPerfTestName);

// NOLINTNEXTLINE
INSTANTIATE_TEST_SUITE_P(SEQ, SentenceCountFuncTests, kSeqValues, kPerfTestName);

}  // namespace

}  // namespace agafonov_i_sentence_count
