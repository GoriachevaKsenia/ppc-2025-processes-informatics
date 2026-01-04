#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "gusev_d_sentence_count/common/include/common.hpp"
#include "gusev_d_sentence_count/mpi/include/ops_mpi.hpp"
#include "gusev_d_sentence_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace gusev_d_sentence_count {

class GusevDSentenceCountFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<2>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    int initialized = 0;
    MPI_Initialized(&initialized);

    if (initialized == 0) {
      return output_data == expected_output_;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
      return output_data == expected_output_;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_ = 0;
};

namespace {

TEST_P(GusevDSentenceCountFuncTests, SentenceBoundaryTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 16> kTestParam = {
    TestType{"", 0, "EmptyString"},
    TestType{"No terminators here", 0, "NoTerminators"},
    TestType{"Sentence one. Sentence two! Sentence three?", 3, "ThreeSentences"},
    TestType{"Is this a question?", 1, "QuestionOnly"},
    TestType{"This is the end.", 1, "PeriodOnly"},
    TestType{"Wow!", 1, "ExclamationOnly"},
    TestType{"Wait, what?", 1, "QuestionWithComma"},
    TestType{"Wait... What?! Stop!", 3, "MixedTerminators"},
    TestType{"Really?? No!", 2, "DoubleQuestion"},
    TestType{"Test. Another one! Last?", 3, "StandardSet"},
    TestType{"Symbols? @ # $ %", 1, "WithSymbols"},
    TestType{"!Start with terminator. End.", 3, "StartWithTerminator"},
    TestType{"The end is near...", 1, "Ellipsis"},
    TestType{"A!B.C?", 3, "SingleChars"},
    TestType{R"(Only terminators???!!!)", 1, "OnlyTerminators"},
    TestType{"A..B.C", 2, "DoublePeriod"}};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<GusevDSentenceCountMPI, InType>(kTestParam, PPC_SETTINGS_gusev_d_sentence_count),
    ppc::util::AddFuncTask<GusevDSentenceCountSEQ, InType>(kTestParam, PPC_SETTINGS_gusev_d_sentence_count));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GusevDSentenceCountFuncTests::PrintFuncTestName<GusevDSentenceCountFuncTests>;

INSTANTIATE_TEST_SUITE_P(SentenceCountBoundaryTests, GusevDSentenceCountFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gusev_d_sentence_count
