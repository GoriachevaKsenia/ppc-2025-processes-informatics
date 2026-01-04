#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <vector>

#include "agafonov_i_sentence_count/common/include/common.hpp"
#include "agafonov_i_sentence_count/mpi/include/ops_mpi.hpp"
#include "agafonov_i_sentence_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace agafonov_i_sentence_count {
namespace {

class SentenceCountPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    input_data_ = GenerateLongText(kTextLength_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data >= 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  const int kTextLength_ = 5000000;
  InType input_data_;

  static std::string GenerateLongText(int length) {
    std::string text;
    text.reserve(length);

    const std::vector<std::string> words = {"Hello", "world",    "this", "is",         "a",
                                            "test",  "sentence", "for",  "performance"};
    const std::vector<std::string> endings = {".", "!", "?"};

    std::size_t word_index = 0;
    std::size_t end_index = 0;

    while (text.length() < static_cast<std::size_t>(length)) {
      text += words[word_index] + " ";
      word_index = (word_index + 1) % words.size();

      if (word_index % 7 == 0 && text.length() < static_cast<std::size_t>(length) - 2) {
        text += endings[end_index] + " ";
        end_index = (end_index + 1) % endings.size();
      }
    }

    if (!text.empty() && text.back() != '.' && text.back() != '!' && text.back() != '?') {
      text += ".";
    }

    return text;
  }
};

TEST_P(SentenceCountPerfTest, RunSentenceCountPerfTests) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SentenceCountMPI, SentenceCountSEQ>(PPC_SETTINGS_agafonov_i_sentence_count);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SentenceCountPerfTest::CustomPerfTestName;

// NOLINTNEXTLINE: макрос GTest генерирует неконстантные переменные
INSTANTIATE_TEST_SUITE_P(SentenceCountPerfTests, SentenceCountPerfTest, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace agafonov_i_sentence_count
