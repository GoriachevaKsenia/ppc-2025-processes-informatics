#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>

#include "gusev_d_sentence_count/common/include/common.hpp"
#include "gusev_d_sentence_count/mpi/include/ops_mpi.hpp"
#include "gusev_d_sentence_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace gusev_d_sentence_count {

namespace {

bool IsTerminator(char c) {
  return c == '.' || c == '!' || c == '?';
}

}  // namespace

class GusevDSentenceCountPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_;
  OutType expected_output_ = 0;

  void SetUp() override {
    std::string abs_path = ppc::util::GetAbsoluteTaskPath("gusev_d_sentence_count", "holy-bible.txt");

    std::ifstream file(abs_path);
    if (file.is_open()) {
      std::ostringstream buffer;
      buffer << file.rdbuf();
      input_data_ = buffer.str();
    } else {
      input_data_ = "Fallback text. Sentence one. Sentence two!";
    }

    size_t count = 0;
    size_t len = input_data_.length();
    for (size_t i = 0; i < len; ++i) {
      if (IsTerminator(input_data_[i])) {
        if (i + 1 == len || !IsTerminator(input_data_[i + 1])) {
          count++;
        }
      }
    }
    expected_output_ = count;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    if (rank == 0) {
      return output_data == expected_output_;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 public:
  GusevDSentenceCountPerfTests() = default;
};

TEST_P(GusevDSentenceCountPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, GusevDSentenceCountMPI, GusevDSentenceCountSEQ>(
    PPC_SETTINGS_gusev_d_sentence_count);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GusevDSentenceCountPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GusevDSentenceCountPerfTests, kGtestValues, kPerfTestName);

class GusevDSentenceCountSmallInputPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_;
  OutType expected_output_ = 0;

  void SetUp() override {
    const std::string base = "A!B.C?   Tail without terminator ";
    input_data_.clear();

    const int k_repeats = 2000;
    for (int i = 0; i < k_repeats; ++i) {
      input_data_ += base;
    }

    size_t count = 0;
    size_t len = input_data_.length();
    for (size_t i = 0; i < len; ++i) {
      if (IsTerminator(input_data_[i])) {
        if (i + 1 == len || !IsTerminator(input_data_[i + 1])) {
          count++;
        }
      }
    }
    expected_output_ = count;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    if (rank == 0) {
      return output_data == expected_output_;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 public:
  GusevDSentenceCountSmallInputPerfTests() = default;
};

TEST_P(GusevDSentenceCountSmallInputPerfTests, RunPerfModesSmallInput) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasksSmallInput =
    ppc::util::MakeAllPerfTasks<InType, GusevDSentenceCountMPI, GusevDSentenceCountSEQ>(
        PPC_SETTINGS_gusev_d_sentence_count);

const auto kGtestValuesSmallInput = ppc::util::TupleToGTestValues(kAllPerfTasksSmallInput);

const auto kPerfTestNameSmallInput = GusevDSentenceCountSmallInputPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTestsSmallInput, GusevDSentenceCountSmallInputPerfTests, kGtestValuesSmallInput,
                         kPerfTestNameSmallInput);

}  // namespace gusev_d_sentence_count
