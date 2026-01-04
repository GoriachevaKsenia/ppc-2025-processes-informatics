#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <tuple>

#include "util/include/func_test_util.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"
#include "volkov_a_odd_even_transposition/common/include/common.hpp"
#include "volkov_a_odd_even_transposition/mpi/include/ops_mpi.hpp"
#include "volkov_a_odd_even_transposition/seq/include/ops_seq.hpp"

namespace volkov_a_odd_even_transposition {

enum class TestType : std::uint8_t { kRandom, kSorted, kReverse, kDuplicates, kSingle };

using TestParam = std::tuple<size_t, TestType>;

class OddEvenSortFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestParam> {
 public:
  static std::string PrintTestParam(const TestParam &param) {
    auto size = std::get<0>(param);
    auto type = std::get<1>(param);
    std::string type_str;
    switch (type) {
      case TestType::kRandom:
        type_str = "Random";
        break;
      case TestType::kSorted:
        type_str = "Sorted";
        break;
      case TestType::kReverse:
        type_str = "Reverse";
        break;
      case TestType::kDuplicates:
        type_str = "Duplicates";
        break;
      case TestType::kSingle:
        type_str = "Single";
        break;
    }
    return type_str + "_" + std::to_string(size);
  }

 protected:
  void SetUp() override {
    input_.clear();
    expected_.clear();

    if (CheckRank()) {
      auto param_tuple = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
      size_t size = std::get<0>(param_tuple);
      TestType type = std::get<1>(param_tuple);

      input_.resize(size);

      if (type == TestType::kSingle) {
        input_.assign(1, 42);
      } else {
        std::random_device rd;
        std::mt19937 gen(rd());

        if (type == TestType::kDuplicates) {
          std::uniform_int_distribution<int> dist(1, 10);
          for (auto &x : input_) {
            x = dist(gen);
          }
        } else {
          std::uniform_int_distribution<int> dist(-10000, 10000);
          for (auto &x : input_) {
            x = dist(gen);
          }
        }

        if (type == TestType::kSorted) {
          std::ranges::sort(input_);
        } else if (type == TestType::kReverse) {
          std::ranges::sort(input_, std::greater<>());
        }
      }

      expected_ = input_;
      std::ranges::sort(expected_);
    }
  }

  bool CheckTestOutputData(OutType &result) override {
    if (CheckRank()) {
      return result == expected_;
    }
    return true;
  }

  InType GetTestInputData() override {
    return input_;
  }

 private:
  InType input_;
  OutType expected_;

  static bool CheckRank() {
    return !ppc::util::IsUnderMpirun() || ppc::util::GetMPIRank() == 0;
  }
};

TEST_P(OddEvenSortFuncTest, CheckSortCorrectness) {
  ExecuteTest(GetParam());
}

const std::array<TestParam, 6> kTestParams = {{{100, TestType::kRandom},
                                               {100, TestType::kSorted},
                                               {100, TestType::kReverse},
                                               {100, TestType::kDuplicates},
                                               {50, TestType::kRandom},
                                               {1, TestType::kSingle}}};

const auto kFuncTasks = std::tuple_cat(
    ppc::util::AddFuncTask<OddEvenSortMPI, InType>(kTestParams, PPC_SETTINGS_volkov_a_odd_even_transposition),
    ppc::util::AddFuncTask<OddEvenSortSeq, InType>(kTestParams, PPC_SETTINGS_volkov_a_odd_even_transposition));

INSTANTIATE_TEST_SUITE_P(VolkovOddEvenFunc, OddEvenSortFuncTest, ppc::util::ExpandToValues(kFuncTasks),
                         OddEvenSortFuncTest::PrintFuncTestName<OddEvenSortFuncTest>);

}  // namespace volkov_a_odd_even_transposition
