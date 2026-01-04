#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <string>
#include <tuple>

#include "rozenberg_a_bubble_odd_even_sort/common/include/common.hpp"
#include "rozenberg_a_bubble_odd_even_sort/mpi/include/ops_mpi.hpp"
#include "rozenberg_a_bubble_odd_even_sort/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace rozenberg_a_bubble_odd_even_sort {

class RozenbergABubbleOddEvenSortFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    input_data_.clear();
    output_data_.clear();
    if (CheckTestAndRank()) {
      TestType filename =
          std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam()) + ".txt";
      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_rozenberg_a_bubble_odd_even_sort, filename);
      std::ifstream file(abs_path);

      if (file.is_open()) {
        int size = 0;
        file >> size;

        InType input_data(size);
        for (int i = 0; i < size; i++) {
          file >> input_data[i];
        }

        OutType output_data(size);
        for (int i = 0; i < size; i++) {
          file >> output_data[i];
        }
        input_data_ = input_data;
        output_data_ = output_data;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (CheckTestAndRank()) {
      return (output_data_ == output_data);
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType output_data_;

  static bool CheckTestAndRank() {
    const std::string &test_name =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(GetParam());
    return (test_name.find("mpi") == std::string::npos || !ppc::util::IsUnderMpirun() || ppc::util::GetMPIRank() == 0);
  }
};

namespace {

TEST_P(RozenbergABubbleOddEvenSortFuncTests, BubbleOddEvenSort) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {"basic_test",       "duplicate_values_test", "inorder_test",
                                            "random_data_test", "random_data_test_2",    "reverse_order_test",
                                            "same_value_test",  "single_element_test"};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<RozenbergABubbleOddEvenSortMPI, InType>(
                                               kTestParam, PPC_SETTINGS_rozenberg_a_bubble_odd_even_sort),
                                           ppc::util::AddFuncTask<RozenbergABubbleOddEvenSortSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_rozenberg_a_bubble_odd_even_sort));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    RozenbergABubbleOddEvenSortFuncTests::PrintFuncTestName<RozenbergABubbleOddEvenSortFuncTests>;

INSTANTIATE_TEST_SUITE_P(BubbleOddEvenSortTests, RozenbergABubbleOddEvenSortFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace rozenberg_a_bubble_odd_even_sort
