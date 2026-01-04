#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "akimov_i_radix_sort_double_batcher_merge/common/include/common.hpp"
#include "akimov_i_radix_sort_double_batcher_merge/mpi/include/ops_mpi.hpp"
#include "akimov_i_radix_sort_double_batcher_merge/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace akimov_i_radix_sort_double_batcher_merge {

class AkimovIRunFuncTestsRadixBatcherSort : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<2>(test_param);
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(param);
    expected_data_ = std::get<1>(param);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }
    if (output_data.size() != expected_data_.size()) {
      return false;
    }
    for (std::size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_data_[i]) {
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
  OutType expected_data_;
};

namespace {

// helper: produce sorted vector using std::ranges::sort
inline OutType SortedOutput(const InType &input) {
  OutType result = input;
  std::ranges::sort(result);
  return result;
}

const std::array<TestType, 15> kTestParam = {
    TestType{InType{}, OutType{}, "Empty"},
    TestType{InType{7.7}, OutType{7.7}, "SingleVal"},
    TestType{InType{-2.5, 0.0}, OutType{-2.5, 0.0}, "TwoSorted"},
    TestType{InType{9.9, 3.3}, OutType{3.3, 9.9}, "TwoReverse"},
    TestType{InType{12.4, 0.5, 99.1, 3.14, 6.28, 42.0}, OutType{0.5, 3.14, 6.28, 12.4, 42.0, 99.1}, "PositiveSet"},
    TestType{InType{-0.1, -100.0, -50.5, -3.3, -7.7}, OutType{-100.0, -50.5, -7.7, -3.3, -0.1}, "NegativeSet"},
    TestType{InType{-2.2, 5.5, -9.9, 0.0, 2.2, -1.1}, OutType{-9.9, -2.2, -1.1, 0.0, 2.2, 5.5}, "MixedSigns"},
    TestType{InType{-0.0, 0.0, -0.0, 0.0}, SortedOutput({-0.0, 0.0, -0.0, 0.0}), "ZeroVariants"},
    TestType{InType{2.5, 1.75, 3.333, 0.125}, OutType{0.125, 1.75, 2.5, 3.333}, "DecimalNumbers"},
    TestType{InType{9e12, 1e3, 5e9, 7e15, 2e11}, OutType{1e3, 5e9, 2e11, 9e12, 7e15}, "LargeRange"},
    TestType{InType{1e-20, 5e-6, 3e-12, 2e-3, 4e-9}, OutType{1e-20, 3e-12, 4e-9, 5e-6, 2e-3}, "SmallRange"},
    TestType{InType{4.4, 1.1, 4.4, 1.1, 4.4}, OutType{1.1, 1.1, 4.4, 4.4, 4.4}, "Duplicates"},
    TestType{InType{-5.0, -1.0, 0.0, 2.0, 3.0}, OutType{-5.0, -1.0, 0.0, 2.0, 3.0}, "AlreadySorted"},
    TestType{InType{100.0, 50.0, 0.0, -50.0, -100.0}, OutType{-100.0, -50.0, 0.0, 50.0, 100.0}, "ReverseSorted"},
    TestType{InType{10.0, 8.0, 6.0, 4.0, 2.0, 0.0, -2.0, -4.0}, OutType{-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0},
             "PowerOfTwoSize"}};

TEST_P(AkimovIRunFuncTestsRadixBatcherSort, Sorting) {
  ExecuteTest(GetParam());
}

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<AkimovIRadixBatcherSortSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_akimov_i_radix_sort_double_batcher_merge),
                                           ppc::util::AddFuncTask<AkimovIRadixBatcherSortMPI, InType>(
                                               kTestParam, PPC_SETTINGS_akimov_i_radix_sort_double_batcher_merge));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = AkimovIRunFuncTestsRadixBatcherSort::PrintFuncTestName<AkimovIRunFuncTestsRadixBatcherSort>;

// instantiate
INSTANTIATE_TEST_SUITE_P(SortingTests, AkimovIRunFuncTestsRadixBatcherSort, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace akimov_i_radix_sort_double_batcher_merge
