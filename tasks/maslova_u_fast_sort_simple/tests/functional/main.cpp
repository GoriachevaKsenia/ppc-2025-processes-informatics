#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "maslova_u_fast_sort_simple/common/include/common.hpp"
#include "maslova_u_fast_sort_simple/mpi/include/ops_mpi.hpp"
#include "maslova_u_fast_sort_simple/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace maslova_u_fast_sort_simple {

class MaslovaUFastSortFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<2>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_ = std::get<1>(params);
  }
  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    int is_initialized = 0;
    MPI_Initialized(&is_initialized);

    if (is_initialized != 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    if (rank != 0) {
      return true;
    }

    if (output_data.size() != expected_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(static_cast<double>(output_data[i]) - static_cast<double>(expected_[i])) > 1e-6) {
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
  OutType expected_;
};

TEST_P(MaslovaUFastSortFuncTests, fastSortSimple) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {
    {std::make_tuple(InType{}, OutType{}, "empty_array"), std::make_tuple(InType{42}, OutType{42}, "single_element"),
     std::make_tuple(InType{5, 2, 9, 1, 5}, OutType{1, 2, 5, 5, 9}, "random_with_duplicates"),
     std::make_tuple(InType{10, 9, 8, 7, 6}, OutType{6, 7, 8, 9, 10}, "reverse_sorted"),
     std::make_tuple(InType{-1, -5, 0, 10, -2}, OutType{-5, -2, -1, 0, 10}, "with_negatives"),
     std::make_tuple(InType(100, 1), InType(100, 1), "large_identical")}};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<MaslovaUFastSortSimpleMPI, InType>(kTestParam, PPC_SETTINGS_maslova_u_fast_sort_simple),
    ppc::util::AddFuncTask<MaslovaUFastSortSimpleSEQ, InType>(kTestParam, PPC_SETTINGS_maslova_u_fast_sort_simple));

INSTANTIATE_TEST_SUITE_P(fastSortTests, MaslovaUFastSortFuncTests, ppc::util::ExpandToValues(kTestTasksList),
                         MaslovaUFastSortFuncTests::PrintFuncTestName<MaslovaUFastSortFuncTests>);

}  // namespace maslova_u_fast_sort_simple
