#include <gtest/gtest.h>
#include <stb/stb_image.h>

// #include <algorithm>
#include <array>
#include <cstddef>
// #include <cstdint>
// #include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
// #include <utility>
// #include <vector>

#include "konstantinov_s_elem_vec_sign_change_count/common/include/common.hpp"
#include "konstantinov_s_elem_vec_sign_change_count/mpi/include/ops_mpi.hpp"
#include "konstantinov_s_elem_vec_sign_change_count/seq/include/ops_seq.hpp"
#include "konstantinov_s_elem_vec_sign_change_count/tests/testgen.h"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace konstantinov_s_elem_vec_sign_change_count {

class KonstantinovSElemVecSignChangeTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    switch (params) {
      case 1: {  // все положительные
        input_data_ = InType{1.5, 2.3, 4.7, 8.9, 12.1};
        result_right_ = 0;
        break;
      }

      case 2: {  // все отрицательные
        std::array<EType, 6> arr = {-1.5, -2.3, -4.7, -8.9, -12.1, -0.5};
        input_data_ = InType(arr.begin(), arr.end());
        result_right_ = 0;
        break;
      }

      case 3: {  // частое чередование
        std::array<EType, 8> arr = {1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, 4.0};
        input_data_ = InType(arr.begin(), arr.end());
        result_right_ = 6;
        break;
      }

      case 4: {  // редкие смены
        std::array<EType, 10> arr = {1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, 4.0, 4.0};
        input_data_ = InType(arr.begin(), arr.end());
        result_right_ = 2;
        break;
      }

      case 5: {  // длинные серии
        std::array<EType, 12> arr = {1.1, 1.2, 1.3, 1.4, 1.5, -2.1, -2.2, -2.3, -2.4, 3.1, 3.2, 3.1};
        input_data_ = InType(arr.begin(), arr.end());
        result_right_ = 2;
        break;
      }

      case 6: {  // большие данные с частыми сменами (размножение)
        const size_t testdatasz = 1000000;
        InType vec(testdatasz);
        std::array<EType, 9> arr = {100.0, -50.0, 75.0, -25.0, 60.0, -60.0, 30.0, -30.0, 30.0};
        const int arrsz = 9;
        const int chngcnt = 8;
        result_right_ = GenerateTestData(arr.data(), arrsz, chngcnt, vec);
        input_data_ = vec;
        break;
      }

      case 7: {  // одинокая смена
        std::array<EType, 7> arr = {0.1, 0.2, 0.3, -0.1, 0.4, 0.5, 0.4};
        input_data_ = InType(arr.begin(), arr.end());
        result_right_ = 2;
        break;
      }

      case 8: {  // с нулями
        std::array<EType, 9> arr = {1.0, 0.0, -1.0, 0.0, 2.0, -3.0, 0.0, 0.0, 5.0};
        input_data_ = InType(arr.begin(), arr.end());
        result_right_ = 4;
        break;
      }

      case 9: {  // большие данные с редкими сменами (размножение)
        const size_t testdatasz = 1500000;
        InType vec(testdatasz);
        std::array<EType, 11> arr = {0.001, 0.002, -0.001, -0.002, 0.003, 0.004, 0.005, -0.003, -0.004, 0.006, 0.006};
        const int arrsz = 11;
        const int chngcnt = 4;
        result_right_ = GenerateTestData(arr.data(), arrsz, chngcnt, vec);
        input_data_ = vec;
        break;
      }

      case 10: {
        std::array<EType, 15> arr = {1, -1, 23, -11, -12, -167, 13, 42, -12, 2, -43, 33, 44, -7, 1};
        input_data_ = InType(arr.begin(), arr.end());
        result_right_ = 10;
        break;
      }

      default:
        throw std::invalid_argument("Invalid test case number");
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // std::cout<<"CHECK: ";
    // for(int i=0;i<input_data_.size();i++)
    //   std::cout<<input_data_[i]<<" ";
    // std::cout<<"CHECK DATA: "<<output_data<<" ?= "<<result_right_<<"\n";
    return (output_data == result_right_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType result_right_{};
};

namespace {

TEST_P(KonstantinovSElemVecSignChangeTests, SignChangeCount) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KonstantinovSElemVecSignChangeMPI, InType>(
                                               kTestParam, PPC_SETTINGS_konstantinov_s_elem_vec_sign_change_count),
                                           ppc::util::AddFuncTask<KonstantinovSElemVecSignChangeSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_konstantinov_s_elem_vec_sign_change_count));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KonstantinovSElemVecSignChangeTests::PrintFuncTestName<KonstantinovSElemVecSignChangeTests>;

INSTANTIATE_TEST_SUITE_P(SignChangeTests, KonstantinovSElemVecSignChangeTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace konstantinov_s_elem_vec_sign_change_count
