#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "savva_d_min_elem_vec/common/include/common.hpp"
#include "savva_d_min_elem_vec/mpi/include/ops_mpi.hpp"
#include "savva_d_min_elem_vec/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace savva_d_min_elem_vec {

class SavvaDMinElemVecFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
  // тест один общий
 public:
  static std::string PrintTestParam(const TestType &test_param) {  // конструктор названия тестов
    const auto &vec = std::get<0>(test_param);
    return "vec_size_" + std::to_string(vec.size()) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {  // здесь данные готовятся - например читаются изображения
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (input_data_.empty()) {
      return false;
    }
    int expected_min = *std::ranges::min_element(input_data_);
    return expected_min == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {
// реализация (но пока не запуск) тестов
TEST_P(SavvaDMinElemVecFuncTests, MatmulFromPic) {  // не изменяется во всех задачах - генерация теста с параметрами
  ExecuteTest(GetParam());
}

// генерация тестовых параметров
const std::array<TestType, 5> kTestParam = {
    std::make_tuple(std::vector<int>{3, 1, 4, 1, 5}, "small"),
    std::make_tuple(std::vector<int>{9, 2, 6, 1, 8, 3, 7}, "medium"),
    std::make_tuple(std::vector<int>{5, 4, 3, 2, 1, 0, -1, -2}, "with_negatives"),
    std::make_tuple(std::vector<int>{-5, -55, -5, -3, -100000, -111111, -9, -111111}, "nagative"),
    std::make_tuple(std::vector<int>{7}, "singular"),
};
// не изменяется (определяет какие тесты будем запускать - сек и мпай )
const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<SavvaDMinElemVecMPI, InType>(kTestParam, PPC_SETTINGS_savva_d_min_elem_vec),
                   ppc::util::AddFuncTask<SavvaDMinElemVecSEQ, InType>(kTestParam, PPC_SETTINGS_savva_d_min_elem_vec));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SavvaDMinElemVecFuncTests::PrintFuncTestName<SavvaDMinElemVecFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, SavvaDMinElemVecFuncTests, kGtestValues,
                         kPerfTestName);  // здесь запуск тестов

}  // namespace

}  // namespace savva_d_min_elem_vec
