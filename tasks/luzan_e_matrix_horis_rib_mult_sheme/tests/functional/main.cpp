#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "luzan_e_matrix_horis_rib_mult_sheme/common/include/common.hpp"
#include "luzan_e_matrix_horis_rib_mult_sheme/mpi/include/ops_mpi.hpp"
#include "luzan_e_matrix_horis_rib_mult_sheme/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace luzan_e_matrix_horis_rib_mult_sheme {

class LuzanEMatrixHorisRibMultShemeFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<int>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int height = std::get<0>(params);
    int width = std::get<1>(params);
    // int vec_len = std::get<2>(params);
    int vec_len = width;
    std::tuple_element_t<0, InType> mat(static_cast<size_t>(height) * static_cast<size_t>(width));
    std::tuple_element_t<3, InType> vec(static_cast<size_t>(vec_len));

    for (int elem = 0; elem < height * width; elem++) {
      mat[elem] = (elem % 42001) * (elem % 421);
    }

    for (int elem = 0; elem < vec_len; elem++) {
      vec[elem] = (elem % 4201) * (elem % 421);
    }

    input_data_ = std::make_tuple(mat, height, width, vec, vec_len);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int height = std::get<1>(input_data_);
    int width = std::get<2>(input_data_);
    // int vec_len = std::get<4>(input_data_);

    std::vector<int> prod(height, 0);
    std::tuple_element_t<0, InType> mat = std::get<0>(input_data_);
    std::tuple_element_t<3, InType> vec = std::get<3>(input_data_);

    int tmp_sum = 0;
    for (int row = 0; row < height; row++) {
      tmp_sum = 0;
      for (int col = 0; col < width; col++) {
        tmp_sum += mat[(width * row) + col] * vec[col];
      }
      prod[row] += tmp_sum;
    }

    for (int i = 0; i < height; i++) {
      if (height > 10) {
        break;
      }
    }
    return (output_data == prod);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(LuzanEMatrixHorisRibMultShemeFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 9> kTestParam = {
    std::make_tuple(3, 3),    std::make_tuple(2, 5),    std::make_tuple(10, 70),
    std::make_tuple(2000, 5), std::make_tuple(5, 2000), std::make_tuple(1, 1),
    std::make_tuple(1, 100),  std::make_tuple(100, 1),  std::make_tuple(1000, 1000)};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LuzanEMatrixHorisRibMultShemeMPI, InType>(
                                               kTestParam, PPC_SETTINGS_luzan_e_matrix_horis_rib_mult_sheme),
                                           ppc::util::AddFuncTask<LuzanEMatrixHorisRibMultShemeSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_luzan_e_matrix_horis_rib_mult_sheme));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    LuzanEMatrixHorisRibMultShemeFuncTests::PrintFuncTestName<LuzanEMatrixHorisRibMultShemeFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, LuzanEMatrixHorisRibMultShemeFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace luzan_e_matrix_horis_rib_mult_sheme
