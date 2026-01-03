#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "luzan_e_matrix_horis_rib_mult_sheme/common/include/common.hpp"
#include "luzan_e_matrix_horis_rib_mult_sheme/mpi/include/ops_mpi.hpp"
#include "luzan_e_matrix_horis_rib_mult_sheme/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace luzan_e_matrix_horis_rib_mult_sheme {

class LuzanEMatrixHorisRibMultShemepERFTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int height_ = 10000;
  const int width_ = 10000;
  const int vec_len_ = 10000;
  InType input_data_;

  void SetUp() override {
    std::tuple_element_t<0, InType> mat(static_cast<size_t>(height_) * static_cast<size_t>(width_));
    std::tuple_element_t<3, InType> vec(static_cast<size_t>(width_));

    for (int elem = 0; elem < height_ * width_; elem++) {
      mat[elem] = (elem % 42001) * (elem % 421);
    }

    for (int elem = 0; elem < vec_len_; elem++) {
      vec[elem] = (elem % 4201) * (elem % 421);
    }

    input_data_ = std::make_tuple(mat, height_, width_, vec, vec_len_);
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

    return (output_data == prod);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LuzanEMatrixHorisRibMultShemepERFTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LuzanEMatrixHorisRibMultShemeMPI, LuzanEMatrixHorisRibMultShemeSEQ>(
        PPC_SETTINGS_luzan_e_matrix_horis_rib_mult_sheme);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LuzanEMatrixHorisRibMultShemepERFTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LuzanEMatrixHorisRibMultShemepERFTests, kGtestValues, kPerfTestName);

}  // namespace luzan_e_matrix_horis_rib_mult_sheme
